"""
pipeline/signals.py — Technical signal computation and target construction.

Three Phase 0 changes:

  TARGET (T1.2 / F8). The target was ``close.shift(-30)``, an absolute price
  level. That choice made the reported error look small for the wrong reason —
  prices are persistent, so a random walk scores ~8% MAPE and the model scored
  ~15% — and it capped every forecast at the maximum price seen in training,
  because gradient-boosted trees cannot extrapolate (measured: 51 of 53 stocks).
  The target is now the forward 30-session log return in EXCESS of the stock's
  benchmark index, which is approximately stationary, bounded, and directly
  answers the question the leaderboard is actually asking.

  BACKFILL (F6). ``compute_and_store`` inserted only dates absent from the
  table. Rows written today carry a null target for the trailing 30 sessions;
  30 sessions later the label is computable, but the row already existed and
  was skipped forever. The labelled training set was therefore frozen at
  (first-run date - 30 sessions) and never grew. Writes are now upserts that
  refresh the target on every run.

  EARNINGS TIMING (F13). Earnings surprise was stamped on the announcement
  date. Indian results are frequently declared after the 15:30 IST close, so
  the model was reading information a trader could not have acted on until the
  next session. The surprise is now shifted to the next trading session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import text
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from data.db import get_engine
from data.tickers import get_benchmark

# Forecast horizon in TRADING SESSIONS, not calendar days. 30 sessions is
# roughly 42 calendar days on the NSE calendar. The previous code used
# shift(-30) while documenting "30 days"; naming it explicitly avoids the
# ambiguity.
HORIZON_SESSIONS = 30

FEATURE_COLS = [
    "rsi", "macd_hist", "bb_width", "obv", "sma_20", "ema_50", "bb_upper",
    "bb_lower", "ema_9", "ema_21", "atr_14", "stoch_k", "williams_r",
    "roc_10", "vroc_10", "prox_52w", "lag1_ret", "lag5_ret", "dev_sma50",
    "hurst", "sector_rel_5d", "sector_rel_10d", "sector_rel_20d",
    "earnings_surprise",
]

TARGET_COLS = ["target_return", "target_excess_return", "benchmark_return"]

_benchmark_cache: dict[str, pd.DataFrame] = {}


# ── Regime estimate ───────────────────────────────────────────────────────────
def compute_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Estimates the Hurst exponent from the scaling of the standard deviation of
    lagged differences (the "variance of increments" estimator, not rescaled
    range — the previous docstring claimed R/S while the code did this).

    Applied to LOG PRICES, so the estimate is scale-free; the previous version
    ran on raw price levels, which makes the exponent depend on the price.

    > 0.5 trending, < 0.5 mean-reverting, ~0.5 random walk. Returns 0.5 when
    the estimate is not computable.
    """
    try:
        values = np.log(np.asarray(series, dtype=float))
        if not np.all(np.isfinite(values)) or len(values) <= max_lag:
            return 0.5

        lags = range(2, max_lag)
        tau = [np.std(values[lag:] - values[:-lag]) for lag in lags]
        if any(t <= 0 for t in tau):
            return 0.5

        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return round(float(np.clip(poly[0], 0.0, 1.0)), 4)
    except Exception:
        return 0.5


def _rolling_hurst(close: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling Hurst over log prices.

    Vectorised over lags rather than calling a Python lambda per window, which
    was measurably slow across a 100-ticker universe.
    """
    log_close = np.log(close.astype(float))
    lags = np.arange(2, 20)
    log_lags = np.log(lags)

    tau_frames = []
    for lag in lags:
        diff = log_close.diff(lag)
        tau_frames.append(diff.rolling(window).std())

    tau = pd.concat(tau_frames, axis=1)
    tau = tau.where(tau > 0)
    log_tau = np.log(tau)

    # Slope of log(tau) on log(lag), computed in closed form per row.
    x_mean = log_lags.mean()
    x_dev = log_lags - x_mean
    denom = float((x_dev ** 2).sum())

    y_mean = log_tau.mean(axis=1)
    cov = (log_tau.sub(y_mean, axis=0) * x_dev).sum(axis=1)

    slope = cov / denom
    return slope.clip(0.0, 1.0).fillna(0.5).round(4)


# ── Benchmark series ──────────────────────────────────────────────────────────
def get_benchmark_series(index_ticker: str, period: str = "10y") -> pd.DataFrame:
    """
    Downloads a benchmark index close series, cached per process.

    The previous code downloaded the sector index once per stock, so a
    100-ticker run made 100 redundant requests for the same handful of indices.
    """
    if index_ticker in _benchmark_cache:
        return _benchmark_cache[index_ticker]

    try:
        data = yf.download(index_ticker, period=period, interval="1d",
                           auto_adjust=True, progress=False)
        if data is None or data.empty:
            raise ValueError("empty response")

        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(c[0]).lower() for c in data.columns]
        else:
            data.columns = [str(c).lower() for c in data.columns]

        out = pd.DataFrame({
            "date": data["date"].astype(str).str[:10],
            "benchmark_close": data["close"].astype(float),
        }).dropna()
    except Exception as exc:                                  # noqa: BLE001
        print(f"[Signals] benchmark {index_ticker} unavailable: {exc}")
        out = pd.DataFrame(columns=["date", "benchmark_close"])

    _benchmark_cache[index_ticker] = out
    return out


def compute_sector_momentum(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Relative momentum over 5/10/20 sessions: stock return minus benchmark
    return. Falls back to 0.0 when the benchmark is unavailable.
    """
    cols = ["sector_rel_5d", "sector_rel_10d", "sector_rel_20d"]

    if benchmark.empty:
        for col in cols:
            df[col] = 0.0
        return df

    df = df.merge(benchmark, on="date", how="left")
    df["benchmark_close"] = df["benchmark_close"].ffill()   # forward only, never bfill

    for window in [5, 10, 20]:
        stock_ret = df["close"].pct_change(window)
        bench_ret = df["benchmark_close"].pct_change(window)
        df[f"sector_rel_{window}d"] = (
            (stock_ret - bench_ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    return df


# ── Earnings ──────────────────────────────────────────────────────────────────
def compute_earnings_surprise(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Earnings surprise, (actual - estimate) / |estimate|, clipped to [-2, 2] and
    forward-filled between quarters.

    The surprise is attached to the first trading session STRICTLY AFTER the
    announcement date (F13). Indian results are commonly declared post-close,
    so stamping the announcement date itself let the model read information one
    session before it was tradable.
    """
    try:
        earnings = yf.Ticker(ticker).earnings_dates
        if earnings is None or len(earnings) == 0:
            df["earnings_surprise"] = 0.0
            return df

        earnings = earnings.reset_index()
        earnings.columns = [c.lower().replace(" ", "_") for c in earnings.columns]

        est_col = next((c for c in earnings.columns if "estimate" in c), None)
        act_col = next((c for c in earnings.columns if "actual" in c or "reported" in c), None)
        date_col = next((c for c in earnings.columns if "date" in c), None)

        if not (est_col and act_col and date_col):
            df["earnings_surprise"] = 0.0
            return df

        earnings["announced"] = pd.to_datetime(
            earnings[date_col], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%d")

        earnings = earnings[["announced", est_col, act_col]].dropna()
        earnings = earnings[earnings[est_col].abs() > 0.001]
        if earnings.empty:
            df["earnings_surprise"] = 0.0
            return df

        earnings["surprise"] = (
            (earnings[act_col] - earnings[est_col]) / earnings[est_col].abs()
        ).clip(-2.0, 2.0)

        # Map each announcement onto the first session strictly after it.
        sessions = df["date"].tolist()
        surprise_by_session: dict[str, float] = {}
        for _, row in earnings.iterrows():
            later = [s for s in sessions if s > row["announced"]]
            if later:
                surprise_by_session[later[0]] = float(row["surprise"])

        df["earnings_surprise"] = df["date"].map(surprise_by_session).ffill().fillna(0.0)

    except Exception as exc:                                  # noqa: BLE001
        print(f"[Signals] earnings surprise failed for {ticker}: {exc}")
        df["earnings_surprise"] = 0.0

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def compute_signals_frame(ticker: str, ohlcv: pd.DataFrame) -> pd.DataFrame | None:
    """
    Computes indicators and the forward excess-return target for one ticker.

    Prices use ``adj_close`` so the whole series shares one adjustment basis.
    Returns None when there is not enough history.
    """
    if ohlcv.empty or len(ohlcv) < 120:
        return None

    df = ohlcv.sort_values("date").reset_index(drop=True)

    # Put the whole OHLC bar on the adjusted basis. Adjusting close alone while
    # leaving high/low raw would corrupt every range-based indicator (ATR,
    # Stochastic, Williams %R, 52-week proximity) across any corporate action.
    raw_close = df["close"].astype(float)
    adj_close = df["adj_close"].astype(float).where(df["adj_close"].notna(), raw_close)
    factor = (adj_close / raw_close).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df["close"] = adj_close
    df["high"] = df["high"].astype(float) * factor
    df["low"] = df["low"].astype(float) * factor
    df["volume"] = df["volume"].astype(float)

    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["macd_hist"] = MACD(close=df["close"]).macd_diff()

    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_width"] = bb.bollinger_wband()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["ema_9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["atr_14"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["stoch_k"] = StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).stoch()
    df["williams_r"] = WilliamsRIndicator(
        high=df["high"], low=df["low"], close=df["close"], lbp=14
    ).williams_r()
    df["roc_10"] = ROCIndicator(close=df["close"], window=10).roc()
    df["vroc_10"] = df["volume"].pct_change(10).replace([np.inf, -np.inf], np.nan)

    rolling_high = df["high"].rolling(252, min_periods=50).max()
    rolling_low = df["low"].rolling(252, min_periods=50).min()
    span = rolling_high - rolling_low
    df["prox_52w"] = np.where(span == 0, 0.5, (df["close"] - rolling_low) / span)

    df["lag1_ret"] = df["close"].pct_change(1)
    df["lag5_ret"] = df["close"].pct_change(5)
    sma_50 = SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["dev_sma50"] = (df["close"] - sma_50) / sma_50 * 100
    df["hurst"] = _rolling_hurst(df["close"])

    # ── Benchmark, relative momentum, and the target ─────────────────────────
    index_ticker, is_sector = get_benchmark(ticker)
    benchmark = get_benchmark_series(index_ticker)

    df = compute_sector_momentum(df, benchmark)
    df = compute_earnings_surprise(ticker, df)

    if "benchmark_close" not in df.columns:
        df = df.merge(benchmark, on="date", how="left")
        df["benchmark_close"] = df["benchmark_close"].ffill()

    h = HORIZON_SESSIONS
    log_close = np.log(df["close"])
    df["target_return"] = log_close.shift(-h) - log_close

    if df["benchmark_close"].notna().any():
        log_bench = np.log(df["benchmark_close"])
        df["benchmark_return"] = log_bench.shift(-h) - log_bench
    else:
        df["benchmark_return"] = np.nan

    df["target_excess_return"] = df["target_return"] - df["benchmark_return"]

    df["benchmark_ticker"] = index_ticker
    df["benchmark_sector_specific"] = 1 if is_sector else 0
    df["ticker"] = ticker

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS)

    keep = (["date", "ticker", "close"] + FEATURE_COLS + TARGET_COLS
            + ["benchmark_ticker", "benchmark_sector_specific"])
    return df[keep].reset_index(drop=True)


def _upsert_signals(conn, ticker: str, df: pd.DataFrame) -> int:
    """
    Writes signal rows, refreshing rows that already exist.

    This replaces the append-only insert that caused F6. Deleting the ticker's
    rows in the recomputed date range and reinserting is the portable way to
    upsert across SQLite and PostgreSQL, and it also picks up any indicator
    correction, not just the target.
    """
    if df.empty:
        return 0
    conn.execute(
        text("DELETE FROM signals WHERE ticker = :t AND date >= :start"),
        {"t": ticker, "start": df["date"].min()},
    )
    df.to_sql("signals", con=conn, if_exists="append", index=False)
    return len(df)


def compute_and_store(single_ticker: str | None = None,
                      tickers: list[str] | None = None) -> int:
    """Computes signals for the given tickers and upserts them."""
    engine = get_engine()

    if single_ticker:
        to_process = [single_ticker]
    elif tickers:
        to_process = list(tickers)
    else:
        from data.universe import get_universe
        to_process = get_universe()

    total = 0
    for ticker in to_process:
        ohlcv = pd.read_sql(
            text("SELECT * FROM ohlcv WHERE ticker = :t ORDER BY date ASC"),
            engine, params={"t": ticker},
        )
        frame = compute_signals_frame(ticker, ohlcv)
        if frame is None or frame.empty:
            print(f"[Signals] {ticker}: insufficient history")
            continue

        with engine.connect() as conn:
            written = _upsert_signals(conn, ticker, frame)
            conn.commit()

        labelled = int(frame["target_excess_return"].notna().sum())
        total += written
        print(f"[Signals] {ticker}: {written} rows ({labelled} labelled)")

    print(f"[Signals] Complete. {total} rows written.")
    return total


def count_labelled_rows(ticker: str | None = None) -> int:
    """
    Number of rows with a usable target. The F6 regression test asserts this
    is non-decreasing across runs.
    """
    engine = get_engine()
    if ticker:
        q = text("SELECT COUNT(*) AS n FROM signals "
                 "WHERE ticker = :t AND target_excess_return IS NOT NULL")
        df = pd.read_sql(q, engine, params={"t": ticker})
    else:
        df = pd.read_sql(
            text("SELECT COUNT(*) AS n FROM signals WHERE target_excess_return IS NOT NULL"),
            engine,
        )
    return int(df["n"].iloc[0]) if not df.empty else 0
