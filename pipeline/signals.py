"""
pipeline/signals.py — Technical signal computation and target construction.

Three Phase 0 changes:

  TARGET (T1.2 / F8). The target was ``close.shift(-30)``, an absolute price
  level. That choice made the reported error look small for the wrong reason —
  prices are persistent, so a random walk scores ~8% MAPE and the model scored
  ~15% — and it capped every forecast at the maximum price seen in training,
  because gradient-boosted trees cannot extrapolate (measured: 51 of 53 stocks).
  The target is now the forward 30-session log return in EXCESS of the stock's
  benchmark index, which is approximately stationary and bounded.

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

from typing import NamedTuple

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import text
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from data.db import get_engine
from pipeline.sector_benchmark import Benchmark

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

#: Features that may legitimately be NULL on a row, and are stored as NULL
#: rather than as a neutral number (2026-09-24, MODEL_VERSION v4). Every one
#: of them used to be filled with a value that sits ON the scale:
#:
#:   sector_rel_*       0.0 when the benchmark was unavailable ("exactly in line
#:                      with the sector"); now NULL for a thin sector, with
#:                      `sector_rel_missing = 1` beside it. See
#:                      pipeline/sector_benchmark.py.
#:   earnings_surprise  0.0 before the vendor's first recorded announcement —
#:                      88-94% of rows in 2016-2020, measured — i.e. "never
#:                      observed" stored as "no surprise"; and, the other way
#:                      round, one old surprise carried for YEARS when the
#:                      vendor stopped recording a ticker (EARNINGS_STALE_
#:                      SESSIONS).
#:   hurst              0.0 for EVERY ticker on 13 dates in late 2016: a
#:                      partial-window artifact (the fit ran over whichever
#:                      lags existed yet), plus 0.5 when not computable.
#:
#: A zero is indistinguishable downstream from a real measurement of zero,
#: which is the silent-neutral defect this project has shipped three times.
#: XGBoost reads NaN natively as MISSING, with a learned default branch, so
#: the tree models take these as NaN; `panel.cross_sectional_zscore` keeps
#: them NaN on request. A row is never DROPPED for one of these being NULL -
#: dropping would lose a labelled row and the write guard would (rightly)
#: refuse the whole ticker.
NULLABLE_FEATURES: tuple[str, ...] = (
    "sector_rel_5d", "sector_rel_10d", "sector_rel_20d",
    "earnings_surprise", "hurst",
)
SECTOR_REL_COLS = ("sector_rel_5d", "sector_rel_10d", "sector_rel_20d")


class LabelLossRefused(RuntimeError):
    """
    Raised instead of committing a write that would leave a ticker with fewer
    labelled rows than it already had.

    _upsert_signals is DELETE-range-then-reinsert, so it is not additive: the
    frame handed to it wholly REPLACES the range it covers. When that frame's
    targets are null — the standard consequence of a benchmark index that
    failed to download, since the target is `stock return - benchmark return`
    — the write erases every label in the range and returns a healthy row
    count. On 2026-08-16 that cost ~2,390 labels each across 22 tickers.

    The F6 monotonicity guard in scheduler.py catches this after the fact, at
    the whole-table level, and by then can only abort the run: the labels are
    already gone, and only a later clean recompute restores them. This is the
    same check moved to the write boundary, which is the last point at which
    refusing is still an option.
    """


class SignalsReport(NamedTuple):
    """
    Outcome of a compute_and_store run.

    `refused` is deliberately separate from `skipped`. A skip means the frame
    was never built; a refusal means it was built, looked writable, and would
    have destroyed labels. The second is the more alarming of the two and used
    to be indistinguishable from success.
    """
    rows_written: int
    processed: list[str]
    skipped: list[str]
    refused: list[str]


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

    NaN UNTIL EVERY LAG HAS A FULL WINDOW (2026-09-24). The slope used to be
    fitted over whichever lags happened to exist yet, against deviations
    computed over ALL of them, which is a different and biased estimator: it
    wrote exactly 0.0 for every ticker on 13 dates in late 2016. And an
    incomputable row used to be filled with 0.5, "a random walk", which is a
    claim, not an absence. Both are NaN now; see NULLABLE_FEATURES.
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
    complete = tau.notna().all(axis=1)
    log_tau = np.log(tau).where(complete, axis=0)

    # Slope of log(tau) on log(lag), computed in closed form per row.
    x_mean = log_lags.mean()
    x_dev = log_lags - x_mean
    denom = float((x_dev ** 2).sum())

    y_mean = log_tau.mean(axis=1)
    cov = (log_tau.sub(y_mean, axis=0) * x_dev).sum(axis=1)

    slope = (cov / denom).where(complete)
    return slope.clip(0.0, 1.0).round(4)


# ── Benchmark and relative momentum ──────────────────────────────────────────


def _window_void(void_cum: pd.Series, steps: int) -> pd.Series:
    """True where the `steps`-row window ending (steps > 0) or starting
    (steps < 0) at this row spans a date the benchmark could not be formed."""
    if steps > 0:
        return (void_cum - void_cum.shift(steps)) > 0
    return (void_cum.shift(steps) - void_cum) > 0


def attach_benchmark(df: pd.DataFrame, benchmark: Benchmark | None) -> pd.DataFrame:
    """The benchmark level and its void counter, on this ticker's own rows.

    Forward-filled onto a date the benchmark's grid lacks, never backward: a
    panel-internal benchmark is built on the universe's own grid so this is a
    no-op for a universe member; it matters only for an external series."""
    if benchmark is None or benchmark.frame.empty:
        df["benchmark_close"] = np.nan
        df["bench_void_cum"] = np.nan
        return df
    frame = benchmark.frame[["date", "benchmark_close", "bench_void_cum"]]
    df = df.merge(frame, on="date", how="left")
    df["benchmark_close"] = df["benchmark_close"].ffill()      # never bfill
    df["bench_void_cum"] = df["bench_void_cum"].ffill()
    return df


def compute_sector_momentum(df: pd.DataFrame, benchmark: Benchmark | None) -> pd.DataFrame:
    """
    Relative momentum over 5/10/20 sessions: the stock's return minus its
    benchmark's over the same rows.

    NULL, NEVER 0.0, WHEN THERE IS NO SECTOR BENCHMARK. This used to write 0.0
    for every row when the index failed to download — "exactly in line with
    the sector", on a scale where that is a real and common value. And before
    that it FORWARD-FILLED a dead index, so for the two months after Yahoo
    stopped publishing it, `sector_rel_*` for 49 tickers was their own raw
    momentum. Now: a thin or unlabelled sector has no sector benchmark, these
    columns are NULL, and `sector_rel_missing` says so. A window that spans a
    date the peers could not form a mean is NULL too, never stretched.
    """
    df = attach_benchmark(df, benchmark)
    usable = benchmark is not None and benchmark.relative_features

    for window in (5, 10, 20):
        col = f"sector_rel_{window}d"
        if not usable:
            df[col] = np.nan
            continue
        stock_ret = df["close"].pct_change(window)
        bench_ret = df["benchmark_close"].pct_change(window)
        rel = (stock_ret - bench_ret).replace([np.inf, -np.inf], np.nan)
        df[col] = rel.where(~_window_void(df["bench_void_cum"], window))
    df["sector_rel_missing"] = df[list(SECTOR_REL_COLS)].isna().any(axis=1).astype(int)
    return df


# ── Earnings ──────────────────────────────────────────────────────────────────

#: `yf.Ticker().earnings_dates` parses HTML through `pandas.read_html`, which
#: needs one of these. yfinance 1.3.0 declares NEITHER, so a perfectly valid
#: install can lack it.
HTML_PARSERS = ("lxml", "html5lib")


class EarningsParserMissing(RuntimeError):
    """No HTML parser, so no earnings dates, so `earnings_surprise` would be a
    constant zero for every ticker — silently, because the fetch is caught."""


def require_earnings_parser() -> None:
    """
    Refuse to compute signals with no HTML parser installed.

    MEASURED, 2026-09-22, on the first CI run after the dependency lock landed.
    Production had been resolving yfinance 1.7.0, which depends on lxml; the
    lock pins the researched 1.3.0, which does not, so the runner had none and
    every ticker logged "earnings surprise failed ... Import lxml failed". The
    except branch below then wrote 0.0 — and `earnings_surprise` is one of the
    15 pooled FACTORS. Nothing failed, nothing was empty, and a whole feature
    column quietly became a constant on every row the job rewrote.

    That is the FinBERT gauge and the FII-flow zeros for the third time: a
    FAILED MEASUREMENT STORED AS A VALID NEUTRAL VALUE. The package is pinned
    in requirements.in now; this makes its absence loud rather than invisible,
    because a pin is only as good as the environment that honours it.
    """
    from importlib.util import find_spec

    if not any(find_spec(name) is not None for name in HTML_PARSERS):
        raise EarningsParserMissing(
            f"none of {list(HTML_PARSERS)} is installed, so "
            f"yfinance.earnings_dates cannot parse and earnings_surprise would "
            f"be written as a constant 0.0 for every ticker. Install the locked "
            f"environment: pip install -r requirements.txt")


#: How long a surprise stays "current" with no new announcement, in sessions.
#: Measured on the vendor tables in the Stage 2 snapshot (1,902 consecutive
#: announcement pairs, 84 tickers): median gap 91 calendar days, 95th
#: percentile 112, 99th 265 — the tail is missed quarters. 85 sessions is ~120
#: calendar days: every ordinary reporting cycle plus a week of slack. Beyond
#: it the value is NULL, never the stale number.
EARNINGS_STALE_SESSIONS = 85


class EarningsUnavailable(RuntimeError):
    """The vendor returned no usable earnings history for one ticker.

    Raised, never absorbed as 0.0. `compute_and_store` then SKIPS the ticker:
    it keeps the signals it already has and is named in the run's `skipped`
    list, which is loud and self-healing. The alternative this replaced wrote
    `earnings_surprise = 0.0` over the ticker's WHOLE history — and because
    the signals write is DELETE-range-then-reinsert, one transient vendor
    failure silently replaced every real surprise value it had."""


def earnings_surprise_from(earnings: pd.DataFrame | None, df: pd.DataFrame,
                           ticker: str = "") -> pd.DataFrame:
    """
    Maps a vendor earnings table onto `df`'s sessions. Pure: no network.

    NULL BEFORE THE FIRST RECORDED ANNOUNCEMENT, NEVER 0.0 (2026-09-24). The
    vendor's history starts around 2020-21; before it, this used to write 0.0
    — measured on 88-94% of rows in 2016-2020 — which says "the result matched
    the estimate" about quarters nobody observed. Forward-filling BETWEEN
    announcements is the design (the last surprise is the current one); filling
    BEFORE the first is invention.
    """
    if earnings is None or len(earnings) == 0:
        raise EarningsUnavailable(f"{ticker}: no earnings history returned")

    earnings = earnings.reset_index()
    earnings.columns = [str(c).lower().replace(" ", "_") for c in earnings.columns]

    est_col = next((c for c in earnings.columns if "estimate" in c), None)
    act_col = next((c for c in earnings.columns if "actual" in c or "reported" in c), None)
    date_col = next((c for c in earnings.columns if "date" in c), None)
    if not (est_col and act_col and date_col):
        raise EarningsUnavailable(
            f"{ticker}: earnings table lacks estimate/actual/date columns: "
            f"{list(earnings.columns)}")

    earnings["announced"] = pd.to_datetime(
        earnings[date_col], errors="coerce", utc=True
    ).dt.strftime("%Y-%m-%d")

    earnings = earnings[["announced", est_col, act_col]].dropna()
    earnings = earnings[earnings[est_col].abs() > 0.001]
    if earnings.empty:
        raise EarningsUnavailable(f"{ticker}: no announcement carries an estimate")

    earnings["surprise"] = (
        (earnings[act_col] - earnings[est_col]) / earnings[est_col].abs()
    ).clip(-2.0, 2.0)

    # Map each announcement onto the first session strictly after it (F13).
    sessions = df["date"].tolist()
    surprise_by_session: dict[str, float] = {}
    for _, row in earnings.iterrows():
        later = [s for s in sessions if s > row["announced"]]
        if later:
            surprise_by_session[later[0]] = float(row["surprise"])

    # CARRIED FORWARD FOR ONE REPORTING CYCLE, NOT FOREVER (2026-09-24). The
    # last surprise IS the current one - until the next announcement is due.
    # Past that, the vendor has missed a quarter and the old value is stale,
    # not current: measured on the Stage 2 snapshot, BAJAJHLDNG's last
    # announcement with an estimate was 2018-10-24 and its surprise had been
    # carried for 1,954 sessions; ADANIPOWER for 1,463. See
    # EARNINGS_STALE_SESSIONS for the measured cycle.
    df["earnings_surprise"] = (df["date"].map(surprise_by_session)
                               .ffill(limit=EARNINGS_STALE_SESSIONS))
    return df


def compute_earnings_surprise(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Earnings surprise, (actual - estimate) / |estimate|, clipped to [-2, 2] and
    forward-filled between quarters.

    The surprise is attached to the first trading session STRICTLY AFTER the
    announcement date (F13). Indian results are commonly declared post-close,
    so stamping the announcement date itself let the model read information one
    session before it was tradable.

    Any failure raises EarningsUnavailable — see there for why that is a skip
    and not a 0.0.
    """
    try:
        earnings = yf.Ticker(ticker).earnings_dates
    except Exception as exc:                                  # noqa: BLE001
        raise EarningsUnavailable(f"{ticker}: earnings fetch failed: {exc}") from exc
    return earnings_surprise_from(earnings, df, ticker)


# ── Main ──────────────────────────────────────────────────────────────────────
def compute_signals_frame(ticker: str, ohlcv: pd.DataFrame,
                          benchmark: Benchmark | None = None) -> pd.DataFrame | None:
    """
    Computes indicators and both forward targets for one ticker.

    Prices use ``adj_close`` so the whole series shares one adjustment basis.
    Returns None when there is not enough history, or when the vendor's
    earnings history is unavailable (EarningsUnavailable: skip, never 0.0).

    `benchmark` comes from `pipeline.sector_benchmark.build_benchmarks`, which
    needs the whole universe's prices and so is built once per run by
    `compute_and_store`, not here. None means "no benchmark": the relative
    features and the excess label are NULL, the absolute label is unaffected.
    """
    if ohlcv.empty or len(ohlcv) < 120:
        print(f"[Signals] {ticker}: insufficient history ({len(ohlcv)} rows)")
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
    # A ZERO-VOLUME SESSION MUST NOT DELETE A DIFFERENT SESSION.
    #
    # pct_change(10) divides by the volume ten rows back, so a single
    # zero-volume day produces inf ten rows LATER, which becomes NaN and then
    # loses that row to dropna(subset=FEATURE_COLS) below. RELIANCE.NS carries
    # five zero-volume sessions and was missing exactly five interior rows
    # because of it -- healthy rows, with normal OHLC, deleted by a defect in a
    # neighbour a fortnight earlier, and silently.
    #
    # The lost rows are the smaller half of the problem. The larger half is
    # that the stored session grid ends up with invisible interior gaps, while
    # target_excess_return is computed on the FULL ohlcv sequence above, before
    # the drop. Anything that steps by stored rows -- a series model asked to
    # forecast 30 steps ahead -- then measures a different horizon from the one
    # the label describes, for every window spanning a gap.
    #
    # Volume ten sessions ago being zero makes the rate of change undefined,
    # not infinite. 0.0 is the honest reading ("no measurable change") and it
    # keeps the row, which is what matters.
    prev_volume = df["volume"].shift(10)
    df["vroc_10"] = np.where(prev_volume > 0,
                             df["volume"] / prev_volume.where(prev_volume > 0) - 1.0,
                             0.0)

    rolling_high = df["high"].rolling(252, min_periods=50).max()
    rolling_low = df["low"].rolling(252, min_periods=50).min()
    span = rolling_high - rolling_low
    df["prox_52w"] = np.where(span == 0, 0.5, (df["close"] - rolling_low) / span)

    df["lag1_ret"] = df["close"].pct_change(1)
    df["lag5_ret"] = df["close"].pct_change(5)
    sma_50 = SMAIndicator(close=df["close"], window=50).sma_indicator()
    df["dev_sma50"] = (df["close"] - sma_50) / sma_50 * 100
    df["hurst"] = _rolling_hurst(df["close"])

    # ── Benchmark, relative momentum, and the targets ────────────────────────
    #
    # THE BENCHMARK IS PANEL-INTERNAL SINCE MODEL_VERSION v4 (2026-09-24): the
    # leave-one-out equal-weighted mean of the stock's sector peers in this
    # universe, or of the whole universe for a thin or unlabelled sector. See
    # pipeline/sector_benchmark.py. The Yahoo sector indices this replaces
    # stopped publishing around 2026-07-20 and left 49 tickers refused.
    #
    # The primary label never depended on it (P1): target_return asks the
    # benchmark nothing, so a missing benchmark costs the excess label and the
    # relative features, and nothing else.
    df = compute_sector_momentum(df, benchmark)
    try:
        df = compute_earnings_surprise(ticker, df)
    except EarningsUnavailable as exc:
        print(f"[Signals] SKIPPED {exc} — keeping the stored signals rather "
              f"than writing a neutral 0.0 over them.")
        return None

    h = HORIZON_SESSIONS
    log_close = np.log(df["close"])

    # THE PRIMARY LABEL, and it depends on nothing but this ticker's own close.
    df["target_return"] = log_close.shift(-h) - log_close

    if benchmark is not None and df["benchmark_close"].notna().any():
        log_bench = np.log(df["benchmark_close"])
        bench_return = log_bench.shift(-h) - log_bench
        # A forward window that spans a date the peers could not form a mean
        # is void, exactly as a backward feature window is.
        df["benchmark_return"] = bench_return.where(
            ~_window_void(df["bench_void_cum"], -h))
        df["target_excess_return"] = df["target_return"] - df["benchmark_return"]
    else:
        print(f"[Signals] {ticker}: no benchmark — the excess-return label and "
              f"the relative features are NULL for this run; the absolute-"
              f"return label is unaffected.")
        df["benchmark_close"] = np.nan
        df["benchmark_return"] = np.nan
        df["target_excess_return"] = np.nan
    df["benchmark_ticker"] = benchmark.name if benchmark is not None else None
    df["benchmark_sector_specific"] = (
        int(benchmark.sector_specific) if benchmark is not None else 0)
    df["ticker"] = ticker

    df = df.replace([np.inf, -np.inf], np.nan)
    # Only a REQUIRED feature drops a row; a NULLABLE one is stored as NULL.
    df = df.dropna(subset=[c for c in FEATURE_COLS if c not in NULLABLE_FEATURES])

    # A row with no benchmark still has close, every feature and the primary
    # label. Dropping on the target columns here would undo the whole point.

    # benchmark_close is persisted alongside the targets, not as one of them.
    # It is an observation at date t, so unlike benchmark_return it is safe to
    # read as an input - see the note in data/db.py.
    keep = (["date", "ticker", "close"] + FEATURE_COLS + TARGET_COLS
            + ["benchmark_close", "benchmark_ticker",
               "benchmark_sector_specific", "sector_rel_missing"])
    return df[keep].reset_index(drop=True)


#: Every label the write boundary protects. The primary target first.
LABEL_COLS = ("target_return", "target_excess_return")


def _labelled_rows_from(conn, ticker: str, start: str) -> dict[str, int]:
    """
    Labelled rows this ticker already holds in the range about to be replaced,
    PER TARGET.

    Both are counted, not just the primary one. Since P1 a dead benchmark index
    costs only the excess label, so a recompute for such a ticker writes a full
    set of target_return values and a column of NULL excess ones. Counting the
    primary target alone would wave that through — and because the write is a
    DELETE over the range, it would erase every historical excess label the
    ticker had from before its index went dark. That is F6 exactly, reappearing
    through the door the target switch opened.

    ONLY LABELS ON DAYS NSE TRADED ARE PROTECTED (2026-09-21). A row stored on
    a holiday Yahoo invented a bar for is a defect, not training data, and
    `pipeline/fetch.py` now removes those days. Counting them here would make
    the first clean recompute a "decrease" for every ticker and refuse the lot.
    Nothing else is excused: a label lost to a vendor gap on a real session,
    or to a feature turning NaN, still refuses the write.
    """
    calendar = session_calendar()
    counts = {}
    for col in LABEL_COLS:
        dates = [r[0] for r in conn.execute(
            text(f"SELECT date FROM signals WHERE ticker = :t "
                 f"AND date >= :start AND {col} IS NOT NULL"),
            {"t": ticker, "start": start},
        )]
        counts[col] = int(calendar.session_mask(dates).sum()) if dates else 0
    return counts


def session_calendar():
    """The NSE calendar the fetch step used this run. Indirected so a test can
    hand the write guard a calendar without a network."""
    from data import nse_calendar

    return nse_calendar.current()[0]


def _upsert_signals(conn, ticker: str, df: pd.DataFrame) -> int:
    """
    Writes signal rows, refreshing rows that already exist.

    This replaces the append-only insert that caused F6. Deleting the ticker's
    rows in the recomputed date range and reinserting is the portable way to
    upsert across SQLite and PostgreSQL, and it also picks up any indicator
    correction, not just the target.

    THE DELETE IS THE DANGEROUS HALF. Because the range is cleared before the
    reinsert, this call is a replacement and not a merge, so a frame carrying
    fewer labels than the rows it displaces is a net loss of training data
    dressed up as a successful write. The counts are therefore compared BEFORE
    the delete and the write refused outright -- the last moment at which the
    existing labels still exist. See LabelLossRefused.

    EVERY label is checked, not only the primary one -- see
    _labelled_rows_from for why the target switch made that necessary.

    The comparison is a decrease check rather than a null-frame check on
    purpose. A null-frame check catches only total destruction, and would also
    wrongly block a genuinely new listing whose forward labels are all still in
    the future. Comparing counts over the same date range catches the partial
    case too -- an index that aligns to only part of the history, or a feature
    that turns NaN and drops labelled rows out through dropna -- while letting
    the new-listing case through, because 0 is not less than 0.
    """
    if df.empty:
        return 0

    start = df["date"].min()
    existing = _labelled_rows_from(conn, ticker, start)
    # Both sides are counted over NSE sessions only, so the comparison is like
    # with like whether or not the incoming frame still carries a phantom row.
    on_session = session_calendar().session_mask(df["date"])

    for col in LABEL_COLS:
        incoming = (int((df[col].notna().to_numpy() & on_session).sum())
                    if col in df.columns else 0)
        if incoming < existing[col]:
            raise LabelLossRefused(
                f"{ticker}: this frame would drop {col} from {existing[col]} to "
                f"{incoming} labelled rows over dates >= {start}. Refusing the "
                f"write; the ticker keeps its existing labels and its signals go "
                f"stale until the next clean run. The usual cause is a benchmark "
                f"index that failed to download or failed to align, which NULLs "
                f"target_excess_return for every stock mapped to it — note that "
                f"target_return is unaffected by that, so a refusal naming the "
                f"excess column alone is the outage and not a data defect."
            )

    conn.execute(
        text("DELETE FROM signals WHERE ticker = :t AND date >= :start"),
        {"t": ticker, "start": start},
    )
    df.to_sql("signals", con=conn, if_exists="append", index=False)
    return len(df)


def build_run_benchmarks(to_process: list[str], engine=None) -> dict[str, Benchmark]:
    """Every processed ticker's panel-internal benchmark, peers drawn from the
    frozen universe (plus the processed tickers themselves)."""
    from data.universe import get_universe
    from pipeline.sector_benchmark import (build_benchmarks, load_universe_prices,
                                           sector_labels)

    engine = engine or get_engine()
    peers = sorted(set(get_universe()) | set(to_process))
    prices = load_universe_prices(peers, engine)
    if prices.empty:
        return {}
    labels = sector_labels(peers, engine)
    unlabelled = sorted(t for t in to_process if labels.get(t) is None)
    if unlabelled:
        print(f"[Signals] {len(unlabelled)} ticker(s) carry no industry label and "
              f"get NO sector benchmark (market fallback for the excess label, "
              f"NULL sector_rel): {', '.join(unlabelled)}")
    return build_benchmarks(prices, labels, tickers=to_process)


def compute_and_store(single_ticker: str | None = None,
                      tickers: list[str] | None = None) -> SignalsReport:
    """
    Computes signals for the given tickers and upserts them.

    A ticker that is skipped or refused keeps whatever it already had, so this
    degrades per ticker rather than per run: one dead benchmark index costs the
    stocks mapped to it a day of freshness, not the other ninety their
    forecast. Both outcomes are returned rather than only printed, because the
    callers in scheduler.py have to decide whether the run is still worth
    continuing and previously could not see either.
    """
    engine = get_engine()
    # BEFORE the loop, not inside it: a missing HTML parser is a configuration
    # error that would otherwise be absorbed per ticker as earnings_surprise =
    # 0.0, one factor column at a time.
    require_earnings_parser()

    from data.universe import get_universe

    if single_ticker:
        to_process = [single_ticker]
    elif tickers:
        to_process = list(tickers)
    else:
        to_process = get_universe()

    # THE BENCHMARKS NEED THE WHOLE UNIVERSE, so they are built once, here,
    # before the loop — one price query and one membership query, however
    # many tickers are processed. Peers are the FROZEN universe, never "who is
    # in the index today", so a sector's composition does not churn under a
    # label. See pipeline/sector_benchmark.py.
    benchmarks = build_run_benchmarks(to_process, engine)

    total = 0
    processed: list[str] = []
    skipped: list[str] = []
    refused: list[str] = []

    for ticker in to_process:
        ohlcv = pd.read_sql(
            text("SELECT * FROM ohlcv WHERE ticker = :t ORDER BY date ASC"),
            engine, params={"t": ticker},
        )
        frame = compute_signals_frame(ticker, ohlcv, benchmarks.get(ticker))
        if frame is None or frame.empty:
            skipped.append(ticker)          # reason already printed by the callee
            continue

        try:
            with engine.connect() as conn:
                written = _upsert_signals(conn, ticker, frame)
                conn.commit()
        except LabelLossRefused as exc:
            # Not fatal to the run, but never quiet: this is the exact failure
            # mode that erased 22 tickers' labels while reporting success.
            refused.append(ticker)
            print(f"[Signals] REFUSED {exc}")
            continue

        labelled = int(frame["target_excess_return"].notna().sum())
        total += written
        processed.append(ticker)
        print(f"[Signals] {ticker}: {written} rows ({labelled} labelled)")

    print(f"[Signals] Complete. {total} rows written across {len(processed)} "
          f"tickers, {len(skipped)} skipped, {len(refused)} refused.")
    if skipped:
        print(f"[Signals] Skipped: {', '.join(skipped)}")
    if refused:
        print(f"[Signals] REFUSED (labels preserved, signals now stale): "
              f"{', '.join(refused)}")

    return SignalsReport(rows_written=total, processed=processed,
                         skipped=skipped, refused=refused)


def count_labelled_rows(ticker: str | None = None) -> int:
    """
    Number of rows carrying the PRIMARY target. The F6 regression test asserts
    this is non-decreasing across runs.

    Counts target_return since P1. The number it reports therefore JUMPS at the
    switch — upward, and by a lot, because every row that had an excess label
    also had an absolute one and 55 tickers had been blocked from writing any
    label at all by a dead index. A jump is safe for the F6 guard, which only
    ever refuses a DECREASE, but it means a count recorded before the switch is
    not comparable with one recorded after. That is what MODEL_VERSION and
    experiment_runs.data_hash exist to make visible.

    SESSIONS ONLY, SINCE 2026-09-22, for the reason `_labelled_rows_from`
    gives, and measured the hard way: the daily job ABORTED TWICE on the first
    run after the phantom-session fix, because each ticker's first clean
    rewrite drops the four or five rows Yahoo invented on NSE holidays and this
    total then fell, 222,514 -> 222,378. The write-boundary guard excused those
    rows; this one did not, and BOTH jobs read it. Left alone it is a trap
    rather than a one-off: 49 tickers are still refused by the benchmark
    outage, so the day their index returns their phantom rows go with it and
    both jobs would abort again — the weekly one before persisting any
    evaluation. A row on a day NSE did not trade is not a label to protect.
    """
    engine = get_engine()
    where = "target_return IS NOT NULL" + (" AND ticker = :t" if ticker else "")
    params = {"t": ticker} if ticker else {}
    dates = pd.read_sql(text(f"SELECT date FROM signals WHERE {where}"),
                        engine, params=params)
    if dates.empty:
        return 0
    return int(session_calendar().session_mask(dates["date"]).sum())
