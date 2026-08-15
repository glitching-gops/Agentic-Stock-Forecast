"""
pipeline/fetch.py — OHLCV ingestion.

Previously this fetched a rolling 2-year window with ``auto_adjust=True`` and
appended only dates it had not seen before. That produced a corrupted series
(audit finding F11): rows written before a split or dividend kept the old
adjustment basis while later rows used the new one, so the stored close price
jumped discontinuously at every corporate action. A 1:5 split introduced an
artificial 80% gap, poisoning both the 30-day target spanning that date and
every indicator for a month either side.

Two changes fix it:

  1. Store BOTH raw and adjusted prices (``auto_adjust=False``). Raw OHLCV is
     immutable history; ``adj_close`` is a derived view that legitimately
     changes when a corporate action occurs.

  2. Overwrite the full window on every run instead of appending unseen dates,
     so the entire stored series always shares one adjustment basis.

``detect_adjustment_breaks()`` reports where the ratio between raw and adjusted
close changes, which the validation gate uses to catch corporate actions that
arrived mid-window.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from data.db import get_engine

PERIOD   = "10y"          # was "2y" — the 30-day target with ~250 usable rows
INTERVAL = "1d"           # per stock left roughly 13 independent observations
BATCH    = 20

OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def _normalise(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flattens a yfinance frame to the stored OHLCV schema."""
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower().replace(" ", "_") for c in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    if "adj_close" not in df.columns:
        # auto_adjust=True was applied upstream, so close is already adjusted.
        df["adj_close"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["date"] = df["date"].astype(str).str[:10]
    df["ticker"] = ticker

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: yfinance frame missing {missing}")

    out = df[OHLCV_COLUMNS].dropna(subset=["close", "adj_close"])
    return out[out["close"] > 0].reset_index(drop=True)


def _download_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Downloads a batch of tickers in one call, returning per-ticker frames."""
    raw = yf.download(
        tickers, period=PERIOD, interval=INTERVAL,
        auto_adjust=False, progress=False, group_by="ticker", threads=True,
    )
    if raw is None or raw.empty:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            sub = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
            if sub is None or sub.dropna(how="all").empty:
                continue
            frames[ticker] = _normalise(sub.dropna(how="all").copy(), ticker)
        except (KeyError, ValueError) as exc:
            print(f"[Fetch] {ticker}: {exc}")
    return frames


def _replace_ticker_history(conn, ticker: str, df: pd.DataFrame) -> int:
    """
    Replaces a ticker's stored history with a freshly fetched window.

    Deleting and reinserting is what keeps the adjustment basis consistent —
    an append-only write is exactly the bug being fixed here. Dates older than
    the fetched window are preserved, so history accumulated over time is not
    lost when the rolling window moves forward.
    """
    if df.empty:
        return 0

    conn.execute(
        text("DELETE FROM ohlcv WHERE ticker = :t AND date >= :start"),
        {"t": ticker, "start": df["date"].min()},
    )
    df.to_sql("ohlcv", con=conn, if_exists="append", index=False)
    return len(df)


def fetch_and_store(single_ticker: str | None = None, tickers: list[str] | None = None) -> int:
    """
    Fetches OHLCV for the given tickers and overwrites the stored window.

    Args:
        single_ticker: fetch one ticker (kept for the per-stock agent path).
        tickers:       explicit list. When both are omitted, the caller must
                       supply a universe — this function no longer reaches for
                       a hard-coded ticker list.
    """
    engine = get_engine()

    if single_ticker:
        to_fetch = [single_ticker]
    elif tickers:
        to_fetch = list(tickers)
    else:
        from data.universe import get_universe
        to_fetch = get_universe()
        if not to_fetch:
            print("[Fetch] Universe is empty — run data.universe.sync_current_membership() first.")
            return 0

    total = 0
    for i in range(0, len(to_fetch), BATCH):
        batch = to_fetch[i:i + BATCH]
        print(f"[Fetch] Downloading {i + 1}-{i + len(batch)} of {len(to_fetch)}...")
        frames = _download_batch(batch)

        with engine.connect() as conn:
            for ticker, df in frames.items():
                try:
                    total += _replace_ticker_history(conn, ticker, df)
                except Exception as exc:                      # noqa: BLE001
                    print(f"[Fetch] {ticker}: write failed — {exc}")
            conn.commit()

        missing = set(batch) - set(frames)
        if missing:
            print(f"[Fetch] no data returned for: {', '.join(sorted(missing))}")

    print(f"[Fetch] Complete. {total} rows written across {len(to_fetch)} tickers.")
    return total


def detect_adjustment_breaks(ticker: str, tolerance: float = 0.01) -> pd.DataFrame:
    """
    Returns dates where the raw/adjusted close ratio shifts by more than
    ``tolerance``, i.e. where a corporate action took effect.

    Used by the validation gate: a break inside the training window means the
    stored series must be re-fetched rather than appended to.
    """
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT date, close, adj_close FROM ohlcv WHERE ticker = :t ORDER BY date ASC"),
        engine, params={"t": ticker},
    )
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=["date", "ratio", "prev_ratio", "shift"])

    df["ratio"] = df["adj_close"].astype(float) / df["close"].astype(float)
    df["prev_ratio"] = df["ratio"].shift(1)
    df["shift"] = (df["ratio"] / df["prev_ratio"] - 1).abs()

    breaks = df[df["shift"] > tolerance].dropna(subset=["shift"])
    return breaks[["date", "ratio", "prev_ratio", "shift"]].reset_index(drop=True)
