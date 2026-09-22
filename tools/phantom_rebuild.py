"""
tools/phantom_rebuild.py — rebuild the panel with and without the NSE calendar,
from ONE snapshot, so the difference is the calendar and nothing else.

WHY A CONTROLLED REBUILD, NOT A FRESH `load_panel()`. The stored panel was built
on 2026-09-07. Rebuilding it today from the database would also pick up a
fortnight of new sessions, every dividend Yahoo has back-adjusted into
`adj_close` since, and whatever the sector-index outage did to the benchmark
series - and every one of those would be read as the phantom fix. So:

  1. ohlcv is read ONCE, read-only, up to the stored panel's last date.
  2. `compute_signals_frame` - the production code, not a copy - runs on it
     twice: CONTROL on the rows as stored, CLEAN with the dates NSE did not
     trade removed by `data.nse_calendar.drop_non_sessions`, the same call
     `pipeline/fetch.py` now makes.
  3. The two inputs that come off the network are FROZEN to what the stored
     panel used: each ticker's benchmark series and its earnings surprise.
  4. Every float goes through the float32 round trip the `signals` table's
     REAL columns impose, as `load_panel` would have read it.

The CONTROL is measured against the stored panel first. Where it matches, the
rebuild is faithful and CLEAN minus CONTROL is the fix alone; where it does not,
the report says how far and in which columns, before anything is scored.

    python tools/phantom_rebuild.py --out-dir .      # writes two parquet files
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pipeline.signals as signals  # noqa: E402
from data import nse_calendar  # noqa: E402
from pipeline.panel import (EXCESS_TARGET, FEATURE_COLS, FEATURES, MACRO_COLS,  # noqa: E402
                            TARGET, TARGETS)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
PANEL_COLS = ["date", "ticker", "close", *FEATURE_COLS, TARGET, EXCESS_TARGET,
              "benchmark_close", "benchmark_ticker"]


def db_real(values) -> np.ndarray:
    """What a float64 becomes after a Postgres REAL column and psycopg2: the
    float32 nearest it, printed shortest-round-trip, parsed back as a double."""
    a = np.asarray(values, dtype=np.float64)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    out[ok] = [float(str(v)) for v in a[ok].astype(np.float32)]
    return out


def read_ohlcv(tickers: list[str], through: str) -> dict[str, pd.DataFrame]:
    from sqlalchemy import text

    from data.db import get_engine

    out = {}
    with get_engine().connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        for t in tickers:
            out[t] = pd.read_sql(
                text("SELECT * FROM ohlcv WHERE ticker = :t AND date <= :d "
                     "ORDER BY date ASC"), conn, params={"t": t, "d": through})
        conn.rollback()
    return out


def freeze_network_inputs(stored: pd.DataFrame) -> None:
    """Point signals' three network calls at what the stored panel used."""
    bench_of = stored.groupby("ticker")["benchmark_ticker"].first().to_dict()
    series = {b: g.drop_duplicates("date").sort_values("date")[["date", "benchmark_close"]]
              .reset_index(drop=True)
              for b, g in stored.groupby("benchmark_ticker")}
    surprise = stored.set_index(["ticker", "date"])["earnings_surprise"]

    def get_benchmark(ticker):
        return bench_of[ticker], bench_of[ticker] != "^NSEI"

    def get_benchmark_series(index_ticker, period="10y"):
        return series[index_ticker].copy()

    def compute_earnings_surprise(ticker, df):
        s = surprise.xs(ticker, level="ticker")
        df["earnings_surprise"] = df["date"].map(s).ffill().fillna(0.0)
        return df

    signals.get_benchmark = get_benchmark
    signals.get_benchmark_series = get_benchmark_series
    signals.compute_earnings_surprise = compute_earnings_surprise


def as_panel(frames: list[pd.DataFrame], stored: pd.DataFrame) -> pd.DataFrame:
    """The frames as `load_panel` would have returned them from the table."""
    p = pd.concat(frames, ignore_index=True)[PANEL_COLS].copy()
    for c in PANEL_COLS:
        if c not in ("date", "ticker", "benchmark_ticker"):
            p[c] = db_real(p[c])
    macro = stored.drop_duplicates("date")[["date", *MACRO_COLS]]
    p = p.merge(macro, on="date", how="left")
    for col in FEATURES:
        p[col] = pd.to_numeric(p[col], errors="coerce") \
            .replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in TARGETS:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p[p["date"] <= stored["date"].max()]
    return p[stored.columns].sort_values(["date", "ticker"]).reset_index(drop=True)


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Row and per-column agreement of two panels on (date, ticker)."""
    m = a.merge(b, on=["date", "ticker"], how="outer", suffixes=("_a", "_b"),
                indicator=True)
    both = m[m["_merge"] == "both"]
    cols = {}
    for c in a.columns:
        if c in ("date", "ticker") or a[c].dtype == object or str(a[c].dtype) == "str":
            continue
        x, y = both[f"{c}_a"].to_numpy(float), both[f"{c}_b"].to_numpy(float)
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        d = np.abs(x - y)
        cols[c] = {"identical_share": float(same.mean()),
                   "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0}
    return {"rows_a": len(a), "rows_b": len(b),
            "only_a": int((m["_merge"] == "left_only").sum()),
            "only_b": int((m["_merge"] == "right_only").sum()),
            "only_a_dates": sorted(m.loc[m["_merge"] == "left_only", "date"].unique())[:12],
            "columns": cols}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--out-dir", default=ROOT)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    stored = pd.read_parquet(args.panel_cache)
    stored["date"] = stored["date"].astype(str)
    through = stored["date"].max()
    tickers = sorted(stored["ticker"].unique())
    calendar = nse_calendar.load()
    print(f"stored panel {len(stored):,} rows, {len(tickers)} tickers, through {through}",
          flush=True)

    ohlcv = read_ohlcv(tickers, through)
    freeze_network_inputs(stored)

    ctrl, clean, removed = [], [], {}
    for t in tickers:
        raw = ohlcv[t]
        kept, gone = nse_calendar.drop_non_sessions(raw, calendar)
        if gone:
            removed[t] = gone
        for bucket, frame in ((ctrl, raw), (clean, kept)):
            f = signals.compute_signals_frame(t, frame.copy())
            if f is not None:
                bucket.append(f)
    print(f"signals recomputed for {len(ctrl)} tickers ({time.time() - t0:.0f}s)", flush=True)

    p_ctrl, p_clean = as_panel(ctrl, stored), as_panel(clean, stored)
    report = {
        "through": through,
        "non_session_dates_removed": sorted({d for v in removed.values() for d in v}),
        "tickers_with_removed_rows": len(removed),
        "control_vs_stored": compare(stored, p_ctrl),
        "clean_vs_control": compare(p_ctrl, p_clean),
    }
    for name, frame in (("panel_cache_control.parquet", p_ctrl),
                        ("panel_cache_clean.parquet", p_clean)):
        frame.to_parquet(os.path.join(args.out_dir, name))
    cs = report["control_vs_stored"]
    print(f"CONTROL vs STORED: rows {cs['rows_a']:,}/{cs['rows_b']:,}, unmatched "
          f"{cs['only_a']}/{cs['only_b']}", flush=True)
    for c, v in cs["columns"].items():
        if v["identical_share"] < 1.0:
            print(f"  {c:22s} identical {v['identical_share']:.4%}  max diff {v['max_abs_diff']:.3e}")
    cc = report["clean_vs_control"]
    print(f"CLEAN vs CONTROL: rows {cc['rows_a']:,} -> {cc['rows_b']:,}; removed dates "
          f"{report['non_session_dates_removed']}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
