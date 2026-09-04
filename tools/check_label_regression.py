"""
tools/check_label_regression.py — would the weekly job abort on F6 today?

    python tools/check_label_regression.py
    python tools/check_label_regression.py --tickers RELIANCE.NS ABB.NS

WHY THIS EXISTS
---------------
Both scheduled jobs guard their writes with the F6 monotonicity check: count
labelled rows, recompute signals, count again, and ABORT if the number fell.
That guard is correct and has already earned its place — on 2026-08-17/18 two
daily runs recomputed signals while the sector indices were transiently
unavailable, NULLed the target for every stock mapped to them, and would have
evaluated on the wreckage.

But the guard is also OPAQUE. It fires after the recompute has already run,
reports one number against another, and names a likely cause rather than the
actual one. The weekly job aborted five times on 2026-09-03 and nothing on
record said WHICH ticker regressed or by how much — so the only way to find out
was to run the hour-long job again and watch.

This does the same comparison READ-ONLY. It recomputes every ticker's signals
frame in memory, compares the labelled-row count against what is stored, and
names the offenders. Nothing is written, so it can be run against production at
any time, including while deciding whether the real job is safe to trigger.

WHAT IT CANNOT TELL YOU
-----------------------
A regression that depends on a transient — a benchmark index that failed to
download during THAT run, a database blip mid-write — will not reproduce here
if the transient has passed. A clean report means "the stored data and the
current vendor responses are consistent right now", not "the abort was
spurious". That distinction is the whole reason the guard aborts rather than
warning: it cannot know which it was either, and the cheap mistake is to keep
the old labels.
"""

from __future__ import annotations

import argparse
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import pandas as pd                                              # noqa: E402
from sqlalchemy import text                                      # noqa: E402

from data.db import get_engine                                   # noqa: E402
from pipeline.panel import EXCESS_TARGET, TARGET                 # noqa: E402
from pipeline.signals import compute_signals_frame               # noqa: E402


def stored_counts(engine) -> pd.DataFrame:
    return pd.read_sql(text(f"""
        SELECT ticker,
               COUNT(*)                  AS rows,
               COUNT({TARGET})           AS labelled,
               COUNT({EXCESS_TARGET})    AS labelled_excess,
               MAX(date)                 AS last_date
        FROM signals GROUP BY ticker
    """), engine)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    engine = get_engine()
    stored = stored_counts(engine).set_index("ticker")

    if args.tickers:
        tickers = args.tickers
    else:
        from data.universe import get_universe
        tickers = get_universe()
    if args.limit:
        tickers = tickers[:args.limit]

    print(f"Recomputing {len(tickers)} tickers in memory. Nothing is written.\n")
    print(f"  {'ticker':16s}{'stored':>8}{'recomp':>8}{'delta':>8}"
          f"{'excess d':>10}  note")

    rows, regressed, missing = [], [], []
    for i, ticker in enumerate(tickers, 1):
        ohlcv = pd.read_sql(
            text("SELECT * FROM ohlcv WHERE ticker = :t ORDER BY date ASC"),
            engine, params={"t": ticker})
        if ohlcv.empty:
            missing.append((ticker, "no ohlcv rows"))
            continue

        try:
            frame = compute_signals_frame(ticker, ohlcv)
        except Exception as exc:                                 # noqa: BLE001
            missing.append((ticker, f"{type(exc).__name__}: {exc}"[:80]))
            continue
        if frame is None or frame.empty:
            missing.append((ticker, "compute_signals_frame returned nothing"))
            continue

        was = int(stored.loc[ticker, "labelled"]) if ticker in stored.index else 0
        was_x = int(stored.loc[ticker, "labelled_excess"]) if ticker in stored.index else 0
        now = int(frame[TARGET].notna().sum()) if TARGET in frame else 0
        now_x = int(frame[EXCESS_TARGET].notna().sum()) if EXCESS_TARGET in frame else 0

        rows.append({"ticker": ticker, "stored": was, "recomputed": now,
                     "delta": now - was, "excess_delta": now_x - was_x})
        # BOTH LABELS, because `_upsert_signals` refuses a decrease in either.
        # A ticker whose absolute label is intact while its excess label
        # collapses is exactly the shape a dead benchmark produces, and
        # counting only the primary label would wave it through.
        if now < was or now_x < was_x:
            regressed.append(rows[-1])
            note = "REGRESSED"
        else:
            note = ""
        if note or i <= 5 or i % 20 == 0:
            print(f"  {ticker:16s}{was:>8}{now:>8}{now - was:>+8}"
                  f"{now_x - was_x:>+10}  {note}")

    print()
    if missing:
        print(f"{len(missing)} ticker(s) produced NO frame — a recompute would "
              f"skip them, which preserves their labels:")
        for ticker, why in missing[:10]:
            print(f"    {ticker:16s} {why}")
        print()

    total_was = sum(r["stored"] for r in rows)
    total_now = sum(r["recomputed"] for r in rows)
    print(f"Across {len(rows)} recomputed tickers: {total_was:,} labelled rows "
          f"stored, {total_now:,} recomputed ({total_now - total_was:+,}).")

    if regressed:
        print(f"\nWOULD ABORT. {len(regressed)} ticker(s) lose labelled rows:")
        for r in sorted(regressed, key=lambda r: r["delta"])[:20]:
            print(f"    {r['ticker']:16s} {r['delta']:+6} absolute, "
                  f"{r['excess_delta']:+6} excess")
        print("\n  `_upsert_signals` refuses each of these individually, so the "
              "labels\n  survive — but the job's own count falls and it aborts "
              "before writing\n  any evaluation. Fix the cause before "
              "triggering the weekly run.")
        return 1

    print("\nNo ticker regresses. The F6 guard would not fire on this data.")
    print("  Note this is a statement about RIGHT NOW: a transient vendor "
          "failure\n  during the real run can still trip it, which is precisely "
          "why the guard\n  aborts rather than warns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
