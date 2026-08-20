"""
tools/run_baselines.py — Score every comparator on identical purged folds.

    python tools/run_baselines.py                     # the whole panel
    python tools/run_baselines.py --no-pooled-xgb     # baselines only
    python tools/run_baselines.py --tickers 20 --folds 3
    python tools/run_baselines.py --json out.json

This is the Phase 2 starting line. Everything the phase adds later — a pooled
cross-sectional model, Chronos-2, TimesFM-2.5 — has to be reported in this
table, on these folds, over these rows, or the comparison is not one.

The measurement itself lives in ``pipeline.baselines.compare_baselines``; this
file only renders it. The weekly job calls that same function, so the table it
records into ``experiment_runs`` cannot drift from the table printed here.

Read the output in this order:

  1. ``zero`` is the floor. It is the random walk in excess-return space.
  2. ``daily_IC`` is the number that matters for a leaderboard, not ``IC``.
     The pooled IC can be moved by knowing which months were good and by
     knowing which fold a row came from; the daily one is computed within each
     date and then averaged, so it measures only cross-sectional ordering.
  3. ``alpha_t`` comes from non-overlapping rebalances at the 30-session
     horizon. Anything below 2 is not evidence, however large the point
     estimate next to it looks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.baselines import compare_baselines          # noqa: E402
from pipeline.signals import HORIZON_SESSIONS             # noqa: E402


def render(results: list[dict]) -> str:
    def f(v, spec=".4f"):
        return "     -" if v is None or not np.isfinite(v) else format(v, spec)

    head = (f"{'comparator':<16s} {'daily_IC':>9s} {'IC_t':>7s} {'pooledIC':>9s} "
            f"{'hit%':>7s} {'maj%':>7s} {'MAE':>8s} {'<naive':>7s} "
            f"{'alpha':>9s} {'alpha_t':>8s} {'L-S':>9s} {'n_reb':>6s}")
    lines = [head, "-" * len(head)]

    for r in results:
        lines.append(
            f"{r['name']:<16s} "
            f"{f(r['daily_rank_ic'], '+.4f'):>9s} "
            f"{f(r['rebalance_ic_t'], '+.2f'):>7s} "
            f"{f(r['rank_ic'], '+.4f'):>9s} "
            f"{f(r['hit_rate'], '.2f'):>7s} "
            f"{f(r['majority_hit_rate'], '.2f'):>7s} "
            f"{f(r['mae'], '.5f'):>8s} "
            f"{str(bool(r['beats_naive_mae'])):>7s} "
            f"{f(r['alpha_vs_equal_weight'], '+.5f'):>9s} "
            f"{f(r['alpha_t'], '+.2f'):>8s} "
            f"{f(r['long_short_spread'], '+.5f'):>9s} "
            f"{r['n_rebalances']:>6d}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", type=int, default=None,
                    help="cap the panel to the N widest histories")
    ap.add_argument("--start", type=str, default=None, help="earliest date")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=500,
                    help="dates before the first test window opens")
    ap.add_argument("--no-pooled-xgb", action="store_true",
                    help="skip the untuned pooled gradient-boosted tree")
    ap.add_argument("--allow-thin", action="store_true",
                    help="run even when the panel is too thin to rank (harness "
                         "smoke test only; the numbers are not results)")
    ap.add_argument("--json", type=str, default=None, help="write results to a file")
    args = ap.parse_args()

    print("Loading panel and scoring comparators ...")
    comparison = compare_baselines(
        start=args.start,
        n_folds=args.folds,
        min_train=args.min_train,
        with_pooled_xgb=not args.no_pooled_xgb,
        max_tickers=args.tickers,
        allow_thin=args.allow_thin,
    )

    cov = comparison.coverage
    if cov:
        print(f"  {cov['rows']:,} rows  |  {cov['tickers']} tickers  |  "
              f"{cov['dates']:,} dates  |  {cov['first_date']} -> {cov['last_date']}")
        print(f"  labelled {cov['labelled_rows']:,}  |  median names/date "
              f"{cov['median_names_per_date']:.0f}")

    if not comparison.ranked:
        # A refusal, not a warning. Below the breadth threshold every feature is
        # zeroed by design and no date can be ranked, so the table that would
        # print here is a grid of zeros and dashes in the exact shape of a real
        # result. Printing it under a caption saying it is unreliable is how a
        # screenshot of it ends up somewhere as evidence.
        print(f"\n  REFUSED: {comparison.note}", file=sys.stderr)
        return 1

    if comparison.note:
        # A comparator that was skipped must say so. Its absence from the table
        # is otherwise indistinguishable from it never having existed.
        print(f"\n  NOTE: {comparison.note}")

    print(f"\n{render(comparison.results)}")
    print("\n  daily_IC - mean of the per-date rank IC. The leaderboard number.")
    print(f"  IC_t     - t-statistic of that IC over {HORIZON_SESSIONS}-session "
          f"non-overlapping rebalances.")
    print("  pooledIC - correlated across every row at once. Moved by market "
          "timing AND by fold")
    print("             identity: a constant-per-fold predictor scores a "
          "non-zero pooled IC with")
    print("             no ranking information at all. Trust daily_IC.")
    print(f"  alpha_t  - from {HORIZON_SESSIONS}-session non-overlapping "
          f"rebalances. Below ~2 is not evidence.")

    if comparison.loadings:
        print("\nLinear factor loadings on the first training window "
              "(standardised inputs, so directly comparable):")
        for col, coef in sorted(comparison.loadings.items(),
                                key=lambda kv: -abs(kv[1])):
            print(f"  {col:<20s} {coef:+.5f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"coverage": cov, "results": comparison.results,
                       "loadings": comparison.loadings}, fh, indent=2, default=str)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
