"""
tools/run_baselines.py — Score every comparator on identical purged folds.

    python tools/run_baselines.py                     # the whole panel
    python tools/run_baselines.py --no-pooled-xgb     # baselines only
    python tools/run_baselines.py --tickers 20 --folds 3
    python tools/run_baselines.py --json out.json
    python tools/run_baselines.py --chronos                # slow: needs torch
    python tools/run_baselines.py --timesfm                # slower: 200M params

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
import logging
import os
import sys

import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.baselines import compare_baselines          # noqa: E402
from pipeline.series import DEFAULT_CONTEXT as SERIES_CONTEXT  # noqa: E402
from pipeline.signals import HORIZON_SESSIONS             # noqa: E402


def render(results: list[dict]) -> str:
    def f(v, spec=".4f"):
        return "     -" if v is None or not np.isfinite(v) else format(v, spec)

    head = (f"{'comparator':<18s} {'daily_IC':>9s} {'reb_IC':>8s} {'reb_t':>7s} {'pooledIC':>9s} "
            f"{'hit%':>7s} {'maj%':>7s} {'MAE':>8s} {'<naive':>7s} "
            f"{'alpha':>9s} {'alpha_t':>8s} {'L-S':>9s} {'n_reb':>6s} {'secs':>7s}")
    lines = [head, "-" * len(head)]

    for r in results:
        lines.append(
            f"{r['name']:<18s} "
            f"{f(r['daily_rank_ic'], '+.4f'):>9s} "
            f"{f(r['rebalance_ic'], '+.4f'):>8s} "
            f"{f(r['rebalance_ic_t'], '+.2f'):>7s} "
            f"{f(r['rank_ic'], '+.4f'):>9s} "
            f"{f(r['hit_rate'], '.2f'):>7s} "
            f"{f(r['majority_hit_rate'], '.2f'):>7s} "
            f"{f(r['mae'], '.5f'):>8s} "
            f"{str(bool(r['beats_naive_mae'])):>7s} "
            f"{f(r['alpha_vs_equal_weight'], '+.5f'):>9s} "
            f"{f(r['alpha_t'], '+.2f'):>8s} "
            f"{f(r['long_short_spread'], '+.5f'):>9s} "
            f"{r['n_rebalances']:>6d} "
            f"{r.get('seconds', 0.0):>7.1f}"
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
    ap.add_argument("--chronos", action="store_true",
                    help="also score Chronos-2 (needs requirements-series.txt; "
                         "roughly 1.7 s per date per 46 tickers at context "
                         "2048, so budget ~1 hour for a full panel)")
    ap.add_argument("--chronos-context", type=int, default=SERIES_CONTEXT,
                    help=f"trailing observations handed to Chronos-2 "
                         f"(default {SERIES_CONTEXT}; cost is linear in this)")
    ap.add_argument("--timesfm", action="store_true",
                    help="also score TimesFM-2.5 (200M decoder-only, a "
                         "different architecture from Chronos rather than "
                         "a different size; needs requirements-series.txt)")
    ap.add_argument("--timesfm-context", type=int, default=SERIES_CONTEXT,
                    help=f"trailing observations handed to TimesFM-2.5 "
                         f"(default {SERIES_CONTEXT})")
    ap.add_argument("--record", action="store_true",
                    help="open an experiment_runs row and store the table in "
                         "it, beside the config_hash and data_hash that say "
                         "whether a movement came from code or from data")
    ap.add_argument("--json", type=str, default=None, help="write results to a file")
    args = ap.parse_args()

    # Without a handler the per-comparator progress goes nowhere, and a Chronos
    # run is ~50 minutes of total silence on a workflow runner — which is
    # indistinguishable from a hung step at the point somebody decides to
    # cancel it.
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout, force=True)

    run_id = None
    if args.record:
        # A measurement nobody can find later is not evidence. The weekly job
        # records its comparison the same way; a Chronos run that only ever
        # existed in a workflow log could not be read beside it.
        from pipeline.tracking import finish_run, start_run
        run_id = start_run("series_comparison")
        print(f"  experiment_runs row {run_id}")

    # Persist after every comparator, not just at the end. A killed
    # two-hour run previously discarded twelve finished comparators
    # because the results only left compare_baselines via its return
    # value. Written to a .partial file so a crash mid-write cannot
    # corrupt a completed table.
    partial_path = (args.json + ".partial") if args.json else None

    def _persist(results, coverage):
        if not partial_path:
            return
        with open(partial_path, "w", encoding="utf-8") as fh:
            json.dump({"coverage": coverage, "results": results,
                       "complete": False}, fh, indent=2, default=str)
        print(f"    ... {len(results)} comparator(s) saved to "
              f"{partial_path}", flush=True)

    print("Loading panel and scoring comparators ...")
    if args.chronos:
        print(f"  Chronos-2 enabled at context {args.chronos_context}.")
    if args.timesfm:
        print(f"  TimesFM-2.5 enabled at context {args.timesfm_context}.")
    if args.chronos or args.timesfm:
        print("  These are the slow paths; the `secs` column reports what "
              "each cost.")
    comparison = compare_baselines(
        start=args.start,
        n_folds=args.folds,
        min_train=args.min_train,
        with_pooled_xgb=not args.no_pooled_xgb,
        with_chronos=args.chronos,
        chronos_context=args.chronos_context,
        with_timesfm=args.timesfm,
        timesfm_context=args.timesfm_context,
        max_tickers=args.tickers,
        allow_thin=args.allow_thin,
        on_result=_persist,
    )

    cov = comparison.coverage
    if cov:
        print(f"  {cov['rows']:,} rows  |  {cov['tickers']} tickers  |  "
              f"{cov['dates']:,} dates  |  {cov['first_date']} -> {cov['last_date']}")
        print(f"  labelled {cov['labelled_rows']:,}  |  median names/date "
              f"{cov['median_names_per_date']:.0f}")

    def _record(status: str) -> None:
        if run_id is None:
            return
        finish_run(run_id, status, metrics={
            "baselines": comparison.to_metrics(),
            # The Chronos settings are NOT inside config_hash, which covers
            # features, target, horizon, eval params and the benchmark mapping.
            # Without them two runs at different contexts would be
            # indistinguishable in experiment_runs — precisely the confusion
            # config_hash exists to prevent everywhere else.
            "series_config": {
                "chronos": bool(args.chronos),
                "chronos_context": args.chronos_context if args.chronos else None,
                "timesfm": bool(args.timesfm),
                "timesfm_context": args.timesfm_context if args.timesfm else None,
                "folds": args.folds,
                "min_train": args.min_train,
                "max_tickers": args.tickers,
                "pooled_xgb": not args.no_pooled_xgb,
            },
        })

    if not comparison.ranked:
        # A refusal, not a warning. Below the breadth threshold every feature is
        # zeroed by design and no date can be ranked, so the table that would
        # print here is a grid of zeros and dashes in the exact shape of a real
        # result. Printing it under a caption saying it is unreliable is how a
        # screenshot of it ends up somewhere as evidence.
        print(f"\n  REFUSED: {comparison.note}", file=sys.stderr)
        _record("REFUSED")
        return 1

    if comparison.note:
        # A comparator that was skipped must say so. Its absence from the table
        # is otherwise indistinguishable from it never having existed.
        print(f"\n  NOTE: {comparison.note}")

    print(f"\n{render(comparison.results)}")
    print("\n  daily_IC - mean per-date rank IC over EVERY out-of-sample date.")
    print("             The point estimate of ordering ability. It carries no")
    print(f"             t-statistic here: consecutive dates share "
          f"{HORIZON_SESSIONS - 1} of their {HORIZON_SESSIONS} sessions, so")
    print("             ~1,900 dates hold only ~60 independent windows "
          "and a naive t is inflated ~5x.")
    print(f"  reb_IC   - the same mean over {HORIZON_SESSIONS}-session NON-OVERLAPPING "
          f"rebalance dates only,")
    print("             and reb_t is its t-statistic. A DIFFERENT sample from")
    print("             daily_IC - the two can and do carry opposite signs.")
    print("             reb_t is the one that supports inference.")
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
                       "loadings": comparison.loadings,
                       "complete": True}, fh, indent=2, default=str)
        print(f"\nWrote {args.json}")

    _record("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
