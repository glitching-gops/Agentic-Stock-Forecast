"""
tools/run_baselines.py — Score every comparator on identical purged folds.

    python tools/run_baselines.py                     # baselines only
    python tools/run_baselines.py --with-pooled-xgb   # add the pooled tree
    python tools/run_baselines.py --tickers 20 --folds 3
    python tools/run_baselines.py --json out.json

This is the Phase 2 starting line. Everything the phase adds later — a pooled
cross-sectional model, Chronos-2, TimesFM-2.5 — has to be reported in this
table, on these folds, over these rows, or the comparison is not one.

Read the output in this order:

  1. ``zero`` is the floor. It is the random walk in excess-return space.
  2. ``daily_IC`` is the number that matters for a leaderboard, not ``IC``.
     The pooled IC can be inflated by knowing which months were good; the
     daily one is computed within each date and then averaged, so it measures
     only cross-sectional ordering.
  3. ``alpha_t`` comes from non-overlapping rebalances at the 30-session
     horizon. Anything below 2 is not evidence, however large the point
     estimate next to it looks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.baselines import BASELINES, FACTORS, LinearFactorModel   # noqa: E402
from pipeline.evaluation import PurgedPanelWalkForward, panel_walk_forward  # noqa: E402
from pipeline.panel import (                                            # noqa: E402
    MIN_NAMES_PER_DATE,
    SCALE_FREE,
    TARGET,
    cross_sectional_zscore,
    load_panel,
    panel_coverage,
)
from pipeline.signals import HORIZON_SESSIONS                           # noqa: E402


def build_panel(n_tickers: int | None, start: str | None) -> pd.DataFrame:
    """Loads the panel and standardises the scale-free columns within each date."""
    panel = load_panel(start=start)
    if panel.empty:
        return panel

    if n_tickers:
        # Widest histories first, so a truncated run still has folds to split.
        counts = panel.groupby("ticker")["date"].count().sort_values(ascending=False)
        panel = panel[panel["ticker"].isin(counts.head(n_tickers).index)]

    return cross_sectional_zscore(panel, SCALE_FREE)


def run(panel: pd.DataFrame, folds: int, min_train: int,
        with_pooled_xgb: bool) -> list[dict]:
    splitter = PurgedPanelWalkForward(
        n_folds=folds, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=min_train,
    )

    runs: list[tuple[str, callable, list[str]]] = [
        (name, factory, FACTORS) for name, factory in BASELINES.items()
    ]

    if with_pooled_xgb:
        from xgboost import XGBRegressor

        def pooled_xgb():
            # Deliberately untuned. The point of this row is to ask whether the
            # extra capacity beats a ridge on the same columns, not to find the
            # best tree — a tuned tree would need its Sharpe deflating for the
            # search before it could be read beside an unsearched linear model.
            return XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, tree_method="hist",
            )

        runs.append(("pooled_xgb", pooled_xgb, FACTORS))

    results = []
    for name, factory, cols in runs:
        t0 = time.time()
        res = panel_walk_forward(
            panel=panel, feature_cols=cols, model_factory=factory,
            splitter=splitter, name=name, target=TARGET,
            rebalance_every=HORIZON_SESSIONS,
        )
        m, xs = res.metrics, res.cross_sectional
        results.append({
            "name": name,
            "n_oos": res.n_predictions,
            "folds": res.n_folds_run,
            "rank_ic": m.get("rank_ic", float("nan")),
            "daily_rank_ic": m.get("daily_rank_ic", float("nan")),
            "hit_rate": m.get("hit_rate", float("nan")),
            "majority_hit_rate": m.get("majority_hit_rate", float("nan")),
            "mae": m.get("mae", float("nan")),
            "mae_naive_zero": m.get("mae_naive_zero", float("nan")),
            "beats_naive_mae": m.get("beats_naive_mae", False),
            "n_rebalances": xs.get("n_rebalances", 0),
            "n_dates_no_ordering": xs.get("n_dates_no_ordering", 0),
            "rebalance_ic": xs.get("mean_rank_ic", float("nan")),
            "rebalance_ic_t": xs.get("rank_ic_t", float("nan")),
            "alpha_vs_equal_weight": xs.get("alpha_vs_equal_weight", float("nan")),
            "alpha_t": xs.get("alpha_t", float("nan")),
            "long_short_spread": xs.get("long_short_spread", float("nan")),
            "spread_t": xs.get("spread_t", float("nan")),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  {name:<16s} {results[-1]['seconds']:>6.1f}s  "
              f"{res.n_predictions:>7,} OOS rows")

    return results


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


def factor_loadings(panel: pd.DataFrame, min_train: int) -> dict[str, float]:
    """Fits the linear comparator once on the earliest training window, to read."""
    grid = sorted(panel["date"].unique())
    if len(grid) <= min_train:
        return {}
    cut = grid[min_train - HORIZON_SESSIONS * 2]
    train = panel[(panel["date"] < cut) & panel[TARGET].notna()]
    if len(train) < 100:
        return {}
    model = LinearFactorModel().fit(train[FACTORS], train[TARGET])
    return model.coefficients()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", type=int, default=None,
                    help="cap the panel to the N widest histories")
    ap.add_argument("--start", type=str, default=None, help="earliest date")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=500,
                    help="dates before the first test window opens")
    ap.add_argument("--with-pooled-xgb", action="store_true",
                    help="also score an untuned pooled gradient-boosted tree")
    ap.add_argument("--allow-thin", action="store_true",
                    help="run even when the panel is too thin to rank (harness "
                         "smoke test only; the numbers are not results)")
    ap.add_argument("--json", type=str, default=None, help="write results to a file")
    args = ap.parse_args()

    print("Loading panel ...")
    panel = build_panel(args.tickers, args.start)
    if panel.empty:
        print("No signals rows. Run the daily pipeline first.", file=sys.stderr)
        return 1

    cov = panel_coverage(panel)
    print(f"  {cov['rows']:,} rows  |  {cov['tickers']} tickers  |  "
          f"{cov['dates']:,} dates  |  {cov['first_date']} -> {cov['last_date']}")
    print(f"  labelled {cov['labelled_rows']:,}  |  median names/date "
          f"{cov['median_names_per_date']:.0f}  |  "
          f"{cov['dates_with_enough_breadth']:,} dates carry >={MIN_NAMES_PER_DATE} names")

    if cov["median_names_per_date"] < MIN_NAMES_PER_DATE and not args.allow_thin:
        # Not a warning. Below MIN_NAMES_PER_DATE, cross_sectional_zscore zeroes
        # every feature by design and cross_sectional_report declines to rank, so
        # the table that would print here is a grid of zeros and dashes in the
        # exact shape of a real result. Printing it under a caption saying it is
        # unreliable is how a screenshot of it ends up somewhere as evidence.
        print(f"\n  REFUSED: the median date holds "
              f"{cov['median_names_per_date']:.0f} names, below the "
              f"{MIN_NAMES_PER_DATE} needed to rank a cross-section. Every feature "
              f"is zeroed at that breadth, so the table would compare six "
              f"constants.", file=sys.stderr)
        print("           Widen the panel, or pass --allow-thin to smoke-test "
              "the harness itself.", file=sys.stderr)
        return 1

    print("\nScoring comparators on identical purged folds ...")
    results = run(panel, args.folds, args.min_train, args.with_pooled_xgb)

    print(f"\n{render(results)}")
    print("\n  daily_IC - mean of the per-date rank IC. The leaderboard number.")
    print(f"  IC_t     - t-statistic of that IC over "
          f"{HORIZON_SESSIONS}-session non-overlapping rebalances.")
    print("  IC       - pooled across every row. Inflated by market timing AND by "
          "fold identity: a constant-per-fold predictor scores a non-zero pooled")
    print("             IC with no ranking information at all. Trust daily_IC; "
          "read IC only beside it.")
    print(f"  alpha_t  - from {HORIZON_SESSIONS}-session non-overlapping rebalances. "
          f"Below ~2 is not evidence.")

    loadings = factor_loadings(panel, args.min_train)
    if loadings:
        print("\nLinear factor loadings on the first training window "
              "(standardised inputs, so directly comparable):")
        for col, coef in sorted(loadings.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {col:<20s} {coef:+.5f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"coverage": cov, "results": results,
                       "loadings": loadings}, fh, indent=2, default=str)
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
