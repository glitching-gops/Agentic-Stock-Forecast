"""
tools/score_news_feature.py — the news comparator and its pre-registered control.

    python tools/score_news_feature.py --coverage        # read this FIRST
    python tools/score_news_feature.py
    python tools/score_news_feature.py --min-train 380 --min-train 500 --min-train 620

THE BAR, FIXED BEFORE THE NUMBER WAS SEEN (2026-09-03, the user's call)
------------------------------------------------------------------------
A news comparator succeeds if BOTH hold:

  1. THE PHASE 2 BAR. Rebalance IC positive with t > 2, and `clears_floor` —
     beats `market` on MAE and `beta_market` on rebalance IC.

  2. THE LATE-FOLD CONTROL. The same comparator, on the IDENTICAL rows, with
     the news columns removed, must NOT score similarly. If plain FACTORS
     reproduces the result on those rows, the effect is the PERIOD, not the
     news.

Why (2) exists, and why it had to be fixed in advance. News coverage grows with
time: measured over the first backfilled tickers, ~0.89 articles per
ticker-month in 2016 against ~6.87 in 2024. So a news feature is effectively
live only in the late folds, and ANY positive result there is confounded with
"the recent period is easier" — a confound this project has already been fooled
by three times, in the opposite direction. Valuation at +3.32, LoRA at +2.37 and
pooled_xgb at +2.42 were every one of them carried by the EARLIEST fold.

The valuation post-mortem is the direct ancestor of this control: adding filing
lag decayed the edge monotonically and looked like clean evidence of a freshness
effect, until it turned out the sample had quietly shrunk from 77,585 rows to
54,304 and the edge was absent at every lag once the rows were held fixed. Hold
the sample fixed BEFORE reading any trend as signal.

AND THE SETTING IS SWEPT, because a result at one cell is not a result. The
valuation finding scored +3.32 at `min_train=380` and +1.00 at the harness
default of 500 on identical rows.
"""

from __future__ import annotations

import argparse
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from pipeline.baselines import (                                 # noqa: E402
    FACTORS, FLOORS, LinearFactorModel, NewsAugmentedFactor, compare_baselines,
)
from pipeline.evaluation import (                                # noqa: E402
    PurgedPanelWalkForward, cross_sectional_report, panel_walk_forward,
)
from pipeline.panel import (                                     # noqa: E402
    MIN_NAMES_PER_DATE, SCALE_FREE, TARGET, attach_news, cross_sectional_zscore,
    load_panel,
)
from pipeline.signals import HORIZON_SESSIONS                    # noqa: E402

#: Below this share of rows searched, the comparison is refused rather than
#: reported. Not a tuning knob: a ridge fitted where most of the news columns
#: are filled zeros restates `linear_factor`, and its null would be quoted as a
#: verdict on news rather than on coverage.
MIN_COVERAGE = 0.60


def _prepare(engine=None, start=None):
    """
    The panel the comparison runs on, prepared EXACTLY as compare_baselines does.

    COVERAGE IS CAPTURED BEFORE THE Z-SCORE, and the first version of this
    function did not do that. `cross_sectional_zscore` fills NaN with 0.0 — by
    design, since the model needs a number and `news_observed` carries the
    missing-ness — so asking "how many rows carry news" afterwards reports
    100% on a panel where seven tickers of eighty-four have any news at all.

    That is this phase's own defect turned on its diagnostic: a filled zero
    read back as a measurement. `news_observed` is the only honest coverage
    signal once standardisation has run, so it is computed first and returned
    alongside.
    """
    from pipeline.news_features import NEWS_COLS

    panel = load_panel(engine=engine, start=start)
    if panel.empty:
        raise SystemExit("no signals rows")
    panel = attach_news(panel, engine)
    panel["news_observed"] = panel["news_count_excess"].notna().astype(float)
    panel["news_has_sentiment"] = panel["news_sent_mean"].notna().astype(float)

    # ANY SCRIPT THAT ATTACKS THE TABLE MUST RUN THE TABLE'S OWN PREPROCESSING.
    # A skeptic sweep that skipped this once read pooled_xgb at +0.99 against a
    # recorded +2.42 and looked like it had overturned the result; the cause was
    # the missing z-score, and beta_market being identical across both runs is
    # what located it.
    panel = cross_sectional_zscore(panel, SCALE_FREE + list(NEWS_COLS))
    return panel, list(NEWS_COLS)


def _score(panel, columns, min_train, n_folds=5):
    """Fits a ridge on `columns` through the shared purged harness."""
    splitter = PurgedPanelWalkForward(
        n_folds=n_folds, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=min_train)
    preds = panel_walk_forward(
        panel, splitter,
        lambda: LinearFactorModel(columns=list(columns)),
        feature_cols=list(columns), target=TARGET)
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-train", type=int, action="append")
    ap.add_argument("--start")
    ap.add_argument("--coverage", action="store_true",
                    help="print per-fold news coverage and exit")
    ap.add_argument("--force", action="store_true",
                    help="score below the coverage floor — wiring checks only")
    args = ap.parse_args()

    settings = args.min_train or [380, 500, 620]
    panel, news_cols = _prepare(start=args.start)

    with_news = list(FACTORS) + news_cols + ["news_observed"]
    without = list(FACTORS)

    print("=" * 78)
    print("NEWS COVERAGE — read this before any result below")
    print("=" * 78)
    # `news_observed` and NOT `news_sent_mean`: the z-score has already filled
    # the latter with 0.0, so reading it here reports 100% on a panel where
    # seven tickers of eighty-four carry any news at all.
    covered = panel["news_observed"] > 0
    scored = panel["news_has_sentiment"] > 0
    print(f"  {int(covered.sum()):,} of {len(panel):,} rows were SEARCHED "
          f"({covered.mean():.1%})")
    print(f"  {int(scored.sum()):,} of {len(panel):,} carry a sentiment "
          f"reading ({scored.mean():.1%})")
    print(f"  {panel['ticker'].nunique()} tickers in the panel, "
          f"{panel.loc[covered, 'ticker'].nunique()} with any coverage")
    by_year = (panel.assign(year=panel["date"].str.slice(0, 4))
               .groupby("year")
               .agg(rows=("news_observed", "size"),
                    searched=("news_observed", "sum"),
                    with_sentiment=("news_has_sentiment", "sum")))
    by_year["coverage"] = by_year["searched"] / by_year["rows"]
    print(by_year.to_string())
    print("\n  Coverage RISING with time is the confound the control below "
          "exists for.\n  A result concentrated where coverage is high is "
          "indistinguishable from\n  'the recent period is easier' until the "
          "same rows are scored without news.")
    if args.coverage:
        return 0

    # REFUSED BELOW A COVERAGE FLOOR, rather than printed with a caveat.
    #
    # A ridge fitted where 96% of the news columns are filled zeros is a ridge
    # fitted on FACTORS with four constant columns attached; its reb_t is a
    # restatement of `linear_factor` and would be read as "news does not help"
    # when the truth is "news was barely present". That is the same error shape
    # as the `linear_factor+val` row that came out identical to five decimals
    # and read as a verdict on valuation rather than on the wiring.
    #
    # A caveat printed above a number does not survive being quoted. The number
    # is withheld.
    if covered.mean() < MIN_COVERAGE and not args.force:
        print(f"\nREFUSING TO SCORE: {covered.mean():.1%} of rows were "
              f"searched, below the {MIN_COVERAGE:.0%} floor.\n"
              f"  {panel.loc[covered, 'ticker'].nunique()} of "
              f"{panel['ticker'].nunique()} tickers carry any news, so a "
              f"comparator fitted here\n  would be `linear_factor` with four "
              f"mostly-constant columns bolted on — and its\n  null would be "
              f"read as a verdict on news rather than on coverage.\n\n"
              f"  Finish the backfill first:\n"
              f"    python tools/backfill_news.py --start 2016-09-01\n"
              f"    python -c \"from pipeline.news_scoring import "
              f"score_unscored; print(score_unscored().summary())\"\n\n"
              f"  --force scores it anyway, for wiring checks only.")
        return 1

    print()
    print("=" * 78)
    print("1. THE PHASE 2 BAR, swept — a result at one cell is not a result")
    print("=" * 78)
    print(f"  {'min_train':>10} {'news reb_IC':>13} {'t':>7} "
          f"{'FACTORS reb_IC':>16} {'t':>7} {'delta t':>9}")

    rows = []
    for mt in settings:
        a = _score(panel, with_news, mt)
        b = _score(panel, without, mt)
        xa = cross_sectional_report(a, rebalance_every=HORIZON_SESSIONS)
        xb = cross_sectional_report(b, rebalance_every=HORIZON_SESSIONS)
        rows.append({"min_train": mt,
                     "news_ic": xa.get("mean_rank_ic", np.nan),
                     "news_t": xa.get("rank_ic_t", np.nan),
                     "base_ic": xb.get("mean_rank_ic", np.nan),
                     "base_t": xb.get("rank_ic_t", np.nan)})
        r = rows[-1]
        print(f"  {mt:>10} {r['news_ic']:>+13.4f} {r['news_t']:>+7.2f} "
              f"{r['base_ic']:>+16.4f} {r['base_t']:>+7.2f} "
              f"{r['news_t'] - r['base_t']:>+9.2f}")

    print()
    print("=" * 78)
    print("2. THE LATE-FOLD CONTROL — identical rows, news columns removed")
    print("=" * 78)
    default = settings[len(settings) // 2]
    a = _score(panel, with_news, default)
    b = _score(panel, without, default)

    merged = a[["date", "ticker", "fold", "y_true", "y_pred"]].merge(
        b[["date", "ticker", "y_pred"]], on=["date", "ticker"],
        suffixes=("_news", "_base"))
    print(f"  scored on {len(merged):,} identical rows at min_train={default}")
    print(f"  {'fold':>5} {'rows':>8} {'news cov':>9} "
          f"{'news reb_IC':>13} {'FACTORS reb_IC':>16} {'delta':>9}")

    for fold, grp in merged.groupby("fold"):
        cov = panel.merge(grp[["date", "ticker"]], on=["date", "ticker"])
        coverage = float(cov["news_sent_mean"].notna().mean()) if len(cov) else np.nan
        ic_a = cross_sectional_report(
            grp.rename(columns={"y_pred_news": "y_pred"}),
            rebalance_every=HORIZON_SESSIONS).get("mean_rank_ic", np.nan)
        ic_b = cross_sectional_report(
            grp.rename(columns={"y_pred_base": "y_pred"}),
            rebalance_every=HORIZON_SESSIONS).get("mean_rank_ic", np.nan)
        print(f"  {fold:>5} {len(grp):>8,} {coverage:>8.0%} "
              f"{ic_a:>+13.4f} {ic_b:>+16.4f} {ic_a - ic_b:>+9.4f}")

    print()
    print("  VERDICT RULE: news succeeds only if its reb_t clears 2 AND it")
    print("  clears both floors AND the delta above is positive in the folds")
    print("  where coverage is high. A delta near zero means the ridge is")
    print("  reading the period, not the news.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
