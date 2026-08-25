"""
tools/score_lora.py - Score Kaggle LoRA predictions through the local harness.

    python tools/score_lora.py --predictions lora_predictions.npz

WHY THE SCORING COMES HOME
--------------------------
The notebook produces predictions and nothing else. Every statistic - the
per-date rank IC, the non-overlapping rebalance IC and its t, the MAE against
the `zero` floor - is computed here, by `cross_sectional_report`, the same
function that produced every other row of the results table.

That is the whole reason the split is drawn where it is. A notebook that also
scored its own output would be a second implementation of the metrics, and the
first thing anyone would ask of a number that finally cleared the bar is
whether it was measured the same way as the numbers it beat.

The predictions carry `row_index` into the exported package, so the mapping
back to (date, ticker) is by construction rather than by a join that could
silently mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.evaluation import cross_sectional_report            # noqa: E402
from pipeline.signals import HORIZON_SESSIONS                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--package", default="lora_package.npz",
                    help="the package the notebook was given; supplies the "
                         "(date, ticker) each prediction belongs to")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pred = np.load(args.predictions, allow_pickle=True)
    pkg = np.load(args.package, allow_pickle=True)
    meta = json.loads(str(pkg["meta"][0]))
    cfg = json.loads(str(pred["config"][0])) if "config" in pred else {}

    idx = pred["row_index"].astype(int)
    values = pred["prediction"].astype(float)
    if len(idx) != len(values):
        print("  REFUSED: row_index and prediction differ in length",
              file=sys.stderr)
        return 1

    tickers = list(pkg["tickers"])
    dates = list(pkg["dates"])

    frame = pd.DataFrame({
        "date": [dates[i] for i in pkg["row_end"][idx]],
        "ticker": [tickers[i] for i in pkg["row_ticker"][idx]],
        "y_pred": values,
        "y_true": pkg["row_target"][idx].astype(float),
    })

    # The notebook must have scored TEST rows and only test rows. A run that
    # predicted its own training data would report a spectacular result.
    split = pkg["row_split"][idx]
    if (split != 1).any():
        print(f"  REFUSED: {(split != 1).sum():,} of {len(split):,} predictions "
              f"are on TRAINING rows. Those are in-sample and the report would "
              f"be meaningless.", file=sys.stderr)
        return 1

    if frame["y_pred"].nunique() <= 1:
        print("  REFUSED: every prediction is identical, so there is no "
              "ordering to score. The run produced a constant.",
              file=sys.stderr)
        return 1

    report = cross_sectional_report(frame, rebalance_every=HORIZON_SESSIONS)

    mae = float(np.mean(np.abs(frame["y_pred"] - frame["y_true"])))
    floor = float(np.mean(np.abs(frame["y_true"])))

    print(f"LoRA predictions: {len(frame):,} rows | "
          f"{frame['ticker'].nunique()} tickers | "
          f"{frame['date'].nunique():,} dates")
    if cfg:
        print(f"  config: rank {cfg.get('lora_rank')} on {cfg.get('targets')}, "
              f"{cfg.get('epochs')} epochs, context {cfg.get('context')}, "
              f"folds_run {cfg.get('folds_run')}")
    print(f"  package: stride {meta['train_stride']}, "
          f"{meta['folds']} folds, min_train {meta['min_train']}")

    print(f"\n  reb_IC     {report.get('mean_rank_ic', float('nan')):+.4f}")
    print(f"  reb_t      {report.get('rank_ic_t', float('nan')):+.2f}"
          f"   <- the pre-registered criterion; below 2 is not evidence")
    print(f"  n_reb      {report.get('n_rebalances', 0)}")
    print(f"  MAE        {mae:.5f}")
    print(f"  MAE floor  {floor:.5f}   ({100 * (mae / floor - 1):+.1f}%)")
    print(f"  pred sd    {frame['y_pred'].std():.5f}  vs target sd "
          f"{frame['y_true'].std():.5f}")

    if report.get("n_dates_no_ordering"):
        print(f"  {report['n_dates_no_ordering']} date(s) carried no ordering "
              f"and were skipped rather than scored")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"report": report, "mae": mae, "mae_floor": floor,
                       "rows": len(frame), "config": cfg}, fh,
                      indent=2, default=str)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
