"""
tools/export_kronos_package.py - Everything a Kaggle notebook needs to run Kronos.

    python tools/export_kronos_package.py --context 512  --out kronos_512.npz
    python tools/export_kronos_package.py --context 2048 --out kronos_2048.npz

WHERE THE LINE IS DRAWN, AND WHY IT IS FURTHER OUT THAN LoRA'S
--------------------------------------------------------------
`export_lora_package.py` ships the folds and `end_index` and lets the notebook
do gradient descent, because gradient descent is irreducibly the notebook's
job. Kronos is ZERO-SHOT, so there is no training at all — and that lets the
line move much further out. The notebook here does exactly two things: slice a
window at an index it is given, and decode. It returns RAW TERMINAL PRICES.

Everything that can be arithmetically wrong stays home:

  * which row of the forecast is t+horizon (an off-by-one forecasts 29 or 31
    sessions against a 30-session label and still renders a full table)
  * which column carries the close
  * the anchor it is differenced against
  * averaging the sampled paths, weighted by chunk and taken in LOG space

None of that is the notebook's to decide. `tools/score_kronos.py` does it, in
the same process and through the same `cross_sectional_report` that produced
every other row of the results table.

WHAT DRIFT WOULD LOOK LIKE IF IT HAPPENED
-----------------------------------------
It would not look like an error. A notebook that re-derived the purged folds,
the as-of slice or the usable-row mask slightly differently would produce a
complete, plausible table that could not be put beside anything else — and the
first question anyone would ask of a number that finally cleared the bar is
whether it was measured the same way as the numbers it beat. So the fold
labels, `end_index`, and the mask all come from the tested code here, and are
shipped as data.

The mask in particular is `kronos_forecaster.usable_mask`, imported rather than
reimplemented, so the rows a notebook is handed are exactly the rows the local
forecaster would have scored.

ONE PACKAGE PER CONTEXT
-----------------------
The candle data is identical across configurations, but the row list is not:
a full context is required, so at 2048 the fourteen youngest listings drop out
that survive at 512. Exporting per context keeps that visible in the file name
rather than buried in a flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.evaluation import PurgedPanelWalkForward, oos_dates   # noqa: E402
from pipeline.kronos_forecaster import (INPUT_COLS,                 # noqa: E402
                                        TOKENIZERS, VENDORED_COMMIT,
                                        load_relative_candles, usable_mask)
from pipeline.panel import TARGET, load_panel                       # noqa: E402
from pipeline.series import _block_ending_at                        # noqa: E402
from pipeline.signals import HORIZON_SESSIONS                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=HORIZON_SESSIONS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"kronos_{args.context}.npz"

    print("  loading panel ...")
    panel = load_panel()
    frames = load_relative_candles(panel)

    index = frames["close"].index
    tickers = list(frames["close"].columns)
    positions = {d: i for i, d in enumerate(index)}

    splitter = PurgedPanelWalkForward(n_folds=args.folds, horizon=args.horizon,
                                      min_train=args.min_train)

    # THE SCORING GRID, chosen by the same function `compare_baselines` uses.
    # Every out-of-sample date, then every `horizon`-th of them — the
    # non-overlapping rebalances, which are the only sample that supports
    # inference. Derived from the splitter and the LABELS alone, so it is
    # identical to the grid the cheap comparators are scored on.
    all_oos = oos_dates(panel, splitter, TARGET)
    grid = all_oos[::args.horizon]
    print(f"  {len(grid)} rebalance dates of {len(all_oos)} out-of-sample")

    # Which fold each date belongs to, so the notebook's output can be broken
    # down per fold at home. That breakdown is not cosmetic: both prior
    # positive results in this project were carried entirely by the earliest
    # folds, and Kronos' own authors warn it has likely seen these periods.
    date_fold: dict[str, int] = {}
    y = panel.sort_values(["date", "ticker"])
    y_dates = y["date"].to_numpy()
    y_vals = np.asarray(y[TARGET], dtype=float)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(y_dates)):
        train_labelled = train_idx[np.isfinite(y_vals[train_idx])]
        test_labelled = test_idx[np.isfinite(y_vals[test_idx])]
        if len(train_labelled) < 100 or len(test_labelled) == 0:
            continue
        for d in np.unique(y_dates[test_labelled]):
            date_fold.setdefault(str(d), fold)

    # The label is READ from the panel, never recomputed from the candles.
    # target_excess_return is what every other comparator was scored against;
    # a locally re-derived version could differ in the last decimal and the
    # table would silently stop being one table.
    target = dict(zip(zip(panel["date"].astype(str),
                          panel["ticker"].astype(str)),
                      np.asarray(panel[TARGET], dtype=float)))

    rows_date, rows_ticker, rows_end, rows_fold, rows_y = [], [], [], [], []
    dropped_short = 0

    for as_of in grid:
        as_of = str(as_of)
        blocks = {c: _block_ending_at(frames[c], positions, as_of, args.context)
                  for c in INPUT_COLS}
        if any(b is None for b in blocks.values()):
            continue
        if len(blocks["close"]) < args.context:
            # Not enough history behind this date for ANY ticker at this
            # context. Counted rather than skipped silently: an absent date is
            # a rebalance the table will not have, and n is the whole basis of
            # the t-statistic.
            dropped_short += 1
            continue

        mask = usable_mask(blocks)
        end = positions[as_of]

        for j, ticker in enumerate(tickers):
            if not mask[:, j].all():
                continue
            label = target.get((as_of, ticker))
            if label is None or not np.isfinite(label):
                continue
            rows_date.append(as_of)
            rows_ticker.append(j)
            rows_end.append(end)
            rows_fold.append(date_fold.get(as_of, -1))
            rows_y.append(label)

    if not rows_date:
        print("  no scorable rows — check benchmark_close coverage")
        return 1

    # The candles as one dense array, ordered exactly as INPUT_COLS. The
    # notebook builds its DataFrame from this order and nothing else, so a
    # column permutation is impossible rather than merely unlikely.
    #
    # FLOAT64, not float32, and it is not about the model — Kronos casts its
    # input to float32 internally either way. It is about the ANCHOR. The
    # scorer differences each forecast against `candles[end, ticker, close]`,
    # and at float32 that value disagrees with the float64 one the local
    # forecaster reads by ~2e-8, so the two paths stop being bit-identical and
    # the round-trip test can only assert "close enough". It costs 4 MB.
    candles = np.stack(
        [frames[c].to_numpy(dtype=np.float64) for c in INPUT_COLS], axis=-1)

    meta = {
        "context": args.context,
        "horizon": args.horizon,
        "folds": args.folds,
        "min_train": args.min_train,
        "input_cols": INPUT_COLS,
        "vendored_commit": VENDORED_COMMIT,
        "tokenizers": {k: list(v) for k, v in TOKENIZERS.items()},
        "n_rebalances": len(set(rows_date)),
        "n_oos_dates": len(all_oos),
        "dates_dropped_short": dropped_short,
    }

    np.savez_compressed(
        out,
        candles=candles,
        tickers=np.array(tickers, dtype=object),
        dates=np.array([str(d) for d in index], dtype=object),
        row_date=np.array(rows_date, dtype=object),
        row_ticker=np.asarray(rows_ticker, dtype=np.int16),
        row_end=np.asarray(rows_end, dtype=np.int32),
        row_fold=np.asarray(rows_fold, dtype=np.int8),
        row_target=np.asarray(rows_y, dtype=np.float32),
        meta=np.array([json.dumps(meta)], dtype=object),
    )

    per_date = len(rows_date) / max(len(set(rows_date)), 1)
    print(f"\n  {len(rows_date):,} rows | {len(set(rows_date))} rebalances | "
          f"{per_date:.1f} names/date | {len(tickers)} tickers in the panel")
    if dropped_short:
        print(f"  {dropped_short} rebalance dates dropped: fewer than "
              f"{args.context} sessions of history behind them")
    print(f"  wrote {out} ({os.path.getsize(out) / 2**20:.1f} MB)")
    print("\n  Upload as a Kaggle dataset, then run tools/kronos_kaggle.py "
          "in a GPU notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
