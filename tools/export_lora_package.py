"""
tools/export_lora_package.py - Everything a Kaggle notebook needs to fine-tune.

    python tools/export_lora_package.py --out lora_package.npz
    python tools/export_lora_package.py --train-stride 10 --context 512

WHY A PACKAGE AND NOT A NOTEBOOK THAT REBUILDS THE PANEL
--------------------------------------------------------
The risk in moving this to Kaggle is not compute, it is COMPARABILITY. A
notebook that re-derives the purged folds, the as-of slice or the target even
slightly differently produces a number that cannot be put in the same table as
everything else - and it would look perfectly reasonable while doing it.

So every decision that could drift is made HERE, by the same tested code that
produced every other row of the results table, and shipped as data:

  * the exact train/test row lists per fold, from `PurgedPanelWalkForward`
  * `end_index`, the position in each ticker's series that a row is as-of.
    The notebook slices `series[end - context + 1 : end + 1]` and nothing else.
    The causal decision - the one an off-by-one turns into a breakthrough - is
    not the notebook's to make.
  * the target, read from the panel rather than recomputed from prices

The notebook's only job is gradient descent.

TRAIN STRIDE, AND WHY IT IS NOT CHEATING
----------------------------------------
Using every date gives 535,069 training windows across the five folds, which is
the figure behind a one-to-two-day estimate. It is also almost entirely
duplication: consecutive dates share 29 of their 30 forward sessions, so a
panel of 1,956 training dates holds roughly sixty independent windows per
ticker, not 1,956. `--train-stride 10` takes every tenth training date, cuts
the work to 53,506 windows and about an hour, and removes duplicated gradient
signal rather than information.

The stride applies to TRAINING ONLY. Every test row is scored, because the
results table must cover the same rows as every other comparator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.evaluation import PurgedPanelWalkForward           # noqa: E402
from pipeline.panel import (TARGET, load_panel,                  # noqa: E402
                            relative_price_frame)
from pipeline.series import DEFAULT_CONTEXT, MIN_CONTEXT         # noqa: E402
from pipeline.signals import HORIZON_SESSIONS                    # noqa: E402


def build_rows(labelled, splitter, series, t_index, d_index, context,
               train_stride, summary, verbose=False):
    """
    One record per scored row: which fold, which split, and WHERE IT IS AS OF.

    `end_index` is the causal boundary and the only thing here that a notebook
    could not recompute safely. The window a row is entitled to see is
    `series[ticker, end - context + 1 : end + 1]` - inclusive of the row's own
    date, exclusive of everything after it. An off-by-one in the other
    direction hands the model the answer it is being asked to predict, produces
    no error, and reads as a breakthrough.

    The train stride is applied to TRAINING dates only. Test rows are never
    strided, because the results table must cover the same rows as every other
    comparator.
    """
    rows_fold, rows_split, rows_t, rows_end, rows_y = [], [], [], [], []

    for fold, (tr, te) in enumerate(splitter.split(labelled["date"].to_numpy())):
        train_dates = sorted(labelled.loc[tr, "date"].unique())
        keep = set(train_dates[::train_stride])

        n_tr = n_te = 0
        for split, idx in (("train", tr), ("test", te)):
            for i in idx:
                d = labelled.at[i, "date"]
                if split == "train" and d not in keep:
                    continue
                ti = t_index.get(labelled.at[i, "ticker"])
                di = d_index.get(d)
                if ti is None or di is None:
                    continue
                window = series[ti, max(0, di - context + 1):di + 1]
                if np.isfinite(window).sum() < MIN_CONTEXT:
                    continue
                rows_fold.append(fold)
                rows_split.append(0 if split == "train" else 1)
                rows_t.append(ti)
                rows_end.append(di)
                rows_y.append(float(labelled.at[i, TARGET]))
                n_tr += split == "train"
                n_te += split == "test"

        summary.append({"fold": fold, "train_rows": n_tr, "test_rows": n_te,
                        "train_dates_kept": len(keep),
                        "train_dates_total": len(train_dates)})
        if verbose:
            print(f"  fold {fold}: {n_tr:>7,} train (every {train_stride}th of "
                  f"{len(train_dates):,} dates)  {n_te:>7,} test")

    return rows_fold, rows_split, rows_t, rows_end, rows_y


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="lora_package.npz")
    ap.add_argument("--context", type=int, default=512,
                    help="trailing observations per window. 512 fits a Kaggle "
                         "session comfortably; 2048 costs roughly 8x because "
                         "attention is quadratic in patch count")
    ap.add_argument("--train-stride", type=int, default=10,
                    help="keep every Nth TRAINING date (test is never strided)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=500)
    args = ap.parse_args()

    panel = load_panel()
    if panel.empty:
        print("  REFUSED: no signals rows", file=sys.stderr)
        return 1

    wide = relative_price_frame(panel)
    if wide.empty:
        print("  REFUSED: no benchmark_close, so no relative-price series",
              file=sys.stderr)
        return 1

    tickers = list(wide.columns)
    t_index = {t: i for i, t in enumerate(tickers)}
    dates = [str(d) for d in wide.index]
    d_index = {d: i for i, d in enumerate(dates)}

    # One flat array per ticker, and the row's position in it. Slicing is then
    # the only thing left to get wrong, and it is one expression.
    series = wide.to_numpy(dtype=np.float32).T          # (n_tickers, n_dates)

    labelled = panel.dropna(subset=[TARGET]).reset_index(drop=True)
    labelled["date"] = labelled["date"].astype(str)

    splitter = PurgedPanelWalkForward(
        n_folds=args.folds, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=args.min_train)

    rows_fold, rows_split, rows_t, rows_end, rows_y = build_rows(
        labelled, splitter, series, t_index, d_index,
        context=args.context, train_stride=args.train_stride,
        summary=(summary := []), verbose=True)

    meta = {
        "context": args.context, "horizon": HORIZON_SESSIONS,
        "min_context": MIN_CONTEXT, "train_stride": args.train_stride,
        "folds": args.folds, "min_train": args.min_train,
        "n_tickers": len(tickers), "n_dates": len(dates),
        "summary": summary,
    }

    np.savez_compressed(
        args.out,
        series=series,
        tickers=np.array(tickers, dtype=object),
        dates=np.array(dates, dtype=object),
        row_fold=np.asarray(rows_fold, dtype=np.int16),
        row_split=np.asarray(rows_split, dtype=np.int8),
        row_ticker=np.asarray(rows_t, dtype=np.int16),
        row_end=np.asarray(rows_end, dtype=np.int32),
        row_target=np.asarray(rows_y, dtype=np.float32),
        meta=np.array([json.dumps(meta)], dtype=object),
    )

    size = os.path.getsize(args.out) / 2**20
    print(f"\n  {len(rows_fold):,} rows | {len(tickers)} tickers | "
          f"{len(dates):,} dates | context {args.context}")
    print(f"  wrote {args.out} ({size:.1f} MB)")
    print("\n  Upload this one file as a Kaggle dataset, then run "
          "tools/lora_kaggle.py in a GPU notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
