"""
tools/export_series_package.py - Everything a Kaggle notebook needs for
Chronos-2 and TimesFM-2.5.

    python tools/export_series_package.py --out series_panel.npz

The Kronos counterpart is `export_kronos_package.py`; this is the univariate
one. Read that file's docstring first — the line about what the notebook is and
is not allowed to compute is the same line, for the same reasons.

THREE THINGS THIS DOES DIFFERENTLY, AND EACH IS A CORRECTION
------------------------------------------------------------
1. ONE PACKAGE, EVERY CONTEXT.

   The Kronos package is per-context because Kronos needs a FULL window: at
   2048 the fourteen youngest listings drop out that survive at 512, so the two
   runs are measured over different universes and the recorded comparison has
   to be read with that caveat. A univariate foundation model has no such
   requirement — it left-pads — so eligibility here is computed once, at
   MIN_CONTEXT, over the history available up to the as-of date, and it does
   not move with the context.

   Consequence: chronos@2048, chronos@512 and timesfm@16384 score IDENTICAL
   ROWS. The context comparison is then a context comparison and nothing else,
   which is exactly what the base@512-vs-mini@2048 landmine says the Kronos
   pair failed to be. `row_avail` records how much real history each row
   actually had, so a run can report its effective context rather than the
   requested one.

2. THE FLOORS TRAVEL WITH THE ROWS.

   `market` and `beta_market` are the P1 floors: on an absolute-return target
   `zero` is beaten by drift and beta before a model opens its eyes, so a
   Kaggle number graded against `zero` is graded against nothing. Both are
   fitted HERE, through the same `panel_walk_forward` on the same folds as
   every other comparator, and shipped per row. `score_series.py` can then say
   `clears_floor` without a second database round trip and without the scorer
   re-deriving a floor slightly differently from the table it is joining.

3. THE NOTEBOOK IMPORTS OUR CODE RATHER THAN A COPY OF IT.

   `pipeline/series.py`, `pipeline/chronos_forecaster.py` and
   `pipeline/timesfm_forecaster.py` are shipped INSIDE the package as source
   text with a sha256 each. `series_kaggle.py` writes them out and imports
   them, so the median-index lookup, the TF32 determinism settings, the
   `truncate_negative` override and the patch-multiple context rounding are the
   tested ones rather than a notebook's approximation of them.

   That matters more here than anywhere: the median is at index 10 on
   `amazon/chronos-2`, 6 on `autogluon/chronos-2-small` and 5 on TimesFM (with
   an off-by-one because column 0 is the point output). Reading the wrong one
   applies a systematic quantile bias to every prediction, raises nothing, and
   renders a perfect table. A hand-copied notebook is precisely where that
   happens.

   The three modules import only numpy, pandas and the standard library at
   module level; torch is imported inside `load_pipeline`. So this costs the
   package about 100 KB and the notebook nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.baselines import (BASELINES,                        # noqa: E402
                                FLOORS, baseline_feature_columns)
from pipeline.evaluation import (PurgedPanelWalkForward,          # noqa: E402
                                 oos_dates, panel_walk_forward)
from pipeline.panel import TARGET, load_panel, price_frame        # noqa: E402
from pipeline.series import MIN_CONTEXT                           # noqa: E402
from pipeline.signals import HORIZON_SESSIONS                     # noqa: E402

# Shipped verbatim so the notebook runs the tested code. Order matters only in
# that `series` must be importable before the two forecasters, which import it.
SHIPPED_MODULES = (
    "pipeline/series.py",
    "pipeline/chronos_forecaster.py",
    "pipeline/timesfm_forecaster.py",
)


def collect_sources(root: str = ".") -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for rel in SHIPPED_MODULES:
        path = os.path.join(root, *rel.split("/"))
        text = open(path, encoding="utf-8").read()
        out[rel] = {
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    return out


def floor_predictions(panel: pd.DataFrame,
                      splitter: PurgedPanelWalkForward,
                      grid: set,
                      horizon: int) -> dict[str, pd.DataFrame]:
    """
    The two floors, fitted on the same folds as everything else.

    `score_dates` narrows the TEST side only, after the split, so the folds,
    the purge and the embargo are untouched — the same guarantee
    `compare_baselines` relies on when it scores an expensive model on a subset
    of dates. `rebalance_every=1` because `grid` is already the non-overlapping
    rebalance set; sub-sampling it again would take every 30th of ~64.
    """
    out: dict[str, pd.DataFrame] = {}
    for name in FLOORS:
        factory = BASELINES[name]
        cols = baseline_feature_columns(name)
        result = panel_walk_forward(
            panel,
            feature_cols=cols,
            model_factory=factory,
            splitter=splitter,
            name=name,
            target=TARGET,
            rebalance_every=1,
            score_dates=grid,
        )
        preds = result.predictions[["date", "ticker", "y_pred"]].copy()
        preds["date"] = preds["date"].astype(str)
        preds["ticker"] = preds["ticker"].astype(str)
        out[name] = preds.rename(columns={"y_pred": name})
    return out


def scorable_rows(values: np.ndarray,
                  positions: dict[str, int],
                  tickers: list[str],
                  grid,
                  target: dict[tuple[str, str], float],
                  date_fold: dict[str, int],
                  min_context: int) -> list[dict]:
    """
    Which (date, ticker) pairs are scorable, and what each one's anchor is.

    Factored out of `main` so the tests exercise THIS, not a re-statement of
    the rule. A test that lifts the rule into itself passes against any
    implementation that agrees with the copy in the test, which is the same
    trap as a test that greps source for a literal line.

    Two decisions live here and both fail silently if they are wrong.

    ELIGIBILITY is `min_context` finite observations at or before the as-of
    date, and it does NOT take the model's context. Kronos needs a full window
    and so its packages are per-context, which is why base@512 covered 90
    tickers and mini@2048 covered 81 and their comparison confounded coverage
    with context. A univariate model left-pads, so fixing eligibility here
    means every context scores identical rows.

    THE ANCHOR is the last FINITE observation at or before the as-of date -
    not `values[end]`. Those differ exactly when a ticker's most recent session
    is missing, and `_history_ending_at` drops non-finite values before the
    forecaster reads `history[-1]`. Using `values[end]` would apply a whole
    session's return to that row alone, keep the prediction the right order of
    magnitude, and move only the fifth decimal of MAE.
    """
    finite = np.isfinite(values)
    # A running count, so eligibility is O(1) per row rather than a slice.
    avail_upto = np.cumsum(finite, axis=0)

    rows: list[dict] = []
    for as_of in grid:
        as_of = str(as_of)
        end = positions.get(as_of)
        if end is None:
            continue
        for j, ticker in enumerate(tickers):
            avail = int(avail_upto[end, j])
            if avail < min_context:
                continue
            label = target.get((as_of, ticker))
            if label is None or not np.isfinite(label):
                continue
            col = values[: end + 1, j]
            anchor = float(col[finite[: end + 1, j]][-1])
            rows.append({
                "date": as_of, "ticker": ticker, "ticker_idx": j,
                "end": end, "fold": date_fold.get(as_of, -1),
                "target": float(label), "anchor": anchor, "avail": avail,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=HORIZON_SESSIONS)
    ap.add_argument("--min-context", type=int, default=MIN_CONTEXT,
                    help="finite observations a row needs to be eligible. "
                         "NOT the model context — that is the notebook's flag. "
                         "This is what makes every context score the same rows.")
    ap.add_argument("--out", default="series_panel.npz")
    args = ap.parse_args()

    print("  loading panel ...")
    panel = load_panel()

    # THE BASIS IS DERIVED FROM THE TARGET, never chosen here. log(close) for
    # the absolute return, log(close/benchmark) for the excess one; the
    # h-session forward difference of what this returns IS the label being
    # scored. Passing the wrong basis raises nothing — the table renders, the
    # magnitudes look right, and the only symptom is a comparator that will not
    # beat the floor.
    series = price_frame(panel, TARGET)
    index = list(series.index)
    tickers = list(series.columns)
    positions = {str(d): i for i, d in enumerate(index)}
    values = series.to_numpy(dtype=np.float64)

    splitter = PurgedPanelWalkForward(n_folds=args.folds, horizon=args.horizon,
                                      min_train=args.min_train)

    all_oos = oos_dates(panel, splitter, TARGET)
    grid = all_oos[::args.horizon]
    print(f"  {len(grid)} rebalance dates of {len(all_oos)} out-of-sample")

    # Fold identity per date, so the notebook's output can be broken down per
    # fold at home. Not cosmetic: both prior positive results in this project
    # were carried entirely by the earliest folds, and a pooled t-statistic hid
    # a negative most-recent fold once already.
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

    # The label is READ from the panel, never recomputed from the series. A
    # locally re-derived version could differ in the last decimal and the table
    # would silently stop being one table.
    target = dict(zip(zip(panel["date"].astype(str),
                          panel["ticker"].astype(str)),
                      np.asarray(panel[TARGET], dtype=float)))

    print("  fitting the floors on the same folds ...")
    floors = floor_predictions(panel, splitter, set(grid), args.horizon)

    rows = scorable_rows(values, positions, tickers, grid, target, date_fold,
                         args.min_context)
    if not rows:
        print("  no scorable rows")
        return 1

    frame = pd.DataFrame(rows)
    for name, preds in floors.items():
        frame = frame.merge(preds, on=["date", "ticker"], how="left")
        missing = int(frame[name].isna().sum())
        if missing:
            # A floor that is absent on some rows cannot grade them, and a
            # silently partial floor is the shape that let `train_mean` look
            # like a winner once. Reported, not hidden.
            print(f"  WARNING: {name} has no prediction on {missing} of "
                  f"{len(frame)} rows")

    meta = {
        "target": TARGET,
        "horizon": args.horizon,
        "folds": args.folds,
        "min_train": args.min_train,
        "min_context": args.min_context,
        "n_rebalances": int(frame["date"].nunique()),
        "n_oos_dates": int(len(all_oos)),
        "n_tickers": int(frame["ticker"].nunique()),
        "floors": list(FLOORS),
        "avail_min": int(frame["avail"].min()),
        "avail_median": int(frame["avail"].median()),
        "avail_max": int(frame["avail"].max()),
    }

    payload = dict(
        series=values,
        tickers=np.array(tickers, dtype=object),
        dates=np.array([str(d) for d in index], dtype=object),
        row_date=np.array(frame["date"].tolist(), dtype=object),
        row_ticker=np.asarray(frame["ticker_idx"], dtype=np.int16),
        row_end=np.asarray(frame["end"], dtype=np.int32),
        row_fold=np.asarray(frame["fold"], dtype=np.int8),
        row_target=np.asarray(frame["target"], dtype=np.float64),
        row_anchor=np.asarray(frame["anchor"], dtype=np.float64),
        row_avail=np.asarray(frame["avail"], dtype=np.int32),
        sources=np.array([json.dumps(collect_sources())], dtype=object),
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    for name in FLOORS:
        payload[f"row_{name}"] = np.asarray(frame[name], dtype=np.float64)

    np.savez_compressed(args.out, **payload)

    size = os.path.getsize(args.out) / 1e6
    print(f"\n  wrote {args.out}  ({size:.1f} MB)")
    print(f"    {len(frame)} rows, {meta['n_tickers']} tickers, "
          f"{meta['n_rebalances']} rebalances, target {TARGET}")
    print(f"    history available per row: min {meta['avail_min']}, "
          f"median {meta['avail_median']}, max {meta['avail_max']}")
    print(f"    a context above {meta['avail_max']} buys nothing on any row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
