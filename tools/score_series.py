"""
tools/score_series.py - Score a Kaggle Chronos/TimesFM run at home.

    python tools/score_series.py --package series_panel.npz \\
        --predictions chronos2_2048.npz timesfm25_2048.npz --json p2_series.json

Every statistic here comes from `cross_sectional_report`, the same function
that produced every other row of the results table, on the rows the package
declared before any model ran. Nothing about the comparison is re-derived from
a notebook's output.

WHAT THIS GRADES AGAINST, AND WHY IT IS NOT `zero`
--------------------------------------------------
The target is the stock's own 30-session log return. 57.67% of those labels are
positive and 32.8% of their variance is shared across the panel, so `zero` is
beaten by drift and beta before a model opens its eyes - it is not a floor, it
is a gift. The package ships the two real floors, fitted on the same folds:

    market        the training mean of the cross-sectional mean. Constant
                  within a date, so it has no ordering by construction, and it
                  bounds MAE.
    beta_market   beta_i x mu_market. Ranks by beta alone and scores a positive
                  IC in a rising market with no company-specific view, so it
                  bounds rank IC.

`clears_floor` means BOTH: MAE below `market` AND reb_IC above `beta_market`.
A comparator that clears `reb_t > 2` while ranking worse than a beta sort has
not found anything about companies.

NEVER ATTEMPTED IS NOT ABSTAINED
--------------------------------
A NaN row is one the notebook never reached - a wall-clock timeout, a
`--limit-dates` smoke test. Those are dropped and reported as coverage. Filling
them with 0.0 would put the `zero` forecast's own prediction into the sample; a
Kronos run covering 1 of 63 rebalances once reported a plausible near-floor null
that was 98% the zero prediction, and was +209.9% worse than the floor over the
rows it actually reached.
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

from pipeline.evaluation import cross_sectional_report               # noqa: E402


def load_run(path: str, package) -> tuple[pd.DataFrame, dict]:
    pred_file = np.load(path, allow_pickle=True)
    run = json.loads(str(pred_file["run"][0]))
    pred = np.asarray(pred_file["pred"], dtype=float)

    n = len(package["row_date"])
    if len(pred) != n:
        raise SystemExit(
            f"{path} holds {len(pred)} predictions but the package declares "
            f"{n} rows. These were produced from different packages and must "
            f"not be scored together."
        )

    tickers = [str(t) for t in package["tickers"]]
    frame = pd.DataFrame({
        "date": [str(d) for d in package["row_date"]],
        "ticker": [tickers[i] for i in package["row_ticker"].astype(int)],
        "fold": package["row_fold"].astype(int),
        "y_true": package["row_target"].astype(float),
        "y_pred": pred,
        "market": package["row_market"].astype(float),
        "beta_market": package["row_beta_market"].astype(float),
    })

    attempted = np.isfinite(frame["y_pred"])
    run["not_scored"] = int((~attempted).sum())
    frame = frame[attempted].reset_index(drop=True)
    run["rows_scored"] = len(frame)
    run["coverage"] = len(frame) / max(n, 1)

    if run["coverage"] < 0.98:
        print(f"  WARNING: {run.get('name', path)} covers "
              f"{run['coverage']:.1%} of the grid. A partial run is not a weak "
              f"result - read it as an incomplete one.")
    return frame, run


def report(frame: pd.DataFrame, label: str, target_sd: float) -> dict:
    # rebalance_every=1: the package already holds ONLY the non-overlapping
    # rebalance dates. Sub-sampling them again would take every 30th of ~64 and
    # compute a t-statistic on n=2 against a table that still renders.
    xs = cross_sectional_report(frame, rebalance_every=1)

    mae = float(np.abs(frame["y_pred"] - frame["y_true"]).mean())
    mkt_mae = float(np.abs(frame["market"] - frame["y_true"]).mean())

    beta_frame = frame.assign(y_pred=frame["beta_market"])
    beta_xs = cross_sectional_report(beta_frame, rebalance_every=1)
    beta_ic = beta_xs.get("mean_rank_ic", float("nan"))

    reb_ic = xs.get("mean_rank_ic", float("nan"))
    beats_mae = mae < mkt_mae
    beats_ic = bool(np.isfinite(reb_ic) and np.isfinite(beta_ic)
                    and reb_ic > beta_ic)

    print(f"\n  {label}")
    print(f"    reb_IC {reb_ic:+.4f}   t {xs.get('rank_ic_t', float('nan')):+.2f}"
          f"   n {xs.get('n_rebalances', 0)}   "
          f"alpha_t {xs.get('alpha_t', float('nan')):+.2f}")
    print(f"    MAE {mae:.5f}  vs market {mkt_mae:.5f} "
          f"({(mae / mkt_mae - 1) * 100:+.1f}%)   "
          f"pred SD {frame['y_pred'].std():.5f} vs target {target_sd:.5f}")
    print(f"    beats market on MAE: {beats_mae}   "
          f"beats beta_market on reb_IC ({beta_ic:+.4f}): {beats_ic}   "
          f"-> {'CLEARS THE FLOOR' if (beats_mae and beats_ic) else 'below the floor'}")

    per_fold = []
    for fold in sorted(frame["fold"].unique()):
        part = frame[frame["fold"] == fold]
        f_xs = cross_sectional_report(part, rebalance_every=1)
        per_fold.append({"fold": int(fold),
                         "n": f_xs.get("n_rebalances", 0),
                         "reb_ic": f_xs.get("mean_rank_ic"),
                         "reb_t": f_xs.get("rank_ic_t")})

    # ALWAYS BROKEN DOWN BY FOLD, never only pooled. Both prior positive
    # results in this project ran +0.0879 -> -0.0307 across the folds while
    # their pooled t-statistics read +3.32 and +2.37. A result that shrinks as
    # training data grows, or that lives in the earliest window, is an artifact
    # of the panel rather than a property of the model.
    print(f"    {'fold':<6}{'n':>5}{'reb_IC':>10}{'reb_t':>9}")
    for f in per_fold:
        print(f"    {f['fold']:<6}{f['n']:>5}"
              f"{(f['reb_ic'] if f['reb_ic'] is not None else float('nan')):>+10.4f}"
              f"{(f['reb_t'] if f['reb_t'] is not None else float('nan')):>+9.2f}")

    return {"reb_ic": reb_ic, "reb_t": xs.get("rank_ic_t"),
            "n": xs.get("n_rebalances"), "alpha_t": xs.get("alpha_t"),
            "mae": mae, "market_mae": mkt_mae, "beta_market_ic": beta_ic,
            "beats_market": beats_mae, "beats_beta_ic": beats_ic,
            "clears_floor": bool(beats_mae and beats_ic),
            "pred_sd": float(frame["y_pred"].std()),
            "per_fold": per_fold}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    package = np.load(args.package, allow_pickle=True)
    meta = json.loads(str(package["meta"][0]))
    target_sd = float(np.std(package["row_target"].astype(float)))

    print(f"  package: target {meta['target']}, {meta['n_rebalances']} "
          f"rebalances, {meta['n_tickers']} tickers, "
          f"{len(package['row_date'])} rows")
    print(f"  target SD {target_sd:.5f}")

    out = {"package": meta, "runs": {}}
    for path in args.predictions:
        frame, run = load_run(path, package)
        label = (f"{run.get('name', os.path.basename(path))} @"
                 f"{run.get('context')}  "
                 f"[{run['rows_scored']} rows, {run['coverage']:.1%} covered, "
                 f"{run.get('seconds', 0) / 60:.0f} min]")
        out["runs"][run.get("name", path)] = {
            **report(frame, label, target_sd), **run}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
