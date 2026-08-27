"""
tools/score_kronos.py - Score Kaggle Kronos output through the local harness.

    python tools/score_kronos.py --predictions kronos_base_512_s0.npz \\
        --package kronos_512.npz
    python tools/score_kronos.py --predictions run_s0.npz run_s1.npz run_s2.npz \\
        --package kronos_512.npz

The notebook returns raw terminal relative closes and nothing else. Every
statistic — the rebalance rank IC and its t, the MAE against the `zero` floor,
the per-fold breakdown — is computed here by `cross_sectional_report`, the same
function that produced every other row of the results table. A notebook that
scored its own output would be a second implementation of the metrics, and the
first thing anyone would ask of a number that finally cleared the bar is
whether it was measured the same way as the numbers it beat.

WHAT THIS PRINTS, AND WHY IT PRINTS IT THAT WAY
-----------------------------------------------
Three things beyond the headline, each answering a specific way this project has
been fooled before.

**Per fold.** Both prior positive results here — valuation at t +3.32 and LoRA
at +2.37 — were carried entirely by the earliest folds and inverted in the most
recent one, and the pooled statistic hid it both times. Kronos' own authors warn
it has likely seen the historical periods a quant team would want to evaluate
on, so an effect concentrated in early folds is the EXPECTED shape of
contamination, not a surprise to be explained away.

**Across seeds.** Kronos samples, so a single `reb_t` is one draw at a fixed
setting. Passing several prediction files reports the spread, which is the only
thing that makes any one of them quotable.

**Predicted dispersion against the target's.** At one sample the predicted
cross-sectional SD measured 0.192 (base@512) and 0.421 (mini@2048) against a
target SD of 0.102 — one name at a predicted +247% excess return. A run whose
dispersion is still far from the target's is reporting sampling noise, and its
MAE is not comparable with a run at a different sample count either way.
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


def load_run(path: str, package) -> tuple[pd.DataFrame, dict]:
    pred = np.load(path, allow_pickle=True)
    run = json.loads(str(pred["run"][0]))

    terminal = np.asarray(pred["terminal"], dtype=float)
    row_index = pred["row_index"].astype(int)

    row_ticker = package["row_ticker"].astype(int)
    tickers = [str(t) for t in package["tickers"]]
    row_end = package["row_end"].astype(int)
    candles = package["candles"]
    input_cols = list(json.loads(str(package["meta"][0]))["input_cols"])
    close_col = input_cols.index("close")

    # THE ANCHOR IS THE AS-OF CLOSE, read here rather than trusted from the
    # notebook. `row_end` is the as-of position, so `candles[end, ticker,
    # close]` is the observed relative close at t — the same value the local
    # forecaster differences against.
    anchor = candles[row_end[row_index], row_ticker[row_index], close_col]
    anchor = np.asarray(anchor, dtype=float)

    # AVERAGED IN LOG SPACE. `predict_batch` averages decoded PRICES inside a
    # chunk; averaging chunks the same way would apply a Jensen bias that grows
    # with dispersion, which is the very quantity the sampling reduces. Each
    # column of `terminal` is one path's terminal price, repeated per chunk
    # member, so a plain mean over columns is already weighted correctly.
    # NEVER ATTEMPTED IS NOT THE SAME AS ABSTAINED, AND THE DIFFERENCE MATTERS.
    #
    # The notebook initialises `terminal` to NaN and fills what it scores, so an
    # ALL-NaN row is one it never reached — a truncated run, a --limit-dates
    # smoke test, a session that hit the wall clock. Those rows are not part of
    # the sample and are dropped.
    #
    # A row with finite samples whose forecast is unusable (a decode to a
    # non-positive price, a non-positive anchor) IS part of the sample, and
    # predicts no excess return — the same claim the `zero` floor makes, so it
    # cannot flatter this comparator against the floor it is measured against.
    #
    # Filling both with 0.0 is the ChronosProbe defect in a new place: a run
    # that scored 1 of 63 rebalances reported MAE 0.06705 against a floor of
    # 0.06532, which reads as a genuine near-floor null and is 98% the zero
    # prediction.
    attempted = np.isfinite(terminal).any(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_terminal = np.where(terminal > 0, np.log(terminal), np.nan)

    # Summed explicitly rather than by nanmean, which warns "Mean of empty
    # slice" on every abstention. A warning that fires routinely is a warning
    # nobody reads.
    usable = np.isfinite(log_terminal)
    n_usable = usable.sum(axis=1)
    total = np.where(usable, log_terminal, 0.0).sum(axis=1)

    mean_log = np.where(n_usable > 0, total / np.maximum(n_usable, 1), np.nan)
    mean_log = np.where(attempted, mean_log, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        y_pred = mean_log - np.log(np.where(anchor > 0, anchor, np.nan))

    frame = pd.DataFrame({
        "date": [str(d) for d in package["row_date"][row_index]],
        "ticker": [tickers[i] for i in row_ticker[row_index]],
        "fold": package["row_fold"].astype(int)[row_index],
        "y_true": package["row_target"].astype(float)[row_index],
        "y_pred": y_pred,
        "attempted": attempted,
    })

    not_scored = int((~frame["attempted"]).sum())
    frame = frame[frame["attempted"]].drop(columns="attempted").reset_index(drop=True)

    abstained = int(frame["y_pred"].isna().sum())
    frame["y_pred"] = frame["y_pred"].fillna(0.0)

    run["abstentions"] = abstained
    run["not_scored"] = not_scored
    run["rows_scored"] = len(frame)
    return frame, run


def report(frame: pd.DataFrame, label: str, target_sd: float) -> dict:
    # rebalance_every=1: the package already holds ONLY the non-overlapping
    # rebalance dates. Sub-sampling them again would take every 30th of 64 and
    # compute a t-statistic on n=2.
    xs = cross_sectional_report(frame, rebalance_every=1)
    mae = float(np.abs(frame["y_pred"] - frame["y_true"]).mean())
    floor = float(np.abs(frame["y_true"]).mean())
    sd = float(frame["y_pred"].std())

    print(f"\n  {label}")
    print(f"    reb_IC {xs.get('mean_rank_ic', float('nan')):+.4f}   "
          f"t {xs.get('rank_ic_t', float('nan')):+.2f}   "
          f"n {xs.get('n_rebalances', 0)}   "
          f"alpha_t {xs.get('alpha_t', float('nan')):+.2f}")
    print(f"    MAE {mae:.5f} vs zero-floor {floor:.5f} "
          f"({(mae / floor - 1) * 100:+.1f}%)   "
          f"pred SD {sd:.5f} vs target {target_sd:.5f}")
    return {"reb_ic": xs.get("mean_rank_ic"), "reb_t": xs.get("rank_ic_t"),
            "n": xs.get("n_rebalances"), "mae": mae, "sd": sd}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only-dates-from", default=None, metavar="PACKAGE.npz",
                    help="score only the rebalance dates present in another "
                         "package. The @2048 package holds 12 rebalances "
                         "against @512's 63, because 52 dates lack 2,048 "
                         "sessions of history behind them — so an unrestricted "
                         "comparison of the two reads a context effect off "
                         "different samples in different folds.")
    args = ap.parse_args()

    package = np.load(args.package, allow_pickle=True)
    target_sd = float(np.std(package["row_target"].astype(float)))

    # MATCHED DATES, OR THE CONTEXT COMPARISON IS NOT A CONTEXT COMPARISON.
    #
    # A longer context needs more history behind each date, so it loses the
    # EARLY rebalances — exactly the folds where both prior positive results in
    # this project lived. Comparing @512 over 63 dates spanning five folds
    # against @2048 over 12 dates in the last one measures the period, the fold
    # mix and n as much as the context. This restricts the wider run to the
    # narrower run's dates so only the context differs.
    keep_dates = None
    if args.only_dates_from:
        other = np.load(args.only_dates_from, allow_pickle=True)
        keep_dates = {str(d) for d in other["row_date"]}
        print(f"  restricted to {len(keep_dates)} dates from "
              f"{os.path.basename(args.only_dates_from)}")

    total_rows = len(package["row_date"])
    summary = []
    for path in args.predictions:
        frame, run = load_run(path, package)

        if run["not_scored"]:
            # Loud, and it names the shortfall as a COVERAGE fact rather than
            # letting it be absorbed into the metrics below. A run that covered
            # a fraction of the grid is not a weaker version of the full run —
            # its rebalances sit wherever the run stopped, which on a
            # walk-forward panel means a different set of folds.
            covered = run["rows_scored"] / max(total_rows, 1)
            print(f"\n  !! {path}: {run['not_scored']:,} of {total_rows:,} rows "
                  f"were never attempted ({covered:.1%} of the grid covered). "
                  f"Metrics below describe ONLY the rows the run reached.")
            if covered < 0.9:
                print("     That is a partial run. Do not put this row in the "
                      "results table beside comparators scored on the full "
                      "grid — the folds it covers are not the same folds.")

        label = (f"{run['model'].rsplit('/', 1)[-1]}@{run['context']} "
                 f"samples={run['samples']} seed={run['seed']} "
                 f"({run['seconds'] / 60:.0f} min"
                 + (f", {run['abstentions']} abstained" if run["abstentions"]
                    else "") + ")")
        if keep_dates is not None:
            frame = frame[frame["date"].isin(keep_dates)].reset_index(drop=True)
            label += f" [{frame['date'].nunique()} matched dates]"
        stats = report(frame, label, target_sd)
        stats["seed"] = run["seed"]
        stats["label"] = label
        summary.append(stats)

        # PER FOLD. A pooled t-statistic hid a monotone decline and a negative
        # most-recent fold twice in this project. Printed always, not on a flag.
        print("      fold   n    reb_IC     t")
        for fold, part in frame.groupby("fold"):
            if len(part) == 0:
                continue
            f_xs = cross_sectional_report(part, rebalance_every=1)
            print(f"      {fold:>4} {f_xs.get('n_rebalances', 0):>3} "
                  f"{f_xs.get('mean_rank_ic', float('nan')):>+8.4f} "
                  f"{f_xs.get('rank_ic_t', float('nan')):>+6.2f}")

    if len(summary) > 1:
        ts = np.array([s["reb_t"] for s in summary], dtype=float)
        print(f"\n  ACROSS {len(ts)} SEEDS: reb_t mean {np.nanmean(ts):+.2f} "
              f"sd {np.nanstd(ts, ddof=1):.2f} "
              f"min {np.nanmin(ts):+.2f} max {np.nanmax(ts):+.2f}")
        print("  A single seed's t is one draw from this spread. The "
              "pre-registered bar is reb_t > 2 — read it against the mean, "
              "and treat a max that clears it while the mean does not as "
              "the sampling artifact it is.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, allow_nan=False, default=float)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
