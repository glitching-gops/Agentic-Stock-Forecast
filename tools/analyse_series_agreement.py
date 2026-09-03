"""
tools/analyse_series_agreement.py - Three questions a results table cannot answer.

    python tools/analyse_series_agreement.py --package series_panel.npz \\
        --predictions c2_2048.npz c2_2048_xl.npz c2_512.npz tfm_2048.npz tfm_512.npz

`score_series.py` grades each run against the floors. This asks what the runs
say about EACH OTHER, which is where the 2026-09-02 verdict actually came from.

1. IS A STRONG FOLD A PERIOD EFFECT OR A CELL COUNT?
   All three Chronos configurations peaked at fold 3 with reb_t +2.02, +1.76
   and +1.72 - and that would have been the first positive result in this
   project not carried by the earliest fold. It is not one. Five configurations
   times five folds is 25 cells; the largest |t| was 2.02, p = 0.066 on 12 df,
   so 1.65 cells are EXPECTED at that level and 1 was seen. The three Chronos
   rows are also one model at three settings rather than three witnesses, which
   is why the breakdown is repeated by ARCHITECTURE: TimesFM's fold 3 read
   -0.43 and +0.36.

2. DO THE MODELS AGREE WITH EACH OTHER, OR WITH THE ANSWER?
   The decisive measurement. Measured: the two architectures correlate with
   each other at Spearman +0.24 to +0.51 and with the TARGET at +0.005 to
   +0.034. They confidently extract a common structure that carries no
   information about the label. Two models agreeing is therefore not evidence
   about the target - they can agree on the same wrong thing - which is the
   inference the earlier "two independent architectures agree" note came close
   to making.

3. DOES AN ENSEMBLE OF NULLS CLEAR ANYTHING?
   Read the ORDERING only. The members are standardised before averaging,
   because they carry different dispersions and a plain mean is a weighted vote
   nobody chose - which also puts the ensemble off the return scale, so its MAE
   is an artifact and is not reported.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.evaluation import cross_sectional_report          # noqa: E402


def architecture(name: str) -> str:
    """The model family a run belongs to, for the witness count in (1)."""
    stem = name.split("@")[0]
    return stem.replace("_xl", "")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--predictions", nargs="+", required=True)
    args = ap.parse_args()

    import json

    pkg = np.load(args.package, allow_pickle=True)
    tickers = [str(t) for t in pkg["tickers"]]
    base = pd.DataFrame({
        "date": [str(d) for d in pkg["row_date"]],
        "ticker": [tickers[i] for i in pkg["row_ticker"].astype(int)],
        "fold": pkg["row_fold"].astype(int),
        "y_true": pkg["row_target"].astype(float),
        "market": pkg["row_market"].astype(float),
        "beta_market": pkg["row_beta_market"].astype(float),
    })

    preds: dict[str, np.ndarray] = {}
    for path in args.predictions:
        f = np.load(path, allow_pickle=True)
        run = json.loads(str(f["run"][0]))
        label = f"{run.get('name', os.path.basename(path))}@{run.get('context')}"
        pred = np.asarray(f["pred"], dtype=float)
        if len(pred) != len(base):
            raise SystemExit(
                f"{path} holds {len(pred)} predictions against the package's "
                f"{len(base)} rows; these came from different exports.")
        preds[label] = pred

    print("=" * 78)
    print("1. IS A STRONG FOLD A PERIOD EFFECT, OR THE NUMBER OF CELLS LOOKED AT?")
    print("=" * 78)
    cells = []
    for name, p in preds.items():
        frame = base.assign(y_pred=p)
        for fold in sorted(frame["fold"].unique()):
            xs = cross_sectional_report(frame[frame["fold"] == fold],
                                        rebalance_every=1)
            t = xs.get("rank_ic_t")
            n = xs.get("n_rebalances", 0)
            if t is not None and np.isfinite(t) and n > 1:
                cells.append({"config": name, "arch": architecture(name),
                              "fold": int(fold), "t": float(t), "n": int(n)})
    if not cells:
        print("  no scorable (config, fold) cells")
        return 1

    cells = pd.DataFrame(cells)
    biggest = float(cells["t"].abs().max())
    df = int(cells.loc[cells["t"].abs().idxmax(), "n"]) - 1
    p_cell = 2 * stats.t.sf(biggest, df=df)
    print(f"  {len(cells)} (config, fold) cells examined")
    print(f"  largest |t| = {biggest:.2f}, two-sided p = {p_cell:.3f} on {df} df")
    print(f"  expected at that level under the null: {len(cells) * p_cell:.2f}")
    print(f"  observed: {int((cells['t'].abs() >= biggest - 1e-9).sum())}")
    print()
    print("  mean t by fold:")
    for fold, grp in cells.groupby("fold"):
        print(f"    fold {fold}   mean {grp['t'].mean():+.2f}   n {len(grp)}   "
              f"[{', '.join(f'{v:+.2f}' for v in grp['t'])}]")
    print()
    print("  by ARCHITECTURE - configurations of one model are not independent")
    print("  witnesses, and a pattern only one family shows is that family's:")
    for arch, grp in cells.groupby("arch"):
        best = grp.loc[grp["t"].idxmax()]
        print(f"    {arch:12s} best cell fold {int(best['fold'])} "
              f"t {best['t']:+.2f}   mean over folds {grp['t'].mean():+.2f}")

    print()
    print("=" * 78)
    print("2. DO THEY AGREE WITH EACH OTHER, OR WITH THE ANSWER?")
    print("=" * 78)
    mat = pd.DataFrame(preds)
    mat["beta_market"] = base["beta_market"]
    mat["TARGET"] = base["y_true"]
    corr = mat.corr(method="spearman")
    cols = list(mat.columns)
    width = max(len(c) for c in cols) + 2
    print(f"  Spearman over all {len(base)} rows")
    print("  " + " " * width + "".join(f"{c:>{width}s}" for c in cols))
    for c in cols:
        print(f"  {c:<{width}s}"
              + "".join(f"{corr.loc[c, d]:>+{width}.3f}" for d in cols))

    model_names = list(preds)
    with_target = corr.loc[model_names, "TARGET"].abs()
    pairwise = [corr.loc[a, b] for i, a in enumerate(model_names)
                for b in model_names[i + 1:]]
    print()
    print(f"  |rho| with the target : {with_target.min():.3f} to {with_target.max():.3f}")
    print(f"  rho between models    : {min(pairwise):+.3f} to {max(pairwise):+.3f}")
    if pairwise and max(pairwise) > 3 * with_target.max():
        print("\n  The models agree with each other far more than with the label.")
        print("  Agreement between models is therefore NOT evidence about the")
        print("  target: they can agree on the same wrong thing.")

    print()
    print("=" * 78)
    print("3. AN ENSEMBLE (ordering only - the members are standardised, so the")
    print("   average is off the return scale and its MAE is an artifact)")
    print("=" * 78)

    beta_ic = cross_sectional_report(
        base.assign(y_pred=base["beta_market"]),
        rebalance_every=1).get("mean_rank_ic", float("nan"))

    def grade(y_pred, label):
        xs = cross_sectional_report(base.assign(y_pred=y_pred), rebalance_every=1)
        ic = xs.get("mean_rank_ic", float("nan"))
        beats = np.isfinite(ic) and np.isfinite(beta_ic) and ic > beta_ic
        print(f"  {label:28s} reb_IC {ic:+.4f}  "
              f"t {xs.get('rank_ic_t', float('nan')):+.2f}  "
              f"-> {'above' if beats else 'below'} beta_market ({beta_ic:+.4f})")

    z = {k: (v - np.nanmean(v)) / np.nanstd(v) for k, v in preds.items()}
    grade(np.mean(list(z.values()), axis=0), f"all {len(z)}, standardised")
    for arch in sorted({architecture(k) for k in z}):
        members = [z[k] for k in z if architecture(k) == arch]
        if len(members) > 1:
            grade(np.mean(members, axis=0), f"{arch} only ({len(members)})")

    print()
    target_sd = float(base["y_true"].std())
    print(f"  dispersion against a target SD of {target_sd:.5f}:")
    for k, v in preds.items():
        sd = float(np.nanstd(v))
        print(f"    {k:24s} SD {sd:.5f}  ({sd / target_sd:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
