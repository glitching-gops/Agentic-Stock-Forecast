"""
tools/stage2_fallback_report.py — Parts B and C, scored against their
pre-registration (docs/stage2-fallback-conformal-preregistration.md, hashed
before any Part B panel was built or any Part C coverage computed).

    panel     B0 (does the production-built v5 panel equal the post-hoc
              diagnostic's fallback panel?) and B1 (is the fingerprint
              structurally gone — do the 11 carry the NULL rate of the 73?)
    report    B2-B5 from the pooled predictions, and Part C: coverage by fold
              and overall for the constant and the spread-normalised interval
              on IDENTICAL rows, in price space, with widths

    STAGE2_PREREG=docs/stage2-fallback-conformal-preregistration.md \\
      python tools/stage2_fallback_report.py panel --v5 stage2/v5/panel_v5.parquet \\
        --fallback stage2/diag/panel_fallback.parquet --json stage2/v5/panel_check.json
    STAGE2_PREREG=... python tools/stage2_fallback_report.py report \\
        --panel stage2/v5/panel_v5.parquet --v5 stage2/v5/v5_a.npz \\
        --old stage2/old_a.npz --v4 stage2/ref_a.npz --real11 stage2/v5/real11.npz \\
        --placebo stage2/v5/placebo_0.npz stage2/v5/placebo_1.npz stage2/v5/placebo_2.npz \\
        --json stage2/v5/report.json

The deciding rules are restated here only as the pre-registration fixes them;
nothing below is a threshold chosen after seeing a number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.stage2_measure import dk, load_preds, per_date_ic, provenance, sha256_file  # noqa: E402

#: Pre-registered (§2.2): the random-11 NULL arms within this of the v5 model,
#: and Part 0's original bound on the benchmark delta.
PLACEBO_BAND = 0.005
DELTA_BOUND = 0.005
T_BOUND = 2.0
#: B1: the thin names' NULL share within this of the others', in share units.
NULL_SHARE_TOL = 0.005


# ── panel ─────────────────────────────────────────────────────────────────────


def cmd_panel(args) -> None:
    from pipeline.baselines import FACTORS
    from pipeline.signals import SECTOR_REL_COLS

    meta = provenance(args.v5)
    v5 = pd.read_parquet(args.v5)
    fb = pd.read_parquet(args.fallback)
    keys = ["date", "ticker"]
    cols = list(FACTORS) + ["target_return", "target_excess_return"]
    m = v5[keys + cols].merge(fb[keys + cols], on=keys, how="outer",
                              suffixes=("_v5", "_fb"), indicator=True)
    both = m["_merge"] == "both"
    diffs = {}
    for c in cols:
        a, b = m.loc[both, f"{c}_v5"].to_numpy(float), m.loc[both, f"{c}_fb"].to_numpy(float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            d = np.abs(a - b)
            diffs[c] = {"rows_differing": int((~same).sum()),
                        "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else None,
                        "nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum()),
                        "tickers": sorted(m.loc[both, "ticker"][~same].unique())[:20]}
    b0 = {"rows_v5": int(len(v5)), "rows_fallback": int(len(fb)),
          "unmatched": int((~both).sum()), "columns_differing": diffs,
          "identical": not diffs and int((~both).sum()) == 0}

    spec = v5.groupby("ticker")["benchmark_sector_specific"].max()
    thin = sorted(spec[spec == 0].index)
    null_any = v5[list(SECTOR_REL_COLS)].isna().any(axis=1)
    # "beyond the warm-up rows every ticker shares": each ticker's first 20
    # rows (the longest sector_rel window) are excluded for everyone.
    rank = v5.sort_values("date").groupby("ticker").cumcount()
    past_warmup = rank.reindex(v5.index) >= 20
    share = lambda mask: float(null_any[mask & past_warmup].mean())   # noqa: E731
    is_thin = v5["ticker"].isin(thin)
    b1 = {"thin": thin, "null_share_thin": share(is_thin),
          "null_share_others": share(~is_thin),
          "null_share_thin_incl_warmup": float(null_any[is_thin].mean()),
          "null_share_others_incl_warmup": float(null_any[~is_thin].mean())}
    b1["holds"] = abs(b1["null_share_thin"] - b1["null_share_others"]) <= NULL_SHARE_TOL
    out = {"provenance": meta, "fallback_sha256": sha256_file(args.fallback),
           "B0": b0, "B1": b1}
    args.json.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


# ── report ────────────────────────────────────────────────────────────────────


def paired(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat({"a": a, "b": b}, axis=1).dropna()
    return dk(j["a"] - j["b"])


def part_c(inv: pd.DataFrame) -> dict:
    from pipeline.pooled_shadow import coverage_gate

    ok = (np.isfinite(inv["pred_return"]) & np.isfinite(inv["y_raw"])
          & np.isfinite(inv["close"]) & (inv["close"] > 0))
    has_spread = ok & np.isfinite(inv["interval_spread"]) & (inv["interval_spread"] > 0)
    rows = inv[has_spread]
    args = (rows["y_raw"], rows["pred_return"], rows["fold"])
    before = coverage_gate(*args, price=rows["close"])
    after = coverage_gate(*args, spread=rows["interval_spread"], price=rows["close"])
    # Realised cross-sectional dispersion of the label per checked fold, for
    # the descriptive width-tracks-dispersion prediction (§3.4).
    disp = (rows.groupby(["fold", "date"])["y_raw"].std().groupby("fold").mean())
    checked = [f["fold"] for f in after["per_fold"]]
    w_after = [f["mean_width_log"] for f in after["per_fold"]]
    d = [float(disp.loc[k]) for k in checked]
    rho = float(pd.Series(w_after).rank().corr(pd.Series(d).rank())) if len(d) > 1 else None
    # within the checked folds, date by date: does the band widen with dispersion?
    by_date = rows[rows["fold"].isin(checked)].groupby("date").agg(
        spread=("interval_spread", "first"), disp=("y_raw", "std"))
    return {
        "rows_with_prediction": int(ok.sum()),
        "rows_excluded_no_spread": int((ok & ~has_spread).sum()),
        "before_constant": before, "after_spread_normalised": after,
        "fold_realised_dispersion": dict(zip(checked, d)),
        "spearman_width_vs_fold_dispersion": rho,
        "date_corr_spread_vs_realised_dispersion": float(
            by_date["spread"].corr(by_date["disp"])),
        "pass": after["status"] == "PASS",
    }


def cmd_report(args) -> None:
    from pipeline import pooled

    meta = provenance(args.panel)
    raw = pd.read_parquet(args.panel)
    causal = pooled.causal_frame(raw)

    def inv(path):
        return pooled.invert(load_preds(path), causal)

    files = {"v5": args.v5, "old": args.old, "v4": args.v4, "real11": args.real11,
             **{f"placebo_{k}": p for k, p in enumerate(args.placebo)}}
    preds = {k: inv(p) for k, p in files.items()}
    ic = {k: per_date_ic(v) for k, v in preds.items()}
    out: dict = {"provenance": meta,
                 "files": {k: {"path": str(p), "sha256": sha256_file(p)}
                           for k, p in files.items()}}
    placebo_d = {k: paired(ic[k], ic["v5"]) for k in ic if k.startswith("placebo_")}
    real_d = paired(ic["real11"], ic["v5"])
    b2 = {"random11_minus_v5": placebo_d, "real11_minus_v5": real_d,
          "holds": (all(abs(d["mean"]) <= PLACEBO_BAND for d in placebo_d.values())
                    and real_d["mean"] > max(d["mean"] for d in placebo_d.values()))}
    b3 = paired(ic["v5"], ic["old"])
    b3["holds"] = abs(b3["mean"]) <= DELTA_BOUND and abs(b3["t"]) < T_BOUND
    b4 = paired(ic["v5"], ic["v4"])
    b5 = dk(ic["v5"])
    b5["holds"] = abs(b5["t"]) < T_BOUND
    out["part_b"] = {"B2": b2, "B3": b3, "B4_descriptive": b4, "B5": b5,
                     "cs_ic": {k: dk(v) for k, v in ic.items()},
                     "by_fold_v5_minus_v4": {
                         int(f): dk(per_date_ic(preds["v5"][preds["v5"]["fold"] == f])
                                    - per_date_ic(preds["v4"][preds["v4"]["fold"] == f]))
                         for f in sorted(preds["v5"]["fold"].unique())},
                     "pass": bool(b2["holds"] and b3["holds"] and b5["holds"])}
    if args.b1_json:
        b1 = json.loads(Path(args.b1_json).read_text())["B1"]
        out["part_b"]["B1"] = b1
        out["part_b"]["pass"] = bool(out["part_b"]["pass"] and b1["holds"])

    v5 = preds["v5"].merge(raw[["date", "ticker", "close"]], on=["date", "ticker"],
                           how="left").merge(pooled.spread_frame(raw), on="date", how="left")
    out["part_c"] = part_c(v5)
    # the same interval method on the NULL (v4) model, for the record only
    v4 = preds["v4"].merge(raw[["date", "ticker", "close"]], on=["date", "ticker"],
                           how="left").merge(pooled.spread_frame(raw), on="date", how="left")
    out["part_c_on_v4_for_reference"] = {
        k: v for k, v in part_c(v4).items() if k in ("before_constant",
                                                      "after_spread_normalised")}
    args.json.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float)[:12000])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("panel")
    p.add_argument("--v5", type=Path, required=True)
    p.add_argument("--fallback", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--panel", type=Path, required=True)
    r.add_argument("--v5", type=Path, required=True)
    r.add_argument("--old", type=Path, required=True)
    r.add_argument("--v4", type=Path, required=True)
    r.add_argument("--real11", type=Path, required=True)
    r.add_argument("--placebo", type=Path, nargs="+", required=True)
    r.add_argument("--b1-json", type=Path, default=None)
    r.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    {"panel": cmd_panel, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
