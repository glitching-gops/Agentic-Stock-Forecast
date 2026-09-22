"""
tools/stage1_rerun.py — the three Stage 1 pilots, re-run under the new
conditions: standardised label, pinned threads, locked libraries, calendar-
clean panel. Terms fixed in `docs/stage1-rerun-preregistration.md`.

The FEATURES, their ingestion and construction are imported unchanged from the
three pilot tools. What changed is only the conditions they run under.

Split into jobs so independent fits can run in parallel processes, each at the
pinned two threads (the result does not depend on how many run at once - the
thread pin is per fit, and each fit is bit-identical across runs):

    python tools/stage1_rerun.py plan                   # list the jobs
    python tools/stage1_rerun.py run  <job> [<job>...]  # fit, save predictions
    python tools/stage1_rerun.py grade <job> [...]      # grade_panel_v3, save
    python tools/stage1_rerun.py report --markdown out.md

Every job writes to --out-dir (default `stage1_rerun/`, gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.baselines import FACTORS  # noqa: E402
from pipeline.delivery import ABNORMAL_COLS, LEVEL_COL, delivery_features  # noqa: E402
from pipeline.earnings import EVENT_COLS, SUE_AGE, SUE_MISSING  # noqa: E402
from pipeline.evidence_panel import BOOTSTRAP_B  # noqa: E402
from pipeline.model import EVAL_TUNE_TRIALS  # noqa: E402
from pipeline.panel import TARGET  # noqa: E402
from pipeline.reversal import RAW_COLS, RESID_COLS  # noqa: E402
from tools.stage0c_close import _from_arm  # noqa: E402
from tools.stage1_reversal import (  # noqa: E402
    COLS, SIGNAL_T, _f, as_data, attach_reversal, grade, paired, per_date_ic,
    reproduction, sha256_file, summary)
from tools.stage1b_delivery import (  # noqa: E402
    DELIVERY_CACHE, attach_delivery, delivery_long, permute_within_date)
from tools.stage1c_sue import (  # noqa: E402
    RESULTS_CACHE, attach_sue, net_of_cost, pead_book, raw_features, sue_announcements)
from tools.stage2b_pooled import cell_metrics, load_cached_panel, run_arm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL = os.path.join(ROOT, "panel_cache_clean.parquet")
BASELINE_NPZ = os.path.join(ROOT, "baseline_clean_oos.npz")
BASELINE_ARM = "new_pinned"
PREREG = os.path.join(ROOT, "docs", "stage1-rerun-preregistration.md")
OUT_DIR = os.path.join(ROOT, "stage1_rerun")

#: The frozen inputs, as recorded in the pre-registration.
PANEL_SHA256 = "c0a26d4593917946bbb7fc2a9a314ab9ccc363828b5569aad8272e2883c93a75"
BASELINE_ARRAYS_SHA256 = "bed813ac312b4165443211f375837708b19baecb0803ac5626ea67aabda17948"


def frozen_inputs_ok(panel_path: str, baseline_npz: str) -> None:
    from tools.stage1_reversal import arm_arrays_sha256

    got_p = sha256_file(panel_path)
    got_b = arm_arrays_sha256(baseline_npz, BASELINE_ARM)
    if got_p != PANEL_SHA256 or got_b != BASELINE_ARRAYS_SHA256:
        raise SystemExit(f"a frozen input changed since the pre-registration: panel "
                         f"{got_p[:16]}, baseline arrays {got_b[:16]}")

#: pilot -> (hypothesis arm, its new columns, the other arm, placebo seeds)
PILOTS = {
    "reversal": {"arms": {"resid_reversal": list(FACTORS) + RESID_COLS,
                          "raw_reversal": list(FACTORS) + RAW_COLS},
                 "hypothesis": "resid_reversal", "new_cols": RESID_COLS,
                 "seeds": tuple(range(20260921, 20260930))},
    "delivery": {"arms": {"abnormal": list(FACTORS) + ABNORMAL_COLS,
                          "level": list(FACTORS) + [LEVEL_COL]},
                 "hypothesis": "abnormal", "new_cols": ABNORMAL_COLS,
                 "seeds": tuple(range(20260914, 20260923))},
    "sue": {"arms": {"sue": list(FACTORS) + EVENT_COLS,
                     "timing": list(FACTORS) + [SUE_AGE, SUE_MISSING]},
            "hypothesis": "sue", "new_cols": EVENT_COLS,
            "seeds": tuple(range(20260920, 20260929))},
}


def jobs() -> list[str]:
    out = []
    for pilot, spec in PILOTS.items():
        out.append(f"{pilot}:baseline")
        out += [f"{pilot}:{arm}" for arm in spec["arms"]]
        out += [f"{pilot}:placebo{s}" for s in spec["seeds"]]
    return out


def pilot_panel(pilot: str, panel_path: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The clean panel with this pilot's features attached exactly as the
    pilot's own tool attaches them. Returns (panel, raw SUE features or None)."""
    raw = load_cached_panel(panel_path)
    tickers = sorted(raw["ticker"].unique())
    grid = sorted(raw["date"].astype(str).unique())
    if pilot == "reversal":
        return attach_reversal(raw), None
    if pilot == "delivery":
        return attach_delivery(raw, delivery_features(
            delivery_long(DELIVERY_CACHE, tickers), grid)), None
    ann, _ = sue_announcements(RESULTS_CACHE)
    feats = raw_features(ann, grid, tickers)
    return attach_sue(raw, feats), feats


def _path(out_dir: str, job: str, ext: str) -> str:
    return os.path.join(out_dir, job.replace(":", "__") + ext)


def run_jobs(names: list[str], panel_path: str, out_dir: str, trials: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache: dict[str, pd.DataFrame] = {}
    for job in names:
        pilot, arm = job.split(":")
        spec = PILOTS[pilot]
        if pilot not in cache:
            cache[pilot] = pilot_panel(pilot, panel_path)[0]
        panel = cache[pilot]
        if arm == "baseline":
            features = list(FACTORS)
        elif arm.startswith("placebo"):
            seed = int(arm[len("placebo"):])
            panel = permute_within_date(panel, spec["new_cols"], seed)
            features = spec["arms"][spec["hypothesis"]]
        else:
            features = spec["arms"][arm]
        t0 = time.time()
        preds, folds = run_arm(panel, "mae", "none", n_trials=trials,
                               features=features, verbose=False)
        np.savez_compressed(_path(out_dir, job, ".npz"),
                            **{c: preds[c].to_numpy() for c in COLS})
        with open(_path(out_dir, job, ".folds.json"), "w", encoding="utf-8") as f:
            json.dump(folds, f, default=float)
        print(f"{job}: {len(preds):,} rows, gammas "
              f"{[round(x['gamma'], 3) for x in folds]} ({time.time() - t0:.0f}s)",
              flush=True)


def load_job(out_dir: str, job: str) -> tuple[pd.DataFrame, list[dict]]:
    z = np.load(_path(out_dir, job, ".npz"), allow_pickle=True)
    preds = pd.DataFrame({c: z[c] for c in COLS}).astype({"date": str, "ticker": str})
    with open(_path(out_dir, job, ".folds.json"), encoding="utf-8") as f:
        return preds, json.load(f)


def grade_jobs(names: list[str], out_dir: str, bootstrap: int) -> None:
    for job in names:
        preds, _ = load_job(out_dir, job)
        t0 = time.time()
        s = summary(grade(as_data(preds), bootstrap,
                          auto_block=not job.split(":")[1].startswith("placebo")))
        with open(_path(out_dir, job, ".grade.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, default=float)
        print(f"{job}: S/W/I {s['strong']}/{s['weak']}/{s['insufficient']} "
              f"({time.time() - t0:.0f}s)", flush=True)


def resolution(preds: pd.DataFrame, folds: list[dict]) -> dict:
    per_date = preds.groupby("date")["y_pred"].nunique()
    names = preds.groupby("date")["ticker"].nunique()
    cells = preds.groupby(["ticker", "fold"])["y_pred"].nunique()
    return {"median_distinct": float(per_date.median()), "min_distinct": int(per_date.min()),
            "median_names": float(names.median()),
            "constant_cells": int((cells <= 1).sum()), "cells": int(len(cells))}


def with_raw_target(preds: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """The book tests trade RETURNS: swap the standardised label for the raw
    30-session log return on the same rows. The rank-IC rules need no swap -
    a within-date z-score is monotone within the date."""
    r = raw[["date", "ticker", TARGET]].astype({"date": str})
    return preds.drop(columns="y_true").merge(r, on=["date", "ticker"]) \
        .rename(columns={TARGET: "y_true"})


def report(out_dir: str, panel_path: str, baseline_npz: str) -> str:
    stored = _from_arm(baseline_npz, BASELINE_ARM)
    base_ic = per_date_ic(stored)
    raw = load_cached_panel(panel_path)
    res: dict = {"inputs": {
        "prereg_sha256": sha256_file(PREREG) if os.path.exists(PREREG) else "missing",
        "panel_sha256": sha256_file(panel_path),
        "baseline_sha256": sha256_file(baseline_npz)}, "pilots": {}}
    for pilot, spec in PILOTS.items():
        P: dict = {"s1": {}, "arms": {}, "placebo": []}
        b, bf = load_job(out_dir, f"{pilot}:baseline")
        rep = reproduction(stored, as_data(b))
        P["s1"] = {**rep, "exact": rep["only_stored"] == 0 and rep["only_rerun"] == 0
                   and rep["max_pred_drift"] == 0.0}
        P["arms"]["baseline"] = {"resolution": resolution(b, bf),
                                 "grade": json.load(open(_path(out_dir, f"{pilot}:baseline",
                                                               ".grade.json")))}
        ics = {}
        for arm in spec["arms"]:
            p, f = load_job(out_dir, f"{pilot}:{arm}")
            ics[arm] = per_date_ic(as_data(p))
            m = cell_metrics(p, f)
            P["arms"][arm] = {
                "resolution": resolution(p, f),
                "grade": json.load(open(_path(out_dir, f"{pilot}:{arm}", ".grade.json"))),
                "vs_baseline": paired(base_ic, ics[arm]),
                "reb_ic": m["reb_ic"], "reb_t": m["reb_t"]}
        for seed in spec["seeds"]:
            job = f"{pilot}:placebo{seed}"
            p, f = load_job(out_dir, job)
            g_path = _path(out_dir, job, ".grade.json")
            P["placebo"].append({"seed": seed, "pair": paired(base_ic, per_date_ic(as_data(p))),
                                 "resolution": resolution(p, f),
                                 "grade": json.load(open(g_path)) if os.path.exists(g_path)
                                 else None})
        h = P["arms"][spec["hypothesis"]]["vs_baseline"]
        pmax = max(r["pair"]["diff"] for r in P["placebo"])
        P["r3"] = bool(np.isfinite(h["t"]) and h["t"] >= SIGNAL_T)
        P["r2"] = bool(h["diff"] > pmax)
        P["placebo_max_diff"] = pmax
        P["placebo_beaten"] = int(sum(h["diff"] > r["pair"]["diff"] for r in P["placebo"]))
        if pilot == "sue":
            P["r5"] = paired(ics["timing"], ics["sue"])
            P["r5_pass"] = bool(P["r5"]["diff"] > 0)
            base_frame = with_raw_target(as_frame_from(stored), raw)
            sue_frame = with_raw_target(load_job(out_dir, "sue:sue")[0], raw)
            P["r4"] = net_of_cost(base_frame, sue_frame)
        P["signal"] = P["r3"] and P["r2"] and P.get("r5_pass", True)
        res["pilots"][pilot] = P
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    return render(res)


def as_frame_from(data: dict) -> pd.DataFrame:
    return pd.DataFrame({"date": np.asarray(data["dates"]).astype(str),
                         "ticker": np.asarray(data["tickers"]).astype(str),
                         "fold": data["folds"], "y_true": data["y_true"],
                         "y_pred": data["y_pred"]})


def render(r: dict) -> str:
    i = r["inputs"]
    o = ["# Stage 1 re-run — report\n",
         f"Pre-registration sha256 `{i['prereg_sha256']}`; panel `{i['panel_sha256'][:16]}`; "
         f"baseline `{i['baseline_sha256'][:16]}`.\n"]
    for pilot, P in r["pilots"].items():
        s1 = P["s1"]
        o += [f"## {pilot}\n",
              f"S1: baseline re-run {'**EXACT**' if s1['exact'] else '**NOT EXACT**'} "
              f"(max drift {s1['max_pred_drift']:.1e}, unmatched "
              f"{s1['only_stored']}/{s1['only_rerun']}).\n",
              "| arm | gain vs baseline | DK SE | t | S/W/I | median distinct / date "
              "| constant cells |", "|---|---|---|---|---|---|---|"]
        for arm, A in P["arms"].items():
            g, rs = A["grade"], A["resolution"]
            pair = A.get("vs_baseline")
            o.append(f"| {arm} | {_f(pair['diff']) if pair else '—'} | "
                     f"{_f(pair['se'], '.5f') if pair else '—'} | "
                     f"{_f(pair['t'], '+.2f') if pair else '—'} | "
                     f"{g['strong']}/{g['weak']}/{g['insufficient']} | "
                     f"{rs['median_distinct']:.0f} (min {rs['min_distinct']}) of "
                     f"{rs['median_names']:.0f} | {rs['constant_cells']}/{rs['cells']} |")
        diffs = sorted(x["pair"]["diff"] for x in P["placebo"])
        o += ["", f"Placebo (new columns permuted within date, retrained, "
                  f"{len(diffs)} seeds): gains {', '.join(_f(d) for d in diffs)}; the "
                  f"hypothesis arm beats {P['placebo_beaten']} of {len(diffs)}.\n",
              f"**R3 {'PASS' if P['r3'] else 'FAIL'}, R2 {'PASS' if P['r2'] else 'FAIL'}"
              + (f", R5 {'PASS' if P['r5_pass'] else 'FAIL'} (sue − timing "
                 f"{_f(P['r5']['diff'])}, t {_f(P['r5']['t'], '+.2f')})" if "r5" in P else "")
              + f" -> {'SIGNAL' if P['signal'] else 'NOT SIGNAL'}.**\n"]
        if "r4" in P:
            q = P["r4"]
            o.append(f"R4 (descriptive here): sue book minus baseline book, net of the "
                     f"{q['round_trip']:.4%} round trip, {_f(q['diff'])} per rebalance "
                     f"(t {_f(q['t'], '+.2f')}, n {q['n_rebalances']}).\n")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("plan", "run", "grade", "report"))
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--panel-cache", default=PANEL)
    ap.add_argument("--baseline-npz", default=BASELINE_NPZ)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--trials", type=int, default=EVAL_TUNE_TRIALS)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    if args.command == "plan":
        print("\n".join(jobs()))
        return
    if not os.path.exists(PREREG):
        raise SystemExit("no pre-registration on disk; refusing to run")
    print(f"pre-registration {sha256_file(PREREG)}", flush=True)
    if not args.out_dir.rstrip("/\\").endswith("_smoke"):
        frozen_inputs_ok(args.panel_cache, args.baseline_npz)
    names = args.jobs or jobs()
    unknown = [j for j in names if j not in jobs()]
    if unknown:
        raise SystemExit(f"unknown jobs {unknown}; see `plan`")
    if args.command == "run":
        run_jobs(names, args.panel_cache, args.out_dir, args.trials)
    elif args.command == "grade":
        grade_jobs(names, args.out_dir, args.bootstrap)
    else:
        text = report(args.out_dir, args.panel_cache, args.baseline_npz)
        print(text)
        if args.markdown:
            with open(args.markdown, "w", encoding="utf-8") as f:
                f.write(text + "\n")


if __name__ == "__main__":
    main()
