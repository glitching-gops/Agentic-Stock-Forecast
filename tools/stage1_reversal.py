"""
tools/stage1_reversal.py — Stage 1, Pilot 1: multi-lookback residual reversal.

Method, arms, the stop condition and every decision rule are fixed in
`docs/stage1-preregistration.md`, written before this ran on real data.

Three arms on the pooled × MAE, no-ticker model, all through the Stage 0c
evidence harness:

    (a) baseline        FACTORS               REUSED from stage2b_pooled_oos.npz
    (b) resid_reversal  FACTORS + rev_resid_k
    (c) raw_reversal    FACTORS + rev_raw_k

Arm (a) is ALSO re-run, and must reproduce the stored predictions exactly
(S1). If it does not, the code or the inputs have moved underneath the stored
baseline, and the pilot stops before (b) and (c) are graded against it.

SANDBOXED. Reads the cached panel and the Stage 2b predictions; writes an .npz
of held-out predictions and a markdown report. Nothing the API, the web app or
either scheduled job reads.

    python tools/stage1_reversal.py --smoke --markdown stage1_smoke.md
    python tools/stage1_reversal.py --markdown stage1_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.baselines import FACTORS  # noqa: E402
from pipeline.evaluation import rank_ic  # noqa: E402
from pipeline.evidence_panel import (  # noqa: E402
    BOOTSTRAP_B,
    driscoll_kraay_se,
    grade_panel_v3,
)
from pipeline.evidence_shrinkage import BLOCK_LENGTH_SESSIONS  # noqa: E402
from pipeline.model import EVAL_TUNE_TRIALS  # noqa: E402
from pipeline.panel import TARGET, cross_sectional_zscore  # noqa: E402
from pipeline.reversal import (  # noqa: E402
    LOOKBACKS,
    RAW_COLS,
    RESID_COLS,
    REVERSAL_COLS,
    reversal_features,
)
from pipeline.signals import HORIZON_SESSIONS  # noqa: E402
from tools.stage0c_close import BREAK_EVEN, _from_arm, shuffled_placebo  # noqa: E402
from tools.stage2b_pooled import cell_metrics, load_cached_panel, run_arm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
STAGE2B_NPZ = os.path.join(ROOT, "stage2b_pooled_oos.npz")
PRED_OUT = os.path.join(ROOT, "stage1_reversal_oos.npz")
PREREG = os.path.join(ROOT, "docs", "stage1-preregistration.md")

# The frozen inputs, as recorded in the pre-registration.
BASELINE_ARM = "pooled_mae_noticker"
BASELINE_SHA256 = "67cc72bee795ea9b7a77ad7c5e49de070b08ff774ade43f09a0613a6e04012e1"
PANEL_SHA256 = "990c0a07d5391e9b6b09942626583b74c5b15321d0bd51391c5a515f810e0ecb"

ARM_FEATURES: dict[str, list[str]] = {
    "baseline": list(FACTORS),
    "resid_reversal": list(FACTORS) + RESID_COLS,
    "raw_reversal": list(FACTORS) + RAW_COLS,
}
REVERSAL_ARMS = ("resid_reversal", "raw_reversal")

PLACEBO_SEEDS: tuple[int, ...] = tuple(range(20260913, 20260922))   # nine
REPRO_TOL = 1e-9
DK_LAGS = HORIZON_SESSIONS          # consecutive dates share 29 of 30 sessions
SIGNAL_T = 2.0
SWEEP_MIN_TRAIN = (380, 420, 460, 500, 540, 580)
COLS = ("date", "ticker", "y_true", "y_pred", "fold")


# ── identities ────────────────────────────────────────────────────────────────


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arm_arrays_sha256(npz_path: str, arm: str) -> str:
    """The hash recorded in the pre-registration, computed the same way."""
    z = np.load(npz_path, allow_pickle=True)
    h = hashlib.sha256()
    for c in COLS:
        a = np.ascontiguousarray(z[f"{arm}__{c}"])
        h.update(a.astype(str if c in ("date", "ticker") else float).tobytes())
    return h.hexdigest()


# ── the panel ─────────────────────────────────────────────────────────────────


def attach_reversal(panel: pd.DataFrame) -> pd.DataFrame:
    """
    The panel with the eight reversal columns joined on and standardised
    within each date by the same ``cross_sectional_zscore`` every other pooled
    feature goes through. Computed from ``close``, which no z-score touches.
    """
    feats = reversal_features(panel)
    out = panel.merge(feats, on=["date", "ticker"], how="left")
    out = cross_sectional_zscore(out, REVERSAL_COLS)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ── predictions ───────────────────────────────────────────────────────────────


def as_data(preds: pd.DataFrame) -> dict:
    return {"dates": preds["date"].astype(str).to_numpy(),
            "tickers": preds["ticker"].astype(str).to_numpy(),
            "folds": preds["fold"].to_numpy(),
            "y_true": preds["y_true"].to_numpy(dtype=float),
            "y_pred": preds["y_pred"].to_numpy(dtype=float)}


def as_frame(data: dict) -> pd.DataFrame:
    return pd.DataFrame({"date": data["dates"], "ticker": data["tickers"],
                         "fold": np.asarray(data["folds"]).astype(int),
                         "y_true": data["y_true"], "y_pred": data["y_pred"]})


def reproduction(stored: dict, rerun: dict) -> dict:
    """S1: the re-run baseline must be the stored baseline, row for row."""
    a, b = as_frame(stored), as_frame(rerun)
    m = a.merge(b, on=["date", "ticker", "fold"], how="outer",
                suffixes=("_stored", "_rerun"), indicator=True)
    both = m[m["_merge"] == "both"]
    drift = (float(np.max(np.abs(both["y_pred_stored"] - both["y_pred_rerun"])))
             if len(both) else float("nan"))
    truth = (float(np.max(np.abs(both["y_true_stored"] - both["y_true_rerun"])))
             if len(both) else float("nan"))
    only_stored = int((m["_merge"] == "left_only").sum())
    only_rerun = int((m["_merge"] == "right_only").sum())
    return {"rows_stored": len(a), "rows_rerun": len(b),
            "only_stored": only_stored, "only_rerun": only_rerun,
            "max_pred_drift": drift, "max_truth_drift": truth,
            "passed": (only_stored == 0 and only_rerun == 0
                       and drift <= REPRO_TOL and truth <= REPRO_TOL)}


# ── grading ───────────────────────────────────────────────────────────────────


def grade(data: dict, bootstrap: int, auto_block: bool):
    """Stage 0c's settings exactly: block 30, seed 20260908, alpha 0.10."""
    return grade_panel_v3(data["dates"], data["tickers"], data["folds"],
                          data["y_true"], data["y_pred"],
                          block=BLOCK_LENGTH_SESSIONS, n_bootstrap=bootstrap,
                          break_even=BREAK_EVEN, compute_auto_block=auto_block)


def summary(g) -> dict:
    c = g.counts()
    return {
        "n_usable": g.n_usable, "tickers": g.tickers_total,
        "mu_hat": g.mu_hat, "se_boot": g.mu_se_bootstrap, "z": g.z_bootstrap,
        "tau2": g.tau2_reml, "counts": c,
        "strong": c.get("STRONG", 0), "weak": c.get("WEAK", 0),
        "insufficient": c.get("INSUFFICIENT", 0),
        "rw": int(sum(r.rw_rejected for r in g.rows)),
        "bh": int(sum(r.bh_significant for r in g.rows)),
        "by": int(sum(r.by_significant for r in g.rows)),
        "cs_ic": g.cross_sectional_ic, "cs_ic_se": g.cross_sectional_ic_se,
        "graded": sorted(r.ticker for r in g.rows if r.grade in ("STRONG", "WEAK")),
        "strong_set": sorted(r.ticker for r in g.rows if r.grade == "STRONG"),
        "degenerate": bool(g.degeneracy.degenerate),
        "block_auto": g.block_length_auto,
        "runtime": g.runtime_seconds,
    }


def per_date_ic(data: dict) -> pd.Series:
    """Per-date cross-sectional rank IC, indexed by date. Never pooled."""
    df = as_frame(data)
    return pd.Series({d: rank_ic(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
                      for d, g in df.groupby("date", sort=True)}, dtype=float)


def paired(a: pd.Series, b: pd.Series) -> dict:
    """(b - a) per date on the dates both define, with DK SEs at 30 lags —
    the pre-registered test — and at the default lag rule, which decides
    nothing."""
    j = pd.concat({"a": a, "b": b}, axis=1).dropna()
    diff = (j["b"] - j["a"]).to_numpy()
    dates = j.index.to_numpy()
    mean, se, lags = driscoll_kraay_se(dates, diff, max_lag=DK_LAGS)
    _, se_def, lags_def = driscoll_kraay_se(dates, diff)
    return {"n_dates": len(j), "mean_a": float(j["a"].mean()),
            "mean_b": float(j["b"].mean()), "diff": mean, "se": se,
            "t": mean / se if se and np.isfinite(se) and se > 0 else float("nan"),
            "lags": lags, "se_default": se_def,
            "t_default": (mean / se_def if se_def and np.isfinite(se_def)
                          and se_def > 0 else float("nan")),
            "lags_default": lags_def}


def feature_ics(panel: pd.DataFrame, dates_oos: set[str]) -> list[dict]:
    """R5: each reversal feature's own per-date rank IC against the target,
    over arm (a)'s out-of-sample dates. Raw features, not z-scored: the z-score
    clips at 3, which alters ranks at the tails."""
    feats = reversal_features(panel)
    f = feats.merge(panel[["date", "ticker", TARGET]], on=["date", "ticker"])
    f = f[f["date"].astype(str).isin(dates_oos) & np.isfinite(f[TARGET])]
    out = []
    for col in REVERSAL_COLS:
        s = pd.Series({d: rank_ic(g[TARGET].to_numpy(dtype=float),
                                  g[col].to_numpy(dtype=float))
                       for d, g in f.groupby("date", sort=True)}, dtype=float).dropna()
        mean, se, _ = driscoll_kraay_se(s.index.to_numpy(), s.to_numpy(),
                                        max_lag=DK_LAGS)
        out.append({"feature": col, "n_dates": len(s), "ic": mean, "se": se,
                    "t": mean / se if se and se > 0 else float("nan")})
    return out


def placebo(data: dict, seeds: tuple[int, ...], bootstrap: int) -> list[dict]:
    rows = []
    for seed in seeds:
        t0 = time.time()
        s = summary(grade(shuffled_placebo(data, seed=seed), bootstrap,
                          auto_block=False))
        rows.append({"seed": seed, **s})
        print(f"    placebo seed {seed}: STRONG {s['strong']} WEAK {s['weak']} "
              f"RW {s['rw']} tau2 {s['tau2']:.5f} mu {s['mu_hat']:+.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return rows


def sweep(panel: pd.DataFrame, arm: str, trials: int) -> list[dict]:
    """R6, run only when R1 and R2 both hold."""
    rows = []
    for mt in SWEEP_MIN_TRAIN:
        for name in ("baseline", arm):
            preds, folds = run_arm(panel, "mae", "none", n_trials=trials,
                                   min_train=mt, features=ARM_FEATURES[name],
                                   verbose=False)
            m = cell_metrics(preds, folds)
            rows.append({"min_train": mt, "arm": name, "reb_ic": m["reb_ic"],
                         "reb_t": m["reb_t"], "cs_ic": m["cs_rank_ic"]})
            print(f"    sweep min_train {mt} {name}: reb t {m['reb_t']:+.2f}",
                  flush=True)
    return rows


# ── the verdicts ──────────────────────────────────────────────────────────────


def verdicts(S: dict, placebo_rows: list[dict], pairs: dict,
             feats: list[dict], repro: dict) -> dict:
    base = set(S["baseline"]["graded"])
    newly = {arm: sorted(set(S[arm]["graded"]) - base) for arm in REVERSAL_ARMS}
    lost = {arm: sorted(base - set(S[arm]["graded"])) for arm in REVERSAL_ARMS}
    b = S["resid_reversal"]
    pmax = ({k: max(r[k] for r in placebo_rows) for k in ("strong", "rw", "tau2")}
            if placebo_rows else {})
    r2 = bool(placebo_rows) and (b["strong"] > pmax["strong"]
                                 and b["rw"] > pmax["rw"]
                                 and b["tau2"] > pmax["tau2"])
    r1 = {arm: bool(newly[arm]) for arm in REVERSAL_ARMS}
    t_ba = pairs["resid_vs_baseline"]["t"]
    t_bc = pairs["resid_vs_raw"]["t"]
    signal = r1["resid_reversal"] and r2 and np.isfinite(t_ba) and t_ba >= SIGNAL_T
    if not np.isfinite(t_bc):
        r4 = "undetermined"
    elif t_bc >= SIGNAL_T:
        r4 = "RESIDUAL BETTER"
    elif t_bc <= -SIGNAL_T:
        r4 = "RAW BETTER"
    else:
        r4 = "COMPARABLE"

    failed = []
    if not r1["resid_reversal"]:
        failed.append("R1 (no ticker moved out of INSUFFICIENT)")
    if not r2:
        failed.append("R2 (did not clear the nine-draw placebo)")
    if not (np.isfinite(t_ba) and t_ba >= SIGNAL_T):
        failed.append(f"R3 (paired cross-sectional t {t_ba:+.2f} < {SIGNAL_T:.1f})")

    pba, pbc = pairs["resid_vs_baseline"], pairs["resid_vs_raw"]
    predictions = {
        "P1 arm (a) reproduces": bool(repro.get("passed")),
        # No "|" in these labels: they are rendered inside a markdown table.
        "P2 no tradeable improvement: abs(d) < 0.005 and abs(t) < 2":
            bool(abs(pba["diff"]) < 0.005 and abs(pba["t"]) < 2),
        "P3 residual and raw comparable: abs(d) < 0.003 and abs(t) < 2":
            bool(abs(pbc["diff"]) < 0.003 and abs(pbc["t"]) < 2),
        "P4 at most 1 STRONG per reversal arm, and R2 fails":
            bool(S["resid_reversal"]["strong"] <= 1
                 and S["raw_reversal"]["strong"] <= 1 and not r2),
        "P5 every feature abs(IC) < 0.02":
            bool(all(abs(f["ic"]) < 0.02 for f in feats)),
    }
    return {"newly": newly, "lost": lost, "r1": r1, "r2": r2, "pmax": pmax,
            "signal": signal, "r4": r4, "failed": failed,
            "predictions": predictions}


# ── the report ────────────────────────────────────────────────────────────────


def _f(v, spec: str = "+.5f") -> str:
    return ("n/a" if v is None or not isinstance(v, (int, float, np.floating))
            or not np.isfinite(v) else format(v, spec))


def render(inputs: dict, repro: dict, S: dict, metrics: dict, pairs: dict,
           feats: list[dict], placebo_rows: list[dict], V: dict,
           sweep_rows: list[dict], args, runtime: float) -> str:
    o = ["# Stage 1, Pilot 1 — multi-lookback residual reversal: run report\n"]
    o.append(f"Pre-registration sha256 at run time: `{inputs['prereg_sha256']}`.  ")
    o.append(f"Panel sha256 {'matches' if inputs['panel_ok'] else '**DOES NOT MATCH**'} "
             f"the frozen value; arm (a) arrays sha256 "
             f"{'match' if inputs['baseline_ok'] else '**DO NOT MATCH**'}.  ")
    o.append(f"B = {args.bootstrap}, trials = {args.trials}, placebo draws = "
             f"{len(placebo_rows)}, DK lags = {DK_LAGS}. "
             f"{'**SMOKE RUN — not a measurement.**' if args.smoke else ''}\n")

    o.append("## S1 — does arm (a) reproduce?\n")
    o.append("| rows stored | rows re-run | only stored | only re-run | max Δ y_pred | max Δ y_true | S1 |")
    o.append("|---|---|---|---|---|---|---|")
    o.append(f"| {repro['rows_stored']:,} | {repro['rows_rerun']:,} | "
             f"{repro['only_stored']} | {repro['only_rerun']} | "
             f"{repro['max_pred_drift']:.1e} | {repro['max_truth_drift']:.1e} | "
             f"**{'PASS' if repro['passed'] else 'FAIL'}** |\n")

    o.append("## The three arms, graded by `grade_panel_v3`\n")
    o.append("| arm | n_usable | mu_hat | boot SE | z | tau2 | STRONG | WEAK | "
             "INSUFF | RW | BH | BY | cs IC (DK SE) |")
    o.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, s in S.items():
        o.append(f"| {arm} | {s['n_usable']}/{s['tickers']} | {_f(s['mu_hat'])} | "
                 f"{_f(s['se_boot'], '.5f')} | {_f(s['z'], '+.2f')} | "
                 f"{_f(s['tau2'], '.5f')} | **{s['strong']}** | {s['weak']} | "
                 f"{s['insufficient']} | {s['rw']} | {s['bh']} | {s['by']} | "
                 f"{_f(s['cs_ic'])} ({_f(s['cs_ic_se'], '.5f')}) |")
    o.append("")
    for arm in REVERSAL_ARMS:
        o.append(f"- **{arm}** graded (STRONG/WEAK): {S[arm]['graded'] or 'none'}; "
                 f"newly out of INSUFFICIENT vs (a): {V['newly'][arm] or 'none'}; "
                 f"lost vs (a): {V['lost'][arm] or 'none'}.")
    o.append(f"- **baseline** graded: {S['baseline']['graded'] or 'none'}.\n")

    o.append("## R2 — the within-date placebo on the residual arm\n")
    o.append("| seed | STRONG | WEAK | RW | tau2 | mu_hat | z |")
    o.append("|---|---|---|---|---|---|---|")
    for r in placebo_rows:
        o.append(f"| {r['seed']} | {r['strong']} | {r['weak']} | {r['rw']} | "
                 f"{_f(r['tau2'], '.5f')} | {_f(r['mu_hat'])} | {_f(r['z'], '+.2f')} |")
    if V["pmax"]:
        b = S["resid_reversal"]
        o.append(f"| **max** | **{V['pmax']['strong']}** |  | **{V['pmax']['rw']}** | "
                 f"**{_f(V['pmax']['tau2'], '.5f')}** |  |  |")
        o.append(f"| **resid arm** | **{b['strong']}** | {b['weak']} | **{b['rw']}** | "
                 f"**{_f(b['tau2'], '.5f')}** | {_f(b['mu_hat'])} | {_f(b['z'], '+.2f')} |")
    o.append(f"\n**R2: {'PASS' if V['r2'] else 'FAIL'}** — the residual arm must "
             f"beat the placebo maximum on STRONG, Romano-Wolf rejections AND "
             f"tau2.\n")

    o.append("## R3 / R4 — paired per-date cross-sectional rank IC\n")
    o.append("| comparison | dates | mean first | mean second | Δ | DK SE (30 lags) "
             "| **t** | SE, default lags | t, default |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    labels = {"resid_vs_baseline": "(b) resid − (a) baseline",
              "raw_vs_baseline": "(c) raw − (a) baseline",
              "resid_vs_raw": "(b) resid − (c) raw"}
    for key, lab in labels.items():
        p = pairs[key]
        o.append(f"| {lab} | {p['n_dates']:,} | {_f(p['mean_a'])} | {_f(p['mean_b'])} | "
                 f"{_f(p['diff'])} | {_f(p['se'], '.5f')} | **{_f(p['t'], '+.2f')}** | "
                 f"{_f(p['se_default'], '.5f')} ({p['lags_default']}) | "
                 f"{_f(p['t_default'], '+.2f')} |")
    o.append(f"\n**R4: {V['r4']}.**\n")

    o.append("## R5 — each feature's own cross-sectional IC, against Da-Liu-Schaumburg\n")
    o.append("| k | raw IC | raw t | resid IC | resid t | abs(resid) / abs(raw) |")
    o.append("|---|---|---|---|---|---|")
    by = {f["feature"]: f for f in feats}
    for k in LOOKBACKS:
        r, e = by[f"rev_raw_{k}"], by[f"rev_resid_{k}"]
        ratio = (abs(e["ic"]) / abs(r["ic"]) if r["ic"] and np.isfinite(r["ic"])
                 and r["ic"] != 0 else float("nan"))
        o.append(f"| {k} | {_f(r['ic'])} | {_f(r['t'], '+.2f')} | {_f(e['ic'])} | "
                 f"{_f(e['t'], '+.2f')} | {_f(ratio, '.2f')} |")
    o.append("\nDLS (2014, US, three-factor residual): residual reversal 1.34%/month "
             "(t 9.28) against raw 0.33% (t 1.37) — about 4x on alpha, 6.8x on t. "
             "Descriptive only; decides nothing.\n")

    o.append("## Stage 2b's cell metrics per arm\n")
    o.append("| arm | constant cells | reb IC | reb t | rebalances | cs IC | OOS MAE | "
             "train/OOS gap | cs IC by fold |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    for arm, m in metrics.items():
        o.append(f"| {arm} | {m['constant_cells']}/{m['cells']} | {_f(m['reb_ic'], '+.4f')} | "
                 f"{_f(m['reb_t'], '+.2f')} | {m['n_rebalances']} | "
                 f"{_f(m['cs_rank_ic'], '+.4f')} | {m['mae']:.5f} | {_f(m['gap'])} | "
                 f"{m['cs_rank_ic_by_fold']} |")
    o.append("")

    o.append("## Verdicts, as pre-registered\n")
    o.append(f"- **R1** residual arm: {'PASS' if V['r1']['resid_reversal'] else 'FAIL'}; "
             f"raw arm: {'PASS' if V['r1']['raw_reversal'] else 'FAIL'}.")
    o.append(f"- **R2** placebo: {'PASS' if V['r2'] else 'FAIL'}.")
    o.append(f"- **R3** reported as SIGNAL: **{'YES' if V['signal'] else 'NO'}**"
             + (f" — failed: {'; '.join(V['failed'])}." if V["failed"] else "."))
    o.append(f"- **R4** residual vs raw: **{V['r4']}**.")
    o.append(f"- **R6** sweep: {'run' if sweep_rows else 'not triggered'}.\n")
    o.append("| prediction | held |")
    o.append("|---|---|")
    for k, v in V["predictions"].items():
        o.append(f"| {k} | {'yes' if v else '**no**'} |")
    o.append("")
    if sweep_rows:
        o.append("## R6 — min_train sweep\n")
        o.append("| min_train | arm | reb IC | reb t | cs IC |")
        o.append("|---|---|---|---|---|")
        for r in sweep_rows:
            o.append(f"| {r['min_train']} | {r['arm']} | {_f(r['reb_ic'], '+.4f')} | "
                     f"{_f(r['reb_t'], '+.2f')} | {_f(r['cs_ic'], '+.4f')} |")
        o.append("")
    o.append(f"Every SE above is likely too small: Politis-White puts the panel's "
             f"dependence at 35.8-62.5 sessions. Runtime {runtime / 60:.1f} min.")
    return "\n".join(o)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--stage2b-npz", default=STAGE2B_NPZ)
    ap.add_argument("--npz", default=PRED_OUT)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--trials", type=int, default=EVAL_TUNE_TRIALS)
    ap.add_argument("--placebo-draws", type=int, default=len(PLACEBO_SEEDS))
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 trial, B = 20, 1 placebo draw, no sweep; S1 not enforced")
    args = ap.parse_args()
    if args.smoke:
        args.trials, args.bootstrap, args.placebo_draws = 1, 20, 1
        args.no_sweep = True
        if args.npz == PRED_OUT:
            args.npz = os.path.join(ROOT, "stage1_smoke_oos.npz")

    started = time.time()
    inputs = {
        "prereg_sha256": sha256_file(PREREG) if os.path.exists(PREREG) else "missing",
        "panel_sha256": sha256_file(args.panel_cache),
        "baseline_sha256": arm_arrays_sha256(args.stage2b_npz, BASELINE_ARM),
    }
    inputs["panel_ok"] = inputs["panel_sha256"].lower() == PANEL_SHA256
    inputs["baseline_ok"] = inputs["baseline_sha256"].lower() == BASELINE_SHA256
    print(f"pre-registration {inputs['prereg_sha256']}\n"
          f"panel ok {inputs['panel_ok']}, arm (a) arrays ok {inputs['baseline_ok']}",
          flush=True)
    if not (inputs["panel_ok"] and inputs["baseline_ok"]) and not args.smoke:
        raise SystemExit("A frozen input has changed since the pre-registration; "
                         "refusing to run against a different baseline.")

    raw_panel = load_cached_panel(args.panel_cache)
    panel = attach_reversal(raw_panel)
    stored = _from_arm(args.stage2b_npz, BASELINE_ARM)
    print(f"panel {panel.shape}; arm (a) stored {len(stored['y_pred']):,} rows",
          flush=True)

    frames: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict] = {}
    print("\n(a) re-running the baseline arm for S1 ...", flush=True)
    preds, folds = run_arm(panel, "mae", "none", n_trials=args.trials,
                           features=ARM_FEATURES["baseline"])
    frames["baseline_repro"] = preds
    metrics["baseline"] = cell_metrics(preds, folds)
    repro = reproduction(stored, as_data(preds))
    print(f"  S1 {'PASS' if repro['passed'] else 'FAIL'}: max drift "
          f"{repro['max_pred_drift']:.2e}, only-stored {repro['only_stored']}, "
          f"only-rerun {repro['only_rerun']}", flush=True)
    if not repro["passed"] and not args.smoke:
        raise SystemExit(
            f"S1 FAILED — the re-run baseline does not reproduce the stored one "
            f"({repro}). Stopping before (b) and (c) are graded, as pre-registered.")

    for arm in REVERSAL_ARMS:
        print(f"\n({'b' if arm == 'resid_reversal' else 'c'}) {arm} ...", flush=True)
        preds, folds = run_arm(panel, "mae", "none", n_trials=args.trials,
                               features=ARM_FEATURES[arm])
        frames[arm] = preds
        metrics[arm] = cell_metrics(preds, folds)
        m = metrics[arm]
        print(f"  -> {m['constant_cells']}/{m['cells']} constant, reb t "
              f"{m['reb_t']:+.2f}, cs IC {m['cs_rank_ic']:+.4f}, MAE {m['mae']:.5f}",
              flush=True)

    np.savez_compressed(args.npz, **{f"{k}__{c}": v[c].to_numpy()
                                     for k, v in frames.items() for c in COLS})
    print(f"wrote {args.npz}", flush=True)

    data = {"baseline": stored,
            "resid_reversal": as_data(frames["resid_reversal"]),
            "raw_reversal": as_data(frames["raw_reversal"])}
    for arm in REVERSAL_ARMS:
        same = len(as_frame(data[arm]).merge(as_frame(stored),
                                             on=["date", "ticker", "fold"]))
        print(f"  {arm}: {same:,} of {len(stored['y_pred']):,} rows shared with (a)",
              flush=True)

    S = {}
    for arm, d in data.items():
        print(f"\ngrading {arm} at B = {args.bootstrap} ...", flush=True)
        S[arm] = summary(grade(d, args.bootstrap, auto_block=not args.smoke))
        s = S[arm]
        print(f"  n_usable {s['n_usable']}  mu {s['mu_hat']:+.5f}  z {s['z']:+.2f}  "
              f"tau2 {s['tau2']:.5f}  {s['counts']}  RW {s['rw']}  "
              f"({s['runtime']:.0f}s)", flush=True)

    ics = {arm: per_date_ic(d) for arm, d in data.items()}
    pairs = {"resid_vs_baseline": paired(ics["baseline"], ics["resid_reversal"]),
             "raw_vs_baseline": paired(ics["baseline"], ics["raw_reversal"]),
             "resid_vs_raw": paired(ics["raw_reversal"], ics["resid_reversal"])}
    feats = feature_ics(raw_panel, set(stored["dates"]))

    print(f"\nplacebo: {args.placebo_draws} within-date shuffles of the residual arm",
          flush=True)
    placebo_rows = placebo(data["resid_reversal"],
                           PLACEBO_SEEDS[:args.placebo_draws], args.bootstrap)

    V = verdicts(S, placebo_rows, pairs, feats, repro)
    sweep_rows: list[dict] = []
    if not args.no_sweep and V["r1"]["resid_reversal"] and V["r2"]:
        print("\nR1 and R2 hold: running the pre-registered min_train sweep",
              flush=True)
        sweep_rows = sweep(panel, "resid_reversal", args.trials)

    text = render(inputs, repro, S, metrics, pairs, feats, placebo_rows, V,
                  sweep_rows, args, time.time() - started)
    print("\n" + text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
