"""
tools/hygiene_repin.py — re-pin the stored baselines under the new label and
the pinned thread count, and measure what the conformal interval costs.

Run against `docs/hygiene-preregistration.md`, hashed before anything here
touched real data. NOTHING HERE IS EXPECTED TO PRODUCE SIGNAL; it is expected
to change stored numbers, and any apparent new signal is to be investigated as
a defect first.

Three arms at h=30, legacy purge, on identical folds and rows:

    committed     the numbers in the repo today: raw label, whatever thread
                  count the machine happened to default to (20 here)
    old_pinned    raw label, XGB_THREADS. Isolates what the PIN alone moved.
    new_pinned    within-date standardised label, XGB_THREADS. The new baseline.

Splitting the pin from the label matters: collapsing them would leave every
difference unattributable between two changes, which is the same reason
`config_hash` and `data_hash` are separate.

THE CONFORMAL ROUND TRIP IS THE PART THAT CAN STOP THE SESSION. A standardised
label means the model predicts relative cross-sectional position, so the
published price band only exists after inverting — and the inverse needs
moments that are NOT knowable at prediction time. So it is done twice:

    realised moments   exact, illegitimate live, reported as the ceiling
    causal moments     a trailing estimate, the only thing available live

Coverage is measured split-conformal: calibrated on the early folds and
checked on the LAST fold, never on the rows it was fitted on.

    python tools/hygiene_repin.py --markdown hygiene.md
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
from pipeline.conformal import check_coverage, fit_conformal  # noqa: E402
from pipeline.determinism import XGB_THREADS, environment_fingerprint  # noqa: E402
from pipeline.evidence_panel import BOOTSTRAP_B, driscoll_kraay_se  # noqa: E402
from pipeline.label import (  # noqa: E402
    CS_MEAN,
    CS_SD,
    attach_causal_moments,
    gamma_range_note,
    inverse_standardise,
    standardise_target,
)
from pipeline.model import EVAL_TUNE_TRIALS  # noqa: E402
from pipeline.panel import TARGET  # noqa: E402
from pipeline.signals import HORIZON_SESSIONS  # noqa: E402
from tools.stage0c_close import _from_arm  # noqa: E402
from tools.stage1_reversal import (  # noqa: E402
    BASELINE_ARM,
    DK_LAGS,
    _f,
    as_data,
    grade,
    per_date_ic,
    reproduction,
    sha256_file,
    summary,
)
from tools.stage2b_pooled import cell_metrics, load_cached_panel, run_arm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
STAGE2B_NPZ = os.path.join(ROOT, "stage2b_pooled_oos.npz")
PREREG = os.path.join(ROOT, "docs", "hygiene-preregistration.md")
PRED_OUT = os.path.join(ROOT, "hygiene_repin_oos.npz")
STATE = os.path.join(ROOT, "hygiene_repin.json")

COLS = ("date", "ticker", "y_true", "y_pred", "fold")
#: Folds calibrated on, versus the fold coverage is CHECKED on. Split
#: conformal measured on its own calibration pool is not a measurement.
CALIBRATION_FOLDS = (0, 1, 2, 3)
CHECK_FOLD = 4


def _ic(preds: pd.DataFrame) -> dict:
    ics = per_date_ic(as_data(preds)).dropna()
    if len(ics) < 3:
        return {"cs_ic": float("nan"), "se": float("nan"), "t": float("nan"),
                "n_dates": len(ics)}
    mean, se, _ = driscoll_kraay_se(ics.index.to_numpy(), ics.to_numpy(),
                                    max_lag=DK_LAGS)
    return {"cs_ic": mean, "se": se,
            "t": mean / se if se and np.isfinite(se) and se > 0 else float("nan"),
            "n_dates": int(len(ics))}


def run_one(panel: pd.DataFrame, trials: int, label: str,
            verbose: bool = True,
            standardise: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    print(f"\n[{label}] h={HORIZON_SESSIONS}, legacy purge, {XGB_THREADS} threads",
          flush=True)
    return run_arm(panel, "mae", "none", n_trials=trials, features=list(FACTORS),
                   horizon=HORIZON_SESSIONS, purge=HORIZON_SESSIONS,
                   verbose=verbose, standardise_label=standardise)


def score_arm(preds: pd.DataFrame, folds: list[dict], bootstrap: int,
              auto_block: bool) -> dict:
    m = cell_metrics(preds, folds, rebalance_every=HORIZON_SESSIONS)
    s = summary(grade(as_data(preds), bootstrap, auto_block=auto_block))
    return {**{k: m[k] for k in ("reb_ic", "reb_t", "n_rebalances",
                                 "n_dates_no_ordering", "cs_rank_ic",
                                 "cs_rank_ic_by_fold", "mae", "gap",
                                 "constant_cells", "cells", "n_rows")},
            **{k: s[k] for k in ("mu_hat", "se_boot", "z", "tau2", "strong",
                                 "weak", "insufficient", "rw", "graded")},
            **{f"dk_{k}": v for k, v in _ic(preds).items()},
            "gammas": [round(f["gamma"], 3) for f in folds]}


# ── the conformal round trip ──────────────────────────────────────────────────


def conformal_on(y_true: np.ndarray, y_pred: np.ndarray,
                 folds: np.ndarray, coverage: float = 0.80) -> dict:
    """
    Split conformal: calibrate on the early folds, CHECK on the last one.

    `check_coverage` on the calibration pool itself reports the quantile's own
    definition back, near-exactly, and would look like a pass.
    """
    cal = np.isin(folds, CALIBRATION_FOLDS)
    chk = folds == CHECK_FOLD
    if cal.sum() < 20 or chk.sum() < 20:
        return {"n": 0, "note": "not enough rows to split"}

    calibration = fit_conformal(y_true[cal], y_pred[cal], coverage=coverage)
    if calibration is None:
        return {"n": 0, "note": "conformal refused the calibration pool"}
    out = check_coverage(calibration, y_true[chk], y_pred[chk])
    out["quantile"] = calibration.quantile
    out["n_calibration"] = int(cal.sum())

    # PER CHECK-FOLD, expanding: calibrate on everything before fold k and
    # check on k. Without this the headline confounds the round trip with the
    # fold it lands on — this panel's target dispersion falls monotonically
    # across the folds (0.108 to 0.077), so a calibration fitted on the early,
    # wilder folds necessarily OVER-covers the last one, and that has nothing
    # to do with standardising anything.
    by_fold = []
    for k in sorted({int(f) for f in np.unique(folds)}):
        earlier, here = folds < k, folds == k
        if earlier.sum() < 20 or here.sum() < 20:
            continue
        c = fit_conformal(y_true[earlier], y_pred[earlier], coverage=coverage)
        if c is None:
            continue
        cov = check_coverage(c, y_true[here], y_pred[here])
        by_fold.append({"fold": k, "n": cov["n"],
                        "coverage": cov["realised_coverage"],
                        "quantile": c.quantile})
    out["by_fold"] = by_fold
    return out


def conformal_round_trip(preds_std: pd.DataFrame, panel_raw: pd.DataFrame,
                         panel_causal: pd.DataFrame) -> dict:
    """
    Coverage of the published interval, after the standardise-and-invert trip.

    The arm predicted a z-score. Both inverses are measured:
      - realised moments, which are exact and are NOT available live;
      - causal moments, which is what a live forecast would actually use.
    Residuals are taken against the RAW h-session label in both cases, so the
    coverage figure is in the units the dashboard publishes.
    """
    keyed = preds_std.astype({"date": str}).copy()

    truth = panel_raw[["date", "ticker", TARGET]].astype({"date": str}) \
        .rename(columns={TARGET: "y_raw"})
    realised = panel_raw[["date", "ticker"]].astype({"date": str}).copy()
    std_moments = standardise_target(panel_raw)
    realised[CS_MEAN] = std_moments[CS_MEAN].to_numpy()
    realised[CS_SD] = std_moments[CS_SD].to_numpy()
    causal = panel_causal[["date", "ticker", CS_MEAN, CS_SD]] \
        .astype({"date": str}) \
        .rename(columns={CS_MEAN: "c_mean", CS_SD: "c_sd"})

    j = keyed.merge(truth, on=["date", "ticker"], how="inner") \
             .merge(realised, on=["date", "ticker"], how="left") \
             .merge(causal, on=["date", "ticker"], how="left")

    out: dict = {"n_rows": int(len(j))}
    for name, mean_col, sd_col in (("realised", CS_MEAN, CS_SD),
                                   ("causal", "c_mean", "c_sd")):
        inverted = inverse_standardise(j["y_pred"], j[mean_col], j[sd_col])
        ok = (np.isfinite(inverted) & np.isfinite(j["y_raw"].to_numpy(dtype=float)))
        res = conformal_on(j["y_raw"].to_numpy(dtype=float)[ok],
                           np.asarray(inverted)[ok],
                           j["fold"].to_numpy()[ok])
        res["n_usable"] = int(ok.sum())
        res["mae"] = float(np.mean(np.abs(
            np.asarray(inverted)[ok] - j["y_raw"].to_numpy(dtype=float)[ok])))
        out[name] = res
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--stage2b-npz", default=STAGE2B_NPZ)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--npz", default=PRED_OUT)
    ap.add_argument("--trials", type=int, default=EVAL_TUNE_TRIALS)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.trials, args.bootstrap = 1, 20
        args.state = os.path.join(ROOT, "hygiene_smoke.json")
        args.npz = os.path.join(ROOT, "hygiene_smoke.npz")

    started = time.time()
    env = environment_fingerprint()
    print(f"environment: {env}", flush=True)
    prereg = sha256_file(PREREG) if os.path.exists(PREREG) else "missing"
    print(f"pre-registration {prereg}", flush=True)
    if prereg == "missing" and not args.smoke:
        raise SystemExit("no pre-registration on disk; refusing to run")

    raw_panel = load_cached_panel(args.panel_cache)
    std_panel = standardise_target(raw_panel)
    causal_panel = attach_causal_moments(raw_panel, horizon=HORIZON_SESSIONS)
    stored = _from_arm(args.stage2b_npz, BASELINE_ARM)

    results: dict = {"inputs": {"prereg_sha256": prereg, "environment": env},
                     "arms": {}, "gamma": {}}

    # ── the two arms ─────────────────────────────────────────────────────────
    frames = {}
    # The raw label, deliberately: this arm exists to isolate what the
    # THREAD PIN alone moved, so the label must not move with it.
    old_preds, old_folds = run_one(raw_panel, args.trials, "old_pinned",
                                   standardise=False)
    frames["old_pinned"] = old_preds
    results["arms"]["old_pinned"] = score_arm(old_preds, old_folds,
                                              args.bootstrap, not args.smoke)
    repro = reproduction(stored, as_data(old_preds))
    results["a0_vs_committed"] = repro
    print(f"  A0 vs the committed predictions: drift {repro['max_pred_drift']:.3e} "
          f"({'identical' if repro['passed'] else 'MOVED — expected, see H3'})",
          flush=True)

    new_preds, new_folds = run_one(std_panel, args.trials, "new_pinned")
    frames["new_pinned"] = new_preds
    results["arms"]["new_pinned"] = score_arm(new_preds, new_folds,
                                              args.bootstrap, not args.smoke)

    # ── reproducible twice in a row, on this machine ─────────────────────────
    print("\n[reproducibility] re-running new_pinned a second time", flush=True)
    again, _ = run_one(std_panel, args.trials, "new_pinned_again", verbose=False)
    rep = reproduction(as_data(new_preds), as_data(again))
    results["reproducible_twice"] = rep
    print(f"  drift {rep['max_pred_drift']:.3e}: "
          f"{'IDENTICAL' if rep['passed'] else 'NOT REPRODUCIBLE'}", flush=True)

    # ── the conformal round trip ─────────────────────────────────────────────
    print("\n[conformal] split-conformal, calibrated on folds 0-3, checked on 4",
          flush=True)
    results["conformal_new"] = conformal_round_trip(new_preds, raw_panel,
                                                    causal_panel)
    d = as_data(old_preds)
    direct = conformal_on(d["y_true"], d["y_pred"], np.asarray(d["folds"]))
    direct["mae"] = float(np.mean(np.abs(d["y_true"] - d["y_pred"])))
    results["conformal_old"] = {"direct": direct}
    for k in ("realised", "causal"):
        c = results["conformal_new"].get(k, {})
        print(f"  new label, {k:8} moments: coverage "
              f"{c.get('realised_coverage', float('nan')):.4f} "
              f"(n {c.get('n', 0)}, half-width {c.get('quantile', float('nan')):.4f})",
              flush=True)
    c = results["conformal_old"]["direct"]
    print(f"  old label, direct           : coverage "
          f"{c.get('realised_coverage', float('nan')):.4f} (n {c.get('n', 0)})",
          flush=True)

    # ── is [0, 5] still the right gamma range? ───────────────────────────────
    results["gamma"] = {"raw": gamma_range_note(raw_panel),
                        "standardised": gamma_range_note(std_panel)}

    np.savez_compressed(args.npz, **{f"{k}__{c}": v[c].to_numpy()
                                     for k, v in frames.items() for c in COLS})
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=float)

    text = render(results, args, time.time() - started)
    print("\n" + text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")


def render(r: dict, args, runtime: float) -> str:
    a = r["arms"]
    o = ["# Hygiene — the re-pin, old against new\n",
         f"Pre-registration sha256 `{r['inputs']['prereg_sha256']}`.  ",
         f"Threads pinned at {r['inputs']['environment']['xgb_threads']}; "
         f"xgboost {r['inputs']['environment']['xgboost']}, "
         f"numpy {r['inputs']['environment']['numpy']}, "
         f"pandas {r['inputs']['environment']['pandas']}.  ",
         f"B = {args.bootstrap}, trials = {args.trials}. "
         f"{'**SMOKE — not a measurement.**' if args.smoke else ''}\n",
         "## The comparator table\n",
         "| arm | cs IC (DK SE) | t | reb IC | reb t | n reb | constant cells | "
         "S/W/I | mu_hat | tau2 | OOS MAE | per-fold gamma |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in a.items():
        o.append(f"| {name} | {_f(m['dk_cs_ic'])} ({_f(m['dk_se'], '.5f')}) | "
                 f"{_f(m['dk_t'], '+.2f')} | {_f(m['reb_ic'], '+.4f')} | "
                 f"{_f(m['reb_t'], '+.2f')} | {m['n_rebalances']} | "
                 f"{m['constant_cells']}/{m['cells']} | "
                 f"{m['strong']}/{m['weak']}/{m['insufficient']} | "
                 f"{_f(m['mu_hat'])} | {_f(m['tau2'], '.5f')} | {m['mae']:.5f} | "
                 f"{m['gammas']} |")

    p = r["a0_vs_committed"]
    o += ["", "## A0 against the committed predictions\n",
          f"{p['rows_stored']:,} stored, {p['rows_rerun']:,} re-run, unmatched "
          f"{p['only_stored']}/{p['only_rerun']}, max prediction drift "
          f"**{p['max_pred_drift']:.3e}**, max label drift "
          f"{p['max_truth_drift']:.3e}.\n"]
    rep = r["reproducible_twice"]
    o.append(f"**Reproducible twice on this machine:** drift "
             f"{rep['max_pred_drift']:.3e} — "
             f"{'IDENTICAL' if rep['passed'] else '**NOT REPRODUCIBLE**'}.\n")

    o += ["## Conformal coverage after the round trip\n",
          "Split conformal: calibrated on folds 0-3, coverage CHECKED on fold 4.\n",
          "| arm | inverse | n checked | half-width | **realised coverage** | "
          "gap (pp) | well calibrated | MAE |",
          "|---|---|---|---|---|---|---|---|"]
    for key, inv in (("conformal_old", "direct"), ("conformal_new", "realised"),
                     ("conformal_new", "causal")):
        c = r.get(key, {}).get(inv, {})
        if not c.get("n"):
            continue
        arm = "old label (raw)" if key == "conformal_old" else "new label (z)"
        o.append(f"| {arm} | {inv} | {c['n']:,} | {_f(c.get('quantile'), '.5f')} | "
                 f"**{c['realised_coverage']:.4f}** | "
                 f"{c['coverage_gap_pp']:+.2f} | "
                 f"{'yes' if c['well_calibrated'] else '**NO**'} | "
                 f"{c.get('mae', float('nan')):.5f} |")

    o += ["", "### Coverage per check-fold, calibrated on everything before it\n",
          "| arm / inverse | fold 1 | fold 2 | fold 3 | fold 4 |",
          "|---|---|---|---|---|"]
    for key, inv in (("conformal_old", "direct"),
                     ("conformal_new", "realised"),
                     ("conformal_new", "causal")):
        c = r.get(key, {}).get(inv, {})
        rows = {int(b["fold"]): b["coverage"] for b in c.get("by_fold", [])}
        if not rows:
            continue
        arm = "old (raw)" if key == "conformal_old" else f"new (z), {inv}"
        cells = " | ".join(f"{rows[k]:.4f}" if k in rows else "n/a"
                           for k in (1, 2, 3, 4))
        o.append(f"| {arm} | {cells} |")

    g = r["gamma"]
    o += ["", "## Is [0, 5] still the right `gamma` range?\n",
          "| label | pooled sd | mean abs | n |", "|---|---|---|---|",
          f"| raw | {g['raw']['pooled_sd']:.5f} | {g['raw']['mean_abs']:.5f} | "
          f"{g['raw']['n']:,} |",
          f"| standardised | {g['standardised']['pooled_sd']:.5f} | "
          f"{g['standardised']['mean_abs']:.5f} | {g['standardised']['n']:,} |", ""]
    o.append(f"Runtime {runtime / 60:.1f} min.")
    return "\n".join(o)


if __name__ == "__main__":
    main()
