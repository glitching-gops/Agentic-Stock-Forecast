"""
tools/stage1b_delivery.py — Stage 1, Pilot 2: NSE delivery % as a feature.

Method, arms, the stop condition and every decision rule are fixed in
`docs/stage1b-preregistration.md`, written before this ran on real data.

    (a) baseline   FACTORS                          REUSED; re-run for S1
    (b) abnormal   FACTORS + deliv_abn_l1, deliv_abn5_l1     THE HYPOTHESIS
    (c) level      FACTORS + deliv_pct_l1                    the identity-risk arm

THE PLACEBO IS THE CORRECTED ONE. Pilot 1 shuffled an arm's PREDICTIONS
within each date. That also destroyed the baseline's own ordering, so the arm
was credited with grades the baseline already had (ICICIBANK's STRONG). Here
the delivery COLUMNS are permuted within each date, jointly so their pairing
survives, and the model is RETRAINED nine times. Every other input is
identical to arm (b), so what is destroyed is exactly the name-to-delivery
link and nothing else.

SANDBOXED. Reads panel_cache.parquet, stage2b_pooled_oos.npz and
delivery_cache.npz. Writes one .npz of held-out predictions and a markdown
report. Nothing the API, the web app or either job reads.

    python tools/stage1b_delivery.py --smoke --markdown <path>
    python tools/stage1b_delivery.py --markdown <path>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.baselines import FACTORS  # noqa: E402
from pipeline.delivery import (  # noqa: E402
    ABNORMAL_COLS,
    DELIVERY_COLS,
    LEVEL_COL,
    delivery_features,
    parse_symbol_changes,
    rank_persistence,
    to_panel_tickers,
)
from pipeline.evaluation import rank_ic  # noqa: E402
from pipeline.evidence_panel import BOOTSTRAP_B, driscoll_kraay_se  # noqa: E402
from pipeline.model import EVAL_TUNE_TRIALS  # noqa: E402
from pipeline.panel import TARGET, cross_sectional_zscore  # noqa: E402
from tools.backfill_delivery import CACHE as DELIVERY_CACHE, load_cache  # noqa: E402
from tools.stage0c_close import _from_arm  # noqa: E402
from tools.stage1_reversal import (  # noqa: E402
    BASELINE_ARM,
    BASELINE_SHA256,
    COLS,
    DK_LAGS,
    PANEL_SHA256,
    SIGNAL_T,
    _f,
    arm_arrays_sha256,
    as_data,
    as_frame,
    grade,
    paired,
    per_date_ic,
    reproduction,
    sha256_file,
    summary,
)
from tools.stage2b_pooled import cell_metrics, load_cached_panel, run_arm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
STAGE2B_NPZ = os.path.join(ROOT, "stage2b_pooled_oos.npz")
PRED_OUT = os.path.join(ROOT, "stage1b_delivery_oos.npz")
PREREG = os.path.join(ROOT, "docs", "stage1b-preregistration.md")

ARM_FEATURES: dict[str, list[str]] = {
    "baseline": list(FACTORS),
    "abnormal": list(FACTORS) + ABNORMAL_COLS,
    "level": list(FACTORS) + [LEVEL_COL],
}
DELIVERY_ARMS = ("abnormal", "level")
PLACEBO_SEEDS: tuple[int, ...] = tuple(range(20260914, 20260923))   # nine
SWEEP_MIN_TRAIN = (380, 420, 460, 500, 540, 580)


# ── the panel ─────────────────────────────────────────────────────────────────


def delivery_long(cache: str, tickers: list[str]) -> pd.DataFrame:
    rec, _, sc_text = load_cache(cache)
    return to_panel_tickers(rec, tickers, parse_symbol_changes(sc_text))


def attach_delivery(panel: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Joins the delivery columns on and standardises them within each date
    through the same ``cross_sectional_zscore`` as every other pooled
    feature. A missing value becomes the date's mean (0 after z-scoring), the
    panel-wide convention for "no information", which is not a stale value."""
    f = feats.copy()
    f["date"] = f["date"].astype(panel["date"].dtype)
    out = panel.merge(f, on=["date", "ticker"], how="left")
    out = cross_sectional_zscore(out, DELIVERY_COLS)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def permute_within_date(panel: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    """The corrected placebo's input: `cols` permuted JOINTLY within each
    date. Distribution, pairing and cross-sectional breadth are all kept; which
    company each row belongs to is destroyed. The frame is sorted by (date,
    ticker), so each date is one contiguous block."""
    rng = np.random.default_rng(seed)
    out = panel.copy()
    vals = out[cols].to_numpy().copy()
    dates = out["date"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        vals[lo:hi] = vals[lo:hi][rng.permutation(hi - lo)]
    out[cols] = vals
    return out


def design_inputs(feats: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Outcome-blind: coverage and rank persistence. Reads no target and no
    prediction, so it may be measured before the pre-registration is written."""
    cells = panel[["date", "ticker"]].astype({"date": str}).merge(
        feats.assign(date=feats["date"].astype(str)), on=["date", "ticker"], how="left")
    cov = {c: float(cells[c].notna().mean()) for c in DELIVERY_COLS}
    by_year = cells.assign(year=cells["date"].str[:4]).groupby("year")[ABNORMAL_COLS[0]] \
        .apply(lambda s: s.notna().mean()).round(3).to_dict()
    pers = {c: rank_persistence(feats, c, lag=250) for c in DELIVERY_COLS}
    return {"coverage": cov, "abn_coverage_by_year": by_year, "rank_persistence_250": pers}


def feature_ics(panel_raw: pd.DataFrame, feats: pd.DataFrame,
                dates_oos: set[str]) -> list[dict]:
    f = feats.assign(date=feats["date"].astype(str)).merge(
        panel_raw[["date", "ticker", TARGET]].astype({"date": str}), on=["date", "ticker"])
    f = f[f["date"].isin(dates_oos) & np.isfinite(f[TARGET])]
    out = []
    for col in DELIVERY_COLS:
        s = pd.Series({d: rank_ic(g[TARGET].to_numpy(dtype=float),
                                  g[col].to_numpy(dtype=float))
                       for d, g in f.groupby("date", sort=True)}, dtype=float).dropna()
        mean, se, _ = driscoll_kraay_se(s.index.to_numpy(), s.to_numpy(), max_lag=DK_LAGS)
        out.append({"feature": col, "n_dates": len(s), "ic": mean, "se": se,
                    "t": mean / se if se and se > 0 else float("nan")})
    return out


# ── the verdicts ──────────────────────────────────────────────────────────────


def verdicts(S, pairs, placebo_rows, feats, repro) -> dict:
    base = set(S["baseline"]["graded"])
    newly = {a: sorted(set(S[a]["graded"]) - base) for a in DELIVERY_ARMS}
    lost = {a: sorted(base - set(S[a]["graded"])) for a in DELIVERY_ARMS}
    b = pairs["abnormal_vs_baseline"]
    p_d = [r["pair"]["diff"] for r in placebo_rows]
    p_t = [r["pair"]["t"] for r in placebo_rows]
    r2 = bool(placebo_rows) and b["diff"] > max(p_d)
    r3 = bool(np.isfinite(b["t"]) and b["t"] >= SIGNAL_T)
    signal = r3 and r2
    failed = []
    if not r3:
        failed.append(f"R3 (paired cross-sectional t {b['t']:+.2f} < {SIGNAL_T:.1f})")
    if not r2:
        failed.append(f"R2 (gain {b['diff']:+.5f} does not beat the nine-draw placebo "
                      f"max {max(p_d) if p_d else float('nan'):+.5f})")
    lv = pairs["level_vs_baseline"]
    predictions = {
        "P1 arm (a) reproduces": bool(repro.get("passed")),
        "P2 no signal: abnormal gain abs(d) < 0.005 and abs(t) < 2":
            bool(abs(b["diff"]) < 0.005 and abs(b["t"]) < 2),
        "P3 the level arm moves more grades than the abnormal arm, with no cs gain":
            bool(len(newly["level"]) > len(newly["abnormal"]) and abs(lv["t"]) < 2),
        "P4 every delivery feature abs(IC) < 0.02":
            bool(all(abs(f["ic"]) < 0.02 for f in feats)),
        "P5 the placebo gains all sit within abs(d) < 0.005":
            bool(p_d and max(abs(x) for x in p_d) < 0.005),
    }
    return {"newly": newly, "lost": lost, "r2": r2, "r3": r3, "signal": signal,
            "failed": failed, "placebo_max_diff": max(p_d) if p_d else float("nan"),
            "placebo_max_t": max(p_t) if p_t else float("nan"),
            "predictions": predictions}


# ── the report ────────────────────────────────────────────────────────────────


def render(inputs, design, repro, S, metrics, pairs, feats, placebo_rows, V,
           sweep_rows, args, runtime) -> str:
    o = ["# Stage 1, Pilot 2 — NSE delivery %: run report\n",
         f"Pre-registration sha256 at run time: `{inputs['prereg_sha256']}`.  ",
         f"Panel {'matches' if inputs['panel_ok'] else '**DOES NOT MATCH**'} its frozen "
         f"hash; arm (a) {'matches' if inputs['baseline_ok'] else '**DOES NOT MATCH**'}; "
         f"delivery cache sha256 `{inputs['delivery_sha256'][:16]}…`.  ",
         f"B = {args.bootstrap}, trials = {args.trials}, placebo retrains = "
         f"{len(placebo_rows)}, DK lags = {DK_LAGS}. "
         f"{'**SMOKE RUN — not a measurement.**' if args.smoke else ''}\n",
         "## Design inputs (outcome-blind)\n",
         f"- coverage of panel rows: {design['coverage']}",
         f"- abnormal-feature coverage by year: {design['abn_coverage_by_year']}",
         f"- within-date rank persistence at 250 sessions: {design['rank_persistence_250']}\n",
         "## S1\n",
         f"{repro['rows_stored']:,} stored, {repro['rows_rerun']:,} re-run, only-stored "
         f"{repro['only_stored']}, only-rerun {repro['only_rerun']}, max drift "
         f"{repro['max_pred_drift']:.1e}: **{'PASS' if repro['passed'] else 'FAIL'}**\n",
         "## The arms, graded by `grade_panel_v3`\n",
         "| arm | mu_hat | boot SE | z | tau2 | STRONG | WEAK | INSUFF | RW | cs IC (DK SE) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, s in S.items():
        o.append(f"| {arm} | {_f(s['mu_hat'])} | {_f(s['se_boot'], '.5f')} | "
                 f"{_f(s['z'], '+.2f')} | {_f(s['tau2'], '.5f')} | {s['strong']} | "
                 f"{s['weak']} | {s['insufficient']} | {s['rw']} | "
                 f"{_f(s['cs_ic'])} ({_f(s['cs_ic_se'], '.5f')}) |")
    o.append("")
    for arm in DELIVERY_ARMS:
        o.append(f"- **{arm}** graded {S[arm]['graded'] or 'none'}; newly out of "
                 f"INSUFFICIENT {V['newly'][arm] or 'none'}; lost {V['lost'][arm] or 'none'}.")
    o.append(f"- **baseline** graded {S['baseline']['graded'] or 'none'}.\n")

    o.append("## R3 — the deciding rule: paired per-date cross-sectional IC\n")
    o.append("| comparison | dates | Δ | DK SE (30 lags) | **t** | t, default lags |")
    o.append("|---|---|---|---|---|---|")
    for key, lab in (("abnormal_vs_baseline", "(b) abnormal − (a)"),
                     ("level_vs_baseline", "(c) level − (a)"),
                     ("abnormal_vs_level", "(b) abnormal − (c) level")):
        p = pairs[key]
        o.append(f"| {lab} | {p['n_dates']:,} | {_f(p['diff'])} | {_f(p['se'], '.5f')} | "
                 f"**{_f(p['t'], '+.2f')}** | {_f(p['t_default'], '+.2f')} |")
    o.append("")

    o.append("## R2 — the corrected placebo: delivery columns permuted within date, retrained\n")
    o.append("| seed | Δ cs IC vs (a) | t | STRONG | WEAK | RW | tau2 |")
    o.append("|---|---|---|---|---|---|---|")
    for r in placebo_rows:
        o.append(f"| {r['seed']} | {_f(r['pair']['diff'])} | {_f(r['pair']['t'], '+.2f')} | "
                 f"{r['strong']} | {r['weak']} | {r['rw']} | {_f(r['tau2'], '.5f')} |")
    b = S["abnormal"]
    o.append(f"| **arm (b)** | **{_f(pairs['abnormal_vs_baseline']['diff'])}** | "
             f"**{_f(pairs['abnormal_vs_baseline']['t'], '+.2f')}** | {b['strong']} | "
             f"{b['weak']} | {b['rw']} | {_f(b['tau2'], '.5f')} |")
    o.append(f"\n**R2: {'PASS' if V['r2'] else 'FAIL'}** — arm (b)'s gain must exceed "
             f"all nine placebo gains (max {_f(V['placebo_max_diff'])}).\n")

    o.append("## Each delivery feature's own cross-sectional IC\n")
    o.append("| feature | dates | IC | t |")
    o.append("|---|---|---|---|")
    for f in feats:
        o.append(f"| {f['feature']} | {f['n_dates']:,} | {_f(f['ic'])} | {_f(f['t'], '+.2f')} |")
    o.append("")

    o.append("## Stage 2b's cell metrics\n")
    o.append("| arm | constant cells | reb IC | reb t | cs IC | OOS MAE | gap | cs IC by fold |")
    o.append("|---|---|---|---|---|---|---|---|")
    for arm, m in metrics.items():
        o.append(f"| {arm} | {m['constant_cells']}/{m['cells']} | {_f(m['reb_ic'], '+.4f')} | "
                 f"{_f(m['reb_t'], '+.2f')} | {_f(m['cs_rank_ic'], '+.4f')} | "
                 f"{m['mae']:.5f} | {_f(m['gap'])} | {m['cs_rank_ic_by_fold']} |")
    o.append("")

    o.append("## Verdicts, as pre-registered\n")
    o.append(f"- **R3** (deciding): {'PASS' if V['r3'] else 'FAIL'}.")
    o.append(f"- **R2** (placebo): {'PASS' if V['r2'] else 'FAIL'}.")
    o.append(f"- **SIGNAL: {'YES' if V['signal'] else 'NO'}**"
             + (f" — failed: {'; '.join(V['failed'])}." if V["failed"] else "."))
    o.append(f"- **R6** sweep: {'run' if sweep_rows else 'not triggered'}.\n")
    o.append("| prediction | held |")
    o.append("|---|---|")
    for k, v in V["predictions"].items():
        o.append(f"| {k} | {'yes' if v else '**no**'} |")
    if sweep_rows:
        o.append("\n## R6 — min_train sweep\n")
        o.append("| min_train | arm | reb IC | reb t | cs IC |")
        o.append("|---|---|---|---|---|")
        for r in sweep_rows:
            o.append(f"| {r['min_train']} | {r['arm']} | {_f(r['reb_ic'], '+.4f')} | "
                     f"{_f(r['reb_t'], '+.2f')} | {_f(r['cs_ic'], '+.4f')} |")
    o.append(f"\nEvery SE is likely too small (Politis-White 35.8-62.5 sessions). "
             f"Runtime {runtime / 60:.1f} min.")
    return "\n".join(o)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--stage2b-npz", default=STAGE2B_NPZ)
    ap.add_argument("--delivery-cache", default=DELIVERY_CACHE)
    ap.add_argument("--npz", default=PRED_OUT)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--trials", type=int, default=EVAL_TUNE_TRIALS)
    ap.add_argument("--placebo-draws", type=int, default=len(PLACEBO_SEEDS))
    ap.add_argument("--design-only", action="store_true",
                    help="print the outcome-blind design inputs and stop")
    ap.add_argument("--smoke", action="store_true",
                    help="1 trial, B = 20, 1 placebo retrain; S1 not enforced")
    args = ap.parse_args()
    if args.smoke:
        args.trials, args.bootstrap, args.placebo_draws = 1, 20, 1
        if args.npz == PRED_OUT:
            args.npz = os.path.join(ROOT, "stage1b_smoke_oos.npz")

    started = time.time()
    raw_panel = load_cached_panel(args.panel_cache)
    tickers = sorted(raw_panel["ticker"].unique())
    grid = sorted(raw_panel["date"].astype(str).unique())
    feats = delivery_features(delivery_long(args.delivery_cache, tickers), grid)
    design = design_inputs(feats, raw_panel)
    print(f"design inputs: {design}", flush=True)
    if args.design_only:
        return

    inputs = {
        "prereg_sha256": sha256_file(PREREG) if os.path.exists(PREREG) else "missing",
        "panel_sha256": sha256_file(args.panel_cache),
        "baseline_sha256": arm_arrays_sha256(args.stage2b_npz, BASELINE_ARM),
        "delivery_sha256": sha256_file(args.delivery_cache),
    }
    inputs["panel_ok"] = inputs["panel_sha256"].lower() == PANEL_SHA256
    inputs["baseline_ok"] = inputs["baseline_sha256"].lower() == BASELINE_SHA256
    print(f"pre-registration {inputs['prereg_sha256']}", flush=True)
    if inputs["prereg_sha256"] == "missing" and not args.smoke:
        raise SystemExit("no pre-registration on disk; refusing to run on real data")
    if not (inputs["panel_ok"] and inputs["baseline_ok"]) and not args.smoke:
        raise SystemExit("a frozen input changed since the pre-registration")

    panel = attach_delivery(raw_panel, feats)
    stored = _from_arm(args.stage2b_npz, BASELINE_ARM)

    frames, metrics = {}, {}
    print("\n(a) re-running the baseline for S1 ...", flush=True)
    preds, folds = run_arm(panel, "mae", "none", n_trials=args.trials,
                           features=ARM_FEATURES["baseline"])
    frames["baseline_repro"] = preds
    metrics["baseline"] = cell_metrics(preds, folds)
    repro = reproduction(stored, as_data(preds))
    print(f"  S1 {'PASS' if repro['passed'] else 'FAIL'} (drift {repro['max_pred_drift']:.1e})",
          flush=True)
    if not repro["passed"] and not args.smoke:
        raise SystemExit(f"S1 FAILED ({repro}); stopping before any delivery arm is read")

    for arm in DELIVERY_ARMS:
        print(f"\n({arm}) ...", flush=True)
        preds, folds = run_arm(panel, "mae", "none", n_trials=args.trials,
                               features=ARM_FEATURES[arm])
        frames[arm] = preds
        metrics[arm] = cell_metrics(preds, folds)
    np.savez_compressed(args.npz, **{f"{k}__{c}": v[c].to_numpy()
                                     for k, v in frames.items() for c in COLS})

    data = {"baseline": stored, "abnormal": as_data(frames["abnormal"]),
            "level": as_data(frames["level"])}
    S = {}
    for arm, d in data.items():
        print(f"grading {arm} ...", flush=True)
        S[arm] = summary(grade(d, args.bootstrap, auto_block=not args.smoke))
    ics = {arm: per_date_ic(d) for arm, d in data.items()}
    pairs = {"abnormal_vs_baseline": paired(ics["baseline"], ics["abnormal"]),
             "level_vs_baseline": paired(ics["baseline"], ics["level"]),
             "abnormal_vs_level": paired(ics["level"], ics["abnormal"])}
    fics = feature_ics(raw_panel, feats, set(stored["dates"]))

    print(f"\ncorrected placebo: {args.placebo_draws} retrains with the delivery "
          f"columns permuted within date", flush=True)
    placebo_rows = []
    for seed in PLACEBO_SEEDS[:args.placebo_draws]:
        t0 = time.time()
        shuffled = permute_within_date(panel, ABNORMAL_COLS, seed)
        preds, _ = run_arm(shuffled, "mae", "none", n_trials=args.trials,
                           features=ARM_FEATURES["abnormal"], verbose=False)
        d = as_data(preds)
        s = summary(grade(d, args.bootstrap, auto_block=False))
        s["pair"] = paired(ics["baseline"], per_date_ic(d))
        placebo_rows.append({"seed": seed, **s})
        print(f"    seed {seed}: gain {s['pair']['diff']:+.5f} (t {s['pair']['t']:+.2f}) "
              f"STRONG {s['strong']} RW {s['rw']} ({time.time() - t0:.0f}s)", flush=True)

    V = verdicts(S, pairs, placebo_rows, fics, repro)
    sweep_rows = []
    if V["signal"] and not args.smoke:
        print("\nSIGNAL: running the pre-registered min_train sweep", flush=True)
        for mt in SWEEP_MIN_TRAIN:
            for name in ("baseline", "abnormal"):
                p, f = run_arm(panel, "mae", "none", n_trials=args.trials,
                               min_train=mt, features=ARM_FEATURES[name], verbose=False)
                m = cell_metrics(p, f)
                sweep_rows.append({"min_train": mt, "arm": name, "reb_ic": m["reb_ic"],
                                   "reb_t": m["reb_t"], "cs_ic": m["cs_rank_ic"]})

    text = render(inputs, design, repro, S, metrics, pairs, fics, placebo_rows, V,
                  sweep_rows, args, time.time() - started)
    print("\n" + text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
