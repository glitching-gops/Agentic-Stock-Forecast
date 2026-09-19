"""
tools/stage1c_sue.py — Stage 1, Pilot 3: seasonal-random-walk SUE as an
event-window feature.

Method, arms, the stop condition and every decision rule are fixed in
`docs/stage1c-preregistration.md`, written before this ran on real data.

    (a) baseline  FACTORS                                  REUSED; re-run for S1
    (b) sue       FACTORS + sue_evt, sue_age, sue_missing  THE HYPOTHESIS
    (c) naive     FACTORS + sue_ffill                      the forward-filled
                                                           construction, descriptive
    (d) timing    FACTORS + sue_age, sue_missing           the identity-risk arm:
                                                           everything but the surprise

Carried forward from Pilot 2, unchanged: S1, R3 deciding, and the corrected
placebo. The three event columns are permuted JOINTLY within each date and
arm (b) is RETRAINED nine times; predictions are never shuffled. New here:
R4, whether the arm's gain survives the 0.2225% round-trip cost in a traded
book, and the raw PEAD sort book, net of the same cost.

SANDBOXED. Reads panel_cache.parquet, stage2b_pooled_oos.npz and
results_cache.npz. Writes one .npz of held-out predictions and a markdown
report. Nothing the API, the web app or either job reads.

    python tools/stage1c_sue.py --design-only
    python tools/stage1c_sue.py --smoke --markdown <path>
    python tools/stage1c_sue.py --markdown <path>
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
from pipeline.delivery import rank_persistence  # noqa: E402
from pipeline.earnings import (  # noqa: E402
    EVENT_COLS,
    EVENT_WINDOW,
    SUE_AGE,
    SUE_EVT,
    SUE_FFILL,
    SUE_MISSING,
    announcements,
    event_features,
    merge_yf_splits,
    impute_cross_sectional_median,
    timing_class,
    usable_session,
)
from pipeline.evaluation import rank_ic  # noqa: E402
from pipeline.evidence_panel import BOOTSTRAP_B, driscoll_kraay_se  # noqa: E402
from pipeline.model import EVAL_TUNE_TRIALS  # noqa: E402
from pipeline.panel import TARGET, cross_sectional_zscore  # noqa: E402
from pipeline.portfolio import CostModel, simulate  # noqa: E402
from tools.backfill_results import CACHE as RESULTS_CACHE, eps_table, load_cache  # noqa: E402
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
from tools.stage1b_delivery import permute_within_date  # noqa: E402
from tools.stage1b_phantom_check import PHANTOMS  # noqa: E402
from tools.stage2b_pooled import cell_metrics, load_cached_panel, run_arm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
STAGE2B_NPZ = os.path.join(ROOT, "stage2b_pooled_oos.npz")
PRED_OUT = os.path.join(ROOT, "stage1c_sue_oos.npz")
PREREG = os.path.join(ROOT, "docs", "stage1c-preregistration.md")

ARM_FEATURES: dict[str, list[str]] = {
    "baseline": list(FACTORS),
    "sue": list(FACTORS) + EVENT_COLS,
    "naive": list(FACTORS) + [SUE_FFILL],
    "timing": list(FACTORS) + [SUE_AGE, SUE_MISSING],
}
SUE_ARMS = ("sue", "naive", "timing")
PLACEBO_SEEDS: tuple[int, ...] = tuple(range(20260920, 20260929))   # nine
SWEEP_MIN_TRAIN = (380, 420, 460, 500, 540, 580)
QUANTILES = 5
#: The names needed on a date before an in-window-only IC is computed.
MIN_LIVE_NAMES = 10


# ── the features ──────────────────────────────────────────────────────────────


def sue_announcements(cache: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    T = load_cache(cache)
    E = eps_table(T)
    E = E[E["eps"].notna()][["symbol", "period_end", "basis", "eps", "disclosed"]]
    if T["yf_splits"].empty:
        raise SystemExit("results cache holds no yfinance split snapshot; run "
                         "tools/backfill_results.py --snapshot-splits")
    actions = merge_yf_splits(T["actions"], T["yf_splits"])
    return announcements(E, actions), actions


def raw_features(ann: pd.DataFrame, grid: list[str], tickers: list[str]) -> pd.DataFrame:
    """Event features on the grid, NaNs imputed to each date's
    cross-sectional median (the missingness lives in sue_missing)."""
    f = event_features(ann, grid, tickers, phantoms=PHANTOMS)
    return impute_cross_sectional_median(f, [SUE_EVT, SUE_AGE, SUE_FFILL])


def attach_sue(panel: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Joins the SUE columns on and standardises them within each date
    through the same ``cross_sectional_zscore`` as every other pooled feature."""
    f = feats.copy()
    f["date"] = f["date"].astype(panel["date"].dtype)
    out = panel.merge(f, on=["date", "ticker"], how="left")
    out = cross_sectional_zscore(out, EVENT_COLS + [SUE_FFILL])
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def design_inputs(ann: pd.DataFrame, raw_unimputed: pd.DataFrame, grid: list[str]) -> dict:
    """Outcome-blind: coverage, timing, the in-window share and rank
    persistence. Reads no target and no prediction."""
    f = raw_unimputed.assign(year=raw_unimputed["date"].str[:4])
    defined = f[SUE_MISSING] == 0
    live = defined & (f[SUE_AGE] < EVENT_WINDOW)
    sessions = set(grid)
    in_span = ann["disclosed"].dt.strftime("%Y-%m-%d").between(grid[0], grid[-1])
    timing = ann.loc[in_span, "disclosed"].map(lambda t: timing_class(t, sessions))
    return {
        "announcements": int(len(ann)),
        "announcements_with_sue": int(ann["sue"].notna().sum()),
        "basis": ann["basis"].fillna("none").value_counts().to_dict(),
        "timing": timing.value_counts().to_dict(),
        "defined_share_by_year": f.assign(d=defined).groupby("year")["d"].mean()
                                  .round(3).to_dict(),
        "live_share": float(live.mean()),
        "rank_persistence_250": {c: rank_persistence(raw_unimputed, c, lag=250)
                                 for c in [SUE_EVT, SUE_AGE, SUE_MISSING, SUE_FFILL]},
        "sue_quantiles": ann["sue"].quantile([0.01, 0.1, 0.5, 0.9, 0.99]).round(2).to_dict(),
    }


def feature_ics(raw: pd.DataFrame, panel_raw: pd.DataFrame, dates_oos: set[str]) -> list[dict]:
    """Each SUE column's own per-date rank IC against the target (DK, 30
    lags), over arm (a)'s out-of-sample dates; and sue_evt's IC among the
    names with a LIVE event only."""
    f = raw.assign(date=raw["date"].astype(str)).merge(
        panel_raw[["date", "ticker", TARGET]].astype({"date": str}), on=["date", "ticker"])
    f = f[f["date"].isin(dates_oos) & np.isfinite(f[TARGET])]
    out = []

    def one(name, g_iter):
        s = pd.Series(dict(g_iter), dtype=float).dropna()
        if len(s) < 3:
            out.append({"feature": name, "n_dates": len(s), "ic": float("nan"),
                        "se": float("nan"), "t": float("nan")})
            return
        mean, se, _ = driscoll_kraay_se(s.index.to_numpy(), s.to_numpy(), max_lag=DK_LAGS)
        out.append({"feature": name, "n_dates": len(s), "ic": mean, "se": se,
                    "t": mean / se if se and se > 0 else float("nan")})

    for col in [SUE_EVT, SUE_AGE, SUE_MISSING, SUE_FFILL]:
        one(col, ((d, rank_ic(g[TARGET].to_numpy(float), g[col].to_numpy(float)))
                  for d, g in f.groupby("date", sort=True)))
    live = f[(f[SUE_MISSING] == 0) & (f[SUE_AGE] < EVENT_WINDOW)]
    one(f"{SUE_EVT} (live names only)",
        ((d, rank_ic(g[TARGET].to_numpy(float), g[SUE_EVT].to_numpy(float)))
         for d, g in live.groupby("date", sort=True) if len(g) >= MIN_LIVE_NAMES))
    return out


# ── money ─────────────────────────────────────────────────────────────────────


def book(frame: pd.DataFrame, costs: CostModel | None = None) -> pd.DataFrame:
    """The long-short top-minus-bottom quintile book on one comparator's
    predictions, net of `costs`, by rebalance date."""
    b = simulate(frame[["date", "ticker", "y_pred", "y_true"]], costs=costs or CostModel(),
                 quantiles=QUANTILES, long_only=False)
    return pd.DataFrame({"date": b.dates, "gross": b.gross_returns,
                         "net": b.net_returns, "turnover": b.turnover})


def _t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def net_of_cost(base: pd.DataFrame, arm: pd.DataFrame) -> dict:
    """R4: arm minus baseline, per non-overlapping rebalance, net of the
    0.2225% round trip. The rebalances do not overlap, so a plain t is the
    right one here."""
    a, b = book(base), book(arm)
    j = a.merge(b, on="date", suffixes=("_a", "_b"))
    d = (j["net_b"] - j["net_a"]).to_numpy()
    return {"n_rebalances": len(j), "net_a": float(j["net_a"].mean()),
            "net_b": float(j["net_b"].mean()), "gross_a": float(j["gross_a"].mean()),
            "gross_b": float(j["gross_b"].mean()), "turn_a": float(j["turnover_a"].mean()),
            "turn_b": float(j["turnover_b"].mean()), "diff": float(np.mean(d)) if len(d) else
            float("nan"), "t": _t(d), "round_trip": CostModel().round_trip}


def pead_book(raw: pd.DataFrame, base: pd.DataFrame) -> dict:
    """The literal PEAD test: sort on sue_evt alone, over arm (a)'s rows,
    long the top quintile and short the bottom, net of the round trip."""
    f = base[["date", "ticker", "y_true"]].astype({"date": str}).merge(
        raw[["date", "ticker", SUE_EVT]].astype({"date": str}), on=["date", "ticker"])
    b = book(f.rename(columns={SUE_EVT: "y_pred"}))
    return {"n_rebalances": len(b), "gross": float(b["gross"].mean()),
            "net": float(b["net"].mean()), "t_gross": _t(b["gross"]), "t_net": _t(b["net"]),
            "turnover": float(b["turnover"].mean()),
            "sharpe_net_annual": float(b["net"].mean() / b["net"].std(ddof=1)
                                       * np.sqrt(252 / 30)) if len(b) > 2 else float("nan")}


# ── the verdicts ──────────────────────────────────────────────────────────────


def verdicts(pairs, placebo_rows, r4, feats, repro, pead) -> dict:
    b = pairs["sue_vs_baseline"]
    d = pairs["sue_vs_timing"]
    p_d = [r["pair"]["diff"] for r in placebo_rows]
    r2 = bool(placebo_rows) and b["diff"] > max(p_d)
    r3 = bool(np.isfinite(b["t"]) and b["t"] >= SIGNAL_T)
    r4_ok = bool(np.isfinite(r4["diff"]) and r4["diff"] > 0)
    r5 = bool(np.isfinite(d["diff"]) and d["diff"] > 0)
    failed = []
    if not r3:
        failed.append(f"R3 (paired cross-sectional t {b['t']:+.2f} < {SIGNAL_T:.1f})")
    if not r2:
        failed.append(f"R2 (gain {b['diff']:+.5f} does not beat the placebo max "
                      f"{max(p_d) if p_d else float('nan'):+.5f})")
    if not r4_ok:
        failed.append(f"R4 (net-of-cost book gain {r4['diff']:+.5f} per rebalance <= 0)")
    if not r5:
        failed.append(f"R5 (the surprise adds {d['diff']:+.5f} over timing and coverage "
                      f"alone, t {d['t']:+.2f})")
    evt = next((f for f in feats if f["feature"] == SUE_EVT), {"ic": float("nan")})
    predictions = {
        "P1 arm (a) reproduces": bool(repro.get("passed")),
        "P2 no signal: sue gain abs(d) < 0.005 and abs(t) < 2":
            bool(abs(b["diff"]) < 0.005 and abs(b["t"]) < 2),
        "P3 the raw PEAD book's net-of-cost return has t < 2":
            bool(not (pead["t_net"] >= 2)),
        "P4 sue_evt's own abs(IC) < 0.02": bool(abs(evt["ic"]) < 0.02),
        "P5 the placebo gains all sit within abs(d) < 0.005":
            bool(p_d and max(abs(x) for x in p_d) < 0.005),
    }
    return {"r2": r2, "r3": r3, "r4": r4_ok, "r5": r5,
            "signal": r3 and r2 and r4_ok and r5,
            "failed": failed, "placebo_max_diff": max(p_d) if p_d else float("nan"),
            "predictions": predictions}


# ── the report ────────────────────────────────────────────────────────────────


def render(inputs, design, repro, S, metrics, pairs, feats, placebo_rows, r4, pead, V,
           sweep_rows, args, runtime) -> str:
    o = ["# Stage 1, Pilot 3 — SRW SUE: run report\n",
         f"Pre-registration sha256 at run time: `{inputs['prereg_sha256']}`.  ",
         f"Panel {'matches' if inputs['panel_ok'] else '**DOES NOT MATCH**'} its frozen hash; "
         f"arm (a) {'matches' if inputs['baseline_ok'] else '**DOES NOT MATCH**'}; results "
         f"cache sha256 `{inputs['results_sha256'][:16]}…`.  ",
         f"B = {args.bootstrap}, trials = {args.trials}, placebo retrains = "
         f"{len(placebo_rows)}, DK lags = {DK_LAGS}, event window = {EVENT_WINDOW}. "
         f"{'**SMOKE RUN — not a measurement.**' if args.smoke else ''}\n",
         "## Design inputs (outcome-blind)\n"]
    for k, v in design.items():
        o.append(f"- {k}: {v}")
    o.append("\n## S1\n")
    o.append(f"{repro['rows_stored']:,} stored, {repro['rows_rerun']:,} re-run, max drift "
             f"{repro['max_pred_drift']:.1e}: **{'PASS' if repro['passed'] else 'FAIL'}**\n")
    o.append("## The arms, graded by `grade_panel_v3` (descriptive)\n")
    o.append("| arm | mu_hat | z | tau2 | STRONG | WEAK | INSUFF | RW | cs IC (DK SE) |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    for arm, s in S.items():
        o.append(f"| {arm} | {_f(s['mu_hat'])} | {_f(s['z'], '+.2f')} | {_f(s['tau2'], '.5f')} "
                 f"| {s['strong']} | {s['weak']} | {s['insufficient']} | {s['rw']} | "
                 f"{_f(s['cs_ic'])} ({_f(s['cs_ic_se'], '.5f')}) |")
    base = set(S["baseline"]["graded"])
    for arm in SUE_ARMS:
        o.append(f"- **{arm}** graded {S[arm]['graded'] or 'none'}; newly out of "
                 f"INSUFFICIENT {sorted(set(S[arm]['graded']) - base) or 'none'}; lost "
                 f"{sorted(base - set(S[arm]['graded'])) or 'none'}.")
    o.append("\n## R3 — the deciding rule: paired per-date cross-sectional IC\n")
    o.append("| comparison | dates | Δ | DK SE (30 lags) | **t** | t, default lags |")
    o.append("|---|---|---|---|---|---|")
    for key, lab in (("sue_vs_baseline", "(b) sue − (a)"), ("naive_vs_baseline", "(c) naive − (a)"),
                     ("timing_vs_baseline", "(d) timing − (a)"),
                     ("sue_vs_timing", "(b) sue − (d) timing  [R5]"),
                     ("sue_vs_naive", "(b) sue − (c) naive")):
        p = pairs[key]
        o.append(f"| {lab} | {p['n_dates']:,} | {_f(p['diff'])} | {_f(p['se'], '.5f')} | "
                 f"**{_f(p['t'], '+.2f')}** | {_f(p['t_default'], '+.2f')} |")
    o.append("\n## R2 — the corrected placebo: event columns permuted within date, retrained\n")
    o.append("| seed | Δ cs IC vs (a) | t | STRONG | WEAK | RW | tau2 |")
    o.append("|---|---|---|---|---|---|---|")
    for r in placebo_rows:
        o.append(f"| {r['seed']} | {_f(r['pair']['diff'])} | {_f(r['pair']['t'], '+.2f')} | "
                 f"{r['strong']} | {r['weak']} | {r['rw']} | {_f(r['tau2'], '.5f')} |")
    b = S["sue"]
    o.append(f"| **arm (b)** | **{_f(pairs['sue_vs_baseline']['diff'])}** | "
             f"**{_f(pairs['sue_vs_baseline']['t'], '+.2f')}** | {b['strong']} | {b['weak']} | "
             f"{b['rw']} | {_f(b['tau2'], '.5f')} |")
    o.append(f"\n**R2: {'PASS' if V['r2'] else 'FAIL'}** (placebo max {_f(V['placebo_max_diff'])}).\n")
    o.append("## R4 — net of the round-trip cost\n")
    o.append(f"Long-short top-minus-bottom quintile, {r4['n_rebalances']} non-overlapping "
             f"rebalances, {r4['round_trip'] * 100:.4f}% round trip on turnover.\n")
    o.append("| book | gross / rebalance | net / rebalance | turnover |")
    o.append("|---|---|---|---|")
    o.append(f"| (a) baseline | {_f(r4['gross_a'])} | {_f(r4['net_a'])} | {r4['turn_a']:.2f} |")
    o.append(f"| (b) sue | {_f(r4['gross_b'])} | {_f(r4['net_b'])} | {r4['turn_b']:.2f} |")
    o.append(f"\n(b) − (a), net: {_f(r4['diff'])} per rebalance, t {_f(r4['t'], '+.2f')}. "
             f"**R4: {'PASS' if V['r4'] else 'FAIL'}.**\n")
    o.append("## The raw PEAD book, net of cost (descriptive)\n")
    o.append(f"Sorted on sue_evt alone over arm (a)'s rows: gross {_f(pead['gross'])} "
             f"(t {_f(pead['t_gross'], '+.2f')}), net {_f(pead['net'])} "
             f"(t {_f(pead['t_net'], '+.2f')}) per rebalance over {pead['n_rebalances']}; "
             f"turnover {pead['turnover']:.2f}; net Sharpe {_f(pead['sharpe_net_annual'], '+.2f')} "
             f"annualised.\n")
    o.append("## Each SUE column's own cross-sectional IC\n")
    o.append("| feature | dates | IC | t |")
    o.append("|---|---|---|---|")
    for f in feats:
        o.append(f"| {f['feature']} | {f['n_dates']:,} | {_f(f['ic'])} | {_f(f['t'], '+.2f')} |")
    o.append("\n## Stage 2b's cell metrics\n")
    o.append("| arm | constant cells | reb IC | reb t | cs IC | OOS MAE | cs IC by fold |")
    o.append("|---|---|---|---|---|---|---|")
    for arm, m in metrics.items():
        o.append(f"| {arm} | {m['constant_cells']}/{m['cells']} | {_f(m['reb_ic'], '+.4f')} | "
                 f"{_f(m['reb_t'], '+.2f')} | {_f(m['cs_rank_ic'], '+.4f')} | {m['mae']:.5f} | "
                 f"{m['cs_rank_ic_by_fold']} |")
    o.append("\n## Verdicts, as pre-registered\n")
    o.append(f"- **R3** (deciding): {'PASS' if V['r3'] else 'FAIL'}.")
    o.append(f"- **R2** (placebo): {'PASS' if V['r2'] else 'FAIL'}.")
    o.append(f"- **R4** (net of cost): {'PASS' if V['r4'] else 'FAIL'}.")
    o.append(f"- **R5** (the surprise beyond timing/coverage): {'PASS' if V['r5'] else 'FAIL'}.")
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
    o.append(f"\nRuntime {runtime / 60:.1f} min.")
    return "\n".join(o)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--stage2b-npz", default=STAGE2B_NPZ)
    ap.add_argument("--results-cache", default=RESULTS_CACHE)
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
            args.npz = os.path.join(ROOT, "stage1c_smoke_oos.npz")

    started = time.time()
    raw_panel = load_cached_panel(args.panel_cache)
    tickers = sorted(raw_panel["ticker"].unique())
    grid = sorted(raw_panel["date"].astype(str).unique())
    ann, _ = sue_announcements(args.results_cache)
    unimputed = event_features(ann, grid, tickers, phantoms=PHANTOMS)
    design = design_inputs(ann, unimputed, grid)
    print(f"design inputs: {design}", flush=True)
    if args.design_only:
        return
    raw = impute_cross_sectional_median(unimputed, [SUE_EVT, SUE_AGE, SUE_FFILL])

    inputs = {
        "prereg_sha256": sha256_file(PREREG) if os.path.exists(PREREG) else "missing",
        "panel_sha256": sha256_file(args.panel_cache),
        "baseline_sha256": arm_arrays_sha256(args.stage2b_npz, BASELINE_ARM),
        "results_sha256": sha256_file(args.results_cache),
    }
    inputs["panel_ok"] = inputs["panel_sha256"].lower() == PANEL_SHA256
    inputs["baseline_ok"] = inputs["baseline_sha256"].lower() == BASELINE_SHA256
    print(f"pre-registration {inputs['prereg_sha256']}", flush=True)
    if inputs["prereg_sha256"] == "missing" and not args.smoke:
        raise SystemExit("no pre-registration on disk; refusing to run on real data")
    if not (inputs["panel_ok"] and inputs["baseline_ok"]) and not args.smoke:
        raise SystemExit("a frozen input changed since the pre-registration")

    panel = attach_sue(raw_panel, raw)
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
        raise SystemExit(f"S1 FAILED ({repro}); stopping before any SUE arm is read")

    for arm in SUE_ARMS:
        print(f"\n({arm}) ...", flush=True)
        preds, folds = run_arm(panel, "mae", "none", n_trials=args.trials,
                               features=ARM_FEATURES[arm])
        frames[arm] = preds
        metrics[arm] = cell_metrics(preds, folds)
    np.savez_compressed(args.npz, **{f"{k}__{c}": v[c].to_numpy()
                                     for k, v in frames.items() for c in COLS})

    data = {"baseline": stored, **{a: as_data(frames[a]) for a in SUE_ARMS}}
    S = {}
    for arm, d in data.items():
        print(f"grading {arm} ...", flush=True)
        S[arm] = summary(grade(d, args.bootstrap, auto_block=not args.smoke))
    ics = {arm: per_date_ic(d) for arm, d in data.items()}
    pairs = {"sue_vs_baseline": paired(ics["baseline"], ics["sue"]),
             "naive_vs_baseline": paired(ics["baseline"], ics["naive"]),
             "timing_vs_baseline": paired(ics["baseline"], ics["timing"]),
             "sue_vs_timing": paired(ics["timing"], ics["sue"]),
             "sue_vs_naive": paired(ics["naive"], ics["sue"])}
    base_frame = as_frame(stored)
    r4 = net_of_cost(base_frame, as_frame(data["sue"]))
    pead = pead_book(raw, base_frame)
    fics = feature_ics(raw, raw_panel, set(stored["dates"]))

    print(f"\ncorrected placebo: {args.placebo_draws} retrains with the event columns "
          f"permuted within date", flush=True)
    placebo_rows = []
    for seed in PLACEBO_SEEDS[:args.placebo_draws]:
        t0 = time.time()
        shuffled = permute_within_date(panel, EVENT_COLS, seed)
        preds, _ = run_arm(shuffled, "mae", "none", n_trials=args.trials,
                           features=ARM_FEATURES["sue"], verbose=False)
        d = as_data(preds)
        s = summary(grade(d, args.bootstrap, auto_block=False))
        s["pair"] = paired(ics["baseline"], per_date_ic(d))
        placebo_rows.append({"seed": seed, **s})
        print(f"    seed {seed}: gain {s['pair']['diff']:+.5f} (t {s['pair']['t']:+.2f}) "
              f"STRONG {s['strong']} RW {s['rw']} ({time.time() - t0:.0f}s)", flush=True)

    V = verdicts(pairs, placebo_rows, r4, fics, repro, pead)
    sweep_rows = []
    if V["signal"] and not args.smoke:
        print("\nSIGNAL: running the pre-registered min_train sweep", flush=True)
        for mt in SWEEP_MIN_TRAIN:
            for name in ("baseline", "sue"):
                p, f = run_arm(panel, "mae", "none", n_trials=args.trials,
                               min_train=mt, features=ARM_FEATURES[name], verbose=False)
                m = cell_metrics(p, f)
                sweep_rows.append({"min_train": mt, "arm": name, "reb_ic": m["reb_ic"],
                                   "reb_t": m["reb_t"], "cs_ic": m["cs_rank_ic"]})

    text = render(inputs, design, repro, S, metrics, pairs, fics, placebo_rows, r4, pead, V,
                  sweep_rows, args, time.time() - started)
    print("\n" + text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
