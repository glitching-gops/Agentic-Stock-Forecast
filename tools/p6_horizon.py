"""
tools/p6_horizon.py — P6: the horizon sweep, and SUE / delivery % re-tested at
five sessions.

Method, the grid, the purge rule, the stop condition and every decision rule
are fixed in `docs/p6-preregistration.md`, written and hashed before this ran
on real data. That file is a dated ADDENDUM to the P6 pre-registration of
2026-09-05; where the two differ it says so.

Part A — the sweep. Full RETRAIN and RELABEL at h in {5, 10, 20, 30} on the
pooled x MAE, no-ticker architecture, under two purge rules:

    legacy    purge = embargo = h        what every pre-P6 result used
    deciding  purge = embargo = max(h, 63)   the Politis-White floor

Part B — SUE and delivery %, each independently, against a properly re-derived
FIVE-SESSION baseline. Never against the 30-session one: that would confound
the horizon change with the feature's own effect.

    (a) baseline@5  FACTORS
    (b) sue@5       FACTORS + sue_evt, sue_age, sue_missing
    (c) timing@5    FACTORS + sue_age, sue_missing        the identity arm (R5)
    (d) delivery@5  FACTORS + deliv_abn_l1, deliv_abn5_l1

The ingestion is REUSED, not touched: `pipeline/earnings.py` and
`pipeline/delivery.py` are exactly what Pilots 2 and 3 built and tested. This
session changes the horizon, the label and the purge, and nothing else.

SANDBOXED. Reads panel_cache.parquet, stage2b_pooled_oos.npz,
delivery_cache.npz and results_cache.npz. Writes .npz predictions, a JSON of
results and a markdown report. Nothing the API, the web app or either
scheduled job reads.

    python tools/p6_horizon.py --smoke --part all --markdown <path>
    python tools/p6_horizon.py --part a --markdown p6_a.md
    python tools/p6_horizon.py --part b --markdown p6_b.md
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
from pipeline.delivery import ABNORMAL_COLS  # noqa: E402
from pipeline.earnings import EVENT_COLS, SUE_AGE, SUE_EVT, SUE_MISSING  # noqa: E402
from pipeline.evaluation import (  # noqa: E402
    POLITIS_WHITE_FLOOR_SESSIONS,
    cross_sectional_report,
    horizon_purge_embargo,
)
from pipeline.label import standardise_target as _standardise_target  # noqa: E402
from pipeline.evidence_panel import (  # noqa: E402
    BOOTSTRAP_B,
    driscoll_kraay_se,
    optimal_block_length,
)
from pipeline.model import EVAL_TUNE_TRIALS  # noqa: E402
from pipeline.panel import TARGET, retarget_horizon  # noqa: E402
from pipeline.portfolio import CostModel, break_even_ic, simulate, synthetic_predictions  # noqa: E402
from tools.backfill_delivery import CACHE as DELIVERY_CACHE  # noqa: E402
from tools.backfill_results import CACHE as RESULTS_CACHE  # noqa: E402
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
    grade,
    paired,
    per_date_ic,
    reproduction,
    sha256_file,
    summary,
)
from tools.stage1b_delivery import attach_delivery, delivery_long, permute_within_date  # noqa: E402
from tools.stage1c_sue import attach_sue, raw_features, sue_announcements  # noqa: E402
from tools.stage2b_pooled import cell_metrics, load_cached_panel, run_arm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
STAGE2B_NPZ = os.path.join(ROOT, "stage2b_pooled_oos.npz")
PREREG = os.path.join(ROOT, "docs", "p6-preregistration.md")
PRED_A = os.path.join(ROOT, "p6_sweep_oos.npz")
PRED_B = os.path.join(ROOT, "p6_five_oos.npz")
STATE = os.path.join(ROOT, "p6_results.json")

HORIZONS: tuple[int, ...] = (5, 10, 20, 30)
RULES: tuple[str, ...] = ("legacy", "deciding")
PLACEBO_SEEDS: tuple[int, ...] = tuple(range(20260920, 20260929))   # nine
SWEEP_MIN_TRAIN = (380, 420, 460, 500, 540, 580)
QUANTILES = 5
FIVE = 5

#: The planted edges used to estimate long-short spread per unit of rank IC at
#: each horizon. P4's `break_even_ic` takes that quantity as an argument and
#: says it "must be estimated from the data rather than assumed" — and it is
#: not transferable across horizons, because a 5-session return is smaller than
#: a 30-session one, so the same IC buys a smaller spread.
PLANTED_ICS = (0.02, 0.05, 0.10, 0.20)

ARM_FEATURES: dict[str, list[str]] = {
    "baseline": list(FACTORS),
    "sue": list(FACTORS) + EVENT_COLS,
    "timing": list(FACTORS) + [SUE_AGE, SUE_MISSING],
    "delivery": list(FACTORS) + list(ABNORMAL_COLS),
}
#: Which columns each Part B arm's placebo permutes, jointly, within each date.
PLACEBO_COLS: dict[str, list[str]] = {
    "sue": list(EVENT_COLS),
    "delivery": list(ABNORMAL_COLS),
}
#: R5 applies to the SUE arm alone. Delivery's two columns are one
#: construction, and Pilot 2 already ran the level arm that plays that role.
R5_ARMS = ("sue",)


# ── the label at one horizon ──────────────────────────────────────────────────


def panel_at(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    The panel with `TARGET` rebuilt as the `horizon`-session forward log
    return, through the exact identity in ``panel.price_frame``.

    Applied at EVERY horizon in the sweep, 30 included, so the four cells share
    one label construction. That matters more than it looks: the STORED
    30-session label was computed from full-precision closes while
    ``retarget_horizon`` rebuilds it from the closes as stored to three
    decimals, and the two differ by up to 5.0e-06 on a label of order 0.1.
    Mixing them would put a 5e-05 relative label difference between the h=30
    cell and its neighbours, which is far smaller than anything measured here
    but is not zero, and "far smaller than the effect" is the argument that
    retired three results in this project.

    The A0 regression pin therefore runs on the STORED label instead, which is
    the only way it can reproduce the frozen predictions at 1e-9.
    """
    out = retarget_horizon(panel, horizon, target=TARGET)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def labelled_rows(panel: pd.DataFrame) -> int:
    return int(pd.to_numeric(panel[TARGET], errors="coerce").notna().sum())


# ── money, at a horizon ───────────────────────────────────────────────────────


def book(frame: pd.DataFrame, horizon: int,
         costs: CostModel | None = None) -> pd.DataFrame:
    """
    The long-short top-minus-bottom quintile book on one arm's predictions,
    rebalanced every `horizon` sessions so successive windows do not overlap.

    `rebalance_every` MUST track the horizon. Left at 30 it would sample every
    30th date of a 5-session label, throwing away five sixths of the
    independent windows the shorter horizon bought.
    """
    b = simulate(frame[["date", "ticker", "y_pred", "y_true"]],
                 costs=costs or CostModel(), rebalance_every=horizon,
                 quantiles=QUANTILES, long_only=False)
    return pd.DataFrame({"date": b.dates, "gross": b.gross_returns,
                         "net": b.net_returns, "turnover": b.turnover})


def _t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def _sharpe(x, horizon: int) -> float:
    """
    Annualised explicitly, from THIS horizon's own rebalance count.

    ``portfolio.REBALANCES_PER_YEAR`` is a module constant pinned at
    252/30 = 8.4 — the defect the 2026-09-05 pre-registration flagged and this
    session deliberately does not fix, because it is production code. Nothing
    here reads it: an h=5 book annualised at 8.4 rebalances a year would
    understate its drag six-fold.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0 / horizon))


def net_of_cost(base: pd.DataFrame, arm: pd.DataFrame, horizon: int) -> dict:
    """R4: arm minus baseline per NON-OVERLAPPING rebalance, net of the
    0.2225% round trip. The windows do not overlap, so a plain t is right."""
    a, b = book(base, horizon), book(arm, horizon)
    j = a.merge(b, on="date", suffixes=("_a", "_b"))
    d = (j["net_b"] - j["net_a"]).to_numpy()
    return {"n_rebalances": len(j), "net_a": float(j["net_a"].mean()),
            "net_b": float(j["net_b"].mean()), "gross_a": float(j["gross_a"].mean()),
            "gross_b": float(j["gross_b"].mean()),
            "turn_a": float(j["turnover_a"].mean()),
            "turn_b": float(j["turnover_b"].mean()),
            "diff": float(np.mean(d)) if len(d) else float("nan"),
            "t": _t(d), "round_trip": CostModel().round_trip}


def raw_sort_book(raw: pd.DataFrame, base: pd.DataFrame, col: str,
                  horizon: int) -> dict:
    """Sort on one feature column alone over the baseline arm's own rows: the
    literal event-drift test at this horizon."""
    f = base[["date", "ticker", "y_true"]].astype({"date": str}).merge(
        raw[["date", "ticker", col]].astype({"date": str}), on=["date", "ticker"])
    if f.empty:
        return {"n_rebalances": 0}
    b = book(f.rename(columns={col: "y_pred"}), horizon)
    return {"feature": col, "n_rebalances": len(b),
            "gross": float(b["gross"].mean()), "net": float(b["net"].mean()),
            "t_gross": _t(b["gross"]), "t_net": _t(b["net"]),
            "turnover": float(b["turnover"].mean()),
            "sharpe_net_annual": _sharpe(b["net"], horizon)}


def spread_per_ic(preds: pd.DataFrame, horizon: int, seed: int = 20260920) -> float:
    """
    Long-short spread produced per unit of rank IC, at THIS horizon, from this
    horizon's own rows.

    Estimated by planting edges of known size with ``synthetic_predictions``
    rather than read off the arm's own ordering: a near-zero measured IC gives
    spread/IC no stable value at all, and P4 took the median over comparators
    that happened to have a positive IC — which at these horizons may be none.
    """
    truth = preds[["date", "ticker", "y_true"]].copy()
    ratios = []
    for target_ic in PLANTED_ICS:
        synth = synthetic_predictions(truth, target_ic, seed=seed)
        xs = cross_sectional_report(synth[["date", "ticker", "y_pred", "y_true"]],
                                    rebalance_every=horizon)
        ic, spread = xs.get("mean_rank_ic"), xs.get("long_short_spread")
        if ic and spread and np.isfinite(ic) and np.isfinite(spread) and ic > 0:
            ratios.append(spread / ic)
    return float(np.median(ratios)) if ratios else float("nan")


# ── one cell ──────────────────────────────────────────────────────────────────


def run_cell(panel: pd.DataFrame, horizon: int, purge: int, features: list[str],
             trials: int, min_train: int = 500,
             verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    preds, folds = run_arm(panel, "mae", "none", n_trials=trials,
                           min_train=min_train, features=features,
                           horizon=horizon, purge=purge, verbose=verbose)
    m = cell_metrics(preds, folds, rebalance_every=horizon)
    return preds, m


def score(preds: pd.DataFrame, horizon: int, bootstrap: int,
          auto_block: bool) -> dict:
    """The panel-level grade, the cross-sectional IC and its DK SE."""
    data = as_data(preds)
    s = summary(grade(data, bootstrap, auto_block=auto_block))
    ics = per_date_ic(data)
    clean = ics.dropna()
    mean, se, lags = driscoll_kraay_se(clean.index.to_numpy(),
                                       clean.to_numpy(), max_lag=DK_LAGS)
    s["cs_ic_dk"] = mean
    s["cs_ic_dk_se"] = se
    s["cs_ic_dk_t"] = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
    s["cs_ic_dk_lags"] = lags
    s["n_ic_dates"] = int(len(clean))
    # Politis-White on THIS horizon's own realised IC series, recomputed after
    # the run rather than before it — a floor derived from the run it is meant
    # to constrain would not be a constraint.
    try:
        s["pw_auto_block"] = optimal_block_length(clean.index.to_numpy(),
                                                  clean.to_numpy())
    except Exception as exc:                                    # noqa: BLE001
        s["pw_auto_block"] = float("nan")
        s["pw_note"] = str(exc)
    return s


# ── Part A ────────────────────────────────────────────────────────────────────


def part_a(raw_panel: pd.DataFrame, stored: dict, args) -> dict:
    out: dict = {"pin": {}, "cells": {}, "sweep": []}

    # A0 — the stop condition, on the STORED label and the legacy rule.
    print("\nA0 — regression pin: h=30, legacy purge, stored label", flush=True)
    # The ONLY opt-out in the codebase: this pin exists to reproduce
    # predictions frozen under the raw label, so it must not standardise.
    preds, folds = run_arm(raw_panel, "mae", "none", n_trials=args.trials,
                           features=ARM_FEATURES["baseline"],
                           horizon=30, purge=30, standardise_label=False)
    repro = reproduction(stored, as_data(preds))
    out["pin"] = repro
    print(f"  A0 {'PASS' if repro['passed'] else 'FAIL'} "
          f"(drift {repro['max_pred_drift']:.1e}, truth {repro['max_truth_drift']:.1e})",
          flush=True)
    if not repro["passed"] and not args.smoke:
        raise SystemExit(f"A0 FAILED ({repro}); stopping before any horizon is read")

    frames: dict[str, pd.DataFrame] = {}
    for h in args.horizons:
        panel_h = panel_at(raw_panel, h)
        n_lab = labelled_rows(panel_h)
        for rule in RULES:
            purge = h if rule == "legacy" else horizon_purge_embargo(h)
            key = f"h{h}_{rule}"
            print(f"\n{key}: purge/embargo {purge}, labelled rows {n_lab:,}", flush=True)
            t0 = time.time()
            preds, m = run_cell(panel_h, h, purge, ARM_FEATURES["baseline"],
                                args.trials)
            frames[key] = preds
            s = score(preds, h, args.bootstrap, auto_block=not args.smoke)
            be = break_even_ic(CostModel(),
                               spread_per_ic=spread_per_ic(preds, h),
                               turnover=m.get("mean_turnover", 0.80) or 0.80)
            row = {"horizon": h, "rule": rule, "purge": purge,
                   "labelled_rows": n_lab, "secs": round(time.time() - t0, 1),
                   **{k: m[k] for k in ("reb_ic", "reb_t", "n_rebalances",
                                        "cs_rank_ic", "cs_rank_ic_by_fold",
                                        "mae", "gap", "constant_cells", "cells",
                                        "n_rows", "n_dates_no_ordering")},
                   **{k: s[k] for k in ("mu_hat", "se_boot", "z", "tau2", "strong",
                                        "weak", "insufficient", "rw", "graded",
                                        "cs_ic_dk", "cs_ic_dk_se", "cs_ic_dk_t",
                                        "n_ic_dates", "pw_auto_block")},
                   "break_even_ic": be["break_even_rank_ic"],
                   "spread_per_ic": be["spread_per_unit_ic"]}
            out["cells"][key] = row
            out["sweep"].append(row)
            print(f"  cs IC {row['cs_ic_dk']:+.5f} (DK SE {row['cs_ic_dk_se']:.5f}) "
                  f"t {row['cs_ic_dk_t']:+.2f} | reb IC {row['reb_ic']:+.4f} "
                  f"t {row['reb_t']:+.2f} over {row['n_rebalances']} | "
                  f"{row['strong']}/{row['weak']}/{row['insufficient']} | "
                  f"PW {row['pw_auto_block']:.1f} | {row['secs']:.0f}s", flush=True)

    np.savez_compressed(args.npz_a, **{f"{k}__{c}": v[c].to_numpy()
                                       for k, v in frames.items() for c in COLS})
    out["signalling"] = [r["horizon"] for r in out["sweep"]
                         if r["rule"] == "deciding"
                         and np.isfinite(r["cs_ic_dk_t"])
                         and r["cs_ic_dk_t"] >= SIGNAL_T]

    # A4 — the min_train sweep, only if A1 fired somewhere.
    out["min_train_sweep"] = []
    if out["signalling"] and not args.smoke:
        for h in sorted(set(out["signalling"])):
            panel_h = panel_at(raw_panel, h)
            purge = horizon_purge_embargo(h)
            print(f"\nA4 — min_train sweep at h={h}", flush=True)
            for mt in SWEEP_MIN_TRAIN:
                preds, m = run_cell(panel_h, h, purge, ARM_FEATURES["baseline"],
                                    args.trials, min_train=mt, verbose=False)
                ics = per_date_ic(as_data(preds)).dropna()
                mean, se, _ = driscoll_kraay_se(ics.index.to_numpy(),
                                                ics.to_numpy(), max_lag=DK_LAGS)
                t = mean / se if se and se > 0 else float("nan")
                out["min_train_sweep"].append(
                    {"horizon": h, "min_train": mt, "cs_ic_dk": mean,
                     "cs_ic_dk_t": t, "reb_ic": m["reb_ic"], "reb_t": m["reb_t"]})
                print(f"    min_train {mt}: cs IC {mean:+.5f}, t {t:+.2f}", flush=True)
    return out


# ── Part B ────────────────────────────────────────────────────────────────────


def part_b(raw_panel: pd.DataFrame, args, standardise: bool = False) -> dict:
    """
    SUE and delivery %, each alone, against a re-derived five-session baseline.

    `standardise` refits every arm on a WITHIN-DATE STANDARDISED label, which
    is Part C's formulation applied to Part B. It is NOT the pre-registered
    run and decides nothing — it exists because the pre-registered run's
    baseline@5 turned out to be constant on every date, and an arm compared
    against a baseline with no ordering at all is not a comparison. Both are
    reported.
    """
    h = FIVE
    purge = horizon_purge_embargo(h)
    tickers = sorted(raw_panel["ticker"].unique())
    grid = sorted(raw_panel["date"].astype(str).unique())

    ann, _ = sue_announcements(args.results_cache)
    sue_raw = raw_features(ann, grid, tickers)
    deliv = delivery_long(args.delivery_cache, tickers)
    from pipeline.delivery import delivery_features
    deliv_feats = delivery_features(deliv, grid)

    panel_h = panel_at(raw_panel, h)
    panel_h = attach_sue(panel_h, sue_raw)
    panel_h = attach_delivery(panel_h, deliv_feats)
    # The REAL h-session label, kept aside before any standardisation, because
    # every arm is scored against it whichever label it was fitted on.
    truth = panel_h[["date", "ticker", TARGET]].astype({"date": str}) \
        .rename(columns={TARGET: "y_real"})
    if standardise:
        panel_h = standardise_target(panel_h)
    print(f"\nPart B panel: {panel_h.shape}, labelled {labelled_rows(panel_h):,}"
          f"{' (within-date standardised label)' if standardise else ''}",
          flush=True)

    out: dict = {"horizon": h, "purge": purge, "standardised": bool(standardise),
                 "arms": {}, "pairs": {}, "placebo": {}, "r4": {},
                 "raw_books": {}, "verdicts": {}}

    def rescore(preds: pd.DataFrame) -> pd.DataFrame:
        """Predictions re-joined to the real h-session label. A no-op unless
        the arm was fitted on the standardised one."""
        if not standardise:
            return preds
        m = preds.astype({"date": str}).merge(truth, on=["date", "ticker"],
                                              how="inner")
        m = m[np.isfinite(m["y_real"])]
        return m.drop(columns=["y_true"]).rename(columns={"y_real": "y_true"})
    frames, preds_by_arm = {}, {}
    for arm in ("baseline", "sue", "timing", "delivery"):
        print(f"\n({arm}@{h}) purge {purge} ...", flush=True)
        t0 = time.time()
        preds, _ = run_cell(panel_h, h, purge, ARM_FEATURES[arm], args.trials)
        preds = rescore(preds)
        m = cell_metrics(preds, rebalance_every=h)
        preds_by_arm[arm] = preds
        frames[arm] = preds
        s = score(preds, h, args.bootstrap, auto_block=not args.smoke)
        out["arms"][arm] = {**{k: m[k] for k in ("reb_ic", "reb_t", "n_rebalances",
                                                 "cs_rank_ic", "mae",
                                                 "n_dates_no_ordering",
                                                 "constant_cells", "cells")},
                            **{k: s[k] for k in ("mu_hat", "se_boot", "z", "tau2",
                                                 "strong", "weak", "insufficient",
                                                 "rw", "graded", "cs_ic_dk",
                                                 "cs_ic_dk_se", "cs_ic_dk_t")},
                            "secs": round(time.time() - t0, 1)}
        print(f"  cs IC {s['cs_ic_dk']:+.5f} t {s['cs_ic_dk_t']:+.2f} | "
              f"{s['strong']}/{s['weak']}/{s['insufficient']} | "
              f"{time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(args.npz_b, **{f"{k}__{c}": v[c].to_numpy()
                                       for k, v in frames.items() for c in COLS})

    ics = {a: per_date_ic(as_data(p)) for a, p in preds_by_arm.items()}
    out["pairs"] = {
        "sue_vs_baseline": paired(ics["baseline"], ics["sue"]),
        "timing_vs_baseline": paired(ics["baseline"], ics["timing"]),
        "delivery_vs_baseline": paired(ics["baseline"], ics["delivery"]),
        "sue_vs_timing": paired(ics["timing"], ics["sue"]),
    }

    for arm in ("sue", "delivery"):
        out["r4"][arm] = net_of_cost(preds_by_arm["baseline"], preds_by_arm[arm], h)

    for col in (SUE_EVT, ABNORMAL_COLS[0]):
        src = sue_raw if col == SUE_EVT else deliv_feats
        src = src.assign(date=src["date"].astype(str))
        out["raw_books"][col] = raw_sort_book(src, preds_by_arm["baseline"], col, h)

    # R2 — the corrected placebo: the arm's own columns permuted jointly
    # within each date, and the model RETRAINED. Predictions are never
    # shuffled; that is the error Pilot 1 recorded.
    for arm in ("sue", "delivery"):
        rows = []
        cols = PLACEBO_COLS[arm]
        print(f"\nR2 placebo for {arm}@{h}: {args.placebo_draws} retrains, "
              f"columns {cols} permuted within date", flush=True)
        for seed in PLACEBO_SEEDS[:args.placebo_draws]:
            t0 = time.time()
            shuffled = permute_within_date(panel_h, cols, seed)
            p, _ = run_cell(shuffled, h, purge, ARM_FEATURES[arm], args.trials,
                            verbose=False)
            d = as_data(rescore(p))
            s = summary(grade(d, args.bootstrap, auto_block=False))
            s["pair"] = paired(ics["baseline"], per_date_ic(d))
            rows.append({"seed": seed, "diff": s["pair"]["diff"],
                         "t": s["pair"]["t"], "strong": s["strong"],
                         "weak": s["weak"], "rw": s["rw"], "tau2": s["tau2"]})
            print(f"    seed {seed}: gain {s['pair']['diff']:+.5f} "
                  f"(t {s['pair']['t']:+.2f}) STRONG {s['strong']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        out["placebo"][arm] = rows

    out["verdicts"] = verdicts_b(out)
    return out


def verdicts_b(b: dict) -> dict:
    v = {}
    for arm in ("sue", "delivery"):
        pair = b["pairs"][f"{arm}_vs_baseline"]
        p_d = [r["diff"] for r in b["placebo"].get(arm, [])]
        r3 = bool(np.isfinite(pair["t"]) and pair["t"] >= SIGNAL_T)
        r2 = bool(p_d) and pair["diff"] > max(p_d)
        r4 = bool(np.isfinite(b["r4"][arm]["diff"]) and b["r4"][arm]["diff"] > 0)
        r5 = None
        if arm in R5_ARMS:
            d = b["pairs"]["sue_vs_timing"]
            r5 = bool(np.isfinite(d["diff"]) and d["diff"] > 0)
        failed = []
        if not r3:
            failed.append(f"R3 (paired cs t {pair['t']:+.2f} < {SIGNAL_T:.1f})")
        if not r2:
            failed.append(f"R2 (gain {pair['diff']:+.5f} does not beat the placebo "
                          f"max {max(p_d) if p_d else float('nan'):+.5f})")
        if not r4:
            failed.append(f"R4 (net book gain {b['r4'][arm]['diff']:+.5f} "
                          f"per rebalance <= 0)")
        if r5 is False:
            failed.append(f"R5 (the surprise adds "
                          f"{b['pairs']['sue_vs_timing']['diff']:+.5f} over timing "
                          f"alone, t {b['pairs']['sue_vs_timing']['t']:+.2f})")
        v[arm] = {"r2": r2, "r3": r3, "r4": r4, "r5": r5,
                  "signal": r3 and r2 and r4 and (r5 is not False),
                  "failed": failed,
                  "placebo_max": max(p_d) if p_d else float("nan")}
    return v


# ── Part C — the scale diagnostic ─────────────────────────────────────────────


# ONE implementation of the transform, in `pipeline/label.py`, imported rather
# than re-derived. It was defined here when it was a post-hoc diagnostic; it is
# the pipeline's default training target now, and two copies of a label
# definition that can disagree is the `_log_price_basis` landmine waiting to
# happen. Re-exported so this module's own tests and callers keep working.
standardise_target = _standardise_target


def part_c(raw_panel: pd.DataFrame, args) -> dict:
    """
    Descriptive, post hoc, and it decides nothing.

    Part A's shorter horizons came back mostly CONSTANT, and a null from a
    model that never split is uninterpretable — the same problem Pilot 3's
    announcement check exists to rule out. This refits each horizon on a
    within-date standardised label, so `gamma` means the same thing at h=5 as
    at h=30, and scores the result against the REAL h-session label.

    If the degeneracy collapses and the IC stays null, the sweep's null is
    about the panel. If the IC moves, the sweep's shorter horizons were
    measuring the tuner rather than the market.
    """
    rows = []
    for h in args.horizons:
        purge = horizon_purge_embargo(h)
        panel_h = panel_at(raw_panel, h)
        truth = panel_h[["date", "ticker", TARGET]].astype({"date": str}) \
            .rename(columns={TARGET: "y_real"})

        print(f"\nPart C — h={h}, within-date standardised label, purge {purge}",
              flush=True)
        preds, m = run_cell(standardise_target(panel_h), h, purge,
                            ARM_FEATURES["baseline"], args.trials)

        # Scored against the REAL h-session label, never the standardised one.
        scored = preds.astype({"date": str}).merge(truth, on=["date", "ticker"],
                                                   how="inner")
        scored = scored[np.isfinite(scored["y_real"])]
        real = scored.drop(columns=["y_true"]).rename(columns={"y_real": "y_true"})
        ics = per_date_ic(as_data(real)).dropna()
        mean, se, _ = driscoll_kraay_se(ics.index.to_numpy(), ics.to_numpy(),
                                        max_lag=DK_LAGS)
        rm = cell_metrics(real, rebalance_every=h)
        rows.append({
            "horizon": h, "purge": purge,
            "constant_cells_std": m["constant_cells"], "cells": m["cells"],
            "n_dates_no_ordering": rm["n_dates_no_ordering"],
            "n_rebalances": rm["n_rebalances"],
            "cs_ic_dk": mean, "cs_ic_dk_se": se,
            "cs_ic_dk_t": mean / se if se and se > 0 else float("nan"),
            "reb_ic": rm["reb_ic"], "reb_t": rm["reb_t"],
        })
        print(f"  constant cells {m['constant_cells']}/{m['cells']}, "
              f"no-ordering dates {rm['n_dates_no_ordering']}, "
              f"cs IC {mean:+.5f} t {rows[-1]['cs_ic_dk_t']:+.2f}", flush=True)
    return {"rows": rows}


# ── the report ────────────────────────────────────────────────────────────────


def render(inputs: dict, A: dict | None, B: dict | None, C: dict | None,
           args, runtime: float, B_std: dict | None = None) -> str:
    o = ["# P6 — the horizon sweep, and SUE / delivery % at five sessions\n",
         f"Pre-registration sha256 at run time: `{inputs['prereg_sha256']}`.  ",
         f"Panel {'matches' if inputs['panel_ok'] else '**DOES NOT MATCH**'} its "
         f"frozen hash; the stored baseline "
         f"{'matches' if inputs['baseline_ok'] else '**DOES NOT MATCH**'}.  ",
         f"B = {args.bootstrap}, trials = {args.trials}, DK lags = {DK_LAGS}, "
         f"Politis-White floor = {POLITIS_WHITE_FLOOR_SESSIONS}. "
         f"{'**SMOKE RUN — not a measurement.**' if args.smoke else ''}\n"]

    if A:
        p = A["pin"]
        o += ["## A0 — the regression pin (h=30, legacy purge, stored label)\n",
              f"{p['rows_stored']:,} stored, {p['rows_rerun']:,} re-run, "
              f"only-stored {p['only_stored']}, only-rerun {p['only_rerun']}, "
              f"max prediction drift {p['max_pred_drift']:.1e}, max label drift "
              f"{p['max_truth_drift']:.1e}: "
              f"**{'PASS' if p['passed'] else 'FAIL'}**\n",
              "## Part A — the sweep\n",
              "| h | rule | purge | rows | OOS dates | **cs IC (DK SE)** | **t** | "
              "reb IC | reb t | n reb | S/W/I | mu_hat | z | tau2 | PW auto | "
              "break-even IC |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in A["sweep"]:
            o.append(
                f"| {r['horizon']} | {r['rule']} | {r['purge']} | {r['n_rows']:,} | "
                f"{r['n_ic_dates']:,} | {_f(r['cs_ic_dk'])} ({_f(r['cs_ic_dk_se'], '.5f')}) | "
                f"**{_f(r['cs_ic_dk_t'], '+.2f')}** | {_f(r['reb_ic'], '+.4f')} | "
                f"{_f(r['reb_t'], '+.2f')} | {r['n_rebalances']} | "
                f"{r['strong']}/{r['weak']}/{r['insufficient']} | "
                f"{_f(r['mu_hat'])} | {_f(r['z'], '+.2f')} | {_f(r['tau2'], '.5f')} | "
                f"{_f(r['pw_auto_block'], '.1f')} | {_f(r['break_even_ic'], '.4f')} |")
        o.append("")
        o.append(f"**A1: horizons reaching t >= {SIGNAL_T:.1f} on the deciding rule: "
                 f"{A['signalling'] or 'NONE'}.**\n")
        o.append("Cross-sectional IC by fold, deciding rule:\n")
        o.append("| h | by fold |")
        o.append("|---|---|")
        for r in A["sweep"]:
            if r["rule"] == "deciding":
                o.append(f"| {r['horizon']} | {r['cs_rank_ic_by_fold']} |")
        o.append("")
        for r in A["sweep"]:
            if r["rule"] == "deciding" and r["graded"]:
                o.append(f"- h={r['horizon']} graded: {r['graded']}")
        o.append("")
        if A.get("min_train_sweep"):
            o += ["### A4 — the min_train sweep\n",
                  "| h | min_train | cs IC | t | reb IC | reb t |",
                  "|---|---|---|---|---|---|"]
            for r in A["min_train_sweep"]:
                o.append(f"| {r['horizon']} | {r['min_train']} | {_f(r['cs_ic_dk'])} | "
                         f"{_f(r['cs_ic_dk_t'], '+.2f')} | {_f(r['reb_ic'], '+.4f')} | "
                         f"{_f(r['reb_t'], '+.2f')} |")
            o.append("")

    for B in [b for b in (B, B_std) if b]:
        h = B["horizon"]
        o += [f"## Part B{' (POST HOC: within-date standardised label)' if B.get('standardised') else ''}"
              f" — SUE and delivery % at h={h}, against a re-derived "
              f"{h}-session baseline\n",
              f"Purge = embargo = {B['purge']}. Every arm below is scored on the "
              f"same folds and the same rows as baseline@{h}.\n",
              "| arm | cs IC (DK SE) | **t** | reb IC | reb t | S / W / I | "
              "constant cells | n reb |",
              "|---|---|---|---|---|---|---|---|"]
        for arm, a in B["arms"].items():
            o.append(f"| {arm}@{h} | {_f(a['cs_ic_dk'])} ({_f(a['cs_ic_dk_se'], '.5f')}) | "
                     f"**{_f(a['cs_ic_dk_t'], '+.2f')}** | {_f(a['reb_ic'], '+.4f')} | "
                     f"{_f(a['reb_t'], '+.2f')} | {a['strong']} / {a['weak']} / "
                     f"{a['insufficient']} | {a['constant_cells']}/{a['cells']} | "
                     f"{a['n_rebalances']} |")
        o += ["", "### R3 — the deciding rule: paired per-date cross-sectional IC\n",
              "| comparison | dates | Δ | DK SE (30 lags) | **t** | t, default lags |",
              "|---|---|---|---|---|---|"]
        for key, lab in (("sue_vs_baseline", f"sue@{h} − baseline@{h} **[R3]**"),
                         ("delivery_vs_baseline", f"delivery@{h} − baseline@{h} **[R3]**"),
                         ("timing_vs_baseline", f"timing@{h} − baseline@{h}"),
                         ("sue_vs_timing", f"sue@{h} − timing@{h} **[R5]**")):
            p = B["pairs"][key]
            o.append(f"| {lab} | {p['n_dates']:,} | {_f(p['diff'])} | "
                     f"{_f(p['se'], '.5f')} | **{_f(p['t'], '+.2f')}** | "
                     f"{_f(p['t_default'], '+.2f')} |")
        o += ["", "### R4 — net of the 0.2225% round trip, per rebalance\n",
              "| arm | n reb | gross base | gross arm | net base | net arm | "
              "Δ net | t | turn base | turn arm |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for arm, r in B["r4"].items():
            o.append(f"| {arm}@{h} | {r['n_rebalances']} | {_f(r['gross_a'], '+.4f')} | "
                     f"{_f(r['gross_b'], '+.4f')} | {_f(r['net_a'], '+.4f')} | "
                     f"{_f(r['net_b'], '+.4f')} | **{_f(r['diff'], '+.5f')}** | "
                     f"{_f(r['t'], '+.2f')} | {r['turn_a']:.2f} | {r['turn_b']:.2f} |")
        o += ["", f"### The raw sort books at h={h} (descriptive)\n",
              "| feature | n reb | gross | t | net | t | turnover | net Sharpe |",
              "|---|---|---|---|---|---|---|---|"]
        for col, r in B["raw_books"].items():
            if not r.get("n_rebalances"):
                continue
            o.append(f"| {col} | {r['n_rebalances']} | {_f(r['gross'], '+.4f')} | "
                     f"{_f(r['t_gross'], '+.2f')} | {_f(r['net'], '+.4f')} | "
                     f"{_f(r['t_net'], '+.2f')} | {r['turnover']:.2f} | "
                     f"{_f(r['sharpe_net_annual'], '+.2f')} |")
        for arm, rows in B["placebo"].items():
            o += ["", f"### R2 — the corrected placebo for {arm}@{h} "
                  f"({PLACEBO_COLS[arm]} permuted within date, retrained)\n",
                  "| seed | Δ cs IC vs baseline | t | STRONG | WEAK | RW | tau2 |",
                  "|---|---|---|---|---|---|---|"]
            for r in rows:
                o.append(f"| {r['seed']} | {_f(r['diff'])} | {_f(r['t'], '+.2f')} | "
                         f"{r['strong']} | {r['weak']} | {r['rw']} | "
                         f"{_f(r['tau2'], '.5f')} |")
            p = B["pairs"][f"{arm}_vs_baseline"]
            a = B["arms"][arm]
            o.append(f"| **arm {arm}** | **{_f(p['diff'])}** | "
                     f"**{_f(p['t'], '+.2f')}** | {a['strong']} | {a['weak']} | "
                     f"{a['rw']} | {_f(a['tau2'], '.5f')} |")
        o += ["", "### Verdicts, as pre-registered\n",
              "| arm | R3 | R2 | R4 | R5 | **SIGNAL** |", "|---|---|---|---|---|---|"]
        for arm, v in B["verdicts"].items():
            r5 = "n/a" if v["r5"] is None else ("PASS" if v["r5"] else "FAIL")
            o.append(f"| {arm}@{h} | {'PASS' if v['r3'] else 'FAIL'} | "
                     f"{'PASS' if v['r2'] else 'FAIL'} | "
                     f"{'PASS' if v['r4'] else 'FAIL'} | {r5} | "
                     f"**{'YES' if v['signal'] else 'NO'}** |")
        o.append("")
        for arm, v in B["verdicts"].items():
            if v["failed"]:
                o.append(f"- **{arm}@{h}** failed: {'; '.join(v['failed'])}.")
        o.append("")

    if C:
        o += ["## Part C — the scale diagnostic (descriptive, decides nothing)\n",
              "Each horizon refitted on a WITHIN-DATE STANDARDISED label, so "
              "`gamma` means the same thing at every horizon, then scored "
              "against the REAL h-session label. A null from a model that never "
              "split is uninterpretable; this is what tells the two apart.\n",
              "| h | constant cells | no-ordering dates | n reb | cs IC (DK SE) | "
              "**t** | reb IC | reb t |",
              "|---|---|---|---|---|---|---|---|"]
        for r in C["rows"]:
            o.append(f"| {r['horizon']} | {r['constant_cells_std']}/{r['cells']} | "
                     f"{r['n_dates_no_ordering']} | {r['n_rebalances']} | "
                     f"{_f(r['cs_ic_dk'])} ({_f(r['cs_ic_dk_se'], '.5f')}) | "
                     f"**{_f(r['cs_ic_dk_t'], '+.2f')}** | "
                     f"{_f(r['reb_ic'], '+.4f')} | {_f(r['reb_t'], '+.2f')} |")
        o.append("")

    o.append(f"Every SE here is likely too small: Politis-White puts this panel's "
             f"dependence at 35.8-62.5 sessions against a block of 30. "
             f"Runtime {runtime / 60:.1f} min.")
    return "\n".join(o)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--stage2b-npz", default=STAGE2B_NPZ)
    ap.add_argument("--delivery-cache", default=DELIVERY_CACHE)
    ap.add_argument("--results-cache", default=RESULTS_CACHE)
    ap.add_argument("--npz-a", default=PRED_A)
    ap.add_argument("--npz-b", default=PRED_B)
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--part", choices=("a", "b", "bstd", "c", "all"),
                    default="all")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--trials", type=int, default=EVAL_TUNE_TRIALS)
    ap.add_argument("--placebo-draws", type=int, default=len(PLACEBO_SEEDS))
    ap.add_argument("--smoke", action="store_true",
                    help="1 trial, B = 20, 1 placebo retrain; A0 not enforced")
    args = ap.parse_args()
    if args.smoke:
        args.trials, args.bootstrap, args.placebo_draws = 1, 20, 1
        if args.npz_a == PRED_A:
            args.npz_a = os.path.join(ROOT, "p6_smoke_a.npz")
        if args.npz_b == PRED_B:
            args.npz_b = os.path.join(ROOT, "p6_smoke_b.npz")
        if args.state == STATE:
            args.state = os.path.join(ROOT, "p6_smoke.json")

    started = time.time()
    inputs = {
        "prereg_sha256": sha256_file(PREREG) if os.path.exists(PREREG) else "missing",
        "panel_sha256": sha256_file(args.panel_cache),
        "baseline_sha256": arm_arrays_sha256(args.stage2b_npz, BASELINE_ARM),
    }
    inputs["panel_ok"] = inputs["panel_sha256"].lower() == PANEL_SHA256
    inputs["baseline_ok"] = inputs["baseline_sha256"].lower() == BASELINE_SHA256
    print(f"pre-registration {inputs['prereg_sha256']}", flush=True)
    if inputs["prereg_sha256"] == "missing" and not args.smoke:
        raise SystemExit("no pre-registration on disk; refusing to run on real data")
    if not (inputs["panel_ok"] and inputs["baseline_ok"]) and not args.smoke:
        raise SystemExit("a frozen input changed since the pre-registration")

    raw_panel = load_cached_panel(args.panel_cache)
    stored = _from_arm(args.stage2b_npz, BASELINE_ARM)

    state = {}
    if os.path.exists(args.state):
        with open(args.state, encoding="utf-8") as f:
            state = json.load(f)

    A = state.get("A")
    B = state.get("B")
    C = state.get("C")
    BS = state.get("B_std")
    if args.part in ("a", "all"):
        A = part_a(raw_panel, stored, args)
        state["A"] = A
        state["inputs"] = inputs
        with open(args.state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, default=float)
    if args.part in ("b", "all"):
        B = part_b(raw_panel, args)
        state["B"] = B
        state["inputs"] = inputs
        with open(args.state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, default=float)
    if args.part in ("bstd", "all"):
        BS = part_b(raw_panel, args, standardise=True)
        state["B_std"] = BS
        state["inputs"] = inputs
        with open(args.state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, default=float)
    if args.part in ("c", "all"):
        C = part_c(raw_panel, args)
        state["C"] = C
        state["inputs"] = inputs
        with open(args.state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, default=float)

    text = render(inputs, A, B, C, args, time.time() - started, B_std=BS)
    print("\n" + text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
