"""
tools/p6_scale_followup.py — the skeptic pass on P6's Part C.

Part C refits each horizon on a WITHIN-DATE STANDARDISED label, so `gamma`
means the same thing at every horizon, and scores against the REAL label. It
was added post hoc, after Part A's two shortest horizons came back with a model
that never split, and it decides nothing on its own.

If it produces a number that looks like signal, THAT NUMBER IS NOT A RESULT
until it has survived this file. The project has been fooled three times by a
single cell — valuation at t +3.32 whose neighbours read +1.30 and +1.18, LoRA
at +2.37 carried entirely by the earliest fold, pooled_xgb at +2.42 that failed
at all six min_train settings. Every one of them was caught by re-running the
same measurement somewhere else, and none by reasoning about it.

Four attacks, all of them standing policy in CLAUDE.md:

  1. the min_train sweep      — a result at one setting is not a result
  2. the per-fold profile     — an effect that lives in fold 0 is the panel
  3. a within-date target permutation, retrained — the corrected placebo,
                                 applied to the target because what is under
                                 test is the model's ranking, not an added
                                 column
  4. net of the 0.2225% round trip, against THIS horizon's own break-even

SANDBOXED. Reads panel_cache.parquet, writes a markdown report and a JSON.
Nothing the API, the web app or either scheduled job reads.

    python tools/p6_scale_followup.py --horizons 5 --markdown p6_followup.md
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

from pipeline.evaluation import horizon_purge_embargo  # noqa: E402
from pipeline.evidence_panel import driscoll_kraay_se  # noqa: E402
from pipeline.model import EVAL_TUNE_TRIALS  # noqa: E402
from pipeline.panel import TARGET  # noqa: E402
from pipeline.portfolio import CostModel  # noqa: E402
from tools.stage1_reversal import DK_LAGS, SIGNAL_T, _f, as_data, per_date_ic  # noqa: E402
from tools.stage2b_pooled import cell_metrics, load_cached_panel, run_arm  # noqa: E402
from tools.p6_horizon import (  # noqa: E402
    ARM_FEATURES,
    PLACEBO_SEEDS,
    SWEEP_MIN_TRAIN,
    _sharpe,
    _t,
    book,
    panel_at,
    spread_per_ic,
    standardise_target,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATE = os.path.join(ROOT, "p6_followup.json")


def permute_target_within_date(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    The corrected placebo, pointed at the TARGET.

    Pilot 1 established that shuffling an arm's PREDICTIONS is the wrong null,
    because it destroys the baseline's ordering too. Pilots 2 and 3 shuffled
    the added COLUMNS, because what was under test was an added column. Here
    what is under test is whether the model can rank names at all, so the thing
    to destroy is the link between a name and its own outcome — and nothing
    else. Permuting the target WITHIN each date keeps the cross-sectional
    distribution, the date effects, every feature and the fold geometry exactly
    as they were.
    """
    rng = np.random.default_rng(seed)
    out = panel.copy()
    y = out[TARGET].to_numpy(dtype=float).copy()
    dates = out["date"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        y[lo:hi] = y[lo:hi][rng.permutation(hi - lo)]
    out[TARGET] = y
    return out


def _ic(preds: pd.DataFrame) -> tuple[float, float, float, int]:
    ics = per_date_ic(as_data(preds)).dropna()
    if len(ics) < 3:
        return float("nan"), float("nan"), float("nan"), len(ics)
    mean, se, _ = driscoll_kraay_se(ics.index.to_numpy(), ics.to_numpy(),
                                    max_lag=DK_LAGS)
    t = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
    return mean, se, t, len(ics)


def fit(panel_std: pd.DataFrame, truth: pd.DataFrame, horizon: int, purge: int,
        trials: int, min_train: int = 500, verbose: bool = False):
    """One fit on the standardised label, scored against the real one."""
    preds, folds = run_arm(panel_std, "mae", "none", n_trials=trials,
                           min_train=min_train, features=ARM_FEATURES["baseline"],
                           horizon=horizon, purge=purge, verbose=verbose)
    m = preds.astype({"date": str}).merge(truth, on=["date", "ticker"], how="inner")
    m = m[np.isfinite(m["y_real"])]
    real = m.drop(columns=["y_true"]).rename(columns={"y_real": "y_true"})
    return real, folds


def run_horizon(raw_panel: pd.DataFrame, horizon: int, args) -> dict:
    purge = horizon_purge_embargo(horizon)
    panel_h = panel_at(raw_panel, horizon)
    truth = panel_h[["date", "ticker", TARGET]].astype({"date": str}) \
        .rename(columns={TARGET: "y_real"})
    panel_std = standardise_target(panel_h)

    out: dict = {"horizon": horizon, "purge": purge}

    # ── the headline, re-measured here so the report is self-contained ────────
    print(f"\nh={horizon}: the cell under attack", flush=True)
    t0 = time.time()
    preds, folds = fit(panel_std, truth, horizon, purge, args.trials, verbose=True)
    mean, se, t, n = _ic(preds)
    m = cell_metrics(preds, folds, rebalance_every=horizon)
    out["headline"] = {"cs_ic": mean, "se": se, "t": t, "n_dates": n,
                       "reb_ic": m["reb_ic"], "reb_t": m["reb_t"],
                       "n_rebalances": m["n_rebalances"],
                       "constant_cells": m["constant_cells"],
                       "cs_ic_by_fold": m["cs_rank_ic_by_fold"],
                       "secs": round(time.time() - t0, 1)}
    print(f"  cs IC {mean:+.5f} t {t:+.2f} over {n} dates; by fold "
          f"{m['cs_rank_ic_by_fold']}", flush=True)

    # ── 1. the min_train sweep ────────────────────────────────────────────────
    print(f"\nh={horizon}: min_train sweep", flush=True)
    rows = []
    for mt in SWEEP_MIN_TRAIN:
        p, f = fit(panel_std, truth, horizon, purge, args.trials, min_train=mt)
        mu, s, tt, nn = _ic(p)
        mm = cell_metrics(p, f, rebalance_every=horizon)
        rows.append({"min_train": mt, "cs_ic": mu, "t": tt, "n_dates": nn,
                     "reb_ic": mm["reb_ic"], "reb_t": mm["reb_t"],
                     "constant_cells": mm["constant_cells"]})
        print(f"    min_train {mt}: cs IC {mu:+.5f}, t {tt:+.2f}, "
              f"constant {mm['constant_cells']}/420", flush=True)
    out["min_train_sweep"] = rows

    # ── 3. the within-date target permutation ─────────────────────────────────
    print(f"\nh={horizon}: {args.placebo_draws} target-permuted retrains", flush=True)
    pl = []
    for seed in PLACEBO_SEEDS[:args.placebo_draws]:
        shuffled = permute_target_within_date(panel_std, seed)
        p, _ = fit(shuffled, truth, horizon, purge, args.trials)
        mu, s, tt, nn = _ic(p)
        pl.append({"seed": seed, "cs_ic": mu, "t": tt, "n_dates": nn})
        print(f"    seed {seed}: cs IC {mu:+.5f}, t {tt:+.2f}", flush=True)
    out["placebo"] = pl

    # ── 4. money ──────────────────────────────────────────────────────────────
    b = book(preds, horizon, CostModel())
    spi = spread_per_ic(preds, horizon)
    be = (CostModel().round_trip * float(b["turnover"].mean()) / spi
          if spi and np.isfinite(spi) and spi > 0 else float("nan"))
    out["money"] = {
        "n_rebalances": len(b), "gross": float(b["gross"].mean()),
        "net": float(b["net"].mean()), "t_gross": _t(b["gross"]),
        "t_net": _t(b["net"]), "turnover": float(b["turnover"].mean()),
        "sharpe_net_annual": _sharpe(b["net"], horizon),
        "spread_per_ic": spi, "break_even_ic": be,
        "clears_break_even": bool(np.isfinite(be) and mean > be),
    }
    print(f"  book: net {out['money']['net']:+.5f}/rebalance (t "
          f"{out['money']['t_net']:+.2f}), turnover {out['money']['turnover']:.2f}, "
          f"break-even IC {be:.4f} vs measured {mean:+.5f}", flush=True)

    # ── the verdict ───────────────────────────────────────────────────────────
    p_ic = [r["cs_ic"] for r in pl if np.isfinite(r["cs_ic"])]
    sweep_t = [r["t"] for r in rows if np.isfinite(r["t"])]
    folds_pos = [v for v in m["cs_rank_ic_by_fold"] if np.isfinite(v)]
    out["verdict"] = {
        "headline_clears": bool(np.isfinite(t) and t >= SIGNAL_T),
        "sweep_majority_clears": bool(sweep_t and
                                      sum(x >= SIGNAL_T for x in sweep_t) > len(sweep_t) / 2),
        "beats_every_placebo": bool(p_ic and np.isfinite(mean) and mean > max(p_ic)),
        "not_only_fold_0": bool(len(folds_pos) > 1 and
                                sum(v > 0 for v in folds_pos[1:]) >= len(folds_pos[1:]) / 2),
        "clears_break_even": out["money"]["clears_break_even"],
        "placebo_max": max(p_ic) if p_ic else float("nan"),
        "sweep_t": sweep_t,
    }
    out["verdict"]["survives"] = all(
        out["verdict"][k] for k in ("headline_clears", "sweep_majority_clears",
                                    "beats_every_placebo", "not_only_fold_0",
                                    "clears_break_even"))
    return out


def render(res: list[dict], args, runtime: float) -> str:
    o = ["# P6 — the skeptic pass on Part C\n",
         "Part C is POST HOC and pre-registered nothing. What follows is the "
         "standing attack list applied to it, so a number that looks like "
         "signal is either a hypothesis worth pre-registering or nothing.\n",
         f"trials = {args.trials}, placebo retrains = {args.placebo_draws}, "
         f"DK lags = {DK_LAGS}.\n"]
    for r in res:
        h = r["horizon"]
        hd, v, mo = r["headline"], r["verdict"], r["money"]
        o += [f"## h = {h} (purge/embargo {r['purge']})\n",
              f"**cs IC {_f(hd['cs_ic'])} (DK SE {_f(hd['se'], '.5f')}), "
              f"t {_f(hd['t'], '+.2f')}** over {hd['n_dates']:,} dates; "
              f"reb IC {_f(hd['reb_ic'], '+.4f')} (t {_f(hd['reb_t'], '+.2f')}) "
              f"over {hd['n_rebalances']} rebalances; "
              f"{hd['constant_cells']}/420 constant cells.\n",
              f"By fold: {hd['cs_ic_by_fold']}\n",
              "### 1. The min_train sweep\n",
              "| min_train | cs IC | t | reb IC | reb t | constant cells |",
              "|---|---|---|---|---|---|"]
        for s in r["min_train_sweep"]:
            o.append(f"| {s['min_train']} | {_f(s['cs_ic'])} | "
                     f"**{_f(s['t'], '+.2f')}** | {_f(s['reb_ic'], '+.4f')} | "
                     f"{_f(s['reb_t'], '+.2f')} | {s['constant_cells']}/420 |")
        o += ["", "### 2. The within-date target permutation, retrained\n",
              "| seed | cs IC | t |", "|---|---|---|"]
        for s in r["placebo"]:
            o.append(f"| {s['seed']} | {_f(s['cs_ic'])} | {_f(s['t'], '+.2f')} |")
        o.append(f"| **the real fit** | **{_f(hd['cs_ic'])}** | "
                 f"**{_f(hd['t'], '+.2f')}** |")
        o += ["", "### 3. Money\n",
              f"- net {_f(mo['net'], '+.5f')} per rebalance (t "
              f"{_f(mo['t_net'], '+.2f')}), gross {_f(mo['gross'], '+.5f')} "
              f"(t {_f(mo['t_gross'], '+.2f')}), turnover {mo['turnover']:.2f}",
              f"- annualised net Sharpe {_f(mo['sharpe_net_annual'], '+.2f')} "
              f"(at 252/{h} rebalances a year, not the module constant)",
              f"- break-even rank IC at this horizon and turnover: "
              f"**{_f(mo['break_even_ic'], '.4f')}** against a measured "
              f"{_f(hd['cs_ic'])}\n",
              "### Verdict\n",
              "| attack | result |", "|---|---|",
              f"| headline t >= {SIGNAL_T:.1f} | {'PASS' if v['headline_clears'] else 'FAIL'} |",
              f"| holds at a majority of min_train settings | {'PASS' if v['sweep_majority_clears'] else 'FAIL'} |",
              f"| beats every target-permuted retrain | {'PASS' if v['beats_every_placebo'] else 'FAIL'} (max {_f(v['placebo_max'])}) |",
              f"| not carried by fold 0 alone | {'PASS' if v['not_only_fold_0'] else 'FAIL'} |",
              f"| clears its own break-even IC | {'PASS' if v['clears_break_even'] else 'FAIL'} |",
              f"| **SURVIVES** | **{'YES' if v['survives'] else 'NO'}** |", ""]
    o.append(f"Even a surviving cell is a HYPOTHESIS, not a result: this whole "
             f"pass is post hoc on a panel carrying ~141 prior trials. It earns "
             f"a pre-registration, not a headline. Runtime {runtime / 60:.1f} min.")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=[5])
    ap.add_argument("--trials", type=int, default=EVAL_TUNE_TRIALS)
    ap.add_argument("--placebo-draws", type=int, default=9)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--state", default=STATE)
    args = ap.parse_args()

    started = time.time()
    raw_panel = load_cached_panel()
    res = [run_horizon(raw_panel, h, args) for h in args.horizons]
    with open(args.state, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    text = render(res, args, time.time() - started)
    print("\n" + text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
