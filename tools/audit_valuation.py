"""
tools/audit_valuation.py - Why the valuation result was retired.

    python tools/audit_valuation.py --experiment all
    python tools/audit_valuation.py --experiment a        # min_train sweep
    python tools/audit_valuation.py --experiment b --draws 40
    python tools/audit_valuation.py --json out.json

WHAT THIS EXISTS TO REPRODUCE
-----------------------------
`pooled_xgb+val` once scored a rebalance IC of +0.0708 at t = +3.32, against a
100-draw per-ticker-constant placebo null giving p = 0.0099 and placing it
+4.35 SD above the null mean. That was the only number in this project to clear
the pre-registered bar of reb_t > 2, and it is now retired. This tool is the
evidence, kept in the repository so the retraction is reproducible rather than
a claim in a document.

THE FOUR EXPERIMENTS, AND WHY EACH WAS NEEDED
---------------------------------------------
A  min_train sweep. +3.32 was measured at min_train=380; the harness default
   everything else is reported at is 500, which scores +1.00 on identical rows.
   The grid spans -0.46 to +3.32, and 380 is a lone spike between neighbours of
   +1.30 and +1.18.

B  The placebo null AT EACH SETTING. Necessary because the null mean itself
   moves with min_train (-0.05 at 500, +0.90 at 380), so raw t-statistics are
   not comparable across settings and A alone cannot convict. Measured against
   its own null at each setting, valuation DOES clear - which is why C and D
   were needed rather than being a formality.

C  Filing-lag sweep. Withhold each figure a further 90/180/365 days on top of
   the 60-day SEBI lag. The edge decayed monotonically, which reads as a clean
   freshness effect - and is the trap.

D  The same lag sweep on a FIXED SAMPLE. C's row count fell 77,585 -> 54,304,
   because a longer lag costs the earliest rows their coverage. Holding the
   sample at the rows surviving the longest lag, the edge is absent at EVERY
   lag, lag 0 included. The decay in C was the loss of the specific rows
   carrying the result, not staleness.

E  Split by date, confirming the same thing from the other side.

THE DURABLE LESSON. Three guards were in place - a purged panel splitter, a
pre-registered t > 2, and a persistence placebo - and the result still got as
far as being called the project's one success. None of the three caught it.
What did was re-running the same measurement at a different arbitrary setting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.baselines import FACTORS, _pooled_xgb_factory      # noqa: E402
from pipeline.evaluation import (PurgedPanelWalkForward,         # noqa: E402
                                 panel_walk_forward)
from pipeline.fundamentals import (FUNDAMENTAL_COLS,             # noqa: E402
                                   attach_fundamentals,
                                   load_fundamentals)
from pipeline.panel import (SCALE_FREE, TARGET,                  # noqa: E402
                            cross_sectional_zscore, load_panel)

MIN_TRAIN_GRID = [300, 340, 380, 420, 460, 500, 540]
NULL_GRID = [340, 380, 420, 460, 500]
LAGS = [0, 90, 180, 365]
LAG_SETTINGS = [380, 460, 500]


def _score(frame, cols, min_train, n_folds=5):
    splitter = PurgedPanelWalkForward(n_folds=n_folds, horizon=30, embargo=30,
                                      min_train=min_train)
    r = panel_walk_forward(panel=frame, feature_cols=cols,
                           model_factory=_pooled_xgb_factory, splitter=splitter,
                           name="x", target=TARGET, rebalance_every=30)
    xs = r.cross_sectional
    return (xs.get("rank_ic_t", float("nan")),
            xs.get("mean_rank_ic", float("nan")),
            xs.get("n_rebalances", 0))


def _restricted(panel):
    """Rows carrying a complete set of fundamentals, z-scored within date."""
    p = panel[panel[FUNDAMENTAL_COLS].notna().all(axis=1)].reset_index(drop=True)
    return cross_sectional_zscore(p, SCALE_FREE + FUNDAMENTAL_COLS)


def _lagged(base_panel, fund, extra_days):
    f = fund.copy()
    f["effective_date"] = (
        pd.to_datetime(f["effective_date"]) + pd.Timedelta(days=extra_days)
    ).dt.date.astype(str)
    p = attach_fundamentals(base_panel, fundamentals=f)
    return p[p[FUNDAMENTAL_COLS].notna().all(axis=1)].reset_index(drop=True)


def experiment_a(panel) -> list[dict]:
    print("\nA. min_train sweep - is +3.32 robust to the training window?")
    print(f"{'min_train':>9s} {'n_reb':>6s} {'FACTORS t':>10s} {'+val t':>8s} "
          f"{'+val IC':>9s}")
    print("-" * 46)
    out = []
    for mt in MIN_TRAIN_GRID:
        t_none, _, _ = _score(panel, FACTORS, mt)
        t_val, ic_val, nreb = _score(panel, FACTORS + FUNDAMENTAL_COLS, mt)
        out.append({"min_train": mt, "n_rebalances": int(nreb),
                    "t_factors": float(t_none), "t_val": float(t_val),
                    "ic_val": float(ic_val)})
        print(f"{mt:>9d} {nreb:>6d} {t_none:>+10.2f} {t_val:>+8.2f} "
              f"{ic_val:>+9.4f}", flush=True)
    ts = [r["t_val"] for r in out]
    print(f"  spread {max(ts) - min(ts):.2f} t-units across the grid "
          f"(min {min(ts):+.2f}, max {max(ts):+.2f})")
    return out


def experiment_b(raw, draws: int) -> list[dict]:
    """
    The placebo replaces valuation with two random per-ticker CONSTANTS.

    Those carry zero information, so whatever they score is what the harness
    awards for ticker identity alone. That is the null valuation must beat,
    because the within-date rank of earnings_yield autocorrelates +0.813 at 250
    sessions - it is very nearly a static label, and a tree can read a static
    label as an identifier and recover which names paid during training.
    """
    print(f"\nB. placebo null at each setting ({draws} draws each)")
    tickers = raw["ticker"].unique()
    real = _restricted(raw)

    # Built once and reused across settings: a draw must be the SAME random
    # labelling at every min_train, or the comparison across settings mixes two
    # sources of variation.
    frames = []
    for i in range(draws):
        rng = np.random.default_rng(10_000 + i)
        p = raw[raw[FUNDAMENTAL_COLS].notna().all(axis=1)].reset_index(drop=True)
        p["pl_a"] = p["ticker"].map({t: rng.normal() for t in tickers})
        p["pl_b"] = p["ticker"].map({t: rng.normal() for t in tickers})
        frames.append(cross_sectional_zscore(p, SCALE_FREE + ["pl_a", "pl_b"]))

    print(f"{'min_train':>9s} {'real t':>8s} {'null mean':>10s} {'sd':>6s} "
          f"{'>=real':>8s} {'p':>7s}")
    print("-" * 52)
    out = []
    for mt in NULL_GRID:
        t_real, _, _ = _score(real, FACTORS + FUNDAMENTAL_COLS, mt)
        ts = np.array([_score(f, FACTORS + ["pl_a", "pl_b"], mt)[0]
                       for f in frames])
        ts = ts[~np.isnan(ts)]
        exceed = int((ts >= t_real).sum())
        p_emp = (1 + exceed) / (1 + len(ts))
        out.append({"min_train": mt, "t_real": float(t_real),
                    "null_mean": float(ts.mean()),
                    "null_sd": float(ts.std(ddof=1)),
                    "exceed": exceed, "n_draws": int(len(ts)),
                    "p_empirical": float(p_emp)})
        print(f"{mt:>9d} {t_real:>+8.2f} {ts.mean():>+10.3f} "
              f"{ts.std(ddof=1):>6.3f} {exceed:>4d}/{len(ts):<3d} {p_emp:>7.3f}",
              flush=True)
    print("  the null MEAN moves with min_train, so raw t is not comparable")
    print("  across settings - only each cell against its own null.")
    return out


def experiment_c(base_panel, fund) -> list[dict]:
    print("\nC. filing-lag sweep - NOTE the row count as it goes")
    print(f"{'extra lag':>9s} {'rows':>8s} " +
          " ".join(f"{'mt=' + str(m):>8s}" for m in LAG_SETTINGS))
    print("-" * 44)
    out = []
    for extra in LAGS:
        p = cross_sectional_zscore(_lagged(base_panel, fund, extra),
                                   SCALE_FREE + FUNDAMENTAL_COLS)
        cells, entry = [], {"extra_lag_days": extra, "rows": int(len(p)),
                            "by_min_train": {}}
        for mt in LAG_SETTINGS:
            t, _, _ = _score(p, FACTORS + FUNDAMENTAL_COLS, mt)
            entry["by_min_train"][str(mt)] = float(t)
            cells.append(f"{t:+.2f}")
        out.append(entry)
        print(f"{extra:>9d} {len(p):>8,} " + " ".join(f"{c:>8s}" for c in cells),
              flush=True)
    print("  the sample SHRANK across this sweep - see D before reading it")
    return out


def experiment_d(base_panel, fund) -> list[dict]:
    print("\nD. the same sweep on a FIXED sample")
    longest = _lagged(base_panel, fund, max(LAGS))
    keep = set(zip(longest["date"].astype(str), longest["ticker"]))
    print(f"  common sample: {len(keep):,} rows surviving a "
          f"{max(LAGS)}-day extra lag")
    print(f"{'extra lag':>9s} {'rows':>8s} " +
          " ".join(f"{'mt=' + str(m):>8s}" for m in LAG_SETTINGS))
    print("-" * 44)
    out = []
    for extra in LAGS:
        p = _lagged(base_panel, fund, extra)
        mask = [(d, t) in keep
                for d, t in zip(p["date"].astype(str), p["ticker"])]
        p = cross_sectional_zscore(p[mask].reset_index(drop=True),
                                   SCALE_FREE + FUNDAMENTAL_COLS)
        cells, entry = [], {"extra_lag_days": extra, "rows": int(len(p)),
                            "by_min_train": {}}
        for mt in LAG_SETTINGS:
            t, _, _ = _score(p, FACTORS + FUNDAMENTAL_COLS, mt)
            entry["by_min_train"][str(mt)] = float(t)
            cells.append(f"{t:+.2f}")
        out.append(entry)
        print(f"{extra:>9d} {len(p):>8,} " + " ".join(f"{c:>8s}" for c in cells),
              flush=True)
    print("  sample constant, so any remaining decay would be the SIGNAL")
    return out


def experiment_e(panel_raw) -> list[dict]:
    print("\nE. split by date")
    p = panel_raw[panel_raw[FUNDAMENTAL_COLS].notna().all(axis=1)]
    p = p.reset_index(drop=True)
    dates = np.array(sorted(p["date"].astype(str).unique()))
    cut = dates[len(dates) // 2]
    print(f"  split at {cut}")
    print(f"{'half':<8s} {'rows':>8s} {'n_reb':>6s} {'FACTORS t':>10s} "
          f"{'+val t':>8s}")
    print("-" * 44)
    out = []
    for label, sub in [("EARLY", p[p["date"].astype(str) <= cut]),
                       ("LATE", p[p["date"].astype(str) > cut])]:
        z = cross_sectional_zscore(sub.reset_index(drop=True),
                                   SCALE_FREE + FUNDAMENTAL_COLS)
        t_none, _, _ = _score(z, FACTORS, 200, n_folds=3)
        t_val, ic_val, nreb = _score(z, FACTORS + FUNDAMENTAL_COLS, 200,
                                     n_folds=3)
        out.append({"half": label, "cut": str(cut), "rows": int(len(z)),
                    "n_rebalances": int(nreb), "t_factors": float(t_none),
                    "t_val": float(t_val), "ic_val": float(ic_val)})
        print(f"{label:<8s} {len(z):>8,} {nreb:>6d} {t_none:>+10.2f} "
              f"{t_val:>+8.2f}", flush=True)
    print("  few rebalances per half - read the SIGN and the contrast, not the t")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="all",
                    choices=["a", "b", "c", "d", "e", "all"])
    ap.add_argument("--draws", type=int, default=40,
                    help="placebo draws per setting in experiment B")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    logging.disable(logging.INFO)

    base_panel = load_panel()
    fund = load_fundamentals()
    raw = attach_fundamentals(base_panel)
    restricted = _restricted(raw)

    if restricted.empty:
        print("\n  REFUSED: no row carries a complete set of fundamentals. "
              "Run tools/sync_fundamentals.py first.", file=sys.stderr)
        return 1

    print(f"panel {len(restricted):,} rows | "
          f"{restricted['ticker'].nunique()} tickers | "
          f"{restricted['date'].nunique():,} dates | "
          f"{restricted['date'].min()} -> {restricted['date'].max()}")

    want = args.experiment
    results = {}
    if want in ("a", "all"):
        results["a_min_train_sweep"] = experiment_a(restricted)
    if want in ("b", "all"):
        results["b_placebo_null"] = experiment_b(raw, args.draws)
    if want in ("c", "all"):
        results["c_lag_sweep"] = experiment_c(base_panel, fund)
    if want in ("d", "all"):
        results["d_lag_fixed_sample"] = experiment_d(base_panel, fund)
    if want in ("e", "all"):
        results["e_period_split"] = experiment_e(raw)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
