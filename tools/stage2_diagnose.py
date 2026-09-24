"""
tools/stage2_diagnose.py — POST HOC diagnostics for Part 0's failed prediction.

NOT PRE-REGISTERED. docs/dashboard-switch-preregistration.md §4 predicted the
panel-internal benchmark would move the pooled baseline's cs IC by |Δ| <= 0.005
at |t| < 2.0; it moved +0.0126 at t +2.31, and the rule there is to investigate
that as a defect first. Restricted to the 73 names that HAVE a sector
benchmark the delta is +0.0022 (t +0.72), so the excess lives in the 11
thin-sector names — whose three sector_rel_* features are NULL on every date:
a persistent group marker, the "missingness flag is a company fingerprint"
landmine. These panels separate that mechanism from information:

    fallback      the 11 thin names get market-relative momentum instead of
                  NULL (same formula, market LOO benchmark) — no fingerprint
    placebo_<k>   fallback, then the NULLs moved onto 11 RANDOM names of the
                  73 — the fingerprint without the names that carry it

    python tools/stage2_diagnose.py --panel stage2/panel_new.parquet --out-dir stage2/diag
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

from pipeline.signals import SECTOR_REL_COLS  # noqa: E402


def market_relative(panel: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """sector_rel_* recomputed against the row's own benchmark_close (the
    market fallback level for a thin name), per ticker over its own rows."""
    out = panel.copy()
    for t in tickers:
        m = out["ticker"] == t
        g = out.loc[m].sort_values("date")
        for w in (5, 10, 20):
            rel = (g["close"].pct_change(w) - g["benchmark_close"].pct_change(w))
            out.loc[g.index, f"sector_rel_{w}d"] = rel.to_numpy()
        out.loc[g.index, "sector_rel_missing"] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    thin = sorted(panel.loc[panel["sector_rel_missing"] == 1, "ticker"].groupby(
        panel["ticker"]).size().loc[lambda s: s > 0.5 * panel.groupby("ticker").size()
                                    .reindex(s.index)].index)
    others = sorted(set(panel["ticker"]) - set(thin))
    fb = market_relative(panel, thin)
    fb.to_parquet(args.out_dir / "panel_fallback.parquet", index=False)
    meta = {"thin": thin, "placebos": {}}
    rng = np.random.default_rng(20260924)
    for k in range(args.draws):
        fake = sorted(rng.choice(others, size=len(thin), replace=False))
        p = fb.copy()
        m = p["ticker"].isin(fake)
        p.loc[m, list(SECTOR_REL_COLS)] = np.nan
        p.loc[m, "sector_rel_missing"] = 1
        p.to_parquet(args.out_dir / f"panel_placebo_{k}.parquet", index=False)
        meta["placebos"][k] = fake
    (args.out_dir / "diagnose.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
