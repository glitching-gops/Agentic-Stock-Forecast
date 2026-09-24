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

FROM v5 ON (2026-09-24, docs/stage2-fallback-conformal-preregistration.md §2)
the production panel IS the fallback arm, so the comparison is re-run from it
the other way round (`--from-v5`):

    real11        the NULLs put back on the REAL 11 thin names — v4 rebuilt
    placebo_<k>   the NULLs on 11 RANDOM names of the 73, as before

    python tools/stage2_diagnose.py --from-v5 --panel stage2/v5/panel_v5.parquet --out-dir stage2/v5
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


def thin_names(panel: pd.DataFrame) -> list[str]:
    """The names benchmarked against the market (v5: flagged by
    benchmark_sector_specific = 0; there is no NULL pattern to find them by)."""
    spec = panel.groupby("ticker")["benchmark_sector_specific"].max()
    return sorted(spec[spec == 0].index)


def with_nulls(panel: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """`panel` with sector_rel_* NULL on every row of `tickers` — the v4
    representation, placed on whichever names are given."""
    p = panel.copy()
    p.loc[p["ticker"].isin(tickers), list(SECTOR_REL_COLS)] = np.nan
    return p


def placebo_draws(thin: list[str], all_tickers: list[str], draws: int) -> list[list[str]]:
    """The same seeded draws of 11 random non-thin names the last session used."""
    others = sorted(set(all_tickers) - set(thin))
    rng = np.random.default_rng(20260924)
    return [sorted(rng.choice(others, size=len(thin), replace=False)) for _ in range(draws)]


def main_from_v5(args) -> None:
    panel = pd.read_parquet(args.panel)
    thin = thin_names(panel)
    with_nulls(panel, thin).to_parquet(args.out_dir / "panel_real11.parquet", index=False)
    meta = {"thin": thin, "placebos": {}}
    for k, fake in enumerate(placebo_draws(thin, list(panel["ticker"].unique()), args.draws)):
        with_nulls(panel, fake).to_parquet(args.out_dir / f"panel_placebo_{k}.parquet",
                                           index=False)
        meta["placebos"][k] = fake
    (args.out_dir / "diagnose.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--from-v5", action="store_true",
                    help="build the real-11 and random-11 NULL arms from a v5 panel")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_v5:
        main_from_v5(args)
        return
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
