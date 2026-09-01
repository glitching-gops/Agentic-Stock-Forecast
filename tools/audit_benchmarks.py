"""
tools/audit_benchmarks.py — is each sector benchmarked against the right index?

WHY THIS EXISTS. The model's target is `target_excess_return`: the stock's
forward 30-session log return MINUS its benchmark's. The benchmark is not a
display choice, it is half the label. Point a sector at the wrong index and you
do not merely mislabel a chart — you subtract another sector's rotation from
every row and call the residual alpha. The mapping in data/tickers.py carried a
comment claiming every entry was "verified to return >1,200 daily rows": a
liveness check, not a fitness check. Nothing had ever tested whether an index
described the stocks pointed at it.

HOW IT DECIDES. Arguing from remembered index constituents is how the mapping
got where it is. This measures instead: the share of a stock's daily log-return
variance explained by a candidate index (squared Pearson correlation, which for
a single regressor IS the OLS R²). A sector's score is the MEDIAN across its
members, so one dominant constituent cannot carry the sector alone.

WHY A BOOTSTRAP, AND WHY DAILY RETURNS. A first pass ranked on overlapping
30-session returns, on the reasoning that the target is a 30-session quantity.
That was wrong in a way worth recording: five years of history holds only ~40
INDEPENDENT 30-day windows, and computing a correlation across ~1,170
overlapping ones overstates the precision by roughly the overlap factor. The
resulting ranking was noise-dominated and produced two conclusions that no
amount of theory supports — Capital Goods best explained by NIFTY Realty, Oil &
Gas by NIFTY Infrastructure. Daily returns give ~1,200 near-independent
observations instead, and a moving-block bootstrap over the daily series puts
an interval on every comparison. A mapping changes only when the interval for
(candidate - incumbent) excludes zero. Sector benchmarks should be stable
across years; a difference this procedure cannot separate from noise is not a
reason to redefine a label.

WHY STYLE INDICES ARE MEASURED BUT NEVER CHOSEN. ^CNX100, ^CNXPSE, ^CNXMNC and
the bank sub-indices classify by size, state ownership or domicile — not by
industry. Three reasons they stay ineligible even when they score highest:

  1. Self-benchmarking. ^CNX100 contains every member of this universe by
     construction, so part of its R² against any stock is that stock's own
     weight in it. The excess return would subtract the stock from itself.
  2. Interpretability. The product's claim is "excess return versus sector". A
     figure computed against an ownership index does not support that sentence,
     and every published forecast would be quietly asserting something false.
  3. Instability. Style membership is reconstituted on rules unrelated to what
     a company does, so the label's meaning drifts invisibly to the model.

They are still scored and printed: a sector whose best explanation is a style
factor is a real finding about that sector, worth knowing before Phase 2 builds
a cross-sectional model on these labels.

    python tools/audit_benchmarks.py
    python tools/audit_benchmarks.py --json out.json --apply-check
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

# Industry indices — the only ones eligible to become a sector benchmark.
SECTOR_CANDIDATES = [
    "^NSEBANK",               # NIFTY Bank
    "NIFTY_FIN_SERVICE.NS",   # NIFTY Financial Services
    "^CNXIT",                 # NIFTY IT
    "^CNXAUTO",               # NIFTY Auto
    "^CNXPHARMA",             # NIFTY Pharma
    "^CNXFMCG",               # NIFTY FMCG
    "^CNXMETAL",              # NIFTY Metal
    "^CNXENERGY",             # NIFTY Energy — petroleum, gas AND power
    "^CNXINFRA",              # NIFTY Infrastructure
    "^CNXREALTY",             # NIFTY Realty
    "^CNXCONSUM",             # NIFTY India Consumption
    "^CNXSERVICE",            # NIFTY Services Sector
    "^CNXMEDIA",              # NIFTY Media
]

# Measured and reported, never selected. See the module docstring.
STYLE_INDICES = ["^CNX100", "^CNXPSE", "^CNXMNC", "^CNXPSUBANK",
                 "NIFTY_PVT_BANK.NS"]

# Probed and absent from this feed under every symbol tried, recorded so the
# gap is not rediscovered: no NIFTY Healthcare, Power, Oil & Gas, Chemicals,
# Capital Goods or Defence index resolves on yfinance.
BROAD = "^NSEI"

BLOCK_DAYS = 21          # ~one trading month; preserves short-run dependence
BOOTSTRAP_DRAWS = 400
CI = 90                  # two-sided interval width, percent

# A candidate must beat the incumbent by this much in the point estimate AND
# have a bootstrap interval excluding zero. The floor exists because an index
# that merely ties adds concentration risk — fewer constituents, more
# single-name influence — for no explanatory gain.
MIN_EDGE = 0.03

# Sectors this small cannot support a median: the "sector" is one or two names
# and any index containing them fits for mechanical reasons.
MIN_MEMBERS_FOR_CONFIDENCE = 3
MIN_EDGE_THIN = 0.12


def _closes(symbols: list[str], period: str) -> pd.DataFrame:
    """Daily adjusted closes for many symbols, one column each."""
    import yfinance as yf

    out: dict[str, pd.Series] = {}
    for i in range(0, len(symbols), 25):
        chunk = symbols[i:i + 25]
        data = yf.download(chunk, period=period, interval="1d",
                           auto_adjust=True, progress=False, group_by="ticker")
        for symbol in chunk:
            try:
                frame = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
                series = frame["Close"].dropna()
            except (KeyError, TypeError):
                continue
            if len(series) > 250:
                out[symbol] = series
    return pd.DataFrame(out)


def _median_r2(matrix: np.ndarray, stock_cols: list[int],
               index_cols: dict[str, int]) -> dict[str, float]:
    """
    Median R² of each index against the sector's stocks, from one return matrix.

    One np.corrcoef over the whole matrix serves every pair, which is what makes
    a 400-draw bootstrap affordable.
    """
    corr = np.corrcoef(matrix, rowvar=False)
    out = {}
    for name, j in index_cols.items():
        r2 = corr[stock_cols, j] ** 2
        r2 = r2[np.isfinite(r2)]
        if r2.size:
            out[name] = float(np.median(r2))
    return out


def _bootstrap(matrix: np.ndarray, stock_cols: list[int],
               index_cols: dict[str, int], rng: np.random.Generator
               ) -> dict[str, np.ndarray]:
    """
    Moving-block bootstrap of the median-R² statistic.

    Resamples contiguous BLOCK_DAYS-long blocks of the daily return series with
    replacement. Blocks rather than individual days because daily returns carry
    short-run dependence (volatility clustering above all) that an i.i.d.
    resample would destroy, producing intervals far too narrow.
    """
    n_rows = matrix.shape[0]
    n_blocks = max(1, n_rows // BLOCK_DAYS)
    starts_max = n_rows - BLOCK_DAYS
    if starts_max <= 0:
        return {}

    draws: dict[str, list[float]] = {name: [] for name in index_cols}
    for _ in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, starts_max, size=n_blocks)
        rows = np.concatenate([np.arange(s, s + BLOCK_DAYS) for s in starts])
        sample = _median_r2(matrix[rows], stock_cols, index_cols)
        for name, value in sample.items():
            draws[name].append(value)
    return {name: np.array(v) for name, v in draws.items() if v}


def audit(period: str = "5y", seed: int = 7) -> dict:
    from data.db import get_engine

    members = pd.read_sql(
        "SELECT ticker, industry FROM index_membership "
        "WHERE effective_to = '9999-12-31'",
        get_engine(),
    )
    members = members[members["industry"].notna() & (members["industry"] != "")]
    tickers = sorted(members["ticker"].unique())

    all_indices = [BROAD] + SECTOR_CANDIDATES + STYLE_INDICES
    print(f"[Audit] {len(tickers)} index members, "
          f"{members['industry'].nunique()} sectors, "
          f"{len(all_indices)} candidate indices ({period})")

    prices = _closes(tickers + all_indices, period)
    daily = np.log(prices).diff()

    available = [c for c in all_indices if c in prices.columns]
    missing = sorted(set(all_indices) - set(available))
    if missing:
        print(f"[Audit] unavailable, excluded: {missing}")
    print(f"[Audit] {BOOTSTRAP_DRAWS} bootstrap draws, "
          f"{BLOCK_DAYS}-day blocks, {CI}% interval")

    from data.tickers import BROAD_MARKET_INDEX, SECTOR_INDICES

    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100 - CI) / 2, 100 - (100 - CI) / 2
    results = {}

    for sector, group in members.groupby("industry"):
        sector_tickers = [t for t in group["ticker"] if t in daily.columns]
        if not sector_tickers:
            continue

        block = daily[sector_tickers + available].dropna()
        if len(block) < 250:
            continue
        matrix = block.to_numpy()
        stock_cols = list(range(len(sector_tickers)))
        index_cols = {name: len(sector_tickers) + k
                      for k, name in enumerate(available)}

        point = _median_r2(matrix, stock_cols, index_cols)
        draws = _bootstrap(matrix, stock_cols, index_cols, rng)

        incumbent = SECTOR_INDICES.get(sector, BROAD_MARKET_INDEX)
        eligible = {k: v for k, v in point.items() if k in SECTOR_CANDIDATES}
        thin = len(sector_tickers) < MIN_MEMBERS_FOR_CONFIDENCE
        required = MIN_EDGE_THIN if thin else MIN_EDGE

        best = max(eligible, key=eligible.get) if eligible else None
        broad_r2 = point.get(BROAD, 0.0)

        def interval(a: str, b: str) -> tuple[float, float] | None:
            if a not in draws or b not in draws or a == b:
                return None
            diff = draws[a] - draws[b]
            return float(np.percentile(diff, lo_q)), float(np.percentile(diff, hi_q))

        def justified(index: str | None) -> bool:
            """
            Does this index earn its place over the broad market?

            An industry index has to beat ^NSEI on the point estimate by
            MIN_EDGE *and* on the bootstrap interval. Anything that cannot is
            not a benchmark, it is a noisier NIFTY 50: NIFTY 50 is ~35%
            financials by weight, so for several sectors it is already the best
            available description, with fifty constituents diluting any one
            stock's influence on its own benchmark rather than twelve.
            """
            if not index or index == BROAD or index not in eligible:
                return False
            ci = interval(index, BROAD)
            return eligible[index] - broad_r2 >= MIN_EDGE and bool(ci and ci[0] > 0)

        vs_broad = interval(best, BROAD) if best else None
        vs_incumbent = interval(best, incumbent) if best else None

        # SWITCHING between two sector indices is held to a higher bar than
        # justifying one, and higher still on a thin sector. The incumbent has
        # a year of labels behind it; changing it rewrites the meaning of every
        # historical target, so a candidate has to be clearly better, not
        # merely ahead. On a one- or two-stock sector, "clearly" means a lot:
        # any index containing those names fits them for mechanical reasons.
        switch_bar = MIN_EDGE_THIN if thin else MIN_EDGE

        if justified(best) and best == incumbent:
            chosen = incumbent                       # already right
        elif (justified(best)
              and eligible[best] - point.get(incumbent, broad_r2) >= switch_bar
              and vs_incumbent and vs_incumbent[0] > 0):
            chosen = best                            # clearly better index
        elif justified(incumbent):
            chosen = incumbent                       # not beaten; leave it
        else:
            chosen = BROAD                           # incumbent does not earn its place

        changed = chosen != incumbent

        style = {k: v for k, v in point.items() if k in STYLE_INDICES}
        best_style = max(style, key=style.get) if style else None

        results[sector] = {
            "members": len(sector_tickers),
            "tickers": sector_tickers,
            "thin": thin,
            "required_edge": required,
            "incumbent": incumbent,
            "incumbent_r2": round(point.get(incumbent, float("nan")), 4),
            "broad_r2": round(broad_r2, 4),
            "best_sector_index": best,
            "best_sector_r2": round(eligible[best], 4) if best else None,
            "edge_over_broad": round(eligible[best] - broad_r2, 4) if best else None,
            "ci_vs_broad": [round(v, 4) for v in vs_broad] if vs_broad else None,
            "ci_vs_incumbent": [round(v, 4) for v in vs_incumbent] if vs_incumbent else None,
            "chosen": chosen,
            "changed": changed,
            "sector_specific": chosen != BROAD,
            "best_style_index": best_style,
            "best_style_r2": round(style[best_style], 4) if best_style else None,
            "r2": {k: round(v, 4) for k, v in
                   sorted(point.items(), key=lambda kv: -kv[1])},
        }

    return results


def report(results: dict) -> list[tuple]:
    print()
    print(f"  {'sector':32s} {'n':>3s} {'incumbent':22s} {'r2':>6s} "
          f"{'best sector idx':22s} {'r2':>6s} "
          f"{'CI vs incumbent':>17s}  {'chosen':22s}")
    print("-" * 136)

    changes = []
    for sector, r in sorted(results.items(), key=lambda kv: -kv[1]["members"]):
        ci = r["ci_vs_incumbent"]
        ci_text = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "-"
        mark = "* " if r["changed"] else "  "
        thin = "~" if r["thin"] else " "
        print(f"{mark}{sector:32s}{thin}{r['members']:3d} "
              f"{r['incumbent']:22s} {r['incumbent_r2']:6.3f} "
              f"{str(r['best_sector_index'] or '-'):22s} "
              f"{r['best_sector_r2'] if r['best_sector_r2'] is not None else float('nan'):6.3f} "
              f"{ci_text:>17s}  {r['chosen']:22s}")
        if r["changed"]:
            changes.append((sector, r["incumbent"], r["chosen"], r["members"]))

    print("\n  * mapping changes    ~ thin sector (below the confidence floor)")
    print("  a change needs the point edge AND a bootstrap interval clear of zero")
    print()
    if changes:
        print(f"{len(changes)} sector(s) change, "
              f"{sum(c[3] for c in changes)} tickers affected:")
        for sector, was, now, n in changes:
            print(f"    {sector:32s} {was:22s} -> {now:22s} ({n:2d} tickers)")
    else:
        print("mapping agrees with the measurement on every sector")

    styled = [(s, r) for s, r in results.items()
              if r["best_style_r2"] and r["best_style_r2"] > (r["best_sector_r2"] or 0)]
    if styled:
        print(f"\nsectors better explained by a style factor than by any "
              f"industry index ({len(styled)}) — reported, never selected:")
        for sector, r in sorted(styled, key=lambda kv: -kv[1]["members"]):
            print(f"    {sector:32s} {r['best_style_index']:12s} "
                  f"{r['best_style_r2']:.3f}  vs  "
                  f"{str(r['best_sector_index'] or '-'):22s} "
                  f"{r['best_sector_r2'] if r['best_sector_r2'] is not None else float('nan'):.3f}")
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, help="write full results here")
    parser.add_argument("--apply-check", action="store_true",
                        help="exit 1 if the live mapping disagrees")
    args = parser.parse_args()

    results = audit(args.period, args.seed)
    changes = report(results)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 1 if (changes and args.apply_check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
