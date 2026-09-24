"""
pipeline/sector_benchmark.py — the sector benchmark, built from the panel itself.

WHY THIS REPLACES YAHOO'S SECTOR INDICES (2026-09-24, MODEL_VERSION v4)
-----------------------------------------------------------------------
Yahoo stopped publishing eight of the ten NSE sector indices around
2026-07-20. The benchmark is half the excess label, so from 2026-09-15 the
write-boundary guard refused every recompute for the 49 tickers mapped to a
dead index — it would have erased their historical excess labels — and their
signals went stale. Before that, `get_benchmark_series` FORWARD-FILLED the
dead index, so for two months `sector_rel_*` for those names quietly became the
stock's own raw momentum: a stale benchmark stored as a valid one.

The replacement asks no vendor anything. A stock's benchmark is the
EQUAL-WEIGHTED MEAN DAILY LOG RETURN OF THE OTHER NAMES IN ITS SECTOR, within
this universe, cumulated into a level:

    r_B(i, t) = mean over p in peers(i), p != i, of  log(P_p,t / P_p,t-1)
    B(i, t)   = exp( sum_{s <= t} r_B(i, s) )

`regime.market_log_returns` is the one definition of the market in this
project; this module reads it and restricts it to a sector, rather than
computing daily returns a second way.

LEAVE-ONE-OUT, ALWAYS. A stock inside its own benchmark is partly subtracted
from itself — the `^CNX100` landmine, arrived at from the other side — and in a
five-name sector that is a fifth of the benchmark.

THIN SECTORS ARE MISSING, NOT FILLED. A leave-one-out mean over ONE peer is a
pairwise spread, not a benchmark. Measured 2026-09-24 on the frozen 84 (20-
session log returns, non-overlapping, nine sectors of >= 5 names): the mean of
m random peers shares a median of

    m = 1: 0.58    m = 2: 0.74    m = 3: 0.85    m = 4: 0.92

of its variance with the full leave-one-out sector mean (lowest sector at
m = 2: 0.65, FMCG). One peer leaves ~42% of the "benchmark" as a single other
company's idiosyncratic move; two is the smallest mean whose variance is
majority-shared with its sector in every sector measured. So
`MIN_SECTOR_PEERS = 2`: a sector needs three names. Today 7 sectors hold one
or two names — 11 of the 84 tickers — and for those the three `sector_rel_*`
features are NULL, with `sector_rel_missing = 1`. Never 0.0: a zero is a
position on the scale, indistinguishable downstream from "exactly in line with
the sector", which is the silent-neutral defect this project has shipped three
times. (3 peers would lose the same 11 today; 4 would lose 15.)

THE LABEL FALLS BACK, AND SAYS SO. `target_excess_return` still needs a
benchmark for a thin-sector name, or the write guard refuses it forever and
the name stays stale. It falls back to the leave-one-out equal-weighted
UNIVERSE mean — the market as `regime.market_log_returns` defines it, minus
the stock — and records `benchmark_sector_specific = 0`, exactly as the old
mapping fell back to NIFTY 50 for sectors without a usable index. The
FEATURE does not fall back: a column called `sector_rel` that silently means
market-relative for a subset of names is a mislabelled column.

"UNKNOWN" IS MISSING, NEVER A SECTOR. An empty or absent industry label would
otherwise pool every unlabelled name into one pseudo-sector and average
unrelated companies — the silent-neutral fingerprint in a new place.

WHAT THIS DOES NOT REMOVE. The sector LABEL still comes from outside: NSE's
own constituent CSV (`data.universe.fetch_nse_constituents`, stored with
`source = 'nse-archives'`). That dependency is loud on death (the download
raises) and, since 2026-09-24, loud on a column rename (the schema check), but
`industry` is written ONCE, when a ticker joins, and never updated — so an NSE
reclassification does not propagate, and the label applied to 2016 is today's.
Stale but stable, and recorded rather than claimed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text

from pipeline.regime import market_log_returns

#: The fewest peers a leave-one-out sector mean may average over, per date.
#: See the module docstring for the measurement behind 2.
MIN_SECTOR_PEERS = 2

#: `benchmark_ticker` values. Not index symbols: nothing here is fetched.
SECTOR_PREFIX = "EW-LOO:"
MARKET_BENCHMARK = "EW-LOO:MARKET"

#: Industry values that mean "no label". Treated as missing, never as a sector.
_NO_LABEL = {"", "unknown", "nan", "none", "null"}


def sector_benchmark_id(sector: str) -> str:
    return f"{SECTOR_PREFIX}{sector}"


def clean_label(value) -> str | None:
    """An industry label, or None when it is absent or a placeholder."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    return None if s.lower() in _NO_LABEL else s


def sector_labels(tickers: list[str], engine=None) -> dict[str, str | None]:
    """
    Each ticker's industry, from its MOST RECENT `index_membership` row.

    Not restricted to open rows, unlike `data.tickers._metadata`: a frozen-
    universe name that leaves the NIFTY 100 has no open row, and reading only
    open rows would silently turn its sector into "Unknown" the day it left.
    """
    from data.db import get_engine

    engine = engine or get_engine()
    rows = pd.read_sql(
        text("SELECT ticker, effective_from, industry FROM index_membership"),
        engine)
    rows = rows[rows["ticker"].isin(tickers)].sort_values(["ticker", "effective_from"])
    latest = rows.groupby("ticker")["industry"].last()
    return {t: clean_label(latest.get(t)) for t in tickers}


@dataclass(frozen=True)
class Benchmark:
    """
    One stock's benchmark: a level series on the universe's date grid.

    `frame` holds `date`, `benchmark_close` and `bench_void_cum`, the running
    count of dates whose daily benchmark return could not be formed (too few
    peers traded). A window whose span contains any such date is void — never
    stretched across the hole, and never filled.
    """
    ticker: str
    name: str                 # stored as signals.benchmark_ticker
    sector_specific: bool     # stored as signals.benchmark_sector_specific
    relative_features: bool   # whether sector_rel_* may be computed against it
    peers: int                # names in the leave-one-out set
    frame: pd.DataFrame


def _price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Adjusted closes on the shared date grid, one column per ticker — the
    same adjustment basis `signals.compute_signals_frame` puts the stock on."""
    px = prices["adj_close"].where(prices["adj_close"].notna(), prices["close"])
    return (prices.assign(_px=pd.to_numeric(px, errors="coerce"))
            .pivot_table(index="date", columns="ticker", values="_px", aggfunc="last")
            .sort_index())


def _level(rets: pd.DataFrame, peers: list[str],
           min_peers: int) -> tuple[pd.Series, pd.Series]:
    """Equal-weighted mean daily log return over `peers` — the stock itself is
    never among them — cumulated into a level; and the void mask.

    Summed over the peers directly, never as (sector total - own): the
    subtraction leaves the stock in its own benchmark at the 1e-16 level, and
    "never in its own benchmark" is a statement that should be exact. A test
    moves the stock's own price and requires its benchmark bit-identical."""
    block = rets[peers]
    count = block.notna().sum(axis=1)
    mean = block.sum(axis=1, min_count=1) / count.where(count > 0)
    void = count < min_peers
    # The first grid date carries no return for anyone; it is the anchor, not
    # a hole, and must not void every window that starts on it.
    void.iloc[0] = False
    level = np.exp(mean.where(~void, 0.0).fillna(0.0).cumsum())
    return level, void


def build_benchmarks(prices: pd.DataFrame, sectors: dict[str, str | None],
                     tickers: list[str] | None = None,
                     min_peers: int = MIN_SECTOR_PEERS) -> dict[str, Benchmark]:
    """
    Every requested ticker's benchmark, from one frame of universe prices.

    `prices` — (date, ticker, close, adj_close) for the WHOLE universe; peers
    are drawn from the tickers it contains. `sectors` — ticker -> industry or
    None (see `sector_labels`). `tickers` — whose benchmarks to build; defaults
    to every ticker in `prices`. A requested ticker absent from `prices` gets a
    benchmark from its peers alone (nothing to leave out).
    """
    wide = _price_matrix(prices)
    rets, _market = market_log_returns(None, wide=wide)
    universe = list(wide.columns)
    tickers = universe if tickers is None else list(tickers)

    by_sector: dict[str, list[str]] = {}
    for t in universe:
        s = clean_label(sectors.get(t))
        if s is not None:
            by_sector.setdefault(s, []).append(t)

    out: dict[str, Benchmark] = {}
    for t in tickers:
        sector = clean_label(sectors.get(t))
        peers = [p for p in by_sector.get(sector, []) if p != t] if sector else []
        if len(peers) >= min_peers:
            name, specific = sector_benchmark_id(sector), True
        else:
            peers = [p for p in universe if p != t]
            name, specific = MARKET_BENCHMARK, False
        n_peers = len(peers)
        level, void = _level(rets, peers, min_peers)
        frame = pd.DataFrame({"date": wide.index.astype(str),
                              "benchmark_close": level.to_numpy(),
                              "bench_void_cum": void.cumsum().to_numpy()})
        out[t] = Benchmark(ticker=t, name=name, sector_specific=specific,
                           relative_features=specific, peers=n_peers, frame=frame)
    return out


def index_benchmark(ticker: str, name: str, series: pd.DataFrame,
                    sector_specific: bool) -> Benchmark:
    """
    A Benchmark from an EXTERNAL level series (date, benchmark_close) — the
    pre-v4 definition, kept so a panel measured under the Yahoo indices can be
    rebuilt for comparison. Forward-filled onto nothing: the caller supplies
    the series already on the grid it wants, exactly as the old code merged
    and forward-filled it onto the stock's own dates.
    """
    frame = series[["date", "benchmark_close"]].copy()
    frame["date"] = frame["date"].astype(str)
    frame["bench_void_cum"] = 0
    return Benchmark(ticker=ticker, name=name, sector_specific=sector_specific,
                     relative_features=True, peers=0, frame=frame)


def load_universe_prices(tickers: list[str], engine=None) -> pd.DataFrame:
    """(date, ticker, close, adj_close) for `tickers`, one query."""
    from data.db import get_engine

    engine = engine or get_engine()
    keys = [f"t{i}" for i in range(len(tickers))]
    sql = ("SELECT date, ticker, close, adj_close FROM ohlcv WHERE ticker IN ("
           + ", ".join(":" + k for k in keys) + ") ORDER BY date ASC")
    return pd.read_sql(text(sql), engine, params=dict(zip(keys, tickers)))
