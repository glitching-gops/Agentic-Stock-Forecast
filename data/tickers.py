"""
data/tickers.py — Ticker metadata and benchmark mapping.

This module answers "what is this ticker" (company name, industry, which index
it is measured against). It does NOT decide which tickers the pipeline
processes — that is ``data.universe.get_universe()``, which applies a
point-in-time rule.

The split matters: the previous design conflated the two, and the ticker list
was produced by ranking stocks on the model's own reported accuracy
(audit finding F4). Metadata lookups carry no such bias; universe selection did.

Benchmark mapping was verified against yfinance rather than assumed. Two
corrections to the previous map:

  - ``^CNXFIN`` returns no usable history; Financial Services now uses
    ``^NSEBANK``.
  - Telecommunication was mapped to ``^CNXMEDIA``, which is the Nifty Media
    index, not a telecom index. NSE publishes no telecom index with reliable
    yfinance history, so telecom names now benchmark against the broad market
    and are flagged as such via ``is_sector_specific=False``.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sqlalchemy import text

BROAD_MARKET_INDEX = "^NSEI"          # NIFTY 50 — fallback benchmark
BROAD_MARKET_NAME  = "NIFTY 50"

# NSE industry classification -> benchmark index.
# Every entry verified to return >1,200 daily rows over 5 years on yfinance.
SECTOR_INDICES: dict[str, str] = {
    "Financial Services":             "^NSEBANK",
    "Capital Goods":                  "^CNXINFRA",
    "Automobile and Auto Components": "^CNXAUTO",
    "Healthcare":                     "^CNXPHARMA",
    "Fast Moving Consumer Goods":     "^CNXFMCG",
    "Metals & Mining":                "^CNXMETAL",
    "Power":                          "^CNXENERGY",
    "Oil Gas & Consumable Fuels":     "^CNXENERGY",
    "Information Technology":         "^CNXIT",
    "Consumer Services":              "^CNXSERVICE",
    "Construction Materials":         "^CNXINFRA",
    "Construction":                   "^CNXINFRA",
    "Services":                       "^CNXSERVICE",
    "Realty":                         "^CNXREALTY",
    "Consumer Durables":              "^CNXCONSUM",
    "Media Entertainment & Publication": "^CNXMEDIA",
    # Deliberately absent, benchmarked against the broad market instead:
    #   "Chemicals"          - no representative NSE index
    #   "Telecommunication"  - no NSE telecom index with usable history
}

# Legacy sector labels from the pre-Phase-0 universe, kept so that historical
# rows written under the old scheme still resolve to a benchmark.
LEGACY_SECTOR_ALIASES: dict[str, str] = {
    "Banking & Finance":      "Financial Services",
    "Automobile":             "Automobile and Auto Components",
    "Pharmaceuticals":        "Healthcare",
    "FMCG":                   "Fast Moving Consumer Goods",
    "Energy":                 "Oil Gas & Consumable Fuels",
    "Infrastructure":         "Capital Goods",
    "Real Estate":            "Realty",
    "Consumer Discretionary": "Consumer Durables",
    "Telecom":                "Telecommunication",
}


@lru_cache(maxsize=1)
def _metadata() -> dict[str, dict[str, str]]:
    """
    Loads ticker metadata from the membership table.

    Cached for the process lifetime; call ``refresh_metadata()`` after a
    universe sync. Returns an empty dict if the table does not exist yet, so
    that importing this module never requires a populated database.
    """
    try:
        from data.db import get_engine
        df = pd.read_sql(
            text("""
                SELECT ticker, company, industry FROM index_membership
                WHERE effective_to = '9999-12-31'
            """),
            get_engine(),
        )
    except Exception:
        return {}

    if df.empty:
        return {}

    return {
        row["ticker"]: {
            "company":  row["company"] or row["ticker"].replace(".NS", ""),
            "industry": row["industry"] or "",
        }
        for _, row in df.iterrows()
    }


def refresh_metadata() -> None:
    """Clears the metadata cache. Call after ``sync_current_membership()``."""
    _metadata.cache_clear()


def get_company(ticker: str) -> str:
    """Company name for a ticker, falling back to the bare symbol."""
    meta = _metadata().get(ticker)
    if meta and meta["company"]:
        return meta["company"]
    return ticker.replace(".NS", "")


def get_sector(ticker: str) -> str:
    """NSE industry classification for a ticker, or 'Unknown'."""
    meta = _metadata().get(ticker)
    if meta and meta["industry"]:
        return meta["industry"]
    return "Unknown"


def get_benchmark(ticker: str, sector: str | None = None) -> tuple[str, bool]:
    """
    Returns ``(index_ticker, is_sector_specific)`` for a stock.

    ``is_sector_specific`` is False when the stock falls back to the broad
    market because no reliable sector index exists. Callers should record this
    alongside the forecast: an "excess return vs sector" figure means something
    different when the benchmark is actually NIFTY 50.
    """
    sector = sector or get_sector(ticker)
    sector = LEGACY_SECTOR_ALIASES.get(sector, sector)

    index = SECTOR_INDICES.get(sector)
    if index:
        return index, True
    return BROAD_MARKET_INDEX, False


def get_benchmark_name(index_ticker: str) -> str:
    """Human-readable benchmark name for display."""
    names = {
        "^NSEI":       "NIFTY 50",
        "^NSEBANK":    "NIFTY Bank",
        "^CNXIT":      "NIFTY IT",
        "^CNXAUTO":    "NIFTY Auto",
        "^CNXPHARMA":  "NIFTY Pharma",
        "^CNXFMCG":    "NIFTY FMCG",
        "^CNXMETAL":   "NIFTY Metal",
        "^CNXENERGY":  "NIFTY Energy",
        "^CNXINFRA":   "NIFTY Infrastructure",
        "^CNXREALTY":  "NIFTY Realty",
        "^CNXCONSUM":  "NIFTY India Consumption",
        "^CNXSERVICE": "NIFTY Services Sector",
        "^CNXMEDIA":   "NIFTY Media",
    }
    return names.get(index_ticker, index_ticker)


def get_all_sectors() -> list[str]:
    """Sorted list of industries present in the current universe."""
    return sorted({m["industry"] for m in _metadata().values() if m["industry"]})


def get_tickers_by_sector(sector: str) -> list[str]:
    """All tickers in the current universe belonging to an industry."""
    return sorted(t for t, m in _metadata().items() if m["industry"] == sector)


def all_known_tickers() -> list[str]:
    """
    Every ticker with metadata. This is a metadata listing, NOT the universe —
    use ``data.universe.get_universe(as_of)`` to decide what to process.
    """
    return sorted(_metadata().keys())
