"""
data/universe.py — Point-in-time universe construction.

Replaces the previous universe definition, which was produced by
``tools/select_top_50.py``: it ranked stocks by composite score and kept the
top 5 per sector. Because composite score is largely a function of the model's
own reported accuracy, the universe was selected on realised performance and
every metric averaged over it was biased upward (audit finding F4).

This module defines the universe by a rule that references no model output:

    Nifty 100 membership as of date D
      AND 20-day median traded value >= LIQUIDITY_FLOOR_INR
      AND at least MIN_LISTING_DAYS of price history

Membership is stored in the ``index_membership`` table as intervals
``(ticker, index_name, effective_from, effective_to)`` so that
``get_universe(as_of)`` answers "who was in the index on that date", not
"who is in it today".

Two sources populate that table:

  1. ``sync_current_membership()`` — fetches the official NSE constituent CSV
     and records it as a snapshot valid from today. Run daily; the project then
     accumulates genuine point-in-time history from this date forward.

  2. ``backfill_membership_from_wayback()`` — reconstructs history from
     Internet Archive snapshots of the same NSE CSV. Optional and best-effort;
     archive.org's CDX endpoint is frequently unavailable, so this is a tool the
     operator runs when it responds rather than a pipeline step.

KNOWN LIMITATION — survivorship bias. Until a backfill succeeds, membership is
only known from the first ``sync_current_membership()`` call onward. Evaluations
over earlier periods use today's membership and therefore exclude companies
that were delisted or demoted. ``describe_universe_bias()`` returns this caveat
so it can be printed alongside any metric rather than left implicit.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from sqlalchemy import text

from data.db import get_engine

# ── Rule parameters ───────────────────────────────────────────────────────────
# These define the universe. They must never reference model output.
INDEX_NAME          = "NIFTY100"
LIQUIDITY_FLOOR_INR = 25_00_00_000      # Rs 25 crore, 20-day median traded value
LIQUIDITY_WINDOW    = 20
MIN_LISTING_DAYS    = 750               # ~3 years of trading sessions

NSE_CONSTITUENT_URLS = {
    "NIFTY50":  "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}

_UA = {"User-Agent": "Mozilla/5.0 (compatible; agentic-stock-forecast/2.0)"}

# Open-ended membership intervals use this sentinel rather than NULL so that
# BETWEEN-style date comparisons work identically on SQLite and PostgreSQL.
OPEN_ENDED = "9999-12-31"


@dataclass(frozen=True)
class UniverseRule:
    """The universe definition, recorded alongside results for reproducibility."""

    index_name: str = INDEX_NAME
    liquidity_floor_inr: int = LIQUIDITY_FLOOR_INR
    liquidity_window: int = LIQUIDITY_WINDOW
    min_listing_days: int = MIN_LISTING_DAYS

    def fingerprint(self) -> str:
        """Stable string identifying this rule, for run metadata."""
        return json.dumps(self.__dict__, sort_keys=True)


DEFAULT_RULE = UniverseRule()


# ── Schema ────────────────────────────────────────────────────────────────────
def init_universe_tables() -> None:
    """Creates the membership table. Safe to call repeatedly."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS index_membership (
                ticker         TEXT NOT NULL,
                index_name     TEXT NOT NULL,
                effective_from TEXT NOT NULL,
                effective_to   TEXT NOT NULL,
                company        TEXT,
                industry       TEXT,
                source         TEXT,
                PRIMARY KEY (ticker, index_name, effective_from)
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_membership_lookup "
            "ON index_membership (index_name, effective_from, effective_to)"
        ))
        conn.commit()


# ── Source 1: current membership from NSE ─────────────────────────────────────
def fetch_nse_constituents(index_name: str = INDEX_NAME) -> pd.DataFrame:
    """
    Downloads the official NSE constituent list.

    Returns a DataFrame with columns: ticker, company, industry.
    Tickers are suffixed with '.NS' to match the yfinance convention used
    throughout the pipeline. Raises on failure — callers decide whether an
    empty universe is acceptable, this function does not silently return one.
    """
    url = NSE_CONSTITUENT_URLS[index_name]
    resp = requests.get(url, headers=_UA, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "symbol" not in df.columns:
        raise ValueError(f"Unexpected NSE CSV schema for {index_name}: {list(df.columns)}")

    out = pd.DataFrame({
        "ticker":   df["symbol"].astype(str).str.strip() + ".NS",
        "company":  df.get("company_name", pd.Series(dtype=str)).astype(str).str.strip(),
        "industry": df.get("industry", pd.Series(dtype=str)).astype(str).str.strip(),
    })
    return out[out["ticker"].str.len() > 3].reset_index(drop=True)


def sync_current_membership(
    index_name: str = INDEX_NAME,
    as_of: str | None = None,
) -> int:
    """
    Records today's index membership as a point-in-time snapshot.

    Opens an interval for tickers that have joined, and closes the interval for
    tickers that have left. Idempotent: calling twice on the same day is a
    no-op. Returns the number of membership changes written.
    """
    init_universe_tables()
    as_of = as_of or date.today().isoformat()

    current = fetch_nse_constituents(index_name)
    current_set = set(current["ticker"])

    engine = get_engine()
    with engine.connect() as conn:
        open_rows = pd.read_sql(
            text(
                "SELECT ticker FROM index_membership "
                "WHERE index_name = :idx AND effective_to = :open"
            ),
            conn, params={"idx": index_name, "open": OPEN_ENDED},
        )
        open_set = set(open_rows["ticker"]) if not open_rows.empty else set()

        joined = current_set - open_set
        left   = open_set - current_set
        changes = 0

        for _, row in current[current["ticker"].isin(joined)].iterrows():
            conn.execute(text("""
                INSERT INTO index_membership
                    (ticker, index_name, effective_from, effective_to, company, industry, source)
                VALUES (:t, :idx, :from, :open, :company, :industry, 'nse-archives')
            """), {
                "t": row["ticker"], "idx": index_name, "from": as_of,
                "open": OPEN_ENDED, "company": row["company"], "industry": row["industry"],
            })
            changes += 1

        for ticker in left:
            conn.execute(text("""
                UPDATE index_membership SET effective_to = :to
                WHERE ticker = :t AND index_name = :idx AND effective_to = :open
            """), {"to": as_of, "t": ticker, "idx": index_name, "open": OPEN_ENDED})
            changes += 1

        conn.commit()

    print(f"[Universe] {index_name} @ {as_of}: {len(joined)} joined, {len(left)} left "
          f"({len(current_set)} current members)")
    return changes


# ── Source 2: optional Wayback backfill ───────────────────────────────────────
def backfill_membership_from_wayback(
    index_name: str = INDEX_NAME,
    start_year: int = 2018,
    max_snapshots: int = 120,
    timeout: int = 45,
) -> int:
    """
    Best-effort reconstruction of historical membership from Internet Archive
    snapshots of the NSE constituent CSV.

    This is a manual tool, not a pipeline step: archive.org's CDX endpoint is
    frequently unavailable, and a partial backfill is worse than none if it is
    mistaken for complete history. Returns the number of snapshots ingested;
    0 means the archive was unreachable or held nothing, and the caller should
    keep treating history as unknown.
    """
    init_universe_tables()
    url = NSE_CONSTITUENT_URLS[index_name].replace("https://", "")

    cdx = (
        "https://web.archive.org/cdx/search/cdx"
        f"?url={url}&output=json&fl=timestamp,statuscode"
        f"&filter=statuscode:200&collapse=digest&from={start_year}"
    )

    try:
        resp = requests.get(cdx, headers=_UA, timeout=timeout)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
        print(f"[Universe] Wayback CDX unavailable ({exc}). "
              f"History remains unknown before the first sync_current_membership() call.")
        return 0

    if len(rows) <= 1:
        print("[Universe] Wayback holds no snapshots of the NSE constituent list.")
        return 0

    snapshots = rows[1:][:max_snapshots]
    ingested = 0

    for timestamp, _status in snapshots:
        snap_date = datetime.strptime(timestamp[:8], "%Y%m%d").date().isoformat()
        snap_url = f"https://web.archive.org/web/{timestamp}id_/{NSE_CONSTITUENT_URLS[index_name]}"
        try:
            r = requests.get(snap_url, headers=_UA, timeout=timeout)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if "symbol" not in df.columns:
                continue
            tickers = (df["symbol"].astype(str).str.strip() + ".NS").tolist()
            _record_snapshot(index_name, snap_date, tickers, source="wayback")
            ingested += 1
            time.sleep(0.5)                       # be polite to archive.org
        except Exception as exc:                  # noqa: BLE001
            print(f"[Universe] snapshot {snap_date} failed: {exc}")
            continue

    print(f"[Universe] Backfilled {ingested} historical snapshots for {index_name}.")
    return ingested


def _record_snapshot(index_name: str, snap_date: str, tickers: list[str], source: str) -> None:
    """Writes one dated membership snapshot, ignoring tickers already open on that date."""
    engine = get_engine()
    with engine.connect() as conn:
        for ticker in tickers:
            exists = conn.execute(text("""
                SELECT 1 FROM index_membership
                WHERE ticker = :t AND index_name = :idx
                  AND effective_from <= :d AND effective_to >= :d
                LIMIT 1
            """), {"t": ticker, "idx": index_name, "d": snap_date}).fetchone()
            if exists:
                continue
            # The existence check above makes a plain INSERT safe on both
            # dialects; a duplicate primary key here means concurrent writers,
            # which the daily pipeline does not have.
            conn.execute(text("""
                INSERT INTO index_membership
                    (ticker, index_name, effective_from, effective_to, source)
                VALUES (:t, :idx, :from, :open, :source)
            """), {
                "t": ticker, "idx": index_name, "from": snap_date,
                "open": OPEN_ENDED, "source": source,
            })
        conn.commit()


# ── The universe rule ─────────────────────────────────────────────────────────
def get_index_members(as_of: str, index_name: str = INDEX_NAME) -> list[str]:
    """Returns index membership as of a date, from recorded intervals only."""
    engine = get_engine()
    df = pd.read_sql(
        text("""
            SELECT DISTINCT ticker FROM index_membership
            WHERE index_name = :idx
              AND effective_from <= :d
              AND effective_to   >  :d
            ORDER BY ticker
        """),
        engine, params={"idx": index_name, "d": as_of},
    )
    return df["ticker"].tolist() if not df.empty else []


def get_universe(
    as_of: str | None = None,
    rule: UniverseRule = DEFAULT_RULE,
    apply_liquidity: bool = True,
) -> list[str]:
    """
    Returns the tradable universe as of a date.

    Applies index membership, then the liquidity floor and listing-history
    minimum, both computed only from data dated on or before ``as_of``. No step
    references model output, forecast accuracy, or realised returns.
    """
    as_of = as_of or date.today().isoformat()
    members = get_index_members(as_of, rule.index_name)

    if not members:
        return []
    if not apply_liquidity:
        return members

    engine = get_engine()
    window_start = (date.fromisoformat(as_of)
                    - timedelta(days=rule.liquidity_window * 2)).isoformat()

    keep: list[str] = []
    no_data: list[str] = []
    too_short: list[str] = []
    illiquid: list[str] = []

    for ticker in members:
        stats = pd.read_sql(
            text("""
                SELECT close, volume FROM ohlcv
                WHERE ticker = :t AND date <= :d AND date >= :w
                ORDER BY date DESC
            """),
            engine, params={"t": ticker, "d": as_of, "w": window_start},
        )
        history = pd.read_sql(
            text("SELECT COUNT(*) AS n FROM ohlcv WHERE ticker = :t AND date <= :d"),
            engine, params={"t": ticker, "d": as_of},
        )
        n_rows = int(history["n"].iloc[0]) if not history.empty else 0

        if n_rows == 0:
            no_data.append(ticker)
            continue
        if n_rows < rule.min_listing_days:
            too_short.append(ticker)
            continue
        if stats.empty:
            no_data.append(ticker)
            continue

        traded_value = stats["close"].astype(float) * stats["volume"].astype(float)
        if float(traded_value.head(rule.liquidity_window).median()) >= rule.liquidity_floor_inr:
            keep.append(ticker)
        else:
            illiquid.append(ticker)

    # Distinguish "screened out" from "never ingested". Silently conflating the
    # two is how a 100-name universe quietly becomes a 5-name one: the liquidity
    # screen reads the ohlcv table, which is only populated for tickers that
    # have already been fetched. Callers must fetch over get_index_members()
    # first, then screen with this function.
    if no_data:
        print(f"[Universe] {len(no_data)} index members have no OHLCV data and were "
              f"excluded. Run pipeline.fetch.fetch_and_store(tickers=get_index_members(...)) "
              f"before screening. Missing: {', '.join(sorted(no_data)[:5])}"
              f"{' ...' if len(no_data) > 5 else ''}")
    if too_short or illiquid:
        print(f"[Universe] screened out {len(too_short)} for short history, "
              f"{len(illiquid)} for liquidity")

    return sorted(keep)


def get_ingest_universe(as_of: str | None = None,
                        rule: UniverseRule = DEFAULT_RULE) -> list[str]:
    """
    Tickers to FETCH data for: raw index membership, before any screen that
    depends on that data existing.

    Use this for ingestion, and ``get_universe()`` for modelling.
    """
    as_of = as_of or date.today().isoformat()
    return get_index_members(as_of, rule.index_name)


def describe_universe_bias(index_name: str = INDEX_NAME) -> dict:
    """
    Returns the survivorship-bias caveat for the recorded membership history.

    Print this next to any metric averaged over the universe. It reports the
    earliest date for which membership is actually known; evaluation windows
    starting before it are survivorship-biased.
    """
    engine = get_engine()
    try:
        df = pd.read_sql(
            text("""
                SELECT MIN(effective_from) AS earliest,
                       COUNT(DISTINCT ticker) AS n_tickers,
                       COUNT(*) AS n_intervals
                FROM index_membership WHERE index_name = :idx
            """),
            engine, params={"idx": index_name},
        )
    except Exception:
        return {"known_from": None, "survivorship_bias": True,
                "note": "No membership history recorded."}

    if df.empty or pd.isna(df["earliest"].iloc[0]):
        return {"known_from": None, "survivorship_bias": True,
                "note": "No membership history recorded. Run sync_current_membership()."}

    earliest = str(df["earliest"].iloc[0])
    return {
        "known_from": earliest,
        "n_tickers": int(df["n_tickers"].iloc[0]),
        "n_intervals": int(df["n_intervals"].iloc[0]),
        "survivorship_bias": True,
        "note": (
            f"Point-in-time membership is known from {earliest}. Evaluation windows "
            f"beginning before that date use present-day membership and therefore "
            f"exclude delisted and demoted companies (survivorship bias). Run "
            f"backfill_membership_from_wayback() to extend recorded history."
        ),
    }


if __name__ == "__main__":
    init_universe_tables()
    sync_current_membership()
    print(json.dumps(describe_universe_bias(), indent=2))
