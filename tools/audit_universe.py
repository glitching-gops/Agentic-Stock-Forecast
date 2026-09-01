"""
tools/audit_universe.py — has the frozen universe drifted away from its rule?

WHY THIS EXISTS. The forecasting universe is frozen in data/frozen_universe.py
as a checked-in list, because a universe recomputed at run time changes the row
set underneath a measurement and makes two runs incomparable. But a frozen list
has the opposite failure: it goes quietly wrong. A name can be delisted, drop
out of the index, dry up in liquidity, or stop being fetched, and the file
still names it.

So the rule stays executable and this tool runs it. It is the same arrangement
as tools/audit_benchmarks.py --apply-check: measurement decides nothing on its
own, it just refuses to let the committed answer diverge from the evidence
without somebody seeing it.

WHAT IT CHECKS, against the live database:

  1. Every frozen ticker still passes the rule (history, liquidity, data).
  2. Every frozen ticker is still fetched to the current session.
  3. Every frozen ticker's session grid has no NEW interior holes.
  4. Which non-frozen names WOULD now qualify — reported, never applied.

WHY NOTHING IS APPLIED AUTOMATICALLY. Adding a ticker changes the panel every
Phase 2 result was measured over. Removing one silently shrinks the sample a
forward record is accumulating on. Both are decisions that must bump
MODEL_VERSION and be taken deliberately, exactly like a change to
SECTOR_INDICES — see the benchmark landmine in CLAUDE.md.

Usage:
    python tools/audit_universe.py
    python tools/audit_universe.py --apply-check     # exit 1 on any drift
    python tools/audit_universe.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.db import get_engine                              # noqa: E402
from data.frozen_universe import (                          # noqa: E402
    FROZEN_MEASUREMENTS,
    FROZEN_UNIVERSE,
    MIN_SESSIONS,
    frozen_fingerprint,
)
from data.universe import DEFAULT_RULE, get_index_members   # noqa: E402

#: A frozen ticker whose last stored session is older than this many calendar
#: days is stale. Deliberately generous: a fortnight covers a long exchange
#: holiday plus a weekend run, while a feed that has genuinely stopped fires.
MAX_STALE_DAYS = 14


def _load() -> tuple[pd.DataFrame, set[str]]:
    """
    Per-ticker session count, span, holes and liquidity, plus the calendar.

    EVERYTHING HERE IS AN AGGREGATE, deliberately. The obvious implementation
    reads `SELECT ticker, date, close, volume FROM ohlcv` and does the work in
    pandas — 240,000 rows — and it dies partway through with `SSL connection
    has been closed unexpectedly`, because DATABASE_URL points at Supabase's
    SESSION pooler on 5432 and a long single query does not survive it. That
    is a recorded landmine, not a surprise. Three GROUP BY queries returning
    ~2,500 / ~120 / ~4,000 rows carry the same information and return in
    milliseconds.
    """
    engine = get_engine()

    # The NSE session calendar, derived from the panel itself rather than from
    # a holiday file: a date on which a majority of the tickers traded. A
    # per-ticker date set cannot distinguish "the exchange was shut" from
    # "this bar is missing", and only the second is a defect.
    per_date = pd.read_sql(
        "SELECT date, COUNT(DISTINCT ticker) AS n FROM ohlcv GROUP BY date", engine)
    n_tickers = pd.read_sql(
        "SELECT COUNT(DISTINCT ticker) AS n FROM ohlcv", engine)["n"].iloc[0]
    calendar = set(per_date.loc[per_date["n"] >= 0.5 * n_tickers, "date"])

    span = pd.read_sql(
        "SELECT ticker, COUNT(*) AS sessions, MIN(date) AS first, MAX(date) AS last "
        "FROM ohlcv GROUP BY ticker", engine).set_index("ticker")

    # Bars a ticker holds that the calendar also holds. Counting stored rows
    # instead would let a bar dated outside the calendar cancel out a genuine
    # hole, and the two errors are not related.
    on_calendar = pd.read_sql(
        text("SELECT ticker, COUNT(*) AS n FROM ohlcv WHERE date IN :cal GROUP BY ticker")
        .bindparams(bindparam("cal", expanding=True)),
        engine, params={"cal": sorted(calendar)}).set_index("ticker")["n"]

    # The liquidity window only needs the tail. Two calendar days per session
    # is ample slack for weekends and holidays.
    cutoff = sorted(calendar)[-DEFAULT_RULE.liquidity_window * 3]
    recent = pd.read_sql(
        text("SELECT ticker, date, close, volume FROM ohlcv WHERE date >= :cutoff"),
        engine, params={"cutoff": cutoff})

    rows = []
    for ticker, row in span.iterrows():
        lo, hi = row["first"], row["last"]
        in_range = sum(1 for c in calendar if lo <= c <= hi)
        window = (recent[recent["ticker"] == ticker]
                  .sort_values("date").tail(DEFAULT_RULE.liquidity_window))
        rows.append({
            "ticker": ticker,
            "sessions": int(row["sessions"]),
            "first": lo,
            "last": hi,
            "holes": max(in_range - int(on_calendar.get(ticker, 0)), 0),
            "adv": float((window["close"] * window["volume"]).median())
                   if not window.empty else 0.0,
        })

    return pd.DataFrame(rows).set_index("ticker"), calendar


def audit() -> dict:
    live, calendar = _load()
    members = set(get_index_members(date.today().isoformat(), DEFAULT_RULE.index_name))
    latest = max(calendar) if calendar else None
    recorded = {t: (n, first, holes, adv)
                for t, n, first, holes, adv in FROZEN_MEASUREMENTS}

    findings: list[dict] = []

    for ticker in FROZEN_UNIVERSE:
        if ticker not in live.index:
            findings.append({"ticker": ticker, "issue": "no_ohlcv_rows",
                             "detail": "frozen but absent from ohlcv entirely"})
            continue

        row = live.loc[ticker]
        was_sessions, _, was_holes, _ = recorded[ticker]

        if int(row.sessions) < MIN_SESSIONS:
            findings.append({"ticker": ticker, "issue": "short_history",
                             "detail": f"{int(row.sessions)} sessions < {MIN_SESSIONS}"})

        # A DECREASE in stored rows is the F6 shape in a new place: ohlcv and
        # signals are both rewritten by range, and a run that LOSES history
        # looks identical afterwards to one that never had it.
        if int(row.sessions) < was_sessions:
            findings.append({"ticker": ticker, "issue": "sessions_lost",
                             "detail": f"{was_sessions} at freeze -> {int(row.sessions)} now"})

        if float(row.adv) < DEFAULT_RULE.liquidity_floor_inr:
            findings.append({"ticker": ticker, "issue": "illiquid",
                             "detail": f"Rs {row.adv / 1e7:.1f} cr median daily value"})

        if latest and row.last < latest:
            gap = (date.fromisoformat(latest) - date.fromisoformat(row.last)).days
            if gap > MAX_STALE_DAYS:
                findings.append({"ticker": ticker, "issue": "stale",
                                 "detail": f"last session {row.last}, {gap} days behind {latest}"})

        if int(row.holes) > was_holes:
            findings.append({"ticker": ticker, "issue": "new_session_holes",
                             "detail": f"{was_holes} holes at freeze -> {int(row.holes)} now"})

    # Names the rule would now admit. Informational: growing the universe is a
    # decision, and it is never taken by a script.
    candidates = sorted(
        t for t in live.index
        if t not in set(FROZEN_UNIVERSE)
        and t in members
        and int(live.loc[t, "sessions"]) >= MIN_SESSIONS
        and float(live.loc[t, "adv"]) >= DEFAULT_RULE.liquidity_floor_inr
    )

    return {
        "fingerprint": frozen_fingerprint(),
        "as_of": latest,
        "n_frozen": len(FROZEN_UNIVERSE),
        "findings": findings,
        "would_now_qualify": candidates,
        "no_longer_index_members": sorted(set(FROZEN_UNIVERSE) - members),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    parser.add_argument("--apply-check", action="store_true",
                        help="exit 1 if the frozen universe has drifted from the rule")
    args = parser.parse_args()

    report = audit()

    print(f"Frozen universe: {report['n_frozen']} tickers  [{report['fingerprint']}]")
    print(f"Latest session in ohlcv: {report['as_of']}")
    print()

    if report["findings"]:
        print(f"DRIFT — {len(report['findings'])} finding(s):")
        for f in report["findings"]:
            print(f"  {f['ticker']:<16} {f['issue']:<20} {f['detail']}")
    else:
        print("No drift. Every frozen ticker still satisfies the rule it was chosen by.")
    print()

    if report["would_now_qualify"]:
        print(f"{len(report['would_now_qualify'])} name(s) would now qualify but are "
              f"NOT in the frozen universe. Adding one changes every panel result and "
              f"must bump MODEL_VERSION:")
        print(f"  {', '.join(report['would_now_qualify'])}")
        print()

    if report["no_longer_index_members"]:
        print(f"{len(report['no_longer_index_members'])} frozen name(s) have left the "
              f"index. They are still fetched and still forecast — that is the freeze "
              f"working, not a fault — but the universe no longer describes NIFTY 100:")
        print(f"  {', '.join(report['no_longer_index_members'])}")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    if args.apply_check and report["findings"]:
        print("FAIL: the frozen universe disagrees with the rule that produced it.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
