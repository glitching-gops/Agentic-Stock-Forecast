"""
tools/backfill_news.py — build the dated archive, back to the start of the panel.

    python tools/backfill_news.py --start 2016-09-01 --end 2026-09-03
    python tools/backfill_news.py --tickers RELIANCE.NS INFY.NS --start 2024-01-01
    python tools/backfill_news.py --report            # what is already stored
    python tools/backfill_news.py --dry-run --limit 3 # cost estimate, no writes

WHY THIS IS POSSIBLE AT ALL
---------------------------
Every plan in this project since Phase 0 recorded that news was unbacktestable
because Google News RSS serves no archive. Measured 2026-09-03, the RSS search
endpoint honours Google's `after:` / `before:` operators and returns correctly
dated articles back to at least 2016-09 — the month the `macro` table starts.

RESUMABLE, AND A RE-RUN IS NEARLY FREE
--------------------------------------
Every window that completes writes a `news_coverage` row, and by default this
skips any (ticker, window) already recorded `ok`. So a run that dies at hour
three resumes at hour three rather than at zero, and re-running after adding a
ticker costs only that ticker. `--refetch` overrides it.

THE COST IS DOMINATED BY THE SPLIT, NOT BY THE TICKER COUNT
-----------------------------------------------------------
Monthly windows are the starting granularity and most names never leave it:
measured, MUTHOOTFIN returned 14 articles for January 2024 and UNIONBANK 9 for
June 2019. Large caps saturate — RELIANCE hit the 100 cap for January 2024 and
split to weeks — so cost scales with how newsworthy a name is rather than with
the calendar. Budget roughly 12k-18k requests at ~0.65 s for 84 names over ten
years, so three to five hours. `--report` afterwards is not optional.

READ THE COVERAGE REPORT BEFORE BELIEVING ANY FEATURE BUILT ON THIS
--------------------------------------------------------------------
A backfill whose holes concentrate in the early panel would manufacture exactly
the early-fold artifact this project has now seen three times — valuation at
+3.32, LoRA at +2.37 and pooled_xgb at +2.42, every one of them carried by the
earliest fold. Articles-per-year and saturated-window counts are the two
numbers that say whether that is happening.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from sqlalchemy import text                                       # noqa: E402

from data.db import get_engine                                    # noqa: E402
from pipeline.news import (                                       # noqa: E402
    GoogleNewsRSS,
    company_aliases,
    company_name_is_unresolved,
    coverage_report,
    iter_windows,
    match_ticker,
    month_starts,
    search_query,
    store_window,
)


#: Abort only on WHOLESALE refusal. Measured on the first real run, the
#: ordinary transient block rate is ~2.7%; a burst far above that, sustained
#: across several tickers, is Google refusing us rather than a flaky window.
MAX_BLOCK_RATE = 0.30
RECENT_TICKERS = 3


def unresolved_remediation(unresolved: list[str], tickers: list[str]) -> str:
    """
    What to tell the reader when `get_company` is handing back bare symbols.

    THE REMEDIATION DEPENDS ON WHY, and the first version of this guessed. It
    said "run refresh_metadata()" while the actual cause was an unreachable
    database, which that call cannot help with — the reader follows it, watches
    it change nothing, and learns nothing. Same class of mistake as
    `TimesFMUnavailable` telling a Kaggle user to install
    requirements-series.txt: confident, specific and wrong.

    A few names missing means the table is populated but stale. EVERY name
    missing means the table is empty or the database is not there, so the first
    thing to check is the connection. `data.tickers._metadata` used to swallow
    the connection error, which is what made the two indistinguishable from
    here; that is fixed at the source, and this says which one it is.
    """
    head = (
        f"\nREFUSING TO RUN: {len(unresolved)} of {len(tickers)} tickers have "
        f"no company name, so `get_company` is\nreturning the bare symbol. The "
        f"query would become the symbol and the archive would\ncome back "
        f"near-empty for each, silently:\n"
        f"  {', '.join(unresolved[:12])}"
        f"{' ...' if len(unresolved) > 12 else ''}\n"
    )
    if len(unresolved) == len(tickers):
        return head + (
            "\nEVERY ticker is affected, which means the metadata table is "
            "empty or UNREACHABLE\nrather than a few names being missing. In "
            "order:\n"
            "  1. Is the database up?   python -c \"from data.db import "
            "get_engine; get_engine().connect()\"\n"
            "  2. Populate membership:  python -c \"import data.universe as u; "
            "u.sync_current_membership()\"\n"
            "  3. Clear the cache:      python -c \"import data.tickers as t; "
            "t.refresh_metadata()\"")
    return head + (
        "\nOnly some names are affected, so the table is populated but "
        "incomplete. Re-sync\nmembership, then clear the cache:\n"
        "  python -c \"import data.universe as u; u.sync_current_membership()\"\n"
        "  python -c \"import data.tickers as t; t.refresh_metadata()\"")


def _completed(engine, provider: str) -> set[tuple[str, str, str]]:
    """
    (ticker, start, end) triples already recorded ok — the resume point.

    A MISSING TABLE is the one soft case, and it is narrowed to exactly that by
    `data.db.is_missing_relation` rather than a bare `except Exception`. The
    wider guard is what let a live database outage be served as an empty
    result and cached as data for a day; here it would silently re-fetch a
    completed backfill from scratch against a database that is merely
    unreachable.
    """
    from data.db import is_missing_relation

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT ticker, window_start, window_end FROM news_coverage "
                "WHERE provider = :p AND status = 'ok'"),
                {"p": provider}).fetchall()
    except Exception as exc:                                     # noqa: BLE001
        if not is_missing_relation(exc):
            raise
        print("  news_coverage does not exist yet — run data.db.init_db() "
              "before a real backfill; treating everything as unfetched.")
        return set()
    return {(r[0], r[1], r[2]) for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2016-09-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--limit", type=int, help="first N tickers only (smoke test)")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="seconds between requests; be polite")
    ap.add_argument("--refetch", action="store_true",
                    help="re-fetch windows already recorded ok")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and count, write nothing")
    ap.add_argument("--report", action="store_true",
                    help="print what is stored and exit")
    args = ap.parse_args()

    engine = get_engine()

    if args.report:
        rep = coverage_report(engine)
        print("=" * 70)
        print("NEWS ARCHIVE")
        print("=" * 70)
        t = rep["totals"]
        print(f"  {t.get('n_articles', 0):,} articles, "
              f"{t.get('first', '-')} .. {t.get('last', '-')}")
        print("\n  articles by year — HOLES CONCENTRATED EARLY ARE THE HAZARD:")
        for row in rep["by_year"]:
            bar = "#" * min(60, row["n_articles"] // 200)
            print(f"    {row['year']}  {row['n_articles']:>7,}  {bar}")
        print("\n  windows by status:")
        for row in rep["by_status"]:
            print(f"    {row['status']:<8} {row['n_windows']:>7,} windows, "
                  f"{row['n_saturated'] or 0:>5} saturated")
        holes = rep.get("shared_holes") or []
        if holes:
            print("\n  WINDOWS THAT FAILED FOR MORE THAN ONE TICKER — a hole in"
                  "\n  the index, not a refusal aimed at us. Re-running will not"
                  "\n  fill these, and a feature reading them sees zero articles"
                  "\n  across the whole universe:")
            for row in holes[:12]:
                print(f"    {row['window_start']} .. {row['window_end']}   "
                      f"{row['n_tickers']} tickers")
            if len(holes) > 12:
                print(f"    ... and {len(holes) - 12} more")

        saturated = sum(r["n_saturated"] or 0 for r in rep["by_status"])
        if saturated:
            print(f"\n  {saturated} SATURATED windows remain. Those were "
                  f"relevance-ranked\n  by a model that has seen the future "
                  f"relative to the window, so their\n  article SELECTION is "
                  f"not point-in-time. They are day-granular and\n  cannot be "
                  f"split further; treat them as a known, bounded dent.")
        return 0

    from data.frozen_universe import FROZEN_UNIVERSE
    from data.tickers import get_company

    tickers = args.tickers or sorted(FROZEN_UNIVERSE)
    if args.limit:
        tickers = tickers[:args.limit]

    # CHECKED BEFORE THE REQUESTS, NOT AFTER THEM. `get_company` falls back to
    # the bare ticker symbol when the metadata table has no row, and nothing
    # raises — the query silently becomes "RELIANCE" instead of "Reliance
    # Industries", the only alias is the all-caps symbol, and the run completes
    # in hours having stored almost nothing. Found the hard way on a scratch
    # database with no metadata: RELIANCE and MUTHOOTFIN kept ZERO articles
    # each and the tool reported success.
    unresolved = [t for t in tickers if company_name_is_unresolved(t, get_company(t))]
    if unresolved:
        # THE REMEDIATION DEPENDS ON WHY, and the first version of this message
        # guessed. It told the reader to run `refresh_metadata()` when in fact
        # the database was unreachable, which that call cannot help with — the
        # same class of mistake as `TimesFMUnavailable` telling a Kaggle user to
        # install requirements-series.txt. `_metadata()` used to swallow the
        # connection error, so "no metadata" and "no database" were genuinely
        # indistinguishable here; that is fixed at the source in data/tickers.py
        # and this now says which one it is.
        print(unresolved_remediation(unresolved, tickers))
        return 1

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    provider = GoogleNewsRSS(delay=args.delay)

    done = set() if args.refetch else _completed(engine, provider.name)
    print(f"{len(tickers)} tickers, {start} .. {end}, "
          f"{len(done):,} windows already complete"
          f"{'  [DRY RUN]' if args.dry_run else ''}")

    began = time.time()
    totals = {"requests": 0, "articles": 0, "kept": 0, "saturated": 0,
              "blocked": 0, "errors": 0, "skipped": 0}
    recent: list[float] = []

    for i, ticker in enumerate(tickers, 1):
        company = get_company(ticker)
        aliases = company_aliases(ticker, company)
        query = search_query(ticker, company)
        kept_for_ticker = 0
        blocked_for_ticker = 0
        windows_for_ticker = 0

        for lo, hi in month_starts(start, end):
            if (ticker, lo.isoformat(), hi.isoformat()) in done:
                totals["skipped"] += 1
                continue

            def fetch(a, b, _q=query):
                totals["requests"] += 1
                result = provider.fetch(_q, a, b)
                result.ticker = ticker
                return result

            for result in iter_windows(lo, hi, fetch):
                totals["articles"] += len(result.articles)
                windows_for_ticker += 1
                if result.saturated:
                    totals["saturated"] += 1
                if result.status == "blocked":
                    totals["blocked"] += 1
                    blocked_for_ticker += 1
                elif result.status == "error":
                    totals["errors"] += 1
                    blocked_for_ticker += 1

                # THE RELEVANCE FILTER IS DETERMINISTIC AND RUNS HERE, not at
                # feature time. An article attributed to the wrong ticker is
                # noise on the wrong row, which is worse than a missing one
                # because it is indistinguishable from signal. `matched_by`
                # records WHICH alias fired, so the rule can be measured
                # against a hand-labelled sample and improved.
                keep = []
                for art in result.articles:
                    alias = match_ticker(art.title, aliases)
                    if alias:
                        keep.append((art, alias))
                result.articles = [a for a, _ in keep]
                totals["kept"] += len(keep)
                kept_for_ticker += len(keep)

                if not args.dry_run:
                    matched = keep[0][1] if keep else None
                    store_window(result, ticker,
                                 matched_by=f"alias:{matched}" if matched else None,
                                 engine=engine)

        rate = totals["requests"] / max(time.time() - began, 1e-9)
        print(f"  [{i:>3}/{len(tickers)}] {ticker:<18} "
              f"kept {kept_for_ticker:>5}  |  {totals['requests']:>6} reqs, "
              f"{rate:.1f}/s, {totals['saturated']} saturated, "
              f"{totals['blocked']} blocked")

        # A RATE, NOT A LIFETIME COUNT. The first version aborted at 20 blocked
        # windows total, which is the same mistake `scheduler` made with
        # `succeeded == 0`: a threshold written for the catastrophe that fires
        # on the ordinary case. Measured on the real run, ~2.7% of windows
        # return a transient 404 under sustained load, so a lifetime cap of 20
        # is reached at roughly request 800 of 12,000 EVERY TIME and the
        # backfill can never finish. It aborted at ticker 7 of 84.
        #
        # What actually needs catching is Google refusing us WHOLESALE, which
        # looks like a burst of consecutive failures rather than a low rate
        # spread over hours. Transient refusals are now retried inside
        # `provider.fetch`, so anything still recorded blocked has survived
        # three spaced attempts.
        recent.append(blocked_for_ticker / max(windows_for_ticker, 1))
        recent[:] = recent[-RECENT_TICKERS:]
        if len(recent) == RECENT_TICKERS and sum(recent) / len(recent) > MAX_BLOCK_RATE:
            print(f"\n  ABORTING: {100 * sum(recent) / len(recent):.0f}% of "
                  f"windows blocked across the last {RECENT_TICKERS} tickers, "
                  f"against a {100 * MAX_BLOCK_RATE:.0f}% ceiling.\n"
                  f"  That is wholesale refusal, not the ordinary transient "
                  f"rate. Wait an hour, then\n  re-run — completed windows are "
                  f"skipped, so nothing is lost.")
            break

    elapsed = time.time() - began
    print(f"\n{totals['requests']:,} requests in {elapsed / 60:.1f} min "
          f"({totals['requests'] / max(elapsed, 1e-9):.1f}/s)")
    print(f"{totals['articles']:,} articles seen, {totals['kept']:,} kept "
          f"({100 * totals['kept'] / max(totals['articles'], 1):.1f}% passed the "
          f"alias filter)")
    print(f"{totals['saturated']} saturated, {totals['blocked']} blocked, "
          f"{totals['errors']} errors, {totals['skipped']:,} skipped")
    print("\nRun --report before building any feature on this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
