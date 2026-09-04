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
Every window that completes writes a `news_coverage` row, and the resume key is
MONTHS COVERED rather than windows matched — the granularity changed once
already, and a window-shaped key would have re-fetched ~1,700 finished windows
against a quota that refuses after ~600. `--refetch` overrides it.

THE REQUEST COUNT IS THE BINDING CONSTRAINT, NOT THE SPEED
-----------------------------------------------------------
Measured 2026-09-04: after roughly 600 requests the endpoint returns HTTP 503
to EVERYTHING, including a bare query with no date operators. Pacing does not
buy much — four tickers ran clean at 0.8 req/s and the wall arrived anyway — so
the only way to finish is to need fewer requests.

So this hands the WHOLE range to `iter_windows` and lets it binary-search on
density, instead of walking 121 fixed monthly windows per ticker. Requests then
follow the ARTICLES rather than the calendar. Simulated against measured
densities, per ticker over ten years:

    1 article/month    ->  3 requests   (monthly grid: 121)
    3 articles/month   ->  7 requests
    7 articles/month   -> 31 requests
    25 articles/month  -> 63 requests

A 2x saving on the densest name and 40x on the sparsest, which is the right
shape: cost lands where the articles are.

A 429/503 STOPS THE RUN IMMEDIATELY and is never retried. Retrying a rate limit
is three times the load on a host that has just said stop, and that is exactly
how one block became 164 blocked windows and 82 minutes of grinding. A 404 is
different — it is specific to one (query, window) pair — and stays retryable.

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
    months_spanned,
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


def _previously_blocked(engine, provider: str) -> set[tuple[str, str, str]]:
    """
    Windows that failed last time, and are therefore being RETRIED.

    THESE MUST NOT COUNT TOWARD THE ABORT RATE, and leaving them in is what
    made the first resume run impossible. A resume attempts only the windows a
    previous run failed on — an adversarially selected sample, since anything
    that worked is skipped. Measured: the resume tried 13 windows, 9 of which
    were already-known failures, and the guard read 81% and aborted at ticker 3.

    The rate exists to detect Google refusing us WHOLESALE, and a known-bad
    window failing again is no evidence of that. 2025-06 in particular returns
    404 for every company ever tried, so it will fail on every run forever and
    would poison the statistic permanently.
    """
    from data.db import is_missing_relation

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT ticker, window_start, window_end FROM news_coverage "
                "WHERE provider = :p AND status <> 'ok'"),
                {"p": provider}).fetchall()
    except Exception as exc:                                     # noqa: BLE001
        if not is_missing_relation(exc):
            raise
        return set()
    return {(r[0], r[1], r[2]) for r in rows}


def _completed_months(engine, provider: str) -> set[tuple[str, str]]:
    """
    (ticker, 'YYYY-MM') pairs fully covered by a window recorded ok.

    The resume key, and it is months rather than windows on purpose. The
    backfill's granularity changed from fixed monthly windows to a recursive
    split, so a window-shaped key would match nothing and re-fetch ~1,700
    completed windows against a quota that refuses after ~600.
    """
    from data.db import is_missing_relation
    from pipeline.news import months_spanned as _months

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT ticker, window_start, window_end FROM news_coverage "
                "WHERE provider = :p AND status = 'ok'"),
                {"p": provider}).fetchall()
    except Exception as exc:                                     # noqa: BLE001
        if not is_missing_relation(exc):
            raise
        return set()

    out: set[tuple[str, str]] = set()
    for ticker, lo, hi in rows:
        a = datetime.strptime(str(lo)[:10], "%Y-%m-%d").date()
        b = datetime.strptime(str(hi)[:10], "%Y-%m-%d").date()
        for m in _months(a, b):
            out.add((ticker, m))
    return out


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
    done_months = set() if args.refetch else _completed_months(engine, provider.name)
    retrying = _previously_blocked(engine, provider.name)
    print(f"{len(tickers)} tickers, {start} .. {end}, "
          f"{len(done):,} windows already complete"
          f"{'  [DRY RUN]' if args.dry_run else ''}")

    began = time.time()
    totals = {"requests": 0, "articles": 0, "kept": 0, "saturated": 0,
              "blocked": 0, "errors": 0, "skipped": 0,
              "retried": 0, "recovered": 0, "rate_limited": 0}
    recent: list[float] = []

    for i, ticker in enumerate(tickers, 1):
        company = get_company(ticker)
        aliases = company_aliases(ticker, company)
        query = search_query(ticker, company)
        kept_for_ticker = 0
        blocked_for_ticker = 0
        windows_for_ticker = 0

        # ONE RECURSIVE WALK OVER THE WHOLE RANGE, not 121 fixed monthly
        # windows. This is the structural fix for the rate limit, and it is
        # worth more than any amount of pacing: the endpoint refuses after
        # roughly 600 requests, so the only way to finish is to need fewer.
        #
        # `iter_windows` already binary-searches on density — it splits only
        # what saturates — so the request count follows the ARTICLES rather
        # than the calendar. A decade of a sparse name is one or two requests
        # instead of 121; a dense name still splits down to weeks where the
        # articles actually are. Estimated over the measured ~46,000 articles:
        # ~460 leaves plus internal nodes, so ~1,400 requests for the whole
        # universe against 10,164 at fixed monthly granularity.
        rate_limited = False

        def fetch(a, b, _q=query):
            totals["requests"] += 1
            result = provider.fetch(_q, a, b)
            result.ticker = ticker
            return result

        def already_done(a, b) -> bool:
            """
            True when every month this window spans is already recorded ok.

            Keyed on MONTHS rather than on the window itself so the work done
            by earlier monthly runs still counts — the granularity changed, the
            coverage did not, and re-fetching 1,700 completed windows to satisfy
            a key format would be the worst possible use of a scarce quota.
            """
            months = months_spanned(a, b)
            return bool(months) and all((ticker, m) in done_months for m in months)

        for lo, hi in [(start, end)]:
            is_retry = False
            for result in iter_windows(lo, hi, fetch, skip=already_done):
                totals["articles"] += len(result.articles)
                if result.saturated:
                    totals["saturated"] += 1
                if result.status == "rate_limited":
                    rate_limited = True
                    totals["rate_limited"] += 1
                elif result.status == "blocked":
                    totals["blocked"] += 1
                elif result.status == "error":
                    totals["errors"] += 1
                if not is_retry:
                    windows_for_ticker += 1
                    if result.status != "ok":
                        blocked_for_ticker += 1
                else:
                    totals["retried"] += 1
                    if result.status == "ok":
                        totals["recovered"] += 1

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
                    # PER ARTICLE, not per window. Stamping the first kept
                    # article's alias onto every other one made `matched_by`
                    # wrong for any window where two different aliases fired,
                    # which is exactly the attribution a precision measurement
                    # needs.
                    store_window(
                        result, ticker, engine=engine,
                        matched_by_article={a.article_id: f"alias:{alias}"
                                            for a, alias in keep})

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
        # Only tickers with FRESH windows contribute a rate. On a pure resume
        # run there are none, so the guard stays silent rather than reading a
        # sample made entirely of known failures.
        if windows_for_ticker:
            recent.append(blocked_for_ticker / windows_for_ticker)
            recent[:] = recent[-RECENT_TICKERS:]
        # A 503 IS NOT A DATA PROBLEM AND HAS NO RATE TO AVERAGE. The host has
        # said stop; every further request extends the block rather than
        # sampling it. Stopping on the FIRST one is what keeps a pause short.
        if rate_limited:
            print(
                f"\n  STOPPING: Google returned a rate-limit status (429/503)"
                f" after {totals['requests']} requests this session.\n"
                f"  The endpoint refuses EVERYTHING for a while once it does "
                f"— including a bare\n  query carrying no date operators — so "
                f"this is a quota, not a data problem.\n"
                f"  Wait ~1 hour and re-run; completed windows are skipped, so "
                f"nothing is lost.")
            break

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
    if totals["retried"]:
        print(f"{totals['retried']} previously-failed windows retried, "
              f"{totals['recovered']} recovered "
              f"({totals['retried'] - totals['recovered']} still failing — "
              f"those are index holes, see --report)")
    print("\nRun --report before building any feature on this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
