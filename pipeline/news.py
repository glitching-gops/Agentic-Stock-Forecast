"""
pipeline/news.py — the dated news archive.

WHAT CHANGED, AND WHY IT MATTERS MORE THAN THE CODE
---------------------------------------------------
Since Phase 0 this project has recorded, in two module docstrings and in the
plan, that news is "unbacktestable by construction because Google News RSS
serves no archive". Measured 2026-09-03, that is **wrong**. The RSS search
endpoint honours Google's own date operators:

    q = "Reliance Industries after:2024-01-01 before:2024-01-31"
        -> 100 entries, every one published inside the window
    q = "Reliance Industries after:2016-09-01 before:2016-09-30"
        ->  77 entries, back to the month the `macro` table starts

Correct publication dates, ~0.65 s a request, no key. The one input P3 needed
and believed it could not have was available the whole time.

WHAT WE HAD BEEN STORING WAS NOT WHAT WE THOUGHT
------------------------------------------------
`pipeline/sentiment.py` stamps `datetime.today()` on every headline — the FETCH
date, never the article's — and takes `entries[:5]` of **100 available**, ranked
by RELEVANCE rather than recency. Measured on the live feed, RELIANCE's top five
spanned 0.1 to 151 days old and TCS's reached 246. So the archive recorded a
five-month-old story as today's news, and because the ranking is by relevance
the same old story reappeared day after day.

That is a PERSISTENT PER-TICKER FEATURE, which is the landmine in CLAUDE.md §7
that scored `pooled_xgb` at a mean rebalance t of +0.77 from two random
constants carrying no information at all. It would not have failed loudly. It
would have produced a plausible number.

THE ONE INVARIANT EVERYTHING RESTS ON
--------------------------------------
Google returns at most `RESULT_CAP` results per query and ranks them by
relevance — a relevance computed TODAY, informed by everything that has happened
since. So for a saturated window, *which* articles you receive has been selected
with hindsight. That is a look-ahead channel into the SAMPLE rather than into
any single value, which makes it the subtle kind: no value is wrong, the
selection is.

`iter_windows` therefore SPLITS any window that saturates, recursively, until
none do. Below the cap the result set is exhaustive and the ranking cannot
matter, because we take all of it. This is P3's `_history_ending_at`: for a
zero-shot series model the entire purged-fold guarantee collapses onto one
slice, and here the entire point-in-time guarantee collapses onto this.

`news_coverage` records `saturated` for every window so the invariant is
AUDITABLE after the fact rather than merely asserted in a docstring.

"NOT OBSERVED" IS NOT "NOTHING HAPPENED"
-----------------------------------------
On 2026-08-20 and 2026-08-28 the live job stored "No news available today." for
all 95 tickers. Ninety-five companies do not go quiet on the same day — the
fetch was blocked and the block was stored as data. Every attempt writes a
`news_coverage` row, so a downstream feature can tell an empty window (a
measurement) from an absent one (no measurement). Same rule that already makes
`get_aggregate_sentiment` return None rather than 0.0.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable, Iterator, Protocol

from sqlalchemy import text

from data.db import get_engine

logger = logging.getLogger(__name__)

#: Google News RSS returns at most this many items per query. A window that
#: reaches it has been relevance-ranked and must be split — see the module
#: docstring. This is not a tuning knob.
RESULT_CAP = 100

#: Stop splitting here. A single day that still saturates is reported rather
#: than split further, because there is nothing below it to split into: the
#: operators have day granularity.
MIN_WINDOW_DAYS = 1

#: Politeness. Google publishes no quota for this endpoint; the backfill makes
#: five figures of requests, so it paces itself rather than finding the limit.
REQUEST_DELAY_SECONDS = 0.4


@dataclass(frozen=True)
class Article:
    """One article, identified by its content rather than by who found it."""

    published_at: str          # ISO date of the ARTICLE, never of the fetch
    title: str
    url: str
    source: str
    provider: str

    @property
    def article_id(self) -> str:
        return article_identity(self.url, self.title)


@dataclass
class WindowResult:
    """What one (ticker, window) attempt produced, including its failures."""

    ticker: str
    start: date
    end: date
    provider: str
    status: str                       # ok | blocked | error
    articles: list[Article] = field(default_factory=list)
    detail: str = ""

    @property
    def saturated(self) -> bool:
        return len(self.articles) >= RESULT_CAP


# ── Identity ──────────────────────────────────────────────────────────────────

_TRACKING = re.compile(r"[?&](utm_[^=]+|fbclid|gclid|ref|source)=[^&]*", re.I)


def article_identity(url: str, title: str) -> str:
    """
    A stable id for an article.

    Built from the URL where there is one and from the title otherwise. Google
    News rewrites publisher links through its own redirector and appends
    tracking parameters that vary between fetches of the SAME article, so a raw
    URL hash would record one story twice and let a re-fetch inflate every
    count. Tracking parameters are stripped and the host is lowercased before
    hashing; the title is the fallback, normalised the same way.
    """
    cleaned = _TRACKING.sub("", (url or "").strip())
    cleaned = cleaned.split("#", 1)[0].rstrip("?&").lower()
    basis = cleaned or re.sub(r"\s+", " ", (title or "").strip().lower())
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


# ── Window splitting: the invariant ───────────────────────────────────────────

def iter_windows(
    start: date,
    end: date,
    fetch: Callable[[date, date], WindowResult],
    min_days: int = MIN_WINDOW_DAYS,
) -> Iterator[WindowResult]:
    """
    Walks [start, end] and yields results, splitting any window that saturates.

    THIS IS THE POINT-IN-TIME GUARANTEE, not an optimisation. A window that
    returns `RESULT_CAP` items has had those items chosen by a relevance model
    that has seen the future relative to the window. Splitting until the count
    falls below the cap makes the retrieved set exhaustive, at which point the
    ranking cannot influence what we keep because we keep everything.

    A saturated window at `min_days` is yielded anyway, flagged, and counted —
    hiding it would be worse than recording an imperfect day, and the operators
    have no sub-day granularity to split into.

    Failures are yielded too. A window that errored is not a window with no
    news, and `news_coverage` has to be able to say which it was.
    """
    if start > end:
        return

    stack: list[tuple[date, date]] = [(start, end)]
    while stack:
        lo, hi = stack.pop()
        result = fetch(lo, hi)

        span = (hi - lo).days + 1
        if result.status == "ok" and result.saturated and span > min_days:
            mid = lo + timedelta(days=span // 2)
            # Pushed so the earlier half is processed first: a resumable
            # backfill that dies mid-run should have made contiguous progress
            # from the start of the period rather than leaving holes.
            stack.append((mid, hi))
            stack.append((lo, mid - timedelta(days=1)))
            continue

        yield result


def month_starts(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Calendar months spanning [start, end], the split's starting granularity."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield max(cur, start), min(nxt - timedelta(days=1), end)
        cur = nxt


# ── Providers ─────────────────────────────────────────────────────────────────

class NewsProvider(Protocol):
    name: str

    def supports_history(self) -> bool: ...

    def fetch(self, query: str, start: date, end: date) -> WindowResult: ...


class GoogleNewsRSS:
    """
    The primary provider, and the only one measured to serve a real archive.

    The keyed vendors were researched and are NOT wired, for a reason that is
    about coverage rather than cost: Polygon.io does not carry NSE equities at
    all, and Finnhub's free tier is US-only for news (international is a paid
    plan). Alpha Vantage's 25 calls/day cannot cover 84 tickers once, never mind
    daily. A registry costs nothing and makes adding one later a class rather
    than a rewrite — but adding one today would buy nothing this does not have.
    """

    name = "google_news_rss"
    BASE = "https://news.google.com/rss/search"

    def __init__(self, delay: float = REQUEST_DELAY_SECONDS, hl: str = "en-IN",
                 gl: str = "IN", ceid: str = "IN:en"):
        self.delay = delay
        self.hl, self.gl, self.ceid = hl, gl, ceid

    def supports_history(self) -> bool:
        return True

    def _url(self, query: str, start: date, end: date) -> str:
        # `after:` is EXCLUSIVE of its own date and `before:` is exclusive of
        # its own, so the bounds are widened by a day each to make the window
        # inclusive. Getting this wrong loses one session per window, which at
        # daily granularity would lose everything and at monthly granularity
        # would be nearly invisible.
        q = (f"{query} after:{start - timedelta(days=1):%Y-%m-%d} "
             f"before:{end + timedelta(days=1):%Y-%m-%d}")
        return (f"{self.BASE}?q={urllib.parse.quote(q)}"
                f"&hl={self.hl}&gl={self.gl}&ceid={self.ceid}")

    def fetch(self, query: str, start: date, end: date) -> WindowResult:
        import feedparser

        out = WindowResult(ticker="", start=start, end=end, provider=self.name,
                           status="ok")
        try:
            feed = feedparser.parse(self._url(query, start, end))
        except Exception as exc:                                    # noqa: BLE001
            out.status, out.detail = "error", f"{type(exc).__name__}: {exc}"[:300]
            return out

        status = getattr(feed, "status", 200)
        if status and int(status) >= 400:
            out.status, out.detail = "blocked", f"HTTP {status}"
            return out

        for entry in feed.entries:
            published = _entry_date(entry)
            if published is None:
                # NO DATE, NO ROW. An article we cannot date is exactly the
                # thing this module exists to stop storing; dropping it costs
                # one headline, keeping it reintroduces the defect.
                continue
            if not (start <= published <= end):
                # Google occasionally returns a near-miss. Keeping it would let
                # an article land in a window whose coverage row does not
                # describe it, so the count and the contents would disagree.
                continue
            out.articles.append(Article(
                published_at=published.isoformat(),
                title=(entry.get("title") or "").strip(),
                url=(entry.get("link") or "").strip(),
                source=((entry.get("source") or {}) or {}).get("title", "") or "",
                provider=self.name,
            ))

        if self.delay:
            time.sleep(self.delay)
        return out


def _entry_date(entry) -> date | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc).date()
    except (TypeError, ValueError):
        return None


# ── Relevance ─────────────────────────────────────────────────────────────────

#: CORPORATE FORM ONLY. "Ltd." carries no identifying information, so
#: "Infosys Ltd." and "Infosys" are one company.
#:
#: `industries`, `enterprises`, `holdings` and `power` are DELIBERATELY ABSENT,
#: and the first version of this pattern stripped them. In an Indian universe
#: those words are the discriminating half of the name, not decoration:
#:
#:     Reliance Industries Ltd.  vs  Reliance Power Ltd.
#:     Adani Enterprises Ltd.    vs  Adani Power Ltd.  vs  Adani Ports
#:
#: Several of those pairs are BOTH in this universe. Stripping the tail
#: collapses them to "Reliance" and "Adani", and every article about one would
#: then be attributed to the other — noise on the wrong row, which is worse
#: than a missing article because it is indistinguishable from signal.
_SUFFIXES = re.compile(
    r"\b(ltd|limited|inc|corp|corporation|plc|co|company)\b\.?", re.I)


def core_name(company: str) -> str:
    """The company name with its corporate form removed, and nothing else."""
    return re.sub(r"\s+", " ", _SUFFIXES.sub(" ", company)).strip(" .,&")


def search_query(ticker: str, company: str) -> str:
    """
    What to ask the provider for.

    NO "NSE" TOKEN, and that is a measured correction rather than a
    simplification. `pipeline/sentiment.py` queried f"{company} NSE" and this
    module copied it unexamined; Google treats the extra word as a required
    term, so it does not disambiguate an Indian listing, it filters to articles
    that happen to contain the string "NSE". Measured over January 2024:

        ticker          query                        with NSE   without
        RELIANCE.NS     Reliance Industries                26        98
        INFY.NS         Infosys                             9        94
        ADANIPOWER.NS   Adani Power                         9        42
        UNIONBANK.NS    Union Bank of India                 5        42
        MUTHOOTFIN.NS   Muthoot Finance                     0        14
        ABB.NS          ABB India                           0        31

    It costs 60-100% of the articles, and for two of six names it costs ALL of
    them. So the live archive has been this sparse the whole time as well, for
    the same reason.

    The disambiguation that word was reaching for is done properly elsewhere and
    for free: the feed URL is already pinned to `gl=IN&ceid=IN:en`, and
    `match_ticker` requires the full core name, so "Reliance Steel & Aluminum"
    cannot match "Reliance Industries".
    """
    return core_name(company) or company.strip()


def company_name_is_unresolved(ticker: str, company: str) -> bool:
    """
    True when `company` is just the ticker symbol rather than a real name.

    `data.tickers.get_company` falls back to the bare symbol when the metadata
    table has no row — a fresh checkout, an un-synced database, a ticker added
    since the last `refresh_metadata()`. Nothing raises.

    The consequence is silent and total. The query becomes "RELIANCE" instead of
    "Reliance Industries", the only alias is the all-caps symbol, and
    `match_ticker` then requires that exact capitalisation in a headline. A
    five-hour backfill would complete, report success, and leave a near-empty
    archive for every affected name — with the holes falling wherever the
    metadata happened to be missing rather than at random. Callers check this
    BEFORE spending the requests.
    """
    stripped = (company or "").strip()
    return not stripped or stripped.upper() == ticker.split(".")[0].upper()


def company_aliases(ticker: str, company: str) -> list[str]:
    """
    Names an article might use for this company, longest first.

    Deliberately conservative. The live probe returned "Lumino Industries IPO
    listing today" under a *Reliance Industries* query — mis-attributing that to
    RELIANCE.NS is noise assigned to the wrong row. The residual is measured
    against a hand-labelled sample before any model is paid to adjudicate it.
    """
    symbol = ticker.split(".")[0]
    names = {company.strip(), core_name(company)}
    names.discard("")
    # The bare symbol is cheap to include and rarely fires in prose. It is
    # restricted to alphabetic symbols of three characters or more so that a
    # short or numeric one cannot match arbitrary text.
    if len(symbol) >= 3 and symbol.isalpha():
        names.add(symbol)
    return sorted(names, key=len, reverse=True)


def match_ticker(title: str, aliases: Iterable[str]) -> str | None:
    """
    Returns the alias that matched, or None. The alias is stored, not a bool.

    AN ALL-CAPS ALIAS IS A TICKER SYMBOL AND MATCHES CASE-SENSITIVELY. Everything
    else is a company name and matches case-insensitively.

    That distinction is not tidiness. Matching the symbol "RELIANCE"
    case-insensitively also matches the ordinary word "Reliance", and therefore
    every headline about Reliance Power, Reliance Capital or Reliance
    Infrastructure — separately listed companies, some of them in this very
    universe. A test caught it: "Reliance Power wins solar tender" was being
    attributed to RELIANCE.NS. Headlines that mean the instrument write it in
    caps ("RELIANCE beats Q3 estimates"), so case is exactly the signal that
    separates the two.
    """
    def normalise(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9 ]+", " ", s)

    padded = f" {normalise(title or '')} "
    lowered = padded.lower()

    for alias in aliases:
        needle = normalise(alias).strip()
        if not needle:
            continue
        if alias.isupper():
            if f" {needle} " in padded:          # case-SENSITIVE: a symbol
                return alias
        elif f" {needle.lower()} " in lowered:   # case-insensitive: a name
            return alias
    return None


# ── Storage ───────────────────────────────────────────────────────────────────

def store_window(result: WindowResult, ticker: str, matched_by: str | None = None,
                 engine=None) -> dict:
    """
    Persists one window's articles, mentions and its coverage row.

    THE COVERAGE ROW IS WRITTEN WHATEVER HAPPENS, including on a blocked or
    errored fetch. That is the whole point: a window with no row is a window
    nobody looked at, and a feature reading zero articles has to be able to
    tell that from a window that was searched and genuinely held none.
    """
    engine = engine or get_engine()
    now = datetime.now(timezone.utc).isoformat()
    stored = mentions = 0

    with engine.connect() as conn:
        for art in result.articles:
            conn.execute(text("""
                INSERT INTO news_articles
                    (article_id, published_at, title, url, source, provider,
                     first_seen)
                VALUES (:aid, :pub, :title, :url, :src, :prov, :seen)
                ON CONFLICT (article_id) DO NOTHING
            """), {"aid": art.article_id, "pub": art.published_at,
                   "title": art.title[:1000], "url": art.url[:2000],
                   "src": art.source[:200], "prov": art.provider, "seen": now})
            stored += 1

            conn.execute(text("""
                INSERT INTO news_mentions
                    (article_id, ticker, matched_by, first_seen)
                VALUES (:aid, :t, :m, :seen)
                ON CONFLICT (article_id, ticker) DO NOTHING
            """), {"aid": art.article_id, "t": ticker,
                   "m": matched_by or "query", "seen": now})
            mentions += 1

        conn.execute(text("""
            INSERT INTO news_coverage
                (ticker, window_start, window_end, provider, status,
                 n_articles, saturated, attempted_at, detail)
            VALUES (:t, :ws, :we, :prov, :st, :n, :sat, :at, :detail)
            ON CONFLICT (ticker, window_start, window_end, provider)
            DO UPDATE SET status = EXCLUDED.status,
                          n_articles = EXCLUDED.n_articles,
                          saturated = EXCLUDED.saturated,
                          attempted_at = EXCLUDED.attempted_at,
                          detail = EXCLUDED.detail
        """), {"t": ticker, "ws": result.start.isoformat(),
               "we": result.end.isoformat(), "prov": result.provider,
               "st": result.status, "n": len(result.articles),
               "sat": 1 if result.saturated else 0, "at": now,
               "detail": result.detail[:500]})
        conn.commit()

    return {"articles": stored, "mentions": mentions,
            "status": result.status, "saturated": result.saturated}


def coverage_report(engine=None) -> dict:
    """
    What the archive actually holds — read this before believing any feature.

    A backfill with holes concentrated in the early panel would manufacture
    exactly the early-fold artifact this project has now seen three times
    (valuation +3.32, LoRA +2.37, pooled_xgb +2.42). The saturated count is the
    other half: any non-zero value means some windows were relevance-ranked and
    the exhaustive-selection invariant is dented, by a knowable amount.
    """
    import pandas as pd

    engine = engine or get_engine()
    with engine.connect() as conn:
        articles = pd.read_sql(text("""
            SELECT substr(published_at, 1, 4) AS year,
                   COUNT(*) AS n_articles
            FROM news_articles GROUP BY 1 ORDER BY 1
        """), conn)
        cov = pd.read_sql(text("""
            SELECT status, COUNT(*) AS n_windows,
                   SUM(saturated) AS n_saturated
            FROM news_coverage GROUP BY status
        """), conn)
        totals = pd.read_sql(text("""
            SELECT COUNT(*) AS n_articles,
                   MIN(published_at) AS first, MAX(published_at) AS last
            FROM news_articles
        """), conn)

    return {"by_year": articles.to_dict("records"),
            "by_status": cov.to_dict("records"),
            "totals": totals.to_dict("records")[0] if not totals.empty else {}}


# ── The daily path ────────────────────────────────────────────────────────────

def fetch_recent(tickers: list[str], days: int = 3, provider: NewsProvider | None = None,
                 engine=None) -> dict:
    """
    The daily ingest: a short trailing window per ticker, dated properly.

    A TRAILING WINDOW RATHER THAN "TODAY", because publication and indexing are
    not simultaneous — an article published yesterday evening appears today, and
    a same-day-only query drops it permanently. The article id makes the overlap
    free: re-seeing a story is a no-op, not a duplicate.
    """
    from data.tickers import get_company

    provider = provider or GoogleNewsRSS()
    engine = engine or get_engine()
    end = date.today()
    start = end - timedelta(days=days)

    totals = {"tickers": 0, "articles": 0, "blocked": 0, "errors": 0}
    for ticker in tickers:
        company = get_company(ticker)
        aliases = company_aliases(ticker, company)
        query = search_query(ticker, company)
        for result in iter_windows(
                start, end,
                lambda lo, hi, q=query: _tagged(provider.fetch(q, lo, hi), ticker)):
            kept = [a for a in result.articles if match_ticker(a.title, aliases)]
            dropped = len(result.articles) - len(kept)
            result.articles = kept
            report = store_window(result, ticker, engine=engine)
            totals["articles"] += len(kept)
            if result.status == "blocked":
                totals["blocked"] += 1
            elif result.status == "error":
                totals["errors"] += 1
            if dropped:
                logger.debug("[news] %s: dropped %d off-topic", ticker, dropped)
        totals["tickers"] += 1

    logger.info("[news] %s", totals)
    return totals


def _tagged(result: WindowResult, ticker: str) -> WindowResult:
    result.ticker = ticker
    return result
