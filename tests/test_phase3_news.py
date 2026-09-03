"""
tests/test_phase3_news.py — the dated news archive.

The guards here protect three things, in descending order of how quietly they
would fail:

1. THE SATURATION SPLIT. Google returns at most 100 results and ranks them by a
   relevance computed today. A window that hits the cap has had its articles
   SELECTED with hindsight, which is a look-ahead channel into the sample
   rather than into any single value — so no value is wrong and nothing raises.
   `iter_windows` splits until nothing saturates. This is P3's equivalent of
   `series._history_ending_at`, where the whole purged-fold guarantee collapses
   onto one slice.

2. PUBLICATION DATE, NOT FETCH DATE. The module this replaces stamped
   `datetime.today()` on articles up to 246 days old and reappearing daily —
   a persistent per-ticker feature, which is measured in CLAUDE.md §7 as worth
   a spurious t of +0.77 from pure noise.

3. "NOT OBSERVED" DISTINCT FROM "NOTHING HAPPENED". Two days in the live
   archive hold the "No news available today." placeholder for all 95 tickers,
   which is a blocked fetch stored as data.

Every test below drives the real code against the real schema from
`data/db.py::init_db`. A fixture carrying its own CREATE TABLE passes happily
while the production schema is wrong — that is how `forecast_outcomes` shipped
missing a column its writer needed.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from pipeline.news import (
    RESULT_CAP,
    Article,
    WindowResult,
    article_identity,
    company_aliases,
    coverage_report,
    iter_windows,
    match_ticker,
    month_starts,
    store_window,
)


@pytest.fixture
def engine(monkeypatch):
    """In-memory SQLite carrying the REAL schema from data/db.py."""
    import data.db as db

    eng = create_engine("sqlite://")
    monkeypatch.setattr(db, "_ENGINE", eng, raising=False)
    db.init_db()
    return eng


def _article(day: str, title: str, url: str = "") -> Article:
    return Article(published_at=day, title=title, url=url or f"http://x/{title}",
                   source="Test Wire", provider="stub")


def _result(start: date, end: date, n: int, status: str = "ok") -> WindowResult:
    return WindowResult(
        ticker="AAA.NS", start=start, end=end, provider="stub", status=status,
        articles=[_article(start.isoformat(), f"item {i}") for i in range(n)],
    )


# ── 1. The saturation split ───────────────────────────────────────────────────

def test_a_saturated_window_is_split_until_it_is_not():
    """
    THE INVARIANT THE WHOLE FEATURE RESTS ON.

    A window returning exactly the cap has been relevance-ranked by a model that
    has seen everything since, so which articles came back is chosen with
    hindsight. Below the cap the set is exhaustive and the ranking cannot
    matter, because we keep all of it.
    """
    calls: list[tuple[date, date]] = []

    def fetch(lo, hi):
        calls.append((lo, hi))
        # Saturates for any window wider than 4 days.
        n = RESULT_CAP if (hi - lo).days + 1 > 4 else 3
        return _result(lo, hi, n)

    out = list(iter_windows(date(2024, 1, 1), date(2024, 1, 31), fetch))

    assert len(calls) > 1, "a saturated window must have been split"
    assert all(not r.saturated for r in out), (
        "every YIELDED window must be below the cap; a saturated one means "
        "articles were selected by relevance, with hindsight"
    )
    # And the split must still cover the whole period exactly once.
    days = sorted(d for r in out
                  for d in pd.date_range(r.start, r.end).date)
    assert days == sorted(set(days)), "windows overlap — articles double-counted"
    assert days[0] == date(2024, 1, 1) and days[-1] == date(2024, 1, 31)
    assert len(days) == 31, "the split lost days from the period"


def test_an_unsaturated_window_is_never_split():
    """Splitting costs a request each time; it must happen only when forced."""
    calls = []

    def fetch(lo, hi):
        calls.append((lo, hi))
        return _result(lo, hi, RESULT_CAP - 1)

    out = list(iter_windows(date(2024, 1, 1), date(2024, 1, 31), fetch))
    assert len(calls) == 1 and len(out) == 1


def test_a_single_day_that_still_saturates_is_reported_not_dropped():
    """
    The operators have day granularity, so there is nothing below a day to
    split into. Such a window is yielded and FLAGGED rather than silently
    discarded or silently kept as though it were exhaustive.
    """
    out = list(iter_windows(
        date(2024, 1, 1), date(2024, 1, 2),
        lambda lo, hi: _result(lo, hi, RESULT_CAP)))

    assert out, "a stubbornly saturated day must still be reported"
    assert all(r.saturated for r in out)
    assert all((r.end - r.start).days == 0 for r in out), "should have split to days"


def test_a_failed_window_is_not_split_and_is_still_reported():
    """
    A blocked fetch returns zero articles, which is not saturation and must not
    be read as a quiet period either. Splitting it would multiply one outage
    into many requests against a host already refusing us.
    """
    calls = []

    def fetch(lo, hi):
        calls.append((lo, hi))
        return _result(lo, hi, 0, status="blocked")

    out = list(iter_windows(date(2024, 1, 1), date(2024, 1, 31), fetch))
    assert len(calls) == 1
    assert out[0].status == "blocked"


def test_a_partial_read_is_not_treated_as_a_saturated_one():
    """
    THE CASE THE TEST ABOVE CANNOT REACH, and a surviving mutant is what found
    it. `GoogleNewsRSS` returns early on failure, so its failed windows always
    hold zero articles and `saturated` is False anyway — which made the
    `status == "ok"` half of the split condition look like a duplicate of the
    saturation check and let it be deleted with the suite green.

    It is not a duplicate, and `iter_windows` takes an arbitrary provider: a
    paginated source that dies partway can return a full page AND an error.
    Splitting that would read a TRANSPORT failure as evidence the window is too
    wide, and would keep splitting for as long as the source stayed broken.
    Only a window we read successfully and completely may be split.
    """
    calls = []

    def fetch(lo, hi):
        calls.append((lo, hi))
        result = _result(lo, hi, RESULT_CAP, status="error")
        result.detail = "connection reset after page 1"
        return result

    out = list(iter_windows(date(2024, 1, 1), date(2024, 1, 31), fetch))
    assert len(calls) == 1, (
        "a window that FAILED must not be split, however many articles it "
        "happened to return before failing"
    )
    assert out[0].status == "error"


# ── 2. Publication date, not fetch date ───────────────────────────────────────

def test_an_article_outside_the_requested_window_is_dropped():
    """
    Keeping a near-miss would let an article land in a window whose coverage row
    does not describe it, so the stored count and the stored contents would
    disagree — and the count is what a feature reads.
    """
    from pipeline.news import GoogleNewsRSS

    provider = GoogleNewsRSS(delay=0)

    class _Feed:
        status = 200
        entries = [
            {"title": "inside", "link": "http://a", "published_parsed": (2024, 1, 15, 0, 0, 0, 0, 0, 0)},
            {"title": "outside", "link": "http://b", "published_parsed": (2023, 6, 1, 0, 0, 0, 0, 0, 0)},
            {"title": "undated", "link": "http://c"},
        ]

    import sys, types
    real = sys.modules.get("feedparser")
    sys.modules["feedparser"] = types.SimpleNamespace(parse=lambda url: _Feed())
    try:
        out = provider.fetch("X", date(2024, 1, 1), date(2024, 1, 31))
    finally:
        if real is not None:
            sys.modules["feedparser"] = real
        else:
            sys.modules.pop("feedparser", None)

    titles = [a.title for a in out.articles]
    assert titles == ["inside"], (
        "an out-of-window article, and one we cannot date at all, must both be "
        "dropped — storing an undated article is the defect this module exists "
        "to stop"
    )
    assert out.articles[0].published_at == "2024-01-15"


def test_the_window_bounds_are_inclusive():
    """
    Google's `after:` and `before:` are EXCLUSIVE of their own dates. Passing
    the raw bounds loses a session at each end — invisible on a monthly window
    and total at daily granularity.
    """
    from pipeline.news import GoogleNewsRSS

    url = GoogleNewsRSS(delay=0)._url("ACME", date(2024, 1, 10), date(2024, 1, 20))
    assert "after%3A2024-01-09" in url, "after: must be widened by a day"
    assert "before%3A2024-01-21" in url, "before: must be widened by a day"


def test_one_article_seen_twice_is_one_row():
    """
    Google rewrites publisher links and appends tracking parameters that vary
    between fetches of the SAME article. A raw URL hash would record one story
    twice and let a daily overlap inflate every count.
    """
    a = article_identity("https://Site.com/story?utm_source=news&gclid=1", "T")
    b = article_identity("https://site.com/story", "T")
    assert a == b

    assert article_identity("", "Reliance beats estimates") == \
           article_identity("", "  reliance   beats estimates ")
    assert article_identity("https://site.com/one", "T") != \
           article_identity("https://site.com/two", "T")


# ── 3. Coverage: not observed vs nothing happened ─────────────────────────────

def test_every_attempt_writes_a_coverage_row_including_the_failures(engine):
    """
    THE 2026-08-20 DEFECT. All 95 tickers stored "No news available today." on
    one day; 95 companies do not go quiet together — the fetch was blocked, and
    it was stored as data. A window that was never attempted and one that was
    searched and held nothing must be distinguishable.
    """
    store_window(_result(date(2024, 1, 1), date(2024, 1, 31), 0, status="blocked"),
                 "AAA.NS", engine=engine)
    store_window(_result(date(2024, 2, 1), date(2024, 2, 29), 0, status="ok"),
                 "AAA.NS", engine=engine)

    rows = pd.read_sql(text(
        "SELECT window_start, status, n_articles FROM news_coverage "
        "ORDER BY window_start"), engine)
    assert len(rows) == 2
    assert rows.iloc[0]["status"] == "blocked"
    assert rows.iloc[1]["status"] == "ok"
    # March was never attempted, so it has no row at all — absence is the
    # third state, and it must not be confused with either of the other two.
    assert "2024-03-01" not in set(rows["window_start"])


def test_the_saturated_flag_is_persisted_so_the_invariant_is_auditable(engine):
    """
    A docstring asserting the invariant is not the invariant. The flag lets a
    coverage report say how many windows were relevance-ranked, after the fact.
    """
    store_window(_result(date(2024, 1, 1), date(2024, 1, 1), RESULT_CAP),
                 "AAA.NS", engine=engine)
    store_window(_result(date(2024, 1, 2), date(2024, 1, 2), 5),
                 "AAA.NS", engine=engine)

    got = pd.read_sql(text(
        "SELECT window_start, saturated FROM news_coverage ORDER BY window_start"),
        engine)
    assert list(got["saturated"]) == [1, 0]
    assert coverage_report(engine)["by_status"][0]["n_saturated"] == 1


def test_re_ingesting_the_same_window_does_not_duplicate_articles(engine):
    """
    The daily path uses a TRAILING window, so consecutive runs overlap by
    design — publication and indexing are not simultaneous. Re-seeing a story
    must be a no-op, or the overlap inflates every count it feeds.
    """
    result = _result(date(2024, 1, 1), date(2024, 1, 3), 4)
    store_window(result, "AAA.NS", engine=engine)
    store_window(result, "AAA.NS", engine=engine)

    n = pd.read_sql(text("SELECT COUNT(*) c FROM news_articles"), engine).iloc[0]["c"]
    m = pd.read_sql(text("SELECT COUNT(*) c FROM news_mentions"), engine).iloc[0]["c"]
    assert n == 4 and m == 4

    # One coverage row too, updated rather than duplicated.
    w = pd.read_sql(text("SELECT COUNT(*) c FROM news_coverage"), engine).iloc[0]["c"]
    assert w == 1


def test_re_ingest_preserves_provenance_rather_than_rewriting_it(engine):
    """
    COUNTING ROWS CANNOT CATCH THIS, and a surviving mutant is what proved it:
    swapping DO NOTHING for DO UPDATE leaves the row count identical, so the
    test above passed against the defect it was written to prevent.

    What must not move is the PROVENANCE. `first_seen` says when we first knew
    of an article and `matched_by` says which rule attributed it to this ticker
    — the second is what a precision measurement against a hand-labelled sample
    reads. Letting a later pass silently rewrite either one destroys the record
    that measurement depends on, for the same reason `fundamental_revisions`
    exists and `forecast_outcomes` never updates a row.
    """
    result = _result(date(2024, 1, 1), date(2024, 1, 1), 1)
    store_window(result, "AAA.NS", matched_by="alias:Acme", engine=engine)

    before = pd.read_sql(text(
        "SELECT a.first_seen AS a_seen, m.first_seen AS m_seen, m.matched_by "
        "FROM news_articles a JOIN news_mentions m USING (article_id)"), engine)

    # A later pass re-attributes the same article by a different rule.
    store_window(result, "AAA.NS", matched_by="llm", engine=engine)

    after = pd.read_sql(text(
        "SELECT a.first_seen AS a_seen, m.first_seen AS m_seen, m.matched_by "
        "FROM news_articles a JOIN news_mentions m USING (article_id)"), engine)

    assert len(after) == 1
    assert after.iloc[0]["matched_by"] == "alias:Acme", (
        "a re-ingest overwrote which rule attributed this article"
    )
    assert after.iloc[0]["m_seen"] == before.iloc[0]["m_seen"]
    assert after.iloc[0]["a_seen"] == before.iloc[0]["a_seen"], (
        "first_seen moved, so 'when did we first know this' is now unanswerable"
    )


def test_one_article_mentioning_two_tickers_is_one_article_two_mentions(engine):
    """
    Identity is the article's CONTENT, not the query that surfaced it. Storing
    it per-ticker would let two copies of one story drift apart the moment one
    is re-scored.
    """
    result = _result(date(2024, 1, 1), date(2024, 1, 1), 2)
    store_window(result, "AAA.NS", engine=engine)
    store_window(WindowResult(ticker="BBB.NS", start=result.start, end=result.end,
                              provider="stub", status="ok",
                              articles=list(result.articles)),
                 "BBB.NS", engine=engine)

    assert pd.read_sql(text("SELECT COUNT(*) c FROM news_articles"),
                       engine).iloc[0]["c"] == 2
    assert pd.read_sql(text("SELECT COUNT(*) c FROM news_mentions"),
                       engine).iloc[0]["c"] == 4


# ── 4. Relevance ──────────────────────────────────────────────────────────────

def test_the_alias_match_rejects_the_article_that_prompted_it():
    """
    Measured on the live feed: a *Reliance Industries* query returned "Lumino
    Industries IPO listing today". Attributing that to RELIANCE.NS is noise
    assigned to the wrong row, which is worse than a missing article because it
    is indistinguishable from signal.
    """
    aliases = company_aliases("RELIANCE.NS", "Reliance Industries Ltd.")

    assert match_ticker("Reliance Industries shares slump nearly 4%", aliases)
    assert match_ticker("RELIANCE beats Q3 estimates", aliases)
    assert match_ticker("Reliance Industries Ltd. announces buyback", aliases)
    assert match_ticker("Lumino Industries IPO listing today", aliases) is None
    assert match_ticker("Tata Motors gains on JLR outlook", aliases) is None


def test_the_matching_alias_is_recorded_rather_than_a_bare_boolean(engine):
    """
    The relevance rule is going to be measured against a hand-labelled sample.
    A measurement you cannot attribute back to the rule that produced it cannot
    improve that rule.
    """
    store_window(_result(date(2024, 1, 1), date(2024, 1, 1), 1), "AAA.NS",
                 matched_by="alias:Acme", engine=engine)
    got = pd.read_sql(text("SELECT matched_by FROM news_mentions"), engine)
    assert got.iloc[0]["matched_by"] == "alias:Acme"


def test_corporate_suffixes_do_not_decide_a_match():
    """"Ltd." is not identifying information; "Infosys" and "Infosys Ltd." are one company."""
    aliases = company_aliases("INFY.NS", "Infosys Ltd.")
    assert match_ticker("Infosys president resigns", aliases)
    assert match_ticker("Infosys Ltd. wins cloud deal", aliases)


def test_industries_and_power_are_part_of_the_name_not_corporate_form():
    """
    THE FIRST VERSION OF THIS STRIPPED THEM, and in an Indian universe that is
    the discriminating half of the name rather than decoration. Several of
    these pairs are BOTH in the frozen universe:

        Reliance Industries Ltd.  vs  Reliance Power Ltd.
        Adani Enterprises Ltd.    vs  Adani Power Ltd.

    Collapsing them to "Reliance" and "Adani" attributes every article about
    one to the other — noise on the wrong row, which is worse than a missing
    article because it is indistinguishable from signal.
    """
    ril = company_aliases("RELIANCE.NS", "Reliance Industries Ltd.")
    assert "Reliance" not in ril, "the bare group name is not a company"
    assert match_ticker("Reliance Power wins solar tender", ril) is None
    assert match_ticker("Reliance Industries shares rally", ril)

    adani = company_aliases("ADANIENT.NS", "Adani Enterprises Ltd.")
    assert match_ticker("Adani Power posts record profit", adani) is None
    assert match_ticker("Adani Enterprises raises funds", adani)


def test_an_unresolved_company_name_is_detectable_before_the_requests():
    """
    `data.tickers.get_company` falls back to the BARE SYMBOL when the metadata
    table has no row, and nothing raises. The damage is silent and total: the
    query becomes "RELIANCE" instead of "Reliance Industries", the only alias is
    the all-caps symbol, and `match_ticker` then demands that capitalisation in
    a headline.

    Measured while smoke-testing the backfill against a database with no
    metadata: RELIANCE.NS and MUTHOOTFIN.NS each kept ZERO articles across two
    months and the tool reported success. A five-hour run would have produced a
    near-empty archive whose holes fall wherever metadata happened to be
    missing — the worst possible shape, because it is not random.
    """
    from pipeline.news import company_name_is_unresolved

    assert company_name_is_unresolved("RELIANCE.NS", "RELIANCE")
    assert company_name_is_unresolved("RELIANCE.NS", "reliance")
    assert company_name_is_unresolved("ABB.NS", "ABB")
    assert company_name_is_unresolved("ABB.NS", "")

    assert not company_name_is_unresolved("RELIANCE.NS", "Reliance Industries Ltd.")
    assert not company_name_is_unresolved("ABB.NS", "ABB India Ltd.")


def test_a_ticker_symbol_matches_on_case_because_the_word_does_not():
    """
    The alias list carries the bare symbol, and matching it case-insensitively
    re-opens the exact hole the test above closes: "RELIANCE" lowercased is the
    ordinary word "Reliance", which appears in every Reliance Power and
    Reliance Capital headline.

    Headlines that mean the instrument write it in caps. So an ALL-CAPS alias
    matches case-sensitively and a company name does not.
    """
    ril = company_aliases("RELIANCE.NS", "Reliance Industries Ltd.")
    assert "RELIANCE" in ril

    assert match_ticker("RELIANCE beats Q3 estimates", ril) == "RELIANCE"
    assert match_ticker("Reliance Power wins solar tender", ril) is None
    assert match_ticker("Reliance Capital resolution approved", ril) is None

    # A company NAME still matches whatever case the publisher used.
    infy = company_aliases("INFY.NS", "Infosys Ltd.")
    assert match_ticker("INFOSYS WINS CLOUD DEAL", infy)
    assert match_ticker("infosys wins cloud deal", infy)


def test_the_search_query_carries_no_extra_required_term():
    """
    `pipeline/sentiment.py` queried f"{company} NSE" and this module copied it
    unexamined. Google treats the extra word as a REQUIRED term, so it does not
    disambiguate an Indian listing — it filters to articles containing the
    literal string "NSE". Measured over January 2024:

        RELIANCE.NS     26 articles with it, 98 without
        INFY.NS          9 with,             94 without
        MUTHOOTFIN.NS    0 with,             14 without
        ABB.NS           0 with,             31 without

    Two of six names lost EVERY article. Disambiguation is done properly and
    for free by `gl=IN&ceid=IN:en` on the feed URL and by `match_ticker`
    requiring the full core name.
    """
    from pipeline.news import search_query

    q = search_query("MUTHOOTFIN.NS", "Muthoot Finance Ltd.")
    assert "NSE" not in q, (
        "an extra required term costs 60-100% of the articles and, for some "
        "names, all of them"
    )
    assert q == "Muthoot Finance"
    assert search_query("RELIANCE.NS", "Reliance Industries Ltd.") == \
        "Reliance Industries"

    # A name that is ONLY a corporate form must not become the empty string —
    # an empty query returns the whole news firehose.
    assert search_query("X.NS", "Ltd.") == "Ltd."


# ── 5. Window arithmetic ──────────────────────────────────────────────────────

def test_months_tile_the_period_exactly():
    """A gap loses articles silently; an overlap double-counts them."""
    spans = list(month_starts(date(2023, 11, 15), date(2024, 2, 10)))
    assert spans[0] == (date(2023, 11, 15), date(2023, 11, 30))
    assert spans[-1] == (date(2024, 2, 1), date(2024, 2, 10))

    days = [d for lo, hi in spans for d in pd.date_range(lo, hi).date]
    assert len(days) == len(set(days)) == (date(2024, 2, 10) - date(2023, 11, 15)).days + 1


def test_an_inverted_range_yields_nothing_rather_than_looping():
    assert list(iter_windows(date(2024, 2, 1), date(2024, 1, 1),
                             lambda lo, hi: _result(lo, hi, 1))) == []


# ── 6. The graph node ─────────────────────────────────────────────────────────

def test_the_node_asks_whether_we_LOOKED_not_whether_we_FOUND(engine, monkeypatch):
    """
    The node decides whether to fetch by asking `news_coverage` — did we ATTEMPT
    this ticker today? — and not `news_articles` — do we HAVE anything today?

    The difference is the whole 2026-08-20 defect. A genuinely quiet day and a
    blocked fetch both produce zero articles, so keying on articles re-fetches
    the quiet ones forever and, worse, records "we looked and there was
    nothing" as an ordinary state indistinguishable from "we could not look".
    A window that was searched and held nothing is a MEASUREMENT and must not
    be repeated; one that was never searched is not.
    """
    import agents.external_data_agent as node
    import pandas as pd

    today = date.today()
    calls: list[list[str]] = []
    monkeypatch.setattr(node, "fetch_recent",
                        lambda tickers, **k: calls.append(list(tickers)))
    monkeypatch.setattr(node, "fetch_macro", lambda: None)
    monkeypatch.setattr(node, "get_engine", lambda: engine)
    monkeypatch.setattr(node, "get_aggregate_sentiment", lambda t: None)

    # We searched today and found NOTHING. That is a measurement.
    store_window(WindowResult(ticker="AAA.NS", start=today, end=today,
                              provider="stub", status="ok", articles=[]),
                 "AAA.NS", engine=engine)

    node.external_data_node({"ticker": "AAA.NS"})
    assert calls == [], (
        "a window we already searched must not be searched again just because "
        "it held no articles"
    )

    # A ticker nobody has looked at today MUST be fetched.
    node.external_data_node({"ticker": "BBB.NS"})
    assert calls == [["BBB.NS"]]
