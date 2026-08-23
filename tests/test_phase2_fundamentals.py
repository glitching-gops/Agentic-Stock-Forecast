"""
tests/test_phase2_fundamentals.py - Point-in-time valuation.

This module guards the only result in the project that has cleared its
pre-registered bar, and it guards the mechanism most able to fake one.

`pipeline/fundamentals.py` shipped with NO tests. That is the same shape of
omission the Phase 0 audit was about: F13 was an earnings figure attached to
the wrong date, and the whole difficulty here is attaching a figure to the
right one. An as-of join that is off by a single row hands the model two months
of information nobody had, produces no error, and reads as skill.

Three groups:

  1. THE AS-OF JOIN. A fundamental must be invisible before its effective date
     and visible on it. Every test here would pass against a lookahead-free
     implementation and fail against a forward-fill from `period_end`.
  2. YIELDS, NOT RATIOS. A loss-making company must not sort as the cheapest
     thing on the exchange.
  3. RESTATEMENTS. The vendor serves statements as restated. We cannot see
     revisions that predate us, but a figure must not be able to move while we
     are watching without leaving a record.

The database tests run `data.db.init_db()` against in-memory SQLite rather than
hand-copying the DDL, so a schema change that breaks the writer breaks these
too. A test carrying its own CREATE TABLE passes happily while the real schema
is wrong.
"""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def engine(monkeypatch):
    """In-memory SQLite carrying the REAL schema from data/db.py."""
    import data.db as db

    eng = create_engine("sqlite://")
    monkeypatch.setattr(db, "_ENGINE", eng, raising=False)
    db.init_db()
    return eng


def _panel(dates, tickers=("AAA.NS",), close=100.0):
    rows = []
    for t in tickers:
        for d in dates:
            rows.append({"date": d, "ticker": t, "close": close})
    return pd.DataFrame(rows)


def _fund(ticker="AAA.NS", period_end="2024-03-31", effective_date="2024-05-30",
          eps=10.0, bvps=50.0):
    return {"ticker": ticker, "period_end": period_end,
            "effective_date": effective_date, "eps": eps,
            "book_value_per_share": bvps}


# ── 1. the as-of join ───────────────────────────────────────────────────────

def test_fundamental_is_invisible_the_day_before_its_effective_date():
    """
    The single most important assertion in this file.

    A backward as-of join on `effective_date` yields NaN here. A merge on
    fiscal year, or a fill from `period_end`, yields 10.0 - two months early.
    """
    from pipeline.fundamentals import attach_fundamentals

    # The day before the EARLIEST figure becomes knowable: nothing is visible.
    out = attach_fundamentals(
        _panel(["2023-05-29"]),
        fundamentals=pd.DataFrame([
            _fund(period_end="2023-03-31", effective_date="2023-05-30", eps=5.0),
            _fund(period_end="2024-03-31", effective_date="2024-05-30", eps=10.0),
        ]),
    )
    assert pd.isna(out.loc[0, "earnings_yield"]), \
        "a figure was attached one day before it could be known"


def test_fundamental_is_visible_on_and_after_its_effective_date():
    from pipeline.fundamentals import attach_fundamentals

    panel = _panel(["2024-05-30", "2024-06-15"])
    out = attach_fundamentals(
        panel,
        fundamentals=pd.DataFrame([
            _fund(period_end="2023-03-31", effective_date="2023-05-30", eps=5.0),
            _fund(period_end="2024-03-31", effective_date="2024-05-30", eps=10.0),
        ]),
    )
    # eps 10 on price 100
    assert out["earnings_yield"].tolist() == pytest.approx([0.10, 0.10])


def test_the_previous_period_is_what_is_visible_before_the_next_one_files():
    """
    Not just "no lookahead" - the RIGHT figure. A join that simply dropped
    unknown periods would also pass the invisibility test while leaving the
    row empty when a perfectly good prior year was available.
    """
    from pipeline.fundamentals import attach_fundamentals

    out = attach_fundamentals(
        _panel(["2024-05-29"]),
        fundamentals=pd.DataFrame([
            _fund(period_end="2023-03-31", effective_date="2023-05-30", eps=5.0),
            _fund(period_end="2024-03-31", effective_date="2024-05-30", eps=10.0),
        ]),
    )
    assert out.loc[0, "earnings_yield"] == pytest.approx(0.05)
    assert out.loc[0, "fundamental_period"] == "2023-03-31"


def test_rows_before_the_first_effective_date_are_nan_not_backfilled():
    from pipeline.fundamentals import attach_fundamentals

    out = attach_fundamentals(
        _panel(["2018-01-02"]),
        fundamentals=pd.DataFrame([
            _fund(period_end="2023-03-31", effective_date="2023-05-30"),
            _fund(period_end="2024-03-31", effective_date="2024-05-30"),
        ]),
    )
    assert pd.isna(out.loc[0, "earnings_yield"]), \
        "the market did not know FY2024 earnings in 2018"


def test_one_ticker_cannot_borrow_another_tickers_fundamental():
    """`by="ticker"` is load-bearing; without it merge_asof matches across names."""
    from pipeline.fundamentals import attach_fundamentals

    panel = _panel(["2024-06-15"], tickers=("AAA.NS", "BBB.NS"))
    out = attach_fundamentals(panel, fundamentals=pd.DataFrame([
        _fund(ticker="AAA.NS", period_end="2023-03-31",
              effective_date="2023-05-30", eps=5.0),
        _fund(ticker="AAA.NS", period_end="2024-03-31",
              effective_date="2024-05-30", eps=10.0),
    ]))
    bbb = out[out["ticker"] == "BBB.NS"]
    assert bbb["earnings_yield"].isna().all(), "BBB borrowed AAA's earnings"


def test_lookahead_guard_raises_on_a_future_match(monkeypatch):
    """
    The guard is not decoration.

    merge_asof's own contract makes a future match impossible, so the only way
    to reach this branch is a mis-sort or a wrong `by` key - both of which
    produce a silently contaminated feature. Forced here by making merge_asof
    return exactly what a broken one would.
    """
    from pipeline import fundamentals as F

    real = pd.merge_asof

    def future_match(*args, **kwargs):
        merged = real(*args, **kwargs)
        merged["_eff"] = pd.Timestamp("2099-01-01")
        return merged

    monkeypatch.setattr(pd, "merge_asof", future_match)

    with pytest.raises(F.LookaheadRefused):
        F.attach_fundamentals(
            _panel(["2024-06-15"]),
            fundamentals=pd.DataFrame([
                _fund(period_end="2023-03-31", effective_date="2023-05-30"),
                _fund(period_end="2024-03-31", effective_date="2024-05-30"),
            ]),
        )


def test_filing_lag_matches_sebi_lodr_reg_33():
    """
    60 days for audited annual results. A shorter lag is a lookahead claim
    about how fast every company in the universe files.
    """
    from pipeline.fundamentals import ANNUAL_FILING_LAG_DAYS

    assert ANNUAL_FILING_LAG_DAYS == 60


def test_fetch_attaches_effective_date_sixty_days_after_period_end(monkeypatch):
    """The lag is applied at fetch, so nothing downstream has to remember it."""
    import sys
    import types

    from pipeline.fundamentals import ANNUAL_FILING_LAG_DAYS, fetch_fundamentals

    period = pd.Timestamp("2024-03-31")
    income = pd.DataFrame({period: [7.0]}, index=["Diluted EPS"])
    balance = pd.DataFrame({period: [1000.0, 100.0]},
                           index=["Stockholders Equity", "Ordinary Shares Number"])

    class _T:
        def __init__(self, _):
            self.income_stmt, self.balance_sheet = income, balance

    monkeypatch.setitem(sys.modules, "yfinance",
                        types.SimpleNamespace(Ticker=_T))

    frame = fetch_fundamentals(["AAA.NS"])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["period_end"] == "2024-03-31"
    expected = (period + pd.Timedelta(days=ANNUAL_FILING_LAG_DAYS)).date().isoformat()
    assert row["effective_date"] == expected == "2024-05-30"
    assert row["eps"] == 7.0
    assert row["book_value_per_share"] == pytest.approx(10.0)  # 1000 / 100


def test_a_ticker_with_one_period_is_left_unattached():
    """MIN_PERIODS: one fiscal year is not a step function, it is a constant."""
    from pipeline.fundamentals import MIN_PERIODS, attach_fundamentals

    assert MIN_PERIODS == 2
    out = attach_fundamentals(
        _panel(["2024-06-15"]),
        fundamentals=pd.DataFrame([_fund()]),
    )
    assert out["earnings_yield"].isna().all()


# ── 2. yields, not ratios ───────────────────────────────────────────────────

def test_a_loss_maker_does_not_sort_as_the_cheapest_stock():
    """
    Why the feature is E/P and not P/E.

    P/E is negative for a loss-making company, so ranking ascending on it puts
    the worst business on the exchange at the top of a value screen. E/P is
    monotone through zero: the loss-maker sorts last, which is correct.
    """
    from pipeline.fundamentals import attach_fundamentals

    panel = pd.DataFrame([
        {"date": "2024-06-15", "ticker": "LOSS.NS", "close": 100.0},
        {"date": "2024-06-15", "ticker": "CHEAP.NS", "close": 100.0},
        {"date": "2024-06-15", "ticker": "RICH.NS", "close": 100.0},
    ])
    fund = pd.DataFrame([
        _fund(ticker=t, period_end=p, effective_date=e, eps=eps)
        for t, eps in [("LOSS.NS", -20.0), ("CHEAP.NS", 20.0), ("RICH.NS", 2.0)]
        for p, e in [("2023-03-31", "2023-05-30"), ("2024-03-31", "2024-05-30")]
    ])
    out = attach_fundamentals(panel, fundamentals=fund).set_index("ticker")

    # Ascending P/E ranks the loss-maker first - the defect.
    assert out["pe_ratio"]["LOSS.NS"] < out["pe_ratio"]["CHEAP.NS"]
    # Descending earnings yield ranks it last - the fix.
    order = out["earnings_yield"].sort_values(ascending=False).index.tolist()
    assert order == ["CHEAP.NS", "RICH.NS", "LOSS.NS"]


def test_zero_and_negative_prices_do_not_produce_infinities():
    from pipeline.fundamentals import FUNDAMENTAL_COLS, attach_fundamentals

    panel = pd.DataFrame([
        {"date": "2024-06-15", "ticker": "AAA.NS", "close": 0.0},
        {"date": "2024-06-16", "ticker": "AAA.NS", "close": -1.0},
    ])
    out = attach_fundamentals(panel, fundamentals=pd.DataFrame([
        _fund(period_end="2023-03-31", effective_date="2023-05-30"),
        _fund(period_end="2024-03-31", effective_date="2024-05-30"),
    ]))
    for col in FUNDAMENTAL_COLS + ["pe_ratio", "pb_ratio"]:
        assert not np.isinf(pd.to_numeric(out[col], errors="coerce")).any(), col


def test_coverage_reports_the_fraction_actually_reached():
    from pipeline.fundamentals import attach_fundamentals, fundamental_coverage

    panel = _panel(["2018-01-02", "2024-06-15"])
    out = attach_fundamentals(panel, fundamentals=pd.DataFrame([
        _fund(period_end="2023-03-31", effective_date="2023-05-30"),
        _fund(period_end="2024-03-31", effective_date="2024-05-30"),
    ]))
    cov = fundamental_coverage(out)
    assert cov["rows"] == 2
    assert cov["with_fundamentals"] == 1
    assert cov["fraction"] == pytest.approx(0.5)


# ── 3. restatements ─────────────────────────────────────────────────────────

def test_material_change_ignores_float_noise_and_catches_real_moves():
    from pipeline.fundamentals import _material_change

    assert not _material_change(10.0, 10.0)
    assert not _material_change(10.0, 10.0 + 1e-12)
    assert not _material_change(None, None)
    assert not _material_change(float("nan"), None)
    assert _material_change(10.0, 11.0)
    assert _material_change(None, 10.0)
    assert _material_change(10.0, None)
    assert _material_change(0.0, 1e-6)


def test_a_restatement_is_recorded_before_it_is_overwritten(engine):
    """
    The point of the whole vintage mechanism.

    A plain upsert leaves the new value and no evidence the old one existed.
    """
    from pipeline.fundamentals import (load_fundamentals, load_revisions,
                                       store_fundamentals)

    first = pd.DataFrame([_fund(eps=10.0)])
    store_fundamentals(first, engine, observed_at="2025-01-01")

    restated = pd.DataFrame([_fund(eps=8.5)])
    counts = store_fundamentals(restated, engine, observed_at="2026-01-01")

    assert counts["revised"] == 1
    assert counts["revisions"] == 1

    # The current view carries the new figure ...
    fund = load_fundamentals(engine)
    assert fund.loc[0, "eps"] == pytest.approx(8.5)

    # ... and the old one survives as evidence.
    rev = load_revisions(engine)
    assert len(rev) == 1
    assert rev.loc[0, "field"] == "eps"
    assert rev.loc[0, "old_value"] == pytest.approx(10.0)
    assert rev.loc[0, "new_value"] == pytest.approx(8.5)
    assert rev.loc[0, "observed_at"] == "2026-01-01"
    assert rev.loc[0, "first_seen"] == "2025-01-01"


def test_an_unchanged_resync_records_nothing(engine):
    """Otherwise the log fills with noise and a real restatement hides in it."""
    from pipeline.fundamentals import load_revisions, store_fundamentals

    store_fundamentals(pd.DataFrame([_fund(eps=10.0)]), engine,
                       observed_at="2025-01-01")
    counts = store_fundamentals(pd.DataFrame([_fund(eps=10.0)]), engine,
                                observed_at="2025-01-08")

    assert counts["unchanged"] == 1
    assert counts["revised"] == 0
    assert load_revisions(engine).empty


def test_first_seen_survives_a_resync_but_fetched_at_moves(engine):
    """
    `first_seen` dates the EVIDENCE, not the last time we looked. If a resync
    reset it, every figure would appear to have been observed this week and the
    restatement log could not say how long a value had stood.
    """
    from pipeline.fundamentals import store_fundamentals

    store_fundamentals(pd.DataFrame([_fund(eps=10.0)]), engine,
                       observed_at="2025-01-01")
    store_fundamentals(pd.DataFrame([_fund(eps=10.0)]), engine,
                       observed_at="2026-06-01")

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT first_seen, fetched_at FROM fundamentals")).fetchone()
    assert row[0] == "2025-01-01"
    assert row[1] == "2026-06-01"


def test_both_fields_are_watched_independently(engine):
    from pipeline.fundamentals import load_revisions, store_fundamentals

    store_fundamentals(pd.DataFrame([_fund(eps=10.0, bvps=50.0)]), engine,
                       observed_at="2025-01-01")
    counts = store_fundamentals(pd.DataFrame([_fund(eps=9.0, bvps=44.0)]),
                                engine, observed_at="2026-01-01")

    assert counts["revisions"] == 2, "one revision per field that moved"
    assert counts["revised"] == 1, "but one period was affected"
    assert set(load_revisions(engine)["field"]) == {"eps", "book_value_per_share"}


def test_a_new_period_is_new_not_a_revision(engine):
    """A fresh fiscal year arriving is not the vendor changing its mind."""
    from pipeline.fundamentals import load_revisions, store_fundamentals

    store_fundamentals(pd.DataFrame([_fund(period_end="2024-03-31")]), engine,
                       observed_at="2025-01-01")
    counts = store_fundamentals(
        pd.DataFrame([_fund(period_end="2025-03-31",
                            effective_date="2025-05-30")]),
        engine, observed_at="2026-01-01")

    assert counts["new"] == 1
    assert counts["revised"] == 0
    assert load_revisions(engine).empty


def test_replaying_the_same_sync_twice_does_not_duplicate_a_revision(engine):
    """The revisions PK includes observed_at, so a re-run is idempotent."""
    from pipeline.fundamentals import load_revisions, store_fundamentals

    store_fundamentals(pd.DataFrame([_fund(eps=10.0)]), engine,
                       observed_at="2025-01-01")
    store_fundamentals(pd.DataFrame([_fund(eps=8.0)]), engine,
                       observed_at="2026-01-01")
    store_fundamentals(pd.DataFrame([_fund(eps=8.0)]), engine,
                       observed_at="2026-01-01")

    assert len(load_revisions(engine)) == 1


def test_restatement_summary_reports_size_not_just_count(engine):
    from pipeline.fundamentals import restatement_summary, store_fundamentals

    store_fundamentals(pd.DataFrame([_fund(eps=10.0, bvps=50.0)]), engine,
                       observed_at="2025-01-01")
    store_fundamentals(pd.DataFrame([_fund(eps=5.0, bvps=50.0)]), engine,
                       observed_at="2026-01-01")

    s = restatement_summary(engine)
    assert s["revisions"] == 1
    assert s["tickers_affected"] == 1
    assert s["periods_tracked"] == 1
    assert s["max_abs_rel_change"] == pytest.approx(0.5)  # 5 moved off a base of 10
    assert s["by_field"] == {"eps": 1}


def test_restatement_summary_is_honest_when_nothing_is_tracked(engine):
    """
    Zero revisions against zero tracked periods is not evidence of no bias,
    and the summary has to say so rather than print a reassuring 0.
    """
    from pipeline.fundamentals import restatement_summary

    s = restatement_summary(engine)
    assert s["revisions"] == 0
    assert s["periods_tracked"] == 0
    assert "note" in s and "YET" in s["note"].upper()


# ── 4. the partial-fetch refusal ────────────────────────────────────────────

def test_sync_refuses_a_partial_fetch_and_writes_nothing(engine, monkeypatch):
    """
    Valuation is standardised WITHIN each date. A cross-section where only the
    tickers whose HTTP call succeeded carry a figure is scored against a mean
    and standard deviation taken over that arbitrary subset - and nothing
    downstream can tell it apart from a real one.
    """
    from pipeline import fundamentals as F

    monkeypatch.setattr(F, "fetch_fundamentals",
                        lambda tickers: pd.DataFrame([_fund(ticker="AAA.NS")]))

    with pytest.raises(F.FetchCoverageRefused):
        F.sync_fundamentals(["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"],
                            engine=engine)

    assert F.load_fundamentals(engine).empty, \
        "a refused sync must leave the table untouched"


def test_sync_writes_when_coverage_clears_the_floor(engine, monkeypatch):
    from pipeline import fundamentals as F

    monkeypatch.setattr(F, "fetch_fundamentals", lambda tickers: pd.DataFrame(
        [_fund(ticker=t) for t in ["AAA.NS", "BBB.NS", "CCC.NS"]]))

    counts = F.sync_fundamentals(["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"],
                                 engine=engine, observed_at="2026-01-01")

    assert counts["tickers_returned"] == 3
    assert counts["fetch_coverage"] == pytest.approx(0.75)
    assert counts["new"] == 3
    assert len(F.load_fundamentals(engine)) == 3


def test_the_coverage_floor_is_a_real_threshold(engine, monkeypatch):
    """A floor of 0.0 would make the guard decoration."""
    from pipeline.fundamentals import MIN_FETCH_COVERAGE

    assert 0.0 < MIN_FETCH_COVERAGE <= 1.0
