"""
The NSE trading calendar, and the three places it keeps holidays out (2026-09-21).

Yahoo answers an NSE holiday with a bar for every ticker - open = high = low =
close = the previous close, volume 0 - and `pipeline/fetch.py` stored it, so the
panel carried 2026-01-15, 05-01, 05-28, 06-26 and 09-14 as sessions. These
tests pin:

  * the committed calendar says what NSE's own records say about the dates
    that matter (the five phantoms, the Muhurat and budget-day sessions);
  * ingestion drops a holiday bar and keeps a real one;
  * the signals write guard excuses a lost row on a holiday and nothing else;
  * `load_panel` never returns a non-session - and the check that says so
    fails when a holiday is injected, so it is not vacuous;
  * the gate sees a market-wide flat bar;
  * the calendar refuses a year it does not cover instead of guessing.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data import nse_calendar
from data.nse_calendar import CalendarNotCurrent

PHANTOMS = ("2026-01-15", "2026-05-01", "2026-05-28", "2026-06-26", "2026-09-14")


@pytest.fixture
def calendar():
    return nse_calendar.load()


def non_sessions_in(dates, calendar) -> list[str]:
    """THE CHECK: every date in `dates` NSE did not trade."""
    dates = pd.Series(list(dates)).astype(str).str[:10]
    return sorted(set(dates[~calendar.session_mask(dates)]))


# ── the committed file ────────────────────────────────────────────────────────


def test_the_calendar_covers_every_year_the_panel_spans(calendar):
    assert set(range(2016, 2027)) <= set(calendar.years)


def test_the_calendar_covers_the_current_year(calendar):
    """Fails on 1 January if nobody has run tools/sync_nse_calendar.py. That is
    the reminder: the live merge in production covers the gap, a stale file
    does not."""
    assert date.today().year in calendar.years, (
        "run `python tools/sync_nse_calendar.py` and commit data/nse_calendar.json")


@pytest.mark.parametrize("day", PHANTOMS)
def test_every_known_phantom_is_not_a_session(calendar, day):
    assert calendar.is_session(day) is False


@pytest.mark.parametrize("day", [
    "2025-02-01",   # Saturday: Union Budget, NSE held a full session
    "2019-10-27",   # Sunday: Muhurat trading
    "2020-11-14",   # Saturday: Muhurat trading
    "2023-11-12",   # Sunday: Muhurat - NSE's LIST calls it a holiday
    "2024-01-20",   # Saturday: special live session
])
def test_special_sessions_are_sessions(calendar, day):
    assert calendar.is_session(day) is True


def test_an_ordinary_week(calendar):
    # Monday 2025-03-17 .. Sunday 2025-03-23: five sessions, two weekend days.
    days = pd.date_range("2025-03-17", "2025-03-23")
    assert calendar.session_mask(days).tolist() == [True] * 5 + [False] * 2


def test_an_ordinary_weekend_is_not_a_session(calendar):
    """Through is_session as well as session_mask: the two are separate code
    paths, and a mutant that made every weekend a session survived a suite that
    only asked the mask."""
    assert calendar.is_session("2025-03-22") is False     # Saturday
    assert calendar.is_session("2025-03-23") is False     # Sunday


def test_2025_03_18_is_a_session_although_yahoo_has_no_prices_for_it(calendar):
    """The one real session Yahoo serves as a flat zero-volume bar. It must stay
    a session: dropping it would lose a day NSE traded."""
    assert calendar.is_session("2025-03-18") is True


def test_an_uncovered_year_raises_rather_than_guessing(calendar):
    with pytest.raises(CalendarNotCurrent, match="sync_nse_calendar"):
        calendar.is_session("2031-06-02")
    with pytest.raises(CalendarNotCurrent):
        calendar.session_mask(["2026-06-01", "2031-06-02"])


# ── NSE's published list ──────────────────────────────────────────────────────


def _payload(*rows):
    return {"CM": [{"tradingDate": d, "description": desc} for d, desc in rows]}


def test_the_holiday_list_parses_and_refuses_an_empty_answer():
    got = nse_calendar.parse_holiday_master(
        _payload(("15-Jan-2026", "Election"), ("26-Jan-2026", "Republic Day")), 2026)
    assert got == {"2026-01-15": "Election", "2026-01-26": "Republic Day"}
    with pytest.raises(ValueError):
        nse_calendar.parse_holiday_master({"CM": []}, 2026)


def test_the_live_list_adds_a_future_closure_and_nothing_already_verified(calendar):
    # An ad hoc closure after verified_through is merged; a date the archive
    # has already ruled on is not re-ruled by the list.
    future = "2026-12-28"                              # a Monday
    past = "2026-03-02"                                # a Monday NSE traded
    live = _payload((pd.Timestamp(future).strftime("%d-%b-%Y"), "ad hoc"),
                    (pd.Timestamp(past).strftime("%d-%b-%Y"), "wrong"))
    merged, note = nse_calendar.live_calendar(
        base=calendar, today=date(2026, 9, 21), fetch=lambda y: live)
    assert merged.is_session(future) is False
    assert merged.is_session(past) is True
    assert "1 closure(s)" in note


def test_the_live_list_falls_back_to_the_file_when_nse_does_not_answer(calendar):
    def down(year):
        raise ConnectionError("no route")
    merged, note = nse_calendar.live_calendar(base=calendar, today=date(2026, 9, 21),
                                              fetch=down)
    assert merged == calendar and "unavailable" in note


# ── ingestion ─────────────────────────────────────────────────────────────────


def _bars(dates, ticker="AAA.NS"):
    n = len(dates)
    return pd.DataFrame({"date": list(dates), "ticker": ticker,
                         "open": 100.0, "high": 101.0, "low": 99.0,
                         "close": np.linspace(100, 101, n), "adj_close": 100.0,
                         "volume": 1000.0})


def test_ingestion_drops_a_holiday_bar_and_keeps_every_session(calendar):
    from pipeline.fetch import sessions_only

    days = ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"]
    kept, dropped = sessions_only({"AAA.NS": _bars(days)}, calendar)
    assert dropped == ["2026-01-15"]
    assert kept["AAA.NS"]["date"].tolist() == ["2026-01-13", "2026-01-14", "2026-01-16"]


def test_ingestion_refuses_a_year_the_calendar_does_not_cover(calendar):
    from pipeline.fetch import sessions_only

    with pytest.raises(CalendarNotCurrent):
        sessions_only({"AAA.NS": _bars(["2031-06-02"])}, calendar)


def test_the_fetch_step_uses_the_calendar(monkeypatch, calendar):
    """End to end through fetch_and_store: a holiday bar never reaches the
    table, and the refresh removes one already stored."""
    import sqlalchemy as sa

    import pipeline.fetch as fetch

    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(sa.text(
            "CREATE TABLE IF NOT EXISTS ohlcv (date TEXT, ticker TEXT, open REAL, "
            "high REAL, low REAL, close REAL, adj_close REAL, volume REAL)"))
        _bars(["2026-01-14", "2026-01-15"]).to_sql("ohlcv", conn, if_exists="append",
                                                   index=False)
    days = ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"]
    monkeypatch.setattr(fetch, "get_engine", lambda: engine)
    monkeypatch.setattr(fetch, "_download_batch", lambda batch: {"AAA.NS": _bars(days)})
    fetch.fetch_and_store(tickers=["AAA.NS"])
    with engine.connect() as conn:
        stored = [r[0] for r in conn.execute(sa.text(
            "SELECT date FROM ohlcv WHERE ticker = 'AAA.NS' ORDER BY date"))]
    assert stored == ["2026-01-13", "2026-01-14", "2026-01-16"]


# ── the signals write guard ───────────────────────────────────────────────────


def _signals_table(dates, labelled=True):
    import sqlalchemy as sa

    engine = sa.create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(sa.text(
            "CREATE TABLE signals (ticker TEXT, date TEXT, close REAL, "
            "target_return REAL, target_excess_return REAL)"))
        for d in dates:
            conn.execute(sa.text("INSERT INTO signals VALUES ('T.NS', :d, 100.0, "
                                 ":v, :v)"), {"d": d, "v": 0.01 if labelled else None})
        conn.commit()
    return engine


def _frame(dates):
    return pd.DataFrame({"ticker": "T.NS", "date": list(dates), "close": 100.0,
                         "target_return": 0.01, "target_excess_return": 0.01})


def test_the_label_guard_excuses_a_row_lost_on_a_holiday():
    """The first clean recompute after the fix: the phantom's labelled row
    disappears, and that must not read as label loss."""
    from pipeline.signals import _upsert_signals

    with_phantom = ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"]
    engine = _signals_table(with_phantom)
    with engine.connect() as conn:
        written = _upsert_signals(conn, "T.NS", _frame(
            [d for d in with_phantom if d != "2026-01-15"]))
        conn.commit()
    assert written == 3


def test_the_label_guard_still_refuses_a_row_lost_on_a_session():
    """Excusing holidays must not excuse anything else: a real session's label
    lost - a vendor gap, a feature turning NaN - still refuses the write."""
    from pipeline.signals import LabelLossRefused, _upsert_signals

    days = ["2026-01-13", "2026-01-14", "2026-01-16"]
    engine = _signals_table(days)
    with engine.connect() as conn:
        with pytest.raises(LabelLossRefused):
            _upsert_signals(conn, "T.NS", _frame(["2026-01-13", "2026-01-16"]))


def test_the_job_level_label_count_ignores_phantom_rows(monkeypatch):
    """
    The guard BOTH jobs read, not the write-boundary one.

    Measured 2026-09-22: the daily job aborted twice on the first run after the
    phantom fix, because removing Yahoo's holiday rows lowered this total
    (222,514 -> 222,378) and it counted them as labels. 49 tickers are still
    refused by the benchmark outage, so without this the same abort returns the
    day their index does.
    """
    import sqlalchemy as sa

    import pipeline.signals as signals

    engine = sa.create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(sa.text(
            "CREATE TABLE signals (ticker TEXT, date TEXT, target_return REAL)"))
        for d in ("2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"):
            conn.execute(sa.text("INSERT INTO signals VALUES ('T.NS', :d, 0.01)"),
                         {"d": d})
        conn.commit()
    monkeypatch.setattr(signals, "get_engine", lambda: engine)

    # Four labelled rows on file, one of them on a day NSE did not trade.
    assert signals.count_labelled_rows() == 3
    assert signals.count_labelled_rows("T.NS") == 3
    assert signals.count_labelled_rows("OTHER.NS") == 0


def test_signals_refuse_to_run_without_an_html_parser(monkeypatch):
    """
    No HTML parser means `earnings_surprise` — a pooled FACTOR — is written as
    a constant 0.0 for every ticker, silently, because the fetch is caught.
    Measured on the first locked CI run (2026-09-22).
    """
    import importlib.util

    import pipeline.signals as signals

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a, **k: None if n in signals.HTML_PARSERS
                        else real(n, *a, **k))
    with pytest.raises(signals.EarningsParserMissing, match="requirements.txt"):
        signals.require_earnings_parser()
    with pytest.raises(signals.EarningsParserMissing):
        signals.compute_and_store(tickers=["T.NS"])

    monkeypatch.undo()
    signals.require_earnings_parser()          # the real environment has one


def test_a_phantom_row_cannot_stand_in_for_a_lost_session_label():
    """Both sides of the comparison count sessions only. If the INCOMING side
    counted every row, a frame that loses a real session's label but carries a
    phantom's would balance the books and be written."""
    from pipeline.signals import LabelLossRefused, _upsert_signals

    engine = _signals_table(["2026-01-13", "2026-01-14", "2026-01-16"])
    swapped = _frame(["2026-01-13", "2026-01-15", "2026-01-16"])   # 01-14 lost, 01-15 phantom
    with engine.connect() as conn:
        with pytest.raises(LabelLossRefused):
            _upsert_signals(conn, "T.NS", swapped)


# ── the panel ─────────────────────────────────────────────────────────────────


def _panel_db(dates):
    import sqlalchemy as sa

    from pipeline.signals import FEATURE_COLS

    engine = sa.create_engine("sqlite://")
    rows = []
    for d in dates:
        for i, t in enumerate(("A.NS", "B.NS")):
            rows.append({"date": d, "ticker": t, "close": 100.0 + i,
                         **{c: 0.5 for c in FEATURE_COLS},
                         "target_return": 0.01, "target_excess_return": 0.0,
                         "benchmark_close": 1.0, "benchmark_ticker": "^NSEI"})
    pd.DataFrame(rows).to_sql("signals", engine, index=False)
    from pipeline.panel import MACRO_COLS

    pd.DataFrame([{"date": d, **{c: 1.0 for c in MACRO_COLS}} for d in dates]) \
        .to_sql("macro", engine, index=False)
    return engine


def test_no_non_trading_session_enters_the_panel(calendar):
    from pipeline.panel import load_panel

    days = ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"]
    panel = load_panel(engine=_panel_db(days))
    assert non_sessions_in(panel["date"], calendar) == []
    assert sorted(panel["date"].unique()) == ["2026-01-13", "2026-01-14", "2026-01-16"]


def test_the_session_check_fails_on_an_injected_holiday(calendar):
    """The check the test above relies on, shown to bite: the same frame with a
    holiday injected is reported, so an empty report means something."""
    days = ["2026-01-13", "2026-01-14", "2026-01-16"]
    assert non_sessions_in(days, calendar) == []
    assert non_sessions_in(days + ["2026-01-15"], calendar) == ["2026-01-15"]
    assert non_sessions_in(days + ["2026-01-17"], calendar) == ["2026-01-17"]  # a Saturday


def test_the_stored_clean_panel_has_no_non_trading_session(calendar):
    """The real data, when this checkout has it. The pre-fix panel_cache.parquet
    is kept for reproducing old results and DOES carry the phantoms."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "panel_cache_clean.parquet"
    if not path.exists():
        pytest.skip("panel_cache_clean.parquet not present (gitignored)")
    dates = pd.read_parquet(path, columns=["date"])["date"].astype(str).unique()
    assert non_sessions_in(dates, calendar) == []


# ── the gate ──────────────────────────────────────────────────────────────────


def test_the_gate_warns_on_a_market_wide_flat_bar(monkeypatch):
    import sqlalchemy as sa

    from pipeline import validation

    today = pd.Timestamp.today().normalize()
    recent = [(today - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in (3, 2)]
    rows = []
    for i in range(12):
        rows.append({"date": recent[0], "ticker": f"T{i}.NS", "open": 10.0 + i,
                     "high": 11.0 + i, "low": 9.0 + i, "close": 10.5 + i,
                     "adj_close": 10.5 + i, "volume": 500.0})
        rows.append({"date": recent[1], "ticker": f"T{i}.NS", "open": 10.5 + i,
                     "high": 10.5 + i, "low": 10.5 + i, "close": 10.5 + i,
                     "adj_close": 10.5 + i, "volume": 0.0})
    engine = sa.create_engine("sqlite://")
    pd.DataFrame(rows).to_sql("ohlcv", engine, index=False)
    check = validation.check_no_market_wide_flat_bars(engine, ["T0.NS"])
    assert check.status == validation.WARN and recent[1] in check.detail
    assert recent[0] not in check.detail

    healthy = pd.DataFrame([r for r in rows if r["date"] == recent[0]])
    engine2 = sa.create_engine("sqlite://")
    healthy.to_sql("ohlcv", engine2, index=False)
    assert validation.check_no_market_wide_flat_bars(engine2, ["T0.NS"]).status \
        == validation.PASS
