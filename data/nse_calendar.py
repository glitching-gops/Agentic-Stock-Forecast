"""
data/nse_calendar.py — which dates NSE actually traded.

WHY THIS EXISTS. Yahoo returns a bar for an NSE holiday: open = high = low =
close = the previous close, volume 0, for every ticker. `pipeline/fetch.py`
stored those bars verbatim, so the panel carried sessions that never traded —
2026-01-15, 05-01, 05-28, 06-26 and 09-14. A 30-row label spanning one measures
29 real sessions, and every rolling feature counts a zero-return day that did
not happen. It is the vroc_10 hole landmine inverted: an extra row, not a
missing one, and `check_sessions_are_contiguous` cannot see it because the grid
IS contiguous.

THE AUTHORITY IS NSE ITSELF, in two parts, because neither is enough alone:

  past dates    NSE's own archive. It publishes a security-wise delivery file
                (`MTO_<ddmmyyyy>.DAT`, the source `pipeline/delivery.py` already
                reads) for every session it held and for no other date. That is
                the only source that gets Muhurat trading right: NSE's holiday
                list shows 2023-11-12 and 2016-10-30 as holidays although an
                evening session ran on both, and the archive has a file for each.
  future dates  NSE's published trading-holiday list,
                `/api/holiday-master?type=trading&year=Y`, CM segment. Published
                each December for the following year and amended for ad hoc
                closures (2026-01-15, a municipal election, is one).

`tools/sync_nse_calendar.py` builds `data/nse_calendar.json` from both, and
`live_calendar()` merges the CURRENT year's published list into it at run time,
so an ad hoc closure announced after the file was committed is still honoured.

THE RULE. A date is a session iff it is a weekday not in `closed_weekdays`, or a
weekend date in `weekend_sessions`. A date in a year the calendar does not
cover raises `CalendarNotCurrent` rather than guessing: a guessed calendar is
exactly how phantom sessions got in.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date as _date
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np
import pandas as pd

CALENDAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "nse_calendar.json")
HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading&year={year}"
SEGMENT = "CM"          # capital market: the segment every ticker here trades in
API_TIMEOUT = 20


class CalendarNotCurrent(RuntimeError):
    """A date fell in a year the NSE calendar does not cover. Refresh it with
    `python tools/sync_nse_calendar.py` and commit data/nse_calendar.json."""


def _iso(day) -> str:
    return pd.Timestamp(day).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class NSECalendar:
    #: Weekdays on which NSE held no session.
    closed_weekdays: frozenset[str]
    #: Saturdays and Sundays on which it did (Muhurat, budget day, DR drills).
    weekend_sessions: frozenset[str]
    #: Calendar years whose holiday list is known.
    years: frozenset[int]
    #: Last date checked against the archive; later dates rest on the list.
    verified_through: str = ""
    #: Where each non-default date came from, for the audit trail.
    notes: dict = field(default_factory=dict, compare=False, hash=False)

    def covers(self, day) -> bool:
        return pd.Timestamp(day).year in self.years

    def is_session(self, day) -> bool:
        ts = pd.Timestamp(day)
        if ts.year not in self.years:
            raise CalendarNotCurrent(
                f"{_iso(ts)}: the NSE calendar covers {sorted(self.years)}, not "
                f"{ts.year}. Run `python tools/sync_nse_calendar.py` and commit "
                f"data/nse_calendar.json.")
        iso = _iso(ts)
        if ts.weekday() >= 5:
            return iso in self.weekend_sessions
        return iso not in self.closed_weekdays

    def session_mask(self, dates: Iterable) -> np.ndarray:
        """True where each date is a session. Raises on an uncovered year."""
        dates = pd.Series(pd.to_datetime(pd.Series(list(dates)).astype(str)))
        if dates.empty:
            return np.zeros(0, dtype=bool)
        uncovered = sorted({int(y) for y in dates.dt.year.unique()} - set(self.years))
        if uncovered:
            raise CalendarNotCurrent(
                f"dates in {uncovered} fall outside the NSE calendar "
                f"({sorted(self.years)}). Run `python tools/sync_nse_calendar.py` "
                f"and commit data/nse_calendar.json.")
        iso = dates.dt.strftime("%Y-%m-%d")
        weekend = dates.dt.weekday.to_numpy() >= 5
        closed = iso.isin(self.closed_weekdays).to_numpy()
        special = iso.isin(self.weekend_sessions).to_numpy()
        return np.where(weekend, special, ~closed)

    def with_closed(self, extra: dict[str, str], year: int) -> "NSECalendar":
        """This calendar plus `extra` weekday closures, covering `year`."""
        wk = {d: v for d, v in extra.items() if pd.Timestamp(d).weekday() < 5}
        return NSECalendar(self.closed_weekdays | frozenset(wk),
                           self.weekend_sessions,
                           self.years | {int(year)},
                           self.verified_through,
                           {**self.notes, **{d: f"live list: {v}" for d, v in wk.items()}})


# ── the file ──────────────────────────────────────────────────────────────────


def from_dict(raw: dict) -> NSECalendar:
    return NSECalendar(frozenset(raw["closed_weekdays"]),
                       frozenset(raw["weekend_sessions"]),
                       frozenset(int(y) for y in raw["years"]),
                       raw.get("verified_through", ""),
                       {**raw["closed_weekdays"], **raw["weekend_sessions"]})


@lru_cache(maxsize=4)
def load(path: str = CALENDAR_PATH) -> NSECalendar:
    with open(path, encoding="utf-8") as f:
        return from_dict(json.load(f))


# ── NSE's published list ──────────────────────────────────────────────────────


def parse_holiday_master(payload: dict, year: int,
                         segment: str = SEGMENT) -> dict[str, str]:
    """
    {iso date: description} for every trading holiday NSE lists in `year`.

    Refuses a payload with no entries for the segment rather than returning an
    empty list: an empty list is indistinguishable from "no holidays", and every
    Yahoo phantom bar in that year would then be accepted as a session.
    """
    rows = payload.get(segment) or []
    out = {}
    for r in rows:
        day = pd.to_datetime(r["tradingDate"], format="%d-%b-%Y")
        if day.year == year:
            out[day.strftime("%Y-%m-%d")] = str(r.get("description", "")).strip()
    if not out:
        raise ValueError(f"NSE holiday list for {year} has no {segment} entries")
    return out


def fetch_holiday_master(year: int, session=None) -> dict:
    """The raw JSON of NSE's trading-holiday list for `year`. Measured
    2026-09-21: answers 200 without cookies, for every year back to 2016."""
    if session is None:
        import requests

        session = requests.Session()
    r = session.get(HOLIDAY_API.format(year=year), timeout=API_TIMEOUT,
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                             "Referer": "https://www.nseindia.com/"})
    r.raise_for_status()
    return r.json()


def live_calendar(base: NSECalendar | None = None, today: _date | None = None,
                  fetch: Callable[[int], dict] | None = None
                  ) -> tuple[NSECalendar, str]:
    """
    The committed calendar plus NSE's CURRENT published list for this year.

    Best effort, and never silently: returns the calendar actually used and a
    note saying what happened. A failed fetch falls back to the committed file
    when that covers the year; when it does not, the uncovered calendar is
    returned and `is_session` will raise at the first date that needs it.

    Only CLOSURES are merged. The live list cannot tell a Muhurat evening from
    a full holiday (both appear with null session fields), so a Muhurat day in
    the current year is excluded until `tools/sync_nse_calendar.py` verifies it
    against the archive — a missing hour-long session costs less than a phantom.
    """
    base = base or load()
    year = (today or _date.today()).year
    fetch = fetch or fetch_holiday_master          # looked up at CALL time
    try:
        listed = parse_holiday_master(fetch(year), year)
    except Exception as exc:                                    # noqa: BLE001
        state = "covers" if year in base.years else "DOES NOT COVER"
        return base, (f"NSE holiday list for {year} unavailable "
                      f"({type(exc).__name__}: {str(exc)[:120]}); the committed "
                      f"calendar {state} {year}")
    new = {d: v for d, v in listed.items()
           if pd.Timestamp(d).weekday() < 5 and d not in base.closed_weekdays
           and d > base.verified_through}
    merged = base.with_closed(new, year)
    note = (f"NSE holiday list for {year}: {len(listed)} entries, "
            f"{len(new)} closure(s) not in the committed file"
            + (f" {sorted(new)}" if new else ""))
    return merged, note


_current: dict[str, tuple[NSECalendar, str]] = {}


def current(today: _date | None = None) -> tuple[NSECalendar, str]:
    """`live_calendar()`, fetched at most once per process per day, so the
    ingestion step and the signals write judge a date by the SAME calendar."""
    key = (today or _date.today()).isoformat()
    if key not in _current:
        _current[key] = live_calendar(today=today)
    return _current[key]


# ── applying it ───────────────────────────────────────────────────────────────


def drop_non_sessions(frame: pd.DataFrame, calendar: NSECalendar,
                      date_col: str = "date") -> tuple[pd.DataFrame, list[str]]:
    """`frame` without rows on dates NSE did not trade, and the dates dropped."""
    if frame.empty:
        return frame, []
    mask = calendar.session_mask(frame[date_col])
    dropped = sorted(set(frame.loc[~mask, date_col].astype(str).str[:10]))
    return frame.loc[mask].reset_index(drop=True), dropped
