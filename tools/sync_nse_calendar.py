"""
tools/sync_nse_calendar.py — build data/nse_calendar.json from NSE's own records.

Run it when NSE publishes next year's holiday list (each December), after any
ad hoc closure, and whenever `CalendarNotCurrent` is raised. Commit the JSON.

TWO SOURCES, BOTH NSE'S (see data/nse_calendar.py for why each is needed):

  1. The published trading-holiday list, one request per year.
  2. The daily delivery archive (MTO_<ddmmyyyy>.DAT), which exists for every
     session NSE held and no other date. Probed, at the archive client's
     3-requests-a-second throttle, for:
       - every Saturday and Sunday (Muhurat, budget-day and DR sessions);
       - every listed holiday that has already happened (a Muhurat evening
         appears on the list as a holiday);
       - every past weekday NOT already confirmed by `delivery_cache.npz`,
         whose manifest records the archive's answer for each panel session.

A probe the archive does not answer cleanly (403, error, malformed) aborts the
build. Nothing is written from a partial read: a calendar with a hole in it
would let Yahoo's phantom bars back in on exactly those dates.

    python tools/sync_nse_calendar.py                 # 2016 .. this year
    python tools/sync_nse_calendar.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.nse_calendar import (  # noqa: E402
    CALENDAR_PATH, SEGMENT, fetch_holiday_master, from_dict, parse_holiday_master)
from pipeline.delivery import ArchiveClient  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DELIVERY_CACHE = os.path.join(ROOT, "delivery_cache.npz")
FIRST_YEAR = 2016          # ohlcv begins 2016-08-16


def cached_sessions(path: str = DELIVERY_CACHE) -> set[str]:
    """Dates the Stage 1b backfill already found an MTO file for."""
    if not os.path.exists(path):
        return set()
    z = np.load(path, allow_pickle=True)
    days = pd.to_datetime(pd.Series(z["man_date"]).astype(str)).dt.strftime("%Y-%m-%d")
    ok = pd.Series(z["man_status"]).astype(str).eq("ok").to_numpy()
    return set(days[ok])


def build(first_year: int, last_day: date, client: ArchiveClient,
          fetch=fetch_holiday_master, confirmed: set[str] | None = None,
          log=print) -> dict:
    confirmed = confirmed or set()
    years = list(range(first_year, last_day.year + 1))

    listed: dict[str, str] = {}
    provenance = {}
    for y in years:
        payload = fetch(y)
        rows = parse_holiday_master(payload, y)
        listed.update(rows)
        provenance[str(y)] = {
            "listed": len(rows),
            "sha256": hashlib.sha256(json.dumps(payload.get(SEGMENT), sort_keys=True)
                                     .encode()).hexdigest()}
        log(f"  {y}: {len(rows)} listed {SEGMENT} holidays")

    # ── the probe set ───────────────────────────────────────────────────────
    days = pd.date_range(f"{first_year}-01-01", last_day, freq="D")
    iso = days.strftime("%Y-%m-%d")
    weekend = days.weekday >= 5
    probe = sorted({d for d, w in zip(iso, weekend)
                    if w or d in listed or d not in confirmed})
    log(f"  probing the archive for {len(probe):,} dates "
        f"(~{len(probe) / 3 / 60:.0f} min at the throttle)")

    traded: dict[str, bool] = {d: True for d in confirmed}
    for i, d in enumerate(probe, 1):
        r = client.fetch_mto(pd.Timestamp(d))
        if r.status == "ok":
            traded[d] = True
        elif r.status == "not_found":
            traded[d] = False
        else:
            raise SystemExit(f"archive answered {r.status} ({r.http}) for {d}: "
                             f"refusing to write a calendar with a hole in it")
        if i % 200 == 0:
            log(f"    {i:,}/{len(probe):,}")

    # ── resolve ─────────────────────────────────────────────────────────────
    closed, weekend_sessions = {}, {}
    for d, w in zip(iso, weekend):
        if w and traded.get(d):
            weekend_sessions[d] = (f"session held; listed as {listed[d]!r}"
                                   if d in listed else "special session")
        elif not w and traded.get(d) is False:
            closed[d] = listed.get(d, "no session in NSE's archive; NOT on the "
                                      "published list")
    verified_through = last_day.strftime("%Y-%m-%d")
    for d, desc in listed.items():                     # the future: the list
        if d > verified_through and pd.Timestamp(d).weekday() < 5:
            closed[d] = desc

    muhurat = sorted(d for d in listed if d <= verified_through and traded.get(d))
    unlisted = sorted(d for d, v in closed.items() if "NOT on the published" in v)
    return {
        "source": ("NSE trading-holiday list (api/holiday-master, CM) for the "
                   "future; NSE's MTO delivery archive for every date up to "
                   "verified_through"),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "years": years,
        "verified_through": verified_through,
        "closed_weekdays": dict(sorted(closed.items())),
        "weekend_sessions": dict(sorted(weekend_sessions.items())),
        "checks": {
            "listed_holidays_that_traded": muhurat,
            "weekday_closures_not_on_the_list": unlisted,
            "probed": len(probe),
            "confirmed_from_delivery_cache": len(confirmed),
            "per_year": provenance,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--through", default=None,
                    help="last date to verify against the archive (default: yesterday)")
    ap.add_argument("--out", default=CALENDAR_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    last = (date.fromisoformat(args.through) if args.through
            else date.today() - timedelta(days=1))
    confirmed = {d for d in cached_sessions() if d <= last.isoformat()}
    print(f"NSE calendar {args.first_year}..{last.year}, verified through {last}; "
          f"{len(confirmed):,} sessions already confirmed by {DELIVERY_CACHE}")
    cal = build(args.first_year, last, ArchiveClient(), confirmed=confirmed)
    from_dict(cal)                                           # it must load
    c = cal["checks"]
    print(f"closed weekdays {len(cal['closed_weekdays'])}, weekend sessions "
          f"{len(cal['weekend_sessions'])}: {sorted(cal['weekend_sessions'])}")
    print(f"listed holidays that traded (Muhurat): {c['listed_holidays_that_traded']}")
    print(f"weekday closures NOT on the list: {c['weekday_closures_not_on_the_list']}")
    if args.dry_run:
        return
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cal, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
