"""
tools/sync_fundamentals.py — Populate the point-in-time valuation table.

    python tools/sync_fundamentals.py                  # the screened universe
    python tools/sync_fundamentals.py --tickers RELIANCE.NS TCS.NS
    python tools/sync_fundamentals.py --dry-run        # fetch, show, write nothing
    python tools/sync_fundamentals.py --restatements   # report only, no fetch

The weekly job calls ``sync_fundamentals`` itself, so this tool exists for two
narrower jobs: populating a fresh checkout before ``run_baselines.py
--fundamentals`` can mean anything, and reading the restatement log without
waiting for Saturday.

WHAT THE RESTATEMENT REPORT IS FOR
----------------------------------
yfinance serves financial statements AS RESTATED, not as originally filed. A
company that restates FY2024 during FY2025 gives us the restated figure and we
attach it to 2024 — information nobody had at the time, in the direction that
flatters the model. That is the one unresolved threat to the only result in
this project that has cleared its pre-registered bar.

It cannot be fixed retrospectively without a vendor that keeps as-reported
vintages. It CAN be measured going forward: every sync compares each incoming
figure against what is already on file and records anything that moved into
``fundamental_revisions``. A year of syncs showing no material revisions is
evidence the bias is small; a year showing many is evidence the valuation
result needs a different data source before it can be believed.

So the honest reading of a zero here is "nothing has moved YET", not "there is
no restatement bias" — the log only sees periods recorded since it began, which
is what ``periods_tracked`` reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.fundamentals import (                            # noqa: E402
    FetchCoverageRefused, fetch_fundamentals, restatement_summary,
    sync_fundamentals,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="tickers to fetch (default: the screened universe)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and summarise, but write nothing")
    ap.add_argument("--restatements", action="store_true",
                    help="print the restatement log and exit; fetches nothing")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="override the fraction of requested tickers that must "
                         "return statements before anything is written")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout, force=True)

    if args.restatements:
        print(json.dumps(restatement_summary(), indent=2, default=str))
        return 0

    tickers = args.tickers
    if not tickers:
        from data.universe import get_universe
        tickers = get_universe()
        print(f"Screened universe: {len(tickers)} tickers")
    if not tickers:
        print("  no tickers to sync", file=sys.stderr)
        return 1

    if args.dry_run:
        # Deliberately calls fetch + a reporting pass rather than sync, so a
        # dry run cannot reach store_fundamentals by any path.
        frame = fetch_fundamentals(tickers)
        got = frame["ticker"].nunique() if len(frame) else 0
        print(f"\n  DRY RUN — nothing written.")
        print(f"  {len(frame)} periods across {got} of {len(tickers)} tickers")
        if len(frame):
            print(f"  effective dates {frame['effective_date'].min()} -> "
                  f"{frame['effective_date'].max()}")
        return 0

    kwargs = {}
    if args.min_coverage is not None:
        kwargs["min_coverage"] = args.min_coverage

    try:
        counts = sync_fundamentals(tickers, **kwargs)
    except FetchCoverageRefused as exc:
        # A refusal, not a warning. Valuation is standardised WITHIN each date,
        # so a partial write produces a cross-section whose mean and standard
        # deviation are taken over whichever HTTP calls happened to succeed —
        # indistinguishable downstream from a real one.
        print(f"\n  REFUSED: {exc}", file=sys.stderr)
        return 1

    print(f"\n  {counts['periods']} periods: {counts['new']} new, "
          f"{counts['revised']} revised, {counts['unchanged']} unchanged")
    print(f"  coverage {counts['tickers_returned']}/"
          f"{counts['tickers_requested']} tickers "
          f"({counts['fetch_coverage']:.0%})")

    summary = restatement_summary()
    print(f"\nRestatement log:")
    print(json.dumps(summary, indent=2, default=str))
    if summary["revisions"]:
        print("\n  A figure already on file was CHANGED by the vendor. Any "
              "measurement\n  taken against the previous value described data "
              "that no longer exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
