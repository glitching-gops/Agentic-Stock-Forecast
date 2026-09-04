"""
tools/audit_news_relevance.py — is the alias filter keeping the right articles?

    python tools/audit_news_relevance.py --sample          # draw a worksheet
    python tools/audit_news_relevance.py --score           # grade it once labelled

WHY THIS IS OWED
----------------
The decision was "alias match first, MEASURE, escalate": use a deterministic
company-name match, then measure its precision against a hand-labelled sample,
and only pay an LLM to adjudicate the residual if the measurement says to. The
first half shipped and the second half did not, so the filter that decides which
articles reach the model has never been checked.

It keeps 52.0% of what Google returns. That number is doing a lot of work and
nobody has looked at it. If precision is poor, half the archive is noise
attributed to the wrong ticker — which is WORSE than a missing article, because
it is indistinguishable from signal. If recall is poor, the null measured in P3
was measured on a fraction of the available news and is not a verdict on news.

RECALL CANNOT BE MEASURED FROM THE ARCHIVE, AND THAT IS A DESIGN CONSEQUENCE
-----------------------------------------------------------------------------
`store_window` persists only the articles that PASSED the filter. The ~48% it
drops were never written anywhere, so the stored archive can answer "of what we
kept, how much was right" and cannot answer "of what was there, how much did we
keep". Fixing that retrospectively would mean re-fetching everything.

So this re-fetches a SMALL SAMPLE of windows and keeps every article, filtered
and unfiltered alike. Roughly thirty requests buys both numbers on a real
sample — against ~2,000 to redo the archive, and against a quota that refuses
after about six hundred.

HOW THE SAMPLE IS DRAWN
-----------------------
Windows are chosen at random across tickers AND across years, because the two
things most likely to break the filter are correlated with both: a company whose
name is a common word, and an era whose coverage is thin. A sample drawn from
the recent dense period would flatter the filter.

The worksheet is written as JSON with `label` left null. Fill it in — `1` if the
article is genuinely about that company, `0` if it is not — and `--score` reads
it back. Labels are stored, so the measurement is re-runnable and a later change
to the filter can be scored against the SAME sample rather than a fresh one.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import date, datetime

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import pandas as pd                                              # noqa: E402
from sqlalchemy import text                                      # noqa: E402

from data.db import get_engine                                   # noqa: E402
from data.tickers import get_company                             # noqa: E402
from pipeline.news import (                                      # noqa: E402
    GoogleNewsRSS, company_aliases, match_ticker, search_query,
)

DEFAULT_WORKSHEET = "news_relevance_sample.json"


def draw_sample(n_windows: int, seed: int, engine=None) -> list[dict]:
    """Re-fetches `n_windows` real (ticker, window) pairs, keeping EVERYTHING."""
    engine = engine or get_engine()
    rng = random.Random(seed)

    windows = pd.read_sql(text("""
        SELECT ticker, window_start, window_end
        FROM news_coverage WHERE status = 'ok' AND n_articles > 0
    """), engine)
    if windows.empty:
        raise SystemExit("no completed windows to sample from")

    # STRATIFIED BY YEAR, not uniform. Coverage density rises over the panel,
    # so a uniform draw would land mostly in the recent dense period and
    # flatter the filter exactly where it matters least.
    windows["year"] = windows["window_start"].str.slice(0, 4)
    by_year = {y: g.index.tolist() for y, g in windows.groupby("year")}
    years = sorted(by_year)
    picks: list[int] = []
    while len(picks) < n_windows and any(by_year.values()):
        for y in years:
            if not by_year[y] or len(picks) >= n_windows:
                continue
            picks.append(by_year[y].pop(rng.randrange(len(by_year[y]))))

    provider = GoogleNewsRSS(delay=0.5, retry_pause=2.0)
    rows: list[dict] = []
    for i, idx in enumerate(picks, 1):
        w = windows.loc[idx]
        ticker = w["ticker"]
        company = get_company(ticker)
        aliases = company_aliases(ticker, company)
        lo = datetime.strptime(w["window_start"], "%Y-%m-%d").date()
        hi = datetime.strptime(w["window_end"], "%Y-%m-%d").date()

        result = provider.fetch(search_query(ticker, company), lo, hi)
        if result.status != "ok":
            print(f"  [{i}/{len(picks)}] {ticker} {lo}..{hi} -> {result.status}")
            if result.status == "rate_limited":
                print("  stopping: the host is refusing us")
                break
            continue

        for art in result.articles:
            alias = match_ticker(art.title, aliases)
            rows.append({
                "ticker": ticker, "company": company,
                "window": f"{lo}..{hi}",
                "published_at": art.published_at,
                "title": art.title,
                "source": art.source,
                # WHAT THE FILTER DECIDED, recorded before any labelling, so the
                # label cannot be influenced by knowing the verdict.
                "kept_by_filter": bool(alias),
                "matched_alias": alias,
                "label": None,          # 1 = really about this company, 0 = not
            })
        print(f"  [{i}/{len(picks)}] {ticker:16s} {lo}..{hi}  "
              f"{len(result.articles):>3} articles, "
              f"{sum(1 for a in result.articles if match_ticker(a.title, aliases)):>3} kept")
    return rows


def score(rows: list[dict]) -> dict:
    """Precision, recall and F1 of the alias filter against the labels."""
    labelled = [r for r in rows if r.get("label") in (0, 1, "0", "1")]
    if not labelled:
        raise SystemExit(
            "no labelled rows. Open the worksheet and set \"label\" to 1 "
            "(really about this company) or 0 (not) on as many rows as you can.")

    for r in labelled:
        r["label"] = int(r["label"])

    tp = sum(1 for r in labelled if r["kept_by_filter"] and r["label"] == 1)
    fp = sum(1 for r in labelled if r["kept_by_filter"] and r["label"] == 0)
    fn = sum(1 for r in labelled if not r["kept_by_filter"] and r["label"] == 1)
    tn = sum(1 for r in labelled if not r["kept_by_filter"] and r["label"] == 0)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall)
          else float("nan"))
    return {"n_labelled": len(labelled), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="draw a new worksheet")
    ap.add_argument("--score", action="store_true", help="grade a filled worksheet")
    ap.add_argument("--windows", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--file", default=DEFAULT_WORKSHEET)
    args = ap.parse_args()

    if args.sample:
        if os.path.exists(args.file):
            raise SystemExit(
                f"{args.file} exists. Delete it or pass --file to draw a new "
                f"sample — overwriting would discard labels, and the point of "
                f"keeping them is that a filter change can be scored on the "
                f"SAME sample rather than a fresh one.")
        rows = draw_sample(args.windows, args.seed)
        with open(args.file, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1, ensure_ascii=False)
        kept = sum(1 for r in rows if r["kept_by_filter"])
        print(f"\n{len(rows)} articles written to {args.file} "
              f"({kept} kept by the filter, {len(rows) - kept} dropped).")
        print("Set \"label\" to 1 or 0 on each row, then run --score.")
        return 0

    if args.score:
        with open(args.file, encoding="utf-8") as fh:
            rows = json.load(fh)
        m = score(rows)
        print("=" * 70)
        print("ALIAS FILTER, against a hand-labelled sample")
        print("=" * 70)
        print(f"  {m['n_labelled']} labelled articles")
        print(f"    kept & correct   (TP) {m['tp']:>5}")
        print(f"    kept & WRONG     (FP) {m['fp']:>5}   <- noise on the wrong row")
        print(f"    dropped & wanted (FN) {m['fn']:>5}   <- news we threw away")
        print(f"    dropped & right  (TN) {m['tn']:>5}")
        print()
        print(f"  precision {m['precision']:.3f}   "
              f"recall {m['recall']:.3f}   F1 {m['f1']:.3f}")
        print()
        print("  HOW TO READ IT. Low PRECISION is the dangerous one: an article "
              "attributed\n  to the wrong ticker is noise the model cannot "
              "distinguish from signal, and\n  it would mean the P3 null was "
              "measured on a contaminated feature. Low\n  RECALL is milder but "
              "changes the claim: it would mean news was never\n  properly "
              "tested, rather than tested and found wanting.")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
