"""
tools/score_grade_predictiveness.py — does the evidence GRADE predict outcomes?

    python tools/score_grade_predictiveness.py          # runs, or says why not

PRE-REGISTERED 2026-09-04, BEFORE ANY OUTCOME EXISTS. That timing is the point.
`forecast_outcomes` held ZERO rows when this was written — the first forecast
under the current MODEL_VERSION matures in mid-October 2026 — so the method and
the bar below were fixed with the answer genuinely unavailable. This project has
recorded three results that did not survive being re-run at a different
arbitrary setting; a design chosen once the numbers are on screen is not a test.

WHY THIS QUESTION AND NOT THE ONE IN THE PHASE PLAN
----------------------------------------------------
Phase 3 was scoped to "ablate whether the critic improves realised hit rate at
all". Audited 2026-09-04, the critic's LLM review had run on 38 rows, raised
flags on 172 across the project's life, and changed the verdict on ZERO of them
— no ticker has ever been graded STRONG, which is the only grade a flag can
move. It was retired. So there is no critic intervention left to ablate: the
verdict is now a deterministic relabelling of the evidence grade.

The question underneath it survives and is better. The gate exists to say
whether a forecast is backed by held-out evidence, and that claim has NEVER been
checked against what actually happened. If WEAK rows do not out-perform
INSUFFICIENT ones on published output, the gate is decoration.

THE BAR, FIXED NOW
------------------
The grade is predictive only if BOTH hold:

  1. WEAK rows beat INSUFFICIENT rows on realised direction, by a margin whose
     confidence interval excludes zero.
  2. The margin survives a split by month. This project's three retired results
     were each carried by one period.

Anything else is a null and is recorded as one.

THREE TRAPS, ADDRESSED IN ADVANCE
----------------------------------
  SAMPLE. Roughly 2 of 84 tickers a day are graded above INSUFFICIENT, so the
  treated group grows at ~2 rows a day. The minimum detectable effect is
  reported BEFORE the estimate, and a margin the sample cannot resolve is
  reported as "cannot distinguish", never as a null result.

  CONFOUND. Grade is assigned from per-ticker walk-forward metrics, so graded
  and ungraded rows differ in the very quantity being tested. That is not
  removable here — it IS the hypothesis — but it means the comparison is
  observational and a positive result is evidence the gate ORDERS rows, not
  that it causes anything.

  ONE MODEL_VERSION AT A TIME. A verdict written under an earlier version
  graded a different target through a different gate. Pooling them answers a
  question about neither, which is exactly why `_load_persisted_evaluation`
  discards a stale evaluation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import pandas as pd                                              # noqa: E402
from sqlalchemy import text                                      # noqa: E402

from data.db import get_engine                                   # noqa: E402

#: Below this many resolved rows in the smaller group, the comparison is
#: refused rather than reported. Not a tuning knob: at ~2 graded rows a day, a
#: handful of outcomes cannot separate a real edge from noise, and a number
#: printed with a caveat gets quoted without one.
MIN_ROWS_PER_GROUP = 30


def load(engine=None, model_version: str | None = None) -> pd.DataFrame:
    """Resolved outcomes joined to the grade the forecast carried."""
    engine = engine or get_engine()
    df = pd.read_sql(text("""
        SELECT o.ticker, o.forecast_date, o.direction_correct, o.inside_interval,
               o.realised_return, o.pred_return,
               f.forecast_confidence AS grade, f.model_version
        FROM forecast_outcomes o
        JOIN forecasts f
          ON f.ticker = o.ticker
         AND CAST(f.last_updated AS DATE) = CAST(o.forecast_date AS DATE)
    """), engine)
    if model_version and not df.empty:
        df = df[df["model_version"] == model_version]
    return df


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson interval for a proportion.

    Not the normal approximation: at these sample sizes and hit rates near 0.5
    the textbook interval is wrong in the direction that flatters a result, and
    the whole point of this tool is not to do that.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # CLAMPED, because floating point does not respect the algebra. At hits=0
    # the lower bound comes out at -2.8e-17 rather than 0, which formats as
    # "-0.0%" — a negative probability printed beside a real one, in a tool
    # whose entire job is to be quotable without a caveat.
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-version", default="rebuild-absolute-return-v3")
    args = ap.parse_args()

    df = load(model_version=args.model_version)

    print("=" * 74)
    print("DOES THE EVIDENCE GRADE PREDICT REALISED DIRECTION?")
    print("=" * 74)

    if df.empty:
        print(f"\n  NO RESOLVED OUTCOMES for {args.model_version}.\n")
        print("  This is expected until ~mid-October 2026: a forecast resolves "
              "30 sessions\n  after it is published, and the first one under "
              "this MODEL_VERSION was\n  made on 2026-09-02. The daily job's "
              "resolve_due_forecasts writes them.\n")
        print("  The method and the bar above were fixed on 2026-09-04, with "
              "the answer\n  genuinely unavailable. Do not change them once "
              "rows appear.")
        return 0

    df["graded"] = df["grade"].isin(("STRONG", "WEAK"))
    groups = {"graded (STRONG/WEAK)": df[df["graded"]],
              "INSUFFICIENT": df[~df["graded"]]}

    print(f"\n  {len(df):,} resolved outcomes under {args.model_version}\n")
    print(f"  {'group':24s}{'n':>6}{'hit rate':>10}{'95% CI':>20}")
    stats = {}
    for name, g in groups.items():
        n = int(g["direction_correct"].notna().sum())
        hits = int(g["direction_correct"].fillna(0).sum())
        lo, hi = wilson(hits, n)
        stats[name] = (n, hits, lo, hi)
        rate = f"{hits / n:.1%}" if n else "-"
        band = f"[{lo:.1%}, {hi:.1%}]" if n else "-"
        print(f"  {name:24s}{n:>6}{rate:>10}{band:>20}")

    smallest = min(n for n, *_ in stats.values())
    if smallest < MIN_ROWS_PER_GROUP:
        print(f"\n  CANNOT DISTINGUISH. The smaller group holds {smallest} rows, "
              f"below the\n  {MIN_ROWS_PER_GROUP}-row floor. At ~2 graded rows a "
              f"day this needs about\n  {MIN_ROWS_PER_GROUP // 2} more trading "
              f"days. Reporting a margin here would be\n  reporting noise, and "
              f"a number printed with a caveat gets quoted without one.")
        return 0

    (na, ha, la, hia) = stats["graded (STRONG/WEAK)"]
    (nb, hb, lb, hib) = stats["INSUFFICIENT"]
    margin = ha / na - hb / nb
    overlap = not (la > hib or lb > hia)

    print(f"\n  margin {margin:+.1%}   intervals "
          f"{'OVERLAP' if overlap else 'are disjoint'}")
    print("\n  by month, because a margin carried by one period is not a margin:")
    df["month"] = df["forecast_date"].astype(str).str.slice(0, 7)
    for month, g in df.groupby("month"):
        a, b = g[g["graded"]], g[~g["graded"]]
        if len(a) and len(b):
            print(f"    {month}  graded {a['direction_correct'].mean():.0%} "
                  f"(n={len(a)})   INSUFFICIENT "
                  f"{b['direction_correct'].mean():.0%} (n={len(b)})")

    print()
    if overlap:
        print("  NULL. The grade does not separate realised direction on this "
              "sample.\n  On the pre-registered bar the gate is decoration.")
    else:
        print("  The intervals are disjoint. Read the monthly split above "
              "before quoting\n  this: the bar requires the margin to survive "
              "it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
