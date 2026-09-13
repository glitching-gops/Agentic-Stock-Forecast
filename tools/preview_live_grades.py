"""
tools/preview_live_grades.py — what the next DAILY run will publish. Read-only.

The weekly job persists each ticker's evaluation to `model_metadata`; the
badge itself is not written until the daily job grades it into
`forecast_current`. Between the two, the grades about to be published exist
only as a function of the database. This runs that function — the real
`_load_persisted_evaluation` and the real `grade_evidence` — and prints the
result beside what the site shows now.

It writes nothing. It assumes every daily forecast succeeds, so its WEAK and
STRONG counts are upper bounds.

    python tools/preview_live_grades.py
"""

from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _state(ev: dict | None) -> dict:
    # The same fields `agents.forecasting_agent.forecasting_node` hands the gate.
    if not ev:
        return {"forecast_available": True}
    return {
        "forecast_available": True,
        "eval_rank_ic": ev.get("rank_ic"),
        "eval_rank_ic_t": ev.get("rank_ic_t"),
        "eval_hit_rate": ev.get("hit_rate"),
        "eval_baseline_hit_rate": ev.get("majority_hit_rate"),
        "eval_beats_naive": ev.get("beats_naive"),
        "eval_evaluated_at": ev.get("evaluated_at"),
    }


def main() -> None:
    import pandas as pd
    from sqlalchemy import text

    from agents.critic_agent import grade_evidence
    from data.db import get_engine
    from data.universe import get_universe
    from pipeline.model import _load_persisted_evaluation

    universe = get_universe()
    live = pd.read_sql(
        text("SELECT ticker, forecast_confidence FROM forecast_current"),
        get_engine())
    live = dict(zip(live["ticker"], live["forecast_confidence"]))

    rows = []
    for ticker in universe:
        ev = _load_persisted_evaluation(ticker)
        grade, _ = grade_evidence(_state(ev))
        rows.append((ticker, live.get(ticker), grade, ev))

    now = collections.Counter(str(r[1]) for r in rows)
    nxt = collections.Counter(r[2] for r in rows)
    stamps = sorted(r[3]["evaluated_at"] for r in rows
                    if r[3] and r[3].get("evaluated_at"))
    print(f"universe: {len(universe)} tickers")
    if stamps:
        print(f"evaluation on file: {stamps[0]} -> {stamps[-1]}")
    print(f"never evaluated: {sum(1 for r in rows if not r[3])}   "
          f"evaluated, rank IC not measured: "
          f"{sum(1 for r in rows if r[3] and r[3].get('rank_ic') is None)}")
    for grade in ("STRONG", "WEAK", "INSUFFICIENT"):
        print(f"  {grade:13s} on the site now {now.get(grade, 0):3d}   "
              f"next daily run {nxt.get(grade, 0):3d}")

    moved = [r for r in rows if r[1] != r[2]]
    print(f"\n{len(moved)} ticker(s) change grade at the next daily run:")
    for ticker, before, after, ev in moved:
        ic = (ev or {}).get("rank_ic")
        t = (ev or {}).get("rank_ic_t")
        print(f"  {ticker:15s} {str(before):12s} -> {after:12s} "
              f"IC {'n/a' if ic is None else f'{ic:+.4f}'}  "
              f"t {'n/a' if t is None else f'{t:+.2f}'}")


if __name__ == "__main__":
    main()
