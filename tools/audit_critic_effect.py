"""
tools/audit_critic_effect.py — has the critic's LLM review ever changed a row?

    python tools/audit_critic_effect.py
    python tools/audit_critic_effect.py --model-version rebuild-absolute-return-v3

WHY THIS COMES BEFORE THE ABLATION
-----------------------------------
Phase 3 is scoped to "ablate whether the critic improves realised hit rate at
all". That question needs resolved outcomes, and `forecast_outcomes` is empty:
forecasts under the current MODEL_VERSION begin 2026-09-02 and resolution takes
30 sessions, so the first one matures in mid-October.

But there is a prior question that needs no outcome data, and if its answer is
what the code suggests, the ablation is moot. The critic reaches a published row
through exactly ONE channel:

    verdict = {"STRONG": "APPROVED", "WEAK": "FLAGGED",
               "INSUFFICIENT": "REJECTED"}[grade]
    if flags and verdict == "APPROVED":
        verdict = "FLAGGED"

A flag can only change something on a row that was going to be APPROVED, and
APPROVED requires STRONG, which requires 3 of 3 evidence checks. So the
measurable question is: **how many published rows has a flag actually moved?**

WHY IT IS SPLIT BY MODEL_VERSION
---------------------------------
The same reason `_load_persisted_evaluation` discards a stale evaluation: a
verdict written under `phase0-excess-return-v1` graded a different target
through a different gate, and pooling it with the current one would answer a
question about neither. The pre-P0 rows also carry the old
High/Medium/Low confidence scale, which is not comparable to
STRONG/WEAK/INSUFFICIENT at all.

WHAT THIS IS NOT
----------------
It is not the ablation. It measures whether the critic CAN have had an effect,
not whether its flags were correct. A critic that never fires is not thereby
proven right or wrong — it is proven inert, which is a different and cheaper
thing to establish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import pandas as pd                                              # noqa: E402
from sqlalchemy import text                                      # noqa: E402

from data.db import get_engine                                   # noqa: E402

#: Grades on which the LLM review is invoked at all. Mirrors the gate in
#: critic_agent.critic_node: EVIDENCE_MULTIPLIER[grade] > 0.0.
GATE_OPEN = ("STRONG", "WEAK")


def has_flags(raw) -> bool:
    """A row carries at least one flag. `critic_flags` is a JSON list as TEXT."""
    if raw is None or (isinstance(raw, float) and raw != raw):
        return False
    if isinstance(raw, (list, tuple)):
        return len(raw) > 0
    s = str(raw).strip()
    if s in ("", "[]", "null", "None"):
        return False
    try:
        return bool(json.loads(s))
    except (json.JSONDecodeError, TypeError):
        return bool(s)


def audit(engine=None, model_version: str | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    df = pd.read_sql(text("""
        SELECT model_version, forecast_confidence, critic_verdict, critic_flags
        FROM forecasts
    """), engine)
    if model_version:
        df = df[df["model_version"] == model_version]
    df["flagged"] = df["critic_flags"].map(has_flags)
    df["gate_open"] = df["forecast_confidence"].isin(GATE_OPEN)
    # THE ONLY ROW A FLAG CAN MOVE: graded STRONG (so the verdict would have
    # been APPROVED) and published as FLAGGED.
    df["changed"] = (df["forecast_confidence"] == "STRONG") & \
                    (df["critic_verdict"] == "FLAGGED")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-version")
    args = ap.parse_args()

    df = audit(model_version=args.model_version)
    if df.empty:
        print("no forecast rows"); return 1

    print("=" * 74)
    print("HAS A CRITIC FLAG EVER CHANGED A PUBLISHED ROW?")
    print("=" * 74)
    print(f"  {'model_version':32s}{'rows':>7}{'gate open':>11}{'flagged':>9}"
          f"{'CHANGED':>9}")
    for mv, g in df.groupby(df["model_version"].fillna("(null, pre-P0)")):
        print(f"  {str(mv):32s}{len(g):>7}{int(g.gate_open.sum()):>11}"
              f"{int(g.flagged.sum()):>9}{int(g.changed.sum()):>9}")
    print(f"  {'TOTAL':32s}{len(df):>7}{int(df.gate_open.sum()):>11}"
          f"{int(df.flagged.sum()):>9}{int(df.changed.sum()):>9}")

    print("\n  grades ever assigned:")
    for grade, n in df["forecast_confidence"].value_counts(dropna=False).items():
        note = ""
        if grade == "STRONG":
            note = "  <- the only grade a flag can move"
        elif str(grade) in ("High", "Medium", "Low"):
            note = "  (pre-P0 scale, not comparable)"
        print(f"    {str(grade):16s}{n:>7}{note}")

    changed = int(df["changed"].sum())
    strong = int((df["forecast_confidence"] == "STRONG").sum())
    calls = int(df["gate_open"].sum())

    print()
    if strong == 0:
        print(f"  NO ROW HAS EVER BEEN GRADED STRONG, so no flag could have "
              f"changed a\n  verdict. The LLM review was invoked on {calls:,} "
              f"rows and its output was\n  incapable of moving any of them.")
    elif changed == 0:
        print(f"  {strong} rows were graded STRONG and NONE was downgraded by a "
              f"flag.\n  The review ran on {calls:,} rows and changed nothing.")
    else:
        print(f"  {changed} of {calls:,} reviewed rows were changed by a flag "
              f"({changed / max(calls, 1):.2%}).")

    print()
    print("  READ IT AS A CAPABILITY, NOT A VERDICT. A critic that never fires "
          "is not\n  thereby right or wrong — it is inert. Whether its flags "
          "would have been\n  CORRECT needs resolved outcomes, and "
          "forecast_outcomes is still empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
