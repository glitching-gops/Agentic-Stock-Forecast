"""
agents/critic_agent.py — Evidence gate plus an LLM signal review.

The previous critic asked an LLM for a verdict and then overwrote it with five
deterministic branches keyed on ``mape`` and ``dir_acc`` (audit finding F9).
Because those were the leaked in-sample values — typically ~3% and ~83% —
Tier 1 (``mape < 6 and dir_acc > 75``) fired for effectively every stock, so the
verdict was a near-constant ``APPROVED`` and the LLM's output never survived to
the database. It contributed 30 of 100 composite points while carrying no
information.

Phase 0 splits the two jobs and makes both auditable:

  ``grade_evidence``  Deterministic, tested, and driven by held-out walk-forward
                      metrics. Answers "has this model demonstrated skill on
                      data it did not see?" This is the gate, and an LLM cannot
                      raise it.

  ``critic_node``     The LLM reviews signal coherence and may add flags. Its
                      flags are stored separately, and it can only downgrade —
                      so it is called only for grades that actually score. A
                      flag on an INSUFFICIENT row cannot move anything: the
                      composite multiplies by 0.0 before the deduction.

That asymmetry is deliberate: the LLM sees numbers it has no way to verify, so
it is allowed to raise doubt but never to certify.

Phase 3 replaces this with a critic that reads dated, attributed evidence and
flags contradictions — verifiable work. Until that exists, this layer stays
small and its limits are stated rather than disguised.
"""

from __future__ import annotations

from dotenv import load_dotenv

from agents.state import AgentState

load_dotenv(override=True)

# Thresholds for the deterministic gate. Deliberately modest: a rank IC of 0.03
# with a t-statistic above 2 is a real but weak edge, which is an honest
# description of what technical signals deliver at a 30-session horizon.
MIN_RANK_IC = 0.02

# SIGNED, not absolute. This is Q4 of the Phase 0 audit, closed 2026-09-02.
#
# The check used to read `abs(ic_t) >= MIN_IC_TSTAT`, and it is the ONLY one of
# the three that tests significance at all - the IC floor and the hit-rate edge
# are point estimates with no inferential content. Taking the absolute value
# therefore made the gate's single inferential check symmetric in a quantity
# that is not symmetric in meaning: a rank IC reliably NEGATIVE at t = -2.3 says
# the ranking is backwards, which is a finding about the model and the opposite
# of evidence for the forecast it is grading.
#
# This was not hypothetical. Measured over all 96 tickers on 2026-08-31, four
# passed this check and ALL FOUR had strongly negative IC:
#
#     MUTHOOTFIN.NS  IC -0.253  t -2.00   hit-base -20.1pp
#     TRENT.NS       IC -0.295  t -2.33   hit-base   0.0pp
#     HDFCAMC.NS     IC -0.321  t -2.18   hit-base   0.0pp
#     LT.NS          IC -0.263  t -2.08   hit-base   0.0pp
#
# The maximum POSITIVE t-statistic anywhere in the universe was +1.84. So the
# gate's only real check was passed exclusively by tickers the model gets
# reliably wrong, and any one of them was a single lucky second check away from
# being graded WEAK and published as validated.
#
# Fixing it makes the board strictly MORE conservative - it can only remove
# grades, never add them - which is why it was safe to do while every ticker is
# being re-evaluated against the new absolute-return target anyway.
#
# An anti-signal is still REPORTED, in its own branch below, because "reliably
# backwards" is worth a reader's attention and is not the same statement as
# "indistinguishable from noise". It just does not count as a check passed.
MIN_IC_TSTAT = 2.0
MIN_HIT_RATE_EDGE_PP = 1.0     # percentage points above the majority baseline

# How many of those checks a forecast must pass to earn each grade.
#
# WEAK used to require ONE of three, which meant the rank-IC floor alone was
# enough — and at +0.02 that floor is very low. The result was a board
# whose visible top was carried by single, statistically insignificant
# correlations: on 2026-08-15, BEL.NS ranked 3rd on a rank IC of +0.049 while
# its hit rate sat 4.7pp BELOW the majority-class baseline, its IC t-statistic
# was +0.39 (indistinguishable from noise), and its mean absolute error was
# worse than a random walk's. Twelve more names held the same badge on the same
# basis. A validation badge that one weak correlation can buy is not reporting
# evidence, it is laundering it — the same shape of overclaim Phase 0 was
# convened to remove, just milder than the leaked in-sample metrics were.
#
# Requiring two independent checks cuts WEAK from 13 names to 5 on that day's
# numbers (DMART, BAJAJFINSV, CANBK, ADANIPOWER, BAJFINANCE — each with both a
# positive IC and a hit rate above its baseline). In practice this means "IC
# and hit-rate edge", because the t-statistic check almost never passes at this
# horizon; that is a fact about the signal, not a reason to lower the bar.
MIN_CHECKS_FOR_WEAK = 2

# STRONG additionally requires that all three checks actually RAN. Grading on
# `passed == checks` alone would hand STRONG to a ticker whose only available
# metric happened to clear its floor, which is a statement about missing data
# rather than about skill.
REQUIRED_CHECKS_FOR_STRONG = 3



def grade_evidence(state: dict) -> tuple[str, list[str]]:
    """
    Grades a forecast on held-out evidence alone.

    Returns ``(grade, reasons)`` where grade is STRONG, WEAK or INSUFFICIENT.
    Pure function of the evaluation metrics — no LLM, no network, unit-testable.
    """
    if not state.get("forecast_available"):
        return "INSUFFICIENT", [state.get("forecast_error") or "no forecast produced"]

    ic = state.get("eval_rank_ic")
    ic_t = state.get("eval_rank_ic_t")
    hit = state.get("eval_hit_rate")
    baseline = state.get("eval_baseline_hit_rate")

    reasons: list[str] = []
    passed = 0
    checks = 0

    if ic is not None:
        checks += 1
        if ic >= MIN_RANK_IC:
            passed += 1
            reasons.append(f"Out-of-sample rank IC {ic:+.3f} clears the {MIN_RANK_IC:+.2f} floor.")
        else:
            reasons.append(f"Out-of-sample rank IC {ic:+.3f} is below the {MIN_RANK_IC:+.2f} floor.")

    if ic_t is not None:
        checks += 1
        if ic_t >= MIN_IC_TSTAT:
            passed += 1
            reasons.append(f"Rank IC t-statistic {ic_t:+.2f} is distinguishable from noise.")
        elif ic_t <= -MIN_IC_TSTAT:
            reasons.append(
                f"Rank IC t-statistic {ic_t:+.2f} is significantly NEGATIVE: the "
                f"out-of-sample ranking is reliably backwards. That is a real "
                f"measurement and it counts AGAINST this forecast, not for it."
            )
        else:
            reasons.append(f"Rank IC t-statistic {ic_t:+.2f} is within noise.")

    if hit is not None and baseline is not None:
        checks += 1
        edge = hit - baseline
        if edge >= MIN_HIT_RATE_EDGE_PP:
            passed += 1
            reasons.append(
                f"Hit rate {hit:.1f}% beats the majority-class baseline "
                f"{baseline:.1f}% by {edge:.1f}pp."
            )
        else:
            reasons.append(
                f"Hit rate {hit:.1f}% does not beat the majority-class baseline "
                f"{baseline:.1f}% ({edge:+.1f}pp)."
            )

    evaluated_at = state.get("eval_evaluated_at")

    if checks == 0:
        reason = ("No held-out evaluation metrics available yet — this ticker "
                 "has not been through a weekly evaluation run.")
        return "INSUFFICIENT", [reason]

    if checks >= REQUIRED_CHECKS_FOR_STRONG and passed == checks:
        grade = "STRONG"
    elif passed >= MIN_CHECKS_FOR_WEAK:
        grade = "WEAK"
    else:
        grade = "INSUFFICIENT"

    reasons.append(
        f"{passed} of {checks} held-out checks passed "
        f"({MIN_CHECKS_FOR_WEAK} needed for WEAK, "
        f"{REQUIRED_CHECKS_FOR_STRONG} for STRONG)."
    )

    # The evaluation behind this forecast is refreshed WEEKLY, not daily
    # (Lever 1) — surfaced here rather than left implicit, since the badge
    # sits next to a price that regenerates every day.
    if evaluated_at:
        reasons.append(f"Evidence last measured {evaluated_at}.")

    # The dashboard shows a rupee price target, which depends on the forecast's
    # MAGNITUDE, while rank IC and hit rate only establish ORDERING and
    # DIRECTION. If mean absolute error is worse than forecasting zero excess
    # return, the magnitude carries no information and the grade is capped —
    # otherwise a STRONG badge would sit next to a price the model cannot
    # actually justify.
    if state.get("eval_beats_naive") is False:
        reasons.append(
            "Mean absolute error is worse than forecasting zero excess return, so "
            "the magnitude carries no information even where the ranking does. "
            "Grade capped at WEAK: treat the ranking as the signal and the rupee "
            "target as illustrative."
        )
        if grade == "STRONG":
            grade = "WEAK"

    return grade, reasons


def critic_node(state: AgentState) -> dict:
    """
    Grades held-out evidence. The verdict is a relabelling of that grade.

    THE LLM SIGNAL REVIEW IS GONE, RETIRED ON MEASUREMENT (2026-09-04).

    It could reach a published row through exactly one channel — a flag
    downgrading APPROVED to FLAGGED — and APPROVED requires STRONG, which
    requires 3 of 3 evidence checks. Audited over all 1,152 forecast rows and
    four model versions by `tools/audit_critic_effect.py`:

        model_version                  rows  gate open  flagged  CHANGED
        (null, pre-P0)                  207          0       50        0
        phase0-excess-return-v1         270         19      118        0
        phase1-benchmark-audited-v2     423         13        0        0
        rebuild-absolute-return-v3      252          6        4        0
        TOTAL                          1152         38      172        0

    NO ROW HAS EVER BEEN GRADED STRONG. The review ran on 38 rows, raised flags
    on 172 across the project's life, and moved ZERO of them. It was not merely
    unhelpful, it was structurally incapable of helping.

    The gate that gated it rested on two closed paths — the verdict and the
    composite score — and the score went with the ranking layer, leaving one
    leg. The audit settles the remaining leg by measurement: the leg holds, and
    it holds so completely that the thing it was gating had no reachable effect.

    The alternative was to re-point the flags at the written narrative, where a
    reader would see them. That is a real option and was not taken here; the
    LLM budget is spent on the narrative itself instead, which every reader
    already sees. Retiring the dead path first keeps the two decisions separate.

    What is left is deterministic: the verdict IS the grade, renamed. That
    redundancy is deliberate and visible rather than hidden behind a step that
    might have changed something.
    """
    grade, reasons = grade_evidence(dict(state))

    verdict = {"STRONG": "APPROVED", "WEAK": "FLAGGED",
               "INSUFFICIENT": "REJECTED"}[grade]

    return {
        "evidence_grade": grade,
        "evidence_reasons": reasons,
        "critic_verdict": verdict,
        "critic_reasoning": " ".join(reasons),
        # Kept as an EMPTY LIST rather than removed. `agents/graph.py` writes
        # this column and the API serialises it; a missing key would become a
        # null where every historical row holds `[]`, and a reader cannot tell
        # "no flags" from "the field stopped being written".
        "critic_flags": [],
        "critic_source": "evidence_gate",
    }
