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

import json
import os
import re

from dotenv import load_dotenv

from agents.llm import DEFAULT_GROQ_MODEL, groq_client
from agents.state import EVIDENCE_MULTIPLIER, AgentState

load_dotenv(override=True)

# Thresholds for the deterministic gate. Deliberately modest: a rank IC of 0.03
# with a t-statistic above 2 is a real but weak edge, which is an honest
# description of what technical signals deliver at a 30-session horizon.
MIN_RANK_IC = 0.02
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


_groq_client = groq_client       # re-exported; see agents/llm.py


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
        if abs(ic_t) >= MIN_IC_TSTAT:
            passed += 1
            reasons.append(f"Rank IC t-statistic {ic_t:+.2f} is distinguishable from noise.")
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


def _llm_review(state: dict, ticker: str) -> tuple[list[str], str]:
    """
    Asks the LLM to flag internal contradictions in the signal snapshot.

    Returns ``(flags, reasoning)``. Failure returns no flags rather than a
    default verdict, so an API outage cannot silently change a stock's grade.
    """
    client = _groq_client()
    if client is None:
        return [], "LLM review skipped (no API key configured)."

    prompt = f"""You are reviewing the inputs to a 30-session relative-return forecast for an Indian (NSE) stock.

Stock: {state.get('company_name', ticker)} ({ticker})
Benchmark: {state.get('benchmark_ticker')}
Predicted excess return: {state.get('pred_return')}
Calibrated probability the stock RISES (unconditional base rate on this universe is 0.577): {state.get('prob_up')}

Signal snapshot:
{state.get('latest_signals', {})}

Narrative: {state.get('signal_narrative', '')}

Raise a flag ONLY where a specific, checkable contradiction is present:
1. SIGNAL CONFLICT — RSI above 75 AND MACD histogram strongly negative, simultaneously.
2. DIRECTION CONFLICT — the narrative describes clear momentum in the opposite
   direction to the predicted excess return.
3. THIN TRADING — OBV essentially flat across the window AND volume ROC near zero.
4. STALE OR DEGENERATE INPUT — signal values that are constant, zero, or implausible.

You are reviewing inputs, not certifying the forecast. You cannot approve anything.
Do not comment on whether the model is accurate; you have no way to verify that.

Respond ONLY with JSON:
{{"flags": ["..."], "reasoning": "2-3 sentences"}}"""

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.2,
        )
        raw = completion.choices[0].message.content.strip()
    except Exception as exc:                                   # noqa: BLE001
        return [], f"LLM review unavailable: {exc}"

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return [], "LLM response was not valid JSON; no flags recorded."

    flags = parsed.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    return [str(f) for f in flags], str(parsed.get("reasoning", ""))


def critic_node(state: AgentState) -> dict:
    """
    Grades held-out evidence, then lets the LLM add flags that can only downgrade.

    The LLM is called only where a flag could change the published row. See the
    comment below for why that is arithmetic rather than an optimisation.
    """
    ticker = state["ticker"]

    grade, reasons = grade_evidence(dict(state))

    # THE LLM ONLY RUNS WHERE ITS OUTPUT CAN CHANGE THE VERDICT.
    #
    # This gate was built when flags reached the published board through two
    # paths, and it was justified by BOTH being closed for an INSUFFICIENT
    # grade: the verdict (flags only downgrade APPROVED, which requires
    # STRONG) and the score (compute_composite_score multiplied by
    # EVIDENCE_MULTIPLIER before deducting per flag, and the multiplier is 0.0
    # for INSUFFICIENT, so zero times anything less a deduction is zero).
    #
    # THE SCORE PATH IS GONE with the ranking layer, so the argument now rests
    # on the verdict alone. That is still sound — a flag cannot move a REJECTED
    # row — and it still saves ~91 of 95 Groq calls a day. But it is one leg
    # rather than two, and it is narrower than it looks: what a flag can no
    # longer change is a NUMBER, and the P4 forecast object is supposed to
    # explain in words why a stock does or does not work. A raised flag is
    # exactly that kind of content. When the written analysis lands, revisit
    # whether "cannot change the verdict" is still the right test, or whether
    # the cost is now buying something a reader wants to see.
    #
    # EVIDENCE_MULTIPLIER is retained as the single definition of "INSUFFICIENT
    # means nothing survives". It no longer multiplies anything.
    if EVIDENCE_MULTIPLIER.get(grade, 0.0) > 0.0:
        flags, llm_reasoning = _llm_review(dict(state), ticker)
    else:
        # Recorded, not silent. A skipped step that says nothing reads as a step
        # that ran and found nothing wrong.
        flags = []
        llm_reasoning = (f"LLM signal review skipped: evidence grade {grade} "
                         f"scores zero, so no flag it raised could change this "
                         f"row.")

    # Map evidence grade to a verdict, then apply LLM flags as a downgrade only.
    verdict = {"STRONG": "APPROVED", "WEAK": "FLAGGED", "INSUFFICIENT": "REJECTED"}[grade]
    source = "evidence_gate"

    if flags and verdict == "APPROVED":
        verdict = "FLAGGED"
        source = "evidence_gate+llm_flags"
    elif flags:
        source = "evidence_gate+llm_flags"

    reasoning = " ".join(reasons)
    if llm_reasoning:
        reasoning = f"{reasoning} LLM signal review: {llm_reasoning}"

    return {
        "evidence_grade": grade,
        "evidence_reasons": reasons,
        "critic_verdict": verdict,
        "critic_reasoning": reasoning,
        "critic_flags": flags,
        "critic_source": source,
    }
