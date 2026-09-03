"""
tests/test_phase3_llm.py — the provider routing table.

This exists because of a measured production failure, not a hypothetical one.
`llama-3.1-8b-instant` was decommissioned at Groq, every call began returning
404, and the daily job failed 64 of 95 tickers for three consecutive days while
reporting OK each time. One place holding one model name has no answer to "that
name stopped existing", and OpenRouter is MORE exposed to this than Groq: its
free lineup rotates, and DeepSeek and Mistral both had free variants that are
now gone.

So the guards here are about degradation, not about model quality:

  - a dead first route must fall through, not fail
  - a route that answers with nothing but reasoning is a DEAD route, not an
    empty answer to return
  - every answer carries which model produced it
  - preflight checks the chain before a run rather than during one
"""

from __future__ import annotations

import json

import pytest

import agents.llm as llm
from agents.llm import (
    GROQ,
    OPENROUTER,
    ROUTES,
    NoRouteAvailable,
    Route,
    complete,
    extract_json,
    preflight,
)


@pytest.fixture
def callers(monkeypatch):
    """Replaces the provider seam; returns a dict the test fills in."""
    table: dict = {}

    def make(provider):
        def call(route, prompt, temperature, schema, timeout):
            handler = table.get((provider, route.model), table.get(provider))
            if handler is None:
                raise RuntimeError(f"no stub for {provider}:{route.model}")
            if isinstance(handler, Exception):
                raise handler
            return handler(route, prompt, schema) if callable(handler) else handler
        return call

    monkeypatch.setattr(llm, "_CALLERS",
                        {GROQ: make(GROQ), OPENROUTER: make(OPENROUTER)})
    return table


# ── Fallback ──────────────────────────────────────────────────────────────────

def test_a_dead_first_route_falls_through_to_the_next(callers):
    """
    THE 64-OF-95 FAILURE. A decommissioned model must cost one fallback, not a
    run. The chain is data, so this needs no try/except at the call site.
    """
    first, second = ROUTES["critic"][0], ROUTES["critic"][1]
    callers[(first.provider, first.model)] = RuntimeError(
        "HTTP 404: model not found")
    callers[(second.provider, second.model)] = '{"ok": true}'

    out = complete("critic", "hello")
    assert out.model == second.model
    assert out.attempts, "the failed attempt must be recorded, not swallowed"
    assert "404" in out.attempts[0]


def test_the_chain_ends_on_the_provider_we_have_actually_run():
    """
    OpenRouter's free lineup rotates; Groq's free tier is a published product.
    Every chain's LAST hop is therefore Groq, so a rotation cannot take out a
    task entirely.
    """
    for task, routes in ROUTES.items():
        assert routes[-1].provider == GROQ, (
            f"{task}'s last resort is {routes[-1].provider}, which can rotate "
            f"its free models out from under us"
        )


def test_every_route_failing_raises_rather_than_returning_something(callers):
    """
    A caller that cannot reach a model must degrade VISIBLY — to the
    deterministic narrative, or to no flags. Returning a default would publish
    something that looks like a reading and is not.
    """
    for route in ROUTES["narrative"]:
        callers[(route.provider, route.model)] = RuntimeError("down")

    with pytest.raises(NoRouteAvailable) as exc:
        complete("narrative", "hello")
    # Every attempt named, or a post-mortem has nothing to read.
    for route in ROUTES["narrative"]:
        assert route.model in str(exc.value)


def test_an_unknown_task_is_refused_rather_than_defaulted(callers):
    """
    A typo'd task name must not quietly borrow another task's model — that
    would route, say, bulk relevance work onto the scarce reasoning budget with
    nothing to see.

    EVERY provider is stubbed to SUCCEED here, deliberately. The first version
    of this test stubbed nothing, so a mutant that fell back to
    `ROUTES["narrative"]` still raised — because that chain had no stub either
    — and the test passed against the exact defect it was written to catch.
    """
    callers[GROQ] = "an answer"
    callers[OPENROUTER] = "an answer"
    assert complete("narrative", "hello").text == "an answer"   # the seam works

    with pytest.raises(NoRouteAvailable) as exc:
        complete("narrativve", "hello")
    assert "narrativve" in str(exc.value)


def test_a_response_that_is_all_reasoning_is_a_dead_route_not_an_answer(callers):
    """
    A reasoning model that spends its whole response thinking returns an empty
    string after stripping. Returning that would publish a blank narrative;
    treating it as this route's failure moves on to one that answers.
    """
    first, second = ROUTES["narrative"][0], ROUTES["narrative"][1]
    callers[(first.provider, first.model)] = "<think>hmm, let me consider</think>"
    callers[(second.provider, second.model)] = "A real three-sentence answer."

    out = complete("narrative", "hello")
    assert out.text == "A real three-sentence answer."
    assert out.model == second.model
    assert "empty after stripping" in out.attempts[0]


# ── Provenance ────────────────────────────────────────────────────────────────

def test_every_answer_records_which_model_produced_it(callers):
    """
    Without this a model swap silently changes the published board and nothing
    on record says so — the same defect as an unversioned MODEL_VERSION, where
    a metric measured against one target kept backing forecasts made against
    another.
    """
    route = ROUTES["regime"][0]
    callers[(route.provider, route.model)] = "clear skies"

    out = complete("regime", "hello")
    assert out.provider == route.provider and out.model == route.model
    assert out.source == f"{route.provider}:{route.model}"


# ── Task placement ────────────────────────────────────────────────────────────

def test_the_high_volume_task_is_on_the_provider_with_request_headroom():
    """
    Groq caps TOKENS (8k/min, 200k/day) and OpenRouter caps REQUESTS (20/min,
    50/day unfunded). Relevance filtering is many short calls, so it belongs on
    the one that does not count requests tightly — putting it on OpenRouter
    would spend the entire daily reasoning budget on mechanical work.
    """
    assert all(r.provider == GROQ for r in ROUTES["relevance"])


def test_the_critic_is_routed_to_a_model_that_can_be_given_a_schema():
    """
    The critic json.loads() its answer and a parse failure records NO FLAGS —
    indistinguishable from a clean review. Only two free OpenRouter models
    advertise structured outputs; the critic's PREFERRED route must be one of
    them, or the schema in critic_agent is decoration.
    """
    assert ROUTES["critic"][0].structured, (
        "the critic's first route cannot be handed a JSON schema, so its "
        "silent-failure mode is merely handled rather than unconstructable"
    )


# ── extract_json ──────────────────────────────────────────────────────────────

def test_unreadable_json_is_none_and_not_an_empty_dict():
    """
    "The model reported nothing" and "the model's answer was unreadable" are
    different states, and collapsing them is exactly how a broken critic review
    reads as a clean one. Same rule that makes get_aggregate_sentiment return
    None rather than 0.0.
    """
    assert extract_json("not json at all") is None
    assert extract_json("[1, 2, 3]") is None, "a list is not a review"
    assert extract_json('{"flags": []}') == {"flags": []}

    fenced = "```json\n" + json.dumps({"flags": ["X"], "reasoning": "y"}) + "\n```"
    assert extract_json(fenced) == {"flags": ["X"], "reasoning": "y"}
    assert extract_json("```\n{\"a\": 1}\n```") == {"a": 1}


# ── Preflight ─────────────────────────────────────────────────────────────────

def test_preflight_reports_a_dead_route_before_the_run(callers, monkeypatch):
    """
    `series_kaggle.preflight()` exists because TimesFM failed on the fourth of
    five configurations, an hour in, on a check that took two seconds. Same
    rule: verify the chain before the ticker loop, not during it.
    """
    monkeypatch.setattr(llm, "groq_client", lambda: object())
    monkeypatch.setattr(llm, "openrouter_key", lambda: "test-key")

    dead = ROUTES["critic"][0]
    callers[(dead.provider, dead.model)] = RuntimeError("HTTP 404: no such model")
    callers[GROQ] = "ok"
    callers[OPENROUTER] = "ok"

    report = preflight(["critic"], probe=True)
    statuses = {r["model"]: r["status"] for r in report["routes"]}
    assert "404" in statuses[dead.model]
    # One dead route does not kill the task — that is what the chain is for.
    assert "critic" in report["usable_tasks"]


def test_preflight_says_when_a_task_has_no_route_left(callers, monkeypatch):
    monkeypatch.setattr(llm, "groq_client", lambda: object())
    monkeypatch.setattr(llm, "openrouter_key", lambda: "test-key")
    for provider in (GROQ, OPENROUTER):
        callers[provider] = RuntimeError("down")

    report = preflight(["narrative"], probe=True)
    assert report["dead_tasks"] == ["narrative"]
    assert report["usable_tasks"] == []


def test_preflight_reports_the_openrouter_request_budget(monkeypatch):
    """
    50 requests/day unfunded is the binding constraint on the whole reasoning
    tier, and it is a limit we chose to run against deliberately. Reporting the
    estimate beside it is how we learn it binds before it breaks.
    """
    monkeypatch.delenv("OPENROUTER_FUNDED", raising=False)
    monkeypatch.setenv("NARRATIVE_SAMPLE_SIZE", "12")
    report = preflight([], probe=False)
    assert "50" in report["openrouter_daily_budget"]
    assert report["reasoning_calls_per_day_estimate"] == 17

    monkeypatch.setenv("OPENROUTER_FUNDED", "1")
    assert "1000" in preflight([], probe=False)["openrouter_daily_budget"]


def test_preflight_can_check_configuration_without_spending_the_budget(monkeypatch):
    """
    Probing costs a real request per distinct route against a 50/day cap. A
    test, or a dry run, must be able to check the table without paying for it.
    """
    monkeypatch.setattr(llm, "groq_client", lambda: object())
    monkeypatch.setattr(llm, "openrouter_key", lambda: "k")

    called = []
    monkeypatch.setattr(llm, "_CALLERS", {
        GROQ: lambda *a: called.append(1),
        OPENROUTER: lambda *a: called.append(1),
    })
    report = preflight(["critic"], probe=False)
    assert called == [], "probe=False must send no requests"
    assert all(r["status"].startswith("configured") for r in report["routes"])


def test_an_unreadable_review_is_recorded_as_unreadable_not_as_clean(callers):
    """
    THE SILENT FAILURE THIS WHOLE ROUTE EXISTS FOR.

    `_llm_review` json.loads() the model's answer. When that fails it records NO
    FLAGS — which, from the board, is indistinguishable from a careful review
    that found nothing wrong. A failure that reads as a pass.

    So "no flags" is not enough to assert. The RECORD has to say the answer was
    unreadable, and name the model that produced it, or a recurring offender is
    averaged into silence.
    """
    from agents.critic_agent import _llm_review

    callers[GROQ] = "I think the RSI looks fine, honestly."
    callers[OPENROUTER] = "I think the RSI looks fine, honestly."

    flags, reasoning = _llm_review({"ticker": "ABB.NS"}, "ABB.NS")
    assert flags == []
    assert "not valid JSON" in reasoning, (
        "an unreadable review must not be recorded the same way as a review "
        "that genuinely found nothing"
    )
    assert "openrouter:" in reasoning or "groq:" in reasoning


def test_a_missing_key_is_reported_as_such_rather_than_as_a_dead_model(monkeypatch):
    """
    "No API key" and "the model is gone" need different fixes, so they must not
    share a message. Conflating them is what sent a reader to the wrong
    remediation on Kaggle.
    """
    monkeypatch.setattr(llm, "groq_client", lambda: None)
    monkeypatch.setattr(llm, "openrouter_key", lambda: None)

    report = preflight(["critic"], probe=True)
    assert {r["status"] for r in report["routes"]} == {"no api key"}
    assert report["keys"] == {GROQ: False, OPENROUTER: False}
