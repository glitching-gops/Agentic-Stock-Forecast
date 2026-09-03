"""
agents/llm.py — provider routing, one table, with a preflight.

WHY THIS IS A TABLE AND NOT A MODEL NAME
-----------------------------------------
It used to be one constant. Then `llama-3.1-8b-instant` was decommissioned at
Groq, every call began returning 404, and the daily job failed 64 of 95 tickers
for three consecutive days while reporting OK each time. The model id was
configured in one place, which was the fix at the time — but one place holding
one name still has no answer to "that name stopped existing".

So: a task maps to an ORDERED LIST of (provider, model). The fallback is data,
not a try/except at each call site, and `preflight()` verifies the chain before
the ticker loop rather than discovering it 95 tickers in. That is the same
lesson `tools/series_kaggle.preflight()` records - TimesFM failed on the fourth
of five configurations, an hour into a run, on a dependency that could have
been checked in two seconds.

THE TWO PROVIDERS CAP DIFFERENT RESOURCES, WHICH IS THE WHOLE DESIGN
--------------------------------------------------------------------
Measured 2026-09-03:

                        Groq free       OpenRouter free
    requests / minute   30              20
    requests / day      1,000           50 unfunded, 1,000 after a one-time $10
    tokens / minute     8,000           NONE
    tokens / day        200,000         NONE

Groq caps TOKENS and is generous with requests. OpenRouter caps REQUESTS and
does not count tokens at all. So the split is not "the better model for the
harder job", it is which resource each job actually spends:

  FEW CALLS, LONG PROMPTS, HARD THINKING  -> OpenRouter.
      A critic review carrying a full signal snapshot costs tokens, not
      requests. On Groq the 8k tokens/minute would pace those to ~13 a minute
      and spend a fifth of the daily budget; OpenRouter does not count them.

  MANY CALLS, SHORT PROMPTS, MECHANICAL   -> Groq.
      84 batched relevance calls at ~400 tokens is 34k tokens - trivial for
      Groq, and 8% of OpenRouter's ENTIRE daily request budget for nothing.

Running unfunded on OpenRouter is a deliberate choice, not an oversight. The
reasoning tier is ~17 calls a day (12 narratives + ~4 critic reviews + 1 regime
read) against a 50/day cap. That fits, with 33 to spare, and `preflight()`
reports the headroom so we learn the cap binds before it breaks rather than
after.

WHY `glm-5.2` HAS THE CRITIC
-----------------------------
The critic `json.loads()` the model's content, and a parse failure records NO
FLAGS - which is indistinguishable, from outside, from a clean review that
found nothing wrong. A silent failure that reads as a pass. `strip_reasoning`
below removes the most common cause; a JSON SCHEMA removes the failure mode
itself. Of the 18 free models on OpenRouter exactly two advertise structured
outputs: `z-ai/glm-5.2:free` and `nvidia/nemotron-3-super-120b-a12b:free`. That
is the argument - not benchmark scores.

`inclusionai/ling-3.0-flash-fin:free` takes the narrative instead: finance-tuned
(5.1B active of 124B), 262k context, and prose about a stock is exactly where a
domain-tuned model should show. It does NOT support structured outputs, so it
cannot have the critic.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Reasoning-block stripping ─────────────────────────────────────────────────
#
# A reasoning model may emit its chain of thought INSIDE the message content
# rather than in a separate field, depending on the provider's default. Neither
# consumer survives that quietly:
#
#   the narrative   publishes `content` verbatim to the dashboard, so the
#                   model's private working reaches a reader as the analysis.
#   the critic      json.loads() it, so a prefix drops every review into the
#                   "not valid JSON" branch, which records NO flags and reads
#                   exactly like a clean review.
#
# The second is the dangerous one. Stripping is a no-op on a provider that
# separates the fields, so it costs nothing to be wrong about.
#
# No word boundary on the opening tag, deliberately: the CLOSING side already
# requires `</think\s*>`, so `<thinker>x</thinker>` cannot match either way. A
# mutation test proved the boundary changed no reachable behaviour, which is
# this codebase's definition of a second copy of a guard that already works.
# It IS load-bearing on the truncation rule below, which has no closing tag.
_REASONING_BLOCK = re.compile(
    r"<(?:think|thinking|reasoning)[^>]*>.*?</(?:think|thinking|reasoning)\s*>",
    re.DOTALL | re.IGNORECASE,
)

# An UNCLOSED opening tag means the response was truncated mid-thought: there is
# no answer after it, and whatever precedes it is a preamble at best.
_UNCLOSED_REASONING = re.compile(
    r"<(?:think|thinking|reasoning)\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(text: str | None) -> str:
    """Removes any inline chain-of-thought block, leaving the answer."""
    if not text:
        return ""
    cleaned = _REASONING_BLOCK.sub("", text)
    cleaned = _UNCLOSED_REASONING.sub("", cleaned)
    return cleaned.strip()


# ── The routing table ─────────────────────────────────────────────────────────

GROQ = "groq"
OPENROUTER = "openrouter"

#: Kept for the two call sites that still name it, and because `.env.example`
#: documents it. The router is what the code should reach for.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    #: Whether this model can be handed a JSON schema. Only the critic needs it,
    #: and only two free OpenRouter models offer it - so a fallback that lacks
    #: it degrades to prompt-and-parse rather than failing, and SAYS so.
    structured: bool = False


#: task -> ordered fallback chain. First entry that answers wins.
#:
#: Every chain ends on Groq `gpt-oss-120b`, which is the one model this project
#: has actually run in production. OpenRouter's free lineup ROTATES - DeepSeek
#: and Mistral both had free variants and now have none - so the last hop is
#: deliberately the provider whose free tier is a published product rather than
#: a rotating pool.
ROUTES: dict[str, tuple[Route, ...]] = {
    "narrative": (
        Route(OPENROUTER, "inclusionai/ling-3.0-flash-fin:free"),
        Route(OPENROUTER, "z-ai/glm-5.2:free", structured=True),
        Route(GROQ, DEFAULT_GROQ_MODEL),
    ),
    "critic": (
        Route(OPENROUTER, "z-ai/glm-5.2:free", structured=True),
        Route(OPENROUTER, "nvidia/nemotron-3-super-120b-a12b:free", structured=True),
        Route(GROQ, DEFAULT_GROQ_MODEL),
    ),
    "regime": (
        Route(OPENROUTER, "z-ai/glm-5.2:free", structured=True),
        Route(GROQ, DEFAULT_GROQ_MODEL),
    ),
    # High volume, short prompts, mechanical. Groq's 1,000 tok/s and its
    # request headroom are what this needs; reasoning quality is not.
    "relevance": (
        Route(GROQ, "openai/gpt-oss-20b"),
        Route(GROQ, DEFAULT_GROQ_MODEL),
    ),
}


@dataclass
class Completion:
    """
    A model's answer, and WHICH MODEL GAVE IT.

    The provenance is not decoration. An LLM output that reaches a stored row
    without recording what produced it means a model swap silently changes the
    published board and nothing on record says so - the same defect as an
    unversioned MODEL_VERSION, where a metric measured against one target kept
    backing forecasts made against another.
    """

    text: str
    provider: str
    model: str
    attempts: list[str]

    @property
    def source(self) -> str:
        return f"{self.provider}:{self.model}"


class NoRouteAvailable(Exception):
    """Every route for a task failed. Callers degrade; they do not guess."""


# ── Providers ─────────────────────────────────────────────────────────────────

def groq_client():
    """Returns a Groq client, or None when no usable key is configured."""
    key = os.getenv("GROQ_API_KEY", "").strip('"').strip("'")
    if not key or key == "your_groq_key_here":
        return None
    from groq import Groq
    return Groq(api_key=key)


def openrouter_key() -> str | None:
    key = os.getenv("OPENROUTER_API_KEY", "").strip('"').strip("'")
    return key or None


def groq_model() -> str:
    """The legacy single-model accessor. Prefer `complete(task=...)`."""
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def _call_groq(route: Route, prompt: str, temperature: float,
               schema: dict | None, timeout: float) -> str:
    client = groq_client()
    if client is None:
        raise NoRouteAvailable("GROQ_API_KEY is not configured")
    kwargs: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "model": route.model,
        "temperature": temperature,
    }
    if schema and route.structured:
        kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


def _call_openrouter(route: Route, prompt: str, temperature: float,
                     schema: dict | None, timeout: float) -> str:
    """
    OpenAI-compatible chat completion over plain `requests`.

    No new dependency on purpose. `requests` is already pinned in
    requirements.txt and is installed on Render, in both workflows and locally;
    adding the `openai` package to reach an OpenAI-shaped endpoint would put a
    second HTTP stack on an instance that has already been OOM-killed once.
    """
    import requests

    key = openrouter_key()
    if not key:
        raise NoRouteAvailable("OPENROUTER_API_KEY is not configured")

    body: dict = {
        "model": route.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if schema and route.structured:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": schema},
        }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json=body, timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"no choices in response: {str(payload)[:200]}")
    return (choices[0].get("message") or {}).get("content") or ""


_CALLERS = {GROQ: _call_groq, OPENROUTER: _call_openrouter}


# ── The one entry point ───────────────────────────────────────────────────────

def complete(task: str, prompt: str, temperature: float = 0.3,
             schema: dict | None = None, timeout: float = 60.0) -> Completion:
    """
    Runs `prompt` through the first route for `task` that answers.

    Raises NoRouteAvailable when the whole chain fails, carrying every attempt's
    reason. It never returns a fabricated or default answer: a caller that
    cannot reach a model must degrade visibly - to the deterministic narrative,
    or to no flags - rather than publish something that looks like a reading.
    """
    routes = ROUTES.get(task)
    if not routes:
        raise NoRouteAvailable(f"no route configured for task {task!r}")

    attempts: list[str] = []
    for route in routes:
        try:
            raw = _CALLERS[route.provider](route, prompt, temperature, schema,
                                           timeout)
        except Exception as exc:                                # noqa: BLE001
            attempts.append(f"{route.provider}:{route.model} -> "
                            f"{type(exc).__name__}: {str(exc)[:120]}")
            continue

        text = strip_reasoning(raw)
        if not text:
            # An empty answer after stripping means the model spent the entire
            # response reasoning. That is a failure of this route, not of the
            # task - so try the next one rather than returning a blank.
            attempts.append(f"{route.provider}:{route.model} -> empty after "
                            f"stripping reasoning")
            continue

        if attempts:
            logger.warning("[llm] %s fell through to %s:%s (%s)",
                           task, route.provider, route.model, "; ".join(attempts))
        return Completion(text=text, provider=route.provider, model=route.model,
                          attempts=attempts)

    raise NoRouteAvailable(f"every route for {task!r} failed: " + "; ".join(attempts))


def extract_json(text: str) -> dict | None:
    """
    Parses a model's JSON answer, tolerating a code fence around it.

    Returns None on failure rather than {} - the caller has to be able to tell
    "the model reported nothing" from "the model's answer was unreadable", and
    collapsing those is precisely how a broken critic review reads as a clean
    one.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ── Preflight ─────────────────────────────────────────────────────────────────

def preflight(tasks: list[str] | None = None, probe: bool = True) -> dict:
    """
    Checks every configured route BEFORE a run, not 95 tickers into one.

    Two failures this exists for, both already paid for. Groq decommissioned a
    model under this project and 404'd every call for days while the job
    reported OK. And OpenRouter's free lineup ROTATES, so a model named here can
    stop existing without any change on our side.

    `probe=True` sends a one-token request per DISTINCT route. That is a handful
    of calls against a 50/day budget, so it is cheap but not free - the daily
    job should probe, and a unit test should not.
    """
    report: dict = {"keys": {}, "routes": [], "usable_tasks": [], "dead_tasks": []}
    report["keys"][GROQ] = groq_client() is not None
    report["keys"][OPENROUTER] = openrouter_key() is not None

    checked: dict[Route, str] = {}
    for task in (tasks or list(ROUTES)):
        alive = []
        for route in ROUTES.get(task, ()):
            if route not in checked:
                if not report["keys"].get(route.provider):
                    checked[route] = "no api key"
                elif not probe:
                    checked[route] = "configured (not probed)"
                else:
                    try:
                        _CALLERS[route.provider](route, "ok", 0.0, None, 20.0)
                        checked[route] = "ok"
                    except Exception as exc:                    # noqa: BLE001
                        checked[route] = f"{type(exc).__name__}: {str(exc)[:100]}"
            status = checked[route]
            report["routes"].append(
                {"task": task, "provider": route.provider, "model": route.model,
                 "status": status})
            if status.startswith(("ok", "configured")):
                alive.append(f"{route.provider}:{route.model}")
        (report["usable_tasks"] if alive else report["dead_tasks"]).append(task)

    # The OpenRouter request budget is the binding constraint on the reasoning
    # tier and it is a DAILY one, so it is reported rather than assumed.
    report["openrouter_daily_budget"] = (
        "1000 (funded)" if os.getenv("OPENROUTER_FUNDED") else "50 (unfunded)")
    report["reasoning_calls_per_day_estimate"] = (
        int(os.getenv("NARRATIVE_SAMPLE_SIZE", "12")) + 5)
    return report
