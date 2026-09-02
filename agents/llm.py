"""
agents/llm.py — shared Groq client and model selection.

Both LLM-using nodes (the narrative writer and the critic's input review) had
their own copy of the client constructor and their own hardcoded model name,
which meant the model was configured in two places and could drift between
them. It is one place now.

MODEL CHOICE, decided 2026-09-02. ``openai/gpt-oss-120b`` on Groq.

The previous default, ``llama-3.1-8b-instant``, was chosen for its allowance
(500,000 tokens/day, 14,400 requests/day) after ``openai/gpt-oss-20b`` ran out
of tokens partway through the 2026-08-15 run. It then began returning
``404 - model does not exist`` on EVERY call, which is what fired the stale
``_rule_based_narrative`` call site and failed 64 of 95 tickers while the job
reported OK. That defect is fixed; the dead model id is what is being replaced
here.

WHY NOT OPENROUTER. It was researched as a migration target and refused on one
number: OpenRouter's free tier is capped at **50 requests per day** on an
unfunded account. The universe is 84 stocks. The daily job cannot run inside
that at all - not "would be tight", cannot. The cap rises to 1,000/day
permanently after a one-time $10 credit purchase, which makes OpenRouter viable
but not free, and nothing there is worth $10 that Groq is not giving away.

WHY THIS MODEL ON GROQ. Free-tier allowances, measured against the workload:

    limit                gpt-oss-120b     what the daily job needs
    requests / minute    30               paced, not binding
    requests / day       1,000            ~12 narratives + ~4 critic reviews
    tokens / minute      8,000            ~600/call, so ~13 calls/min
    tokens / day         200,000          ~10,000

The 200k TPD is 2.5x smaller than the 8B model's 500k, and that is not a
problem any more: the narrative gate (forecasting_agent._narrative_sample) caps
written narratives at a fixed daily sample and the critic's LLM review is gated
on the evidence grade, so the daily total is ~16 calls rather than ~190. The
budget was the constraint when every ticker made two unconditional calls. It no
longer is, which is what makes it affordable to spend the allowance on a much
better model instead of a much larger allowance.

TPM is now the binding limit, not TPD - 8,000 tokens/minute paces the run to
roughly 13 calls a minute. A full sample is one minute of a job that already
takes forty.

It is a reasoning model with configurable effort, so its reasoning tokens count
against the same budget. Both prompts here are short and bounded (a
three-sentence signal summary and a contradiction check), which is why the
default effort is left alone rather than pinned low.

Override with the GROQ_MODEL environment variable.
"""

from __future__ import annotations

import os
import re

from groq import Groq

# A reasoning model may emit its chain of thought INSIDE the message content
# rather than in a separate field, depending on the provider's default
# reasoning format. Neither consumer here can survive that quietly:
#
#   the narrative   publishes `content` verbatim to the dashboard, so the
#                   model's private working would be presented to a reader as
#                   the analysis.
#   the critic      json.loads() the content, so a prefix makes every review
#                   land in the "not valid JSON" branch - which records NO
#                   flags and reads, from the outside, exactly like a clean
#                   review that found nothing wrong.
#
# The second is the dangerous one: a silent failure that looks like a pass.
# Stripping is a no-op when the content is already clean, which is the case for
# a provider that separates the fields, so this costs nothing to be wrong about.
# No word boundary on the opening tag here, and that is deliberate rather
# than an oversight: the CLOSING side already requires `</think\s*>`, so a
# tag that merely starts with the keyword (`<thinker>x</thinker>`) can never
# match whether the boundary is present or not. A mutation test proved it -
# removing the boundary changed no behaviour any input could reach, which is
# this codebase's definition of a guard that is really a second copy of one
# that already works. The boundary IS load-bearing on the truncation rule
# below, which has no closing tag to constrain it.
_REASONING_BLOCK = re.compile(
    r"<(?:think|thinking|reasoning)[^>]*>.*?</(?:think|thinking|reasoning)\s*>",
    re.DOTALL | re.IGNORECASE,
)

# An UNCLOSED opening tag means the response was truncated mid-thought: there
# is no answer after it to keep, and whatever precedes it is a preamble at
# best. Anchored to the END of the string so a well-formed block above has
# already been removed by the time this runs.
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


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def groq_client() -> Groq | None:
    """Returns a client, or None when no usable key is configured."""
    key = os.getenv("GROQ_API_KEY", "").strip('"').strip("'")
    if not key or key == "your_groq_key_here":
        return None
    return Groq(api_key=key)


def groq_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
