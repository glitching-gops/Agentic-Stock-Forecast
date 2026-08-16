"""
agents/llm.py — shared Groq client and model selection.

Both LLM-using nodes (the narrative writer and the critic's input review) had
their own copy of the client constructor and their own hardcoded model name,
which meant the model was configured in two places and could drift between
them. It is one place now.

MODEL CHOICE. The default was ``openai/gpt-oss-20b``, whose Groq free-tier
allowance is 200,000 tokens/day and 1,000 requests/day. The daily job makes two
calls per ticker, so at 95 tickers that is 190 requests and a budget of roughly
1,050 tokens per call — and the 2026-08-15 run duly ran out partway through,
with the tail of the universe logging ``narrative generation failed`` and
falling back to the deterministic narrative.

``llama-3.1-8b-instant`` carries 500,000 tokens/day and 14,400 requests/day on
the same free tier: 2.5x the tokens and 14.4x the requests, at no cost. Nothing
else on the free tier beats 200k TPD — gpt-oss-120b and qwen3.6-27b match it,
and llama-3.3-70b-versatile is half of it.

It is a smaller model, and that is a real trade: it writes the three-sentence
signal summary and the input-contradiction check, neither of which is a
reasoning-heavy task, but neither is it free of quality risk. The narrative gate
in forecasting_agent._narrative matters more than the model swap for staying
inside the budget; the swap is headroom on top of it.

Override with the GROQ_MODEL environment variable.
"""

from __future__ import annotations

import os

from groq import Groq

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def groq_client() -> Groq | None:
    """Returns a client, or None when no usable key is configured."""
    key = os.getenv("GROQ_API_KEY", "").strip('"').strip("'")
    if not key or key == "your_groq_key_here":
        return None
    return Groq(api_key=key)


def groq_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
