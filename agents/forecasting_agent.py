"""
agents/forecasting_agent.py — Generates the forecast and a signal narrative.

The warm path is gone. ``_generate_forecast_from_existing`` wrote a column
named ``sentiment`` while ``FEATURES`` expected ``sentiment_score``, so
``dropna(subset=FEATURES)`` raised ``KeyError``, a bare ``except`` swallowed it,
and the caller's fallback persisted ``forecast_price = current_price`` with
``mape = 100`` as though it were a real forecast (audit finding F10). Any second
pipeline pass in a day wrote a fabricated flat forecast with no error surfaced.

Rather than repair that path, it is removed. ``pipeline.model.forecast_ticker_daily``
is the single entry point, and it validates its own feature frame. Re-running is
idempotent and cheap enough; a silently wrong number is not.

This graph node calls the DAILY (Lever 1) path only — cached hyperparameters,
one XGBoost fit, no Optuna search. It must never call ``evaluate_and_persist_ticker``
or ``forecast_ticker_full``: this node is reachable from Render's admin HTTP
routes (``/run/{ticker}``, ``/run-all``), and running the expensive weekly
evaluation from there is exactly what OOM-killed the first production
deployment. The weekly evaluation runs externally — see
.github/workflows/weekly-evaluation.yml.

The LLM's role here is unchanged and deliberately limited: it writes a
narrative summarising signals it is given. It does not produce, adjust, or
review any number.
"""

from __future__ import annotations

import os
import time

from agents.llm import DEFAULT_GROQ_MODEL, groq_client
from agents.state import AgentState
from pipeline.model import forecast_ticker_daily

_groq_client = groq_client       # re-exported; see agents/llm.py


def _failed_forecast(reason: str) -> dict:
    """
    Explicit failure state.

    Distinct from a real forecast in every field a consumer might read, so a
    failure can never be mistaken for a flat prediction — which is exactly how
    F10 stayed invisible.
    """
    return {
        "forecast_available": False,
        "forecast_error": reason,
        "forecast_price": None,
        "forecast_direction": "UNAVAILABLE",
        "forecast_change_pct": None,
        "pred_return": None,
        "interval_low": None,
        "interval_high": None,
        "interval_coverage": None,
        "prob_up": None,
        "random_walk_price": None,
        "benchmark_ticker": None,
        "benchmark_sector_specific": None,
        "eval_rank_ic": None,
        "eval_rank_ic_t": None,
        "eval_hit_rate": None,
        "eval_baseline_hit_rate": None,
        "eval_beats_naive": None,
        "eval_evaluated_at": None,
        "model_version": None,
    }


def forecasting_node(state: AgentState) -> dict:
    ticker = state["ticker"]

    try:
        forecast = forecast_ticker_daily(ticker)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[{ticker}] forecast failed: {exc}")
        return {**_failed_forecast(f"{type(exc).__name__}: {exc}"),
                "signal_narrative": "Forecast unavailable."}

    if forecast is None:
        return {**_failed_forecast("insufficient history or no out-of-sample folds"),
                "signal_narrative": "Forecast unavailable."}

    view = forecast.price_view
    ev = forecast.evaluation

    updates = {
        "forecast_available": True,
        "forecast_error": None,
        "forecast_price": view["implied_price"],
        # UP / DOWN, not OUTPERFORM / UNDERPERFORM. The forecast is the
        # stock's own return since P1 and is not relative to anything, so the
        # comparative words would name a comparison that is no longer made.
        "forecast_direction": "UP" if view["pred_return"] > 0 else "DOWN",
        "forecast_change_pct": view["implied_change_pct"],
        "pred_return": view["pred_return"],
        "interval_low": view.get("interval_low"),
        "interval_high": view.get("interval_high"),
        "interval_coverage": view.get("interval_coverage"),
        "prob_up": view.get("prob_up"),
        "random_walk_price": view["random_walk_price"],
        "benchmark_ticker": forecast.benchmark_ticker,
        "benchmark_sector_specific": forecast.benchmark_sector_specific,
        "eval_rank_ic": ev.get("rank_ic"),
        "eval_rank_ic_t": ev.get("rank_ic_t"),
        "eval_hit_rate": ev.get("hit_rate"),
        "eval_baseline_hit_rate": ev.get("majority_hit_rate"),
        "eval_beats_naive": ev.get("beats_naive"),
        "eval_evaluated_at": ev.get("evaluated_at"),
        "model_version": forecast.model_version,
        "current_price": forecast.current_price,
    }

    updates["signal_narrative"] = _narrative(state, ticker, updates)
    return updates


def _deserves_a_written_narrative(updates: dict) -> bool:
    """
    Whether this ticker is worth spending a Groq call on.

    The daily job runs the whole universe, two LLM calls per ticker, against a
    free-tier daily token budget. On 2026-08-15 that budget ran out partway
    through and the tail of the alphabet fell back to the rule-based narrative
    anyway — so the cap was already choosing which stocks got a written
    narrative, by arrival order, which is the worst possible selection rule.

    Choosing deliberately instead: a narrative is only ever read next to a
    ranked row, and a row can only rank if it has a persisted evaluation behind
    it AND a positive predicted excess return (compute_composite_score floored
    both of its components at zero, so a predicted underperformer scored 0.0
    and sank regardless of what the narrative said). Both facts are already in
    ``updates`` by the time this is called. That takes NARRATIVE calls from one
    per ticker (~95) to roughly 15.

    THIS RATIONALE IS NOW STALE AND THE GATE IS DELIBERATELY LEFT ALONE. There
    are no ranked rows: every stock in the frozen universe gets a forecast, and
    each is read on its own rather than against the others. So "only ever read
    next to a ranked row" is false, and the narrower half of the condition —
    `pred > 0` — now withholds the written analysis from precisely the stocks a
    reader most needs it for, the ones forecast to fall.

    It is not changed here because changing it is not free and not this phase's
    call. It multiplies narrative calls by ~6, the configured model is
    decommissioned and 404s on every request, and the replacement has not been
    chosen. The P4 forecast object is meant to explain every stock, including
    why one does NOT work; that is where this gate gets rewritten, alongside
    the model decision it depends on.

    Note this gates one of the two LLM calls per ticker. critic_agent's
    _llm_review still runs unconditionally, so the daily total is ~95 + ~15
    rather than ~15 — comfortably inside 500k tokens/day, but the critic is now
    the larger consumer of the two.

    Everything else still gets _rule_based_narrative, which is deterministic
    and states only what the signals say — not a degraded output, just an
    unwritten one.
    """
    if not updates.get("eval_evaluated_at"):
        return False

    pred = updates.get("pred_return")
    return pred is not None and pred > 0


def _narrative(state: AgentState, ticker: str, updates: dict) -> str:
    """
    Asks the LLM for a plain-English read of the signals.

    The prompt withholds the forecast on purpose. Handing the model its own
    prediction invites a narrative written to justify the number rather than to
    describe the evidence.
    """
    signals = state.get("latest_signals", {}) or {}

    if not _deserves_a_written_narrative(updates):
        return _rule_based_narrative(ticker, signals)

    client = _groq_client()
    if client is None:
        return _rule_based_narrative(ticker, signals)

    interesting = {
        k: v for k, v in signals.items()
        if k in {"rsi", "macd_hist", "bb_width", "atr_14", "stoch_k", "williams_r",
                 "roc_10", "prox_52w", "dev_sma50", "hurst",
                 "sector_rel_5d", "sector_rel_20d", "close"}
    }

    prompt = f"""You are a quantitative analyst summarising technical signals for an Indian (NSE) stock.

Stock: {state.get('company_name', ticker)} ({ticker})
Benchmark: {updates.get('benchmark_ticker')}
Latest signal values:
{interesting}

Write exactly 3 sentences describing what these signals collectively suggest about
near-term momentum relative to the benchmark. Reference specific signals by name and value.
Do not state a price target, a percentage move, or a buy/sell recommendation."""

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                temperature=0.3,
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:                               # noqa: BLE001
            if "429" in str(exc) and attempt == 0:
                time.sleep(2)
                continue
            print(f"[{ticker}] narrative generation failed: {exc}")
            break

    # TWO arguments. The `sentiment` parameter was dropped from
    # _rule_based_narrative and the two call sites above were updated; this
    # one was not, and it is the ONLY site reachable when the Groq call
    # fails. So it stayed green everywhere the LLM was absent (client is
    # None takes the correct path above) and raised TypeError on every
    # ticker the moment the model id went stale in production.
    return _rule_based_narrative(ticker, signals)


def _rule_based_narrative(ticker: str, signals: dict) -> str:
    """
    Deterministic fallback so a narrative is never fabricated by guesswork.

    Took a ``sentiment`` argument it never referenced. Dropped rather than
    wired in: sentiment is unscored (see pipeline/sentiment.py), so the only
    honest use of it here would have been to say nothing about it.
    """
    rsi = float(signals.get("rsi", 50) or 50)
    macd = float(signals.get("macd_hist", 0) or 0)
    rel20 = float(signals.get("sector_rel_20d", 0) or 0)

    zone = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
    momentum = "positive" if macd > 0 else "negative"
    relative = "ahead of" if rel20 > 0 else "behind"

    return (
        f"RSI is {rsi:.1f}, placing {ticker} in {zone} territory. "
        f"The MACD histogram is {momentum} at {macd:.3f}. "
        f"Over the last 20 sessions the stock has traded {relative} its benchmark "
        f"by {abs(rel20) * 100:.1f}%."
    )
