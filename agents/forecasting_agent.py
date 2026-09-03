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
from datetime import date

from agents.llm import (
    DEFAULT_GROQ_MODEL,
    NoRouteAvailable,
    complete,
    groq_client,
    strip_reasoning,
)
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


# How many stocks get an LLM-written narrative on any one day.
#
# Not "all of them", and no longer "the ones forecast to rise". See
# _deserves_a_written_narrative for why the old condition had to go and why
# this is a SAMPLE rather than the whole universe.
#
# 12 divides 84 exactly, so the rotation below tiles the frozen universe in
# seven windows and every stock gets a written narrative once a week, on a
# known day. Change the size and the rotation still works, it just stops
# tiling evenly.
NARRATIVE_SAMPLE_SIZE = int(os.getenv("NARRATIVE_SAMPLE_SIZE", "12"))


def _narrative_sample(day: date | None = None,
                      size: int | None = None) -> tuple[str, ...]:
    """
    The stocks whose narrative is WRITTEN today.

    A contiguous window over the sorted frozen universe, advanced by one full
    window per calendar day. Deterministic, so a re-run of the same day writes
    the same narratives and costs the same tokens; and a pure function of the
    DATE and the universe file, so it cannot see a forecast.

    That independence is the whole design. Any rule that reads the prediction
    to decide who gets explained selects the sample on the outcome, and this
    project has spent a phase removing exactly that shape of mistake.
    """
    from data.frozen_universe import FROZEN_UNIVERSE

    universe = sorted(FROZEN_UNIVERSE)
    if not universe:
        return ()

    n = max(0, min(size if size is not None else NARRATIVE_SAMPLE_SIZE,
                   len(universe)))
    if n == 0:
        return ()

    start = ((day or date.today()).toordinal() * n) % len(universe)
    return tuple(universe[(start + i) % len(universe)] for i in range(n))


def _deserves_a_written_narrative(ticker: str, updates: dict) -> bool:
    """
    Whether this ticker is worth spending a Groq call on today.

    HISTORY, because the shape of the mistake matters more than the rule.
    The daily job ran the whole universe at two LLM calls per ticker against a
    free-tier daily token budget. On 2026-08-15 that budget ran out partway
    through and the tail of the alphabet fell back to the rule-based narrative
    - so the cap was already choosing which stocks got written up, by arrival
    order, which is the worst possible selection rule.

    The rule that replaced it was "a row that can rank": a persisted evaluation
    AND a positive predicted return. Neither half survives P0 and P1. Nothing
    ranks any more, so "only read beside a ranked row" is false. And on an
    absolute-return target `pred > 0` withheld the written analysis from
    exactly the stocks a reader most needs it for - the ones forecast to fall -
    while making the sample a function of the model's own output.

    WHAT IT IS NOW (2026-09-02): membership of today's rotating sample, and
    nothing else. Two conditions were dropped deliberately.

      `pred > 0`            gone. It selected on the outcome, and a forecast
                            reading is not more worth explaining because it is
                            bullish.

      `eval_evaluated_at`   gone, and this one is less obvious. The prompt
                            withholds the forecast on purpose (see _narrative):
                            what the model is asked to describe is the SIGNAL
                            state, which every ticker has whether or not a
                            weekly evaluation has graded it. Requiring evidence
                            here would also mean zero narratives for the whole
                            of P1, since MODEL_VERSION moved and every ticker
                            sits at INSUFFICIENT until the weekly job re-runs.
                            The evidence grade is published beside the
                            narrative; it does not need to gate it.

    The cost is bounded by the sample, not by the gate's opinion of a stock -
    ~12 calls a day against a 1,000/day allowance, with the critic's review
    gated separately on the evidence grade.
    """
    return ticker in _narrative_sample()


def _narrative(state: AgentState, ticker: str, updates: dict) -> str:
    """
    Asks the LLM for a plain-English read of the signals.

    The prompt withholds the forecast on purpose. Handing the model its own
    prediction invites a narrative written to justify the number rather than to
    describe the evidence.
    """
    signals = state.get("latest_signals", {}) or {}

    if not _deserves_a_written_narrative(ticker, updates):
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

    # The router handles model choice, the fallback chain and the empty-after-
    # stripping case; the retry stays here because a 429 is a PACING problem
    # rather than a broken route, and falling through to the next model on one
    # would spend a scarcer budget to work around a limit that clears in a
    # second. See agents/llm.py for which provider carries this task and why.
    for attempt in range(2):
        try:
            return complete("narrative", prompt, temperature=0.3).text
        except NoRouteAvailable as exc:
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
