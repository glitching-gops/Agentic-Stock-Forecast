"""
agents/graph.py — LangGraph orchestration and forecast persistence.

The composite score is rebuilt. Previously it summed 30 pts directional
accuracy + 30 pts critic verdict + 15 pts confidence + 25 pts upside. The first
three were all functions of the leaked in-sample metrics, and the verdict was a
near-constant APPROVED, so roughly 75 of 100 points did not vary across stocks
and the ranking reduced to sorting by predicted upside (audit finding F9).

The replacement separates two things that were tangled together:

    signal    — how strong is the predicted excess return?
    evidence  — has this model shown skill on data it did not see?

Ranking is driven by signal, gated by evidence. A stock whose model failed its
held-out checks cannot outrank one that passed simply by predicting a bigger
move. The score is a ranking heuristic and is documented as such; it is not an
expected return.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from sqlalchemy import bindparam, text

from agents.critic_agent import critic_node
from agents.external_data_agent import external_data_node
from agents.forecasting_agent import forecasting_node
from agents.state import AgentState
from agents.trading_data_agent import trading_data_node

EVIDENCE_MULTIPLIER = {"STRONG": 1.0, "WEAK": 0.5, "INSUFFICIENT": 0.0}


def compute_composite_score(
    pred_excess_return: float | None,
    evidence_grade: str,
    prob_outperform: float | None,
    n_flags: int = 0,
) -> float:
    """
    Ranking heuristic in [0, 100]. NOT an expected return.

    signal      (0-60)  predicted excess return, saturating at +10% over 30
                        sessions so one extreme forecast cannot dominate.
    conviction  (0-40)  distance of P(outperform) from a coin flip.
    evidence            multiplies the total, so a model that failed its
                        held-out checks scores 0 regardless of its prediction.
    flags               5-point deduction each, floored at zero.
    """
    if pred_excess_return is None:
        return 0.0

    signal = min(max(pred_excess_return, 0.0) / 0.10, 1.0) * 60.0

    if prob_outperform is None:
        conviction = 0.0
    else:
        conviction = min(max(prob_outperform - 0.5, 0.0) / 0.25, 1.0) * 40.0

    raw = (signal + conviction) * EVIDENCE_MULTIPLIER.get(evidence_grade, 0.0)
    return round(max(raw - 5.0 * n_flags, 0.0), 2)


def save_forecast_to_db(state: dict) -> None:
    """Persists the forecast, its evidence, and the leaderboard row."""
    from data.db import get_engine, to_native_params
    from data.tickers import get_benchmark_name, get_company, get_sector
    from data.universe import DEFAULT_RULE

    engine = get_engine()
    ticker = state.get("ticker", "")
    now = datetime.now(timezone.utc)

    composite = compute_composite_score(
        pred_excess_return=state.get("pred_excess_return"),
        evidence_grade=state.get("evidence_grade", "INSUFFICIENT"),
        prob_outperform=state.get("prob_outperform"),
        n_flags=len(state.get("critic_flags", []) or []),
    )

    benchmark = state.get("benchmark_ticker") or ""
    payload = {
        "ticker": ticker,
        "company": get_company(ticker),
        "sector": get_sector(ticker),
        "current_price": state.get("current_price"),
        "forecast_price": state.get("forecast_price"),
        "direction": state.get("forecast_direction"),
        "change_pct": state.get("forecast_change_pct"),
        "pred_excess_return": state.get("pred_excess_return"),
        # Deliberately null: the model predicts an EXCESS return and has no
        # view on the benchmark, so a total return cannot be derived from it.
        # Copying the excess value into this column would mislabel it exactly
        # the way the old xgb_mape column mislabelled its contents.
        "pred_return": None,
        "interval_low": state.get("interval_low"),
        "interval_high": state.get("interval_high"),
        "interval_coverage": state.get("interval_coverage"),
        "prob_outperform": state.get("prob_outperform"),
        "random_walk_price": state.get("random_walk_price"),
        "benchmark_ticker": benchmark,
        "benchmark_name": get_benchmark_name(benchmark) if benchmark else None,
        "benchmark_sector_specific": int(bool(state.get("benchmark_sector_specific"))),
        "eval_rank_ic": state.get("eval_rank_ic"),
        "eval_hit_rate": state.get("eval_hit_rate"),
        "eval_baseline_hit_rate": state.get("eval_baseline_hit_rate"),
        "eval_beats_random_walk": int(bool(state.get("eval_beats_naive"))),
        "model_version": state.get("model_version"),
        "universe_rule": DEFAULT_RULE.fingerprint(),
        "evaluated_at": state.get("eval_evaluated_at"),
        "forecast_confidence": state.get("evidence_grade", "INSUFFICIENT"),
        "signal_narrative": state.get("signal_narrative"),
        "critic_verdict": state.get("critic_verdict", "REJECTED"),
        "critic_reasoning": state.get("critic_reasoning"),
        "critic_flags": json.dumps(state.get("critic_flags", []) or []),
        "critic_confidence_adjustment": state.get("critic_source", "evidence_gate"),
        "last_updated": now,
        # mape / directional_accuracy are retained as columns for schema
        # compatibility but are no longer written with in-sample values.
        "mape": None,
        "directional_accuracy": state.get("eval_hit_rate"),
        "upside_pct": state.get("forecast_change_pct"),
        "composite_score": composite,
    }

    forecast_cols = [
        "ticker", "company", "sector", "current_price", "forecast_price", "direction",
        "change_pct", "mape", "directional_accuracy", "forecast_confidence",
        "signal_narrative", "critic_verdict", "critic_reasoning", "critic_flags",
        "critic_confidence_adjustment", "last_updated", "pred_excess_return",
        "pred_return", "interval_low", "interval_high", "interval_coverage",
        "prob_outperform", "random_walk_price", "benchmark_ticker", "benchmark_name",
        "benchmark_sector_specific", "eval_rank_ic", "eval_hit_rate",
        "eval_baseline_hit_rate", "eval_beats_random_walk", "model_version",
        "universe_rule", "evaluated_at",
    ]

    leaderboard_cols = [
        "ticker", "company", "sector", "current_price", "forecast_price", "upside_pct",
        "composite_score", "critic_verdict", "forecast_confidence", "mape",
        "directional_accuracy", "last_updated", "pred_excess_return", "interval_low",
        "interval_high", "interval_coverage", "prob_outperform", "random_walk_price",
        "benchmark_ticker", "benchmark_name", "benchmark_sector_specific",
        "eval_rank_ic", "eval_hit_rate", "eval_baseline_hit_rate",
        "eval_beats_random_walk", "model_version", "universe_rule", "evaluated_at",
    ]

    def _insert(cols: list[str], table: str) -> str:
        names = ", ".join(cols)
        binds = ", ".join(f":{c}" for c in cols)
        return f"INSERT INTO {table} ({names}) VALUES ({binds})"

    # to_native_params is the last line of defence before psycopg2. Much of
    # this payload originates in pandas/numpy (the persisted evaluation is
    # read with pd.read_sql; the conformal interval and probability are
    # computed in numpy), and a numpy scalar reaching the driver produces a
    # repr-mangled SQL literal rather than a type error — see data.db.to_native.
    with engine.connect() as conn:
        conn.execute(text(_insert(forecast_cols, "forecasts")),
                     to_native_params({c: payload.get(c) for c in forecast_cols}))

        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in leaderboard_cols if c != "ticker")
        conn.execute(
            text(f"{_insert(leaderboard_cols, 'leaderboard')} "
                 f"ON CONFLICT (ticker) DO UPDATE SET {updates}"),
            to_native_params({c: payload.get(c) for c in leaderboard_cols}),
        )
        conn.commit()


def prune_leaderboard(universe: list[str]) -> int:
    """
    Deletes leaderboard rows for tickers no longer in the tradable universe.

    save_forecast_to_db upserts per ticker and never deletes, so a name that
    leaves the index keeps its last row forever. That is not merely untidy:
    those rows carry a composite_score computed under the pre-Phase-0 formula,
    which had no evidence gate and routinely produced scores near 100, while
    every score written today is gated by evidence and is 0.0 whenever the
    weekly evaluation has not yet cleared the ticker. Sorted by score, the
    orphans outrank the entire live universe — the leaderboard was still
    headed by IDEA.NS and SAIL.NS, neither of which is in the NIFTY 100,
    months after they left it.

    Returns the number of rows removed.
    """
    from data.db import get_engine

    if not universe:                    # never prune on an empty universe
        return 0

    engine = get_engine()
    stmt = text("DELETE FROM leaderboard WHERE ticker NOT IN :tickers").bindparams(
        bindparam("tickers", expanding=True)
    )
    with engine.connect() as conn:
        result = conn.execute(stmt, {"tickers": list(universe)})
        conn.commit()
    return result.rowcount or 0


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("trading_data", trading_data_node)
    workflow.add_node("external_data", external_data_node)
    workflow.add_node("forecasting", forecasting_node)
    workflow.add_node("critic", critic_node)

    workflow.add_edge(START, "trading_data")
    workflow.add_edge(START, "external_data")
    workflow.add_edge(["trading_data", "external_data"], "forecasting")
    workflow.add_edge("forecasting", "critic")
    workflow.add_edge("critic", END)

    return workflow.compile()


graph = build_graph()


def run_graph(ticker: str) -> dict:
    """
    Runs the agent graph for one ticker and persists the result.

    On failure it returns an explicit failure state rather than a plausible
    flat forecast — the distinction that F10 erased.
    """
    from data.tickers import get_company

    print(f"\n--- Running agent graph for {ticker} ---")

    try:
        initial: AgentState = {
            "ticker": ticker,
            "company_name": get_company(ticker),
            "current_price": 0.0,
            "signals_df": [],
            "latest_signals": {},
            "sentiment_score": 0.0,
            "macro_df": [],
            "forecast_available": False,
        }

        final_state = graph.invoke(initial)
        save_forecast_to_db(final_state)
        print(f"--- Completed {ticker} "
              f"({final_state.get('evidence_grade')}, "
              f"{final_state.get('critic_verdict')}) ---\n")
        return final_state

    except Exception as exc:                                   # noqa: BLE001
        safe = str(exc).encode("ascii", "backslashreplace").decode("ascii")
        print(f"--- FAILED {ticker}: {safe} ---\n")
        import traceback
        traceback.print_exc()
        return {
            "ticker": ticker,
            "company_name": get_company(ticker),
            "forecast_available": False,
            "forecast_error": safe,
            "forecast_direction": "UNAVAILABLE",
            "evidence_grade": "INSUFFICIENT",
            "critic_verdict": "REJECTED",
            "critic_reasoning": f"Pipeline failure: {safe}",
            "critic_flags": ["Pipeline Error"],
            "critic_source": "exception",
        }
