"""
agents/graph.py — LangGraph orchestration and forecast persistence.

THERE IS NO RANKING ANY MORE. The product is one forecast per stock over a
fixed universe, not an ordering of stocks against each other, and the scoring
layer that produced that ordering is gone: `composite_score`, `score_basis`,
`_score_parts` and the SQL window rank behind `/api/leaderboard`.

Worth recording WHY, because the score was itself a Phase 0 repair and was not
obviously wrong. It was rebuilt once already — the original summed 30 pts
directional accuracy + 30 pts critic verdict + 15 pts confidence + 25 pts
upside, of which the first three were functions of leaked in-sample metrics
that barely varied across stocks, so the ranking reduced to sorting by
predicted upside (audit finding F9). The replacement separated signal from
evidence and gated one by the other, which was the right correction.

What it could not fix is that ranking 95 names needs 95 comparable numbers,
and the evidence gate produces 3 — a yield indistinguishable from chance
(pass rates 0.385/0.042/0.042 give 3.12 expected under independence; three
observed, Poisson p = 0.60). A ranking over 3 real numbers and 92 zeros is a
table that invites a reader to compare rows the measurement cannot separate.
The honest object is a forecast per stock, each carrying its own evidence.

`forecast_confidence` (STRONG / WEAK / INSUFFICIENT) survives and still says
what the held-out evidence supports. It grades a single forecast; it does not
order one against another.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from sqlalchemy import bindparam, text

from agents.critic_agent import critic_node
from agents.external_data_agent import external_data_node
from agents.forecasting_agent import forecasting_node
from agents.state import EVIDENCE_MULTIPLIER, AgentState
from agents.trading_data_agent import trading_data_node

# Refuse to prune more of the universe than this in one run. See
# prune_forecast_current().
PRUNE_MAX_FRACTION = 0.25


def save_forecast_to_db(state: dict) -> None:
    """
    Persists the forecast: one append-only row in `forecasts`, one upserted
    row in `forecast_current`.

    The two tables are the history and the present view of the same object.
    `forecast_current` was called `leaderboard` and is renamed rather than
    dropped — the ranking lived in three of its columns, not in the table. The
    row itself is the product.
    """
    from data.db import get_engine, to_native_params
    from data.frozen_universe import frozen_fingerprint
    from data.tickers import get_benchmark_name, get_company, get_sector

    engine = get_engine()
    ticker = state.get("ticker", "")
    now = datetime.now(timezone.utc)

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
        "eval_rank_ic_t": state.get("eval_rank_ic_t"),
        "eval_hit_rate": state.get("eval_hit_rate"),
        "eval_baseline_hit_rate": state.get("eval_baseline_hit_rate"),
        "eval_beats_random_walk": int(bool(state.get("eval_beats_naive"))),
        "model_version": state.get("model_version"),
        # The FROZEN universe's fingerprint, not the rule's. The rule no longer
        # decides who is forecast, so recording it would say nothing about
        # which rows this run covered — and that is the whole purpose of the
        # column. See data/frozen_universe.py.
        "universe_rule": frozen_fingerprint(),
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
    }

    forecast_cols = [
        "ticker", "company", "sector", "current_price", "forecast_price", "direction",
        "change_pct", "mape", "directional_accuracy", "forecast_confidence",
        "signal_narrative", "critic_verdict", "critic_reasoning", "critic_flags",
        "critic_confidence_adjustment", "last_updated", "pred_excess_return",
        "pred_return", "interval_low", "interval_high", "interval_coverage",
        "prob_outperform", "random_walk_price", "benchmark_ticker", "benchmark_name",
        "benchmark_sector_specific", "eval_rank_ic", "eval_rank_ic_t", "eval_hit_rate",
        "eval_baseline_hit_rate", "eval_beats_random_walk", "model_version",
        "universe_rule", "evaluated_at",
    ]

    # `change_pct`, not `upside_pct`. Same quantity, and the old name was the
    # ranking's vocabulary: "upside" asserts a long recommendation, which is
    # exactly the claim this layer no longer makes. A forecast can point down.
    current_cols = [
        "ticker", "company", "sector", "current_price", "forecast_price", "direction",
        "change_pct", "critic_verdict", "forecast_confidence", "mape",
        "directional_accuracy", "last_updated", "pred_excess_return", "interval_low",
        "interval_high", "interval_coverage", "prob_outperform", "random_walk_price",
        "benchmark_ticker", "benchmark_name", "benchmark_sector_specific",
        "eval_rank_ic", "eval_rank_ic_t", "eval_hit_rate", "eval_baseline_hit_rate",
        "eval_beats_random_walk", "model_version", "universe_rule", "evaluated_at",
        "signal_narrative",
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

        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in current_cols if c != "ticker")
        conn.execute(
            text(f"{_insert(current_cols, 'forecast_current')} "
                 f"ON CONFLICT (ticker) DO UPDATE SET {updates}"),
            to_native_params({c: payload.get(c) for c in current_cols}),
        )
        conn.commit()


def prune_forecast_current(universe: list[str],
                           max_fraction: float = PRUNE_MAX_FRACTION) -> int:
    """
    Deletes forecast_current rows for tickers outside the frozen universe.

    save_forecast_to_db upserts per ticker and never deletes, so a name that
    leaves the universe keeps its last row forever, dated whenever it was last
    forecast. That is worse than untidy: the row renders exactly like a live
    one. The board was still headed by IDEA.NS and SAIL.NS — neither in the
    NIFTY 100 — months after they left it, because their stale rows carried a
    pre-Phase-0 composite near 100 while every gated score written since was
    0.0. The score is gone; the stale row is not, and a reader has no way to
    tell a forecast published today from one abandoned last May.

    ``max_fraction`` bounds the blast radius. It mattered more when the input
    varied: get_universe() used to screen freshly fetched OHLCV, so a partial
    yfinance response produced a SHORT universe rather than an empty one, and
    every healthy name missing from it looked exactly like a delisting. The
    universe is frozen now and cannot shrink by accident, which removes that
    failure mode — but not the guard, because the frozen list is edited by
    hand and a typo in a checked-in file deletes rows just as effectively as a
    bad download did. A universe change of any size is deliberate and can be
    re-run; a quarter of the table vanishing unnoticed cannot be undone.

    Returns the number of rows removed (0 when the prune is refused).
    """
    from data.db import get_engine

    if not universe:                    # never prune on an empty universe
        return 0

    engine = get_engine()

    def outside_universe(sql: str):
        """`sql` must contain the `:tickers` placeholder for the IN list."""
        return text(sql).bindparams(bindparam("tickers", expanding=True))

    params = {"tickers": list(universe)}

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM forecast_current")).scalar() or 0
        if not total:
            return 0

        doomed = conn.execute(
            outside_universe("SELECT COUNT(*) FROM forecast_current "
                             "WHERE ticker NOT IN :tickers"),
            params,
        ).scalar() or 0

        if doomed > total * max_fraction:
            print(f"[Forecasts] REFUSING to prune {doomed} of {total} rows "
                  f"(>{max_fraction:.0%}) against a universe of {len(universe)} "
                  f"tickers — treating this as a bad universe, not a mass "
                  f"delisting. forecast_current left untouched.")
            return 0

        result = conn.execute(
            outside_universe("DELETE FROM forecast_current WHERE ticker NOT IN :tickers"),
            params,
        )
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
            "sentiment_score": None,
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
