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
from agents.state import EVIDENCE_MULTIPLIER, AgentState
from agents.trading_data_agent import trading_data_node

# Refuse to prune more of the leaderboard than this in one run. See
# prune_leaderboard().
PRUNE_MAX_FRACTION = 0.25


def _score_parts(
    pred_excess_return: float,
    prob_outperform: float | None,
) -> tuple[float, float]:
    """
    The two ungated components of the composite, before evidence and flags.

    signal      (0-60)  predicted excess return, saturating at +10% over 30
                        sessions so one extreme forecast cannot dominate.
    conviction  (0-40)  distance of P(outperform) from a coin flip, but ONLY
                        when the point forecast agrees that the move is up.

    Both floor at zero, which is what makes the composite a LONG-ONLY ranking:
    a confidently predicted underperformer scores exactly the same as a
    confidently predicted flat one. classify_score_basis() exists to keep that
    fact visible rather than buried in a 0.0.

    Conviction used to be computed independently of the point forecast, and
    that was not long-only at all. PNB.NS ranked THIRD on the live leaderboard
    on 2026-08-17 while forecasting a 1.69% UNDERPERFORMANCE: signal floored to
    0.00 as intended, but conviction still collected 10.75 points from a
    prob_outperform of 0.567, and 0.38 survived the flag deduction. The two
    inputs disagreed and the score quietly sided with the one that scored
    higher.

    They disagree for a real reason — prob_positive(-0.0169) is the fraction of
    calibration residuals above +0.0169, so a value over 0.5 means the model is
    biased LOW for that ticker — but "the point forecast says down, the
    calibrated probability says up" is not a ranking signal. It is a statement
    that the model contradicts itself, and the honest response is to rank it
    nowhere rather than to pick the cheerier half.
    """
    signal = min(max(pred_excess_return, 0.0) / 0.10, 1.0) * 60.0

    if pred_excess_return <= 0.0 or prob_outperform is None:
        conviction = 0.0
    else:
        conviction = min(max(prob_outperform - 0.5, 0.0) / 0.25, 1.0) * 40.0

    return signal, conviction


def compute_composite_score(
    pred_excess_return: float | None,
    evidence_grade: str,
    prob_outperform: float | None,
    n_flags: int = 0,
) -> float:
    """
    Ranking heuristic in [0, 100]. NOT an expected return.

    signal + conviction, multiplied by an evidence grade so a model that failed
    its held-out checks scores 0 regardless of its prediction, less a 5-point
    deduction per critic flag, floored at zero.
    """
    if pred_excess_return is None:
        return 0.0

    signal, conviction = _score_parts(pred_excess_return, prob_outperform)
    raw = (signal + conviction) * EVIDENCE_MULTIPLIER.get(evidence_grade, 0.0)
    return round(max(raw - 5.0 * n_flags, 0.0), 2)


def classify_score_basis(
    pred_excess_return: float | None,
    evidence_grade: str,
    prob_outperform: float | None,
    n_flags: int = 0,
) -> str:
    """
    Why a row scored what it scored — in particular, why it scored zero.

    A composite of 0.0 was previously ambiguous across four unrelated
    situations, and the leaderboard is mostly zeros: of 95 rows on 2026-08-15,
    85 scored 0.0. A stock the model actively predicted would UNDERPERFORM
    (evidence in hand, conviction real) was indistinguishable from one the
    weekly evaluation had simply never reached. Sorting by score put them in
    the same undifferentiated block, so the one number a reader is invited to
    rank on silently conflated "no view" with "negative view".

    The score itself is unchanged — this only names the reason:

        RANKED        scored above zero and ranks normally.
        NO_FORECAST   the model produced no prediction at all.
        NO_EVIDENCE   the evidence gate zeroed it; the weekly walk-forward
                      has not cleared this ticker (grade INSUFFICIENT).
        NOT_LONG      evidence exists, but the prediction is flat-to-negative,
                      and the composite only ranks long candidates.
        FLAGGED_OUT   a real long signal that critic flags drove to zero.
    """
    if pred_excess_return is None:
        return "NO_FORECAST"

    if EVIDENCE_MULTIPLIER.get(evidence_grade, 0.0) == 0.0:
        return "NO_EVIDENCE"

    signal, conviction = _score_parts(pred_excess_return, prob_outperform)
    if signal + conviction <= 0.0:
        return "NOT_LONG"

    if compute_composite_score(pred_excess_return, evidence_grade,
                               prob_outperform, n_flags) <= 0.0:
        return "FLAGGED_OUT"

    return "RANKED"


def save_forecast_to_db(state: dict) -> None:
    """Persists the forecast, its evidence, and the leaderboard row."""
    from data.db import get_engine, to_native_params
    from data.tickers import get_benchmark_name, get_company, get_sector
    from data.universe import DEFAULT_RULE

    engine = get_engine()
    ticker = state.get("ticker", "")
    now = datetime.now(timezone.utc)

    score_args = {
        "pred_excess_return": state.get("pred_excess_return"),
        "evidence_grade": state.get("evidence_grade", "INSUFFICIENT"),
        "prob_outperform": state.get("prob_outperform"),
        "n_flags": len(state.get("critic_flags", []) or []),
    }
    composite = compute_composite_score(**score_args)
    score_basis = classify_score_basis(**score_args)

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
        "score_basis": score_basis,
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
        "universe_rule", "evaluated_at", "score_basis",
    ]

    leaderboard_cols = [
        "ticker", "company", "sector", "current_price", "forecast_price", "upside_pct",
        "composite_score", "critic_verdict", "forecast_confidence", "mape",
        "directional_accuracy", "last_updated", "pred_excess_return", "interval_low",
        "interval_high", "interval_coverage", "prob_outperform", "random_walk_price",
        "benchmark_ticker", "benchmark_name", "benchmark_sector_specific",
        "eval_rank_ic", "eval_rank_ic_t", "eval_hit_rate", "eval_baseline_hit_rate",
        "eval_beats_random_walk", "model_version", "universe_rule", "evaluated_at",
        "score_basis",
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


def prune_leaderboard(universe: list[str],
                      max_fraction: float = PRUNE_MAX_FRACTION) -> int:
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

    ``max_fraction`` bounds the blast radius. Refusing to prune an EMPTY
    universe is not enough of a guard, because the input is the same
    get_universe() call whose output already varies run to run: it applies a
    liquidity floor and a listing-history floor over freshly fetched OHLCV, so
    a partial yfinance response or a half-written ohlcv table yields a short
    universe rather than an empty one. At that point every healthy name
    missing from the short list looks exactly like a delisting, and this
    function would delete the live leaderboard on the strength of a bad
    download. A legitimate index reconstitution moves a handful of names; a
    quarter of the table disappearing at once is a broken input, so it is
    reported and skipped rather than executed.

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
        total = conn.execute(text("SELECT COUNT(*) FROM leaderboard")).scalar() or 0
        if not total:
            return 0

        doomed = conn.execute(
            outside_universe("SELECT COUNT(*) FROM leaderboard "
                             "WHERE ticker NOT IN :tickers"),
            params,
        ).scalar() or 0

        if doomed > total * max_fraction:
            print(f"[Leaderboard] REFUSING to prune {doomed} of {total} rows "
                  f"(>{max_fraction:.0%}) against a universe of {len(universe)} "
                  f"tickers — treating this as a bad universe, not a mass "
                  f"delisting. Leaderboard left untouched.")
            return 0

        result = conn.execute(
            outside_universe("DELETE FROM leaderboard WHERE ticker NOT IN :tickers"),
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
