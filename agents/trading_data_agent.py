"""
agents/trading_data_agent.py — Loads technical signals for a ticker.

This is an ETL step rather than an agent in any meaningful sense; the audit
noted as much. It is kept as a graph node so the pipeline stays legible, but it
makes no decisions and calls no model.

SQL now uses bound parameters. The previous f-string interpolation was reachable
from ``/api/admin/run/{ticker}``, which accepts a user-supplied ticker
(audit finding F15).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from agents.state import AgentState
from data.db import get_engine
from pipeline.signals import compute_and_store


def trading_data_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    engine = get_engine()

    def _load() -> pd.DataFrame:
        return pd.read_sql(
            text("SELECT * FROM signals WHERE ticker = :t ORDER BY date ASC"),
            engine, params={"t": ticker},
        )

    signals_df = _load()

    # Recompute when the stored history is empty or has no unlabelled tail —
    # the tail is where the forecast row comes from.
    needs_refresh = (
        signals_df.empty
        or "target_excess_return" not in signals_df.columns
        or signals_df["target_excess_return"].isna().sum() == 0
    )
    if needs_refresh:
        print(f"[{ticker}] signals missing or stale — recomputing")
        compute_and_store(ticker)
        signals_df = _load()

    if signals_df.empty:
        return {"signals_df": [], "latest_signals": {}, "current_price": 0.0}

    latest = signals_df.iloc[-1].to_dict()
    return {
        "signals_df": signals_df.to_dict(orient="records"),
        "latest_signals": latest,
        "current_price": float(latest.get("close", 0.0) or 0.0),
    }
