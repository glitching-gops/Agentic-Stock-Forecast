"""
agents/external_data_agent.py — Loads sentiment and macro context.

NOTE on sentiment. ``sentiment_score`` is carried in state for display but is
NOT a model feature any more. It only ever existed for the current date, so
every training row held 0.0 while the row being predicted held a real value
(audit finding F7) — a train/serve mismatch at exactly the row that matters,
and unbacktestable by construction because Google News RSS serves no archive.

It returns as a feature once a dated news archive exists (Phase 2, T2.4). Until
then it is context for a human reader, and the model does not see it.

SQL uses bound parameters (F15).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text

from agents.state import AgentState
from data.db import get_engine
from pipeline.macro import fetch_and_store as fetch_macro
from pipeline.sentiment import fetch_and_score, get_aggregate_sentiment


def external_data_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    engine = get_engine()
    today = date.today().isoformat()

    existing = pd.read_sql(
        text("SELECT 1 FROM sentiment WHERE ticker = :t AND date = :d LIMIT 1"),
        engine, params={"t": ticker, "d": today},
    )
    if existing.empty:
        print(f"[{ticker}] fetching today's news sentiment")
        fetch_and_score(ticker)

    macro_df = pd.read_sql(text("SELECT * FROM macro ORDER BY date ASC"), engine)
    if macro_df.empty or str(macro_df.iloc[-1]["date"]) != today:
        print("[Macro] refreshing macro window")
        fetch_macro()
        macro_df = pd.read_sql(text("SELECT * FROM macro ORDER BY date ASC"), engine)

    return {
        "sentiment_score": get_aggregate_sentiment(ticker),
        "macro_df": macro_df.to_dict(orient="records") if not macro_df.empty else [],
    }
