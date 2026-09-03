"""
agents/external_data_agent.py — Loads sentiment and macro context.

NOTE on sentiment. ``sentiment_score`` is carried in state for display but is
NOT a model feature. It only ever existed for the current date, so every
training row held 0.0 while the row being predicted held a real value (audit
finding F7) — a train/serve mismatch at exactly the row that matters.

THE SECOND HALF OF THAT NOTE USED TO SAY "unbacktestable by construction
because Google News RSS serves no archive", AND THAT WAS WRONG. Measured
2026-09-03, the RSS search endpoint honours Google's `after:` / `before:`
operators and returns correctly-dated articles back to at least 2016-09 — the
month the `macro` table starts. The claim had stood since Phase 0 and had
shaped the plan for as long.

So ingestion now goes through ``pipeline.news``, which stores each article
under ITS OWN publication date rather than the fetch date, records every window
it attempted, and refuses to keep a result set that was relevance-ranked. The
old ``pipeline.sentiment`` path is not called here any more; its table is left
intact as a record of what was served to readers, because its `date` column is
the day we fetched and there is no transformation from that to the truth.

``sentiment_score`` stays None until a scorer exists (P3c). None is
"unavailable"; 0.0 would be "measured as neutral", and collapsing the two is
what let the dashboard report NEUTRAL for every stock in the universe
indefinitely.

SQL uses bound parameters (F15).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text

from agents.state import AgentState
from data.db import get_engine
from pipeline.macro import fetch_and_store as fetch_macro
from pipeline.news import fetch_recent
from pipeline.sentiment import get_aggregate_sentiment


def external_data_node(state: AgentState) -> dict:
    ticker = state["ticker"]
    engine = get_engine()
    today = date.today().isoformat()

    # KEYED ON COVERAGE, NOT ON ARTICLES. Asking "are there rows for today?"
    # cannot tell a day we already fetched and found nothing from a day we
    # never fetched — and on 2026-08-20 and 2026-08-28 the old path stored a
    # placeholder for all 95 tickers, so it looked like the former while being
    # the latter. `news_coverage` records the attempt itself.
    attempted = pd.read_sql(
        text("SELECT 1 FROM news_coverage WHERE ticker = :t "
             "AND window_end >= :d LIMIT 1"),
        engine, params={"t": ticker, "d": today},
    )
    if attempted.empty:
        print(f"[{ticker}] fetching recent news")
        fetch_recent([ticker], engine=engine)

    macro_df = pd.read_sql(text("SELECT * FROM macro ORDER BY date ASC"), engine)
    if macro_df.empty or str(macro_df.iloc[-1]["date"]) != today:
        print("[Macro] refreshing macro window")
        fetch_macro()
        macro_df = pd.read_sql(text("SELECT * FROM macro ORDER BY date ASC"), engine)

    return {
        "sentiment_score": get_aggregate_sentiment(ticker),
        "macro_df": macro_df.to_dict(orient="records") if not macro_df.empty else [],
    }
