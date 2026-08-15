"""
main.py — Local entry point.

Order matters: the universe is synced from its point-in-time rule before any
data is fetched, because every downstream step now takes an explicit ticker
list rather than reaching for a hard-coded one.
"""

import os
import subprocess
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

from data.db import get_engine, init_db
from data.tickers import refresh_metadata
from data.universe import (
    describe_universe_bias,
    get_ingest_universe,
    get_universe,
    init_universe_tables,
    sync_current_membership,
)


def is_db_empty() -> bool:
    engine = get_engine()
    try:
        count = pd.read_sql("SELECT COUNT(*) AS c FROM ohlcv", con=engine).iloc[0]["c"]
        return count == 0
    except Exception:
        return True


if __name__ == "__main__":
    print("=== ZeRO Agentic Stock Forecast ===\n")

    print("[1/5] Initialising database...")
    init_db()
    init_universe_tables()

    print("[2/5] Syncing point-in-time universe...")
    sync_current_membership()
    refresh_metadata()
    ingest_list = get_ingest_universe()
    print(f"      Index members: {len(ingest_list)}")

    bias = describe_universe_bias()
    print(f"      NOTE: {bias['note']}\n")

    if is_db_empty():
        print("[3/5] Database is empty. Running the initial full pipeline "
              "(this takes a while)...")
        from pipeline.fetch import fetch_and_store
        from pipeline.macro import fetch_and_store as fetch_macro
        from pipeline.sentiment import fetch_and_score
        from pipeline.signals import compute_and_store
        from agents.graph import run_graph

        # Fetch over raw membership, then screen — the liquidity filter reads
        # the table this step populates.
        fetch_and_store(tickers=ingest_list)
        fetch_macro()

        universe = get_universe()
        print(f"      Tradable universe after screening: {len(universe)}")

        compute_and_store(tickers=universe)
        fetch_and_score(tickers=universe)

        # run_graph(), not pipeline.model.train_and_forecast(): only run_graph
        # (via its critic node) populates forecasts/leaderboard. See
        # scheduler.run_pipeline_job's docstring for why this matters.
        for ticker in universe:
            run_graph(ticker)
    else:
        print("[3/5] Database already contains data. Skipping initial fetch.")
        universe = get_universe()
        print(f"      Tradable universe: {len(universe)}")

    print("[4/5] Starting background scheduler...")
    from scheduler import start_scheduler
    start_scheduler()

    print("[5/5] Launching dashboard...\n")
    dashboard_path = os.path.join(os.path.dirname(__file__), "app", "main.py")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
    except KeyboardInterrupt:
        print("\nShutting down...")
