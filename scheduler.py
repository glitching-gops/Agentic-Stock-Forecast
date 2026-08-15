# scheduler.py — Runs the pipeline daily at 6:30 PM IST

import time
import logging
import os
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# Imports deferred to job execution to save memory


# Setup logging to file
os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "logs", "scheduler.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lock file approach removed — APScheduler job_defaults handle
# concurrent run prevention natively via coalesce + max_instances.

def run_pipeline_job():
    """
    Daily pipeline.

    Step 0 is new: the universe is resynced from its point-in-time rule before
    anything else runs, and the labelled-row count is asserted non-decreasing
    afterwards. That assertion is the guard against F6 silently returning — the
    old append-only writer froze the training labels without any visible symptom.
    """
    logger.info("Starting scheduled pipeline run...")
    try:
        from data.universe import (
            get_ingest_universe, get_universe, sync_current_membership,
        )
        from data.tickers import refresh_metadata
        from pipeline.fetch import fetch_and_store
        from pipeline.signals import compute_and_store, count_labelled_rows
        from pipeline.sentiment import fetch_and_score
        from pipeline.macro import fetch_and_store as fetch_macro
        from pipeline.model import train_and_forecast

        logger.info("[0/5] Syncing point-in-time universe...")
        sync_current_membership()
        refresh_metadata()

        # Ingest over raw index membership; the liquidity screen runs afterwards
        # because it reads the very table this step populates.
        ingest_list = get_ingest_universe()
        logger.info(f"[0/5] Index members to ingest: {len(ingest_list)}")

        if not ingest_list:
            logger.error("Index membership is empty — aborting run.")
            return

        labelled_before = count_labelled_rows()

        logger.info("[1/5] Fetching OHLCV data...")
        fetch_and_store(tickers=ingest_list)

        universe = get_universe()
        logger.info(f"[1/5] Tradable universe after screening: {len(universe)}")
        if not universe:
            logger.error("Universe is empty after screening — aborting run.")
            return

        logger.info("[2/5] Computing signals...")
        compute_and_store(tickers=universe)

        labelled_after = count_labelled_rows()
        if labelled_after < labelled_before:
            logger.error(
                f"Labelled rows fell from {labelled_before} to {labelled_after}. "
                f"Target backfill is broken (regression of F6) — aborting before "
                f"any forecast is written."
            )
            return
        logger.info(f"[2/5] Labelled rows: {labelled_before} -> {labelled_after}")

        logger.info("[3/5] Fetching news sentiment...")
        fetch_and_score(tickers=universe)

        logger.info("[4/5] Fetching macro data...")
        fetch_macro()

        logger.info("[5/5] Training and forecasting...")
        train_and_forecast(tickers=universe)

        logger.info("Scheduled pipeline run completed successfully.")
    except Exception as e:
        logger.error(f"Error during scheduled pipeline run: {e}", exc_info=True)

def weekly_retune_all():
    """
    Re-tunes the PRODUCTION model's hyperparameters for every ticker.

    These parameters are used only to fit the model that generates the next
    forecast. Reported metrics never come from them: ``walk_forward`` re-tunes
    inside each evaluation fold, so a configuration is never chosen with sight
    of the rows it is later scored on (audit finding F2).

    The LSTM retrain is gone — that module is archived because it never wrote a
    checkpoint (F5). See pipeline/archived/README.md.
    """
    from data.universe import get_universe
    from pipeline.model import load_features_for_ticker, FEATURES, TARGET
    from pipeline.signals import HORIZON_SESSIONS
    from pipeline.tuning import tune_and_cache

    universe = get_universe()
    logger.info(f"[Scheduler] Weekly retune started for {len(universe)} stocks")

    for ticker in universe:
        try:
            df = load_features_for_ticker(ticker)
            labelled = df[df[TARGET].notna()] if not df.empty and TARGET in df else df
            if labelled.empty or len(labelled) < 300:
                logger.info(f"[Scheduler] {ticker}: insufficient labelled data, skipping")
                continue

            tune_and_cache(ticker, labelled[FEATURES], labelled[TARGET],
                           horizon=HORIZON_SESSIONS, force=True)
            logger.info(f"[Scheduler] {ticker}: retuned on {len(labelled)} rows")
        except Exception as e:
            logger.error(f"[Scheduler] {ticker}: retuning failed — {e}")

    logger.info("[Scheduler] Weekly retune complete")

def start_scheduler():
    """Starts the APScheduler background scheduler.

    Uses APScheduler's native coalesce + max_instances to prevent duplicate
    runs — no lock files needed or created.
    """
    ist_tz = pytz.timezone("Asia/Kolkata")

    scheduler = BackgroundScheduler(
        timezone=ist_tz,
        job_defaults={
            "coalesce":           True,   # merge multiple missed runs into one
            "max_instances":      1,      # never run the same job twice simultaneously
            "misfire_grace_time": 3600,   # allow up to 1 hour late start
        }
    )

    # Daily pipeline at 18:30 IST
    scheduler.add_job(
        run_pipeline_job,
        "cron",
        hour=18,
        minute=30,
        id="pipeline_job",
        replace_existing=True,
    )

    # Weekly full Optuna retune — Sunday 02:00 IST
    scheduler.add_job(
        weekly_retune_all,
        trigger="cron",
        day_of_week="sun",
        hour=2,
        minute=0,
        id="weekly_retune",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started. Pipeline will run daily at 18:30 IST.")
    return scheduler

if __name__ == "__main__":
    scheduler = start_scheduler()
    if scheduler:
        try:
            while True:
                time.sleep(2)
        except (KeyboardInterrupt, SystemExit):
            pass
