# scheduler.py — Daily forecast job and weekly evaluation job.
#
# Split into two cadences (Lever 1). The first production run on Render ran
# the full purged walk-forward evaluation (hundreds of XGBoost fits per
# ticker) for every ticker, EVERY DAY — the instance starved its one free-tier
# CPU core for over an hour and eventually OOM-killed. "Does this model form
# have skill" doesn't need daily re-measurement; it's a slow-moving property.
#
#   run_pipeline_job()          — DAILY. Fresh data in, cheap forecast out.
#                                  No Optuna search runs here.
#   run_weekly_evaluation_job() — WEEKLY. The expensive purged walk-forward
#                                  evaluation + hyperparameter retune, once.
#
# As of Lever 4, the primary trigger for both is an external GitHub Actions
# workflow that runs this module's functions directly against Supabase (see
# .github/workflows/daily-pipeline.yml and weekly-evaluation.yml) — not
# Render. start_scheduler() below still exists for local development, but
# api/main.py no longer calls it: an in-process scheduler on the same
# single-core instance that serves API requests is exactly the CPU
# contention that caused the OOM crash, and running it there AS WELL AS the
# external GH Actions trigger risks both firing at once.

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


def run_pipeline_job():
    """
    Daily pipeline: universe sync, fresh data, cheap per-ticker forecast.

    Calls pipeline.model.train_and_forecast(), which internally uses
    forecast_ticker_daily() — cached hyperparameters, one XGBoost fit per
    ticker, no Optuna search. The evidence grade attached to each forecast
    comes from whatever run_weekly_evaluation_job() last persisted, which may
    be up to a week old; that staleness is surfaced via `evaluated_at`, not
    hidden.
    """
    logger.info("Starting daily pipeline run...")
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

        logger.info("[5/5] Forecasting (cached hyperparameters, no search)...")
        train_and_forecast(tickers=universe)

        logger.info("Daily pipeline run completed successfully.")
    except Exception as e:
        logger.error(f"Error during daily pipeline run: {e}", exc_info=True)


def run_weekly_evaluation_job():
    """
    Weekly evaluation: the expensive purged walk-forward run, once per ticker.

    For each ticker: runs evaluate_ticker() (nested-tuned purged walk-forward),
    calibrates conformal intervals on the out-of-sample residuals, refreshes
    the cached production hyperparameters (force=True), and persists all of
    it to model_metadata. The daily job reads this all week; it never
    recomputes it.
    """
    from data.universe import get_universe, sync_current_membership
    from data.tickers import refresh_metadata
    from pipeline.model import evaluate_and_persist_universe

    # Idempotent and cheap even if the daily job already did this today —
    # the weekly workflow runs in its own environment (Lever 4) and should
    # not assume a daily run has happened recently.
    try:
        sync_current_membership()
        refresh_metadata()
    except Exception as exc:                                        # noqa: BLE001
        logger.warning(f"[Scheduler] universe sync failed, using last known "
                       f"membership: {exc}")

    universe = get_universe()
    logger.info(f"[Scheduler] Weekly evaluation started for {len(universe)} stocks")

    if not universe:
        logger.error("[Scheduler] Universe is empty — aborting weekly evaluation.")
        return

    results = evaluate_and_persist_universe(tickers=universe)
    logger.info(f"[Scheduler] Weekly evaluation complete: "
               f"{len(results)}/{len(universe)} tickers evaluated")


def start_scheduler():
    """
    Starts the in-process APScheduler background scheduler.

    Local development only. Production (Render) does not call this — see the
    module docstring. Uses APScheduler's native coalesce + max_instances to
    prevent duplicate runs if it ever IS used somewhere long-running.
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

    # Weekly evaluation — Saturday 08:30 IST. Saturday rather than Sunday
    # 02:00 (the old weekly_retune_all schedule) so a run that takes hours
    # has the whole weekend as buffer before Monday's market open, and NSE
    # is closed both days so there's no competing daily run same-day.
    scheduler.add_job(
        run_weekly_evaluation_job,
        trigger="cron",
        day_of_week="sat",
        hour=8,
        minute=30,
        id="weekly_evaluation",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started (local). Daily pipeline at 18:30 IST, "
               "weekly evaluation Saturday 08:30 IST.")
    return scheduler


if __name__ == "__main__":
    scheduler = start_scheduler()
    if scheduler:
        try:
            while True:
                time.sleep(2)
        except (KeyboardInterrupt, SystemExit):
            pass
