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

    The final step runs agents.graph.run_graph() per ticker rather than
    calling pipeline.model.train_and_forecast() directly. This matters:
    train_and_forecast() only ever wrote to model_metadata — it never touched
    the forecasts/leaderboard tables the dashboard actually reads. Only
    run_graph() (via its critic node, which computes the composite score and
    calls save_forecast_to_db) does that. That gap was latent since Phase 0 —
    nothing scheduled ever called run_graph, only the manual /run-all admin
    route did — which is why the dashboard sat on a months-old row even while
    this job ran successfully every day. run_graph()'s forecasting_node still
    calls forecast_ticker_daily() internally, so this remains cached
    hyperparameters, no Optuna search, per audit finding for this rewrite.

    The evidence grade attached to each forecast comes from whatever
    run_weekly_evaluation_job() last persisted, which may be up to a week
    old; that staleness is surfaced via `evaluated_at`, not hidden.
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
        from agents.graph import prune_leaderboard, run_graph

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

        logger.info(f"[5/5] Forecasting {len(universe)} tickers "
                   f"(cached hyperparameters, no search)...")
        succeeded, failed = 0, 0
        for ticker in universe:
            try:
                state = run_graph(ticker)
                if state.get("forecast_available"):
                    succeeded += 1
                else:
                    failed += 1
                    logger.warning(f"[5/5] {ticker}: {state.get('forecast_error')}")
            except Exception as exc:                                # noqa: BLE001
                failed += 1
                logger.error(f"[5/5] {ticker}: run_graph failed — {exc}")
        logger.info(f"[5/5] Forecasting complete: {succeeded} succeeded, {failed} failed")

        # Names that have left the index keep their last leaderboard row
        # otherwise, and those rows carry pre-Phase-0 composite scores that
        # outrank every evidence-gated score written today. See
        # agents.graph.prune_leaderboard.
        removed = prune_leaderboard(universe)
        if removed:
            logger.info(f"[5/5] Pruned {removed} leaderboard row(s) for tickers "
                       f"no longer in the universe")

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

    INGESTS ITS OWN DATA FIRST. This job used to sync index membership and then
    evaluate straight out of the `signals` table, which only the DAILY job ever
    writes. That made a silent ordering dependency, and on 2026-08-15 it bit:
    the run finished normally in 14 minutes and reported "33/95 tickers
    evaluated", with the other 62 refused for insufficient history — including
    INFY, TCS, RELIANCE and HDFCBANK, none of which is short of history. The
    signals table was simply half-built at the time, and the daily run 90
    minutes later populated the rest and forecast all 62 without complaint.

    Nothing in the output distinguished "this ticker has no track record" from
    "this table was not ready yet", so an infrastructure race was reported as a
    data-quality verdict, and the leaderboard sat on 33 evaluated names for a
    week. Fetching and recomputing here costs minutes against an evaluation
    measured in hours, and makes the job's result depend on the database rather
    than on what ran before it — which is what the membership sync below
    already assumed.
    """
    from data.universe import (
        get_ingest_universe, get_universe, sync_current_membership,
    )
    from data.tickers import refresh_metadata
    from pipeline.fetch import fetch_and_store
    from pipeline.signals import compute_and_store, count_labelled_rows
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

    ingest_list = get_ingest_universe()
    if not ingest_list:
        logger.error("[Scheduler] Index membership is empty — aborting weekly "
                     "evaluation.")
        return

    logger.info(f"[Scheduler] Refreshing OHLCV for {len(ingest_list)} index members...")
    fetch_and_store(tickers=ingest_list)

    universe = get_universe()
    if not universe:
        logger.error("[Scheduler] Universe is empty after screening — aborting "
                     "weekly evaluation.")
        return

    # Same F6 monotonicity check the daily job runs. It belongs here for the
    # same reason it belongs there, and its absence cost real data: the first
    # run of this rewritten job (2026-08-16) recomputed signals while three
    # sector indices were transiently unavailable, wrote a null target over
    # every row of the 22 tickers benchmarked to them, and evaluated anyway —
    # reporting them as "no out-of-sample predictions". The daily job would
    # have aborted on the same input. Giving this job the same write power
    # without the same guard is what let it through.
    labelled_before = count_labelled_rows()

    logger.info(f"[Scheduler] Recomputing signals for {len(universe)} tickers...")
    compute_and_store(tickers=universe)

    labelled_after = count_labelled_rows()
    if labelled_after < labelled_before:
        logger.error(
            f"Labelled rows fell from {labelled_before} to {labelled_after} "
            f"after recomputing signals — aborting before any evaluation is "
            f"persisted. The most likely cause is a benchmark index that "
            f"failed to download, which NULLs the excess-return target for "
            f"every stock mapped to it (regression of F6)."
        )
        return
    logger.info(f"[Scheduler] Labelled rows: {labelled_before} -> {labelled_after}")

    logger.info(f"[Scheduler] Weekly evaluation started for {len(universe)} stocks")
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
