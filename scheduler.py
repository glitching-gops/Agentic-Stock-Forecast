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


class PipelineAbort(RuntimeError):
    """
    A precondition failed and the job stopped without writing its output.

    Every abort path in this module used to `logger.error(...)` and then
    `return`, and the module has never contained a `sys.exit`, a `raise`, or a
    non-zero return anywhere -- so the interpreter exited 0 and GitHub Actions
    marked the run green. On 2026-08-17 and 2026-08-18 two scheduled daily runs
    aborted on the F6 monotonicity guard, wrote no forecast at all, finished in
    a third of the normal time, and both reported success. The staleness was
    eventually noticed on the dashboard, not in CI, two days later.

    A job that produced nothing must fail loudly enough for the runner to see
    it. Both workflows invoke these functions through `python -c`, so an
    exception is exactly what turns the step red.
    """


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
    run_id, gate = None, None
    try:
        from data.universe import (
            get_ingest_universe, get_universe, sync_current_membership,
        )
        from data.tickers import refresh_metadata
        from pipeline.corporate_actions import fetch_and_store as fetch_actions
        from pipeline.fetch import fetch_and_store
        from pipeline.outcomes import resolve_due_forecasts
        from pipeline.signals import compute_and_store, count_labelled_rows
        from pipeline.sentiment import fetch_and_score
        from pipeline.macro import fetch_and_store as fetch_macro
        from pipeline.tracking import finish_run, start_run
        from pipeline.validation import FAIL, run_gate
        from agents.graph import prune_leaderboard, run_graph

        logger.info("[1/8] Syncing point-in-time universe...")
        sync_current_membership()
        refresh_metadata()

        # Ingest over raw index membership; the liquidity screen runs afterwards
        # because it reads the very table this step populates.
        ingest_list = get_ingest_universe()
        logger.info(f"[1/8] Index members to ingest: {len(ingest_list)}")

        if not ingest_list:
            raise PipelineAbort("Index membership is empty — aborting run.")

        labelled_before = count_labelled_rows()

        logger.info("[2/8] Fetching OHLCV data...")
        fetch_and_store(tickers=ingest_list)

        universe = get_universe()
        logger.info(f"[2/8] Tradable universe after screening: {len(universe)}")
        if not universe:
            raise PipelineAbort(
                "Universe is empty after screening — aborting run.")

        # Opened here rather than at the top of the function because the run's
        # identity includes its universe and its data hash, and neither exists
        # until the screen has run. Everything that can go wrong after this
        # point is recorded against this row; the two aborts above are loud and
        # immediate by comparison.
        run_id = start_run("daily", universe)

        # Splits and dividends, before the gate that uses them to tell a
        # corporate action apart from a broken adjustment basis (F11).
        logger.info("[3/8] Refreshing corporate actions...")
        try:
            fetch_actions(tickers=universe)
        except Exception as exc:                                # noqa: BLE001
            # Degraded, not fatal: the gate downgrades the price-break check to
            # a warning when this table is stale, and no other step reads it.
            logger.warning(f"[3/8] corporate actions refresh failed: {exc}")

        logger.info("[4/8] Computing signals...")
        signals_report = compute_and_store(tickers=universe)

        # A skip or a refusal preserves that ticker's labels and leaves its
        # signals stale for the day. That is a degradation, not a failure, and
        # aborting the whole run over it would cost the other ninety tickers
        # their forecast for no benefit — but it must not pass silently, since
        # a benchmark index that stays down is a data-quality problem that
        # compounds every day it goes unreported.
        if signals_report.skipped or signals_report.refused:
            logger.error(
                f"[4/8] Signals degraded: {len(signals_report.skipped)} skipped "
                f"{signals_report.skipped}, {len(signals_report.refused)} "
                f"refused {signals_report.refused}. Labels for these tickers "
                f"are intact but their signals are stale; the usual cause is a "
                f"benchmark index that failed to download."
            )
        if not signals_report.processed:
            raise PipelineAbort(
                f"Signals were written for 0 of {len(universe)} tickers — "
                f"aborting before any forecast is made."
            )

        labelled_after = count_labelled_rows()
        if labelled_after < labelled_before:
            # With the write-boundary guard in pipeline.signals this should now
            # be unreachable. It stays as the backstop it always was: if it
            # ever fires again, a destructive write got past that guard.
            raise PipelineAbort(
                f"Labelled rows fell from {labelled_before} to {labelled_after}. "
                f"Target backfill is broken (regression of F6) — aborting before "
                f"any forecast is written."
            )
        logger.info(f"[4/8] Labelled rows: {labelled_before} -> {labelled_after}")

        # THE GATE. Everything above has written to the database; nothing below
        # has published yet. This is the only point at which a data defect can
        # still be caught before it reaches a reader, which is why it sits here
        # and not at the end.
        logger.info("[5/8] Running the validation gate...")
        gate = run_gate(universe)
        for check in gate.checks:
            (logger.error if check.status == FAIL else logger.info)(
                f"[5/8] {check}")
        logger.info(f"[5/8] Gate: {gate.summary()}")
        if gate.status == FAIL:
            raise PipelineAbort(
                f"Validation gate FAILED before publishing: "
                f"{'; '.join(c.detail for c in gate.failures)}"
            )

        logger.info("[6/8] Fetching news sentiment and macro data...")
        fetch_and_score(tickers=universe)
        fetch_macro()

        logger.info(f"[7/8] Forecasting {len(universe)} tickers "
                   f"(cached hyperparameters, no search)...")
        succeeded, failed = 0, 0
        for ticker in universe:
            try:
                state = run_graph(ticker)
                if state.get("forecast_available"):
                    succeeded += 1
                else:
                    failed += 1
                    logger.warning(f"[7/8] {ticker}: {state.get('forecast_error')}")
            except Exception as exc:                                # noqa: BLE001
                failed += 1
                logger.error(f"[7/8] {ticker}: run_graph failed — {exc}")
        logger.info(f"[7/8] Forecasting complete: {succeeded} succeeded, {failed} failed")

        # Every ticker failing is the same outcome as aborting — nothing was
        # published — and used to report the same way aborting did: green.
        if succeeded == 0:
            raise PipelineAbort(
                f"All {failed} tickers failed to forecast; no leaderboard row "
                f"was written. Not reporting this run as successful."
            )

        # Names that have left the index keep their last leaderboard row
        # otherwise, and those rows carry pre-Phase-0 composite scores that
        # outrank every evidence-gated score written today. See
        # agents.graph.prune_leaderboard.
        removed = prune_leaderboard(universe)
        if removed:
            logger.info(f"[7/8] Pruned {removed} leaderboard row(s) for tickers "
                       f"no longer in the universe")

        # Score the forecasts whose 30 sessions have now elapsed. This is the
        # only measurement in the system taken on PUBLISHED output rather than
        # on held-out folds, and `forecast_outcomes` had no writer at all until
        # now — so nothing had ever checked whether a forecast came true.
        logger.info("[8/8] Resolving matured forecasts...")
        outcomes = resolve_due_forecasts()
        logger.info(f"[8/8] Outcomes: {outcomes.summary()}")

        finish_run(run_id, "OK", gate=gate, metrics={
            "forecasts_succeeded": succeeded,
            "forecasts_failed": failed,
            "signals_skipped": signals_report.skipped,
            "signals_refused": signals_report.refused,
            "labelled_rows": labelled_after,
            "outcomes_resolved": outcomes.resolved,
            "leaderboard_pruned": removed,
        })
        logger.info("Daily pipeline run completed successfully.")
    except Exception as e:
        # Log first so the traceback lands in scheduler.log, then re-raise so
        # the process exits non-zero. Swallowing here is what let two days of
        # no-op runs report success.
        logger.error(f"Error during daily pipeline run: {e}", exc_info=True)
        if run_id:
            from pipeline.tracking import finish_run as _finish
            _finish(run_id,
                    "ABORTED" if isinstance(e, PipelineAbort) else "FAILED",
                    gate=gate, notes=str(e)[:500])
        raise


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
    from pipeline.tracking import finish_run, start_run
    from pipeline.validation import FAIL, run_gate

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
        raise PipelineAbort("Index membership is empty — aborting weekly "
                            "evaluation.")

    logger.info(f"[Scheduler] Refreshing OHLCV for {len(ingest_list)} index members...")
    fetch_and_store(tickers=ingest_list)

    universe = get_universe()
    if not universe:
        raise PipelineAbort("Universe is empty after screening — aborting "
                            "weekly evaluation.")

    # Same F6 monotonicity check the daily job runs. It belongs here for the
    # same reason it belongs there, and its absence cost real data: the first
    # run of this rewritten job (2026-08-16) recomputed signals while three
    # sector indices were transiently unavailable, wrote a null target over
    # every row of the 22 tickers benchmarked to them, and evaluated anyway —
    # reporting them as "no out-of-sample predictions". The daily job would
    # have aborted on the same input. Giving this job the same write power
    # without the same guard is what let it through.
    labelled_before = count_labelled_rows()

    run_id = start_run("weekly", universe)

    logger.info(f"[Scheduler] Recomputing signals for {len(universe)} tickers...")
    signals_report = compute_and_store(tickers=universe)

    if signals_report.skipped or signals_report.refused:
        logger.error(
            f"[Scheduler] Signals degraded: {len(signals_report.skipped)} "
            f"skipped {signals_report.skipped}, {len(signals_report.refused)} "
            f"refused {signals_report.refused}. Those tickers keep their "
            f"existing labels and will be evaluated on slightly stale signals."
        )
    if not signals_report.processed:
        finish_run(run_id, "ABORTED", notes="no ticker had signals written")
        raise PipelineAbort(
            f"Signals were written for 0 of {len(universe)} tickers — aborting "
            f"before any evaluation is persisted."
        )

    labelled_after = count_labelled_rows()
    if labelled_after < labelled_before:
        finish_run(run_id, "ABORTED", notes="labelled rows regressed (F6)")
        raise PipelineAbort(
            f"Labelled rows fell from {labelled_before} to {labelled_after} "
            f"after recomputing signals — aborting before any evaluation is "
            f"persisted. The most likely cause is a benchmark index that "
            f"failed to download, which NULLs the excess-return target for "
            f"every stock mapped to it (regression of F6)."
        )
    logger.info(f"[Scheduler] Labelled rows: {labelled_before} -> {labelled_after}")

    # The gate belongs here for the same reason it belongs in the daily job:
    # everything above has written to the database and nothing below has
    # published. The weekly job is the one that persists the evidence the
    # leaderboard gates on, so a data defect reaching it is worse, not better —
    # a corrupted evaluation stays on the board for a week.
    gate = run_gate(universe)
    for check in gate.checks:
        (logger.error if check.status == FAIL else logger.info)(
            f"[Scheduler] {check}")
    logger.info(f"[Scheduler] Gate: {gate.summary()}")
    if gate.status == FAIL:
        finish_run(run_id, "ABORTED", gate=gate,
                   notes="validation gate failed before evaluation")
        raise PipelineAbort(
            f"Validation gate FAILED before evaluating: "
            f"{'; '.join(c.detail for c in gate.failures)}"
        )

    logger.info(f"[Scheduler] Weekly evaluation started for {len(universe)} stocks")
    try:
        results = evaluate_and_persist_universe(tickers=universe)
    except Exception as exc:                                        # noqa: BLE001
        finish_run(run_id, "FAILED", gate=gate, notes=str(exc)[:500])
        raise
    logger.info(f"[Scheduler] Weekly evaluation complete: "
               f"{len(results)}/{len(universe)} tickers evaluated")

    finish_run(run_id, "OK", gate=gate, metrics={
        "tickers_evaluated": len(results),
        "tickers_in_universe": len(universe),
        "labelled_rows": labelled_after,
        "signals_skipped": signals_report.skipped,
        "signals_refused": signals_report.refused,
    })


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
