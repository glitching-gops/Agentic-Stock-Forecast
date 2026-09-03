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
from collections import Counter
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


# Below this share of the universe forecasting successfully, the run publishes a
# board that is mostly STALE rows and says nothing about it. See the
# rationale at the check itself in run_pipeline_job.
MIN_FORECAST_SUCCESS_RATE = 0.5


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
    the forecasts/forecast_current tables the dashboard actually reads. Only
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
        from pipeline.news import fetch_recent
        from pipeline.macro import fetch_and_store as fetch_macro
        from agents.llm import preflight as llm_preflight
        from pipeline.tracking import finish_run, start_run
        from pipeline.validation import FAIL, run_gate
        from agents.graph import prune_forecast_current, run_graph

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

        logger.info("[6/8] Fetching news and macro data...")
        news_totals = fetch_recent(universe)
        logger.info(f"[6/8] News: {news_totals}")
        fetch_macro()

        # A LOUD WARNING, NOT AN ABORT. A blocked news fetch degrades the
        # written narrative; it does not corrupt anything published, so the
        # "a job that writes nothing must raise" rule does not apply. But it
        # must not be invisible either: two days in August stored the "no news"
        # placeholder for every ticker and nothing anywhere said the fetch had
        # been refused.
        if news_totals.get("blocked"):
            logger.warning(
                f"[6/8] {news_totals['blocked']} news windows were BLOCKED. "
                f"Those are gaps in the archive, not quiet days — "
                f"news_coverage records which.")

        # THE MODELS ARE CHECKED BEFORE THE TICKER LOOP, NOT DURING IT.
        # `llama-3.1-8b-instant` was decommissioned under this project and
        # 404'd every call for days while the job reported OK. OpenRouter is
        # more exposed still: its free lineup rotates. Two seconds here answers
        # what ninety-five failures otherwise answer an hour later.
        try:
            flight = llm_preflight()
            for row in flight["routes"]:
                if not row["status"].startswith(("ok", "configured")):
                    logger.warning(f"[6/8] LLM route {row['task']} "
                                   f"{row['provider']}:{row['model']} — {row['status']}")
            if flight["dead_tasks"]:
                logger.warning(f"[6/8] LLM tasks with NO usable route: "
                               f"{flight['dead_tasks']} — those degrade to the "
                               f"deterministic path")
            logger.info(f"[6/8] OpenRouter budget {flight['openrouter_daily_budget']}, "
                        f"reasoning tier needs ~"
                        f"{flight['reasoning_calls_per_day_estimate']}/day")
        except Exception as exc:                                # noqa: BLE001
            logger.warning(f"[6/8] LLM preflight failed: {exc}")

        logger.info(f"[7/8] Forecasting {len(universe)} tickers "
                   f"(cached hyperparameters, no search)...")
        succeeded, failed = 0, 0
        # WHY the reasons are collected and not merely logged. On 2026-08-26/27/28
        # this loop failed 64 of 95 tickers with an identical TypeError, three
        # runs running, and every run finished OK. The count reached
        # experiment_runs; the REASON existed only in a workflow log that expires,
        # so nothing on record could answer "failed at what?" — the same
        # "non-fatal must not mean invisible" rule the baseline step obeys.
        reasons: Counter[str] = Counter()
        for ticker in universe:
            try:
                state = run_graph(ticker)
                if state.get("forecast_available"):
                    succeeded += 1
                else:
                    failed += 1
                    detail = str(state.get("forecast_error"))
                    reasons[detail[:120]] += 1
                    logger.warning(f"[7/8] {ticker}: {detail}")
            except Exception as exc:                                # noqa: BLE001
                failed += 1
                reasons[f"{type(exc).__name__}: {exc}"[:120]] += 1
                logger.error(f"[7/8] {ticker}: run_graph failed — {exc}")
        logger.info(f"[7/8] Forecasting complete: {succeeded} succeeded, {failed} failed")
        for detail, n in reasons.most_common(5):
            logger.warning(f"[7/8]   {n:>3} x {detail}")

        # A RATE, not just the all-fail case. `succeeded == 0` was written for
        # the total outage and is the weakest guard that could have been chosen:
        # it passed a 67% failure rate for three consecutive days while the
        # published board froze on rows a fortnight old, every one of them still
        # carrying a superseded MODEL_VERSION. What a reader sees is a stale
        # forecast indistinguishable from a fresh one, which is exactly the
        # class of defect the validation gate calls FAIL.
        #
        # 0.5 is deliberately loose rather than tight: ~12 names carry too little
        # history to forecast at all, so a healthy run sits near 87% and a gate
        # at 90% would fire on ordinary universe churn. A gate that cries wolf
        # gets switched off.
        rate = succeeded / max(len(universe), 1)
        if rate < MIN_FORECAST_SUCCESS_RATE:
            top = "; ".join(f"{n}x {d}" for d, n in reasons.most_common(3))
            raise PipelineAbort(
                f"Only {succeeded} of {len(universe)} tickers forecast "
                f"({rate:.0%}, floor {MIN_FORECAST_SUCCESS_RATE:.0%}). The "
                f"board keeps stale rows for the other {failed}. "
                f"Top reasons: {top}"
            )

        # Names outside the frozen universe keep their last row otherwise, and
        # a stale row renders exactly like a live one. See
        # agents.graph.prune_forecast_current.
        removed = prune_forecast_current(universe)
        if removed:
            logger.info(f"[7/8] Pruned {removed} forecast_current row(s) for "
                       f"tickers no longer in the universe")

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
            "forecast_errors": dict(reasons.most_common(10)),
            "signals_skipped": signals_report.skipped,
            "signals_refused": signals_report.refused,
            "labelled_rows": labelled_after,
            "outcomes_resolved": outcomes.resolved,
            "forecast_rows_pruned": removed,
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
    data-quality verdict, and the board sat on 33 evaluated names for a
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
    # the evidence gate reads, so a data defect reaching it is worse, not better —
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

    # ── Fundamentals sync ────────────────────────────────────────────────────
    #
    # Here rather than in the daily job because annual statements step once a
    # year per ticker: a daily refresh would be five times the vendor calls for
    # a figure that cannot have moved. Here rather than a workflow of its own
    # because the reader is twenty lines below — writer and reader in one job
    # cannot drift apart, and a sync that silently stopped running would
    # otherwise be invisible until someone wondered why coverage was stale.
    #
    # Runs BEFORE the comparisons, which consume it, and AFTER the per-ticker
    # evaluation has persisted, so a vendor outage cannot cost the expensive
    # work. Non-fatal for the same reason the baseline step is: this writes to
    # `fundamentals` only, which nothing published reads. It is recorded either
    # way — a step that can fail silently reads as "it was fine".
    fundamentals_metrics: dict = {}
    try:
        from pipeline.fundamentals import restatement_summary, sync_fundamentals

        fundamentals_metrics = sync_fundamentals(universe)
        fundamentals_metrics["restatements"] = restatement_summary()
        revised = fundamentals_metrics.get("revised", 0)
        if revised:
            # A restatement means the vendor changed a figure we had already
            # attached to a date and scored a model against. It does not
            # invalidate the run, but it is the one event in this step that
            # bears on whether the valuation result is real.
            logger.warning(
                f"[Scheduler] {revised} fiscal period(s) were RESTATED by the "
                f"vendor this week; see the fundamental_revisions table")
    except Exception as exc:                                        # noqa: BLE001
        logger.error(f"[Scheduler] Fundamentals sync failed: {exc}")
        fundamentals_metrics = {"note": f"sync raised: {str(exc)[:300]}"}

    # ── Baseline comparison ──────────────────────────────────────────────────
    #
    # Regenerated here rather than left to a manual tool run, for the same
    # reason report_performance.py exists: a number that is not regenerated
    # goes stale and then stays wrong. The per-ticker evaluation above says how
    # each model did against ITS OWN baselines; this says how the whole panel
    # does against a common set of comparators on one set of folds, which is
    # the only form in which a Phase 2 addition can be shown to have improved
    # anything.
    #
    # NOT FATAL, and deliberately so. It reads; it never writes to signals,
    # forecasts or model_metadata. The "a job that writes nothing must raise"
    # rule exists to stop a run PUBLISHING nothing while reporting success —
    # this step publishes nothing by design, so failing the week's evaluation
    # over it would trade a real result for a measurement. It is recorded
    # either way, so a silent failure is still a visible one: `baselines.note`
    # carries the reason into experiment_runs.
    #
    # It runs in the cheap tail of the job (~20s against an evaluation measured
    # in tens of minutes), after the persist, so a defect in it cannot cost the
    # expensive work. Only the valuation comparison — which reads the table the
    # fundamentals sync above just wrote — comes after it.
    baseline_metrics: dict = {}
    try:
        from pipeline.baselines import compare_baselines

        comparison = compare_baselines(tickers=universe)
        baseline_metrics = comparison.to_metrics()
        (logger.info if comparison.ranked else logger.error)(
            f"[Scheduler] {comparison.summary()}")
    except Exception as exc:                                        # noqa: BLE001
        logger.error(f"[Scheduler] Baseline comparison failed: {exc}")
        baseline_metrics = {"note": f"comparison raised: {str(exc)[:300]}",
                            "comparators": []}

    # ── Valuation comparison ─────────────────────────────────────────────────
    #
    # A SECOND comparison rather than a flag on the first, because
    # with_fundamentals restricts the panel to rows that carry a fundamental —
    # roughly a third of it, starting in 2022. That restriction is correct (the
    # with/without A/B has to run over identical rows or it measures the sample
    # change instead) but it must not displace the full-panel table, or the
    # weekly number would quietly start describing a different set of rows.
    #
    # Cheap: no foundation model, a third of the rows, ~10s.
    valuation_metrics: dict = {}
    try:
        from pipeline.baselines import compare_baselines

        valuation = compare_baselines(tickers=universe, with_fundamentals=True)
        valuation_metrics = valuation.to_metrics()
        (logger.info if valuation.ranked else logger.error)(
            f"[Scheduler] valuation: {valuation.summary()}")
    except Exception as exc:                                        # noqa: BLE001
        logger.error(f"[Scheduler] Valuation comparison failed: {exc}")
        valuation_metrics = {"note": f"comparison raised: {str(exc)[:300]}",
                             "comparators": []}

    # SCORING RUNS HERE AND NOWHERE ELSE, so one checkpoint scores the archive
    # and the live tail alike. Splitting them across two venues or two models
    # is F7 in a new costume - the historical feature and the published feature
    # would be different quantities, and nothing measured in a purged fold
    # would describe what a reader sees.
    #
    # Non-fatal and lazily imported: torch is absent from requirements.txt on
    # purpose, so a runner that has not installed requirements-scoring.txt gets
    # a recorded skip rather than a failed weekly evaluation.
    scoring_metrics: dict = {}
    try:
        from pipeline.news_scoring import ScorerUnavailable, score_unscored

        try:
            scoring = score_unscored()
            scoring_metrics = {"scored": scoring.scored,
                               "scorer_id": scoring.scorer_id,
                               "device": scoring.device,
                               "seconds": round(scoring.seconds, 1)}
            logger.info(f"[Scheduler] news scoring: {scoring.summary()}")
        except ScorerUnavailable as exc:
            scoring_metrics = {"note": f"scorer unavailable: {str(exc)[:200]}"}
            logger.warning(f"[Scheduler] news scoring skipped: {exc}")
    except Exception as exc:                                        # noqa: BLE001
        scoring_metrics = {"note": f"scoring raised: {str(exc)[:300]}"}
        logger.error(f"[Scheduler] News scoring failed: {exc}")

    # THE NEWS AND REGIME COMPARISONS ARE RECORDED SEPARATELY from the headline
    # table, for the same reason `baselines_valuation` is: they change the
    # feature set, so folding them into `baselines` would make the weekly
    # number describe a different model week to week with nothing saying so.
    enriched_metrics: dict = {}
    try:
        from pipeline.baselines import compare_baselines

        enriched = compare_baselines(tickers=universe, with_news=True,
                                     with_regime=True)
        enriched_metrics = enriched.to_metrics()
        logger.info(f"[Scheduler] news+regime: {enriched.summary()}")
    except Exception as exc:                                        # noqa: BLE001
        logger.error(f"[Scheduler] News/regime comparison failed: {exc}")
        enriched_metrics = {"note": f"comparison raised: {str(exc)[:300]}",
                            "comparators": []}

    finish_run(run_id, "OK", gate=gate, metrics={
        "tickers_evaluated": len(results),
        "tickers_in_universe": len(universe),
        "labelled_rows": labelled_after,
        "signals_skipped": signals_report.skipped,
        "signals_refused": signals_report.refused,
        "baselines": baseline_metrics,
        "baselines_valuation": valuation_metrics,
        "baselines_news_regime": enriched_metrics,
        "fundamentals": fundamentals_metrics,
        "news_scoring": scoring_metrics,
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
