"""
Regression tests for the daily/weekly split (Lever 1), the tuning-budget cut
(Lever 2), and moving compute off Render (Lever 4).

Context: the first production run on Render ran the full purged
walk-forward evaluation — hundreds of XGBoost fits per ticker — for every
ticker, every day, in the same process that serves API requests. It starved
the instance's one free-tier CPU core for over an hour and then OOM-killed
it. These tests exist so that regression never ships again silently: the
daily path must never run an Optuna search, and Render must never be the
thing running this work.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


# ── Lever 1: the daily path must never search ──────────────────────────────────

def test_forecast_ticker_daily_never_calls_evaluate_or_tunes():
    """
    The whole point of the split: forecast_ticker_daily() must not invoke
    evaluate_ticker() (the nested-tuned purged walk-forward runner) or force
    a hyperparameter search. It only fits with cached params.
    """
    from pipeline import model

    source = inspect.getsource(model.forecast_ticker_daily)
    assert "evaluate_ticker(" not in source
    assert "evaluate_and_persist" not in source
    assert "force_tune=True" not in source
    assert "force=True" not in source
    # Must load cached params only.
    assert "fit_production_model(ticker, df, force_tune=False)" in source


def test_daily_batch_driver_calls_the_daily_function_not_the_weekly_one():
    from pipeline import model

    source = inspect.getsource(model.train_and_forecast)
    assert "forecast_ticker_daily(ticker)" in source
    assert "evaluate_and_persist_ticker" not in source
    assert "evaluate_ticker(" not in source


def test_daily_write_never_touches_weekly_owned_columns():
    """
    _record_daily_fit()'s DB write (called from forecast_ticker_daily) must
    only ever set last_trained (+ model_version). If it also wrote
    eval_rank_ic/conformal_*/evaluated_at, a daily run could silently clobber
    the weekly evaluation's staleness signal with values that were never
    actually recomputed.
    """
    from pipeline import model

    source = inspect.getsource(model._record_daily_fit)
    insert_start = source.index("INSERT INTO model_metadata")
    insert_block = source[insert_start:insert_start + 400]

    for forbidden in ["eval_rank_ic", "eval_hit_rate", "conformal_quantile",
                     "conformal_residuals", "evaluated_at"]:
        assert forbidden not in insert_block, (
            f"daily write touches '{forbidden}', a weekly-owned column"
        )
    assert "last_trained" in insert_block


def test_forecast_ticker_daily_records_its_own_fit():
    """_record_daily_fit must be owned by the unit of work, not a batch
    wrapper — so it fires no matter which caller reaches this function."""
    from pipeline import model

    source = inspect.getsource(model.forecast_ticker_daily)
    assert "_record_daily_fit(ticker)" in source


def test_scheduler_populates_forecasts_and_leaderboard_not_just_model_metadata():
    """
    The bug this guards against: scheduler.run_pipeline_job() used to call
    pipeline.model.train_and_forecast() directly, which only ever writes
    model_metadata. Only agents.graph.run_graph() (via its critic node)
    populates the forecasts/leaderboard tables the dashboard actually reads.
    That gap was latent since Phase 0 -- nothing scheduled ever called
    run_graph, so the dashboard sat on a months-old row even while the daily
    job ran successfully every day. This asserts the daily job calls
    run_graph per ticker, not the lower-level batch driver.
    """
    import scheduler

    source = inspect.getsource(scheduler.run_pipeline_job)
    assert "run_graph(ticker)" in source
    assert "train_and_forecast(tickers=universe)" not in source


def test_main_bootstrap_also_populates_forecasts_not_just_model_metadata():
    """main.py's local first-run bootstrap had the identical gap."""
    source = (REPO / "main.py").read_text(encoding="utf-8")
    assert "run_graph(ticker)" in source
    assert "train_and_forecast(tickers=universe)" not in source


def test_weekly_function_does_the_expensive_work():
    """evaluate_and_persist_ticker must run the real evaluation and retune."""
    from pipeline import model

    source = inspect.getsource(model.evaluate_and_persist_ticker)
    assert "evaluate_ticker(ticker, df)" in source
    assert "force=True" in source          # refreshes cached production params
    assert "_persist_evaluation(" in source


# ── Agent graph and admin routes must only use the cheap path ─────────────────

def test_forecasting_agent_imports_the_daily_function():
    from agents import forecasting_agent

    assert hasattr(forecasting_agent, "forecast_ticker_daily")
    assert not hasattr(forecasting_agent, "forecast_ticker")
    assert not hasattr(forecasting_agent, "evaluate_and_persist_ticker")

    source = inspect.getsource(forecasting_agent)
    assert "forecast_ticker_daily(ticker)" in source


def test_admin_run_and_run_all_never_call_the_weekly_path():
    """
    /run/{ticker} and /run-all are reachable from the public internet (behind
    the admin key) and must stay on the cheap path. Only the dedicated
    /run-weekly-evaluation route may call the expensive one.
    """
    admin_source = (REPO / "api" / "routers" / "admin.py").read_text(encoding="utf-8")

    run_fn_start = admin_source.index("def _run_pipeline(")
    run_fn_end = admin_source.index("def _run_daily_pipeline(")
    run_fn_body = admin_source[run_fn_start:run_fn_end]

    assert "evaluate_and_persist" not in run_fn_body
    assert "run_weekly_evaluation" not in run_fn_body
    assert "run_graph" in run_fn_body   # -> forecasting_node -> forecast_ticker_daily


def test_admin_has_a_separate_weekly_endpoint():
    from api.routers import admin

    paths = [r.path for r in admin.router.routes]
    assert "/run-daily-pipeline" in paths
    assert "/run-weekly-evaluation" in paths
    assert "/run/{ticker}" in paths
    assert "/run-all" in paths


# ── Lever 4: Render must not run this in-process ───────────────────────────────

def test_render_lifespan_does_not_start_the_scheduler():
    """
    api/main.py must not IMPORT or CALL start_scheduler(). An in-process
    scheduler on the same single-core instance that serves API requests is
    what caused the OOM crash; GitHub Actions is the trigger now (see
    .github/workflows/). Parsed via AST rather than string search, so the
    module docstring is free to mention the name in prose explaining why
    it's absent without tripping this check.
    """
    import ast

    tree = ast.parse((REPO / "api" / "main.py").read_text(encoding="utf-8"))

    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "start_scheduler" not in imported_names
    assert "start_scheduler" not in called_names


def test_local_scheduler_entrypoint_still_exists_for_dev_use():
    """start_scheduler() itself must still exist -- only Render's lifespan
    should avoid calling it, local dev (main.py, `python scheduler.py`) may."""
    import scheduler
    assert callable(scheduler.start_scheduler)
    assert callable(scheduler.run_pipeline_job)
    assert callable(scheduler.run_weekly_evaluation_job)


# ── Workflow files exist and are wired to the right functions ─────────────────

def _load_workflow(name: str) -> dict:
    import yaml
    path = REPO / ".github" / "workflows" / name
    assert path.exists(), f"{name} does not exist"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_daily_workflow_calls_the_daily_job_and_has_a_schedule():
    wf = _load_workflow("daily-pipeline.yml")
    text = (REPO / ".github" / "workflows" / "daily-pipeline.yml").read_text(encoding="utf-8")

    assert "run_pipeline_job" in text
    assert "run_weekly_evaluation_job" not in text
    assert "DATABASE_URL" in text
    assert "secrets.DATABASE_URL" in text

    trigger = wf.get(True) or wf.get("on")
    assert "schedule" in trigger
    assert "workflow_dispatch" in trigger
    assert trigger["schedule"][0]["cron"] == "0 13 * * 1-5"


def test_weekly_workflow_calls_the_weekly_job_and_has_a_schedule():
    wf = _load_workflow("weekly-evaluation.yml")
    text = (REPO / ".github" / "workflows" / "weekly-evaluation.yml").read_text(encoding="utf-8")

    assert "run_weekly_evaluation_job" in text
    assert "run_pipeline_job()" not in text
    assert "DATABASE_URL" in text

    trigger = wf.get(True) or wf.get("on")
    assert "schedule" in trigger
    assert trigger["schedule"][0]["cron"] == "0 3 * * 6"


def test_workflows_persist_tuned_params_across_runs():
    """Both workflows must cache tuned_params/ or every run pays the full
    Optuna cost again — defeating the point of the daily/weekly split."""
    for name in ["daily-pipeline.yml", "weekly-evaluation.yml"]:
        text = (REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "tuned_params" in text
        assert "actions/cache" in text


# ── Persistence round-trip ──────────────────────────────────────────────────────

def test_persisted_calibration_round_trips_through_reconstruction():
    """
    _reconstruct_calibration() must rebuild a ConformalCalibration that
    behaves identically to the one that was persisted -- same interval, same
    probability -- from exactly the fields _load_persisted_evaluation() reads
    back from the database.
    """
    from pipeline.conformal import fit_conformal
    from pipeline.model import _reconstruct_calibration

    rng = np.random.default_rng(11)
    y_pred = rng.normal(size=500) * 0.02
    y_true = y_pred + rng.normal(size=500) * 0.05
    original = fit_conformal(y_true, y_pred, coverage=0.80)
    assert original is not None

    persisted = {
        "conformal_quantile": original.quantile,
        "conformal_coverage": original.coverage,
        "conformal_residuals": original.residuals.tolist(),
        "conformal_n": original.n,
    }
    rebuilt = _reconstruct_calibration(persisted)

    assert rebuilt is not None
    assert rebuilt.quantile == pytest.approx(original.quantile)
    assert rebuilt.coverage == original.coverage
    assert rebuilt.interval(0.01) == pytest.approx(original.interval(0.01))
    assert rebuilt.prob_positive(0.0) == pytest.approx(original.prob_positive(0.0))


def test_reconstruction_returns_none_when_never_evaluated():
    from pipeline.model import _reconstruct_calibration
    assert _reconstruct_calibration({}) is None
    assert _reconstruct_calibration({"conformal_quantile": 0.05}) is None  # partial


# ── Critic surfaces staleness rather than hiding it ────────────────────────────

def test_grade_evidence_reports_when_it_was_last_measured():
    from agents.critic_agent import grade_evidence

    state = {
        "forecast_available": True,
        "eval_rank_ic": 0.05, "eval_rank_ic_t": 2.5,
        "eval_hit_rate": 58.0, "eval_baseline_hit_rate": 52.0,
        "eval_beats_naive": True,
        "eval_evaluated_at": "2026-08-09T03:00:00+00:00",
    }
    _, reasons = grade_evidence(state)
    assert any("2026-08-09" in r for r in reasons)


def test_grade_evidence_insufficient_message_explains_never_evaluated():
    from agents.critic_agent import grade_evidence

    grade, reasons = grade_evidence({"forecast_available": True})
    assert grade == "INSUFFICIENT"
    assert any("weekly evaluation" in r.lower() for r in reasons)


# ── Lever 2: tuning budget is the agreed moderate cut, not a silent drift ─────

def test_tuning_budget_matches_the_agreed_moderate_cut():
    from pipeline import model, tuning

    assert tuning.N_TRIALS == 25
    assert tuning.INNER_FOLDS == 3
    assert model.EVAL_N_FOLDS == 5
    assert model.EVAL_TUNE_TRIALS == 10
