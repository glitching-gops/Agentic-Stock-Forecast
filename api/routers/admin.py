"""
Admin routes — protected by ADMIN_API_KEY, triggers pipeline work as a
background task so the HTTP response returns immediately.

/run/{ticker} and /run-all drive the LangGraph agent path (trading_data_node,
external_data_node, forecasting_node, critic_node) for one or all tickers.
As of Lever 1, that path calls pipeline.model.forecast_ticker_daily() — cached
hyperparameters, no Optuna search — so these stay cheap even triggered from
Render's single free-tier core.

/run-daily-pipeline is the light daily job (see scheduler.run_pipeline_job).

/run-weekly-evaluation is the EXPENSIVE purged walk-forward evaluation
(scheduler.run_weekly_evaluation_job) — hundreds of XGBoost fits per ticker.
It is what OOM-killed this instance when it ran daily. As of Lever 4, an
external GitHub Actions workflow runs this directly against Supabase instead
of through Render (see .github/workflows/weekly-evaluation.yml); this route
exists only as a manual/fallback trigger and should not be called routinely
from here — prefer re-running the GH Actions workflow.
"""
from fastapi import APIRouter, Depends, BackgroundTasks

from api.dependencies import verify_api_key

router = APIRouter()


def _run_pipeline(ticker: str):
    try:
        from agents.graph import run_graph
        run_graph(ticker)
    except Exception as e:                                      # noqa: BLE001
        print(f"[Admin] Pipeline failed for {ticker}: {e}")


def _run_daily_pipeline():
    try:
        from scheduler import run_pipeline_job
        run_pipeline_job()
    except Exception as e:                                      # noqa: BLE001
        print(f"[Admin] Daily pipeline failed: {e}")


def _run_weekly_evaluation():
    try:
        from scheduler import run_weekly_evaluation_job
        run_weekly_evaluation_job()
    except Exception as e:                                      # noqa: BLE001
        print(f"[Admin] Weekly evaluation failed: {e}")


@router.post("/run/{ticker}", dependencies=[Depends(verify_api_key)])
def trigger_pipeline(ticker: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_pipeline, ticker.upper())
    return {
        "status": "accepted",
        "message": f"Pipeline triggered for {ticker.upper()} in the background"
    }


@router.post("/run-all", dependencies=[Depends(verify_api_key)])
def trigger_all_pipelines(background_tasks: BackgroundTasks):
    from data.universe import get_universe

    universe = get_universe()
    for ticker in universe:
        background_tasks.add_task(_run_pipeline, ticker)
    return {
        "status": "accepted",
        "message": f"Pipeline triggered for all {len(universe)} stocks in the background"
    }


@router.post("/run-daily-pipeline", dependencies=[Depends(verify_api_key)])
def trigger_daily_pipeline(background_tasks: BackgroundTasks):
    """
    Runs the light daily pipeline: universe sync -> fetch OHLCV -> compute
    signals -> fetch sentiment -> fetch macro -> forecast with cached
    hyperparameters. No Optuna search. Safe to trigger from Render.
    """
    background_tasks.add_task(_run_daily_pipeline)
    return {
        "status": "accepted",
        "message": "Daily pipeline triggered in the background",
    }


@router.post("/run-weekly-evaluation", dependencies=[Depends(verify_api_key)])
def trigger_weekly_evaluation(background_tasks: BackgroundTasks):
    """
    Runs the expensive purged walk-forward evaluation for the whole universe.

    Manual/fallback only. This is the workload that OOM-killed a free-tier
    Render instance running it daily; the primary path is the GitHub Actions
    weekly-evaluation.yml workflow, which runs this against Supabase without
    touching Render's CPU/RAM at all. Prefer re-running that workflow over
    calling this endpoint unless GH Actions is unavailable.
    """
    background_tasks.add_task(_run_weekly_evaluation)
    return {
        "status": "accepted",
        "message": ("Weekly evaluation triggered in the background. This is "
                    "expensive — prefer the GitHub Actions workflow when possible."),
    }


# ── The pooled model in SHADOW (Stage 2, 2026-09-24) ──────────────────────────
#
# Read-only inspection of what pipeline/pooled_shadow.py writes. Admin-gated
# and linked from nowhere: the shadow model is NOT published, and nothing here
# may be read by a public page until the cutover session (docs/
# dashboard-switch-preregistration.md). The booster itself is never returned.

_SHADOW_MODEL_COLS = ("model_id, pooled_version, trained_at, train_first_date, "
                      "train_last_date, n_train_rows, params_json, coverage_json, "
                      "coverage_status, walk_forward_json, config_hash, data_hash, "
                      "env_hash, env_json, git_sha, runtime_seconds")


def _shadow_read(sql: str, params: dict | None = None):
    import pandas as pd
    from sqlalchemy import text

    from api.serialization import records
    from data.db import get_engine, is_missing_relation

    try:
        return records(pd.read_sql(text(sql), get_engine(), params=params or {}))
    except Exception as exc:                                    # noqa: BLE001
        if is_missing_relation(exc):
            return None
        raise


@router.get("/shadow/summary", dependencies=[Depends(verify_api_key)])
def shadow_summary():
    """The latest shadow model, its panel statement and its grade counts."""
    models = _shadow_read(f"SELECT {_SHADOW_MODEL_COLS} FROM shadow_models "
                          f"ORDER BY trained_at DESC LIMIT 1")
    if not models:
        return {"model": None, "note": "no shadow model has been written yet"}
    model = models[0]
    statement = _shadow_read("SELECT * FROM shadow_panel_statements WHERE model_id = :m",
                             {"m": model["model_id"]})
    latest = _shadow_read("SELECT forecast_date, COUNT(*) AS n FROM shadow_forecasts "
                          "WHERE model_id = :m GROUP BY forecast_date "
                          "ORDER BY forecast_date DESC LIMIT 5",
                          {"m": model["model_id"]})
    return {"model": model, "panel_statement": (statement or [None])[0],
            "recent_forecast_dates": latest or []}


@router.get("/shadow/forecasts", dependencies=[Depends(verify_api_key)])
def shadow_forecasts(date: str | None = None):
    """Every shadow forecast for one date (default: the latest)."""
    if date is None:
        latest = _shadow_read("SELECT MAX(forecast_date) AS d FROM shadow_forecasts")
        date = latest[0]["d"] if latest else None
    if date is None:
        return {"date": None, "forecasts": []}
    rows = _shadow_read("SELECT * FROM shadow_forecasts WHERE forecast_date = :d "
                        "ORDER BY ticker", {"d": date})
    return {"date": date, "forecasts": rows or []}
