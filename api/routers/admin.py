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
