"""
Admin routes — protected by ADMIN_API_KEY, triggers pipeline work as a
background task so the HTTP response returns immediately.

/run/{ticker} and /run-all drive the LangGraph agent path (trading_data_node,
external_data_node, forecasting_node, critic_node) for one or all tickers.
That path recomputes signals from whatever is ALREADY in the ohlcv table —
it does not fetch new prices. Useful for re-forecasting after a code change,
not for a daily refresh.

/run-daily-pipeline is the one an external scheduler should call. It runs the
full sequence the in-process APScheduler job runs (universe sync, fetch,
signals, sentiment, macro, train) — added because the in-process scheduler
only fires if the instance happens to be awake at 18:30 IST. On Render's free
tier, an idle instance sleeps, the scheduled job silently never runs, and
nothing surfaces the gap. An external cron hitting this endpoint is what
actually guarantees the pipeline runs daily; see .github/workflows/.
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
    Runs the full daily pipeline: universe sync -> fetch OHLCV -> compute
    signals -> fetch sentiment -> fetch macro -> train and forecast.

    Call this from an external scheduler (see .github/workflows/daily-pipeline.yml)
    rather than relying on the in-process APScheduler job, which does not fire
    if the instance is asleep when 18:30 IST arrives.
    """
    background_tasks.add_task(_run_daily_pipeline)
    return {
        "status": "accepted",
        "message": "Daily pipeline triggered in the background",
    }
