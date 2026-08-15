"""
POST /api/admin/run/{ticker} — triggers an on-demand pipeline run.
Protected by API key header.
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from api.dependencies import verify_api_key

router = APIRouter()

def _run_pipeline(ticker: str):
    try:
        from agents.graph import run_graph
        run_graph(ticker)
    except Exception as e:
        print(f"[Admin] Pipeline failed for {ticker}: {e}")

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
