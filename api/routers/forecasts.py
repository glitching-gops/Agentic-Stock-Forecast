"""
GET /api/forecasts/{ticker} — the latest stored forecast for one stock.

Reads from the database rather than re-running the pipeline per request.
"""
import json

import pandas as pd
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from api.schemas.forecast import EvaluationEvidence, ForecastResponse
from data.db import get_engine

router = APIRouter()


def _to_bool(value) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return bool(value)


@router.get("/{ticker}", response_model=ForecastResponse)
def get_forecast(ticker: str):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM forecasts WHERE ticker = :ticker "
                     "ORDER BY last_updated DESC LIMIT 1"),
                {"ticker": ticker.upper()},
            ).mappings().first()
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for {ticker}. Run the pipeline first.",
        )

    data = dict(row)

    raw_flags = data.get("critic_flags") or "[]"
    if isinstance(raw_flags, str):
        try:
            flags = json.loads(raw_flags)
        except (json.JSONDecodeError, TypeError):
            flags = []
    else:
        flags = list(raw_flags)

    evaluation = EvaluationEvidence(
        rank_ic=data.get("eval_rank_ic"),
        hit_rate=data.get("eval_hit_rate"),
        baseline_hit_rate=data.get("eval_baseline_hit_rate"),
        beats_random_walk=_to_bool(data.get("eval_beats_random_walk")),
        model_version=data.get("model_version"),
        evaluated_at=str(data["evaluated_at"]) if data.get("evaluated_at") else None,
    )

    return ForecastResponse(
        ticker=data["ticker"],
        company=data.get("company"),
        sector=data.get("sector"),
        pred_excess_return=data.get("pred_excess_return"),
        benchmark_ticker=data.get("benchmark_ticker"),
        benchmark_name=data.get("benchmark_name"),
        benchmark_sector_specific=_to_bool(data.get("benchmark_sector_specific")),
        current_price=data.get("current_price"),
        forecast_price=data.get("forecast_price"),
        direction=data.get("direction"),
        change_pct=data.get("change_pct"),
        random_walk_price=data.get("random_walk_price"),
        interval_low=data.get("interval_low"),
        interval_high=data.get("interval_high"),
        interval_coverage=data.get("interval_coverage"),
        prob_outperform=data.get("prob_outperform"),
        evaluation=evaluation,
        forecast_confidence=data.get("forecast_confidence"),
        signal_narrative=data.get("signal_narrative"),
        critic_verdict=data.get("critic_verdict"),
        critic_reasoning=data.get("critic_reasoning"),
        critic_flags=[str(f) for f in flags],
        critic_source=data.get("critic_confidence_adjustment"),
        forecast_available=data.get("direction") != "UNAVAILABLE",
        universe_rule=data.get("universe_rule"),
        last_updated=str(data.get("last_updated")) if data.get("last_updated") else None,
    )
