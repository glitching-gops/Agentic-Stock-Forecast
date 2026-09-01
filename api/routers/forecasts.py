"""
GET /api/forecasts          — the current forecast for every stock.
GET /api/forecasts/{ticker} — the latest stored forecast for one stock.

Reads from the database rather than re-running the pipeline per request.

THE LIST ENDPOINT REPLACES /api/leaderboard, and the difference is not the
route name. That endpoint computed a competition RANK with a SQL window
function and ordered by `composite_score`. Both are gone: the evidence gate
clears three of ninety-six tickers, which is what chance produces, so an
ordering over the remaining ninety-three tied zeros published a comparison the
measurement cannot support. This returns the same rows with no ordering
claim — sorted by ticker, which asserts nothing.
"""
import json
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import inspect, text

from api.schemas.forecast import (
    CurrentForecast,
    EvaluationEvidence,
    ForecastListResponse,
    ForecastResponse,
)
from data.db import get_engine, is_missing_relation

router = APIRouter()

VERDICTS = {"APPROVED", "FLAGGED", "REJECTED"}
EVIDENCE_GRADES = {"STRONG", "WEAK", "INSUFFICIENT"}


def _to_bool(value) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return bool(value)


def _current_columns() -> frozenset[str]:
    """
    Columns actually present on forecast_current.

    Several are added lazily by data.db's ALTER-TABLE-and-swallow idiom, so a
    database that predates one would 500 on a query naming it. NOT cached:
    the old leaderboard version memoised this for the process lifetime, which
    was fine while the schema only moved at deploy time and is a trap now that
    a rename runs inside init_db. is_missing_relation is the only thing allowed
    to open the soft path — an outage must reach the 503 handler.
    """
    try:
        return frozenset(c["name"] for c in
                         inspect(get_engine()).get_columns("forecast_current"))
    except Exception as exc:                                    # noqa: BLE001
        if not is_missing_relation(exc):
            raise
        return frozenset()


@router.get("", response_model=ForecastListResponse)
def list_forecasts(
    sector:   Optional[str] = Query(None),
    verdict:  Optional[str] = Query(None),
    evidence: Optional[str] = Query(None, description="STRONG / WEAK / INSUFFICIENT"),
    limit:    int = Query(200, ge=1, le=500),
):
    columns = _current_columns()
    if not columns:
        return ForecastListResponse(forecasts=[], total=0,
                                    last_updated=datetime.now().isoformat(),
                                    filters_applied={})

    filters: dict = {}
    clauses: list[str] = []
    params: dict = {"limit": limit}

    if sector:
        clauses.append("sector = :sector")
        params["sector"] = sector
        filters["sector"] = sector

    if verdict:
        upper = verdict.upper()
        if upper == "APPROVED_OR_FLAGGED":
            # Literal constants, not user input — safe to inline.
            clauses.append("critic_verdict IN ('APPROVED', 'FLAGGED')")
            filters["verdict"] = upper
        elif upper in VERDICTS:
            clauses.append("critic_verdict = :verdict")
            params["verdict"] = upper
            filters["verdict"] = upper

    if evidence:
        upper = evidence.upper()
        if upper in EVIDENCE_GRADES:
            clauses.append("forecast_confidence = :evidence")
            params["evidence"] = upper
            filters["evidence"] = upper

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    # ORDER BY ticker, and nothing is interpolated from the request. The old
    # endpoint took a `sort_by` from the query string and reached ORDER BY with
    # it as an identifier, which cannot be a bind parameter — so it needed a
    # SORTABLE allowlist acting as an injection boundary. Removing the ordering
    # removed the boundary along with it.
    #
    # `total` still comes from a window function so it counts the whole filtered
    # set rather than the page: computing it as len(rows) after LIMIT was a real
    # defect once.
    sql = text(f"""
        SELECT *,
               COUNT(*) OVER () AS "total_matching",
               MAX(last_updated) OVER () AS "table_last_updated"
        FROM forecast_current
        {where_sql}
        ORDER BY ticker
        LIMIT :limit
    """)

    try:
        df = pd.read_sql(sql, con=get_engine(), params=params)
    except Exception as exc:                                    # noqa: BLE001
        # Only the case this was written for: a column the deployed schema does
        # not carry yet. A connection failure must NOT be served as an empty
        # list with a 200 — an outage cached as data is what left the dashboard
        # showing zero stocks for a day after the database recovered.
        if not is_missing_relation(exc):
            raise
        return ForecastListResponse(forecasts=[], total=0,
                                    last_updated=datetime.now().isoformat(),
                                    filters_applied=filters)

    if df.empty:
        return ForecastListResponse(forecasts=[], total=0,
                                    last_updated=datetime.now().isoformat(),
                                    filters_applied=filters)

    total_matching = int(df["total_matching"].iloc[0])
    last_updated = df["table_last_updated"].iloc[0]

    forecasts = []
    for _, row in df.iterrows():
        record = row.where(row.notna(), None).to_dict()
        record["benchmark_sector_specific"] = _to_bool(row.get("benchmark_sector_specific"))
        record["eval_beats_random_walk"] = _to_bool(row.get("eval_beats_random_walk"))
        for key in ("evaluated_at", "last_updated"):
            if record.get(key) is not None:
                record[key] = str(record[key])
        forecasts.append(CurrentForecast(**{
            k: v for k, v in record.items() if k in CurrentForecast.model_fields
        }))

    return ForecastListResponse(
        forecasts=forecasts,
        total=total_matching,
        last_updated=str(last_updated if last_updated is not None else datetime.now()),
        filters_applied=filters,
    )


@router.get("/{ticker}", response_model=ForecastResponse)
def get_forecast(ticker: str):
    engine = get_engine()
    # No try/except around the query. It used to raise a 500 whose detail was
    # str(exc), which on a connection failure is the database hostname and its
    # resolved IP — published to any caller. A DBAPIError now reaches the
    # application handler in api/main.py and is served as a 503.
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM forecasts WHERE ticker = :ticker "
                 "ORDER BY last_updated DESC LIMIT 1"),
            {"ticker": ticker.upper()},
        ).mappings().first()

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
