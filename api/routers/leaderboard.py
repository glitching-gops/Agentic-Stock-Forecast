"""
GET /api/leaderboard — stocks ranked by composite score.

The composite is now predicted excess return and conviction, multiplied by an
evidence grade from purged walk-forward evaluation. Previously ~75 of its 100
points came from leaked in-sample metrics that barely varied across stocks
(audit finding F9).
"""
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Query
from sqlalchemy import text

from api.schemas.leaderboard import LeaderboardEntry, LeaderboardResponse
from data.db import get_engine

router = APIRouter()

SORTABLE = {
    "composite_score": False,
    "upside_pct": False,
    "pred_excess_return": False,
    "prob_outperform": False,
    "eval_rank_ic": False,
    "eval_hit_rate": False,
}


@router.get("", response_model=LeaderboardResponse)
def get_leaderboard(
    sector:     Optional[str] = Query(None),
    verdict:    Optional[str] = Query(None),
    evidence:   Optional[str] = Query(None, description="STRONG / WEAK / INSUFFICIENT"),
    sort_by:    str = Query("composite_score"),
    limit:      int = Query(20, ge=1, le=200),
):
    engine = get_engine()
    df = pd.read_sql(text("SELECT * FROM leaderboard"), con=engine)

    if df.empty:
        return LeaderboardResponse(entries=[], total=0,
                                   last_updated=datetime.now().isoformat(),
                                   filters_applied={})

    filters: dict = {}

    if sector:
        df = df[df["sector"] == sector]
        filters["sector"] = sector

    if verdict:
        upper = verdict.upper()
        if upper == "APPROVED_OR_FLAGGED":
            df = df[df["critic_verdict"].isin(["APPROVED", "FLAGGED"])]
            filters["verdict"] = upper
        elif upper in {"APPROVED", "FLAGGED", "REJECTED"}:
            df = df[df["critic_verdict"] == upper]
            filters["verdict"] = upper

    if evidence:
        upper = evidence.upper()
        if upper in {"STRONG", "WEAK", "INSUFFICIENT"}:
            df = df[df["forecast_confidence"] == upper]
            filters["evidence"] = upper

    key = sort_by if sort_by in SORTABLE else "composite_score"
    if key in df.columns:
        df = df.sort_values(key, ascending=SORTABLE[key], na_position="last")

    df = df.head(limit).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    def _bool(value):
        return None if value is None or pd.isna(value) else bool(value)

    entries = []
    for _, row in df.iterrows():
        record = row.where(row.notna(), None).to_dict()
        record["benchmark_sector_specific"] = _bool(row.get("benchmark_sector_specific"))
        record["eval_beats_random_walk"] = _bool(row.get("eval_beats_random_walk"))
        entries.append(LeaderboardEntry(**{
            k: v for k, v in record.items()
            if k in LeaderboardEntry.model_fields
        }))

    last_updated = df["last_updated"].max() if "last_updated" in df.columns else datetime.now()

    return LeaderboardResponse(
        entries=entries,
        total=len(entries),
        last_updated=str(last_updated),
        filters_applied=filters,
    )
