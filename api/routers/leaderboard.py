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

    # Count matches BEFORE the limit. Reporting len(entries) instead made
    # `total` a restatement of the page size, so a leaderboard holding only 5
    # rows and one holding 500 both reported whatever `limit` happened to be —
    # exactly the signal needed to tell "the pipeline wrote nothing" apart
    # from "the page is capped".
    total_matching = len(df)

    # Competition ranking on the sort key, computed BEFORE the page is sliced.
    #
    # Positional numbering asserted an ordering that does not exist. Under the
    # 2-of-3 evidence gate, 93 of 95 rows share composite_score 0.0, and
    # `range(1, len+1)` handed them ranks 3 through 95 purely from the order
    # pandas happened to leave them in — so the API published "rank 47" as a
    # fact about a stock the score cannot separate from 92 others. Tied rows
    # now share a rank (1, 2, 3, 3, 3, ...), which says what is true: the
    # ordering ran out. A row with no value for the sort key gets no rank.
    if key in df.columns:
        df["rank"] = (df[key].rank(method="min", ascending=SORTABLE[key])
                             .astype("Int64"))
    else:
        df["rank"] = pd.array(range(1, len(df) + 1), dtype="Int64")

    df = df.head(limit).reset_index(drop=True)

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
        total=total_matching,
        last_updated=str(last_updated),
        filters_applied=filters,
    )
