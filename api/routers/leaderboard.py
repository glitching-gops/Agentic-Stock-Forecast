"""
GET /api/leaderboard — stocks ranked by composite score.

The composite is now predicted excess return and conviction, multiplied by an
evidence grade from purged walk-forward evaluation. Previously ~75 of its 100
points came from leaked in-sample metrics that barely varied across stocks
(audit finding F9).

Filtering, ranking, counting and paging all happen in ONE query. This endpoint
used to `SELECT * FROM leaderboard` and do all four in pandas, which is O(whole
table) per request no matter how small the page. That is affordable at 95 rows
and stops being affordable well before it stops being correct, so it is fixed
while the row count still makes it a cheap change.
"""
from datetime import datetime
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Query
from sqlalchemy import inspect, text

from api.schemas.leaderboard import LeaderboardEntry, LeaderboardResponse
from data.db import get_engine, is_missing_relation

router = APIRouter()

# Column -> ascending?  This dict is also the SQL injection boundary: `sort_by`
# arrives from the query string and reaches the ORDER BY clause as an
# identifier, which cannot be a bind parameter. Nothing may be interpolated
# into the statement unless it is a key of this dict.
SORTABLE = {
    "composite_score": False,
    "upside_pct": False,
    "pred_excess_return": False,
    "prob_outperform": False,
    "eval_rank_ic": False,
    "eval_hit_rate": False,
}

VERDICTS = {"APPROVED", "FLAGGED", "REJECTED"}
EVIDENCE_GRADES = {"STRONG", "WEAK", "INSUFFICIENT"}


@lru_cache(maxsize=1)
def _leaderboard_columns() -> frozenset[str]:
    """
    Columns actually present on the leaderboard table.

    The old pandas path could check `key in df.columns` for free because it had
    already read the whole table. Sorting in SQL means naming a column that may
    not exist yet: several are added lazily by data.db's ALTER-TABLE-and-swallow
    idiom, so a database that predates a column would 500 rather than degrade.
    Cached for the process lifetime — the ALTERs run once at startup, and a
    deploy is a new process.
    """
    try:
        return frozenset(c["name"] for c in
                         inspect(get_engine()).get_columns("leaderboard"))
    except Exception as exc:                                    # noqa: BLE001
        if not is_missing_relation(exc):
            raise
        return frozenset()


def _empty(filters: dict) -> LeaderboardResponse:
    return LeaderboardResponse(entries=[], total=0,
                               last_updated=datetime.now().isoformat(),
                               filters_applied=filters)


@router.get("", response_model=LeaderboardResponse)
def get_leaderboard(
    sector:     Optional[str] = Query(None),
    verdict:    Optional[str] = Query(None),
    evidence:   Optional[str] = Query(None, description="STRONG / WEAK / INSUFFICIENT"),
    sort_by:    str = Query("composite_score"),
    limit:      int = Query(20, ge=1, le=200),
):
    columns = _leaderboard_columns()
    if not columns:
        return _empty({})

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

    key = sort_by if sort_by in SORTABLE else "composite_score"
    if key not in columns:
        key = "composite_score" if "composite_score" in columns else None

    if key is None:
        order_sql = ""
        rank_sql = "NULL AS \"rank\""
    else:
        # `key` is a SORTABLE dict key that also exists on the table, so this
        # interpolation cannot carry anything from the request.
        direction = "ASC" if SORTABLE[key] else "DESC"
        order_sql = f"ORDER BY {key} {direction} NULLS LAST"

        # Competition ranking over the FULL filtered set. Window functions are
        # evaluated before LIMIT, so this is the rank among all matches rather
        # than within the page — the same distinction that made `total` wrong
        # when it was computed as len(entries).
        #
        # Tied rows SHARE a rank (1, 2, 3, 3, 3, ...). Positional numbering
        # asserted an ordering that does not exist: under the 2-of-3 evidence
        # gate 93 of 95 rows share composite_score 0.0, and numbering them off
        # published "rank 47" as a fact about a stock the score cannot separate
        # from 92 others. A row with no value for the sort key gets no rank.
        rank_sql = (f'CASE WHEN {key} IS NULL THEN NULL '
                    f'ELSE RANK() OVER ({order_sql}) END AS "rank"')

    last_updated_sql = ('MAX(last_updated) OVER () AS "table_last_updated"'
                        if "last_updated" in columns
                        else 'NULL AS "table_last_updated"')

    sql = text(f"""
        SELECT *,
               COUNT(*) OVER () AS "total_matching",
               {last_updated_sql},
               {rank_sql}
        FROM leaderboard
        {where_sql}
        {order_sql}
        LIMIT :limit
    """)

    try:
        df = pd.read_sql(sql, con=get_engine(), params=params)
    except Exception as exc:                                    # noqa: BLE001
        # Only the case this was written for: a column named in ORDER BY or
        # WHERE that the deployed schema does not carry yet. A connection
        # failure must not be served as an empty leaderboard with a 200.
        if not is_missing_relation(exc):
            raise
        _leaderboard_columns.cache_clear()   # schema may have moved under us
        return _empty(filters)

    if df.empty:
        return _empty(filters)

    total_matching = int(df["total_matching"].iloc[0])
    last_updated = df["table_last_updated"].iloc[0]
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")

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

    return LeaderboardResponse(
        entries=entries,
        total=total_matching,
        last_updated=str(last_updated if last_updated is not None else datetime.now()),
        filters_applied=filters,
    )
