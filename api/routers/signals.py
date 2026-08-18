"""
GET /api/signals/{ticker} — the historical signals frame plus its most recent
row as `latest_signals`, used by the dashboard chart and signals view. Reads
directly from the signals table.
"""
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from api.serialization import json_safe, records
from data.db import get_engine

router = APIRouter()


@router.get("/{ticker}")
def get_signals(ticker: str, days: int = Query(200, ge=1, le=2000)):
    """
    Up to `days` rows for `ticker`, oldest first, plus `latest_signals`.

    Bound parameters only. The previous version tried a bound query and, on any
    exception, fell back to a second query with the ticker and row count
    interpolated straight into the SQL string — the exact pattern audit finding
    F15 removed from the neighbouring router, left behind here. It was not
    reachable in normal operation (the bound query does not raise), which is
    precisely what let it survive: an injection sink one unrelated exception
    away from being live. There is no fallback now.
    """
    try:
        df = pd.read_sql(
            text("SELECT * FROM signals WHERE ticker = :ticker "
                 "ORDER BY date DESC LIMIT :days"),
            con=get_engine(),
            params={"ticker": ticker.upper(), "days": days},
        )
    except Exception as exc:                                    # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch signals for {ticker}: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No signal data found for {ticker}. Run the pipeline first.",
        )

    df = df.sort_values("date", ascending=True)

    # NaN must be replaced on the way out, not masked with df.where(): pandas
    # coerces None back to NaN in a float column, and json.dumps then refuses
    # the response with a 500. See api/serialization.py.
    rows = records(df)

    latest = rows[-1]
    latest_signals = {
        key: json_safe(value)
        for key, value in latest.items()
        if key not in ("date", "ticker", "target")
    }

    return {
        "ticker": ticker.upper(),
        "signals_df": rows,
        "latest_signals": latest_signals,
        "rows": len(rows),
    }
