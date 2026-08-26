from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from data.db import get_engine

router = APIRouter()

@router.get("/{ticker}/headlines")
def get_headlines(ticker: str):
    """
    Returns the 5 most recent headlines for a given ticker from the sentiment table.
    """
    engine = get_engine()
    try:
        query = text("""
            SELECT headline, sentiment_label, sentiment_score, date 
            FROM sentiment 
            WHERE ticker = :ticker 
            ORDER BY date DESC 
            LIMIT 5
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"ticker": ticker.upper()})
            rows = result.mappings().all()
            
        return [dict(row) for row in rows]
    except Exception as e:
        # Not str(e): a connection failure carries the database hostname and its
        # resolved IP, and this response is public. Re-raising DBAPIError is what
        # lets api/main.py serve an outage as 503 + no-store — catching it here
        # turned a dead database into a 500 that named the host, which is the
        # defect the other three routers were fixed for. This one was missed.
        if isinstance(e, DBAPIError):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch headlines for {ticker}.",
        ) from e
