"""
GET /api/stocks — returns the full list of tickers with company name and sector.
"""
from fastapi import APIRouter
from api.schemas.stock import StockList, StockInfo
from api.serialization import records
from data.tickers import get_company, get_sector

router = APIRouter()

@router.get("", response_model=StockList)
def get_stocks():
    """
    The current tradable universe.

    Sourced from data.universe, which applies a point-in-time rule (index
    membership + liquidity floor + listing history). The previous hard-coded
    list was produced by ranking stocks on the model's own reported accuracy
    (audit finding F4).
    """
    from data.universe import get_universe

    stocks = [
        StockInfo(ticker=t, company=get_company(t), sector=get_sector(t))
        for t in get_universe()
    ]
    return StockList(stocks=stocks, total=len(stocks))

@router.get("/{ticker}/signals")
def get_signals(ticker: str, days: int = 30):
    """
    Returns the last N days of signal data for a ticker.
    Only queries columns that exist in the signals table.
    """
    from data.db import get_engine
    from sqlalchemy.exc import DBAPIError
    from fastapi import HTTPException
    from sqlalchemy import text
    import pandas as pd

    days = max(1, min(days, 2000))
    engine = get_engine()

    # Bound parameters only. The previous fallback branch interpolated the
    # ticker straight into SQL, and this endpoint takes it from the URL
    # (audit finding F15).
    try:
        df = pd.read_sql(
            text("""
                SELECT date, close, rsi, macd_hist, bb_width, obv,
                       sma_20, ema_9, ema_21, ema_50, atr_14,
                       stoch_k, williams_r, roc_10, vroc_10,
                       prox_52w, lag1_ret, lag5_ret, dev_sma50,
                       bb_upper, bb_lower, hurst,
                       sector_rel_5d, sector_rel_10d, sector_rel_20d,
                       earnings_surprise,
                       target_return, target_excess_return, benchmark_return,
                       benchmark_ticker
                FROM signals
                WHERE ticker = :ticker
                ORDER BY date DESC
                LIMIT :days
            """),
            con=engine,
            params={"ticker": ticker.upper(), "days": days},
        )
    except Exception as e:
        # Not str(e): a connection failure carries the database hostname and
        # its resolved IP, and this response is public. api/main.py serves
        # DBAPIError as a 503.
        if isinstance(e, DBAPIError):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch signals for {ticker}.",
        ) from e

    df = df.sort_values("date", ascending=True)
    # df.where(df.notna(), None) claimed to do this and did not: pandas coerces
    # None back to NaN in a float column, and json.dumps then 500s the whole
    # response. See api/serialization.py.
    return records(df)
