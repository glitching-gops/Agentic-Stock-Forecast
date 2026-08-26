"""
FastAPI application entry point.
Initialises the database, mounts routers, and configures CORS.

Does NOT start the in-process APScheduler. The first production run pegged
this instance's one free-tier CPU core for over an hour running Optuna
searches and eventually OOM-killed the process — running that same work
in-process, in the same container that serves API requests, is what caused
it. As of Lever 4, an external GitHub Actions workflow runs the daily and
weekly jobs directly against Supabase (see .github/workflows/), and this
process only ever serves reads plus on-demand admin-triggered forecasts.
scheduler.start_scheduler() still exists for local development (see
scheduler.py's module docstring) but is intentionally not called here.
"""
import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pandas.errors import DatabaseError as PandasDatabaseError
from sqlalchemy.exc import DBAPIError
from data.db import init_db

logger = logging.getLogger(__name__)

from api.routers import stocks, forecasts, leaderboard, admin, signals, sentiment

@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB init is handled by render_start.sh. Nothing else to do on startup —
    # see the module docstring for why the scheduler is not started here.
    yield
    # Run on shutdown (if needed)

app = FastAPI(
    title="Indian Stock Market Forecasting API",
    description="Agentic multi-agent system for NSE stock price forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Restrict origins. The wildcard was paired with an admin route whose
# /run-all variant kicks off a full pipeline run for the entire universe
# (audit finding F15). Set ALLOWED_ORIGINS as a comma-separated list.
#
# The two Streamlit origins were dropped with the app/ directory. Note what the
# Next.js frontend does NOT need: every page there renders on the server and
# revalidates on a timer, so the fetches reach this API from Vercel's runtime
# rather than from a browser, and CORS never applies to them. The default below
# is local development only. If a client component is ever given a direct fetch,
# its deployed origin has to be added to ALLOWED_ORIGINS on Render or it will
# fail in the browser and work everywhere else — including in `next build`.
_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(stocks.router,     prefix="/api/stocks",     tags=["Stocks"])
app.include_router(forecasts.router,  prefix="/api/forecasts",  tags=["Forecasts"])
app.include_router(leaderboard.router,prefix="/api/leaderboard",tags=["Leaderboard"])
app.include_router(admin.router,      prefix="/api/admin",      tags=["Admin"])
app.include_router(signals.router,    prefix="/api/signals",    tags=["Signals"])
app.include_router(sentiment.router,  prefix="/api/sentiment",  tags=["Sentiment"])


@app.exception_handler(DBAPIError)
async def database_unavailable(request: Request, exc: DBAPIError) -> JSONResponse:
    """
    A database failure is an OUTAGE, and must be served as one.

    Every read path here is behind a cache: the Next.js frontend revalidates on
    a timer and keeps whatever it last received. So the failure mode that
    matters is not the error itself, it is a caller storing the error as
    though it were data. 503 plus no-store says "this is not an answer";
    a 200 with an empty list says "there are no stocks", which is what this
    API did during a live outage.

    The detail is deliberately generic. `str(exc)` on a connection failure
    contains the database hostname and its resolved IP, and this response is
    public.
    """
    logger.error("database unavailable on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The forecast database is unavailable. "
                           "No data could be read; this is an outage rather "
                           "than an empty result."},
        headers={"Retry-After": "30", "Cache-Control": "no-store"},
    )


@app.exception_handler(PandasDatabaseError)
async def pandas_wrapped_database_error(request: Request,
                                        exc: PandasDatabaseError) -> JSONResponse:
    """
    pandas raises its OWN DatabaseError from a driver failure inside read_sql.

    That class is not a DBAPIError, so it does not reach the handler above.
    Only failures at CONNECT time arrive unwrapped; a pooled connection that
    has gone stale fails mid-query and arrives like this, which is exactly the
    shape a pooler outage takes once a connection has already been handed out.

    Only the wrapped driver failure is served as an outage. A DatabaseError
    raised for any other reason is our own defect and must stay a 500.
    """
    if isinstance(exc.__cause__, DBAPIError):
        return await database_unavailable(request, exc.__cause__)
    raise exc

@app.get("/")
def read_root():
    return {"message": "ZeRO Stock Forecast API is running", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
