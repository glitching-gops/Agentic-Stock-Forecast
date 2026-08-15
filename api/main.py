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
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from data.db import init_db

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
_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "https://glitching-gops-zer0.streamlit.app,"
    "https://agentic-stock-forecast.streamlit.app,"
    "http://localhost:8501",
)

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

@app.get("/")
def read_root():
    return {"message": "ZeRO Stock Forecast API is running", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
