#!/bin/bash
set -e

echo "=== ZeRO Stock Forecast — Render Startup ==="
echo "Port: $PORT"

# The pre-trained-model download step that used to live here is gone.
# Under Phase 0, pipeline.model.fit_production_model() always retrains fresh
# and only ever WRITES to models/joblib/*.joblib — nothing reads an existing
# file to skip training (that warm-path was removed because it caused bug
# F10: a stale/missing joblib silently produced a flat forecast). Downloading
# a 21MB archive from GitHub's Contents API added 45-90s to every cold start
# and, on a slow or interrupted transfer, would crash the whole deploy — that
# is exactly what took the site down: curl exited 18 (partial transfer) and
# `set -e` killed the script before uvicorn ever started.
mkdir -p models/joblib

echo "Initialising database..."
# Use a timeout for DB init to prevent hanging
timeout 30s python3 -c "from data.db import init_db; init_db(); print('DB init OK')" || echo "DB init timed out or failed (ignoring to allow startup)"

echo "Starting FastAPI server on port $PORT..."
# exec ensures uvicorn receives signals directly from Render
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info
