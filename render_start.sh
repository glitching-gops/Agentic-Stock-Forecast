#!/bin/bash
set -e

echo "=== ZeRO Stock Forecast — Render Startup ==="
echo "Port: $PORT"
echo "Checking model files..."

# Ensure models directory exists
mkdir -p models/joblib models/lstm models/meta

# Only download if models directory is empty
if [ ! -f "models/joblib/RELIANCE.NS.joblib" ]; then
    echo "Models not found — downloading from model-store branch..."

    if [ -z "$GITHUB_TOKEN" ]; then
        echo "ERROR: GITHUB_TOKEN is not set. Cannot download models."
        exit 1
    fi

    # Download zip (forcing HTTP/1.1 to avoid protocol errors)
    curl -L --http1.1 \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3.raw" \
        "https://api.github.com/repos/glitching-gops/Agentic-Stock-Forecast/contents/models_store.zip?ref=model-store" \
        -o models_store.zip

    echo "Download complete. Size: $(du -h models_store.zip | cut -f1). Extracting..."
    # Validate every member path before writing. extractall() honours absolute
    # paths and '..' segments in the archive, so a tampered zip could write
    # outside the working directory (audit finding F15).
    python3 -c "
import os
import zipfile

DEST = os.path.abspath('.')
print('Starting extraction...')
with zipfile.ZipFile('models_store.zip', 'r') as zf:
    for member in zf.infolist():
        target = os.path.abspath(os.path.join(DEST, member.filename))
        if not target.startswith(DEST + os.sep) and target != DEST:
            raise SystemExit(f'Refusing unsafe archive path: {member.filename}')
    zf.extractall(DEST)
print('Extraction complete.')
"
    rm models_store.zip
    echo "Models ready."
else
    echo "Models already present — skipping download."
fi

echo "Initialising database..."
# Use a timeout for DB init to prevent hanging
timeout 30s python3 -c "from data.db import init_db; init_db(); print('DB init OK')" || echo "DB init timed out or failed (ignoring to allow startup)"

echo "Starting FastAPI server on port $PORT..."
# exec ensures uvicorn receives signals directly from Render
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info
