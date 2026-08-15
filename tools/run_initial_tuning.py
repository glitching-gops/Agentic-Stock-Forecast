"""
run_initial_tuning.py — run once to generate tuned_params/ for all 100 stocks.
Place this in the scripts/ directory.
"""
import sys
import os
sys.path.append(os.getcwd())

from data.universe import get_universe
from pipeline.model import load_features_for_ticker
from pipeline.tuning import tune_hyperparameters
from data.db import get_engine
import time

engine = get_engine()
results = []

print(f"Starting initial tuning for {len(get_universe())} stocks...")

for i, ticker in enumerate(get_universe()):
    start = time.time()
    try:
        X, y = load_features_for_ticker(ticker, engine)
        if len(X) < 100:
            print(f"[{i+1}/100] {ticker}: skipped — only {len(X)} rows")
            continue
        best_params = tune_hyperparameters(ticker, X, y, force=False)
        elapsed = round(time.time() - start, 1)
        print(f"[{i+1}/100] {ticker}: done in {elapsed}s")
        results.append({"ticker": ticker, "status": "OK", "time": elapsed})
    except Exception as e:
        print(f"[{i+1}/100] {ticker}: FAILED — {e}")
        results.append({"ticker": ticker, "status": "FAILED", "error": str(e)})

import pandas as pd
df = pd.DataFrame(results)
print("\n=== Tuning Summary ===")
print(df.to_string())
print(f"\nSuccessful: {(df['status']=='OK').sum()}")
print(f"Failed:     {(df['status']=='FAILED').sum()}")
