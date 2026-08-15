import subprocess
import time
import os
import sys
from dotenv import load_dotenv

load_dotenv()

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from data.universe import get_universe

api_key    = os.getenv("ADMIN_API_KEY", "admin_secret_key_here")
tickers    = get_universe()
batch_size = 5

print(f"Running pipeline for {len(tickers)} stocks in batches of {batch_size}...\n")

for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i + batch_size]
    print(f"Batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}: {batch}")

    for ticker in batch:
        result = subprocess.run([
            "curl", "-s", "-X", "POST",
            f"http://localhost:8000/api/admin/run/{ticker}",
            "-H", f"x-api-key: {api_key}"
        ], capture_output=True, text=True)
        status = "OK" if "accepted" in result.stdout.lower() else "ERR"
        print(f"  {status} {ticker}: {result.stdout.strip()[:60]}")

    if i + batch_size < len(tickers):
        print(f"  Waiting 20 seconds before next batch...")
        time.sleep(20)

print("\nAll batches triggered.")
print("Wait 10 minutes for all pipelines to complete before running verification.")
