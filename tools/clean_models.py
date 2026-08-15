import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
    
from data.universe import get_universe

keep = set(get_universe())

def ticker_to_safe(ticker: str) -> str:
    return ticker.replace(".", "_")

deleted = []
kept    = []

# Also delete tuned_params JSON for dropped stocks
tuned_dir = "tuned_params"

all_tickers_on_disk = set()
for d in ["models/joblib", "models/lstm", "models/meta"]:
    if os.path.exists(d):
        for f in os.listdir(d):
            # Extract ticker from filename
            safe = f.replace(".joblib", "").replace("_lstm.pt", "") \
                    .replace("_scaler.joblib", "").replace("_meta.joblib", "").replace("_lstm", "")
            # Convert safe name back to ticker format
            ticker = safe.replace("_NS", ".NS")
            all_tickers_on_disk.add(ticker)

for d in ["models/joblib", "models/lstm", "models/meta"]:
    if not os.path.exists(d):
        continue
    for fname in os.listdir(d):
        fpath = os.path.join(d, fname)
        # Check if this file belongs to a kept ticker
        belongs_to_kept = any(
            ticker_to_safe(t) in fname for t in keep
        )
        if not belongs_to_kept:
            os.remove(fpath)
            deleted.append(fpath)
        else:
            kept.append(fpath)

# Delete tuned params for dropped stocks
if os.path.exists(tuned_dir):
    for fname in os.listdir(tuned_dir):
        if fname.endswith("_params.json"):
            belongs_to_kept = any(
                ticker_to_safe(t) in fname for t in keep
            )
            if not belongs_to_kept:
                os.remove(os.path.join(tuned_dir, fname))
                deleted.append(os.path.join(tuned_dir, fname))

print(f"Deleted: {len(deleted)} model files")
print(f"Kept:    {len(kept)} model files")
print(f"\nExpected kept: ~{len(keep) * 4} files (4 per stock: joblib, lstm, scaler, meta)")
