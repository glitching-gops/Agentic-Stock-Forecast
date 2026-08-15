import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from data.db import get_engine
from data.universe import get_universe
from sqlalchemy import text

engine  = get_engine()
keep    = get_universe()
placeholders = ",".join([f"'{t}'" for t in keep])

tables_with_ticker = [
    "ohlcv", "signals", "sentiment", "forecasts",
    "leaderboard", "model_metadata"
]

with engine.connect() as conn:
    for table in tables_with_ticker:
        try:
            result = conn.execute(text(
                f"DELETE FROM {table} "
                f"WHERE ticker NOT IN ({placeholders})"
            ))
            conn.commit()
            print(f"Cleaned {table:20s}: {result.rowcount} rows deleted")
        except Exception as e:
            print(f"Error cleaning {table}: {e}")

# macro table has no ticker column — leave it unchanged
print("\nmacro table: unchanged (no ticker column)")
