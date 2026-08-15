import sys, os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from data.db import get_engine
from data.universe import get_universe
import pandas as pd

engine = get_engine()

lb = pd.read_sql("""
    SELECT ticker, sector, mape, directional_accuracy,
           composite_score, critic_verdict, forecast_confidence,
           last_updated
    FROM leaderboard
    ORDER BY composite_score DESC
""", con=engine)

print(f"=== Final {len(get_universe())}-Stock Leaderboard ===\n")
print(f"Total stocks:        {len(lb)}")
print(f"Mean MAPE:           {lb['mape'].mean():.2f}%")
print(f"Mean Dir Accuracy:   {lb['directional_accuracy'].mean():.2f}%")
print(f"Mean Composite:      {lb['composite_score'].mean():.2f}")
print(f"APPROVED:            {(lb['critic_verdict'] == 'APPROVED').sum()}")
print(f"FLAGGED:             {(lb['critic_verdict'] == 'FLAGGED').sum()}")
print(f"REJECTED:            {(lb['critic_verdict'] == 'REJECTED').sum()}")

print(f"\n=== Per-sector summary ===")
sector_summary = lb.groupby("sector").agg(
    stocks=("ticker", "count"),
    avg_mape=("mape", "mean"),
    avg_dir=("directional_accuracy", "mean"),
    avg_score=("composite_score", "mean"),
    approved=("critic_verdict", lambda x: (x == "APPROVED").sum())
).round(2)
print(sector_summary.to_string())

# Check for missing tickers
all_tickers   = set(get_universe())
lb_tickers    = set(lb["ticker"].tolist())
missing       = all_tickers - lb_tickers
print(f"\nMissing from leaderboard: {sorted(missing) if missing else 'None'}")
