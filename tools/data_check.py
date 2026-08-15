import yfinance as yf
from data.universe import get_universe

existing_20 = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "SBIN.NS", "BAJFINANCE.NS", "TCS.NS", "INFY.NS", "WIPRO.NS",
    "HCLTECH.NS", "TECHM.NS", "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS",
    "ONGC.NS", "BPCL.NS", "POWERGRID.NS", "MARUTI.NS", "TATAMOTORS.NS", "LT.NS"
]

new_tickers = [t for t in get_universe() if t not in existing_20]
print(f"Checking {len(new_tickers)} new tickers...\n")

sufficient = []
insufficient = []

for ticker in new_tickers:
    try:
        df = yf.download(ticker, period="2y", interval="1d",
                         auto_adjust=True, progress=False)
        row_count = len(df)
        if row_count >= 200:
            sufficient.append((ticker, row_count))
            print(f"OK     {ticker}: {row_count} rows")
        else:
            insufficient.append((ticker, row_count))
            print(f"SKIP   {ticker}: only {row_count} rows")
    except Exception as e:
        insufficient.append((ticker, 0))
        print(f"ERROR  {ticker}: {e}")

print(f"\n=== Summary ===")
print(f"Sufficient data (>=200 rows): {len(sufficient)}")
print(f"Insufficient or error:        {len(insufficient)}")
if insufficient:
    print(f"Skipping: {[t for t, _ in insufficient]}")
