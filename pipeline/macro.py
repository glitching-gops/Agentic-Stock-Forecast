# macro.py — Fetches macroeconomic indicators
# Uses yfinance to pull USDINR, INDIAVIX, and NIFTY 50
# Stores results in the macro table

import yfinance as yf
import pandas as pd
from data.db import get_engine

def fetch_fii_dii_flows() -> pd.DataFrame:
    """
    Fetches FII and DII net cash market flow data from NSE India.
    Returns a DataFrame with columns: date, fii_net_flow, dii_net_flow.
    Values are in crores (Indian Rupees).
    Falls back to an empty DataFrame if the fetch fails.

    Source: NSE India FII/DII trade data endpoint.
    Note: NSE requires a session cookie — the function establishes
    a browser-like session before making the API call.
    """
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":  "application/json",
        "Referer": "https://www.nseindia.com/",
    }

    try:
        session = requests.Session()
        session.get(
            "https://www.nseindia.com",
            headers=headers,
            timeout=10
        )

        response = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        records = []
        for entry in data:
            try:
                records.append({
                    "date": entry.get("date", ""),
                    "fii_net_flow": float(
                        str(entry.get("fiiNetFlow", "0")).replace(",", "")
                    ),
                    "dii_net_flow": float(
                        str(entry.get("diiNetFlow", "0")).replace(",", "")
                    ),
                })
            except (ValueError, TypeError):
                continue

        if not records:
            print("[Macro] FII/DII: no records parsed from response")
            return pd.DataFrame(columns=["date", "fii_net_flow", "dii_net_flow"])

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(
            df["date"], format="%d-%b-%Y", errors="coerce"
        )
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df.sort_values("date", ascending=True).reset_index(drop=True)

    except Exception as e:
        print(f"[Macro] FII/DII fetch failed: {e}")
        return pd.DataFrame(columns=["date", "fii_net_flow", "dii_net_flow"])

def fetch_and_store():
    engine = get_engine()
    print("Fetching macroeconomic data...")
    
    try:
        # Match the OHLCV window (pipeline.fetch.PERIOD). A shorter macro
        # window silently truncates the training set at the join in
        # load_features_for_ticker.
        tickers = ["USDINR=X", "^INDIAVIX", "^NSEI"]
        data = yf.download(tickers, period="10y", interval="1d", auto_adjust=True)
        
        if data.empty:
            print("[Macro] No macro data returned from yfinance. Skipping update.")
            return 0
            
        # Extract the Close prices
        close_df = data["Close"].copy()
        
        # Flatten columns if multi-index (happens when downloading multiple tickers)
        if isinstance(close_df.columns, pd.MultiIndex):
            close_df.columns = [str(col[0]) for col in close_df.columns]
            
        # Forward fill only. The previous code also called bfill(), which fills
        # a missing value from a LATER observation — look-ahead bias (audit
        # finding F12). Leading gaps, where USDINR and NSE calendars diverge at
        # the start of the window, are dropped instead of being back-filled.
        close_df.ffill(inplace=True)
        close_df = close_df.dropna(how="any")

        # Reset index to get Date as column
        df = close_df.reset_index()
        df.rename(columns={
            "Date": "date",
            "USDINR=X": "usdinr",
            "^INDIAVIX": "india_vix",
            "^NSEI": "nifty"
        }, inplace=True)
        
        df["date"] = df["date"].astype(str).str[:10]
        
        import numpy as np
        # Compute Nifty returns and protect against division by zero (inf)
        df["nifty_5d_return"] = df["nifty"].pct_change(5)
        df["nifty_20d_return"] = df["nifty"].pct_change(20)
        
        df["nifty_5d_return"] = df["nifty_5d_return"].replace([np.inf, -np.inf], np.nan)
        df["nifty_20d_return"] = df["nifty_20d_return"].replace([np.inf, -np.inf], np.nan)
        
        # Drop the original nifty column as it's not in the schema and drop NaNs from pct_change
        df = df.drop(columns=["nifty"])
        df.dropna(inplace=True)
        
        # Merge FII/DII flows
        fii_dii = fetch_fii_dii_flows()
        if not fii_dii.empty:
            df = df.merge(fii_dii, on="date", how="left")
            df["fii_net_flow"] = df["fii_net_flow"].fillna(0.0)
            df["dii_net_flow"] = df["dii_net_flow"].fillna(0.0)
        else:
            df["fii_net_flow"] = 0.0
            df["dii_net_flow"] = 0.0
        
        # yfinance can return more than one row for the current session (a
        # partial intraday bar alongside the daily one). Keep the last per date.
        df = df.drop_duplicates(subset=["date"], keep="last")

        # Overwrite the refreshed window rather than appending unseen dates.
        # Appending leaves rows computed under an older Nifty adjustment basis
        # in place, the same defect as F11 in the OHLCV path.
        from sqlalchemy import text

        with engine.connect() as conn:
            conn.execute(text("DELETE FROM macro WHERE date >= :start"),
                         {"start": df["date"].min()})
            df.to_sql("macro", con=conn, if_exists="append", index=False)
            conn.commit()

        print(f"Stored {len(df)} macro rows (window refreshed from {df['date'].min()}).")
        return len(df)


    except Exception as e:
        print(f"Error fetching macro data: {e}")
        return 0
