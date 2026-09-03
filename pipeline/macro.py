# macro.py — Fetches macroeconomic indicators
# Uses yfinance to pull USDINR, INDIAVIX, and NIFTY 50
# Stores results in the macro table

import yfinance as yf
import pandas as pd
from data.db import get_engine

class FlowParseError(Exception):
    """The response parsed to nothing usable. Raised rather than defaulted."""


def parse_fii_dii(payload: list[dict]) -> dict | None:
    """
    Turns NSE's FII/DII response into one row, or None.

    THE FIELD NAMES WERE WRONG FOR THE LIFE OF THIS PROJECT, and the failure was
    silent by construction. The old parser read `fiiNetFlow` and `diiNetFlow`
    with a default of "0"; NSE serves neither key. It returns two ROWS
    discriminated by `category`, with the figure in `netValue`:

        [{"category":"DII","date":"02-Sep-2026","netValue":"2812.98",
          "buyValue":"17639.89","sellValue":"14826.91"},
         {"category":"FII/FPI", ...}]

    So `.get(..., "0")` defaulted every figure to zero, `fillna(0.0)` and the
    empty-frame branch painted the rest of history the same, and the result was
    two columns holding exactly ONE distinct value across all 2,601 macro rows.
    Both were in `model.FEATURES` and `panel.MACRO_COLS`, so every per-ticker
    model has been fitted on two constant-zero inputs — under a comment claiming
    they were wired in "rather than left as dead columns (audit finding F15)".

    This is the third instance of one defect class here, after FinBERT's
    permanently-neutral gauge: A FAILED MEASUREMENT STORED AS A VALID NEUTRAL
    VALUE. 100% non-null, structurally constant, and invisible to every check
    that counts nulls. Hence no default: a category we cannot parse raises, and
    a row missing either side returns None rather than half a reading.
    """
    def num(raw) -> float | None:
        try:
            return float(str(raw).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    row: dict = {}
    seen_date = None
    for entry in payload or []:
        category = str(entry.get("category", "")).upper()
        if "FII" in category or "FPI" in category:
            prefix = "fii"
        elif "DII" in category:
            prefix = "dii"
        else:
            continue
        seen_date = seen_date or entry.get("date")
        row[f"{prefix}_net"] = num(entry.get("netValue"))
        row[f"{prefix}_buy"] = num(entry.get("buyValue"))
        row[f"{prefix}_sell"] = num(entry.get("sellValue"))

    if not seen_date or row.get("fii_net") is None or row.get("dii_net") is None:
        return None

    parsed = pd.to_datetime(seen_date, format="%d-%b-%Y", errors="coerce")
    if pd.isna(parsed):
        return None

    row["date"] = parsed.strftime("%Y-%m-%d")
    return row


def fetch_fii_dii_flows(expected_date: str | None = None) -> dict | None:
    """
    Fetches the latest FII/DII cash-market figures from NSE.

    ONE SESSION ONLY, AND THE ENDPOINT LIES ABOUT IT. `?date=` is accepted and
    silently IGNORED — measured 2026-09-03, a request for 01-Sep-2026 returned
    02-Sep-2026. So a backfill loop over historical dates would fetch the same
    latest row for every date and write today's flows across ten years of
    history, with nothing raising. `expected_date` is the guard: when a caller
    believes it asked for a specific date, a mismatch returns None instead of
    the wrong day's data.
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
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=headers, timeout=10,
        )
        response.raise_for_status()
        row = parse_fii_dii(response.json())
    except Exception as e:
        print(f"[Macro] FII/DII fetch failed: {e}")
        return None

    if row is None:
        print("[Macro] FII/DII: response held no parsable FII and DII pair")
        return None

    if expected_date and row["date"] != expected_date:
        print(f"[Macro] FII/DII: asked for {expected_date}, served "
              f"{row['date']} — NSE ignores ?date=; refusing to store it")
        return None

    return row


def store_flows(row: dict | None, engine=None) -> int:
    """
    Appends one session's flows to `market_flows`. Never updates.

    ON CONFLICT DO NOTHING, for the same reason `forecast_outcomes` never
    updates: NSE marks these figures provisional and revises them after
    custodial confirmation, so an upsert would silently rewrite history. And it
    lives outside `macro` because `fetch_and_store` refreshes that table with a
    DELETE-range — a flow figure published once and never served again would be
    destroyed by the next price refresh.
    """
    if not row:
        return 0
    from datetime import datetime, timezone
    from sqlalchemy import text as _text

    from data.db import get_engine as _get_engine, to_native_params

    engine = engine or _get_engine()
    params = to_native_params({
        "date": row["date"],
        "fii_net": row.get("fii_net"), "fii_buy": row.get("fii_buy"),
        "fii_sell": row.get("fii_sell"), "dii_net": row.get("dii_net"),
        "dii_buy": row.get("dii_buy"), "dii_sell": row.get("dii_sell"),
        "source": "nse_fiidiiTradeReact",
        "first_seen": datetime.now(timezone.utc).isoformat(),
    })
    with engine.connect() as conn:
        result = conn.execute(_text("""
            INSERT INTO market_flows
                (date, fii_net, fii_buy, fii_sell, dii_net, dii_buy, dii_sell,
                 source, first_seen)
            VALUES (:date, :fii_net, :fii_buy, :fii_sell, :dii_net, :dii_buy,
                    :dii_sell, :source, :first_seen)
            ON CONFLICT (date) DO NOTHING
        """), params)
        conn.commit()
    return int(result.rowcount or 0)

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
        
        # FII/DII NO LONGER TOUCHES THIS FRAME. It goes to `market_flows`, which
        # is append-only and is not wiped by the DELETE-range below. The merge
        # that used to live here is what painted ten years of zeros: a left join
        # onto a frame the endpoint can only ever supply ONE date for, followed
        # by fillna(0.0). The columns stay on the `macro` TABLE so old rows still
        # read; nothing writes them any more.
        store_flows(fetch_fii_dii_flows())


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
