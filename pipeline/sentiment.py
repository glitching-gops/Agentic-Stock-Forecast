"""
pipeline/sentiment.py — News headline ingestion from Google News RSS.

SCORING IS DELIBERATELY GONE. This module used to run every headline through
ProsusAI/FinBERT, but FinBERT needs torch, and torch was removed from
requirements in Phase 0 (it existed only for the archived LSTM, which never
wrote a checkpoint and so never produced a forecast). The load therefore failed
on every single production run:

    Error loading FinBERT: name 'torch' is not defined

and ``fetch_and_score`` returned 0 immediately, so no headline was ever stored
either. Meanwhile the dashboard kept rendering a sentiment gauge, and
``get_aggregate_sentiment`` kept returning 0.0 — which the UI displayed as
NEUTRAL. A number that is structurally always neutral, presented as a reading
of the news, is the same defect the audit was convened over: a displayed value
that does not mean what it appears to mean.

So the headlines are now stored unscored, and ``get_aggregate_sentiment``
returns None rather than 0.0 when nothing has been scored — None is
"unavailable", 0.0 is "measured as neutral", and the two must not share a
representation. Headlines themselves are still worth showing a reader.

Sentiment was never a model feature and still is not; it was removed as one
under audit finding F7 (it existed only for the current date, so every training
row held 0.0 while the row being predicted held a real value). Restoring a
scorer is a Phase 2 item (T2.4) and needs a dated news archive to be
backtestable at all — Google News RSS serves no archive.
"""

import pandas as pd
import feedparser
from datetime import datetime
import urllib.parse
from sqlalchemy import text

from data.db import get_engine
from data.tickers import get_company

# Label written for every headline while no scorer exists. Distinct from
# "neutral", which would be a claim about the headline's content.
UNSCORED = "unscored"


def fetch_and_score(single_ticker=None, tickers=None):
    """
    Fetches recent headlines and stores them unscored. See the module
    docstring for why no sentiment model runs here.
    """
    engine = get_engine()
    if single_ticker:
        tickers_to_process = [single_ticker]
    elif tickers:
        tickers_to_process = list(tickers)
    else:
        from data.universe import get_universe
        tickers_to_process = get_universe()

    total_new_rows = 0
    today_str = datetime.today().strftime('%Y-%m-%d')
    
    for ticker in tickers_to_process:
        company_name = get_company(ticker)
        print(f"Fetching news for {company_name} ({ticker})...")
        
        query = urllib.parse.quote(f"{company_name} NSE")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        try:
            feed = feedparser.parse(url)
            entries = feed.entries[:5]  # Top 5 headlines
            
            if not entries:
                print(f"No news found for {ticker}.")
                df = pd.DataFrame([{
                    "date": today_str,
                    "ticker": ticker,
                    "headline": "No news available today.",
                    "sentiment_label": UNSCORED,
                    "sentiment_score": None,
                }])
            else:
                df = pd.DataFrame([{
                    "date": today_str,
                    "ticker": ticker,
                    "headline": entry.title,
                    "sentiment_label": UNSCORED,
                    "sentiment_score": None,
                } for entry in entries])
            
            # Check existing to avoid duplicates
            # Read all headlines for this ticker and date
            existing = pd.read_sql(
                text("SELECT headline FROM sentiment WHERE ticker = :t AND date = :d"),
                con=engine, params={"t": ticker, "d": today_str},
            )["headline"].tolist()
            
            new_rows = df[~df["headline"].isin(existing)]
            
            if not new_rows.empty:
                new_rows.to_sql("sentiment", con=engine, if_exists="append", index=False)
                total_new_rows += len(new_rows)
                print(f"Stored {len(new_rows)} new sentiment rows for {ticker}.")
            else:
                print(f"No new sentiment rows for {ticker}.")
                
        except Exception as e:
            safe_err = str(e).encode("ascii", "backslashreplace").decode("ascii")
            print(f"Error processing sentiment for {ticker}: {safe_err}")

    print(f"Sentiment processing complete. Total new rows: {total_new_rows}")
    return total_new_rows

def get_aggregate_sentiment(ticker, date=None) -> float | None:
    """
    Aggregate sentiment for a ticker on a date, or None if nothing is scored.

    Returns None — not 0.0 — when no scored headline exists, which is currently
    always, because no scorer runs (see the module docstring). 0.0 is a
    measurement meaning "the news balanced out"; None means "there is no
    measurement". Collapsing the second into the first is what let the
    dashboard report NEUTRAL for every stock in the universe indefinitely.

    Positive labels add their score, negative labels subtract it, and any other
    label (including UNSCORED) is ignored rather than counted as neutral —
    otherwise a batch of unscored headlines would drag a real reading toward
    zero.
    """
    engine = get_engine()

    if date is None:
        dates_df = pd.read_sql(
            text("SELECT MAX(date) AS max_date FROM sentiment WHERE ticker = :t"),
            con=engine, params={"t": ticker},
        )
        if dates_df.empty or pd.isna(dates_df.iloc[0]["max_date"]):
            return None
        date = dates_df.iloc[0]["max_date"]

    df = pd.read_sql(
        text("SELECT sentiment_label, sentiment_score FROM sentiment "
             "WHERE ticker = :t AND date = :d"),
        con=engine, params={"t": ticker, "d": date},
    )
    if df.empty:
        return None

    scored = df[df["sentiment_label"].isin(["positive", "negative"])
                & df["sentiment_score"].notna()]
    if scored.empty:
        return None

    signed = scored.apply(
        lambda row: float(row["sentiment_score"])
        * (1.0 if row["sentiment_label"] == "positive" else -1.0),
        axis=1,
    )
    return max(-1.0, min(1.0, float(signed.mean())))
