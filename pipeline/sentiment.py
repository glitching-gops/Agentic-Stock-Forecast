# sentiment.py — Fetches news headlines and computes sentiment using FinBERT
# Uses Google News RSS and ProsusAI/finbert
# Stores results in the sentiment table

import pandas as pd
import feedparser
from datetime import datetime
import urllib.parse
from sqlalchemy import text

from data.db import get_engine
from data.tickers import get_company

# FinBERT model lazy loading
finbert = None

def get_finbert():
    global finbert
    if finbert is None:
        print("Lazy loading FinBERT model...")
        try:
            from transformers import pipeline
            finbert = pipeline("text-classification", model="ProsusAI/finbert")
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
    return finbert

def fetch_and_score(single_ticker=None, tickers=None):
    engine = get_engine()
    if single_ticker:
        tickers_to_process = [single_ticker]
    elif tickers:
        tickers_to_process = list(tickers)
    else:
        from data.universe import get_universe
        tickers_to_process = get_universe()

    model = get_finbert()
    if model is None:
        print("FinBERT model not loaded. Skipping sentiment analysis.")
        return 0
        
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
                print(f"No news found for {ticker}. Inserting neutral fallback.")
                df = pd.DataFrame([{
                    "date": today_str,
                    "ticker": ticker,
                    "headline": "No news available today.",
                    "sentiment_label": "neutral",
                    "sentiment_score": 0.0
                }])
            else:
                headlines = [entry.title for entry in entries]
                # Run FinBERT
                results = model(headlines)
                
                rows_to_insert = []
                for headline, result in zip(headlines, results):
                    label = result['label']
                    score = result['score']
                    
                    rows_to_insert.append({
                        "date": today_str,
                        "ticker": ticker,
                        "headline": headline,
                        "sentiment_label": label,
                        "sentiment_score": score
                    })
                    
                df = pd.DataFrame(rows_to_insert)
            
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

def get_aggregate_sentiment(ticker, date=None):
    """
    Returns the aggregate sentiment score for a ticker on a given date.
    Positive = +score, Negative = -score, Neutral = 0
    """
    engine = get_engine()

    if date is None:
        dates_df = pd.read_sql(
            text("SELECT MAX(date) AS max_date FROM sentiment WHERE ticker = :t"),
            con=engine, params={"t": ticker},
        )
        if dates_df.empty or pd.isna(dates_df.iloc[0]["max_date"]):
            return 0.0
        date = dates_df.iloc[0]["max_date"]

    df = pd.read_sql(
        text("SELECT * FROM sentiment WHERE ticker = :t AND date = :d"),
        con=engine, params={"t": ticker, "d": date},
    )
    
    if df.empty:
        return 0.0
        
    # Calculate aggregate score
    agg_score = 0.0
    for _, row in df.iterrows():
        if row["sentiment_label"] == "positive":
            agg_score += row["sentiment_score"]
        elif row["sentiment_label"] == "negative":
            agg_score -= row["sentiment_score"]
            
    # Return average over number of headlines, strictly bounded between -1 and 1
    avg_score = agg_score / len(df)
    return max(-1.0, min(1.0, avg_score))
