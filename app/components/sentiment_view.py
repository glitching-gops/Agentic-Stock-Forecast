import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import requests
def render_sentiment_view(ticker: str, sentiment_score: float | None) -> None:
    """
    Renders the 5 most recent headlines for the ticker, plus an aggregate
    sentiment gauge when an aggregate actually exists.

    ``sentiment_score`` is None when no headline has been scored, which is
    currently always — no scorer runs (see pipeline/sentiment.py). The gauge is
    suppressed in that case rather than drawn at 0.0: a needle parked in the
    middle of a red/grey/green dial reads as "we measured the news and it came
    out neutral", which would be a claim this pipeline cannot support.
    """
    if sentiment_score is None:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem; padding: 2rem; background-color: #142252; border-radius: 12px; border: 1px solid #1E3A6E;">
            <h2 style="color: #94A3B8; font-weight: 400;">Overall sentiment for {ticker}</h2>
            <h1 style="color: #94A3B8; font-size: 3rem; margin-top: 0.5rem;">NOT SCORED</h1>
            <p style="color: #94A3B8; margin: 0;">Headlines are collected but not scored — no sentiment
            model runs in this pipeline. Sentiment is not a model input either way.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        sentiment_text = "POSITIVE" if sentiment_score > 0.2 else "NEGATIVE" if sentiment_score < -0.2 else "NEUTRAL"
        sent_color = "#06D6A0" if sentiment_text == "POSITIVE" else "#EF476F" if sentiment_text == "NEGATIVE" else "#94A3B8"

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem; padding: 2rem; background-color: #142252; border-radius: 12px; border: 1px solid #1E3A6E;">
            <h2 style="color: #94A3B8; font-weight: 400;">Overall sentiment for {ticker}</h2>
            <h1 style="color: {sent_color}; font-size: 3rem; margin-top: 0.5rem;">{sentiment_text}</h1>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        if sentiment_score is None:
            st.markdown("""
            <div class="stCard" style="padding: 1rem;">
                <p style="margin: 0; color: #94A3B8;">No aggregate to plot.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            fig3 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sentiment_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#FFFFFF"},
                    'bar': {'color': "#00B4D8"},
                    'bgcolor': "#0D1B3E",
                    'borderwidth': 2,
                    'bordercolor': "#1E3A6E",
                    'steps': [
                        {'range': [-1, -0.2], 'color': "#EF476F"},
                        {'range': [-0.2, 0.2], 'color': "#475569"},
                        {'range': [0.2, 1], 'color': "#06D6A0"}
                    ],
                }
            ))
            fig3.update_layout(height=350, paper_bgcolor="#0D1B3E", font=dict(color="#FFFFFF"))
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    with col2:
        st.markdown("<h3 style='margin-bottom: 1rem; color:#FFFFFF;'>Recent Headlines</h3>", unsafe_allow_html=True)
        
        # Replace DB call with API fetch gracefully
        api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        try:
            r = requests.get(f"{api_url}/api/sentiment/{ticker}/headlines", timeout=60)
            if r.status_code == 200:
                headlines_df = pd.DataFrame(r.json())
            else:
                headlines_df = pd.DataFrame()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            st.warning(
                "⚠️ Backend is taking too long to respond (likely cold start). "
                "Please wait 60 seconds and refresh."
            )
            headlines_df = pd.DataFrame()
        except Exception:
            headlines_df = pd.DataFrame()
        
        if not headlines_df.empty:
            for _, row in headlines_df.iterrows():
                headline = row['headline']
                label = str(row['sentiment_label'] or "unscored").lower()
                score = row['sentiment_score']

                if label == "positive":
                    tag_class = "tag-bullish"
                elif label == "negative":
                    tag_class = "tag-bearish"
                else:
                    tag_class = "tag-neutral"

                # No scorer runs, so score is NULL for every row written now.
                # Printing "0.00" beside an UNSCORED badge would contradict it.
                score_text = "" if pd.isna(score) else f"{float(score):.2f}"

                st.markdown(f"""
                <div class="stCard" style="padding: 1rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;">
                        <span style="font-weight: 500; color: #FFFFFF; font-size: 0.95rem; line-height: 1.4;">{headline}</span>
                        <div>
                            <span class="{tag_class}" style="white-space: nowrap; margin-right: 8px;">{label.upper()}</span>
                            <span style="color: #94A3B8; font-size: 0.8rem;">{score_text}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stCard" style="padding: 1rem;">
                <p style="margin: 0; color: #94A3B8;">No recent headlines found in the database for this stock.</p>
            </div>
            """, unsafe_allow_html=True)
