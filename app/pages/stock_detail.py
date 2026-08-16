# app/pages/stock_detail.py
import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def _get_api_base_url() -> str:
    """
    Reads API_BASE_URL from Streamlit secrets (Streamlit Cloud)
    with fallback to environment variable (local development)
    and finally to localhost for offline testing.
    """
    try:
        return st.secrets["API_BASE_URL"]
    except (KeyError, FileNotFoundError):
        return os.getenv("API_BASE_URL", "http://localhost:8000")

API_BASE_URL = _get_api_base_url()

# Check if a ticker was pre-selected from the leaderboard navigation
if "selected_ticker" in st.session_state:
    preselected = st.session_state["selected_ticker"]
    # Clear it so subsequent manual selections work correctly
    del st.session_state["selected_ticker"]
else:
    preselected = None

def fetch_forecast(ticker: str) -> dict:
    """Fetches the latest forecast for a ticker from the FastAPI backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/forecasts/{ticker}", timeout=60)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def fetch_signals(ticker: str) -> dict:
    """Fetches historical signals from the FastAPI backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/signals/{ticker}", timeout=60)
        if response.status_code == 404:
            return {"signals_df": [], "latest_signals": {}}
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"signals_df": [], "latest_signals": {}}

def fetch_leaderboard(sector=None, verdict=None, confidence=None, sort_by="composite_score") -> dict:
    """Fetches the leaderboard from the FastAPI backend with optional filters."""
    try:
        params = {"sort_by": sort_by, "limit": 100}
        if sector:     params["sector"]     = sector
        if verdict:    params["verdict"]    = verdict
        if confidence: params["evidence"] = confidence
        response = requests.get(f"{API_BASE_URL}/api/leaderboard", params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except:
        return {"entries": [], "last_updated": "N/A", "total": 0}

from app.components.chart import render_price_chart
from app.components.signals_view import render_signals_view
from app.components.sentiment_view import render_sentiment_view

# inject_custom_css() # Injected globally in main.py

# --- Sidebar ---
st.sidebar.markdown("<h2 style='color:#00B4D8; margin-bottom:0;'>ZeRO</h2><p style='color:#475569; margin-top:0;'>Agentic Stock Forecast v3</p>", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_stocks() -> list[dict]:
    """
    Fetches the full list of available stocks from the FastAPI backend.
    Returns a list of dicts with ticker, company, and sector keys.
    Falls back to an empty list if the API is unreachable.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/stocks",
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("stocks", [])
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.warning(
            "⚠️ Backend is taking too long to respond (likely cold start). "
            "Please wait 60 seconds and refresh."
        )
        return []
    except Exception as e:
        st.error(f"⚠️ Error fetching stock list: {e}")
        return []

stocks = fetch_stocks()

if not stocks:
    st.sidebar.error("No stocks available. Backend may be offline.")
    st.stop()

# Build display name → ticker mapping
company_to_ticker = {s["company"]: s["ticker"] for s in stocks}
ticker_to_company = {v: k for k, v in company_to_ticker.items()}
ticker_to_sector  = {s["ticker"]: s["sector"]  for s in stocks}

# Determine default index
default_company = (
    ticker_to_company.get(preselected, sorted(company_to_ticker.keys())[0])
    if preselected
    else sorted(company_to_ticker.keys())[0]
)
default_idx = sorted(company_to_ticker.keys()).index(default_company)

selected_company = st.sidebar.selectbox(
    "Select Stock",
    options=sorted(company_to_ticker.keys()),
    index=default_idx,
    key="stock_detail_selector"
)
selected_ticker = company_to_ticker[selected_company]
selected_sector = ticker_to_sector[selected_ticker]

# Stock info card
st.sidebar.markdown(f"""
<div class="stCard">
    <div style="font-size: 0.85rem; color: #94A3B8;">Sector</div>
    <div style="font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;">{selected_sector}</div>
    <div style="font-size: 0.85rem; color: #94A3B8;">Market Cap</div>
    <div style="font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;">Large Cap</div>
    <div style="font-size: 0.85rem; color: #94A3B8;">Exchange</div>
    <div style="font-weight: 600; color: #FFFFFF;">NSE</div>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("Refresh Data"):
    with st.spinner(f"Triggering pipeline for {selected_company}..."):
        try:
            requests.post(f"{API_BASE_URL}/api/admin/run/{selected_ticker}", headers={"x-api-key": os.getenv("ADMIN_API_KEY", "")})
        except:
            pass
        st.cache_data.clear()
        st.rerun()

# --- Data Loading ---
@st.cache_data(ttl=3600, show_spinner="Fetching forecast from backend...")
def get_agent_state(ticker):
    forecast = fetch_forecast(ticker)
    if not forecast:
        return None
    signals_data = fetch_signals(ticker)
    # Merge signals data into the state dict for the components
    forecast["signals_df"] = signals_data.get("signals_df", [])
    forecast["latest_signals"] = signals_data.get("latest_signals", {})
    return forecast

try:
    state = get_agent_state(selected_ticker)
except Exception as e:
    st.error(f"Error fetching data from API: {e}")
    st.stop()

if not state:
    st.warning(f"No forecast available for {selected_company}. Trigger the pipeline from the backend.")
    st.stop()

evidence_grade = state.get("forecast_confidence") or "INSUFFICIENT"
evaluation = state.get("evaluation") or {}
hit_rate = evaluation.get("hit_rate")
baseline_hit = evaluation.get("baseline_hit_rate")

if evidence_grade != "STRONG":
    edge_text = ""
    if hit_rate is not None and baseline_hit is not None:
        edge_text = (f" Out-of-sample hit rate {hit_rate:.1f}% against a "
                     f"{baseline_hit:.1f}% majority-class baseline.")
    st.sidebar.warning(
        f"⚠️ Held-out evidence for {selected_company} is **{evidence_grade}**."
        f"{edge_text} See the Agent Analysis tab for the full evidence trail."
    )

# df could be empty list of dicts, let's load it to DataFrame
signals_raw = state.get("signals_df", [])
if isinstance(signals_raw, pd.DataFrame):
    df = signals_raw
else:
    df = pd.DataFrame(signals_raw)

# Agent Status
last_updated = "N/A"
is_today = False
if not df.empty and "date" in df.columns:
    last_updated = df["date"].iloc[-1]
    is_today = last_updated == datetime.today().strftime('%Y-%m-%d')
    
status_color = "#06D6A0" if is_today else "#FFB703"

st.sidebar.markdown(f"""
<div class="stCard" style="padding: 1rem;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {status_color};"></div>
        <span style="font-size: 0.9rem; font-weight: 600; color:#FFFFFF;">Data Sync Status</span>
    </div>
    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px; padding-left: 18px;">Last updated: {state.get('last_updated', 'Unknown')}</div>
</div>
""", unsafe_allow_html=True)

if st.button("← Back to Leaderboard"):
    st.switch_page("pages/leaderboard.py")

st.title(f"{selected_company} ({selected_ticker})")

verdict = state.get("critic_verdict", "FLAGGED")
if verdict == "REJECTED":
    st.error("⚠️ **This forecast has been rejected by the Critic Agent.** Review the Agent Analysis tab before acting on this data.")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Signals", "Sentiment", "Agent Analysis"])

# ==========================================
# TAB 1: Overview
# ==========================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    current_price   = state.get("current_price") or 0.0
    forecast_price  = state.get("forecast_price")
    excess_return   = state.get("pred_excess_return")
    direction       = state.get("direction", "UNAVAILABLE")
    interval_low    = state.get("interval_low")
    interval_high   = state.get("interval_high")
    coverage        = state.get("interval_coverage")
    prob_outperform = state.get("prob_outperform")
    random_walk     = state.get("random_walk_price")
    benchmark_name  = state.get("benchmark_name") or state.get("benchmark_ticker") or "benchmark"
    sector_specific = state.get("benchmark_sector_specific")

    dir_color = ("#06D6A0" if direction == "OUTPERFORM"
                 else "#EF476F" if direction == "UNDERPERFORM" else "#94A3B8")
    dir_symbol = ("▲" if direction == "OUTPERFORM"
                  else "▼" if direction == "UNDERPERFORM" else "–")

    def _money(value) -> str:
        return f"₹{value:,.2f}" if value is not None else "—"

    with col1:
        st.markdown(f"""
        <div class="stCard">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem;">Current Price</div>
            <div style="color: #FFB703; font-size: 2rem; font-weight: 700;">{_money(current_price)}</div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 6px;">
                Random-walk forecast: {_money(random_walk)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        excess_text = f"{excess_return * 100:+.1f}%" if excess_return is not None else "—"
        interval_text = (f"{_money(interval_low)} – {_money(interval_high)}"
                         if interval_low is not None and interval_high is not None else "—")
        coverage_text = f"{coverage:.0%} interval" if coverage else "interval"

        st.markdown(f"""
        <div class="stCard">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem;">Implied 30-Session Target</div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                <div style="color: #FFB703; font-size: 2rem; font-weight: 700;">{_money(forecast_price)}</div>
                <div style="background-color: rgba(0,0,0,0.2); padding: 4px 8px; border-radius: 6px; color: {dir_color}; font-weight: 600;">{dir_symbol} {excess_text}</div>
            </div>
            <div style="color: #94A3B8; font-size: 0.8rem;">{coverage_text}: {interval_text}</div>
            <div style="color: #64748B; font-size: 0.72rem; margin-top: 6px;">
                Assumes {benchmark_name} is flat. The model forecasts relative performance only.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        prob_text = f"{prob_outperform:.0%}" if prob_outperform is not None else "—"
        prob_color = ("#06D6A0" if (prob_outperform or 0) > 0.55
                      else "#EF476F" if (prob_outperform or 1) < 0.45 else "#94A3B8")
        bench_note = ("" if sector_specific
                      else " (no sector index available — measured against NIFTY 50)")
        st.markdown(f"""
        <div class="stCard">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem;">P(outperform)</div>
            <div style="color: {prob_color}; font-size: 2rem; font-weight: 700;">{prob_text}</div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 6px;">
                vs {benchmark_name}{bench_note}. 50% = coin flip.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        grade_color = {"STRONG": "#06D6A0", "WEAK": "#FFB703",
                       "INSUFFICIENT": "#EF476F"}.get(evidence_grade, "#94A3B8")
        ic = evaluation.get("rank_ic")
        ic_text = f"Rank IC {ic:+.3f}" if ic is not None else "No IC recorded"
        hit_text = (f"Hit {hit_rate:.1f}% vs {baseline_hit:.1f}% baseline"
                    if hit_rate is not None and baseline_hit is not None else "")
        st.markdown(f"""
        <div class="stCard">
            <div style="color: #94A3B8; font-size: 0.9rem; margin-bottom: 0.5rem;">Held-Out Evidence</div>
            <div style="color: {grade_color}; font-size: 2rem; font-weight: 700;">{evidence_grade}</div>
            <div style="color: #64748B; font-size: 0.75rem; margin-top: 6px;">
                {ic_text}<br>{hit_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Performance figures come from purged walk-forward validation with a "
        "30-session embargo — folds the model never trained on — and are stated "
        "before transaction costs."
    )

    if not df.empty:
        render_price_chart(df)

# ==========================================
# TAB 2: Signals
# ==========================================
with tab2:
    latest = state.get("latest_signals", {})
    render_signals_view(df, latest)

# ==========================================
# TAB 3: Sentiment
# ==========================================
with tab3:
    agg_score = state.get("sentiment_score")   # None when nothing is scored
    render_sentiment_view(selected_ticker, agg_score)


# ==========================================
# TAB 4: Agent Analysis
# ==========================================
with tab4:
    st.markdown("""
    <div class="stInfoCard">
        <h4 style="color: #00B4D8; display: flex; align-items: center; gap: 8px; margin-top: 0;"><span style="font-size: 1.5rem;">🤖</span> Forecasting Agent — Signal Narrative</h4>
        <p style="color: #FFFFFF; font-size: 1.1rem; line-height: 1.6; margin-bottom: 0;">""" + state.get("signal_narrative", "No narrative generated.") + """</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem; color:#FFFFFF;'>Critic Agent Review</h3>", unsafe_allow_html=True)
    
    c_verdict = state.get("critic_verdict", "FLAGGED")
    v_class = f"badge-{c_verdict.lower()}"
    v_icon = "✓" if c_verdict == "APPROVED" else "⚠" if c_verdict == "FLAGGED" else "✕"
    
    st.markdown(f"""
<div style="margin-left: 20px; border-left: 2px solid #1E3A6E; padding-left: 20px;">
<div style="position: relative;">
<div style="position: absolute; left: -30px; top: 0; background-color: #0D1B3E; padding: 4px;">
<div style="width: 16px; height: 16px; border-radius: 50%; background-color: #00B4D8;"></div>
</div>
<div class="{v_class}" style="margin-bottom: 1rem;">{v_icon} VERDICT: {c_verdict}</div>
</div>
<div style="position: relative; margin-top: 20px;">
<div style="position: absolute; left: -30px; top: 0; background-color: #0D1B3E; padding: 4px;">
<div style="width: 16px; height: 16px; border-radius: 50%; background-color: #1E3A6E;"></div>
</div>
<div class="stCard" style="margin-bottom: 1rem;">
<div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Reasoning</div>
<div style="color: #FFFFFF; line-height: 1.6;">{state.get('critic_reasoning', '')}</div>
</div>
</div>
""", unsafe_allow_html=True)
    
    flags = state.get("critic_flags", [])
    if flags:
        st.markdown("""
<div style="position: relative; margin-top: 20px;">
<div style="position: absolute; left: -30px; top: 0; background-color: #0D1B3E; padding: 4px;">
<div style="width: 16px; height: 16px; border-radius: 50%; background-color: #EF476F;"></div>
</div>
<div style="margin-bottom: 1rem;">
<div style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Flags Raised</div>
""", unsafe_allow_html=True)
        
        for flag in flags:
            st.markdown(f"""
<div class="stCard" style="border-left: 4px solid #EF476F !important; padding: 1rem !important; margin-bottom: 0.5rem !important;">
<span style="color: #EF476F; font-weight: 600; margin-right: 8px;">⚠</span> <span style="color: #FFFFFF;">{flag}</span>
</div>
""", unsafe_allow_html=True)
            
        st.markdown("</div></div>", unsafe_allow_html=True)
        
    st.markdown(f"""
<div style="position: relative; margin-top: 20px;">
<div style="position: absolute; left: -30px; top: 0; background-color: #0D1B3E; padding: 4px;">
<div style="width: 16px; height: 16px; border-radius: 50%; background-color: #FFB703;"></div>
</div>
<div>
<span style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-right: 1rem;">Verdict Set By</span>
<span style="background-color: rgba(255, 183, 3, 0.2); color: #FFB703; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.9rem;">{state.get('critic_source') or 'evidence_gate'}</span>
<div style="color: #64748B; font-size: 0.75rem; margin-top: 8px;">
The evidence gate is deterministic and driven by held-out walk-forward metrics.
The LLM review can add flags and downgrade a verdict, but never raise one.
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Raw Shared LangGraph State (JSON)"):
        safe_state = {}
        for k, v in state.items():
            if k in ["signals_df", "macro_df"] and isinstance(v, list):
                safe_state[k] = f"List of {len(v)} records"
            else:
                safe_state[k] = v
                
        import numpy as np
        def default_serializer(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        st.code(json.dumps(safe_state, indent=2, default=default_serializer), language="json")
