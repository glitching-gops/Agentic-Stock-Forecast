"""
app/pages/portfolio.py

Portfolio Optimizer page.
Currently a structured placeholder showing the top 10 stocks
by composite score with a simple equal-weight allocation.
Full Modern Portfolio Theory optimization planned for Stage 5.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os

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


@st.cache_data(ttl=3600)
def fetch_top_stocks(limit: int = 20) -> list[dict]:
    """Fetches the top stocks by composite score from the leaderboard."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/leaderboard",
            params={
                "sort_by": "composite_score",
                "limit":   limit,
                "verdict": "APPROVED_OR_FLAGGED",
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("entries", [])
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.warning(
            "⚠️ Unable to reach the forecast API. "
            "The backend service may be starting up (Render cold start) — "
            "please wait 30-60 seconds and refresh the page."
        )
        return []
    except Exception as e:
        st.error(f"⚠️ Error fetching portfolio data: {e}")
        return []


def main():
    st.markdown("## 💼 Portfolio")
    st.markdown(
        "Equal-weight allocation across the top-ranked stocks. This is a view "
        "of the ranking, **not an optimised portfolio and not a backtest**."
    )

    st.info(
        "Position sizing, a cost model, and risk metrics (Sharpe, Sortino, "
        "max drawdown, Calmar) measured against NIFTY 50 TR are Phase 4 of the "
        "improvement roadmap. Until those exist, nothing on this page has been "
        "shown to make money."
    )

    st.divider()

    n_stocks = st.slider(
        "Number of stocks in portfolio",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    stocks = fetch_top_stocks(limit=n_stocks)

    if not stocks:
        st.warning("No portfolio data available.")
        return

    df = pd.DataFrame(stocks)

    # Simple equal-weight allocation
    df["allocation_pct"] = round(100.0 / len(df), 2)

    # Aggregate the model's actual output: predicted EXCESS return relative to
    # each stock's benchmark. This is not an expected portfolio return — it
    # excludes market direction, transaction costs and slippage entirely.
    excess = pd.to_numeric(df.get("pred_excess_return"), errors="coerce")
    weighted_excess = float((excess.fillna(0.0) * df["allocation_pct"] / 100).sum())

    avg_ic = pd.to_numeric(df.get("eval_rank_ic"), errors="coerce").mean()
    avg_hit = pd.to_numeric(df.get("eval_hit_rate"), errors="coerce").mean()
    avg_base = pd.to_numeric(df.get("eval_baseline_hit_rate"), errors="coerce").mean()
    n_strong = int((df.get("forecast_confidence") == "STRONG").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks", n_stocks)
    c2.metric("Weighted Excess Signal", f"{weighted_excess * 100:+.2f}%",
              help="Average predicted 30-session return relative to each stock's "
                   "benchmark. NOT an expected portfolio return: it excludes "
                   "market direction, transaction costs and slippage.")
    c3.metric("Avg Rank IC", f"{avg_ic:+.3f}",
              help="Out-of-sample rank correlation. 0 = no skill.")
    c4.metric("Strong Evidence", f"{n_strong}/{len(df)}",
              delta=f"Hit {avg_hit:.1f}% vs {avg_base:.1f}% baseline"
                    if pd.notna(avg_hit) and pd.notna(avg_base) else None,
              delta_color="off")

    st.warning(
        "**This is a ranking, not a backtested strategy.** No transaction costs "
        "(Indian delivery round trips run roughly 30–60 bps), slippage, "
        "liquidity limits, or benchmark comparison are modelled. A cost-aware "
        "portfolio simulator with Sharpe, Sortino, max drawdown and NIFTY 50 TR "
        "comparison is Phase 4 of the improvement roadmap."
    )

    st.divider()

    st.markdown("### Suggested Allocation")

    display_cols = {
        "company": "Company",
        "sector": "Sector",
        "current_price": "Price (₹)",
        "forecast_price": "Implied Target (₹)",
        "pred_excess_return": "Excess vs Benchmark",
        "prob_outperform": "P(outperform)",
        "forecast_confidence": "Evidence",
        "composite_score": "Score",
        "allocation_pct": "Allocation %",
    }
    available = {k: v for k, v in display_cols.items() if k in df.columns}

    table = df[list(available)].rename(columns=available)
    if "Excess vs Benchmark" in table.columns:
        table["Excess vs Benchmark"] = pd.to_numeric(
            table["Excess vs Benchmark"], errors="coerce") * 100

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price (₹)":            st.column_config.NumberColumn(format="₹%.2f"),
            "Implied Target (₹)":   st.column_config.NumberColumn(
                format="₹%.2f", help="Assumes the benchmark index is flat."),
            "Excess vs Benchmark":  st.column_config.NumberColumn(format="%+.1f%%"),
            "P(outperform)":        st.column_config.NumberColumn(format="%.2f"),
            "Score":                st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"),
            "Allocation %":         st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    # Allocation pie chart
    st.markdown("### Sector Allocation")
    sector_alloc = (
        df.groupby("sector")["allocation_pct"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        sector_alloc,
        names="sector",
        values="allocation_pct",
        title="Portfolio Allocation by Sector",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F0FE",
        title_font_color="#E8F0FE",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


main()
