"""
app/pages/leaderboard.py

Leaderboard landing page.
Displays all 100 stocks ranked by composite score with filtering
and sector breakdown chart.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from datetime import datetime

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


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_leaderboard(
    sector: str | None     = None,
    verdict: str | None    = None,
    confidence: str | None = None,
    sort_by: str           = "composite_score",
    limit: int             = 100,
) -> dict | None:
    """Fetches leaderboard data from the FastAPI backend."""
    try:
        params = {"sort_by": sort_by, "limit": limit}
        if sector and sector != "All":
            params["sector"] = sector
        if verdict and verdict != "All":
            params["verdict"] = verdict
        if confidence and confidence != "All":
            # The API parameter is `evidence` — the field holds an evidence
            # grade (STRONG/WEAK/INSUFFICIENT), not the old High/Medium/Low
            # confidence label derived from leaked metrics.
            params["evidence"] = confidence

        r = requests.get(
            f"{API_BASE_URL}/api/leaderboard",
            params=params,
            timeout=30
        )
        r.raise_for_status()
        return r.json()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.warning(
            "⚠️ Unable to reach the forecast API. "
            "The backend service may be starting up (Render cold start) — "
            "please wait 30-60 seconds and refresh the page."
        )
        return {"entries": [], "last_updated": "N/A", "total": 0}
    except Exception as e:
        st.error(f"⚠️ Error fetching leaderboard: {e}")
        return {"entries": [], "last_updated": "N/A", "total": 0}




# ── Page header ───────────────────────────────────────────────────────────────

def render_header(last_updated: str, total: int) -> None:
    """Renders the page header with title and last updated timestamp."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("## 🏆 Stock Leaderboard")
        st.markdown(
            "Nifty 100 stocks ranked by composite score — "
            "combining forecast upside, directional accuracy, "
            "model confidence, and Critic Agent verdict."
        )

    with col2:
        try:
            ts = datetime.fromisoformat(last_updated.replace("Z", ""))
            formatted = ts.strftime("%d %b %Y, %I:%M %p IST")
        except Exception:
            formatted = last_updated

        st.markdown(
            f"""
            <div style='text-align:right; padding-top:8px;'>
                <div style='color:#94A3B8; font-size:0.75rem;'>LAST UPDATED</div>
                <div style='color:#00B4D8; font-weight:600;'>{formatted}</div>
                <div style='color:#94A3B8; font-size:0.75rem;'>{total} stocks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()


# ── Summary metric cards ──────────────────────────────────────────────────────

def render_summary_cards(df: pd.DataFrame) -> None:
    """
    Summary cards above the leaderboard.

    Directional accuracy is now shown NEXT TO the majority-class baseline it
    has to beat. Reporting it alone is what let an 85% figure look impressive
    while the baseline on the same window was ~59%.
    """
    strong = (df["forecast_confidence"] == "STRONG").sum()
    weak = (df["forecast_confidence"] == "WEAK").sum()
    insufficient = (df["forecast_confidence"] == "INSUFFICIENT").sum()

    avg_score = df["composite_score"].mean()
    avg_ic = df["eval_rank_ic"].mean() if "eval_rank_ic" in df.columns else float("nan")
    avg_hit = df["eval_hit_rate"].mean() if "eval_hit_rate" in df.columns else float("nan")
    avg_base = (df["eval_baseline_hit_rate"].mean()
                if "eval_baseline_hit_rate" in df.columns else float("nan"))

    top_stock = df.iloc[0]["company"] if len(df) > 0 else "N/A"
    top_score = df.iloc[0]["composite_score"] if len(df) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Avg Composite Score", f"{avg_score:.1f}/100",
                  help="Ranking heuristic, not an expected return.")
    with c2:
        st.metric("Evidence Grade",
                  f"{strong} strong · {weak} weak · {insufficient} none",
                  help="From purged walk-forward evaluation on held-out folds.")
    with c3:
        edge = avg_hit - avg_base
        st.metric("Directional Accuracy", f"{avg_hit:.1f}%",
                  delta=f"{edge:+.1f}pp vs {avg_base:.1f}% baseline",
                  help="The baseline is always predicting the more common "
                       "direction. Accuracy only means something relative to it.")
    with c4:
        st.metric("Avg Rank IC", f"{avg_ic:+.3f}",
                  delta=f"Top: {top_stock[:18]} ({top_score:.0f})",
                  delta_color="off",
                  help="Out-of-sample Spearman correlation between predicted "
                       "and realised excess return. 0 = no skill.")

    st.caption(
        "All performance figures are out-of-sample, measured by purged "
        "walk-forward validation with a 30-session embargo, before transaction "
        "costs. Predicted moves are relative to each stock's benchmark index."
    )

    st.divider()


# ── Filter controls ───────────────────────────────────────────────────────────

def render_filters() -> tuple:
    """
    Renders the filter and sort controls above the leaderboard table.
    Returns (sector, verdict, confidence, sort_by, show_all).
    """
    c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1.5, 1.5, 1])

    with c1:
        # NSE industry classifications, matching data/tickers.SECTOR_INDICES.
        sectors = [
            "All", "Financial Services", "Information Technology",
            "Oil Gas & Consumable Fuels", "Fast Moving Consumer Goods",
            "Automobile and Auto Components", "Healthcare", "Metals & Mining",
            "Capital Goods", "Power", "Telecommunication", "Realty",
            "Consumer Durables", "Consumer Services", "Construction Materials",
            "Chemicals", "Services", "Construction",
        ]
        sector = st.selectbox("Sector", sectors, index=0)

    with c2:
        verdict = st.selectbox(
            "Critic Verdict",
            ["All", "APPROVED", "FLAGGED", "REJECTED", "APPROVED_OR_FLAGGED"],
            index=0
        )

    with c3:
        confidence = st.selectbox(
            "Evidence Grade",
            ["All", "STRONG", "WEAK", "INSUFFICIENT"],
            index=0,
            help="Held-out evidence that this stock's model has skill.",
        )

    with c4:
        sort_by = st.selectbox(
            "Sort By",
            [
                "composite_score", "pred_excess_return",
                "prob_outperform", "eval_rank_ic",
            ],
            format_func=lambda x: {
                "composite_score":    "Composite Score",
                "pred_excess_return": "Predicted Excess Return",
                "prob_outperform":    "P(outperform)",
                "eval_rank_ic":       "Out-of-sample Rank IC",
            }.get(x, x),
            index=0
        )

    with c5:
        st.markdown("<br>", unsafe_allow_html=True)
        show_all = st.toggle("Show All", value=False)

    return sector, verdict, confidence, sort_by, show_all


# ── Leaderboard table ─────────────────────────────────────────────────────────

def render_leaderboard_table(df: pd.DataFrame, show_all: bool) -> str | None:
    """
    Renders the main leaderboard table using st.dataframe with
    sparkline column config. Returns the ticker of the selected row
    if the user clicks the stock detail button, otherwise None.
    """
    display_df = df if show_all else df.head(20)

    display_df = display_df.copy()
    display_df["rank"]      = range(1, len(display_df) + 1)

    # Verdict emoji column
    verdict_map = {
        "APPROVED": "✅ APPROVED",
        "FLAGGED":  "⚠️ FLAGGED",
        "REJECTED": "❌ REJECTED",
    }
    display_df["verdict_display"] = display_df["critic_verdict"].map(
        lambda v: verdict_map.get(v, v)
    )

    # Predicted move, relative to the stock's benchmark index.
    display_df["excess_display"] = display_df.apply(
        lambda row: (
            f"🟢 {row['pred_excess_return'] * 100:+.1f}%"
            if pd.notna(row.get("pred_excess_return")) and row["pred_excess_return"] >= 0
            else f"🔴 {row['pred_excess_return'] * 100:+.1f}%"
            if pd.notna(row.get("pred_excess_return")) else "—"
        ),
        axis=1
    )

    # The 80% conformal interval, in rupees.
    display_df["interval_display"] = display_df.apply(
        lambda row: (
            f"₹{row['interval_low']:,.0f} – ₹{row['interval_high']:,.0f}"
            if pd.notna(row.get("interval_low")) and pd.notna(row.get("interval_high"))
            else "—"
        ),
        axis=1
    )

    table_cols = [
        "rank", "company", "sector",
        "current_price", "forecast_price", "interval_display",
        "excess_display", "prob_outperform", "composite_score",
        "eval_rank_ic", "verdict_display", "forecast_confidence",
    ]
    table_cols = [c for c in table_cols if c in display_df.columns]

    st.dataframe(
        display_df[table_cols],
        use_container_width=True,
        hide_index=True,
        height=600 if show_all else 400,
        column_config={
            "rank": st.column_config.NumberColumn(
                "Rank", width="small", format="%d"
            ),
            "company": st.column_config.TextColumn(
                "Company", width="medium"
            ),
            "sector": st.column_config.TextColumn(
                "Sector", width="medium"
            ),
            "current_price": st.column_config.NumberColumn(
                "Price (₹)", format="₹%.2f", width="small"
            ),
            "forecast_price": st.column_config.NumberColumn(
                "Implied Target (₹)", format="₹%.2f", width="small",
                help="Assumes the benchmark index is flat. The model forecasts "
                     "relative performance, not the level of the market."
            ),
            "interval_display": st.column_config.TextColumn(
                "80% Interval", width="medium",
                help="Conformal prediction interval calibrated on out-of-sample "
                     "residuals. Covers the outcome ~80% of the time."
            ),
            "excess_display": st.column_config.TextColumn(
                "Excess vs Benchmark", width="small",
                help="The model's actual output: predicted 30-session return "
                     "relative to the stock's sector index."
            ),
            "prob_outperform": st.column_config.NumberColumn(
                "P(outperform)", format="%.2f", width="small",
                help="Calibrated probability of beating the benchmark. "
                     "0.50 = a coin flip."
            ),
            "composite_score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100,
                format="%.1f", width="medium",
                help="Ranking heuristic: signal and conviction, gated by "
                     "held-out evidence. Not an expected return."
            ),
            "eval_rank_ic": st.column_config.NumberColumn(
                "Rank IC", format="%.3f", width="small",
                help="Out-of-sample rank correlation. 0 = no skill."
            ),
            "verdict_display": st.column_config.TextColumn(
                "Verdict", width="medium"
            ),
            "forecast_confidence": st.column_config.TextColumn(
                "Evidence", width="small"
            ),
        }
    )

    if not show_all and len(df) > 20:
        st.caption(
            f"Showing top 20 of {len(df)} stocks. "
            f"Toggle 'Show All' to see the full leaderboard."
        )


# ── Sector breakdown chart ────────────────────────────────────────────────────

def render_sector_breakdown(df: pd.DataFrame) -> None:
    """
    Renders a horizontal bar chart showing average composite score
    by sector, and a pie chart showing verdict distribution.
    """
    st.markdown("### Sector Analysis")

    c1, c2 = st.columns(2)

    with c1:
        sector_scores = (
            df.groupby("sector")["composite_score"]
            .mean()
            .sort_values(ascending=True)
            .reset_index()
        )

        fig = px.bar(
            sector_scores,
            x="composite_score",
            y="sector",
            orientation="h",
            title="Average Composite Score by Sector",
            color="composite_score",
            color_continuous_scale=["#1E3A6E", "#00B4D8", "#06D6A0"],
            range_color=[0, 100],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F0FE",
            title_font_color="#E8F0FE",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0),
            height=380,
        )
        fig.update_xaxes(
            showgrid=True, gridcolor="#1E3A6E", range=[0, 100]
        )
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        verdict_counts = df["critic_verdict"].value_counts().reset_index()
        verdict_counts.columns = ["verdict", "count"]

        colour_map = {
            "APPROVED": "#06D6A0",
            "FLAGGED":  "#FFB703",
            "REJECTED": "#EF476F",
        }

        fig2 = px.pie(
            verdict_counts,
            names="verdict",
            values="count",
            title="Critic Verdict Distribution",
            color="verdict",
            color_discrete_map=colour_map,
            hole=0.45,
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F0FE",
            title_font_color="#E8F0FE",
            margin=dict(l=0, r=0, t=40, b=0),
            height=380,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
            ),
        )
        fig2.update_traces(textfont_color="#0D1B3E")
        st.plotly_chart(fig2, use_container_width=True)


# ── Navigate to stock detail ──────────────────────────────────────────────────

def render_stock_navigation(df: pd.DataFrame) -> None:
    """
    Renders a quick-navigate selectbox that lets the user jump
    directly to a stock's detail page from the leaderboard.
    """
    st.divider()
    st.markdown("### 🔍 Quick Navigate to Stock Detail")

    company_to_ticker = {
        row["company"]: row["ticker"]
        for _, row in df.iterrows()
    }

    selected = st.selectbox(
        "Select a stock to view full analysis",
        options=["— select a stock —"] + sorted(company_to_ticker.keys()),
        index=0,
        key="leaderboard_nav"
    )

    if selected != "— select a stock —":
        ticker = company_to_ticker[selected]
        st.session_state["selected_ticker"] = ticker
        st.switch_page("pages/stock_detail.py")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sector, verdict, confidence, sort_by, show_all = render_filters()

    data = fetch_leaderboard(
        sector=sector,
        verdict=verdict,
        confidence=confidence,
        sort_by=sort_by,
        limit=100,
    )

    if data is None:
        st.info(
            "No leaderboard data available. "
            "Run the pipeline first via the admin endpoint."
        )
        return

    df = pd.DataFrame(data["entries"])

    if df.empty:
        st.warning("No stocks match the current filters.")
        return

    render_header(
        last_updated=data.get("last_updated", ""),
        total=data.get("total", len(df))
    )
    render_summary_cards(df)
    render_leaderboard_table(df, show_all)
    render_sector_breakdown(df)
    render_stock_navigation(df)


main()
