import streamlit as st
import pandas as pd
import plotly.express as px

def render_signals_view(df: pd.DataFrame, latest_signals: dict) -> None:
    """
    Renders a table of the latest signal values with an Interpretation
    column, followed by a 30-day normalised signal heatmap.
    Expects the full signals dataframe and the latest_signals dict from state.
    """
    if not latest_signals:
        st.warning("No latest signals available.")
        return

    def get_interp(sig, val, ref=None):
        if sig == "RSI": return "Bearish" if val > 70 else "Bullish" if val < 30 else "Neutral"
        if sig == "MACD Hist": return "Bullish" if val > 0 else "Bearish" if val < 0 else "Neutral"
        if sig == "BB Width": return "Bearish" if val > ref else "Bullish" if val < ref else "Neutral" # ref can be average
        if sig == "OBV": return "Bullish" if val > ref else "Bearish" if val < ref else "Neutral"
        if sig == "Stoch %K": return "Bearish" if val > 80 else "Bullish" if val < 20 else "Neutral"
        if sig == "Williams %R": return "Bearish" if val > -20 else "Bullish" if val < -80 else "Neutral"
        if sig == "ROC-10": return "Bullish" if val > 0 else "Bearish" if val < 0 else "Neutral"
        if sig == "Price vs SMA-20": return "Bullish" if val > ref else "Bearish" if val < ref else "Neutral"
        if sig == "Price vs EMA-50": return "Bullish" if val > ref else "Bearish" if val < ref else "Neutral"
        if sig == "Price vs BB Upper": return "Bearish" if val > ref else "Neutral"
        if sig == "Price vs BB Lower": return "Bullish" if val < ref else "Neutral"
        if sig == "EMA-9 vs EMA-21": return "Bullish" if val > ref else "Bearish" if val < ref else "Neutral"
        return "Neutral"
        
    cprice = latest_signals.get("close", 0)
    sma20 = latest_signals.get("sma_20", 0)
    ema50 = latest_signals.get("ema_50", 0)
    bbupper = latest_signals.get("bb_upper", float('inf'))
    bblower = latest_signals.get("bb_lower", float('-inf'))
    ema9 = latest_signals.get("ema_9", 0)
    ema21 = latest_signals.get("ema_21", 0)
    
    cards = [
        ("RSI (14)", latest_signals.get("rsi", 0), get_interp("RSI", latest_signals.get("rsi", 0))),
        ("MACD Hist", latest_signals.get("macd_hist", 0), get_interp("MACD Hist", latest_signals.get("macd_hist", 0))),
        ("Stochastic %K", latest_signals.get("stoch_k", 0), get_interp("Stoch %K", latest_signals.get("stoch_k", 0))),
        ("Williams %R", latest_signals.get("williams_r", 0), get_interp("Williams %R", latest_signals.get("williams_r", 0))),
        ("ROC (10)", latest_signals.get("roc_10", 0), get_interp("ROC-10", latest_signals.get("roc_10", 0))),
        ("Price vs SMA-20", cprice, get_interp("Price vs SMA-20", cprice, sma20)),
        ("Price vs EMA-50", cprice, get_interp("Price vs EMA-50", cprice, ema50)),
        ("Price vs BB Upper", cprice, get_interp("Price vs BB Upper", cprice, bbupper)),
        ("Price vs BB Lower", cprice, get_interp("Price vs BB Lower", cprice, bblower)),
        ("EMA-9", ema9, get_interp("EMA-9 vs EMA-21", ema9, ema21)),
    ]
    
    st.markdown("<h3 style='margin-bottom: 1rem; color:#FFFFFF;'>Latest Technical Signals</h3>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, (name, val, interp) in enumerate(cards):
        with cols[i % 3]:
            tag_class = f"tag-{interp.lower()}"
            st.markdown(f"""
            <div class="stCard" style="padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #94A3B8; font-weight: 600;">{name}</span>
                    <span class="{tag_class}">{interp}</span>
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #FFFFFF; margin-top: 0.5rem;">{val:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1rem; color:#FFFFFF;'>Signal Heatmap (Last 30 Days)</h3>", unsafe_allow_html=True)
    if not df.empty:
        heatmap_df = df.tail(30).copy()
        if "date" in heatmap_df.columns:
            heatmap_df.set_index("date", inplace=True)
            
        # Signal columns only — exclude identifiers, targets and benchmark
        # bookkeeping so the heatmap never plots a forward-looking label.
        exclude_cols = [
            "date", "ticker",
            "target", "target_return", "target_excess_return", "benchmark_return",
            "benchmark_ticker", "benchmark_sector_specific",
        ]
        cols_to_plot = [c for c in heatmap_df.columns if c not in exclude_cols]
        
        if cols_to_plot:
            heatmap_df = heatmap_df[cols_to_plot]
            # Normalise each signal column to 0-1 using min-max scaling across the last 30 rows
            norm_df = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min() + 1e-9)
            
            fig2 = px.imshow(norm_df.T, color_continuous_scale=["#00B4D8", "#0D1B3E", "#FFB703"], aspect="auto")
            fig2.update_layout(
                height=500, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#0D1B3E", plot_bgcolor="#0D1B3E",
                font=dict(color="#FFFFFF")
            )
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
