"""
app/main.py

Multi-page Streamlit application entry point.
Defines the navigation structure and injects global CSS.
Run with: streamlit run app/main.py
"""

import streamlit as st
import os

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZeRO Stock Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "styles", "main.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Define pages ──────────────────────────────────────────────────────────────
leaderboard_page = st.Page(
    "pages/leaderboard.py",
    title="Leaderboard",
    icon="🏆",
    default=True,
)
stock_detail_page = st.Page(
    "pages/stock_detail.py",
    title="Stock Detail",
    icon="📊",
)
portfolio_page = st.Page(
    "pages/portfolio.py",
    title="Portfolio",
    icon="💼",
)
about_page = st.Page(
    "pages/about.py",
    title="About",
    icon="ℹ️",
)

# ── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation(
    pages=[
        leaderboard_page,
        stock_detail_page,
        portfolio_page,
        about_page,
    ],
    position="sidebar",
)

# ── Run selected page ─────────────────────────────────────────────────────────
pg.run()
