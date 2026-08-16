"""
app/pages/about.py

About page — what the system predicts, how it is evaluated, and what its
measured performance actually is.

Performance figures are read live from the leaderboard rather than hard-coded.
The previous version displayed "~4.3% MAPE / ~85% directional accuracy" as
validated performance; those numbers came from fitting a Ridge meta-learner on
the validation set and scoring it on that same set. Numbers typed into a page
cannot be audited and go stale silently, so this page now shows whatever the
purged walk-forward harness last measured — including when that is unflattering.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@st.cache_data(ttl=900)
def fetch_measured_performance() -> pd.DataFrame:
    """Pulls the held-out evaluation metrics recorded for each stock."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/leaderboard",
            params={"limit": 200, "sort_by": "composite_score"},
            timeout=60,
        )
        response.raise_for_status()
        return pd.DataFrame(response.json().get("entries", []))
    except Exception:
        return pd.DataFrame()


def main():
    st.markdown("## About ZeRO Agentic Stock Forecast")
    st.markdown(
        "ZeRO ranks NSE stocks by their predicted **30-session return relative "
        "to their sector benchmark**. It runs a LangGraph pipeline over a "
        "point-in-time universe, a gradient-boosted model evaluated by purged "
        "walk-forward validation, and conformal prediction intervals. Every "
        "performance figure on this site is out-of-sample and shown next to the "
        "baseline it has to beat."
    )

    st.divider()

    # ── Measured performance ─────────────────────────────────────────────────
    st.markdown("### Measured Performance")
    st.caption(
        "Purged walk-forward validation with a 30-session embargo. "
        "Hyperparameters are tuned inside each training fold, so no "
        "configuration is chosen with sight of the rows it is later scored on. "
        "Figures are before transaction costs."
    )

    df = fetch_measured_performance()

    if df.empty:
        st.info("No evaluation metrics available yet — run the pipeline first.")
    else:
        ic = pd.to_numeric(df.get("eval_rank_ic"), errors="coerce")
        hit = pd.to_numeric(df.get("eval_hit_rate"), errors="coerce")
        base = pd.to_numeric(df.get("eval_baseline_hit_rate"), errors="coerce")
        beats = df.get("eval_beats_random_walk")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Rank IC", f"{ic.mean():+.3f}",
                  help="Out-of-sample Spearman correlation between predicted "
                       "and realised excess return. 0 = no skill. Values around "
                       "0.02–0.05 are typical for real technical signals.")
        c2.metric("Directional Accuracy", f"{hit.mean():.1f}%",
                  delta=f"{hit.mean() - base.mean():+.1f}pp vs baseline",
                  help="The baseline is always predicting the more common "
                       "direction over the same window.")
        c3.metric("Beats Random Walk",
                  f"{int(pd.Series(beats).fillna(False).astype(bool).sum())}/{len(df)}"
                  if beats is not None else "—",
                  help="Stocks where mean absolute error beats forecasting zero "
                       "excess return.")
        c4.metric("Stocks Covered", f"{len(df)}")

        beats_baseline = int((hit > base).sum()) if hit.notna().any() else 0

        if ic.mean() <= 0 or beats_baseline < len(df) / 2:
            st.error(
                "**The model does not currently beat its baselines.** Mean rank "
                "IC is at or below zero and directional accuracy trails the "
                "majority-class rate on most stocks. Treat every forecast on "
                "this site as a research output with no demonstrated edge. "
                "The interval and probability are calibrated; the point "
                "forecast is not informative."
            )
        else:
            st.warning(
                "**Read these honestly.** A rank IC near 0.05 is a weak signal — "
                "real, but not a licence to trade. Where the model does not beat "
                "a random walk on magnitude, treat the ranking as the output and "
                "the rupee target as illustrative only. No transaction costs, "
                "slippage or liquidity limits are modelled anywhere in this system."
            )

    st.divider()

    # ── What it predicts ─────────────────────────────────────────────────────
    st.markdown("### What the Model Actually Predicts")
    st.markdown("""
    The target is the **30-session log return in excess of the stock's benchmark
    index** — not an absolute price.

    That choice matters. An earlier version predicted the closing price 30
    sessions ahead, which made the error look small for the wrong reason: prices
    are persistent, so simply repeating today's price scores well on MAPE. It
    also capped every forecast at the highest price seen in training, because
    gradient-boosted trees cannot extrapolate beyond their training range.

    The rupee target shown on each stock page is **derived** from the excess-return
    forecast and assumes the benchmark index is flat over the horizon. The model
    forecasts relative performance; it has nothing to say about where the market goes.
    """)

    st.divider()

    # ── How it works ─────────────────────────────────────────────────────────
    st.markdown("### How It Works")

    with st.expander("① Universe — point-in-time construction", expanded=False):
        st.markdown("""
        The tradable universe comes from a rule that references no model output:

        - NIFTY 100 membership as of the date in question
        - 20-day median traded value above ₹25 crore
        - At least ~3 years of listed price history

        Membership is stored as dated intervals, so the system can ask
        *"who was in the index then?"* rather than *"who is in it now?"*

        **Known limitation:** membership is only recorded from the first sync
        onward. Evaluation windows starting before that date use present-day
        membership and are therefore survivorship-biased. The API reports the
        date from which membership is genuinely known.
        """)

    with st.expander("② Trading Data Agent — signals", expanded=False):
        st.markdown("""
        Fetches 10 years of daily OHLCV and computes 24 signals:

        - **Momentum:** RSI-14, Stochastic %K, Williams %R, ROC-10, lag-1/5 returns
        - **Trend:** SMA-20, EMA-9/21/50, SMA-50 deviation, 52-week proximity
        - **Volatility:** Bollinger width/upper/lower, ATR-14
        - **Volume:** OBV, volume ROC
        - **Regime:** Hurst exponent over log prices
        - **Relative:** sector-relative momentum over 5/10/20 sessions
        - **Fundamental:** quarterly EPS surprise, attached to the first session
          *after* the announcement (Indian results are often declared post-close)

        Prices are stored raw and adjusted separately, so a split or dividend
        cannot splice two adjustment bases into one series.
        """)

    with st.expander("③ Forecasting Agent — model and uncertainty", expanded=False):
        st.markdown("""
        **XGBoost**, heavily regularised, trained per stock on the excess-return
        target. Hyperparameters are searched with Optuna inside each training
        fold and the study is seeded, so a tuning run is reproducible.

        This search runs **weekly, not daily** — every ticker was originally
        re-evaluated with a full nested Optuna search every single day, which
        starved a production server's CPU for over an hour and eventually
        crashed it. "Does this model form have skill" doesn't change day to
        day, so it's now measured once a week; each day just fits fresh with
        the hyperparameters that search already found. The evidence badge on
        every forecast states exactly when it was last measured, which can be
        up to a week before the price it sits next to.

        **Conformal prediction** calibrates an 80% interval on out-of-sample
        residuals and yields a probability that the stock beats its benchmark.
        Coverage is measured, not assumed — if the 80% interval does not cover
        ~80% of held-out outcomes, that is reported. This calibration is also
        refreshed weekly, alongside the hyperparameter search.

        The LLM writes a plain-English read of the signals. It does not produce,
        adjust or review any number, and it is not shown the forecast when
        writing the narrative.

        *An LSTM and a Ridge meta-learner were previously advertised here. Both
        are archived: the LSTM never wrote a checkpoint and so never produced a
        forecast, and the meta-learner was the source of the inflated accuracy
        figures. They return only if an experiment shows they beat a linear
        baseline out-of-sample.*
        """)

    with st.expander("④ Critic Agent — evidence gate", expanded=False):
        st.markdown("""
        Two separate jobs, deliberately not mixed:

        **The evidence gate** is deterministic and tested. It asks only:
        has this model shown skill on folds it never trained on? It checks rank
        IC, its t-statistic (corrected for overlapping labels), and hit rate
        against the majority-class baseline, then grades the forecast
        **STRONG / WEAK / INSUFFICIENT**.

        **The LLM review** looks for contradictions in the signal snapshot and
        may add flags. It can only *downgrade*. It sees numbers it has no way to
        verify, so it is allowed to raise doubt but never to certify.

        Previously the LLM's verdict was overwritten by thresholds keyed on the
        inflated accuracy figures, which meant nearly every stock came out
        APPROVED and the verdict carried no information.
        """)

    st.divider()

    # ── Composite score ──────────────────────────────────────────────────────
    st.markdown("### Leaderboard Composite Score")
    st.markdown(
        "A **ranking heuristic**, not an expected return. Signal and conviction "
        "are multiplied by an evidence grade, so a model that failed its "
        "held-out checks scores zero no matter how large a move it predicts."
    )

    st.dataframe(
        pd.DataFrame({
            "Component": ["Predicted excess return", "Conviction",
                          "Evidence grade", "LLM flags"],
            "Effect": ["0–60 pts", "0–40 pts", "×1.0 / ×0.5 / ×0", "−5 pts each"],
            "Description": [
                "Predicted 30-session return vs benchmark, saturating at +10%",
                "How far P(outperform) sits from a coin flip",
                "STRONG / WEAK / INSUFFICIENT, from purged walk-forward folds",
                "Contradictions raised by the LLM signal review",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Evaluation methodology ───────────────────────────────────────────────
    st.markdown("### Evaluation Methodology")
    st.markdown("""
    - **Purged walk-forward validation.** Each label spans 30 sessions, so
      training rows whose label reaches into the test window are removed, plus a
      further 30-session embargo. Without this, the last 30 training rows carry
      labels drawn from inside the test window.
    - **Nested tuning.** Hyperparameters are searched inside each training fold
      only. The tuner is structurally incapable of seeing test rows.
    - **Baselines always reported.** Directional accuracy next to the
      majority-class rate; error next to the naive zero-excess forecast.
    - **Overlap-corrected t-statistics.** Consecutive 30-session labels are ~97%
      overlapping, so the effective sample is roughly *n / 30*. Treating all rows
      as independent inflates every t-statistic by about 5.5×.
    - **Measured interval coverage.** Conformal intervals claim 80%; realised
      coverage is checked against that claim.
    """)

    st.divider()

    # ── Known limitations ────────────────────────────────────────────────────
    st.markdown("### Known Limitations")
    st.markdown("""
    Stated plainly, because a system that hides these is not worth trusting:

    - **No transaction costs.** Indian delivery round trips run roughly 30–60 bps
      before market impact. On a monthly rebalance that is comparable to the
      entire measured edge.
    - **No portfolio backtest.** Forecast accuracy and investment profitability
      are different questions; only the first is currently measured.
    - **Survivorship bias.** Point-in-time index membership is only recorded from
      the first universe sync onward.
    - **News sentiment is not a model feature, and is not scored at all.**
      As a feature it only ever existed for the current date, so it was zero
      across the training set and non-zero only for the row being predicted.
      The scorer itself is also gone: FinBERT required torch, which was removed
      in Phase 0, so it failed to load on every run. Headlines are collected and
      shown unscored. Both return when a dated news archive exists.
    - **T+1 settlement is not modelled.** A signal computed at the 15:30 close is
      actionable at the next open at the earliest.
    - **No demonstrated edge.** As measured, the model does not beat a
      majority-class baseline on direction or a zero-excess forecast on
      magnitude. The calibrated intervals hold up; the point forecast does not.
    """)

    st.divider()

    # ── Tech stack ───────────────────────────────────────────────────────────
    st.markdown("### Tech Stack")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Modelling**
        - XGBoost + Optuna (seeded, nested tuning)
        - Split-conformal prediction (intervals, calibrated probabilities)
        - scipy / scikit-learn (metrics)

        **Agents**
        - LangGraph (orchestration)
        - Groq API — `openai/gpt-oss-20b` (narrative + signal review)

        **Data**
        - yfinance (OHLCV, benchmarks, macro)
        - NSE archives (index constituents)
        - `ta` (technical indicators)
        """)

    with col2:
        st.markdown("""
        **Infrastructure**
        - FastAPI + Uvicorn (backend — reads only, no scheduled compute)
        - Streamlit + Plotly (dashboard)
        - Supabase PostgreSQL
        - Render / Streamlit Community Cloud
        - GitHub Actions (daily forecast + weekly evaluation,
          run directly against Supabase)
        - pytest (leakage and regression suite)

        **Repository**
        - [github.com/glitching-gops/Agentic-Stock-Forecast](https://github.com/glitching-gops/Agentic-Stock-Forecast)
        """)

    st.divider()

    st.caption(
        "⚠️ ZeRO is a research and portfolio project. Forecasts are generated by "
        "statistical models and are not financial advice. Measured out-of-sample "
        "performance is weak and is reported before transaction costs. Past "
        "performance does not guarantee future results. Do your own research."
    )

    st.divider()
    st.markdown(
        "Built by **Venu Gopal Battula** · "
        "[github.com/glitching-gops](https://github.com/glitching-gops)"
    )


main()
