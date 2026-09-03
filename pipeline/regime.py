"""
pipeline/regime.py — market state, and the only way it can reach the model.

WHY A BARE REGIME COLUMN IS DEAD WEIGHT, MEASURED RATHER THAN ARGUED
---------------------------------------------------------------------
Every ticker sees the same market on a given date. `cross_sectional_zscore`
standardises WITHIN each date, so any market-wide column is identically ZERO
after it and carries no ranking information whatsoever. That is already
recorded for the six macro columns, and it is exactly how `fii_net_flow` spent
years in `FEATURES` contributing nothing — except that one was constant for a
second reason too, so nobody noticed.

So the regime enters as an INTERACTION with a ticker-varying quantity, or it
does not enter at all.

WHY THE INTERACTION IS WITH BETA
---------------------------------
`beta_market` is the only comparator on this panel that ranks anything —
rebalance IC +0.0464 at t +1.51 on the absolute target, better than the tuned
tree's +0.0389 — and it holds NO company-specific view: it sorts by beta and
rides a rising market. A high-beta name is supposed to behave differently from
a low-beta name when volatility spikes or the market is in drawdown, and that
statement is testable, cross-sectionally varying, and about the one effect this
panel actually exhibits.

`beta_i x regime_vol` is therefore the most defensible thing available, not the
most promising-looking. Widening to `rsi x regime_vol`, `sector_rel_20d x
regime_disp` and the rest was considered and refused: this project has been
fooled three times by a result that lived in one cell of a grid, and a wide
interaction set is that grid with more cells.

EVERYTHING HERE IS BACKTESTABLE TODAY
--------------------------------------
Unlike news, which has ~1 article per ticker-month before 2022, the regime is
computed from `ohlcv` and `macro`, both of which run to 2016-09 with full
coverage. It is the one half of P3 that can be scored across all five purged
folds immediately.

CAUSALITY: every column is a trailing statistic of data available AT the date.
No forward window, no centred rolling, no bfill. `macro` is already forward
filled only (F12), and breadth is computed from each ticker's own trailing
50-session mean.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.db import get_engine

logger = logging.getLogger(__name__)

#: Trailing windows, in sessions.
VOL_SHORT, VOL_LONG = 20, 60
DRAWDOWN_LOOKBACK = 252
BREADTH_MA = 50

#: The market-wide state columns. NONE of these may enter FEATURES directly —
#: see the module docstring. `regime_columns()` is what a caller wires in.
REGIME_STATE: list[str] = [
    "regime_vol",        # NIFTY realised vol, 20d, annualised
    "regime_vol_ratio",  # 20d vol over 60d vol: is volatility RISING?
    "regime_drawdown",   # NIFTY distance below its own trailing 252d high
    "regime_breadth",    # fraction of the universe above its own 50d mean
    "regime_disp",       # cross-sectional SD of that date's daily returns
]

#: What actually reaches the panel. Each is a market-wide state multiplied by a
#: ticker-varying quantity, so it survives cross-sectional standardisation.
REGIME_INTERACTIONS: list[str] = [
    "beta_x_regime_vol",
    "beta_x_regime_drawdown",
]


def compute_market_state(panel: pd.DataFrame, engine=None) -> pd.DataFrame:
    """
    One row per date: the market's state as of that session's close.

    Breadth and dispersion are computed from the PANEL rather than from an
    index, so they describe this universe rather than a different one. That
    matters — NIFTY 50 is roughly a third financials by weight, and this
    universe is 84 names chosen for data quality.
    """
    engine = engine or get_engine()
    if panel.empty:
        return pd.DataFrame(columns=["date"] + REGIME_STATE)

    macro = pd.read_sql(text(
        "SELECT date, nifty_5d_return, nifty_20d_return, india_vix "
        "FROM macro ORDER BY date ASC"), engine)

    wide = (panel.pivot_table(index="date", columns="ticker", values="close",
                              aggfunc="last").sort_index())
    rets = np.log(wide / wide.shift(1))

    market = rets.mean(axis=1)                      # equal-weighted, this universe
    level = market.cumsum()

    state = pd.DataFrame(index=wide.index)
    state["regime_vol"] = market.rolling(VOL_SHORT, min_periods=VOL_SHORT).std() * np.sqrt(252)
    long_vol = market.rolling(VOL_LONG, min_periods=VOL_LONG).std() * np.sqrt(252)
    state["regime_vol_ratio"] = state["regime_vol"] / long_vol.replace(0, np.nan)

    peak = level.rolling(DRAWDOWN_LOOKBACK, min_periods=VOL_SHORT).max()
    state["regime_drawdown"] = level - peak         # <= 0 by construction

    above = (wide > wide.rolling(BREADTH_MA, min_periods=BREADTH_MA).mean())
    state["regime_breadth"] = above.sum(axis=1) / wide.notna().sum(axis=1).replace(0, np.nan)

    state["regime_disp"] = rets.std(axis=1)

    state = state.replace([np.inf, -np.inf], np.nan).reset_index()

    if not macro.empty:
        state = state.merge(macro[["date", "india_vix"]], on="date", how="left")
        state["india_vix"] = state["india_vix"].ffill()   # never bfill (F12)
        state = state.drop(columns=["india_vix"])

    return state[["date"] + REGIME_STATE]


def rolling_beta(panel: pd.DataFrame, window: int = DRAWDOWN_LOOKBACK) -> pd.DataFrame:
    """
    Each ticker's trailing beta to the equal-weighted universe return.

    TRAILING, and estimated only from sessions at or before the row's own date.
    `pipeline.baselines.BetaMarketForecast` fits its betas inside each training
    fold, which is right for a comparator; this one has to be a FEATURE
    available at every date, so it is a rolling window rather than a fold-wide
    fit. A fold-wide beta used as a feature would be F2 in miniature — a
    quantity estimated partly from the test window.
    """
    wide = (panel.pivot_table(index="date", columns="ticker", values="close",
                              aggfunc="last").sort_index())
    rets = np.log(wide / wide.shift(1))
    market = rets.mean(axis=1)

    cov = rets.rolling(window, min_periods=VOL_LONG).cov(market)
    var = market.rolling(window, min_periods=VOL_LONG).var()
    beta = cov.div(var.replace(0, np.nan), axis=0)

    out = (beta.stack(future_stack=True).rename("beta").reset_index()
           .rename(columns={"level_1": "ticker"}))
    return out.replace([np.inf, -np.inf], np.nan)


def build_regime_features(panel: pd.DataFrame, engine=None) -> pd.DataFrame:
    """
    Returns (date, ticker, REGIME_INTERACTIONS...).

    The market state alone is NOT returned for the panel to use. Returning it
    would put a column in reach that is identically zero after within-date
    standardisation, and this project already has one feature list that spent
    years carrying exactly that.
    """
    engine = engine or get_engine()
    if panel.empty:
        return pd.DataFrame(columns=["date", "ticker"] + REGIME_INTERACTIONS)

    state = compute_market_state(panel, engine)
    betas = rolling_beta(panel)

    merged = betas.merge(state, on="date", how="left")
    merged["beta_x_regime_vol"] = merged["beta"] * merged["regime_vol"]
    merged["beta_x_regime_drawdown"] = merged["beta"] * merged["regime_drawdown"]

    return merged[["date", "ticker"] + REGIME_INTERACTIONS]


def describe_regime(state: pd.DataFrame, as_of: str | None = None) -> dict:
    """
    The market state on one date, for the written narrative.

    Percentile ranks are computed against the HISTORY UP TO that date only, so
    "volatility is in its 90th percentile" is a statement a reader could have
    made at the time rather than one that needs the rest of the sample.
    """
    if state.empty:
        return {}
    as_of = as_of or str(state["date"].max())
    upto = state[state["date"] <= as_of]
    if upto.empty:
        return {}

    row = upto.iloc[-1]
    out = {"date": as_of}
    for col in REGIME_STATE:
        value = row.get(col)
        if value is None or not np.isfinite(value):
            out[col] = None
            continue
        history = upto[col].dropna()
        out[col] = {
            "value": float(value),
            "pctile": float((history <= value).mean()) if len(history) > 1 else None,
        }
    return out
