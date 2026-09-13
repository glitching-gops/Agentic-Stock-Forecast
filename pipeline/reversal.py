"""
pipeline/reversal.py — multi-lookback reversal, raw and market-residualised.

Stage 1, Pilot 1. The method, arms and decision rules are fixed in
`docs/stage1-preregistration.md`, which was written before any of this ran on
real data.

THE FEATURE, AND THE SESSION IT SKIPS
-------------------------------------
For a lookback k and a skip of one session:

    r_k(i, t)    = log close(i, t-1) - log close(i, t-1-k)
    m_k(t)       = the same window's sum of the equal-weighted universe return
    rev_raw_k    = -r_k
    rev_resid_k  = -(r_k - beta(i, t-1) * m_k)

Session t itself is skipped. At daily frequency the most recent return carries
bid-ask bounce (Conrad, Gultekin & Kaul 1997; Lo & MacKinlay 1990), a
microstructure reversal that is not investable. The beta is lagged by one
session too, so NOTHING in the feature at t depends on session t. A test
corrupts every close from t onward and requires the feature at t unchanged.

A missing session is NOT stepped over. The windows are taken on the shared
date grid, so a hole in one ticker's series voids every window that spans it
rather than silently making it one session longer. That is the vroc_10
landmine, where a row-stepped 30-session horizon measured 31 across a hole.

WHY THIS REUSES regime.rolling_beta AND NOT neutralise
-------------------------------------------------------
The residual is a TIME-SERIES one: the stock's return, net of what its own
trailing beta times the market's return over the same window explains. That
is exactly `regime.rolling_beta` (trailing 252 sessions, min 60, point-in-time,
and already pinned by a test that corrupts the future). `pipeline.neutralise`
regresses ACROSS names within a date, which is right for neutralising a target
and is a different operator. The market series is `regime.market_log_returns`,
the one definition that rolling_beta and the regime state also use.

WHAT THE BASELINE ALREADY HOLDS
-------------------------------
`FACTORS` carries lag1_ret, lag5_ret, roc_10 and sector_rel_{5,10,20}d, and the
pooled panel is z-scored within each date, which removes the market's common
LEVEL from any raw return. So the market-only residual can remove only
(beta_i - mean beta) * m_k, far less than Da, Liu & Schaumburg's (2014)
three-factor residual. The raw arm exists to measure what that is worth here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import regime

#: Lookbacks, in sessions.
LOOKBACKS: tuple[int, ...] = (1, 5, 10, 20)

#: Sessions skipped before each window. One: session t is never read.
SKIP_SESSIONS = 1


def raw_columns(lookbacks: tuple[int, ...] = LOOKBACKS) -> list[str]:
    return [f"rev_raw_{k}" for k in lookbacks]


def resid_columns(lookbacks: tuple[int, ...] = LOOKBACKS) -> list[str]:
    return [f"rev_resid_{k}" for k in lookbacks]


RAW_COLS: list[str] = raw_columns()
RESID_COLS: list[str] = resid_columns()
REVERSAL_COLS: list[str] = RAW_COLS + RESID_COLS


def reversal_features(panel: pd.DataFrame,
                      lookbacks: tuple[int, ...] = LOOKBACKS,
                      skip: int = SKIP_SESSIONS) -> pd.DataFrame:
    """
    Returns (date, ticker, rev_raw_k..., rev_resid_k...) on the shared grid.

    Reads only ``date``, ``ticker`` and ``close``. The result is NOT
    standardised; a pooled caller z-scores it within each date exactly as it
    does every other feature.
    """
    cols = raw_columns(lookbacks) + resid_columns(lookbacks)
    if panel.empty:
        return pd.DataFrame(columns=["date", "ticker"] + cols)
    if skip < 1:
        raise ValueError("skip must be >= 1: a feature at t may not read session t")

    rets, market = regime.market_log_returns(panel)
    beta = (regime.rolling_beta(panel)
            .pivot(index="date", columns="ticker", values="beta")
            .reindex(index=rets.index, columns=rets.columns))
    # Known at t-skip, so the residual reads no session the raw return skips.
    beta_known = beta.shift(skip)

    frames: dict[str, pd.DataFrame] = {}
    for k in lookbacks:
        # rolling(k).sum() at s is log close(s) - log close(s-k); shifting by
        # `skip` moves the window's end from t back to t-skip. min_periods=k
        # voids any window that contains a missing session.
        r_k = rets.rolling(k, min_periods=k).sum().shift(skip)
        m_k = market.rolling(k, min_periods=k).sum().shift(skip)
        frames[f"rev_raw_{k}"] = -r_k
        frames[f"rev_resid_{k}"] = -(r_k - beta_known.mul(m_k, axis=0))

    long = pd.concat({c: frames[c].stack(future_stack=True) for c in cols}, axis=1)
    long.index = long.index.set_names(["date", "ticker"])
    out = long.reset_index().replace([np.inf, -np.inf], np.nan)
    return out[["date", "ticker"] + cols]
