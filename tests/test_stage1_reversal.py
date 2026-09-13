"""
Stage 1, Pilot 1 — the multi-lookback reversal features.

The contract is small, and every clause of it fails silently when broken:
- the feature at t reads nothing from session t;
- the residual is built from the project's existing trailing beta and market
  series, not from a copy;
- a hole in a series voids a window rather than stretching it;
- the baseline arm is exactly the Stage 2b factor set.

See docs/stage1-preregistration.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import regime
from pipeline.reversal import (
    LOOKBACKS,
    RAW_COLS,
    RESID_COLS,
    REVERSAL_COLS,
    reversal_features,
)

N_DATES = 160
TICKERS = [f"T{i:02d}" for i in range(24)]


def _panel(n_dates: int = N_DATES, tickers: list[str] = TICKERS,
           seed: int = 0) -> pd.DataFrame:
    """Random walks with a common market factor and a spread of betas."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates).strftime("%Y-%m-%d")
    market = rng.normal(0.0005, 0.01, n_dates)
    frames = []
    for j, ticker in enumerate(tickers):
        beta = 0.5 + j / len(tickers)
        r = beta * market + rng.normal(0.0, 0.015, n_dates)
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker,
                                    "close": 100.0 * np.exp(np.cumsum(r))}))
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["date", "ticker"]).reset_index(drop=True))


def _wide(features: pd.DataFrame, col: str) -> pd.DataFrame:
    return features.pivot(index="date", columns="ticker", values=col)


# ── the skipped session ───────────────────────────────────────────────────────

def test_the_skip_is_exactly_one_session():
    """Session t never reaches the feature at t — and DOES reach it at t+1, so
    the skip is one session: not two, and not zero."""
    panel = _panel()
    dates = sorted(panel["date"].unique())
    t, t_next = dates[120], dates[121]

    base = reversal_features(panel)
    shocked = panel.copy()
    later = shocked["date"] >= t
    shocked.loc[later, "close"] *= np.random.default_rng(1).uniform(
        0.5, 2.0, int(later.sum()))
    after = reversal_features(shocked)

    for col in REVERSAL_COLS:
        pd.testing.assert_series_equal(
            _wide(base, col).loc[t], _wide(after, col).loc[t],
            obj=f"{col} at t moved when only session t and later changed")
    assert not np.allclose(_wide(base, "rev_raw_1").loc[t_next],
                           _wide(after, "rev_raw_1").loc[t_next]), (
        "the feature at t+1 must read session t; if it does not, the skip is "
        "wider than one session")


def test_the_raw_feature_is_the_negated_skip_one_log_return():
    panel = _panel()
    features = reversal_features(panel)
    log_close = np.log(panel.pivot(index="date", columns="ticker", values="close"))
    for k in LOOKBACKS:
        expected = -(log_close.shift(1) - log_close.shift(1 + k))
        pd.testing.assert_frame_equal(_wide(features, f"rev_raw_{k}"), expected,
                                      check_names=False)


def test_a_missing_session_voids_the_window_instead_of_stretching_it():
    panel = _panel()
    dates = sorted(panel["date"].unique())
    hole = dates[80]
    holed = panel[~((panel["ticker"] == "T00") & (panel["date"] == hole))]

    rev5 = _wide(reversal_features(holed), "rev_raw_5")
    # The returns AT the hole and at the session after it are undefined. The
    # feature at t sums returns t-5..t-1, so every t in dates[81..86] is void.
    assert rev5.loc[dates[81]:dates[86], "T00"].isna().all(), (
        "a window spanning a missing session must be void, not one session "
        "longer (the vroc_10 landmine)")
    assert np.isfinite(rev5.loc[dates[87], "T00"])
    assert np.isfinite(rev5.loc[dates[81], "T01"]), "other names are untouched"


# ── the residual reuses the project's beta and market ─────────────────────────

def _constant_beta(value: float, calls: list, real):
    """A rolling_beta that reports `value` everywhere. It wraps the REAL
    function, captured once, so that stacking two patches cannot chain one
    fake through the other."""
    def fake(panel, *args, **kwargs):
        calls.append(value)
        return real(panel, *args, **kwargs).assign(beta=value)
    return fake


def test_the_residual_is_built_from_the_projects_rolling_beta(monkeypatch):
    panel = _panel()
    real = regime.rolling_beta
    calls: list = []

    monkeypatch.setattr(regime, "rolling_beta", _constant_beta(0.0, calls, real))
    zero = reversal_features(panel)
    for raw, resid in zip(RAW_COLS, RESID_COLS):
        pd.testing.assert_frame_equal(_wide(zero, resid), _wide(zero, raw),
                                      check_names=False)

    monkeypatch.setattr(regime, "rolling_beta", _constant_beta(1.0, calls, real))
    one = reversal_features(panel)
    _, market = regime.market_log_returns(panel)
    for k, raw, resid in zip(LOOKBACKS, RAW_COLS, RESID_COLS):
        m_k = market.rolling(k, min_periods=k).sum().shift(1)
        # -(r - 1 * m) = raw + m
        expected = _wide(one, raw).add(m_k, axis=0)
        pd.testing.assert_frame_equal(_wide(one, resid).iloc[k + 2:],
                                      expected.iloc[k + 2:], check_names=False)

    assert calls == [0.0, 1.0], "rolling_beta must be called once per build"


def test_the_residual_uses_the_beta_known_at_t_minus_one(monkeypatch):
    panel = _panel()
    dates = sorted(panel["date"].unique())
    t0, t1 = dates[100], dates[101]
    real = regime.rolling_beta
    monkeypatch.setattr(regime, "rolling_beta", lambda p, *a, **kw: (
        real(p, *a, **kw).assign(beta=lambda d: np.where(d["date"] < t0, 1.0, 3.0))))

    features = reversal_features(panel)
    _, market = regime.market_log_returns(panel)
    m1 = market.shift(1)                 # the k = 1 window ends at t-1
    gap = _wide(features, "rev_resid_1") - _wide(features, "rev_raw_1")   # beta * m

    np.testing.assert_allclose(gap.loc[t0].to_numpy(),
                               np.full(len(TICKERS), 1.0 * m1.loc[t0]),
                               err_msg="at t0 the beta known is t0-1's, which is 1.0")
    np.testing.assert_allclose(gap.loc[t1].to_numpy(),
                               np.full(len(TICKERS), 3.0 * m1.loc[t1]))


def test_the_market_has_one_definition(monkeypatch):
    real = regime.market_log_returns
    calls: list = []

    def spy(panel, *args, **kwargs):
        calls.append(1)
        return real(panel, *args, **kwargs)

    monkeypatch.setattr(regime, "market_log_returns", spy)
    reversal_features(_panel())
    assert len(calls) == 2, (
        "the residual's market AND rolling_beta's must both come from "
        f"regime.market_log_returns; saw {len(calls)} call(s)")


def test_rolling_beta_is_unchanged_by_moving_the_market_into_one_helper():
    panel = _panel()
    wide = panel.pivot_table(index="date", columns="ticker", values="close",
                             aggfunc="last").sort_index()
    rets = np.log(wide / wide.shift(1))
    market = rets.mean(axis=1)
    cov = rets.rolling(252, min_periods=60).cov(market)
    var = market.rolling(252, min_periods=60).var()
    reference = cov.div(var.replace(0, np.nan), axis=0)

    got = regime.rolling_beta(panel).pivot(index="date", columns="ticker",
                                           values="beta")
    pd.testing.assert_frame_equal(got, reference, check_names=False)


# ── the arms and their preprocessing ──────────────────────────────────────────

def test_the_baseline_arm_is_exactly_the_stage2b_factors():
    from pipeline.baselines import FACTORS
    from tools.stage1_reversal import ARM_FEATURES
    from tools.stage2b_pooled import feature_columns

    assert list(FACTORS) == [
        "roc_10", "sector_rel_20d", "sector_rel_5d", "sector_rel_10d",
        "lag1_ret", "lag5_ret", "rsi", "stoch_k", "williams_r", "dev_sma50",
        "prox_52w", "bb_width", "vroc_10", "hurst", "earnings_surprise"]
    assert feature_columns("none") == list(FACTORS)
    assert ARM_FEATURES["baseline"] == list(FACTORS)
    assert ARM_FEATURES["resid_reversal"] == list(FACTORS) + RESID_COLS
    assert ARM_FEATURES["raw_reversal"] == list(FACTORS) + RAW_COLS


def test_the_new_columns_are_standardised_like_every_other_pooled_feature():
    from tools.stage1_reversal import attach_reversal

    panel = _panel()
    out = attach_reversal(panel)
    pd.testing.assert_frame_equal(out[["date", "ticker", "close"]],
                                  panel[["date", "ticker", "close"]])

    # Judged only on dates where the feature is DEFINED. The residual needs
    # rolling_beta's 60-return minimum plus the one-session lag, so its first
    # ~62 sessions are undefined and standardise to zero, as any missing
    # feature does on this panel.
    raw = reversal_features(panel)
    for col in REVERSAL_COLS:
        defined = raw.groupby("date")[col].count()
        dates = defined[defined >= 10].index
        assert len(dates) > 50, f"{col}: fixture too short to judge"
        by_date = out[out["date"].isin(dates)].groupby("date")[col]
        assert by_date.mean().abs().max() < 0.2, col
        assert by_date.std().between(0.8, 1.1).all(), col
        assert out[col].abs().max() <= 3.0, f"{col} must be clipped like the rest"
