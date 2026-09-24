"""
Stage 2, Part 1 — a failed measurement must never be stored as a valid neutral
value (2026-09-24).

Three instances shipped before this: FinBERT's neutral gauge, the FII/DII
zeros, and `earnings_surprise = 0.0` after the lock dropped lxml. These pin:

  * the standing gate check FIRES on an injected constant-at-neutral column
    and on a date constant across every ticker, and stays quiet on a healthy
    frame;
  * an earnings fetch that fails or returns nothing SKIPS the ticker instead
    of writing 0.0 over its whole history;
  * earnings_surprise is NULL before the vendor's first announcement;
  * Hurst is NULL during its warm-up, never a partial-window 0.0 or a 0.5;
  * the loaders keep a NULLABLE feature NaN instead of refilling 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import silent_neutral as sn


def _frame(n_dates=150, n_tickers=12, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_dates).strftime("%Y-%m-%d")
    return pd.DataFrame([{"date": d, "ticker": f"T{i}.NS", "x": rng.normal()}
                         for d in dates for i in range(n_tickers)])


def test_the_standing_check_is_quiet_on_a_healthy_column():
    assert sn.standing_findings(_frame(), ["x"]) == []


def test_it_fires_on_a_ticker_constant_at_a_neutral_value():
    df = _frame()
    df.loc[df["ticker"] == "T3.NS", "x"] = 0.0          # the lxml fingerprint
    found = sn.standing_findings(df, ["x"])
    assert found and "T3.NS" in found[0] and "neutral" in found[0]


def test_it_fires_on_a_date_constant_across_every_ticker():
    df = _frame()
    day = df["date"].iloc[-1]
    df.loc[df["date"] == day, "x"] = 0.0                # the FII/DII fingerprint
    assert any("same value" in f for f in sn.standing_findings(df, ["x"]))


def test_a_constant_at_a_non_neutral_value_is_not_the_fingerprint():
    df = _frame()
    df.loc[df["ticker"] == "T3.NS", "x"] = 0.4321
    assert not any("neutral" in f for f in sn.standing_findings(df, ["x"]))


def test_the_audit_reports_the_indicators():
    df = _frame()
    df.loc[df["ticker"] == "T3.NS", "x"] = 0.0
    row = sn.audit(df, ["x"]).iloc[0]
    assert row["const_neutral_tickers"] == ["T3.NS"]
    assert row["neutral_value"] == 0.0
    assert row["longest_run"] >= 150


def test_the_gate_carries_the_standing_check(tmp_path, monkeypatch):
    from sqlalchemy import create_engine

    from pipeline import validation
    from pipeline.signals import FEATURE_COLS

    assert validation.check_no_silent_neutral_signals in validation.CHECKS
    df = _frame(n_tickers=12)
    for c in FEATURE_COLS:
        df[c] = np.random.default_rng(len(c)).normal(size=len(df))
    df.loc[df["ticker"] == "T5.NS", "earnings_surprise"] = 0.0
    eng = create_engine(f"sqlite:///{(tmp_path / 'g.sqlite').as_posix()}")
    df.drop(columns=["x"]).to_sql("signals", eng, index=False)
    universe = sorted(df["ticker"].unique())
    check = validation.check_no_silent_neutral_signals(eng, universe)
    assert check.status == validation.WARN
    assert "earnings_surprise" in check.detail and "T5.NS" in check.detail
    df.loc[df["ticker"] == "T5.NS", "earnings_surprise"] = 0.3 + np.arange(
        (df["ticker"] == "T5.NS").sum()) * 1e-3
    df.drop(columns=["x"]).to_sql("signals", eng, index=False, if_exists="replace")
    assert validation.check_no_silent_neutral_signals(eng, universe).status == validation.PASS


# ── earnings ──────────────────────────────────────────────────────────────────


def _sessions(n=60):
    return pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=n)
                         .strftime("%Y-%m-%d")})


def _vendor(dates, est, act):
    return pd.DataFrame({"EPS Estimate": est, "Reported EPS": act},
                        index=pd.Index(pd.to_datetime(dates).tz_localize("UTC"),
                                       name="Earnings Date"))


def test_earnings_are_null_before_the_first_announcement_not_zero():
    from pipeline.signals import earnings_surprise_from

    df = earnings_surprise_from(_vendor(["2024-02-01"], [10.0], [11.0]), _sessions())
    before = df[df["date"] <= "2024-02-01"]["earnings_surprise"]
    after = df[df["date"] > "2024-02-01"]["earnings_surprise"]
    assert before.isna().all()
    np.testing.assert_allclose(after.to_numpy(), 0.1)


@pytest.mark.parametrize("vendor", [None, pd.DataFrame(),
                                    _vendor(["2024-02-01"], [0.0], [1.0])])
def test_no_usable_earnings_history_is_refused_not_zeroed(vendor):
    from pipeline.signals import EarningsUnavailable, earnings_surprise_from

    with pytest.raises(EarningsUnavailable):
        earnings_surprise_from(vendor, _sessions(), "X.NS")


def test_a_failed_earnings_fetch_skips_the_ticker(monkeypatch):
    import pipeline.signals as signals

    class Boom:
        def __init__(self, *_):
            pass

        @property
        def earnings_dates(self):
            raise ConnectionError("429 Too Many Requests")

    monkeypatch.setattr(signals.yf, "Ticker", Boom)
    n = 300
    dates = pd.bdate_range("2021-01-01", periods=n).strftime("%Y-%m-%d")
    px = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, n)) + 5)
    ohlcv = pd.DataFrame({"date": dates, "ticker": "X.NS", "open": px, "high": px * 1.01,
                          "low": px * 0.99, "close": px, "adj_close": px,
                          "volume": 1e6})
    assert signals.compute_signals_frame("X.NS", ohlcv, None) is None


def test_hurst_is_null_in_its_warm_up_not_zero_or_half():
    from pipeline.signals import _rolling_hurst

    close = pd.Series(np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, 200)) + 4))
    h = _rolling_hurst(close)
    # 60-row window over 19-row differences: the first complete row is 78.
    assert h.iloc[:78].isna().all()
    assert h.iloc[78:].notna().all()
    assert not ((h.iloc[:78] == 0.0) | (h.iloc[:78] == 0.5)).any()


def test_the_loaders_keep_a_nullable_feature_missing():
    from pipeline.panel import cross_sectional_zscore

    df = pd.DataFrame({"date": ["d"] * 12, "ticker": [f"T{i}" for i in range(12)],
                       "sector_rel_20d": [np.nan, np.nan] + list(np.arange(10.0))})
    kept = cross_sectional_zscore(df, ["sector_rel_20d"], keep_missing=True)
    filled = cross_sectional_zscore(df, ["sector_rel_20d"])
    assert kept["sector_rel_20d"].iloc[:2].isna().all()
    assert (filled["sector_rel_20d"].iloc[:2] == 0.0).all()
    np.testing.assert_array_equal(kept["sector_rel_20d"].iloc[2:],
                                  filled["sector_rel_20d"].iloc[2:])


def test_a_surprise_goes_stale_after_one_reporting_cycle():
    """BAJAJHLDNG carried a 2018 surprise for 1,954 sessions until this."""
    from pipeline.signals import EARNINGS_STALE_SESSIONS, earnings_surprise_from

    sessions = _sessions(400)
    df = earnings_surprise_from(_vendor(["2024-01-02"], [10.0], [11.0]), sessions)
    s = df["earnings_surprise"]
    first = int(s.first_valid_index())
    assert s.iloc[first:first + EARNINGS_STALE_SESSIONS + 1].notna().all()
    assert s.iloc[first + EARNINGS_STALE_SESSIONS + 1:].isna().all()


def test_the_contiguity_check_flags_surplus_rows_not_only_gaps(tmp_path):
    """It passed for the WRONG reason after the phantom fix: ohlcv lost the
    holidays, refused tickers' signals kept them, the difference went
    negative, and a one-directional check read that as clean."""
    from sqlalchemy import create_engine

    from pipeline import validation

    eng = create_engine(f"sqlite:///{(tmp_path / 'c.sqlite').as_posix()}")
    days = pd.bdate_range("2026-01-01", periods=40).strftime("%Y-%m-%d")
    pd.DataFrame({"date": days, "ticker": "A.NS"}).to_sql("ohlcv", eng, index=False)
    pd.DataFrame({"date": days, "ticker": "A.NS"}).to_sql("signals", eng, index=False)
    assert validation.check_sessions_are_contiguous(eng, ["A.NS"]).status == validation.PASS
    # ohlcv drops a phantom session the stored signals still hold
    pd.DataFrame({"date": [d for d in days if d != days[20]], "ticker": "A.NS"}) \
        .to_sql("ohlcv", eng, index=False, if_exists="replace")
    check = validation.check_sessions_are_contiguous(eng, ["A.NS"])
    assert check.status == validation.WARN and "A.NS(+1)" in check.detail
