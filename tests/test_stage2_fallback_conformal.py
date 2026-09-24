"""
Stage 2, continued — the thin-sector fallback and the spread-normalised
interval (2026-09-24, docs/stage2-fallback-conformal-preregistration.md).

  * no flag, and not `benchmark_sector_specific`, reaches any feature matrix —
    if the model can read "this name is benchmarked against the market", the
    fingerprint survives the fallback;
  * the leave-one-out market mean never contains the stock;
  * the fallback feature and the excess label read ONE reference;
  * the interval's spread is PAST-ONLY — the test corrupts every label not yet
    realised and requires the spread unchanged, which a spread read on the
    forecast's own date fails;
  * coverage against a hand-checked case, in price space and in log space;
  * the fingerprint arms (real 11 / random 11) rebuild deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import pooled
from pipeline import sector_benchmark as sb
from pipeline.conformal import (ConformalCalibration, ScaledConformalCalibration,
                                expanding_fold_coverage, fit_scaled_conformal)
from pipeline.signals import SECTOR_REL_COLS, compute_signals_frame

from tests.test_stage2_pooled_shadow import PARAMS, synthetic
from tests.test_stage2_sector_benchmark import SECTORS, _ohlcv, _prices

FLAGS = ("sector_rel_missing", "benchmark_sector_specific", "benchmark_ticker")


@pytest.fixture
def frozen_earnings(monkeypatch):
    import pipeline.signals as signals

    def fake(ticker, df):
        df["earnings_surprise"] = 0.1
        return df
    monkeypatch.setattr(signals, "compute_earnings_surprise", fake)


# ── Part B ────────────────────────────────────────────────────────────────────


def test_no_benchmark_flag_reaches_any_feature_matrix():
    from pipeline.baselines import FACTORS
    from pipeline.model import FEATURES as PER_TICKER
    from pipeline.panel import FEATURES as PANEL, SCALE_FREE
    from pipeline.signals import FEATURE_COLS

    for cols in (FEATURE_COLS, PER_TICKER, PANEL, SCALE_FREE, FACTORS, pooled.FEATURES):
        assert not set(FLAGS) & set(cols)
    # ...and what the pooled booster was actually fitted on, with the flags
    # present in the frame it was handed.
    raw = synthetic()
    raw["benchmark_sector_specific"] = (raw["ticker"] < "T05").astype(int)
    raw["sector_rel_missing"] = raw["benchmark_sector_specific"]
    fitted = pooled.fit_final(pooled.prepare(raw), params=PARAMS)
    names = fitted.model().get_booster().feature_names
    assert names == list(pooled.FEATURES)
    assert not set(FLAGS) & set(names)


def test_the_market_fallback_is_leave_one_out_exactly():
    prices = _prices()
    b = sb.build_benchmarks(prices, SECTORS)["T0.NS"]
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    others = [t for t in wide.columns if t != "T0.NS"]
    expected = np.log(wide[others] / wide[others].shift(1)).mean(axis=1).fillna(0).cumsum()
    np.testing.assert_allclose(np.log(b.frame["benchmark_close"].to_numpy()),
                               expected.to_numpy(), atol=1e-12)
    # the stock's own price never moves its market reference
    moved = prices.copy()
    own = moved["ticker"] == "T0.NS"
    moved.loc[own, ["close", "adj_close"]] *= np.linspace(1, 4, own.sum())[:, None]
    after = sb.build_benchmarks(moved, SECTORS)["T0.NS"].frame
    np.testing.assert_array_equal(b.frame["benchmark_close"], after["benchmark_close"])


def test_the_fallback_feature_and_the_excess_label_share_one_reference(frozen_earnings):
    prices = _prices()
    b = sb.build_benchmarks(prices, SECTORS)["T0.NS"]
    f = compute_signals_frame("T0.NS", _ohlcv(prices, "T0.NS"), b).reset_index(drop=True)
    level = b.frame.set_index("date")["benchmark_close"].reindex(f["date"]).to_numpy()
    np.testing.assert_allclose(f["benchmark_close"].to_numpy(), level, rtol=0, atol=0)
    for w in (5, 10, 20):
        # recomputed on the stored rows, so defined from the w-th stored row on
        rel = f["close"].pct_change(w) - f["benchmark_close"].pct_change(w)
        m = rel.notna().to_numpy()
        assert m.sum() > len(f) - 25
        np.testing.assert_allclose(f[f"sector_rel_{w}d"].to_numpy()[m],
                                   rel.to_numpy()[m], atol=1e-15)
    lb = np.log(f["benchmark_close"])
    np.testing.assert_allclose(f["benchmark_return"].to_numpy(),
                               (lb.shift(-30) - lb).to_numpy(), atol=1e-15)
    assert (f["benchmark_ticker"] == sb.MARKET_BENCHMARK).all()


def test_the_fingerprint_arms_rebuild_deterministically():
    from tools.stage2_diagnose import placebo_draws, thin_names, with_nulls

    raw = synthetic(n_tickers=30)
    raw["benchmark_sector_specific"] = (~raw["ticker"].isin(
        ["T03.NS", "T11.NS", "T20.NS"])).astype(int)
    for c in SECTOR_REL_COLS:
        raw[c] = 0.01
    thin = thin_names(raw)
    assert thin == ["T03.NS", "T11.NS", "T20.NS"]
    real = with_nulls(raw, thin)
    assert real.loc[real["ticker"].isin(thin), list(SECTOR_REL_COLS)].isna().all().all()
    assert real.loc[~real["ticker"].isin(thin), list(SECTOR_REL_COLS)].notna().all().all()
    a = placebo_draws(thin, list(raw["ticker"].unique()), 3)
    b = placebo_draws(thin, list(raw["ticker"].unique())[::-1], 3)
    assert a == b and all(len(d) == 3 and not set(d) & set(thin) for d in a)


# ── Part C ────────────────────────────────────────────────────────────────────


def test_the_interval_spread_is_past_only():
    """Corrupt every label NOT yet realised at `t` (a label at date d realises
    at d + 30); the spread at `t` must not move. A spread read on `t`'s own
    date, or any horizon short of 30, reads a corrupted label and fails."""
    raw = synthetic()
    dates = sorted(raw["date"].unique())
    base = pooled.spread_frame(raw).set_index("date")["interval_spread"]
    for t in (dates[200], dates[300], dates[-1]):
        i = dates.index(t)
        tampered = raw.copy()
        tampered.loc[tampered["date"] > dates[i - 30], "target_return"] = 7.0
        after = pooled.spread_frame(tampered).set_index("date")["interval_spread"]
        assert np.isfinite(base[t])
        assert after[t] == base[t], t
    with pytest.raises(ValueError, match="RAW"):
        pooled.spread_frame(pooled.prepare(raw))


def test_the_interval_spread_is_the_trailing_mean_of_realised_dispersion():
    raw = synthetic()
    dates = sorted(raw["date"].unique())
    sd = raw.groupby("date")["target_return"].std()
    t = dates[250]
    i = dates.index(t)
    window = dates[i - 30 - pooled.SPREAD_LOOKBACK + 1: i - 30 + 1]
    expected = sd.loc[window].mean()
    got = pooled.spread_frame(raw).set_index("date").loc[t, "interval_spread"]
    assert got == pytest.approx(expected, rel=1e-12)
    assert pooled.SPREAD_LOOKBACK == 21


def _hand_case():
    """Fold 0 calibrates, fold 1 is checked. Fold 0: spread 2, residuals
    2 x (1..20), so normalised scores are 1..20 and the rank
    ceil(21 x 0.8) = 17 gives q = 17. Fold 1: spread 0.5, ten residuals of
    0.5 x 8 (score 8: inside) and ten of 0.5 x 18 (score 18: outside)."""
    y0 = 2.0 * np.arange(1, 21) * np.where(np.arange(20) % 2, 1, -1)
    y1 = np.r_[np.full(10, 0.5 * 8), np.full(10, -0.5 * 18)]
    y = np.r_[y0, y1] / 100          # log returns
    spread = np.r_[np.full(20, 2.0), np.full(20, 0.5)] / 100
    folds = np.r_[np.zeros(20), np.ones(20)].astype(int)
    return y, np.zeros_like(y), spread, folds


def test_coverage_against_a_hand_checked_case():
    y, pred, spread, folds = _hand_case()
    scaled = expanding_fold_coverage(y, pred, folds, spread=spread)
    assert scaled["method"] == "spread-normalised"
    (f1,) = scaled["per_fold"]
    assert f1["quantile"] == pytest.approx(17.0)
    assert f1["coverage"] == 0.5 and scaled["overall"] == 0.5
    assert f1["mean_width_log"] == pytest.approx(2 * 17 * 0.005)
    # The constant method on the SAME rows: |residual| 2..40 (/100) gives
    # q = 0.34, which covers every fold-1 residual (0.04 and 0.09).
    const = expanding_fold_coverage(y, pred, folds)
    assert const["per_fold"][0]["quantile"] == pytest.approx(0.34)
    assert const["overall"] == 1.0
    # In PRICE space the event is the same.
    price = np.full(y.size, 250.0)
    in_price = expanding_fold_coverage(y, pred, folds, spread=spread, price=price)
    assert in_price["overall"] == scaled["overall"]
    assert in_price["per_fold"][0]["mean_width_pct"] == pytest.approx(
        (np.exp(0.085) - np.exp(-0.085)) * 100)


def test_price_space_and_log_space_coverage_agree_on_random_data():
    rng = np.random.default_rng(3)
    n = 5000
    folds = np.repeat(np.arange(5), n // 5)
    spread = np.exp(rng.normal(-2.3, 0.3, n))
    y = rng.standard_t(4, n) * spread
    pred = rng.normal(0, 0.01, n)
    price = np.exp(rng.normal(6, 1, n))
    a = expanding_fold_coverage(y, pred, folds, spread=spread)
    b = expanding_fold_coverage(y, pred, folds, spread=spread, price=price)
    assert a["overall"] == b["overall"]
    assert [f["coverage"] for f in a["per_fold"]] == [f["coverage"] for f in b["per_fold"]]


def test_the_scaled_band_follows_dispersion_where_the_constant_one_cannot():
    """The failure the method exists for: a late fold far calmer than the
    calibration pool. With the spread known, the scaled band keeps ~80%."""
    rng = np.random.default_rng(0)
    folds = np.repeat(np.arange(5), 4000)
    spread = np.where(folds == 4, 0.3, 1.0)
    y = rng.normal(size=folds.size) * spread
    from pipeline.pooled_shadow import GATE_FAIL, GATE_PASS, coverage_gate
    assert coverage_gate(y, np.zeros_like(y), folds)["status"] == GATE_FAIL
    ok = coverage_gate(y, np.zeros_like(y), folds, spread=spread)
    assert ok["status"] == GATE_PASS
    widths = [f["mean_width_log"] for f in ok["per_fold"]]
    assert widths[-1] < 0.5 * widths[0]


def test_a_scaled_calibration_withholds_an_interval_without_a_spread():
    cal = fit_scaled_conformal(np.arange(1, 41) / 100, np.zeros(40), np.full(40, 0.02))
    assert isinstance(cal, ScaledConformalCalibration)
    for bad in (float("nan"), 0.0, -1.0, None):
        assert cal.at(bad) is None
    at = cal.at(0.04)
    assert isinstance(at, ConformalCalibration)
    assert at.quantile == pytest.approx(cal.quantile * 0.04)
    np.testing.assert_allclose(at.residuals, cal.scores * 0.04)



def test_an_existing_shadow_forecasts_table_gains_the_spread_column(tmp_path):
    """The v1 table shipped without `interval_spread`; CREATE IF NOT EXISTS
    would leave it so, and the daily step's insert would then fail."""
    from sqlalchemy import create_engine, inspect, text

    from pipeline.pooled_shadow import init_shadow_tables

    eng = create_engine(f"sqlite:///{(tmp_path / 'v1.sqlite').as_posix()}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE shadow_forecasts (forecast_date TEXT, ticker TEXT, "
                       "model_id TEXT, interval_coverage REAL)"))
    init_shadow_tables(eng)
    init_shadow_tables(eng)                                   # idempotent
    cols = {c["name"] for c in inspect(eng).get_columns("shadow_forecasts")}
    assert "interval_spread" in cols
