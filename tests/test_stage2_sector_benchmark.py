"""
Stage 2, Part 0 — the panel-internal sector benchmark (2026-09-24).

Pins the properties that make it a benchmark rather than a number:

  * leave-one-out: a stock's own price never reaches its own benchmark;
  * it IS the mean of the peers' daily log returns, cumulated;
  * a thin sector (fewer than MIN_SECTOR_PEERS peers) gets NO sector features
    — NULL with sector_rel_missing = 1, never 0.0 — and its excess label falls
    back to the market, flagged sector_specific = 0;
  * "Unknown" / an empty label is missing, never a pseudo-sector;
  * a window spanning a date the peers could not form a mean is void;
  * NSE's constituent CSV without an industry column is refused, loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import sector_benchmark as sb
from pipeline.signals import SECTOR_REL_COLS, compute_signals_frame


def _prices(n_dates=300, tickers=None, seed=0):
    rng = np.random.default_rng(seed)
    tickers = tickers or [f"S{i}.NS" for i in range(5)] + [f"T{i}.NS" for i in range(2)]
    dates = pd.bdate_range("2021-01-01", periods=n_dates).strftime("%Y-%m-%d")
    rows = []
    for t in tickers:
        lp = np.cumsum(rng.normal(0, 0.01, n_dates)) + 5
        rows.append(pd.DataFrame({"date": dates, "ticker": t, "close": np.exp(lp),
                                  "adj_close": np.exp(lp)}))
    return pd.concat(rows, ignore_index=True)


SECTORS = {**{f"S{i}.NS": "Big" for i in range(5)},
           "T0.NS": "Thin", "T1.NS": "Thin"}


def test_a_stock_never_enters_its_own_benchmark():
    prices = _prices()
    before = sb.build_benchmarks(prices, SECTORS)["S0.NS"].frame
    moved = prices.copy()
    own = moved["ticker"] == "S0.NS"
    moved.loc[own, ["close", "adj_close"]] *= np.linspace(1, 3, own.sum())[:, None]
    after = sb.build_benchmarks(moved, SECTORS)["S0.NS"].frame
    np.testing.assert_array_equal(before["benchmark_close"], after["benchmark_close"])
    # ...while a PEER's move does reach it.
    peer = prices.copy()
    p = peer["ticker"] == "S1.NS"
    peer.loc[p, ["close", "adj_close"]] *= np.linspace(1, 3, p.sum())[:, None]
    assert not np.allclose(before["benchmark_close"],
                           sb.build_benchmarks(peer, SECTORS)["S0.NS"].frame["benchmark_close"])


def test_the_benchmark_is_the_mean_of_the_peers_log_returns():
    prices = _prices()
    b = sb.build_benchmarks(prices, SECTORS)["S0.NS"]
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    peers = [f"S{i}.NS" for i in range(1, 5)]
    expected = np.log(wide[peers] / wide[peers].shift(1)).mean(axis=1).fillna(0).cumsum()
    np.testing.assert_allclose(np.log(b.frame["benchmark_close"].to_numpy()),
                               expected.to_numpy(), atol=1e-12)
    assert b.name == "EW-LOO:Big" and b.sector_specific and b.peers == 4


def test_a_thin_sector_gets_no_sector_features_and_a_flagged_market_label():
    b = sb.build_benchmarks(_prices(), SECTORS)["T0.NS"]
    assert sb.MIN_SECTOR_PEERS == 2
    assert b.name == sb.MARKET_BENCHMARK
    assert not b.sector_specific and not b.relative_features
    # market fallback: every other name in the universe
    assert b.peers == 6


def test_unknown_is_missing_never_a_pseudo_sector():
    # THREE "Unknown" names: enough to form a sector with two peers each if
    # "Unknown" were ever read as a label — one alone would be too thin to
    # tell the two readings apart (a mutant survived exactly that fixture).
    sectors = {**SECTORS, "S2.NS": "Unknown", "S3.NS": "Unknown", "T0.NS": "Unknown",
               "S4.NS": "", "T1.NS": "nan"}
    out = sb.build_benchmarks(_prices(), sectors)
    for t in ("S2.NS", "S3.NS", "S4.NS", "T0.NS", "T1.NS"):
        assert out[t].name == sb.MARKET_BENCHMARK, t
        assert not out[t].relative_features
    assert all(not b.name.endswith("Unknown") for b in out.values())
    # The two remaining "Big" names are a sector of 2: one peer each, thin.
    assert out["S0.NS"].name == sb.MARKET_BENCHMARK


def test_a_window_spanning_a_date_the_peers_missed_is_void():
    prices = _prices(tickers=["A.NS", "B.NS", "C.NS"])
    sectors = {"A.NS": "X", "B.NS": "X", "C.NS": "X"}
    hole = sorted(prices["date"].unique())[200]
    # B trades on the hole date but C does not: A has ONE peer that day.
    prices = prices[~((prices["ticker"] == "C.NS") & (prices["date"] == hole))]
    b = sb.build_benchmarks(prices, sectors)["A.NS"]
    f = b.frame.set_index("date")
    assert f.loc[hole, "bench_void_cum"] > f["bench_void_cum"].iloc[0]


def test_a_void_date_voids_the_features_and_the_label_that_span_it(frozen_earnings):
    """Through `compute_signals_frame`: every backward feature window and every
    forward label window that spans the hole is NULL — never stretched."""
    prices = _prices(tickers=["A.NS", "B.NS", "C.NS"])
    sectors = {"A.NS": "X", "B.NS": "X", "C.NS": "X"}
    dates = sorted(prices["date"].unique())
    hole = dates[200]
    thin = prices[~((prices["ticker"] == "C.NS") & (prices["date"] == hole))]
    b = sb.build_benchmarks(thin, sectors)["A.NS"]
    f = compute_signals_frame("A.NS", _ohlcv(prices, "A.NS"), b).set_index("date")
    i = dates.index(hole)
    # C's return is missing ON the hole and the day after it: both void.
    assert f.loc[dates[i], "sector_rel_5d"] != f.loc[dates[i], "sector_rel_5d"]
    assert pd.isna(f.loc[dates[i + 4], "sector_rel_5d"])
    assert pd.notna(f.loc[dates[i + 7], "sector_rel_5d"])
    assert pd.isna(f.loc[dates[i - 5], "benchmark_return"])     # its 30 sessions span the hole
    assert pd.notna(f.loc[dates[i - 40], "benchmark_return"])
    assert pd.isna(f.loc[dates[i - 5], "target_excess_return"])


def _ohlcv(prices, ticker):
    p = prices[prices["ticker"] == ticker].copy()
    p["open"] = p["high"] = p["low"] = p["close"]
    p["high"] *= 1.01
    p["low"] *= 0.99
    p["volume"] = 1_000_000.0
    return p.reset_index(drop=True)


@pytest.fixture
def frozen_earnings(monkeypatch):
    import pipeline.signals as signals

    def fake(ticker, df):
        df["earnings_surprise"] = 0.1
        return df
    monkeypatch.setattr(signals, "compute_earnings_surprise", fake)


def test_thin_sector_features_are_null_with_an_indicator_never_zero(frozen_earnings):
    prices = _prices()
    bm = sb.build_benchmarks(prices, SECTORS)
    thin = compute_signals_frame("T0.NS", _ohlcv(prices, "T0.NS"), bm["T0.NS"])
    assert thin[list(SECTOR_REL_COLS)].isna().all().all()
    assert (thin["sector_rel_missing"] == 1).all()
    assert (thin[list(SECTOR_REL_COLS)] != 0.0).all().all()
    # the excess label still exists, against the market, and says so
    assert thin["target_excess_return"].notna().sum() > 0
    assert (thin["benchmark_sector_specific"] == 0).all()
    assert (thin["benchmark_ticker"] == sb.MARKET_BENCHMARK).all()
    # no row was DROPPED for the NULL features
    big = compute_signals_frame("S0.NS", _ohlcv(prices, "S0.NS"), bm["S0.NS"])
    assert len(thin) == len(big)
    assert big[list(SECTOR_REL_COLS)].notna().all().all()
    assert (big["sector_rel_missing"] == 0).all()


def test_the_excess_label_is_stock_minus_benchmark_over_the_same_window(frozen_earnings):
    prices = _prices()
    b = sb.build_benchmarks(prices, SECTORS)["S0.NS"]
    f = compute_signals_frame("S0.NS", _ohlcv(prices, "S0.NS"), b).set_index("date")
    lvl = b.frame.set_index("date")["benchmark_close"]
    d = f.index[10]
    later = lvl.index[lvl.index.get_loc(d) + 30]
    expected = np.log(lvl[later] / lvl[d])
    assert f.loc[d, "benchmark_return"] == pytest.approx(expected, abs=1e-12)
    assert f.loc[d, "target_excess_return"] == pytest.approx(
        f.loc[d, "target_return"] - expected, abs=1e-12)


def test_no_benchmark_means_null_relative_features_and_excess_label(frozen_earnings):
    prices = _prices()
    f = compute_signals_frame("S0.NS", _ohlcv(prices, "S0.NS"), None)
    assert f["target_excess_return"].isna().all()
    assert f[list(SECTOR_REL_COLS)].isna().all().all()
    assert f["target_return"].notna().sum() > 0


def test_sector_labels_read_the_latest_row_even_when_closed(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, text

    eng = create_engine(f"sqlite:///{(tmp_path / 'm.sqlite').as_posix()}")
    with eng.begin() as c:
        c.execute(text("CREATE TABLE index_membership (ticker TEXT, index_name TEXT, "
                       "effective_from TEXT, effective_to TEXT, company TEXT, "
                       "industry TEXT, source TEXT)"))
        c.execute(text("INSERT INTO index_membership VALUES "
                       "('A.NS','NIFTY100','2020-01-01','2024-01-01','A','Old','x'),"
                       "('A.NS','NIFTY100','2024-01-01','2025-06-01','A','New','x'),"
                       "('B.NS','NIFTY100','2020-01-01','9999-12-31','B','','x')"))
    labels = sb.sector_labels(["A.NS", "B.NS", "C.NS"], eng)
    assert labels == {"A.NS": "New", "B.NS": None, "C.NS": None}


def test_nse_constituents_without_an_industry_column_are_refused(monkeypatch):
    import data.universe as universe

    class Resp:
        text = "Company Name,Symbol,Series\nFoo Ltd,FOO,EQ\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(universe.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(ValueError, match="industry"):
        universe.fetch_nse_constituents()


def test_benchmark_names_are_human_readable():
    from data.tickers import get_benchmark_name

    assert get_benchmark_name("EW-LOO:Healthcare") == "Healthcare peers (equal-weighted)"
    assert "rest of the universe" in get_benchmark_name(sb.MARKET_BENCHMARK)
    assert get_benchmark_name("^CNXIT") == "NIFTY IT"
