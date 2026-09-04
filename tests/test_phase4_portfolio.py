"""
tests/test_phase4_portfolio.py — the cost-aware simulator.

A backtest is the easiest thing in this repository to get wrong in a way that
LOOKS right, so the guards here are ordered by how quietly each would fail.

1. THE INSTRUMENT MUST DETECT A PLANTED EDGE. A null from an untested simulator
   is indistinguishable from a broken one. This is the `series_zero` pattern —
   it reproduced the `zero` baseline to the row, which is what stopped the
   adapter being what was measured.
2. COSTS ON ACTUAL TURNOVER. Charging an assumed 100% each rebalance overstates
   the drag by the overlap, which for a 30-session ordering is most of the book.
3. NO ORDERING, NO PORTFOLIO. A stable sort once handed `zero` and `train_mean`
   an alphabetical long-short spread of +0.01744 at t +1.19.
4. THE REBALANCE GRID, NEVER THE DAILY ONE. Consecutive dates share 29 of 30
   forward sessions; a daily Sharpe here is inflated by roughly the overlap.
5. THE DEFLATION'S UNITS. Its expected-maximum term is in units of the SPREAD of
   trial Sharpes, not raw Sharpe. Leaving that at 1.0 sets the bar at +1.98 and
   deflates everything into the floor for a units reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation import (
    cross_sectional_report, deflated_sharpe_note, rebalance_books,
)
from pipeline.portfolio import (
    REBALANCES_PER_YEAR, CostModel, break_even_ic, simulate,
    synthetic_predictions,
)
from pipeline.signals import HORIZON_SESSIONS


def _panel(n_rebalances=40, n_tickers=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in
             pd.bdate_range("2020-01-01", periods=HORIZON_SESSIONS * n_rebalances)]
    tickers = [f"T{i:02d}.NS" for i in range(n_tickers)]
    df = pd.DataFrame([{"date": d, "ticker": t,
                        "y_true": float(rng.normal(0, 0.09))}
                       for d in dates for t in tickers])
    df["y_pred"] = 0.0
    return df


# ── 1. The instrument ─────────────────────────────────────────────────────────

def test_the_simulator_detects_a_planted_edge():
    """
    THE PRECONDITION FOR QUOTING ANY NULL. If net Sharpe does not rise with a
    planted edge, this tool cannot distinguish "no signal" from "broken", and
    its null on real data means nothing.
    """
    truth = _panel()
    sharpes = []
    for ic in (0.0, 0.05, 0.20):
        preds = synthetic_predictions(truth, ic, seed=7)
        sharpes.append(simulate(preds, CostModel()).metrics()["net"]["sharpe"])

    assert sharpes[0] < sharpes[1] < sharpes[2], (
        f"net Sharpe {sharpes} is not increasing in the planted edge; the "
        f"simulator cannot see an edge that is really there")
    assert sharpes[0] < 0.4, "a zero-edge prediction should not look skilful"


def test_a_planted_edge_survives_costs_only_above_the_break_even():
    """
    Measured: at a planted IC of ~0.02 the net Sharpe is roughly zero, and the
    break-even calculation independently puts the required IC near 0.005-0.03
    depending on impact. The two must agree in ORDER OF MAGNITUDE, or one of
    them is wrong.
    """
    be = break_even_ic(CostModel(), spread_per_ic=0.35, turnover=0.80)
    assert 0.001 < be["break_even_rank_ic"] < 0.05, (
        f"break-even IC {be['break_even_rank_ic']:.4f} is implausible for a "
        f"0.22% round trip at 0.8 turnover")
    # And it must RISE with impact cost, or the sweep is not doing anything.
    worse = break_even_ic(CostModel(impact=0.0050), spread_per_ic=0.35,
                          turnover=0.80)
    assert worse["break_even_rank_ic"] > be["break_even_rank_ic"] * 3


# ── 2. Costs on actual turnover ───────────────────────────────────────────────

def test_costs_are_charged_on_the_holdings_that_actually_changed():
    """
    A name in the top quintile on two consecutive rebalances is HELD, not sold
    and re-bought. Charging it twice would overstate the drag by the overlap,
    which for a 30-session ordering is most of the book.
    """
    truth = _panel(n_rebalances=12, n_tickers=40, seed=3)
    # A prediction that is CONSTANT PER TICKER: the same names top the book at
    # every rebalance, so after the first there is nothing to trade.
    rng = np.random.default_rng(1)
    fixed = {t: float(rng.normal()) for t in truth["ticker"].unique()}
    preds = truth.assign(y_pred=truth["ticker"].map(fixed))

    book = simulate(preds, CostModel())
    # TURNOVER IS IN ROUND TRIPS, because that is the unit costs are charged in.
    # Opening the book buys 100% and sells nothing, which is HALF a round trip,
    # so 0.5 is correct and 1.0 would be double-charging the first rebalance.
    assert book.turnover[0] == pytest.approx(0.5), (
        "opening the book is a one-way purchase: half a round trip")
    assert max(book.turnover[1:]) == pytest.approx(0.0, abs=1e-12), (
        "an unchanged book was charged turnover; costs are being applied to an "
        "assumed 100% rather than to what actually traded")

    # With no trading after the first rebalance, gross and net converge.
    later_gap = [g - n for g, n in zip(book.gross_returns[1:], book.net_returns[1:])]
    assert max(later_gap) == pytest.approx(0.0, abs=1e-12)


def test_the_cost_model_matches_the_measured_indian_schedule():
    """
    Zerodha equity-delivery, NSE, verified 2026-09-04. STT is the dominant term
    and applies to BOTH sides, which is what most cost assumptions get wrong.
    """
    c = CostModel()
    assert c.stt_buy == c.stt_sell == 0.001, "STT applies to both sides"
    assert c.round_trip == pytest.approx(0.002225, abs=5e-6), (
        f"round trip {c.round_trip:.6f} does not match the measured schedule")
    # Impact is NOT part of the fee schedule and must default to zero so a
    # caller has to choose and sweep it.
    assert c.impact == 0.0
    assert CostModel(impact=0.0025).round_trip > c.round_trip


# ── 3. No ordering, no portfolio ──────────────────────────────────────────────

def test_a_constant_prediction_earns_no_book_at_all():
    """
    THE ALPHABETICAL-ALPHA LANDMINE. pandas' sort is stable, so tied predictions
    keep their incoming (date, ticker) order. That once gave `zero`,
    `train_mean` and `majority` a long-short spread of +0.01744 at t +1.19 —
    entirely the return of the alphabetically-first fifth of the universe.
    """
    truth = _panel(n_rebalances=20)
    book = simulate(truth, CostModel())        # y_pred is 0.0 everywhere

    assert book.n_rebalances == 0, (
        "a prediction with no ordering was given a portfolio")
    assert book.n_no_ordering > 0, "the degenerate dates were not counted"
    assert book.metrics()["n_rebalances"] == 0


def test_the_simulator_trades_exactly_what_the_report_scored():
    """
    Both consume `rebalance_books`, so the names traded are the names scored.
    If each did its own sorting and tie-breaking they would eventually diverge
    with nothing to see, because both would still render.
    """
    preds = synthetic_predictions(_panel(n_rebalances=20), 0.10, seed=5)
    xs = cross_sectional_report(preds, rebalance_every=HORIZON_SESSIONS)
    book = simulate(preds, CostModel(), long_only=False)

    assert book.n_rebalances == xs["n_rebalances"]
    # The long-short gross mean IS the report's spread, by construction.
    assert float(np.mean(book.gross_returns)) == pytest.approx(
        xs["long_short_spread"], rel=1e-9)


# ── 4. The rebalance grid ─────────────────────────────────────────────────────

def test_statistics_come_off_the_non_overlapping_grid():
    """
    Consecutive dates share 29 of their 30 forward sessions. Sampling daily
    would multiply the observation count ~30x and inflate every t-statistic by
    roughly its square root — the trap `audit_benchmarks.py` recorded.
    """
    preds = synthetic_predictions(_panel(n_rebalances=30), 0.10, seed=5)
    book = simulate(preds, CostModel(), rebalance_every=HORIZON_SESSIONS)
    assert book.n_rebalances <= 30, (
        "more books than rebalances: the simulator is sampling the daily grid")

    daily = simulate(preds, CostModel(), rebalance_every=1)
    assert daily.n_rebalances > 10 * book.n_rebalances, (
        "the guard case is unreachable, so this test proves nothing")

    assert REBALANCES_PER_YEAR == pytest.approx(252 / HORIZON_SESSIONS)


# ── 5. The deflation's units ──────────────────────────────────────────────────

def test_the_deflation_is_expressed_in_the_spread_of_trial_sharpes():
    """
    THE BUG THIS FOUND. `deflated_sharpe_note` had no caller and no test, and
    its expected-maximum term was left in raw Sharpe units — putting the hurdle
    at +1.98 for 24 trials, which no honest per-rebalance Sharpe reaches. Every
    strategy was therefore "deflated" into the floor for a units reason rather
    than a statistical one.
    """
    observed, n_obs = 0.465, 64

    wrong = deflated_sharpe_note(24, observed, n_obs, sharpe_std=1.0)
    right = deflated_sharpe_note(24, observed, n_obs, sharpe_std=0.199)

    assert wrong["expected_max_sharpe_under_null"] > 1.9
    assert right["expected_max_sharpe_under_null"] < 0.5
    assert right["deflated_statistic"] > wrong["deflated_statistic"], (
        "scaling by the trial spread must RAISE the statistic; if it does not, "
        "the hurdle is not being scaled at all")

    # More trials is a higher hurdle, always.
    assert (deflated_sharpe_note(40, observed, n_obs, sharpe_std=0.199)
            ["expected_max_sharpe_under_null"] >
            right["expected_max_sharpe_under_null"])

    # A nonsensical spread is refused rather than silently treated as 1.0.
    assert "note" in deflated_sharpe_note(24, observed, n_obs, sharpe_std=0.0)
    assert "deflated_statistic" not in deflated_sharpe_note(
        24, observed, n_obs, sharpe_std=-1.0)


def test_this_is_a_measurement_and_publishes_no_book():
    """
    P0 removed the leaderboard and the portfolio from the PRODUCT. This module
    is a measurement and must not quietly reinstate them: nothing it returns is
    a current or forward-looking holding, and every book it builds is dated and
    scored against what actually happened next.
    """
    preds = synthetic_predictions(_panel(n_rebalances=10), 0.05, seed=2)
    book = simulate(preds, CostModel())

    assert set(vars(book)) == {
        "name", "n_rebalances", "gross_returns", "net_returns", "turnover",
        "dates", "n_no_ordering"}, (
        "BookResult grew a field; if it now carries holdings, this stopped "
        "being a measurement and became the thing P0 deleted")
    assert len(book.dates) == book.n_rebalances
    assert all(isinstance(d, str) for d in book.dates)
