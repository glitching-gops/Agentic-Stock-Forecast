"""
Phase 5 — stripping the beta channel, and the guards that make the result
readable rather than merely produced.

The measurement these test is worth nothing without the guards: a null from a
broken neutraliser is indistinguishable from a null from a working one, which is
the `series_zero` lesson this project has already paid for once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.neutralise import (
    CALIBRATION_MAX_T,
    CALIBRATION_TOLERANCE,
    NeutralisationRefused,
    _residualise_within_date,
    calibration_gate,
    fold_null_band,
    neutralise,
    residual_report,
)
from pipeline.portfolio import CostModel, _turn, simulate_hedged

N_NAMES = 24
N_DATES = 12


def _panel(seed: int = 0, beta_strength: float = 1.0, alpha_strength: float = 0.0):
    """
    A panel whose target is BUILT from a beta channel plus an optional
    company-specific one, so the right answer is known before anything runs.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(N_NAMES)]
    betas = np.linspace(0.5, 1.8, N_NAMES)
    alpha = rng.normal(size=N_NAMES)          # persistent company-specific edge

    rows = []
    for d in range(N_DATES):
        mkt = 0.02 + 0.01 * rng.normal()
        noise = rng.normal(scale=0.01, size=N_NAMES)
        y = beta_strength * betas * mkt + alpha_strength * alpha * 0.02 + noise
        for i, t in enumerate(tickers):
            rows.append({"date": f"2024-01-{d + 1:02d}", "ticker": t,
                         "y_true": y[i], "beta": betas[i], "alpha": alpha[i],
                         "mkt": mkt, "fold": 0})
    return pd.DataFrame(rows)


def _beta_frame(panel: pd.DataFrame, scale: float = 1.0, shift: float = 0.0):
    """The regressor, optionally affinely transformed."""
    out = panel[["date", "ticker", "beta"]].copy()
    out["beta_basis"] = shift + scale * out["beta"]
    return out


# ---------------------------------------------------------------- the residual

def test_residualising_removes_the_beta_ordering_it_was_given():
    p = _panel(beta_strength=1.0, alpha_strength=0.0)
    p["y_pred"] = p["beta"]                       # a pure beta sort
    r = neutralise(p, _beta_frame(p))
    rep = residual_report(r.frame, rebalance_every=1)

    assert abs(rep["residual_rank_ic"]) < 0.05, (
        "a pure beta ordering must lose its rank IC against a beta-neutralised "
        f"target; kept {rep['residual_rank_ic']:+.4f}")
    assert r.r2_mean > 0.3, "the planted beta channel should be real in this panel"


def test_a_company_specific_edge_SURVIVES_neutralisation():
    """
    The other half, and the one that stops the module being a shredder. If
    neutralisation destroyed every ordering it would 'prove' a null on any input.
    """
    p = _panel(beta_strength=1.0, alpha_strength=3.0)
    p["y_pred"] = p["alpha"]
    rep = residual_report(neutralise(p, _beta_frame(p)).frame, rebalance_every=1)

    assert rep["residual_rank_ic"] > 0.3, (
        "a real company-specific edge must survive beta neutralisation; got "
        f"{rep['residual_rank_ic']:+.4f}")


@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (0.02, 0.0),
                                         (-3.0, 1.5), (1e4, -7.0)])
def test_the_residual_is_invariant_to_an_affine_rescale_of_the_regressor(scale, shift):
    """
    THE CLAIM THE PRIMARY BASIS RESTS ON. `beta_market` predicts
    `beta_i * mu_market`, and mu_market is constant within a date — so
    residualising on the floor's prediction is identical to residualising on beta
    itself. If that were false, §1 would be measuring something other than what
    its heading says, including for a NEGATIVE mu.
    """
    p = _panel()
    p["y_pred"] = p["alpha"]
    base = neutralise(p, _beta_frame(p)).frame.sort_values(["date", "ticker"])
    other = neutralise(p, _beta_frame(p, scale, shift)).frame.sort_values(
        ["date", "ticker"])

    np.testing.assert_allclose(base["y_true"].to_numpy(),
                               other["y_true"].to_numpy(), atol=1e-10)


def test_a_constant_regressor_is_refused_rather_than_passed_through():
    """
    THE QUIET FAILURE THIS MODULE EXISTS TO PREVENT. mu_market of zero makes the
    regressor constant; returning the target unchanged would put an
    UNNEUTRALISED number under a neutralised heading and nothing would raise.
    """
    p = _panel()
    p["y_pred"] = p["alpha"]
    flat = p[["date", "ticker"]].copy()
    flat["beta_basis"] = 0.0

    with pytest.raises(NeutralisationRefused, match="constant"):
        neutralise(p, flat)


def test_one_dead_date_is_counted_not_silently_dropped():
    p = _panel()
    p["y_pred"] = p["alpha"]
    bf = _beta_frame(p)
    bf.loc[bf["date"] == "2024-01-01", "beta_basis"] = 0.0

    r = neutralise(p, bf)
    assert r.n_dates_refused == 1
    assert r.n_dates == N_DATES - 1
    assert any("refused" in n for n in r.notes)


def test_residualise_reports_nan_not_the_input_when_it_cannot_fit():
    resid, r2 = _residualise_within_date(np.arange(5.0), np.zeros(5))
    assert np.isnan(resid).all(), "a failed fit must not return the input"
    assert np.isnan(r2)


# ------------------------------------------------------- what may be read off

def test_the_residual_report_refuses_to_produce_an_MAE():
    """
    A residual has strictly smaller variance than its target, so an MAE against
    it looks like a large improvement and means nothing. Computing one and
    captioning it is the `daily_IC`-beside-`reb_t` error; not computing it is the
    only version that cannot be misquoted.
    """
    p = _panel()
    p["y_pred"] = p["alpha"]
    rep = residual_report(neutralise(p, _beta_frame(p)).frame, rebalance_every=1)

    forbidden = {"mae", "mae_naive_zero", "top_quintile_return",
                 "alpha_vs_equal_weight", "long_short_spread"}
    assert not forbidden & set(rep), (
        f"return-space statistics are meaningless against a residual: {rep}")
    assert {"residual_rank_ic", "residual_ic_t"} <= set(rep)


# ------------------------------------------------------------ the gate itself

def test_the_calibration_gate_passes_when_the_regressor_is_actually_beta():
    p = _panel()
    p["y_pred"] = p["beta"] * 0.02              # beta_market's own prediction
    gate = calibration_gate(p, _beta_frame(p), rebalance_every=1)

    assert gate["passed"], gate["note"]
    # A LINEAR residualisation does not exactly zero a RANK correlation, so the
    # right assertion is that the leftover is indistinguishable from noise —
    # not that it is inside a fixed band, which mis-sizes itself by sample.
    assert abs(gate["observed_residual_t"]) < CALIBRATION_MAX_T


def test_the_calibration_gate_FAILS_when_the_regressor_is_misaligned():
    """
    The gate has to be able to fail, or it is decoration. Merging the betas onto
    the wrong tickers is the realistic version of the mistake — a join key that
    is right in shape and wrong in content, which nothing else here would catch.
    """
    p = _panel()
    p["y_pred"] = p["beta"] * 0.02
    scrambled = _beta_frame(p)
    rng = np.random.default_rng(1)
    scrambled["beta_basis"] = rng.permutation(scrambled["beta_basis"].to_numpy())

    gate = calibration_gate(p, scrambled, rebalance_every=1)
    assert not gate["passed"]
    assert "not beta" in gate["note"]
    # A REAL MISALIGNMENT FAILS BOTH HALVES OF THE RULE, which is what makes the
    # "or" safe: it is not passing because one loose test carried it.
    assert abs(gate["observed_residual_t"]) >= CALIBRATION_MAX_T
    assert abs(gate["observed_residual_ic"]) > CALIBRATION_TOLERANCE


# ------------------------------------------------------------ the null band

def test_the_permutation_null_centres_on_zero_in_every_fold():
    p = _panel()
    p["y_pred"] = p["alpha"]
    bands = fold_null_band(p, n_draws=60, rebalance_every=1, seed=3)

    assert bands, "expected at least one fold"
    for fold, b in bands.items():
        assert abs(b["null_mean"]) < 0.05, (
            f"fold {fold} null is off-centre at {b['null_mean']:+.4f}; the "
            f"permutation is probably not within-date")


def test_a_thinner_cross_section_earns_a_WIDER_null_band():
    """
    THE MECHANISM THE PHASE IS TESTING FOR. Three retired results lived entirely
    in fold 0, which holds fewer names per date — so its rank IC is a noisier
    statistic and a given value there is less surprising than the same value
    late. If breadth did not widen the band, that explanation would be dead.
    """
    p = _panel()
    p["y_pred"] = p["alpha"]
    thin = p[p["ticker"] < "T12"].copy()
    thin["fold"] = 1

    wide_band = fold_null_band(p, n_draws=120, rebalance_every=1, seed=5)["0"]
    thin_band = fold_null_band(thin, n_draws=120, rebalance_every=1, seed=5)["1"]

    assert thin_band["median_names_per_date"] < wide_band["median_names_per_date"]
    assert thin_band["null_sd"] > wide_band["null_sd"], (
        f"thin fold sd {thin_band['null_sd']:.4f} should exceed wide "
        f"{wide_band['null_sd']:.4f}")


# ----------------------------------------------------------- the hedged book

def test_the_hedged_book_removes_the_market_channel():
    """
    The money-space twin of the calibration gate: a pure beta ordering, held
    beta-neutral, should return ~0 gross because it is short exactly what it is
    long.
    """
    p = _panel(beta_strength=1.0, alpha_strength=0.0)
    p["y_pred"] = p["beta"]
    beta = p[["date", "ticker", "beta"]]

    hedged = simulate_hedged(p, beta, CostModel(), rebalance_every=1)
    plain = np.mean(hedged.gross_returns)

    assert hedged.n_rebalances >= 3
    assert abs(plain) < 0.01, (
        f"a beta-neutral book on a pure beta panel should be flat; got {plain:+.4f}")


def test_the_hedge_leg_is_charged_and_not_free():
    """
    The hedge notional changes between rebalances and those trades cost money.
    Omitting them understates the drag with nothing visible, because the book
    still renders.
    """
    p = _panel(alpha_strength=2.0)
    rng = np.random.default_rng(9)
    p["y_pred"] = rng.normal(size=len(p))        # churns the book every date
    beta = p[["date", "ticker", "beta"]]

    hedged = simulate_hedged(p, beta, CostModel(), rebalance_every=1)
    gross = np.mean(hedged.gross_returns)
    net = np.mean(hedged.net_returns)

    assert net < gross, "costs were not charged at all"
    assert np.mean(hedged.turnover) > 0.0


def test_the_hedge_leg_is_charged_EVEN_WHEN_THE_LONG_BOOK_NEVER_TRADES():
    """
    ISOLATES THE HEDGE LEG, because the previous test cannot. When the ordering
    churns, long-leg turnover dominates and a completely free hedge would still
    leave net below gross — so that test passes against the defect it looks like
    it covers. Here the long book holds the SAME names at every rebalance (long
    turnover 0 after the open) while the betas drift, so any turnover beyond the
    opening trade can only be the hedge.
    """
    p = _panel(alpha_strength=2.0)
    p["y_pred"] = p["alpha"]                     # same ordering every date
    beta = p[["date", "ticker", "beta"]].copy()
    # Drift the hedge ratio over time without touching which names are held.
    day_index = beta["date"].str.slice(8).astype(int)
    beta["beta"] = beta["beta"] * (1.0 + 0.25 * day_index)

    hedged = simulate_hedged(p, beta, CostModel(), rebalance_every=1)
    after_open = hedged.turnover[1:]

    assert all(t > 0 for t in after_open), (
        "the long book never trades after the open, so every non-zero turnover "
        f"here is the hedge leg — got {after_open}")
    assert np.mean(hedged.net_returns) < np.mean(hedged.gross_returns)


def test_simulate_hedged_refuses_a_scaled_prediction_as_a_beta():
    """
    A hedge RATIO is not scale-free. Handing it `beta_market`'s prediction
    (beta * mu_market, ~0.02x) would short 2% of the right amount and report a
    plausible number, so the column name is enforced rather than assumed.
    """
    p = _panel()
    p["y_pred"] = p["alpha"]
    with pytest.raises(ValueError, match="beta"):
        simulate_hedged(p, p[["date", "ticker"]].assign(beta_basis=1.0))


# --------------------------------------------------------- the shared helper

def test_both_simulators_share_one_turnover_rule():
    """
    `_turn` is at module scope so `simulate` and `simulate_hedged` cannot drift
    on what a round trip is. Two copies would both render and only disagree in
    the cost column.
    """
    assert _turn(set("ab"), set()) == 0.5, "opening a book is half a round trip"
    assert _turn(set("ab"), set("ab")) == 0.0
    assert _turn(set("ab"), set("cd")) == 1.0


def test_an_uncomputable_gate_says_so_instead_of_blaming_the_regressor():
    """
    Failing closed is right; blaming the wrong cause is not. Too few rebalances
    and a misaligned regressor need opposite remedies, and a gate that reports
    the second when it means the first sends the reader to rewrite correct code.
    """
    p = _panel()
    p["y_pred"] = p["beta"] * 0.02
    gate = calibration_gate(p, _beta_frame(p), rebalance_every=999)

    assert not gate["passed"], "must still fail closed"
    assert gate["computed"] is False
    assert "not beta" not in gate["note"]
    assert "could not be computed" in gate["note"]
