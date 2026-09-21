"""
The within-date standardised training target, and its inverse.

The transform is now the pipeline's default label, so the round trip is
load-bearing in a way it was not when it lived inside one diagnostic tool:
everything the dashboard shows has to come back out of it. Two inverses exist
and only one of them is legitimate at prediction time, which is the thing most
likely to be got wrong later — so it is pinned from both directions here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.label import (
    CS_MEAN,
    CS_SD,
    attach_causal_moments,
    causal_moments,
    cross_sectional_moments,
    inverse_standardise,
    standardise_target,
)
from pipeline.panel import MIN_NAMES_PER_DATE, TARGET


def _panel(n_dates: int = 400, n_tickers: int = 40, seed: int = 7,
           drift: bool = True) -> pd.DataFrame:
    """
    A panel whose cross-sectional mean AND dispersion both move over time —
    which is what makes the moments worth estimating rather than assuming.
    """
    rng = np.random.default_rng(seed)
    dates = [f"D{i:04d}" for i in range(n_dates)]
    scale = np.linspace(0.03, 0.18, n_dates) if drift else np.full(n_dates, 0.1)
    centre = np.linspace(-0.04, 0.06, n_dates) if drift else np.zeros(n_dates)
    rows = []
    for i, d in enumerate(dates):
        for j in range(n_tickers):
            rows.append({"date": d, "ticker": f"T{j:02d}.NS",
                         TARGET: centre[i] + scale[i] * rng.normal()})
    return pd.DataFrame(rows)


# ── the transform ─────────────────────────────────────────────────────────────

def test_the_standardised_label_is_a_z_score_within_each_date():
    out = standardise_target(_panel())
    by_date = out.groupby("date")[TARGET]
    assert np.allclose(by_date.mean().to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(by_date.std().to_numpy(), 1.0, atol=1e-12)


def test_standardising_changes_no_within_date_ranking():
    """
    The whole justification for calling this a defect fix rather than a new
    model: a positive affine map inside each date moves no rank, so the
    per-date rank IC of any fixed ordering is untouched. What changes is the
    SCALE the loss — and therefore `gamma` — is measured against.
    """
    from pipeline.evaluation import rank_ic

    panel = _panel()
    rng = np.random.default_rng(11)
    pred = rng.normal(size=len(panel))
    out = standardise_target(panel)

    for d in panel["date"].unique()[:40]:
        m = (panel["date"] == d).to_numpy()
        assert rank_ic(panel[TARGET].to_numpy()[m], pred[m]) == pytest.approx(
            rank_ic(out[TARGET].to_numpy()[m], pred[m]))


def test_the_pooled_dispersion_rises_to_roughly_one():
    """
    The measured reason the fix works. `gamma` is a minimum loss reduction in
    the loss's own units, so what matters to the tuner is the label's spread.
    """
    panel = _panel()
    raw = pd.to_numeric(panel[TARGET], errors="coerce")
    std = pd.to_numeric(standardise_target(panel)[TARGET], errors="coerce")

    assert raw.std() < 0.2
    assert std.std() == pytest.approx(1.0, abs=0.05)
    assert std.std() / raw.std() > 5


# ── the inverse ───────────────────────────────────────────────────────────────

def test_the_round_trip_with_realised_moments_is_exact():
    panel = _panel()
    raw = pd.to_numeric(panel[TARGET], errors="coerce").to_numpy(dtype=float)

    out = standardise_target(panel)
    back = inverse_standardise(out[TARGET], out[CS_MEAN], out[CS_SD])

    ok = np.isfinite(back) & np.isfinite(raw)
    assert ok.sum() == len(raw)
    assert np.max(np.abs(back[ok] - raw[ok])) < 1e-12


def test_the_moments_travel_with_the_standardised_label():
    """A standardised label with no moments beside it is a one-way door."""
    out = standardise_target(_panel())
    assert CS_MEAN in out.columns and CS_SD in out.columns
    assert out[[CS_MEAN, CS_SD]].notna().all().all()
    # And they are the date's own moments, not the panel's.
    moments = cross_sectional_moments(_panel())
    assert len(moments) == out["date"].nunique()
    assert moments[CS_SD].std() > 0, "a fixture with constant dispersion proves nothing"


def test_inverting_a_z_score_of_zero_returns_the_cross_sectional_mean():
    """The sanity case, and the one a reader can check by hand."""
    assert inverse_standardise(0.0, 0.05, 0.10) == pytest.approx(0.05)
    assert inverse_standardise(1.0, 0.05, 0.10) == pytest.approx(0.15)
    assert inverse_standardise(-2.0, 0.0, 0.10) == pytest.approx(-0.20)


# ── the causal inverse, which is the one that matters live ────────────────────

def test_the_causal_moments_are_blind_to_the_window_they_invert():
    """
    THE LEAKAGE CONTRACT, and the single most important test in this file.

    The label at date t spans [t, t+h], so its cross-sectional moments are not
    knowable until t+h. Inverting date t's FORECAST with date t's own realised
    moments uses the answer — it would read as a large improvement in both MAE
    and coverage, and it is F1 in a new place.

    Corrupting every moment from the cut day onward must leave every causal
    estimate up to the cut unchanged.
    """
    horizon = 30
    panel = _panel(n_dates=500)
    grid = sorted(panel["date"].unique())
    cut = grid[350]

    clean = causal_moments(panel, horizon=horizon)

    poisoned = panel.copy()
    after = poisoned["date"] >= cut
    poisoned.loc[after, TARGET] = 99.0
    dirty = causal_moments(poisoned, horizon=horizon)

    merged = clean.merge(dirty, on="date", suffixes=("_clean", "_dirty"))
    upto = merged[merged["date"] <= cut]
    for col in (CS_MEAN, CS_SD):
        a = upto[f"{col}_clean"].to_numpy(dtype=float)
        b = upto[f"{col}_dirty"].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        assert both.sum() > 100, "too few comparable dates; the test would be weak"
        assert np.array_equal(a[both], b[both]), (
            f"{col} at or before the cut moved when only the FUTURE was "
            f"corrupted; the causal estimate is not causal")

    # And it must actually differ somewhere after the cut, or the corruption
    # never reached the estimator and the test proves nothing.
    later = merged[merged["date"] > cut]
    assert not np.allclose(
        later[f"{CS_MEAN}_clean"].fillna(0).to_numpy(),
        later[f"{CS_MEAN}_dirty"].fillna(0).to_numpy())


def test_the_causal_moments_lag_by_at_least_the_horizon():
    """
    Measured on the dates themselves rather than asserted from the code: the
    first date with a usable estimate must sit at least `horizon` dates after
    the first date with a realised moment.
    """
    horizon = 30
    panel = _panel(n_dates=500)
    est = causal_moments(panel, horizon=horizon, lookback=252, min_dates=20)
    grid = sorted(panel["date"].unique())

    first_usable = est.loc[est[CS_SD].notna(), "date"].min()
    assert grid.index(first_usable) >= horizon + 20 - 1


def test_the_causal_inverse_is_approximate_and_says_so():
    """
    It is an estimate, so the round trip through it does NOT close. That is not
    a defect — it is the cost of the transform, and the conformal layer is what
    turns it into a measured interval rather than an assumed one.
    """
    panel = _panel(n_dates=500)
    raw = pd.to_numeric(panel[TARGET], errors="coerce").to_numpy(dtype=float)
    std = standardise_target(panel)
    est = attach_causal_moments(panel)

    back = inverse_standardise(std[TARGET], est[CS_MEAN], est[CS_SD])
    ok = np.isfinite(back) & np.isfinite(raw)
    assert ok.sum() > 1000

    err = np.abs(back[ok] - raw[ok])
    assert err.max() > 1e-6, (
        "the causal inverse closed exactly, which means it used the realised "
        "moments — check the lag")
    # It is an estimate of the right thing, though: the error must be far
    # smaller than the label's own spread.
    assert err.mean() < np.std(raw[ok])


def test_attaching_causal_moments_replaces_rather_than_duplicates():
    panel = standardise_target(_panel())
    out = attach_causal_moments(panel)
    assert list(out.columns).count(CS_MEAN) == 1
    assert list(out.columns).count(CS_SD) == 1
    assert len(out) == len(panel)


# ── refusals ──────────────────────────────────────────────────────────────────

def test_an_undefined_z_score_is_nan_and_never_infinite():
    """
    The standardiser carries no explicit guard for a zero denominator, because
    mutation testing showed no input could reach one: 0/0 is already NaN and a
    non-finite label poisons its date's mean so `inf - inf` is NaN too. This
    test is what stands in place of those guards — if a future pandas changes
    the semantics, an infinite z-score would sail into the model as an extreme
    ordering, and it must fail here instead.
    """
    from pipeline.panel import MIN_NAMES_PER_DATE

    n = MIN_NAMES_PER_DATE + 2
    for name, values in (
        ("constant", [0.1] * n),
        ("constant, awkward float", [1.0 / 3.0] * n),
        ("carries an inf", [0.01 * i for i in range(n - 1)] + [np.inf]),
        ("carries a nan", [0.01 * i for i in range(n - 1)] + [np.nan]),
    ):
        panel = pd.DataFrame({
            "date": ["d0"] * n,
            "ticker": [f"T{j:02d}" for j in range(n)],
            TARGET: values,
        })
        z = standardise_target(panel)[TARGET].to_numpy(dtype=float)
        assert not np.isinf(z).any(), f"{name}: produced an infinite z-score"
        if name.startswith("constant"):
            assert np.isnan(z).all(), f"{name}: fabricated an ordering"


def test_a_thin_or_constant_cross_section_earns_no_z_score():
    rng = np.random.default_rng(13)
    thin, wide = MIN_NAMES_PER_DATE - 1, MIN_NAMES_PER_DATE + 5
    panel = pd.DataFrame({
        "date": (["thin"] * thin + ["flat"] * wide + ["ok"] * wide),
        "ticker": ([f"T{j:02d}" for j in range(thin)]
                   + [f"T{j:02d}" for j in range(wide)] * 2),
        TARGET: (list(rng.normal(0, 0.05, thin)) + [0.02] * wide
                 + list(rng.normal(0, 0.05, wide))),
    })
    out = standardise_target(panel)
    assert out[out["date"] == "thin"][TARGET].isna().all()
    assert out[out["date"] == "flat"][TARGET].isna().all()
    assert out[out["date"] == "ok"][TARGET].notna().all()
