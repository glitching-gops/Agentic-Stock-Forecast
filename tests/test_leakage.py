"""
Regression tests for the leakage defects found in the Phase 0 audit.

Each test corresponds to a finding and fails if that defect is reintroduced.
These are the tests whose absence let F1-F3 ship and reach a public dashboard.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation import (
    PurgedWalkForward,
    assert_no_leakage,
    compute_metrics,
    effective_sample_size,
    majority_hit_rate,
)

HORIZON = 30


# ── F3: purging and embargo ───────────────────────────────────────────────────

def test_folds_leave_a_purge_gap_of_at_least_the_horizon():
    """Training must end at least `horizon + embargo` rows before the test window."""
    splitter = PurgedWalkForward(n_folds=5, horizon=HORIZON,
                                 embargo=HORIZON, min_train=250)
    n = 1200
    folds = list(splitter.split(n))
    assert folds, "splitter produced no folds"

    for train_idx, test_idx in folds:
        gap = int(test_idx.min()) - int(train_idx.max()) - 1
        assert gap >= HORIZON + HORIZON, (
            f"gap of {gap} rows is smaller than horizon+embargo "
            f"({HORIZON + HORIZON}); a 30-session label would straddle the split"
        )


def test_train_indices_always_precede_test_indices():
    splitter = PurgedWalkForward(n_folds=6, horizon=HORIZON, min_train=300)
    for train_idx, test_idx in splitter.split(1500):
        assert train_idx.max() < test_idx.min()


def test_assert_no_leakage_catches_a_contiguous_split():
    """
    The exact defect from pipeline/model.py: an 85/15 contiguous split whose
    last training label lands inside the test window.
    """
    dates = [f"d{i:04d}" for i in range(500)]
    split = int(500 * 0.85)
    train, test = dates[:split], dates[split:]

    with pytest.raises(AssertionError, match="Label leakage"):
        assert_no_leakage(train, test, HORIZON, dates)


def test_assert_no_leakage_accepts_a_purged_split():
    dates = [f"d{i:04d}" for i in range(500)]
    train = dates[:300]
    test = dates[300 + HORIZON + HORIZON:]
    assert_no_leakage(train, test, HORIZON, dates)   # must not raise


# ── F2: nested tuning ─────────────────────────────────────────────────────────

def test_tuner_never_sees_the_rows_its_own_fold_is_scored_on():
    """
    F2 was tuning on the full labelled set and then reporting a slice of it as
    held out. The contract is per fold: the tuner receives that fold's training
    slice only, separated from that fold's test slice by the purge gap.

    Reusing an EARLIER fold's test rows as a LATER fold's training rows is
    correct rolling-origin behaviour and is not leakage — each model is still
    fitted only on data preceding its own test window.
    """
    from pipeline.evaluation import walk_forward
    from sklearn.linear_model import Ridge

    n = 900
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.normal(size=n))
    dates = [f"d{i:04d}" for i in range(n)]

    train_rows_per_call: list[set] = []

    def spying_tuner(X_train, y_train):
        train_rows_per_call.append(set(X_train.index))
        return {}

    result = walk_forward(
        X=X, y=y, dates=dates,
        model_factory=lambda: Ridge(alpha=1.0),
        splitter=PurgedWalkForward(n_folds=4, horizon=HORIZON, min_train=300),
        tuner=spying_tuner,
    )

    assert result.n_folds_run > 0
    assert len(train_rows_per_call) == result.n_folds_run, "tuner call count mismatch"

    for position, fold_id in enumerate(sorted(result.predictions["fold"].unique())):
        fold_tests = result.predictions[result.predictions["fold"] == fold_id]
        test_rows = {int(d[1:]) for d in fold_tests["date"]}
        train_rows = train_rows_per_call[position]

        overlap = train_rows & test_rows
        assert not overlap, (
            f"fold {fold_id}: tuner saw {len(overlap)} of the rows this fold is "
            f"scored on"
        )

        gap = min(test_rows) - max(train_rows) - 1
        assert gap >= HORIZON, (
            f"fold {fold_id}: only {gap} rows between the tuner's training data "
            f"and its test window; a {HORIZON}-session label spans the gap"
        )


# ── Overlapping-label t-statistic correction ──────────────────────────────────

def test_effective_sample_size_discounts_overlap():
    assert effective_sample_size(3000, 30) == pytest.approx(100.0)
    assert effective_sample_size(100, 1) == pytest.approx(100.0)
    assert effective_sample_size(5, 30) == pytest.approx(1.0)


def test_t_statistic_uses_effective_sample_size():
    """
    A t-statistic computed on raw row count would be ~sqrt(30) times larger.
    This is what turned a rank IC of 0.18 into an apparent t of 7.9.
    """
    rng = np.random.default_rng(1)
    n = 2000
    y_true = rng.normal(size=n)
    y_pred = y_true * 0.2 + rng.normal(size=n)

    m = compute_metrics(y_true, y_pred, horizon=30)
    naive_t = m["rank_ic"] * np.sqrt(n - 1)

    assert m["rank_ic_t"] < naive_t / 4, (
        "t-statistic does not appear to discount overlapping labels"
    )
    assert m["n_effective"] == pytest.approx(n / 30, rel=0.01)


# ── Baseline reporting ────────────────────────────────────────────────────────

def test_metrics_always_carry_a_majority_baseline():
    """
    A directional accuracy figure without its baseline is uninterpretable —
    the previous system reported 85% while the majority baseline was ~59%.
    """
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=500)
    y_pred = rng.normal(size=500)

    m = compute_metrics(y_true, y_pred, horizon=30)
    assert "majority_hit_rate" in m
    assert "mae_naive_zero" in m
    assert "beats_naive_mae" in m


def test_majority_baseline_is_never_below_fifty_percent():
    for p_up in [0.1, 0.3, 0.5, 0.7, 0.95]:
        rng = np.random.default_rng(3)
        y = np.where(rng.random(1000) < p_up, 1.0, -1.0)
        assert majority_hit_rate(y) >= 50.0


def test_a_zero_skill_model_reports_near_zero_ic():
    """Sanity check that the harness cannot manufacture skill from noise."""
    rng = np.random.default_rng(4)
    y_true = rng.normal(size=3000)
    y_pred = rng.normal(size=3000)

    m = compute_metrics(y_true, y_pred, horizon=30)
    assert abs(m["rank_ic"]) < 0.06
    assert abs(m["rank_ic_t"]) < 2.0
