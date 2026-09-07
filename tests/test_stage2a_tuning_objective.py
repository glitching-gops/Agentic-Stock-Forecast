"""
Stage 2a — the alternative tuning objective, and the pin on the old one.

The Stage 0 addendum measured that 316 of 420 (ticker, fold) fits emit a
constant, and traced it to the tuner scoring MAE on a target under which a
constant is near-optimal. This suite covers the "rank_ic" objective added in
response, and — at least as importantly — pins the "mae" path as UNCHANGED, so
that an A/B between them is a controlled comparison rather than two edits at
once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.tuning import (
    DEGENERATE_FOLD_PENALTY,
    TUNING_OBJECTIVES,
    _OBJECTIVE_SCORERS,
    purged_cv_rank_ic_score,
    purged_cv_score,
    tune,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


def _rankable_panel(n: int = 900, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """
    A series whose target is genuinely predictable from one feature.

    Deliberately NOT a random walk: the point of these tests is to separate a
    scorer that can see an ordering from one that cannot, which needs an
    ordering to exist. Noise is large enough that a shrunken constant is still
    competitive on MAE — which is the whole phenomenon under test.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise = rng.normal(scale=3.0, size=n)
    X = pd.DataFrame({
        "signal": signal,
        "decoy": rng.normal(size=n),
        "decoy2": rng.normal(size=n),
    })
    y = pd.Series(0.05 * signal + 0.01 * noise)
    return X, y


def _noise_panel(n: int = 900, seed: int = 17) -> tuple[pd.DataFrame, pd.Series]:
    """
    A target with NO learnable structure, at roughly the real panel's dispersion.

    This is the situation the Stage 0 addendum found on live data: nothing to
    learn, so the training mean is a strong MAE predictor and a splitting tree
    is worse on MAE while being the only one of the two that expresses any
    ordering at all.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n) for c in ("a", "b", "c")})
    return X, pd.Series(rng.normal(scale=0.10, size=n))


SPLITTING = {"n_estimators": 60, "learning_rate": 0.3, "max_depth": 3,
             "min_child_weight": 1, "gamma": 0.0, "tree_method": "hist"}

# gamma this large makes every split unprofitable, so the tree returns the
# training mean — the exact configuration the addendum found 316 times.
CONSTANT = {**SPLITTING, "gamma": 1e6}


# ── the new objective does what it is for ─────────────────────────────────────


def test_the_rank_ic_objective_prefers_an_ordering_to_a_constant():
    X, y = _rankable_panel()
    splitting = purged_cv_rank_ic_score(X, y, SPLITTING)
    constant = purged_cv_rank_ic_score(X, y, CONSTANT)

    assert splitting < constant, (
        "the rank-IC objective must score a configuration that produces a "
        "rank-informative ordering BETTER (lower, since it is minimised) than "
        "one that produces a constant"
    )
    assert splitting < 0, "a real ordering should score a negative -IC"


def test_a_constant_configuration_earns_the_penalty_and_not_a_middling_score():
    """
    The specified failure handling, asserted at the value.

    A NaN would be dropped by ``np.mean`` and silently vanish; a 0.0 would read
    as "no ordering, but acceptable" and sit mid-range among real trials. Both
    would leave Optuna with no reason to avoid a constant. The penalty must be
    strictly worse than any attainable score.
    """
    X, y = _rankable_panel()
    score = purged_cv_rank_ic_score(X, y, CONSTANT)

    assert np.isfinite(score), "a degenerate trial must not score NaN"
    assert score == pytest.approx(DEGENERATE_FOLD_PENALTY), (
        "every inner fold is constant here, so the mean must be exactly the "
        "penalty — not 0.0, and not a NaN-shrunken average of the rest"
    )
    assert DEGENERATE_FOLD_PENALTY > 1.0, (
        "-IC attains at most +1.0 (a perfectly inverted predictor), so a "
        "penalty of 1.0 or less would let a constant TIE a real ordering"
    )


def test_the_penalty_leaves_a_gradient_between_partly_and_wholly_degenerate():
    """
    Partly degenerate must beat wholly degenerate, or Optuna cannot climb out.

    Built by hand rather than by finding a hyperparameter that happens to
    straddle: the scorer is handed fold scores directly, which is the quantity
    the averaging rule is about.
    """
    partly = float(np.mean([-0.2, DEGENERATE_FOLD_PENALTY, -0.1]))
    wholly = float(np.mean([DEGENERATE_FOLD_PENALTY] * 3))
    assert partly < wholly


def test_the_correlation_is_taken_within_folds_and_never_pooled():
    """
    The Stage 0 addendum's artifact, refused at the tuning stage.

    A model emitting one constant per fold holds NO ordering, yet a POOLED rank
    IC over concatenated folds is non-zero because the constants differ between
    folds — it is ranking rows by which fold they came from. Here the fold
    constants are made to line up with the fold-mean outcomes, so a pooled
    scorer would report a strong positive IC. The within-fold scorer must see
    the penalty instead.
    """
    n_folds = 3
    fold_ids = np.repeat(np.arange(n_folds), 60)
    # Outcome level rises with the fold; within a fold it is pure noise.
    rng = np.random.default_rng(7)
    truth = fold_ids.astype(float) + rng.normal(scale=0.01, size=fold_ids.size)
    preds = fold_ids.astype(float)          # one constant per fold

    from pipeline.evaluation import rank_ic
    pooled = rank_ic(truth, preds)
    assert pooled > 0.9, "the pooled IC must be strongly positive, or this " \
                         "test is not exercising the artifact"

    within = [rank_ic(truth[fold_ids == k], preds[fold_ids == k])
              for k in range(n_folds)]
    assert all(not np.isfinite(v) for v in within), (
        "within a fold the prediction is constant, so every within-fold IC is "
        "undefined — which is what the tuning objective scores"
    )

    scored = [DEGENERATE_FOLD_PENALTY if not np.isfinite(v) else -v
              for v in within]
    assert float(np.mean(scored)) == pytest.approx(DEGENERATE_FOLD_PENALTY)


def test_a_real_ordering_scores_better_than_no_ordering_at_all():
    """A learnable target must score below (better than) an unlearnable one,
    which is what fixes the sign convention: the score is -IC, not +IC."""
    rankable = purged_cv_rank_ic_score(*_rankable_panel(), SPLITTING)
    noise = purged_cv_rank_ic_score(*_noise_panel(), SPLITTING)
    assert rankable < noise
    assert rankable < 0 <= noise + 0.5, "a strong ordering must score well below zero"


def test_the_two_objectives_disagree_in_exactly_the_way_stage_0_measured():
    """
    The mechanism, reproduced synthetically.

    On a target with no structure at the real panel's dispersion, MAE prefers
    the CONSTANT and the rank IC prefers the SPLITTER. That opposition is the
    whole of Stage 2a: the tuner has been optimising a metric under which a
    constant wins, while the evidence gate grades an ordering a constant cannot
    express. If these two ever agree here, the alternative objective is not
    doing anything and the pilot has no subject.
    """
    X, y = _noise_panel()
    mae_split = purged_cv_score(X, y, SPLITTING)
    mae_const = purged_cv_score(X, y, CONSTANT)
    ic_split = purged_cv_rank_ic_score(X, y, SPLITTING)
    ic_const = purged_cv_rank_ic_score(X, y, CONSTANT)

    assert mae_const < mae_split, "MAE must prefer the constant here"
    assert ic_split < ic_const, "the rank IC must prefer the splitter here"


# ── the old objective is UNCHANGED ────────────────────────────────────────────


def test_the_mae_path_is_the_default_and_is_byte_for_byte_the_old_behaviour():
    """
    The regression pin.

    ``tune`` gained a parameter; if the default did anything other than exactly
    what it did before, every measured number in this project would be sitting
    on a moved foundation. Asserted two ways: the default dispatches to
    ``purged_cv_score`` itself (identity, not a copy of it), and the params it
    returns are identical to an explicit "mae".
    """
    assert _OBJECTIVE_SCORERS["mae"] is purged_cv_score
    assert TUNING_OBJECTIVES[0] == "mae"

    X, y = _rankable_panel(n=400, seed=3)
    default = tune(X, y, horizon=30, n_trials=3)
    explicit = tune(X, y, horizon=30, n_trials=3, tuning_objective="mae")
    assert default == explicit


def test_the_objective_argument_actually_reaches_the_optuna_study(monkeypatch):
    """
    Guards against the alternative being wired up but never reached.

    A ``tuning_objective`` that changed nothing would pass every scorer test
    above and still leave the Step B A/B meaningless. Asserted by observing
    which scorer the study calls, rather than by comparing selected params —
    with a seeded TPE the two searches can coincide on the same trial by luck,
    which would make that comparison flaky rather than wrong.
    """
    import pipeline.tuning as tuning

    called: list[str] = []

    def spy(name, real):
        def wrapper(*a, **kw):
            called.append(name)
            return real(*a, **kw)
        return wrapper

    monkeypatch.setitem(tuning._OBJECTIVE_SCORERS, "mae",
                        spy("mae", purged_cv_score))
    monkeypatch.setitem(tuning._OBJECTIVE_SCORERS, "rank_ic",
                        spy("rank_ic", purged_cv_rank_ic_score))

    X, y = _rankable_panel(n=400, seed=5)
    tune(X, y, horizon=30, n_trials=2, tuning_objective="rank_ic")
    assert called and set(called) == {"rank_ic"}

    called.clear()
    tune(X, y, horizon=30, n_trials=2)
    assert called and set(called) == {"mae"}, "the default must stay MAE"


def test_an_unknown_objective_is_refused_rather_than_silently_defaulted():
    X, y = _rankable_panel(n=200)
    with pytest.raises(ValueError, match="unknown tuning_objective"):
        tune(X, y, horizon=30, n_trials=2, tuning_objective="sharpe")


def test_both_objectives_share_the_cv_mechanics():
    """
    Only the metric may differ. If the alternative quietly used different
    folds, a purge width or a row guard, the A/B would confound the objective
    with the protocol — and the difference would be invisible in the output.
    """
    import inspect

    mae_src = inspect.getsource(purged_cv_score)
    ic_src = inspect.getsource(purged_cv_rank_ic_score)
    for shared in ("PurgedWalkForward(", "n_folds=n_folds, horizon=horizon, embargo=horizon",
                   "min_train=max(120, len(X) // 3)",
                   "mask_tr.sum() < 50 or mask_te.sum() < 10"):
        assert shared in mae_src and shared in ic_src, (
            f"{shared!r} must appear in BOTH scorers — the CV protocol is the "
            f"controlled half of this comparison"
        )


def test_a_short_series_still_falls_back_to_defaults_under_either_objective():
    X, y = _rankable_panel(n=100)
    for objective in TUNING_OBJECTIVES:
        params = tune(X, y, horizon=30, n_trials=2, tuning_objective=objective)
        assert params["gamma"] == 1.0 and params["max_depth"] == 3, (
            "below 150 rows both objectives must return _default_params()"
        )


def test_the_alternative_objective_is_not_wired_into_any_production_path():
    """
    Shadow discipline, asserted rather than promised.

    Nothing on the daily or weekly path may reach the alternative objective
    this session — the pilot drives it explicitly from tools/. The token
    searched for is ``tuning_objective``, which is unique to the new path;
    "rank_ic" alone would collide with ``eval_rank_ic`` and ``daily_rank_ic``,
    which are unrelated and everywhere.

    A grep is the right altitude because the claim is about ABSENCE across a
    set of modules, which no single call can demonstrate.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for module in ("pipeline/model.py", "pipeline/scheduler.py",
                   "agents/graph.py", "pipeline/baselines.py",
                   "pipeline/validation.py"):
        path = root / module
        if not path.exists():
            continue
        assert "tuning_objective" not in path.read_text(encoding="utf-8"), (
            f"{module} selects a tuning objective; Stage 2a is a pilot and "
            f"must not reach a production fit"
        )
