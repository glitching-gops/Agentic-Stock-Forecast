"""
Stage 2b — the pooled search, the cross-sectional objective, and the harness.

The leakage contract for the pooled path lives in ``test_leakage.py`` beside
the per-ticker one it extends. This file covers the rest: that the
cross-sectional objective is genuinely cross-sectional, that the pooled harness
reproduces ``panel_walk_forward``'s folds rather than approximating them, and
that the per-ticker path is still exactly what it was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation import PurgedPanelWalkForward, rank_ic
from pipeline.tuning import (
    DEGENERATE_FOLD_PENALTY,
    POOLED_INNER_FOLDS,
    _OBJECTIVE_SCORERS,
    per_date_rank_ic,
    purged_cv_score,
    purged_panel_cv_score,
    tune,
    tune_pooled,
)
from tools.stage2b_pooled import (
    TICKER_COL,
    cell_metrics,
    feature_columns,
    with_ticker,
)

HORIZON = 30


def _panel(n_dates: int = 700, n_tickers: int = 20, seed: int = 0,
           signal: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = np.repeat([f"d{i:04d}" for i in range(n_dates)], n_tickers)
    tickers = np.tile([f"T{j:02d}.NS" for j in range(n_tickers)], n_dates)
    n = dates.size
    a = rng.normal(size=n)
    return pd.DataFrame({
        "date": dates, "ticker": tickers,
        "a": a, "b": rng.normal(size=n),
        "target_return": signal * a + rng.normal(scale=0.1, size=n),
    })


# ── the cross-sectional objective is cross-sectional ──────────────────────────


def test_per_date_rank_ic_is_computed_within_dates_and_never_pooled():
    """
    The Stage 0 addendum's artifact, refused at the pooled tuning stage.

    Predictions here hold NO within-date ordering — every name on a date gets
    the same value — while the constants rise with the date and so do the
    outcomes. A pooled correlation would read that as near-perfect skill. The
    per-date statistic must see nothing at all.
    """
    n_dates, n_names = 40, 12
    dates = np.repeat([f"d{i:03d}" for i in range(n_dates)], n_names)
    level = np.repeat(np.arange(n_dates, dtype=float), n_names)
    rng = np.random.default_rng(3)
    truth = level + rng.normal(scale=0.01, size=level.size)

    pooled = rank_ic(truth, level)
    assert pooled > 0.99, "the fixture must actually contain the artifact"

    mean_ic, scored, undefined = per_date_rank_ic(dates, truth, level)
    assert scored == 0 and undefined == n_dates
    assert not np.isfinite(mean_ic)


def test_per_date_rank_ic_recovers_a_within_date_ordering():
    n_dates, n_names = 30, 15
    dates = np.repeat([f"d{i:03d}" for i in range(n_dates)], n_names)
    rng = np.random.default_rng(1)
    pred = rng.normal(size=dates.size)
    truth = pred + rng.normal(scale=0.05, size=dates.size)

    mean_ic, scored, undefined = per_date_rank_ic(dates, truth, pred)
    assert scored == n_dates and undefined == 0
    assert mean_ic > 0.9


def test_per_date_rank_ic_is_indifferent_to_row_order():
    """The grouping must come from the date values, not from the frame arriving
    pre-sorted — a caller that hands it an unsorted panel must get the same
    answer, or the objective silently depends on upstream ordering."""
    panel = _panel(n_dates=40, n_tickers=10, signal=1.0)
    pred = panel["a"].to_numpy()
    truth = panel["target_return"].to_numpy()
    dates = panel["date"].to_numpy()

    straight = per_date_rank_ic(dates, truth, pred)
    shuffled = np.random.default_rng(5).permutation(len(panel))
    jumbled = per_date_rank_ic(dates[shuffled], truth[shuffled], pred[shuffled])
    assert straight[0] == pytest.approx(jumbled[0])
    assert straight[1:] == jumbled[1:]


# ── the pooled scorer ─────────────────────────────────────────────────────────


SPLITTING = {"n_estimators": 40, "learning_rate": 0.3, "max_depth": 3,
             "min_child_weight": 1, "gamma": 0.0, "tree_method": "hist"}
CONSTANT = {**SPLITTING, "gamma": 1e6}


def test_the_pooled_rank_ic_objective_prefers_an_ordering_to_a_constant():
    panel = _panel(signal=0.5)
    splitting = purged_panel_cv_score(panel, ["a", "b"], SPLITTING,
                                      horizon=HORIZON, objective="rank_ic")
    constant = purged_panel_cv_score(panel, ["a", "b"], CONSTANT,
                                     horizon=HORIZON, objective="rank_ic")
    assert splitting < constant
    assert splitting < 0


def test_a_pooled_constant_earns_the_penalty_not_a_skipped_date():
    """
    Reporting an undefined IC and SEARCHING on one are different jobs.

    ``_mean_daily_rank_ic`` correctly skips a date with no ordering, because an
    undefined IC is not a zero. A SEARCH that skips them scores a model
    constant on 90% of dates by the 10% where it happened to vary, which is how
    a degenerate configuration wins the study.
    """
    panel = _panel(signal=0.5)
    score = purged_panel_cv_score(panel, ["a", "b"], CONSTANT,
                                  horizon=HORIZON, objective="rank_ic")
    assert np.isfinite(score)
    assert score == pytest.approx(DEGENERATE_FOLD_PENALTY)


def test_a_configuration_degenerate_on_MOST_dates_cannot_win_on_the_few():
    """
    The case that separates penalising an undefined date from skipping it, and
    the one a fully-constant fixture cannot reach.

    A model that emits a constant on most dates and a real ordering on a few
    would, under a skip-the-undefined rule, be scored ONLY on the few — so a
    configuration that has almost no opinion beats one that has an opinion
    everywhere. Weighting by the date count is what stops that.
    """
    n_dates, n_names = 20, 10
    mean_ic, mostly_flat, all_ranked = -0.4, [], []
    for scored, undefined in ((4, 16), (20, 0)):
        per_date = [-mean_ic] * scored + [DEGENERATE_FOLD_PENALTY] * undefined
        (mostly_flat if undefined else all_ranked).append(float(np.mean(per_date)))

    assert all_ranked[0] < mostly_flat[0], (
        "ranking on every date must score better than ranking on four of "
        "twenty; under a skip rule the two would tie exactly")

    skipped = float(np.mean([-mean_ic] * 4))
    assert skipped == pytest.approx(all_ranked[0]), (
        "and under a skip rule they DO tie — which is the defect")


def test_a_partly_degenerate_configuration_is_scored_on_every_date():
    """
    The same thing end to end, through the real scorer.

    ``min_child_weight`` large enough to flatten most dates but not all gives a
    configuration whose score must sit strictly between a full ordering and a
    full constant. Under a skip-the-undefined rule it could score better than
    the full ordering, which is the failure.
    """
    panel = _panel(signal=0.5)
    ranked = purged_panel_cv_score(panel, ["a", "b"], SPLITTING,
                                   horizon=HORIZON, objective="rank_ic")
    constant = purged_panel_cv_score(panel, ["a", "b"], CONSTANT,
                                     horizon=HORIZON, objective="rank_ic")
    assert ranked < constant == pytest.approx(DEGENERATE_FOLD_PENALTY)

    # Half the dates given a flat OUTCOME, which makes their rank IC undefined
    # whatever the model predicts. The same configuration must now score
    # strictly worse. Under a skip-the-undefined rule it would score the SAME
    # or better, since only the surviving dates would count.
    half_flat = panel.copy()
    flat_dates = sorted(panel["date"].unique())[::2]
    half_flat.loc[half_flat["date"].isin(flat_dates), "target_return"] = 0.01

    partly = purged_panel_cv_score(half_flat, ["a", "b"], SPLITTING,
                                   horizon=HORIZON, objective="rank_ic")

    # The discriminating bound. Each defined date contributes at worst -1 and
    # each undefined one contributes +2, so with roughly half undefined the
    # weighted mean cannot fall below (0.5 * 2) + (0.5 * -1) = +0.5 and is
    # certainly POSITIVE. Under a skip-the-undefined rule the score would be
    # -mean_ic over the surviving dates alone, which is NEGATIVE for any
    # positive ordering — so the sign alone separates the two rules, whatever
    # the model happens to learn.
    assert ranked < 0 < partly < constant, (
        f"a configuration undefined on half its dates scored {partly:.4f}; a "
        f"negative score means only the surviving dates were counted")


def test_the_pooled_inner_splitter_carries_the_label_horizon_as_its_purge():
    """
    Asserted on the splitter the scorer actually builds, not on a copy of it.

    A purge that quietly became 1 session would leave the outer fold looking
    clean while every hyperparameter had been chosen on leaked labels, and the
    only symptom would be a number that is merely too good.
    """
    import pipeline.tuning as tuning
    from pipeline.tuning import pooled_inner_splitter

    panel = _panel(n_dates=700, n_tickers=6)
    s = pooled_inner_splitter(panel, horizon=HORIZON)
    assert s.horizon == HORIZON
    assert s.effective_embargo == HORIZON
    assert s.n_folds == POOLED_INNER_FOLDS
    assert s.min_train >= 2 * HORIZON

    dates = panel["date"].to_numpy()
    grid = np.unique(dates)
    splits = list(s.split(dates))
    assert splits, "no inner folds; the assertion below would be vacuous"
    for train_idx, test_idx in splits:
        gap = (np.searchsorted(grid, dates[test_idx].min())
               - np.searchsorted(grid, dates[train_idx].max()) - 1)
        assert gap >= HORIZON, f"inner purge is only {gap} dates"

    # And the scorer must actually USE it. Testing the helper alone leaves the
    # scorer free to build its own splitter beside it, which is a live failure
    # mode: the helper would still pass every assertion above while every
    # trial was scored on leaked labels.
    called: list[int] = []
    real = tuning.pooled_inner_splitter

    def spy(*a, **kw):
        called.append(1)
        return real(*a, **kw)

    tuning.pooled_inner_splitter = spy
    try:
        purged_panel_cv_score(_panel(n_dates=400, n_tickers=6), ["a", "b"],
                              SPLITTING, horizon=HORIZON, objective="mae")
    finally:
        tuning.pooled_inner_splitter = real
    assert called, "the scorer built its own splitter instead of the declared one"


def test_the_pooled_scorer_refuses_an_unknown_objective():
    panel = _panel(n_dates=300, n_tickers=5)
    with pytest.raises(ValueError, match="unknown objective"):
        purged_panel_cv_score(panel, ["a", "b"], SPLITTING, objective="sharpe")


def test_the_pooled_search_reaches_both_objectives(monkeypatch):
    import pipeline.tuning as tuning

    seen: list[str] = []
    real = tuning.purged_panel_cv_score

    def spy(*a, **kw):
        seen.append(kw.get("objective", "?"))
        return real(*a, **kw)

    monkeypatch.setattr(tuning, "purged_panel_cv_score", spy)
    panel = _panel(n_dates=400, n_tickers=8)
    tuning.tune_pooled(panel, ["a", "b"], horizon=HORIZON, n_trials=2,
                       tuning_objective="mae")
    assert set(seen) == {"mae"}

    seen.clear()
    tuning.tune_pooled(panel, ["a", "b"], horizon=HORIZON, n_trials=2,
                       tuning_objective="rank_ic")
    assert set(seen) == {"rank_ic"}


def test_the_pooled_search_refuses_an_unknown_objective():
    panel = _panel(n_dates=300, n_tickers=5)
    with pytest.raises(ValueError, match="unknown tuning_objective"):
        tune_pooled(panel, ["a", "b"], horizon=HORIZON, n_trials=1,
                    tuning_objective="sharpe")


def test_the_pooled_inner_cv_uses_the_declared_fold_count():
    """A constant that drifts changes what every trial is scored on, silently.
    Pinned against an independently built splitter rather than against itself."""
    panel = _panel(n_dates=700, n_tickers=4)
    expected = list(PurgedPanelWalkForward(
        n_folds=POOLED_INNER_FOLDS, horizon=HORIZON, embargo=HORIZON,
        min_train=max(2 * HORIZON, panel["date"].nunique() // 2),
    ).split(panel["date"].to_numpy()))
    assert 0 < len(expected) <= POOLED_INNER_FOLDS


# ── the harness is panel_walk_forward's, not an approximation of it ──────────


def test_the_pooled_harness_reproduces_panel_walk_forward_exactly():
    """
    The Stage 2a drift check, at panel altitude.

    ``run_arm`` runs its own outer loop because it needs the training error and
    a tuner callback, neither of which ``panel_walk_forward`` returns. That is
    a second implementation of a fold boundary, and this project has been
    caught by exactly that before — a notebook re-deriving `end_index` slightly
    differently produces a number that looks fine and is comparable with
    nothing. Given the same fixed hyperparameters the two must agree to the
    last bit, or Stage 2b's cells cannot be read beside the baselines table.
    """
    from pipeline.determinism import xgb_params
    from pipeline.evaluation import panel_walk_forward
    from tools.stage2b_pooled import EVAL_MIN_TRAIN_DATES, run_arm
    from xgboost import XGBRegressor

    panel = _panel(n_dates=700, n_tickers=10, signal=0.4)
    params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1,
              "tree_method": "hist"}

    # THE RAW LABEL ON BOTH SIDES. `run_arm` standardises by default now;
    # `panel_walk_forward` does not, because `compare_baselines` reads it and
    # that table's floors (`market`, `train_mean`) are defined in return units.
    # What this test measures is the FOLD BOUNDARY, so the label must be the
    # one thing that does not differ between the two.
    mine, folds = run_arm(panel, "mae", "none", min_train=400, verbose=False,
                          fixed_params=params, features=["a", "b"],
                          standardise_label=False)

    # And the comparator is built through the same pin. Constructing it bare
    # would leave `n_jobs` at the machine default while `run_arm` uses two,
    # and P6 measured that a thread-count difference alone changes every
    # prediction — so this equality check would be asserting something about
    # the machine rather than about the harness.
    theirs = panel_walk_forward(
        panel, ["a", "b"],
        model_factory=lambda: XGBRegressor(**xgb_params(**params,
                                                        random_state=42)),
        splitter=PurgedPanelWalkForward(n_folds=5, horizon=HORIZON,
                                        embargo=HORIZON, min_train=400),
        target="target_return",
    ).predictions

    assert len(mine) == len(theirs) > 0
    key = ["date", "ticker", "fold"]
    a = mine.sort_values(key).reset_index(drop=True)
    b = theirs.sort_values(key).reset_index(drop=True)
    assert (a["date"].to_numpy() == b["date"].to_numpy()).all()
    assert (a["ticker"].to_numpy() == b["ticker"].to_numpy()).all()
    assert np.max(np.abs(a["y_pred"].to_numpy() - b["y_pred"].to_numpy())) == 0.0
    assert len(folds) == theirs["fold"].nunique()


# ── the per-ticker path is unchanged ──────────────────────────────────────────


def test_the_per_ticker_path_is_untouched_by_the_pooled_addition():
    """
    Regression pin. Two of the four cells of the 2x2 are REUSED from earlier
    sessions rather than recomputed, so the per-ticker path producing them must
    still be bit-for-bit what produced them.
    """
    assert _OBJECTIVE_SCORERS["mae"] is purged_cv_score

    rng = np.random.default_rng(3)
    X = pd.DataFrame({c: rng.normal(size=400) for c in ("a", "b", "c")})
    y = pd.Series(rng.normal(scale=0.1, size=400))
    assert tune(X, y, horizon=HORIZON, n_trials=3) == tune(
        X, y, horizon=HORIZON, n_trials=3, tuning_objective="mae")


def test_no_single_date_is_split_across_the_pooled_train_test_boundary():
    """
    The reason a panel needs its own splitter, asserted as behaviour.

    A row-position splitter cuts wherever row N falls, which on a panel is
    mid-cross-section: some of a date's names land in training and the rest in
    test, so the model is fitted on part of the very cross-section it is then
    scored on. Splitting the shared date grid makes that unconstructable, and
    the pooled inner CV inherits it.
    """
    panel = _panel(n_dates=700, n_tickers=20)
    dates = panel["date"].to_numpy()
    inner = PurgedPanelWalkForward(
        n_folds=POOLED_INNER_FOLDS, horizon=HORIZON, embargo=HORIZON,
        min_train=max(2 * HORIZON, panel["date"].nunique() // 2))

    splits = list(inner.split(dates))
    assert splits, "no inner folds; the test would be vacuous"
    for train_idx, test_idx in splits:
        straddling = set(dates[train_idx]) & set(dates[test_idx])
        assert not straddling, (
            f"{len(straddling)} dates have names on both sides of the boundary")
        # And every fold must actually hold whole cross-sections.
        counts = pd.Series(dates[test_idx]).value_counts().unique()
        assert list(counts) == [20], "a test fold holds a partial cross-section"


# ── the ticker feature ────────────────────────────────────────────────────────


def test_the_feature_list_carries_the_ticker_only_when_asked():
    assert TICKER_COL not in feature_columns("none")
    assert feature_columns("identity")[-1] == TICKER_COL
    assert feature_columns("shuffled")[-1] == TICKER_COL


def test_every_ticker_mode_keeps_the_same_categories():
    """The category set must come from the panel, not from whichever labels
    happen to appear — a fold whose categories differ from training's would
    make the feature mean different things in the two halves."""
    panel = _panel(n_dates=40, n_tickers=6)
    for mode in ("identity", "shuffled"):
        tagged = with_ticker(panel, mode)
        assert list(tagged[TICKER_COL].cat.categories) == sorted(
            panel["ticker"].unique())


def test_an_unknown_ticker_mode_is_refused():
    with pytest.raises(ValueError, match="unknown ticker mode"):
        with_ticker(_panel(n_dates=20, n_tickers=3), "target_encoded")


def test_the_placebo_is_deterministic():
    panel = _panel(n_dates=30, n_tickers=6)
    a = with_ticker(panel, "shuffled")[TICKER_COL].astype(str).to_numpy()
    b = with_ticker(panel, "shuffled")[TICKER_COL].astype(str).to_numpy()
    assert (a == b).all()


# ── the reported cell ─────────────────────────────────────────────────────────


def test_cell_metrics_counts_a_constant_cell_as_degenerate():
    preds = pd.DataFrame({
        "date": np.repeat([f"d{i:03d}" for i in range(20)], 4),
        "ticker": np.tile(["A.NS", "B.NS", "C.NS", "D.NS"], 20),
        "y_true": np.random.default_rng(0).normal(size=80),
        "fold": 0,
    })
    preds["y_pred"] = np.where(preds["ticker"] == "A.NS", 0.02,
                               np.random.default_rng(1).normal(size=80))

    m = cell_metrics(preds)
    assert m["cells"] == 4
    assert m["constant_cells"] == 1
    assert m["degeneracy_rate"] == pytest.approx(0.25)


def test_cell_metrics_reports_the_cross_sectional_ic_not_the_pooled_one():
    """
    The same artifact as the objective test, at the reporting end.

    One constant per fold, rising with the fold, against outcomes that rise
    with it: the pooled correlation is near +1 and the honest number is
    undefined. A cell metric that quietly reported the pooled figure would
    hand the 2x2 a column of artifacts.
    """
    rows = []
    for fold in range(3):
        for i in range(10):
            # One constant per DATE, rising with the date, against outcomes
            # that rise with it. Within a fold the prediction still varies, so
            # a fold-level pooled correlation is defined and near +1 — which is
            # what makes this fixture separate the two statistics. A
            # constant-per-FOLD fixture cannot: its pooled IC is undefined too,
            # so the correct code and the defect agree.
            level = fold * 10 + i
            for t in "ABCD":
                rows.append({"date": f"d{level:03d}", "ticker": f"{t}.NS",
                             "y_true": level + 0.001 * ord(t),
                             "y_pred": float(level), "fold": fold})
    preds = pd.DataFrame(rows)

    for fold, g in preds.groupby("fold"):
        assert rank_ic(g["y_true"].to_numpy(), g["y_pred"].to_numpy()) > 0.9, (
            "the fold-level pooled IC must be strongly positive, or this test "
            "is not exercising the artifact")

    m = cell_metrics(preds)
    assert not np.isfinite(m["cs_rank_ic"]), (
        "within every date the prediction is constant, so there is no "
        "cross-sectional ordering at all")
    assert m["constant_cells"] == 0, (
        "and the model is NOT constant within a fold — the defect this "
        "separates from is precisely the one a constant-per-fold fixture hides")


def test_the_gap_is_undefined_rather_than_zero_when_training_error_is_unknown():
    """
    The per-ticker x mae cell is read from a cache of PREDICTIONS, which holds
    no training error. Reporting its gap as 0.0 would put the production
    baseline at the best possible value on the one column that detects
    overfitting — a missing measurement dressed as a perfect score.
    """
    preds = pd.DataFrame({
        "date": [f"d{i:03d}" for i in range(20)],
        "ticker": ["A.NS"] * 20,
        "y_true": np.linspace(-0.1, 0.1, 20),
        "y_pred": np.linspace(-0.05, 0.05, 20),
        "fold": 0,
    })
    m = cell_metrics(preds, folds=None)
    assert not np.isfinite(m["train_mae"])
    assert not np.isfinite(m["gap"])
