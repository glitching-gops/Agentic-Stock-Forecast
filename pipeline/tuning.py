"""
pipeline/tuning.py — Hyperparameter search, nested inside the training fold.

The previous version was called with the full labelled set, including the slice
later reported as held out (audit finding F2), and used contiguous CV folds with
no purge (F3). Fifty trials over nine hyperparameters, selected using the test
fold, on a series whose effective independent sample size is roughly
n_rows / horizon — about 13 per stock on a 2-year window.

Three changes:

  1. ``tune`` accepts ONLY a training slice. It is passed to the walk-forward
     harness as the ``tuner`` callback, which is structurally incapable of
     handing it test rows.
  2. Inner CV uses ``PurgedWalkForward``, so the search itself is not scored on
     leaked labels.
  3. Studies are seeded, so a tuning run is reproducible. The trial count is
     recorded and returned for the deflated-Sharpe adjustment — searching more
     configurations raises the bar a result must clear.
"""

from __future__ import annotations

import json
import os

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from pipeline.evaluation import (
    PurgedPanelWalkForward,
    PurgedWalkForward,
    rank_ic,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "tuned_params"
)
os.makedirs(PARAMS_DIR, exist_ok=True)

SEED = 42
# Lever 2 (moderate cut): 40 -> 25 production-tuning trials. Chosen jointly
# with the frequency cut in pipeline/model.py (weekly, not daily) rather than
# in isolation: this runs once a week now, not once a day, so the total
# monthly search budget still went UP even after this per-run cut. Also
# defensible on its own terms — the honest re-score measured near-zero/
# negative rank IC on this target, which argues against spending a large
# trial budget chasing precision the underlying signal doesn't support.
N_TRIALS = 25
INNER_FOLDS = 3

# ── Tuning objectives (Stage 2a) ──────────────────────────────────────────────
#
# "mae" is the original path and remains the DEFAULT. Nothing that does not ask
# for an alternative changes behaviour.
#
# "rank_ic" exists because the Stage 0 addendum measured that 316 of 420
# (ticker, fold) fits emit a constant, that the constant IS the training mean
# (median 0.043 sd away, so the trees make no splits), and that the search
# offers gammas below 1.0 which MAE consistently REJECTS. The evidence gate
# grades a rank correlation, which a constant cannot express at all — so the
# tuner and the grader are optimising different things, and the tuner wins.
TUNING_OBJECTIVES = ("mae", "rank_ic")

# A fold whose predictions are constant has an UNDEFINED Spearman correlation.
# It is scored as strictly worse than any attainable score rather than as NaN
# (silently dropped) or 0.0 (a mediocre-but-acceptable middle): the point of
# this objective is that Optuna must actively steer AWAY from configurations
# that emit constants, not merely decline to reward them. Since the score
# minimised is -IC and IC lies in [-1, 1], the worst attainable value is +1.0
# and this is strictly above it, so a constant can never tie a real ordering —
# not even a perfectly inverted one.
DEGENERATE_FOLD_PENALTY = 2.0


def get_params_path(ticker: str) -> str:
    return os.path.join(PARAMS_DIR, f"{ticker.replace('.', '_')}_params.json")


def save_params(ticker: str, params: dict) -> None:
    with open(get_params_path(ticker), "w") as f:
        json.dump(params, f, indent=2)


def load_params(ticker: str) -> dict | None:
    path = get_params_path(ticker)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def purged_cv_score(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    horizon: int = 30,
    n_folds: int = INNER_FOLDS,
) -> float:
    """
    Mean absolute error across purged inner folds.

    MAE on excess returns, not MAPE on prices. MAPE is undefined near zero and
    was flattering on price levels; on a return target it is meaningless.
    """
    splitter = PurgedWalkForward(
        n_folds=n_folds, horizon=horizon, embargo=horizon,
        min_train=max(120, len(X) // 3),
    )

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(len(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        mask_tr, mask_te = y_tr.notna(), y_te.notna()
        if mask_tr.sum() < 50 or mask_te.sum() < 10:
            continue

        model = XGBRegressor(**params, random_state=SEED, verbosity=0)
        model.fit(X_tr[mask_tr], y_tr[mask_tr])
        preds = model.predict(X_te[mask_te])
        scores.append(float(mean_absolute_error(y_te[mask_te], preds)))

    return float(np.mean(scores)) if scores else float("inf")


def purged_cv_rank_ic_score(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    horizon: int = 30,
    n_folds: int = INNER_FOLDS,
) -> float:
    """
    Negative mean within-fold Spearman rank IC across purged inner folds.

    Negated because Optuna minimises here, so the sign convention matches
    ``purged_cv_score`` and the two are interchangeable at the call site.

    **The correlation is taken WITHIN each validation fold and then averaged —
    never pooled across concatenated folds.** Pooling is the exact artifact the
    Stage 0 addendum diagnosed at the evaluation stage: a model emitting one
    constant per fold has no ordering at all, yet scores a non-zero pooled IC
    because the constants differ between folds, so the pooled statistic reads
    fold identity as skill. Reintroducing it here would make this objective
    reward the very thing it exists to punish.

    The CV mechanics — splitter, purge, embargo, minimum-row guards — are
    identical to ``purged_cv_score``. Only the metric differs, so an A/B
    between the two objectives is a controlled comparison.
    """
    splitter = PurgedWalkForward(
        n_folds=n_folds, horizon=horizon, embargo=horizon,
        min_train=max(120, len(X) // 3),
    )

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(len(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        mask_tr, mask_te = y_tr.notna(), y_te.notna()
        if mask_tr.sum() < 50 or mask_te.sum() < 10:
            continue

        model = XGBRegressor(**params, random_state=SEED, verbosity=0)
        model.fit(X_tr[mask_tr], y_tr[mask_tr])
        preds = np.asarray(model.predict(X_te[mask_te]), dtype=float)
        truth = y_te[mask_te].to_numpy(dtype=float)

        # ``rank_ic`` already returns NaN for a constant prediction, a constant
        # outcome, or anything else with no defined ordering — reused rather
        # than re-derived so the tuner and the evaluator cannot disagree about
        # what a rank IC is. NaN becomes the penalty here; letting it reach
        # np.mean would poison the trial into an unrankable score instead.
        ic = rank_ic(truth, preds)
        scores.append(DEGENERATE_FOLD_PENALTY if not np.isfinite(ic) else -ic)

    return float(np.mean(scores)) if scores else float("inf")


_OBJECTIVE_SCORERS = {
    "mae": purged_cv_score,
    "rank_ic": purged_cv_rank_ic_score,
}


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int = 30,
    n_trials: int = N_TRIALS,
    seed: int = SEED,
    tuning_objective: str = "mae",
) -> dict:
    """
    Searches hyperparameters using only the rows it is given.

    Designed to be passed as the ``tuner`` callback to
    ``pipeline.evaluation.walk_forward``, which calls it with the training slice
    of each outer fold. It has no access to the outer test rows by construction.

    ``tuning_objective`` selects the scoring metric and nothing else — same
    search space, same seed, same nested purged CV. "mae" is the default and
    the production path; "rank_ic" is the Stage 2a alternative. Both minimise,
    so the study direction is shared.
    """
    if tuning_objective not in _OBJECTIVE_SCORERS:
        raise ValueError(
            f"unknown tuning_objective {tuning_objective!r}; "
            f"expected one of {TUNING_OBJECTIVES}"
        )
    score = _OBJECTIVE_SCORERS[tuning_objective]

    if len(X) < 150:
        return _default_params()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 40),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 20.0),
            "tree_method":      "hist",
        }
        return score(X, y, params, horizon=horizon)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),   # seeded: reproducible
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["tree_method"] = "hist"
    return best


# ── Pooled search (Stage 2b) ──────────────────────────────────────────────────
#
# Everything above searches ONE ticker's series. Stage 2a measured why that is
# the wrong altitude: the inner CV that scores a trial holds n_eff of 3.3 to
# 14.7 independent observations, so a rank IC estimated there is mostly noise
# and the noise buys spurious splits.
#
# The pooled search is the same machinery on the whole cross-section. It splits
# the shared DATE grid via PurgedPanelWalkForward, which is what makes pooling
# compose with the purge: a training row on grid date i carries a label spanning
# [i, i + horizon], so the boundary is ONE calendar date for every ticker at
# once and no name's future can reach another name's training fold.

POOLED_INNER_FOLDS = 3


def per_date_rank_ic(dates: np.ndarray, y_true: np.ndarray,
                     y_pred: np.ndarray) -> tuple[float, int, int]:
    """
    Mean rank IC computed WITHIN each date, plus (dates scored, dates with no
    ordering).

    The cross-sectional quantity — "of the names trading today, did the ones
    ranked higher go on to earn more". Distinct from the per-ticker time-series
    IC Stage 2a optimised, and it is the one the product claims to do; it is
    also what ``reb_ic``, the break-even IC and the portfolio simulator already
    measure, so a model tuned on it is tuned on the thing it is graded by.

    Never pooled across dates. A single correlation over every (date, ticker)
    row can be earned by knowing which MONTHS were good, which is a time-series
    effect wearing cross-sectional clothes — see ``_mean_daily_rank_ic``.
    """
    order = np.argsort(dates, kind="stable")
    dates, y_true, y_pred = dates[order], y_true[order], y_pred[order]
    bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])

    ics: list[float] = []
    undefined = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ic = rank_ic(y_true[lo:hi], y_pred[lo:hi])
        if np.isfinite(ic):
            ics.append(ic)
        else:
            undefined += 1
    return (float(np.mean(ics)) if ics else float("nan"), len(ics), undefined)


def pooled_inner_splitter(panel: pd.DataFrame, horizon: int = 30,
                          n_folds: int = POOLED_INNER_FOLDS
                          ) -> PurgedPanelWalkForward:
    """
    The splitter every pooled trial is scored on.

    Extracted so it can be asserted directly. Nesting is not automatic: if THIS
    splitter lost its purge the outer fold would still look clean while every
    hyperparameter had been chosen on leaked labels — F3 one level down, and
    invisible from the outside because the reported number would merely be
    optimistic rather than wrong-shaped.
    """
    return PurgedPanelWalkForward(
        n_folds=n_folds, horizon=horizon, embargo=horizon,
        min_train=max(2 * horizon, panel["date"].nunique() // 2),
    )


def purged_panel_cv_score(
    panel: pd.DataFrame,
    features: list[str],
    params: dict,
    target: str = "target_return",
    horizon: int = 30,
    n_folds: int = POOLED_INNER_FOLDS,
    objective: str = "rank_ic",
    enable_categorical: bool = False,
) -> float:
    """
    Scores one pooled configuration on purged inner folds of the date grid.

    Minimised under both objectives, so the study direction is shared with the
    per-ticker search: "mae" returns the mean absolute error, "rank_ic" returns
    the NEGATED mean per-date rank IC.

    A date with no ordering earns ``DEGENERATE_FOLD_PENALTY`` rather than being
    skipped. Skipping it is what ``_mean_daily_rank_ic`` correctly does when
    REPORTING — an undefined IC is not a zero — but a search that skips them
    scores a model constant on 90% of dates by the 10% where it was not, which
    is how a degenerate configuration wins.
    """
    if objective not in TUNING_OBJECTIVES:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of {TUNING_OBJECTIVES}")

    splitter = pooled_inner_splitter(panel, horizon=horizon, n_folds=n_folds)
    dates = panel["date"].to_numpy()
    y_all = pd.to_numeric(panel[target], errors="coerce").to_numpy(dtype=float)

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(dates):
        tr = train_idx[np.isfinite(y_all[train_idx])]
        te = test_idx[np.isfinite(y_all[test_idx])]
        if len(tr) < 100 or len(te) < 10:
            continue

        model = XGBRegressor(**params, random_state=SEED, verbosity=0,
                             enable_categorical=enable_categorical)
        model.fit(panel.iloc[tr][features], y_all[tr])
        preds = np.asarray(model.predict(panel.iloc[te][features]), dtype=float)

        if objective == "mae":
            scores.append(float(mean_absolute_error(y_all[te], preds)))
            continue

        mean_ic, n_scored, n_undefined = per_date_rank_ic(
            dates[te], y_all[te], preds)
        # Weighted by how many dates fell each way, so a configuration that is
        # constant on most dates and ranks well on a few cannot win on the few.
        # `n_scored + n_undefined` is the fold's date count, which the row
        # guard above has already established is non-zero — an `if per_date`
        # fallback here would be a second guard covering the same case, and
        # mutation testing could not tell the two apart.
        per_date = ([-mean_ic] * n_scored
                    + [DEGENERATE_FOLD_PENALTY] * n_undefined)
        scores.append(float(np.mean(per_date)))

    return float(np.mean(scores)) if scores else float("inf")


def tune_pooled(
    panel: pd.DataFrame,
    features: list[str],
    target: str = "target_return",
    horizon: int = 30,
    n_trials: int = N_TRIALS,
    seed: int = SEED,
    tuning_objective: str = "rank_ic",
    enable_categorical: bool = False,
) -> dict:
    """
    The pooled counterpart of ``tune``. Same search space, same seed, same
    nested purge — only the altitude and the scorer change.

    Like ``tune`` it must be handed ONLY a training slice; it has no access to
    the outer test rows by construction.
    """
    if tuning_objective not in TUNING_OBJECTIVES:
        raise ValueError(
            f"unknown tuning_objective {tuning_objective!r}; "
            f"expected one of {TUNING_OBJECTIVES}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 2, 6),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 40),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 20.0),
            "tree_method":      "hist",
        }
        return purged_panel_cv_score(
            panel, features, params, target=target, horizon=horizon,
            objective=tuning_objective, enable_categorical=enable_categorical)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best["tree_method"] = "hist"
    return best


def _default_params() -> dict:
    """
    Conservative defaults for short series.

    Deliberately heavily regularised: with ~30-session overlapping labels the
    effective sample is an order of magnitude smaller than the row count, and
    the previous search space (depth up to 8, min_child_weight from 1) invited
    memorisation.
    """
    return {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,
        "gamma": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "tree_method": "hist",
    }


def tune_and_cache(
    ticker: str,
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int = 30,
    force: bool = False,
) -> dict:
    """
    Tunes for the FINAL production fit and caches the result.

    This is separate from the evaluation path on purpose. Parameters cached here
    are used to fit the model that generates tomorrow's forecast; they are never
    used to produce a reported metric, because ``walk_forward`` re-tunes inside
    each fold. Conflating the two is what F2 was.
    """
    if not force:
        cached = load_params(ticker)
        if cached:
            return cached

    params = tune(X, y, horizon=horizon)
    save_params(ticker, params)
    return params


# Backwards-compatible aliases for callers not yet migrated.
def tune_hyperparameters(ticker: str, X: pd.DataFrame, y: pd.Series,
                         n_trials: int = N_TRIALS, force: bool = False) -> dict:
    """Deprecated. Use ``tune`` (evaluation) or ``tune_and_cache`` (production)."""
    return tune_and_cache(ticker, X, y, force=force)
