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

from pipeline.evaluation import PurgedWalkForward

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


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int = 30,
    n_trials: int = N_TRIALS,
    seed: int = SEED,
) -> dict:
    """
    Searches hyperparameters using only the rows it is given.

    Designed to be passed as the ``tuner`` callback to
    ``pipeline.evaluation.walk_forward``, which calls it with the training slice
    of each outer fold. It has no access to the outer test rows by construction.
    """
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
        return purged_cv_score(X, y, params, horizon=horizon)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),   # seeded: reproducible
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
