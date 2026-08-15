"""
pipeline/evaluation.py — Purged, embargoed walk-forward evaluation.

This module exists because the previous evaluation reported numbers that could
not be true. Three defects compounded (audit findings F1-F3):

  - The Ridge meta-learner was fitted on the validation set and then scored on
    that same validation set. Re-running that procedure on live NSE data
    reproduces the README's headline figures (3.0% MAPE, 83% directional
    accuracy) from pure in-sample fit.
  - Optuna tuned on the full labelled set, including the slice later reported
    as held out.
  - Folds were contiguous, so with a 30-session label the last 30 training rows
    carried labels drawn from inside the test window.

Everything here follows one rule: **a model never scores a row whose label
overlaps anything it was trained on.** Purging removes training rows whose
label window reaches into the test window; the embargo widens that gap.

Every result is reported next to a baseline. A forecasting number without a
random-walk or majority-class comparison is not interpretable, and the whole
point of Phase 0 is that the project stops publishing those.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# ── Fold construction ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PurgedWalkForward:
    """
    Rolling-origin splitter with purging and an embargo.

    For a test window beginning at position ``t``, any training row at position
    ``i`` whose label spans ``[i, i + horizon]`` overlaps the test window when
    ``i + horizon >= t``. Those rows are purged. The embargo drops a further
    ``embargo`` rows so that serial correlation just before the boundary cannot
    leak either.

    Args:
        n_folds:   number of successive test windows.
        horizon:   label length in rows. Must match the target's horizon.
        embargo:   extra rows dropped from the end of training. Defaults to the
                   horizon, which is the conservative choice.
        min_train: row offset at which the FIRST test window begins. The first
                   fold therefore trains on ``min_train - horizon - embargo``
                   rows, since the purge gap is carved out of the training end.
                   Set it comfortably above the longest feature lookback (252
                   rows for 52-week proximity).
    """

    n_folds: int = 6
    horizon: int = 30
    embargo: int | None = None
    min_train: int = 250

    @property
    def effective_embargo(self) -> int:
        return self.horizon if self.embargo is None else self.embargo

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yields (train_idx, test_idx) with purging and embargo applied."""
        usable = n_samples - self.min_train
        if usable <= 0:
            return

        step = max(1, usable // self.n_folds)

        for fold in range(self.n_folds):
            test_start = self.min_train + fold * step
            test_end = min(test_start + step, n_samples)
            if test_end <= test_start:
                continue

            # Purge + embargo: training must end far enough before the test
            # window that no training label reaches into it.
            train_end = test_start - self.horizon - self.effective_embargo
            if train_end < self.min_train // 2:
                continue

            yield np.arange(0, train_end), np.arange(test_start, test_end)

    def describe(self) -> dict:
        return {
            "splitter": "PurgedWalkForward",
            "n_folds": self.n_folds,
            "horizon": self.horizon,
            "embargo": self.effective_embargo,
            "min_train": self.min_train,
        }


def assert_no_leakage(
    train_dates: Sequence[str],
    test_dates: Sequence[str],
    horizon_sessions: int,
    all_dates: Sequence[str],
) -> None:
    """
    Raises if any training label window reaches into the test window.

    Called by the regression tests and by ``walk_forward`` in debug runs. This
    is the assertion whose absence allowed F3 to persist unnoticed.
    """
    if len(train_dates) == 0 or len(test_dates) == 0:
        return

    index = {d: i for i, d in enumerate(all_dates)}
    last_train = max(index[d] for d in train_dates)
    first_test = min(index[d] for d in test_dates)

    if last_train + horizon_sessions >= first_test:
        raise AssertionError(
            f"Label leakage: last training row at position {last_train} has a "
            f"{horizon_sessions}-session label reaching position "
            f"{last_train + horizon_sessions}, but the test window starts at "
            f"{first_test}. Purge and embargo are not being applied."
        )


# ── Results ───────────────────────────────────────────────────────────────────


@dataclass
class WalkForwardResult:
    """Out-of-sample predictions and metrics from a purged walk-forward run."""

    ticker: str
    n_folds_run: int
    n_predictions: int
    predictions: pd.DataFrame = field(repr=False)      # date, y_true, y_pred, fold
    metrics: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    protocol: dict = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "predictions"}
        return json.dumps(payload, indent=2, default=str)


# ── Metrics ───────────────────────────────────────────────────────────────────


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between prediction and outcome."""
    if len(y_true) < 3:
        return float("nan")
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return float("nan")

    # A constant series has no rank ordering, so correlation is undefined
    # rather than zero. Checking here avoids a scipy warning and, more
    # importantly, stops a degenerate baseline (e.g. the always-up predictor)
    # from being reported as though it had been measured.
    if np.ptp(y_pred[valid]) == 0 or np.ptp(y_true[valid]) == 0:
        return float("nan")

    result = stats.spearmanr(y_pred[valid], y_true[valid]).correlation
    return float(result) if result is not None and np.isfinite(result) else float("nan")


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions whose sign matches the outcome, as a percentage."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean((y_pred[valid] > 0) == (y_true[valid] > 0)) * 100)


def majority_hit_rate(y_true: np.ndarray) -> float:
    """
    Accuracy of always predicting the more common direction.

    This is the number a directional-accuracy claim must beat. The previous
    system reported 85% without ever computing it; on the same windows the
    majority baseline is around 59%.
    """
    valid = np.isfinite(y_true)
    if valid.sum() == 0:
        return float("nan")
    up = float(np.mean(y_true[valid] > 0))
    return float(max(up, 1 - up) * 100)


def effective_sample_size(n: int, horizon: int) -> float:
    """
    Independent-observation count for overlapping labels.

    Consecutive rows share horizon-1 of their horizon days, so a 30-session
    target makes neighbouring observations ~97% overlapping. Treating all n
    rows as independent inflates every t-statistic by roughly sqrt(horizon) —
    about 5.5x at a 30-session horizon.

    This is the same family of error as F1-F3: a number that looks decisive
    because the sample was counted wrongly. n/horizon is the standard
    conservative correction.
    """
    return max(float(n) / max(horizon, 1), 1.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    horizon: int = 30) -> dict:
    """Forecast-quality metrics for an excess-return target, with baselines."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[valid], y_pred[valid]

    if len(yt) < 3:
        return {"n": int(len(yt))}

    ic = rank_ic(yt, yp)
    # Under the null of no skill IC is approximately N(0, 1/sqrt(n_eff - 1)),
    # where n_eff discounts the overlap between successive labels.
    n_eff = effective_sample_size(len(yt), horizon)
    ic_t = float(ic * np.sqrt(max(n_eff - 1, 1))) if np.isfinite(ic) else float("nan")

    # The naive forecast for an EXCESS return is zero: "this stock will track
    # its benchmark". That is the random walk in this target space.
    mae_model = float(np.mean(np.abs(yt - yp)))
    mae_naive = float(np.mean(np.abs(yt)))

    return {
        "n": int(len(yt)),
        "n_effective": round(n_eff, 1),
        "rank_ic": ic,
        "rank_ic_t": ic_t,
        "hit_rate": hit_rate(yt, yp),
        "majority_hit_rate": majority_hit_rate(yt),
        "mae": mae_model,
        "mae_naive_zero": mae_naive,
        "beats_naive_mae": bool(mae_model < mae_naive),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "mean_pred": float(np.mean(yp)),
        "mean_actual": float(np.mean(yt)),
    }


def baseline_predictions(
    y_true: np.ndarray,
    features: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """
    The comparison set every result is reported against.

    zero      — predict no excess return, i.e. "this stock tracks its
                benchmark". This is the random walk in excess-return space and
                the single most important number to beat.
    majority  — predict the majority direction at unit magnitude. Only its hit
                rate is meaningful; its MAE is not, since the magnitude is
                arbitrary.
    momentum  — recent relative strength continues. A real, non-trivial
                baseline: if the model cannot beat 20-session sector-relative
                momentum, it has added nothing to a one-line heuristic.
    """
    baselines = {
        "zero": np.zeros_like(y_true, dtype=float),
        "majority": np.full_like(y_true, 1.0 if np.mean(y_true > 0) >= 0.5 else -1.0,
                                 dtype=float),
    }
    if features is not None and "sector_rel_20d" in features.columns:
        baselines["momentum"] = features["sector_rel_20d"].to_numpy(dtype=float)
    return baselines


# ── The harness ───────────────────────────────────────────────────────────────


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    dates: Sequence[str],
    model_factory: Callable[[], object],
    splitter: PurgedWalkForward | None = None,
    ticker: str = "",
    tuner: Callable[[pd.DataFrame, pd.Series], dict] | None = None,
    verify: bool = True,
) -> WalkForwardResult:
    """
    Runs purged walk-forward evaluation and returns out-of-sample predictions.

    Args:
        model_factory: returns a fresh unfitted estimator. Called per fold, so
                       no state survives between folds.
        tuner:         optional. Receives ONLY the training slice and returns
                       hyperparameters. This is the nested-tuning contract that
                       F2 violated — the tuner never sees test rows.
        verify:        run the leakage assertion on every fold.
    """
    splitter = splitter or PurgedWalkForward()
    dates = list(dates)

    rows: list[pd.DataFrame] = []
    folds_run = 0

    for fold, (train_idx, test_idx) in enumerate(splitter.split(len(X))):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        labelled = y_train.notna()
        if labelled.sum() < 50:
            continue
        X_train, y_train = X_train[labelled], y_train[labelled]

        if verify:
            assert_no_leakage(
                [dates[i] for i in train_idx[labelled.to_numpy()]],
                [dates[i] for i in test_idx],
                splitter.horizon,
                dates,
            )

        params = tuner(X_train, y_train) if tuner else {}
        model = model_factory()
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train)

        block = pd.DataFrame({
            "date": [dates[i] for i in test_idx],
            "y_true": y_test.to_numpy(dtype=float),
            "y_pred": np.asarray(model.predict(X_test), dtype=float),
            "fold": fold,
        })
        # Carry the momentum feature through so the momentum baseline can be
        # scored on exactly the same rows as the model.
        if "sector_rel_20d" in X_test.columns:
            block["sector_rel_20d"] = X_test["sector_rel_20d"].to_numpy(dtype=float)

        rows.append(block)
        folds_run += 1

    if not rows:
        return WalkForwardResult(ticker, 0, 0, pd.DataFrame(
            columns=["date", "y_true", "y_pred", "fold"]),
            protocol=splitter.describe())

    preds = pd.concat(rows, ignore_index=True)
    preds = preds[np.isfinite(preds["y_true"])].reset_index(drop=True)

    y_true = preds["y_true"].to_numpy()
    metrics = compute_metrics(y_true, preds["y_pred"].to_numpy(),
                              horizon=splitter.horizon)

    features = preds[["sector_rel_20d"]] if "sector_rel_20d" in preds.columns else None
    baselines = {
        name: compute_metrics(y_true, values, horizon=splitter.horizon)
        for name, values in baseline_predictions(y_true, features).items()
    }

    return WalkForwardResult(
        ticker=ticker,
        n_folds_run=folds_run,
        n_predictions=len(preds),
        predictions=preds,
        metrics=metrics,
        baselines=baselines,
        protocol=splitter.describe(),
    )


# ── Cross-sectional evaluation ────────────────────────────────────────────────


def cross_sectional_report(
    panel: pd.DataFrame,
    rebalance_every: int = 30,
    quantiles: int = 5,
) -> dict:
    """
    Evaluates the panel the way the leaderboard is actually used: as a ranking.

    ``panel`` needs columns ``date``, ``ticker``, ``y_pred``, ``y_true``, all
    out-of-sample. Returns per-rebalance rank IC and quantile spreads with
    t-statistics, so a weak signal is reported as weak rather than as a headline.

    ``rebalance_every`` defaults to the 30-session horizon so that successive
    rebalances hold non-overlapping return windows. Sampling more frequently
    than the horizon makes the observations overlap, which inflates the
    t-statistics on ``alpha`` and ``ic``.
    """
    required = {"date", "ticker", "y_pred", "y_true"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"cross_sectional_report needs columns {missing}")

    panel = panel.dropna(subset=["y_pred", "y_true"])
    if panel.empty:
        return {"n_rebalances": 0, "note": "no out-of-sample panel rows"}

    dates = sorted(panel["date"].unique())[::rebalance_every]
    records = []

    for dt in dates:
        day = panel[panel["date"] == dt]
        if len(day) < max(10, quantiles * 2):
            continue
        day = day.sort_values("y_pred", ascending=False)
        k = max(2, len(day) // quantiles)
        records.append({
            "date": dt,
            "n": len(day),
            "top": float(day.head(k)["y_true"].mean()),
            "bottom": float(day.tail(k)["y_true"].mean()),
            "all": float(day["y_true"].mean()),
            "ic": rank_ic(day["y_true"].to_numpy(), day["y_pred"].to_numpy()),
        })

    if not records:
        return {"n_rebalances": 0, "note": "not enough breadth per date to rank"}

    R = pd.DataFrame(records)
    alpha = R["top"] - R["all"]
    spread = R["top"] - R["bottom"]

    def _t(series: pd.Series) -> tuple[float, float]:
        clean = series.dropna()
        if len(clean) < 3:
            return float("nan"), float("nan")
        t, p = stats.ttest_1samp(clean, 0)
        return float(t), float(p)

    alpha_t, alpha_p = _t(alpha)
    ic_t, ic_p = _t(R["ic"])
    spread_t, spread_p = _t(spread)

    return {
        "n_rebalances": len(R),
        "mean_rank_ic": float(R["ic"].mean()),
        "rank_ic_t": ic_t,
        "rank_ic_p": ic_p,
        "top_quintile_return": float(R["top"].mean()),
        "bottom_quintile_return": float(R["bottom"].mean()),
        "equal_weight_return": float(R["all"].mean()),
        "alpha_vs_equal_weight": float(alpha.mean()),
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "long_short_spread": float(spread.mean()),
        "spread_t": spread_t,
        "spread_p": spread_p,
        "win_rate_vs_equal_weight": float((R["top"] > R["all"]).mean()),
        "significant_at_5pct": bool(np.isfinite(alpha_p) and alpha_p < 0.05),
    }


def deflated_sharpe_note(n_trials: int, observed_sharpe: float, n_obs: int) -> dict:
    """
    Reports how much of an observed Sharpe ratio is explained by having tried
    many configurations (Bailey & Lopez de Prado).

    The expected maximum Sharpe under the null of no skill grows with the
    number of trials, so a backtest that searched 50 Optuna configurations per
    stock needs a materially higher bar than one that searched none.
    """
    if n_trials < 2 or n_obs < 3:
        return {"n_trials": n_trials, "note": "too few trials or observations"}

    euler = 0.5772156649
    e_max = (
        (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials)
        + euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    deflated = (observed_sharpe - e_max) * np.sqrt(max(n_obs - 1, 1))

    return {
        "n_trials": n_trials,
        "observed_sharpe": float(observed_sharpe),
        "expected_max_sharpe_under_null": float(e_max),
        "deflated_statistic": float(deflated),
        "clears_null": bool(deflated > 1.96),
        "note": (
            f"With {n_trials} configurations tried, a Sharpe of {e_max:.3f} is "
            f"expected from luck alone."
        ),
    }


def format_report(result: WalkForwardResult) -> str:
    """Renders a walk-forward result with its baselines side by side."""
    if result.n_predictions == 0:
        return f"{result.ticker}: no out-of-sample predictions (insufficient history)"

    m = result.metrics
    zero = result.baselines.get("zero", {})

    lines = [
        f"{result.ticker or 'model'}  —  {result.n_predictions} OOS predictions "
        f"across {result.n_folds_run} purged folds "
        f"(n_eff = {m.get('n_effective', float('nan'))} after overlap correction)",
        f"  rank IC          {m.get('rank_ic', float('nan')):+.4f}  "
        f"(t = {m.get('rank_ic_t', float('nan')):+.2f})",
        f"  hit rate         {m.get('hit_rate', float('nan')):.2f}%   "
        f"majority baseline {m.get('majority_hit_rate', float('nan')):.2f}%",
        f"  MAE              {m.get('mae', float('nan')):.5f}   "
        f"naive (zero excess) {zero.get('mae', float('nan')):.5f}",
        f"  beats naive MAE  {m.get('beats_naive_mae', False)}",
    ]
    return "\n".join(lines)
