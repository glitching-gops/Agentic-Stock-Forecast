"""
pipeline/baselines.py — The comparator set.

Phase 2 exists to attack a null result: 65 of 95 per-ticker models score below
their own majority-class baseline, and only 17 clear a random walk. Before a
pooled model or a time-series foundation model can be said to have improved on
that, there has to be something to improve *on*, measured on identical folds
over identical rows.

That is what this module is. Every estimator here implements the same
``fit(X, y)`` / ``predict(X)`` pair that XGBoost does, so all of them —
baselines, the linear factor comparator, and whatever Phase 2 adds later — run
through ``pipeline.evaluation.panel_walk_forward`` unchanged. No comparator gets
its own harness, its own folds, or its own row filter, because a comparator
scored under different conditions is not a comparator.

Two of these deserve a note.

``ZeroForecast`` is the one that matters most. The target is an excess return,
so predicting zero is the claim "this stock will track its benchmark" — the
random walk in this target space, and the floor a forecast has to clear before
any of its other properties are worth discussing. It is trivially implemented
and it is not trivially beaten.

``MajorityDirection`` is fitted on the TRAINING fold, unlike
``evaluation.majority_hit_rate``, which reads the majority off the test set.
The latter is an oracle: it cannot be implemented in advance and it sets a
deliberately conservative bar. This one is the honest version — what a person
who had only seen the training data would actually have predicted — and it is
usually the easier of the two to beat. Both are reported, because the gap
between them is itself informative about how much the direction distribution
drifts between train and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pipeline.panel import SCALE_FREE

# ── Factor set for the linear comparator ──────────────────────────────────────

# Grouped by what they are supposed to measure, so a coefficient can be read
# rather than merely reported. Every column here is scale-free (a ratio, an
# oscillator, a percentage or a return) — see pipeline/panel.py on why the
# price-denominated columns are excluded from anything pooled.
#
# This is a TECHNICAL factor model, not Fama-French. The database holds no
# fundamentals: no book-to-market, no market capitalisation, no profitability,
# no accruals. Calling it a factor model without saying so would overstate it.
# The honest description is "the standard cross-sectional technical predictors,
# fitted linearly" — which is exactly the thing a gradient-boosted tree on the
# same columns has to beat to justify its complexity.
FACTOR_GROUPS: dict[str, list[str]] = {
    "momentum":        ["roc_10", "sector_rel_20d"],
    "relative_strength": ["sector_rel_5d", "sector_rel_10d"],
    "short_reversal":  ["lag1_ret", "lag5_ret"],
    "oscillator":      ["rsi", "stoch_k", "williams_r"],
    "trend_deviation": ["dev_sma50", "prox_52w"],
    "volatility":      ["bb_width"],
    "volume":          ["vroc_10"],
    "persistence":     ["hurst"],
    "event":           ["earnings_surprise"],
}

FACTORS: list[str] = [c for group in FACTOR_GROUPS.values() for c in group]

# Sanity: every factor must be in the scale-free set, or the linear model is
# being fitted on rupees. Enforced at import rather than in a test so it cannot
# be broken by an edit that never runs the suite.
_leaked_scale = [c for c in FACTORS if c not in SCALE_FREE]
if _leaked_scale:
    raise ImportError(
        f"FACTORS contains price-denominated columns {_leaked_scale}; a pooled "
        f"linear model on those fits ticker identity, not signal."
    )


# ── Baselines ─────────────────────────────────────────────────────────────────


class ZeroForecast:
    """
    Predicts no excess return: "this stock tracks its benchmark".

    The random walk in excess-return space, and the single most important
    number in the whole comparison. A model that does not beat this has not
    demonstrated that it knows anything about relative performance, whatever
    its MAE looks like in absolute terms.
    """

    name = "zero"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ZeroForecast":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class TrainMeanForecast:
    """
    Predicts the training set's mean excess return for every name.

    Distinct from ``ZeroForecast`` in one specific way: if the universe as a
    whole has drifted against its benchmarks — which it can, since the
    benchmark mapping sends 24 tickers to the broad market and NIFTY 50 is not
    the same portfolio as the NIFTY 100 — then a constant non-zero prediction
    beats a constant zero on MAE while containing no cross-sectional
    information at all. Reporting both separates "the level moved" from "the
    ranking worked".
    """

    name = "train_mean"

    def __init__(self) -> None:
        self.mu_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TrainMeanForecast":
        clean = pd.Series(y).replace([np.inf, -np.inf], np.nan).dropna()
        self.mu_ = float(clean.mean()) if len(clean) else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mu_, dtype=float)


class MajorityDirection:
    """
    Predicts the training majority direction at unit magnitude.

    Only its hit rate is meaningful; the magnitude is arbitrary, so its MAE and
    RMSE are not comparable with anything and should be ignored. Its rank IC is
    undefined by construction, since a constant prediction has no ordering —
    ``rank_ic`` returns NaN for it rather than 0, which is the distinction
    between "no skill measured" and "skill measured as zero".
    """

    name = "majority"

    def __init__(self) -> None:
        self.sign_ = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityDirection":
        clean = pd.Series(y).replace([np.inf, -np.inf], np.nan).dropna()
        up = float((clean > 0).mean()) if len(clean) else 0.5
        self.sign_ = 1.0 if up >= 0.5 else -1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.sign_, dtype=float)


class SingleFactor:
    """
    Univariate OLS of the target on one column.

    Used for the momentum baseline. A raw passthrough of ``sector_rel_20d``
    would rank identically but sit on the wrong scale, making its MAE
    meaningless next to everything else; fitting the slope on the training fold
    puts it in the same units as the target at no cost to the ranking. If the
    slope comes out negative, that is a finding — 20-session relative strength
    reverses rather than persists — and it is preserved rather than clamped.
    """

    def __init__(self, column: str = "sector_rel_20d") -> None:
        self.column = column
        self.name = f"factor:{column}"
        self.slope_ = 0.0
        self.intercept_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SingleFactor":
        if self.column not in X.columns:
            return self
        x = pd.to_numeric(X[self.column], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(t)
        if ok.sum() < 30 or np.ptp(x[ok]) == 0:
            return self
        self.slope_, self.intercept_ = np.polyfit(x[ok], t[ok], 1)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.column not in X.columns:
            return np.zeros(len(X), dtype=float)
        x = pd.to_numeric(X[self.column], errors="coerce").to_numpy(dtype=float)
        return np.nan_to_num(self.intercept_ + self.slope_ * x, nan=0.0)


class LinearFactorModel:
    """
    Ridge regression on the cross-sectional technical factors.

    The comparator Phase 2 actually turns on. It is the model a reviewer would
    reach for first, it has no hyperparameter search worth deflating for, and
    it cannot overfit the way a boosted tree can — so if the tree does not beat
    it, the tree's extra capacity is buying noise.

    Ridge rather than OLS because the factor set is deliberately collinear:
    ``sector_rel_5d``, ``_10d`` and ``_20d`` overlap by construction, as do the
    three oscillators. OLS would give those groups unstable, alternating-sign
    coefficients that flip between folds and make the fitted model unreadable
    without changing its predictions much. The penalty is small enough not to
    shrink a real signal away and large enough to keep the coefficients
    interpretable.

    Inputs are expected to be cross-sectionally standardised already — see
    ``pipeline.panel.cross_sectional_zscore``. That is what makes the fitted
    coefficients comparable to each other, since every input then has unit
    within-date variance.
    """

    name = "linear_factor"

    def __init__(self, alpha: float = 1.0, columns: list[str] | None = None) -> None:
        self.alpha = alpha
        self.columns = list(columns) if columns else list(FACTORS)
        self.model_ = Ridge(alpha=alpha, fit_intercept=True)
        self.used_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearFactorModel":
        self.used_ = [c for c in self.columns if c in X.columns]
        if not self.used_:
            return self
        A = X[self.used_].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        t = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0.0)
        self.model_.fit(A, t)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.used_:
            return np.zeros(len(X), dtype=float)
        A = X[self.used_].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return np.asarray(self.model_.predict(A), dtype=float)

    def coefficients(self) -> dict[str, float]:
        """Fitted loadings, for reading which factors the model leaned on."""
        if not self.used_ or not hasattr(self.model_, "coef_"):
            return {}
        return dict(zip(self.used_, (float(c) for c in self.model_.coef_)))


# ── Registry ──────────────────────────────────────────────────────────────────

# What every Phase 2 result is reported against. Ordered from least to most
# informed, so a comparison table reads top to bottom as increasing claim
# strength: if a row does not beat the row above it, the extra machinery in it
# is not earning its place.
BASELINES: dict[str, callable] = {
    "zero":          ZeroForecast,
    "train_mean":    TrainMeanForecast,
    "majority":      MajorityDirection,
    "momentum_20d":  lambda: SingleFactor("sector_rel_20d"),
    "reversal_5d":   lambda: SingleFactor("lag5_ret"),
    "linear_factor": LinearFactorModel,
}


def baseline_feature_columns(name: str) -> list[str]:
    """
    Columns a given baseline needs handed to it.

    Kept explicit so the harness passes each comparator the same frame it would
    receive in production, rather than quietly widening the feature set for the
    ones that can cope with it.
    """
    if name == "linear_factor":
        return list(FACTORS)
    if name.startswith("momentum"):
        return ["sector_rel_20d"]
    if name.startswith("reversal"):
        return ["lag5_ret"]
    return []
