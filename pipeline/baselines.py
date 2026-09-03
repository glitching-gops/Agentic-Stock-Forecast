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

**THE FLOOR IS NO LONGER `zero`.** It was, and correctly, while the target was
an excess return: predicting zero is the claim "this stock will track its
benchmark", the label already had the market subtracted out of it, and beating
zero therefore required saying something about the individual company.

P1 moved the target to the ABSOLUTE 30-session return, and that breaks the
argument in two places at once. 57.67% of absolute returns on this universe are
positive, so a constant "up" beats a coin flip by nearly eight points; and
32.8% of their variance is COMMON across stocks, so predicting the market alone
explains a third of the target with no company-specific information at all. A
comparator measured against `zero` on this target is being credited with drift
and with beta.

So the floors are ``market`` (the level, bounding MAE) and ``beta_market`` (the
ordering, bounding rank IC) — see ``FLOORS``, and ``annotate_against_floors``
for where a comparator is graded against them. ``zero`` and ``always_up`` stay
in the table as the degenerate references: they are what shows how large the
gift was.

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

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pipeline.evaluation import (PurgedPanelWalkForward, oos_dates,
                                 panel_walk_forward)
from pipeline.panel import (
    EXCESS_TARGET,
    attach_news,
    attach_regime,
    MIN_NAMES_PER_DATE,
    SCALE_FREE,
    TARGET,
    cross_sectional_zscore,
    load_panel,
    panel_coverage,
    price_frame,
    retarget_horizon,
)
from pipeline.series import (
    DEFAULT_CONTEXT as SERIES_CONTEXT,
    SERIES_BASELINES,
    adapter_factory,
)
from pipeline.signals import HORIZON_SESSIONS

logger = logging.getLogger(__name__)

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


class AlwaysUp:
    """
    Predicts a positive return for every stock, every day. THE DIRECTIONAL FLOOR.

    Distinct from ``MajorityDirection``, which LEARNS its sign from the
    training fold and would land on +1 here anyway. This one is pinned, so the
    bar it sets cannot drift with the sample: on this universe 57.67% of
    30-session absolute returns are positive, measured over 84 tickers and
    205,973 windows, and any model whose directional accuracy is below that has
    learned less than "shares tend to go up".

    That is a materially higher bar than the excess target set. There the
    majority baseline was ~52%, so a model at 55% looked like it had found
    something; on absolute return the same 55% is three points WORSE than
    saying nothing. Switching target without moving the floor would have
    handed every model in the project roughly six free points.

    Only its hit rate is meaningful. The magnitude is arbitrary, so its MAE and
    RMSE are not comparable with anything, and its rank IC is undefined by
    construction because a constant has no ordering.
    """

    name = "always_up"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlwaysUp":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=float)


class MarketForecast:
    """
    Predicts the training-period MARKET return for every stock. THE LEVEL FLOOR.

    The market return on a date is the EQUAL-WEIGHTED cross-sectional mean of
    the target, and this predicts the average of that over the training fold.
    Equal-weighted per date and then averaged, rather than a flat mean over all
    training rows: those differ whenever the panel is unbalanced, and it is —
    tickers enter at different dates, so a plain row mean tilts toward whichever
    names have the longest history.

    WHY THIS EXISTS AT ALL. 32.8% of 30-session return variance on this panel is
    common across stocks. A model that captured only that — no stock-specific
    information whatsoever — would still explain a third of the target and post
    an MAE far below `zero`. On the excess target that common component was
    subtracted out by the label itself and `zero` was the honest floor; on the
    absolute target it is not, and reporting against `zero` would credit every
    comparator with the market.

    A constant, so its rank IC is undefined. It bounds MAE, not ordering. For
    the ordering floor see ``BetaMarket``.
    """

    name = "market"

    def __init__(self) -> None:
        self.mu_ = 0.0

    @staticmethod
    def _market_by_date(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Equal-weighted cross-sectional mean of the target, per date."""
        frame = pd.DataFrame({
            "date": X["date"].to_numpy() if "date" in X.columns else 0,
            "y": pd.to_numeric(pd.Series(y).reset_index(drop=True),
                               errors="coerce").to_numpy(),
        })
        return frame.dropna(subset=["y"]).groupby("date")["y"].mean()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarketForecast":
        by_date = self._market_by_date(X, y)
        self.mu_ = float(by_date.mean()) if len(by_date) else 0.0
        if not np.isfinite(self.mu_):
            self.mu_ = 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mu_, dtype=float)


class BetaMarket:
    """
    Predicts ``beta_i * mu_market``. THE ORDERING FLOOR, and the important one.

    ``beta_i`` is the OLS slope of ticker i's target on the market's, fitted on
    the training fold; ``mu_market`` is the training-period mean market return.
    Both quantities are contemporaneous in label space — the market return on a
    date is the cross-section's mean 30-session forward return, the same window
    the label spans — so no future information enters.

    THIS IS THE COMPARATOR THE TARGET SWITCH MADE NECESSARY. Unlike ``market``
    it is NOT constant within a date: it orders names by beta. And in a market
    that drifted up over the training fold, mu_market is positive, so sorting
    by beta produces a positive cross-sectional rank IC — from a quantity that
    contains no view about any individual company. A model whose IC does not
    exceed this one has demonstrated nothing except that it noticed which
    stocks are volatile.

    Beta is per-ticker and persistent, which puts this in the same family as the
    valuation result the project already retired: a PERSISTENT per-ticker
    feature earns a positive t-statistic from nothing, measured at a mean
    rebalance t of +0.77 over 24 draws of random per-ticker constants. The
    difference is that beta is not a placebo — it is the real mechanism, so it
    belongs in the table rather than in a separate null.

    A ticker unseen in training gets beta 1.0: the market's own forecast, which
    is the least informative available answer rather than zero (which would
    assert a view) or a skipped row (which would change the sample).
    """

    name = "beta_market"

    def __init__(self, min_obs: int = 20) -> None:
        self.min_obs = min_obs
        self.mu_ = 0.0
        self.beta_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BetaMarket":
        if "date" not in X.columns or "ticker" not in X.columns:
            return self

        frame = pd.DataFrame({
            "date": X["date"].to_numpy(),
            "ticker": X["ticker"].to_numpy(),
            "y": pd.to_numeric(pd.Series(y).reset_index(drop=True),
                               errors="coerce").to_numpy(),
        }).dropna(subset=["y"])
        if frame.empty:
            return self

        market = frame.groupby("date")["y"].mean().rename("mkt")
        self.mu_ = float(market.mean())
        if not np.isfinite(self.mu_):
            self.mu_ = 0.0

        joined = frame.join(market, on="date")
        for ticker, group in joined.groupby("ticker", sort=False):
            x = group["mkt"].to_numpy(dtype=float)
            t = group["y"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(t)
            # np.ptp guards the degenerate case a single training date
            # produces: one market value, zero variance, and a slope that is
            # either a divide-by-zero or an arbitrarily large number fitted to
            # noise.
            if ok.sum() < self.min_obs or np.ptp(x[ok]) == 0:
                continue
            var = float(np.var(x[ok]))
            if var <= 0:
                continue
            beta = float(np.cov(x[ok], t[ok], bias=True)[0, 1] / var)
            if np.isfinite(beta):
                self.beta_[str(ticker)] = beta
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "ticker" not in X.columns:
            return np.full(len(X), self.mu_, dtype=float)
        betas = np.array([self.beta_.get(str(t), 1.0) for t in X["ticker"]],
                         dtype=float)
        return betas * self.mu_


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


class NewsAugmentedFactor(LinearFactorModel):
    """
    The technical factor set PLUS the news columns, fitted the same way.

    A SUBCLASS AND NOT A FLAG, so the base comparator's columns are untouched
    and the two rows in the table differ by exactly one thing. `LinearFactorModel`
    reads `self.columns`, NOT `X` — passing extra columns through `feature_cols`
    alone silently does nothing, which is how a `linear_factor+val` row once came
    out identical to `linear_factor` to five decimal places and read as
    "valuation does not help" rather than "valuation was never supplied". The
    extra columns therefore reach the CONSTRUCTOR here.

    NULL IS NOT ZERO, AND FILLING IT IS THE WHOLE PROBLEM. A row with no news
    has no measurement; `LinearFactorModel.fit` fills NaN with 0.0, which for a
    signed sentiment score means "measured as neutral". So the news columns are
    accompanied by `news_observed`, an explicit indicator, and the model can
    learn that the zero is a placeholder rather than a reading. Without it, the
    early panel — where coverage is ~1 article per ticker-month — would be
    training the model that the market was permanently neutral before 2022.
    """

    name = "news_factor"

    def __init__(self, alpha: float = 1.0, columns: list[str] | None = None) -> None:
        from pipeline.news_features import NEWS_COLS
        cols = list(columns) if columns else list(FACTORS) + list(NEWS_COLS) + ["news_observed"]
        super().__init__(alpha=alpha, columns=cols)


class RegimeAugmentedFactor(LinearFactorModel):
    """
    The technical factor set PLUS the beta x regime interactions.

    Only interactions, never the market state itself: every ticker sees the
    same market on a date, so a bare regime column is identically zero after
    `cross_sectional_zscore` and carries no ranking information. That is not a
    theory — `fii_net_flow` sat in FEATURES for years being exactly that.
    """

    name = "regime_factor"

    def __init__(self, alpha: float = 1.0, columns: list[str] | None = None) -> None:
        from pipeline.regime import REGIME_INTERACTIONS
        cols = list(columns) if columns else list(FACTORS) + list(REGIME_INTERACTIONS)
        super().__init__(alpha=alpha, columns=cols)


# ── Registry ──────────────────────────────────────────────────────────────────

# What every Phase 2 result is reported against. Ordered from least to most
# informed, so a comparison table reads top to bottom as increasing claim
# strength: if a row does not beat the row above it, the extra machinery in it
# is not earning its place.
BASELINES: dict[str, callable] = {
    "zero":          ZeroForecast,
    "always_up":     AlwaysUp,
    "train_mean":    TrainMeanForecast,
    "majority":      MajorityDirection,
    "market":        MarketForecast,
    "beta_market":   BetaMarket,
    "momentum_20d":  lambda: SingleFactor("sector_rel_20d"),
    "reversal_5d":   lambda: SingleFactor("lag5_ret"),
    "linear_factor": LinearFactorModel,
    "news_factor":   NewsAugmentedFactor,
    "regime_factor": RegimeAugmentedFactor,
}

#: What a comparator has to beat before its number means anything. NOT `zero`.
#:
#: `zero` was the right floor for an excess return: predicting no excess return
#: is the claim "this stock tracks its benchmark", the label already had the
#: market subtracted out of it, and beating zero therefore required saying
#: something about the individual company. On an absolute return none of that
#: holds. 57.67% of the labels are positive and 32.8% of their variance is
#: shared, so `zero` is beaten by drift alone and a comparator that clears it
#: has shown only that it noticed the market goes up.
#:
#: Two floors, because they bound different things and a comparator can clear
#: one while failing the other:
#:
#:   market       the LEVEL. A constant, so it has no ordering and its role is
#:                MAE. Beating it means the forecast's magnitude carries
#:                something beyond the average.
#:   beta_market  the ORDERING. Ranks by beta, which is information-free about
#:                any individual company, and scores a positive rank IC
#:                whenever the training market drifted up. Beating it is what
#:                separates stock-specific skill from beta.
#:
#: `zero` and `always_up` stay in the table and stay reported. They are the
#: DEGENERATE references - the arithmetic floor and the directional floor - and
#: dropping them would hide the size of the gift the target switch handed every
#: model.
FLOORS: tuple[str, ...] = ("market", "beta_market")


def baseline_feature_columns(name: str) -> list[str]:
    """
    Columns a given baseline needs handed to it.

    Kept explicit so the harness passes each comparator the same frame it would
    receive in production, rather than quietly widening the feature set for the
    ones that can cope with it.
    """
    if name == "linear_factor":
        return list(FACTORS)
    if name == "news_factor":
        from pipeline.news_features import NEWS_COLS
        return list(FACTORS) + list(NEWS_COLS) + ["news_observed"]
    if name == "regime_factor":
        from pipeline.regime import REGIME_INTERACTIONS
        return list(FACTORS) + list(REGIME_INTERACTIONS)
    if name.startswith("momentum"):
        return ["sector_rel_20d"]
    if name.startswith("reversal"):
        return ["lag5_ret"]
    if name in ("market", "beta_market"):
        # Identifiers, not indicators - the same shape SeriesAdapter takes.
        # `market` needs `date` to weight each cross-section equally rather
        # than each row; `beta_market` needs both to regress a ticker's target
        # on the market's.
        return ["date", "ticker"]
    return []


# ── The comparison ────────────────────────────────────────────────────────────


@dataclass
class BaselineComparison:
    """
    Every comparator scored on one set of folds over one set of rows.

    `note` is not decoration. A comparison run on a panel too thin to rank
    returns an empty `results` and an explanation, rather than a table of zeros
    in the shape of a result — the same refusal the CLI makes, moved here so
    that the weekly job inherits it instead of reimplementing it.
    """

    coverage: dict = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    loadings: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def ranked(self) -> bool:
        return bool(self.results)

    def best(self, metric: str = "daily_rank_ic") -> dict | None:
        """The strongest comparator on `metric`, ignoring the degenerate ones."""
        scored = [r for r in self.results
                  if np.isfinite(r.get(metric, np.nan))]
        return max(scored, key=lambda r: r[metric]) if scored else None

    def to_metrics(self) -> dict:
        """
        The compact form written into ``experiment_runs.metrics``.

        Deliberately not a new table. ``experiment_runs`` already carries
        ``config_hash`` and ``data_hash`` beside every row, which is the whole
        mechanism for telling whether a metric moved because the code changed
        or because the data did. A separate baseline table would either
        duplicate those hashes or lose that attribution, and would need its own
        writer, its own migration and its own entry in the landmines list.
        """
        keep = ("name", "n_oos", "folds", "daily_rank_ic", "rebalance_ic_t",
                "hit_rate", "majority_hit_rate", "mae", "mae_naive_zero",
                "beats_naive_mae", "mae_vs_market", "beats_market",
                "beats_beta_ic", "clears_floor",
                "alpha_vs_equal_weight", "alpha_t",
                "n_rebalances")
        return {
            "panel_tickers": self.coverage.get("tickers", 0),
            "panel_dates": self.coverage.get("dates", 0),
            "panel_labelled_rows": self.coverage.get("labelled_rows", 0),
            "median_names_per_date": self.coverage.get("median_names_per_date", 0.0),
            "note": self.note,
            "comparators": [{k: r.get(k) for k in keep} for r in self.results],
            "loadings": self.loadings,
        }

    def floors(self) -> dict:
        """The two numbers every comparator is measured against. See FLOORS."""
        market = next((r for r in self.results if r["name"] == "market"), {})
        beta = next((r for r in self.results if r["name"] == "beta_market"), {})
        return {
            "market_mae": market.get("mae"),
            "beta_market_ic": beta.get("rebalance_ic"),
            "beta_market_ic_t": beta.get("rebalance_ic_t"),
        }

    def summary(self) -> str:
        if not self.ranked:
            return f"baselines: not scored - {self.note}"
        best = self.best()
        f = self.floors()
        cleared = [r["name"] for r in self.results if r.get("clears_floor")]
        # `zero` is reported beside them rather than as the floor. It is the
        # degenerate reference now: on an absolute return it is beaten by drift,
        # and a summary that led with it would overstate every comparator.
        zero = next((r for r in self.results if r["name"] == "zero"), {})

        def _fmt(value, spec):
            return "n/a" if value is None or not np.isfinite(value) else format(value, spec)

        return (
            f"baselines: {len(self.results)} comparators over "
            f"{self.coverage.get('tickers', 0)} tickers; best daily IC "
            f"{best['name']} {best['daily_rank_ic']:+.4f} "
            f"(t {best.get('rebalance_ic_t', float('nan')):+.2f}); "
            f"floors market MAE {_fmt(f['market_mae'], '.5f')} / "
            f"beta_market reb_IC {_fmt(f['beta_market_ic'], '+.4f')} "
            f"(zero MAE {_fmt(zero.get('mae'), '.5f')}); "
            f"cleared BOTH floors: {cleared or 'none'}"
        )


# Fraction of labelled rows that must carry benchmark_close before the series
# comparators are scored at all. Below this the relative-price series has holes
# large enough that an abstention, not a forecast, is what would be measured.
MIN_BENCHMARK_LEVEL_COVERAGE = 0.90


def _pooled_xgb_factory():
    """
    An untuned pooled gradient-boosted tree.

    Untuned on purpose. This row answers "does the extra capacity beat a ridge
    on the same columns", and a searched tree could not be read beside an
    unsearched linear model without deflating its result for the search first
    (see ``evaluation.deflated_sharpe_note``).
    """
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, tree_method="hist",
    )


def fit_factor_loadings(panel: pd.DataFrame, min_train: int = 500,
                        horizon: int = HORIZON_SESSIONS,
                        columns: list[str] | None = None) -> dict[str, float]:
    """
    Fits the linear comparator once on the earliest training window, to read.

    Inputs are already standardised within each date, so the coefficients are
    directly comparable to one another — which is the only reason printing them
    beside each other means anything.
    """
    grid = sorted(panel["date"].unique())
    cut_at = min_train - horizon * 2
    if cut_at <= 0 or len(grid) <= min_train:
        return {}
    train = panel[(panel["date"] < grid[cut_at]) & panel[TARGET].notna()]
    if len(train) < 100:
        return {}
    # Same default-columns trap as the comparator: LinearFactorModel reads
    # self.columns, not X. Left unpassed, this table silently omits every
    # column outside FACTORS while claiming to report the fitted model.
    cols = list(columns) if columns else list(FACTORS)
    cols = [c for c in cols if c in train.columns]
    return LinearFactorModel(columns=cols).fit(train[cols], train[TARGET]).coefficients()


def annotate_against_floors(results: list[dict]) -> list[dict]:
    """
    Adds each comparator's standing against `market` and `beta_market`.

    Done HERE and not inside the per-comparator report, because a floor is a
    relation between comparators and the report only ever sees one. That is
    also why `beats_naive_mae` could be computed row by row: `zero`'s MAE is
    just mean(|y|), knowable without running `zero` at all. The market's is not.

    Three fields are added:

      mae_vs_market   percentage by which this comparator's MAE exceeds the
                      market's. NEGATIVE is better. Reported rather than
                      thresholded because the size is the interesting part.
      beats_market    MAE strictly below the market forecast's.
      beats_beta_ic   rebalance IC strictly above beta_market's.
      clears_floor    both of the above. THE P1 CRITERION.

    A missing floor leaves the fields None rather than defaulting to False. A
    run that did not score `market` cannot say whether anything beat it, and
    False would read as "measured, and it lost".
    """
    def _finite(value):
        return value is not None and np.isfinite(value)

    by_name = {r["name"]: r for r in results}
    market = by_name.get("market")
    beta = by_name.get("beta_market")

    market_mae = market.get("mae") if market else None
    beta_ic = beta.get("rebalance_ic") if beta else None

    for r in results:
        r["mae_vs_market"] = (
            float((r["mae"] / market_mae - 1.0) * 100.0)
            if _finite(market_mae) and _finite(r.get("mae")) and market_mae > 0
            else None
        )
        r["beats_market"] = (
            bool(r["mae"] < market_mae)
            if _finite(market_mae) and _finite(r.get("mae")) else None
        )
        r["beats_beta_ic"] = (
            bool(r["rebalance_ic"] > beta_ic)
            if _finite(beta_ic) and _finite(r.get("rebalance_ic")) else None
        )
        r["clears_floor"] = (
            bool(r["beats_market"] and r["beats_beta_ic"])
            if r["beats_market"] is not None and r["beats_beta_ic"] is not None
            else None
        )
    return results


def compare_baselines(
    tickers: list[str] | None = None,
    engine=None,
    start: str | None = None,
    n_folds: int = 5,
    min_train: int = 500,
    with_pooled_xgb: bool = True,
    with_series: bool = True,
    with_chronos: bool = False,
    chronos_context: int = SERIES_CONTEXT,
    with_timesfm: bool = False,
    timesfm_context: int = SERIES_CONTEXT,
    with_kronos: bool = False,
    kronos_models: Sequence[tuple[str, int]] = (),
    kronos_seed: int = 0,
    kronos_sample_count: int = 1,
    rebalance_only: bool = False,
    horizon: int = HORIZON_SESSIONS,
    with_fundamentals: bool = False,
    with_news: bool = False,
    with_regime: bool = False,
    on_result=None,
    max_tickers: int | None = None,
    allow_thin: bool = False,
) -> BaselineComparison:
    """
    Loads the panel and scores every comparator on identical purged folds.

    ``on_result`` is called with the running results list after EVERY
    comparator finishes. A foundation-model table is a two-hour run and the
    return value is the only thing that carries results out, so an
    interruption at comparator 13 of 13 discards all twelve that succeeded.
    That happened. The callback lets a caller persist as it goes.

    Read only. Nothing here writes to the database — the caller decides what to
    record — which is what makes it safe to run inside a job whose expensive
    work has already been persisted.
    """
    if with_kronos and not rebalance_only:
        # NOT a default that can be forgotten. Kronos decodes autoregressively
        # with no KV cache — 30 sequential passes over the full context per
        # date — and it MEASURED 643 s for one 46-ticker cross-section at
        # context 512 on CPU against a ridge's ~1 s for its entire run. The
        # full ~2,400-date grid is 429 hours. Refused rather than allowed with
        # a warning, because the warning would be read after the run started.
        raise ValueError(
            "with_kronos requires rebalance_only=True. Scoring every date "
            "costs ~429 CPU-hours and buys only `daily_IC`, which cannot "
            "support inference — ~1,900 overlapping dates hold ~60 independent "
            "windows. The rebalance grid preserves reb_IC and its t-statistic, "
            "which is the pre-registered criterion, exactly."
        )

    if (with_chronos or with_timesfm or with_kronos) and not with_series:
        # Chronos reads the relative-price series and nothing else, so this
        # combination cannot mean anything. Raised rather than silently
        # dropped: a run that was asked for a foundation model and quietly
        # returned six linear comparators reads as "it did not help".
        raise ValueError(
            "a foundation-model comparator requires with_series=True: "
            "Chronos-2, TimesFM-2.5 and Kronos all rest on the price series "
            "panel.price_frame builds for the current target — Kronos on a "
            "candle derived from it — and with_series=False switches off the "
            "check that says the series exists"
        )

    panel = load_panel(tickers=tickers, engine=engine, start=start)
    if panel.empty:
        return BaselineComparison(note="no signals rows in the database")

    # The stored label is the 30-session return. Any other horizon is derived
    # exactly from the price-series identity - see panel.retarget_horizon. Done
    # BEFORE coverage and z-scoring so every downstream statistic describes the
    # horizon actually being scored.
    if horizon != HORIZON_SESSIONS:
        panel = retarget_horizon(panel, horizon, target=TARGET)

    # Valuation is the first information here that is not a transform of the
    # same OHLCV series. Restricting the panel to rows that CARRY it is the
    # point: the comparison has to be with and without fundamentals over
    # identical rows, or it measures the sample change instead.
    fundamental_cols: list[str] = []
    if with_fundamentals:
        from pipeline.fundamentals import (
            FUNDAMENTAL_COLS, attach_fundamentals, fundamental_coverage)

        panel = attach_fundamentals(panel, engine)
        fcov = fundamental_coverage(panel)
        panel = panel[panel[FUNDAMENTAL_COLS].notna().all(axis=1)].reset_index(drop=True)
        if panel.empty:
            return BaselineComparison(
                note=('no row carries a complete set of fundamentals; run pipeline.fundamentals.sync_fundamentals first'))
        fundamental_cols = list(FUNDAMENTAL_COLS)
        logger.info(
            f"[Baselines] fundamentals cover {fcov['fraction']:.1%} of rows; "
            f"panel restricted to {len(panel):,} rows from "
            f"{panel['date'].min()}")

    # NEWS AND REGIME DO NOT RESTRICT THE PANEL, and that is the opposite of
    # what `with_fundamentals` does above. Valuation is either present for a row
    # or the row cannot be scored on it, so restricting is correct there. News
    # coverage instead GROWS WITH TIME - measured, ~1 article per ticker-month
    # in 2016 against ~7 in 2024 - so restricting to covered rows would silently
    # delete the early panel and leave a comparison run entirely on the recent
    # period. That is the valuation post-mortem's lesson stated forwards: a
    # sweep that changes the row count measures two things at once.
    extra_cols: list[str] = []
    if with_news:
        from pipeline.news_features import NEWS_COLS
        panel = attach_news(panel, engine)
        # AN EXPLICIT "WE LOOKED" INDICATOR. The model fills NaN with 0.0, and
        # for a signed sentiment score 0.0 means "measured as neutral" — so
        # without this the early panel would teach it that the market was
        # permanently neutral before 2022. Same None-vs-0.0 rule that made
        # get_aggregate_sentiment return None.
        #
        # Derived from the COUNT column, not the sentiment column. A window we
        # searched that happened to be quiet has a count and no sentiment;
        # keying this on sentiment would file it under "never looked" and throw
        # away the very distinction `news_coverage` exists to preserve.
        panel["news_observed"] = panel["news_count_excess"].notna().astype(float)
        extra_cols += list(NEWS_COLS)
        covered = float(panel["news_sent_mean"].notna().mean())
        logger.info(f"[Baselines] news covers {covered:.1%} of panel rows")
    if with_regime:
        from pipeline.regime import REGIME_INTERACTIONS
        panel = attach_regime(panel, engine)
        extra_cols += list(REGIME_INTERACTIONS)

    if max_tickers:
        counts = panel.groupby("ticker")["date"].count().sort_values(ascending=False)
        panel = panel[panel["ticker"].isin(counts.head(max_tickers).index)]

    panel = cross_sectional_zscore(panel, SCALE_FREE + fundamental_cols + extra_cols)
    coverage = panel_coverage(panel)

    if coverage["median_names_per_date"] < MIN_NAMES_PER_DATE and not allow_thin:
        return BaselineComparison(
            coverage=coverage,
            note=(f"panel too thin to rank: the median date holds "
                  f"{coverage['median_names_per_date']:.0f} names, below the "
                  f"{MIN_NAMES_PER_DATE} needed. Every feature is zeroed at "
                  f"that breadth, so the comparison would be between constants."),
        )

    splitter = PurgedPanelWalkForward(
        n_folds=n_folds, horizon=horizon,
        embargo=horizon, min_train=min_train,
    )

    # baseline_feature_columns, NOT a blanket FACTORS. That function has always
    # existed and declared exactly this, and this line ignored it: every
    # baseline received the full factor set regardless of what it asked for.
    #
    # It was invisible while every comparator either ignored X entirely (`zero`,
    # `train_mean`, `majority`) or read one named column out of it (`momentum`,
    # `reversal`) - a widened frame changes nothing for either. `market` and
    # `beta_market` are the first comparators that need columns FACTORS does not
    # contain, and the symptom was not an error: BetaMarket fitted no betas,
    # defaulted every ticker to 1.0, emitted a constant, and was recorded with
    # `n_dates_no_ordering` on every date and a blank rank IC. A floor that
    # silently reports "no ordering" is a floor nothing can fail to clear.
    #
    # Same shape as the LinearFactorModel landmine: a comparator that does not
    # receive the columns it needs produces a complete, plausible row.
    runs: list[tuple[str, object, list[str]]] = [
        (name, factory, baseline_feature_columns(name) or FACTORS)
        for name, factory in BASELINES.items()
    ]
    if with_pooled_xgb:
        runs.append(("pooled_xgb", _pooled_xgb_factory, FACTORS))

    # Added as SEPARATE rows rather than by widening FACTORS, so the table
    # carries the with/without contrast on identical folds and rows. A
    # comparator that improved because the sample moved would otherwise be
    # indistinguishable from one that improved because valuation helped.
    if fundamental_cols:
        with_val = FACTORS + fundamental_cols
        # LinearFactorModel defaults its column list to FACTORS and reads only
        # those, so the extra columns must be passed to the CONSTRUCTOR and not
        # merely present in X. Handing them only through feature_cols produced a
        # row identical to `linear_factor` to five decimal places.
        runs.append(("linear_factor+val",
                     lambda cols=with_val: LinearFactorModel(columns=cols),
                     with_val))
        if with_pooled_xgb:
            runs.append(("pooled_xgb+val", _pooled_xgb_factory, with_val))

    # Series comparators read a PRICE SERIES, not the feature matrix. They are
    # added only when that series actually exists.
    #
    # A SeriesAdapter handed an all-NaN series does not fail — it declines every
    # ticker and predicts 0.0, which is the correct default for an abstention
    # and which would appear in this table as a row indistinguishable from
    # `zero`. So a run whose series could not be built would report three extra
    # comparators that all silently measured nothing.
    #
    # WHICH series is decided by the TARGET, never chosen here. `price_frame`
    # returns log(close) for the absolute target and log(close / benchmark)
    # for the excess one, and in both cases the h-session forward difference of
    # what it returns IS the label being scored. Passing the relative series
    # while scoring the absolute label is the failure this indirection exists
    # to make unconstructable: it raises nothing, the table renders, the
    # magnitudes look right, and the only symptom is a foundation model that
    # inexplicably will not beat the floor.
    #
    # The benchmark-coverage gate therefore applies only to the excess target.
    # The absolute series needs `close`, which every row has by construction,
    # so the vendor outage that emptied eight of ten NSE sector indices cannot
    # reach it.
    series_note = ""
    labelled = max(coverage.get("labelled_rows", 0), 1)
    level_coverage = coverage.get("rows_with_benchmark_close", 0) / labelled
    needs_benchmark = TARGET == EXCESS_TARGET

    if not with_series:
        series_note = "series comparators disabled by the caller"
    elif needs_benchmark and level_coverage < MIN_BENCHMARK_LEVEL_COVERAGE:
        series_note = (
            f"series comparators skipped: benchmark_close is present on "
            f"{level_coverage:.1%} of labelled rows, below the "
            f"{MIN_BENCHMARK_LEVEL_COVERAGE:.0%} needed. Without the benchmark "
            f"LEVEL there is no relative-price series, and every series "
            f"forecaster would abstain and report as `zero`. Recompute signals."
        )
    else:
        series = price_frame(panel, TARGET)
        for name, cls in SERIES_BASELINES.items():
            runs.append((name, adapter_factory(cls, series, horizon=horizon),
                         ["date", "ticker"]))

        # Chronos-2 rides the same series and the same folds as the three
        # known-answer forecasters above, which is the point: `series_zero`
        # reproducing `zero` exactly on this very panel is what licenses
        # reading a foundation model's row as the model rather than the
        # plumbing. Off by default because it needs torch — see
        # requirements-series.txt — and every other caller of this function
        # runs where torch is deliberately absent.
        if with_chronos:
            from pipeline.chronos_forecaster import (
                CHRONOS_VARIANTS, Chronos2Forecaster)

            for name, kwargs in CHRONOS_VARIANTS.items():
                runs.append((
                    name,
                    adapter_factory(Chronos2Forecaster, series,
                                    horizon=horizon,
                                    context=chronos_context, **kwargs),
                    ["date", "ticker"],
                ))

        # TimesFM-2.5 is a DIFFERENT ARCHITECTURE, not a different size of the
        # same one: 200M decoder-only from Google against 28M encoder-only from
        # Amazon, different corpus, different objective. Two independent
        # architectures agreeing on this panel is evidence about the target
        # rather than about either model, which is the whole reason it is here.
        if with_timesfm:
            from pipeline.timesfm_forecaster import (
                TIMESFM_VARIANTS, TimesFM25Forecaster)

            for name, kwargs in TIMESFM_VARIANTS.items():
                runs.append((
                    name,
                    adapter_factory(TimesFM25Forecaster, series,
                                    horizon=horizon,
                                    context=timesfm_context, **kwargs),
                    ["date", "ticker"],
                ))

        # Kronos is the FINANCE-PRETRAINED comparator: 12B K-line records from
        # 45 exchanges, against Chronos' and TimesFM's general time-series
        # corpora. Two general architectures already agree this target is not
        # forecastable zero-shot, so a model trained on candlesticks is the one
        # remaining argument that the CORPUS rather than the TARGET was the
        # limitation. It rides its own adapter because it is multivariate — see
        # pipeline/kronos_forecaster.py for the synthetic relative candle that
        # keeps the excess-return identity intact.
        if with_kronos:
            from pipeline.kronos_forecaster import (
                adapter_factory as kronos_factory, load_candles)

            # TARGET, not the relative candle: the basis must match the label
            # being scored, or Kronos measures a quantity nobody asked for.
            frames = load_candles(panel, engine=engine, target=TARGET)
            for model_id, ctx in kronos_models:
                factory = kronos_factory(
                    frames, horizon=horizon, context=ctx, model_id=model_id,
                    seed=kronos_seed, sample_count=kronos_sample_count)
                runs.append((factory().name, factory, ["date", "ticker"]))

    # ── The scoring grid, chosen ONCE and applied to every comparator ───────
    #
    # Derived from the splitter and the labels alone, so it is identical for
    # each row of the table by construction — an expensive model scored on a
    # subset and a ridge scored on everything would not be comparable, and the
    # difference would look like a result.
    score_dates = None
    rebalance_every = horizon
    if rebalance_only:
        grid = oos_dates(panel, splitter, TARGET)[::horizon]
        score_dates = set(grid.tolist())
        rebalance_every = 1
        logger.info("[Baselines] rebalance-only: %d of %d out-of-sample dates",
                    len(score_dates), len(oos_dates(panel, splitter, TARGET)))

    results = []
    for i, (name, factory, feature_cols) in enumerate(runs, 1):
        started = time.time()
        # A Chronos variant runs for the better part of an hour and prints
        # nothing while it does. On a workflow runner that is indistinguishable
        # from a hung step, and the reflex is to cancel it at 40 minutes.
        logger.info(f"[Baselines] {i}/{len(runs)} {name} ...")
        res = panel_walk_forward(
            panel=panel, feature_cols=feature_cols, model_factory=factory,
            splitter=splitter, name=name, target=TARGET,
            rebalance_every=rebalance_every, score_dates=score_dates,
        )
        m, xs = res.metrics, res.cross_sectional
        results.append({
            "name": name,
            "n_oos": res.n_predictions,
            "folds": res.n_folds_run,
            "rank_ic": m.get("rank_ic", float("nan")),
            "daily_rank_ic": m.get("daily_rank_ic", float("nan")),
            "hit_rate": m.get("hit_rate", float("nan")),
            "majority_hit_rate": m.get("majority_hit_rate", float("nan")),
            "mae": m.get("mae", float("nan")),
            "mae_naive_zero": m.get("mae_naive_zero", float("nan")),
            "beats_naive_mae": bool(m.get("beats_naive_mae", False)),
            "n_rebalances": xs.get("n_rebalances", 0),
            "n_dates_no_ordering": xs.get("n_dates_no_ordering", 0),
            "rebalance_ic": xs.get("mean_rank_ic", float("nan")),
            "rebalance_ic_t": xs.get("rank_ic_t", float("nan")),
            "alpha_vs_equal_weight": xs.get("alpha_vs_equal_weight", float("nan")),
            "alpha_t": xs.get("alpha_t", float("nan")),
            "long_short_spread": xs.get("long_short_spread", float("nan")),
            "spread_t": xs.get("spread_t", float("nan")),
            "seconds": round(time.time() - started, 1),
        })

        if on_result is not None:
            # Never let persistence kill the measurement it is recording.
            try:
                on_result(results, coverage)
            except Exception as exc:                            # noqa: BLE001
                logger.error(f"[Baselines] on_result failed: {str(exc)[:200]}")

    if not any(r["n_oos"] for r in results):
        return BaselineComparison(
            coverage=coverage, results=[],
            note=(f"no fold produced an out-of-sample prediction: the panel "
                  f"holds {coverage['dates']} dates against a min_train of "
                  f"{min_train}"),
        )

    return BaselineComparison(
        coverage=coverage,
        results=annotate_against_floors(results),
        loadings=fit_factor_loadings(panel, min_train, horizon,
                                     FACTORS + fundamental_cols),
        note=series_note,
    )
