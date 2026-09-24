"""
pipeline/conformal.py — Distribution-free intervals and calibrated probabilities.

The old system reported a three-level "confidence" label derived by
thresholding MAPE and directional accuracy. Because both were leaked in-sample
figures, the label measured the evaluation bug rather than forecast
uncertainty, and it fired "High" for nearly every stock.

Split-conformal prediction replaces it with something checkable. Given
out-of-sample residuals from the purged walk-forward run, the interval

    [pred - q, pred + q],   q = the ceil((n+1)(1-alpha))/n quantile of |residual|

has finite-sample coverage of at least 1 - alpha under exchangeability, with no
distributional assumption. The same residual pool yields a calibrated
probability that the excess return is positive, which is what the dashboard
shows instead of a confidence word.

Coverage is an empirical claim, so ``check_coverage()`` measures it. If the
80% interval does not cover ~80% of held-out outcomes, that is a reportable
failure rather than something to hide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConformalCalibration:
    """Residual quantiles fitted on out-of-sample walk-forward residuals."""

    quantile: float                # half-width in log-excess-return units
    coverage: float                # nominal coverage, e.g. 0.80
    residuals: np.ndarray          # calibration pool, kept for probabilities
    n: int

    def interval(self, prediction: float) -> tuple[float, float]:
        """Prediction interval in log excess return space."""
        return prediction - self.quantile, prediction + self.quantile

    def prob_positive(self, prediction: float) -> float:
        """
        Calibrated probability that the realised excess return exceeds zero.

        The outcome is modelled as ``prediction + residual``, so
        P(outcome > 0) = P(residual > -prediction), estimated as the empirical
        fraction of calibration residuals above that threshold. Distribution
        free, and it degrades to 0.5 when the prediction is small relative to
        residual spread — which is the honest answer for a weak signal.
        """
        if self.n == 0:
            return 0.5
        return float(np.mean(self.residuals > -prediction))


def fit_conformal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverage: float = 0.80,
) -> ConformalCalibration | None:
    """
    Fits split-conformal calibration on out-of-sample predictions.

    The inputs MUST come from the purged walk-forward harness. Passing in-sample
    residuals produces intervals that are too narrow — the same class of error
    as F1, one layer up.
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    residuals = np.asarray(y_true)[valid] - np.asarray(y_pred)[valid]
    n = len(residuals)

    if n < 20:
        return None

    # Conformal quantile with the finite-sample correction.
    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        rank = n
    q = float(np.sort(np.abs(residuals))[rank - 1])

    return ConformalCalibration(quantile=q, coverage=coverage, residuals=residuals, n=n)


def check_coverage(
    calibration: ConformalCalibration,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Measures realised coverage on a held-out set.

    Reported alongside the forecast so the interval's claim is falsifiable.
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = np.asarray(y_true)[valid], np.asarray(y_pred)[valid]
    if len(yt) == 0:
        return {"n": 0}

    lo = yp - calibration.quantile
    hi = yp + calibration.quantile
    inside = (yt >= lo) & (yt <= hi)

    realised = float(np.mean(inside))
    return {
        "n": int(len(yt)),
        "nominal_coverage": calibration.coverage,
        "realised_coverage": realised,
        "coverage_gap_pp": float((realised - calibration.coverage) * 100),
        "well_calibrated": bool(abs(realised - calibration.coverage) <= 0.05),
    }


@dataclass
class ScaledConformalCalibration:
    """
    Split-conformal calibration on SPREAD-NORMALISED residuals.

    The score is ``(y - pred) / spread``, where `spread` is a PAST-ONLY
    estimate of the cross-sectional dispersion of the label on the forecast's
    date (``pooled.spread_frame``). The interval is ``pred ± quantile ×
    spread``: wide when the market is dispersed, narrow when it is calm.

    WHY (docs/stage2-fallback-conformal-preregistration.md §3). A constant
    half-width calibrated on the wild early folds over-covered the calm late
    ones: 0.868 overall and 0.903 in fold 4 against a nominal 0.80. The
    quantity that drifted is the dispersion, so the band is expressed in units
    of it. The spread must never be the date's REALISED dispersion — that is
    a property of the 30 sessions after the forecast.
    """

    quantile: float                # half-width in units of the spread
    coverage: float
    scores: np.ndarray             # normalised calibration residuals
    n: int

    def at(self, spread: float) -> ConformalCalibration | None:
        """The ordinary calibration for one date, in log-return units; None
        when that date's spread is unknown — an interval is withheld rather
        than priced at a made-up width."""
        spread = float(spread) if spread is not None else float("nan")
        if not np.isfinite(spread) or spread <= 0:
            return None
        return ConformalCalibration(quantile=self.quantile * spread,
                                    coverage=self.coverage,
                                    residuals=self.scores * spread, n=self.n)


def fit_scaled_conformal(y_true: np.ndarray, y_pred: np.ndarray,
                         spread: np.ndarray, coverage: float = 0.80
                         ) -> ScaledConformalCalibration | None:
    """`fit_conformal`'s rule — the same finite-sample rank — applied to the
    residuals divided by each row's past-only spread."""
    y_true, y_pred, spread = (np.asarray(a, dtype=float) for a in (y_true, y_pred, spread))
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(spread) & (spread > 0)
    scores = (y_true[ok] - y_pred[ok]) / spread[ok]
    base = fit_conformal(scores, np.zeros_like(scores), coverage=coverage)
    if base is None:
        return None
    return ScaledConformalCalibration(quantile=base.quantile, coverage=coverage,
                                      scores=base.residuals, n=base.n)


def expanding_fold_coverage(y_true: np.ndarray, y_pred: np.ndarray,
                            folds: np.ndarray, coverage: float = 0.80,
                            spread: np.ndarray | None = None,
                            price: np.ndarray | None = None) -> dict:
    """
    Coverage measured the only honest way on a walk-forward: for each fold k,
    calibrate on every fold BEFORE k and check on k. Plus the pooled figure over
    every checked row.

    Checking on the calibration pool itself reports the quantile's own
    definition back and always looks like a pass. And the per-fold split is
    what exposes drift: this panel's target dispersion falls across folds, so a
    band calibrated on the wilder early period over-covers the calmer late one
    — which the pooled number averages away (Hygiene, 2026-09-21: 0.8166 at
    the first checkable fold, 0.87-0.91 after).

    `spread` (optional): score and scale by a PAST-ONLY dispersion estimate per
    row (`ScaledConformalCalibration`); rows without a positive spread are
    excluded. `price` (optional): the close at the forecast date, so coverage
    is checked in PRICE space, ``P·exp(lo) <= P·exp(y) <= P·exp(hi)`` — the
    same event as log-return coverage, since the map is monotone — and the
    width is reported as a percentage of that price.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    folds = np.asarray(folds)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if spread is not None:
        spread = np.asarray(spread, dtype=float)
        ok &= np.isfinite(spread) & (spread > 0)
    if price is not None:
        price = np.asarray(price, dtype=float)
        ok &= np.isfinite(price) & (price > 0)
    per_fold, inside_all = [], []
    for k in sorted({int(f) for f in np.unique(folds[ok])}):
        earlier, here = ok & (folds < k), ok & (folds == k)
        if earlier.sum() < 20 or here.sum() < 20:
            continue
        if spread is None:
            cal = fit_conformal(y_true[earlier], y_pred[earlier], coverage=coverage)
            if cal is None:
                continue
            q = cal.quantile
            half = np.full(int(here.sum()), q)
        else:
            cal = fit_scaled_conformal(y_true[earlier], y_pred[earlier],
                                       spread[earlier], coverage=coverage)
            if cal is None:
                continue
            q = cal.quantile
            half = q * spread[here]
        yt, yp = y_true[here], y_pred[here]
        if price is None:
            inside = np.abs(yt - yp) <= half
        else:
            p = price[here]
            inside = ((p * np.exp(yp - half) <= p * np.exp(yt))
                      & (p * np.exp(yt) <= p * np.exp(yp + half)))
        inside_all.append(inside)
        row = {"fold": k, "n": int(here.sum()),
               "coverage": float(inside.mean()),
               "quantile": q,
               "n_calibration": int(earlier.sum()),
               "mean_width_log": float(np.mean(2 * half))}
        if price is not None:
            row["mean_width_pct"] = float(np.mean(
                (np.exp(yp + half) - np.exp(yp - half)) * 100))
        per_fold.append(row)
    pooled = np.concatenate(inside_all) if inside_all else np.zeros(0, dtype=bool)
    return {"nominal": coverage, "per_fold": per_fold,
            "method": "constant" if spread is None else "spread-normalised",
            "overall": float(pooled.mean()) if pooled.size else float("nan"),
            "n_checked": int(pooled.size)}


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Brier score for the P(excess return > 0) forecasts. Lower is better."""
    valid = np.isfinite(y_true) & np.isfinite(probabilities)
    if valid.sum() == 0:
        return float("nan")
    outcomes = (np.asarray(y_true)[valid] > 0).astype(float)
    return float(np.mean((np.asarray(probabilities)[valid] - outcomes) ** 2))


def calibration_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Predicted probability versus observed frequency, for a reliability plot."""
    valid = np.isfinite(y_true) & np.isfinite(probabilities)
    df = pd.DataFrame({
        "p": np.asarray(probabilities)[valid],
        "outcome": (np.asarray(y_true)[valid] > 0).astype(float),
    })
    if df.empty:
        return pd.DataFrame(columns=["bin_mid", "predicted", "observed", "n"])

    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), observed=("outcome", "mean"), n=("outcome", "size")
    ).reset_index()
    grouped["bin_mid"] = grouped["bin"].apply(lambda b: (b.left + b.right) / 2)
    return grouped[["bin_mid", "predicted", "observed", "n"]]


# ── Presentation ──────────────────────────────────────────────────────────────


def to_price_view(
    current_price: float,
    pred_return: float,
    calibration: ConformalCalibration | None,
) -> dict:
    """
    Converts a log return forecast into the rupee view the dashboard shows.

    ``pred_return`` is the ABSOLUTE 30-session log return since P1, so the
    implied price is a plain price target and needs no caveat about the index.
    That is a real simplification and worth stating: the previous version
    forecast an EXCESS return, from which a rupee figure could only be derived
    by ASSUMING the benchmark stayed flat - an assumption nobody believes, that
    had to travel with every number, and that made the headline figure on every
    stock page conditional on something the model had no view about.

    ``prob_up`` is P(the stock rises), not P(it beats its benchmark). Read it
    against 57.67%, the measured unconditional rate of a positive 30-session
    return on this universe - NOT against 50%. A 0.55 here is BEARISH.
    """
    implied = float(current_price * math.exp(pred_return))

    view = {
        "current_price": float(current_price),
        "pred_return": float(pred_return),
        "implied_price": implied,
        "implied_change_pct": float((implied / current_price - 1) * 100),
        "random_walk_price": float(current_price),
        "assumption": "Implied price is the model's point forecast; the interval around it is the calibrated part.",
    }

    if calibration is None:
        view.update({
            "interval_low": None,
            "interval_high": None,
            "interval_coverage": None,
            "prob_up": None,
            "note": "Not enough out-of-sample residuals to calibrate an interval.",
        })
        return view

    lo, hi = calibration.interval(pred_return)
    view.update({
        "interval_low": float(current_price * math.exp(lo)),
        "interval_high": float(current_price * math.exp(hi)),
        "interval_coverage": calibration.coverage,
        "prob_up": calibration.prob_positive(pred_return),
    })
    return view
