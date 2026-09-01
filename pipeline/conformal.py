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
