"""
pipeline/label.py — the within-date standardised training target, and its inverse.

WHY THE TARGET IS STANDARDISED AT ALL
--------------------------------------
P6 measured the defect this fixes. ``gamma`` is the minimum loss reduction a
split must buy, **in the units of the loss**, and for an MAE objective those
units are the label's own scale. The label's dispersion falls with the horizon:

    h            5        10       20       30
    label sd     0.0476   0.0678   0.0973   0.1196

while ``tuning.tune_pooled`` searches ``gamma`` over a fixed ``[0, 5]`` at every
horizon. So at a short horizon every split looks unprofitable, the search walks
to the top of the range, and the model emits a constant. Measured at h=5: **five
distinct predicted values across 162,535 rows** — one per fold — and 340 of 420
(ticker, fold) cells constant. Refit on a standardised label, the same
configuration produced 0 of 420 constant cells at every horizon.

A within-date z-score is a positive affine map inside each date, so it changes
no cross-sectional ranking and no per-date rank IC of any fixed ordering. What
it changes is the scale the loss is measured on — which is the whole point, and
the reason this is a defect fix rather than a new model.

WHAT IT COSTS, AND IT IS NOT FREE
----------------------------------
The model now predicts RELATIVE CROSS-SECTIONAL POSITION, not an absolute
return. Getting back to a return needs that date's cross-sectional mean and
standard deviation — and at prediction time **both are unknown**, because they
are properties of the h-session window that has not happened yet. The mean is
the market's move over the next h sessions; the sd is the cross-section's
dispersion over it.

So the inverse has two forms and they are not interchangeable:

  - ``inverse_standardise`` with the REALISED moments. Exact, and the right
    thing for scoring history, backtests and any round-trip test.
  - ``inverse_standardise`` with ``causal_moments``. A trailing estimate from
    data available at t, and the only thing available when forecasting. It is
    an approximation, and the conformal layer is what makes its error
    measurable rather than assumed: calibrate on the residuals of the INVERTED
    prediction against the RAW label, and the interval absorbs the moment
    error along with everything else.

Never invert a live forecast with realised moments. That is the F1 shape — a
quantity from the future used to dress up a prediction of it — and it would
read as a large improvement in both MAE and coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.panel import MIN_NAMES_PER_DATE, TARGET

#: Columns carrying the moments the inverse needs, written alongside the
#: standardised target so a frame can always be turned back. A standardised
#: label with no moments beside it is a one-way door, and the round trip is
#: exactly what the conformal layer has to be able to make.
CS_MEAN = "target_cs_mean"
CS_SD = "target_cs_sd"
MOMENT_COLS = (CS_MEAN, CS_SD)

#: Trailing window for the causal moment estimate, in SESSIONS of the shared
#: date grid. One trading year: long enough that the dispersion estimate is not
#: itself noise, short enough to track a volatility regime. It is a lookback on
#: the moments, not on the label, so it costs no rows at the start of a fold —
#: the estimate simply uses whatever history exists, down to `MIN_MOMENT_DATES`.
MOMENT_LOOKBACK = 252
MIN_MOMENT_DATES = 20


def cross_sectional_moments(panel: pd.DataFrame,
                            target: str = TARGET,
                            min_names: int = MIN_NAMES_PER_DATE) -> pd.DataFrame:
    """
    Per-date mean and standard deviation of `target` across the cross-section.

    A date carrying fewer than `min_names` labelled observations gets NaN
    moments rather than a mean of six numbers — the same refusal
    ``cross_sectional_zscore`` makes, and for the same reason.
    """
    y = pd.to_numeric(panel[target], errors="coerce")
    frame = pd.DataFrame({"date": panel["date"].to_numpy(), "_y": y.to_numpy()})
    grouped = frame.groupby("date", sort=True)["_y"]
    out = pd.DataFrame({
        CS_MEAN: grouped.mean(),
        CS_SD: grouped.std(),
        "_n": grouped.count(),
    })
    thin = out["_n"] < min_names
    out.loc[thin, [CS_MEAN, CS_SD]] = np.nan
    return out.drop(columns="_n").reset_index()


def standardise_target(panel: pd.DataFrame, target: str = TARGET,
                       min_names: int = MIN_NAMES_PER_DATE) -> pd.DataFrame:
    """
    `target` replaced by its own within-date z-score, with the moments attached.

    A date whose cross-section is constant, or too thin to standardise, yields
    NaN rather than a fabricated ordering: dividing by a zero standard
    deviation must not invent one, and `run_arm` drops non-finite labels.
    """
    if all(c in panel.columns for c in MOMENT_COLS):
        # ALREADY STANDARDISED, so this is a no-op rather than a second pass.
        # The transform is very nearly idempotent — the z-score of a z-score is
        # itself — but a second pass would overwrite the moments with 0 and 1
        # and destroy the only route back to the target's own units. Detecting
        # it by the presence of the moment columns is what lets the transform
        # be a DEFAULT at the training entry point without every caller having
        # to know whether someone upstream already applied it.
        return panel

    out = panel.copy()
    moments = cross_sectional_moments(out, target=target, min_names=min_names)
    out = out.merge(moments, on="date", how="left")

    y = pd.to_numeric(out[target], errors="coerce")
    # THE INFINITY GUARD IS LOAD-BEARING, and it took a mutation to prove it.
    #
    # The obvious reading is that a constant date divides 0 by 0 and is NaN
    # already. That is true for most values and FALSE for the ones that matter:
    # the mean of twelve copies of 1/3 is not exactly 1/3, so the numerator is
    # about -5.5e-17 while the standard deviation rounds to exactly 0.0 — and
    # the result is **-inf on every name**, which is not a missing ordering but
    # a fabricated one, identical across the cross-section and extreme.
    #
    # Deleting this line left the whole suite green, which said the TEST was
    # missing rather than the guard being redundant. It is
    # `test_an_undefined_z_score_is_nan_and_never_infinite` now.
    out[target] = ((y - out[CS_MEAN]) / out[CS_SD]).replace(
        [np.inf, -np.inf], np.nan)
    return out


def inverse_standardise(z, mean, sd):
    """
    Back to the target's own units: ``z * sd + mean``.

    Element-wise over arrays or scalars. Exact when `mean` and `sd` are the
    realised moments of the date being inverted; approximate when they are
    `causal_moments` estimates, which is the only option at prediction time.
    """
    z = np.asarray(z, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    with np.errstate(invalid="ignore"):
        return z * sd + mean


def causal_moments(panel: pd.DataFrame, target: str = TARGET,
                   lookback: int = MOMENT_LOOKBACK,
                   min_dates: int = MIN_MOMENT_DATES,
                   horizon: int = 30,
                   min_names: int = MIN_NAMES_PER_DATE) -> pd.DataFrame:
    """
    Per-date moment estimates usable AT that date, for inverting a forecast.

    Two lags are applied and both are load-bearing:

    1. The label at date ``t`` spans ``[t, t + horizon]``, so its moments are
       not knowable until ``t + horizon``. The rolling window is therefore
       shifted by `horizon` dates — using date ``t``'s own realised moment to
       invert date ``t``'s forecast is using the answer.
    2. The window then averages the `lookback` dates before that, so the
       estimate is a trailing property of the panel rather than one period's
       accident.

    Returned for every date on the grid, NaN where there is not yet enough
    history. Callers fall back to the panel-wide trailing values, never to the
    full-sample ones.
    """
    moments = cross_sectional_moments(panel, target=target, min_names=min_names)
    moments = moments.sort_values("date").reset_index(drop=True)

    shifted = moments[[CS_MEAN, CS_SD]].shift(horizon)
    rolled = shifted.rolling(lookback, min_periods=min_dates).mean()

    out = pd.DataFrame({"date": moments["date"]})
    out[CS_MEAN] = rolled[CS_MEAN].to_numpy()
    out[CS_SD] = rolled[CS_SD].to_numpy()
    return out


def attach_causal_moments(panel: pd.DataFrame, target: str = TARGET,
                          **kwargs) -> pd.DataFrame:
    """The panel with causal moment columns joined on, for inverting."""
    est = causal_moments(panel, target=target, **kwargs)
    out = panel.drop(columns=[c for c in MOMENT_COLS if c in panel.columns])
    return out.merge(est, on="date", how="left")


def gamma_range_note(panel: pd.DataFrame, target: str = TARGET) -> dict:
    """
    Whether ``[0, 5]`` is still the right ``gamma`` search range once the label
    is standardised. Reported, not enforced.

    ``gamma`` is compared against the loss reduction a split buys, so the
    quantity that matters is the label's dispersion. Standardising sets the
    WITHIN-DATE sd to exactly 1 by construction; what this measures is the
    POOLED sd the tuner actually fits against, which is not 1 — dates with
    thin or constant cross-sections drop out, and the clipping of extreme
    z-scores is not applied to the target.
    """
    y = pd.to_numeric(panel[target], errors="coerce")
    y = y[np.isfinite(y)]
    return {
        "pooled_sd": float(y.std()) if len(y) else float("nan"),
        "mean_abs": float(y.abs().mean()) if len(y) else float("nan"),
        "n": int(len(y)),
        "gamma_range": (0.0, 5.0),
    }
