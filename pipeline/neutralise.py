"""
pipeline/neutralise.py — strip the beta channel and see what ordering is left.

THE QUESTION THIS EXISTS TO ANSWER
-----------------------------------
Phases 2, 3 and 4 all closed as nulls and they closed on the SAME comparator.
`beta_market` — which sorts by beta and holds no company-specific view — outranks
everything measured on this panel (reb_IC +0.0464 against `pooled_xgb`'s +0.0389),
and P4 restated it in money: of the +8.95%/yr the best long-only book made over
the equal-weighted floor, the pure beta sort took +7.99%.

So every model here has been competing on a quantity it may simply be a noisier
estimate of, and nothing measured so far can tell the two apart. This module
removes that channel from the TARGET and re-scores the ordering against what is
left. A comparator that keeps its rank IC on the residual has company-specific
information; one that loses it never had any.

WHY THE FLOOR'S OWN PREDICTION IS THE RIGHT REGRESSOR
------------------------------------------------------
`BetaMarket.predict` returns ``beta_i * mu_market``, and ``mu_market`` is a
per-FOLD constant — so within any single date it is an AFFINE function of
``beta_i``. OLS residuals are invariant to affine rescaling of the regressor, so
residualising on the floor's prediction is exactly identical to residualising on
beta itself, while requiring no second beta estimate that could disagree with
the floor's. The neutralisation is complementary to the floor by construction.

That invariance is also why the sign of ``mu_market`` does not matter and why a
NEGATIVE mu is not a special case. What does matter is a mu of ZERO: the
regressor is then constant, the residual is the target unchanged, and a table
would report an unneutralised number under a neutralised heading. That is
refused, per date, rather than silently passed through.

WHAT MAY AND MAY NOT BE READ OFF THE OUTPUT
--------------------------------------------
REPORT reb_IC AND reb_t. NEVER MAE. The residual has strictly smaller variance
than the target it came from, so an MAE against it is not comparable to any MAE
in any existing table in this project — it will look like a large improvement
and mean nothing. That is the `daily_IC`-beside-`reb_t` error class, and it is
the single easiest way to make this table lie. `residual_report` therefore does
not compute an MAE at all, rather than computing one and captioning it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.evaluation import rank_ic, rebalance_books
from pipeline.signals import HORIZON_SESSIONS

logger = logging.getLogger(__name__)

#: Below this spread in the regressor, a within-date OLS is fitting nothing and
#: the "residual" is the target unchanged. Not an epsilon on the fit — an
#: epsilon on the REGRESSOR, which is where the degeneracy actually lives.
MIN_REGRESSOR_SPREAD = 1e-12

#: The calibration gate's MATERIALITY band, in rank IC.
#:
#: A LINEAR RESIDUALISATION DOES NOT EXACTLY ZERO A RANK CORRELATION, and this
#: caught the first version of the gate. The residual is orthogonal to the
#: regressor in the OLS sense, but rank IC is a SPEARMAN correlation and
#: orthogonality is not preserved through the rank transform. So a correctly
#: wired gate lands NEAR zero with sampling noise around it, never at zero, and
#: the noise scales as 1/sqrt(names)/sqrt(rebalances) — about 0.06 on a small
#: fixture and about 0.013 on the real panel. A fixed absolute band therefore
#: mis-sizes itself against whatever sample it is handed.
#:
#: The gate consequently fails only when the leak is BOTH statistically real and
#: materially large; see `calibration_gate`.
CALIBRATION_TOLERANCE = 0.02

#: t-statistic above which a residual rank IC is treated as a real leak rather
#: than sampling noise. The project's standing threshold, used here for the same
#: reason it is used everywhere else.
CALIBRATION_MAX_T = 2.0


class NeutralisationRefused(RuntimeError):
    """
    Raised when the neutralisation could not be performed as described.

    A null from a broken neutraliser is indistinguishable from a null from a
    working one — the `series_zero` problem — so every failure here is loud
    rather than a quietly unneutralised row.
    """


@dataclass
class ResidualPanel:
    """A panel whose target (and optionally prediction) has had beta removed."""

    frame: pd.DataFrame
    basis: str
    n_dates: int = 0
    n_dates_refused: int = 0
    neutralised_prediction: bool = False
    #: Fraction of the target's within-date variance the regressor explained.
    #: This IS the size of the beta channel, and it is the number that says how
    #: much of the panel the rest of the phase is arguing about.
    r2_mean: float = float("nan")
    notes: list[str] = field(default_factory=list)


def _residualise_within_date(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """
    OLS residual of ``y`` on ``x`` with an intercept, plus the fit's R-squared.

    One regressor and one intercept against ~84-116 names, so the two degrees of
    freedom consumed are immaterial. Written out rather than delegated because
    the degenerate branch — a constant regressor — has to be DETECTABLE by the
    caller, and every library form of this quietly returns the input instead.
    """
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3:
        return np.full_like(y, np.nan, dtype=float), float("nan")

    xs, ys = x[ok], y[ok]
    if float(np.ptp(xs)) <= MIN_REGRESSOR_SPREAD:
        # A CONSTANT REGRESSOR IS NOT A NEUTRALISATION. Returning `y` here would
        # put an untouched target under a "beta-neutralised" heading, which is
        # the whole failure this module is built to make impossible.
        return np.full_like(y, np.nan, dtype=float), float("nan")

    xc = xs - xs.mean()
    slope = float(np.dot(xc, ys - ys.mean()) / np.dot(xc, xc))
    fitted = ys.mean() + slope * xc
    resid = ys - fitted

    var = float(np.var(ys))
    r2 = float(1.0 - np.var(resid) / var) if var > 0 else float("nan")

    out = np.full_like(y, np.nan, dtype=float)
    out[ok] = resid
    return out, r2


def neutralise(predictions: pd.DataFrame, beta: pd.DataFrame,
               basis: str = "beta_market",
               neutralise_prediction: bool = False) -> ResidualPanel:
    """
    Removes the beta channel from ``y_true``, within each date.

    ``predictions`` is what `panel_walk_forward` returns — date, ticker, y_pred,
    y_true. ``beta`` carries (date, ticker, beta_basis): either `beta_market`'s
    own out-of-sample prediction (the primary basis, exactly complementary to
    the floor) or `regime.rolling_beta`'s trailing estimate (the robustness one,
    and the only one a live system could actually use).

    ``neutralise_prediction`` also residualises ``y_pred``. The two variants ask
    DIFFERENT questions and both are reported rather than the better one quoted:
    target-only asks "does this ordering predict the part of return beta cannot
    explain", both asks "does the non-beta part of this ordering predict it".
    """
    required = {"date", "ticker", "y_pred", "y_true"}
    missing = required - set(predictions.columns)
    if missing:
        raise NeutralisationRefused(f"predictions lack columns {sorted(missing)}")
    if "beta_basis" not in beta.columns:
        raise NeutralisationRefused("beta frame needs a 'beta_basis' column")

    merged = predictions.merge(beta[["date", "ticker", "beta_basis"]],
                               on=["date", "ticker"], how="left")

    kept, refused, r2s = [], 0, []
    for dt, day in merged.groupby("date", sort=True):
        x = day["beta_basis"].to_numpy(dtype=float)
        resid, r2 = _residualise_within_date(day["y_true"].to_numpy(dtype=float), x)
        if not np.isfinite(resid).any():
            refused += 1
            continue

        day = day.assign(y_true=resid)
        if neutralise_prediction:
            presid, _ = _residualise_within_date(
                day["y_pred"].to_numpy(dtype=float), x)
            if not np.isfinite(presid).any():
                refused += 1
                continue
            day = day.assign(y_pred=presid)

        if np.isfinite(r2):
            r2s.append(r2)
        kept.append(day.dropna(subset=["y_true", "y_pred"]))

    if not kept:
        raise NeutralisationRefused(
            f"every date refused neutralisation on basis '{basis}' — the "
            f"regressor is constant or absent on all {len(merged['date'].unique())} "
            f"of them, so nothing here would be neutralised")

    frame = pd.concat(kept, ignore_index=True)
    panel = ResidualPanel(
        frame=frame, basis=basis, n_dates=frame["date"].nunique(),
        n_dates_refused=refused, neutralised_prediction=neutralise_prediction,
        r2_mean=float(np.mean(r2s)) if r2s else float("nan"))
    if refused:
        panel.notes.append(
            f"{refused} date(s) refused: constant or missing {basis}")
    return panel


def residual_report(panel: pd.DataFrame,
                    rebalance_every: int = HORIZON_SESSIONS,
                    quantiles: int = 5) -> dict:
    """
    Rank IC of an ordering against a residual target, over non-overlapping dates.

    Deliberately NOT `cross_sectional_report`. That function returns MAE-adjacent
    and return-space quantities — top-quintile return, alpha vs equal weight — and
    every one of them is meaningless against a residual whose mean is zero within
    each date by construction. Returning them with a caption would invite exactly
    the misreading the module docstring refuses. Only the scale-invariant
    statistics survive residualisation, so only they are computed.

    Books come from `evaluation.rebalance_books`, so the dates scored here are
    the same dates every other table in this project scores.
    """
    required = {"date", "ticker", "y_pred", "y_true"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"residual_report needs columns {sorted(missing)}")

    panel = panel.dropna(subset=["y_pred", "y_true"])
    if panel.empty:
        return {"n_rebalances": 0, "note": "no rows after neutralisation"}

    ics, degenerate = [], 0
    for book in rebalance_books(panel, rebalance_every, quantiles):
        if book.degenerate:
            degenerate += 1
            continue
        day = book.day
        ics.append(rank_ic(day["y_true"].to_numpy(), day["y_pred"].to_numpy()))

    clean = pd.Series(ics, dtype=float).dropna()
    if len(clean) < 3:
        return {"n_rebalances": len(clean), "n_dates_no_ordering": degenerate,
                "note": "too few rebalances to support a t-statistic"}

    t, p = stats.ttest_1samp(clean, 0)
    return {
        "n_rebalances": int(len(clean)),
        "n_dates_no_ordering": degenerate,
        "residual_rank_ic": float(clean.mean()),
        "residual_ic_t": float(t),
        "residual_ic_p": float(p),
    }


def calibration_gate(beta_predictions: pd.DataFrame, beta: pd.DataFrame,
                     basis: str = "beta_market",
                     tolerance: float = CALIBRATION_TOLERANCE,
                     max_t: float = CALIBRATION_MAX_T,
                     rebalance_every: int = HORIZON_SESSIONS) -> dict:
    """
    THE CHECK THAT EARNS THE RIGHT TO QUOTE EVERY OTHER ROW.

    `beta_market`'s own ordering, scored against its own beta-neutralised
    residual, must come back at ~0: the residual is orthogonal to the regressor
    by construction, and `beta_market`'s prediction IS an affine function of that
    regressor within each date. If this does not land at zero, the regressor is
    not what the code believes it is — wired to the wrong column, merged on the
    wrong key, or misaligned by a date — and every other row in the table is
    measuring something nobody has named.

    A null from an unvalidated instrument is indistinguishable from a null from a
    broken one. This is the `series_zero` pattern, and the direct analogue of
    `run_portfolio`'s planted-edge gate: the caller runs this FIRST and prints
    nothing real if it fails.
    """
    resid = neutralise(beta_predictions, beta, basis=basis)
    report = residual_report(resid.frame, rebalance_every=rebalance_every)
    observed = report.get("residual_rank_ic", float("nan"))
    t = report.get("residual_ic_t", float("nan"))
    computed = np.isfinite(observed) and np.isfinite(t)

    # PASS IF THE LEAK IS NOT REAL, OR IS REAL BUT IMMATERIAL. Requiring both
    # would fail a correctly wired gate on a small sample, where rank-transform
    # noise alone reaches 0.06; requiring neither would let a genuine
    # misalignment through on a large one. A misaligned regressor fails both
    # tests at once, which is what the accompanying test pins.
    passed = bool(computed and (abs(t) < max_t or abs(observed) <= tolerance))

    # TWO WAYS TO FAIL, AND THEY NEED DIFFERENT REMEDIES. A gate that reports
    # "the regressor is not beta" when the real problem is too few rebalances
    # sends the reader to rewrite correct code — the same wasted trip as
    # `backfill_news` blaming missing metadata during a database outage. Fail
    # closed for both, but say which one happened.
    if passed:
        note = ""
    elif not computed:
        note = (f"the gate could not be computed: {report.get('note', 'no report')} "
                f"({report.get('n_rebalances', 0)} rebalances at "
                f"rebalance_every={rebalance_every}). This is NOT evidence the "
                f"neutralisation is wrong, and it is not permission to proceed.")
    else:
        note = (f"beta_market retains rank IC {observed:+.4f} (t {t:+.2f}) "
                f"against its own residual — both significant and larger than "
                f"+/-{tolerance}. The regressor is not beta. Every other row is "
                f"meaningless until this is fixed.")

    return {
        "basis": basis,
        "observed_residual_ic": observed,
        "observed_residual_t": t,
        "tolerance": tolerance,
        "max_t": max_t,
        "n_rebalances": report.get("n_rebalances", 0),
        "beta_channel_r2": resid.r2_mean,
        "passed": passed,
        "computed": bool(computed),
        "note": note,
    }


def fold_null_band(panel: pd.DataFrame, n_draws: int = 200,
                   rebalance_every: int = HORIZON_SESSIONS,
                   quantiles: int = 5, seed: int = 0) -> dict:
    """
    Each fold's OWN null distribution of rebalance rank IC, by permutation.

    WHY THIS EXISTS. Three unrelated methods produced their entire apparent
    signal in the earliest fold — valuation reb_t +3.32, LoRA +2.37, pooled_xgb
    +2.42 — and each was judged against a POOLED null. Fold 0 holds fewer names
    per date and fewer rebalances than fold 4, so its rank IC is a noisier
    statistic, and a lone +0.0879 there is less surprising than the same number
    late. That has been the standing explanation for three retired results and it
    has never once been measured.

    NOTE WHAT THIS DOES NOT TEST. Target dispersion falls monotonically across
    the folds (0.108 -> 0.077) and it is tempting to blame that. It cannot be the
    cause: rank IC is invariant to a monotone transform of the target within a
    date, so scaling a whole cross-section leaves it exactly unchanged.
    Dispersion moves MAE and Sharpe, not this. Breadth and window count move
    this, which is why those are what is reported beside the band.

    Permutation is WITHIN DATE, which preserves each date's breadth, its label
    distribution and the fold's rebalance count, and destroys only the pairing
    between prediction and outcome. A permutation across dates would also destroy
    the date structure and would produce a narrower, wrong band.
    """
    if "fold" not in panel.columns:
        raise ValueError("fold_null_band needs a 'fold' column")

    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}

    for fold, block in panel.groupby("fold", sort=True):
        block = block.dropna(subset=["y_pred", "y_true"])
        dates = sorted(block["date"].unique())[::rebalance_every]
        scored = block[block["date"].isin(dates)]
        if scored.empty:
            continue

        draws = []
        for _ in range(n_draws):
            ics = []
            for _, day in scored.groupby("date", sort=False):
                y = day["y_true"].to_numpy(dtype=float)
                if len(y) < max(10, quantiles * 2):
                    continue
                ics.append(rank_ic(y, rng.permutation(y)))
            clean = [v for v in ics if np.isfinite(v)]
            if clean:
                draws.append(float(np.mean(clean)))

        if not draws:
            continue
        d = np.asarray(draws, dtype=float)
        per_date = scored.groupby("date").size()
        out[str(fold)] = {
            "n_rebalances": int(len(per_date)),
            "median_names_per_date": float(per_date.median()),
            "target_dispersion": float(
                scored.groupby("date")["y_true"].std().mean()),
            "null_mean": float(d.mean()),
            "null_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "null_p95_abs": float(np.percentile(np.abs(d), 95)),
        }
    return out
