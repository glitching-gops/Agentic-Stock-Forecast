"""
pipeline/evidence_panel.py — Stage 0c: the evidence layer, closed.

EVIDENCE-GRADING REDESIGN, STAGE 0c. Last session on this track. Separate
numbering from the project's own Phase 0-6 roadmap.

--------------------------------------------------------------------------------
WHAT STAGE 0b LEFT OPEN, AND WHAT THIS FIXES
--------------------------------------------------------------------------------

Stage 0b corrected the per-ticker IC from "concatenate every walk-forward fold
and correlate once" to "correlate WITHIN each fold and average". That was a real
bug — the fold prediction levels run against the fold realised returns at
rho = -0.600 on this panel, so the pooled figure read -0.05988 while every
fold's internal ranking was positive — and it is fixed. Two problems survived it,
and this module closes both.

**1. THE GRADED QUANTITY WAS STILL A RAW TIME-SERIES IC.** "Within this stock,
did the dates it ranked higher pay more" is answered YES for every name at once
whenever a common market factor moves them together. That is exactly the pattern
Stage 0b's table showed: 58 of 84 tickers positive under `pooled_rank_ic`, with
a cross-sectional rebalance t of +0.86 on the same predictions. Every economic
bar this project owns — `reb_IC`, the break-even 0.00512, the portfolio
simulator, the deflated Sharpe — is CROSS-SECTIONAL, so the gate was grading a
quantity nothing else in the project measures.

The fix is CROSS-SECTIONAL RANK-DEMEANING of both sides. Within each date, rank
the predictions across the names present and map to [-1, 1]; do the same to the
targets. What is left for each ticker is its position RELATIVE TO ITS OWN
CROSS-SECTION, which is what a long-short book actually trades. Convention
follows Gu, Kelly & Xiu (2020) and Kelly, Pruitt & Su (2019): rank
period-by-period, map into [-1, 1].

**2. mu_hat's STANDARD ERROR TREATED 84 TICKERS AS INDEPENDENT.** They share one
market and roughly 64 independent 30-session windows. The precision-weighted
`var_mu = 1 / sum(w_i)` is a fixed-effect formula that assumes independence
across units, and it produced a z of +5.72 that nobody should have believed in
either direction.

The fix is a DATE-LEVEL CIRCULAR BLOCK BOOTSTRAP that re-runs the whole
empirical-Bayes pipeline inside every replicate: resample dates within fold,
recompute every ticker's fold-averaged IC, recompute every sigma2_i, re-estimate
tau2, recompute mu_hat and every posterior. The spread across replicates is the
standard error, and it contains within-fold, between-fold AND cross-sectional
dependence at once.

**BECAUSE THE BOOTSTRAP ALREADY CONTAINS THE BETWEEN-FOLD COMPONENT, STAGE 0b's
ADDITIVE max(within, between) TERM IS SWITCHED OFF WHERE THE BOOTSTRAP IS IN
FORCE.** Keeping both double-counts. It survives only as the fallback for
tickers below `MIN_FOLDS_FOR_ESTIMATE`, and which path a ticker took is recorded
on its row rather than inferred.

--------------------------------------------------------------------------------
THE SUM-TO-ZERO CONSTRAINT
--------------------------------------------------------------------------------

Cross-sectional demeaning applies the centering matrix

    C = I_N - (1/N) 11'

which is idempotent with rank N - 1. It induces an EXACT pairwise correlation of
-1/(N-1) between the demeaned residuals — **-0.01205 at N = 84** — and removes
one degree of freedom per date. This is the oldest known artifact of its kind:
Pearson (1897) on spurious correlation from indices, Chayes (1960) on
constant-sum correlation in petrology, Aitchison (1986) on compositional data.

It needs NO separate correction here, because the date-level bootstrap resamples
whole cross-sections and therefore carries the constraint through every
replicate. That is an argument, not a proof, so it is TESTED: a test builds
i.i.d. cross-sections, demeans them, and requires the empirical mean off-diagonal
correlation to match -1/(N-1).

--------------------------------------------------------------------------------
MULTIPLICITY
--------------------------------------------------------------------------------

Grading 84 tickers is 84 simultaneous tests and a nominal per-ticker t > 2 is not
evidence of skill. Benjamini-Hochberg is not safe here: BH needs PRDS, and this
panel has a common market factor (positive dependence) PLUS the -1/(N-1)
negative dependence demeaning induces. Romano-Wolf stepdown (Romano & Wolf 2005)
makes no dependence assumption at all — it takes the dependence from the same
bootstrap that produced the statistics — so it is the headline, with BH and
Benjamini-Yekutieli reported beside it as sensitivity bounds.

Harvey, Liu & Zhu (2016) argue a t of 3.0 for a NEW factor after the volume of
testing the literature has done. This project has additionally run 103+
configurations across its own history. The bar here should be read in that light.

--------------------------------------------------------------------------------
HETEROGENEITY
--------------------------------------------------------------------------------

tau2 is estimated by REML rather than DerSimonian-Laird: DL is negatively biased
at small K and Langan et al. (2019) recommend REML over it (and over
Paule-Mandel, which over-corrects when study sizes differ a lot). Intervals on
the grand mean carry the Hartung-Knapp-Sidik-Jonkman adjustment — a t reference
with K-1 degrees of freedom — which IntHout, Ioannidis & Borm (2014) show
outperforms the standard DL interval, with extra caution at five or fewer very
unequal units. THIS PANEL IS IN THAT CAUTION ZONE and the output says so.

**AND THERE IS AN EXPLICIT tau2 ~ 0 DETECTOR.** At the zero boundary the
random-effects model has collapsed to a fixed-effect one: every shrinkage weight
is 1, every posterior IS the grand mean, and all 84 tickers receive an identical
grade. Printing that as 84 per-ticker findings misrepresents ONE fact as 84 —
which is exactly what happened when Stage 2b's `pooled_rank_ic` reported 84
identical ANTI_SIGNAL grades at tau2 = 0.0003. When the detector fires this
module emits one panel-level statement and suppresses the duplicates.

--------------------------------------------------------------------------------
DEPENDENCIES, AND WHY THEY ARE NOT IN requirements.txt
--------------------------------------------------------------------------------

`arch` (Romano-Wolf, optimal block length) and `linearmodels` (Driscoll-Kraay)
are imported LAZILY and live in `requirements-evidence.txt`, not
`requirements.txt`. That file is installed by Render, by the daily pipeline and
by the weekly evaluation; torch was removed from it in Phase 0 as the largest
contributor to memory pressure on an instance that had already been OOM-killed,
and the rule that followed is not "no big libraries" but "nothing on the serving
path that the serving path does not use". Render serves reads and never grades a
panel. A test asserts neither package is imported by loading this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from pipeline.evidence_shrinkage import (
    BLOCK_LENGTH_SESSIONS,
    FDR_Q,
    MIN_FOLDS_FOR_ESTIMATE,
    STRONG_POSTERIOR_THRESHOLD,
    benjamini_hochberg,
    posterior_probability_positive,
    precision_weighted_mean,
    shrink,
)

logger = logging.getLogger(__name__)

# ── Constants, fixed in docs/stage0c-preregistration.md before the run ────────

#: A date with fewer names than this cannot support a cross-sectional
#: demeaning — the centering is over that date's own cross-section, and a
#: handful of names gives a mean and a rank scale that are mostly noise. 20
#: mirrors the minimum-firms convention used in Fama-MacBeth cross-sectional
#: regressions. Dates below it are DROPPED and counted, never imputed.
MIN_CROSS_SECTION = 20

#: Bootstrap replicates. Each one is a full empirical-Bayes re-run, not a
#: resample of a finished statistic.
BOOTSTRAP_B = 1000

#: Seed for the date-level resample. Distinct from Stage 0's ticker-level seed
#: so the two layers cannot accidentally share a draw.
PANEL_BOOTSTRAP_SEED = 20260908

#: Familywise error rate for Romano-Wolf. Deliberately the same 0.10 the FDR
#: procedures use, so the three are comparable as bounds on the same panel.
FWER_ALPHA = FDR_Q

#: REML fixed point: iterations and the convergence tolerance on tau2.
REML_MAX_ITER = 200
REML_TOL = 1e-12

#: tau2 is called degenerate at or below this. Not exactly zero: REML lands on
#: the boundary numerically rather than analytically, and a tau2 of 1e-9 has the
#: same consequence as a tau2 of 0 — every posterior collapses onto the grand
#: mean and the 84 grades become one grade printed 84 times.
TAU2_ZERO_TOL = 1e-6

GRADER_VERSION_V3 = "stage0c-panel-rw-v4"


# ── Rank-demeaning ────────────────────────────────────────────────────────────


def rank_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """
    Ranks mapped to [-1, 1], the Gu-Kelly-Xiu / Kelly-Pruitt-Su convention.

    A single value maps to 0.0 rather than dividing by zero: one name has no
    cross-sectional position, and 0 is the centre of the scale.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return values
    if n == 1:
        return np.zeros(1)
    ranks = stats.rankdata(values)                # average ranks for ties
    return 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0


def rank_demean_by_date(dates: np.ndarray, y_true: np.ndarray,
                        y_pred: np.ndarray,
                        min_cross_section: int = MIN_CROSS_SECTION
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Cross-sectionally rank-demean BOTH sides, within each date.

    Returns ``(keep_mask, demeaned_true, demeaned_pred, report)`` where the
    arrays are aligned to the ORIGINAL rows and are NaN where ``keep_mask`` is
    False.

    BOTH SIDES, NOT TARGETS ONLY. Demeaning the target alone removes the market
    from what is being predicted but leaves it in the prediction, so a model
    that forecasts nothing but the market level still scores a positive
    correlation against the residual through its own level drift. Demeaning both
    is what makes the statistic "did this name beat its own cross-section",
    which is the quantity a long-short book trades and the one every economic
    bar in this project already measures.

    A date below ``min_cross_section`` is dropped and counted. Not imputed, and
    no cross-sectional mean is carried forward from an adjacent date: a
    demeaning against a different day's cross-section is not a demeaning.
    """
    dates = np.asarray(dates)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    order = np.argsort(dates, kind="stable")
    keep = np.zeros(dates.size, dtype=bool)
    dt = np.full(dates.size, np.nan)
    dp = np.full(dates.size, np.nan)

    sorted_dates = dates[order]
    bounds = np.flatnonzero(np.r_[True, sorted_dates[1:] != sorted_dates[:-1], True])

    dropped_small = 0
    dropped_rows = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = order[lo:hi]
        finite = np.isfinite(y_true[idx]) & np.isfinite(y_pred[idx])
        idx = idx[finite]
        if idx.size < min_cross_section:
            dropped_small += 1
            dropped_rows += int(idx.size)
            continue
        keep[idx] = True
        dt[idx] = rank_to_unit_interval(y_true[idx])
        dp[idx] = rank_to_unit_interval(y_pred[idx])

    return keep, dt, dp, {
        "dates_total": int(len(bounds) - 1),
        "dates_dropped_small_cross_section": dropped_small,
        "rows_dropped": dropped_rows,
        "min_cross_section": int(min_cross_section),
    }


def induced_pairwise_correlation(n: int) -> float:
    """The exact correlation cross-sectional demeaning induces: -1/(N-1)."""
    return -1.0 / (n - 1.0) if n > 1 else float("nan")


# ── A NaN-aware vectorised Spearman, down rows ────────────────────────────────


def nan_rank_ic_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Spearman correlation of each ROW of ``a`` against each row of ``b``,
    ignoring positions where either is NaN.

    ``evidence_shrinkage.rank_ic_rows`` is the same computation without the NaN
    handling, and this panel needs it: a ticker is present on 81 to 84 of the
    names on a date and its fold's date grid is the union across tickers, so a
    rectangular (ticker x date) matrix necessarily has holes. Filling them would
    invent an ordering; dropping the whole date would discard 80 good names to
    accommodate one absence.

    A row with fewer than 3 usable pairs, or no ordering on either side, yields
    NaN rather than 0.0.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    valid = np.isfinite(a) & np.isfinite(b)

    # Rank within each row over the valid entries only. Pushing invalid entries
    # to +inf puts them last under argsort, so their ranks are the tail and can
    # be masked out afterwards without disturbing the valid ones.
    def _ranks(x: np.ndarray) -> np.ndarray:
        masked = np.where(valid, x, np.inf)
        order = np.argsort(masked, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=float)
        rows = np.arange(x.shape[0])[:, None]
        ranks[rows, order] = np.arange(x.shape[1], dtype=float)[None, :]
        return ranks

    ra, rb = _ranks(a), _ranks(b)

    # Average ranks for ties, computed only where it matters: ties are common
    # after rank-demeaning only when a date had tied predictions, which the
    # degenerate-cell guards elsewhere already treat as no ordering. Ties are
    # handled by re-ranking the valid slice with scipy for the affected rows.
    counts = valid.sum(axis=1)
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        m = valid[i]
        n = int(counts[i])
        if n < 3:
            continue
        av, bv = a[i, m], b[i, m]
        if np.ptp(av) == 0 or np.ptp(bv) == 0:
            continue
        ar = stats.rankdata(av)
        br = stats.rankdata(bv)
        ar -= ar.mean()
        br -= br.mean()
        den = np.sqrt((ar * ar).sum() * (br * br).sum())
        if den > 0:
            out[i] = float((ar * br).sum() / den)
    return out


# ── The panel, folded into matrices the bootstrap can resample ────────────────


@dataclass
class FoldMatrix:
    """One fold as a (date x ticker) grid, with NaN where a name is absent."""

    fold: int
    dates: np.ndarray            # (D,)
    true: np.ndarray             # (D, T)
    pred: np.ndarray             # (D, T)


@dataclass
class PanelMatrices:
    tickers: np.ndarray
    folds: list[FoldMatrix]
    demean_report: dict


def build_panel(dates: np.ndarray, tickers: np.ndarray, folds: np.ndarray,
                y_true: np.ndarray, y_pred: np.ndarray,
                min_cross_section: int = MIN_CROSS_SECTION) -> PanelMatrices:
    """
    Rank-demean within date, then lay each fold out as a (date x ticker) grid.

    The grid is what makes the date-level bootstrap cheap AND correct: a
    replicate is a selection of ROWS, so every ticker sees the same dates in the
    same replicate — which is the whole point, because the cross-sectional
    dependence is what the naive standard error was missing.
    """
    dates = np.asarray(dates)
    tickers = np.asarray(tickers)
    folds = np.asarray(folds)

    keep, dt, dp, report = rank_demean_by_date(
        dates, y_true, y_pred, min_cross_section=min_cross_section)

    uniq_tickers = np.unique(tickers)
    tix = {t: i for i, t in enumerate(uniq_tickers)}

    out: list[FoldMatrix] = []
    for k in np.unique(folds):
        m = (folds == k) & keep
        if not m.any():
            continue
        fd = np.unique(dates[m])
        dix = {d: i for i, d in enumerate(fd)}
        true = np.full((fd.size, uniq_tickers.size), np.nan)
        pred = np.full((fd.size, uniq_tickers.size), np.nan)
        rows = np.flatnonzero(m)
        r = np.fromiter((dix[d] for d in dates[rows]), dtype=int, count=rows.size)
        c = np.fromiter((tix[t] for t in tickers[rows]), dtype=int, count=rows.size)
        true[r, c] = dt[rows]
        pred[r, c] = dp[rows]
        out.append(FoldMatrix(int(k), fd, true, pred))

    return PanelMatrices(uniq_tickers, out, report)


def per_ticker_fold_ics(panel: PanelMatrices,
                        selections: list[np.ndarray] | None = None
                        ) -> np.ndarray:
    """
    (n_folds x n_tickers) matrix of within-fold, rank-demeaned ICs.

    ``selections`` optionally gives, per fold, the row indices to use — which is
    how a bootstrap replicate is expressed. ``None`` means the real sample.
    """
    n_t = panel.tickers.size
    out = np.full((len(panel.folds), n_t), np.nan)
    for j, fm in enumerate(panel.folds):
        sel = slice(None) if selections is None else selections[j]
        out[j] = nan_rank_ic_rows(fm.pred[sel].T, fm.true[sel].T)
    return out


# ── The date-level circular block bootstrap ───────────────────────────────────


def circular_block_indices(n: int, block: int, n_draws: int,
                           rng: np.random.Generator) -> np.ndarray:
    """
    One circular-block resample of ``n`` positions.

    CIRCULAR, not moving-block. A moving-block bootstrap can only start a block
    at positions 0..n-block, so the first and last few observations appear in
    fewer blocks than the middle ones and are systematically under-sampled.
    Wrapping the series makes every position equally likely (Politis & Romano
    1992), which matters here because the fold boundaries ARE the ends and the
    walk-forward's most recent fold is the one a reader cares about most.
    """
    if n <= 0 or block <= 0:
        return np.zeros(0, dtype=int)
    block = min(block, n)
    starts = rng.integers(0, n, size=n_draws)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return idx.reshape(-1)


def resample_selections(panel: PanelMatrices, block: int,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """
    One replicate: dates resampled WITHIN EACH FOLD, never across.

    Resampling across the whole sample would mix a 2019 date into the 2025 fold
    and destroy the walk-forward ordering the purge and embargo exist to
    protect. Within-fold draws leave the fold structure, the ordering and those
    guarantees exactly where they were.
    """
    sels = []
    for fm in panel.folds:
        n = fm.dates.size
        n_draws = int(np.ceil(n / min(block, max(n, 1))))
        idx = circular_block_indices(n, block, n_draws, rng)[:n]
        sels.append(idx)
    return sels


# ── REML tau2, HKSJ, and the degeneracy detector ──────────────────────────────


def reml_tau2(hat: np.ndarray, sigma2: np.ndarray,
              max_iter: int = REML_MAX_ITER, tol: float = REML_TOL) -> float:
    """
    Restricted maximum likelihood estimate of the between-ticker variance.

    The standard random-effects fixed point (Viechtbauer 2005, eq. 21):

        w_i    = 1 / (sigma2_i + tau2)
        mu     = sum(w_i y_i) / sum(w_i)
        tau2'  = [ sum(w_i^2 ((y_i - mu)^2 + 1/sum(w) - sigma2_i)) ] / sum(w_i^2)

    iterated to convergence and truncated at zero.

    REML rather than DerSimonian-Laird because DL is NEGATIVELY BIASED at small
    K, and this panel's K is 5 folds' worth of information spread over 84 names.
    Langan et al. (2019) compare sixteen estimators and recommend REML;
    Paule-Mandel over-corrects when unit precisions differ a lot, which they do
    here. Both are reported side by side rather than one being quoted.
    """
    hat = np.asarray(hat, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    ok = np.isfinite(hat) & np.isfinite(sigma2) & (sigma2 > 0)
    hat, sigma2 = hat[ok], sigma2[ok]
    k = hat.size
    if k < 2:
        return 0.0

    tau2 = max(float(np.var(hat, ddof=1) - np.mean(sigma2)), 0.0)
    for _ in range(max_iter):
        w = 1.0 / (sigma2 + tau2)
        sw = w.sum()
        mu = float((w * hat).sum() / sw)
        num = (w ** 2 * ((hat - mu) ** 2 + 1.0 / sw - sigma2)).sum()
        den = (w ** 2).sum()
        new = max(float(num / den), 0.0) if den > 0 else 0.0
        if abs(new - tau2) < tol:
            tau2 = new
            break
        tau2 = new
    return float(tau2)


def cochran_q(hat: np.ndarray, sigma2: np.ndarray) -> tuple[float, int]:
    """Cochran's Q and its degrees of freedom, on the usable units."""
    hat = np.asarray(hat, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    ok = np.isfinite(hat) & np.isfinite(sigma2) & (sigma2 > 0)
    hat, sigma2 = hat[ok], sigma2[ok]
    if hat.size < 2:
        return float("nan"), 0
    w = 1.0 / sigma2
    mu = float((w * hat).sum() / w.sum())
    return float((w * (hat - mu) ** 2).sum()), int(hat.size - 1)


def hksj_interval(hat: np.ndarray, sigma2: np.ndarray, tau2: float,
                  alpha: float = 0.05) -> tuple[float, float, float, float]:
    """
    Hartung-Knapp-Sidik-Jonkman interval on the grand mean.

    Returns ``(mu, se_hksj, lo, hi)``. The HKSJ variance rescales the
    random-effects standard error by the observed dispersion of the units around
    the mean and refers it to a t distribution with K-1 degrees of freedom,
    instead of the standard normal the plain DL interval uses.

    IntHout, Ioannidis & Borm (2014) show it outperforms DL across a wide range
    of conditions, with the caveat that at five or fewer units of very unequal
    size it can be anti-conservative. **THIS PANEL IS IN THAT ZONE** — the
    effective number of independent periods is about five — so the interval is
    reported as approximate and the bootstrap is the primary.
    """
    hat = np.asarray(hat, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    ok = np.isfinite(hat) & np.isfinite(sigma2) & (sigma2 > 0)
    hat, sigma2 = hat[ok], sigma2[ok]
    k = hat.size
    if k < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    w = 1.0 / (sigma2 + tau2)
    mu = float((w * hat).sum() / w.sum())
    q = float((w * (hat - mu) ** 2).sum() / (k - 1))
    se = float(np.sqrt(max(q, 0.0) / w.sum()))
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 1))
    return mu, se, mu - crit * se, mu + crit * se


@dataclass
class DegeneracyVerdict:
    """Whether the random-effects model has collapsed to a fixed-effect one."""

    degenerate: bool
    tau2: float
    q_statistic: float
    q_df: int
    reason: str

    def statement(self, mu: float, grade: str, n: int) -> str:
        return (
            f"Between-ticker heterogeneity is statistically indistinguishable "
            f"from zero (tau2 = {self.tau2:.2e}, Q = {self.q_statistic:.1f} on "
            f"{self.q_df} df). The data cannot separate individual tickers, so "
            f"all {n} names carry the pooled grade {grade} at a common "
            f"posterior mean of {mu:+.5f}. This is ONE finding about the panel, "
            f"not {n} findings about {n} companies."
        )


def detect_tau2_degeneracy(tau2: float, hat: np.ndarray,
                           sigma2: np.ndarray,
                           tol: float = TAU2_ZERO_TOL) -> DegeneracyVerdict:
    """
    Fires at the tau2 = 0 boundary, or when Q is at or below its own df.

    Both conditions describe the same collapse. At tau2 = 0 the shrinkage weight
    B_i = sigma2_i / (sigma2_i + tau2) is 1 for every ticker, so every posterior
    mean IS the grand mean and every p_positive is identical — the panel makes a
    single statement and the code then prints it once per name.

    Q <= df is the classical "no more dispersion than sampling error alone
    predicts" test and is included because REML can land marginally above the
    numerical tolerance while the data still carry no separable heterogeneity.
    """
    q, df = cochran_q(hat, sigma2)
    at_boundary = not np.isfinite(tau2) or tau2 <= tol
    no_dispersion = np.isfinite(q) and df > 0 and q <= df

    if at_boundary:
        reason = (f"tau2 = {tau2:.3e} is at or within {tol:.0e} of the zero "
                  f"boundary")
    elif no_dispersion:
        reason = (f"Cochran Q = {q:.2f} does not exceed its {df} degrees of "
                  f"freedom, so observed dispersion is no larger than sampling "
                  f"error alone predicts")
    else:
        reason = ""

    return DegeneracyVerdict(bool(at_boundary or no_dispersion), float(tau2),
                             q, df, reason)


# ── Romano-Wolf stepdown ──────────────────────────────────────────────────────


def romano_wolf(stat: np.ndarray, boot: np.ndarray,
                alpha: float = FWER_ALPHA,
                two_sided: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Romano-Wolf (2005) stepdown, driven by the panel's own bootstrap.

    ``stat`` is the (T,) vector of per-ticker studentised statistics and ``boot``
    the (B, T) matrix of the same statistics on each replicate, CENTRED on the
    observed values so each column is a draw from that ticker's null.

    Returns ``(rejected, adjusted_p)``.

    Chosen over Benjamini-Hochberg because BH controls FDR under positive
    regression dependence (PRDS), and this panel violates it in both directions
    at once: a common market factor is positive dependence, and cross-sectional
    demeaning induces the exact NEGATIVE correlation -1/(N-1). Romano-Wolf makes
    no dependence assumption — it reads the dependence off the joint bootstrap
    distribution — and it controls the familywise error rate, which is the
    stricter guarantee.

    Implemented directly rather than through ``arch.bootstrap.StepM`` because
    StepM wants to own the resampling, and this bootstrap is not resamplable by
    it: a replicate here is a re-run of the whole empirical-Bayes pipeline on a
    date-block draw, not an i.i.d. draw over a loss series. The algorithm is the
    published one and is tested against a synthetic case with known strong and
    null units.
    """
    stat = np.asarray(stat, dtype=float)
    boot = np.atleast_2d(np.asarray(boot, dtype=float))
    t = stat.size
    if boot.shape[1] != t:
        raise ValueError(
            f"romano_wolf: {boot.shape[1]} bootstrap columns against "
            f"{t} statistics")

    s = np.abs(stat) if two_sided else stat
    bs = np.abs(boot) if two_sided else boot

    rejected = np.zeros(t, dtype=bool)
    adjusted = np.ones(t, dtype=float)

    remaining = np.flatnonzero(np.isfinite(s))
    order = remaining[np.argsort(-s[remaining])]

    running = 0.0
    for position, i in enumerate(order):
        active = order[position:]
        # The stepdown maximum: over the hypotheses NOT yet rejected only.
        with np.errstate(invalid="ignore"):
            maxima = np.nanmax(bs[:, active], axis=1)
        p = float(np.mean(maxima >= s[i]))
        # Monotonicity: an adjusted p-value may never fall below one already
        # assigned to a larger statistic.
        running = max(running, p)
        adjusted[i] = running
        if running <= alpha:
            rejected[i] = True
        else:
            break                      # stepdown stops at the first failure

    return rejected, adjusted


def benjamini_yekutieli(pvalues: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """
    BH with the Benjamini-Yekutieli (2001) harmonic penalty, valid under ANY
    dependence. Reported as the conservative sensitivity bound beside BH's
    optimistic one.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    return benjamini_hochberg(p, q=q / c_m)


# ── Driscoll-Kraay cross-check ────────────────────────────────────────────────


def driscoll_kraay_se(dates: np.ndarray, values: np.ndarray,
                      max_lag: int | None = None) -> tuple[float, float, int]:
    """
    Driscoll-Kraay (1998) standard error of the mean of a panel quantity.

    Returns ``(mean, se, lags)``.

    For the mean of a scalar, Driscoll-Kraay reduces exactly to a Newey-West
    HAC applied to the SERIES OF DATE-LEVEL CROSS-SECTIONAL AVERAGES: averaging
    within a date absorbs arbitrary cross-sectional dependence, and the kernel
    then handles serial correlation across dates. That reduction is what makes
    this an INDEPENDENT check on the bootstrap rather than a second version of
    it — no resampling is involved at any point.

    ``max_lag`` defaults to the Newey-West rule of thumb 4*(T/100)^(2/9).
    """
    dates = np.asarray(dates)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    dates, values = dates[ok], values[ok]
    if values.size == 0:
        return float("nan"), float("nan"), 0

    order = np.argsort(dates, kind="stable")
    d, v = dates[order], values[order]
    bounds = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])
    means = np.array([v[lo:hi].mean() for lo, hi in zip(bounds[:-1], bounds[1:])])

    t = means.size
    if t < 3:
        return float(means.mean()), float("nan"), 0
    lags = int(max_lag if max_lag is not None
               else np.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, t - 1))

    e = means - means.mean()
    gamma0 = float((e * e).sum() / t)
    total = gamma0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)                    # Bartlett
        total += 2.0 * w * float((e[lag:] * e[:-lag]).sum() / t)
    total = max(total, 0.0)
    return float(means.mean()), float(np.sqrt(total / t)), lags


def _rank_product_panel(panel: "PanelMatrices") -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray]:
    """
    Per-(ticker, date) contributions whose PANEL MEAN is the mean per-date
    cross-sectional rank IC.

    A date's Spearman between two already-ranked vectors is
    ``sum_i r_p r_t / (N * s2_N)`` where ``s2_N`` is the mean squared rank on
    that date. Dividing each product by ``s2_N`` therefore makes the mean of the
    contributions ON THAT DATE equal that date's IC, and the grand mean equal
    the average of the per-date ICs — which is the quantity being cross-checked.

    Returns ``(entity, time, value)`` as long arrays.
    """
    ent, tim, val = [], [], []
    for fm in panel.folds:
        for d in range(fm.dates.size):
            p, t = fm.pred[d], fm.true[d]
            m = np.isfinite(p) & np.isfinite(t)
            if m.sum() < 3:
                continue
            s2 = float(np.mean(p[m] ** 2) * np.mean(t[m] ** 2)) ** 0.5
            if s2 <= 0:
                continue
            ent.append(np.flatnonzero(m))
            tim.append(np.full(int(m.sum()), fm.dates[d]))
            val.append(p[m] * t[m] / s2)
    if not ent:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=object), np.zeros(0)
    return (np.concatenate(ent), np.concatenate(tim), np.concatenate(val))


def driscoll_kraay_panel(panel: "PanelMatrices",
                         bandwidth: int | None = None) -> dict:
    """
    Driscoll-Kraay (1998) standard error on the panel mean, via
    ``linearmodels.PanelOLS`` with ``cov_type="kernel"``.

    This is the INDEPENDENT cross-check on the bootstrap that §3.3 of the
    Stage 0c spec calls for. Independent in the sense that matters: no
    resampling enters it anywhere. DK first averages within each time period —
    which absorbs cross-sectional dependence of arbitrary form, including both
    the common market factor and the -1/(N-1) the demeaning induces — and then
    applies a Bartlett kernel across periods for serial correlation.

    **If this and the bootstrap disagree substantially, neither is to be
    trusted and both are reported.** Picking the more favourable one is the
    error this whole track exists to stop making.

    ``linearmodels`` is imported lazily; see the module docstring on why it is
    not in ``requirements.txt``.
    """
    import pandas as pd
    from linearmodels.panel import PanelOLS

    ent, tim, val = _rank_product_panel(panel)
    if val.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n_obs": 0,
                "n_dates": 0, "bandwidth": 0, "engine": "none"}

    # An ORDINAL time index, not a parsed date. linearmodels only needs the
    # ordering to build the kernel, and `to_datetime` on a synthetic or
    # non-ISO label falls back to dateutil per element — slow, and a warning
    # that would fire on every test run.
    codes = {d: i for i, d in enumerate(np.unique(tim))}
    idx = pd.MultiIndex.from_arrays(
        [pd.Index(ent, name="entity"),
         pd.Index([codes[d] for d in tim], name="time")])
    frame = pd.DataFrame({"y": val, "const": 1.0}, index=idx)
    # Duplicate (entity, time) pairs cannot occur — one row per ticker per date.
    frame = frame[~frame.index.duplicated()]

    n_dates = frame.index.get_level_values("time").nunique()
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n_dates / 100.0) ** (2.0 / 9.0)))

    res = PanelOLS(frame["y"], frame[["const"]]).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=bandwidth)
    return {
        "mean": float(res.params["const"]),
        "se": float(res.std_errors["const"]),
        "n_obs": int(frame.shape[0]),
        "n_dates": int(n_dates),
        "bandwidth": int(bandwidth),
        "engine": "linearmodels.PanelOLS(cov_type='kernel')",
    }


def optimal_block_length(dates: np.ndarray, values: np.ndarray) -> float:
    """
    Politis-White (2004) automatic block length, with the Patton-Politis-White
    (2009) correction, on the date-level average series.

    Reported BESIDE the 30-session label horizon rather than instead of it. If
    the data-driven length materially exceeds 30 that is evidence of dependence
    beyond the labels themselves, which is a finding rather than a parameter to
    adopt silently.

    ``arch`` is imported lazily — see the module docstring on why it is not in
    requirements.txt.
    """
    from arch.bootstrap import optimal_block_length as _obl
    import pandas as pd

    dates = np.asarray(dates)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    dates, values = dates[ok], values[ok]
    order = np.argsort(dates, kind="stable")
    d, v = dates[order], values[order]
    bounds = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])
    means = np.array([v[lo:hi].mean() for lo, hi in zip(bounds[:-1], bounds[1:])])
    return float(_obl(pd.Series(means))["circular"].iloc[0])


# ── The graded panel ──────────────────────────────────────────────────────────


@dataclass
class TickerRow:
    ticker: str
    n_folds_scored: int
    hat_ic: float
    sigma2: float
    sigma2_source: str            # "bootstrap" or "between-fold fallback"
    theta: float
    shrinkage: float
    p_positive: float
    boot_p: float
    rw_adjusted_p: float
    rw_rejected: bool
    bh_significant: bool
    by_significant: bool
    grade: str
    reason: str


@dataclass
class PanelGradingV3:
    rows: list[TickerRow]
    tickers_total: int
    n_usable: int
    mu_hat: float
    mu_se_bootstrap: float
    mu_se_driscoll_kraay: float
    mu_se_naive: float
    z_bootstrap: float
    tau2_reml: float
    tau2_dl: float
    degeneracy: DegeneracyVerdict
    hksj: tuple[float, float, float, float]
    block_length: int
    block_length_auto: float
    n_bootstrap: int
    demean_report: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0
    grader_version: str = GRADER_VERSION_V3
    notes: list[str] = field(default_factory=list)
    #: The bootstrap distribution of mu_hat itself, kept so the report can look
    #: at its SHAPE. A standard error summarises a distribution as one number,
    #: and a bimodal or outlier-date-dominated one is not summarisable that way.
    boot_mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: The panel-level headline: mean per-date CROSS-SECTIONAL rank IC, which is
    #: a different quantity from the per-ticker grade and is never conflated
    #: with it.
    cross_sectional_ic: float = float("nan")
    cross_sectional_ic_se: float = float("nan")
    cross_sectional_dates: int = 0
    #: The linearmodels Driscoll-Kraay result, beside the closed-form one.
    dk_panel: dict = field(default_factory=dict)
    dk_lags: int = 0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.grade] = out.get(r.grade, 0) + 1
        return out

    def headline(self) -> str:
        """
        ONE statement when the panel is degenerate, the per-ticker table when it
        is not.

        This is the whole point of the detector: at tau2 = 0 every row carries
        an identical grade, and rendering 84 of them presents one fact as 84.
        """
        if self.n_usable == 0:
            return (f"NO ticker on this panel is gradeable: none has "
                    f"{MIN_FOLDS_FOR_ESTIMATE} walk-forward folds carrying an "
                    f"ordering after cross-sectional demeaning. There is "
                    f"nothing to grade, which is a finding rather than an "
                    f"error.")
        if self.degeneracy.degenerate and self.rows:
            grade = next((r.grade for r in self.rows
                          if r.grade != "INSUFFICIENT"), self.rows[0].grade)
            return self.degeneracy.statement(self.mu_hat, grade, len(self.rows))
        n_strong = self.counts().get("STRONG", 0)
        return (f"{n_strong} of {self.n_usable} gradeable tickers survive "
                f"Romano-Wolf at alpha = {FWER_ALPHA:.2f}; "
                f"mu_hat {self.mu_hat:+.5f} "
                f"(bootstrap SE {self.mu_se_bootstrap:.5f}, "
                f"z {self.z_bootstrap:+.2f}).")


# ── The orchestrator ──────────────────────────────────────────────────────────


def cross_sectional_ic_series(panel: PanelMatrices
                              ) -> tuple[np.ndarray, np.ndarray]:
    """
    The per-date cross-sectional rank IC — the PANEL-LEVEL headline, and a
    DIFFERENT quantity from the per-ticker one the dashboard grades.

    Per-ticker: "within this name, did the dates it ranked higher pay more".
    Per-date:   "on this date, did the names it ranked higher pay more".

    The second is what a long-short book earns and what every economic bar in
    this project measures; the first is what an individual company's badge
    claims. Both are reported, never conflated — Stage 0b's closing argument was
    precisely that the gate had been grading the first while the project was
    being measured on the second.
    """
    all_dates, all_ics = [], []
    for fm in panel.folds:
        ics = nan_rank_ic_rows(fm.pred, fm.true)
        keep = np.isfinite(ics)
        all_dates.append(fm.dates[keep])
        all_ics.append(ics[keep])
    if not all_dates:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(all_dates), np.concatenate(all_ics)


def _fallback_sigma2(fold_ics: np.ndarray) -> float:
    """
    Stage 0b's between-fold heuristic, kept ONLY for the low-fold fallback.

    ``var(fold ICs) / K`` is unbiased for the total variance of a mean over K
    periods — each fold IC already carries its own sampling noise — but it is
    very noisy at K = 5 and undefined below K = 2.

    **Where the bootstrap is in force this must NOT be added on top of it.** The
    bootstrap already contains the between-fold component, so summing the two
    double-counts. Which path a ticker took is recorded on its row.
    """
    ok = fold_ics[np.isfinite(fold_ics)]
    if ok.size < 2:
        return float("nan")
    return float(np.var(ok, ddof=1) / ok.size)


def _eb_pass(fold_ics: np.ndarray, sigma2: np.ndarray
             ) -> tuple[float, float, float, np.ndarray]:
    """
    One empirical-Bayes pass: REML tau2, precision-weighted mean, posteriors.

    Called once on the real sample and once inside EVERY bootstrap replicate,
    which is what makes the bootstrap an estimate of the whole pipeline's
    uncertainty rather than of its last step alone.
    """
    hat = np.nanmean(fold_ics, axis=0)
    ok = np.isfinite(hat) & np.isfinite(sigma2) & (sigma2 > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan"), 0.0, np.full(hat.size, np.nan)

    tau2 = reml_tau2(hat[ok], sigma2[ok])
    mu, mu_var = precision_weighted_mean(hat[ok], sigma2[ok] + tau2)

    theta = np.full(hat.size, np.nan)
    for i in np.flatnonzero(ok):
        theta[i] = shrink(float(hat[i]), float(sigma2[i]), mu, mu_var, tau2)[0]
    return mu, mu_var, tau2, theta


def grade_panel_v3(
    dates: np.ndarray,
    tickers: np.ndarray,
    folds: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    block: int = BLOCK_LENGTH_SESSIONS,
    n_bootstrap: int = BOOTSTRAP_B,
    seed: int = PANEL_BOOTSTRAP_SEED,
    alpha: float = FWER_ALPHA,
    min_cross_section: int = MIN_CROSS_SECTION,
    break_even: float = 0.00512363994209475,
    compute_auto_block: bool = True,
) -> PanelGradingV3:
    """
    The whole Stage 0c layer: demean, fold, bootstrap, shrink, adjust, grade.

    EVERY STANDARD ERROR PRODUCED HERE IS APPROXIMATE, and the output says so.
    Clustering is on the DATE dimension — about 64 independent 30-session
    windows on this panel — never on the fold dimension, because five clusters
    is nowhere near asymptotic. Cameron & Miller (2015) note there is no firm
    threshold, but applied practice treats fewer than ~30-40 clusters as the
    danger zone; at ~64 this panel is borderline rather than comfortable.
    """
    import time

    started = time.time()
    panel = build_panel(dates, tickers, folds, y_true, y_pred,
                        min_cross_section=min_cross_section)
    n_t = panel.tickers.size
    notes: list[str] = []

    # ── the real sample ───────────────────────────────────────────────────────
    fold_ics = per_ticker_fold_ics(panel)
    n_scored = np.isfinite(fold_ics).sum(axis=0)
    with np.errstate(invalid="ignore"):
        hat = np.nanmean(fold_ics, axis=0)

    # ── the bootstrap: a full EB re-run per replicate ─────────────────────────
    rng = np.random.default_rng(seed)
    boot_hat = np.full((n_bootstrap, n_t), np.nan)
    boot_mu = np.full(n_bootstrap, np.nan)
    boot_tau2 = np.full(n_bootstrap, np.nan)

    for b in range(n_bootstrap):
        sels = resample_selections(panel, block, rng)
        f_ic = per_ticker_fold_ics(panel, sels)
        s2 = np.array([_fallback_sigma2(f_ic[:, i]) for i in range(n_t)])
        mu_b, _, tau2_b, _ = _eb_pass(f_ic, s2)
        with np.errstate(invalid="ignore"):
            boot_hat[b] = np.nanmean(f_ic, axis=0)
        boot_mu[b] = mu_b
        boot_tau2[b] = tau2_b

    # sigma2_i IS the bootstrap spread of that ticker's own estimate. This is
    # the switch the pre-registration named: where the bootstrap is in force the
    # between-fold term is NOT added on top, because the bootstrap contains it.
    with np.errstate(invalid="ignore"):
        sigma2_boot = np.nanvar(boot_hat, axis=0, ddof=1)
    sigma2 = np.where(np.isfinite(sigma2_boot) & (sigma2_boot > 0),
                      sigma2_boot, np.nan)
    source = np.where(np.isfinite(sigma2), "bootstrap", "between-fold fallback")

    for i in range(n_t):
        if not np.isfinite(sigma2[i]):
            sigma2[i] = _fallback_sigma2(fold_ics[:, i])
            source[i] = "between-fold fallback"
            if np.isfinite(sigma2[i]):
                logger.info(
                    "stage0c: %s falls back to the between-fold variance; the "
                    "bootstrap spread was undefined", panel.tickers[i])

    usable = ((n_scored >= MIN_FOLDS_FOR_ESTIMATE) & np.isfinite(hat)
              & np.isfinite(sigma2) & (sigma2 > 0))

    # ── the panel statistics ──────────────────────────────────────────────────
    from pipeline.evidence_shrinkage import dersimonian_laird_tau2

    if usable.sum() >= 2:
        tau2 = reml_tau2(hat[usable], sigma2[usable])
        tau2_dl, _q_dl = dersimonian_laird_tau2(hat[usable], sigma2[usable])
        mu, mu_var = precision_weighted_mean(hat[usable], sigma2[usable] + tau2)
    else:
        tau2, tau2_dl = 0.0, 0.0
        mu, mu_var = float("nan"), float("nan")

    se_boot = float(np.nanstd(boot_mu, ddof=1)) if n_bootstrap > 1 else float("nan")
    z_boot = (float(mu / se_boot)
              if np.isfinite(se_boot) and se_boot > 0 else float("nan"))
    se_naive = (float(np.sqrt(mu_var))
                if np.isfinite(mu_var) and mu_var > 0 else float("nan"))

    cs_dates, cs_ics = cross_sectional_ic_series(panel)
    # TWO Driscoll-Kraay figures, deliberately. `driscoll_kraay_panel` is the
    # one the spec asked for — linearmodels.PanelOLS with a Bartlett kernel on
    # the (ticker, date) panel. `driscoll_kraay_se` is the closed-form
    # reduction: for the mean of a scalar, DK collapses to Newey-West on the
    # series of per-date averages. They should agree; if they do not, the
    # machinery is doing something the arithmetic does not.
    dk_mean, dk_se, dk_lags = driscoll_kraay_se(cs_dates, cs_ics)
    try:
        dk_panel = driscoll_kraay_panel(panel)
    except Exception as exc:                                  # noqa: BLE001
        dk_panel = {"mean": float("nan"), "se": float("nan"),
                    "engine": f"unavailable: {exc}"}
        notes.append(f"linearmodels Driscoll-Kraay unavailable: {exc}")

    auto_block = float("nan")
    if compute_auto_block:
        try:
            auto_block = optimal_block_length(cs_dates, cs_ics)
        except Exception as exc:                              # noqa: BLE001
            notes.append(f"automatic block length unavailable: {exc}")

    degeneracy = detect_tau2_degeneracy(tau2, hat[usable], sigma2[usable])
    hksj = hksj_interval(hat[usable], sigma2[usable], tau2)

    # ── multiplicity, on studentised statistics ───────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        se_i = np.sqrt(sigma2)
        stat = np.where(usable, hat / se_i, np.nan)
        boot_stat = (boot_hat - hat[None, :]) / se_i[None, :]
    boot_stat[:, ~usable] = np.nan
    rejected, rw_p = romano_wolf(stat, boot_stat, alpha=alpha)

    boot_p = np.full(n_t, np.nan)
    for i in np.flatnonzero(usable):
        col = boot_hat[:, i]
        col = col[np.isfinite(col)]
        if col.size:
            # A one-sided bootstrap p-value against theta <= 0, centred on the
            # observed estimate so the null is "this ticker has no edge".
            boot_p[i] = float(np.mean((col - hat[i]) >= hat[i]))

    finite_p = np.where(np.isfinite(boot_p), boot_p, 1.0)
    bh = benjamini_hochberg(finite_p, q=FDR_Q)
    by = benjamini_yekutieli(finite_p, q=FDR_Q)

    # ── posteriors and grades ─────────────────────────────────────────────────
    rows: list[TickerRow] = []
    for i, ticker in enumerate(panel.tickers):
        if not usable[i]:
            rows.append(TickerRow(
                str(ticker), int(n_scored[i]), float(hat[i]),
                float(sigma2[i]), str(source[i]),
                float("nan"), float("nan"), float("nan"), float("nan"),
                float("nan"), False, False, False, "INSUFFICIENT",
                f"only {int(n_scored[i])} of {fold_ics.shape[0]} folds carry an "
                f"ordering; {MIN_FOLDS_FOR_ESTIMATE} are required"))
            continue

        theta, b_i, post_var, _ = shrink(float(hat[i]), float(sigma2[i]),
                                         mu, mu_var, tau2)
        p_pos = posterior_probability_positive(theta, post_var)

        if bool(rejected[i]) and theta > break_even \
                and p_pos >= STRONG_POSTERIOR_THRESHOLD:
            grade = "STRONG"
            reason = (f"survives Romano-Wolf at alpha {alpha:.2f} (adjusted p "
                      f"{rw_p[i]:.3f}), posterior {theta:+.4f} clears the "
                      f"{break_even:.5f} break-even, P(theta>0) = {p_pos:.3f}")
        elif theta > 0 and p_pos >= STRONG_POSTERIOR_THRESHOLD:
            grade = "WEAK"
            reason = (f"posterior {theta:+.4f} is positive with P(theta>0) = "
                      f"{p_pos:.3f}, but it does not survive Romano-Wolf "
                      f"(adjusted p {rw_p[i]:.3f}) across {int(usable.sum())} "
                      f"simultaneous tests")
        else:
            grade = "INSUFFICIENT"
            reason = (f"posterior {theta:+.4f}, P(theta>0) = {p_pos:.3f}; no "
                      f"evidence at this sample size")

        rows.append(TickerRow(
            str(ticker), int(n_scored[i]), float(hat[i]), float(sigma2[i]),
            str(source[i]), float(theta), float(b_i), float(p_pos),
            float(boot_p[i]), float(rw_p[i]), bool(rejected[i]),
            bool(bh[i]), bool(by[i]), grade, reason))

    if degeneracy.degenerate:
        notes.append(
            "tau2 is at the zero boundary: every posterior collapses onto the "
            "grand mean, so the per-ticker rows carry ONE panel-level finding "
            "repeated, not one finding per company. Read headline().")
    if 0 < usable.sum() < n_t:
        notes.append(
            f"n_usable = {int(usable.sum())} of {n_t}. A degenerate fold has an "
            f"undefined correlation, so missingness depends on the MODEL'S OWN "
            f"behaviour - informative (MNAR), not MCAR. A variant grading few "
            f"names has not demonstrated broad skill and its mu_hat is not "
            f"comparable to one grading all of them.")
    n_windows = max(len(np.unique(cs_dates)) // max(block, 1), 1)
    notes.append(
        f"Every standard error here is APPROXIMATE. Clustering is on ~"
        f"{n_windows} independent {block}-session date-windows, which is "
        f"borderline for cluster-robust asymptotics (applied practice treats "
        f"below ~30-40 as the danger zone), and the HKSJ interval refers to t "
        f"on {max(int(usable.sum()) - 1, 0)} df while the panel holds about "
        f"five genuinely independent periods.")

    return PanelGradingV3(
        rows=rows, tickers_total=int(n_t), n_usable=int(usable.sum()),
        mu_hat=float(mu), mu_se_bootstrap=se_boot,
        mu_se_driscoll_kraay=float(dk_se), mu_se_naive=se_naive,
        z_bootstrap=z_boot, tau2_reml=float(tau2), tau2_dl=float(tau2_dl),
        degeneracy=degeneracy, hksj=hksj, block_length=int(block),
        block_length_auto=auto_block, n_bootstrap=int(n_bootstrap),
        demean_report=panel.demean_report,
        runtime_seconds=round(time.time() - started, 1), notes=notes,
        boot_mu=boot_mu, cross_sectional_ic=float(dk_mean),
        cross_sectional_ic_se=float(dk_se),
        cross_sectional_dates=int(cs_dates.size),
        dk_panel=dk_panel, dk_lags=int(dk_lags))
