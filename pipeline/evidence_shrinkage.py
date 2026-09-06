"""
pipeline/evidence_shrinkage.py — the evidence gate, measured honestly.

EVIDENCE-GRADING REDESIGN, STAGE 0. This is a separate track from the project's
own Phase 0-6 roadmap and neither renumbers the other. Stage 0 replaces the
MEASUREMENT, not the model: nothing here changes a feature, a target, a fold, a
horizon, a cost assumption or an agent. It reads the same held-out track record
the live gate reads and grades it a different way, in shadow, beside it.

--------------------------------------------------------------------------------
THE PROBLEM THIS EXISTS FOR
--------------------------------------------------------------------------------

`agents.critic_agent.grade_evidence` asks, of each ticker on its own, a question
that ticker's sample size cannot answer. Its only inferential check is
`eval_rank_ic_t >= 2.0`, and that t-statistic is built from

    n_effective = n_oos / horizon = 1909 / 30 = 63.6

independent observations, because consecutive 30-session labels share 29 of
their 30 sessions. At n = 64, t ~ IC * sqrt(n - 1), so a t of 2.0 demands a rank
IC of about **0.25**, and 80% power against it demands about **0.34**. Real
cross-sectional equity signals run 0.02-0.05. Detecting an IC of 0.04 at t > 2
needs on the order of 2,500 independent observations; this panel has 64 per
name.

So the check is not merely strict, it is asking for an effect size that does not
exist in monthly equity prediction. Measured over all 84 tickers, the three
checks pass at rates 0.385 / 0.042 / 0.042, giving an expected 3.12 names
clearing 2-of-3 under independence — and 3 clear it. The gate's yield is
indistinguishable from what chance produces, and the reason is arithmetic
rather than a defect in the code.

That is the textbook setting for PARTIAL POOLING: many units, each with too few
observations to speak for itself, all plausibly drawn from a common
distribution. The fix is not a lower threshold. It is to stop grading each
ticker in isolation and let the panel lend it precision.

--------------------------------------------------------------------------------
THE METHOD, AND THE EXACT ALGEBRA
--------------------------------------------------------------------------------

Closed-form empirical Bayes — James-Stein (1961) / Efron-Morris (1973, 1975)
shrinkage, with the between-unit variance estimated by the DerSimonian-Laird
(1986) method of moments, the standard random-effects meta-analysis estimator.
No MCMC: the whole thing is four lines of algebra over 84 pairs of numbers, and
a closed form that can be verified by hand is worth more here than a sampler
nobody re-derives.

For each ticker i, from its own out-of-sample series:

    hat_ic_i    the observed rank IC (point estimate, on the real series)
    sigma2_i    its sampling variance, from a MOVING-BLOCK bootstrap at block
                length = the 30-session label horizon

Then, over the panel:

    w_i      = 1 / sigma2_i
    mu_hat   = sum(w_i * hat_ic_i) / sum(w_i)          precision-weighted mean
    var_mu   = 1 / sum(w_i)                            its own variance
    Q        = sum(w_i * (hat_ic_i - mu_hat)^2)        Cochran's Q
    c        = sum(w_i) - sum(w_i^2) / sum(w_i)
    tau2     = max(0, (Q - (k - 1)) / c)               DerSimonian-Laird

and per ticker:

    B_i      = sigma2_i / (sigma2_i + tau2)            shrinkage toward mu_hat
    theta_i  = B_i * mu_hat + (1 - B_i) * hat_ic_i     posterior mean
    v_i      = sigma2_i * tau2 / (sigma2_i + tau2)  +  B_i^2 * var_mu
    p_i      = Phi(theta_i / sqrt(v_i))                P(theta_i > 0 | data)

**THE SECOND TERM IN v_i IS LOAD-BEARING AND IS NOT DECORATION.** The classical
Efron-Morris posterior variance is the first term alone, which conditions on
mu_hat and tau2 as though they were known. At tau2 = 0 — the single most likely
outcome on this panel, because the live per-ticker t-statistics already look
like standard normal draws — that first term is EXACTLY ZERO for every ticker.
Every theta_i collapses onto mu_hat with infinite confidence, every p_i becomes
0 or 1, and the run reports 84 STRONG grades manufactured out of a degenerate
limit. Morris' correction for the estimation of mu_hat (Morris 1983) supplies
the right limit: at tau2 = 0, B_i = 1 and v_i = var_mu, so the panel is saying
"the best estimate for every name is the common mean, known this precisely" —
which is a real statement, and a modest one.

That failure mode is written into `docs/stage0-preregistration.md` as the first
thing to check if the STRONG count jumps. It is exactly the shape of artifact
this project has been fooled by before: a plausible table, no error anywhere,
and a number produced by a division that should never have been performed.

--------------------------------------------------------------------------------
WHY A BLOCK BOOTSTRAP, AND WHY THE BLOCK IS 30
--------------------------------------------------------------------------------

The per-ticker out-of-sample series is CONTIGUOUS: `PurgedWalkForward` opens
each test window exactly where the previous one closed, and the purge and the
embargo are carved out of the TRAINING end. So there is no gap inside the series
for a resample to jump, and the only dependence a bootstrap has to respect is
the overlap between neighbouring labels.

Block length = `HORIZON_SESSIONS` = 30 is that overlap. A shorter block would
break rows apart that share 29 of their 30 forward sessions and hand back a
sigma2 that is too small — which would then make every posterior too confident,
which would then produce more STRONG grades. That is the same class of error as
quoting `daily_IC` beside `reb_t`, and it is the second thing to check if the
count jumps.

--------------------------------------------------------------------------------
WHAT THIS DOES NOT FIX
--------------------------------------------------------------------------------

The quantity being graded is inherited from the live gate and is not improved
here: a pooled per-ticker rank IC over ~1,900 dates can be moved by knowing
which fold a row came from, because each fold is a different fitted model and a
constant level shift between folds shows up as ordering. `TrainMeanForecast`
scored a pooled IC of -0.007 on exactly that mechanism. Stage 0's job is to
measure the gate's own quantity honestly, not to replace it; the replacement is
a later stage's question.

The posterior is a NORMAL approximation to an empirical-Bayes posterior with
tau2 and mu_hat plugged in. `p_positive` is therefore a good decision statistic
and not a calibrated probability, and it is reported as such.

--------------------------------------------------------------------------------
SHADOW MODE
--------------------------------------------------------------------------------

Nothing here writes to `forecast_current`, `forecasts`, `model_metadata` or any
column the public API serves. `forecast_confidence` is untouched and the old
gate keeps running unmodified as the baseline. Grades land in
`evidence_grades_v2`, keyed by run, APPEND-ONLY for the same reason
`forecast_outcomes` is: a grade that can be quietly improved after the fact is
not a record of what the measurement said.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import inspect, text

from data.db import get_engine, to_native_params
from pipeline.signals import HORIZON_SESSIONS

# ── Constants ─────────────────────────────────────────────────────────────────
#
# Every one of these is fixed in `docs/stage0-preregistration.md`, committed
# before the layer was run on a single real ticker. None of them may be moved
# after seeing a result in order to produce more STRONG grades.

#: Moving-block bootstrap resamples per ticker. 2,000 puts the Monte-Carlo error
#: on sigma2 an order of magnitude below the between-ticker spread that matters,
#: at a few seconds for the whole panel. More would be measurement theatre.
BOOTSTRAP_N_RESAMPLES = 2000

#: Seeded, and seeded PER TICKER off this base, so the panel result does not
#: depend on the order tickers happen to arrive in. Unseeded randomness in a
#: grading layer means the grade moves when nothing else did.
BOOTSTRAP_SEED = 20260906

#: Block length in rows. This is the label horizon, the purge width and the
#: embargo width — the same 30 — because it is the number of sessions two
#: neighbouring observations share. See the module docstring.
BLOCK_LENGTH_SESSIONS = HORIZON_SESSIONS

#: A ticker needs at least this many out-of-sample rows to be estimated at all.
#: Three blocks is the floor at which a moving-block bootstrap has anything to
#: move; below it the resample is close to the original sample every time and
#: sigma2 is an artifact of the block length rather than a measurement.
MIN_OOS_ROWS = 3 * BLOCK_LENGTH_SESSIONS

#: Fraction of bootstrap resamples that must yield a defined rank IC. A
#: constant prediction has no ordering, so its IC is undefined by design (the
#: same reason `zero` and `majority` produce NaN in the baseline table); a
#: ticker whose resamples mostly go undefined has not been measured and must
#: say so rather than return a variance computed from the survivors.
MIN_USABLE_RESAMPLE_FRACTION = 0.5

#: Benjamini-Hochberg false-discovery rate across the panel. FDR rather than
#: Bonferroni because the goal is to surface as many REAL weak effects as the
#: data support while controlling the proportion of discoveries that are false
#: — Bonferroni at 84 names controls something nobody asked about and would
#: bury every effect of realistic size.
FDR_Q = 0.10

#: Posterior probability of positive skill required for a positive grade.
STRONG_POSTERIOR_THRESHOLD = 0.90

#: Long-short spread produced per unit of rank IC, MEASURED on this panel by
#: `tools/run_portfolio.py` and recorded to `experiment_runs` on 2026-09-04.
#: Used only as the fallback when that row cannot be read; `economic_bar()`
#: prefers the recorded value. It is not a constant of nature.
FALLBACK_SPREAD_PER_IC = 0.3474

#: Turnover assumed when converting costs into a break-even IC. The same 0.80
#: `tools/run_portfolio.py` uses, so the two report the same bar.
ASSUMED_TURNOVER = 0.80

#: Version of THIS grading layer, stored on every row.
#:
#: `config_hash` and `data_hash` record the model's configuration and the
#: database's contents; neither moves when the GRADER changes, so without this
#: two runs of different grading logic would sit in one append-only table with
#: nothing to tell them apart. Bump it whenever a guard, a constant or the
#: decision table changes.
#:
#:   v1  first run, 2026-09-06 (run_id d692631c). No within-fold ordering
#:       guard: WIPRO.NS was graded ANTI_SIGNAL on a pooled IC built entirely
#:       from five fold-level constants.
#:   v2  adds the no-within-fold-ordering guard in `block_bootstrap_ic`.
GRADER_VERSION = "stage0-eb-shrinkage-v2"

GRADES = ("STRONG", "WEAK", "INSUFFICIENT", "ANTI_SIGNAL")


class EvidenceGradingRefused(RuntimeError):
    """Raised when the panel cannot support a grading run at all."""


# ── Inputs ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TickerTrack:
    """
    One ticker's held-out track record — the input this module consumes.

    Produced by the EXISTING per-ticker purged walk-forward
    (`pipeline.model.evaluate_ticker`), unmodified. This module never fits a
    model, never chooses a fold and never touches how these rows are made.
    """

    ticker: str
    dates: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    #: Which walk-forward fold each row came from. OPTIONAL, and supplying it
    #: enables the no-within-fold-ordering guard in `block_bootstrap_ic`. See
    #: that guard for why it matters and what it caught.
    folds: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not (len(self.dates) == len(self.y_true) == len(self.y_pred)):
            raise ValueError(
                f"{self.ticker}: dates/y_true/y_pred lengths disagree "
                f"({len(self.dates)}/{len(self.y_true)}/{len(self.y_pred)})")
        if self.folds is not None and len(self.folds) != len(self.y_true):
            raise ValueError(
                f"{self.ticker}: folds has {len(self.folds)} entries against "
                f"{len(self.y_true)} rows")


@dataclass
class BootstrapEstimate:
    """A ticker's point estimate and its sampling variance."""

    ticker: str
    n_oos: int
    n_blocks: int
    hat_ic: float
    sigma2: float
    n_valid_resamples: int
    usable: bool
    reason: str = ""


@dataclass
class TickerEvidence:
    """One graded row. Everything needed to re-derive the grade by hand."""

    ticker: str
    n_oos: int
    n_blocks: int
    hat_ic: float
    sigma2: float
    shrinkage_weight: float          # B_i, weight ON the grand mean
    theta: float                     # posterior mean
    post_var: float                  # Morris-corrected
    post_var_naive: float            # classical Efron-Morris, for audit
    p_positive: float
    p_two_sided: float
    bh_significant: bool
    grade_v2: str
    reason: str
    old_grade: str | None = None
    old_reasons: list[str] = field(default_factory=list)


@dataclass
class PanelGrading:
    """The whole run: per-ticker rows plus the panel-level diagnostics."""

    rows: list[TickerEvidence]
    mu_hat: float
    mu_var: float
    tau2: float
    q_statistic: float
    mu_hat_random_effects: float
    break_even_ic: float
    n_usable: int
    n_unusable: int
    fdr_q: float = FDR_Q
    posterior_threshold: float = STRONG_POSTERIOR_THRESHOLD
    block_length: int = BLOCK_LENGTH_SESSIONS
    n_resamples: int = BOOTSTRAP_N_RESAMPLES
    run_id: str = ""
    graded_at: str = ""
    grader_version: str = GRADER_VERSION
    model_version: str = ""
    config_hash: str | None = None
    data_hash: str | None = None

    @property
    def degenerate(self) -> bool:
        """
        True when ``tau2_hat`` came back at zero — no detectable between-ticker
        variation in skill.

        THIS IS A REAL AND LIKELY OUTCOME, NOT AN ERROR, AND IT HAS TO BE SAID
        OUT LOUD. At tau2 = 0 the shrinkage weight is 1 for every ticker, so
        every posterior mean IS the grand mean and every ticker receives an
        IDENTICAL p-value and an identical grade. The panel is then making one
        statement about the universe and printing it 84 times; no name has been
        distinguished from any other, and a badge that every row carries
        conveys nothing about the row it sits on.

        It also means the whole board can flip together. If `mu_hat` were
        significantly positive and above the break-even bar, this code would
        grade all 84 names STRONG off a single common estimate — which is the
        arithmetic working correctly and would still be a badge nobody should
        read as per-ticker evidence. Callers must surface this flag beside the
        counts rather than reporting the counts alone.
        """
        return not (self.tau2 > 0)

    def counts(self) -> dict[str, int]:
        out = {g: 0 for g in GRADES}
        for row in self.rows:
            out[row.grade_v2] += 1
        return out

    def crosstab(self) -> pd.DataFrame:
        """Old grade (rows) against new grade (columns), over every ticker."""
        frame = pd.DataFrame([{"old": r.old_grade or "UNGRADED",
                               "new": r.grade_v2} for r in self.rows])
        if frame.empty:
            return pd.DataFrame()
        table = pd.crosstab(frame["old"], frame["new"])
        old_order = [g for g in ("STRONG", "WEAK", "INSUFFICIENT", "UNGRADED")
                     if g in table.index]
        new_order = [g for g in GRADES if g in table.columns]
        return table.loc[old_order, new_order]

    def to_metrics(self) -> dict:
        """The shape `experiment_runs.metrics` gets, mirroring `to_metrics()`
        elsewhere in the pipeline so the weekly row reads the same way."""
        return {
            "mu_hat": self.mu_hat,
            "mu_var": self.mu_var,
            "tau2_hat": self.tau2,
            "q_statistic": self.q_statistic,
            "mu_hat_random_effects": self.mu_hat_random_effects,
            "break_even_ic": self.break_even_ic,
            "n_usable": self.n_usable,
            "n_unusable": self.n_unusable,
            "fdr_q": self.fdr_q,
            "posterior_threshold": self.posterior_threshold,
            "block_length": self.block_length,
            "n_resamples": self.n_resamples,
            "degenerate_panel": self.degenerate,
            "grade_counts": self.counts(),
            "old_grade_counts": _old_counts(self.rows),
        }

    def summary(self) -> str:
        c = self.counts()
        return (f"mu_hat {self.mu_hat:+.4f} (sd {np.sqrt(self.mu_var):.4f})  "
                f"tau2 {self.tau2:.6f}  break-even {self.break_even_ic:.4f}  "
                f"-> STRONG {c['STRONG']} / WEAK {c['WEAK']} / "
                f"ANTI {c['ANTI_SIGNAL']} / INSUFFICIENT {c['INSUFFICIENT']}")


def _old_counts(rows: list[TickerEvidence]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = row.old_grade or "UNGRADED"
        out[key] = out.get(key, 0) + 1
    return out


# ── Rank IC, vectorised over bootstrap resamples ──────────────────────────────


def rank_ic_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Spearman rank correlation of each ROW of ``a`` against each row of ``b``.

    A SECOND IMPLEMENTATION OF A QUANTITY THAT ALREADY HAS ONE, and that is a
    debt rather than a feature. `pipeline.evaluation.rank_ic` is the definition;
    calling it 2,000 times per ticker across 84 tickers is ~170,000 scipy calls,
    which is minutes of wall clock for a number that is four lines of linear
    algebra on ranks. The debt is paid by a test that runs both over random
    inputs and requires agreement to 1e-12 — the same round-trip guarantee
    `tools/kronos_kaggle.py` carries against `tools/score_kronos.py`, and for
    the same reason: two copies of a formula drift silently, and the only thing
    that stops them is a test that would fail if they did.

    Ties take average ranks, which is what `scipy.stats.spearmanr` does. A row
    with no ordering on either side yields NaN rather than 0.0, because an
    undefined correlation is not a measured zero.
    """
    ra = stats.rankdata(a, axis=1).astype(float)
    rb = stats.rankdata(b, axis=1).astype(float)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    return out


def block_bootstrap_ic(
    track: TickerTrack,
    block: int = BLOCK_LENGTH_SESSIONS,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapEstimate:
    """
    Point estimate and sampling variance of one ticker's out-of-sample rank IC.

    Moving-block bootstrap: ``ceil(n / block)`` blocks of ``block`` consecutive
    rows are drawn with replacement from every position the series admits, and
    concatenated to length n. Both series are resampled with the SAME index, so
    a prediction always travels with the outcome it was made for — resampling
    them independently would destroy the very correlation being measured and
    return a null variance around a null estimate.

    The seed is derived from the TICKER as well as the base seed, so the run is
    reproducible and independent of the order tickers are processed in.
    """
    n = len(track.y_true)
    valid = np.isfinite(track.y_true) & np.isfinite(track.y_pred)
    y_true = track.y_true[valid]
    y_pred = track.y_pred[valid]
    n_valid = len(y_true)

    if n_valid < MIN_OOS_ROWS:
        return BootstrapEstimate(
            track.ticker, n, 0, float("nan"), float("nan"), 0, False,
            f"only {n_valid} finite out-of-sample rows, need {MIN_OOS_ROWS} "
            f"({MIN_OOS_ROWS // block} blocks of {block})")

    hat = rank_ic_rows(y_true[None, :], y_pred[None, :])[0]
    if not np.isfinite(hat):
        return BootstrapEstimate(
            track.ticker, n, 0, float("nan"), float("nan"), 0, False,
            "the out-of-sample prediction (or the outcome) is constant, so "
            "rank IC is undefined rather than zero")

    # A PREDICTION WITH NO ORDERING MUST EARN NO RANKING RESULT, and the check
    # above is not enough to enforce it.
    #
    # It asks whether the prediction is constant over the WHOLE out-of-sample
    # series. A per-ticker model that emits one constant per FOLD passes that
    # test easily — the five constants differ — while containing no ordering
    # information anywhere. Its pooled rank IC is then 100% the arrangement of
    # five fold-level constants against five realised period returns: a
    # correlation over FIVE points, reported as a per-company track record.
    #
    # THIS IS NOT HYPOTHETICAL, AND IT COST THIS RUN ITS ONLY NON-NULL GRADE.
    # Measured 2026-09-06 over the 84-ticker panel: 316 of 420 (ticker, fold)
    # cells emit a constant, 21 tickers do so in ALL FIVE folds, and the panel's
    # mean per-ticker IC is -0.070 pooled against +0.126 within folds — opposite
    # signs. WIPRO.NS was graded ANTI_SIGNAL on a pooled IC of -0.3914 whose
    # within-fold IC is UNDEFINED in every one of its five folds.
    #
    # It is the `TrainMeanForecast` finding in a new place: that comparator
    # scored a pooled IC of -0.007 while emitting one constant per fold and
    # holding no ranking information at all, and `daily_rank_ic` was added to
    # the baseline table precisely because the pooled figure could not tell the
    # difference. This guard is the per-ticker equivalent.
    #
    # Added AFTER the first run of this layer, and it can only REMOVE a grade,
    # never create one — the same property that made the live gate's signed-t
    # fix safe to ship mid-flight. It is not a pre-registered threshold and
    # moves none of them.
    if track.folds is not None:
        fold_ids = np.asarray(track.folds)[valid]
        if fold_ids.size and all(
                np.ptp(y_pred[fold_ids == k]) == 0
                for k in np.unique(fold_ids)):
            return BootstrapEstimate(
                track.ticker, n, 0, float("nan"), float("nan"), 0, False,
                f"the prediction is CONSTANT within every one of its "
                f"{len(np.unique(fold_ids))} walk-forward folds, so it holds no "
                f"ordering anywhere; the pooled rank IC of {hat:+.4f} is "
                f"entirely the arrangement of those fold-level constants "
                f"against the realised period returns")

    n_blocks = int(np.ceil(n_valid / block))
    n_starts = n_valid - block + 1

    rng = np.random.default_rng(
        [seed, int.from_bytes(track.ticker.encode("utf-8"), "little")])
    starts = rng.integers(0, n_starts, size=(n_resamples, n_blocks))
    # (n_resamples, n_blocks, block) -> (n_resamples, n_blocks * block), then
    # truncated back to the original length so every resample is the same size
    # as the sample it estimates.
    idx = (starts[:, :, None] + np.arange(block)[None, None, :])
    idx = idx.reshape(n_resamples, n_blocks * block)[:, :n_valid]

    boot = rank_ic_rows(y_true[idx], y_pred[idx])
    n_ok = int(np.isfinite(boot).sum())

    if n_ok < MIN_USABLE_RESAMPLE_FRACTION * n_resamples:
        return BootstrapEstimate(
            track.ticker, n, n_blocks, float(hat), float("nan"), n_ok, False,
            f"only {n_ok} of {n_resamples} bootstrap resamples produced a "
            f"defined rank IC")

    sigma2 = float(np.nanvar(boot, ddof=1))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return BootstrapEstimate(
            track.ticker, n, n_blocks, float(hat), float("nan"), n_ok, False,
            "bootstrap sampling variance is zero or non-finite")

    return BootstrapEstimate(track.ticker, n, n_blocks, float(hat), sigma2,
                             n_ok, True)


# ── Empirical Bayes ───────────────────────────────────────────────────────────


def precision_weighted_mean(hat_ic: np.ndarray,
                            sigma2: np.ndarray) -> tuple[float, float]:
    """
    The fixed-effect grand mean and its variance — the shrinkage target.

    Precision-weighted, NOT a plain average: a ticker measured to sigma2 twice
    another's carries half the weight, which is the whole content of the
    Efron-Morris construction. A simple mean would let the noisiest names drag
    the target every ticker is shrunk toward.

    **ONE DISCLOSED APPROXIMATION.** ``var_mu = 1 / sum(w)`` is the FIXED-EFFECT
    variance of this mean. Under a random-effects model the same weighted mean
    has variance ``sum(w^2 * (sigma2 + tau2)) / sum(w)^2``, which is larger
    whenever ``tau2 > 0``, so this understates it — an anti-conservative
    approximation, stated rather than hidden. Two things bound the damage: at
    ``tau2 = 0``, which is the expected outcome on this panel, the two are
    IDENTICAL; and where ``tau2 > 0`` this quantity enters the posterior only
    through ``B_i^2 * var_mu``, which is second-order beside the
    ``sigma2*tau2/(sigma2+tau2)`` term that then dominates. The formula is the
    one fixed in `docs/stage0-preregistration.md` and is left alone rather than
    quietly improved after the fact.
    """
    w = 1.0 / sigma2
    total = float(w.sum())
    return float((w * hat_ic).sum() / total), float(1.0 / total)


def dersimonian_laird_tau2(hat_ic: np.ndarray,
                           sigma2: np.ndarray) -> tuple[float, float]:
    """
    Between-ticker variance by DerSimonian-Laird (1986) method of moments.

    Returns ``(tau2, Q)``. Cochran's Q has expectation ``k - 1`` under the null
    that every ticker shares one true IC, so ``Q - (k - 1)`` is the excess
    dispersion the sampling error alone cannot explain, and ``c`` converts it
    from Q's units into a variance.

    **Clipped at zero, by the estimator's own convention.** A negative
    method-of-moments estimate means the observed spread is SMALLER than
    sampling error alone predicts — there is no between-ticker variation to
    detect, and zero is the honest report. On this panel that is the expected
    outcome, and it is why the posterior variance carries Morris' correction:
    see the module docstring.
    """
    k = len(hat_ic)
    if k < 2:
        return 0.0, float("nan")

    w = 1.0 / sigma2
    mu = float((w * hat_ic).sum() / w.sum())
    q = float((w * (hat_ic - mu) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    if c <= 0:
        return 0.0, q
    return max(0.0, (q - (k - 1)) / c), q


def shrink(hat_ic: float, sigma2: float, mu_hat: float, mu_var: float,
           tau2: float) -> tuple[float, float, float, float]:
    """
    One ticker's posterior. Returns ``(theta, B, post_var, post_var_naive)``.

        B        = sigma2 / (sigma2 + tau2)
        theta    = B * mu_hat + (1 - B) * hat_ic
        naive    = sigma2 * tau2 / (sigma2 + tau2)
        post_var = naive + B^2 * mu_var

    See the module docstring for why the second term of ``post_var`` is not
    optional. At ``tau2 == 0`` it is the only term left, and without it the
    posterior variance is exactly zero for every ticker on the panel.
    """
    denom = sigma2 + tau2
    b = 1.0 if denom <= 0 else sigma2 / denom
    theta = b * mu_hat + (1.0 - b) * hat_ic
    naive = 0.0 if denom <= 0 else sigma2 * tau2 / denom
    return float(theta), float(b), float(naive + b * b * mu_var), float(naive)


def posterior_probability_positive(theta: float, post_var: float) -> float:
    """
    ``P(theta_i > 0 | data)`` under a Normal approximation to the posterior.

    A DECISION STATISTIC, not a calibrated probability: tau2 and mu_hat are
    plugged in as though known (beyond the mean-estimation term already carried
    in ``post_var``), and the posterior of a bounded correlation is not exactly
    Normal. Reported as such rather than dressed up.
    """
    if not np.isfinite(theta) or not np.isfinite(post_var) or post_var <= 0:
        return float("nan")
    return float(stats.norm.cdf(theta / np.sqrt(post_var)))


def benjamini_hochberg(pvalues: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """
    Benjamini-Hochberg (1995) step-up procedure. Returns a boolean mask.

    Sorted p-values are compared against ``q * rank / m``; every hypothesis up
    to and including the LARGEST rank that clears its threshold is rejected —
    the step-up part, and the part a naive implementation gets wrong by
    stopping at the first failure instead of the last success.

    Implemented here rather than taken from `statsmodels` deliberately.
    `requirements.txt` is installed by Render, by the daily pipeline and by the
    weekly evaluation, and this project has a standing rule against growing that
    file for anything the live path does not need (the same rule that keeps
    torch out of it). Six lines against a dependency on three deployment targets
    is not a close call. `tests/test_stage0_evidence_shrinkage.py` cross-checks
    this against `statsmodels.stats.multitest.multipletests` when that package
    happens to be importable, and against a hand-verified vector when it is not.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    out = np.zeros(m, dtype=bool)
    if m == 0:
        return out
    if not np.isfinite(p).all():
        raise ValueError("benjamini_hochberg received a non-finite p-value; "
                         "unusable tickers must be excluded from the family "
                         "before it is formed, not passed in as NaN")

    order = np.argsort(p, kind="mergesort")
    thresholds = q * np.arange(1, m + 1) / m
    passing = np.nonzero(p[order] <= thresholds)[0]
    if len(passing):
        out[order[: passing.max() + 1]] = True
    return out


# ── The economic bar ──────────────────────────────────────────────────────────


def economic_bar(engine=None, spread_per_ic: float | None = None,
                 turnover: float = ASSUMED_TURNOVER) -> tuple[float, dict]:
    """
    The rank IC an ordering must carry to pay for its own trading costs.

    REUSES `pipeline.portfolio.break_even_ic` rather than recomputing it — that
    function owns the measured Zerodha/NSE cost schedule and the annualisation,
    and a second copy of a cost model is a second thing to keep in step.
    `spread_per_ic` is read from the most recent `experiment_runs` row that
    recorded one (`tools/run_portfolio.py --record`, 2026-09-04: 0.3474),
    because it is a measurement on this panel and not a constant.

    **THE TWO QUANTITIES ARE NOT THE SAME AND THIS IS DISCLOSED RATHER THAN
    SMOOTHED OVER.** `break_even_ic` is derived from a CROSS-SECTIONAL
    long-short book: the spread it earns per unit of rank IC was measured by
    sorting the panel on each date. What is being graded here is a per-ticker
    TIME-SERIES rank IC, which would be traded by timing one name. The bar is
    therefore an ANALOGY, not an equivalence, and the direction of the
    approximation is unknown.

    It is used accordingly: as an extra hurdle that STRONG must clear, never as
    something that can create a grade. Like the signed-t fix before it, this
    check can only remove grades, never add them — which is what makes it safe
    to apply a bar whose transfer between the two quantities is unproven.
    """
    from pipeline.portfolio import CostModel, break_even_ic

    detail: dict = {"spread_per_ic_source": "argument"}
    if spread_per_ic is None:
        spread_per_ic, source = _recorded_spread_per_ic(engine)
        detail["spread_per_ic_source"] = source

    bar = break_even_ic(CostModel(), spread_per_ic=spread_per_ic,
                        turnover=turnover)
    detail.update(bar)
    return float(bar["break_even_rank_ic"]), detail


def _recorded_spread_per_ic(engine=None) -> tuple[float, str]:
    """The last measured spread-per-IC, or the recorded fallback with a reason."""
    try:
        engine = engine or get_engine()
        row = pd.read_sql(
            text("SELECT metrics FROM experiment_runs "
                 "WHERE metrics LIKE '%spread_per_ic%' "
                 "ORDER BY started_at DESC LIMIT 1"),
            engine,
        )
        if not row.empty:
            metrics = json.loads(row.iloc[0]["metrics"])
            value = metrics.get("portfolio", {}).get("spread_per_ic")
            if value and np.isfinite(float(value)) and float(value) > 0:
                return float(value), "experiment_runs"
    except Exception as exc:                                    # noqa: BLE001
        return FALLBACK_SPREAD_PER_IC, f"fallback ({str(exc)[:80]})"
    return FALLBACK_SPREAD_PER_IC, "fallback (no recorded run)"


# ── Grading ───────────────────────────────────────────────────────────────────


def assign_grade(bh_significant: bool, theta: float, p_positive: float,
                 break_even: float,
                 posterior_threshold: float = STRONG_POSTERIOR_THRESHOLD,
                 ) -> tuple[str, str]:
    """
    The decision table, pre-registered on 2026-09-06 and quoted here verbatim.

    | BH significant | sign(theta) | p_positive >= thr | theta >= break-even | grade       |
    |----------------|-------------|-------------------|---------------------|-------------|
    | yes            | negative    | -                 | -                   | ANTI_SIGNAL |
    | yes            | positive    | yes               | yes                 | STRONG      |
    | yes            | positive    | yes               | no                  | WEAK        |
    | yes            | positive    | no                | -                   | WEAK        |
    | no             | -           | yes               | -                   | WEAK        |
    | no             | -           | no                | -                   | INSUFFICIENT|

    ANTI_SIGNAL EXISTS BECAUSE THE LIVE GATE HAD TO LEARN THIS ONCE ALREADY.
    Its t-check read `abs(ic_t) >= 2.0` until 2026-09-02, and measured over 96
    tickers the only four names that ever passed it had strongly NEGATIVE IC —
    MUTHOOTFIN -0.253, TRENT -0.295, HDFCAMC -0.321, LT -0.263 — each one lucky
    second check away from being published as validated. The fix made the check
    signed. The symmetric mistake is to fold "reliably backwards" back into
    "no evidence", which throws away a real measurement, so it gets its own
    branch and its own label.

    The fourth row is unreachable at the configured constants — BH significance
    at q = 0.10 on a two-sided p implies a tail below 0.05, hence p_positive
    above 0.95 — and is written out anyway rather than assumed away, because
    both constants are named and either could move.
    """
    if bh_significant and theta < 0:
        return "ANTI_SIGNAL", (
            f"Posterior IC {theta:+.4f} is negative and survives "
            f"Benjamini-Hochberg control across the panel: the out-of-sample "
            f"ordering is reliably backwards. A real measurement, and it counts "
            f"against this forecast rather than for it.")

    if bh_significant and theta > 0:
        if p_positive >= posterior_threshold and theta >= break_even:
            return "STRONG", (
                f"Posterior IC {theta:+.4f} with P(skill) {p_positive:.3f} "
                f"survives BH control at the panel level AND clears the "
                f"{break_even:.4f} break-even cost bar.")
        if p_positive >= posterior_threshold:
            return "WEAK", (
                f"Posterior IC {theta:+.4f} with P(skill) {p_positive:.3f} "
                f"survives BH control, but sits below the {break_even:.4f} "
                f"break-even bar: statistically real, economically below its "
                f"own trading costs.")
        return "WEAK", (
            f"Posterior IC {theta:+.4f} survives BH control but P(skill) "
            f"{p_positive:.3f} is below the {posterior_threshold:.2f} bar.")

    if p_positive >= posterior_threshold:
        return "WEAK", (
            f"Posterior IC {theta:+.4f} with P(skill) {p_positive:.3f}, but it "
            f"does not survive Benjamini-Hochberg control across the panel — "
            f"at 84 names, this many would be expected by chance.")

    return "INSUFFICIENT", (
        f"Posterior IC {theta:+.4f} with P(skill) {p_positive:.3f}: after "
        f"shrinking toward the panel, indistinguishable from no skill.")


def grade_panel(
    tracks: list[TickerTrack],
    old_grades: dict[str, tuple[str, list[str]]] | None = None,
    engine=None,
    block: int = BLOCK_LENGTH_SESSIONS,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    fdr_q: float = FDR_Q,
    posterior_threshold: float = STRONG_POSTERIOR_THRESHOLD,
    spread_per_ic: float | None = None,
    break_even: float | None = None,
) -> PanelGrading:
    """
    Bootstrap, pool, shrink, control and grade — the whole Stage 0 layer.

    A ticker that cannot be estimated is graded INSUFFICIENT with its reason
    and is EXCLUDED FROM THE BH FAMILY. Including it as a NaN would either
    crash the procedure or, worse, inflate the family size with hypotheses that
    were never tested — which makes every real p-value harder to reject for a
    bookkeeping reason. The family is the set of tickers actually measured, and
    its size is reported.
    """
    from pipeline.model import MODEL_VERSION

    estimates = [block_bootstrap_ic(t, block=block, n_resamples=n_resamples,
                                    seed=seed) for t in tracks]
    usable = [e for e in estimates if e.usable]
    if len(usable) < 2:
        raise EvidenceGradingRefused(
            f"only {len(usable)} of {len(estimates)} tickers produced a usable "
            f"IC estimate — partial pooling needs a panel to pool over")

    hat = np.array([e.hat_ic for e in usable], dtype=float)
    s2 = np.array([e.sigma2 for e in usable], dtype=float)

    mu_hat, mu_var = precision_weighted_mean(hat, s2)
    tau2, q_stat = dersimonian_laird_tau2(hat, s2)

    # Reported as a diagnostic only. The random-effects mean is the more usual
    # shrinkage target in meta-analysis; the fixed-effect mean is what this
    # stage pre-registered, and printing both makes the choice auditable rather
    # than invisible. At tau2 = 0 they are identical by construction.
    w_re = 1.0 / (s2 + tau2)
    mu_re = float((w_re * hat).sum() / w_re.sum())

    if break_even is None:
        break_even, _ = economic_bar(engine=engine, spread_per_ic=spread_per_ic)

    posteriors: dict[str, tuple[float, float, float, float, float]] = {}
    p_two = np.empty(len(usable), dtype=float)
    for i, est in enumerate(usable):
        theta, b, post_var, naive = shrink(est.hat_ic, est.sigma2, mu_hat,
                                           mu_var, tau2)
        p_pos = posterior_probability_positive(theta, post_var)
        p_two[i] = 2.0 * min(p_pos, 1.0 - p_pos)
        posteriors[est.ticker] = (theta, b, post_var, naive, p_pos)

    significant = dict(zip([e.ticker for e in usable],
                           benjamini_hochberg(p_two, fdr_q)))
    two_sided = dict(zip([e.ticker for e in usable], p_two))

    rows: list[TickerEvidence] = []
    for est in estimates:
        old = (old_grades or {}).get(est.ticker)
        if not est.usable:
            rows.append(TickerEvidence(
                ticker=est.ticker, n_oos=est.n_oos, n_blocks=est.n_blocks,
                hat_ic=est.hat_ic, sigma2=est.sigma2, shrinkage_weight=float("nan"),
                theta=float("nan"), post_var=float("nan"),
                post_var_naive=float("nan"), p_positive=float("nan"),
                p_two_sided=float("nan"), bh_significant=False,
                grade_v2="INSUFFICIENT",
                reason=f"Not measurable: {est.reason}. Excluded from the "
                       f"Benjamini-Hochberg family rather than counted as a "
                       f"test that was run.",
                old_grade=old[0] if old else None,
                old_reasons=list(old[1]) if old else []))
            continue

        theta, b, post_var, naive, p_pos = posteriors[est.ticker]
        grade, reason = assign_grade(significant[est.ticker], theta, p_pos,
                                     break_even, posterior_threshold)
        if tau2 <= 0:
            # The disclosure travels WITH the grade into the stored row, not
            # just into a report someone may not read. See PanelGrading.degenerate.
            reason += (" NOTE: tau2_hat came back at zero, so this grade is the "
                       "PANEL's grade — every measured ticker in this run shares "
                       "the same posterior and the same badge, and nothing here "
                       "distinguishes this name from any other.")
        rows.append(TickerEvidence(
            ticker=est.ticker, n_oos=est.n_oos, n_blocks=est.n_blocks,
            hat_ic=est.hat_ic, sigma2=est.sigma2, shrinkage_weight=b,
            theta=theta, post_var=post_var, post_var_naive=naive,
            p_positive=p_pos, p_two_sided=two_sided[est.ticker],
            bh_significant=bool(significant[est.ticker]),
            grade_v2=grade, reason=reason,
            old_grade=old[0] if old else None,
            old_reasons=list(old[1]) if old else []))

    cfg = data = None
    try:
        from pipeline.tracking import config_hash, data_hash
        cfg, _ = config_hash()
        data, _ = data_hash([t.ticker for t in tracks], engine)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[EvidenceV2] provenance hashes unavailable: {exc}")

    return PanelGrading(
        rows=rows, mu_hat=mu_hat, mu_var=mu_var, tau2=tau2, q_statistic=q_stat,
        mu_hat_random_effects=mu_re, break_even_ic=float(break_even),
        n_usable=len(usable), n_unusable=len(estimates) - len(usable),
        fdr_q=fdr_q, posterior_threshold=posterior_threshold, block_length=block,
        n_resamples=n_resamples, run_id=uuid.uuid4().hex[:16],
        graded_at=datetime.now(timezone.utc).isoformat(),
        model_version=MODEL_VERSION, config_hash=cfg, data_hash=data)


# ── Storage: shadow, and append-only ──────────────────────────────────────────


def init_evidence_tables(engine=None) -> None:
    """
    Creates `evidence_grades_v2`. Module-local DDL, following
    `data.universe.init_universe_tables` rather than `data.db.init_db`.

    Deliberate: `data/db.py` is imported by the API, so putting the DDL there
    would make a shadow-mode research table a reason to redeploy Render. This
    table is written by one weekly step and read by one tool; nothing the public
    API serves knows it exists, which is what shadow mode means.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS evidence_grades_v2 (
                run_id                    TEXT NOT NULL,
                ticker                    TEXT NOT NULL,
                graded_at                 TEXT NOT NULL,
                model_version             TEXT,
                grader_version            TEXT,
                config_hash               TEXT,
                data_hash                 TEXT,
                n_oos                     INTEGER,
                n_blocks                  INTEGER,
                block_length              INTEGER,
                n_resamples               INTEGER,
                hat_ic                    REAL,
                sigma2                    REAL,
                mu_hat                    REAL,
                mu_var                    REAL,
                tau2_hat                  REAL,
                shrinkage_weight          REAL,
                theta_posterior_mean      REAL,
                theta_posterior_var       REAL,
                theta_posterior_var_naive REAL,
                p_positive_skill          REAL,
                p_two_sided               REAL,
                bh_significant            INTEGER,
                fdr_q                     REAL,
                break_even_ic             REAL,
                grade_v2                  TEXT,
                grade_v2_reason           TEXT,
                old_grade                 TEXT,
                PRIMARY KEY (run_id, ticker)
            )
        """))

        # Additive column migration, the `data.db.init_db` pattern.
        #
        # `CREATE TABLE IF NOT EXISTS` is a no-op against a table that already
        # exists, so a column added to the DDL above never reaches a database
        # created before it. Measured the hard way on 2026-09-06: `store_grading`
        # raised `UndefinedColumn: grader_version` against the table its own
        # first run had created twenty minutes earlier. Loud, and therefore the
        # good version of this failure — but only because the INSERT names its
        # columns explicitly.
        existing = {c["name"] for c in inspect(engine).get_columns(
            "evidence_grades_v2")}
        for column, coltype in (("grader_version", "TEXT"),):
            if column not in existing:
                conn.execute(text(f"ALTER TABLE evidence_grades_v2 "
                                  f"ADD COLUMN {column} {coltype}"))

        conn.commit()


def store_grading(grading: PanelGrading, engine=None) -> int:
    """
    Writes one graded run. APPEND-ONLY, keyed by ``(run_id, ticker)``.

    Never an upsert on ticker alone, for the same reason `forecast_outcomes` is
    written with ON CONFLICT DO NOTHING: the point of a measurement record is
    that a past reading cannot be quietly improved. Two runs a week apart are
    two rows, and the difference between them is the finding.
    """
    engine = engine or get_engine()
    init_evidence_tables(engine)

    cols = ("run_id, ticker, graded_at, model_version, grader_version, "
            "config_hash, data_hash, "
            "n_oos, n_blocks, block_length, n_resamples, hat_ic, sigma2, "
            "mu_hat, mu_var, tau2_hat, shrinkage_weight, theta_posterior_mean, "
            "theta_posterior_var, theta_posterior_var_naive, p_positive_skill, "
            "p_two_sided, bh_significant, fdr_q, break_even_ic, grade_v2, "
            "grade_v2_reason, old_grade")
    binds = ", ".join(f":{c.strip()}" for c in cols.split(","))

    written = 0
    with engine.connect() as conn:
        for row in grading.rows:
            conn.execute(
                text(f"INSERT INTO evidence_grades_v2 ({cols}) VALUES ({binds})"),
                to_native_params({
                    "run_id": grading.run_id, "ticker": row.ticker,
                    "graded_at": grading.graded_at,
                    "model_version": grading.model_version,
                    "grader_version": grading.grader_version,
                    "config_hash": grading.config_hash,
                    "data_hash": grading.data_hash,
                    "n_oos": row.n_oos, "n_blocks": row.n_blocks,
                    "block_length": grading.block_length,
                    "n_resamples": grading.n_resamples,
                    "hat_ic": _none_if_nan(row.hat_ic),
                    "sigma2": _none_if_nan(row.sigma2),
                    "mu_hat": _none_if_nan(grading.mu_hat),
                    "mu_var": _none_if_nan(grading.mu_var),
                    "tau2_hat": _none_if_nan(grading.tau2),
                    "shrinkage_weight": _none_if_nan(row.shrinkage_weight),
                    "theta_posterior_mean": _none_if_nan(row.theta),
                    "theta_posterior_var": _none_if_nan(row.post_var),
                    "theta_posterior_var_naive": _none_if_nan(row.post_var_naive),
                    "p_positive_skill": _none_if_nan(row.p_positive),
                    "p_two_sided": _none_if_nan(row.p_two_sided),
                    "bh_significant": int(row.bh_significant),
                    "fdr_q": grading.fdr_q,
                    "break_even_ic": _none_if_nan(grading.break_even_ic),
                    "grade_v2": row.grade_v2,
                    "grade_v2_reason": row.reason,
                    "old_grade": row.old_grade,
                }))
            written += 1
        conn.commit()
    return written


def _none_if_nan(value) -> float | None:
    """
    NaN out, NULL in — the `tracking.json_safe` rule at a different boundary.

    An undefined statistic is a null, not a number. Writing a bare NaN into a
    REAL column gives Postgres a value that compares false against everything
    including itself, which is a worse lie than saying it was not measured.
    """
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else value
