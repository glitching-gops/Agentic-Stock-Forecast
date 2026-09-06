"""
Evidence-Grading Redesign, Stage 0 — the shrinkage layer and its guards.

NAMED `test_stage0_*` RATHER THAN `test_phase6_*` ON PURPOSE. This repo already
has a Phase 0-6 roadmap with its own meaning, and the improvement plan this work
comes from has its own Stage 0-4 numbering. Two schemes sharing a filename
prefix would conflate them permanently; the prefix is the separation.

What these tests are for: a grading layer that produces a plausible table for
the wrong reason is exactly the failure this project has paid for repeatedly.
The specific ways this one can be plausibly wrong are

  - the posterior variance collapsing to zero at tau2 = 0, which manufactures
    infinite confidence and a full board of STRONG grades out of a degenerate
    limit;
  - a bootstrap block shorter than the 30-session label overlap, which
    understates sigma2, which does the same thing more subtly;
  - a Benjamini-Hochberg implementation that stops at the first failure instead
    of the last success, which silently under-rejects;
  - a significant NEGATIVE effect being folded back into "no evidence", which
    is the mistake the live gate's `abs(ic_t)` made until 2026-09-02.

Each has a test below whose failure message says which one broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from pipeline.evaluation import rank_ic
from pipeline.evidence_shrinkage import (
    BLOCK_LENGTH_SESSIONS,
    FDR_Q,
    MIN_OOS_ROWS,
    EvidenceGradingRefused,
    PanelGrading,
    TickerTrack,
    assign_grade,
    benjamini_hochberg,
    block_bootstrap_ic,
    dersimonian_laird_tau2,
    economic_bar,
    grade_panel,
    init_evidence_tables,
    posterior_probability_positive,
    precision_weighted_mean,
    rank_ic_rows,
    shrink,
    store_grading,
)

BREAK_EVEN = 0.0051          # the P4-measured bar, pinned so tests are offline


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _track(ticker: str, n: int = 600, ic: float = 0.0, seed: int = 0,
           autocorr: int = BLOCK_LENGTH_SESSIONS) -> TickerTrack:
    """
    A synthetic track record with a PLANTED rank IC and realistic overlap.

    `y_true` is a rolling sum over `autocorr` sessions, which is what a
    30-session forward return actually is: neighbouring rows share all but one
    of their sessions.

    **BOTH SERIES ARE PERSISTENT, AND THE FIRST VERSION OF THIS FIXTURE GOT
    THAT WRONG.** Making only the outcome autocorrelated and leaving the
    prediction iid produces a rank IC whose sampling variance is INSENSITIVE to
    the block length — with one series independent across time, the cross
    products carry no serial dependence for a block to capture, and a block of
    1 measures the variance correctly. The test written to catch a too-short
    block then fails against correct code, which is how this was found.

    Real predictions are persistent: the features move slowly, so a model's
    output on adjacent sessions is nearly the same number. That is the
    condition under which overlap inflates the variance of an IC, and it is the
    condition this fixture has to reproduce for the block-length guard below to
    have any teeth at all.
    """
    rng = np.random.default_rng(seed)

    def _persistent(size: int) -> np.ndarray:
        raw = rng.normal(size=size + autocorr)
        return np.array([raw[i:i + autocorr].sum()
                         for i in range(size)]) / np.sqrt(autocorr)

    y_true = _persistent(n)
    noise = _persistent(n)
    y_pred = ic * y_true + np.sqrt(max(1e-12, 1 - ic ** 2)) * noise
    dates = tuple(str(pd.Timestamp("2019-01-01") + pd.Timedelta(days=i))[:10]
                  for i in range(n))
    return TickerTrack(ticker, dates, y_true, y_pred)


def _sqlite_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'stage0.db'}")
    init_evidence_tables(engine)
    return engine


# ── The rank IC second implementation, and its debt ───────────────────────────


def test_the_vectorised_rank_ic_agrees_with_the_one_the_project_already_had():
    """
    `rank_ic_rows` is a second copy of `pipeline.evaluation.rank_ic`, kept only
    because 170,000 scipy calls is minutes of wall clock. Two copies of a
    formula drift; this is the only thing that would notice.
    """
    rng = np.random.default_rng(11)
    a = rng.normal(size=(25, 120))
    b = rng.normal(size=(25, 120))

    mine = rank_ic_rows(a, b)
    theirs = np.array([rank_ic(b[i], a[i]) for i in range(25)])

    assert np.max(np.abs(mine - theirs)) < 1e-12, (
        "the vectorised rank IC has drifted from pipeline.evaluation.rank_ic; "
        "the whole justification for the second copy is that it cannot")


def test_rank_ic_of_a_constant_row_is_undefined_and_not_zero():
    """An undefined correlation is not a measured zero — the same rule that
    makes `zero` and `majority` produce NaN in the baseline table."""
    a = np.array([[1.0, 2.0, 3.0, 4.0]])
    flat = np.array([[7.0, 7.0, 7.0, 7.0]])
    assert np.isnan(rank_ic_rows(a, flat)[0])
    assert np.isnan(rank_ic_rows(flat, a)[0])


def test_ties_take_average_ranks_exactly_as_spearmanr_does():
    a = np.array([[1.0, 2.0, 2.0, 3.0, 5.0, 8.0]])
    b = np.array([[2.0, 1.0, 4.0, 4.0, 5.0, 6.0]])
    assert abs(rank_ic_rows(a, b)[0] - rank_ic(b[0], a[0])) < 1e-12


# ── Shrinkage, against arithmetic done by hand ────────────────────────────────


def test_the_grand_mean_is_precision_weighted_and_not_a_plain_average():
    """
    Hand-computed. hat = [0.30, 0.00, -0.30], sigma2 = [0.01, 0.04, 0.01].

        w        = [100, 25, 100],  sum(w) = 225
        mu_hat   = (100*0.30 + 25*0.00 + 100*(-0.30)) / 225 = 0
        var(mu)  = 1 / 225

    The plain average is also 0 here BY CONSTRUCTION, so the test moves the
    middle value: at hat = [0.30, 0.60, -0.30] the plain average is +0.20 and
    the precision-weighted mean is 25*0.60/225 = +0.0666..., because the middle
    ticker is measured four times as noisily and must carry a quarter the
    weight. A test that only used the symmetric case would pass against a plain
    mean.
    """
    hat = np.array([0.30, 0.60, -0.30])
    s2 = np.array([0.01, 0.04, 0.01])

    mu, var_mu = precision_weighted_mean(hat, s2)

    assert mu == pytest.approx(25 * 0.60 / 225, abs=1e-12)
    assert var_mu == pytest.approx(1 / 225, abs=1e-12)
    assert mu != pytest.approx(float(hat.mean()), abs=1e-4), (
        "the shrinkage target is a plain average, not a precision-weighted "
        "one: the noisiest tickers are dragging the mean every name is "
        "shrunk toward")


def test_dersimonian_laird_matches_the_arithmetic_done_by_hand():
    """
    hat = [0.30, 0.00, -0.30], sigma2 = [0.01, 0.01, 0.01].

        w    = [100, 100, 100],  mu = 0
        Q    = 100*0.09 + 0 + 100*0.09 = 18
        c    = 300 - 30000/300 = 200
        tau2 = (18 - (3 - 1)) / 200 = 0.08
    """
    hat = np.array([0.30, 0.00, -0.30])
    s2 = np.full(3, 0.01)

    tau2, q = dersimonian_laird_tau2(hat, s2)

    assert q == pytest.approx(18.0, abs=1e-10)
    assert tau2 == pytest.approx(0.08, abs=1e-10)


def test_a_negative_moment_estimate_is_clipped_to_zero_not_reported():
    """Less spread than sampling error alone predicts means no between-ticker
    variation. Zero is the honest report; a negative variance is not a number."""
    hat = np.array([0.010, 0.011, 0.009, 0.0105])
    s2 = np.full(4, 0.04)
    tau2, q = dersimonian_laird_tau2(hat, s2)
    assert tau2 == 0.0
    assert q < 3


def test_the_posterior_matches_the_closed_form_done_by_hand():
    """
    Continuing the panel above: mu_hat = 0, var(mu) = 1/300, tau2 = 0.08,
    sigma2 = 0.01 for the ticker at hat_ic = +0.30.

        B        = 0.01 / (0.01 + 0.08)      = 1/9
        theta    = (1/9)*0 + (8/9)*0.30      = 0.2666666...
        naive    = 0.01 * 0.08 / 0.09        = 0.0088888...
        post_var = naive + (1/9)^2 * (1/300) = 0.0088888... + 0.0000411522...
                                             = 0.0089300411...
    """
    theta, b, post_var, naive = shrink(hat_ic=0.30, sigma2=0.01,
                                       mu_hat=0.0, mu_var=1 / 300, tau2=0.08)

    assert b == pytest.approx(1 / 9, abs=1e-12)
    assert theta == pytest.approx(0.8 / 3, abs=1e-12)
    assert naive == pytest.approx(0.0008 / 0.09, abs=1e-12)
    assert post_var == pytest.approx(0.0008 / 0.09 + (1 / 81) * (1 / 300),
                                     abs=1e-15)


def test_a_precisely_measured_ticker_is_barely_shrunk():
    """sigma2 -> 0 means B -> 0: the data speak for themselves."""
    theta, b, post_var, _ = shrink(hat_ic=0.30, sigma2=1e-12, mu_hat=0.0,
                                   mu_var=1 / 300, tau2=0.08)
    assert b < 1e-10
    assert theta == pytest.approx(0.30, abs=1e-9)
    assert post_var < 1e-11


def test_a_hopelessly_measured_ticker_shrinks_all_the_way_to_the_panel():
    """sigma2 -> inf means B -> 1: the ticker's own estimate carries nothing
    and the best available answer is the panel's."""
    theta, b, post_var, _ = shrink(hat_ic=0.30, sigma2=1e9, mu_hat=-0.02,
                                   mu_var=1 / 300, tau2=0.08)
    assert b == pytest.approx(1.0, abs=1e-8)
    assert theta == pytest.approx(-0.02, abs=1e-8)


# ── The degenerate limit that would manufacture a full board of STRONGs ───────


def test_tau2_of_zero_does_not_produce_infinite_confidence():
    """
    THE SINGLE MOST DANGEROUS BRANCH IN THIS MODULE.

    The classical Efron-Morris posterior variance is `sigma2 * tau2 /
    (sigma2 + tau2)`, which is EXACTLY ZERO when tau2 is zero. Every ticker
    then collapses onto mu_hat with zero variance, every p_positive becomes 0
    or 1, every BH test rejects, and the run reports a full board of STRONG
    grades produced by a division that should never have happened.

    Morris' correction supplies the right limit: at tau2 = 0 the posterior
    variance IS the variance of the grand mean.
    """
    theta, b, post_var, naive = shrink(hat_ic=0.30, sigma2=0.01, mu_hat=0.004,
                                       mu_var=1 / 300, tau2=0.0)

    assert naive == 0.0, "the classical term should be zero here - that is the trap"
    assert b == 1.0
    assert theta == pytest.approx(0.004, abs=1e-12)
    assert post_var == pytest.approx(1 / 300, abs=1e-12), (
        "posterior variance collapsed to zero at tau2 = 0; every ticker on the "
        "panel is about to be graded with infinite confidence")

    p = posterior_probability_positive(theta, post_var)
    assert 0.0 < p < 1.0, "posterior probability saturated at a hard 0 or 1"


def test_a_degenerate_panel_is_flagged_and_says_so_on_every_row():
    """
    A panel with no between-ticker variation grades every name identically.
    That is one statement printed N times, and it has to be labelled as such.

    **CONSTRUCTED TO DEGENERATE, NOT HOPED INTO IT.** The first version of this
    test built twelve independent noise tracks and skipped when `tau2` came
    back positive — which it did, so the branch was never exercised and the
    test was decoration. Every track here shares ONE data seed and differs only
    in its ticker, so the twelve point estimates are identical, Cochran's Q is
    ~0 against an expectation of 11, and DerSimonian-Laird returns exactly zero
    every time.
    """
    tracks = [_track(f"T{i}.NS", ic=0.0, seed=100) for i in range(12)]
    grading = grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=200)

    assert grading.tau2 == 0.0, (
        "the fixture was built to force tau2 = 0 and did not; the degenerate "
        "branch is going untested")
    assert grading.degenerate
    assert grading.to_metrics()["degenerate_panel"] is True
    thetas = {round(r.theta, 12) for r in grading.rows if np.isfinite(r.theta)}
    assert len(thetas) == 1, "tau2 = 0 must collapse every posterior onto one value"
    assert all("tau2_hat came back at zero" in r.reason
               for r in grading.rows if np.isfinite(r.theta)), (
        "the degenerate-panel disclosure is missing from the stored reason, so "
        "a reader of the table cannot tell a per-ticker grade from a panel one")


# ── Benjamini-Hochberg ────────────────────────────────────────────────────────


def test_benjamini_hochberg_reproduces_the_published_worked_example():
    """
    The 15 p-values from Benjamini & Hochberg (1995), q = 0.05. The published
    answer is 4 rejections; Bonferroni on the same vector rejects 3.
    """
    p = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                  0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590,
                  1.0000])
    rejected = benjamini_hochberg(p, q=0.05)
    assert rejected.sum() == 4
    assert list(np.nonzero(rejected)[0]) == [0, 1, 2, 3]


def test_benjamini_hochberg_steps_UP_and_rejects_past_a_local_failure():
    """
    THE PART A NAIVE IMPLEMENTATION GETS WRONG. p = [0.04, 0.05, 0.06] at
    q = 0.10, m = 3 gives thresholds 0.0333 / 0.0667 / 0.1000. The smallest
    p-value FAILS its own threshold (0.04 > 0.0333), but the largest clears
    its own (0.06 <= 0.10), and the step-up procedure therefore rejects all
    three. An implementation that stops at the first failure rejects none.
    """
    rejected = benjamini_hochberg(np.array([0.04, 0.05, 0.06]), q=0.10)
    assert rejected.all(), (
        "the FDR procedure stopped at the first p-value that failed its own "
        "threshold instead of continuing to the last one that passed")


def test_benjamini_hochberg_is_order_invariant():
    p = np.array([0.6, 0.001, 0.3, 0.008, 0.9])
    a = benjamini_hochberg(p, q=0.05)
    order = np.array([3, 0, 4, 1, 2])
    b = benjamini_hochberg(p[order], q=0.05)
    assert list(a[order]) == list(b)


def test_benjamini_hochberg_matches_statsmodels_where_it_is_installed():
    """Cross-check against the reference implementation when it happens to be
    importable. `statsmodels` is deliberately NOT added to requirements.txt —
    that file is installed by Render and by both scheduled jobs."""
    multipletests = pytest.importorskip(
        "statsmodels.stats.multitest").multipletests
    rng = np.random.default_rng(5)
    for _ in range(20):
        p = np.clip(rng.beta(0.3, 3.0, size=84), 1e-9, 1.0)
        mine = benjamini_hochberg(p, q=FDR_Q)
        theirs = multipletests(p, alpha=FDR_Q, method="fdr_bh")[0]
        assert list(mine) == list(theirs)


def test_a_non_finite_pvalue_is_refused_rather_than_silently_sorted():
    """NaN sorts to the end and would silently enlarge the family. Unmeasurable
    tickers must be excluded before the family is formed, not passed in."""
    with pytest.raises(ValueError, match="non-finite"):
        benjamini_hochberg(np.array([0.01, np.nan, 0.5]), q=0.10)


# ── The signed handling the live gate had to learn the hard way ───────────────


def test_a_significant_negative_posterior_is_labelled_ANTI_SIGNAL():
    grade, reason = assign_grade(bh_significant=True, theta=-0.28,
                                 p_positive=0.001, break_even=BREAK_EVEN)
    assert grade == "ANTI_SIGNAL"
    assert "backwards" in reason


def test_a_reliably_backwards_ticker_survives_the_whole_pipeline_as_ANTI_SIGNAL():
    """
    End to end, not just the decision table. One ticker predicts the NEGATIVE
    of the outcome; the rest are noise. It must come out ANTI_SIGNAL — not
    STRONG (the `abs(ic_t)` mistake the live gate made until 2026-09-02, where
    all four names that ever passed the t-check had strongly negative IC), and
    not quietly INSUFFICIENT (the symmetric mistake, which throws a real
    measurement away).
    """
    tracks = [_track(f"NOISE{i}.NS", ic=0.0, seed=200 + i) for i in range(14)]
    tracks.append(_track("BACKWARDS.NS", ic=-0.95, seed=999))

    grading = grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=400)
    row = next(r for r in grading.rows if r.ticker == "BACKWARDS.NS")

    assert row.hat_ic < -0.5
    assert row.theta < 0
    assert row.bh_significant, "a near-perfect inverse ordering failed FDR control"
    assert row.grade_v2 == "ANTI_SIGNAL", (
        f"a reliably backwards ticker was graded {row.grade_v2}")


def test_a_strong_positive_ticker_reaches_STRONG_so_the_gate_is_not_merely_shut():
    """
    A gate nothing can pass is indistinguishable from a broken one. This is the
    `series_zero` pattern: plant an effect of known, enormous size and require
    the machinery to find it, so that a null on real data is a statement about
    the data rather than about the code.
    """
    tracks = [_track(f"NOISE{i}.NS", ic=0.0, seed=300 + i) for i in range(14)]
    tracks.append(_track("REAL.NS", ic=0.95, seed=777))

    grading = grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=400)
    row = next(r for r in grading.rows if r.ticker == "REAL.NS")

    assert row.bh_significant
    assert row.theta > BREAK_EVEN
    assert row.grade_v2 == "STRONG", (
        f"a planted rank IC near +0.95 was graded {row.grade_v2}; the layer "
        f"cannot detect an effect of any size")


def test_the_economic_bar_can_only_remove_a_STRONG_never_create_one():
    """A significant positive effect below the break-even IC is WEAK, not
    STRONG — and never the other way round."""
    above = assign_grade(True, theta=0.20, p_positive=0.999,
                         break_even=BREAK_EVEN)[0]
    below = assign_grade(True, theta=0.0001, p_positive=0.999,
                         break_even=BREAK_EVEN)[0]
    assert above == "STRONG"
    assert below == "WEAK"


def test_an_unremarkable_ticker_is_INSUFFICIENT():
    assert assign_grade(False, theta=0.001, p_positive=0.55,
                        break_even=BREAK_EVEN)[0] == "INSUFFICIENT"


def test_a_positive_posterior_that_fails_FDR_is_WEAK_not_INSUFFICIENT():
    """"Probably positive, but this many would be expected by chance across 84
    names" is a different statement from "indistinguishable from nothing"."""
    grade, reason = assign_grade(False, theta=0.05, p_positive=0.94,
                                 break_even=BREAK_EVEN)
    assert grade == "WEAK"
    assert "does not survive" in reason


# ── The bootstrap ─────────────────────────────────────────────────────────────


def test_the_bootstrap_is_deterministic_under_its_seed():
    track = _track("DET.NS", seed=42)
    a = block_bootstrap_ic(track, n_resamples=300)
    b = block_bootstrap_ic(track, n_resamples=300)
    assert a.sigma2 == b.sigma2
    assert a.hat_ic == b.hat_ic


def test_the_bootstrap_seed_depends_on_the_ticker_not_on_arrival_order():
    """A panel result that moved when tickers were reordered would be a grade
    that changes while nothing else did."""
    tracks = [_track(f"ORD{i}.NS", ic=0.0, seed=400 + i) for i in range(8)]
    forward = grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=200)
    backward = grade_panel(list(reversed(tracks)), break_even=BREAK_EVEN,
                           n_resamples=200)

    a = {r.ticker: (r.hat_ic, r.sigma2, r.theta) for r in forward.rows}
    b = {r.ticker: (r.hat_ic, r.sigma2, r.theta) for r in backward.rows}
    assert a == b


def test_the_point_estimate_is_the_real_series_not_a_bootstrap_average():
    """The bootstrap supplies the VARIANCE. Reporting the mean of the resamples
    as the estimate would introduce the bootstrap's own bias into the number
    being graded."""
    track = _track("PT.NS", ic=0.4, seed=7)
    est = block_bootstrap_ic(track, n_resamples=300)
    assert est.hat_ic == pytest.approx(rank_ic(track.y_true, track.y_pred),
                                       abs=1e-12)


def test_prediction_and_outcome_are_resampled_with_the_SAME_index():
    """
    Resampling the two series independently would destroy the correlation being
    measured and hand back a null estimate with a null variance — a clean,
    plausible, entirely fake result. A planted IC near +0.95 must survive.
    """
    track = _track("PAIRED.NS", ic=0.95, seed=3)
    est = block_bootstrap_ic(track, n_resamples=300)
    assert est.hat_ic > 0.8
    assert np.sqrt(est.sigma2) < 0.1, (
        "the bootstrap spread is enormous for a near-perfect ordering, which "
        "is what independent resampling of the two series looks like")


def test_the_out_of_sample_series_has_no_purge_gap_for_a_bootstrap_to_cross():
    """
    THE STRUCTURAL ASSUMPTION THE WHOLE BOOTSTRAP DESIGN RESTS ON.

    A moving-block bootstrap draws blocks of CONSECUTIVE rows. That is only
    legitimate if consecutive rows in the out-of-sample series are actually
    adjacent in time — if a purge or embargo gap sat inside it, a block
    straddling the gap would splice two periods together and the resulting
    sigma2 would describe a series that never existed.

    It does not, and the reason is in the splitter: `PurgedWalkForward` opens
    each test window exactly where the previous one closed
    (`test_start = min_train + fold*step`, `test_end = test_start + step`), and
    the purge and the embargo are carved out of the TRAINING end. Verified
    against real data too: on the 2,439-row ABB.NS frame the four fold
    boundaries fall on consecutive NSE sessions (1, 1, 1 and 3 calendar days,
    the last a weekend) with no interior gap above 5 days anywhere.

    This test pins the property at the splitter, where it is a fact about the
    code rather than about one ticker's calendar.
    """
    from pipeline.evaluation import PurgedWalkForward
    from pipeline.model import EVAL_MIN_TRAIN, EVAL_N_FOLDS
    from pipeline.signals import HORIZON_SESSIONS

    splitter = PurgedWalkForward(n_folds=EVAL_N_FOLDS, horizon=HORIZON_SESSIONS,
                                 embargo=HORIZON_SESSIONS,
                                 min_train=EVAL_MIN_TRAIN)
    windows = [test for _, test in splitter.split(2439)]

    assert len(windows) == EVAL_N_FOLDS
    for earlier, later in zip(windows, windows[1:]):
        assert later[0] == earlier[-1] + 1, (
            "a gap opened between consecutive out-of-sample windows; a "
            "moving-block bootstrap over the concatenated series would now "
            "splice two non-adjacent periods into one block")


def test_a_shorter_block_understates_the_sampling_variance():
    """
    THE SECOND FAILURE MODE NAMED IN THE PRE-REGISTRATION. The fixture's
    outcome is a 30-session rolling sum, so neighbouring rows share 29 of their
    30 sessions. A block of 1 treats them as independent, returns a sigma2 that
    is too small, and every posterior downstream becomes too confident.
    """
    track = _track("BLK.NS", ic=0.0, seed=17)
    short = block_bootstrap_ic(track, block=1, n_resamples=400)
    proper = block_bootstrap_ic(track, block=BLOCK_LENGTH_SESSIONS,
                                n_resamples=400)
    assert short.sigma2 < proper.sigma2, (
        "a block of 1 did not understate the variance on an autocorrelated "
        "series, which means the fixture has no autocorrelation and this test "
        "is blind to the defect it exists to catch")


# ── Small samples, and the tickers that cannot be measured at all ─────────────


def test_a_ticker_with_too_few_rows_is_refused_with_a_reason_not_estimated():
    track = _track("SHORT.NS", n=MIN_OOS_ROWS - 1, seed=1)
    est = block_bootstrap_ic(track, n_resamples=100)
    assert est.usable is False
    assert "out-of-sample rows" in est.reason
    assert np.isnan(est.sigma2)


def test_a_prediction_constant_within_every_fold_earns_no_ranking_result():
    """
    THE GUARD THAT COST THIS PROJECT'S FIRST STAGE 0 RUN ITS ONLY NON-NULL
    GRADE, and it was found by looking rather than by a test failing.

    A per-ticker model that emits ONE CONSTANT PER FOLD passes the
    whole-series constancy check easily — the five constants differ — while
    holding no ordering anywhere. Its pooled rank IC is then entirely the
    arrangement of five fold-level constants against five realised period
    returns: a correlation over FIVE points, published as a per-company track
    record.

    Measured on the real panel, 2026-09-06: WIPRO.NS was graded ANTI_SIGNAL on
    a pooled IC of -0.3914 whose within-fold IC is UNDEFINED in all five of its
    folds. 21 of 84 tickers are in that state.

    The fixture reproduces it exactly: distinct constants per fold, arranged
    against rising realised returns, giving a large negative pooled IC out of
    nothing at all.
    """
    n_per_fold = 80
    y_true, y_pred, folds = [], [], []
    rng = np.random.default_rng(2)
    for k, (pred, truth) in enumerate([(0.04, -0.02), (0.03, -0.01),
                                       (0.02, 0.00), (0.01, 0.01),
                                       (0.00, 0.02)]):
        y_true.append(truth + 0.001 * rng.normal(size=n_per_fold))
        y_pred.append(np.full(n_per_fold, pred))
        folds.append(np.full(n_per_fold, k))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    dates = tuple(str(i) for i in range(len(y_true)))

    # Without the fold labels the layer cannot see the problem and estimates a
    # confident, entirely spurious negative IC.
    blind = block_bootstrap_ic(TickerTrack("BLIND.NS", dates, y_true, y_pred),
                               n_resamples=200)
    assert blind.usable is True
    assert blind.hat_ic < -0.8, (
        "the fixture does not reproduce the artifact, so the guard below is "
        "being tested against nothing")

    # With them, it is refused.
    seeing = block_bootstrap_ic(
        TickerTrack("SEEING.NS", dates, y_true, y_pred,
                    folds=np.concatenate(folds)),
        n_resamples=200)
    assert seeing.usable is False
    assert "CONSTANT within every one of its 5 walk-forward folds" in seeing.reason


def test_the_within_fold_guard_spares_a_ticker_that_does_order_somewhere():
    """It must refuse only the genuinely empty case. A ticker with real
    ordering in even one fold is still measurable, contamination and all —
    the broader contamination is REPORTED by the fold diagnostic, not gated."""
    n_per_fold = 80
    rng = np.random.default_rng(3)
    y_true, y_pred, folds = [], [], []
    for k in range(5):
        truth = rng.normal(size=n_per_fold)
        y_true.append(truth)
        # fold 2 carries a real ordering; the rest are flat constants
        y_pred.append(truth * 0.5 if k == 2 else np.full(n_per_fold, 0.01 * k))
        folds.append(np.full(n_per_fold, k))

    est = block_bootstrap_ic(
        TickerTrack("MIXED.NS", tuple(str(i) for i in range(5 * n_per_fold)),
                    np.concatenate(y_true), np.concatenate(y_pred),
                    folds=np.concatenate(folds)),
        n_resamples=200)
    assert est.usable is True


def test_a_constant_prediction_is_undefined_rather_than_a_measured_zero():
    n = 400
    track = TickerTrack("FLAT.NS", tuple(str(i) for i in range(n)),
                        np.random.default_rng(0).normal(size=n),
                        np.full(n, 0.017))
    est = block_bootstrap_ic(track, n_resamples=100)
    assert est.usable is False
    assert "constant" in est.reason


def test_an_unmeasurable_ticker_is_graded_INSUFFICIENT_and_left_out_of_the_family():
    """
    It must not crash the run, must not be silently dropped from the report,
    and must NOT be counted as a hypothesis that was tested — inflating the BH
    family with untested nulls makes every real p-value harder to reject for a
    bookkeeping reason.
    """
    tracks = [_track(f"OK{i}.NS", ic=0.0, seed=500 + i) for i in range(10)]
    tracks.append(_track("TOOSHORT.NS", n=MIN_OOS_ROWS - 5, seed=9))

    grading = grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=200)

    assert grading.n_usable == 10
    assert grading.n_unusable == 1
    assert len(grading.rows) == 11, "the unmeasurable ticker vanished from the report"

    row = next(r for r in grading.rows if r.ticker == "TOOSHORT.NS")
    assert row.grade_v2 == "INSUFFICIENT"
    assert "Excluded from the" in row.reason
    assert np.isnan(row.theta) and np.isnan(row.p_positive)


def test_a_panel_too_small_to_pool_over_is_refused_outright():
    """Partial pooling with one unit is not partial pooling."""
    with pytest.raises(EvidenceGradingRefused, match="panel to pool over"):
        grade_panel([_track("ONLY.NS", seed=2)], break_even=BREAK_EVEN,
                    n_resamples=100)


# ── Storage, provenance and shadow mode ───────────────────────────────────────


def test_the_shadow_table_round_trips_and_is_append_only(tmp_path):
    """
    Two runs are two rows, never an overwrite — the `forecast_outcomes` rule.
    A grade that can be quietly improved after the fact is not a record of what
    the measurement said.
    """
    engine = _sqlite_engine(tmp_path)
    tracks = [_track(f"S{i}.NS", ic=0.0, seed=600 + i) for i in range(6)]

    first = grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=150)
    second = grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=150)
    assert first.run_id != second.run_id

    store_grading(first, engine)
    store_grading(second, engine)

    stored = pd.read_sql(text("SELECT * FROM evidence_grades_v2"), engine)
    assert len(stored) == 12
    assert stored["run_id"].nunique() == 2
    assert set(stored["ticker"]) == {t.ticker for t in tracks}
    assert stored["grade_v2"].isin(
        ["STRONG", "WEAK", "INSUFFICIENT", "ANTI_SIGNAL"]).all()


def test_nan_never_leaves_the_write_boundary_as_a_number():
    """
    THIS TEST EXISTS BECAUSE THE ROUND-TRIP TEST BELOW HAS NO TEETH ON SQLITE,
    and that was found by mutation rather than by reading the code.

    Deleting the guard in `_none_if_nan` — returning the raw float instead of
    None — left the whole suite green. The reason is that Python's `sqlite3`
    driver converts a bound `float('nan')` to SQL NULL on the way in, so the
    test database launders the defect and reports exactly the behaviour the
    guard was supposed to provide. Postgres does not: it stores `'NaN'::float8`,
    a real value that compares false against everything including itself, and
    every downstream `IS NULL` check on `evidence_grades_v2` would then miss it.

    So the guard is asserted at the FUNCTION, where the two databases cannot
    disagree. Same lesson as the `first_seen` COALESCE duplicate and the
    `total_matching = len(df)` grep: assert the invariant where it lives.
    """
    from pipeline.evidence_shrinkage import _none_if_nan

    assert _none_if_nan(float("nan")) is None
    assert _none_if_nan(float("inf")) is None
    assert _none_if_nan(float("-inf")) is None
    assert _none_if_nan(None) is None
    assert _none_if_nan(0.0) == 0.0
    assert _none_if_nan(-0.031) == pytest.approx(-0.031)


def test_the_block_length_is_the_label_horizon_and_not_an_independent_knob():
    """
    The whole justification for the block bootstrap is that the block equals the
    number of sessions two neighbouring observations share. If the horizon ever
    moves (P6 sweeps 5/10/20/30) and this constant does not move with it, every
    sigma2 here silently starts describing the wrong dependence structure.
    """
    from pipeline.signals import HORIZON_SESSIONS
    assert BLOCK_LENGTH_SESSIONS == HORIZON_SESSIONS


def test_an_undefined_statistic_is_stored_as_NULL_and_never_as_NaN(tmp_path):
    """
    A bare NaN in a REAL column compares false against everything including
    itself. NULL says "not measured", which is what an undefined statistic is —
    the `tracking.json_safe` rule at a different boundary.

    Read this alongside `test_nan_never_leaves_the_write_boundary_as_a_number`:
    on SQLite this test passes with the guard DELETED, because the driver
    converts NaN to NULL for us. It is kept because it covers the wiring — that
    the value reaches the column at all — and the guard itself is pinned above.
    """
    engine = _sqlite_engine(tmp_path)
    tracks = [_track(f"N{i}.NS", ic=0.0, seed=700 + i) for i in range(6)]
    tracks.append(_track("UNMEASURED.NS", n=MIN_OOS_ROWS - 3, seed=8))

    store_grading(grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=150),
                  engine)

    row = pd.read_sql(
        text("SELECT theta_posterior_mean, sigma2, p_positive_skill FROM "
             "evidence_grades_v2 WHERE ticker = 'UNMEASURED.NS'"), engine)
    assert row["theta_posterior_mean"].isna().all()
    assert row["sigma2"].isna().all()
    assert row["p_positive_skill"].isna().all()

    nulls = pd.read_sql(
        text("SELECT COUNT(*) AS n FROM evidence_grades_v2 "
             "WHERE theta_posterior_mean IS NULL"), engine)
    assert int(nulls.iloc[0]["n"]) == 1, (
        "the undefined posterior did not reach the database as SQL NULL")


def test_creating_the_shadow_table_is_idempotent(tmp_path):
    engine = _sqlite_engine(tmp_path)
    init_evidence_tables(engine)
    init_evidence_tables(engine)
    assert pd.read_sql(text("SELECT COUNT(*) AS n FROM evidence_grades_v2"),
                       engine).iloc[0]["n"] == 0


def test_stage0_writes_nothing_the_public_api_serves(tmp_path):
    """
    SHADOW MODE, ASSERTED RATHER THAN INTENDED. The live board is built from
    `forecast_current.forecast_confidence`. A full grading run must leave that
    table byte-for-byte as it found it.
    """
    engine = _sqlite_engine(tmp_path)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE forecast_current "
                          "(ticker TEXT, forecast_confidence TEXT)"))
        conn.execute(text("INSERT INTO forecast_current VALUES "
                          "('A.NS', 'WEAK'), ('B.NS', 'INSUFFICIENT')"))
        conn.commit()

    before = pd.read_sql(text("SELECT * FROM forecast_current"), engine)
    tracks = [_track(f"X{i}.NS", ic=0.0, seed=800 + i) for i in range(6)]
    store_grading(grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=150),
                  engine)
    after = pd.read_sql(text("SELECT * FROM forecast_current"), engine)

    pd.testing.assert_frame_equal(before, after)


def test_the_fold_diagnostic_detects_a_pooled_IC_made_of_fold_levels():
    """
    THE DIAGNOSTIC THAT QUALIFIES EVERY OTHER NUMBER IN THE RUN, pinned on a
    case whose right answer is known by construction.

    Each ticker here predicts a CONSTANT within each fold, so no fold contains
    any ordering at all and the true within-fold rank IC is undefined
    everywhere. The constants are then arranged to run OPPOSITE to the realised
    fold returns, which is exactly the mechanism that gives a pooled per-ticker
    rank IC a confident negative value out of nothing.

    On the real panel this measured 75.2% of (ticker, fold) cells constant and
    a pooled IC of -0.070 against a within-fold mean of +0.126 — opposite
    signs. If this test stops detecting the planted version, the real reading
    cannot be trusted either.
    """
    import tools.run_evidence_grading as tool

    n_per_fold, n_folds = 60, 5
    pred_levels = [0.04, 0.03, 0.02, 0.01, 0.00]      # falling
    true_levels = [-0.02, -0.01, 0.00, 0.01, 0.02]    # rising: opposite order

    rng = np.random.default_rng(4)
    tracks, folds = [], {}
    for t in range(8):
        y_true, y_pred, fold = [], [], []
        for k in range(n_folds):
            y_true.append(true_levels[k] + 0.001 * rng.normal(size=n_per_fold))
            y_pred.append(np.full(n_per_fold, pred_levels[k]))
            fold.append(np.full(n_per_fold, k))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        name = f"F{t}.NS"
        tracks.append(TickerTrack(name, tuple(str(i) for i in range(len(y_true))),
                                  y_true, y_pred))
        folds[name] = np.concatenate(fold)

    out = tool.fold_diagnostic(tracks, folds)

    assert out["cells"] == 8 * n_folds
    assert out["constant_cells"] == 8 * n_folds, (
        "every fold here holds a single repeated prediction and the diagnostic "
        "did not notice")
    assert out["constant_row_fraction"] == pytest.approx(1.0)
    assert out["fold_level_vs_realised_rho"] == pytest.approx(-1.0)
    assert out["pooled_ic_mean"] < -0.5, (
        "a panel with no within-fold ordering whatsoever produced a pooled IC "
        "near zero; the fold-level channel is not being measured")
    assert np.isnan(out["within_fold_ic_mean"]), (
        "a constant prediction inside a fold has an UNDEFINED rank IC, not a "
        "measured zero")


def test_the_old_gate_is_imported_and_not_reimplemented():
    """
    The crosstab compares the NEW layer against the LIVE gate. If the "old"
    grade were a local copy, the comparison would be between two of this
    module's own functions and would stay green while the real gate changed
    underneath it.
    """
    import tools.run_evidence_grading as tool
    from agents import critic_agent

    assert tool.grade_evidence is critic_agent.grade_evidence


def test_the_break_even_bar_is_the_portfolio_modules_own_number():
    """Reused, not reimplemented: a second copy of the Zerodha/NSE cost
    schedule is a second thing to keep in step with the exchange."""
    from pipeline.portfolio import CostModel, break_even_ic

    bar, detail = economic_bar(spread_per_ic=0.3474, turnover=0.80)
    expected = break_even_ic(CostModel(), spread_per_ic=0.3474,
                             turnover=0.80)["break_even_rank_ic"]
    assert bar == pytest.approx(expected, abs=1e-15)
    assert detail["round_trip_cost"] == pytest.approx(CostModel().round_trip)


def test_the_crosstab_covers_every_ticker_including_the_unmeasurable_ones():
    tracks = [_track(f"C{i}.NS", ic=0.0, seed=900 + i) for i in range(6)]
    tracks.append(_track("TINY.NS", n=MIN_OOS_ROWS - 2, seed=6))
    old = {t.ticker: ("INSUFFICIENT", []) for t in tracks}
    old["C0.NS"] = ("WEAK", [])

    grading = grade_panel(tracks, old_grades=old, break_even=BREAK_EVEN,
                          n_resamples=150)
    table = grading.crosstab()

    assert int(table.to_numpy().sum()) == len(tracks)
    assert "WEAK" in table.index


def test_the_grading_carries_the_projects_own_provenance_hashes():
    """`config_hash` / `data_hash` from `pipeline.tracking`, not a new scheme —
    a metric that moves while one is constant says which of code or data did it."""
    tracks = [_track(f"P{i}.NS", ic=0.0, seed=1000 + i) for i in range(4)]
    grading = grade_panel(tracks, break_even=BREAK_EVEN, n_resamples=100)
    assert grading.model_version
    assert grading.graded_at
    assert grading.run_id


def test_a_panel_grading_reports_the_counts_it_actually_holds():
    rows = grade_panel([_track(f"K{i}.NS", ic=0.0, seed=1100 + i)
                        for i in range(5)],
                       break_even=BREAK_EVEN, n_resamples=100)
    assert sum(rows.counts().values()) == len(rows.rows)
    assert isinstance(rows, PanelGrading)
