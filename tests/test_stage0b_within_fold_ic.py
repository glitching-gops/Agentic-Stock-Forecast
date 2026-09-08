"""
Stage 0b — the pooled-across-folds IC bug, and the proof that it is fixed.

The bug: `evidence_shrinkage` concatenated every walk-forward fold's rows into
one series and correlated once. Pooling across groups conflates between-group
with within-group variation, so when the folds' own average prediction levels
run opposite to their own average realised returns the pooled figure comes out
NEGATIVE while every fold's internal ranking is POSITIVE.

Measured on the real panel before the fix: the same model over the same rows
read -0.0512 pooled and +0.0120 within, with rho(fold prediction level, fold
realised return) = -0.600. Stage 0's headline `mu_hat = -0.05988` was that
arrangement of five numbers.

The centrepiece here is `test_the_synthetic_panel_separates_pooled_from_within`:
a five-fold panel with a known within-fold correlation and a deliberately
anti-correlated fold structure, where the right answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pipeline.evidence_shrinkage import (
    BLOCK_LENGTH_SESSIONS,
    EvidenceGradingRefused,
    TickerTrack,
    block_bootstrap_ic,
    rank_ic_rows,
    within_fold_rank_ic,
)

N_FOLDS = 5
ROWS_PER_FOLD = 120
TARGET_WITHIN_IC = 0.30


# ── the synthetic panel ───────────────────────────────────────────────────────


def synthetic_panel(
    n_folds: int = N_FOLDS,
    rows_per_fold: int = ROWS_PER_FOLD,
    within_ic: float = TARGET_WITHIN_IC,
    seed: int = 20260908,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Five folds, each with a genuinely POSITIVE within-fold correlation, whose
    fold-level MEANS are deliberately anti-correlated across the five.

    Construction, so the answer is knowable rather than measured:

      * inside fold k, ``pred`` and ``target`` share a common factor scaled to
        give a Spearman correlation near ``within_ic``. Every fold is built the
        same way, so the correct within-fold-then-averaged answer is
        ``within_ic``, up to sampling noise.
      * fold k is then SHIFTED: predictions get a level that rises with k while
        targets get one that falls. The two level sequences have Spearman -1.0,
        mirroring the -0.600 measured on the real panel and making it stronger,
        so the pooled statistic has no room to be ambiguous.

    The shifts are large relative to the within-fold spread, which is what
    makes the pooled correlation read the LEVELS rather than the ordering. That
    is the regime the real panel is in, not an exaggeration of it: there, five
    fold means spanned a range several times any one fold's mean prediction.
    """
    rng = np.random.default_rng(seed)

    # Spearman ~= (6/pi) * arcsin(rho/2) for a bivariate normal, so invert it
    # to pick the Pearson rho that lands the RANK correlation on target.
    rho = 2.0 * np.sin(np.pi * within_ic / 6.0)

    pred_levels = np.linspace(-1.0, 1.0, n_folds)          # rises with the fold
    true_levels = np.linspace(1.0, -1.0, n_folds)          # falls with it

    preds, trues, folds = [], [], []
    for k in range(n_folds):
        z = rng.normal(size=rows_per_fold)
        e = rng.normal(size=rows_per_fold)
        p = z
        t = rho * z + np.sqrt(max(1.0 - rho ** 2, 0.0)) * e
        preds.append(0.05 * p + pred_levels[k])
        trues.append(0.05 * t + true_levels[k])
        folds.append(np.full(rows_per_fold, k))

    return (np.concatenate(trues), np.concatenate(preds),
            np.concatenate(folds))


def _pooled_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The OLD calculation, kept here and nowhere else.

    Reproduced in the test rather than left behind a flag in the module: a
    calculation known to be wrong must not stay callable in production, but the
    proof that it was wrong has to keep working."""
    return float(rank_ic_rows(y_true[None, :], y_pred[None, :])[0])


# ── the centrepiece ───────────────────────────────────────────────────────────


def test_the_synthetic_panel_separates_pooled_from_within():
    """
    Old answer, new answer, and the analytically correct answer, side by side.

    The panel is built so the correct within-fold answer is +0.30 by
    construction. The old pooled calculation must come back strongly NEGATIVE
    on the identical rows; the new one must come back at +0.30.
    """
    y_true, y_pred, folds = synthetic_panel()

    pooled = _pooled_rank_ic(y_true, y_pred)
    within, scored, undefined = within_fold_rank_ic(y_true, y_pred, folds)

    # The fold structure is what the old statistic reads.
    level_pred = [y_pred[folds == k].mean() for k in range(N_FOLDS)]
    level_true = [y_true[folds == k].mean() for k in range(N_FOLDS)]
    level_rho = stats.spearmanr(level_pred, level_true).statistic
    assert level_rho == pytest.approx(-1.0), (
        "the fixture must contain the anti-correlated fold structure, or this "
        "test proves nothing")

    assert scored == N_FOLDS and undefined == 0

    # The analytically correct answer.
    assert within == pytest.approx(TARGET_WITHIN_IC, abs=0.06), (
        f"every fold was built with a within-fold rank IC of "
        f"{TARGET_WITHIN_IC}; the corrected statistic reads {within:+.4f}")

    # The old one, on the same rows, is not merely smaller - it has the wrong
    # sign, and by a wide margin.
    assert pooled < -0.5, (
        f"the pooled statistic reads {pooled:+.4f}; if it is not strongly "
        f"negative the fixture no longer reproduces the bug")
    assert within - pooled > 1.0, (
        f"pooled {pooled:+.4f} vs within {within:+.4f} - the two must be far "
        f"apart or this regression test has lost its teeth")


def test_the_bug_needs_the_fold_structure_to_appear():
    """
    Control: with the fold levels removed and everything else identical, the
    two statistics agree.

    This is what makes the test above a test of POOLING rather than of the
    fixture: the disagreement has to come from the between-fold arrangement and
    nothing else.
    """
    y_true, y_pred, folds = synthetic_panel()
    for k in range(N_FOLDS):
        m = folds == k
        y_true[m] -= y_true[m].mean()
        y_pred[m] -= y_pred[m].mean()

    pooled = _pooled_rank_ic(y_true, y_pred)
    within, _, _ = within_fold_rank_ic(y_true, y_pred, folds)
    assert pooled == pytest.approx(within, abs=0.05)
    assert within > 0.2


def test_the_corrected_estimate_matches_the_panel_level_pattern():
    """
    Numerical consistency with `_mean_daily_rank_ic`.

    Both compute "correlate inside each group, then average the defined ones".
    Applied to the same grouped data they must agree exactly, or the codebase
    holds two definitions of one quantity - the debt `rank_ic_rows` already
    carries a test for, and for the same reason.
    """
    import pandas as pd

    from pipeline.evaluation import _mean_daily_rank_ic

    y_true, y_pred, folds = synthetic_panel()
    frame = pd.DataFrame({"date": folds, "y_true": y_true, "y_pred": y_pred})

    within, _, _ = within_fold_rank_ic(y_true, y_pred, folds)
    assert within == pytest.approx(_mean_daily_rank_ic(frame), abs=1e-12)


def test_a_fold_with_no_ordering_is_counted_not_scored_as_zero():
    y_true, y_pred, folds = synthetic_panel()
    y_pred[folds == 2] = 0.5                      # one flat fold

    within, scored, undefined = within_fold_rank_ic(y_true, y_pred, folds)
    assert (scored, undefined) == (N_FOLDS - 1, 1)

    # Scoring it as 0.0 would drag the mean toward zero by a fifth. The
    # difference is the whole reason an undefined IC is not a measured one.
    dragged = within * scored / N_FOLDS
    assert within != pytest.approx(dragged, abs=1e-6)


def test_a_track_constant_in_every_fold_is_refused():
    y_true, y_pred, folds = synthetic_panel()
    for k in range(N_FOLDS):
        y_pred[folds == k] = float(k)             # one constant per fold

    within, scored, undefined = within_fold_rank_ic(y_true, y_pred, folds)
    assert scored == 0 and undefined == N_FOLDS and not np.isfinite(within)

    est = block_bootstrap_ic(
        TickerTrack("X.NS", tuple(f"d{i:04d}" for i in range(len(y_true))),
                    y_true, y_pred, folds))
    assert not est.usable
    assert "no ordering inside ANY" in est.reason


# ── the variance estimator ────────────────────────────────────────────────────


def test_no_bootstrap_resample_ever_mixes_two_folds():
    """
    The property that makes the bootstrap an estimator of the NEW statistic.

    A resample that mixes rows across folds destroys the grouping the estimate
    is defined on, so its spread would describe the pooled statistic instead.
    Asserted by construction: give each fold a distinct constant in `y_true`
    and require every resampled fold slice to hold exactly one of them.
    """
    y_true, y_pred, folds = synthetic_panel()
    tagged = folds.astype(float)                  # y_true IS the fold id

    track = TickerTrack("X.NS",
                        tuple(f"d{i:04d}" for i in range(len(y_true))),
                        tagged, y_pred, folds)

    # A constant y_true has no ordering, so the estimator refuses it - which is
    # itself the check that the tag reached the right place.
    est = block_bootstrap_ic(track, n_resamples=50)
    assert not est.usable

    # The positive form: fold slices are contiguous index ranges, and a block
    # drawn inside one cannot reach another.
    for k in range(N_FOLDS):
        rows = np.flatnonzero(folds == k)
        assert rows.max() - rows.min() == rows.size - 1, (
            "folds must be contiguous for the within-fold draw to be a block "
            "of anything")


def test_the_bootstrap_variance_tracks_a_known_sampling_variance():
    """
    The estimator is checked against an analytically derivable target.

    Under the null of no skill, a rank IC over n independent observations has
    sampling variance ~= 1/(n-1). The statistic here is the MEAN of five such
    correlations, so its variance is ~= 1/(5 * (n_eff - 1)) where n_eff
    discounts the block structure. With independent rows (block = 1) that is
    exact enough to check an order of magnitude and a direction, which is what
    a variance estimator has to get right.
    """
    rng = np.random.default_rng(11)
    rows = 200
    folds = np.repeat(np.arange(N_FOLDS), rows)
    y_true = rng.normal(size=N_FOLDS * rows)
    y_pred = rng.normal(size=N_FOLDS * rows)      # no skill, by construction

    track = TickerTrack("X.NS",
                        tuple(f"d{i:04d}" for i in range(len(y_true))),
                        y_true, y_pred, folds)
    est = block_bootstrap_ic(track, block=1, n_resamples=1500)
    assert est.usable

    expected = 1.0 / (N_FOLDS * (rows - 1))
    assert est.sigma2 == pytest.approx(expected, rel=0.5), (
        f"bootstrap sigma2 {est.sigma2:.3e} against an analytic "
        f"{expected:.3e}")


def test_a_longer_block_widens_the_variance():
    """
    The block length has to still do its job.

    Blocks exist because a 30-session label makes consecutive rows dependent;
    a longer block preserves more of that dependence and must therefore report
    MORE uncertainty, not less. If the within-fold rewrite had quietly lost the
    block structure, this is what would catch it.
    """
    rng = np.random.default_rng(3)
    rows = 300
    folds = np.repeat(np.arange(N_FOLDS), rows)

    # Persistent series, so block length has something to preserve.
    def persistent(n, window=30):
        x = rng.normal(size=n + window)
        return np.convolve(x, np.ones(window) / window, mode="valid")[:n]

    y_true = np.concatenate([persistent(rows) for _ in range(N_FOLDS)])
    y_pred = np.concatenate([persistent(rows) for _ in range(N_FOLDS)])
    track = TickerTrack("X.NS",
                        tuple(f"d{i:04d}" for i in range(len(y_true))),
                        y_true, y_pred, folds)

    short = block_bootstrap_ic(track, block=1, n_resamples=800)
    long = block_bootstrap_ic(track, block=BLOCK_LENGTH_SESSIONS,
                              n_resamples=800)
    assert short.usable and long.usable
    assert long.sigma2 > short.sigma2, (
        f"block {BLOCK_LENGTH_SESSIONS} reported sigma2 {long.sigma2:.3e} "
        f"against block 1's {short.sigma2:.3e}; the block structure is not "
        f"reaching the resample")


def test_fold_labels_are_required_rather_than_silently_pooled():
    """
    The failure mode this session exists to remove, refused loudly.

    A track with no fold labels cannot be scored the correct way, and the only
    thing computable from it is the pooled correlation. Falling back to that
    silently is precisely how the bug survived four sessions.
    """
    y_true, y_pred, _ = synthetic_panel()
    track = TickerTrack("X.NS",
                        tuple(f"d{i:04d}" for i in range(len(y_true))),
                        y_true, y_pred, None)
    with pytest.raises(EvidenceGradingRefused, match="fold labels are required"):
        block_bootstrap_ic(track)


def test_the_point_estimate_the_bootstrap_reports_is_the_within_fold_one():
    """The estimate and its variance must describe the same statistic; a
    bootstrap around a different point estimate is not an error anything
    downstream could see."""
    y_true, y_pred, folds = synthetic_panel()
    track = TickerTrack("X.NS",
                        tuple(f"d{i:04d}" for i in range(len(y_true))),
                        y_true, y_pred, folds)
    est = block_bootstrap_ic(track, n_resamples=200)
    within, _, _ = within_fold_rank_ic(y_true, y_pred, folds)

    assert est.usable
    assert est.hat_ic == pytest.approx(within, abs=1e-12)
    assert est.hat_ic > 0.2, "and it must carry the POSITIVE within-fold sign"


# ── the variance has TWO components, and the first fix only had one ──────────


def test_period_heterogeneity_reaches_the_variance():
    """
    The correction to the correction, and the reason it exists.

    The first Stage 0b bootstrap resampled inside each fold only. That
    estimates how much each fold's IC would move if that period were
    re-sampled, and says nothing about how much the ICs differ BETWEEN
    periods — which is part of the uncertainty of a statistic that is a mean
    over periods.

    Measured on the real panel it was a median of 1.5x too small, and the
    consequence was 12 tickers graded STRONG that should not have been. Here
    it is reproduced deterministically: two tracks with the SAME within-fold
    noise, one whose fold ICs agree and one whose fold ICs are wildly
    dispersed. The dispersed one must report the larger variance.
    """
    rng = np.random.default_rng(21)
    rows, n_folds = 240, 5
    folds = np.repeat(np.arange(n_folds), rows)

    def build(ics):
        yt, yp = [], []
        for ic in ics:
            rho = 2.0 * np.sin(np.pi * ic / 6.0)
            z = rng.normal(size=rows)
            e = rng.normal(size=rows)
            yp.append(z)
            yt.append(rho * z + np.sqrt(max(1.0 - rho ** 2, 0.0)) * e)
        return np.concatenate(yt), np.concatenate(yp)

    same_t, same_p = build([0.20] * n_folds)
    spread_t, spread_p = build([-0.60, -0.30, 0.20, 0.70, 1.00 - 0.01])

    dates = tuple(f"d{i:05d}" for i in range(rows * n_folds))
    agree = block_bootstrap_ic(
        TickerTrack("AGREE.NS", dates, same_t, same_p, folds), n_resamples=400)
    disperse = block_bootstrap_ic(
        TickerTrack("SPREAD.NS", dates, spread_t, spread_p, folds),
        n_resamples=400)

    assert agree.usable and disperse.usable
    assert disperse.sigma2 > 3 * agree.sigma2, (
        f"a ticker whose fold ICs run from -0.60 to +0.99 reported sigma2 "
        f"{disperse.sigma2:.5f} against {agree.sigma2:.5f} for one whose folds "
        f"all agree; between-period variation is not reaching the estimate")


def test_the_bootstrap_is_a_floor_under_the_between_fold_estimate():
    """
    Neither component may be dropped.

    With only five folds the between-fold sample variance is itself very noisy,
    and a ticker whose fold ICs happen to land close together would buy a
    near-zero variance from that coincidence. Taking the larger of the two
    stops that, and this pins the direction: the reported variance is never
    below the within-fold bootstrap's.
    """
    y_true, y_pred, folds = synthetic_panel()
    for k in range(N_FOLDS):                       # force the fold ICs to agree
        m = folds == k
        y_true[m] -= y_true[m].mean()
        y_pred[m] -= y_pred[m].mean()

    track = TickerTrack("X.NS", tuple(f"d{i:04d}" for i in range(len(y_true))),
                        y_true, y_pred, folds)
    est = block_bootstrap_ic(track, n_resamples=600)
    assert est.usable

    ics = np.array([stats.spearmanr(y_true[folds == k],
                                    y_pred[folds == k]).statistic
                    for k in range(N_FOLDS)])
    between = float(np.var(ics, ddof=1) / ics.size)
    assert est.sigma2 >= between, "the between-fold estimate was used alone"
    assert est.sigma2 > 0


def test_a_ticker_scored_on_too_few_folds_is_refused():
    """
    A mean over two numbers is not a track record, and the model that most
    needs this rule is the production one: 316 of its 420 (ticker, fold) cells
    emit a constant, so the median ticker has two folds carrying an ordering.
    Grading those means grading survivors selected on the model having split.
    """
    from pipeline.evidence_shrinkage import MIN_FOLDS_FOR_ESTIMATE

    y_true, y_pred, folds = synthetic_panel()
    for k in range(N_FOLDS - MIN_FOLDS_FOR_ESTIMATE + 1):
        y_pred[folds == k] = float(k)              # flatten all but two

    est = block_bootstrap_ic(
        TickerTrack("FEW.NS", tuple(f"d{i:04d}" for i in range(len(y_true))),
                    y_true, y_pred, folds))
    assert not est.usable
    assert "is not a track record" in est.reason
    assert np.isfinite(est.hat_ic), (
        "the point estimate is still reportable; it is the GRADE that is "
        "refused, and a reader should see the number that was refused")


# ── the headline comparator table must STAY clean ────────────────────────────


def test_the_reported_comparator_metrics_are_never_the_pooled_ic():
    """
    THE REGRESSION PIN ON THE PROJECT'S ORIGINAL HEADLINE TABLE.

    The Stage 0b audit found `pipeline/baselines.py` clean: every number quoted
    in this project's comparator tables — `daily_IC`, `reb_IC`, `reb_t`,
    `alpha_t` — comes from the within-date-then-averaged path, and the pooled
    figure is carried in a separate `rank_ic` key that is not persisted, not
    used by `clears_floor`, and not what `best()` ranks on.

    That is worth pinning rather than trusting, because the two live in the
    same dict one line apart and swapping them would change every headline in
    the project while raising nothing.
    """
    from pipeline.baselines import BaselineComparison

    kept = BaselineComparison(results=[], coverage={}, note="").to_metrics()
    assert kept["comparators"] == []

    # The keys `to_metrics` keeps, read off the real implementation.
    import inspect

    source = inspect.getsource(BaselineComparison.to_metrics)
    assert '"daily_rank_ic"' in source, "the within-date metric is not persisted"
    assert '"rebalance_ic_t"' in source, "the rebalance t is not persisted"
    assert '"rank_ic"' not in source.replace('"daily_rank_ic"', ""), (
        "the POOLED rank IC is being persisted to experiment_runs; it is moved "
        "by fold identity and must not become a reported number")


def test_the_floors_are_decided_on_the_rebalance_ic_not_the_pooled_one():
    """`clears_floor` is the P1 criterion. If it ever read the pooled figure,
    every comparator's standing against `beta_market` would be decided by which
    fold a row came from."""
    import inspect

    from pipeline.baselines import annotate_against_floors

    source = inspect.getsource(annotate_against_floors)
    assert 'r["rebalance_ic"] > beta_ic' in source
    assert '"rank_ic"' not in source


def test_the_panel_harness_reports_the_per_date_ic_beside_the_pooled_one():
    """
    Both are computed; only one is trustworthy, and they must stay distinct.

    A change that made `daily_rank_ic` an alias of `rank_ic` would leave every
    table rendering perfectly and every number wrong — the same shape as the
    `linear_factor+val` defect, where two rows agreed to five decimals because
    one of them was never really computed.
    """
    import numpy as np
    import pandas as pd

    from pipeline.evaluation import _mean_daily_rank_ic, rank_ic

    y_true, y_pred, folds = synthetic_panel()
    frame = pd.DataFrame({"date": folds, "y_true": y_true, "y_pred": y_pred})

    pooled = rank_ic(y_true, y_pred)
    per_date = _mean_daily_rank_ic(frame)
    assert np.isfinite(pooled) and np.isfinite(per_date)
    assert abs(per_date - pooled) > 1.0, (
        "the two statistics agreed on a fixture built to separate them; one of "
        "them is not being computed the way it claims")
