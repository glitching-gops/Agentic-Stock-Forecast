"""
Evidence-Grading Redesign, Stage 0 ADDENDUM — the degeneracy sensitivity sweep.

The addendum's whole job is to say how much of Stage 0's `mu_hat = -0.05988`
survives once partially-degenerate folds are excluded too, not just the fully
degenerate tickers the v2 guard already refuses. Two things have to be true for
that answer to mean anything, and each has a test here whose failure message
says which one broke:

  1. **tau = 1.00 must reproduce Stage 0 exactly.** It excludes nothing, so it
     is a pure re-derivation of a committed number by a different code path. If
     it drifts, the sweep is measuring its own reimplementation and every row
     below it is void.
  2. **The bootstrap must not draw blocks across the seam a dropped fold
     leaves.** Rows either side of an excluded fold are months apart; a block
     spanning that join manufactures continuity that is not in the data, which
     is the same class of error as a block shorter than the label overlap.

Stage 0b made the second point STRUCTURAL. The corrected estimator in
`pipeline/evidence_shrinkage.py` draws its blocks within each fold, so no
resample can span a seam whatever was dropped, and the `respect_fold_gaps`
flag this file used to exercise is gone with the helpers behind it.

**The sweep's own tau-grid numbers are SUPERSEDED by that fix.** They were
computed on the pooled-across-folds statistic Stage 0b replaced, so the
addendum's table is history rather than a current measurement. This file tests
the sweep's MECHANICS - selection, restriction, bookkeeping - which are
unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.evidence_shrinkage import (
    BLOCK_LENGTH_SESSIONS,
    TickerTrack,
    block_bootstrap_ic,
    dersimonian_laird_tau2,
    grade_panel,
    precision_weighted_mean,
)
from tools.stage0_degeneracy_sweep import (
    STAGE0_MU_HAT,
    STAGE0_N_USABLE,
    STAGE0_TAU2,
    TRUST_MIN_TICKERS,
    cell_scores,
    mode_share,
    restrict,
    sweep,
    unique_fraction,
)

CACHE = Path("evidence_oos.npz")


# ── The two degeneracy metrics ────────────────────────────────────────────────


def test_mode_share_on_hand_built_folds():
    """Fully constant, partially repeated, fully varied — the three shapes."""
    assert mode_share(np.full(10, 0.017)) == 1.0
    assert mode_share(np.arange(10.0)) == pytest.approx(0.1)
    # 7 of 10 rows share a value, the other 3 are distinct.
    assert mode_share(np.array([1.0] * 7 + [2.0, 3.0, 4.0])) == pytest.approx(0.7)
    assert np.isnan(mode_share(np.array([])))


def test_mode_share_uses_exact_equality_not_a_tolerance():
    """
    A tree that made no splits returns the SAME float for every row, bit for
    bit. Clustering nearby values would classify a genuinely varying but
    low-dispersion fold as degenerate and exclude real ordering.
    """
    nearly = np.full(10, 0.017)
    nearly[0] = np.nextafter(0.017, 1.0)
    assert mode_share(nearly) == pytest.approx(0.9)


def test_unique_fraction_on_hand_built_folds():
    assert unique_fraction(np.full(10, 0.017)) == pytest.approx(0.1)
    assert unique_fraction(np.arange(10.0)) == pytest.approx(1.0)
    assert unique_fraction(np.array([1.0] * 7 + [2.0, 3.0, 4.0])) == pytest.approx(0.4)


def test_the_two_metrics_disagree_where_they_should():
    """
    They are not redundant, which is why both are reported. A fold split evenly
    between TWO values is mild by mode-share (0.5) and extreme by unique
    fraction (0.02) — so a sweep on mode-share alone would keep it. The report
    measures their agreement rather than assuming it.
    """
    two_valued = np.array([1.0] * 50 + [2.0] * 50)
    assert mode_share(two_valued) == pytest.approx(0.5)
    assert unique_fraction(two_valued) == pytest.approx(0.02)


# ── The seam a dropped fold leaves is now handled STRUCTURALLY ────────────────
#
# Six tests lived here, covering `contiguous_segments`, `_block_start_pool` and
# the `respect_fold_gaps` flag. Stage 0b deleted all three: the corrected
# bootstrap draws its blocks WITHIN each fold, so a resample cannot span a seam
# whatever folds were dropped. The property they bought is not gone - it is no
# longer optional, and the tests below assert it on the estimator itself rather
# than on a helper that has to be remembered to switch on.


def test_no_resample_can_span_the_seam_a_dropped_fold_left():
    """
    The addendum's guarantee, asserted at the altitude that now provides it.

    Folds 0, 1, 3, 4 with fold 2 removed: rows either side of the join are
    months apart. Blocks are drawn inside each fold, so the join is
    unreachable — and this is checked by construction rather than by a flag,
    because a flag has to be passed and this cannot be forgotten.
    """
    rng = np.random.default_rng(4)
    n = 800
    folds = np.repeat([0, 1, 3, 4], n // 4)
    track = TickerTrack("SEAM.NS", tuple(str(i) for i in range(n)),
                        rng.normal(size=n), rng.normal(size=n), folds=folds)

    est = block_bootstrap_ic(track, n_resamples=200)
    assert est.usable

    # Every fold is a contiguous index range, and the estimator only ever
    # indexes inside one of them, so no block can hold two fold ids.
    for k in np.unique(folds):
        rows = np.flatnonzero(folds == k)
        assert rows.max() - rows.min() == rows.size - 1


def test_a_fold_shorter_than_the_block_is_dropped_rather_than_shrunk():
    """
    A short fold cannot be resampled at its own altitude. Shrinking the block
    to fit would understate the autocorrelation the block length exists to
    preserve, so the fold is dropped and the estimate says so when nothing is
    left.
    """
    rng = np.random.default_rng(0)
    n = 200
    folds = np.repeat([0, 2, 4, 6, 8], 40)       # every fold is 40 rows...
    track = TickerTrack("SHARD.NS", tuple(str(i) for i in range(n)),
                        rng.normal(size=n), rng.normal(size=n), folds=folds)

    est = block_bootstrap_ic(track, block=60, n_resamples=50)   # ...block is 60
    assert est.usable is False
    assert "no fold holds both an ordering" in est.reason


def test_dropping_a_fold_changes_the_estimate_but_never_the_grouping():
    """
    Restriction has to reach the statistic — otherwise the whole tau sweep is
    measuring nothing — while leaving each surviving fold scored on its own.
    """
    rng = np.random.default_rng(9)
    n = 1000
    folds = np.repeat([0, 1, 2, 3, 4], n // 5)
    y_true, y_pred = rng.normal(size=n), rng.normal(size=n)
    full = TickerTrack("ALL.NS", tuple(str(i) for i in range(n)),
                       y_true, y_pred, folds=folds)

    keep = folds != 2
    dropped = TickerTrack("ALL.NS", tuple(str(i) for i in np.flatnonzero(keep)),
                          y_true[keep], y_pred[keep], folds=folds[keep])

    a = block_bootstrap_ic(full, n_resamples=200)
    b = block_bootstrap_ic(dropped, n_resamples=200)
    assert a.usable and b.usable
    assert a.hat_ic != b.hat_ic, "dropping a fold did not reach the estimate"


# ── Exclusion and bookkeeping ─────────────────────────────────────────────────


def _panel(spec: dict[str, list[float]], n_per_fold: int = 120, seed: int = 1):
    """
    A synthetic panel with KNOWN degeneracy. ``spec`` maps ticker -> per-fold
    mode-share targets, where 1.0 means a fully constant fold.
    """
    rng = np.random.default_rng(seed)
    tracks, folds = [], {}
    for ticker, shares in spec.items():
        y_true, y_pred, fold = [], [], []
        for k, share in enumerate(shares):
            truth = rng.normal(size=n_per_fold)
            n_const = int(round(share * n_per_fold))
            pred = np.concatenate([np.full(n_const, 0.01 * (k + 1)),
                                   rng.normal(size=n_per_fold - n_const)])
            y_true.append(truth)
            y_pred.append(pred)
            fold.append(np.full(n_per_fold, k))
        name = ticker
        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        tracks.append(TickerTrack(name, tuple(str(i) for i in range(len(y_true))),
                                  y_true, y_pred,
                                  folds=np.concatenate(fold)))
        folds[name] = np.concatenate(fold)
    return tracks, folds


def test_cell_scores_recovers_the_planted_degeneracy():
    tracks, folds = _panel({"A.NS": [1.0, 0.5, 0.0], "B.NS": [0.0, 0.0, 1.0]})
    scores = cell_scores(tracks, folds)

    assert len(scores) == 6
    a = scores[scores["ticker"] == "A.NS"].sort_values("fold")["mode_share"].tolist()
    assert a[0] == pytest.approx(1.0)          # planted fully constant
    assert a[1] == pytest.approx(0.5)          # planted half constant
    # Planted fully varied: 120 distinct draws, so the modal value is one row.
    assert a[2] == pytest.approx(1 / 120, abs=1e-6)


def test_the_right_folds_are_excluded_at_each_tau():
    """A fold is dropped when its mode-share EXCEEDS tau — so tau = 1.00 drops
    nothing, which is what makes it a sanity check rather than a data point."""
    tracks, folds = _panel({"A.NS": [1.0, 0.6, 0.2, 0.0, 0.0]})
    scores = cell_scores(tracks, folds)
    shares = scores.set_index("fold")["mode_share"]

    for tau, expected_kept in ((1.00, {0, 1, 2, 3, 4}),
                               (0.90, {1, 2, 3, 4}),
                               (0.50, {2, 3, 4}),
                               (0.10, {3, 4})):
        kept = set(shares[shares <= tau].index)
        assert kept == expected_kept, f"wrong folds kept at tau={tau}"


def test_restrict_keeps_the_original_fold_ids_so_the_seam_stays_visible():
    """
    Renumbering survivors 0..n would erase which fold each row came from.

    That mattered for seam detection before Stage 0b and matters MORE now: the
    corrected estimator groups by fold id to compute the statistic at all, so a
    renumbering would silently merge nothing but would make the reported fold
    ids disagree with the sweep's own bookkeeping.
    """
    tracks, folds = _panel({"A.NS": [0.0, 0.0, 1.0, 0.0, 0.0]}, n_per_fold=50)
    restricted = restrict(tracks[0], folds["A.NS"], keep={0, 1, 3, 4})

    assert set(restricted.folds.tolist()) == {0, 1, 3, 4}
    assert restricted.folds.size == 200
    # And the ids stay in date order, which is what makes each fold a
    # contiguous block of rows for the within-fold draw.
    assert np.all(np.diff(restricted.folds) >= 0)


def test_a_ticker_that_loses_every_fold_is_counted_and_never_imputed():
    """
    The bookkeeping the addendum turns on: as tau falls, tickers drop out of
    the panel entirely. Silently omitting them would shrink `n_effective` with
    no trace; imputing a value would invent evidence.
    """
    tracks, folds = _panel({
        "GONE.NS": [1.0, 1.0, 1.0, 1.0, 1.0],       # every fold constant
        "KEEP1.NS": [0.0, 0.0, 0.0, 0.0, 0.0],
        "KEEP2.NS": [0.0, 0.0, 0.0, 0.0, 0.0],
        "KEEP3.NS": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    scores = cell_scores(tracks, folds)
    table, per_ticker = sweep(tracks, folds, scores, taus=(0.90,),
                              n_resamples=100)

    row = table.iloc[0]
    assert row["tickers_no_folds_left"] == 1
    assert row["n_effective"] == 3

    gone = per_ticker[per_ticker["ticker"] == "GONE.NS"].iloc[0]
    assert gone["status"] == "no_folds_left"
    assert np.isnan(gone["hat_ic"]), "a dropped ticker was given an imputed IC"
    assert gone["n_folds_kept"] == 0


def test_the_trust_flag_fires_below_half_the_panel():
    tracks, folds = _panel({f"T{i}.NS": [0.0] * 3 for i in range(4)})
    scores = cell_scores(tracks, folds)
    table, _ = sweep(tracks, folds, scores, taus=(1.00,), n_resamples=100)
    # 4 effective tickers is far below TRUST_MIN_TICKERS
    assert TRUST_MIN_TICKERS > 4
    assert table.iloc[0]["trust"] == "UNTRUSTED"


def test_tau_of_one_reproduces_the_unrestricted_computation():
    """
    THE SANITY CHECK, in a form that runs without the 84-ticker cache.

    tau = 1.00 excludes nothing, so the sweep must land on exactly what
    `grade_panel` computes over the same tracks. This is the synthetic twin of
    the real-cache assertion below, and it is the one that runs in CI.
    """
    tracks, folds = _panel({f"T{i}.NS": [0.0] * 5 for i in range(6)}, seed=3)
    scores = cell_scores(tracks, folds)
    table, _ = sweep(tracks, folds, scores, taus=(1.00,), n_resamples=200)

    reference = grade_panel(tracks, break_even=0.00512363994209475,
                            n_resamples=200)

    assert table.iloc[0]["mu_hat"] == pytest.approx(reference.mu_hat, abs=1e-15)
    assert table.iloc[0]["tau2_hat"] == pytest.approx(reference.tau2, abs=1e-15)
    assert table.iloc[0]["n_effective"] == reference.n_usable


def test_the_sweep_is_deterministic():
    tracks, folds = _panel({f"T{i}.NS": [0.0, 1.0, 0.0, 0.0, 0.0]
                            for i in range(5)}, seed=5)
    scores = cell_scores(tracks, folds)
    a, _ = sweep(tracks, folds, scores, taus=(0.90, 0.50), n_resamples=150)
    b, _ = sweep(tracks, folds, scores, taus=(0.90, 0.50), n_resamples=150)
    pd.testing.assert_frame_equal(a, b)


# ── Against the real cache, when it is present ────────────────────────────────


@pytest.mark.skipif(not CACHE.exists(),
                    reason="evidence_oos.npz absent (it is gitignored; rebuild "
                           "with tools/run_evidence_grading.py --rebuild)")
def test_tau_of_one_excludes_nothing_and_no_longer_reproduces_stage0():
    """
    The sanity check, and the record of what Stage 0b did to it.

    Before Stage 0b this asserted that tau = 1.00 re-derived Stage 0's
    committed `mu_hat = -0.05988188592526484` to the last digit — the check
    that made the rest of the sweep readable. That constant is now SUPERSEDED:
    it was computed by pooling every fold's rows into one series and
    correlating once, and the corrected estimator averages the correlation
    WITHIN each fold instead.

    What survives, and is still worth pinning, is the STRUCTURAL half: tau =
    1.00 must exclude nothing. The value half is deliberately inverted — the
    corrected figure must NOT equal the old one, because a fix that left it
    unchanged would not have reached the statistic.
    """
    from tools.run_evidence_grading import load_cache

    cache = load_cache(CACHE)
    scores = cell_scores(cache["tracks"], cache["folds"])
    table, _ = sweep(cache["tracks"], cache["folds"], scores, taus=(1.00,))

    row = table.iloc[0]
    assert row["folds_dropped"] == 0, "tau = 1.00 excluded something"
    assert abs(row["mu_hat"] - STAGE0_MU_HAT) > 0.01, (
        f"mu_hat came back at {row['mu_hat']:+.5f}, within a hair of the "
        f"superseded {STAGE0_MU_HAT:+.5f}; the within-fold correction did not "
        f"reach the sweep")
    assert np.isfinite(row["mu_hat"]) and np.isfinite(row["tau2_hat"])
