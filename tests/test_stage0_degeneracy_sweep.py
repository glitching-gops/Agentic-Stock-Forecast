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

The `respect_fold_gaps` flag added to `pipeline/evidence_shrinkage.py` for this
is STRICTLY ADDITIVE, and the bit-identity test below is what makes that claim
checkable rather than asserted — Stage 0's numbers are committed and must not
move underneath them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.evidence_shrinkage import (
    BLOCK_LENGTH_SESSIONS,
    TickerTrack,
    _block_start_pool,
    block_bootstrap_ic,
    contiguous_segments,
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


# ── Segment handling: the seam a dropped fold leaves ──────────────────────────


def test_contiguous_segments_breaks_only_where_a_fold_was_removed():
    """
    Fold ids are consecutive and `PurgedWalkForward`'s test windows abut, so
    3 -> 4 is a neighbour and 1 -> 3 is a seam.
    """
    intact = np.repeat([0, 1, 2, 3, 4], 10)
    assert contiguous_segments(intact) == [(0, 50)]

    fold_two_dropped = np.repeat([0, 1, 3, 4], 10)
    assert contiguous_segments(fold_two_dropped) == [(0, 20), (20, 40)]

    alternating = np.repeat([0, 2, 4], 10)
    assert contiguous_segments(alternating) == [(0, 10), (10, 20), (20, 30)]


def test_no_block_start_ever_spans_a_seam():
    """The guarantee itself, not a proxy for it."""
    folds = np.repeat([0, 1, 3, 4], 50)          # seam at index 100
    pool = _block_start_pool(len(folds), block=30, fold_ids=folds)
    for start in pool:
        block = folds[start:start + 30]
        assert np.all(np.diff(block) <= 1), (
            f"a block starting at {start} spans the seam left by fold 2")


def test_a_segment_shorter_than_the_block_contributes_no_starts():
    folds = np.repeat([0, 2], [100, 10])         # second segment is 10 rows
    pool = _block_start_pool(len(folds), block=30, fold_ids=folds)
    assert pool.max() + 30 <= 100


def test_a_ticker_with_no_run_long_enough_is_refused_not_estimated():
    rng = np.random.default_rng(0)
    n = 200
    folds = np.repeat([0, 2, 4, 6, 8], 40)       # every run is 40 rows...
    track = TickerTrack("SHARD.NS", tuple(str(i) for i in range(n)),
                        rng.normal(size=n), rng.normal(size=n), folds=folds)
    est = block_bootstrap_ic(track, block=60, n_resamples=50,
                             respect_fold_gaps=True)
    assert est.usable is False
    assert "no surviving contiguous run" in est.reason


# ── The additive change must not have moved Stage 0 ───────────────────────────


def test_respecting_fold_gaps_is_bit_identical_when_there_are_none():
    """
    THE GUARANTEE THAT PROTECTS STAGE 0'S COMMITTED NUMBERS.

    With no seam the legal-start pool is `arange(0, n - block + 1)`, so
    `pool[rng.integers(0, len(pool))]` draws exactly what
    `rng.integers(0, n_starts)` drew before — same bounds, same shape, same
    stream. Anything less than bit-identity here means the addendum silently
    rewrote the number it exists to check against.
    """
    rng = np.random.default_rng(7)
    n = 900
    folds = np.repeat([0, 1, 2, 3, 4], n // 5)
    track = TickerTrack("SAME.NS", tuple(str(i) for i in range(n)),
                        rng.normal(size=n), rng.normal(size=n), folds=folds)

    off = block_bootstrap_ic(track, n_resamples=300, respect_fold_gaps=False)
    on = block_bootstrap_ic(track, n_resamples=300, respect_fold_gaps=True)

    assert on.hat_ic == off.hat_ic
    assert on.sigma2 == off.sigma2, (
        "the fold-gap-aware path changed the bootstrap on a series that has no "
        "gaps; Stage 0's committed mu_hat is no longer reproducible")


def test_the_default_path_is_untouched_by_the_addendum():
    """`respect_fold_gaps` defaults False, so nothing that existed before the
    addendum takes the new branch even when fold labels are present."""
    rng = np.random.default_rng(8)
    n = 600
    folds = np.repeat([0, 1, 3, 4], n // 4)      # deliberately has a seam
    track = TickerTrack("DEF.NS", tuple(str(i) for i in range(n)),
                        rng.normal(size=n), rng.normal(size=n), folds=folds)

    default = block_bootstrap_ic(track, n_resamples=200)
    explicit_off = block_bootstrap_ic(track, n_resamples=200,
                                      respect_fold_gaps=False)
    seam_aware = block_bootstrap_ic(track, n_resamples=200,
                                    respect_fold_gaps=True)

    assert default.sigma2 == explicit_off.sigma2
    assert seam_aware.sigma2 != default.sigma2, (
        "the seam-aware path made no difference on a series that HAS a seam, "
        "so it is not doing anything")


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
    """Renumbering survivors 0..n would erase exactly the information
    `contiguous_segments` needs to find the gap."""
    tracks, folds = _panel({"A.NS": [0.0, 0.0, 1.0, 0.0, 0.0]}, n_per_fold=50)
    restricted = restrict(tracks[0], folds["A.NS"], keep={0, 1, 3, 4})

    assert set(restricted.folds.tolist()) == {0, 1, 3, 4}
    assert contiguous_segments(restricted.folds) == [(0, 100), (100, 200)]


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
                            n_resamples=200, respect_fold_gaps=True)

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
def test_tau_of_one_reproduces_stage0s_committed_mu_hat_exactly():
    """
    The real thing. Stage 0's `mu_hat = -0.05988188592526484` is committed and
    reported; tau = 1.00 excludes nothing and must re-derive it to the last
    digit through the addendum's own code path. Anything else means the sweep
    is measuring its own reimplementation.
    """
    from tools.run_evidence_grading import load_cache

    cache = load_cache(CACHE)
    scores = cell_scores(cache["tracks"], cache["folds"])
    table, _ = sweep(cache["tracks"], cache["folds"], scores, taus=(1.00,))

    row = table.iloc[0]
    assert row["mu_hat"] == pytest.approx(STAGE0_MU_HAT, abs=1e-15)
    assert row["tau2_hat"] == pytest.approx(STAGE0_TAU2, abs=1e-15)
    assert int(row["n_effective"]) == STAGE0_N_USABLE
    assert row["folds_dropped"] == 0, "tau = 1.00 excluded something"
