"""
Stage 0c — the corrected evidence layer, and the guards that make it readable.

Four things changed at once in this layer and each can be plausibly wrong in a
way that produces a complete, well-formed table:

  - cross-sectional demeaning applied to one side only, which leaves the market
    factor in the prediction and scores a model that forecasts nothing else;
  - a bootstrap that resamples across folds, destroying the walk-forward
    ordering the purge exists to protect;
  - the between-fold variance term added ON TOP of a bootstrap that already
    contains it, which double-counts and shrinks every interval;
  - a tau2 at the zero boundary rendered as 84 per-ticker findings instead of
    one panel-level one.

Each has a test below whose failure message says which one broke.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pipeline.evidence_panel import (
    BLOCK_LENGTH_SESSIONS,
    MIN_CROSS_SECTION,
    TAU2_ZERO_TOL,
    benjamini_yekutieli,
    build_panel,
    circular_block_indices,
    cochran_q,
    cross_sectional_ic_series,
    detect_tau2_degeneracy,
    driscoll_kraay_se,
    grade_panel_v3,
    hksj_interval,
    induced_pairwise_correlation,
    nan_rank_ic_rows,
    per_ticker_fold_ics,
    rank_demean_by_date,
    rank_to_unit_interval,
    reml_tau2,
    resample_selections,
    romano_wolf,
)

N_TICKERS = 84
N_FOLDS = 5


# ── fixtures ──────────────────────────────────────────────────────────────────


def synthetic_panel(n_dates: int = 400, n_tickers: int = N_TICKERS,
                    n_folds: int = N_FOLDS, ic: float = 0.0,
                    seed: int = 0) -> dict:
    """
    A balanced panel with a planted CROSS-SECTIONAL IC and a common market
    factor on top.

    The market factor is the point: it moves every name together, so a raw
    per-ticker time-series IC sees it and a cross-sectionally demeaned one does
    not. Any test that did not include it could not tell the two apart.
    """
    rng = np.random.default_rng(seed)
    dates = np.repeat([f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" if i < 300
                       else f"d{i:05d}" for i in range(n_dates)], n_tickers)
    dates = np.array([f"d{i:05d}" for i in range(n_dates)]).repeat(n_tickers)
    tickers = np.tile([f"T{j:02d}.NS" for j in range(n_tickers)], n_dates)
    folds = np.repeat(np.arange(n_folds).repeat(n_dates // n_folds), n_tickers)
    folds = folds[:dates.size]

    n = dates.size
    signal = rng.normal(size=n)
    market = np.repeat(rng.normal(scale=3.0, size=n_dates), n_tickers)
    rho = 2.0 * np.sin(np.pi * ic / 6.0) if ic else 0.0
    noise = rng.normal(size=n)

    y_pred = signal + market
    y_true = rho * signal + np.sqrt(max(1.0 - rho ** 2, 0.0)) * noise + market
    return {"dates": dates, "tickers": tickers, "folds": folds,
            "y_true": y_true, "y_pred": y_pred}


# ── §3.1 rank-demeaning, and the -1/(N-1) it induces ─────────────────────────


def test_ranks_map_to_the_closed_unit_interval():
    out = rank_to_unit_interval(np.array([3.0, 1.0, 2.0, 5.0]))
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(+1.0)
    assert out.mean() == pytest.approx(0.0, abs=1e-12)
    # Order preserved, spacing uniform: this is the Gu-Kelly-Xiu convention.
    assert list(np.argsort(out)) == [1, 2, 0, 3]


def test_demeaning_induces_the_exact_minus_one_over_n_minus_one_correlation():
    """
    THE COMPOSITIONAL ARTIFACT, MEASURED RATHER THAN ASSUMED.

    Cross-sectional demeaning applies C = I - (1/N)11', idempotent with rank
    N-1, so the demeaned residuals of N i.i.d. names carry an EXACT pairwise
    correlation of -1/(N-1) and one lost degree of freedom per date. At N = 84
    that is -0.01205.

    Pearson (1897) found it in ratios of independent quantities; Chayes (1960)
    in constant-sum geochemical data; Aitchison (1986) built compositional data
    analysis around it. It needs no correction HERE because the date-level
    bootstrap resamples whole cross-sections and carries it through — but that
    is an argument, and this is the measurement.
    """
    n_names, n_draws = N_TICKERS, 4000
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(n_draws, n_names))
    demeaned = raw - raw.mean(axis=1, keepdims=True)

    corr = np.corrcoef(demeaned, rowvar=False)
    off = corr[~np.eye(n_names, dtype=bool)]
    expected = induced_pairwise_correlation(n_names)

    assert expected == pytest.approx(-1.0 / (n_names - 1))
    assert expected == pytest.approx(-0.01205, abs=1e-5)
    assert off.mean() == pytest.approx(expected, abs=2e-3), (
        f"demeaned i.i.d. cross-sections show a mean off-diagonal correlation "
        f"of {off.mean():+.5f} against the exact {expected:+.5f}")

    # And the rank drops by exactly one.
    assert np.linalg.matrix_rank(demeaned[:n_names]) == n_names - 1


def test_both_sides_are_demeaned_not_just_the_target():
    """
    Demeaning the target alone leaves the market IN the prediction, so a model
    that forecasts nothing but the market level still scores against the
    residual through its own drift. This is the difference between "did this
    name beat its cross-section" and "did this name go up".
    """
    data = synthetic_panel(n_dates=120, n_tickers=30, seed=3)
    keep, dt, dp, _ = rank_demean_by_date(
        data["dates"], data["y_true"], data["y_pred"], min_cross_section=10)

    for arr, label in ((dt, "target"), (dp, "prediction")):
        by_date: dict[str, list[float]] = {}
        for d, v, k in zip(data["dates"], arr, keep):
            if k:
                by_date.setdefault(d, []).append(v)
        means = np.array([np.mean(v) for v in by_date.values()])
        assert np.abs(means).max() < 1e-12, (
            f"the {label} side is not centred within every date")


def test_a_thin_cross_section_is_dropped_and_counted_never_imputed():
    """
    A demeaning is against THAT date's own cross-section. Carrying a mean
    forward from an adjacent date, or imputing the missing names, would centre
    against something that was not observed.
    """
    n_names = 30
    dates = np.repeat(["d0", "d1"], n_names)
    y = np.arange(2 * n_names, dtype=float)
    # Blank out all but five names on d1.
    y[n_names + 5:] = np.nan

    keep, dt, dp, report = rank_demean_by_date(dates, y, y.copy(),
                                               min_cross_section=20)
    assert report["dates_dropped_small_cross_section"] == 1
    assert report["rows_dropped"] == 5
    assert keep[:n_names].all()
    assert not keep[n_names:].any()
    assert np.isnan(dt[n_names:]).all() and np.isnan(dp[n_names:]).all()


def test_the_minimum_cross_section_constant_is_the_declared_one():
    assert MIN_CROSS_SECTION == 20


# ── §3.2 the date-level circular block bootstrap ─────────────────────────────


def test_the_block_bootstrap_is_circular_and_reaches_the_series_ends():
    """
    A MOVING-block bootstrap can only start a block at 0..n-block, so the first
    and last observations appear in fewer blocks than the middle and are
    systematically under-sampled. Here the ends ARE the fold boundaries and the
    most recent fold is the one a reader cares about most, so the wrap is not a
    detail.
    """
    n, block = 50, 10
    rng = np.random.default_rng(0)
    idx = circular_block_indices(n, block, n_draws=2000, rng=rng)
    counts = np.bincount(idx, minlength=n)

    assert counts.min() > 0, "some positions are unreachable"
    # Uniform to within sampling noise: the ends must not be under-represented.
    assert counts[:3].mean() == pytest.approx(counts[n // 2 - 2:n // 2 + 1].mean(),
                                              rel=0.15)


def test_a_replicate_never_mixes_dates_across_folds():
    """
    Resampling across the whole sample would put a fold-0 date into fold 4 and
    destroy the walk-forward ordering the purge and embargo exist to protect.
    """
    data = synthetic_panel(n_dates=300, n_tickers=25, seed=1)
    panel = build_panel(data["dates"], data["tickers"], data["folds"],
                        data["y_true"], data["y_pred"], min_cross_section=10)
    rng = np.random.default_rng(0)
    sels = resample_selections(panel, BLOCK_LENGTH_SESSIONS, rng)

    assert len(sels) == len(panel.folds)
    for fm, sel in zip(panel.folds, sels):
        assert sel.max() < fm.dates.size, (
            "a replicate indexed outside its own fold's date grid")
        assert sel.size == fm.dates.size


def test_the_bootstrap_is_deterministic_under_its_seed():
    data = synthetic_panel(n_dates=200, n_tickers=20, seed=5)
    kw = dict(block=BLOCK_LENGTH_SESSIONS, n_bootstrap=40,
              min_cross_section=10, compute_auto_block=False)
    a = grade_panel_v3(data["dates"], data["tickers"], data["folds"],
                       data["y_true"], data["y_pred"], **kw)
    b = grade_panel_v3(data["dates"], data["tickers"], data["folds"],
                       data["y_true"], data["y_pred"], **kw)
    assert a.mu_hat == pytest.approx(b.mu_hat, abs=1e-15)
    assert a.mu_se_bootstrap == pytest.approx(b.mu_se_bootstrap, abs=1e-15)
    assert a.counts() == b.counts()


def test_the_between_fold_term_is_not_added_on_top_of_the_bootstrap():
    """
    THE DOUBLE-COUNTING GUARD.

    Stage 0b's sigma2 was max(within-fold bootstrap, between-fold var/K).
    Stage 0c's bootstrap already contains the between-fold component — it
    resamples dates and re-runs the whole pipeline — so adding the other term
    on top would inflate every variance and shrink every posterior toward the
    mean for a bookkeeping reason.

    Asserted through the reported source: where the bootstrap succeeded the row
    says "bootstrap", and the value IS the bootstrap spread rather than a
    maximum of two things.
    """
    data = synthetic_panel(n_dates=250, n_tickers=25, seed=11)
    g = grade_panel_v3(data["dates"], data["tickers"], data["folds"],
                       data["y_true"], data["y_pred"], n_bootstrap=60,
                       min_cross_section=10, compute_auto_block=False)

    from pipeline.evidence_panel import _fallback_sigma2, build_panel as _bp

    panel = _bp(data["dates"], data["tickers"], data["folds"],
                data["y_true"], data["y_pred"], min_cross_section=10)
    fold_ics = per_ticker_fold_ics(panel)

    bootstrapped = [r for r in g.rows if r.sigma2_source == "bootstrap"]
    assert bootstrapped, "no ticker took the bootstrap path; test is vacuous"

    disagreements = 0
    for r in bootstrapped:
        i = int(np.flatnonzero(panel.tickers == r.ticker)[0])
        fb = _fallback_sigma2(fold_ics[:, i])
        if np.isfinite(fb) and fb > r.sigma2:
            disagreements += 1
    assert disagreements > 0, (
        "every bootstrap sigma2 is at least the between-fold term, which is "
        "what taking a maximum would produce; the two are not being separated")


def test_the_low_fold_fallback_is_used_and_labelled_when_it_is():
    """The heuristic survives only for tickers the bootstrap cannot estimate,
    and the row says which path it took rather than leaving it inferable."""
    from pipeline.evidence_panel import _fallback_sigma2

    assert not np.isfinite(_fallback_sigma2(np.array([0.3])))
    assert not np.isfinite(_fallback_sigma2(np.array([np.nan, np.nan])))
    got = _fallback_sigma2(np.array([0.1, 0.3, 0.2]))
    assert got == pytest.approx(np.var([0.1, 0.3, 0.2], ddof=1) / 3)


# ── §3.5 REML, HKSJ, and the degeneracy detector ─────────────────────────────


def test_reml_recovers_a_planted_between_unit_variance():
    """
    REML against a case whose answer is known by construction: units drawn from
    a normal with variance tau2_true, each observed with its own known sampling
    variance. The estimate must land near tau2_true and must not be the
    negatively-biased DL figure.
    """
    rng = np.random.default_rng(3)
    k, tau2_true = 200, 0.04
    sigma2 = rng.uniform(0.005, 0.02, size=k)
    theta = rng.normal(0.1, np.sqrt(tau2_true), size=k)
    hat = theta + rng.normal(0, np.sqrt(sigma2))

    got = reml_tau2(hat, sigma2)
    assert got == pytest.approx(tau2_true, rel=0.25), (
        f"REML returned {got:.5f} against a planted {tau2_true}")


def test_reml_returns_exactly_zero_on_a_homogeneous_panel():
    """No between-unit variation at all: REML must land ON the boundary, not
    at a small positive number that would leave the posteriors spread out."""
    rng = np.random.default_rng(4)
    k = 120
    sigma2 = np.full(k, 0.01)
    hat = 0.05 + rng.normal(0, np.sqrt(sigma2))
    assert reml_tau2(hat, sigma2) == pytest.approx(0.0, abs=1e-6)


def test_reml_and_dersimonian_laird_are_reported_together_and_differ():
    """DL is negatively biased at small K (Langan et al. 2019 recommend REML).
    Both are computed on the same data so the gap is visible rather than a
    claim in a docstring."""
    from pipeline.evidence_shrinkage import dersimonian_laird_tau2

    rng = np.random.default_rng(9)
    k, tau2_true = 12, 0.05
    sigma2 = rng.uniform(0.01, 0.15, size=k)          # very unequal precisions
    hat = rng.normal(0.1, np.sqrt(tau2_true), size=k) + rng.normal(0, np.sqrt(sigma2))

    reml = reml_tau2(hat, sigma2)
    dl, _ = dersimonian_laird_tau2(hat, sigma2)
    assert reml >= 0 and dl >= 0
    assert reml != pytest.approx(dl, abs=1e-9), (
        "REML and DL returned the same number on unequal precisions at K = 12, "
        "where they are known to differ")


def test_hksj_widens_the_interval_when_units_disagree():
    """
    HKSJ rescales the standard error by the observed dispersion and refers it
    to t with K-1 df. On units that disagree more than their sampling errors
    explain it must be WIDER than the unadjusted normal interval, which is the
    whole reason IntHout et al. (2014) recommend it.
    """
    hat = np.array([0.30, -0.20, 0.45, -0.10, 0.35])
    sigma2 = np.full(5, 0.002)                        # tiny, so units disagree
    tau2 = reml_tau2(hat, sigma2)

    mu, se_hksj, lo, hi = hksj_interval(hat, sigma2, tau2)
    naive_se = float(np.sqrt(1.0 / np.sum(1.0 / (sigma2 + tau2))))
    naive_width = 2 * 1.959964 * naive_se

    assert (hi - lo) > naive_width, (
        f"HKSJ width {hi - lo:.5f} is no wider than the unadjusted "
        f"{naive_width:.5f} on units that plainly disagree")
    assert mu == pytest.approx(np.mean(hat), abs=1e-9)   # equal weights here


def test_the_tau2_detector_fires_at_the_boundary_and_on_low_dispersion():
    hat = np.array([0.05, 0.05, 0.05, 0.05])
    sigma2 = np.full(4, 0.01)

    fired = detect_tau2_degeneracy(0.0, hat, sigma2)
    assert fired.degenerate and "boundary" in fired.reason

    q, df = cochran_q(hat, sigma2)
    assert q == pytest.approx(0.0, abs=1e-12) and df == 3

    # Above the tolerance but with Q still under its df: the second condition.
    low_q = detect_tau2_degeneracy(1e-3, hat, sigma2)
    assert low_q.degenerate and "Cochran" in low_q.reason

    real = detect_tau2_degeneracy(0.05, np.array([0.4, -0.3, 0.5, -0.2]),
                                  np.full(4, 0.001))
    assert not real.degenerate


def test_a_homogeneous_panel_emits_ONE_statement_not_eighty_four():
    """
    THE FAILURE THIS DETECTOR EXISTS FOR, end to end.

    Stage 2b's `pooled_rank_ic` reported 84 identical ANTI_SIGNAL grades at
    tau2 = 0.0003. At the zero boundary every shrinkage weight is 1, so every
    posterior IS the grand mean and every grade is the same grade — one fact
    about the panel, printed once per company.
    """
    data = synthetic_panel(n_dates=250, n_tickers=40, ic=0.0, seed=21)
    g = grade_panel_v3(data["dates"], data["tickers"], data["folds"],
                       data["y_true"], data["y_pred"], n_bootstrap=60,
                       min_cross_section=10, compute_auto_block=False)

    assert g.tau2_reml <= 1e-2, "fixture is not homogeneous enough"
    if g.degeneracy.degenerate:
        head = g.headline()
        assert "ONE finding about the panel" in head or "NO ticker" in head
        assert str(len(g.rows)) in head
        distinct = {r.grade for r in g.rows}
        assert len(distinct) == 1, (
            f"tau2 is at the boundary but the rows carry {len(distinct)} "
            f"different grades, which the arithmetic cannot produce")


# ── §3.4 Romano-Wolf ─────────────────────────────────────────────────────────


def test_romano_wolf_finds_planted_units_and_rejects_the_nulls():
    """
    A synthetic case with known strong and null units. The strong ones must be
    rejected; the nulls must not be, and the adjusted p-values must be
    monotone in the statistic.
    """
    rng = np.random.default_rng(2)
    t, b = 50, 2000
    boot = rng.normal(size=(b, t))
    stat = np.zeros(t)
    stat[:3] = 6.0                       # unmistakably strong
    stat[3:] = rng.normal(scale=0.5, size=t - 3)

    rejected, adj = romano_wolf(stat, boot, alpha=0.10)
    assert rejected[:3].all(), "the planted units were not rejected"
    assert rejected[3:].sum() <= 2, (
        f"{int(rejected[3:].sum())} of {t - 3} null units were rejected at "
        f"alpha 0.10; familywise control is not holding")

    order = np.argsort(-stat)
    assert np.all(np.diff(adj[order]) >= -1e-12), (
        "adjusted p-values are not monotone in the statistic")


def test_romano_wolf_is_stricter_than_the_unadjusted_comparison():
    """The whole point: 84 simultaneous tests must not be graded at a nominal
    per-ticker threshold."""
    rng = np.random.default_rng(6)
    t, b = 84, 1500
    boot = rng.normal(size=(b, t))
    stat = rng.normal(size=t)
    stat[0] = 2.6                       # would clear a nominal 2.0 easily

    rejected, adj = romano_wolf(stat, boot, alpha=0.10)
    nominal = stat > 2.0
    assert nominal.sum() >= 1
    assert rejected.sum() <= nominal.sum(), (
        "Romano-Wolf rejected more than the unadjusted rule, which is "
        "impossible if the adjustment is doing anything")


def test_benjamini_yekutieli_is_the_conservative_bound():
    from pipeline.evidence_shrinkage import benjamini_hochberg

    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9])
    bh = benjamini_hochberg(p, q=0.10)
    by = benjamini_yekutieli(p, q=0.10)
    assert by.sum() <= bh.sum(), "BY rejected more than BH, which cannot happen"
    assert bh.sum() >= 1


# ── §3.3 Driscoll-Kraay ──────────────────────────────────────────────────────


def test_driscoll_kraay_widens_with_serial_correlation():
    """
    DK exists because the naive standard error assumes independence across
    dates. On a persistent series it must report MORE uncertainty than the
    i.i.d. formula, or it is not doing its job.
    """
    rng = np.random.default_rng(8)
    n = 600
    dates = np.array([f"d{i:04d}" for i in range(n)])
    iid = rng.normal(size=n)
    persistent = np.convolve(rng.normal(size=n + 30),
                             np.ones(30) / 30, mode="valid")[:n]

    _, se_iid, _ = driscoll_kraay_se(dates, iid)
    _, se_ac, lags = driscoll_kraay_se(dates, persistent)
    naive_ac = float(np.std(persistent, ddof=1) / np.sqrt(n))

    assert lags > 0
    assert se_ac > naive_ac, (
        f"DK SE {se_ac:.5f} is no larger than the i.i.d. {naive_ac:.5f} on a "
        f"strongly autocorrelated series")
    assert se_iid == pytest.approx(float(np.std(iid, ddof=1) / np.sqrt(n)),
                                   rel=0.5)


# ── the panel-level quantity is NOT the per-ticker one ───────────────────────


def test_the_two_quantities_are_computed_separately_and_can_disagree():
    """
    Per-ticker: "within this name, did the dates it ranked higher pay more".
    Per-date:   "on this date, did the names it ranked higher pay more".

    Stage 0b's closing argument was that the gate graded the first while the
    project was measured on the second. They must never be the same number by
    construction.
    """
    data = synthetic_panel(n_dates=300, n_tickers=30, ic=0.25, seed=13)
    panel = build_panel(data["dates"], data["tickers"], data["folds"],
                        data["y_true"], data["y_pred"], min_cross_section=10)

    per_ticker = np.nanmean(per_ticker_fold_ics(panel), axis=0)
    cs_dates, cs_ics = cross_sectional_ic_series(panel)

    assert cs_dates.size > 100 and np.isfinite(cs_ics).all()
    assert per_ticker.size == 30
    # Both should see the planted cross-sectional edge, and they are different
    # aggregations of it rather than the same array under two names.
    assert np.nanmean(per_ticker) > 0.05
    assert cs_ics.mean() > 0.05
    assert not np.allclose(np.nanmean(per_ticker), cs_ics.mean(), atol=1e-9)


def test_nan_rank_ic_ignores_missing_names_rather_than_dropping_the_date():
    """A ticker is present on 81-84 of 84 names on a given date. Filling the
    holes would invent an ordering; dropping the whole date would discard 80
    good names to accommodate one absence."""
    a = np.array([[1.0, 2.0, np.nan, 4.0, 5.0]])
    b = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert nan_rank_ic_rows(a, b)[0] == pytest.approx(1.0)

    flat = np.array([[7.0, 7.0, 7.0, 7.0, 7.0]])
    assert np.isnan(nan_rank_ic_rows(flat, b)[0])

    too_few = np.array([[1.0, np.nan, np.nan, np.nan, np.nan]])
    assert np.isnan(nan_rank_ic_rows(too_few, b)[0])


def test_it_agrees_with_scipy_on_a_complete_row():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(6, 40))
    b = rng.normal(size=(6, 40))
    mine = nan_rank_ic_rows(a, b)
    theirs = np.array([stats.spearmanr(a[i], b[i]).statistic for i in range(6)])
    assert np.max(np.abs(mine - theirs)) < 1e-12


# ── shadow discipline ────────────────────────────────────────────────────────


def test_arch_and_linearmodels_are_not_imported_by_loading_the_module():
    """
    They are grading-time dependencies, not serving-time ones. `requirements.txt`
    is installed by Render, which serves reads and never grades a panel — the
    rule torch's removal established. Both are imported lazily inside the two
    functions that need them.
    """
    import subprocess
    import sys

    code = ("import sys; import pipeline.evidence_panel; "
            "print('arch' in sys.modules, 'linearmodels' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", (
        f"importing evidence_panel pulled in a grading-only dependency: "
        f"{out.stdout.strip()}")
