"""
tools/stage0_degeneracy_sweep.py — Stage 0 ADDENDUM: how much of mu_hat survives
the partially-degenerate folds?

EVIDENCE-GRADING REDESIGN, STAGE 0 ADDENDUM. Separate track from the project's
Phase 0-6 roadmap. This is a DIAGNOSTIC that sits BESIDE Stage 0's committed
numbers, never in place of them: it recomputes nothing in `evidence_grades_v2`,
touches no pre-registered threshold, and writes no database row.

--------------------------------------------------------------------------------
THE GAP THIS FILLS
--------------------------------------------------------------------------------

Stage 0 measured that 316 of 420 (ticker, fold) cells emit a CONSTANT
prediction, and that 21 tickers are constant in all five of their folds. The
grader's v2 guard refuses exactly those 21, because a ticker with no ordering in
any fold has a pooled rank IC that is entirely the arrangement of its
fold-level constants against the realised period returns.

That leaves **211 degenerate cells spread across the other 63 tickers**
(316 - 21*5). Those tickers ARE graded, and they still feed `mu_hat = -0.05988`.
The guard is binary and catches only the total case; this asks how much of the
reported number survives once partial degeneracy is accounted for as well.

--------------------------------------------------------------------------------
A SWEEP, NOT A NUMBER
--------------------------------------------------------------------------------

The deliverable is a threshold sweep and the SHAPE of it. Picking one cutoff and
reporting one adjusted mu_hat would be the "result at one cell" error this
project has already paid for twice — valuation at t +3.32 measured at
`min_train=380` and +1.00 at 500 on identical rows, and the P5 regime split that
produced t +5.21 between neighbours of +1.78 and +1.21. So every tau is
reported, and the reading is whether the trend is smooth or abrupt.

TWO INDEPENDENT DEGENERACY METRICS, checked for agreement rather than assumed:

  mode_share       fraction of a fold's held-out predictions equal to that
                   fold's single most common predicted value. 1.00 = constant.
  unique_fraction  distinct predicted values as a fraction of the fold's row
                   count. Near 0 = degenerate. Scale-free across fold sizes.

They can disagree: a fold that is 95% one value plus 5% all-distinct values
scores mode_share 0.95 (very degenerate) and unique_fraction 0.05 (also
degenerate) — but a fold split evenly between two values scores mode_share 0.50
(mild) and unique_fraction ~0.001 (extreme). The report measures how often they
classify the same cells and says so.

EXCLUSION IS BY WHOLE FOLD, AND THE BOOTSTRAP IS TOLD ABOUT THE SEAM. Dropping
fold 2 of 5 leaves rows either side that are months apart. `block_bootstrap_ic`
is called with `respect_fold_gaps=True` so no resampled block spans that join —
otherwise the sweep would manufacture continuity that is not in the data, which
is the same class of error as the too-short block the main run swept for.

THE EFFECTIVE TICKER COUNT IS REPORTED AT EVERY TAU, AND IT IS THE CAVEAT. As
tau falls, tickers lose every fold and drop out of the panel entirely — which
shrinks the very thing that makes partial pooling meaningful. A mu_hat over 20
surviving tickers is not evidence of the same weight as one over 63.
`TRUST_MIN_TICKERS` is the declared cutoff and cells below it are flagged
UNTRUSTED in the output rather than left to be read as equals.

Usage
-----
    python tools/stage0_degeneracy_sweep.py                    # the sweep
    python tools/stage0_degeneracy_sweep.py --block-spot-check # + one at 60
    python tools/stage0_degeneracy_sweep.py --markdown out.md
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.evaluation import rank_ic                            # noqa: E402
from pipeline.evidence_shrinkage import (                          # noqa: E402
    BLOCK_LENGTH_SESSIONS, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, FDR_Q, GRADES,
    TickerTrack, assign_grade, benjamini_hochberg, block_bootstrap_ic,
    dersimonian_laird_tau2, posterior_probability_positive,
    precision_weighted_mean, shrink,
)

#: Stage 0's measured break-even rank IC (spread_per_ic 0.3474, turnover 0.80,
#: zero impact). Pinned rather than re-read from the database so the addendum
#: is offline and its grade counts are comparable to Stage 0's to the digit.
BREAK_EVEN_IC = 0.00512363994209475

#: Stage 0's committed headline, reproduced at tau = 1.00 as a sanity check.
#: If the first row of the sweep does not match this to the last digit, the
#: reimplementation is wrong and nothing below it can be believed.
STAGE0_MU_HAT = -0.05988188592526484
STAGE0_TAU2 = 0.0022093500537861445
STAGE0_N_USABLE = 63

#: The exclusion grid. A fold is dropped when its mode-share EXCEEDS tau, so
#: tau = 1.00 drops nothing (mode-share is a fraction and cannot exceed 1).
TAU_GRID = (1.00, 0.90, 0.75, 0.50, 0.30, 0.10)

#: Below this many tickers feeding mu_hat, the cell is flagged UNTRUSTED.
#: Half the frozen 84-name universe. Partial pooling borrows strength from the
#: panel; once most of the panel is gone there is little left to borrow, the
#: survivors are a set SELECTED on the very property being swept, and the
#: precision-weighted mean starts describing that selection instead of the
#: universe. The number is declared here rather than chosen after the fact.
TRUST_MIN_TICKERS = 42


# ── Degeneracy scoring ────────────────────────────────────────────────────────


def mode_share(values: np.ndarray) -> float:
    """
    Fraction of ``values`` equal to the single most common value.

    1.0 means the fold emitted one repeated number and holds no ordering at
    all. Computed on exact float equality, which is the right test here: a tree
    that made no splits returns the SAME float for every row, bit for bit, not
    a cluster of nearby ones.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max() / values.size)


def unique_fraction(values: np.ndarray) -> float:
    """
    Distinct predicted values as a fraction of the row count.

    Scale-free, so folds of different lengths are comparable — which is why the
    raw distinct COUNT is not used: fold 0 and fold 4 hold different numbers of
    rows and a count would rank the longer fold as richer for that reason alone.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(len(np.unique(values)) / values.size)


def cell_scores(tracks, folds: dict) -> pd.DataFrame:
    """One row per (ticker, fold), with both degeneracy metrics."""
    rows = []
    for track in tracks:
        f = np.asarray(folds[track.ticker])
        for k in sorted(set(f.tolist())):
            mask = f == k
            rows.append({
                "ticker": track.ticker,
                "fold": int(k),
                "n_rows": int(mask.sum()),
                "mode_share": mode_share(track.y_pred[mask]),
                "unique_fraction": unique_fraction(track.y_pred[mask]),
                "pred_level": float(np.mean(track.y_pred[mask])),
                "realised": float(np.mean(track.y_true[mask])),
                "target_sd": float(np.std(track.y_true[mask])),
            })
    return pd.DataFrame(rows)


def metric_agreement(scores: pd.DataFrame, tau: float) -> dict:
    """
    Do the two metrics flag the SAME cells?

    `mode_share > tau` selects some number of cells. The comparison takes the
    same number of most-degenerate cells by `unique_fraction` (lowest first) and
    reports the overlap. Matching the COUNT rather than picking a second
    arbitrary threshold is what makes this an agreement test rather than two
    unrelated cutoffs compared.
    """
    flagged = scores["mode_share"] > tau
    n = int(flagged.sum())
    if n == 0:
        return {"n_flagged": 0, "overlap": float("nan")}
    by_unique = set(scores["unique_fraction"].nsmallest(n, keep="all").index[:n])
    overlap = len(set(scores.index[flagged]) & by_unique) / n
    return {"n_flagged": n, "overlap": float(overlap)}


def _grade_counts(hat: np.ndarray, s2: np.ndarray, mu: float, mu_var: float,
                  tau2: float, break_even: float = BREAK_EVEN_IC) -> dict:
    """
    What Stage 0's own decision table would grade at this tau.

    Reuses `shrink`, `posterior_probability_positive`, `benjamini_hochberg` and
    `assign_grade` unchanged - the addendum changes WHICH ROWS feed the
    calculation, never the calculation. Included because the sweep's real
    question is not "does mu_hat move" but "does anything become gradeable",
    and a mu_hat that turns positive while the grade count stays at zero is a
    materially different result from one that does not.
    """
    thetas, pvars, p_pos = [], [], []
    for h, v in zip(hat, s2):
        theta, _b, pv, _n = shrink(h, v, mu, mu_var, tau2)
        thetas.append(theta)
        pvars.append(pv)
        p_pos.append(posterior_probability_positive(theta, pv))

    p_two = np.array([2.0 * min(p, 1.0 - p) for p in p_pos])
    sig = benjamini_hochberg(p_two, FDR_Q)

    counts = {g: 0 for g in GRADES}
    for theta, p, s in zip(thetas, p_pos, sig):
        counts[assign_grade(bool(s), theta, p, break_even)[0]] += 1
    return counts


# ── The sweep ─────────────────────────────────────────────────────────────────


def restrict(track: TickerTrack, folds: np.ndarray,
             keep: set[int]) -> TickerTrack | None:
    """
    The same ticker with the excluded folds removed.

    The surviving rows keep their ORIGINAL fold ids, which is what lets
    `contiguous_segments` find the seam a dropped fold left behind. Renumbering
    them 0..n would erase exactly the information the bootstrap needs.
    """
    mask = np.isin(folds, list(keep))
    if not mask.any():
        return None
    return TickerTrack(
        ticker=track.ticker,
        dates=tuple(np.asarray(track.dates)[mask].tolist()),
        y_true=track.y_true[mask],
        y_pred=track.y_pred[mask],
        folds=folds[mask],
    )


def sweep(tracks, folds: dict, scores: pd.DataFrame,
          taus=TAU_GRID, block: int = BLOCK_LENGTH_SESSIONS,
          n_resamples: int = BOOTSTRAP_N_RESAMPLES,
          seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Recompute mu_hat and tau2_hat at each exclusion threshold."""
    by_ticker = {t.ticker: t for t in tracks}
    rows = []

    for tau in taus:
        kept_ic, kept_s2, per_ticker = [], [], []
        n_folds_dropped = n_rows_dropped = 0
        n_no_folds_left = n_refused = 0

        for ticker, group in scores.groupby("ticker", sort=False):
            track = by_ticker[ticker]
            f = np.asarray(folds[ticker])
            keep = set(group.loc[group["mode_share"] <= tau, "fold"].tolist())
            dropped = group.loc[group["mode_share"] > tau]
            n_folds_dropped += len(dropped)
            n_rows_dropped += int(dropped["n_rows"].sum())

            restricted = restrict(track, f, keep)
            if restricted is None:
                # EVERY fold excluded. Not imputed, not silently dropped: the
                # ticker has no estimate at this tau and is counted as such.
                n_no_folds_left += 1
                per_ticker.append({"tau": tau, "ticker": ticker,
                                   "status": "no_folds_left", "hat_ic": np.nan,
                                   "sigma2": np.nan, "n_folds_kept": 0})
                continue

            est = block_bootstrap_ic(restricted, block=block,
                                     n_resamples=n_resamples, seed=seed,
                                     respect_fold_gaps=True)
            if not est.usable:
                n_refused += 1
                per_ticker.append({"tau": tau, "ticker": ticker,
                                   "status": "refused", "hat_ic": est.hat_ic,
                                   "sigma2": np.nan,
                                   "n_folds_kept": len(keep)})
                continue

            kept_ic.append(est.hat_ic)
            kept_s2.append(est.sigma2)
            per_ticker.append({"tau": tau, "ticker": ticker, "status": "ok",
                               "hat_ic": est.hat_ic, "sigma2": est.sigma2,
                               "n_folds_kept": len(keep)})

        if len(kept_ic) >= 2:
            hat = np.array(kept_ic)
            s2 = np.array(kept_s2)
            mu, mu_var = precision_weighted_mean(hat, s2)
            tau2, q = dersimonian_laird_tau2(hat, s2)
            unweighted = float(hat.mean())
            grades = _grade_counts(hat, s2, mu, mu_var, tau2)
        else:
            mu = mu_var = tau2 = q = unweighted = float("nan")
            grades = {g: 0 for g in GRADES}

        # HOW MANY FOLDS ARE LEFT IS HOW MUCH FOLD LEVEL IS LEFT. A ticker down
        # to ONE surviving fold has no cross-fold level at all, so its pooled IC
        # IS its within-fold IC and the artifact is gone by construction. At two
        # or more, one level contrast survives per extra fold. This column is
        # what says whether a positive mu_hat further down the sweep is a
        # cleaner measurement or simply a shorter one.
        kept_counts = [r["n_folds_kept"] for r in per_ticker
                       if r["tau"] == tau and r["status"] == "ok"]

        rows.append({
            "tau": tau,
            "folds_dropped": n_folds_dropped,
            "rows_dropped": n_rows_dropped,
            "tickers_no_folds_left": n_no_folds_left,
            "tickers_refused": n_refused,
            "n_effective": len(kept_ic),
            "mu_hat": mu,
            "mu_sd": float(np.sqrt(mu_var)) if np.isfinite(mu_var) else np.nan,
            "tau2_hat": tau2,
            "q_statistic": q,
            "mean_hat_ic_unweighted": unweighted,
            "mu_z": float(mu / np.sqrt(mu_var)) if np.isfinite(mu_var) and mu_var > 0
                    else float("nan"),
            "mean_folds_kept": float(np.mean(kept_counts)) if kept_counts else np.nan,
            "STRONG": grades["STRONG"], "WEAK": grades["WEAK"],
            "ANTI_SIGNAL": grades["ANTI_SIGNAL"],
            "trust": "ok" if len(kept_ic) >= TRUST_MIN_TICKERS else "UNTRUSTED",
        })

    return pd.DataFrame(rows), pd.DataFrame(per_ticker)


# ── Reconciliation with the -0.070 -> +0.126 figure ───────────────────────────


def wholesale_within_fold(tracks, folds: dict) -> dict:
    """
    The figure Stage 0 already reported, recomputed here so the two are
    side by side rather than quoted from a different script.

    IT IS A DIFFERENT OPERATION FROM THE SWEEP and that is the point of showing
    both. This scores each fold SEPARATELY and averages the per-fold rank ICs,
    so the fold level is removed from every ticker at once and a fold with no
    ordering simply contributes nothing. The sweep instead DROPS degenerate
    folds and re-pools the survivors, which keeps the fold level inside whatever
    survives. The first is an upper bound on what removing the level can do; the
    second is what a defensible exclusion rule actually buys.
    """
    pooled, within = [], []
    for track in tracks:
        f = np.asarray(folds[track.ticker])
        pooled.append(rank_ic(track.y_true, track.y_pred))
        ics = [rank_ic(track.y_true[f == k], track.y_pred[f == k])
               for k in sorted(set(f.tolist()))]
        within.append(float(np.nanmean(ics)) if np.any(np.isfinite(ics))
                      else np.nan)
    pooled, within = np.array(pooled), np.array(within)
    return {
        "pooled_mean": float(np.nanmean(pooled)),
        "within_mean": float(np.nanmean(within)),
        "n_pooled": int(np.isfinite(pooled).sum()),
        "n_within": int(np.isfinite(within).sum()),
    }


def fold_level_significance(tracks, folds: dict) -> pd.DataFrame:
    """
    THE WITHIN-FOLD IC, TESTED AT THE RIGHT ALTITUDE.

    The sweep's `mu_hat` is a precision-weighted mean over tickers, and its
    standard error treats those tickers as INDEPENDENT. They are not: every
    name in a fold trades the same market over the same months, so a common
    factor moves the whole cross-section together. The panel holds five
    periods, not 52 independent observations, and BOTH the -4.87 z at
    tau = 1.00 and the +2.63 at tau = 0.75 are overstated for that reason.

    This is the same family of error as counting 1,900 overlapping dates as
    1,900 observations - the one `effective_sample_size` exists to correct and
    the one `rebalance_ic_t` was added to the baseline table to avoid.

    So: score each NON-DEGENERATE (ticker, fold) cell within its own fold, take
    the mean per fold, and treat the FIVE FOLD MEANS as the units. That is a
    small sample and a blunt test, which is the honest shape of the evidence
    rather than a flattering one.
    """
    from scipy import stats as _st

    per_fold = {}
    for track in tracks:
        f = np.asarray(folds[track.ticker])
        for k in sorted(set(f.tolist())):
            mask = f == k
            if np.ptp(track.y_pred[mask]) == 0:
                continue
            ic = rank_ic(track.y_true[mask], track.y_pred[mask])
            if np.isfinite(ic):
                per_fold.setdefault(int(k), []).append(ic)

    rows = []
    for k in sorted(per_fold):
        v = np.array(per_fold[k])
        rows.append({"fold": k, "cells": len(v), "mean_ic": float(v.mean()),
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                     "positive": int((v > 0).sum())})
    table = pd.DataFrame(rows)

    means = table["mean_ic"].to_numpy()
    t = float(means.mean() / (means.std(ddof=1) / np.sqrt(len(means))))
    table.attrs["fold_mean"] = float(means.mean())
    table.attrs["fold_sd"] = float(means.std(ddof=1))
    table.attrs["t"] = t
    table.attrs["p"] = float(2 * (1 - _st.t.cdf(abs(t), len(means) - 1)))
    table.attrs["all_same_sign"] = bool(np.all(means > 0) or np.all(means < 0))
    return table


# ── Root-cause context (secondary, explicitly not a finding) ──────────────────


def degeneracy_context(scores: pd.DataFrame, tracks, folds: dict) -> pd.DataFrame:
    """Where the degenerate cells sit. Context for Stage 1 vs Stage 2, not a fix."""
    scores = scores.copy()
    scores["degenerate"] = scores["mode_share"] == 1.0
    by_fold = scores.groupby("fold").agg(
        cells=("degenerate", "size"),
        degenerate=("degenerate", "sum"),
        mean_mode_share=("mode_share", "mean"),
        mean_rows=("n_rows", "mean"),
        mean_target_sd=("target_sd", "mean"),
    )
    by_fold["pct_degenerate"] = 100 * by_fold["degenerate"] / by_fold["cells"]
    return by_fold


def constant_equals_prior_mean(tracks, folds: dict) -> dict:
    """
    IS THE CONSTANT THE TRAINING MEAN? — a falsifiable root-cause probe.

    A regression tree that finds no split worth making returns the mean of its
    training target for every row. If the degenerate folds' constants sit close
    to the mean of the target over the period preceding them, the mechanism is
    "Optuna selected a no-split configuration", not "the model broke".

    That would be a coherent story rather than a mystery: `pipeline.tuning`
    searches `gamma` over [0, 5] and `min_child_weight` over [5, 40], and
    scores candidates on MEAN ABSOLUTE ERROR of a 30-session return whose
    dispersion is ~0.10. The total loss reduction any split can offer on that
    target is tiny, so a gamma anywhere above ~0 makes every split unprofitable
    — and MAE positively REWARDS the resulting constant, because on a
    near-random-walk target the training mean is a strong MAE predictor. The
    search would then be selecting no-split configurations on purpose.

    Which lands on a structural mismatch worth stating plainly: the tuner
    optimises MAE, under which a constant is close to optimal, while the gate
    grades RANK IC, which a constant cannot express at all.

    Uses earlier out-of-sample folds as the proxy for the training window,
    since the training rows themselves are not in the cache. Fold 0 has no
    earlier fold and is skipped.
    """
    diffs, rel = [], []
    for track in tracks:
        f = np.asarray(folds[track.ticker])
        keys = sorted(set(f.tolist()))
        for k in keys[1:]:
            mask = f == k
            if np.ptp(track.y_pred[mask]) != 0:
                continue
            prior = track.y_true[f < k]
            if prior.size < 100:
                continue
            constant = float(track.y_pred[mask][0])
            prior_mean = float(prior.mean())
            diffs.append(abs(constant - prior_mean))
            rel.append(abs(constant - prior_mean) / max(float(prior.std()), 1e-9))
    return {
        "n_cells": len(diffs),
        "median_abs_diff": float(np.median(diffs)) if diffs else float("nan"),
        "median_diff_in_sd": float(np.median(rel)) if rel else float("nan"),
        "pct_within_quarter_sd": (100 * float(np.mean(np.array(rel) < 0.25))
                                  if rel else float("nan")),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────


def render(table: pd.DataFrame, scores: pd.DataFrame, wholesale: dict,
           fold_sig: pd.DataFrame, context: pd.DataFrame, probe: dict,
           block: int) -> str:
    out = []
    w = out.append

    w("=" * 78)
    w("STAGE 0 ADDENDUM - DEGENERACY SENSITIVITY SWEEP")
    w("=" * 78)
    w(f"  block length            {block} sessions (Stage 0's reported run)")
    w(f"  (ticker, fold) cells    {len(scores)}")
    w(f"  fully constant cells    {int((scores['mode_share'] == 1.0).sum())}")
    w(f"  trust cutoff            n_effective >= {TRUST_MIN_TICKERS} "
      f"(half the frozen universe)")

    w("")
    w("=" * 78)
    w("1. DO THE TWO DEGENERACY METRICS AGREE?")
    w("=" * 78)
    rho = scores[["mode_share", "unique_fraction"]].corr(method="spearman")
    w(f"  Spearman(mode_share, unique_fraction) over all cells: "
      f"{rho.iloc[0, 1]:+.4f}")
    w(f"  {'tau':>7}{'cells flagged':>16}{'same cells by unique_fraction':>32}")
    for tau in TAU_GRID:
        a = metric_agreement(scores, tau)
        pct = "n/a" if not np.isfinite(a["overlap"]) else f"{100*a['overlap']:.1f}%"
        w(f"  {tau:>7.2f}{a['n_flagged']:>16}{pct:>32}")

    w("")
    w("=" * 78)
    w("2. THE SWEEP")
    w("=" * 78)
    w(f"  {'tau':>6}{'folds':>7}{'gone':>6}{'refus':>7}{'n_eff':>7}"
      f"{'folds/tk':>10}{'mu_hat':>11}{'sd':>9}{'z':>7}{'tau2_hat':>10}"
      f"{'STR':>5}{'WEAK':>6}{'ANTI':>6}  trust")
    for r in table.itertuples(index=False):
        w(f"  {r.tau:>6.2f}{r.folds_dropped:>7}"
          f"{r.tickers_no_folds_left:>6}{r.tickers_refused:>7}"
          f"{r.n_effective:>7}{r.mean_folds_kept:>10.2f}{r.mu_hat:>+11.5f}"
          f"{r.mu_sd:>9.5f}{r.mu_z:>+7.2f}{r.tau2_hat:>10.6f}"
          f"{r.STRONG:>5}{r.WEAK:>6}{r.ANTI_SIGNAL:>6}  {r.trust}")
    w("")
    w("  folds = folds dropped   folds/tk = mean surviving folds per graded")
    w("          ticker (1.00 means NO cross-fold level is left at all)")
    w("  STR/WEAK/ANTI = what Stage 0's own decision table grades at this tau")
    w("  gone  = tickers left with ZERO surviving folds (excluded from mu_hat,")
    w("          counted here, never imputed)")
    w("  refus = tickers the grader still refused (constant in every SURVIVING")
    w("          fold, or too few surviving rows to bootstrap)")

    first = table.iloc[0]
    ok = (abs(first["mu_hat"] - STAGE0_MU_HAT) < 1e-15
          and abs(first["tau2_hat"] - STAGE0_TAU2) < 1e-15
          and int(first["n_effective"]) == STAGE0_N_USABLE)
    w("")
    w(f"  SANITY CHECK, tau = 1.00 excludes nothing and must reproduce Stage 0:")
    w(f"    mu_hat   {first['mu_hat']!r}")
    w(f"    Stage 0  {STAGE0_MU_HAT!r}")
    w(f"    tau2_hat {first['tau2_hat']!r}  vs  {STAGE0_TAU2!r}")
    w(f"    n_eff    {int(first['n_effective'])}  vs  {STAGE0_N_USABLE}")
    w(f"    -> {'EXACT MATCH' if ok else '*** MISMATCH - STOP AND DEBUG ***'}")

    w("")
    w("=" * 78)
    w("3. RECONCILIATION WITH THE -0.070 -> +0.126 FIGURE")
    w("=" * 78)
    w(f"  wholesale within-fold scoring (Stage 0's operation, recomputed here):")
    w(f"    per-ticker IC pooled over folds   {wholesale['pooled_mean']:+.4f}"
      f"   over {wholesale['n_pooled']} tickers")
    w(f"    per-ticker IC averaged within     {wholesale['within_mean']:+.4f}"
      f"   over {wholesale['n_within']} tickers")
    w("")
    w("  the sweep's unweighted mean hat_ic, for comparison on the same scale:")
    for r in table.itertuples(index=False):
        w(f"    tau {r.tau:.2f}   {r.mean_hat_ic_unweighted:+.4f}"
          f"   (n_eff {r.n_effective}, {r.trust})")

    w("")
    w("=" * 78)
    w("4. THE WITHIN-FOLD IC AT THE FOLD LEVEL - the right altitude")
    w("=" * 78)
    w("  Non-degenerate cells only, each scored inside its own fold, so the")
    w("  fold-level artifact is removed by construction rather than swept for.")
    w(f"  {'fold':>6}{'cells':>8}{'mean IC':>11}{'sd':>9}{'positive':>11}")
    for r in fold_sig.itertuples(index=False):
        w(f"  {r.fold:>6}{r.cells:>8}{r.mean_ic:>+11.4f}{r.sd:>9.4f}"
          f"{f'{r.positive}/{r.cells}':>11}")
    a = fold_sig.attrs
    w("")
    w(f"  five fold means as the units:  mean {a['fold_mean']:+.4f}"
      f"   sd {a['fold_sd']:.4f}   t {a['t']:+.2f}   p {a['p']:.4f}")
    w(f"  all five carry the same sign:  {a['all_same_sign']}"
      f"   (P = 1/32 = 0.031 under a symmetric null)")
    w("")
    w("  READ THE CAVEATS BEFORE READING THE NUMBER. These 104 cells are the")
    w("  SURVIVORS of a 420-cell panel, selected on the model having made any")
    w("  split at all - and whether that selection is independent of whether")
    w("  the model was RIGHT is untested. It is also a per-ticker TIME-SERIES")
    w("  IC, a different quantity from the cross-sectional rebalance IC every")
    w("  other null in this project is stated in, and one a common market")
    w("  factor moves for all names at once.")

    w("")
    w("=" * 78)
    w("5. CONTEXT: WHERE THE DEGENERATE CELLS SIT (not a finding, not a fix)")
    w("=" * 78)
    w(f"  {'fold':>6}{'cells':>8}{'constant':>10}{'%':>8}{'mean rows':>11}"
      f"{'mean mode_share':>18}{'mean target sd':>17}")
    for fold, r in context.iterrows():
        w(f"  {fold:>6}{int(r['cells']):>8}{int(r['degenerate']):>10}"
          f"{r['pct_degenerate']:>8.1f}{r['mean_rows']:>11.0f}"
          f"{r['mean_mode_share']:>18.4f}{r['mean_target_sd']:>17.5f}")
    w("")
    w("  IS THE CONSTANT THE TRAINING MEAN? (proxied by earlier OOS folds)")
    w(f"    degenerate cells testable        {probe['n_cells']}")
    w(f"    median |constant - prior mean|   {probe['median_abs_diff']:.5f}")
    w(f"    ... in units of the prior sd     {probe['median_diff_in_sd']:.3f}")
    w(f"    within a quarter of an sd        {probe['pct_within_quarter_sd']:.1f}%")

    return "\n".join(out)


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("evidence_oos.npz"))
    ap.add_argument("--block", type=int, default=BLOCK_LENGTH_SESSIONS)
    ap.add_argument("--block-spot-check", action="store_true",
                    help="repeat the sweep at one other block length; a "
                         "robustness spot-check, not the deliverable")
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    from tools.run_evidence_grading import load_cache
    cache = load_cache(args.cache)
    tracks, folds = cache["tracks"], cache["folds"]
    print(f"[Addendum] {len(tracks)} tickers from {args.cache}: {cache['meta']}")

    scores = cell_scores(tracks, folds)
    table, per_ticker = sweep(tracks, folds, scores, block=args.block)
    wholesale = wholesale_within_fold(tracks, folds)
    context = degeneracy_context(scores, tracks, folds)
    probe = constant_equals_prior_mean(tracks, folds)

    fold_sig = fold_level_significance(tracks, folds)
    report = render(table, scores, wholesale, fold_sig, context, probe,
                    args.block)
    print(report)

    if args.block_spot_check:
        other = 60 if args.block != 60 else 45
        spot, _ = sweep(tracks, folds, scores, block=other)
        print()
        print("=" * 78)
        print(f"6. ROBUSTNESS SPOT-CHECK AT BLOCK {other} (not the deliverable)")
        print("=" * 78)
        print(f"  {'tau':>6}{'n_eff':>7}{'mu_hat':>11}{'tau2_hat':>11}  trust")
        for r in spot.itertuples(index=False):
            print(f"  {r.tau:>6.2f}{r.n_effective:>7}{r.mu_hat:>+11.5f}"
                  f"{r.tau2_hat:>11.6f}  {r.trust}")

    if args.csv:
        per_ticker.to_csv(args.csv, index=False)
        print(f"\n  per-(tau, ticker) rows -> {args.csv}")
    if args.markdown:
        args.markdown.write_text(report, encoding="utf-8")
        print(f"  report -> {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
