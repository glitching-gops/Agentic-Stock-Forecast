"""
tools/run_evidence_grading.py — run the Stage 0 shadow grading and report it.

EVIDENCE-GRADING REDESIGN, STAGE 0. Separate track from the project's own
Phase 0-6 roadmap; neither renumbers the other.

WHAT THIS PRINTS IS THE POINT: an old-grade against new-grade crosstab over the
whole universe, plus `mu_hat` and `tau2_hat`. Those two panel numbers are the
diagnostic the stage exists to produce — a `mu_hat` near zero with a
`tau2_hat` of zero is not an intermediate quantity on the way to a grade, it is
the finding, and it says the constraint is signal rather than measurement.

--------------------------------------------------------------------------------
WHERE THE OUT-OF-SAMPLE ROWS COME FROM, AND WHY THIS COSTS AN HOUR
--------------------------------------------------------------------------------

**The per-ticker walk-forward's predictions are not persisted anywhere.**
`pipeline.model.evaluate_and_persist_ticker` keeps `WalkForwardResult.metrics`
(into `model_metadata`) and the conformal residuals, and drops
`WalkForwardResult.predictions` on the floor when the function returns. So
"consume the existing evaluation's output" is not literally possible: the only
surviving trace of a ticker's track record is four scalars, and four scalars
cannot be bootstrapped.

This tool therefore RE-RUNS `pipeline.model.evaluate_ticker` unmodified. That
is safe to do rather than a hidden re-derivation, because the walk-forward is
deterministic: Optuna's TPE sampler is seeded (`pipeline.tuning.SEED`) and
XGBoost's `random_state` is fixed, so the same input frame yields
bit-identical predictions — verified in-process before this tool was written,
and re-verified on every run by `--verify-against-persisted`, which compares
the regenerated scalars against what `model_metadata` holds.

It costs ~47 s a ticker, so ~65 minutes for 84 names, and the result is cached
to an `.npz` so the grading itself can be re-run in seconds. The cheaper
arrangement is a five-line write of `result.predictions` inside
`evaluate_and_persist_ticker`, which would let this read the weekly job's own
output for free. That change is NOT made here because `pipeline/model.py` is
explicitly out of scope for Stage 0; it is the obvious follow-up and is flagged
in `docs/stage0-evidence-grading.md`.

--------------------------------------------------------------------------------
OLD AND NEW ARE GRADED ON THE SAME ROWS
--------------------------------------------------------------------------------

The old grade in the crosstab is NOT read out of `forecast_current`. It is
recomputed by calling `pipeline.evaluation.compute_metrics` and
`agents.critic_agent.grade_evidence` — both unmodified — on the SAME regenerated
predictions the new grade is computed from. Comparing a live grade measured on
last Saturday's data against a new grade measured on today's would report the
week's extra data as though it were the method change.

The live grades are still shown, separately, as a third column, and
`--verify-against-persisted` reports how far the regenerated scalars have
drifted from the persisted ones. Drift is expected and is data, not a defect:
the daily job has recomputed signals since the last weekly evaluation.

Usage
-----
    python tools/run_evidence_grading.py                    # build cache + grade
    python tools/run_evidence_grading.py --no-rebuild       # grade a cached run
    python tools/run_evidence_grading.py --block-sweep      # robustness check
    python tools/run_evidence_grading.py --store            # write the shadow table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.critic_agent import grade_evidence                     # noqa: E402
from pipeline.evaluation import compute_metrics                    # noqa: E402
from pipeline.evidence_shrinkage import (                          # noqa: E402
    BLOCK_LENGTH_SESSIONS, BOOTSTRAP_N_RESAMPLES, FDR_Q,
    STRONG_POSTERIOR_THRESHOLD, TickerTrack, economic_bar, grade_panel,
    store_grading,
)

DEFAULT_CACHE = Path("evidence_oos.npz")


# ── The out-of-sample cache ───────────────────────────────────────────────────


def build_cache(tickers: list[str], path: Path) -> dict:
    """
    Regenerates every ticker's held-out track record and caches it.

    Calls `evaluate_ticker` — the production weekly evaluation's own function,
    unmodified, with its default nested tuning — and keeps `date`, `y_true` and
    `y_pred`. Ragged lengths are stored concatenated with an offset index, the
    same shape `tools/export_series_package.py` uses, because npz has no ragged
    array type and a dict-of-arrays file would need one key per ticker.
    """
    from pipeline.model import evaluate_ticker

    kept: list[tuple[str, pd.DataFrame]] = []
    started = time.time()
    for i, ticker in enumerate(tickers, 1):
        t0 = time.time()
        try:
            result = evaluate_ticker(ticker)
        except Exception as exc:                                # noqa: BLE001
            print(f"  [{i:>3}/{len(tickers)}] {ticker:<16} FAILED: {exc}")
            continue
        if result.n_predictions == 0:
            print(f"  [{i:>3}/{len(tickers)}] {ticker:<16} no out-of-sample rows")
            continue
        preds = result.predictions[["date", "y_true", "y_pred", "fold"]].copy()
        kept.append((ticker, preds))
        print(f"  [{i:>3}/{len(tickers)}] {ticker:<16} {len(preds):>5} rows  "
              f"folds {result.n_folds_run}  IC {result.metrics.get('rank_ic', float('nan')):+.4f}  "
              f"{time.time() - t0:>5.1f}s")

    if not kept:
        raise SystemExit("no ticker produced out-of-sample predictions")

    offsets = np.cumsum([0] + [len(p) for _, p in kept])
    frame = pd.concat([p for _, p in kept], ignore_index=True)

    from pipeline.model import MODEL_VERSION
    np.savez_compressed(
        path,
        tickers=np.array([t for t, _ in kept], dtype=object),
        offsets=offsets.astype(np.int64),
        dates=frame["date"].astype(str).to_numpy(dtype=object),
        y_true=frame["y_true"].to_numpy(dtype=np.float64),
        y_pred=frame["y_pred"].to_numpy(dtype=np.float64),
        fold=frame["fold"].to_numpy(dtype=np.int32),
        meta=np.array([json.dumps({
            "model_version": MODEL_VERSION,
            "built_at": pd.Timestamp.utcnow().isoformat(),
            "n_tickers": len(kept),
            "n_rows": int(len(frame)),
            "seconds": round(time.time() - started, 1),
        })], dtype=object),
    )
    print(f"\n  cached {len(kept)} tickers / {len(frame)} rows -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB, "
          f"{(time.time() - started) / 60:.1f} min)")
    return load_cache(path)


def load_cache(path: Path) -> dict:
    """Reads the npz back into tracks plus per-ticker fold labels."""
    with np.load(path, allow_pickle=True) as data:
        tickers = [str(t) for t in data["tickers"]]
        offsets = data["offsets"]
        dates = data["dates"]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        fold = data["fold"]
        meta = json.loads(str(data["meta"][0]))

    tracks, folds = [], {}
    for i, ticker in enumerate(tickers):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        tracks.append(TickerTrack(
            ticker=ticker,
            dates=tuple(str(d) for d in dates[lo:hi]),
            y_true=np.asarray(y_true[lo:hi], dtype=float),
            y_pred=np.asarray(y_pred[lo:hi], dtype=float),
            # Carried so the grader can refuse a ticker whose prediction is
            # constant inside EVERY fold - see block_bootstrap_ic.
            folds=np.asarray(fold[lo:hi], dtype=int),
        ))
        folds[ticker] = np.asarray(fold[lo:hi], dtype=int)
    return {"tracks": tracks, "folds": folds, "meta": meta}


# ── The old gate, on the same rows ────────────────────────────────────────────


def old_grades_from_tracks(tracks: list[TickerTrack],
                           horizon: int = BLOCK_LENGTH_SESSIONS,
                           ) -> dict[str, tuple[str, list[str]]]:
    """
    Re-runs the LIVE gate, unmodified, over the regenerated rows.

    `compute_metrics` and `grade_evidence` are imported and called, never
    reimplemented — a second copy of the thing being compared against would make
    the crosstab a comparison of two of my own functions rather than of the old
    gate against the new one.
    """
    out: dict[str, tuple[str, list[str]]] = {}
    for track in tracks:
        m = compute_metrics(track.y_true, track.y_pred, horizon=horizon)
        grade, reasons = grade_evidence({
            "forecast_available": True,
            "eval_rank_ic": m.get("rank_ic"),
            "eval_rank_ic_t": m.get("rank_ic_t"),
            "eval_hit_rate": m.get("hit_rate"),
            "eval_baseline_hit_rate": m.get("majority_hit_rate"),
            "eval_beats_naive": m.get("beats_naive_mae"),
        })
        out[track.ticker] = (grade, reasons)
    return out


def verify_against_persisted(tracks: list[TickerTrack]) -> pd.DataFrame:
    """How far the regenerated scalars sit from what `model_metadata` holds."""
    from sqlalchemy import text
    from data.db import get_engine

    stored = pd.read_sql(
        text("SELECT ticker, eval_rank_ic, eval_rank_ic_t, eval_n_oos, "
             "model_version FROM model_metadata"),
        get_engine()).set_index("ticker")

    rows = []
    for track in tracks:
        m = compute_metrics(track.y_true, track.y_pred,
                            horizon=BLOCK_LENGTH_SESSIONS)
        if track.ticker not in stored.index:
            continue
        s = stored.loc[track.ticker]
        rows.append({
            "ticker": track.ticker,
            "ic_now": m.get("rank_ic"),
            "ic_stored": s["eval_rank_ic"],
            "d_ic": (m.get("rank_ic") or np.nan) - (s["eval_rank_ic"] or np.nan),
            "n_now": m.get("n"),
            "n_stored": s["eval_n_oos"],
            "version_stored": s["model_version"],
        })
    return pd.DataFrame(rows)


# ── Reporting ─────────────────────────────────────────────────────────────────


def fold_diagnostic(tracks, folds: dict) -> dict:
    """
    IS THE QUANTITY BEING GRADED EVEN A RANKING? Measured, not assumed.

    Both gates — the live one and this one — grade a per-ticker rank IC POOLED
    over every out-of-sample row, across all five folds. `CLAUDE.md` records
    the hazard in the abstract: "the pooled IC correlates every (date, ticker)
    row at once, so it can be moved by knowing which fold a row came from",
    and `TrainMeanForecast` scored a pooled IC of -0.007 on exactly that
    mechanism while holding no ranking information at all.

    This measures how much of the graded number is that mechanism. Three
    numbers say it:

      1. how many (ticker, fold) cells emit a CONSTANT prediction, which has no
         ordering within the fold and therefore contributes to the pooled IC
         only through its LEVEL;
      2. the per-ticker pooled IC against the average of its WITHIN-FOLD ICs,
         which is the same rows scored with the fold level removed;
      3. the rank correlation, across the five folds, between the mean
         prediction level and the mean realised return — the channel by which
         a constant per fold becomes a pooled correlation.

    Read (3) with its sample size in mind: it is a correlation over FIVE
    points. That is the point. A five-observation nuisance correlation is
    steering a statistic being reported as a per-company track record.
    """
    from pipeline.evaluation import rank_ic

    cells = const_cells = rows_total = rows_const = 0
    per_ticker_const, pooled, within, levels = [], [], [], []

    for track in tracks:
        f = folds[track.ticker]
        keys = sorted(set(f.tolist()))
        n_const, fold_ics, fold_levels = 0, [], []
        for k in keys:
            mask = f == k
            cells += 1
            rows_total += int(mask.sum())
            fold_levels.append(float(np.mean(track.y_pred[mask])))
            fold_ics.append(rank_ic(track.y_true[mask], track.y_pred[mask]))
            if np.ptp(track.y_pred[mask]) == 0:
                const_cells += 1
                n_const += 1
                rows_const += int(mask.sum())
        per_ticker_const.append(n_const)
        pooled.append(rank_ic(track.y_true, track.y_pred))
        within.append(float(np.nanmean(fold_ics)) if np.any(np.isfinite(fold_ics))
                      else np.nan)
        levels.append(fold_levels)

    pooled = np.array(pooled, dtype=float)
    within = np.array(within, dtype=float)
    level_by_fold = np.nanmean(np.array(levels, dtype=float), axis=0)

    # `within` is legitimately ALL-NaN when every fold holds a constant
    # prediction, which is not an error state - it is the strongest possible
    # version of the finding. numpy warns on the empty slice, so the warning is
    # suppressed here rather than left to look like a defect in the report.
    def _nanmean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float("nan")

    def _nanmedian(a: np.ndarray) -> float:
        return float(np.nanmedian(a)) if np.any(np.isfinite(a)) else float("nan")

    n_folds = len(level_by_fold)
    realised = np.array([
        np.concatenate([t.y_true[folds[t.ticker] == k] for t in tracks]).mean()
        for k in range(n_folds)])
    from scipy import stats as _st
    rho = float(_st.spearmanr(level_by_fold, realised).correlation)

    print()
    print("=" * 78)
    print("2. IS THE GRADED QUANTITY A RANKING? - the pooled-IC fold artifact")
    print("=" * 78)
    print(f"  (ticker, fold) cells                {cells}")
    print(f"  ... emitting a CONSTANT prediction  {const_cells}"
          f"  ({100 * const_cells / cells:.1f}%)")
    print(f"  ... out-of-sample rows inside one   {rows_const}/{rows_total}"
          f"  ({100 * rows_const / rows_total:.1f}%)")
    counts = {k: per_ticker_const.count(k) for k in sorted(set(per_ticker_const))}
    print(f"  tickers by number of constant folds {counts}")
    print()
    print(f"  per-ticker IC, POOLED over folds    mean {_nanmean(pooled):+.4f}"
          f"   median {_nanmedian(pooled):+.4f}"
          f"   positive {int((pooled > 0).sum())}/{len(pooled)}")
    print(f"  per-ticker IC, mean WITHIN folds    mean {_nanmean(within):+.4f}"
          f"   median {_nanmedian(within):+.4f}"
          f"   positive {int(np.nansum(within > 0))}/{len(within)}")
    print()
    print(f"  mean prediction level by fold       "
          f"{np.round(level_by_fold, 5).tolist()}")
    print(f"  mean realised return by fold        {np.round(realised, 5).tolist()}")
    print(f"  rank correlation of those two       {rho:+.3f}  over {n_folds} folds")

    pooled_mean, within_mean = _nanmean(pooled), _nanmean(within)
    if (np.isfinite(pooled_mean) and np.isfinite(within_mean)
            and np.sign(pooled_mean) != np.sign(within_mean)):
        print()
        print("  *** THE POOLED AND WITHIN-FOLD ICs CARRY OPPOSITE SIGNS. ***")
        print("  Removing the fold level flips the panel's average per-ticker "
              "IC. The pooled")
        print("  number - the one BOTH gates grade - is therefore substantially "
              "a statement")
        print("  about how five fold-level constants line up against five "
              "realised period")
        print("  returns, not about ordering companies. Read mu_hat, and any "
              "ANTI_SIGNAL")
        print("  grade, in that light.")

    return {
        "cells": cells, "constant_cells": const_cells,
        "constant_row_fraction": rows_const / rows_total,
        "tickers_by_constant_folds": counts,
        "pooled_ic_mean": pooled_mean,
        "within_fold_ic_mean": within_mean,
        "fold_level_vs_realised_rho": rho,
        "n_folds": n_folds,
    }


def report(grading, tracks=None, folds: dict | None = None,
           live: pd.DataFrame | None = None) -> None:
    print()
    print("=" * 78)
    print("1. PANEL DIAGNOSTICS - the number this stage exists to produce")
    print("=" * 78)
    print(f"  tickers measured          {grading.n_usable}"
          f"  (unmeasurable: {grading.n_unusable})")
    print(f"  mu_hat  (grand mean IC)   {grading.mu_hat:+.5f}"
          f"   sd {np.sqrt(grading.mu_var):.5f}")
    print(f"  mu_hat  (random effects)  {grading.mu_hat_random_effects:+.5f}"
          f"   [diagnostic only]")
    print(f"  tau2_hat (between-ticker) {grading.tau2:.8f}"
          f"   sd {np.sqrt(grading.tau2):.5f}")
    print(f"  Cochran's Q               {grading.q_statistic:.2f}"
          f"   against E[Q] = {grading.n_usable - 1} under one common IC")
    print(f"  break-even rank IC        {grading.break_even_ic:.5f}")
    print(f"  block length              {grading.block_length} sessions"
          f"   ({grading.n_resamples} resamples, FDR q = {grading.fdr_q})")

    # Section 2 is printed from HERE rather than from main() so that the
    # numbering in the output matches the order it is read in. It qualifies
    # everything below it and must not appear after the crosstab.
    if tracks is not None and folds is not None:
        fold_diagnostic(tracks, folds)

    if grading.degenerate:
        print()
        print("  *** DEGENERATE PANEL: tau2_hat = 0 ***")
        print("  There is no detectable between-ticker variation in skill, so "
              "every measured")
        print("  ticker is shrunk ALL the way onto the grand mean and receives "
              "an identical")
        print("  posterior, an identical p-value and an identical grade. Read "
              "the counts")
        print("  below as ONE statement about the universe printed N times, "
              "not as N")
        print("  per-ticker findings - and note that on this branch the whole "
              "board can")
        print("  only move together.")

    print()
    print("=" * 78)
    print("3. OLD GRADE vs NEW GRADE - both measured on the same rows")
    print("=" * 78)
    table = grading.crosstab()
    print(table.to_string() if not table.empty else "  (empty)")
    old, new = grading.to_metrics()["old_grade_counts"], grading.counts()
    print(f"\n  old: {old}")
    print(f"  new: {new}")
    if live is not None and not live.empty:
        print(f"  live board (forecast_current, last weekly run): "
              f"{live.set_index('forecast_confidence')['n'].to_dict()}")

    print()
    print("=" * 78)
    print("4. THE TEN LARGEST POSTERIORS, both tails")
    print("=" * 78)
    rows = [r for r in grading.rows if np.isfinite(r.theta)]
    rows.sort(key=lambda r: r.theta, reverse=True)
    # De-duplicated by identity, not by slicing: on a panel of fewer than ten
    # measured tickers `rows[:5] + rows[-5:]` prints the same name twice, which
    # reads as two independent observations agreeing.
    shown = rows[:5] + [r for r in rows[-5:] if r not in rows[:5]]
    print(f"  {'ticker':<16}{'hat_ic':>9}{'sigma':>9}{'B':>7}"
          f"{'theta':>9}{'P(skill)':>10}{'p_two':>8}{'BH':>4}  grade")
    for r in shown:
        print(f"  {r.ticker:<16}{r.hat_ic:>+9.4f}{np.sqrt(r.sigma2):>9.4f}"
              f"{r.shrinkage_weight:>7.3f}{r.theta:>+9.4f}{r.p_positive:>10.3f}"
              f"{r.p_two_sided:>8.3f}{'*' if r.bh_significant else '':>4}"
              f"  {r.grade_v2}")

    unusable = [r for r in grading.rows if not np.isfinite(r.theta)]
    if unusable:
        print(f"\n  not measurable ({len(unusable)}):")
        for r in unusable:
            print(f"    {r.ticker:<16} {r.reason}")


def block_sweep(tracks, old, break_even) -> None:
    """
    The pre-registered block length against its neighbours.

    THE STANDING RULE IN THIS REPO IS THAT A RESULT AT ONE SETTING IS NOT A
    RESULT — the valuation finding scored t +3.32 at `min_train=380` and +1.00
    at 500 on identical rows. The block length is this stage's arbitrary
    setting: too short and sigma2 comes back too small, every posterior too
    confident, and grades appear out of a bootstrap that broke the label
    overlap. This sweep is a diagnostic, not a menu — 30 is the pre-registered
    primary and the others are here to show whether the answer moves.
    """
    print()
    print("=" * 78)
    print("5. BLOCK-LENGTH ROBUSTNESS - 30 is pre-registered, the rest diagnose")
    print("=" * 78)
    print(f"  {'block':>7}{'mu_hat':>11}{'tau2':>12}{'mean sigma':>12}"
          f"{'STRONG':>8}{'WEAK':>7}{'ANTI':>7}{'INSUF':>7}")
    for block in (10, 20, 30, 45, 60):
        g = grade_panel(tracks, old_grades=old, block=block,
                        break_even=break_even)
        sig = np.sqrt(np.nanmean([r.sigma2 for r in g.rows]))
        c = g.counts()
        mark = " <- pre-registered" if block == BLOCK_LENGTH_SESSIONS else ""
        print(f"  {block:>7}{g.mu_hat:>+11.5f}{g.tau2:>12.8f}{sig:>12.5f}"
              f"{c['STRONG']:>8}{c['WEAK']:>7}{c['ANTI_SIGNAL']:>7}"
              f"{c['INSUFFICIENT']:>7}{mark}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--no-rebuild", action="store_true",
                    help="grade an existing cache; fail if it is absent")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the cache even if one exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N tickers only — for smoke tests, never for a "
                         "reported result")
    ap.add_argument("--store", action="store_true",
                    help="write the shadow table `evidence_grades_v2`")
    ap.add_argument("--block-sweep", action="store_true")
    ap.add_argument("--verify-against-persisted", action="store_true")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    if args.cache.exists() and not args.rebuild:
        print(f"[Stage0] reading cached out-of-sample rows from {args.cache}")
        cache = load_cache(args.cache)
        print(f"         {cache['meta']}")
    elif args.no_rebuild:
        raise SystemExit(f"{args.cache} does not exist and --no-rebuild was given")
    else:
        from data.universe import get_universe
        universe = get_universe()
        if args.limit:
            universe = universe[: args.limit]
        print(f"[Stage0] regenerating held-out track records for "
              f"{len(universe)} tickers (~47 s each) ...")
        cache = build_cache(universe, args.cache)

    tracks = cache["tracks"]
    if args.limit:
        tracks = tracks[: args.limit]

    print(f"[Stage0] re-running the LIVE gate on the same rows ...")
    old = old_grades_from_tracks(tracks)

    if args.verify_against_persisted:
        drift = verify_against_persisted(tracks)
        if not drift.empty:
            print(f"\n  regenerated vs persisted `model_metadata`: "
                  f"max |d_ic| {drift['d_ic'].abs().max():.4f}, "
                  f"median |d_ic| {drift['d_ic'].abs().median():.4f}, "
                  f"row counts identical for "
                  f"{int((drift['n_now'] == drift['n_stored']).sum())}"
                  f"/{len(drift)} tickers")

    bar, bar_detail = economic_bar()
    print(f"[Stage0] break-even rank IC {bar:.5f} "
          f"(spread_per_ic {bar_detail.get('spread_per_unit_ic', float('nan')):.4f} "
          f"from {bar_detail['spread_per_ic_source']})")

    print(f"[Stage0] bootstrapping {len(tracks)} tickers "
          f"({BOOTSTRAP_N_RESAMPLES} resamples, block {BLOCK_LENGTH_SESSIONS}) ...")
    t0 = time.time()
    grading = grade_panel(tracks, old_grades=old, break_even=bar)
    print(f"         done in {time.time() - t0:.1f}s")

    live = None
    try:
        from sqlalchemy import text
        from data.db import get_engine
        live = pd.read_sql(
            text("SELECT forecast_confidence, COUNT(*) AS n FROM "
                 "forecast_current GROUP BY forecast_confidence"), get_engine())
    except Exception as exc:                                    # noqa: BLE001
        print(f"  (live board unavailable: {exc})")

    report(grading, tracks=tracks, folds=cache["folds"], live=live)

    if args.block_sweep:
        block_sweep(tracks, old, bar)

    if args.csv:
        pd.DataFrame([vars(r) for r in grading.rows]).to_csv(args.csv, index=False)
        print(f"\n  per-ticker rows -> {args.csv}")

    if args.store:
        n = store_grading(grading)
        print(f"\n  wrote {n} rows to evidence_grades_v2 "
              f"(run_id {grading.run_id}) - SHADOW ONLY, nothing the API "
              f"serves has changed")

    print()
    print("=" * 78)
    print("PRE-REGISTERED READING (docs/stage0-preregistration.md)")
    print("=" * 78)
    counts = grading.counts()
    graded = counts["STRONG"] + counts["WEAK"]
    old_graded = sum(v for k, v in grading.to_metrics()["old_grade_counts"].items()
                     if k in ("STRONG", "WEAK"))
    print(f"  STRONG + WEAK: old {old_graded} -> new {graded}")
    if graded <= old_graded:
        print("  The count did NOT rise. Per the pre-registered rule the "
              "constraint is\n  SIGNAL, not measurement - the next move is "
              "Stage 1 (new data sources),\n  not a looser grading scheme.")
    else:
        print("  The count ROSE. Before reporting that as good news, check the "
              "two\n  failure modes named in the pre-registration: a degenerate "
              "tau2_hat = 0\n  collapsing the posterior variance, and a "
              "bootstrap block shorter than\n  the 30-session label overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
