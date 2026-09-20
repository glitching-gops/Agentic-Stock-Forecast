"""
tools/stage2b_pooled.py — Stage 2b: the pooled cross-sectional model.

Stage 2a established that the per-ticker tuner is over-specified: its inner CV
holds n_eff of 3.3 to 14.7 independent observations, so selecting among ten
configurations on a rank IC estimated there is mostly noise, and the noise buys
spurious splits. This tests the obvious consequence — pool the training across
all 84 tickers, and the same objective has orders of magnitude more to go on.

What already existed, and what this adds
----------------------------------------
``pipeline/baselines.py::_pooled_xgb_factory`` is ALREADY one XGBoost fitted on
the whole pooled panel through ``panel_walk_forward``, on the 15 scale-free
FACTORS columns. It is UNTUNED by design and carries NO ticker identity. So
this session is an audit and an extension, not a rewrite: it adds the nested
pooled hyperparameter search (both objectives), the ticker categorical, and the
degeneracy / train-gap diagnostics Stage 2a used to catch overfitting.

SANDBOXED. Reads a cached panel, writes a markdown report, a CSV and an .npz of
held-out predictions. No hyperparameter cache, no ``model_metadata``, no
forecast, nothing the API or the web app serves.

Usage
-----
    python tools/stage2b_pooled.py --build-cache          # once, ~40 s
    python tools/stage2b_pooled.py --markdown stage2b.md --csv stage2b.csv
    python tools/stage2b_pooled.py --arms pooled_mae --trials 3   # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.baselines import FACTORS  # noqa: E402
from pipeline.evaluation import (  # noqa: E402
    PurgedPanelWalkForward,
    cross_sectional_report,
    effective_sample_size,
    rank_ic,
)
from pipeline.model import EVAL_N_FOLDS, EVAL_TUNE_TRIALS  # noqa: E402
from pipeline.panel import (  # noqa: E402
    SCALE_FREE,
    TARGET,
    cross_sectional_zscore,
)
from pipeline.signals import HORIZON_SESSIONS  # noqa: E402
from pipeline.tuning import per_date_rank_ic, tune_pooled  # noqa: E402
from tools.stage2a_gamma_spotcheck import mode_share, unique_fraction  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
OOS_CACHE = os.path.join(ROOT, "evidence_oos.npz")
PRED_OUT = os.path.join(ROOT, "stage2b_pooled_oos.npz")

EVAL_MIN_TRAIN_DATES = 500
BREAK_EVEN_IC = 0.00512363994209475     # P4, zero market impact
NULL_DRAWS = 400
BLOCK = HORIZON_SESSIONS

TICKER_COL = "ticker_cat"

# The arms. Each is (label, objective, ticker mode).
ARMS: dict[str, tuple[str, str]] = {
    "pooled_mae":            ("mae", "identity"),
    "pooled_rank_ic":        ("rank_ic", "identity"),
    "pooled_mae_noticker":   ("mae", "none"),
    "pooled_rank_ic_noticker": ("rank_ic", "none"),
    "pooled_rank_ic_placebo": ("rank_ic", "shuffled"),
    "pooled_mae_placebo":     ("mae", "shuffled"),
}


# ── the panel ─────────────────────────────────────────────────────────────────


def build_cache(path: str = PANEL_CACHE) -> None:
    from data.universe import get_universe
    from pipeline.panel import load_panel

    panel = load_panel(get_universe())
    panel.to_parquet(path)
    print(f"cached {panel.shape} -> {path}")


def load_cached_panel(path: str = PANEL_CACHE) -> pd.DataFrame:
    """
    The panel, standardised within each date exactly as ``compare_baselines``
    does before fitting anything.

    Applied once, before splitting, and that is safe rather than sloppy: the
    z-score is computed from a single date's own cross-section, so every input
    was observable at the time and there is no lookahead to purge. A script
    that attacks a table must run the table's own preprocessing — the P2 sweep
    that omitted this read `pooled_xgb` at +0.99 against a table saying +2.42.
    """
    if not os.path.exists(path):
        raise SystemExit(f"no panel cache at {path}; run with --build-cache")
    panel = pd.read_parquet(path)
    panel = cross_sectional_zscore(panel, SCALE_FREE)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def with_ticker(panel: pd.DataFrame, mode: str, seed: int = 20260908) -> pd.DataFrame:
    """
    Attaches the ticker categorical in one of three modes.

    ``identity``  — the real name, as a pandas category. XGBoost 3.x reads it
                    natively with ``enable_categorical=True``, so there is no
                    encoding to fit and therefore no encoding that could be fit
                    across a fold boundary.
    ``none``      — absent. The control that says what the feature is worth.
    ``shuffled``  — labels permuted WITHIN each date. Cardinality and per-date
                    frequency are preserved exactly; the link between a name and
                    its own returns is destroyed. A PARTIAL placebo: it also
                    destroys the label's persistence across dates, which the
                    real feature has, so it bounds the null from below.

    No relabeling can placebo-test pure identity — any bijection of 84 names
    onto 84 categories carries identical information — which is why the
    with/without comparison, not this, is what the ticker feature is judged on.
    """
    if mode == "none":
        return panel

    out = panel.copy()
    if mode == "identity":
        labels = out["ticker"].to_numpy()
    elif mode == "shuffled":
        rng = np.random.default_rng(seed)
        labels = np.empty(len(out), dtype=object)
        # The frame is sorted by (date, ticker), so each date is one contiguous
        # block and a per-block permutation is a within-date shuffle.
        dates = out["date"].to_numpy()
        bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            labels[lo:hi] = rng.permutation(out["ticker"].to_numpy()[lo:hi])
    else:
        raise ValueError(f"unknown ticker mode {mode!r}")

    # The category set is the panel's own names in sorted order, because both
    # modes carry every name and pandas sorts the distinct labels it is given.
    # An explicit `categories=` argument was written first and REMOVED: no
    # input could distinguish it from this, so it was a redundant guard rather
    # than a belt-and-braces one — the same call as the fundamentals
    # `first_seen` duplicate. The property itself is pinned by a test.
    out[TICKER_COL] = pd.Categorical(labels)
    return out


def feature_columns(mode: str) -> list[str]:
    return list(FACTORS) + ([] if mode == "none" else [TICKER_COL])


# ── the effective sample size behind one hyperparameter decision ──────────────


def _block_permute(values: np.ndarray, rng, block: int = BLOCK) -> np.ndarray:
    """Moving-block permutation at the label horizon, so the surrogate keeps a
    real prediction's persistence instead of being iid — an iid null understates
    the objective's spread and would flatter the pooled arm."""
    n = values.size
    starts = rng.integers(0, max(n - block, 1), size=n // block + 1)
    out = np.concatenate([values[s:s + block] for s in starts])[:n]
    return out if out.size == n else np.resize(out, n)


def objective_null_se(panel: pd.DataFrame, draws: int = NULL_DRAWS,
                      seed: int = 7, outer_fold: int = 2) -> dict:
    """
    How much each objective moves under a prediction that knows nothing.

    This is the number the whole hypothesis rests on, and it is the honest form
    of "effective sample size": not a row count, but the standard error of the
    statistic a TRIAL IS SELECTED ON. A search cannot resolve a real difference
    between two configurations that is smaller than this, so anything it
    resolves below it is noise being spent on hyperparameters.

    Measured at the INNER-CV altitude for both arms, because that is where
    selection happens — the outer fold is where the winner is reported, not
    where it is chosen. The surrogate is a moving-block permutation at the
    label horizon, so it carries a real prediction's persistence; an iid
    surrogate would understate both spreads and flatter the pooled arm most.
    """
    rng = np.random.default_rng(seed)
    dates = panel["date"].to_numpy()
    y = pd.to_numeric(panel[TARGET], errors="coerce").to_numpy(dtype=float)

    outer = PurgedPanelWalkForward(
        n_folds=EVAL_N_FOLDS, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=EVAL_MIN_TRAIN_DATES)
    train_idx, _ = list(outer.split(dates))[outer_fold]
    tr = train_idx[np.isfinite(y[train_idx])]
    slice_ = panel.iloc[tr]

    # ── pooled: the inner block purged_panel_cv_score actually scores ────────
    inner = PurgedPanelWalkForward(
        n_folds=3, horizon=HORIZON_SESSIONS, embargo=HORIZON_SESSIONS,
        min_train=max(2 * HORIZON_SESSIONS, slice_["date"].nunique() // 2))
    inner_dates = slice_["date"].to_numpy()
    _, inner_test = list(inner.split(inner_dates))[0]
    blk = slice_.iloc[inner_test]
    d, truth = blk["date"].to_numpy(), pd.to_numeric(
        blk[TARGET], errors="coerce").to_numpy(dtype=float)
    tick = blk["ticker"].to_numpy()

    order = np.argsort(tick, kind="stable")
    bounds = np.flatnonzero(np.r_[True, tick[order][1:] != tick[order][:-1], True])
    pooled = []
    for _ in range(draws):
        fake = np.empty(len(blk))
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            idx = order[lo:hi]
            fake[idx] = _block_permute(rng.normal(size=idx.size), rng)
        pooled.append(per_date_rank_ic(d, truth, fake)[0])

    # ── per-ticker: the inner block purged_cv_rank_ic_score scores ───────────
    # Stage 2a measured these at 98 / 184 / 270 / 356 / 442 rows across the
    # five outer folds; the matching one is taken here.
    from pipeline.evaluation import PurgedWalkForward

    one = slice_[slice_["ticker"] == sorted(slice_["ticker"].unique())[0]]
    n_one = len(one)
    inner_ts = PurgedWalkForward(n_folds=3, horizon=HORIZON_SESSIONS,
                                 embargo=HORIZON_SESSIONS,
                                 min_train=max(120, n_one // 3))
    splits = list(inner_ts.split(n_one))
    ts_truth = pd.to_numeric(one[TARGET], errors="coerce").to_numpy(dtype=float)
    ts_idx = splits[0][1] if splits else np.arange(min(270, n_one))

    per_ticker = []
    for _ in range(draws):
        per_ticker.append(rank_ic(
            ts_truth[ts_idx], _block_permute(rng.normal(size=ts_idx.size), rng)))

    n_dates, n_rows = len(np.unique(d)), len(blk)
    return {
        "pooled_rows": n_rows,
        "pooled_dates": n_dates,
        "pooled_names_per_date": n_rows / max(n_dates, 1),
        "n_eff_rows_over_horizon": effective_sample_size(n_rows, HORIZON_SESSIONS),
        "n_eff_dates_over_horizon": effective_sample_size(n_dates, HORIZON_SESSIONS),
        "pooled_null_se": float(np.nanstd(pooled)),
        "per_ticker_rows": int(ts_idx.size),
        "per_ticker_n_eff": effective_sample_size(int(ts_idx.size), HORIZON_SESSIONS),
        "per_ticker_null_se": float(np.nanstd(per_ticker)),
    }


# ── one arm ───────────────────────────────────────────────────────────────────


def run_arm(panel: pd.DataFrame, objective: str, ticker_mode: str,
            n_trials: int = EVAL_TUNE_TRIALS,
            n_folds: int = EVAL_N_FOLDS,
            min_train: int = EVAL_MIN_TRAIN_DATES,
            verbose: bool = True,
            fixed_params: dict | None = None,
            features: list[str] | None = None,
            horizon: int = HORIZON_SESSIONS,
            purge: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    A pooled purged walk-forward with the search nested inside each training
    fold. The splitter and its constants are ``panel_walk_forward``'s, so the
    fold boundaries are the ones every other comparator in this project was
    scored on.

    `horizon` is the LABEL width in sessions, and the caller is responsible for
    having retargeted ``panel[TARGET]`` to it (``panel.retarget_horizon``). It
    is passed in rather than inferred from the target so the two cannot
    disagree silently - the same reason ``_log_price_basis`` derives the price
    series from the target's name instead of taking it as an argument.

    `purge` is the gap carved out of the end of training, applied as BOTH the
    purge and the embargo, so the last training date sits ``2 * purge`` dates
    before the first test date. ``None`` means `horizon`, which is the legacy
    rule every pre-P6 result in this project was measured under. P6 passes
    ``horizon_purge_embargo(horizon)``, which never goes below the panel's own
    measured serial dependence.
    """
    purge = horizon if purge is None else int(purge)
    if purge < horizon:
        raise ValueError(
            f"purge {purge} is narrower than the {horizon}-session label it "
            f"must span; training labels would reach into the test window")
    from xgboost import XGBRegressor

    frame = with_ticker(panel, ticker_mode)
    features = feature_columns(ticker_mode) if features is None else features
    categorical = ticker_mode != "none"

    splitter = PurgedPanelWalkForward(
        n_folds=n_folds, horizon=purge,
        embargo=purge, min_train=min_train)
    dates = frame["date"].to_numpy()
    y = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype=float)

    blocks: list[pd.DataFrame] = []
    folds: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dates)):
        tr = train_idx[np.isfinite(y[train_idx])]
        te = test_idx[np.isfinite(y[test_idx])]
        if len(tr) < 100 or len(te) == 0:
            continue

        t0 = time.time()
        # ``fixed_params`` skips the search. It exists so the outer loop can be
        # pinned against ``panel_walk_forward`` on identical inputs — a harness
        # that quietly split differently would make every number below
        # incomparable with the rest of this project's tables, and Stage 2a
        # showed that check is worth having (its refit matched the Stage 0
        # cache at drift 0.0e+00, which is what made the sweep readable).
        # The INNER search is purged at the same width as the outer split.
        # Nesting is not automatic: if this argument stayed at the module's 30
        # while the outer fold widened, every hyperparameter would be chosen
        # across a boundary the outer fold refuses to trust - F3 one level
        # down, and invisible from outside, because the reported number would
        # merely be optimistic rather than wrong-shaped.
        params = fixed_params if fixed_params is not None else tune_pooled(
            frame.iloc[tr], features, target=TARGET, horizon=purge,
            n_trials=n_trials, tuning_objective=objective,
            enable_categorical=categorical)

        model = XGBRegressor(**params, random_state=42, verbosity=0,
                             enable_categorical=categorical)
        model.fit(frame.iloc[tr][features], y[tr])
        pred = np.asarray(model.predict(frame.iloc[te][features]), dtype=float)
        in_sample = np.asarray(model.predict(frame.iloc[tr][features]), dtype=float)

        blocks.append(pd.DataFrame({
            "date": dates[te],
            "ticker": frame.iloc[te]["ticker"].to_numpy(),
            "y_true": y[te],
            "y_pred": pred,
            "fold": fold,
        }))
        folds.append({
            "fold": fold,
            "gamma": float(params.get("gamma", float("nan"))),
            "max_depth": int(params.get("max_depth", 0)),
            "train_rows": int(tr.size),
            "train_mae": float(np.mean(np.abs(y[tr] - in_sample))),
            "secs": round(time.time() - t0, 1),
        })
        if verbose:
            print(f"    fold {fold}: {tr.size:,} train rows, gamma "
                  f"{params.get('gamma', float('nan')):.3f}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    return pd.concat(blocks, ignore_index=True), folds


# ── scoring one cell ──────────────────────────────────────────────────────────


def cell_metrics(preds: pd.DataFrame, folds: list[dict] | None = None,
                 rebalance_every: int = HORIZON_SESSIONS) -> dict:
    """
    Every quantity the pre-registration named, for one cell of the 2x2.

    Degeneracy is measured per (ticker, fold) cell with the SAME two metrics the
    Stage 0 addendum and Stage 2a used, so the three sets of numbers are one
    series rather than three.
    """
    cells = []
    for (ticker, fold), g in preds.groupby(["ticker", "fold"]):
        p = g["y_pred"].to_numpy()
        cells.append({
            "ticker": ticker, "fold": int(fold),
            "mode_share": mode_share(p),
            "unique_fraction": unique_fraction(p),
            "ts_rank_ic": rank_ic(g["y_true"].to_numpy(), p),
        })
    cells = pd.DataFrame(cells)

    cs_by_fold = []
    for fold, g in preds.groupby("fold"):
        cs_by_fold.append(per_date_rank_ic(g["date"].to_numpy(),
                                           g["y_true"].to_numpy(),
                                           g["y_pred"].to_numpy())[0])

    train_mae = (float(np.mean([f["train_mae"] for f in folds]))
                 if folds else float("nan"))
    mae = float(np.mean(np.abs(preds["y_true"] - preds["y_pred"])))

    def _mean(values) -> float:
        """NaN rather than a warning when nothing is defined — an all-constant
        cell has no IC, and averaging an empty list must say so."""
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else float("nan")

    # THE NON-OVERLAPPING STATISTIC, WHICH IS THE ONE THAT SUPPORTS INFERENCE.
    #
    # `cs_rank_ic` above is the mean over EVERY out-of-sample date, the
    # `daily_IC` analogue: consecutive dates share 29 of their 30 forward
    # sessions, so a t-statistic on it is inflated roughly 5x. `reb_ic` is the
    # mean over the ~64 non-overlapping rebalance dates and `reb_t` is its
    # t-statistic. CLAUDE.md records these as DIFFERENT SAMPLES that have
    # carried opposite signs, so they are reported in separate columns and the
    # t belongs only to the second. Computed by `cross_sectional_report`, the
    # same function every comparator in this project has been scored by.
    # `rebalance_every` IS the horizon being scored, so the sampled dates are
    # non-overlapping at that horizon. Leaving it at 30 while sweeping h would
    # take every 30th date of a 5-session label and throw away five sixths of
    # the independent windows the shorter horizon bought; the t column would
    # then be measuring the sampling rule rather than the panel.
    report = cross_sectional_report(preds[["date", "ticker", "y_pred", "y_true"]],
                                    rebalance_every=rebalance_every)

    return {
        "cells": len(cells),
        "constant_cells": int((cells["mode_share"] == 1.0).sum()),
        "reb_ic": float(report.get("mean_rank_ic", float("nan"))),
        "reb_t": float(report.get("rank_ic_t", float("nan"))),
        "n_rebalances": int(report.get("n_rebalances", 0)),
        "n_dates_no_ordering": int(report.get("n_dates_no_ordering", 0)),
        "degeneracy_rate": float((cells["mode_share"] == 1.0).mean()),
        "mean_unique_fraction": float(cells["unique_fraction"].mean()),
        "mae": mae,
        "train_mae": train_mae,
        "gap": mae - train_mae,
        "cs_rank_ic": _mean(cs_by_fold),
        "cs_rank_ic_by_fold": [round(v, 4) for v in cs_by_fold],
        "ts_rank_ic": _mean(cells["ts_rank_ic"]),
        "n_rows": len(preds),
        "n_tickers": preds["ticker"].nunique(),
    }


def restrict_to_cache_rows(preds: pd.DataFrame,
                           cache_path: str = OOS_CACHE) -> pd.DataFrame:
    """
    The pooled predictions on exactly the (ticker, date) pairs the per-ticker
    arm produced.

    The per-ticker and pooled protocols split different grids — one on a
    ticker's own row positions, the other on the shared date grid — so their
    natural row sets differ. Comparing MAE across them unrestricted would be the
    sweep-that-changes-the-row-count error that retired the valuation lag
    result. This is the matched read.
    """
    if not os.path.exists(cache_path):
        return pd.DataFrame(columns=preds.columns)
    z = np.load(cache_path, allow_pickle=True)
    tickers, offsets = [str(t) for t in z["tickers"]], z["offsets"]
    dates = z["dates"]
    keys = set()
    for i, ticker in enumerate(tickers):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        keys.update((ticker, str(d)) for d in dates[lo:hi])
    mask = [(t, str(d)) in keys
            for t, d in zip(preds["ticker"], preds["date"])]
    return preds[np.asarray(mask)].reset_index(drop=True)


def per_ticker_cells(cache_path: str = OOS_CACHE) -> dict:
    """The per-ticker x mae cell, read from the Stage 0 cache rather than
    recomputed — it is the production baseline and re-running it could only
    introduce a difference."""
    z = np.load(cache_path, allow_pickle=True)
    tickers, offsets = [str(t) for t in z["tickers"]], z["offsets"]
    frame = pd.DataFrame({
        "ticker": np.repeat(tickers, np.diff(offsets)),
        "date": z["dates"], "y_true": z["y_true"],
        "y_pred": z["y_pred"], "fold": z["fold"],
    })
    return cell_metrics(frame)


# ── reporting ─────────────────────────────────────────────────────────────────


def render(results: dict[str, dict], null: dict, baseline: dict,
           matched: dict[str, dict]) -> str:
    o: list[str] = []
    o.append("# Stage 2b — the pooled cross-sectional model\n")
    o.append("Pre-registration: `docs/stage2b-preregistration.md`. "
             "Tool: `tools/stage2b_pooled.py`. Shadow only.\n")

    o.append("\n## The effective sample size behind one hyperparameter decision\n")
    o.append("| | per-ticker (Stage 2a) | pooled (Stage 2b) |")
    o.append("|---|---|---|")
    o.append(f"| training rows in the fold | {null['per_ticker_rows']:,} | "
             f"{null['pooled_rows']:,} |")
    o.append(f"| dates | — | {null['pooled_dates']:,} "
             f"({null['pooled_names_per_date']:.0f} names/date) |")
    o.append(f"| n_eff, rows / horizon | {null['per_ticker_n_eff']:.1f} | "
             f"{null['n_eff_rows_over_horizon']:.0f} |")
    o.append(f"| n_eff, dates / horizon | — | "
             f"{null['n_eff_dates_over_horizon']:.0f} |")
    o.append(f"| **null SE of the rank-IC objective** | "
             f"**{null['per_ticker_null_se']:.4f}** | "
             f"**{null['pooled_null_se']:.4f}** |")
    ratio = null["per_ticker_null_se"] / max(null["pooled_null_se"], 1e-12)
    o.append(f"\nMeasured over {NULL_DRAWS} moving-block ({BLOCK}-session) "
             f"surrogate predictions, so the null keeps a real prediction's "
             f"persistence. **The objective's standard error falls "
             f"{ratio:.0f}x.**\n")

    o.append("\n## The 2x2, each cell on its own protocol\n")
    o.append("| cell | tickers | rows | constant cells | degeneracy | "
             "cross-sec IC | time-series IC | OOS MAE | train MAE | gap |")
    o.append("|---|---|---|---|---|---|---|---|---|---|")

    def row(label: str, m: dict) -> str:
        train = "—" if not np.isfinite(m["train_mae"]) else f"{m['train_mae']:.5f}"
        gap = "—" if not np.isfinite(m["gap"]) else f"{m['gap']:+.5f}"
        return (f"| {label} | {m['n_tickers']} | {m['n_rows']:,} | "
                f"{m['constant_cells']}/{m['cells']} | "
                f"{100 * m['degeneracy_rate']:.0f}% | "
                f"{m['cs_rank_ic']:+.4f} | {m['ts_rank_ic']:+.4f} | "
                f"{m['mae']:.5f} | {train} | {gap} |")

    o.append(row("**per-ticker x mae** (production)", baseline))
    for name, m in results.items():
        o.append(row(f"**{name}**", m))

    o.append("\n## Matched rows — pooled scored only where the per-ticker arm scored\n")
    o.append("| cell | rows | constant cells | cross-sec IC | OOS MAE |")
    o.append("|---|---|---|---|---|")
    o.append(f"| per-ticker x mae | {baseline['n_rows']:,} | "
             f"{baseline['constant_cells']}/{baseline['cells']} | "
             f"{baseline['cs_rank_ic']:+.4f} | {baseline['mae']:.5f} |")
    for name, m in matched.items():
        o.append(f"| {name} | {m['n_rows']:,} | "
                 f"{m['constant_cells']}/{m['cells']} | "
                 f"{m['cs_rank_ic']:+.4f} | {m['mae']:.5f} |")

    o.append("\n## Cross-sectional IC per fold\n")
    o.append("| cell | fold 0 | 1 | 2 | 3 | 4 |")
    o.append("|---|---|---|---|---|---|")
    for label, m in [("per-ticker x mae", baseline)] + list(results.items()):
        cells = m["cs_rank_ic_by_fold"] + [float("nan")] * 5
        o.append(f"| {label} | " + " | ".join(f"{v:+.4f}" for v in cells[:5]) + " |")

    o.append(f"\nBreak-even rank IC at zero market impact is "
             f"**{BREAK_EVEN_IC:.4f}** (P4).\n")
    return "\n".join(o)


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--trials", type=int, default=EVAL_TUNE_TRIALS)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--npz", default=PRED_OUT)
    args = ap.parse_args()

    if args.build_cache:
        build_cache()
        return

    panel = load_cached_panel()
    print(f"panel {panel.shape}, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} dates")

    print("\nmeasuring the objective's null standard error...")
    null = objective_null_se(panel)
    print(f"  per-ticker {null['per_ticker_null_se']:.4f}   "
          f"pooled {null['pooled_null_se']:.4f}")

    baseline = per_ticker_cells()
    print(f"per-ticker x mae (from cache): {baseline['constant_cells']}"
          f"/{baseline['cells']} constant, cs IC {baseline['cs_rank_ic']:+.4f}")

    results, matched, frames = {}, {}, {}
    for name in args.arms:
        objective, ticker_mode = ARMS[name]
        print(f"\n{name}  (objective={objective}, ticker={ticker_mode})")
        preds, folds = run_arm(panel, objective, ticker_mode,
                               n_trials=args.trials)
        results[name] = cell_metrics(preds, folds)
        sub = restrict_to_cache_rows(preds)
        if len(sub):
            matched[name] = cell_metrics(sub)
        frames[name] = preds
        m = results[name]
        print(f"  -> {m['constant_cells']}/{m['cells']} constant, "
              f"cs IC {m['cs_rank_ic']:+.4f}, MAE {m['mae']:.5f}, "
              f"gap {m['gap']:+.5f}")

    text = render(results, null, baseline, matched)
    print("\n" + text)

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.markdown}")
    if args.csv:
        pd.DataFrame(results).T.to_csv(args.csv)
        print(f"wrote {args.csv}")
    if args.npz and frames:
        np.savez_compressed(
            args.npz,
            **{f"{k}__{c}": v[c].to_numpy() for k, v in frames.items()
               for c in ("date", "ticker", "y_true", "y_pred", "fold")})
        print(f"wrote {args.npz}")


if __name__ == "__main__":
    main()
