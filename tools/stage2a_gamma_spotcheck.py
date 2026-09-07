"""
tools/stage2a_gamma_spotcheck.py — Stage 2a, Step A: is the tuner the cause?

The Stage 0 addendum measured that 316 of 420 (ticker, fold) fits emit a
CONSTANT prediction, that the constant is the training mean to within 0.043 of
a standard deviation — so the trees make no splits at all — and that
``pipeline/tuning.py`` searches ``gamma`` over [0, 5] while scoring candidates
on MAE of a target whose dispersion is ~0.10.

This is the cheap, falsifiable test of the obvious consequence: hold every
other hyperparameter at whatever the nested Optuna search chose for that exact
cell, vary ``gamma`` alone, and see whether constancy goes away and the
within-fold rank IC improves.

Nothing here fits anything a forecast is served from. It reads the Stage 0
out-of-sample cache, re-runs the UNMODIFIED seeded tuner on training slices,
and refits in memory. `pipeline/tuning.py` and `pipeline/model.py` are not
modified by Step A.

Method and cell-selection rule are fixed in ``docs/stage2a-preregistration.md``,
written before this was ever run on real data.

Usage
-----
    python tools/stage2a_gamma_spotcheck.py --markdown docs/stage2a-step-a.md
    python tools/stage2a_gamma_spotcheck.py --limit 2      # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.evaluation import PurgedWalkForward, rank_ic  # noqa: E402
from pipeline.model import (  # noqa: E402
    EVAL_MIN_TRAIN,
    EVAL_N_FOLDS,
    EVAL_TUNE_TRIALS,
    FEATURES,
    TARGET,
    _model_factory,
    load_features_for_ticker,
)
from pipeline.signals import HORIZON_SESSIONS  # noqa: E402
from pipeline.tuning import tune  # noqa: E402

CACHE_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "evidence_oos.npz"
)

# The sweep. A single gamma would be the "result at one cell" error the project
# has been fooled by three times.
GAMMA_GRID = (0.0, 0.5, 1.0, 2.0)

# Pre-registered: 2 cells per fold, 5 folds, 10 distinct tickers.
CELLS_PER_FOLD = 2

# Pre-registered: "most sampled cells" means at least this many of the 10.
SUPPORT_THRESHOLD = 6

# The refit must reproduce the cached prediction, or the harness is not
# measuring what Stage 0 recorded. Same role tau = 1.00 played in the addendum.
REPRODUCTION_TOL = 1e-9


# ── degeneracy metrics (re-derived here so Stage 2a is self-contained) ─────────


def mode_share(values: np.ndarray) -> float:
    """
    Fraction of predictions equal to the modal value.

    Exact float equality on purpose: a tree that makes no splits returns the
    same float bit for bit, so a tolerance would only blur the thing being
    measured. Carried over from the Stage 0 addendum unchanged.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max() / values.size)


def unique_fraction(values: np.ndarray) -> float:
    """Distinct predicted values over row count. Scale-free, and independent
    of ``mode_share`` in the sense that neither is computed from the other."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.unique(values).size / values.size)


# ── cell selection ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    ticker: str
    fold: int
    mode_share: float
    n_rows: int


def load_cells(cache_path: str = CACHE_PATH) -> list[Cell]:
    """Every (ticker, fold) cell in the Stage 0 out-of-sample cache, scored."""
    z = np.load(cache_path, allow_pickle=True)
    tickers = [str(t) for t in z["tickers"]]
    offsets = z["offsets"]
    y_pred, folds = z["y_pred"], z["fold"]

    cells: list[Cell] = []
    for i, ticker in enumerate(tickers):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        p, f = y_pred[lo:hi], folds[lo:hi]
        for k in np.unique(f):
            block = p[f == k]
            cells.append(Cell(ticker, int(k), mode_share(block), int(block.size)))
    return cells


def select_cells(cells: list[Cell], per_fold: int = CELLS_PER_FOLD) -> list[Cell]:
    """
    The pre-registered, outcome-blind rule.

    For each fold in order, the first ``per_fold`` tickers alphabetically among
    that fold's FULLY constant cells, skipping tickers already taken. Spans
    every fold, clusters in none, and cannot be steered by what the refit shows
    because it looks only at the cached prediction's constancy.
    """
    degenerate = [c for c in cells if c.mode_share == 1.0]
    chosen: list[Cell] = []
    taken: set[str] = set()
    for fold in sorted({c.fold for c in degenerate}):
        pool = sorted((c for c in degenerate if c.fold == fold),
                      key=lambda c: c.ticker)
        for cell in pool:
            if cell.ticker in taken:
                continue
            chosen.append(cell)
            taken.add(cell.ticker)
            if sum(1 for c in chosen if c.fold == fold) >= per_fold:
                break
    return chosen


# ── the refit ─────────────────────────────────────────────────────────────────


def fold_slices(n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The exact folds ``evaluate_ticker`` uses. Not a reimplementation of the
    splitter — the splitter itself, at the evaluation constants."""
    splitter = PurgedWalkForward(
        n_folds=EVAL_N_FOLDS, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=EVAL_MIN_TRAIN,
    )
    return list(splitter.split(n_rows))


def _fit_predict(X_train, y_train, X_test, params: dict) -> np.ndarray:
    model = _model_factory()()
    if params:
        model.set_params(**params)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def sweep_cell(
    cell: Cell,
    df: pd.DataFrame,
    cached_pred: np.ndarray,
    gammas=GAMMA_GRID,
    tune_trials: int = EVAL_TUNE_TRIALS,
) -> tuple[list[dict], dict]:
    """
    Re-tunes one cell, verifies the refit against the cache, then sweeps gamma.

    Returns (rows, as_tuned_params). Raises if the reproduction check fails —
    a harness that cannot reproduce what Stage 0 recorded is measuring
    something else, and every number below it would be uninterpretable.
    """
    X, y = df[FEATURES], df[TARGET]
    train_idx, test_idx = fold_slices(len(X))[cell.fold]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    labelled = y_train.notna()
    X_train, y_train = X_train[labelled], y_train[labelled]

    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    finite = np.isfinite(y_test.to_numpy(dtype=float))
    X_test, y_true = X_test[finite], y_test.to_numpy(dtype=float)[finite]

    # The seeded, UNMODIFIED tuner on this fold's own training slice — the same
    # call walk_forward made, so this is the configuration that produced the
    # cached constant.
    params = tune(X_train, y_train, horizon=HORIZON_SESSIONS, n_trials=tune_trials)

    as_tuned = _fit_predict(X_train, y_train, X_test, params)
    if as_tuned.size != cached_pred.size:
        raise RuntimeError(
            f"{cell.ticker} fold {cell.fold}: refit produced {as_tuned.size} rows "
            f"against the cache's {cached_pred.size} — the fold boundary moved."
        )
    drift = float(np.max(np.abs(as_tuned - cached_pred)))
    if drift > REPRODUCTION_TOL:
        raise RuntimeError(
            f"{cell.ticker} fold {cell.fold}: refit drifts {drift:.3e} from the "
            f"cached prediction (tolerance {REPRODUCTION_TOL:.0e}). STOP — the "
            f"refit harness is not reproducing what Stage 0 recorded."
        )

    rows: list[dict] = []
    for gamma in gammas:
        pred = _fit_predict(X_train, y_train, X_test,
                            {**params, "gamma": float(gamma)})
        rows.append({
            "ticker": cell.ticker,
            "fold": cell.fold,
            "gamma": float(gamma),
            "as_tuned_gamma": float(params.get("gamma", 0.0)),
            "n_test_rows": int(y_true.size),
            "unique_fraction": unique_fraction(pred),
            "mode_share": mode_share(pred),
            "rank_ic": rank_ic(y_true, pred),
            "mae": float(np.mean(np.abs(y_true - pred))),
            "pred_sd": float(np.std(pred)),
        })

    return rows, {"drift": drift, "params": params,
                  "as_tuned_rank_ic": rank_ic(y_true, as_tuned),
                  "as_tuned_mae": float(np.mean(np.abs(y_true - as_tuned)))}


# ── verdict ───────────────────────────────────────────────────────────────────


def cell_verdict(cell_rows: list[dict], as_tuned: dict) -> dict:
    """
    Does this one cell support H1?

    The pre-registered rule is that constancy must fall at some swept gamma AND
    the within-fold rank IC must improve on the cell's own as-tuned value.

    Only the second is tested, because on any consistent input the first is
    implied by it: ``mode_share == 1.0`` means every prediction is the same
    float, which is exactly the condition under which ``rank_ic`` returns NaN.
    A cell that never splits therefore has no finite IC to improve with. An
    explicit ``and split_at`` conjunct was written first and removed once
    mutation testing showed no input could distinguish it — the same call as
    the fundamentals ``first_seen`` duplicate and the probe's tie check. Two
    guards covering one failure are one untestable guard.

    ``splits_anywhere`` is still REPORTED, because a reader wants to see it.
    """
    base_ic = as_tuned["as_tuned_rank_ic"]
    best = max(cell_rows, key=lambda r: (-1.0 if not np.isfinite(r["rank_ic"])
                                         else r["rank_ic"]))
    split_at = [r for r in cell_rows if r["mode_share"] < 1.0]
    ic_improved = (np.isfinite(best["rank_ic"])
                   and (not np.isfinite(base_ic) or best["rank_ic"] > base_ic))
    return {
        "ticker": cell_rows[0]["ticker"],
        "fold": cell_rows[0]["fold"],
        "splits_anywhere": bool(split_at),
        "best_gamma": best["gamma"],
        "best_rank_ic": best["rank_ic"],
        "as_tuned_rank_ic": base_ic,
        "as_tuned_mae": as_tuned["as_tuned_mae"],
        "best_mae": best["mae"],
        "ic_improved": bool(ic_improved),
        "supports_h1": bool(ic_improved),
    }


# ── driver ────────────────────────────────────────────────────────────────────


def run(limit: int | None = None, cache_path: str = CACHE_PATH,
        tune_trials: int = EVAL_TUNE_TRIALS) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = load_cells(cache_path)
    selected = select_cells(cells)
    if limit:
        selected = selected[:limit]

    z = np.load(cache_path, allow_pickle=True)
    tickers = [str(t) for t in z["tickers"]]
    offsets, y_pred, folds = z["offsets"], z["y_pred"], z["fold"]

    all_rows: list[dict] = []
    verdicts: list[dict] = []
    for n, cell in enumerate(selected, 1):
        t0 = time.time()
        i = tickers.index(cell.ticker)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        cached = y_pred[lo:hi][folds[lo:hi] == cell.fold]

        df = load_features_for_ticker(cell.ticker)
        rows, as_tuned = sweep_cell(cell, df, cached, tune_trials=tune_trials)
        all_rows.extend(rows)
        verdicts.append(cell_verdict(rows, as_tuned))
        print(f"  [{n}/{len(selected)}] {cell.ticker} fold {cell.fold}: "
              f"as-tuned gamma {as_tuned['params'].get('gamma', float('nan')):.3f}, "
              f"drift {as_tuned['drift']:.1e}, "
              f"supports H1 {verdicts[-1]['supports_h1']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    return pd.DataFrame(all_rows), pd.DataFrame(verdicts)


def render(sweep: pd.DataFrame, verdicts: pd.DataFrame) -> str:
    out: list[str] = []
    out.append("# Stage 2a, Step A — the gamma spot-check\n")
    out.append("Pre-registration: `docs/stage2a-preregistration.md`. "
               "Tool: `tools/stage2a_gamma_spotcheck.py`.\n")

    out.append("\n## Per cell, per gamma\n")
    out.append("| ticker | fold | rows | as-tuned gamma | gamma | "
               "unique frac | mode share | rank IC | MAE |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in sweep.iterrows():
        out.append(
            f"| {r['ticker']} | {int(r['fold'])} | {int(r['n_test_rows'])} | "
            f"{r['as_tuned_gamma']:.3f} | {r['gamma']:.1f} | "
            f"{r['unique_fraction']:.4f} | {r['mode_share']:.4f} | "
            f"{r['rank_ic']:+.4f} | {r['mae']:.5f} |"
        )

    out.append("\n## Verdict per cell\n")
    out.append("| ticker | fold | splits at some gamma | best gamma | "
               "as-tuned IC | best IC | as-tuned MAE | MAE there | supports H1 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in verdicts.iterrows():
        base = ("undefined" if not np.isfinite(r["as_tuned_rank_ic"])
                else f"{r['as_tuned_rank_ic']:+.4f}")
        out.append(
            f"| {r['ticker']} | {int(r['fold'])} | {r['splits_anywhere']} | "
            f"{r['best_gamma']:.1f} | {base} | {r['best_rank_ic']:+.4f} | "
            f"{r['as_tuned_mae']:.5f} | {r['best_mae']:.5f} | "
            f"**{r['supports_h1']}** |"
        )

    n_support = int(verdicts["supports_h1"].sum())
    out.append(f"\n**{n_support} of {len(verdicts)} cells support H1** "
               f"(pre-registered threshold: {SUPPORT_THRESHOLD}). "
               f"{'PROCEED to Step B.' if n_support >= SUPPORT_THRESHOLD else 'STOP at Step A.'}\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N selected cells (smoke test)")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print("Stage 2a Step A — gamma spot-check on fully-degenerate cells")
    sweep, verdicts = run(limit=args.limit, cache_path=args.cache)
    text = render(sweep, verdicts)
    print("\n" + text)

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.markdown}")
    if args.csv:
        sweep.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
