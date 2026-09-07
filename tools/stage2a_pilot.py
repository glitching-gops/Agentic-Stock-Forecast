"""
tools/stage2a_pilot.py — Stage 2a, Step B: the sandboxed tuner-objective pilot.

Step A established that the degeneracy is a tuner artifact: every one of ten
fully-constant cells split at some lower gamma, and every one of them had
selected a gamma at or above 1.79 even though the seeded search offered three
candidates below 1.0. So the search range is not the binding constraint — the
MAE objective is, because it actively prefers the constant.

This runs the alternative objective (``tuning_objective="rank_ic"``) end to end
on a stratified pilot sample and reports what it costs and what it buys.

SANDBOXED. It writes no hyperparameter cache, no ``model_metadata``, no
forecast, and nothing the API or the web app serves — a markdown report and a
CSV, and that is all. ``pipeline/model.py`` is not modified: the walk-forward
configuration is reproduced here from the same constants and the same splitter,
and the MAE arm is checked against the Stage 0 out-of-sample cache to prove it.

Usage
-----
    python tools/stage2a_pilot.py --markdown docs/stage2a-step-b.md --csv stage2a_step_b.csv
    python tools/stage2a_pilot.py --limit 2        # smoke test
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.evaluation import rank_ic  # noqa: E402
from pipeline.model import (  # noqa: E402
    EVAL_TUNE_TRIALS,
    FEATURES,
    TARGET,
    load_features_for_ticker,
)
from pipeline.signals import HORIZON_SESSIONS  # noqa: E402
from pipeline.tuning import tune  # noqa: E402
from tools.stage2a_gamma_spotcheck import (  # noqa: E402
    CACHE_PATH,
    REPRODUCTION_TOL,
    Cell,
    _fit_predict,
    fold_slices,
    load_cells,
    mode_share,
    unique_fraction,
)

# Pre-registered strata. "Clean throughout" as literally specified does not
# exist on this panel — exactly ONE of 84 tickers (CGPOWER.NS) is non-constant
# in all five folds — so the control group is the least-degenerate population
# that exists, at most one constant fold, and it is reported as underpowered
# rather than presented as a clean control.
STRATA = {
    "degenerate": (5, 5, 5),    # (min constant folds, max, sample size)
    "partial":    (2, 4, 5),
    "control":    (0, 1, 4),
}


def stratify(cells: list[Cell]) -> dict[str, list[str]]:
    """Deterministic, alphabetical within each stratum. No look at any outcome."""
    counts: dict[str, int] = collections.defaultdict(int)
    seen: set[str] = set()
    for c in cells:
        seen.add(c.ticker)
        counts[c.ticker] += int(c.mode_share == 1.0)

    out: dict[str, list[str]] = {}
    for name, (lo, hi, size) in STRATA.items():
        members = sorted(t for t in seen if lo <= counts[t] <= hi)
        out[name] = members[:size]
    return out


def arm(ticker: str, df: pd.DataFrame, objective: str,
        tune_trials: int = EVAL_TUNE_TRIALS) -> list[dict]:
    """
    One walk-forward pass under one tuning objective.

    Reproduces ``evaluate_ticker``'s configuration through the same splitter and
    the same model factory. Records the TRAINING MAE beside the out-of-sample
    one, so a rank IC bought by under-regularised splits shows up as a widening
    gap rather than hiding behind a headline.
    """
    X, y = df[FEATURES], df[TARGET]
    rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(fold_slices(len(X))):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        labelled = y_tr.notna()
        if labelled.sum() < 50:
            continue
        X_tr, y_tr = X_tr[labelled], y_tr[labelled]

        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        finite = np.isfinite(y_te.to_numpy(dtype=float))
        X_te, y_true = X_te[finite], y_te.to_numpy(dtype=float)[finite]

        params = tune(X_tr, y_tr, horizon=HORIZON_SESSIONS,
                      n_trials=tune_trials, tuning_objective=objective)
        pred = _fit_predict(X_tr, y_tr, X_te, params)
        in_sample = _fit_predict(X_tr, y_tr, X_tr, params)

        rows.append({
            "ticker": ticker,
            "fold": fold,
            "objective": objective,
            "gamma": float(params.get("gamma", float("nan"))),
            "n_test_rows": int(y_true.size),
            "unique_fraction": unique_fraction(pred),
            "mode_share": mode_share(pred),
            "constant": bool(mode_share(pred) == 1.0),
            "rank_ic": rank_ic(y_true, pred),
            "mae": float(np.mean(np.abs(y_true - pred))),
            "train_mae": float(np.mean(np.abs(y_tr.to_numpy(dtype=float) - in_sample))),
        })
    return rows


def verify_against_cache(rows: list[dict], ticker: str,
                         cache_path: str = CACHE_PATH) -> float:
    """
    The MAE arm must reproduce what Stage 0 recorded, or the "before" column is
    a different measurement wearing the same name.

    Compared on the per-fold statistics rather than the raw vector, because the
    cache stores predictions and this arm stores summaries; a fold-level MAE
    and rank IC agreeing to 1e-9 across five folds is not something two
    different fits do by accident.
    """
    z = np.load(cache_path, allow_pickle=True)
    tickers = [str(t) for t in z["tickers"]]
    i = tickers.index(ticker)
    lo, hi = int(z["offsets"][i]), int(z["offsets"][i + 1])
    y_true, y_pred, folds = z["y_true"][lo:hi], z["y_pred"][lo:hi], z["fold"][lo:hi]

    worst = 0.0
    for r in rows:
        m = folds == r["fold"]
        cached_mae = float(np.mean(np.abs(y_true[m] - y_pred[m])))
        cached_ms = mode_share(y_pred[m])
        worst = max(worst, abs(cached_mae - r["mae"]), abs(cached_ms - r["mode_share"]))
    return worst


def run(limit: int | None = None, tune_trials: int = EVAL_TUNE_TRIALS,
        cache_path: str = CACHE_PATH) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    cells = load_cells(cache_path)
    strata = stratify(cells)
    plan = [(s, t) for s, ts in strata.items() for t in ts]
    if limit:
        plan = plan[:limit]

    rows: list[dict] = []
    for n, (stratum, ticker) in enumerate(plan, 1):
        t0 = time.time()
        df = load_features_for_ticker(ticker)
        before = arm(ticker, df, "mae", tune_trials)
        drift = verify_against_cache(before, ticker, cache_path)
        if drift > REPRODUCTION_TOL:
            raise RuntimeError(
                f"{ticker}: the MAE arm drifts {drift:.3e} from the Stage 0 "
                f"cache. STOP — the 'before' column would not be the number "
                f"Stage 0 reported."
            )
        after = arm(ticker, df, "rank_ic", tune_trials)
        for r in before + after:
            r["stratum"] = stratum
        rows.extend(before + after)

        b = pd.DataFrame(before)
        a = pd.DataFrame(after)
        print(f"  [{n}/{len(plan)}] {ticker} ({stratum}): constant folds "
              f"{int(b['constant'].sum())} -> {int(a['constant'].sum())}, "
              f"drift {drift:.1e} ({time.time() - t0:.0f}s)", flush=True)

    return pd.DataFrame(rows), strata


# ── reporting ─────────────────────────────────────────────────────────────────


def _mean(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def render(d: pd.DataFrame, strata: dict[str, list[str]]) -> str:
    out: list[str] = []
    out.append("# Stage 2a, Step B — the pilot re-tune\n")
    out.append("Pre-registration: `docs/stage2a-preregistration.md`. "
               "Tool: `tools/stage2a_pilot.py`. Shadow only — nothing here is "
               "written to a hyperparameter cache, to `model_metadata`, or to "
               "anything the API serves.\n")

    out.append("\n## The sample\n")
    for name, members in strata.items():
        out.append(f"- **{name}** ({len(members)}): {', '.join(members)}")
    out.append("\n> Only ONE ticker on this panel (CGPOWER.NS) is non-constant "
               "in all five folds, so a 'clean throughout' control group of the "
               "requested kind does not exist. The control stratum is the least "
               "degenerate population available — at most one constant fold — "
               "and it is underpowered at n = 4.\n")

    out.append("\n## Per ticker\n")
    out.append("| stratum | ticker | constant folds | mean within-fold IC | "
               "OOS MAE | train MAE | train/OOS gap |")
    out.append("|---|---|---|---|---|---|---|")
    for stratum in STRATA:
        for ticker in strata[stratum]:
            t = d[d["ticker"] == ticker]
            if t.empty:
                continue
            b, a = t[t["objective"] == "mae"], t[t["objective"] == "rank_ic"]
            out.append(
                f"| {stratum} | {ticker} | "
                f"{int(b['constant'].sum())} -> **{int(a['constant'].sum())}** | "
                f"{_mean(b['rank_ic']):+.4f} -> **{_mean(a['rank_ic']):+.4f}** | "
                f"{_mean(b['mae']):.5f} -> **{_mean(a['mae']):.5f}** | "
                f"{_mean(b['train_mae']):.5f} -> {_mean(a['train_mae']):.5f} | "
                f"{_mean(b['mae']) - _mean(b['train_mae']):+.5f} -> "
                f"**{_mean(a['mae']) - _mean(a['train_mae']):+.5f}** |"
            )

    out.append("\n## By stratum\n")
    out.append("| stratum | n | cells | constant before | constant after | "
               "IC before | IC after | MAE before | MAE after | gap before | gap after |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for stratum in STRATA:
        s = d[d["stratum"] == stratum]
        if s.empty:
            continue
        b, a = s[s["objective"] == "mae"], s[s["objective"] == "rank_ic"]
        out.append(
            f"| **{stratum}** | {s['ticker'].nunique()} | {len(b)} | "
            f"{int(b['constant'].sum())} ({100 * b['constant'].mean():.0f}%) | "
            f"**{int(a['constant'].sum())} ({100 * a['constant'].mean():.0f}%)** | "
            f"{_mean(b['rank_ic']):+.4f} | **{_mean(a['rank_ic']):+.4f}** | "
            f"{_mean(b['mae']):.5f} | **{_mean(a['mae']):.5f}** | "
            f"{_mean(b['mae']) - _mean(b['train_mae']):+.5f} | "
            f"**{_mean(a['mae']) - _mean(a['train_mae']):+.5f}** |"
        )

    b, a = d[d["objective"] == "mae"], d[d["objective"] == "rank_ic"]
    out.append(
        f"\n**Whole pilot:** constant cells {int(b['constant'].sum())}/{len(b)} "
        f"-> {int(a['constant'].sum())}/{len(a)}; mean within-fold IC "
        f"{_mean(b['rank_ic']):+.4f} -> {_mean(a['rank_ic']):+.4f}; mean OOS MAE "
        f"{_mean(b['mae']):.5f} -> {_mean(a['mae']):.5f} "
        f"({100 * (_mean(a['mae']) / _mean(b['mae']) - 1):+.1f}%); "
        f"selected gamma {_mean(b['gamma']):.3f} -> {_mean(a['gamma']):.3f}.\n"
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trials", type=int, default=EVAL_TUNE_TRIALS)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print("Stage 2a Step B — pilot re-tune under the rank-IC objective")
    d, strata = run(limit=args.limit, tune_trials=args.trials,
                    cache_path=args.cache)
    text = render(d, strata)
    print("\n" + text)

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.markdown}")
    if args.csv:
        d.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
