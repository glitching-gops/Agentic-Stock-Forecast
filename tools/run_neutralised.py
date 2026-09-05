"""
tools/run_neutralised.py — Phase 5. Is there anything here that is not beta?

Runs, in this order and refusing to continue if the first fails:

  1. THE CALIBRATION GATE. `beta_market` scored against its own beta-neutralised
     residual must be indistinguishable from zero. A null from an unvalidated
     neutraliser is indistinguishable from a null from a broken one, so this
     earns the right to print everything below it. Same contract as
     `run_portfolio.run_synthetic`.
  2. THE RESIDUAL TABLE (§1). Every comparator's rank IC against the
     beta-neutralised target, on both bases and both variants.
  3. THE HEDGED BOOKS (§2). The same orderings held beta-neutral, in money.
  4. THE PER-FOLD NULL BAND (§3). Was fold 0's apparent signal ever surprising
     against fold 0's OWN null?

Usage
-----
    $py tools/run_neutralised.py                    # everything
    $py tools/run_neutralised.py --skip-books       # sections 1, 2, 4
    $py tools/run_neutralised.py --min-train 380    # a sweep cell

READ ONLY. Nothing here writes to the database.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.baselines import FLOORS                                  # noqa: E402
from pipeline.neutralise import (                                      # noqa: E402
    NeutralisationRefused,
    calibration_gate,
    fold_null_band,
    neutralise,
    residual_report,
)
from pipeline.news_features import NEWS_COLS                            # noqa: E402
from pipeline.panel import (                                           # noqa: E402
    SCALE_FREE, TARGET, attach_news, attach_regime, cross_sectional_zscore,
    load_panel,
)
from pipeline.regime import REGIME_INTERACTIONS                        # noqa: E402
from pipeline.portfolio import CostModel, simulate_hedged              # noqa: E402
from pipeline.regime import rolling_beta                               # noqa: E402
from pipeline.signals import HORIZON_SESSIONS                          # noqa: E402
from tools.run_portfolio import _predictions                           # noqa: E402

#: Scored on the residual. `beta_market` is included deliberately — it is the
#: calibration case, and seeing it at ~0 in the table itself is worth more than
#: seeing it only in a gate the reader has to trust.
COMPARATORS = ["beta_market", "momentum_20d", "reversal_5d", "linear_factor",
               "regime_factor", "news_factor", "pooled_xgb"]


#: Walk-forward is the expensive step and every comparator is needed twice —
#: once for the residual table, once for its hedged book. Caching also removes
#: any chance the two sections score DIFFERENT predictions for the same name,
#: which would render fine and be wrong.
_CACHE: dict[tuple, pd.DataFrame | None] = {}


def _cached(panel: pd.DataFrame, name: str, min_train: int, n_folds: int):
    key = (name, min_train, n_folds)
    if key not in _CACHE:
        _CACHE[key] = _predictions(panel, name, min_train, n_folds)
    return _CACHE[key]


def _beta_market_basis(panel: pd.DataFrame, min_train: int,
                       n_folds: int = 5) -> pd.DataFrame:
    """
    The primary regressor: the floor's OWN out-of-sample prediction.

    Within a date this is `beta_i * mu_market` with mu constant, i.e. an affine
    function of beta — and OLS residuals are invariant to that, so this is
    numerically identical to residualising on beta while being complementary to
    the floor by construction rather than by a second estimate that could
    disagree with it.
    """
    preds = _cached(panel, "beta_market", min_train, n_folds)
    if preds is None:
        raise NeutralisationRefused("beta_market produced no predictions")
    return preds.rename(columns={"y_pred": "beta_basis"})[
        ["date", "ticker", "beta_basis"]]


def _rolling_basis(panel: pd.DataFrame) -> pd.DataFrame:
    """The robustness regressor: a trailing beta a live book could have used."""
    b = rolling_beta(panel)
    return b.rename(columns={"beta": "beta_basis"})[["date", "ticker", "beta_basis"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--skip-books", action="store_true")
    ap.add_argument("--regime", action="store_true",
                    help="the PRE-REGISTERED regime-conditional test (section 4)")
    ap.add_argument("--sweep", action="store_true",
                    help="attack any positive residual across min_train and by "
                         "fold, per the project's standing skeptic checks")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    print("Loading panel ...")
    panel = load_panel()
    if panel.empty:
        print("REFUSING: empty panel")
        return 1

    # THE TABLE'S OWN PREPROCESSING, in the table's own order. A script that
    # attacks a result and skips a step of the pipeline that produced it is
    # measuring a different panel: the recorded instance here read pooled_xgb at
    # +0.99 against a table saying +2.42, purely because a sweep omitted
    # `cross_sectional_zscore`. `beta_market` was identical across both runs —
    # it reads only (date, ticker) and the target — which is what located it.
    panel = attach_news(panel)
    panel["news_observed"] = panel["news_count_excess"].notna().astype(float)
    panel = attach_regime(panel)
    raw = panel.copy()                       # rolling_beta needs raw `close`
    panel = cross_sectional_zscore(
        panel, SCALE_FREE + list(NEWS_COLS) + list(REGIME_INTERACTIONS))
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique():,} dates, target={TARGET}, floors={FLOORS}")

    out: dict = {"min_train": args.min_train, "n_folds": args.n_folds}

    # ---------------------------------------------------------------- 1. gate
    print("\n[1/4] CALIBRATION GATE — beta_market against its own residual")
    try:
        basis = _beta_market_basis(panel, args.min_train, args.n_folds)
        bm = _cached(panel, "beta_market", args.min_train, args.n_folds)
        gate = calibration_gate(bm, basis)
    except NeutralisationRefused as exc:
        print(f"  REFUSED: {exc}")
        return 1

    out["gate"] = gate
    print(f"  residual rank IC {gate['observed_residual_ic']:+.4f} "
          f"(t {gate['observed_residual_t']:+.2f}) over "
          f"{gate['n_rebalances']} rebalances")
    print(f"  beta channel R^2 {gate['beta_channel_r2']:.3f}  "
          f"<- how much of the within-date target beta explains")
    if not gate["passed"]:
        print(f"\n  GATE FAILED. {gate['note']}")
        print("  Refusing to print residual results.")
        return 1
    print("  PASSED — the neutralisation removes what it claims to remove.")

    # ------------------------------------------------------------ 2. residual
    print(f"\n[2/4] RESIDUAL RANK IC  (min_train={args.min_train})")
    bases = {"beta_market": basis}
    try:
        bases["rolling_beta"] = _rolling_basis(raw)
    except Exception as exc:                                     # noqa: BLE001
        print(f"  rolling_beta unavailable ({type(exc).__name__}: {exc}); "
              f"primary basis only")

    rows, trials = [], 0
    for name in COMPARATORS:
        preds = _cached(panel, name, args.min_train, args.n_folds)
        if preds is None:
            continue
        plain = residual_report(preds)                      # un-neutralised
        for bname, bframe in bases.items():
            for neut_pred in (False, True):
                try:
                    r = neutralise(preds, bframe, basis=bname,
                                   neutralise_prediction=neut_pred)
                except NeutralisationRefused as exc:
                    print(f"  {name}/{bname}: REFUSED — {exc}")
                    continue
                rep = residual_report(r.frame)
                trials += 1
                rows.append({
                    "comparator": name, "basis": bname,
                    "neutralised_prediction": neut_pred,
                    "raw_rank_ic": plain.get("residual_rank_ic", float("nan")),
                    "raw_t": plain.get("residual_ic_t", float("nan")),
                    "residual_rank_ic": rep.get("residual_rank_ic", float("nan")),
                    "residual_t": rep.get("residual_ic_t", float("nan")),
                    "n_rebalances": rep.get("n_rebalances", 0),
                })

    table = pd.DataFrame(rows)
    out["residual"] = rows
    out["trials_added"] = trials
    if table.empty:
        print("  no comparator produced a residual table")
        return 1

    for bname in table["basis"].unique():
        for neut in (False, True):
            block = table[(table["basis"] == bname)
                          & (table["neutralised_prediction"] == neut)]
            if block.empty:
                continue
            what = "target+prediction" if neut else "target only"
            print(f"\n  basis={bname}, neutralised: {what}")
            print(f"    {'comparator':<16}{'raw IC':>10}{'raw t':>8}"
                  f"{'resid IC':>11}{'resid t':>9}{'kept':>7}")
            for _, r in block.iterrows():
                kept = (r["residual_rank_ic"] / r["raw_rank_ic"]
                        if r["raw_rank_ic"] not in (0, np.nan)
                        and np.isfinite(r["raw_rank_ic"]) else float("nan"))
                print(f"    {r['comparator']:<16}{r['raw_rank_ic']:>+10.4f}"
                      f"{r['raw_t']:>+8.2f}{r['residual_rank_ic']:>+11.4f}"
                      f"{r['residual_t']:>+9.2f}"
                      f"{(f'{kept:.0%}' if np.isfinite(kept) else '   n/a'):>7}")

    print("\n  NOTE: no MAE column, deliberately. A residual has smaller variance")
    print("  than its target, so an MAE against it is not comparable to any MAE")
    print("  in this project's other tables.")

    # --------------------------------------------------------------- 3. books
    if not args.skip_books:
        print("\n[3/4] BETA-NEUTRAL BOOKS — the same orderings, in money")
        try:
            rb = rolling_beta(raw)[["date", "ticker", "beta"]]
        except Exception as exc:                                 # noqa: BLE001
            print(f"  SKIPPED — rolling_beta failed: {type(exc).__name__}: {exc}")
            rb = None

        if rb is not None:
            books = []
            print(f"    {'comparator':<16}{'gross/reb':>12}{'net/reb':>10}"
                  f"{'Sharpe':>9}{'turn':>7}{'n':>5}")
            for name in COMPARATORS:
                preds = _cached(panel, name, args.min_train, args.n_folds)
                if preds is None:
                    continue
                book = simulate_hedged(preds, rb, CostModel())
                m = book.metrics()
                if book.n_rebalances < 3:
                    continue
                trials += 1
                books.append({"comparator": name, **m["net"],
                              "n_rebalances": book.n_rebalances,
                              "mean_turnover": m["mean_turnover"]})
                print(f"    {name:<16}{np.mean(book.gross_returns):>+12.4f}"
                      f"{np.mean(book.net_returns):>+10.4f}"
                      f"{m['net']['sharpe']:>+9.2f}"
                      f"{m['mean_turnover']:>7.2f}{book.n_rebalances:>5}")
            out["books"] = books
            print("\n  These are SPREADS, not self-financing portfolios: no 'vs floor',")
            print("  and their drawdown is not the quantity that word usually names.")

    # ---------------------------------------------------------- 4. null bands
    print(f"\n[4/4] PER-FOLD NULL BAND — {args.draws} within-date permutations")
    ref = _cached(panel, "linear_factor", args.min_train, args.n_folds)
    if ref is not None and "fold" in ref.columns:
        bands = fold_null_band(ref, n_draws=args.draws)
        out["fold_null"] = bands
        print(f"    {'fold':<6}{'names/date':>12}{'rebals':>8}"
              f"{'disp':>9}{'null sd':>10}{'|null| p95':>12}")
        for fold, b in bands.items():
            print(f"    {fold:<6}{b['median_names_per_date']:>12.0f}"
                  f"{b['n_rebalances']:>8}{b['target_dispersion']:>9.4f}"
                  f"{b['null_sd']:>10.4f}{b['null_p95_abs']:>12.4f}")
        print("\n  A WIDER band in the early folds is the mechanism behind three")
        print("  retired results. Dispersion is printed for completeness and is NOT")
        print("  the explanation: rank IC is invariant to scaling the target.")
    else:
        print("  SKIPPED — no fold column available")

    # ------------------------------------------------------------- 5. attack
    if args.sweep:
        print("\n[5/5] THE STANDING SKEPTIC CHECKS")
        print("  A result at ONE (min_train, fold) cell is not a result. The")
        print("  retired valuation finding scored +3.32 at min_train 380 and")
        print("  +1.00 at 500 on IDENTICAL rows; pooled_xgb's own RAW t has")
        print("  already run +2.47/+1.12/+1.15/+2.42/+0.86/+1.35 across this grid.")

        grid = [380, 420, 460, 500, 540, 580]
        watch = ["beta_market", "linear_factor", "regime_factor", "pooled_xgb"]

        print("\n  min_train sweep - residual t (basis=beta_market, target only)")
        print(f"    {'comparator':<16}" + "".join(f"{g:>8}" for g in grid))
        sweep = {}
        for name in watch:
            cells = []
            for mt in grid:
                pr = _cached(panel, name, mt, args.n_folds)
                if pr is None:
                    cells.append(float("nan"))
                    continue
                try:
                    bs = _beta_market_basis(panel, mt, args.n_folds)
                    rr = residual_report(
                        neutralise(pr, bs, basis="beta_market").frame)
                except NeutralisationRefused:
                    cells.append(float("nan"))
                    continue
                cells.append(rr.get("residual_ic_t", float("nan")))
            sweep[name] = dict(zip(grid, cells))
            print(f"    {name:<16}" + "".join(
                (f"{c:>+8.2f}" if np.isfinite(c) else f"{'n/a':>8}")
                for c in cells))
        out["sweep"] = sweep

        print(f"\n  per-fold residual rank IC (min_train={args.min_train})")
        basis_d = _beta_market_basis(panel, args.min_train, args.n_folds)
        perfold = {}
        for name in watch:
            pr = _cached(panel, name, args.min_train, args.n_folds)
            if pr is None or "fold" not in pr.columns:
                continue
            r = neutralise(pr, basis_d, basis="beta_market").frame
            cells = []
            for f in sorted(r["fold"].dropna().unique()):
                rep = residual_report(r[r["fold"] == f])
                cells.append(rep.get("residual_rank_ic", float("nan")))
            perfold[name] = cells
            print(f"    {name:<16}" + "".join(
                (f"{c:>+10.4f}" if np.isfinite(c) else f"{'n/a':>10}")
                for c in cells))
        out["per_fold"] = perfold
        print("\n  A result concentrated in fold 0 is the shape that already")
        print("  retired valuation (+3.32), LoRA (+2.37) and pooled_xgb (+2.42).")

    # ------------------------------------------- 6. regime-conditional (P4 §4)
    #
    # PRE-REGISTERED, and the declarations are here in the source rather than in
    # a message: SPLIT COLUMN `regime_vol`, THRESHOLD its own EXPANDING median up
    # to each date (causal — a full-sample median would be F2 in miniature), and
    # the bar is the project's standing one: residual reb_IC positive with t > 2,
    # surviving the min_train sweep. Fixed before the first number was seen.
    if args.regime:
        print("\n[6/6] REGIME-CONDITIONAL — pre-registered, on the residual")
        print("  split=regime_vol, threshold=own expanding median (causal),")
        print("  bar=residual t > 2 AND surviving the min_train sweep.")
        from pipeline.regime import compute_market_state

        state = compute_market_state(raw).sort_values("date")
        state["_med"] = state["regime_vol"].expanding(min_periods=60).median()
        state["_hi"] = state["regime_vol"] > state["_med"]
        regime_of = dict(zip(state["date"].astype(str), state["_hi"]))

        basis_r = _beta_market_basis(panel, args.min_train, args.n_folds)
        print(f"\n    {'comparator':<16}{'high-vol IC':>13}{'t':>7}{'n':>5}"
              f"{'low-vol IC':>13}{'t':>7}{'n':>5}")
        reg = {}
        for name in ["beta_market", "linear_factor", "regime_factor", "pooled_xgb"]:
            pr = _cached(panel, name, args.min_train, args.n_folds)
            if pr is None:
                continue
            r = neutralise(pr, basis_r, basis="beta_market").frame
            hi_mask = r["date"].astype(str).map(regime_of)
            cells = {}
            for label, sub in (("high", r[hi_mask.fillna(False)]),
                               ("low", r[~hi_mask.fillna(True)])):
                cells[label] = residual_report(sub) if not sub.empty else {}
            reg[name] = cells
            h, lo = cells["high"], cells["low"]
            print(f"    {name:<16}"
                  f"{h.get('residual_rank_ic', float('nan')):>+13.4f}"
                  f"{h.get('residual_ic_t', float('nan')):>+7.2f}"
                  f"{h.get('n_rebalances', 0):>5}"
                  f"{lo.get('residual_rank_ic', float('nan')):>+13.4f}"
                  f"{lo.get('residual_ic_t', float('nan')):>+7.2f}"
                  f"{lo.get('n_rebalances', 0):>5}")
        out["regime"] = reg
        print("\n  Each half holds ~half the rebalances, so these t-statistics")
        print("  rest on ~30 independent windows apiece and every cell here is")
        print("  another trial against the cumulative deflation count.")

    out["trials_added"] = trials
    print(f"\nTRIALS ADDED BY THIS RUN: {trials}. Carry this into the deflated")
    print("Sharpe count cumulatively (P4 stood at N=40); a phase that slices the")
    print("sample further must pay for the slicing.")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str),
                                   encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
