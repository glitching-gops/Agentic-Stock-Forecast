"""
tools/run_portfolio.py — what the ordering would have cost, in money.

    python tools/run_portfolio.py --synthetic     # validate the instrument FIRST
    python tools/run_portfolio.py
    python tools/run_portfolio.py --impact-bps 0 10 25 50

THREE OUTPUTS, IN THIS ORDER, AND THE ORDER IS THE POINT
---------------------------------------------------------
1. SYNTHETIC VALIDATION. Plant an edge of known size and confirm the net Sharpe
   rises with it. A null from an untested simulator is indistinguishable from a
   broken simulator, and this project has already been caught by a clean-looking
   null that described a learning rate rather than the data. `--synthetic` runs
   this alone; a full run does it first regardless.

2. THE NULL, QUANTIFIED. Every comparator's long-only and long-short book,
   gross and net of measured Indian costs, against NIFTY 50. Expected to be
   negative — nothing in this project clears either floor.

3. THE BREAK-EVEN EDGE. What rank IC would be needed to cover costs at this
   horizon and turnover. This is the reusable part: it turns every P2/P3
   statistic into an economic statement.

DEFLATED SHARPE UNDER TWO TRIAL COUNTS, and the larger is the honest one. Every
comparator here was tried on the same panel chasing the same target, which is
exactly the multiple testing Bailey & Lopez de Prado correct for. Reporting only
the small count would be choosing the flattering number.
"""

from __future__ import annotations

import argparse
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from pipeline.baselines import (                                 # noqa: E402
    BASELINES, FACTORS, baseline_feature_columns,
)
from pipeline.evaluation import (                                # noqa: E402
    PurgedPanelWalkForward, cross_sectional_report,
    deflated_sharpe_note, panel_walk_forward,
)
from pipeline.panel import (                                     # noqa: E402
    SCALE_FREE, TARGET, cross_sectional_zscore, load_panel,
)
from pipeline.portfolio import (                                 # noqa: E402
    CostModel, REBALANCES_PER_YEAR, break_even_ic, simulate,
    synthetic_predictions,
)
from pipeline.signals import HORIZON_SESSIONS                    # noqa: E402

#: EVERY configuration tried on this panel, for the honest deflation. Counted
#: from CLAUDE.md rather than guessed: 6 foundation-model configurations on the
#: excess target, 5 on the absolute target, the linear probe at 2 contexts,
#: LoRA, the ridge, the tree, valuation, news, regime, the Kronos pair, and the
#: min_train sweeps that were run over several of them.
TRIALS_ALL = 40

#: What P4 itself tries: comparators x {long_only, long_short} x impact levels.
#: The conventional count, and the flattering one.
TRIALS_P4 = 24

#: Comparators worth trading. `zero`, `train_mean` and `majority` are constants
#: and produce no ordering at all — `rebalance_books` reports them degenerate
#: rather than handing them an alphabetical portfolio.
TRADEABLE = ("beta_market", "pooled_xgb", "linear_factor", "momentum_20d",
             "reversal_5d", "news_factor", "regime_factor")


def _predictions(panel, name, min_train=500, n_folds=5):
    """
    One comparator's out-of-sample predictions, through the shared harness.

    REFUSES A COMPARATOR WHOSE DECLARED COLUMNS ARE ABSENT, rather than fitting
    whatever happens to be present. `LinearFactorModel.fit` keeps only the
    columns it finds in X, so a panel missing the news block silently produces
    a `news_factor` row IDENTICAL to `linear_factor` — which reads as "news
    does not help" instead of "news was never supplied". That exact defect
    already shipped once here as `linear_factor+val`, identical to five decimal
    places, and the first run of this tool reproduced it: news_factor and
    regime_factor came back byte-identical to linear_factor.
    """
    factory = BASELINES.get(name)
    if factory is None:
        if name == "pooled_xgb":
            # THE FUNCTION, not its result. `panel_walk_forward` calls the
            # factory once per fold to get a fresh unfitted model; handing it an
            # already-built XGBRegressor makes it call the model.
            from pipeline.baselines import _pooled_xgb_factory
            factory = _pooled_xgb_factory
        else:
            print(f"    {name}: SKIPPED — not in BASELINES")
            return None
    cols = baseline_feature_columns(name) or list(FACTORS)
    absent = [c for c in cols if c not in panel.columns]
    if absent:
        print(f"    {name}: SKIPPED — panel lacks {absent[:3]}"
              f"{' ...' if len(absent) > 3 else ''}. Fitting without them would "
              f"restate linear_factor.")
        return None
    splitter = PurgedPanelWalkForward(
        n_folds=n_folds, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=min_train)
    try:
        return panel_walk_forward(
            panel, cols, factory, splitter=splitter, target=TARGET,
            rebalance_every=HORIZON_SESSIONS).predictions
    except Exception as exc:                                     # noqa: BLE001
        print(f"    {name}: {type(exc).__name__}: {exc}")
        return None


def _annualise_list(returns) -> float:
    from pipeline.portfolio import _annualise
    return _annualise(float(np.mean(returns))) if len(returns) else float("nan")


def _benchmark_per_rebalance(panel) -> list[float] | None:
    """
    The equal-weighted panel return on each rebalance date.

    This IS `pipeline.baselines.MarketForecast`'s quantity — buy everything,
    hold 30 sessions — and it is the floor `market` plays in every other table
    here. Computed off the same rebalance grid so it lines up row for row.
    """
    from pipeline.panel import TARGET as _T
    rows = panel.dropna(subset=[_T])
    if rows.empty:
        return None
    dates = sorted(rows["date"].unique())[::HORIZON_SESSIONS]
    out = [float(rows[rows["date"] == d][_T].mean()) for d in dates]
    return [v for v in out if np.isfinite(v)]


def run_synthetic(seed: int = 7) -> bool:
    """Plants edges of known size; returns whether the simulator recovered them."""
    rng = np.random.default_rng(0)
    dates = [d.strftime("%Y-%m-%d")
             for d in pd.bdate_range("2019-01-01", periods=HORIZON_SESSIONS * 60)]
    tickers = [f"T{i:02d}.NS" for i in range(84)]
    truth = pd.DataFrame([
        {"date": d, "ticker": t, "y_true": float(rng.normal(0, 0.09))}
        for d in dates for t in tickers])
    truth["y_pred"] = 0.0

    print("=" * 78)
    print("1. CAN THIS SIMULATOR SEE AN EDGE THAT IS REALLY THERE?")
    print("=" * 78)
    print(f"  {'planted IC':>11}{'measured IC':>13}{'gross Sharpe':>14}"
          f"{'net Sharpe':>12}{'turnover':>10}")

    sharpes = []
    for ic in (0.0, 0.02, 0.05, 0.10, 0.20):
        p = synthetic_predictions(truth, ic, seed=seed)
        xs = cross_sectional_report(p, rebalance_every=HORIZON_SESSIONS)
        m = simulate(p, CostModel(), long_only=True).metrics()
        sharpes.append(m["net"]["sharpe"])
        print(f"  {ic:>11.2f}{xs.get('mean_rank_ic', float('nan')):>13.4f}"
              f"{m['gross']['sharpe']:>14.2f}{m['net']['sharpe']:>12.2f}"
              f"{m['mean_turnover']:>10.2f}")

    ok = all(b > a for a, b in zip(sharpes, sharpes[1:]))
    print(f"\n  net Sharpe is {'strictly increasing' if ok else 'NOT monotone'} "
          f"in the planted edge.")
    if not ok:
        print("  REFUSING to report real results: an instrument that cannot "
              "detect a\n  planted edge cannot be quoted as evidence of its "
              "absence.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="run the validation only and stop")
    ap.add_argument("--impact-bps", type=float, nargs="+", default=[0, 10, 25, 50])
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--record", action="store_true",
                    help="write the result into experiment_runs")
    args = ap.parse_args()

    if not run_synthetic():
        return 1
    if args.synthetic:
        return 0

    print()
    print("Loading panel ...")
    panel = load_panel()
    # The same enrichment `compare_baselines` performs, in the same order, so a
    # book traded here holds the names that table scored.
    from pipeline.news_features import NEWS_COLS
    from pipeline.panel import attach_news, attach_regime
    from pipeline.regime import REGIME_INTERACTIONS
    panel = attach_news(panel)
    panel["news_observed"] = panel["news_count_excess"].notna().astype(float)
    panel = attach_regime(panel)
    panel = cross_sectional_zscore(
        panel, SCALE_FREE + list(NEWS_COLS) + list(REGIME_INTERACTIONS))
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

    # THE FLOOR THE WHOLE TABLE IS READ AGAINST. An Indian equity panel over
    # 2019-2026 rose a lot; the top quintile of ANY ordering makes money in a
    # rising market, which is why `market` and `beta_market` are the floors in
    # every other table this project publishes. A return with no benchmark
    # beside it is the shape P0 was convened over.
    bench = _benchmark_per_rebalance(panel)

    base = CostModel()
    print(f"\n  cost model: {base.describe()}")
    print(f"  {REBALANCES_PER_YEAR:.1f} rebalances a year at a "
          f"{HORIZON_SESSIONS}-session horizon")

    print()
    print("=" * 78)
    print("2. THE NULL, QUANTIFIED — net of measured Indian costs")
    print("=" * 78)
    if bench is not None:
        print(f"  equal-weight panel (buy everything): "
              f"{_annualise_list(bench):+.2%}/yr — THE FLOOR\n")
    print(f"  {'comparator':16s}{'book':12s}{'net/yr':>9}{'vs floor':>10}"
          f"{'SR':>6}{'Sortino':>9}{'maxDD':>8}{'Calmar':>8}{'turn':>7}")

    results: dict[str, dict] = {}
    for name in TRADEABLE:
        preds = _predictions(panel, name, args.min_train)
        if preds is None or preds.empty:
            continue
        for long_only in (True, False):
            book = simulate(preds, base, long_only=long_only)
            m = book.metrics()
            if book.n_rebalances < 3:
                continue
            key = f"{name}:{'long_only' if long_only else 'long_short'}"
            results[key] = m
            # A MARKET-NEUTRAL BOOK IS NOT MEASURED AGAINST A LONG-ONLY FLOOR.
            # The long-short spread already has the market subtracted out by
            # construction, so differencing it against "buy everything" reports
            # a -20% that is nothing but the market it never held. Its own
            # excess IS its return, so the column is blank for it.
            vs = (m["net"]["annualised_return"] - _annualise_list(bench)
                  if (long_only and bench is not None) else float("nan"))
            n = m["net"]
            print(f"  {name:16s}{'long-only' if long_only else 'long-short':12s}"
                  f"{n['annualised_return']:>+9.2%}"
                  f"{(f'{vs:+.2%}' if np.isfinite(vs) else 'n/a'):>10}"
                  f"{n['sharpe']:>6.2f}{n['sortino']:>9.2f}"
                  f"{n['max_drawdown']:>8.1%}{n['calmar']:>8.2f}"
                  f"{m['mean_turnover']:>7.2f}")

    if not results:
        print("  no comparator produced a tradeable ordering")
        return 1

    print()
    print("=" * 78)
    print("3. THE BREAK-EVEN EDGE — what would have to be true")
    print("=" * 78)

    # spread per unit of IC, ESTIMATED FROM THIS PANEL rather than assumed.
    pairs = []
    for name in TRADEABLE:
        preds = _predictions(panel, name, args.min_train)
        if preds is None or preds.empty:
            continue
        xs = cross_sectional_report(preds, rebalance_every=HORIZON_SESSIONS)
        ic, spread = xs.get("mean_rank_ic"), xs.get("long_short_spread")
        if ic and spread and np.isfinite(ic) and np.isfinite(spread) and ic > 0:
            pairs.append(spread / ic)
    spread_per_ic = float(np.median(pairs)) if pairs else 1.0
    print(f"  measured on this panel: {spread_per_ic:.3f} of long-short spread "
          f"per unit of rank IC\n  (median over {len(pairs)} comparators with a "
          f"positive IC)")

    print(f"\n  {'impact':>8}{'round trip':>12}{'cost/yr @1.0 turn':>20}"
          f"{'break-even IC':>15}")
    for bps in args.impact_bps:
        cm = CostModel(impact=bps / 1e4)
        be = break_even_ic(cm, spread_per_ic=spread_per_ic, turnover=0.80)
        print(f"  {bps:>6.0f}bp{cm.round_trip:>12.4%}"
              f"{be['annual_cost_drag']:>20.2%}{be['break_even_rank_ic']:>15.4f}")

    print()
    print("  Against that bar, the comparators measured on this panel:")
    print("    beta_market +0.0464   pooled_xgb +0.0413   linear_factor +0.0249")
    print("    regime_factor +0.0277   news_factor +0.0083")

    print()
    print("=" * 78)
    print("4. DEFLATED SHARPE — the same number under two trial counts")
    print("=" * 78)
    best = max(results.items(), key=lambda kv: kv[1]["net"]["sharpe"])
    name, m = best
    n_obs = m["n_rebalances"]
    # PER-REBALANCE, NOT ANNUALISED. `deflated_sharpe_note` compares the
    # observed Sharpe against the expected MAXIMUM of `n_trials` draws from a
    # standard normal, which is a per-observation quantity. Handing it an
    # annualised Sharpe compares a yearly number against a per-rebalance null
    # and deflates every strategy into the floor for a units reason rather than
    # a statistical one — the same shape as reading `daily_IC` beside `reb_t`.
    per_rebalance = m["net"]["sharpe"] / np.sqrt(REBALANCES_PER_YEAR)

    # THE SPREAD OF SHARPES ACROSS THE TRIALS ACTUALLY RUN. The expected-maximum
    # term is expressed in THESE units, not raw Sharpe units, so leaving it at
    # 1.0 puts the hurdle at +1.98 for N=24 — a level no honest per-rebalance
    # Sharpe reaches, which deflates everything into the floor for a units
    # reason rather than a statistical one.
    trial_sharpes = [v["net"]["sharpe"] / np.sqrt(REBALANCES_PER_YEAR)
                     for v in results.values()
                     if np.isfinite(v["net"]["sharpe"])]
    sharpe_std = (float(np.std(trial_sharpes, ddof=1))
                  if len(trial_sharpes) > 2 else 1.0)

    print(f"  best net Sharpe: {name} at {m['net']['sharpe']:+.3f} annualised, "
          f"= {per_rebalance:+.3f} per rebalance over {n_obs} rebalances")
    print(f"  spread of per-rebalance Sharpes across the "
          f"{len(trial_sharpes)} books run here: {sharpe_std:.3f}\n")
    for label, n in (("P4's own variants (flattering)", TRIALS_P4),
                     ("every trial on this panel (HONEST)", TRIALS_ALL)):
        d = deflated_sharpe_note(n, per_rebalance, n_obs, sharpe_std=sharpe_std)
        print(f"  {label:36s} N={n:<4} expected-max {d['expected_max_sharpe_under_null']:+.3f} "
              f"-> deflated {d['deflated_statistic']:+.2f}  "
              f"{'CLEARS' if d['clears_null'] else 'does not clear'}")
    print("\n  The larger count is the honest one: every comparator above was "
          "tried on\n  the same panel chasing the same target.")

    if args.record:
        from pipeline.tracking import finish_run, start_run
        run_id = start_run("manual", sorted(panel["ticker"].unique()))
        finish_run(run_id, "OK", metrics={"portfolio": {
            "books": results, "spread_per_ic": spread_per_ic,
            "cost_round_trip": base.round_trip,
            "rebalances_per_year": REBALANCES_PER_YEAR}})
        print(f"\n  recorded to experiment_runs row {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
