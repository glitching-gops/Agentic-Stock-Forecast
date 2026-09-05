"""
pipeline/portfolio.py — what the ordering would have cost, after Indian costs.

WHAT THIS IS AND IS NOT
------------------------
P0 removed the leaderboard and the portfolio from the PRODUCT, because ranking
84 names needs 84 comparable numbers and the evidence gate produces 3. That
removal stands. This is a MEASUREMENT: it converts a rank IC into rupees so the
size of the null can be read by someone who does not think in rank ICs. It
publishes no holdings, recommends nothing, and its headline output is expected
to be negative.

The distinction matters enough to be enforced rather than asserted: nothing
here returns a current or forward-looking book. Every book it builds is dated,
historical, and scored against what actually happened next.

WHY A BACKTEST OF THIS MODEL IS A BACKTEST OF NOISE
----------------------------------------------------
Nothing clears either floor. `beta_market` — which sorts by beta and holds no
company-specific view — scores reb_IC +0.0464 and out-ranks everything;
`pooled_xgb` reaches t +2.51 and fails both floors. So the question is not
"what did it return". It is:

  1. what the absence of a signal costs, in money, after real costs;
  2. what rank IC WOULD be needed to clear those costs (`break_even_ic`);
  3. whether this simulator could detect an edge if one existed
     (`synthetic_predictions`, and it is run BEFORE any real number is quoted).

THE COST MODEL IS MEASURED, NOT ASSUMED
----------------------------------------
Zerodha equity-delivery schedule, NSE, verified 2026-09-04. STT is the term
that dominates and it applies to BOTH sides, which is the fact most cost
assumptions get wrong:

    brokerage        Rs 0        (discount broker, delivery)
    STT              0.100%      BOTH buy and sell
    stamp duty       0.015%      buy only
    exchange txn     0.00307%    both
    SEBI             Rs 10/crore both
    GST              18% on (brokerage + SEBI + exchange)

That is ~0.222% round trip. THE HORIZON IS 30 SESSIONS, NOT 30 DAYS, so a year
holds 252/30 = 8.4 rebalances, not 12 — an easy slip that overstates the drag by
40%. At full turnover that is ~1.9% a year before impact; at the ~0.6 turnover
these orderings actually generate, ~1.1%.

IMPACT COST IS SWEPT, NOT ASSUMED. It is the term that actually varies by name
and the one no fee schedule can tell you, so `simulate` takes it as a parameter
and the caller reports a table across values. A single assumed impact is a
result at one cell, which is the error that produced the retired valuation
finding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.evaluation import rebalance_books
from pipeline.signals import HORIZON_SESSIONS

logger = logging.getLogger(__name__)

#: Sessions in a trading year, for annualising rebalance-frequency statistics.
SESSIONS_PER_YEAR = 252

#: Rebalances a year at the model's own horizon. ~8.4 at 30 sessions.
REBALANCES_PER_YEAR = SESSIONS_PER_YEAR / HORIZON_SESSIONS


@dataclass(frozen=True)
class CostModel:
    """
    Indian equity-delivery costs, as fractions of traded value.

    Defaults are the measured Zerodha/NSE schedule (2026-09-04). `impact` is
    NOT part of that schedule and defaults to zero so a caller must choose it
    deliberately and sweep it — see the module docstring.
    """

    stt_buy: float = 0.001000          # 0.100%, and it applies to BOTH sides
    stt_sell: float = 0.001000
    stamp_duty_buy: float = 0.000150   # 0.015%, buy only
    exchange: float = 0.0000307        # 0.00307%
    sebi: float = 0.0000010            # Rs 10 per crore
    gst_rate: float = 0.18             # on brokerage + SEBI + exchange
    brokerage: float = 0.0             # Rs 0 on delivery at a discount broker
    impact: float = 0.0                # basis points of slippage, caller's call

    def one_way(self, side: str) -> float:
        """Cost of trading one rupee of notional, buying or selling."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        stt = self.stt_buy if side == "buy" else self.stt_sell
        stamp = self.stamp_duty_buy if side == "buy" else 0.0
        fees = self.brokerage + self.sebi + self.exchange
        return stt + stamp + fees + self.gst_rate * fees + self.impact

    @property
    def round_trip(self) -> float:
        return self.one_way("buy") + self.one_way("sell")

    def describe(self) -> str:
        return (f"{self.round_trip * 100:.4f}% round trip "
                f"({self.impact * 1e4:.0f} bps impact included)")


@dataclass
class BookResult:
    """One strategy's realised record. Every field is historical."""

    name: str
    n_rebalances: int
    gross_returns: list[float] = field(default_factory=list)
    net_returns: list[float] = field(default_factory=list)
    turnover: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    n_no_ordering: int = 0

    def metrics(self, benchmark: list[float] | None = None) -> dict:
        return _metrics(self, benchmark)


def _annualise(mean_per_rebalance: float) -> float:
    """Compounds a per-rebalance log return to a yearly figure."""
    return float(np.expm1(mean_per_rebalance * REBALANCES_PER_YEAR))


def _drawdown(returns: list[float]) -> float:
    """Worst peak-to-trough on the compounded curve, as a positive fraction."""
    if not returns:
        return float("nan")
    curve = np.cumsum(np.asarray(returns, dtype=float))   # log space
    peak = np.maximum.accumulate(curve)
    return float(np.max(np.expm1(peak - curve))) if len(curve) else float("nan")


def _metrics(book: BookResult, benchmark: list[float] | None = None) -> dict:
    """
    Risk and return, computed on the NON-OVERLAPPING rebalance returns only.

    Never on a daily grid. Consecutive dates share 29 of their 30 forward
    sessions, so a daily Sharpe here would be inflated by roughly the square
    root of the overlap — the same trap `audit_benchmarks.py` recorded and the
    reason `reb_t` exists beside `daily_IC`.
    """
    out: dict = {"name": book.name, "n_rebalances": book.n_rebalances,
                 "n_no_ordering": book.n_no_ordering}
    if book.n_rebalances < 3:
        out["note"] = "too few rebalances to compute risk statistics"
        return out

    for label, series in (("gross", book.gross_returns), ("net", book.net_returns)):
        r = np.asarray(series, dtype=float)
        sd = float(r.std(ddof=1))
        ann_ret = _annualise(float(r.mean()))
        ann_vol = sd * np.sqrt(REBALANCES_PER_YEAR)
        sharpe = (float(r.mean()) / sd * np.sqrt(REBALANCES_PER_YEAR)
                  if sd > 0 else float("nan"))
        downside = r[r < 0]
        dsd = float(downside.std(ddof=1)) if len(downside) > 2 else float("nan")
        sortino = (float(r.mean()) / dsd * np.sqrt(REBALANCES_PER_YEAR)
                   if dsd and np.isfinite(dsd) and dsd > 0 else float("nan"))
        mdd = _drawdown(list(r))
        out[label] = {
            "annualised_return": ann_ret,
            "annualised_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": mdd,
            "calmar": float(ann_ret / mdd) if mdd and mdd > 0 else float("nan"),
            "hit_rate": float((r > 0).mean()),
        }

    out["mean_turnover"] = float(np.mean(book.turnover)) if book.turnover else 0.0
    out["cost_drag_annual"] = out["gross"]["annualised_return"] - \
        out["net"]["annualised_return"]

    if benchmark and len(benchmark) == book.n_rebalances:
        b = np.asarray(benchmark, dtype=float)
        n = np.asarray(book.net_returns, dtype=float)
        out["benchmark"] = {
            "annualised_return": _annualise(float(b.mean())),
            "excess_vs_benchmark_annual": _annualise(float((n - b).mean())),
            "beat_rate": float((n > b).mean()),
        }
    return out


def _turn(new: set[str], old: set[str]) -> float:
    """
    Fraction of a book doing a ROUND TRIP, the unit costs are charged in.

    A rebalance that replaces half the names sells 50% and buys 50% — one round
    trip on half the book, so 0.5. Opening the book buys 100% and sells nothing,
    which is half a round trip, so 0.5 as well; reporting it as 1.0 there would
    double-charge the first rebalance.

    AT MODULE SCOPE BECAUSE TWO SIMULATORS CHARGE COSTS. `simulate` and
    `simulate_hedged` must agree on what a round trip is, and two copies of this
    rule would drift with nothing to see — both books would still render, and
    only the cost drag would silently disagree. Same argument that pulled
    `rebalance_books` out of `cross_sectional_report`.
    """
    if not new and not old:
        return 0.0
    entered = len(new - old)
    exited = len(old - new)
    base = max(len(new), len(old), 1)
    return (entered + exited) / (2.0 * base)


def simulate(predictions: pd.DataFrame, costs: CostModel | None = None,
             rebalance_every: int = HORIZON_SESSIONS, quantiles: int = 5,
             long_only: bool = True) -> BookResult:
    """
    Trades one comparator's ordering and returns its realised record.

    `predictions` is what `panel_walk_forward` returns: date, ticker, y_pred,
    y_true, all out-of-sample. Books come from `evaluation.rebalance_books`, so
    the names traded here are exactly the names `cross_sectional_report` scored.

    TURNOVER IS COMPUTED FROM THE HOLDINGS THAT ACTUALLY CHANGED, name by name,
    not assumed to be 100% each rebalance. A name in the top quintile on two
    consecutive dates is held, not sold and re-bought, and charging it twice
    would overstate the drag by roughly the overlap — which for a 30-session
    momentum-ish ordering is substantial.
    """
    costs = costs or CostModel()
    name = ("long_only" if long_only else "long_short") + f"@{quantiles}"
    book = BookResult(name=name, n_rebalances=0)

    held: set[str] = set()          # long leg
    held_short: set[str] = set()

    for rb in rebalance_books(predictions, rebalance_every, quantiles):
        if rb.degenerate:
            book.n_no_ordering += 1
            continue

        longs = set(rb.top["ticker"])
        shorts = set() if long_only else set(rb.bottom["ticker"])

        gross = float(rb.top["y_true"].mean())
        if not long_only:
            gross -= float(rb.bottom["y_true"].mean())

        turn = _turn(longs, held)
        if not long_only:
            turn = 0.5 * (turn + _turn(shorts, held_short))

        # Entering costs a buy, exiting costs a sell; a full round trip on the
        # replaced fraction.
        cost = turn * costs.round_trip
        book.gross_returns.append(gross)
        book.net_returns.append(gross - cost)
        book.turnover.append(turn)
        book.dates.append(rb.date)
        book.n_rebalances += 1
        held, held_short = longs, shorts

    return book


def break_even_ic(costs: CostModel | None = None, spread_per_ic: float = 1.0,
                  turnover: float = 1.0) -> dict:
    """
    What rank IC would be needed for the ordering to cover its own costs.

    THE MOST REUSABLE THING HERE, because it converts every statistic this
    project has reported into an economic statement. `spread_per_ic` is the
    measured long-short spread produced per unit of rank IC on THIS panel — it
    is not a constant of nature and must be estimated from the data rather than
    assumed, which `tools/run_portfolio.py` does before calling this.
    """
    costs = costs or CostModel()
    per_rebalance_cost = turnover * costs.round_trip
    needed = per_rebalance_cost / spread_per_ic if spread_per_ic > 0 else float("nan")
    return {
        "round_trip_cost": costs.round_trip,
        "assumed_turnover": turnover,
        "cost_per_rebalance": per_rebalance_cost,
        "annual_cost_drag": float(np.expm1(per_rebalance_cost * REBALANCES_PER_YEAR)),
        "spread_per_unit_ic": spread_per_ic,
        "break_even_rank_ic": needed,
    }


def synthetic_predictions(truth: pd.DataFrame, target_ic: float,
                          seed: int = 0) -> pd.DataFrame:
    """
    Predictions of KNOWN strength, for validating the simulator before use.

    A null from an untested simulator is indistinguishable from a broken one.
    This is the `series_zero` pattern: that comparator reproduced the `zero`
    baseline to the row, which is what stopped the adapter being what was
    measured. Here, planting an edge of known size and confirming the net Sharpe
    rises with it is what earns the right to quote a null on real data.

    Blends the true label with noise inside each date, so the planted signal is
    CROSS-SECTIONAL — which is the only kind this panel's metrics can see.
    """
    rng = np.random.default_rng(seed)
    out = truth.copy()
    parts = []
    for _, day in out.groupby("date", sort=False):
        y = day["y_true"].to_numpy(dtype=float)
        z = (y - np.nanmean(y)) / (np.nanstd(y) or 1.0)
        noise = rng.standard_normal(len(day))
        blended = target_ic * z + np.sqrt(max(1.0 - target_ic ** 2, 0.0)) * noise
        d = day.copy()
        d["y_pred"] = blended
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def simulate_hedged(predictions: pd.DataFrame, beta: pd.DataFrame,
                    costs: CostModel | None = None,
                    rebalance_every: int = HORIZON_SESSIONS,
                    quantiles: int = 5) -> BookResult:
    """
    The same ordering, held BETA-NEUTRAL. The money-space twin of
    `pipeline.neutralise`.

    P4's finding was that the long-only excess over the equal-weighted floor is
    almost entirely a beta tilt: `beta_market`, which holds no company-specific
    view at all, took +7.99%/yr of the best book's +8.95%. This holds the top
    quintile and shorts the equal-weighted panel sized to the book's own mean
    beta, so the market channel is removed by construction and what remains is
    whatever the ordering knew about individual companies.

    WHY THIS NEEDS A REAL BETA AND `neutralise` DOES NOT. Rank IC is
    scale-invariant, so §1 can residualise on `beta_market`'s prediction
    (``beta_i * mu_market``) and get numerically identical answers to
    residualising on ``beta_i``. A HEDGE RATIO IS NOT SCALE-FREE: shorting
    ``beta_i * mu_market`` units of the market is shorting the wrong amount by a
    factor of mu_market, which is ~0.02. So this takes a dimensionally real beta
    — `regime.rolling_beta`, trailing, and the only estimate a live book could
    actually have used.

    COSTS ARE DENOMINATED IN LONG-BOOK UNITS, and so is the return: both are per
    unit of long notional, which keeps them comparable to each other and to
    `simulate`. The hedge leg is charged the full equity round trip, which
    OVERSTATES it — an index future or ETF pays no STT on the buy side and lower
    exchange fees. Modelling a second cost schedule for one leg would be a
    second set of numbers to keep true; erring against the strategy is the safe
    direction for a measurement whose expected answer is a null.

    Returns a `BookResult` whose series is a SPREAD, not a self-financing
    portfolio. It therefore has no meaningful "vs floor" and its drawdown is not
    the quantity that word usually names — the same suppression the P4 table
    applies to its long-short rows.
    """
    costs = costs or CostModel()
    if "beta" not in beta.columns:
        raise ValueError("simulate_hedged needs a 'beta' column (a real slope, "
                         "not beta_market's scaled prediction)")

    lookup = {(str(d), str(t)): float(b) for d, t, b
              in zip(beta["date"], beta["ticker"], beta["beta"])
              if np.isfinite(b)}

    book = BookResult(name=f"beta_neutral@{quantiles}", n_rebalances=0)
    held: set[str] = set()
    prev_hedge = 0.0

    for rb in rebalance_books(predictions, rebalance_every, quantiles):
        if rb.degenerate:
            book.n_no_ordering += 1
            continue

        longs = set(rb.top["ticker"])
        # A name with no trailing beta gets 1.0 — the market's own exposure,
        # which is the least assertive available answer. Dropping it instead
        # would change which names the book holds and stop it trading the
        # ordering `cross_sectional_report` scored.
        betas = [lookup.get((rb.date, str(t)), 1.0) for t in rb.top["ticker"]]
        hedge = float(np.mean(betas)) if betas else 0.0

        long_ret = float(rb.top["y_true"].mean())
        market_ret = float(rb.day["y_true"].mean())
        gross = long_ret - hedge * market_ret

        # Long leg trades names; hedge leg trades NOTIONAL. A one-sided notional
        # change of size |d| against a book of 1.0 is half a round trip, which is
        # the same convention `_turn` uses for entering a full book.
        turn = _turn(longs, held) + abs(hedge - prev_hedge) / 2.0
        cost = turn * costs.round_trip

        book.gross_returns.append(gross)
        book.net_returns.append(gross - cost)
        book.turnover.append(turn)
        book.dates.append(rb.date)
        book.n_rebalances += 1
        held, prev_hedge = longs, hedge

    return book
