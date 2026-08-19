"""
pipeline/validation.py — the gate a run must pass before it publishes.

WHY A GATE RATHER THAN MORE ASSERTIONS. Every data defect this project has hit
was silent at the point of failure and loud somewhere far away: F6 froze the
labelled set and surfaced as flat accuracy months later; F11 spliced two
adjustment bases and surfaced as an unexplainable price jump; the 2026-08-16
benchmark outage erased 22 tickers' labels and surfaced as "insufficient
history" in a weekly report. In each case the pipeline had everything it needed
to notice at the time, and no place to put the check.

This is that place. Each check answers one question about the state of the
database, returns PASS, WARN or FAIL with a readable detail line, and the whole
report is written to `experiment_runs.gate_report` so a run's data quality is
recoverable after the fact rather than reconstructed from logs.

FAIL VERSUS WARN. FAIL means the defect would corrupt what the run publishes:
duplicate training rows, non-finite features, timestamps from the future. The
run stops. WARN means something is degraded but the output is still honest —
one ticker short of history, a price break that no recorded corporate action
explains. The run continues and the warning is recorded. The distinction
matters because a gate that fails on everything gets disabled, and a gate that
fails on nothing is decoration.

WHAT THIS GATE DOES NOT DO. It does not check whether the model is any good.
Model quality is measured by the weekly purged walk-forward evaluation and
gated by the evidence rules in agents.critic_agent. Mixing the two would let a
data-quality pass be read as a performance claim, which is the confusion the
whole Phase 0 audit was about.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.db import get_engine

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# OHLCV older than this means ingestion is broken, not that the market was shut.
# Long enough to survive a Diwali-week cluster of holidays plus a weekend.
MAX_DATA_AGE_DAYS = 6

# A daily log return this large is either a corporate action or a data error.
# 0.35 is ~42% in one session; genuine single-session moves that large are rare
# enough on NIFTY 100 names to be worth a look every time.
PRICE_BREAK_LOG_RETURN = 0.35

# Below this, a ticker cannot produce even one evaluation fold.
MIN_LABELLED_ROWS_PER_TICKER = 505

# Fraction of the universe that may be short of labels before it stops being a
# per-ticker problem and starts being a pipeline problem.
MAX_FRACTION_UNDERLABELLED = 0.25


class Check(NamedTuple):
    name: str
    status: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.status:4s}] {self.name}: {self.detail}"


class GateReport(NamedTuple):
    status: str
    checks: list[Check]

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    def to_json(self) -> str:
        return json.dumps([c._asdict() for c in self.checks])

    def summary(self) -> str:
        return (f"{self.status} — {len(self.failures)} failed, "
                f"{len(self.warnings)} warned, {len(self.checks)} checks")


def _scalar(engine, sql: str, params: dict | None = None):
    with engine.connect() as conn:
        return conn.execute(text(sql), params or {}).scalar()


# ── Individual checks ─────────────────────────────────────────────────────────

def check_no_duplicate_signal_rows(engine, universe) -> Check:
    """
    A repeated (ticker, date) is a duplicated training example. It reweights the
    fit toward whatever was duplicated and inflates any metric averaged over
    rows, invisibly.
    """
    n = _scalar(engine, """
        SELECT COUNT(*) FROM (
            SELECT ticker, date FROM signals
            GROUP BY ticker, date HAVING COUNT(*) > 1
        ) AS d
    """)
    n = int(n or 0)
    return Check("no_duplicate_signal_rows",
                 PASS if n == 0 else FAIL,
                 "no duplicates" if n == 0
                 else f"{n} (ticker, date) pairs appear more than once")


def check_no_future_dates(engine, universe) -> Check:
    """
    A row dated ahead of today means either a timezone error or a bad parse, and
    a forward-looking label computed from it would be lookahead by definition.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = int(_scalar(engine, "SELECT COUNT(*) FROM signals WHERE date > :t",
                    {"t": today}) or 0)
    return Check("no_future_dates",
                 PASS if n == 0 else FAIL,
                 "no rows dated ahead of today" if n == 0
                 else f"{n} signal rows are dated after {today}")


def check_data_freshness(engine, universe) -> Check:
    """Is the price history actually being refreshed?"""
    latest = _scalar(engine, "SELECT MAX(date) FROM ohlcv")
    if not latest:
        return Check("data_freshness", FAIL, "ohlcv is empty")

    age = (datetime.now(timezone.utc).date()
           - pd.Timestamp(str(latest)[:10]).date()).days
    if age > MAX_DATA_AGE_DAYS:
        return Check("data_freshness", FAIL,
                     f"newest OHLCV row is {latest} ({age} days old, "
                     f"limit {MAX_DATA_AGE_DAYS})")
    return Check("data_freshness", PASS, f"newest OHLCV row is {latest} ({age}d)")


def check_features_are_finite(engine, universe) -> Check:
    """
    XGBoost tolerates NaN by design but not infinity, and an inf reaching the
    API also breaks JSON serialisation (see api/serialization.py). Checked on
    the most recent rows, which are the ones a forecast is actually made from.
    """
    from pipeline.signals import FEATURE_COLS

    df = pd.read_sql(
        text(f"SELECT ticker, date, {', '.join(FEATURE_COLS)} FROM signals "
             f"WHERE date >= :cut"),
        engine,
        params={"cut": (datetime.now(timezone.utc) - timedelta(days=30))
                .strftime("%Y-%m-%d")},
    )
    if df.empty:
        return Check("features_are_finite", WARN, "no recent rows to check")

    numeric = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    bad = numeric.columns[np.isinf(numeric.to_numpy(dtype=float)).any(axis=0)].tolist()
    return Check("features_are_finite",
                 PASS if not bad else FAIL,
                 f"{len(df)} recent rows finite" if not bad
                 else f"non-finite values in {bad}")


def check_label_coverage(engine, universe) -> Check:
    """
    Enough labelled history per ticker to produce evaluation folds at all.

    Per-ticker shortfalls are normal — a recent listing genuinely has no history
    — so this warns until a quarter of the universe is affected, at which point
    the cause is the pipeline rather than the constituents.
    """
    if not universe:
        return Check("label_coverage", WARN, "empty universe")

    counts = pd.read_sql(
        text("SELECT ticker, COUNT(*) AS n FROM signals "
             "WHERE target_excess_return IS NOT NULL GROUP BY ticker"),
        engine,
    ).set_index("ticker")["n"].to_dict()

    short = [t for t in universe
             if counts.get(t, 0) < MIN_LABELLED_ROWS_PER_TICKER]
    fraction = len(short) / len(universe)
    zero = [t for t in universe if counts.get(t, 0) == 0]

    detail = (f"{len(short)}/{len(universe)} below "
              f"{MIN_LABELLED_ROWS_PER_TICKER} labelled rows")
    if zero:
        detail += f"; {len(zero)} with ZERO labels: {', '.join(zero[:8])}"

    if fraction > MAX_FRACTION_UNDERLABELLED:
        return Check("label_coverage", FAIL, detail)
    return Check("label_coverage", PASS if not short else WARN, detail)


def check_benchmark_coverage(engine, universe) -> Check:
    """
    Every ticker must carry a benchmark return, or its target is not an excess
    return at all. This is the 2026-08-16 failure expressed as a check.
    """
    if not universe:
        return Check("benchmark_coverage", WARN, "empty universe")

    df = pd.read_sql(
        text("SELECT ticker, COUNT(benchmark_return) AS n FROM signals "
             "GROUP BY ticker"),
        engine,
    ).set_index("ticker")["n"].to_dict()

    missing = [t for t in universe if int(df.get(t, 0)) == 0]
    return Check("benchmark_coverage",
                 PASS if not missing else FAIL,
                 f"all {len(universe)} tickers carry a benchmark return"
                 if not missing
                 else f"{len(missing)} tickers have no benchmark return at all: "
                      f"{', '.join(missing[:10])}")


def check_price_breaks_are_explained(engine, universe) -> Check:
    """
    Any session where the adjusted price moves more than PRICE_BREAK_LOG_RETURN
    should correspond to a recorded corporate action. An unexplained break is
    the signature of F11 — two adjustment bases spliced at a seam.

    Warns rather than fails: the corporate-actions table is populated by a
    separate step that can legitimately be behind, and a genuine 40% session
    does occasionally happen.
    """
    from pipeline.corporate_actions import explained_by_action

    n_actions = int(_scalar(engine, "SELECT COUNT(*) FROM corporate_actions") or 0)
    if n_actions == 0:
        return Check("price_breaks_are_explained", WARN,
                     "corporate_actions is empty — cannot attribute price breaks")

    df = pd.read_sql(
        text("SELECT ticker, date, close FROM signals ORDER BY ticker, date"),
        engine,
    )
    if df.empty:
        return Check("price_breaks_are_explained", WARN, "no signal rows")

    df["r"] = np.log(df.groupby("ticker")["close"].transform(
        lambda s: s / s.shift(1)))
    breaks = df[df["r"].abs() > PRICE_BREAK_LOG_RETURN]

    unexplained = [(r.ticker, r.date) for r in breaks.itertuples(index=False)
                   if not explained_by_action(r.ticker, str(r.date)[:10], engine=engine)]

    if not unexplained:
        return Check("price_breaks_are_explained", PASS,
                     f"{len(breaks)} large moves, all attributable")
    shown = ", ".join(f"{t}@{d}" for t, d in unexplained[:6])
    return Check("price_breaks_are_explained", WARN,
                 f"{len(unexplained)} of {len(breaks)} large moves unexplained: {shown}")


def check_target_distribution(engine, universe) -> Check:
    """
    A forward 30-session excess log return should sit inside single digits of
    absolute value for a NIFTY 100 name. Values far outside that are the
    fingerprint of an adjustment splice rather than a market event, and they
    dominate a squared-error objective completely.
    """
    df = pd.read_sql(
        text("SELECT target_excess_return AS t FROM signals "
             "WHERE target_excess_return IS NOT NULL"),
        engine,
    )
    if df.empty:
        return Check("target_distribution", WARN, "no labelled rows")

    t = df["t"].astype(float)
    extreme = int((t.abs() > 1.5).sum())            # >1.5 in log terms: ~4.5x
    detail = (f"n={len(t)}, sd={t.std():.4f}, "
              f"1%={t.quantile(0.01):+.3f}, 99%={t.quantile(0.99):+.3f}, "
              f"{extreme} beyond ±1.5")
    return Check("target_distribution",
                 PASS if extreme == 0 else WARN, detail)


CHECKS = [
    check_no_duplicate_signal_rows,
    check_no_future_dates,
    check_data_freshness,
    check_features_are_finite,
    check_label_coverage,
    check_benchmark_coverage,
    check_price_breaks_are_explained,
    check_target_distribution,
]


def run_gate(universe: list[str] | None = None, engine=None) -> GateReport:
    """
    Runs every check and returns the combined report.

    A check that raises is itself a FAIL: an exception here means the database
    is not in a state the check could even interrogate, which is not a reason to
    proceed.
    """
    engine = engine or get_engine()
    if universe is None:
        from data.universe import get_universe
        universe = get_universe()

    checks: list[Check] = []
    for check in CHECKS:
        try:
            checks.append(check(engine, universe))
        except Exception as exc:                                # noqa: BLE001
            checks.append(Check(check.__name__.replace("check_", ""), FAIL,
                                f"check raised {type(exc).__name__}: {exc}"))

    status = FAIL if any(c.status == FAIL for c in checks) else (
        WARN if any(c.status == WARN for c in checks) else PASS)
    return GateReport(status, checks)
