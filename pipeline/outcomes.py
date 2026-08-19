"""
pipeline/outcomes.py — did the published forecast come true?

THE GAP THIS FILLS. `forecast_outcomes` has existed since Phase 0 (data/db.py)
and nothing has ever written to it. Every accuracy number the system reports is
therefore a BACKTEST number: measured on held-out folds of history, by the same
process that fitted the model. Nothing measured whether a forecast that was
actually published, on a specific date, to a specific reader, turned out to be
right. Those are different claims, and the audit exists because the first was
mistaken for the second once already.

HOW A FORECAST IS RESOLVED. Not by recomputing returns here — by reading the
label. `signals.target_excess_return` at date D is, by construction, the forward
HORIZON_SESSIONS log return from D in excess of the benchmark: exactly the
quantity the model was trained to predict, computed by exactly the code that
computes the training label. A forecast made on D is therefore resolved the
moment that row's target stops being null, and predicted-versus-realised is a
comparison of like with like. Reimplementing the arithmetic here would create a
second definition of "excess return" that could drift from the first, which is
the class of defect F8 and F13 both belong to.

IDEMPOTENT AND APPEND-ONLY. `forecast_outcomes` is keyed (ticker,
forecast_date) and written with ON CONFLICT DO NOTHING. Re-running resolves
only what is newly due; a resolved outcome is never rewritten, because the
whole point of the table is to be a record that cannot be quietly improved
after the fact.

BENCHMARK DRIFT IS RECORDED, NOT HIDDEN. The realised excess return depends on
which index the ticker was benchmarked against, and that mapping can change
(tools/audit_benchmarks.py changed it for 36 tickers on 2026-08-19). A forecast
published under one benchmark and resolved under another is not a fair test of
the forecast, so the resolution is skipped and counted rather than scored.
"""
from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from sqlalchemy import text

from data.db import get_engine, to_native_params


class OutcomeReport(NamedTuple):
    """What one resolution pass did."""
    resolved: int
    already_resolved: int
    not_due: int
    benchmark_changed: int

    def summary(self) -> str:
        return (f"{self.resolved} resolved, {self.already_resolved} already on "
                f"record, {self.not_due} not yet due, "
                f"{self.benchmark_changed} skipped (benchmark changed)")


# One row per ticker per forecast DATE. The pipeline appends to `forecasts` on
# every run, so a re-run on the same day would otherwise produce two candidate
# outcomes for one published forecast; the latest row for that date wins.
_DUE_FORECASTS = """
    SELECT f.ticker,
           f.forecast_date,
           f.pred_excess_return,
           f.interval_low,
           f.interval_high,
           f.current_price,
           f.benchmark_ticker,
           f.forecast_id
    FROM (
        SELECT ticker,
               {date_expr}                             AS forecast_date,
               pred_excess_return,
               interval_low,
               interval_high,
               current_price,
               benchmark_ticker,
               id                                      AS forecast_id,
               ROW_NUMBER() OVER (
                   PARTITION BY ticker, {date_expr}
                   ORDER BY last_updated DESC, id DESC
               )                                       AS rn
        FROM forecasts
        WHERE pred_excess_return IS NOT NULL
    ) AS f
    WHERE f.rn = 1
"""


def _date_expr(dialect: str) -> str:
    """`last_updated` is a timestamp; the outcome is keyed on its date."""
    if dialect.startswith("postgres"):
        return "CAST(last_updated AS DATE)"
    return "DATE(last_updated)"


def resolve_due_forecasts(engine=None, horizon_label: str = "target_excess_return"
                          ) -> OutcomeReport:
    """
    Writes an outcome row for every published forecast whose horizon has closed.

    Returns counts rather than raising on a partial pass: a forecast that is not
    yet due is the normal case, not an error, and the daily job should not fail
    because most of the book is still open.
    """
    engine = engine or get_engine()
    dialect = engine.dialect.name

    forecasts = pd.read_sql(
        text(_DUE_FORECASTS.format(date_expr=_date_expr(dialect))), engine)
    if forecasts.empty:
        return OutcomeReport(0, 0, 0, 0)

    # The realised label, straight from the signals table.
    labels = pd.read_sql(
        text(f"SELECT ticker, date AS forecast_date, {horizon_label} AS realised_excess, "
             f"target_return AS realised_return, benchmark_return, "
             f"benchmark_ticker AS label_benchmark, close AS price_at_forecast "
             f"FROM signals WHERE {horizon_label} IS NOT NULL"),
        engine,
    )

    existing = pd.read_sql(
        text("SELECT ticker, forecast_date FROM forecast_outcomes"), engine)

    for frame in (forecasts, labels, existing):
        if not frame.empty:
            frame["forecast_date"] = frame["forecast_date"].astype(str).str[:10]

    total = len(forecasts)
    if not existing.empty:
        seen = set(map(tuple, existing[["ticker", "forecast_date"]].to_numpy()))
        keep = [tuple(r) not in seen for r in
                forecasts[["ticker", "forecast_date"]].to_numpy()]
        forecasts = forecasts[keep]
    already = total - len(forecasts)

    due = forecasts.merge(labels, on=["ticker", "forecast_date"], how="inner")
    not_due = len(forecasts) - len(due)

    # A forecast published against one index and resolved against another is
    # not a test of the forecast. Skip and count it.
    if not due.empty:
        same = (due["benchmark_ticker"].fillna("") == due["label_benchmark"].fillna("")) \
               | (due["benchmark_ticker"].fillna("") == "")
        changed = int((~same).sum())
        due = due[same]
    else:
        changed = 0

    if due.empty:
        return OutcomeReport(0, already, not_due, changed)

    # Resolution date: the session HORIZON_SESSIONS after the forecast, read
    # from the ticker's own trading calendar rather than assumed to be 30
    # calendar days — the NSE calendar is what the label was built on.
    resolution = _resolution_dates(engine, due)
    due = due.merge(resolution, on=["ticker", "forecast_date"], how="left")

    rows = []
    for r in due.itertuples(index=False):
        pred, realised = r.pred_excess_return, r.realised_excess
        direction = None
        if pred is not None and pd.notna(pred) and pd.notna(realised) and pred != 0:
            direction = int((pred > 0) == (realised > 0))

        # The published interval is a PRICE band. The realised price follows
        # from the realised TOTAL return, not the excess one.
        inside = None
        if (pd.notna(r.interval_low) and pd.notna(r.interval_high)
                and pd.notna(r.realised_return) and pd.notna(r.price_at_forecast)):
            import math
            realised_price = float(r.price_at_forecast) * math.exp(float(r.realised_return))
            inside = int(float(r.interval_low) <= realised_price <= float(r.interval_high))

        rows.append({
            "forecast_id": int(r.forecast_id) if pd.notna(r.forecast_id) else None,
            "ticker": r.ticker,
            "forecast_date": r.forecast_date,
            "resolution_date": getattr(r, "resolution_date", None),
            "pred_excess_return": float(pred) if pd.notna(pred) else None,
            "realised_excess_return": float(realised) if pd.notna(realised) else None,
            "realised_return": float(r.realised_return) if pd.notna(r.realised_return) else None,
            "benchmark_return": float(r.benchmark_return) if pd.notna(r.benchmark_return) else None,
            "direction_correct": direction,
            "inside_interval": inside,
        })

    columns = list(rows[0])
    conflict = ("ON CONFLICT (ticker, forecast_date) DO NOTHING"
                if dialect.startswith(("postgres", "sqlite")) else "")
    statement = text(
        f"INSERT INTO forecast_outcomes ({', '.join(columns)}) "
        f"VALUES ({', '.join(':' + c for c in columns)}) {conflict}"
    )

    with engine.connect() as conn:
        for row in rows:
            conn.execute(statement, to_native_params(row))
        conn.commit()

    return OutcomeReport(len(rows), already, not_due, changed)


def _resolution_dates(engine, due: pd.DataFrame) -> pd.DataFrame:
    """
    The session HORIZON_SESSIONS trading days after each forecast date.

    Derived from the ticker's own rows in `signals`, which is the same calendar
    the forward label was shifted along. Deriving it from calendar arithmetic
    would drift on every exchange holiday.
    """
    from pipeline.signals import HORIZON_SESSIONS

    out = []
    for ticker, group in due.groupby("ticker"):
        sessions = pd.read_sql(
            text("SELECT date FROM signals WHERE ticker = :t ORDER BY date ASC"),
            engine, params={"t": ticker},
        )["date"].astype(str).str[:10].tolist()
        position = {d: i for i, d in enumerate(sessions)}
        for forecast_date in group["forecast_date"]:
            i = position.get(forecast_date)
            target = i + HORIZON_SESSIONS if i is not None else None
            out.append({
                "ticker": ticker,
                "forecast_date": forecast_date,
                "resolution_date": (sessions[target]
                                    if target is not None and target < len(sessions)
                                    else None),
            })
    return pd.DataFrame(out) if out else pd.DataFrame(
        columns=["ticker", "forecast_date", "resolution_date"])


def realised_accuracy(engine=None, ticker: str | None = None) -> dict:
    """
    Observed performance of published forecasts. The honest counterpart to the
    `eval_*` columns, which are backtest figures.

    Returns Nones rather than zeros when nothing has resolved yet. A hit rate of
    0.0 and "no forecast has matured" are very different statements, and the
    leaderboard already carries one column (`composite_score`) where a single
    value covers several meanings — see agents.graph.classify_score_basis.
    """
    engine = engine or get_engine()
    where, params = "", {}
    if ticker:
        where, params = "WHERE ticker = :t", {"t": ticker.upper()}

    df = pd.read_sql(
        text(f"""
            SELECT COUNT(*)                          AS n,
                   AVG(CAST(direction_correct AS FLOAT)) AS hit_rate,
                   AVG(CAST(inside_interval AS FLOAT))   AS coverage,
                   AVG(realised_excess_return)       AS mean_realised_excess,
                   AVG(pred_excess_return)           AS mean_pred_excess
            FROM forecast_outcomes {where}
        """),
        engine, params=params,
    )
    row = df.iloc[0].to_dict() if not df.empty else {}
    n = int(row.get("n") or 0)
    if n == 0:
        return {"n": 0, "hit_rate": None, "interval_coverage": None,
                "mean_realised_excess": None, "mean_pred_excess": None}
    return {
        "n": n,
        "hit_rate": _f(row.get("hit_rate")),
        "interval_coverage": _f(row.get("coverage")),
        "mean_realised_excess": _f(row.get("mean_realised_excess")),
        "mean_pred_excess": _f(row.get("mean_pred_excess")),
    }


def _f(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)
