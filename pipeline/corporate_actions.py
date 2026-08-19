"""
pipeline/corporate_actions.py — splits and dividends, stored rather than assumed.

WHY THIS TABLE EXISTS. Audit finding F11: OHLCV ingestion used to append only
unseen dates, so a series fetched before a split and topped up after it spliced
two different adjustment bases together at an invisible seam. The fix was to
delete and rewrite the whole series every run, which keeps one basis — but it
fixes the symptom blind. Nothing in the system knows a split HAPPENED, so:

  * a 1:2 split shows up as a 50% overnight fall in any un-adjusted view, and
    nothing can distinguish it from a real 50% fall;
  * the validation gate cannot tell a data error from a corporate action, so it
    must either alarm on every split or stay silent on genuine breaks;
  * `adj_close` is trusted absolutely, with no independent record to check it
    against, even though the whole excess-return target rests on it.

Storing the actions makes the adjustment auditable. A price jump is either
explained by a row in this table or it is a data-quality finding.

DIVIDENDS ARE RECORDED BUT NOT ACTED ON. The target is built from `adj_close`,
which is already total-return adjusted, so dividends are inside the label
whether or not this table exists. They are stored because a dividend explains a
price gap on the ex-date, which is exactly what the gate needs to know.

THE DATE IS THE EX-DATE, AND IT IS NOT SHIFTED. Unlike earnings (F13), where
the announcement can land after the close and the surprise had to move to the
next session, a split or dividend takes effect ON the ex-date by definition:
the price opens adjusted. There is no post-close ambiguity to correct for.
"""
from __future__ import annotations

import time
from typing import NamedTuple

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from data.db import get_engine, to_native_params

FETCH_ATTEMPTS = 2

# A split ratio outside this band is almost certainly a feed error rather than
# a corporate action. Recorded but flagged, never silently dropped.
PLAUSIBLE_SPLIT_RANGE = (0.01, 100.0)


class ActionsReport(NamedTuple):
    tickers: int
    splits: int
    dividends: int
    failed: list[str]

    def summary(self) -> str:
        return (f"{self.splits} splits and {self.dividends} dividends across "
                f"{self.tickers} tickers, {len(self.failed)} failed")


def fetch_and_store(tickers: list[str] | None = None) -> ActionsReport:
    """
    Refreshes the corporate-actions record for the given tickers.

    Rewrites each ticker's rows wholesale rather than appending, for the same
    reason F11 forced OHLCV to: an amended or withdrawn action must be able to
    disappear, and an append-only history cannot express that.
    """
    engine = get_engine()

    if tickers is None:
        from data.universe import get_universe
        tickers = get_universe()

    total_splits = total_dividends = 0
    failed: list[str] = []

    for ticker in tickers:
        actions = None
        for attempt in range(1, FETCH_ATTEMPTS + 1):
            try:
                actions = yf.Ticker(ticker).actions
                break
            except Exception as exc:                            # noqa: BLE001
                print(f"[Actions] {ticker} attempt {attempt}/{FETCH_ATTEMPTS} "
                      f"failed: {exc}")
                if attempt < FETCH_ATTEMPTS:
                    time.sleep(2 * attempt)

        if actions is None:
            failed.append(ticker)
            continue
        if len(actions) == 0:
            continue

        rows = []
        frame = actions.reset_index()
        date_col = frame.columns[0]
        for record in frame.to_dict("records"):
            date = str(record[date_col])[:10]

            split = float(record.get("Stock Splits") or 0.0)
            if split and split > 0:
                lo, hi = PLAUSIBLE_SPLIT_RANGE
                rows.append({
                    "ticker": ticker, "date": date, "action_type": "SPLIT",
                    "ratio": split, "amount": None,
                    "implausible": int(not (lo <= split <= hi)),
                })

            dividend = float(record.get("Dividends") or 0.0)
            if dividend and dividend > 0:
                rows.append({
                    "ticker": ticker, "date": date, "action_type": "DIVIDEND",
                    "ratio": None, "amount": dividend, "implausible": 0,
                })

        if not rows:
            continue

        with engine.connect() as conn:
            conn.execute(text("DELETE FROM corporate_actions WHERE ticker = :t"),
                         {"t": ticker})
            for row in rows:
                conn.execute(
                    text("INSERT INTO corporate_actions "
                         "(ticker, date, action_type, ratio, amount, implausible) "
                         "VALUES (:ticker, :date, :action_type, :ratio, :amount, "
                         ":implausible)"),
                    to_native_params(row),
                )
            conn.commit()

        total_splits += sum(r["action_type"] == "SPLIT" for r in rows)
        total_dividends += sum(r["action_type"] == "DIVIDEND" for r in rows)

    report = ActionsReport(len(tickers), total_splits, total_dividends, failed)
    print(f"[Actions] {report.summary()}")
    if failed:
        print(f"[Actions] failed: {', '.join(failed)}")
    return report


def actions_for(ticker: str, engine=None) -> pd.DataFrame:
    """Every recorded action for one ticker, oldest first."""
    engine = engine or get_engine()
    return pd.read_sql(
        text("SELECT date, action_type, ratio, amount, implausible "
             "FROM corporate_actions WHERE ticker = :t ORDER BY date ASC"),
        engine, params={"t": ticker},
    )


def explained_by_action(ticker: str, date: str, window: int = 1,
                        engine=None) -> bool:
    """
    Is a price break on this date attributable to a recorded corporate action?

    The window exists because the ex-date recorded by the data provider and the
    session on which the gap appears in an un-adjusted series can differ by one
    trading day across a weekend or holiday.
    """
    engine = engine or get_engine()
    target = pd.Timestamp(date)
    lo = (target - pd.Timedelta(days=window + 3)).strftime("%Y-%m-%d")
    hi = (target + pd.Timedelta(days=window + 3)).strftime("%Y-%m-%d")
    n = pd.read_sql(
        text("SELECT COUNT(*) AS n FROM corporate_actions "
             "WHERE ticker = :t AND date BETWEEN :lo AND :hi"),
        engine, params={"t": ticker, "lo": lo, "hi": hi},
    )["n"].iloc[0]
    return int(n) > 0
