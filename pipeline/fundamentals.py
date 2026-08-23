"""
pipeline/fundamentals.py — Valuation signals, attached point-in-time.

Phase 2. The first information in this project that is not derived from the
same OHLCV series as everything else.

Every one of the 24 technical features is a transformation of price and volume,
so no model can extract from them information that price and volume do not
carry — which is the standing explanation for why two independent foundation
models, a gradient-boosted tree and a ridge all failed to beat a naive floor.
Valuation is a genuinely different axis, and it is also the first feature set
that varies *across tickers on the same date*, unlike the six macro columns
which are identically zero after within-date standardisation.

THE HAZARD, WHICH IS THE WHOLE DIFFICULTY
─────────────────────────────────────────
A fundamental is known to the market on the day it is FILED, not on the day the
fiscal period ended. Attaching FY2025 earnings to 31 March 2025 hands the model
two months of information nobody had. That is F13 exactly — the earnings-surprise
defect, where a figure was attached to the announcement date although Indian
results are commonly declared post-close — with a longer lookahead window.

SEBI (LODR) Regulation 33 requires audited annual results within **60 days** of
the financial year end. Every figure here is therefore attached to
``period_end + 60 days``, and never earlier. That is conservative: a company
filing on day 45 has its number withheld for a further 15 days, which costs a
little signal and cannot manufacture any.

A SECOND HAZARD, NOT FULLY SOLVED
─────────────────────────────────
yfinance reports statements **as restated**, not as originally reported. If a
company restates FY2024 earnings during FY2025, we see the restated figure and
attach it to 2024. The direction of that error flatters the model. It cannot be
fixed without a vendor that keeps as-reported vintages, so it is recorded here
and in the report rather than papered over. It is mild for EPS and book value,
which are rarely restated materially, and it would be severe for anything
accrual-based — which is one reason this module stops at two quantities.

WHY YIELDS, NOT RATIOS
──────────────────────
The features are ``earnings_yield = EPS / price`` and
``book_to_market = BVPS / price``, not P/E and P/B.

P/E is discontinuous and non-monotone at zero earnings: a company losing money
has a *negative* P/E, which sorts below every profitable company as though it
were the cheapest thing on the exchange. Ranking on it is meaningless exactly
where it matters most. E/P is continuous through zero and monotone in value
throughout. Fama–French use book-to-market for the same reason. The ratios are
still stored and exposed, because P/E is what a human reader expects to see.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.db import get_engine, to_native_params

logger = logging.getLogger(__name__)

# SEBI (LODR) Regulation 33: audited annual results within 60 days of the
# financial year end. Quarterly is 45. We use annual statements, so 60.
ANNUAL_FILING_LAG_DAYS = 60

# The features this module contributes. Both are yields — see the module
# docstring on why not P/E and P/B.
FUNDAMENTAL_COLS = ["earnings_yield", "book_to_market"]

# Below this, a ticker has too few fiscal years to carry a usable step
# function and is left unattached rather than half-filled.
MIN_PERIODS = 2

_INCOME_EPS_ROWS = ["Diluted EPS", "Basic EPS"]
_EQUITY_ROWS = ["Stockholders Equity", "Total Equity Gross Minority Interest"]
_SHARES_ROW = "Ordinary Shares Number"


class LookaheadRefused(RuntimeError):
    """Raised when a fundamental would be attached before it could be known."""


def _first_present(frame: pd.DataFrame, rows: list[str], col) -> float | None:
    for r in rows:
        if r in frame.index:
            v = frame.loc[r, col]
            if pd.notna(v):
                return float(v)
    return None


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Annual EPS and book value per share per ticker, with an effective date.

    Annual rather than quarterly because that is what is actually available:
    yfinance returns roughly five fiscal years of annual statements against
    three to five *quarters* of quarterly ones. Four usable years of an
    annually-stepping cross-sectional feature is worth more than fifteen months
    of a quarterly one.
    """
    import yfinance as yf

    records = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            income, balance = t.income_stmt, t.balance_sheet
            if income is None or balance is None or income.empty or balance.empty:
                logger.warning(f"[Fundamentals] {ticker}: no statements")
                continue

            for period in sorted(set(income.columns) & set(balance.columns)):
                eps = _first_present(income, _INCOME_EPS_ROWS, period)
                equity = _first_present(balance, _EQUITY_ROWS, period)
                shares = (float(balance.loc[_SHARES_ROW, period])
                          if _SHARES_ROW in balance.index
                          and pd.notna(balance.loc[_SHARES_ROW, period]) else None)

                bvps = (equity / shares
                        if equity is not None and shares and shares > 0 else None)
                if eps is None and bvps is None:
                    continue

                period_end = pd.Timestamp(period).date()
                records.append({
                    "ticker": ticker,
                    "period_end": period_end.isoformat(),
                    "effective_date": (period_end + timedelta(
                        days=ANNUAL_FILING_LAG_DAYS)).isoformat(),
                    "eps": eps,
                    "book_value_per_share": bvps,
                    "shares": shares,
                    "source": "yfinance:annual",
                })
        except Exception as exc:                                # noqa: BLE001
            logger.warning(f"[Fundamentals] {ticker}: {str(exc)[:120]}")

    frame = pd.DataFrame(records)
    logger.info(f"[Fundamentals] {len(frame)} periods across "
                f"{frame['ticker'].nunique() if len(frame) else 0} tickers")
    return frame


def store_fundamentals(frame: pd.DataFrame, engine=None) -> int:
    """Upsert by (ticker, period_end). Restatements overwrite; see the docstring."""
    if frame.empty:
        return 0
    engine = engine or get_engine()
    with engine.begin() as conn:
        for row in frame.to_dict("records"):
            conn.execute(text("""
                INSERT INTO fundamentals
                    (ticker, period_end, effective_date, eps,
                     book_value_per_share, shares, source)
                VALUES
                    (:ticker, :period_end, :effective_date, :eps,
                     :book_value_per_share, :shares, :source)
                ON CONFLICT (ticker, period_end) DO UPDATE SET
                    effective_date = EXCLUDED.effective_date,
                    eps = EXCLUDED.eps,
                    book_value_per_share = EXCLUDED.book_value_per_share,
                    shares = EXCLUDED.shares,
                    source = EXCLUDED.source
            """), to_native_params(row))
    return len(frame)


def load_fundamentals(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(
                "SELECT ticker, period_end, effective_date, eps, "
                "book_value_per_share FROM fundamentals"), conn)
    except Exception:                                           # noqa: BLE001
        return pd.DataFrame()


def attach_fundamentals(panel: pd.DataFrame, engine=None,
                        fundamentals: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    As-of join: each row gets the latest figure whose EFFECTIVE date has passed.

    ``merge_asof`` with ``direction="backward"`` is the whole point — it takes
    the most recent record at or before the row's date and never a later one.
    A plain merge on fiscal year, or a forward fill from ``period_end``, is the
    lookahead this module exists to avoid.

    Rows before a ticker's first effective date get NaN rather than the oldest
    available figure. That is correct and it is also why the feature only
    covers the recent part of the panel: the market genuinely did not know
    FY2023 earnings in 2018.
    """
    if panel.empty:
        return panel

    fund = fundamentals if fundamentals is not None else load_fundamentals(engine)
    out = panel.copy()
    if fund is None or fund.empty:
        for c in FUNDAMENTAL_COLS + ["pe_ratio", "pb_ratio"]:
            out[c] = np.nan
        logger.warning("[Fundamentals] none stored; columns attached as NaN")
        return out

    fund = fund.copy()
    fund["_eff"] = pd.to_datetime(fund["effective_date"], errors="coerce")
    fund = fund.dropna(subset=["_eff"])

    enough = fund.groupby("ticker")["period_end"].transform("count") >= MIN_PERIODS
    fund = fund[enough]

    out["_d"] = pd.to_datetime(out["date"], errors="coerce")
    left = out.sort_values("_d", kind="mergesort")
    right = fund.sort_values("_eff", kind="mergesort")

    merged = pd.merge_asof(
        left, right[["ticker", "_eff", "eps", "book_value_per_share", "period_end"]],
        left_on="_d", right_on="_eff", by="ticker", direction="backward",
        allow_exact_matches=True,
    )

    # The guard, not decoration: merge_asof's contract is exactly this, so a
    # violation means the frames were mis-sorted or the `by` key was wrong —
    # both of which produce a silently lookahead-contaminated feature.
    known = merged["_eff"].notna()
    if known.any() and (merged.loc[known, "_eff"] > merged.loc[known, "_d"]).any():
        raise LookaheadRefused(
            "a fundamental was matched to a date before its effective date; "
            "refusing to return a lookahead-contaminated frame"
        )

    close = pd.to_numeric(merged["close"], errors="coerce")
    price = close.where(close > 0)
    eps = pd.to_numeric(merged["eps"], errors="coerce")
    bvps = pd.to_numeric(merged["book_value_per_share"], errors="coerce")

    # Yields, so the feature is continuous and monotone through zero earnings.
    merged["earnings_yield"] = (eps / price).replace([np.inf, -np.inf], np.nan)
    merged["book_to_market"] = (bvps / price).replace([np.inf, -np.inf], np.nan)

    # Ratios for display only. Undefined at zero, and negative where earnings
    # are negative — which is precisely why they are not the modelled feature.
    merged["pe_ratio"] = (price / eps.where(eps != 0)).replace([np.inf, -np.inf], np.nan)
    merged["pb_ratio"] = (price / bvps.where(bvps > 0)).replace([np.inf, -np.inf], np.nan)

    merged = merged.rename(columns={"period_end": "fundamental_period"})
    return merged.drop(columns=["_d", "_eff"]).sort_index()


def fundamental_coverage(panel: pd.DataFrame) -> dict:
    """How much of the panel the valuation features actually reach."""
    if panel.empty or "earnings_yield" not in panel.columns:
        return {"rows": 0, "with_fundamentals": 0, "fraction": 0.0}

    have = panel["earnings_yield"].notna()
    covered = panel.loc[have]
    return {
        "rows": int(len(panel)),
        "with_fundamentals": int(have.sum()),
        "fraction": float(have.mean()),
        "tickers_covered": int(covered["ticker"].nunique()) if len(covered) else 0,
        "first_covered_date": str(covered["date"].min()) if len(covered) else None,
        "last_covered_date": str(covered["date"].max()) if len(covered) else None,
    }


def sync_fundamentals(tickers: list[str], engine=None) -> int:
    """Fetch and store in one call. Safe to re-run; upserts by period."""
    frame = fetch_fundamentals(tickers)
    return store_fundamentals(frame, engine)
