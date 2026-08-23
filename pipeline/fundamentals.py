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


def _material_change(old, new, rtol: float = 1e-9) -> bool:
    """
    Did a figure actually move, as opposed to arriving with different float noise?

    ``rtol`` is deliberately tiny. The job here is to filter identical values
    that round-tripped through the database and back, not to decide what counts
    as a big restatement — that judgement belongs at analysis time, on the
    recorded values, not at the write boundary where it would silently discard
    evidence.
    """
    old_missing = old is None or (isinstance(old, float) and not np.isfinite(old))
    new_missing = new is None or (isinstance(new, float) and not np.isfinite(new))
    if old_missing and new_missing:
        return False
    if old_missing or new_missing:
        return True
    scale = max(abs(float(old)), abs(float(new)), 1e-12)
    return abs(float(old) - float(new)) / scale > rtol


def _existing_rows(conn, tickers: list[str]) -> dict:
    """Current (ticker, period_end) -> figures already on file."""
    if not tickers:
        return {}
    rows = conn.execute(text(
        "SELECT ticker, period_end, eps, book_value_per_share, first_seen "
        "FROM fundamentals"
    )).fetchall()
    wanted = set(tickers)
    return {(r[0], r[1]): {"eps": r[2], "book_value_per_share": r[3],
                           "first_seen": r[4]}
            for r in rows if r[0] in wanted}


def store_fundamentals(frame: pd.DataFrame, engine=None,
                       observed_at: str | None = None) -> dict:
    """
    Upsert by (ticker, period_end), recording every figure that MOVED.

    The upsert is a current view: one row per fiscal period, holding the latest
    values we have. What is new here is that a change to a figure already on
    file is written to ``fundamental_revisions`` BEFORE it is overwritten.

    That matters because yfinance serves statements as restated rather than as
    originally filed — see the module docstring. We cannot recover restatements
    that predate our first look, but from here on a figure cannot move without
    leaving a row saying so, which is what turns the size of that bias into
    something measurable rather than something disclaimed.

    Returns counts rather than a row total, because "300 periods stored" hides
    the only interesting number in it: how many of them changed.
    """
    if frame.empty:
        return {"periods": 0, "new": 0, "revised": 0, "unchanged": 0,
                "revisions": 0}

    engine = engine or get_engine()
    observed_at = observed_at or datetime.now().date().isoformat()
    tickers = sorted(set(frame["ticker"]))

    new = revised = unchanged = 0
    revisions: list[dict] = []

    with engine.begin() as conn:
        existing = _existing_rows(conn, tickers)

        for row in frame.to_dict("records"):
            key = (row["ticker"], row["period_end"])
            prior = existing.get(key)

            if prior is None:
                new += 1
            else:
                moved = [
                    (field, prior[field], row.get(field))
                    for field in ("eps", "book_value_per_share")
                    if _material_change(prior[field], row.get(field))
                ]
                for field, old_value, new_value in moved:
                    revisions.append({
                        "ticker": row["ticker"],
                        "period_end": row["period_end"],
                        "observed_at": observed_at,
                        "field": field,
                        "old_value": old_value,
                        "new_value": new_value,
                        "first_seen": prior["first_seen"],
                        "source": row.get("source"),
                    })
                if moved:
                    revised += 1
                else:
                    unchanged += 1

            # The bind dict is built explicitly rather than splatted from the
            # row: a frame missing an optional column raised a bind-parameter
            # error naming SQLAlchemy rather than the column, and a frame
            # carrying an extra one would have reached the statement
            # uninspected.
            conn.execute(text("""
                INSERT INTO fundamentals
                    (ticker, period_end, effective_date, eps,
                     book_value_per_share, shares, source,
                     first_seen, fetched_at)
                VALUES
                    (:ticker, :period_end, :effective_date, :eps,
                     :book_value_per_share, :shares, :source,
                     :first_seen, :fetched_at)
                ON CONFLICT (ticker, period_end) DO UPDATE SET
                    effective_date = EXCLUDED.effective_date,
                    eps = EXCLUDED.eps,
                    book_value_per_share = EXCLUDED.book_value_per_share,
                    shares = EXCLUDED.shares,
                    source = EXCLUDED.source,
                    fetched_at = EXCLUDED.fetched_at,
                    first_seen = COALESCE(fundamentals.first_seen,
                                          EXCLUDED.first_seen)
            """), to_native_params({
                "ticker": row["ticker"],
                "period_end": row["period_end"],
                "effective_date": row["effective_date"],
                "eps": row.get("eps"),
                "book_value_per_share": row.get("book_value_per_share"),
                "shares": row.get("shares"),
                "source": row.get("source"),
                # Always the current date. Preserving an EARLIER first_seen is
                # the COALESCE in the statement above, and deliberately only
                # there: this was preserved in Python as well, and the two
                # guards covered for each other so completely that breaking
                # either one alone left every test green. A redundant guard is
                # an untestable one.
                "first_seen": observed_at,
                "fetched_at": observed_at,
            }))

        # Written AFTER the figures they describe and in the SAME transaction:
        # a crash between the two would otherwise leave a revision claiming a
        # change the table does not show, or a silent overwrite with no record.
        for rev in revisions:
            conn.execute(text("""
                INSERT INTO fundamental_revisions
                    (ticker, period_end, observed_at, field,
                     old_value, new_value, first_seen, source)
                VALUES
                    (:ticker, :period_end, :observed_at, :field,
                     :old_value, :new_value, :first_seen, :source)
                ON CONFLICT (ticker, period_end, field, observed_at)
                DO NOTHING
            """), to_native_params(rev))

    if revisions:
        # Loud on purpose. A restatement is the one thing in this module that
        # can invalidate a published result, so it must not arrive as a debug
        # line nobody reads.
        logger.warning(
            f"[Fundamentals] {len(revisions)} figure(s) RESTATED across "
            f"{revised} period(s) - the vendor changed values already on "
            f"file; see fundamental_revisions")

    return {"periods": int(len(frame)), "new": new, "revised": revised,
            "unchanged": unchanged, "revisions": len(revisions)}


def load_revisions(engine=None) -> pd.DataFrame:
    """Every restatement observed since vintage tracking began."""
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(
                "SELECT ticker, period_end, observed_at, field, old_value, "
                "new_value, first_seen FROM fundamental_revisions"), conn)
    except Exception:                                           # noqa: BLE001
        return pd.DataFrame()


def restatement_summary(engine=None) -> dict:
    """
    How much the vendor has revised what it already told us.

    This is the measurement the second hazard in the module docstring calls
    unsolved. It stays unsolved for HISTORY — we cannot see restatements that
    happened before we started recording — but it stops being unmeasurable
    going forward, and a summary showing zero revisions after a year of syncs
    is itself the evidence that the bias is small.
    """
    rev = load_revisions(engine)
    fund = load_fundamentals(engine)
    tracked = 0
    if not fund.empty and "first_seen" in fund.columns:
        tracked = int(fund["first_seen"].notna().sum())

    if rev.empty:
        return {"revisions": 0, "periods_tracked": tracked,
                "tickers_affected": 0, "median_abs_rel_change": None,
                "max_abs_rel_change": None, "by_field": {},
                "note": ("no restatement observed yet; only periods carrying a "
                         "non-null first_seen are being watched")}

    old = pd.to_numeric(rev["old_value"], errors="coerce")
    new = pd.to_numeric(rev["new_value"], errors="coerce")
    scale = np.maximum(old.abs(), new.abs()).replace(0.0, np.nan)
    rel = ((new - old).abs() / scale).replace([np.inf, -np.inf], np.nan)

    return {
        "revisions": int(len(rev)),
        "periods_tracked": tracked,
        "tickers_affected": int(rev["ticker"].nunique()),
        "median_abs_rel_change": (float(rel.median())
                                  if rel.notna().any() else None),
        "max_abs_rel_change": (float(rel.max())
                               if rel.notna().any() else None),
        "by_field": {str(k): int(v)
                     for k, v in rev["field"].value_counts().items()},
    }


def load_fundamentals(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            # first_seen/fetched_at are selected because restatement_summary
            # counts how many periods are actually being WATCHED. Omitting
            # them made it report zero tracked periods however many were on
            # file, which reads as "nothing to see" — the exact reassuring
            # answer this log exists to avoid giving falsely.
            return pd.read_sql(text(
                "SELECT ticker, period_end, effective_date, eps, "
                "book_value_per_share, first_seen, fetched_at "
                "FROM fundamentals"), conn)
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


# Below this share of requested tickers coming back with statements, the sync
# refuses to write. yfinance failing wholesale looks exactly like a universe of
# companies that stopped reporting, and the upsert would happily record the
# handful that did answer — leaving a cross-section where coverage correlates
# with which HTTP calls happened to succeed that morning.
MIN_FETCH_COVERAGE = 0.50


class FetchCoverageRefused(RuntimeError):
    """Raised when too few tickers returned statements to trust the batch."""


def sync_fundamentals(tickers: list[str], engine=None,
                      min_coverage: float = MIN_FETCH_COVERAGE,
                      observed_at: str | None = None) -> dict:
    """
    Fetch and store in one call. Safe to re-run; upserts by period.

    Refuses to write a batch that covers less than ``min_coverage`` of the
    requested tickers. A partial write is worse than no write here: valuation
    is a CROSS-SECTIONAL feature, standardised within each date, so a date on
    which only the tickers whose fetch succeeded carry a figure is scored
    against a mean and standard deviation taken over that arbitrary subset.
    Nothing downstream can tell that apart from a real cross-section.
    """
    frame = fetch_fundamentals(tickers)

    requested = len(set(tickers))
    returned = int(frame["ticker"].nunique()) if len(frame) else 0
    coverage = (returned / requested) if requested else 0.0

    if requested and coverage < min_coverage:
        raise FetchCoverageRefused(
            f"only {returned} of {requested} tickers returned statements "
            f"({coverage:.0%}, floor {min_coverage:.0%}) — refusing to write a "
            f"partial cross-section. The stored table is unchanged."
        )

    counts = store_fundamentals(frame, engine, observed_at=observed_at)
    counts["tickers_requested"] = requested
    counts["tickers_returned"] = returned
    counts["fetch_coverage"] = round(coverage, 4)
    logger.info(
        f"[Fundamentals] sync: {returned}/{requested} tickers, "
        f"{counts['periods']} periods ({counts['new']} new, "
        f"{counts['revised']} revised, {counts['unchanged']} unchanged)")
    return counts
