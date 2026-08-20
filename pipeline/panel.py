"""
pipeline/panel.py — The cross-sectional panel.

Everything before Phase 2 modelled one ticker at a time: 95 independent
XGBoost models, each fitted on its own ~2,400 rows, each measured on its own
walk-forward. That is not how the output is used. The leaderboard ranks
tickers *against each other on the same day*, so the quantity that matters is
cross-sectional ordering, and a per-ticker model is never trained on it.

This module builds the substrate for models that are: a long panel indexed by
(date, ticker), loaded in one query, with the macro block aligned onto the
panel's own date grid.

Two design decisions here are load-bearing.

**Price-level features are not usable pooled.** Nine of the twenty-four
technical columns are denominated in rupees or in share counts:

    macd_hist, obv, sma_20, ema_9, ema_21, ema_50, bb_upper, bb_lower, atr_14

Per ticker they are at least on a consistent scale. Pooled they are not: a
`sma_20` of 3,000 versus 50 says which company you are looking at, not what is
about to happen to it. `obv` is worse than the rest — it is a cumulative sum
running from the first row of the series, so its level encodes how long the
history is and how heavily the name trades. A pooled learner handed these
columns will partition on ticker identity and report the in-sample fit that
follows. That is the same failure mode as F1, arrived at from a different
direction, so the split is made explicit here (`SCALE_FREE` / `PRICE_SCALED`)
rather than left for a modeller to notice.

**Cross-sectional standardisation does not leak, time-series standardisation
does.** Z-scoring a column within a single date uses only values that were all
observable on that date, so it is safe to apply before splitting. Z-scoring the
same column over its whole history uses a mean and standard deviation computed
partly from the future. The first is what `cross_sectional_zscore` does; the
second is what it must never be relaxed into. tests pin this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.db import get_engine
from pipeline.signals import FEATURE_COLS

# Macro columns joined onto every row by date. Kept here rather than imported
# from pipeline.model so that the panel does not pull in XGBoost; a test pins
# the two lists together so they cannot drift apart silently.
#
# Note what these are in a CROSS-SECTIONAL model: constants. Every ticker sees
# the same USDINR and the same India VIX on a given date, so after
# cross_sectional_zscore they are identically zero and carry no ranking
# information whatsoever. They can only ever help a model that is timing the
# market as a whole, which is not what the leaderboard is. They stay in
# FEATURES because the per-ticker models are fitted on time series and can use
# them there; they are excluded from the pooled factor set for this reason and
# not by oversight.
MACRO_COLS = [
    "usdinr", "india_vix", "nifty_5d_return", "nifty_20d_return",
    "fii_net_flow", "dii_net_flow",
]

FEATURES = FEATURE_COLS + MACRO_COLS

TARGET = "target_excess_return"

# Denominated in rupees or share counts, so their level differs between
# tickers for reasons that have nothing to do with the forecast. See the
# module docstring: these are excluded from pooled feature sets by default.
PRICE_SCALED = [
    "macd_hist", "obv", "sma_20", "ema_9", "ema_21", "ema_50",
    "bb_upper", "bb_lower", "atr_14",
]

# Ratios, oscillators, percentages and returns. Comparable across tickers as
# they stand, which is what makes them candidates for a pooled model.
SCALE_FREE = [c for c in FEATURE_COLS if c not in PRICE_SCALED]

# A date carrying fewer names than this cannot support a cross-sectional
# statistic worth computing: a z-score over four observations is noise, and a
# quintile spread over four names is two names against two.
MIN_NAMES_PER_DATE = 10

# Winsorisation bound for cross-sectional z-scores, in standard deviations.
# One name gapping 20% on results day otherwise dominates the fit for that
# date under any squared-error objective.
ZSCORE_CLIP = 3.0


def load_panel(
    tickers: list[str] | None = None,
    engine=None,
    start: str | None = None,
) -> pd.DataFrame:
    """
    Loads every ticker's signals into one long frame, macro included.

    One query for the signals and one for the macro block, joined in pandas.
    Deliberately not a loop over tickers: `get_universe()` used to run two
    queries per ticker and cost ~200 sequential round trips per request, which
    is what made `/api/stocks` a 51-second call.

    Returns columns: date, ticker, close, FEATURES..., target_excess_return,
    benchmark_ticker — sorted by (date, ticker).
    """
    engine = engine or get_engine()

    cols = ", ".join(["date", "ticker", "close", *FEATURE_COLS, TARGET,
                      "benchmark_ticker"])
    where, params = [], {}
    if start:
        where.append("date >= :start")
        params["start"] = start
    if tickers:
        keys = [f"t{i}" for i in range(len(tickers))]
        where.append(f"ticker IN ({', '.join(':' + k for k in keys)})")
        params.update(dict(zip(keys, tickers)))

    sql = f"SELECT {cols} FROM signals"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date ASC, ticker ASC"

    panel = pd.read_sql(text(sql), engine, params=params)
    if panel.empty:
        return panel

    panel = _attach_macro(panel, engine)

    for col in FEATURES:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        panel[col] = panel[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    panel[TARGET] = pd.to_numeric(panel[TARGET], errors="coerce")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def _attach_macro(panel: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Joins the macro block onto the panel's date grid.

    The forward fill happens on the macro frame indexed by date, BEFORE the
    merge — never after. Filling after the merge would carry a value down the
    long frame from one ticker's row into the next ticker's row, since the
    panel is not one series. The reindex spans the union of both date sets so
    that a macro observation on a date the panel does not contain still
    propagates forward to the dates it does.
    """
    macro = pd.read_sql(text("SELECT * FROM macro ORDER BY date ASC"), engine)

    if macro.empty:
        for col in MACRO_COLS:
            panel[col] = 0.0
        return panel

    macro = macro.drop_duplicates(subset="date").set_index("date").sort_index()
    for col in MACRO_COLS:
        if col not in macro.columns:
            macro[col] = np.nan

    grid = pd.Index(sorted(panel["date"].unique()))
    aligned = (macro[MACRO_COLS]
               .reindex(macro.index.union(grid))
               .sort_index()
               .ffill()                      # never bfill: that imports the future (F12)
               .reindex(grid))
    aligned.index.name = "date"

    return panel.merge(aligned.reset_index(), on="date", how="left")


def cross_sectional_zscore(
    panel: pd.DataFrame,
    cols: list[str],
    min_names: int = MIN_NAMES_PER_DATE,
    clip: float = ZSCORE_CLIP,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Standardises `cols` within each date.

    This is the transform that makes a pooled model answer "which of today's
    names looks best" rather than "what is the unconditional level of this
    indicator". It is computed per date from that date's own cross-section, so
    every input was observable at the time — there is no lookahead to purge,
    and it is therefore safe to apply once, before splitting.

    A date with fewer than `min_names` observations is left at zero rather than
    standardised: dividing by the standard deviation of six numbers manufactures
    outliers instead of removing them. A column that is constant across a date
    is likewise zeroed, since it carries no cross-sectional information that day.
    """
    out = panel.copy()
    present = [c for c in cols if c in out.columns]
    if not present:
        return out

    grouped = out.groupby("date")[present]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    count = grouped.transform("count")

    z = (out[present] - mean) / std.replace(0.0, np.nan)
    z = z.where(count >= min_names, 0.0)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip, clip)

    for col in present:
        out[col + suffix] = z[col]
    return out


def usable_dates(panel: pd.DataFrame, min_names: int = MIN_NAMES_PER_DATE) -> list[str]:
    """Dates carrying enough labelled names to be ranked."""
    labelled = panel[panel[TARGET].notna()]
    if labelled.empty:
        return []
    counts = labelled.groupby("date")["ticker"].nunique()
    return sorted(counts[counts >= min_names].index.tolist())


def panel_coverage(panel: pd.DataFrame) -> dict:
    """
    Describes the panel's shape, so a result can be read next to the breadth
    that produced it.

    `median_names_per_date` is the number that decides whether a
    cross-sectional claim means anything. A rank IC computed across 12 names is
    a different measurement from one computed across 95, and reporting them as
    the same number is how a thin panel gets read as a strong one.
    """
    if panel.empty:
        return {"rows": 0, "tickers": 0, "dates": 0}

    labelled = panel[panel[TARGET].notna()]
    per_date = (labelled.groupby("date")["ticker"].nunique()
                if not labelled.empty else pd.Series(dtype=int))

    return {
        "rows": int(len(panel)),
        "labelled_rows": int(len(labelled)),
        "tickers": int(panel["ticker"].nunique()),
        "dates": int(panel["date"].nunique()),
        "first_date": str(panel["date"].min()),
        "last_date": str(panel["date"].max()),
        "median_names_per_date": float(per_date.median()) if len(per_date) else 0.0,
        "min_names_per_date": int(per_date.min()) if len(per_date) else 0,
        "max_names_per_date": int(per_date.max()) if len(per_date) else 0,
        "dates_with_enough_breadth": int((per_date >= MIN_NAMES_PER_DATE).sum()),
        "benchmarks": sorted(panel["benchmark_ticker"].dropna().unique().tolist())
        if "benchmark_ticker" in panel.columns else [],
    }
