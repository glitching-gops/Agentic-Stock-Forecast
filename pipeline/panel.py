"""
pipeline/panel.py — The cross-sectional panel.

Everything before Phase 2 modelled one ticker at a time: 95 independent
XGBoost models, each fitted on its own ~2,400 rows, each measured on its own
walk-forward. Phase 2 measures those models against each other on the SAME
DAY, across the cross-section, because that is the only comparison in which a
common market move cancels out. A per-ticker model is never trained on that
ordering, so it has no reason to be good at it.

(The product no longer ranks stocks — that layer is gone. The cross-sectional
measurement stays, because it is still the right way to separate skill from
beta: roughly a third of return variance is common across these names, and a
per-ticker metric credits a model for capturing it.)

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
# market as a whole, which is not what this measures. They stay in
# FEATURES because the per-ticker models are fitted on time series and can use
# them there; they are excluded from the pooled factor set for this reason and
# not by oversight.
#
# `fii_net_flow` / `dii_net_flow` WERE HERE AND ARE NOT ANY MORE. Measured
# 2026-09-03 they held one distinct value across all 2,601 macro rows — 0.0 —
# because the NSE parser read field names the endpoint does not serve. Real
# flows now accumulate in `market_flows`; they rejoin a feature list only when
# there is history, because a column that is constant until 2026-09 and real
# after it is a structural break, not a feature. See pipeline/macro.py.
MACRO_COLS = [
    "usdinr", "india_vix", "nifty_5d_return", "nifty_20d_return",
]

FEATURES = FEATURE_COLS + MACRO_COLS

#: What the model predicts. ABSOLUTE 30-session log return since P1.
#:
#: The excess-return label is not deleted — it is still computed, still stored,
#: and still loaded into every panel as EXCESS_TARGET. Dropping it would make
#: every Phase 2 result permanently unreadable: six foundation-model
#: configurations, a linear probe, LoRA, a ridge, a tree and the valuation
#: experiment were all scored against it, and a number you cannot re-measure is
#: a number you cannot compare against.
#:
#: THE SWITCH MAKES THE TARGET EASIER TO FAKE, and that is the thing to hold on
#: to. Measured over 84 tickers and 205,973 windows: 57.67% of 30-session
#: absolute returns are positive, against ~52% for excess return, so a model
#: that always says "up" gains ~5.7 points of apparent accuracy over the excess
#: target while learning nothing. And 32.8% of return variance is COMMON across
#: stocks, so predicting the market captures a third of the target with zero
#: stock-specific information. `zero` stops being the right floor the moment
#: the target has drift in it — see baselines.always_up / market / beta_market,
#: which exist for exactly this.
TARGET = "target_return"

#: The previous target, kept alongside. Loaded into every panel so a Phase 2
#: comparison can be re-run on the label it was originally measured against.
EXCESS_TARGET = "target_excess_return"

TARGETS = (TARGET, EXCESS_TARGET)

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

    Returns columns: date, ticker, close, FEATURES..., target_return,
    target_excess_return, benchmark_ticker — sorted by (date, ticker).

    BOTH targets are loaded, always. The excess label is the only comparable
    record of everything Phase 2 already tried, and a panel that carried only
    the current target would make re-running any of it impossible without a
    second loader.
    """
    engine = engine or get_engine()

    # benchmark_close arrives via the safe-migration list in data/db.py, which
    # runs at the start of both workflows. Selecting it unconditionally would
    # turn "the migration has not run here yet" into a hard failure of the whole
    # comparison, when the correct behaviour is a panel with no relative-price
    # series and a note saying so. Both jobs call init_db() first, so this is a
    # transitional gap rather than a permanent one — but a read path should not
    # depend on a write path having run.
    available = set(_table_columns(engine, "signals"))
    optional = [c for c in ("benchmark_close", "benchmark_ticker")
                if c in available]

    # EXCESS_TARGET is optional for the same reason benchmark_close is: a
    # database that predates the column should produce a panel without it and
    # a note, not a hard failure of the whole comparison.
    optional_targets = [c for c in (EXCESS_TARGET,) if c in available]

    cols = ", ".join(["date", "ticker", "close", *FEATURE_COLS, TARGET,
                      *optional_targets, *optional])
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

    for col in ("benchmark_close", "benchmark_ticker"):
        if col not in panel.columns:
            panel[col] = np.nan

    panel = _attach_macro(panel, engine)

    for col in FEATURES:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        panel[col] = panel[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for col in TARGETS:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
        else:
            panel[col] = np.nan
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def _table_columns(engine, table: str) -> list[str]:
    """Column names on `table`, or an empty list if it cannot be inspected."""
    try:
        from sqlalchemy import inspect

        return [c["name"] for c in inspect(engine).get_columns(table)]
    except Exception:                                          # noqa: BLE001
        return []


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


def retarget_horizon(panel: pd.DataFrame, horizon: int,
                     min_coverage: float = 0.90,
                     target: str = TARGET) -> pd.DataFrame:
    """
    Replaces `target` with the same quantity measured over `horizon` sessions.

    Free, and exact, because of the identity in ``price_frame``: the h-session
    forward difference of that frame IS the h-session label, for any h.
    Nothing has to be refetched and no approximation is introduced — the stored
    30-session label is simply one value of h among many.

    This is what makes a horizon sweep affordable. The alternative would be
    recomputing labels through ``pipeline.signals`` at each horizon, which
    means refetching every benchmark and rewriting the signals table four
    times.

    The coverage refusal applies ONLY to the excess target, and that asymmetry
    is the point rather than an oversight. The relative series needs
    ``benchmark_close``, and a ticker whose level was never backfilled produces
    an all-NaN column, drops silently out of the labelled set, and leaves the
    sweep comparing horizons over different universes. The absolute series
    needs only ``close``, which every row has by construction, so there is no
    coverage question to ask.
    """
    wide = price_frame(panel, target)
    if wide.empty:
        raise ValueError(
            f"retarget_horizon found no usable price series for {target}. "
            f"Recompute signals before sweeping horizons."
        )

    if target == EXCESS_TARGET:
        labelled = int(pd.to_numeric(panel[target], errors="coerce").notna().sum())
        have_level = int(panel["benchmark_close"].notna().sum())
        if labelled and have_level / labelled < min_coverage:
            raise ValueError(
                f"benchmark_close covers {have_level / labelled:.1%} of labelled "
                f"rows, below {min_coverage:.0%}. Retargeting would drop the "
                f"uncovered tickers and compare horizons over different universes."
            )

    # Shifted WITHIN each ticker, never across the shared date grid.
    #
    # The wide frame's index is the UNION of every ticker's dates, so a ticker
    # absent on a date another one trades — a later listing, a suspension, a
    # dropped row — has a placeholder there. `wide.shift(-h)` would step h rows
    # of that union, which is more than h sessions for such a ticker, and the
    # label would silently measure a longer horizon for exactly the names whose
    # history is most irregular. Measured: it broke the identity by 4.8e-02
    # against a stored label that reproduces at 1e-15 when shifted per ticker.
    rel = panel[["date", "ticker"]].copy()
    rel["_rel"] = _log_price_basis(panel, target)

    rel = rel.sort_values(["ticker", "date"], kind="mergesort")
    grouped = rel.groupby("ticker", sort=False)["_rel"]
    rel["_retarget"] = grouped.shift(-horizon) - rel["_rel"]

    out = panel.drop(columns=[target]).merge(
        rel[["date", "ticker", "_retarget"]], on=["date", "ticker"], how="left")
    return out.rename(columns={"_retarget": target})


def _log_price_basis(panel: pd.DataFrame, target: str) -> pd.Series:
    """
    The log series whose h-session forward difference IS `target`.

    Absolute: ``log(close)``. Excess: ``log(close / benchmark_close)``.

    ONE function, because the two must not be able to disagree. A series model
    handed the relative basis while being scored against the absolute label is
    measuring a quantity nobody asked for, and it fails SILENTLY: the table
    renders, the numbers are the right order of magnitude, and the only symptom
    is a comparator that mysteriously will not beat the floor. Deriving the
    basis from the target name makes that combination unconstructable.
    """
    close = pd.to_numeric(panel["close"], errors="coerce")
    if target == TARGET:
        return np.log(close.where(close > 0))
    if target == EXCESS_TARGET:
        bench = pd.to_numeric(panel["benchmark_close"], errors="coerce")
        ratio = (close / bench).replace([np.inf, -np.inf], np.nan)
        return np.log(ratio.where(ratio > 0))
    raise ValueError(f"no price basis is defined for target {target!r}")


def price_frame(panel: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """
    Wide log-price series per ticker, whose forward difference IS `target`.

    Returned wide: index of dates, one column per ticker, sorted. A ticker with
    no usable basis is all-NaN rather than absent, so a caller sees a gap
    instead of silently scoring a smaller universe.
    """
    if panel.empty:
        return pd.DataFrame()
    if target == EXCESS_TARGET and "benchmark_close" not in panel.columns:
        return pd.DataFrame()

    frame = panel[["date", "ticker"]].assign(_rel=_log_price_basis(panel, target))
    wide = frame.pivot_table(index="date", columns="ticker", values="_rel",
                             aggfunc="last")
    return wide.sort_index()


def relative_price_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """
    The log relative-price series per ticker: ``log(close / benchmark_close)``.

    Retained as the EXCESS_TARGET case of ``price_frame`` because the identity
    below is worth stating in full somewhere, and because it is what every
    Phase 2 series result was measured on.

    THE POINT OF THIS FUNCTION IS AN EXACT IDENTITY. For a horizon h,

        log(rel[t+h]) - log(rel[t])
            = (log S[t+h] - log S[t]) - (log B[t+h] - log B[t])
            = target_return - benchmark_return
            = target_excess_return

    so the h-session forward log return of this series **is** the label, with no
    approximation. A time-series model handed this series predicts the excess
    return directly. The alternative — forecast the stock, forecast the index,
    subtract — compounds two independent errors into a quantity smaller than
    either of them, and is the main reason a univariate foundation model looks
    unsuited to a benchmark-relative target. It is not, given the right series.

    Returned wide: index of dates, one column per ticker, sorted. A ticker whose
    benchmark_close has not been backfilled yet is all-NaN rather than absent,
    so a caller sees a gap instead of silently scoring a smaller universe.

    The absolute analogue is ``log(close)`` and needs no benchmark at all,
    which is the second thing the target switch buys: eight of ten NSE sector
    indices stopped publishing around 2026-07-20, and that outage cannot reach
    a label that never asks an index anything.
    """
    return price_frame(panel, EXCESS_TARGET)


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
        # Rows carrying a benchmark LEVEL, which is what the relative-price
        # series needs. Distinct from label coverage: a row written before
        # benchmark_close was persisted has a perfectly good target and no way
        # to reconstruct the series it came from.
        "rows_with_benchmark_close": (
            int(pd.to_numeric(panel["benchmark_close"], errors="coerce").notna().sum())
            if "benchmark_close" in panel.columns else 0),
    }
