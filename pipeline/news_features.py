"""
pipeline/news_features.py — point-in-time news features for the panel.

FOUR RULES, EACH ONE PAID FOR BY A DEFECT THIS PROJECT ALREADY SHIPPED
-----------------------------------------------------------------------

1. THE WINDOW IS 30 SESSIONS, NOT 5, AND THAT IS FORCED BY THE DATA.
   Measured over the first backfilled tickers, articles per ticker-month run
   0.89 in 2016 and 6.87 in 2024 — a 7.7x gradient (2-4x for established names,
   which is the honest figure; the early sample is Adani-heavy). At ~1 article
   a month a five-session window is EMPTY for most rows before ~2022, so the
   feature would be almost entirely missing exactly where the panel is longest.
   Thirty sessions also happens to match the forecast horizon, which is the
   conceptually right choice independently.

2. NOTHING ENTERS AS A LEVEL. A ticker's average news volume is a near-constant
   per-ticker attribute, and CLAUDE.md §7 measures that shape as worth a
   `pooled_xgb` rebalance t of +0.77 from two RANDOM CONSTANTS carrying no
   information at all. The tree identifies the ticker from the constant and
   learns which names paid in the training window. So counts are expressed
   against the ticker's OWN trailing baseline, never raw.

3. "NOT OBSERVED" IS NOT "NO NEWS". A window we searched and found empty is a
   measurement and its count is 0. A window we never searched, or one Google
   refused, has no measurement and its features are NULL. Collapsing those is
   how the live archive came to record a blocked fetch for all 95 tickers as a
   market-wide silent day, and how a dead FinBERT loader displayed a confident
   NEUTRAL gauge for months.

4. THE AS-OF BOUNDARY IS PUBLICATION TIME. A row dated t may use articles
   published on or before t and no others. This is the same guarantee
   `series._history_ending_at` carries, and it fails the same way: an off-by-one
   hands the model tomorrow's news and reads as a breakthrough.

WHAT THIS DOES NOT DO
---------------------
It does not decide whether news helps. `pipeline.baselines` scores a comparator
built on these columns against the same folds, rows and floors as everything
else, and the pre-registered bar carries a late-fold control — because coverage
grows with time, so any positive result lands where the panel is densest and is
otherwise indistinguishable from "the recent period is easier".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.db import get_engine
from pipeline.signals import HORIZON_SESSIONS

logger = logging.getLogger(__name__)

#: Sessions of news behind each row. Matches the forecast horizon; see rule 1.
NEWS_WINDOW_SESSIONS = HORIZON_SESSIONS

#: Sessions of history used for a ticker's OWN baseline, against which its
#: current volume is expressed. A year, so the baseline is not itself moved by
#: the window it is normalising.
BASELINE_SESSIONS = 252

#: The columns this module produces. Kept here so `panel` and `baselines` cannot
#: hold different ideas of what exists.
NEWS_COLS: list[str] = [
    "news_count_excess",     # volume against the ticker's own trailing median
    "news_sent_mean",        # mean signed score over the window
    "news_sent_disp",        # dispersion — disagreement, not direction
    "news_sent_momentum",    # this window's mean minus the previous window's
]


def load_scored_articles(engine=None, scorer_id: str | None = None) -> pd.DataFrame:
    """
    One row per (published_at, ticker): article count and score statistics.

    SCORED UNDER EXACTLY ONE scorer_id. Mixing two checkpoints in one column
    would put two different quantities under one name, which is the failure
    that got sentiment removed the first time (F7). The default is whichever
    scorer holds the most rows, so a half-finished re-score cannot silently
    contaminate a feature.
    """
    from pipeline.news_scoring import current_scorer_id

    engine = engine or get_engine()
    scorer_id = scorer_id or current_scorer_id(engine)
    if not scorer_id:
        return pd.DataFrame(columns=["date", "ticker", "n", "mean", "sd"])

    # AGGREGATED IN PANDAS, NOT IN SQL. `STDDEV_SAMP` is a Postgres function
    # and SQLite has no equivalent, so a SQL-side aggregate would work in
    # production and raise in every test — which is the reverse of useful, and
    # the test suite caught it on the first run. The row count is modest
    # (~46,000 scored articles for the whole universe), so pulling them and
    # grouping here costs nothing and behaves identically on both engines.
    raw = pd.read_sql(text("""
        SELECT substr(a.published_at, 1, 10) AS date,
               m.ticker                      AS ticker,
               s.score                       AS score
        FROM news_articles a
        JOIN news_mentions m ON m.article_id = a.article_id
        JOIN news_scores  s ON s.article_id = a.article_id
                           AND s.scorer_id = :sid
    """), engine, params={"sid": scorer_id})
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "n", "mean", "sd"])

    grouped = (raw.groupby(["date", "ticker"], as_index=False)["score"]
               .agg(n="size", mean="mean", sd="std"))
    # A single article has no sample SD; zero is the right value for pooling,
    # and it is a real statement — one observation has no dispersion.
    grouped["sd"] = grouped["sd"].fillna(0.0)
    return grouped


def load_observed_months(engine=None) -> set[tuple[str, str]]:
    """
    (ticker, 'YYYY-MM') pairs we actually searched and got an answer for.

    Rule 3 lives here. Only windows recorded `ok` count as observed: a blocked
    or errored window is a period we could not see, and a feature over it must
    be NULL rather than zero.
    """
    engine = engine or get_engine()
    rows = pd.read_sql(text(
        "SELECT ticker, window_start, window_end FROM news_coverage "
        "WHERE status = 'ok'"), engine)
    observed: set[tuple[str, str]] = set()
    for _, r in rows.iterrows():
        for month in pd.period_range(str(r["window_start"])[:10],
                                     str(r["window_end"])[:10], freq="M"):
            observed.add((r["ticker"], str(month)))
    return observed


def build_news_features(panel: pd.DataFrame, engine=None,
                        scorer_id: str | None = None,
                        window: int = NEWS_WINDOW_SESSIONS,
                        baseline: int = BASELINE_SESSIONS) -> pd.DataFrame:
    """
    Returns (date, ticker, NEWS_COLS...) aligned to the panel's own date grid.

    Rolling windows are counted in SESSIONS off the panel's grid, not in
    calendar days. A calendar window silently changes length across holidays
    and long weekends, and this project has already lost rows to a horizon that
    was row-stepped across holes in the grid.
    """
    engine = engine or get_engine()
    if panel.empty:
        return pd.DataFrame(columns=["date", "ticker"] + NEWS_COLS)

    daily = load_scored_articles(engine, scorer_id)
    observed = load_observed_months(engine)

    grid = sorted(panel["date"].astype(str).unique())
    grid_index = pd.Index(grid, name="date")
    out_frames = []

    for ticker, rows in panel.groupby("ticker", sort=False):
        dates = sorted(rows["date"].astype(str).unique())
        mine = daily[daily["ticker"] == ticker]

        # Reindex onto the FULL grid so a rolling window spans sessions even
        # where this ticker has no row, then restrict at the end.
        counts = pd.Series(0.0, index=grid_index)
        totals = pd.Series(0.0, index=grid_index)
        sqsums = pd.Series(0.0, index=grid_index)
        if not mine.empty:
            agg = mine.set_index("date")
            common = agg.index.intersection(grid_index)
            counts.loc[common] = agg.loc[common, "n"].astype(float)
            totals.loc[common] = (agg.loc[common, "n"] * agg.loc[common, "mean"]).astype(float)
            # Sum of squares, so a window mean and SD can both be pooled from
            # daily aggregates without re-reading every article.
            sqsums.loc[common] = (
                agg.loc[common, "n"] *
                (agg.loc[common, "mean"] ** 2 + agg.loc[common, "sd"] ** 2)
            ).astype(float)

        # THE AS-OF BOUNDARY. `closed="right"` on a trailing window includes
        # the current session and nothing after it — articles published ON date
        # t are usable at t, articles published at t+1 are not.
        n_win = counts.rolling(window, min_periods=1).sum()
        s_win = totals.rolling(window, min_periods=1).sum()
        q_win = sqsums.rolling(window, min_periods=1).sum()

        mean = np.where(n_win > 0, s_win / n_win.replace(0, np.nan), np.nan)
        var = np.where(n_win > 1,
                       np.maximum(q_win / n_win.replace(0, np.nan) - mean ** 2, 0.0),
                       np.nan)

        # Rule 2: volume against the ticker's OWN trailing median, never raw.
        base = counts.rolling(baseline, min_periods=window).median()
        excess = n_win / float(window) - base

        prev_mean = pd.Series(mean, index=grid_index).shift(window)

        frame = pd.DataFrame({
            "date": grid_index,
            "ticker": ticker,
            "news_count_excess": excess.to_numpy(),
            "news_sent_mean": mean,
            "news_sent_disp": np.sqrt(var),
            "news_sent_momentum": pd.Series(mean, index=grid_index) - prev_mean,
        })

        # Rule 3: a date whose month we never successfully searched has NO
        # measurement, so every column is NULL rather than zero.
        months = pd.Index(frame["date"]).str.slice(0, 7)
        seen = np.array([(ticker, m) in observed for m in months])
        frame.loc[~seen, NEWS_COLS] = np.nan

        out_frames.append(frame[frame["date"].isin(dates)])

    result = pd.concat(out_frames, ignore_index=True) if out_frames else \
        pd.DataFrame(columns=["date", "ticker"] + NEWS_COLS)
    return result


def news_coverage_by_fold(panel: pd.DataFrame, features: pd.DataFrame,
                          fold_col: str = "fold") -> pd.DataFrame:
    """
    What fraction of each fold's rows actually carry a news feature.

    READ THIS BEFORE ANY RESULT. Coverage grows with time, so a comparator
    built on these columns is effectively live only in the late folds — and a
    positive number there is indistinguishable from "the recent period is
    easier" unless the same rows are also scored with the feature removed. That
    control is the pre-registered condition on this whole line of work.
    """
    if fold_col not in panel.columns:
        merged = panel.merge(features, on=["date", "ticker"], how="left")
        merged["fold"] = 0
    else:
        merged = panel.merge(features, on=["date", "ticker"], how="left")

    rows = []
    for fold, grp in merged.groupby("fold"):
        rows.append({
            "fold": fold,
            "n_rows": len(grp),
            "with_news": int(grp["news_sent_mean"].notna().sum()),
            "coverage": float(grp["news_sent_mean"].notna().mean()),
            "first_date": grp["date"].min(),
            "last_date": grp["date"].max(),
        })
    return pd.DataFrame(rows)
