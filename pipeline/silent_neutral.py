"""
pipeline/silent_neutral.py — find a failed measurement stored as a valid value.

THREE TIMES, THE SAME DEFECT
----------------------------
1. FinBERT's loader failed, and the dashboard showed a confident NEUTRAL
   sentiment gauge for months.
2. The NSE FII/DII parser read field names the endpoint does not serve,
   defaulted to "0", and two macro columns held ONE distinct value — 0.0 —
   across all 2,601 rows since 2016, while sitting in the feature list.
3. The dependency lock dropped `lxml`, `earnings_dates` raised for every
   ticker, the exception branch wrote `earnings_surprise = 0.0`, and ~85,600
   rows of a pooled FACTOR became a constant (2026-09-22).

None of them raised, none was NULL, and every null check passed. What they
share is a SHAPE in the data, and that shape is what this module measures:

  * a large share of rows sitting on ONE value (a continuous signal almost
    never repeats a float exactly);
  * that value being an exact NEUTRAL point (0.0, 0.5, 50.0);
  * a TICKER whose column never moves (zero variance across its history);
  * a DATE on which every ticker carries the same value (zero cross-sectional
    variance), which is how a market-wide default looks.

A genuine signal can show some of these — `earnings_surprise` is meant to be
flat between announcements, and a bounded oscillator touches its bounds —
so the audit REPORTS indicators and never decides. `standing_check` is the
pipeline's automatic version: WARN-only (see the WARN/FAIL landmine in
CLAUDE.md), scoped to the recent window where a new failure lands, and
looking only for the two indicators no genuine signal in this project shows:
a whole ticker constant at a neutral point, or a whole date constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Values a failed measurement has historically been written as. A signal
#: sitting on one of these for a large share of rows is the fingerprint.
NEUTRAL_POINTS: tuple[float, ...] = (0.0, 0.5, 50.0, 1.0)

#: A run of this many consecutive identical values within one ticker counts
#: as a "long run". 20 sessions is a month; a continuous signal repeating a
#: float that long has stopped measuring anything.
LONG_RUN = 20

#: Minimum rows before a ticker's or a date's variance is judged at all.
MIN_ROWS = 10


def _longest_runs(values: pd.Series, groups: pd.Series) -> tuple[int, int]:
    """(longest run of identical consecutive values in any group, rows that sit
    in runs >= LONG_RUN). NaN never extends a run."""
    frame = pd.DataFrame({"g": groups.to_numpy(), "v": values.to_numpy(dtype=float)})
    longest, in_long = 0, 0
    for _, v in frame.groupby("g", sort=False)["v"]:
        a = v.to_numpy()
        if a.size == 0:
            continue
        change = np.ones(a.size, dtype=bool)
        change[1:] = ~((a[1:] == a[:-1]) & np.isfinite(a[1:]))
        run_id = np.cumsum(change)
        lengths = np.bincount(run_id)[run_id]
        lengths = np.where(np.isfinite(a), lengths, 0)
        longest = max(longest, int(lengths.max()))
        in_long += int((lengths >= LONG_RUN).sum())
    return longest, in_long


def column_indicators(df: pd.DataFrame, col: str) -> dict:
    """The silent-neutral indicators for one column of a (date, ticker) frame."""
    s = pd.to_numeric(df[col], errors="coerce")
    n = int(len(s))
    finite = s[np.isfinite(s)]
    out: dict = {"column": col, "rows": n,
                 "null_share": float(1 - len(finite) / n) if n else float("nan")}
    if finite.empty:
        return {**out, "modal_value": None, "modal_share": float("nan"),
                "neutral_value": None, "neutral_share": 0.0,
                "const_tickers": 0, "const_neutral_tickers": [],
                "const_dates": 0, "const_date_share": 0.0,
                "longest_run": 0, "long_run_share": 0.0, "distinct": 0}

    counts = finite.value_counts()
    modal, modal_n = float(counts.index[0]), int(counts.iloc[0])
    neutral_hits = {p: int((finite == p).sum()) for p in NEUTRAL_POINTS}
    neutral_value = max(neutral_hits, key=neutral_hits.get)

    frame = df[["date", "ticker"]].assign(v=s)
    per_ticker = frame.groupby("ticker")["v"].agg(["count", "nunique", "first"])
    per_ticker = per_ticker[per_ticker["count"] >= MIN_ROWS]
    const_t = per_ticker[per_ticker["nunique"] <= 1]
    const_neutral = sorted(const_t[const_t["first"].isin(NEUTRAL_POINTS)].index)

    per_date = frame.groupby("date")["v"].agg(["count", "nunique"])
    per_date = per_date[per_date["count"] >= MIN_ROWS]
    const_d = int((per_date["nunique"] <= 1).sum())

    ordered = frame.sort_values(["ticker", "date"], kind="mergesort")
    longest, in_long = _longest_runs(ordered["v"], ordered["ticker"])

    return {**out,
            "distinct": int(counts.size),
            "modal_value": modal,
            "modal_share": modal_n / len(finite),
            "neutral_value": neutral_value,
            "neutral_share": neutral_hits[neutral_value] / len(finite),
            "const_tickers": int(len(const_t)),
            "const_neutral_tickers": const_neutral,
            "const_dates": const_d,
            "const_date_share": const_d / max(len(per_date), 1),
            "longest_run": longest,
            "long_run_share": in_long / len(finite)}


def audit(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One row of indicators per column."""
    return pd.DataFrame([column_indicators(df, c) for c in cols if c in df.columns])


# ── the standing check ────────────────────────────────────────────────────────

#: The recent window the standing check reads. A new silent failure lands on
#: the rows the job just rewrote, and the lxml incident rewrote whole
#: histories — so ~six months is enough to see it and short enough that the
#: known, genuine pre-coverage history of a column does not trip it forever.
STANDING_WINDOW_SESSIONS = 120


def standing_findings(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """
    The two indicators no genuine signal in this project shows over a recent
    window, per column:

      * a ticker whose value never moves AND sits on a neutral point
        (the earnings_surprise = 0.0 fingerprint), and
      * a date on which every ticker carries the identical value
        (the FII/DII = 0.0 fingerprint).

    Returns human-readable findings; empty means clean.
    """
    findings: list[str] = []
    if df.empty:
        return findings
    recent_dates = sorted(df["date"].unique())[-STANDING_WINDOW_SESSIONS:]
    recent = df[df["date"].isin(recent_dates)]
    for col in cols:
        if col not in recent.columns:
            continue
        ind = column_indicators(recent, col)
        if ind["const_neutral_tickers"]:
            names = ind["const_neutral_tickers"]
            findings.append(
                f"{col}: {len(names)} ticker(s) constant at a neutral value over "
                f"the last {len(recent_dates)} sessions ({', '.join(names[:6])}"
                f"{'...' if len(names) > 6 else ''})")
        if ind["const_dates"]:
            findings.append(
                f"{col}: {ind['const_dates']} date(s) where every ticker carries "
                f"the same value")
    return findings
