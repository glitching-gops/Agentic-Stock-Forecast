"""
tools/stage1c_announcement_check.py — does the market react to this SUE at
all? The instrument check behind Stage 1c's null.

Descriptive, added AFTER the pre-registered run, and decides nothing. A
seasonal-random-walk SUE that is measured correctly should line up with the
announcement return itself: the move from the last close BEFORE the filing
was public to the close of the first session it was usable. A SUE that
the market ignores on the day would make a drift null uninterpretable,
because it could mean "no PEAD" or "no surprise measured".

The announcement return is taken net of the equal-weighted panel over the
same window, and every window spanning a bonus or split ex-date is dropped.

    python tools/stage1c_announcement_check.py [--markdown out.md]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.earnings import usable_session  # noqa: E402
from tools.backfill_results import CACHE as RESULTS_CACHE  # noqa: E402
from tools.stage1b_phantom_check import PHANTOMS  # noqa: E402
from tools.stage1c_sue import sue_announcements  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")


def announcement_returns(ann: pd.DataFrame, actions: pd.DataFrame,
                         panel: pd.DataFrame) -> pd.DataFrame:
    grid = sorted(set(panel["date"]) - set(PHANTOMS))
    close = panel.pivot_table(index="date", columns="ticker", values="close").reindex(grid)
    lr = np.log(close)
    market = lr.diff().mean(axis=1)
    pos = {d: i for i, d in enumerate(grid)}
    splits = actions[actions["kind"].isin(["bonus", "split", "split_yf"])]
    out = []
    for _, a in ann.dropna(subset=["sue"]).iterrows():
        t = a["symbol"] + ".NS"
        if t not in close.columns:
            continue
        u = usable_session(a["disclosed"], grid)
        if u is None:
            continue
        day = pd.Timestamp(a["disclosed"]).strftime("%Y-%m-%d")
        i1 = pos[u]
        i0 = int(np.searchsorted(np.asarray(grid), day, side="left")) - 1
        if i0 < 0 or i1 <= i0:
            continue
        d0, d1 = grid[i0], grid[i1]
        sp = splits[(splits["symbol"] == a["symbol"])
                    & (splits["ex_date"] > pd.Timestamp(d0))
                    & (splits["ex_date"] <= pd.Timestamp(d1))]
        if len(sp):
            continue
        r = lr.at[d1, t] - lr.at[d0, t]
        m = market.iloc[i0 + 1:i1 + 1].sum()
        if np.isfinite(r) and np.isfinite(m):
            out.append({"symbol": a["symbol"], "period_end": a["period_end"],
                        "sue": a["sue"], "ann_ret": r - m, "sessions": i1 - i0,
                        "year": d1[:4]})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--results-cache", default=RESULTS_CACHE)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    panel = pd.read_parquet(args.panel_cache, columns=["date", "ticker", "close"])
    panel["date"] = panel["date"].astype(str)
    ann, actions = sue_announcements(args.results_cache)
    r = announcement_returns(ann, actions, panel)

    rho, p = spearmanr(r["sue"], r["ann_ret"])
    r["q"] = pd.qcut(r["sue"], 5, labels=[1, 2, 3, 4, 5])
    by_q = r.groupby("q", observed=True)["ann_ret"].agg(["mean", "count"])
    # one Spearman per reporting season (calendar quarter of the period end),
    # so the t treats seasons, not events, as the independent units
    seasons = r.groupby("period_end").filter(lambda g: len(g) >= 20) \
        .groupby("period_end").apply(lambda g: spearmanr(g["sue"], g["ann_ret"])[0])
    t_seasons = seasons.mean() / (seasons.std(ddof=1) / np.sqrt(len(seasons)))
    o = ["## Does the market react to the SUE? (descriptive, post hoc)\n",
         f"{len(r):,} announcements with a SUE and a clean window (median "
         f"{int(r['sessions'].median())} session(s)), return net of the equal-weighted panel.\n",
         f"- Pooled Spearman(SUE, announcement return): **{rho:+.3f}** (p {p:.1e}).",
         f"- Per reporting season: mean {seasons.mean():+.3f} over {len(seasons)} seasons, "
         f"t **{t_seasons:+.2f}**, positive in {int((seasons > 0).sum())} of {len(seasons)}.\n",
         "| SUE quintile | mean announcement return | events |", "|---|---|---|"]
    for q, row in by_q.iterrows():
        o.append(f"| {q} | {row['mean']:+.4f} | {int(row['count']):,} |")
    o.append(f"\nTop minus bottom quintile: "
             f"{by_q['mean'].iloc[-1] - by_q['mean'].iloc[0]:+.4f}.")
    yr = r.groupby("year").apply(lambda g: spearmanr(g["sue"], g["ann_ret"])[0])
    o.append("By year: " + ", ".join(f"{y} {v:+.2f}" for y, v in yr.items()) + ".")
    text = "\n".join(o)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
