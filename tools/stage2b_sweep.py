"""
tools/stage2b_sweep.py — the min_train sweep for Stage 2b's pooled cells.

Standing policy in CLAUDE.md: **a result measured at ONE hyperparameter setting
is not a result.** The valuation finding scored t +3.32 at `min_train=380` and
+1.00 at the harness default of 500 on identical rows, and that is what retired
it. Every positive number this project has produced since is swept before it is
quoted, and the pooled cells are no exception.

Reports the NON-OVERLAPPING rebalance IC and its t-statistic, which is the
statistic the pre-registered Phase 2 bar was written against — not the
all-dates `daily_IC`, whose t is inflated roughly 5x because consecutive dates
share 29 of their 30 forward sessions.

Usage
-----
    python tools/stage2b_sweep.py --arms pooled_mae pooled_rank_ic \\
        --markdown stage2b_sweep.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.stage2b_pooled import (  # noqa: E402
    ARMS,
    BREAK_EVEN_IC,
    cell_metrics,
    load_cached_panel,
    run_arm,
)

# Centred on the harness default of 500 DATES, and spanning enough either side
# that a lone spike between flat neighbours is visible as one.
MIN_TRAIN_GRID = (380, 440, 500, 560, 620)


def sweep(arms: list[str], grid=MIN_TRAIN_GRID,
          n_trials: int | None = None) -> pd.DataFrame:
    panel = load_cached_panel()
    rows: list[dict] = []
    for arm in arms:
        objective, ticker_mode = ARMS[arm]
        for min_train in grid:
            t0 = time.time()
            kw = {} if n_trials is None else {"n_trials": n_trials}
            preds, folds = run_arm(panel, objective, ticker_mode,
                                   min_train=min_train, verbose=False, **kw)
            m = cell_metrics(preds, folds)
            rows.append({
                "arm": arm, "min_train": min_train,
                "constant_cells": m["constant_cells"],
                "reb_ic": m["reb_ic"], "reb_t": m["reb_t"],
                "n_rebalances": m["n_rebalances"],
                "cs_rank_ic": m["cs_rank_ic"],
                "mae": m["mae"], "gap": m["gap"],
                "clears_break_even": bool(m["reb_ic"] > BREAK_EVEN_IC),
            })
            print(f"  {arm} @ {min_train}: reb_IC {m['reb_ic']:+.4f} "
                  f"(t {m['reb_t']:+.2f}, n={m['n_rebalances']}), "
                  f"MAE {m['mae']:.5f}, {m['constant_cells']} constant "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def render(d: pd.DataFrame) -> str:
    o = ["\n## min_train sweep — the non-overlapping rebalance IC\n",
         "A real effect varies smoothly with the setting. An isolated spike "
         "between flat neighbours is the shape that retired the valuation "
         "result, and it has now appeared six times on this panel.\n"]

    for stat, fmt, label in (("reb_ic", "{:+.4f}", "rebalance IC"),
                             ("reb_t", "{:+.2f}", "rebalance t"),
                             ("mae", "{:.5f}", "held-out MAE"),
                             ("constant_cells", "{:.0f}", "constant cells")):
        o.append(f"\n**{label}**\n")
        grid = sorted(d["min_train"].unique())
        o.append("| arm | " + " | ".join(str(g) for g in grid) + " |")
        o.append("|---" * (len(grid) + 1) + "|")
        for arm in d["arm"].unique():
            s = d[d["arm"] == arm].set_index("min_train")[stat]
            o.append(f"| {arm} | " + " | ".join(
                fmt.format(s.get(g, float("nan"))) for g in grid) + " |")

    o.append(f"\nBreak-even rank IC at zero market impact: "
             f"**{BREAK_EVEN_IC:.4f}**; at 25bp, 0.0166 (P4). The "
             f"pre-registered Phase 2 bar is a positive rebalance IC with "
             f"**t > 2**.\n")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["pooled_mae", "pooled_rank_ic"])
    ap.add_argument("--grid", nargs="*", type=int, default=list(MIN_TRAIN_GRID))
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    d = sweep(args.arms, grid=tuple(args.grid), n_trials=args.trials)
    text = render(d)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.markdown}")
    if args.csv:
        d.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
