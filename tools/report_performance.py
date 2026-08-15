"""
tools/report_performance.py

Regenerates the README's performance block from measured evaluation metrics.

The original README carried hand-typed figures ("~4.3% MAPE, ~85% directional
accuracy") that were wrong and stayed wrong because nothing regenerated them.
Run this after a pipeline run and paste the output between the
<!-- PERFORMANCE_BLOCK --> markers, or pass --write to substitute it in place.

    python tools/report_performance.py
    python tools/report_performance.py --write
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import pandas as pd
from sqlalchemy import text

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from data.db import get_engine
from data.universe import describe_universe_bias

MARKER = "<!-- PERFORMANCE_BLOCK -->"
README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def load_metrics() -> pd.DataFrame:
    engine = get_engine()
    try:
        return pd.read_sql(
            text("""
                SELECT ticker, eval_rank_ic, eval_rank_ic_t, eval_hit_rate,
                       eval_baseline_hit_rate, eval_mae, eval_mae_naive,
                       eval_n_oos, model_version, last_trained
                FROM model_metadata
                WHERE eval_rank_ic IS NOT NULL
            """),
            engine,
        )
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not read model_metadata: {exc}", file=sys.stderr)
        return pd.DataFrame()


def build_block(df: pd.DataFrame) -> str:
    if df.empty:
        return (
            f"{MARKER}\n\n"
            "_No evaluation metrics recorded yet. Run the pipeline, then "
            "regenerate this block with `python tools/report_performance.py`._\n"
        )

    n = len(df)
    ic = df["eval_rank_ic"].astype(float)
    ic_t = df["eval_rank_ic_t"].astype(float)
    hit = df["eval_hit_rate"].astype(float)
    base = df["eval_baseline_hit_rate"].astype(float)
    mae = df["eval_mae"].astype(float)
    mae_naive = df["eval_mae_naive"].astype(float)

    beats_baseline = int((hit > base).sum())
    beats_naive = int((mae < mae_naive).sum())
    ic_positive = int((ic > 0).sum())
    ic_significant = int((ic_t.abs() >= 2).sum())

    bias = describe_universe_bias()
    version = df["model_version"].dropna().iloc[0] if df["model_version"].notna().any() else "n/a"

    verdict = (
        "The ranking carries a weak positive signal; the magnitude does not."
        if ic.mean() > 0 and beats_naive < n / 2
        else "The model beats its baselines on both ranking and magnitude."
        if beats_naive >= n / 2 and ic.mean() > 0
        else "The model does not currently beat its baselines."
    )

    return f"""{MARKER}

Purged walk-forward validation, 30-session embargo, hyperparameters tuned inside
each training fold. Every figure below is out-of-sample and stated **before
transaction costs**. Measured {date.today().isoformat()} on {n} stocks
(model `{version}`).

| Metric | Model | Baseline | Beats baseline |
|---|---|---|---|
| Mean rank IC | **{ic.mean():+.4f}** | 0.0000 (no skill) | {ic_positive}/{n} stocks |
| Rank IC t-statistic | {ic_t.mean():+.2f} | \\|t\\| ≥ 2 | {ic_significant}/{n} stocks |
| Directional accuracy | {hit.mean():.2f}% | {base.mean():.2f}% (majority class) | {beats_baseline}/{n} stocks |
| Mean absolute error | {mae.mean():.5f} | {mae_naive.mean():.5f} (zero excess) | {beats_naive}/{n} stocks |

**How to read this.** {verdict} A rank IC around 0.05 is a real but fragile
edge — comparable to published technical-signal results and not, on its own, a
tradable strategy. t-statistics are corrected for overlapping labels: successive
30-session targets share 29 of their 30 days, so the effective sample is roughly
`n / 30`. Skipping that correction inflates every t-statistic by about 5.5×.

**Universe.** {bias.get('note', 'No membership history recorded.')}

Regenerate this block with `python tools/report_performance.py --write`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="substitute the block into README.md in place")
    args = parser.parse_args()

    block = build_block(load_metrics())

    if not args.write:
        print(block)
        return

    with open(README, encoding="utf-8") as f:
        content = f.read()

    if MARKER not in content:
        print(f"{MARKER} not found in README.md", file=sys.stderr)
        sys.exit(1)

    start = content.index(MARKER)
    end = content.find("\n---", start)
    if end == -1:
        end = len(content)

    updated = content[:start] + block.rstrip() + "\n" + content[end:]
    with open(README, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"README.md performance block updated ({len(block)} chars).")


if __name__ == "__main__":
    main()
