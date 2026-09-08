"""
tools/stage0c_live_gate.py — what the corrected gate does to the published grades.

The live path is `pipeline.evaluation.compute_metrics` -> `eval_rank_ic` ->
`agents.critic_agent.grade_evidence` -> `forecast_confidence`. Stage 0c changes
it twice:

  1. `rank_ic` becomes the WITHIN-FOLD average instead of one correlation over
     the pooled folds (Stage 0b's fix, now on the live path);
  2. STRONG additionally requires `eval_rw_significant` — the panel-level
     Romano-Wolf rejection, which is the only altitude multiplicity can be
     controlled at.

This RUNS THE REAL GRADER on both the old and the new inputs and reports every
ticker whose published badge would move. It writes nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.critic_agent import grade_evidence  # noqa: E402
from pipeline.evaluation import (  # noqa: E402
    effective_sample_size,
    hit_rate,
    majority_hit_rate,
    rank_ic,
)
from pipeline.evidence_panel import grade_panel_v3  # noqa: E402
from pipeline.evidence_shrinkage import rank_ic_rows  # noqa: E402
from pipeline.signals import HORIZON_SESSIONS  # noqa: E402
from tools.stage0c_close import _from_offsets  # noqa: E402

PER_TICKER_NPZ = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "evidence_oos.npz")


def _state(ic: float, n_eff: float, yt: np.ndarray, yp: np.ndarray,
           rw: bool | None) -> dict:
    ic_t = (float(ic * np.sqrt(max(n_eff - 1, 1)))
            if np.isfinite(ic) else float("nan"))
    return {
        "forecast_available": True,
        "eval_rank_ic": None if not np.isfinite(ic) else float(ic),
        "eval_rank_ic_t": None if not np.isfinite(ic_t) else ic_t,
        "eval_hit_rate": hit_rate(yt, yp),
        "eval_baseline_hit_rate": majority_hit_rate(yt),
        "eval_rw_significant": rw,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    data = _from_offsets(PER_TICKER_NPZ)

    # The panel-level adjustment the corrected gate consults.
    print(f"grading the panel at B = {args.bootstrap} ...", flush=True)
    panel = grade_panel_v3(data["dates"], data["tickers"], data["folds"],
                           data["y_true"], data["y_pred"],
                           n_bootstrap=args.bootstrap, compute_auto_block=False)
    rw = {r.ticker: r.rw_rejected for r in panel.rows}
    demeaned_ic = {r.ticker: r.hat_ic for r in panel.rows}

    rows = []
    for ticker in sorted(set(data["tickers"])):
        m = data["tickers"] == ticker
        yt, yp, folds = data["y_true"][m], data["y_pred"][m], data["folds"][m]
        ok = np.isfinite(yt) & np.isfinite(yp)
        yt, yp, folds = yt[ok], yp[ok], folds[ok]
        n_eff = effective_sample_size(yt.size, HORIZON_SESSIONS)

        # BEFORE: the pooled-across-folds correlation, no adjustment.
        pooled = float(rank_ic_rows(yt[None, :], yp[None, :])[0])
        before, _ = grade_evidence(_state(pooled, n_eff, yt, yp, rw=True))

        # AFTER: the within-fold, rank-demeaned IC and the panel adjustment.
        after, reasons = grade_evidence(
            _state(demeaned_ic.get(ticker, float("nan")), n_eff, yt, yp,
                   rw=rw.get(ticker, False)))
        rows.append((ticker, before, after, pooled,
                     demeaned_ic.get(ticker, float("nan")),
                     rw.get(ticker, False)))

    def _dist(i: int) -> dict:
        out: dict[str, int] = {}
        for r in rows:
            out[r[i]] = out.get(r[i], 0) + 1
        return out

    moved = [r for r in rows if r[1] != r[2]]

    o = ["\n## The LIVE gate, before and after Stage 0c\n",
         "| grade | pooled IC, unadjusted (live today) | within-fold demeaned "
         "+ Romano-Wolf |", "|---|---|---|"]
    for g in ("STRONG", "WEAK", "INSUFFICIENT"):
        o.append(f"| {g} | {_dist(1).get(g, 0)} | {_dist(2).get(g, 0)} |")
    o.append(f"\n**{len(moved)} of {len(rows)} tickers change grade.**\n")
    if moved:
        o.append("| ticker | before | after | pooled IC | demeaned IC | RW |")
        o.append("|---|---|---|---|---|---|")
        for t, b, a, p, d, r in moved:
            o.append(f"| {t} | {b} | **{a}** | {p:+.4f} | "
                     f"{d:+.4f} | {'yes' if r else 'no'} |")

    text = "\n".join(o)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
