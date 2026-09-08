"""
tools/stage0b_regrade.py — every model variant, re-graded on the CORRECTED IC.

Nothing is retrained. Every set of held-out predictions this reads already
exists; the only thing that changes is how they are graded.

For each variant this reports, side by side:

  * the OLD statistic — one rank correlation over the pooled out-of-sample
    series — reproduced here so the comparison is arithmetic rather than
    remembered. It is deliberately NOT importable from `evidence_shrinkage`
    any more: a calculation known to be wrong must not stay callable in
    production, but the record of what it said has to keep working.
  * the NEW statistic — the mean of the WITHIN-fold rank ICs, which is what
    `pipeline/evidence_shrinkage.py` now computes.
  * the full empirical-Bayes grading built on the new one: `mu_hat`, its
    standard error, `tau2_hat`, and the STRONG / WEAK / ANTI_SIGNAL /
    INSUFFICIENT distribution.

It also measures what the LIVE gate would do. `agents/critic_agent.grade_evidence`
reads `eval_rank_ic` and `eval_rank_ic_t`, which `pipeline/evaluation.py`
computes the pooled way — a defect this session found and deliberately did NOT
fix, because changing it changes `forecast_confidence` on the next weekly run
and that is a production change, not a shadow one. Measuring the difference is
free and is what the decision needs.

Usage
-----
    python tools/stage0b_regrade.py --markdown stage0b_regrade.md
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PER_TICKER_NPZ = os.path.join(ROOT, "evidence_oos.npz")
POOLED_NPZ = os.path.join(ROOT, "stage2b_pooled_oos.npz")
STAGE2A_NPZ = os.path.join(ROOT, "stage2a_pilot_oos.npz")

# What Stage 0 and the addendum reported on the pooled statistic, for the
# before column. Committed numbers; not recomputed.
STAGE0_MU_HAT = -0.05988188592526484
STAGE0_TAU2 = 0.0022093500537861445
STAGE0_N_USABLE = 63


def _old_pooled_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The superseded calculation, reproduced for comparison only."""
    from pipeline.evidence_shrinkage import rank_ic_rows

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return float("nan")
    return float(rank_ic_rows(y_true[valid][None, :], y_pred[valid][None, :])[0])


# ── loading the variants ──────────────────────────────────────────────────────


def _track(ticker, dates, y_true, y_pred, folds):
    from pipeline.evidence_shrinkage import TickerTrack

    order = np.argsort(dates, kind="stable")
    return TickerTrack(ticker=ticker, dates=tuple(dates[order]),
                       y_true=y_true[order], y_pred=y_pred[order],
                       folds=folds[order])


def tracks_from_offsets(path: str) -> list:
    """The Stage 0 cache layout: one flat array per column plus offsets."""
    z = np.load(path, allow_pickle=True)
    tickers, offsets = [str(t) for t in z["tickers"]], z["offsets"]
    return [
        _track(t,
               np.asarray([str(d) for d in z["dates"][int(offsets[i]):int(offsets[i + 1])]]),
               z["y_true"][int(offsets[i]):int(offsets[i + 1])],
               z["y_pred"][int(offsets[i]):int(offsets[i + 1])],
               z["fold"][int(offsets[i]):int(offsets[i + 1])])
        for i, t in enumerate(tickers)
    ]


def tracks_from_arm(path: str, arm: str) -> list:
    """The Stage 2b layout: `<arm>__<column>` keys over the whole panel."""
    z = np.load(path, allow_pickle=True)
    prefix = f"{arm}__"
    if prefix + "ticker" not in z:
        raise SystemExit(
            f"{os.path.basename(path)} has no arm {arm!r}; available: "
            f"{sorted({k.split('__')[0] for k in z})}")

    dates = np.asarray([str(d) for d in z[prefix + "date"]])
    tickers = np.asarray([str(t) for t in z[prefix + "ticker"]])
    y_true, y_pred = z[prefix + "y_true"], z[prefix + "y_pred"]
    folds = z[prefix + "fold"]

    return [_track(t, dates[m], y_true[m], y_pred[m], folds[m])
            for t in sorted(set(tickers)) for m in [tickers == t]]


def variants(include_stage2a: bool = True) -> list[tuple[str, list]]:
    out: list[tuple[str, list]] = [
        ("per-ticker x mae (production)", tracks_from_offsets(PER_TICKER_NPZ)),
    ]
    if include_stage2a and os.path.exists(STAGE2A_NPZ):
        out.append(("per-ticker x rank_ic (Stage 2a, 14)",
                    tracks_from_offsets(STAGE2A_NPZ)))
    for arm in ("pooled_mae", "pooled_rank_ic",
                "pooled_mae_noticker", "pooled_rank_ic_noticker"):
        out.append((arm, tracks_from_arm(POOLED_NPZ, arm)))
    return out


# ── grading ───────────────────────────────────────────────────────────────────


def grade(tracks: list, label: str, n_resamples: int | None = None) -> dict:
    from pipeline.evidence_shrinkage import (
        BOOTSTRAP_N_RESAMPLES,
        grade_panel,
        within_fold_rank_ic,
    )

    old = [_old_pooled_ic(t.y_true, t.y_pred) for t in tracks]
    new = [within_fold_rank_ic(t.y_true, t.y_pred, np.asarray(t.folds))[0]
           for t in tracks]
    old, new = np.array(old, float), np.array(new, float)

    g = grade_panel(tracks, n_resamples=n_resamples or BOOTSTRAP_N_RESAMPLES)
    counts = g.counts()
    return {
        "label": label,
        "n_tickers": len(tracks),
        "n_usable": g.n_usable,
        "old_mean_ic": float(np.nanmean(old)),
        "old_positive": int(np.nansum(old > 0)),
        "new_mean_ic": float(np.nanmean(new)),
        "new_positive": int(np.nansum(new > 0)),
        "mu_hat": g.mu_hat,
        "mu_sd": float(np.sqrt(g.mu_var)) if g.mu_var > 0 else float("nan"),
        "tau2_hat": g.tau2,
        "degenerate": g.degenerate,
        "break_even": g.break_even_ic,
        "counts": counts,
    }


# ── the live gate, measured but not changed ───────────────────────────────────


def live_gate_impact(tracks: list) -> dict:
    """
    What `grade_evidence` would say if `eval_rank_ic` were computed correctly.

    `pipeline/evaluation.compute_metrics` pools across folds, so the live gate
    grades on the same broken statistic Stage 0 did. This changes nothing — it
    runs the REAL grader twice, once on each statistic, and counts the rows
    that would move.
    """
    from agents.critic_agent import grade_evidence
    from pipeline.evaluation import effective_sample_size, hit_rate, \
        majority_hit_rate
    from pipeline.evidence_shrinkage import within_fold_rank_ic
    from pipeline.signals import HORIZON_SESSIONS

    moved, before, after = [], {}, {}
    for t in tracks:
        valid = np.isfinite(t.y_true) & np.isfinite(t.y_pred)
        yt, yp = t.y_true[valid], t.y_pred[valid]
        n_eff = effective_sample_size(len(yt), HORIZON_SESSIONS)

        def _state(ic: float) -> dict:
            ic_t = (float(ic * np.sqrt(max(n_eff - 1, 1)))
                    if np.isfinite(ic) else float("nan"))
            return {
                "forecast_available": True,
                "eval_rank_ic": None if not np.isfinite(ic) else float(ic),
                "eval_rank_ic_t": None if not np.isfinite(ic_t) else ic_t,
                "eval_hit_rate": hit_rate(yt, yp),
                "eval_baseline_hit_rate": majority_hit_rate(yt),
            }

        pooled = _old_pooled_ic(t.y_true, t.y_pred)
        within = within_fold_rank_ic(yt, yp, np.asarray(t.folds)[valid])[0]

        g_before = grade_evidence(_state(pooled))[0]
        g_after = grade_evidence(_state(within))[0]
        before[g_before] = before.get(g_before, 0) + 1
        after[g_after] = after.get(g_after, 0) + 1
        if g_before != g_after:
            moved.append((t.ticker, g_before, g_after))

    return {"before": before, "after": after, "moved": moved}


# ── reporting ─────────────────────────────────────────────────────────────────


def render(rows: list[dict], gate: dict | None) -> str:
    o = ["# Stage 0b — every variant re-graded on the corrected IC\n",
         "Nothing retrained. The predictions are the ones already on file; "
         "only the grading calculation changed.\n",
         "\n## Per-ticker IC, old statistic against new\n",
         "| variant | tickers | pooled-over-folds | positive | "
         "within-fold | positive |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        o.append(
            f"| {r['label']} | {r['n_tickers']} | "
            f"**{r['old_mean_ic']:+.4f}** | {r['old_positive']}/{r['n_tickers']} | "
            f"**{r['new_mean_ic']:+.4f}** | {r['new_positive']}/{r['n_tickers']} |")

    o.append("\n## Empirical-Bayes grading on the corrected statistic\n")
    o.append("| variant | n_usable | mu_hat | sd | z | tau2_hat | degenerate | "
             "STRONG | WEAK | ANTI | INSUFF |")
    o.append("|---|---|---|---|---|---|---|---|---|---|---|")
    o.append(f"| *Stage 0 as reported (pooled)* | {STAGE0_N_USABLE} | "
             f"*{STAGE0_MU_HAT:+.5f}* | 0.01229 | −4.87 | *{STAGE0_TAU2:.5f}* | "
             f"no | 0 | 0 | 0 | 84 |")
    for r in rows:
        z = r["mu_hat"] / r["mu_sd"] if np.isfinite(r["mu_sd"]) else float("nan")
        c = r["counts"]
        o.append(
            f"| {r['label']} | {r['n_usable']} | **{r['mu_hat']:+.5f}** | "
            f"{r['mu_sd']:.5f} | {z:+.2f} | {r['tau2_hat']:.5f} | "
            f"{'YES' if r['degenerate'] else 'no'} | "
            f"{c.get('STRONG', 0)} | {c.get('WEAK', 0)} | "
            f"{c.get('ANTI_SIGNAL', 0)} | {c.get('INSUFFICIENT', 0)} |")
    o.append(f"\nBreak-even rank IC: **{rows[0]['break_even']:.5f}**.\n")

    if gate is not None:
        o.append("\n## The LIVE gate, measured and NOT changed\n")
        o.append("`agents/critic_agent.grade_evidence` reads `eval_rank_ic` "
                 "from `pipeline/evaluation.compute_metrics`, which pools "
                 "across folds. This is what correcting it would do to the "
                 "published grades:\n")
        o.append("| grade | on the pooled IC (live today) | on the within-fold IC |")
        o.append("|---|---|---|")
        for g in sorted(set(gate["before"]) | set(gate["after"])):
            o.append(f"| {g} | {gate['before'].get(g, 0)} | "
                     f"{gate['after'].get(g, 0)} |")
        o.append(f"\n**{len(gate['moved'])} of 84 tickers change grade.**")
        for ticker, b, a in gate["moved"][:20]:
            o.append(f"- {ticker}: {b} -> {a}")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resamples", type=int, default=None)
    ap.add_argument("--no-stage2a", action="store_true")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    rows = []
    for label, tracks in variants(include_stage2a=not args.no_stage2a):
        print(f"grading {label} ({len(tracks)} tickers) ...", flush=True)
        rows.append(grade(tracks, label, n_resamples=args.resamples))
        r = rows[-1]
        print(f"  IC {r['old_mean_ic']:+.4f} -> {r['new_mean_ic']:+.4f}; "
              f"mu_hat {r['mu_hat']:+.5f}; {r['counts']}", flush=True)

    print("measuring the live gate ...", flush=True)
    gate = live_gate_impact(tracks_from_offsets(PER_TICKER_NPZ))

    text = render(rows, gate)
    print("\n" + text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
