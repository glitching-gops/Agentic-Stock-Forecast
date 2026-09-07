"""
tools/stage2b_panel_diagnostic.py — the Stage 0 panel statistics, recomputed on
the POOLED model's held-out predictions.

Stage 0 measured `mu_hat = -0.05988` on the per-ticker model's track records,
and the Stage 0 addendum showed that number is dominated by fold-level
constants: 316 of 420 cells emitted one repeated value, so the per-ticker
pooled rank IC was ranking rows substantially by WHICH FOLD THEY CAME FROM.

Stage 2b's pooled model emits no constants at all. So the question this answers
is narrow and worth asking: with the artifact gone by construction rather than
by exclusion, what does the panel's measured skill actually look like?

A NOTE ON WHERE THE CODE LIVES, AND WHY THIS IMPORTS LAZILY
-----------------------------------------------------------
`pipeline/evidence_shrinkage.py` is Stage 0's module and it lives on the branch
`stage0-evidence-grading`, which is deliberately NOT merged into `main` and not
into this lineage. This tool REUSES it rather than reimplementing the shrinkage
— reimplementing an empirical-Bayes estimator to avoid a branch switch is how
two versions of a number start disagreeing — so it must be materialised in the
working tree first:

    git show stage0-evidence-grading:pipeline/evidence_shrinkage.py > pipeline/evidence_shrinkage.py

That file is gitignored on this branch, so materialising it cannot accidentally
commit Stage 0's code into Stage 2b's history. Delete it when done.

Usage
-----
    python tools/stage2b_panel_diagnostic.py --arm pooled_rank_ic
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POOLED_NPZ = os.path.join(ROOT, "stage2b_pooled_oos.npz")
PER_TICKER_NPZ = os.path.join(ROOT, "evidence_oos.npz")

MATERIALISE = (
    "git show stage0-evidence-grading:pipeline/evidence_shrinkage.py "
    "> pipeline/evidence_shrinkage.py"
)


def _shrinkage():
    """Imported lazily, with the remediation in the error rather than in a
    comment somebody has to go and find."""
    try:
        import pipeline.evidence_shrinkage as es
    except ImportError as exc:                          # pragma: no cover
        raise SystemExit(
            f"pipeline/evidence_shrinkage.py is not in the working tree.\n"
            f"It is Stage 0's module and lives on `stage0-evidence-grading`, "
            f"which is not merged here on purpose. Materialise it with:\n\n"
            f"    {MATERIALISE}\n\n"
            f"(it is gitignored on this branch, so it cannot be committed by "
            f"accident). Original error: {exc}"
        ) from exc
    return es


def tracks_from_pooled(arm: str, path: str = POOLED_NPZ) -> list:
    """
    One ``TickerTrack`` per ticker, from the pooled model's held-out rows.

    Rows are ordered by date within each ticker, which is what the moving-block
    bootstrap assumes — a block drawn from a date-shuffled series would not be
    a block of anything.
    """
    es = _shrinkage()
    z = np.load(path, allow_pickle=True)
    prefix = f"{arm}__"
    missing = [c for c in ("date", "ticker", "y_true", "y_pred", "fold")
               if prefix + c not in z]
    if missing:
        raise SystemExit(f"{path} has no arm {arm!r} (missing {missing}); "
                         f"available: {sorted({k.split('__')[0] for k in z})}")

    dates = np.asarray([str(d) for d in z[prefix + "date"]])
    tickers = np.asarray([str(t) for t in z[prefix + "ticker"]])
    y_true, y_pred = z[prefix + "y_true"], z[prefix + "y_pred"]
    folds = z[prefix + "fold"]

    tracks = []
    for ticker in sorted(set(tickers)):
        m = tickers == ticker
        order = np.argsort(dates[m], kind="stable")
        tracks.append(es.TickerTrack(
            ticker=ticker,
            dates=tuple(dates[m][order]),
            y_true=y_true[m][order],
            y_pred=y_pred[m][order],
            folds=folds[m][order],
        ))
    return tracks


def tracks_from_per_ticker(path: str = PER_TICKER_NPZ) -> list:
    """The Stage 0 input, rebuilt from the same cache Stage 0 used, so the two
    numbers below are produced by one code path and differ only in the model
    that made the predictions."""
    es = _shrinkage()
    z = np.load(path, allow_pickle=True)
    tickers, offsets = [str(t) for t in z["tickers"]], z["offsets"]
    return [
        es.TickerTrack(
            ticker=t,
            dates=tuple(str(d) for d in z["dates"][int(offsets[i]):int(offsets[i + 1])]),
            y_true=z["y_true"][int(offsets[i]):int(offsets[i + 1])],
            y_pred=z["y_pred"][int(offsets[i]):int(offsets[i + 1])],
            folds=z["fold"][int(offsets[i]):int(offsets[i + 1])],
        )
        for i, t in enumerate(tickers)
    ]


def summarise(tracks: list, label: str) -> dict:
    es = _shrinkage()
    grading = es.grade_panel(tracks)
    counts = grading.counts()
    return {
        "label": label,
        "n_tickers": len(tracks),
        "n_usable": grading.n_usable,
        "mu_hat": grading.mu_hat,
        "mu_sd": float(np.sqrt(grading.mu_var)) if grading.mu_var > 0 else float("nan"),
        "tau2_hat": grading.tau2,
        "degenerate": grading.degenerate,
        "break_even": grading.break_even_ic,
        "counts": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="pooled_rank_ic")
    ap.add_argument("--also", nargs="*", default=["pooled_mae"])
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    rows = [summarise(tracks_from_per_ticker(), "per-ticker x mae (Stage 0)")]
    for arm in [args.arm] + list(args.also):
        rows.append(summarise(tracks_from_pooled(arm), arm))

    out = ["\n## Panel diagnostic — Stage 0's statistics on each model's own "
           "held-out predictions\n",
           "| model | tickers | n_usable | mu_hat | sd | z | tau2_hat | "
           "STRONG | WEAK | ANTI | INSUFF |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        z = r["mu_hat"] / r["mu_sd"] if np.isfinite(r["mu_sd"]) else float("nan")
        c = r["counts"]
        out.append(
            f"| {r['label']} | {r['n_tickers']} | {r['n_usable']} | "
            f"{r['mu_hat']:+.5f} | {r['mu_sd']:.5f} | {z:+.2f} | "
            f"{r['tau2_hat']:.5f} | {c.get('STRONG', 0)} | {c.get('WEAK', 0)} | "
            f"{c.get('ANTI_SIGNAL', 0)} | {c.get('INSUFFICIENT', 0)} |")

    text = "\n".join(out)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
