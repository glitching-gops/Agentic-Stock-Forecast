"""
tools/stage0c_close.py — Stage 0c: run the corrected layer over every variant.

Nothing is retrained. Every set of held-out predictions this reads already
exists; only the grading changes, and this time it changes in four ways at once
— cross-sectional rank-demeaning, a date-level block bootstrap that re-runs the
whole empirical-Bayes pipeline per replicate, Romano-Wolf stepdown across the 84
simultaneous tests, and REML with an explicit tau2 ~ 0 detector.

**EVERY ONE OF THOSE IS EXPECTED TO REDUCE THE GRADED COUNT.** A closing table
with fewer, defensible grades is the intended outcome, and the pre-registration
says so before the first number was seen.

THE PLACEBO IS NOT OPTIONAL HERE, and the reason is specific to this session.
Cross-sectional demeaning turns a prediction that is CONSTANT IN TIME into one
whose cross-sectional RANK moves every day, purely because the other names move
around it. That restores gradeability to tickers Stage 0b refused — which is
either a genuine improvement or a manufactured one, and the only way to tell is
to grade a predictor that is constant by construction and carries no information
at all. `--placebo` does exactly that.

Usage
-----
    python tools/stage0c_close.py --markdown stage0c_report.md
    python tools/stage0c_close.py --bootstrap 50 --variants pooled_mae   # smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.evidence_panel import (  # noqa: E402
    BOOTSTRAP_B,
    FWER_ALPHA,
    MIN_CROSS_SECTION,
    grade_panel_v3,
)
from pipeline.evidence_shrinkage import BLOCK_LENGTH_SESSIONS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PER_TICKER_NPZ = os.path.join(ROOT, "evidence_oos.npz")
STAGE2A_NPZ = os.path.join(ROOT, "stage2a_pilot_oos.npz")
POOLED_NPZ = os.path.join(ROOT, "stage2b_pooled_oos.npz")

BREAK_EVEN = 0.00512363994209475

# What Stage 0 and Stage 0b reported, for the superseded column. Committed
# numbers, never recomputed here.
STAGE0_MU_HAT = -0.05988188592526484
STAGE0B = {
    "per-ticker x mae (production)": (10, 0.15028, 4.62, 10),
    "per-ticker x rank_ic (Stage 2a)": (13, 0.12866, 4.89, 6),
    "pooled_mae": (84, 0.01575, 1.55, 0),
    "pooled_rank_ic": (84, 0.05640, 5.72, 3),
    "pooled_mae_noticker": (84, 0.02102, 2.01, 0),
    "pooled_rank_ic_noticker": (84, 0.01282, 1.30, 0),
}


# ── loading ───────────────────────────────────────────────────────────────────


def _from_offsets(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    tickers = [str(t) for t in z["tickers"]]
    return {
        "dates": np.array([str(d) for d in z["dates"]]),
        "tickers": np.repeat(tickers, np.diff(z["offsets"])),
        "folds": z["fold"], "y_true": z["y_true"], "y_pred": z["y_pred"],
    }


def _from_arm(path: str, arm: str) -> dict:
    z = np.load(path, allow_pickle=True)
    p = f"{arm}__"
    if p + "ticker" not in z:
        raise SystemExit(f"{os.path.basename(path)} has no arm {arm!r}")
    return {
        "dates": np.array([str(d) for d in z[p + "date"]]),
        "tickers": np.array([str(t) for t in z[p + "ticker"]]),
        "folds": z[p + "fold"], "y_true": z[p + "y_true"],
        "y_pred": z[p + "y_pred"],
    }


def variants(names: list[str] | None = None) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = [
        ("per-ticker x mae (production)", _from_offsets(PER_TICKER_NPZ))]
    if os.path.exists(STAGE2A_NPZ):
        out.append(("per-ticker x rank_ic (Stage 2a)", _from_offsets(STAGE2A_NPZ)))
    for arm in ("pooled_mae", "pooled_rank_ic",
                "pooled_mae_noticker", "pooled_rank_ic_noticker"):
        out.append((arm, _from_arm(POOLED_NPZ, arm)))
    if names:
        out = [(n, d) for n, d in out if any(k in n for k in names)]
    return out


def constant_placebo(data: dict, seed: int = 20260908) -> dict:
    """
    Every ticker predicts ONE number, forever. Zero information by construction.

    THE CONTROL THIS SESSION NEEDS. After cross-sectional demeaning a constant
    prediction still has a cross-sectional RANK that moves daily — because the
    other 83 names move around it — so it is no longer refused as degenerate.
    If that manufactured variation grades names, the demeaning is producing the
    grades rather than revealing them, and the whole corrected layer would be
    reporting an artifact.

    The constants are drawn once per ticker from a fixed seed, so the placebo is
    reproducible and carries no relation to any outcome.
    """
    rng = np.random.default_rng(seed)
    out = dict(data)
    uniq = np.unique(data["tickers"])
    level = {t: float(rng.normal()) for t in uniq}
    out["y_pred"] = np.array([level[t] for t in data["tickers"]])
    return out


def shuffled_placebo(data: dict, seed: int = 20260908) -> dict:
    """
    Predictions PERMUTED WITHIN EACH DATE. The decisive null for this layer.

    It preserves the cross-sectional distribution of predictions exactly, so the
    demeaning geometry — the ranks, the [-1, 1] scale, the induced -1/(N-1) —
    is identical to the real run. What it destroys is the only thing that
    matters: which name each prediction belonged to. Every ticker still has a
    cross-sectional rank that moves daily, so nothing is refused for degeneracy
    and the layer has every opportunity to grade.

    If this grades names STRONG, the corrected layer is manufacturing grades out
    of the demeaning rather than revealing them, and none of the real numbers
    can be believed. It is the control the whole session turns on.
    """
    rng = np.random.default_rng(seed)
    out = dict(data)
    dates = np.asarray(data["dates"])
    pred = np.array(data["y_pred"], dtype=float)
    order = np.argsort(dates, kind="stable")
    sd = dates[order]
    bounds = np.flatnonzero(np.r_[True, sd[1:] != sd[:-1], True])
    shuffled = pred.copy()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        idx = order[lo:hi]
        shuffled[idx] = rng.permutation(pred[idx])
    out["y_pred"] = shuffled
    return out


# ── reporting ─────────────────────────────────────────────────────────────────


def _fmt(v: float, spec: str = "+.5f") -> str:
    return "n/a" if v is None or not np.isfinite(v) else format(v, spec)


def render(results: list[tuple[str, object]], common: dict,
           args: argparse.Namespace) -> str:
    o: list[str] = []
    o.append("# Stage 0c — the evidence layer, closed\n")
    o.append(f"Bootstrap B = {args.bootstrap}, block = {BLOCK_LENGTH_SESSIONS} "
             f"sessions, Romano-Wolf alpha = {FWER_ALPHA:.2f}, "
             f"MIN_CROSS_SECTION = {MIN_CROSS_SECTION}. Nothing retrained.\n")

    o.append("\n## The panel, per variant\n")
    o.append("| variant | n_usable | mu_hat | boot SE | naive SE | **z (boot)** "
             "| tau2 REML | tau2 DL | tau2~0 | STRONG | WEAK | INSUFF |")
    o.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, g in results:
        c = g.counts()
        o.append(
            f"| {name} | **{g.n_usable}/{g.tickers_total}** | "
            f"{_fmt(g.mu_hat)} | {_fmt(g.mu_se_bootstrap, '.5f')} | "
            f"{_fmt(g.mu_se_naive, '.5f')} | **{_fmt(g.z_bootstrap, '+.2f')}** | "
            f"{g.tau2_reml:.5f} | {g.tau2_dl:.5f} | "
            f"{'YES' if g.degeneracy.degenerate else 'no'} | "
            f"**{c.get('STRONG', 0)}** | {c.get('WEAK', 0)} | "
            f"{c.get('INSUFFICIENT', 0)} |")

    o.append("\n## Against what Stage 0b reported on the same predictions\n")
    o.append("| variant | n_usable 0b -> 0c | mu_hat 0b -> 0c | z 0b -> 0c | "
             "STRONG 0b -> 0c |")
    o.append("|---|---|---|---|---|")
    for name, g in results:
        if name not in STAGE0B:
            continue
        n0, m0, z0, s0 = STAGE0B[name]
        o.append(f"| {name} | {n0} -> **{g.n_usable}** | "
                 f"{m0:+.5f} -> **{_fmt(g.mu_hat)}** | "
                 f"{z0:+.2f} -> **{_fmt(g.z_bootstrap, '+.2f')}** | "
                 f"{s0} -> **{g.counts().get('STRONG', 0)}** |")

    o.append("\n## The two quantities, never conflated\n")
    o.append("| variant | per-ticker mu_hat (the badge) | per-date cross-sectional "
             "IC (the book) | DK SE | dates |")
    o.append("|---|---|---|---|---|")
    for name, g in results:
        o.append(f"| {name} | {_fmt(g.mu_hat)} | "
                 f"{_fmt(g.cross_sectional_ic)} | "
                 f"{_fmt(g.cross_sectional_ic_se, '.5f')} | "
                 f"{g.cross_sectional_dates} |")
    o.append(f"\nBreak-even cross-sectional rank IC at zero market impact: "
             f"**{BREAK_EVEN:.5f}** (P4).\n")

    o.append("\n## Multiplicity — Romano-Wolf against the FDR bounds\n")
    o.append("| variant | RW rejected | BH significant | BY significant |")
    o.append("|---|---|---|---|")
    for name, g in results:
        o.append(f"| {name} | **{sum(r.rw_rejected for r in g.rows)}** | "
                 f"{sum(r.bh_significant for r in g.rows)} | "
                 f"{sum(r.by_significant for r in g.rows)} |")
    o.append("\nRomano-Wolf controls the FAMILYWISE error rate and assumes "
             "nothing about dependence; BH controls FDR and needs PRDS, which "
             "a common market factor plus the demeaning-induced −1/(N−1) "
             "breaks in both directions at once. BH is the optimistic bound, "
             "BY the conservative one, RW the headline.\n")

    if common:
        o.append("\n## The COMMON gradeable subset — the only fair cross-variant read\n")
        o.append(f"{common['n']} tickers are gradeable in every variant.\n")
        o.append("| variant | mu_hat on the common subset | own-population mu_hat "
                 "| own n_usable |")
        o.append("|---|---|---|---|")
        for name, g in results:
            o.append(f"| {name} | **{_fmt(common['mu'].get(name, float('nan')))}** "
                     f"| {_fmt(g.mu_hat)} | {g.n_usable} |")
        o.append("\nA variant grading few names has NOT demonstrated broad "
                 "skill, and its own-population mu_hat is not comparable to a "
                 "variant grading all of them. Missingness here is the model's "
                 "own behaviour — a degenerate fold has an undefined "
                 "correlation — so it is informative (MNAR), not MCAR, and a "
                 "complete-case comparison across different populations is "
                 "biased. No inverse-probability weighting is attempted: at "
                 "N = 84 the missingness model cannot be credibly estimated and "
                 "would add false precision.\n")

    o.append("\n## Block length, and the bootstrap distribution\n")
    o.append("| variant | block used | Politis-White automatic | boot mu_hat "
             "mean | sd | skew | 5th | 95th |")
    o.append("|---|---|---|---|---|---|---|---|")
    for name, g in results:
        b = g.boot_mu[np.isfinite(g.boot_mu)]
        if b.size:
            from scipy import stats as _st
            o.append(f"| {name} | {g.block_length} | "
                     f"{_fmt(g.block_length_auto, '.1f')} | "
                     f"{b.mean():+.5f} | {b.std(ddof=1):.5f} | "
                     f"{_st.skew(b):+.2f} | {np.percentile(b, 5):+.5f} | "
                     f"{np.percentile(b, 95):+.5f} |")

    o.append("\n## Per-variant notes\n")
    for name, g in results:
        o.append(f"\n**{name}** — {g.headline()}")
        o.append(f"\n- HKSJ interval on the grand mean: "
                 f"{_fmt(g.hksj[0])} ± {_fmt(g.hksj[1], '.5f')} "
                 f"-> [{_fmt(g.hksj[2])}, {_fmt(g.hksj[3])}]")
        o.append(f"- Driscoll-Kraay (linearmodels): "
                 f"{_fmt(g.dk_panel.get('mean', float('nan')))} "
                 f"± {_fmt(g.dk_panel.get('se', float('nan')), '.5f')} "
                 f"({g.dk_panel.get('engine', 'n/a')})")
        o.append(f"- variance source: "
                 f"{sum(1 for r in g.rows if r.sigma2_source == 'bootstrap')} "
                 f"bootstrap, "
                 f"{sum(1 for r in g.rows if r.sigma2_source != 'bootstrap')} "
                 f"fallback")
        o.append(f"- runtime {g.runtime_seconds:.0f}s")
        for note in g.notes:
            o.append(f"- {note}")

    return "\n".join(o)


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--block", type=int, default=BLOCK_LENGTH_SESSIONS)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--placebo", action="store_true", default=True)
    ap.add_argument("--no-placebo", dest="placebo", action="store_false")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    plan = variants(args.variants)
    if args.placebo and plan:
        base = plan[0][1]
        plan.append(("PLACEBO: predictions shuffled within date",
                     shuffled_placebo(base)))
        plan.append(("PLACEBO: per-ticker constants", constant_placebo(base)))

    started = time.time()
    results = []
    for name, data in plan:
        print(f"grading {name} ...", flush=True)
        g = grade_panel_v3(data["dates"], data["tickers"], data["folds"],
                           data["y_true"], data["y_pred"],
                           block=args.block, n_bootstrap=args.bootstrap,
                           break_even=BREAK_EVEN)
        results.append((name, g))
        print(f"  n_usable {g.n_usable}/{g.tickers_total}  "
              f"mu_hat {_fmt(g.mu_hat)}  z {_fmt(g.z_bootstrap, '+.2f')}  "
              f"tau2 {g.tau2_reml:.5f}  {g.counts()}  "
              f"({g.runtime_seconds:.0f}s)", flush=True)

    # The common gradeable subset.
    # Variants with NOTHING gradeable are excluded from the intersection rather
    # than allowed to empty it. The Stage 2a pilot carries 14 names per date,
    # below MIN_CROSS_SECTION, so no date survives the demeaning and n_usable
    # is 0 — a structural exclusion, not a comparison.
    gradeable = [set(r.ticker for r in g.rows if r.n_folds_scored >= 3)
                 for _, g in results if g.n_usable > 0]
    common_set = set.intersection(*gradeable) if gradeable else set()
    common = {}
    if common_set:
        from pipeline.evidence_panel import reml_tau2
        from pipeline.evidence_shrinkage import precision_weighted_mean
        mu = {}
        for name, g in results:
            rows = [r for r in g.rows if r.ticker in common_set
                    and np.isfinite(r.hat_ic) and np.isfinite(r.sigma2)
                    and r.sigma2 > 0]
            if len(rows) >= 2:
                h = np.array([r.hat_ic for r in rows])
                s = np.array([r.sigma2 for r in rows])
                t2 = reml_tau2(h, s)
                mu[name] = precision_weighted_mean(h, s + t2)[0]
        common = {"n": len(common_set), "mu": mu}

    text = render(results, common, args)
    print("\n" + text)
    print(f"\ntotal runtime {time.time() - started:.0f}s for {len(plan)} variants "
          f"at B = {args.bootstrap}")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
