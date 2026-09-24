"""
tools/stage2_measure.py — every Stage 2 measurement, on the frozen snapshot.

Run against docs/dashboard-switch-preregistration.md, which was written and
hashed BEFORE any of this touched the reference platform. The tool refuses to
run without it, and records its hash, the snapshot's content hash, the panel's
hash and the environment (platform included) on every output.

    pooled    the pooled x mae, no-ticker, standardised-label walk-forward on a
              snapshot panel -> predictions (.npz) + folds/env (.json)
    repro     are two pooled runs bit-identical? (drift EXACTLY 0.0)
    pertick   the live per-ticker model's walk-forward, every ticker, on the
              same panel (production `evaluate_ticker`, unmodified)
    report    Part 0's delta, pooled vs per-ticker on identical rows, the
              conformal gate, and the Stage 0c grades for both

    python tools/stage2_measure.py pooled --panel stage2/panel_new.parquet --out stage2/ref_a
    python tools/stage2_measure.py repro stage2/ref_a.npz stage2/ref_b.npz
    python tools/stage2_measure.py report --pooled stage2/ref_a.npz \
        --old stage2/old_a.npz --pertick stage2/pertick.npz \
        --panel stage2/panel_new.parquet --json stage2/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = Path(__file__).resolve().parents[1]
#: The pre-registration this run is measured against. Default: the Stage 2
#: dashboard switch; STAGE2_PREREG names a later one (e.g. the 2026-09-24
#: fallback + conformal step). The tool refuses without the file either way.
PREREG = ROOT / os.environ.get("STAGE2_PREREG", "docs/dashboard-switch-preregistration.md")
SNAPSHOT_MANIFEST = ROOT / "stage2" / "snapshot" / "manifest.json"
COLS = ("date", "ticker", "y_true", "y_pred", "fold")
DK_LAGS = 30


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(panel_path: Path | None = None) -> dict:
    from pipeline.determinism import environment_fingerprint

    if not PREREG.exists():
        raise SystemExit(f"no pre-registration at {PREREG}; refusing to measure")
    out = {"prereg": PREREG.name, "prereg_sha256": sha256_file(PREREG),
           "environment": environment_fingerprint(),
           "python": sys.version.split()[0], "platform": sys.platform}
    if SNAPSHOT_MANIFEST.exists():
        out["snapshot"] = json.loads(SNAPSHOT_MANIFEST.read_text())["content_hash"]
    if panel_path is not None:
        out["panel_sha256"] = sha256_file(panel_path)
    return out


def save_preds(preds: pd.DataFrame, out: Path, meta: dict) -> None:
    arrays = {c: preds[c].to_numpy() for c in preds.columns}
    arrays = {c: (a.astype(str) if a.dtype == object else a) for c, a in arrays.items()}
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float),
                                        encoding="utf-8")


def load_preds(path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=False)
    df = pd.DataFrame({k: z[k] for k in z.files})
    df["date"] = df["date"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    return df


# ── pooled ────────────────────────────────────────────────────────────────────


def cmd_pooled(args) -> None:
    from pipeline import pooled

    meta = provenance(args.panel)
    t0 = time.time()
    raw = pd.read_parquet(args.panel)
    prepared = pooled.prepare(raw)
    preds, folds = pooled.walk_forward(prepared, n_trials=args.trials, verbose=True,
                                       min_train=args.min_train)
    meta.update({"pooled_version": pooled.POOLED_MODEL_VERSION, "folds": folds,
                 "min_train": args.min_train,
                 "rows": int(len(preds)), "runtime_s": round(time.time() - t0, 1),
                 "trials": args.trials})
    preds_hash = hashlib.sha256()
    for c in COLS:
        preds_hash.update(np.ascontiguousarray(
            preds[c].to_numpy().astype(str if c in ("date", "ticker") else float)).tobytes())
    meta["predictions_sha256"] = preds_hash.hexdigest()
    save_preds(preds, args.out, meta)
    print(f"pooled on {args.panel.name}: {len(preds):,} rows, gammas "
          f"{[round(f['gamma'], 3) for f in folds]}, {meta['runtime_s']}s, "
          f"predictions sha256 {meta['predictions_sha256'][:16]}", flush=True)


def cmd_repro(args) -> None:
    a, b = load_preds(args.a), load_preds(args.b)
    m = a.merge(b, on=["date", "ticker", "fold"], how="outer", indicator=True,
                suffixes=("_a", "_b"))
    both = m[m["_merge"] == "both"]
    drift = float(np.max(np.abs(both["y_pred_a"] - both["y_pred_b"]))) if len(both) else np.nan
    truth = float(np.max(np.abs(both["y_true_a"] - both["y_true_b"]))) if len(both) else np.nan
    exact = (m["_merge"] != "both").sum() == 0 and drift == 0.0 and truth == 0.0
    print(f"rows {len(a):,} / {len(b):,}, unmatched {(m['_merge'] != 'both').sum()}, "
          f"max prediction drift {drift:.3e}, max label drift {truth:.3e}")
    print(f"EXACT REPRODUCTION: {'YES' if exact else 'NO'}")
    sys.exit(0 if exact else 1)


# ── per-ticker ────────────────────────────────────────────────────────────────


def _one_ticker(args):
    ticker, df = args
    from agents.critic_agent import grade_evidence
    from pipeline.model import evaluate_ticker

    t0 = time.time()
    res = evaluate_ticker(ticker, df.reset_index(drop=True))
    if res.n_predictions == 0:
        return ticker, None, {"note": "no out-of-sample predictions"}
    p = res.predictions.copy()
    p["ticker"] = ticker
    m = res.metrics
    grade, _ = grade_evidence({
        "forecast_available": True, "eval_rank_ic": m.get("rank_ic"),
        "eval_rank_ic_t": m.get("rank_ic_t"), "eval_hit_rate": m.get("hit_rate"),
        "eval_baseline_hit_rate": m.get("majority_hit_rate")})
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
             for k, v in m.items() if isinstance(v, (int, float, type(None)))}
    return ticker, p, {"metrics": clean, "live_gate_grade": grade,
                       "secs": round(time.time() - t0, 1)}


def per_ticker_frame(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """What `load_features_for_ticker` would return, built from the panel."""
    from pipeline.model import FEATURES
    from pipeline.signals import NULLABLE_FEATURES

    df = panel[panel["ticker"] == ticker].sort_values("date").copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col not in NULLABLE_FEATURES:
            df[col] = df[col].fillna(0.0)
    return df


def cmd_pertick(args) -> None:
    import multiprocessing as mp

    meta = provenance(args.panel)
    t0 = time.time()
    panel = pd.read_parquet(args.panel)
    tickers = sorted(panel["ticker"].unique())
    jobs = [(t, per_ticker_frame(panel, t)) for t in tickers]
    out, info = [], {}
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(args.workers) as pool:
        for t, p, i in pool.imap_unordered(_one_ticker, jobs):
            info[t] = i
            if p is not None:
                out.append(p)
            print(f"  {t}: {i.get('live_gate_grade', i.get('note'))} "
                  f"({i.get('secs', 0)}s)", flush=True)
    preds = pd.concat(out, ignore_index=True)[list(COLS)]
    preds["date"] = preds["date"].astype(str)
    meta.update({"tickers": info, "rows": int(len(preds)),
                 "runtime_s": round(time.time() - t0, 1)})
    save_preds(preds, args.out, meta)
    print(f"per-ticker: {len(preds):,} rows over {len(info)} tickers, "
          f"{meta['runtime_s']}s", flush=True)


# ── report ────────────────────────────────────────────────────────────────────


def per_date_ic(df: pd.DataFrame, pred: str = "y_pred", truth: str = "y_raw",
                min_names: int = 20) -> pd.Series:
    """Per-date cross-sectional Spearman rank IC, never pooled."""
    out = {}
    for d, g in df.groupby("date", sort=True):
        g = g[np.isfinite(g[pred]) & np.isfinite(g[truth])]
        if len(g) < min_names or g[pred].nunique() < 2 or g[truth].nunique() < 2:
            continue
        out[d] = g[pred].rank().corr(g[truth].rank())
    return pd.Series(out, dtype=float)


def dk(series: pd.Series) -> dict:
    from pipeline.evidence_panel import driscoll_kraay_se

    s = series.dropna()
    mean, se, lags = driscoll_kraay_se(s.index.to_numpy(), s.to_numpy(), max_lag=DK_LAGS)
    return {"mean": float(mean), "se": float(se), "n_dates": int(len(s)),
            "t": float(mean / se) if se and np.isfinite(se) and se > 0 else float("nan")}


def degeneracy(df: pd.DataFrame) -> dict:
    cells = df.groupby(["ticker", "fold"])["y_pred"].nunique()
    per_date = df.groupby("date")["y_pred"].nunique()
    return {"cells": int(len(cells)), "constant_cells": int((cells <= 1).sum()),
            "median_distinct_per_date": float(per_date.median()),
            "median_names_per_date": float(df.groupby("date")["ticker"].nunique().median())}


def grades(df: pd.DataFrame, bootstrap: int) -> dict:
    from pipeline.evidence_panel import grade_panel_v3

    g = grade_panel_v3(df["date"].to_numpy(), df["ticker"].to_numpy(),
                       df["fold"].to_numpy(), df["y_raw"].to_numpy(dtype=float),
                       df["y_pred"].to_numpy(dtype=float), n_bootstrap=bootstrap)
    c = g.counts()
    return {"counts": c, "strong": c.get("STRONG", 0), "weak": c.get("WEAK", 0),
            "insufficient": c.get("INSUFFICIENT", 0),
            "rw_rejected": int(sum(r.rw_rejected for r in g.rows)),
            "strong_set": sorted(r.ticker for r in g.rows if r.grade == "STRONG"),
            "weak_set": sorted(r.ticker for r in g.rows if r.grade == "WEAK"),
            "n_usable": g.n_usable, "mu_hat": g.mu_hat, "z": g.z_bootstrap,
            "tau2": g.tau2_reml, "degenerate": bool(g.degeneracy.degenerate),
            "degeneracy_reason": g.degeneracy.reason, "headline": g.headline(),
            "runtime_s": g.runtime_seconds}


def cmd_report(args) -> None:
    from pipeline import pooled
    from pipeline.pooled_shadow import coverage_gate

    meta = provenance(args.panel)
    raw = pd.read_parquet(args.panel)
    causal = pooled.causal_frame(raw)
    truth = raw[["date", "ticker", "target_return"]].rename(columns={"target_return": "y_raw"})
    report: dict = {"provenance": meta}

    pooled_p = pooled.invert(load_preds(args.pooled), causal)
    report["pooled_file"] = {"path": str(args.pooled), "sha256": sha256_file(args.pooled)}

    # ── Part 0: the delta, attributed to the benchmark alone ─────────────────
    if args.old:
        old_p = load_preds(args.old)
        new_ic, old_ic = per_date_ic(pooled_p), per_date_ic(old_p)
        j = pd.concat({"new": new_ic, "old": old_ic}, axis=1).dropna()
        report["part0"] = {"new": dk(j["new"]), "old": dk(j["old"]),
                           "delta_new_minus_old": dk(j["new"] - j["old"]),
                           "old_file_sha256": sha256_file(args.old),
                           "rows_identical": bool(len(old_p) == len(pooled_p))}

    # ── pooled vs per-ticker on IDENTICAL rows ────────────────────────────────
    models = {"pooled": pooled_p}
    if args.pertick:
        pt = load_preds(args.pertick).merge(truth, on=["date", "ticker"], how="left")
        pt["pred_return"] = pt["y_pred"]            # per-ticker predicts the raw return
        models["per_ticker"] = pt
    keys = None
    for df in models.values():
        k = set(zip(df["date"], df["ticker"]))
        keys = k if keys is None else keys & k
    table = {}
    for name, df in models.items():
        common = df[[(d, t) in keys for d, t in zip(df["date"], df["ticker"])]]
        ok = np.isfinite(common["pred_return"]) & np.isfinite(common["y_raw"])
        table[name] = {
            "rows_own": int(len(df)), "rows_common": int(len(common)),
            "degeneracy_own": degeneracy(df),
            "cs_ic_common": dk(per_date_ic(common)),
            "cs_ic_own": dk(per_date_ic(df)),
            "mae_log_return_common": float(np.mean(np.abs(
                common.loc[ok, "pred_return"] - common.loc[ok, "y_raw"]))),
            "conformal_own": coverage_gate(df.loc[np.isfinite(df["pred_return"])
                                                  & np.isfinite(df["y_raw"]), "y_raw"],
                                           df.loc[np.isfinite(df["pred_return"])
                                                  & np.isfinite(df["y_raw"]), "pred_return"],
                                           df.loc[np.isfinite(df["pred_return"])
                                                  & np.isfinite(df["y_raw"]), "fold"]),
        }
        if args.bootstrap:
            table[name]["grades_own"] = grades(df, args.bootstrap)
        print(f"{name}: done", flush=True)
    if "per_ticker" in models and args.pertick:
        info = json.loads(Path(args.pertick).with_suffix(".json").read_text())
        live = {}
        for t, i in info["tickers"].items():
            g = i.get("live_gate_grade", "NO_EVALUATION")
            live[g] = live.get(g, 0) + 1
        table["per_ticker"]["live_gate_grades"] = live
    report["comparison"] = table
    args.json.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps(report, indent=1, default=float)[:6000])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pooled")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--min-train", type=int, default=None,
                   help="dates before the first fold (default pooled.MIN_TRAIN_DATES)")
    r = sub.add_parser("repro")
    r.add_argument("a")
    r.add_argument("b")
    t = sub.add_parser("pertick")
    t.add_argument("--panel", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--workers", type=int, default=6)
    q = sub.add_parser("report")
    q.add_argument("--pooled", type=Path, required=True)
    q.add_argument("--old", type=Path, default=None)
    q.add_argument("--pertick", type=Path, default=None)
    q.add_argument("--panel", type=Path, required=True)
    q.add_argument("--bootstrap", type=int, default=1000)
    q.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "pooled":
        from pipeline.pooled import N_TRIALS
        args.trials = N_TRIALS if args.trials is None else args.trials
    {"pooled": cmd_pooled, "repro": cmd_repro, "pertick": cmd_pertick,
     "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
