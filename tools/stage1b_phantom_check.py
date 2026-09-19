"""
tools/stage1b_phantom_check.py — does the Stage 1b verdict survive removing
the four phantom 2026 holiday sessions?

The panel carries 2026-01-15, 05-01, 05-28 and 06-26 as trading days, and NSE
has no delivery file for any of them. Every close on those dates repeats the
previous session's. This tool:

1. Rebuilds `run_arm`'s splits and reports whether any phantom reaches a
   fold's training rows or their label windows. If none does, the fitted
   models cannot depend on the defect.
2. Re-scores R3 on the STORED predictions three ways:
   - V0, as run;
   - V1, labels recomputed over 30 real sessions with the phantom rows dropped;
   - V2, truncated before any scored row's label or lookback can touch one.

No retraining, no database, and nothing written except the optional markdown.

    python tools/stage1b_phantom_check.py [--markdown out.md]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.evaluation import PurgedPanelWalkForward  # noqa: E402
from pipeline.model import EVAL_N_FOLDS  # noqa: E402
from pipeline.signals import HORIZON_SESSIONS  # noqa: E402
from tools.stage0c_close import _from_arm  # noqa: E402
from tools.stage1_reversal import SIGNAL_T, _f, paired, per_date_ic  # noqa: E402
from tools.stage2b_pooled import EVAL_MIN_TRAIN_DATES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PANEL_CACHE = os.path.join(ROOT, "panel_cache.parquet")
PRED_NPZ = os.path.join(ROOT, "stage1b_delivery_oos.npz")

PHANTOMS: tuple[str, ...] = ("2026-01-15", "2026-05-01", "2026-05-28", "2026-06-26")
ARMS = ("baseline_repro", "abnormal", "level")
LABEL_ATOL = 1e-5


def confirm_phantoms(panel: pd.DataFrame, phantoms=PHANTOMS) -> dict[str, float]:
    """Share of tickers whose close on each phantom equals the previous row's.
    A phantom is only accepted as one if that share is at least 0.95."""
    p = panel.sort_values(["ticker", "date"])
    same = p.groupby("ticker")["close"].diff().eq(0.0)
    out = {d: float(same[p["date"] == d].mean()) for d in phantoms}
    bad = {d: s for d, s in out.items() if not s >= 0.95}
    if bad:
        raise SystemExit(f"not phantom on this panel: {bad}")
    return out


def training_exposure(dates_all: np.ndarray, phantoms=PHANTOMS) -> list[dict]:
    """Per fold: the training dates' span, and whether a phantom lies inside
    the training rows or their HORIZON-session label windows."""
    grid = np.array(sorted(set(dates_all)))
    pos = {d: i for i, d in enumerate(grid)}
    splitter = PurgedPanelWalkForward(
        n_folds=EVAL_N_FOLDS, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=EVAL_MIN_TRAIN_DATES)
    out = []
    for fold, (tr, te) in enumerate(splitter.split(dates_all)):
        tr_last = max(dates_all[tr])
        reach = grid[min(pos[tr_last] + HORIZON_SESSIONS, len(grid) - 1)]
        out.append({"fold": fold, "train_first": min(dates_all[tr]),
                    "train_last": tr_last, "label_reach": reach,
                    "test_first": min(dates_all[te]), "test_last": max(dates_all[te]),
                    "phantom_in_training": any(p <= reach for p in phantoms)})
    return out


def corrected_labels(panel: pd.DataFrame, phantoms=PHANTOMS,
                     horizon: int = HORIZON_SESSIONS) -> pd.DataFrame:
    """log(close) `horizon` REAL sessions ahead, per ticker, phantom rows
    dropped. Also returns the naive row-stepped label, which is what was
    stored, so the recomputation can be checked against it."""
    p = panel[["date", "ticker", "close"]].sort_values(["ticker", "date"])
    lc = np.log(p["close"].astype(float))
    naive = lc.groupby(p["ticker"]).shift(-horizon) - lc
    real = p[~p["date"].isin(phantoms)]
    lr = np.log(real["close"].astype(float))
    fixed = lr.groupby(real["ticker"]).shift(-horizon) - lr
    out = p[["date", "ticker"]].assign(naive=naive.to_numpy())
    return out.merge(real[["date", "ticker"]].assign(fixed=fixed.to_numpy()),
                     on=["date", "ticker"], how="left")


def touched(dates: pd.Series, grid: list[str], phantoms=PHANTOMS,
            horizon: int = HORIZON_SESSIONS) -> pd.Series:
    """True where a row's `horizon`-row forward label window includes a
    phantom, or the row IS one."""
    pos = {d: i for i, d in enumerate(grid)}
    ph = np.array(sorted(pos[p] for p in phantoms))
    i = dates.map(pos).to_numpy()
    hit = np.zeros(len(i), dtype=bool)
    for k in ph:
        hit |= (i <= k) & (k <= i + horizon)
    return pd.Series(hit, index=dates.index)


def _data(frame: pd.DataFrame) -> dict:
    return {"dates": frame["date"].to_numpy(), "tickers": frame["ticker"].to_numpy(),
            "folds": frame["fold"].to_numpy(), "y_true": frame["y_true"].to_numpy(),
            "y_pred": frame["y_pred"].to_numpy()}


def r3(frames: dict[str, pd.DataFrame]) -> dict[str, dict]:
    ic = {a: per_date_ic(_data(f)) for a, f in frames.items()}
    return {"abnormal": paired(ic["baseline_repro"], ic["abnormal"]),
            "level": paired(ic["baseline_repro"], ic["level"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--npz", default=PRED_NPZ)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel_cache, columns=["date", "ticker", "close"])
    panel["date"] = panel["date"].astype(str)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    grid = sorted(panel["date"].unique())
    o = ["## Phantom-session sensitivity\n"]

    shares = confirm_phantoms(panel)
    o.append("Share of tickers whose close repeats the previous session's: "
             + ", ".join(f"{d} {s:.1%}" for d, s in shares.items()) + ".\n")

    exp = training_exposure(panel["date"].to_numpy())
    o.append("| fold | train first | train last | train labels reach | test first "
             "| test last | phantom in training |")
    o.append("|---|---|---|---|---|---|---|")
    for r in exp:
        o.append(f"| {r['fold']} | {r['train_first']} | {r['train_last']} | "
                 f"{r['label_reach']} | {r['test_first']} | {r['test_last']} | "
                 f"{'**YES**' if r['phantom_in_training'] else 'no'} |")
    in_test = {p: next((r["fold"] for r in exp
                        if r["test_first"] <= p <= r["test_last"]), None)
               for p in PHANTOMS}
    o.append(f"\nPhantoms inside a scored test window: {in_test}.\n")

    frames = {}
    for a in ARMS:
        d = _from_arm(args.npz, a)
        f = pd.DataFrame({"date": np.asarray(d["dates"]).astype(str),
                          "ticker": d["tickers"], "fold": d["folds"],
                          "y_true": d["y_true"], "y_pred": d["y_pred"]})
        frames[a] = f

    labels = corrected_labels(panel)
    base = frames["baseline_repro"].merge(labels, on=["date", "ticker"], how="left")
    hit = touched(base["date"], grid)
    clean = base[~hit]
    # The panel stores close rounded to 3 decimals, so a label recomputed
    # from it can only agree with the stored one to ~1e-6, not to float
    # precision. 1e-5 is still four orders below the label's dispersion (~0.1).
    match = np.isclose(clean["naive"], clean["y_true"], atol=LABEL_ATOL, rtol=0)
    match_rate = float(match.mean())
    max_dev = float((clean["naive"] - clean["y_true"]).abs().max())
    if match_rate < 0.99:
        raise SystemExit(f"label recomputation reproduces only {match_rate:.2%} "
                         "of the untouched stored labels; stopping")
    changed = base[hit & ~base["date"].isin(PHANTOMS)]
    o.append(f"Label recomputation reproduces the stored label on "
             f"{match_rate:.4%} of {len(clean):,} untouched rows (max abs deviation "
             f"{max_dev:.1e}, tolerance {LABEL_ATOL:.0e}). "
             f"{int(hit.sum()):,} scored rows ({base.loc[hit, 'date'].nunique()} dates) "
             f"have a label window touching a phantom or sit on one; "
             f"{int(base['date'].isin(PHANTOMS).sum()):,} sit on one. Median "
             f"abs label change on the rest: "
             f"{float((changed['fixed'] - changed['y_true']).abs().median()):.5f}.\n")

    first = grid.index(PHANTOMS[0])
    cutoff = grid[first - HORIZON_SESSIONS]
    variants: dict[str, dict[str, pd.DataFrame]] = {"V0 as run": frames}
    v1 = {}
    for a, f in frames.items():
        g = f.merge(labels[["date", "ticker", "fixed"]], on=["date", "ticker"], how="left")
        g = g[~g["date"].isin(PHANTOMS)]
        g = g.assign(y_true=g["fixed"]).dropna(subset=["y_true"])
        v1[a] = g.drop(columns="fixed")
    variants["V1 labels corrected"] = v1
    variants[f"V2 truncated before {cutoff}"] = {a: f[f["date"] < cutoff]
                                                 for a, f in frames.items()}

    o.append("| variant | scored dates | (b) abnormal − (a): Δ | DK SE | **t** "
             "| (c) level − (a): Δ | t |")
    o.append("|---|---|---|---|---|---|---|")
    changes = False
    for name, fr in variants.items():
        res = r3(fr)
        b, c = res["abnormal"], res["level"]
        changes |= bool(name != "V0 as run" and np.isfinite(b["t"]) and b["t"] >= SIGNAL_T)
        o.append(f"| {name} | {b['n_dates']:,} | {_f(b['diff'])} | {_f(b['se'], '.5f')} "
                 f"| **{_f(b['t'], '+.2f')}** | {_f(c['diff'])} | {_f(c['t'], '+.2f')} |")
    o.append(f"\n**Verdict {'CHANGES' if changes else 'STANDS'}**: the rule fixed before "
             f"the run is that it changes iff R3's t reaches +{SIGNAL_T:.1f} in V1 or V2.")
    text = "\n".join(o)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
