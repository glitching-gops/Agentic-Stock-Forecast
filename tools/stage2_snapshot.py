"""
tools/stage2_snapshot.py — ONE frozen snapshot, and every Stage 2 panel built from it.

WHY A SNAPSHOT (docs/dashboard-switch-preregistration.md). Yahoo re-adjusts
past prices between rebuilds: recomputing the 2026-09-07 panel through the
production code moved the baseline's cs IC by t +1.20 — more than the phantom
fix it was measuring. So everything in Stage 2 (Part 0's delta, the Linux
baseline, the per-ticker re-measure, the conformal gate) is built from the
inputs read ONCE here, and a panel is identified by the snapshot hash plus the
code that built it.

    build     read every input once, read-only, and write it with a manifest
    verify    re-hash the files against the manifest
    panel     build a panel from the snapshot through the PRODUCTION signals
              code: --arm new (panel-internal sector benchmark; since v5 the
              thin-sector names are market-relative, not NULL) or
              --arm old (the Yahoo indices, as stored, forward-filled as the
              pre-v4 code did)
    guard     simulate the signals write guard for every ticker against the
              stored label counts: does each of the 84 write under --arm new?

Inputs captured by `build`, each a parquet file with its sha256:
    ohlcv          every row for the frozen universe, <= the cutoff
    membership     index_membership, all rows (the industry labels)
    macro          the macro table
    signals        the stored signals table for the universe: the old arm's
                   index levels, and the label counts the guard compares to
    earnings       yf.Ticker(t).earnings_dates for each ticker, fetched ONCE —
                   the only network input the signals code has

The manifest carries VENDOR DEFECT TAGS (CLAUDE.md rule 8): a known defect in
the inputs is recorded here, and every table built from the snapshot repeats
it, instead of the work waiting for a clean vendor window that may never come.
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
DEFAULT_DIR = ROOT / "stage2" / "snapshot"
FILES = ("ohlcv", "membership", "macro", "signals", "earnings")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(manifest: dict) -> str:
    """One hash over the per-file hashes, in a fixed order: the snapshot's id."""
    joined = "".join(f"{k}:{manifest['files'][k]['sha256']};" for k in FILES)
    return hashlib.sha256(joined.encode()).hexdigest()


# ── build ─────────────────────────────────────────────────────────────────────


def fetch_earnings(tickers: list[str]) -> tuple[pd.DataFrame, dict]:
    """The vendor earnings tables, long, and per-ticker failures."""
    import yfinance as yf

    from pipeline.signals import require_earnings_parser

    require_earnings_parser()
    frames, failures = [], {}
    for t in tickers:
        for attempt in range(3):
            try:
                e = yf.Ticker(t).earnings_dates
                if e is None or len(e) == 0:
                    raise ValueError("empty")
                e = e.reset_index()
                e.columns = [str(c) for c in e.columns]
                e.insert(0, "ticker", t)
                # Keep the vendor's column NAMES: the signals code finds them by
                # substring, and the snapshot must feed it what the vendor gave.
                e = e.astype({c: "float64" for c in e.columns
                              if c not in ("ticker",) and e[c].dtype != object
                              and not str(e[c].dtype).startswith("datetime")})
                date_col = next(c for c in e.columns if "date" in c.lower())
                e[date_col] = pd.to_datetime(e[date_col], utc=True).astype(str)
                frames.append(e)
                failures.pop(t, None)
                break
            except Exception as exc:                          # noqa: BLE001
                failures[t] = f"{type(exc).__name__}: {exc}"[:200]
                time.sleep(2 * (attempt + 1))
        time.sleep(0.3)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), failures


def build(out: Path, cutoff: str | None) -> dict:
    from sqlalchemy import text

    from data.db import get_engine
    from data.frozen_universe import FROZEN_UNIVERSE
    from pipeline.determinism import environment_fingerprint

    out.mkdir(parents=True, exist_ok=True)
    tickers = sorted(FROZEN_UNIVERSE)
    keys = {f"t{i}": t for i, t in enumerate(tickers)}
    in_list = ", ".join(":" + k for k in keys)

    with get_engine().connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        cutoff = cutoff or conn.execute(text(
            f"SELECT MAX(date) FROM ohlcv WHERE ticker IN ({in_list})"), keys).scalar()
        params = {**keys, "cut": cutoff}
        frames = {
            "ohlcv": pd.read_sql(text(
                f"SELECT * FROM ohlcv WHERE ticker IN ({in_list}) AND date <= :cut "
                f"ORDER BY ticker, date"), conn, params=params),
            "membership": pd.read_sql(text(
                "SELECT * FROM index_membership ORDER BY ticker, effective_from"), conn),
            "macro": pd.read_sql(text("SELECT * FROM macro ORDER BY date"), conn),
            "signals": pd.read_sql(text(
                f"SELECT * FROM signals WHERE ticker IN ({in_list}) AND date <= :cut "
                f"ORDER BY ticker, date"), conn, params=params),
        }
        conn.rollback()

    earnings, failures = fetch_earnings(tickers)
    frames["earnings"] = earnings

    manifest: dict = {"created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                      "cutoff": cutoff, "tickers": tickers,
                      "environment": environment_fingerprint(), "files": {}}
    for name in FILES:
        path = out / f"{name}.parquet"
        frames[name].to_parquet(path, index=False)
        manifest["files"][name] = {"rows": int(len(frames[name])),
                                   "sha256": sha256_file(path)}
    manifest["defects"] = vendor_defects(frames, failures)
    manifest["content_hash"] = content_hash(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str),
                                       encoding="utf-8")
    return manifest


def vendor_defects(frames: dict, earnings_failures: dict) -> list[dict]:
    """Known defects in the captured inputs, recorded rather than waited for."""
    from data import nse_calendar

    defects: list[dict] = []
    o = frames["ohlcv"]
    flat = o[(o["volume"] == 0) & (o["open"] == o["high"]) & (o["high"] == o["low"])
             & (o["low"] == o["close"])]
    by_date = flat.groupby("date")["ticker"].nunique()
    n = o.groupby("date")["ticker"].nunique()
    cal = nse_calendar.load()
    for d, k in by_date.items():
        if k >= 0.9 * n.get(d, 0) and n.get(d, 0) >= 10:
            defects.append({"kind": "market_wide_flat_bar", "date": d, "tickers": int(k),
                            "nse_session": bool(cal.is_session(d)),
                            "effect": "every return into/out of this date is fiction"})
    for t, why in sorted(earnings_failures.items()):
        defects.append({"kind": "earnings_fetch_failed", "ticker": t, "detail": why,
                        "effect": "ticker is SKIPPED by the signals code (never 0.0)"})
    s = frames["signals"]
    last = s.groupby("ticker")["date"].max()
    stale = last[last < last.max()]
    if len(stale):
        defects.append({"kind": "stored_signals_stale", "tickers": sorted(stale.index),
                        "through": {t: d for t, d in stale.items()},
                        "effect": "old-arm index levels end here and are forward-filled "
                                  "beyond it, as the pre-v4 code did; the new arm reads "
                                  "ohlcv and is unaffected"})
    return defects


def verify(out: Path) -> bool:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    ok = True
    for name in FILES:
        got = sha256_file(out / f"{name}.parquet")
        if got != manifest["files"][name]["sha256"]:
            print(f"MISMATCH {name}: {got} != {manifest['files'][name]['sha256']}")
            ok = False
    if content_hash(manifest) != manifest["content_hash"]:
        print("MISMATCH content hash")
        ok = False
    print(f"snapshot {manifest['content_hash']} cutoff {manifest['cutoff']}: "
          f"{'VERIFIED' if ok else 'FAILED'}")
    return ok


def load_snapshot(out: Path = DEFAULT_DIR) -> tuple[dict, dict]:
    if not verify(out):
        raise SystemExit(f"snapshot at {out} does not match its manifest; refusing")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    frames = {n: pd.read_parquet(out / f"{n}.parquet") for n in FILES}
    for n in ("ohlcv", "macro", "signals"):
        frames[n]["date"] = frames[n]["date"].astype(str)
    return manifest, frames


# ── panel ─────────────────────────────────────────────────────────────────────


def _db_real(values) -> np.ndarray:
    """A float64 after a Postgres REAL column and psycopg2 — see
    tools/phantom_rebuild.db_real, whose rule this is."""
    from tools.phantom_rebuild import db_real

    return db_real(values)


def old_benchmarks(frames: dict, tickers: list[str]) -> dict:
    """The pre-v4 benchmark for each ticker: its stored Yahoo index level,
    forward-filled onto the universe grid exactly as the old code did."""
    from pipeline.sector_benchmark import index_benchmark

    s = frames["signals"]
    grid = pd.DataFrame({"date": sorted(frames["ohlcv"]["date"].unique())})
    series = {}
    for b, g in s.dropna(subset=["benchmark_close"]).groupby("benchmark_ticker"):
        lv = g.groupby("date")["benchmark_close"].median().rename("benchmark_close")
        series[b] = grid.merge(lv.reset_index(), on="date", how="left").ffill()
    bench_of = (s.dropna(subset=["benchmark_ticker"]).sort_values("date")
                .groupby("ticker")["benchmark_ticker"].last().to_dict())
    return {t: index_benchmark(t, bench_of[t], series[bench_of[t]],
                               sector_specific=bench_of[t] != "^NSEI")
            for t in tickers if t in bench_of}


def new_benchmarks(frames: dict, tickers: list[str]) -> dict:
    from pipeline.sector_benchmark import build_benchmarks, clean_label

    m = frames["membership"].sort_values(["ticker", "effective_from"])
    labels = {t: clean_label(v) for t, v in m.groupby("ticker")["industry"].last().items()}
    labels = {t: labels.get(t) for t in tickers}
    prices = frames["ohlcv"][["date", "ticker", "close", "adj_close"]]
    return build_benchmarks(prices, labels, tickers=tickers)


def build_panel(frames: dict, arm: str) -> tuple[pd.DataFrame, dict]:
    """The panel `load_panel` would return had the production signals code run
    on the snapshot, under the chosen benchmark arm."""
    import pipeline.signals as signals
    from data import nse_calendar
    from pipeline.panel import (EXCESS_TARGET, FEATURES, MACRO_COLS, TARGET, TARGETS)
    from pipeline.signals import FEATURE_COLS, NULLABLE_FEATURES

    tickers = sorted(frames["ohlcv"]["ticker"].unique())
    benchmarks = (new_benchmarks if arm == "new" else old_benchmarks)(frames, tickers)
    earn = {t: g.drop(columns=["ticker"]) for t, g in frames["earnings"].groupby("ticker")}

    original = signals.compute_earnings_surprise

    def frozen_earnings(ticker, df):
        table = earn.get(ticker)
        if table is not None:
            date_col = next(c for c in table.columns if "date" in c.lower())
            table = table.set_index(date_col)
        return signals.earnings_surprise_from(table, df, ticker)

    signals.compute_earnings_surprise = frozen_earnings
    cal = nse_calendar.load()
    out, report = [], {"arm": arm, "skipped": [], "non_sessions_dropped": {}}
    try:
        for t in tickers:
            raw = frames["ohlcv"][frames["ohlcv"]["ticker"] == t]
            kept, gone = nse_calendar.drop_non_sessions(raw, cal)
            if gone:
                report["non_sessions_dropped"][t] = gone
            f = signals.compute_signals_frame(t, kept.copy(), benchmarks.get(t))
            if f is None:
                report["skipped"].append(t)
                continue
            out.append(f)
    finally:
        signals.compute_earnings_surprise = original

    cols = ["date", "ticker", "close", *FEATURE_COLS, TARGET, EXCESS_TARGET,
            "benchmark_close", "benchmark_ticker", "benchmark_sector_specific"]
    p = pd.concat(out, ignore_index=True)[cols].copy()
    for c in cols:
        if c not in ("date", "ticker", "benchmark_ticker", "benchmark_sector_specific"):
            p[c] = _db_real(p[c])

    macro = frames["macro"].drop_duplicates("date").set_index("date").sort_index()
    grid = pd.Index(sorted(p["date"].unique()))
    aligned = (macro.reindex(columns=MACRO_COLS)
               .reindex(macro.index.union(grid)).sort_index().ffill().reindex(grid))
    aligned.index.name = "date"
    p = p.merge(aligned.reset_index(), on="date", how="left")
    for col in FEATURES:
        p[col] = pd.to_numeric(p[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col not in NULLABLE_FEATURES:
            p[col] = p[col].fillna(0.0)
    for col in TARGETS:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p.sort_values(["date", "ticker"]).reset_index(drop=True)

    report["rows"] = int(len(p))
    report["tickers"] = int(p["ticker"].nunique())
    report["labelled"] = {c: int(p[c].notna().sum()) for c in TARGETS}
    report["benchmarks"] = {t: {"name": b.name, "sector_specific": b.sector_specific,
                                "peers": b.peers} for t, b in benchmarks.items()}
    # v5: a thin-sector name's features are market-relative, not NULL.
    report["market_benchmark_tickers"] = sorted(
        p.groupby("ticker")["benchmark_sector_specific"].max().loc[lambda s: s == 0].index)
    report["null_share"] = {c: float(p[c].isna().mean()) for c in NULLABLE_FEATURES}
    return p, report


def guard(frames: dict, panel: pd.DataFrame) -> dict:
    """
    The write guard, simulated per ticker against the STORED label counts in
    the snapshot: would `_upsert_signals` accept this ticker's frame?

    Same arithmetic as `signals._upsert_signals`: labels counted over NSE
    sessions only, from the frame's first date onward, per label column.
    """
    from data import nse_calendar
    from pipeline.signals import LABEL_COLS

    cal = nse_calendar.load()
    stored = frames["signals"]
    verdicts = {}
    for t, f in panel.groupby("ticker"):
        start = f["date"].min()
        s = stored[(stored["ticker"] == t) & (stored["date"] >= start)]
        rows = {}
        for col in LABEL_COLS:
            have = s.loc[s[col].notna(), "date"]
            existing = int(cal.session_mask(have).sum()) if len(have) else 0
            inc = f.loc[f[col].notna(), "date"]
            incoming = int(cal.session_mask(inc).sum()) if len(inc) else 0
            rows[col] = {"existing": existing, "incoming": incoming,
                         "ok": incoming >= existing}
        verdicts[t] = rows
    refused = sorted(t for t, v in verdicts.items() if not all(r["ok"] for r in v.values()))
    return {"refused": refused, "n_ok": len(verdicts) - len(refused),
            "n": len(verdicts), "per_ticker": verdicts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "verify", "panel", "guard"])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--arm", choices=["new", "old"], default="new")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.cmd == "build":
        m = build(args.dir, args.cutoff)
        print(json.dumps({k: m[k] for k in ("cutoff", "content_hash", "files")}, indent=1))
        print(f"{len(m['defects'])} vendor defect tag(s):")
        for d in m["defects"]:
            print("  ", json.dumps(d, default=str)[:300])
        return
    if args.cmd == "verify":
        sys.exit(0 if verify(args.dir) else 1)

    manifest, frames = load_snapshot(args.dir)
    t0 = time.time()
    panel, report = build_panel(frames, args.arm)
    report["snapshot"] = manifest["content_hash"]
    report["seconds"] = round(time.time() - t0, 1)
    if args.cmd == "guard":
        g = guard(frames, panel)
        report["guard"] = {k: g[k] for k in ("refused", "n_ok", "n")}
        report["guard_per_ticker"] = g["per_ticker"]
        print(f"guard, arm {args.arm}: {g['n_ok']} of {g['n']} tickers would write; "
              f"refused: {g['refused']}")
    out = args.out or (args.dir.parent / f"panel_{args.arm}.parquet")
    panel.to_parquet(out, index=False)
    report["panel_path"] = str(out)
    report["panel_sha256"] = sha256_file(out)
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=1, default=str),
                                          encoding="utf-8")
    print(f"panel {args.arm}: {report['rows']:,} rows, {report['tickers']} tickers, "
          f"labelled {report['labelled']}, skipped {report['skipped']}, "
          f"sha256 {report['panel_sha256'][:16]}, {report['seconds']}s")
    print(f"  market benchmark for {len(report['market_benchmark_tickers'])} "
          f"tickers: {report['market_benchmark_tickers']}")
    print(f"  null shares: " + ", ".join(f"{k} {v:.3f}" for k, v in report['null_share'].items()))


if __name__ == "__main__":
    main()
