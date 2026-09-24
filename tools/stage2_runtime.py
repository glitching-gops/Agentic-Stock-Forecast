"""
tools/stage2_runtime.py — does the shadow step fit a GitHub Actions runner?

Loads a Stage 2 snapshot panel into a throwaway SQLite database with the real
schema, then runs the PRODUCTION `run_weekly_shadow` (full settings: nested
search, B = 1000 bootstrap) and `run_daily_shadow` against it, timing each.
Run it under `taskset -c 0-3` and `/usr/bin/time -v` to match a public-repo
`ubuntu-latest` runner (4 vCPU, 16 GB) and read peak memory:

    /usr/bin/time -v taskset -c 0-3 python tools/stage2_runtime.py \
        --panel stage2/panel_new.parquet --db /tmp/shadow.sqlite

Writes only to the SQLite file it is given. Never touches a real database.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.db.exists():
        args.db.unlink()
    url = f"sqlite:///{args.db.as_posix()}"
    os.environ["DATABASE_URL"] = url
    from sqlalchemy import create_engine

    import data.db as db
    engine = create_engine(url)
    db.get_engine = lambda: engine
    if engine.dialect.name != "sqlite":
        raise SystemExit("refusing: this tool only ever writes to SQLite")
    db.init_db()

    from pipeline.panel import MACRO_COLS
    from pipeline.pooled_shadow import run_daily_shadow, run_weekly_shadow

    panel = pd.read_parquet(args.panel)
    signal_cols = [c for c in panel.columns if c not in MACRO_COLS]
    panel[signal_cols].to_sql("signals", engine, index=False, if_exists="append")
    panel.drop_duplicates("date")[["date", *MACRO_COLS]].to_sql(
        "macro", engine, index=False, if_exists="append")
    universe = sorted(panel["ticker"].unique())

    t0 = time.time()
    weekly = run_weekly_shadow(universe, engine)
    t_weekly = time.time() - t0
    t1 = time.time()
    daily = run_daily_shadow(universe, engine)
    t_daily = time.time() - t1
    out = {"weekly": weekly, "weekly_seconds": round(t_weekly, 1),
           "daily": daily, "daily_seconds": round(t_daily, 1),
           "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    print(json.dumps(out, indent=1, default=str))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
