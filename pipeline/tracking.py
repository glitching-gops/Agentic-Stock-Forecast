"""
pipeline/tracking.py — one row per run: what code, what configuration, what data.

THE QUESTION THIS ANSWERS. "The numbers moved — why?" Between any two runs the
model version, the feature list, the forecast horizon, the evaluation settings,
the universe rule, the benchmark mapping and the labelled set can all change,
and until now none of it was recorded anywhere. Reconstructing it meant reading
git log against a log file, which is exactly how the Phase 0 headline metrics
survived unexamined for months.

TWO HASHES, DELIBERATELY SEPARATE.

  config_hash covers everything a human chose: features, target, horizon,
  evaluation folds and embargo, model version, and the sector-benchmark mapping.
  It changes only when someone edits the code.

  data_hash covers what the database actually held: which tickers, how much
  history each had, and how many rows carried a label. It changes every day by
  design.

Keeping them apart is the point. A metric that moves while config_hash is
constant is a data or market effect; one that moves while data_hash is constant
is a code effect. Hashing them together would collapse the only distinction
worth having.

THE BENCHMARK MAPPING IS PART OF THE CONFIG HASH. It is half the label: change
`Financial Services -> ^NSEBANK` to `-> ^NSEI` and every historical
target_excess_return for 22 tickers means something different. A run before and
after that change is not comparable, and the hash is what makes that visible
instead of a mystery.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from data.db import get_engine, to_native_params


def json_safe(value):
    """
    Replaces non-finite floats with None, recursively.

    ``json.dumps`` serialises ``float('nan')`` as a bare ``NaN`` token. Python's
    own ``json.loads`` accepts that as an extension, so it round-trips locally
    and looks correct; every strict parser rejects it — JavaScript's
    ``JSON.parse``, ``jq``, and a Postgres ``::jsonb`` cast alike. The
    ``metrics`` column is TEXT, so nothing raises at write time and the damage
    only surfaces wherever the value is eventually read.

    NaN is not rare in what gets recorded here: a comparator with no ordering
    has an undefined rank IC by design, so a baseline table containing `zero`
    and `majority` produces one on every run. None is the right translation —
    JSON null means "not measured", which is exactly what an undefined
    statistic is.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha(payload) -> str:
    """Short, stable digest of a JSON-serialisable structure."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def git_sha() -> str | None:
    """
    The commit the run executed. GITHUB_SHA is authoritative on Actions; the
    subprocess fallback covers local runs and is allowed to fail silently,
    because a missing git binary is not a reason to abort a pipeline.
    """
    env = os.getenv("GITHUB_SHA")
    if env:
        return env[:40]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10,
                             cwd=os.path.dirname(os.path.dirname(__file__)))
        return out.stdout.strip()[:40] or None
    except Exception:                                           # noqa: BLE001
        return None


def config_hash() -> tuple[str, dict]:
    """Everything a human chose, and its digest."""
    from data.tickers import BROAD_MARKET_INDEX, SECTOR_INDICES
    from pipeline.model import (
        EVAL_MIN_TRAIN, EVAL_N_FOLDS, EVAL_TUNE_TRIALS, FEATURES,
        MIN_ROWS_FOR_EVALUATION, MIN_ROWS_FOR_FORECAST, MODEL_VERSION, TARGET,
    )
    from pipeline.signals import HORIZON_SESSIONS

    config = {
        "model_version": MODEL_VERSION,
        "features": sorted(FEATURES),
        "target": TARGET,
        "horizon_sessions": HORIZON_SESSIONS,
        "eval_n_folds": EVAL_N_FOLDS,
        "eval_min_train": EVAL_MIN_TRAIN,
        "eval_tune_trials": EVAL_TUNE_TRIALS,
        "min_rows_forecast": MIN_ROWS_FOR_FORECAST,
        "min_rows_evaluation": MIN_ROWS_FOR_EVALUATION,
        # Half the label. See the module docstring.
        "benchmarks": dict(sorted(SECTOR_INDICES.items())),
        "broad_market": BROAD_MARKET_INDEX,
    }
    return _sha(config), config


def data_hash(universe: list[str] | None = None, engine=None) -> tuple[str, dict]:
    """
    What the database held, and its digest.

    Deliberately built from per-ticker coverage rather than from row contents:
    hashing 220,000 rows on every run would cost more than it tells us, and the
    failures this project actually has — a frozen labelled set, an erased
    ticker, a truncated history — all show up in the counts.
    """
    engine = engine or get_engine()
    if universe is None:
        from data.universe import get_universe
        universe = get_universe()

    coverage = pd.read_sql(
        text("SELECT ticker, COUNT(*) AS rows, "
             "COUNT(target_excess_return) AS labelled, "
             "MIN(date) AS first_date, MAX(date) AS last_date "
             "FROM signals GROUP BY ticker"),
        engine,
    )
    per_ticker = {
        r.ticker: [int(r.rows), int(r.labelled), str(r.first_date)[:10],
                   str(r.last_date)[:10]]
        for r in coverage.itertuples(index=False)
        if r.ticker in set(universe)
    }
    payload = {"universe": sorted(universe), "coverage": per_ticker}
    stats = {
        "n_tickers": len(universe),
        "labelled_rows": sum(v[1] for v in per_ticker.values()),
        "total_rows": sum(v[0] for v in per_ticker.values()),
        "last_date": max((v[3] for v in per_ticker.values()), default=None),
    }
    return _sha(payload), stats


def start_run(job: str, universe: list[str] | None = None, engine=None) -> str:
    """
    Opens an experiment_runs row and returns its id.

    Written at the START, with status RUNNING, so that a job which dies without
    reaching finish_run leaves evidence. A row that stays RUNNING is itself a
    finding: it means the process was killed rather than aborted — which is what
    an OOM looks like, and this project has had one.
    """
    engine = engine or get_engine()
    run_id = uuid.uuid4().hex[:16]

    from data.universe import DEFAULT_RULE
    from pipeline.model import MODEL_VERSION

    cfg, _ = config_hash()
    try:
        data, stats = data_hash(universe, engine)
    except Exception as exc:                                    # noqa: BLE001
        data, stats = None, {"error": str(exc)}

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO experiment_runs
                    (run_id, job, started_at, status, git_sha, model_version,
                     config_hash, data_hash, universe_rule, n_tickers,
                     labelled_rows)
                VALUES
                    (:run_id, :job, :started_at, 'RUNNING', :git_sha,
                     :model_version, :config_hash, :data_hash, :universe_rule,
                     :n_tickers, :labelled_rows)
            """),
            to_native_params({
                "run_id": run_id,
                "job": job,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": git_sha(),
                "model_version": MODEL_VERSION,
                "config_hash": cfg,
                "data_hash": data,
                "universe_rule": DEFAULT_RULE.fingerprint(),
                "n_tickers": stats.get("n_tickers"),
                "labelled_rows": stats.get("labelled_rows"),
            }),
        )
        conn.commit()

    print(f"[Tracking] run {run_id} ({job}) config={cfg} data={data}")
    return run_id


def finish_run(run_id: str, status: str, gate=None, metrics: dict | None = None,
               notes: str | None = None, engine=None) -> None:
    """Closes the row. Never raises — tracking must not be able to fail a run."""
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE experiment_runs
                       SET finished_at = :finished_at,
                           status      = :status,
                           gate_status = :gate_status,
                           gate_report = :gate_report,
                           metrics     = :metrics,
                           notes       = :notes
                     WHERE run_id = :run_id
                """),
                to_native_params({
                    "run_id": run_id,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "gate_status": gate.status if gate else None,
                    "gate_report": gate.to_json() if gate else None,
                    "metrics": json.dumps(json_safe(metrics or {}),
                                          default=str, allow_nan=False),
                    "notes": notes,
                }),
            )
            conn.commit()
    except Exception as exc:                                    # noqa: BLE001
        print(f"[Tracking] failed to close run {run_id}: {exc}")


def recent_runs(limit: int = 20, engine=None) -> pd.DataFrame:
    """The run log, newest first — the thing to read when metrics move."""
    engine = engine or get_engine()
    return pd.read_sql(
        text("SELECT run_id, job, started_at, finished_at, status, gate_status, "
             "git_sha, config_hash, data_hash, n_tickers, labelled_rows "
             "FROM experiment_runs ORDER BY started_at DESC LIMIT :n"),
        engine, params={"n": limit},
    )
