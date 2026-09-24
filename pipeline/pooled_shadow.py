"""
pipeline/pooled_shadow.py — the pooled model in production, IN SHADOW.

WHAT SHADOW MEANS HERE
----------------------
Real forecasts, real price intervals and real grades, produced on the same
cadence and from the same data as the live per-ticker model — and written to
four tables NO PUBLIC ENDPOINT READS:

    shadow_models            one row per weekly fit: the booster, its params,
                             its conformal calibration and coverage gate
    shadow_evaluations       one row per (model, ticker): the Stage 0c grade
    shadow_panel_statements  one row per model: the panel-level evidence
                             statement, including the tau2 ~ 0 detector
    shadow_forecasts         one row per (date, ticker, model): the forecast

`forecast_current`, `forecasts`, `model_metadata` and every `/api/*` public
route are untouched; a test holds the public responses byte-identical with and
without shadow rows present. The only reader is the admin-gated, unlinked
`/api/admin/shadow/*`. The DDL lives here, not in `data/db.py` — the
`evidence_grades_v2` pattern — so the tables need no Render redeploy of
their own.

The cutover is a later session: after two clean weekly shadow cycles and the
user's explicit go-ahead. See docs/dashboard-switch-preregistration.md for
what two cycles CAN and CANNOT show.

EVERY ROW CARRIES ITS PROVENANCE: `pooled_version` (POOLED_MODEL_VERSION),
`config_hash` (what a human chose), `data_hash` (what the database held) and
`env_hash` (what the machine was — the platform included, because Windows and
Linux give different numbers from identical code).

THE INVERSE NEVER SEES THE FUTURE. The model predicts a within-date z-score;
the published return, price and interval need that date's cross-sectional
mean and sd, which are properties of the 30 sessions AFTER the forecast. Only
the causal, past-only estimate (`pooled.causal_frame`, lagged by the horizon)
is used here — never the realised moments, which would be F1 in a new place.

ROMANO-WOLF IS WIRED, NOT ASSUMED. The live gate caps a ticker at WEAK when
`eval_rw_significant` is absent — and it is still never written, so the live
per-ticker model cannot reach STRONG at all. Here the grade comes straight
from `grade_panel_v3`, whose STRONG REQUIRES the Romano-Wolf rejection it
computed on the same bootstrap; `rw_rejected` and the adjusted p are stored
on every evaluation row.

ONE STATEMENT, NOT 84. When tau2 is at the zero boundary every posterior is
the grand mean and every ticker carries the same grade. Then the evaluation
rows store grade NULL and the panel statement carries the one finding.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from data.db import get_engine

#: THE CONFORMAL GATE, pre-registered in docs/dashboard-switch-preregistration.md
#: BEFORE it was measured on the reference platform. Coverage of the 80%
#: interval, in PRICE space (the interval is monotone in the log return, so
#: covering the realised price and covering the realised log return are the
#: same event), measured by expanding calibration (`conformal.
#: expanding_fold_coverage`).
#:
#: THE BANDS ARE UNCHANGED FROM THE GATE THE CONSTANT-WIDTH INTERVAL FAILED
#: (0.868 overall, 0.903 in fold 4). Since pooled v2 the interval is SPREAD-
#: NORMALISED (docs/stage2-fallback-conformal-preregistration.md §3): the
#: residual is scored in units of a past-only cross-sectional spread
#: (`pooled.spread_frame`) and the band is `pred ± q × spread`. The same bands
#: judge it; widening them after a failure would be tuning the gate.
COVERAGE_NOMINAL = 0.80
#: Overall: the ±5pp rule `conformal.check_coverage` already names
#: `well_calibrated`, and the band inside which "80%" is an honest description.
COVERAGE_BAND_OVERALL = (0.75, 0.85)
#: Per fold: ±10pp. Each fold holds ~13 non-overlapping 30-session windows and
#: a common market shock moves every name's residual at once, so a fold's
#: coverage is a noisy estimate — but a fold outside ±10pp is a band a reader
#: relying on "80%" would find materially wrong for that whole period.
COVERAGE_BAND_FOLD = (0.70, 0.90)

GATE_PASS, GATE_FAIL, GATE_UNMEASURED = "PASS", "FAIL", "UNMEASURED"


def coverage_gate(y_true, y_pred, folds, spread=None, price=None) -> dict:
    """The pre-registered conformal gate. Never tuned after measuring.
    `spread`: score the spread-normalised interval instead of the constant one."""
    from pipeline.conformal import expanding_fold_coverage

    cov = expanding_fold_coverage(y_true, y_pred, folds, coverage=COVERAGE_NOMINAL,
                                  spread=spread, price=price)
    lo, hi = COVERAGE_BAND_OVERALL
    flo, fhi = COVERAGE_BAND_FOLD
    reasons = []
    if not cov["per_fold"] or not np.isfinite(cov["overall"]):
        return {**cov, "status": GATE_UNMEASURED,
                "reasons": ["no fold could be checked"]}
    if not lo <= cov["overall"] <= hi:
        reasons.append(f"overall coverage {cov['overall']:.4f} outside "
                       f"[{lo:.2f}, {hi:.2f}]")
    for f in cov["per_fold"]:
        if not flo <= f["coverage"] <= fhi:
            reasons.append(f"fold {f['fold']} coverage {f['coverage']:.4f} outside "
                           f"[{flo:.2f}, {fhi:.2f}]")
    return {**cov, "status": GATE_FAIL if reasons else GATE_PASS,
            "reasons": reasons}


# ── schema ────────────────────────────────────────────────────────────────────

SHADOW_TABLES = ("shadow_models", "shadow_evaluations", "shadow_panel_statements",
                 "shadow_forecasts")


def init_shadow_tables(engine=None) -> None:
    """Idempotent DDL for the four shadow tables (SQLite and Postgres)."""
    engine = engine or get_engine()
    blob = "BYTEA" if engine.dialect.name == "postgresql" else "BLOB"
    ddl = [
        f"""CREATE TABLE IF NOT EXISTS shadow_models (
            model_id         TEXT PRIMARY KEY,
            pooled_version   TEXT NOT NULL,
            trained_at       TEXT NOT NULL,
            train_first_date TEXT,
            train_last_date  TEXT,
            n_train_rows     INTEGER,
            features_json    TEXT,
            params_json      TEXT,
            booster          {blob} NOT NULL,
            calibration_json TEXT,
            coverage_json    TEXT,
            coverage_status  TEXT,
            walk_forward_json TEXT,
            config_hash      TEXT,
            data_hash        TEXT,
            env_hash         TEXT,
            env_json         TEXT,
            git_sha          TEXT,
            runtime_seconds  REAL
        )""",
        """CREATE TABLE IF NOT EXISTS shadow_evaluations (
            model_id        TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            pooled_version  TEXT NOT NULL,
            evaluated_at    TEXT NOT NULL,
            n_folds_scored  INTEGER,
            hat_ic          REAL,
            sigma2          REAL,
            theta           REAL,
            p_positive      REAL,
            rw_adjusted_p   REAL,
            rw_rejected     INTEGER,
            bh_significant  INTEGER,
            grade           TEXT,
            reason          TEXT,
            config_hash     TEXT,
            data_hash       TEXT,
            env_hash        TEXT,
            PRIMARY KEY (model_id, ticker)
        )""",
        """CREATE TABLE IF NOT EXISTS shadow_panel_statements (
            model_id         TEXT PRIMARY KEY,
            pooled_version   TEXT NOT NULL,
            evaluated_at     TEXT NOT NULL,
            tickers_total    INTEGER,
            n_usable         INTEGER,
            mu_hat           REAL,
            mu_se_bootstrap  REAL,
            z_bootstrap      REAL,
            tau2_reml        REAL,
            degenerate       INTEGER,
            degeneracy_reason TEXT,
            statement        TEXT,
            cs_ic            REAL,
            cs_ic_se         REAL,
            grade_counts_json TEXT,
            notes_json       TEXT,
            config_hash      TEXT,
            data_hash        TEXT,
            env_hash         TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS shadow_forecasts (
            forecast_date    TEXT NOT NULL,
            ticker           TEXT NOT NULL,
            model_id         TEXT NOT NULL,
            pooled_version   TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            pred_z           REAL,
            causal_mean      REAL,
            causal_sd        REAL,
            pred_return      REAL,
            current_price    REAL,
            implied_price    REAL,
            interval_low     REAL,
            interval_high    REAL,
            interval_coverage REAL,
            interval_spread  REAL,
            prob_up          REAL,
            grade            TEXT,
            panel_statement  INTEGER,
            coverage_status  TEXT,
            config_hash      TEXT,
            data_hash        TEXT,
            env_hash         TEXT,
            PRIMARY KEY (forecast_date, ticker, model_id)
        )""",
    ]
    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))
    # Columns added after a table first shipped. CREATE IF NOT EXISTS leaves an
    # existing table as it was, so each is added here when absent.
    from sqlalchemy import inspect

    have = {c["name"] for c in inspect(engine).get_columns("shadow_forecasts")}
    with engine.begin() as conn:
        for col, typ in (("interval_spread", "REAL"),):     # pooled v2
            if col not in have:
                conn.execute(text(f"ALTER TABLE shadow_forecasts ADD COLUMN {col} {typ}"))


# ── provenance ────────────────────────────────────────────────────────────────


def pooled_config_hash() -> tuple[str, dict]:
    """Everything a human chose about the pooled model, and its digest."""
    from pipeline import pooled
    from pipeline.evidence_panel import BLOCK_LENGTH_SESSIONS, BOOTSTRAP_B
    from pipeline.label import MOMENT_LOOKBACK
    from pipeline.sector_benchmark import MIN_SECTOR_PEERS
    from pipeline.tracking import _sha

    config = {
        "pooled_version": pooled.POOLED_MODEL_VERSION,
        "features": list(pooled.FEATURES),
        "objective": pooled.OBJECTIVE,
        "label": "within-date standardised target_return",
        "missing": "nullable features reach XGBoost as NaN",
        "horizon": pooled.HORIZON,
        "n_folds": pooled.N_FOLDS,
        "n_trials": pooled.N_TRIALS,
        "min_train_dates": pooled.MIN_TRAIN_DATES,
        "causal_moment_lookback": MOMENT_LOOKBACK,
        "interval": {"method": "spread-normalised split conformal",
                     "spread_lookback": pooled.SPREAD_LOOKBACK},
        "bootstrap_b": BOOTSTRAP_B,
        "block": BLOCK_LENGTH_SESSIONS,
        "coverage": [COVERAGE_NOMINAL, list(COVERAGE_BAND_OVERALL),
                     list(COVERAGE_BAND_FOLD)],
        "min_sector_peers": MIN_SECTOR_PEERS,
    }
    return _sha(config), config


def _hashes(universe, engine) -> dict:
    from pipeline.tracking import data_hash, environment_hash

    c, _ = pooled_config_hash()
    d, _ = data_hash(universe, engine)
    e, env = environment_hash()
    return {"config_hash": c, "data_hash": d, "env_hash": e, "env": env}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── the weekly step ───────────────────────────────────────────────────────────


def evaluation_rows(grading, model_id: str, hashes: dict, evaluated_at: str
                    ) -> tuple[list[dict], dict]:
    """
    `grade_panel_v3`'s per-ticker rows as shadow rows, plus the panel statement.

    The Romano-Wolf producer IS this mapping: `rw_rejected` and the adjusted p
    go onto every row, and the grade is the one `grade_panel_v3` assigned, so a
    ticker it graded STRONG (which it only does on a Romano-Wolf rejection) is
    STRONG here — never capped for want of a flag nobody wrote.

    A degenerate panel (tau2 at zero, or nothing gradeable) is ONE finding:
    each row's grade is NULL and the statement carries it.
    """
    from pipeline.pooled import POOLED_MODEL_VERSION

    one_statement = bool(grading.degeneracy.degenerate) or grading.n_usable == 0

    def f(x):
        return float(x) if x is not None and np.isfinite(x) else None

    rows = [{
        "model_id": model_id, "ticker": r.ticker,
        "pooled_version": POOLED_MODEL_VERSION, "evaluated_at": evaluated_at,
        "n_folds_scored": int(r.n_folds_scored), "hat_ic": f(r.hat_ic),
        "sigma2": f(r.sigma2), "theta": f(r.theta), "p_positive": f(r.p_positive),
        "rw_adjusted_p": f(r.rw_adjusted_p), "rw_rejected": int(bool(r.rw_rejected)),
        "bh_significant": int(bool(r.bh_significant)),
        "grade": None if one_statement else r.grade,
        "reason": ("see the panel statement: " + grading.headline()[:200]
                   if one_statement else r.reason),
        "config_hash": hashes["config_hash"], "data_hash": hashes["data_hash"],
        "env_hash": hashes["env_hash"],
    } for r in grading.rows]

    statement = {
        "model_id": model_id, "pooled_version": POOLED_MODEL_VERSION,
        "evaluated_at": evaluated_at, "tickers_total": int(grading.tickers_total),
        "n_usable": int(grading.n_usable), "mu_hat": f(grading.mu_hat),
        "mu_se_bootstrap": f(grading.mu_se_bootstrap),
        "z_bootstrap": f(grading.z_bootstrap), "tau2_reml": f(grading.tau2_reml),
        "degenerate": int(one_statement),
        "degeneracy_reason": grading.degeneracy.reason or None,
        "statement": grading.headline(),
        "cs_ic": f(grading.cross_sectional_ic),
        "cs_ic_se": f(grading.cross_sectional_ic_se),
        "grade_counts_json": json.dumps(grading.counts()),
        "notes_json": json.dumps(grading.notes),
        "config_hash": hashes["config_hash"], "data_hash": hashes["data_hash"],
        "env_hash": hashes["env_hash"],
    }
    return rows, statement


def _insert(conn, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    conn.execute(text(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES "
        f"({', '.join(':' + c for c in cols)}) ON CONFLICT DO NOTHING"), rows)


def run_weekly_shadow(universe: list[str] | None = None, engine=None,
                      n_trials: int | None = None, bootstrap: int | None = None,
                      verbose: bool = True) -> dict:
    """
    Train, evaluate, grade and calibrate the pooled model; write the shadow
    model, its evaluations and its panel statement. Returns a metrics dict
    for `experiment_runs`.
    """
    from pipeline import pooled
    from pipeline.conformal import fit_scaled_conformal
    from pipeline.evidence_panel import BOOTSTRAP_B, grade_panel_v3
    from pipeline.panel import load_panel
    from pipeline.tracking import git_sha

    engine = engine or get_engine()
    if universe is None:
        from data.universe import get_universe
        universe = get_universe()
    n_trials = pooled.N_TRIALS if n_trials is None else n_trials
    bootstrap = BOOTSTRAP_B if bootstrap is None else bootstrap
    started = time.time()
    init_shadow_tables(engine)

    raw = load_panel(universe, engine=engine)
    prepared = pooled.prepare(raw)
    preds, folds = pooled.walk_forward(prepared, n_trials=n_trials, verbose=verbose)
    t_wf = time.time() - started

    inv = pooled.invert(preds, pooled.causal_frame(raw)).merge(
        pooled.spread_frame(raw), on="date", how="left")
    ok = (np.isfinite(inv["pred_return"]) & np.isfinite(inv["y_raw"])
          & np.isfinite(inv["interval_spread"]))
    gate = coverage_gate(inv.loc[ok, "y_raw"], inv.loc[ok, "pred_return"],
                         inv.loc[ok, "fold"], spread=inv.loc[ok, "interval_spread"])
    calibration = fit_scaled_conformal(inv.loc[ok, "y_raw"].to_numpy(),
                                       inv.loc[ok, "pred_return"].to_numpy(),
                                       inv.loc[ok, "interval_spread"].to_numpy(),
                                       coverage=COVERAGE_NOMINAL)

    grading = grade_panel_v3(preds["date"].astype(str).to_numpy(),
                             preds["ticker"].astype(str).to_numpy(),
                             preds["fold"].to_numpy(),
                             preds["y_raw"].to_numpy(dtype=float),
                             preds["y_pred"].to_numpy(dtype=float),
                             n_bootstrap=bootstrap)
    t_grade = time.time() - started - t_wf

    fitted = pooled.fit_final(prepared, n_trials=n_trials)
    model_id = hashlib.sha256(fitted.booster).hexdigest()[:16]
    hashes = _hashes(universe, engine)
    evaluated_at = _now()
    ev_rows, statement = evaluation_rows(grading, model_id, hashes, evaluated_at)

    model_row = {
        "model_id": model_id, "pooled_version": pooled.POOLED_MODEL_VERSION,
        "trained_at": evaluated_at, "train_first_date": fitted.train_first_date,
        "train_last_date": fitted.train_last_date,
        "n_train_rows": fitted.n_train_rows,
        "features_json": json.dumps(fitted.features),
        "params_json": json.dumps(fitted.params), "booster": fitted.booster,
        "calibration_json": json.dumps(None if calibration is None else {
            "method": "spread-normalised",
            "spread_lookback": pooled.SPREAD_LOOKBACK,
            "quantile": calibration.quantile, "coverage": calibration.coverage,
            "n": calibration.n,
            "scores": [round(float(r), 6) for r in calibration.scores]}),
        "coverage_json": json.dumps({k: v for k, v in gate.items()}, default=float),
        "coverage_status": gate["status"],
        "walk_forward_json": json.dumps(
            [{k: v for k, v in f.items() if k != "params"} for f in folds]),
        "config_hash": hashes["config_hash"], "data_hash": hashes["data_hash"],
        "env_hash": hashes["env_hash"], "env_json": json.dumps(hashes["env"]),
        "git_sha": git_sha(), "runtime_seconds": round(time.time() - started, 1),
    }
    with engine.begin() as conn:
        _insert(conn, "shadow_models", [model_row])
        _insert(conn, "shadow_evaluations", ev_rows)
        _insert(conn, "shadow_panel_statements", [statement])

    counts = grading.counts()
    return {
        "model_id": model_id, "pooled_version": pooled.POOLED_MODEL_VERSION,
        "oos_rows": int(len(preds)), "folds": len(folds),
        "gammas": [round(f["gamma"], 3) for f in folds],
        "constant_cells": int((preds.groupby(["ticker", "fold"])["y_pred"]
                               .nunique() <= 1).sum()),
        "cs_ic": statement["cs_ic"], "cs_ic_se": statement["cs_ic_se"],
        "grades": counts, "panel_statement": statement["degenerate"] == 1,
        "tau2": statement["tau2_reml"], "statement": statement["statement"][:300],
        "coverage_status": gate["status"], "coverage_overall": gate["overall"],
        "coverage_method": gate.get("method"),
        "coverage_by_fold": [round(f["coverage"], 4) for f in gate["per_fold"]],
        "env_hash": hashes["env_hash"],
        "seconds_walk_forward": round(t_wf, 1), "seconds_grading": round(t_grade, 1),
        "seconds_total": round(time.time() - started, 1),
    }


# ── the daily step ────────────────────────────────────────────────────────────


def latest_model(engine=None) -> dict | None:
    engine = engine or get_engine()
    df = pd.read_sql(text("SELECT * FROM shadow_models ORDER BY trained_at DESC LIMIT 1"),
                     engine)
    return None if df.empty else df.iloc[0].to_dict()


def run_daily_shadow(universe: list[str] | None = None, engine=None) -> dict:
    """
    Forecast every name from the latest shadow model into `shadow_forecasts`.
    Cheap: one panel load, one predict, no search.
    """
    from pipeline import pooled
    from pipeline.conformal import (ConformalCalibration,
                                    ScaledConformalCalibration, to_price_view)
    from pipeline.panel import load_panel

    engine = engine or get_engine()
    if universe is None:
        from data.universe import get_universe
        universe = get_universe()
    init_shadow_tables(engine)
    model = latest_model(engine)
    if model is None:
        return {"note": "no shadow model yet; the weekly shadow step writes one"}

    fitted = pooled.FittedPooled(
        booster=bytes(model["booster"]), params=json.loads(model["params_json"]),
        n_train_rows=int(model["n_train_rows"]),
        train_first_date=str(model["train_first_date"]),
        train_last_date=str(model["train_last_date"]),
        features=json.loads(model["features_json"]))
    cal = json.loads(model["calibration_json"] or "null")
    scaled = constant = None
    if cal and cal.get("method") == "spread-normalised":
        scaled = ScaledConformalCalibration(
            quantile=float(cal["quantile"]), coverage=float(cal["coverage"]),
            scores=np.asarray(cal["scores"], dtype=float), n=int(cal["n"]))
    elif cal:                                    # a pooled v1 calibration
        constant = ConformalCalibration(
            quantile=float(cal["quantile"]), coverage=float(cal["coverage"]),
            residuals=np.asarray(cal["residuals"], dtype=float), n=int(cal["n"]))

    raw = load_panel(universe, engine=engine)
    prepared = pooled.prepare(raw)
    as_of = str(prepared["date"].max())
    preds = pooled.predict_on(fitted, prepared, as_of)
    inv = pooled.invert(preds, pooled.causal_frame(raw), z_col="pred_z").merge(
        pooled.spread_frame(raw), on="date", how="left")

    grades = pd.read_sql(text("SELECT ticker, grade FROM shadow_evaluations "
                              "WHERE model_id = :m"), engine,
                         params={"m": model["model_id"]}).set_index("ticker")["grade"]
    stmt = pd.read_sql(text("SELECT degenerate FROM shadow_panel_statements "
                            "WHERE model_id = :m"), engine,
                       params={"m": model["model_id"]})
    one_statement = bool(len(stmt) and int(stmt["degenerate"].iloc[0]))
    hashes = _hashes(universe, engine)
    created = _now()

    rows, unpriced = [], []
    for r in inv.itertuples(index=False):
        ret, price = float(r.pred_return), float(r.close)
        spread = float(r.interval_spread)
        # The interval is priced at THIS date's past-only spread; with no
        # spread there is no interval, never one at a made-up width.
        calibration = scaled.at(spread) if scaled is not None else constant
        if not (np.isfinite(ret) and np.isfinite(price) and price > 0):
            unpriced.append(r.ticker)
            view = {}
        else:
            view = to_price_view(price, ret, calibration)
        g = grades.get(r.ticker)
        rows.append({
            "forecast_date": as_of, "ticker": r.ticker, "model_id": model["model_id"],
            "pooled_version": model["pooled_version"], "created_at": created,
            "pred_z": float(r.pred_z),
            "causal_mean": float(r.causal_mean) if np.isfinite(r.causal_mean) else None,
            "causal_sd": float(r.causal_sd) if np.isfinite(r.causal_sd) else None,
            "pred_return": ret if np.isfinite(ret) else None,
            "current_price": price if np.isfinite(price) else None,
            "implied_price": view.get("implied_price"),
            "interval_low": view.get("interval_low"),
            "interval_high": view.get("interval_high"),
            "interval_coverage": view.get("interval_coverage"),
            "interval_spread": (spread if scaled is not None and np.isfinite(spread)
                                else None),
            "prob_up": view.get("prob_up"),
            "grade": None if (one_statement or g is None or pd.isna(g)) else str(g),
            "panel_statement": int(one_statement),
            "coverage_status": model["coverage_status"],
            "config_hash": hashes["config_hash"], "data_hash": hashes["data_hash"],
            "env_hash": hashes["env_hash"],
        })
    with engine.begin() as conn:
        _insert(conn, "shadow_forecasts", rows)
    return {"model_id": model["model_id"], "forecast_date": as_of,
            "forecasts": len(rows), "unpriced": unpriced,
            "missing_tickers": sorted(set(universe) - set(inv["ticker"])),
            "coverage_status": model["coverage_status"],
            "panel_statement": one_statement}
