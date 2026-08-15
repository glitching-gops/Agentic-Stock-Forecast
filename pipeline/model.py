"""
pipeline/model.py — Training, honest evaluation, and forecast generation.

Split into two cadences (Lever 1), after the first production run on Render
starved a single free-tier CPU core and eventually OOM-killed the instance:
purged walk-forward evaluation with nested Optuna tuning was running in full,
for every ticker, every single day — even though "does this model form have
skill" is a slow-moving property that doesn't need daily re-measurement.

  WEEKLY  (``evaluate_and_persist_ticker``) — the expensive path. Runs the
          full purged walk-forward evaluation, refreshes cached production
          hyperparameters, calibrates conformal intervals, and PERSISTS all
          of it to ``model_metadata``. This is what actually costs the
          hundreds of XGBoost fits per ticker.

  DAILY   (``forecast_ticker_daily``) — the cheap path. Fetches the day's
          fresh price, fits the production model with already-cached
          hyperparameters (one fit, not a search), predicts, and READS the
          persisted evaluation + calibration rather than recomputing them.

A forecast's evidence grade can therefore be up to a week stale relative to
its price. That staleness is surfaced (``evaluated_at``, distinct from
``last_updated``) rather than hidden — see agents/critic_agent.py and the
``evaluated_at`` column on the forecasts/leaderboard tables.

Nothing on the daily path runs Optuna. tests/test_scheduling.py asserts this.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from xgboost import XGBRegressor

from data.db import get_engine
from pipeline.conformal import ConformalCalibration, fit_conformal, to_price_view
from pipeline.evaluation import (
    PurgedWalkForward,
    WalkForwardResult,
    format_report,
    walk_forward,
)
from pipeline.signals import FEATURE_COLS, HORIZON_SESSIONS
from pipeline.tuning import tune, tune_and_cache

MODEL_VERSION = "phase0-excess-return-v1"

FEATURES = FEATURE_COLS + [
    # Macro. `fii_net_flow` / `dii_net_flow` were scraped and stored by
    # macro.py but never referenced by the old FEATURES list; they are wired
    # in here rather than left as dead columns (audit finding F15).
    "usdinr", "india_vix", "nifty_5d_return", "nifty_20d_return",
    "fii_net_flow", "dii_net_flow",
]

TARGET = "target_excess_return"

MODELS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models", "joblib"
)

# Lever 2 (moderate cut): weekly evaluation budget. See pipeline/tuning.py
# for the matching production-tuning cut (N_TRIALS 40 -> 25).
EVAL_N_FOLDS = 5        # was 6
EVAL_TUNE_TRIALS = 10   # per-fold nested Optuna trials, was 15


@dataclass
class TickerForecast:
    """A forecast and the held-out evidence for trusting it."""

    ticker: str
    forecast_date: str
    current_price: float
    price_view: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    benchmark_ticker: str = ""
    benchmark_sector_specific: bool = False
    model_version: str = MODEL_VERSION
    n_train_rows: int = 0


# ── Feature assembly ──────────────────────────────────────────────────────────


def load_features_for_ticker(ticker: str, engine=None) -> pd.DataFrame:
    """
    Loads signals joined to macro data for one ticker, sorted by date.

    ``sentiment_score`` is deliberately ABSENT. It only ever existed for the
    current date, so every training row held 0.0 while the single row being
    predicted held a real value (audit finding F7) — a train/serve mismatch at
    exactly the row that matters. It returns once a dated news archive exists.
    """
    engine = engine or get_engine()

    signals = pd.read_sql(
        text("SELECT * FROM signals WHERE ticker = :t ORDER BY date ASC"),
        engine, params={"t": ticker},
    )
    if signals.empty:
        return pd.DataFrame()

    macro = pd.read_sql(text("SELECT * FROM macro ORDER BY date ASC"), engine)

    if macro.empty:
        for col in ["usdinr", "india_vix", "nifty_5d_return",
                    "nifty_20d_return", "fii_net_flow", "dii_net_flow"]:
            signals[col] = 0.0
        df = signals
    else:
        df = signals.merge(macro, on="date", how="left")
        macro_cols = [c for c in macro.columns if c != "date"]
        # Forward fill only — never bfill, which would import future values (F12).
        df[macro_cols] = df[macro_cols].ffill()

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df.sort_values("date").reset_index(drop=True)


def _model_factory(params: dict | None = None):
    """Returns a factory producing fresh, identically configured estimators."""
    def factory():
        model = XGBRegressor(random_state=42, verbosity=0, tree_method="hist")
        if params:
            model.set_params(**params)
        return model
    return factory


def _latest_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    The most recent row whose label is not yet knowable — the forecast row.
    Returns a single-row DataFrame (not a Series) so `model.predict(row[FEATURES])`
    gets the 2D shape XGBoost expects without an extra reshape at the call site.
    """
    unlabelled = df[df[TARGET].isna()]
    return unlabelled.iloc[[-1]] if not unlabelled.empty else df.iloc[[-1]]


# ── Evaluation (the raw building block; called weekly) ────────────────────────


def evaluate_ticker(
    ticker: str,
    df: pd.DataFrame | None = None,
    n_folds: int = EVAL_N_FOLDS,
    tune_inside_folds: bool = True,
    tune_trials: int = EVAL_TUNE_TRIALS,
) -> WalkForwardResult:
    """
    Purged walk-forward evaluation. This is the ONLY source of reported metrics.

    Hyperparameter search runs inside each training fold via the ``tuner``
    callback, so no configuration is ever chosen with sight of the rows it is
    later scored on. Expensive — hundreds of XGBoost fits per call — which is
    why it is on the weekly path, not the daily one.
    """
    df = load_features_for_ticker(ticker) if df is None else df
    if df.empty or len(df) < 350:
        return WalkForwardResult(ticker, 0, 0,
                                 pd.DataFrame(columns=["date", "y_true", "y_pred", "fold"]))

    X = df[FEATURES]
    y = df[TARGET] if TARGET in df.columns else pd.Series(np.nan, index=df.index)

    tuner = None
    if tune_inside_folds:
        def tuner(X_train: pd.DataFrame, y_train: pd.Series) -> dict:   # noqa: E306
            return tune(X_train, y_train, horizon=HORIZON_SESSIONS,
                        n_trials=tune_trials)

    return walk_forward(
        X=X, y=y, dates=df["date"].tolist(),
        model_factory=_model_factory(),
        # min_train=500 (~2 years) so the first fold is not fitted on a
        # window shorter than the feature lookbacks it depends on.
        splitter=PurgedWalkForward(n_folds=n_folds, horizon=HORIZON_SESSIONS,
                                   embargo=HORIZON_SESSIONS, min_train=500),
        ticker=ticker,
        tuner=tuner,
    )


# ── Persistence: weekly writes, daily reads ────────────────────────────────────


def _persist_evaluation(ticker: str, payload: dict) -> None:
    """Upserts the weekly evaluation + conformal calibration for one ticker."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO model_metadata (
                ticker, eval_rank_ic, eval_rank_ic_t, eval_hit_rate,
                eval_baseline_hit_rate, eval_mae, eval_mae_naive,
                eval_n_oos, eval_n_effective, eval_protocol,
                conformal_quantile, conformal_coverage, conformal_n,
                conformal_residuals, model_version, evaluated_at,
                ensemble_in_use
            ) VALUES (
                :ticker, :ic, :ic_t, :hit, :baseline, :mae, :mae_naive,
                :n_oos, :n_eff, :protocol,
                :cq, :cc, :cn, :cr, :version, :evaluated_at, 0
            )
            ON CONFLICT (ticker) DO UPDATE SET
                eval_rank_ic           = EXCLUDED.eval_rank_ic,
                eval_rank_ic_t         = EXCLUDED.eval_rank_ic_t,
                eval_hit_rate          = EXCLUDED.eval_hit_rate,
                eval_baseline_hit_rate = EXCLUDED.eval_baseline_hit_rate,
                eval_mae               = EXCLUDED.eval_mae,
                eval_mae_naive         = EXCLUDED.eval_mae_naive,
                eval_n_oos             = EXCLUDED.eval_n_oos,
                eval_n_effective       = EXCLUDED.eval_n_effective,
                eval_protocol          = EXCLUDED.eval_protocol,
                conformal_quantile     = EXCLUDED.conformal_quantile,
                conformal_coverage     = EXCLUDED.conformal_coverage,
                conformal_n            = EXCLUDED.conformal_n,
                conformal_residuals    = EXCLUDED.conformal_residuals,
                model_version          = EXCLUDED.model_version,
                evaluated_at           = EXCLUDED.evaluated_at
        """), {
            "ticker": ticker,
            "ic": payload.get("rank_ic"),
            "ic_t": payload.get("rank_ic_t"),
            "hit": payload.get("hit_rate"),
            "baseline": payload.get("majority_hit_rate"),
            "mae": payload.get("mae"),
            "mae_naive": payload.get("mae_naive_zero"),
            "n_oos": payload.get("n_oos_predictions"),
            "n_eff": payload.get("n_effective"),
            "protocol": json.dumps(payload.get("protocol", {})),
            "cq": payload.get("conformal_quantile"),
            "cc": payload.get("conformal_coverage"),
            "cn": payload.get("conformal_n"),
            "cr": json.dumps(payload.get("conformal_residuals"))
                  if payload.get("conformal_residuals") is not None else None,
            "version": MODEL_VERSION,
            "evaluated_at": payload.get("evaluated_at"),
        })
        conn.commit()


def _load_persisted_evaluation(ticker: str) -> dict | None:
    """
    Reads the last weekly evaluation for a ticker. Returns None if it has
    never been evaluated (e.g. newly added to the universe) — the daily path
    then correctly reports INSUFFICIENT evidence rather than fabricating a
    grade (agents.critic_agent.grade_evidence treats missing metrics as
    INSUFFICIENT).
    """
    engine = get_engine()
    row = pd.read_sql(
        text("SELECT * FROM model_metadata WHERE ticker = :t"),
        engine, params={"t": ticker},
    )
    if row.empty or pd.isna(row.iloc[0].get("eval_rank_ic")):
        return None

    r = row.iloc[0]

    residuals = None
    if r.get("conformal_residuals"):
        try:
            residuals = np.array(json.loads(r["conformal_residuals"]))
        except (json.JSONDecodeError, TypeError):
            residuals = None

    protocol = {}
    if r.get("eval_protocol"):
        try:
            protocol = json.loads(r["eval_protocol"])
        except (json.JSONDecodeError, TypeError):
            protocol = {}

    return {
        "rank_ic": r.get("eval_rank_ic"),
        "rank_ic_t": r.get("eval_rank_ic_t"),
        "hit_rate": r.get("eval_hit_rate"),
        "majority_hit_rate": r.get("eval_baseline_hit_rate"),
        "mae": r.get("eval_mae"),
        "mae_naive_zero": r.get("eval_mae_naive"),
        "beats_naive": (bool(r["eval_mae"] < r["eval_mae_naive"])
                        if pd.notna(r.get("eval_mae")) and pd.notna(r.get("eval_mae_naive"))
                        else None),
        "n_oos_predictions": r.get("eval_n_oos"),
        "n_effective": r.get("eval_n_effective"),
        "protocol": protocol,
        "evaluated_at": str(r["evaluated_at"]) if pd.notna(r.get("evaluated_at")) else None,
        "conformal_quantile": r.get("conformal_quantile"),
        "conformal_coverage": r.get("conformal_coverage"),
        "conformal_n": r.get("conformal_n"),
        "conformal_residuals": residuals,
    }


def _reconstruct_calibration(persisted: dict) -> ConformalCalibration | None:
    """
    Rebuilds a ConformalCalibration from persisted fields (no recomputation).

    Coerces residuals to an ndarray regardless of what shape they arrive in —
    a plain Python list (e.g. straight from a dict payload, before any JSON
    round-trip) fails inside ConformalCalibration.prob_positive(), which does
    elementwise comparison against the residuals array.
    """
    quantile = persisted.get("conformal_quantile")
    coverage = persisted.get("conformal_coverage")
    residuals = persisted.get("conformal_residuals")
    n = persisted.get("conformal_n")

    if quantile is None or coverage is None or residuals is None or n is None:
        return None

    return ConformalCalibration(
        quantile=float(quantile), coverage=float(coverage),
        residuals=np.asarray(residuals, dtype=float), n=int(n),
    )


# ── Weekly path: evaluate, retune, calibrate, persist ─────────────────────────


def evaluate_and_persist_ticker(ticker: str, df: pd.DataFrame | None = None) -> dict | None:
    """
    The expensive weekly job for one ticker.

    Runs purged walk-forward evaluation, refreshes the cached production
    hyperparameters (force=True), calibrates conformal intervals on the
    resulting out-of-sample residuals, and persists all of it. Does NOT fit
    or save a production joblib model — the daily job does that with that
    day's fresher data, using the hyperparameters this call just refreshed.
    """
    df = load_features_for_ticker(ticker) if df is None else df
    if df.empty or len(df) < 350:
        print(f"[Model] {ticker}: insufficient history for weekly evaluation")
        return None

    result = evaluate_ticker(ticker, df)
    if result.n_predictions == 0:
        print(f"[Model] {ticker}: no out-of-sample predictions")
        return None

    calibration = fit_conformal(
        result.predictions["y_true"].to_numpy(),
        result.predictions["y_pred"].to_numpy(),
        coverage=0.80,
    )

    labelled = df[df[TARGET].notna()]
    if len(labelled) >= 200:
        tune_and_cache(ticker, labelled[FEATURES], labelled[TARGET],
                       horizon=HORIZON_SESSIONS, force=True)
    else:
        print(f"[Model] {ticker}: too few labelled rows to retune production params")

    m = result.metrics
    payload = {
        "rank_ic": m.get("rank_ic"),
        "rank_ic_t": m.get("rank_ic_t"),
        "hit_rate": m.get("hit_rate"),
        "majority_hit_rate": m.get("majority_hit_rate"),
        "mae": m.get("mae"),
        "mae_naive_zero": result.baselines.get("zero", {}).get("mae"),
        "n_oos_predictions": result.n_predictions,
        "n_effective": m.get("n_effective"),
        "protocol": result.protocol,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if calibration is not None:
        payload.update({
            "conformal_quantile": calibration.quantile,
            "conformal_coverage": calibration.coverage,
            "conformal_n": calibration.n,
            "conformal_residuals": calibration.residuals.tolist(),
        })

    _persist_evaluation(ticker, payload)

    ic = payload.get("rank_ic")
    hit = payload.get("hit_rate")
    base = payload.get("majority_hit_rate")
    print(f"[Model] {ticker}: evaluated — IC={ic if ic is None else f'{ic:+.3f}'} "
          f"hit={hit if hit is None else f'{hit:.1f}%'} "
          f"(baseline {base if base is None else f'{base:.1f}%'})")

    return payload


# ── Production fit (used by both paths) ────────────────────────────────────────


def fit_production_model(ticker: str, df: pd.DataFrame, force_tune: bool = False):
    """
    Fits on all labelled rows to generate the next forecast.

    With ``force_tune=False`` (the daily path), this loads already-cached
    hyperparameters and performs exactly ONE XGBoost fit — no search. Returns
    (model, n_train_rows). Produces NO metrics: a model fitted on everything
    has no held-out data left to be scored on, and pretending otherwise is
    what audit finding F1 was.
    """
    labelled = df[df[TARGET].notna()]
    if len(labelled) < 200:
        return None, 0

    X, y = labelled[FEATURES], labelled[TARGET]
    params = tune_and_cache(ticker, X, y, horizon=HORIZON_SESSIONS, force=force_tune)

    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(X, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES, "version": MODEL_VERSION,
         "trained_at": datetime.now(timezone.utc).isoformat(), "n_train": len(labelled)},
        os.path.join(MODELS_DIR, f"{ticker}.joblib"),
    )
    return model, len(labelled)


# ── Daily path: fit fresh, predict, read persisted evidence ───────────────────


def forecast_ticker_daily(ticker: str) -> TickerForecast | None:
    """
    The cheap daily job for one ticker. No Optuna search runs here.

    Fits the production model with cached hyperparameters (one fit),
    predicts the latest row, and attaches whatever evaluation + calibration
    the last WEEKLY run persisted — which may be up to a week old. That
    staleness is reported via ``evaluation["evaluated_at"]``, not hidden.
    """
    df = load_features_for_ticker(ticker)
    if df.empty or len(df) < 350:
        print(f"[Model] {ticker}: insufficient history ({len(df)} rows)")
        return None

    model, n_train = fit_production_model(ticker, df, force_tune=False)
    if model is None:
        return None

    latest = _latest_row(df)
    pred_excess = float(model.predict(latest[FEATURES])[0])
    current_price = float(latest["close"].iloc[0])
    forecast_date = str(latest["date"].iloc[0])

    persisted = _load_persisted_evaluation(ticker)
    calibration = _reconstruct_calibration(persisted) if persisted else None
    price_view = to_price_view(current_price, pred_excess, calibration)

    if persisted:
        evaluation = {
            "rank_ic": persisted.get("rank_ic"),
            "rank_ic_t": persisted.get("rank_ic_t"),
            "hit_rate": persisted.get("hit_rate"),
            "majority_hit_rate": persisted.get("majority_hit_rate"),
            "mae": persisted.get("mae"),
            "mae_naive_zero": persisted.get("mae_naive_zero"),
            "beats_naive": persisted.get("beats_naive"),
            "n_oos_predictions": persisted.get("n_oos_predictions"),
            "n_effective": persisted.get("n_effective"),
            "protocol": persisted.get("protocol", {}),
            "evaluated_at": persisted.get("evaluated_at"),
        }
    else:
        evaluation = {"evaluated_at": None}

    bench_ticker = str(latest.get("benchmark_ticker", pd.Series([""])).iloc[0] or "")
    bench_specific = bool(latest.get("benchmark_sector_specific", pd.Series([0])).iloc[0])

    return TickerForecast(
        ticker=ticker,
        forecast_date=forecast_date,
        current_price=current_price,
        price_view=price_view,
        evaluation=evaluation,
        benchmark_ticker=bench_ticker,
        benchmark_sector_specific=bench_specific,
        n_train_rows=n_train,
    )


def forecast_ticker_full(ticker: str) -> TickerForecast | None:
    """
    Convenience wrapper for manual/one-off use: runs the weekly evaluation
    and the daily forecast back to back for a single ticker.

    NOT what the scheduled jobs call, and NOT wired into the LangGraph agent
    or any admin HTTP route — those must stay on the cheap daily path only.
    This exists for local debugging (e.g. `python -c "from pipeline.model
    import forecast_ticker_full; forecast_ticker_full('RELIANCE.NS')"`) or a
    newly-added ticker that has no persisted evaluation yet.
    """
    df = load_features_for_ticker(ticker)
    if df.empty or len(df) < 350:
        return None
    evaluate_and_persist_ticker(ticker, df)
    return forecast_ticker_daily(ticker)


# ── Batch drivers ──────────────────────────────────────────────────────────────


def train_and_forecast(single_ticker: str | None = None,
                       tickers: list[str] | None = None) -> dict[str, TickerForecast]:
    """
    The DAILY batch driver: forecasts every ticker using cached hyperparameters.

    Runs no Optuna search. Updates only ``last_trained`` in model_metadata —
    the eval_*/conformal_*/evaluated_at columns are weekly-owned and must not
    be touched here, or the staleness signal they provide becomes meaningless.
    """
    if single_ticker:
        to_process = [single_ticker]
    elif tickers:
        to_process = list(tickers)
    else:
        from data.universe import get_universe
        to_process = get_universe()

    engine = get_engine()
    results: dict[str, TickerForecast] = {}

    for ticker in to_process:
        try:
            forecast = forecast_ticker_daily(ticker)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[Model] {ticker}: failed — {exc}")
            continue

        if forecast is None:
            continue

        results[ticker] = forecast
        ev = forecast.evaluation
        ic = ev.get("rank_ic")
        hit = ev.get("hit_rate")
        base = ev.get("majority_hit_rate")
        print(f"[Model] {ticker}: excess={forecast.price_view['pred_excess_return']:+.4f} "
              f"IC={'n/a' if ic is None else f'{ic:+.3f}'} "
              f"hit={'n/a' if hit is None else f'{hit:.1f}%'} "
              f"(evaluated_at={ev.get('evaluated_at') or 'never'})")

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO model_metadata (ticker, model_version, last_trained, ensemble_in_use)
                VALUES (:ticker, :version, :trained, 0)
                ON CONFLICT (ticker) DO UPDATE SET
                    model_version = EXCLUDED.model_version,
                    last_trained  = EXCLUDED.last_trained
            """), {
                "ticker": ticker,
                "version": forecast.model_version,
                "trained": datetime.now(timezone.utc),
            })
            conn.commit()

    return results


def evaluate_and_persist_universe(tickers: list[str] | None = None) -> dict[str, dict]:
    """
    The WEEKLY batch driver: runs the expensive evaluation for every ticker
    and persists the results for the daily job to read all week.
    """
    if tickers is None:
        from data.universe import get_universe
        tickers = get_universe()

    results: dict[str, dict] = {}
    for ticker in tickers:
        try:
            payload = evaluate_and_persist_ticker(ticker)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[Model] {ticker}: weekly evaluation failed — {exc}")
            continue
        if payload is not None:
            results[ticker] = payload

    return results


def evaluate_universe(tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Research/reporting tool: runs purged walk-forward across the universe and
    returns a per-ticker table plus the pooled out-of-sample panel, for the
    cross-sectional report. Not part of the scheduled daily/weekly split —
    this is for `tools/report_performance.py`-style manual analysis.
    """
    if tickers is None:
        from data.universe import get_universe
        tickers = get_universe()

    rows, panels = [], []
    for ticker in tickers:
        result = evaluate_ticker(ticker, tune_inside_folds=False)
        if result.n_predictions == 0:
            continue
        print(format_report(result))

        m, zero = result.metrics, result.baselines.get("zero", {})
        rows.append({
            "ticker": ticker,
            "n_oos": result.n_predictions,
            "rank_ic": m.get("rank_ic"),
            "rank_ic_t": m.get("rank_ic_t"),
            "hit_rate": m.get("hit_rate"),
            "majority_hit_rate": m.get("majority_hit_rate"),
            "mae": m.get("mae"),
            "mae_naive": zero.get("mae"),
            "beats_naive": m.get("beats_naive_mae"),
        })

        panel = result.predictions.copy()
        panel["ticker"] = ticker
        panels.append(panel)

    table = pd.DataFrame(rows)
    if panels:
        table.attrs["panel"] = pd.concat(panels, ignore_index=True)
    return table
