"""
pipeline/model.py — Training, honest evaluation, and forecast generation.

The rewrite removes the metric path that produced the project's headline
numbers. Previously:

  - a Ridge meta-learner was fitted on the validation set and scored on that
    same validation set, and its output overwrote the XGBoost metrics (F1);
  - Optuna saw the test slice before it was reported as held out (F2);
  - folds were contiguous, so 30-session labels straddled the split (F3);
  - the final production model was fitted on all data, then a *different*
    model's accuracy was reported next to its forecast.

Now there are two clearly separated paths:

  EVALUATION   ``evaluate_ticker`` runs purged walk-forward with tuning nested
               inside each training fold. Everything reported comes from here,
               always beside a baseline.

  PRODUCTION   ``fit_production_model`` fits on all labelled data to generate
               tomorrow's forecast. It produces NO metrics. A forecast carries
               the evaluation metrics measured on held-out folds, which is the
               only honest thing to attach to it.
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
from pipeline.conformal import fit_conformal, to_price_view
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
    # macro.py but never referenced by the old FEATURES list; they are wired in
    # here rather than left as dead columns (audit finding F15).
    "usdinr", "india_vix", "nifty_5d_return", "nifty_20d_return",
    "fii_net_flow", "dii_net_flow",
]

TARGET = "target_excess_return"

MODELS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models", "joblib"
)


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


# ── Evaluation path ───────────────────────────────────────────────────────────


def evaluate_ticker(
    ticker: str,
    df: pd.DataFrame | None = None,
    n_folds: int = 6,
    tune_inside_folds: bool = True,
    tune_trials: int = 15,
) -> WalkForwardResult:
    """
    Purged walk-forward evaluation. This is the ONLY source of reported metrics.

    Hyperparameter search runs inside each training fold via the ``tuner``
    callback, so no configuration is ever chosen with sight of the rows it is
    later scored on.

    ``tune_trials`` is the per-fold Optuna budget. Cost is
    ``n_folds x tune_trials x inner_folds`` model fits per ticker, so a
    universe-wide evaluation is expensive; lower it for breadth and record the
    value alongside the result. What matters for validity is that tuning is
    NESTED, not that it is exhaustive — and a smaller search is the conservative
    direction, since fewer configurations tried means a lower bar for the result
    to clear (see ``evaluation.deflated_sharpe_note``).
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


# ── Production path ───────────────────────────────────────────────────────────


def fit_production_model(ticker: str, df: pd.DataFrame, force_tune: bool = False):
    """
    Fits on all labelled rows to generate the next forecast.

    Returns (model, n_train_rows). Produces NO metrics: a model fitted on
    everything has no held-out data left to be scored on, and pretending
    otherwise is precisely what F1 did.
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


def forecast_ticker(ticker: str, force_tune: bool = False) -> TickerForecast | None:
    """
    Produces one forecast with its held-out evidence attached.

    Sequence: evaluate on purged folds, calibrate conformal intervals on those
    out-of-sample residuals, then fit the production model and predict the most
    recent unlabelled row.
    """
    df = load_features_for_ticker(ticker)
    if df.empty or len(df) < 350:
        print(f"[Model] {ticker}: insufficient history ({len(df)} rows)")
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

    model, n_train = fit_production_model(ticker, df, force_tune=force_tune)
    if model is None:
        return None

    # Predict the most recent row whose label is not yet knowable.
    unlabelled = df[df[TARGET].isna()]
    latest = unlabelled.iloc[[-1]] if not unlabelled.empty else df.iloc[[-1]]

    pred_excess = float(model.predict(latest[FEATURES])[0])
    current_price = float(latest["close"].iloc[0])
    forecast_date = str(latest["date"].iloc[0])

    price_view = to_price_view(current_price, pred_excess, calibration)

    m = result.metrics
    evaluation = {
        "rank_ic": m.get("rank_ic"),
        "rank_ic_t": m.get("rank_ic_t"),
        "hit_rate": m.get("hit_rate"),
        "majority_hit_rate": m.get("majority_hit_rate"),
        "n_effective": m.get("n_effective"),
        "mae": m.get("mae"),
        "mae_naive_zero": result.baselines.get("zero", {}).get("mae"),
        "beats_naive": m.get("beats_naive_mae", False),
        "n_oos_predictions": result.n_predictions,
        "n_folds": result.n_folds_run,
        "protocol": result.protocol,
    }

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


# ── Batch driver ──────────────────────────────────────────────────────────────


def train_and_forecast(single_ticker: str | None = None,
                       tickers: list[str] | None = None,
                       force_tune: bool = False) -> dict[str, TickerForecast]:
    """Generates forecasts for a set of tickers and records model metadata."""
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
            forecast = forecast_ticker(ticker, force_tune=force_tune)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[Model] {ticker}: failed — {exc}")
            continue

        if forecast is None:
            continue

        results[ticker] = forecast
        ev = forecast.evaluation
        print(f"[Model] {ticker}: excess={forecast.price_view['pred_excess_return']:+.4f} "
              f"IC={ev.get('rank_ic', float('nan')):+.3f} "
              f"hit={ev.get('hit_rate', float('nan')):.1f}% "
              f"(majority {ev.get('majority_hit_rate', float('nan')):.1f}%)")

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO model_metadata (
                    ticker, eval_rank_ic, eval_rank_ic_t, eval_hit_rate,
                    eval_baseline_hit_rate, eval_mae, eval_mae_naive,
                    eval_n_oos, model_version, eval_protocol,
                    ensemble_in_use, last_trained
                ) VALUES (
                    :ticker, :ic, :ic_t, :hit, :baseline, :mae, :mae_naive,
                    :n_oos, :version, :protocol, 0, :trained
                )
                ON CONFLICT (ticker) DO UPDATE SET
                    eval_rank_ic           = EXCLUDED.eval_rank_ic,
                    eval_rank_ic_t         = EXCLUDED.eval_rank_ic_t,
                    eval_hit_rate          = EXCLUDED.eval_hit_rate,
                    eval_baseline_hit_rate = EXCLUDED.eval_baseline_hit_rate,
                    eval_mae               = EXCLUDED.eval_mae,
                    eval_mae_naive         = EXCLUDED.eval_mae_naive,
                    eval_n_oos             = EXCLUDED.eval_n_oos,
                    model_version          = EXCLUDED.model_version,
                    eval_protocol          = EXCLUDED.eval_protocol,
                    ensemble_in_use        = 0,
                    last_trained           = EXCLUDED.last_trained
            """), {
                "ticker":   ticker,
                "ic":       ev.get("rank_ic"),
                "ic_t":     ev.get("rank_ic_t"),
                "hit":      ev.get("hit_rate"),
                "baseline": ev.get("majority_hit_rate"),
                "mae":      ev.get("mae"),
                "mae_naive": ev.get("mae_naive_zero"),
                "n_oos":    ev.get("n_oos_predictions"),
                "version":  forecast.model_version,
                "protocol": json.dumps(ev.get("protocol", {})),
                "trained":  datetime.now(timezone.utc),
            })
            conn.commit()

    return results


def evaluate_universe(tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Runs purged walk-forward across the universe and returns a per-ticker table
    plus the pooled out-of-sample panel, for the cross-sectional report.
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
