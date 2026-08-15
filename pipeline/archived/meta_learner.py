"""
pipeline/meta_learner.py

Ridge regression meta-learner that combines XGBoost and LSTM predictions
into a final ensemble forecast.

The meta-learner is trained on out-of-fold validation predictions from
both models. It receives three inputs per sample:
  1. XGBoost predicted price (from walk-forward validation)
  2. LSTM predicted price (from validation set)
  3. Hurst exponent (regime signal — helps the meta-learner weight
     LSTM higher in trending regimes and XGBoost in mean-reverting ones)

The output is a single ensemble price prediction.

A 25% price change cap is applied to the final output relative to the
current price to prevent implausible forecasts from reaching the dashboard.

Saved as {ticker}_meta.joblib in models/meta/.
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge

MODELS_DIR  = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "models", "meta"
)
MAX_CHANGE_PCT = 0.25   # 25% cap on forecast change from current price


def get_meta_path(ticker: str) -> str:
    """Returns the path to the saved meta-learner joblib for a ticker."""
    safe = ticker.replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}_meta.joblib")


def train_meta_learner(
    ticker: str,
    xgb_val_preds: np.ndarray,
    lstm_val_preds: np.ndarray,
    hurst_val: np.ndarray,
    y_val: np.ndarray,
    force: bool = False
) -> Ridge:
    """
    Trains a Ridge regression meta-learner on validation predictions
    from XGBoost and LSTM models.

    Args:
        ticker:         NSE ticker string
        xgb_val_preds:  XGBoost predictions on validation set (n_samples,)
        lstm_val_preds: LSTM predictions on validation set (n_samples,)
        hurst_val:      Hurst exponent values for validation rows (n_samples,)
        y_val:          True target prices for validation set (n_samples,)
        force:          If True, retrain even if saved model exists

    Returns:
        Trained Ridge model
    """
    meta_path = get_meta_path(ticker)

    if not force and os.path.exists(meta_path):
        return joblib.load(meta_path)

    # Align lengths — LSTM may have fewer samples due to sequence offset
    min_len = min(
        len(xgb_val_preds),
        len(lstm_val_preds),
        len(hurst_val),
        len(y_val)
    )

    if min_len < 10:
        # Not enough validation samples to train a meta-learner
        # Fall back to equal weighting
        print(
            f"[Meta] {ticker}: only {min_len} validation samples — "
            f"using equal weight fallback"
        )
        return None

    X_meta = np.column_stack([
        xgb_val_preds[-min_len:],
        lstm_val_preds[-min_len:],
        hurst_val[-min_len:],
    ])
    y_meta = y_val[-min_len:]

    meta = Ridge(alpha=1.0)
    meta.fit(X_meta, y_meta)

    joblib.dump(meta, meta_path)
    print(
        f"[Meta] {ticker}: trained | "
        f"XGB coef={meta.coef_[0]:.3f} | "
        f"LSTM coef={meta.coef_[1]:.3f} | "
        f"Hurst coef={meta.coef_[2]:.3f}"
    )

    return meta


def predict_ensemble(
    ticker: str,
    xgb_price: float,
    lstm_price: float | None,
    current_hurst: float,
    current_price: float,
) -> float:
    """
    Generates the final ensemble price prediction.

    If LSTM prediction is unavailable, falls back to XGBoost alone.
    If meta-learner is not found, uses equal weighting of XGBoost and LSTM.
    Applies a 25% price change cap relative to current price.

    Args:
        ticker:        NSE ticker string
        xgb_price:     XGBoost predicted price
        lstm_price:    LSTM predicted price, or None if unavailable
        current_hurst: Most recent Hurst exponent value for this stock
        current_price: Current closing price (used for cap)

    Returns:
        Final ensemble price prediction (float)
    """
    # If LSTM is unavailable, return XGBoost prediction directly
    if lstm_price is None:
        return _apply_cap(xgb_price, current_price)

    meta_path = get_meta_path(ticker)

    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        X    = np.array([[xgb_price, lstm_price, current_hurst]])
        pred = float(meta.predict(X)[0])
    else:
        # Equal weight fallback if meta-learner not trained yet
        pred = (xgb_price + lstm_price) / 2.0

    return _apply_cap(pred, current_price)


def _apply_cap(price: float, current_price: float) -> float:
    """
    Clips the forecast price to within MAX_CHANGE_PCT of the current price.
    Prevents implausible forecasts from reaching the dashboard.

    Example: current=1000, cap=25% → forecast clipped to [750, 1250]
    """
    if current_price <= 0:
        return price

    upper = current_price * (1 + MAX_CHANGE_PCT)
    lower = current_price * (1 - MAX_CHANGE_PCT)
    return float(np.clip(price, lower, upper))
