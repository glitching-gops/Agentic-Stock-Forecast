"""
pipeline/lstm_model.py

Single-layer LSTM model for 30-day forward price prediction.
Trained per stock using a 30-day input sequence of 10 price
and momentum features. Uses GPU acceleration via CUDA.

Architecture:
  - Input:   (batch, seq_len=30, n_features=10)
  - LSTM:    hidden_size=128, num_layers=1, dropout=0.2
  - Linear:  128 → 1
  - Output:  scalar price prediction

Training:
  - 100 epochs with early stopping (patience=10)
  - Adam optimiser, lr=1e-3
  - MSE loss
  - Train/val split: 80/20 on time-ordered data
  - Saved as {ticker}_lstm.pt in models/lstm/

Features used (10 price and momentum signals):
  close, rsi, macd_hist, obv, ema_21,
  sector_rel_5d, sentiment_score,
  lag1_ret, lag5_ret, hurst
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN    = 30
HIDDEN     = 128
DROPOUT    = 0.2
EPOCHS     = 100
PATIENCE   = 10
LR         = 1e-3
BATCH_SIZE = 32

LSTM_FEATURES = [
    "close", "rsi", "macd_hist", "obv", "ema_21",
    "sector_rel_5d", "sentiment_score",
    "lag1_ret", "lag5_ret", "hurst"
]

MODELS_DIR  = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "models", "lstm"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model definition ──────────────────────────────────────────────────────────
class StockLSTM(nn.Module):
    """
    Single-layer LSTM with a linear output head.
    Predicts a single scalar: the 30-day forward closing price.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = HIDDEN,
        dropout: float = DROPOUT
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # dropout on single layer has no effect — applied manually
        )
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, 1) predicted price
        """
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])  # take last timestep output
        return self.linear(out)


# ── Sequence builder ──────────────────────────────────────────────────────────
def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a flat dataframe into overlapping sequences for LSTM input.

    For each row i (where i >= seq_len), the input sequence is
    df[feature_cols].iloc[i-seq_len:i] and the target is
    df[target_col].iloc[i].

    Returns:
        X: (n_samples, seq_len, n_features)
        y: (n_samples,)
    """
    data    = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)

    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


# ── Training ──────────────────────────────────────────────────────────────────
def train_lstm(
    ticker: str,
    df: pd.DataFrame,
    force: bool = False
) -> dict:
    """
    Trains a StockLSTM for a single ticker and saves the model and scaler.

    If a saved model exists and force=False, loads and returns it without
    retraining. Set force=True for weekly scheduled retraining.

    Args:
        ticker: NSE ticker string e.g. 'RELIANCE.NS'
        df:     DataFrame with LSTM_FEATURES + 'target' columns,
                sorted ascending by date, NaN-free
        force:  If True, always retrain from scratch

    Returns:
        dict with keys: val_loss, val_mape, epochs_trained, device
    """
    model_path  = get_lstm_path(ticker)
    scaler_path = get_scaler_path(ticker)

    if not force and os.path.exists(model_path):
        print(f"[LSTM] {ticker}: model exists, skipping retrain")
        return load_lstm_result(ticker, df)

    # ── Validate features ────────────────────────────────────────────────────
    missing = [f for f in LSTM_FEATURES if f not in df.columns]
    if missing:
        print(f"[LSTM] {ticker}: missing features {missing} — filling with 0")
        for f in missing:
            df[f] = 0.0

    df = df.copy()
    for col in LSTM_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(df) < SEQ_LEN + 30:
        raise ValueError(
            f"[LSTM] {ticker}: insufficient rows ({len(df)}) "
            f"for seq_len={SEQ_LEN}"
        )

    # ── Scale features ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    df[LSTM_FEATURES] = scaler.fit_transform(df[LSTM_FEATURES])
    joblib.dump(scaler, scaler_path)

    # ── Build sequences ──────────────────────────────────────────────────────
    X, y = build_sequences(df, LSTM_FEATURES)

    # 80/20 time-ordered split
    split   = int(len(X) * 0.8)
    X_train = torch.tensor(X[:split],  dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y[:split],  dtype=torch.float32).to(DEVICE)
    X_val   = torch.tensor(X[split:],  dtype=torch.float32).to(DEVICE)
    y_val   = torch.tensor(y[split:],  dtype=torch.float32).to(DEVICE)

    train_ds     = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Model, optimiser, loss ───────────────────────────────────────────────
    model     = StockLSTM(n_features=len(LSTM_FEATURES)).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # ── Training loop with early stopping ───────────────────────────────────
    best_val_loss   = float("inf")
    patience_count  = 0
    epochs_trained  = 0

    for epoch in range(EPOCHS):
        # Training pass
        model.train()
        for X_batch, y_batch in train_loader:
            optimiser.zero_grad()
            preds = model(X_batch).squeeze()
            loss  = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).squeeze().cpu().numpy()
            val_true  = y_val.cpu().numpy()
            val_loss  = float(criterion(
                torch.tensor(val_preds),
                torch.tensor(val_true)
            ))

        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(
                    f"[LSTM] {ticker}: early stop at epoch {epoch + 1} "
                    f"(best val_loss={best_val_loss:.4f})"
                )
                break

    # ── Compute validation MAPE ──────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        final_preds = model(X_val).squeeze().cpu().numpy()

    val_mape = float(np.mean(
        np.abs((val_true - final_preds) / (np.abs(val_true) + 1e-8))
    ) * 100)

    print(
        f"[LSTM] {ticker}: trained {epochs_trained} epochs | "
        f"val_loss={best_val_loss:.4f} | val_MAPE={val_mape:.2f}% | "
        f"device={DEVICE}"
    )

    return {
        "val_loss":      best_val_loss,
        "val_mape":      val_mape,
        "epochs_trained": epochs_trained,
        "device":        str(DEVICE),
    }


# ── Inference ─────────────────────────────────────────────────────────────────
_model_cache = {}
_scaler_cache = {}

def predict_lstm(ticker: str, df: pd.DataFrame) -> float | None:
    """
    Generates a 30-day forward price prediction using the saved LSTM model.

    Takes the most recent SEQ_LEN rows from df as the input sequence.
    Returns the predicted price as a float, or None if the model is
    not found or inference fails.

    Args:
        ticker: NSE ticker string
        df:     DataFrame with LSTM_FEATURES columns, sorted ascending,
                must have at least SEQ_LEN rows

    Returns:
        Predicted price (float) or None
    """
    model_path  = get_lstm_path(ticker)
    scaler_path = get_scaler_path(ticker)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[LSTM] {ticker}: no saved model found — skipping prediction")
        return None

    try:
        if ticker not in _scaler_cache:
            _scaler_cache[ticker] = joblib.load(scaler_path)
        scaler = _scaler_cache[ticker]
        
        # Fast numpy path
        data = df[LSTM_FEATURES].values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        data = scaler.transform(data)

        if len(data) < SEQ_LEN:
            print(f"[LSTM] {ticker}: only {len(data)} rows, need {SEQ_LEN}")
            return None

        sequence = data[-SEQ_LEN:].astype(np.float32)
        X        = torch.tensor(sequence).unsqueeze(0).to(DEVICE)

        if ticker not in _model_cache:
            model = StockLSTM(n_features=len(LSTM_FEATURES)).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            _model_cache[ticker] = model
        
        model = _model_cache[ticker]

        with torch.no_grad():
            pred = model(X).squeeze().cpu().item()

        return float(pred)

    except Exception as e:
        print(f"[LSTM] {ticker}: inference failed — {e}")
        return None


# ── Path helpers ──────────────────────────────────────────────────────────────
def get_lstm_path(ticker: str) -> str:
    """Returns the path to the saved LSTM model .pt file for a ticker."""
    safe = ticker.replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}_lstm.pt")


def get_scaler_path(ticker: str) -> str:
    """Returns the path to the saved StandardScaler joblib for a ticker."""
    safe = ticker.replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}_scaler.joblib")


def load_lstm_result(ticker: str, df: pd.DataFrame) -> dict:
    """
    Returns a minimal result dict for an already-trained LSTM model
    without retraining. Used when force=False and model already exists.
    """
    return {
        "val_loss":       None,
        "val_mape":       None,
        "epochs_trained": 0,
        "device":         str(DEVICE),
    }
