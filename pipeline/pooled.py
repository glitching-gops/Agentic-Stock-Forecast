"""
pipeline/pooled.py — the pooled, standardised-label model as a production path.

WHAT THIS IS FOR, STATED THE WAY THE DASHBOARD MUST STATE IT
------------------------------------------------------------
A CORRECTNESS UPGRADE, NOT AN ACCURACY CLAIM. Every panel test on this universe
is null, and this model is not expected to change that: its cross-sectional
rank IC has sat at +0.01 with a Driscoll-Kraay t below 1 in every measurement
since Stage 2b. The reason to carry it forward is what the live per-ticker
model does instead of ranking — it emits ONE constant prediction in 316 of 420
(ticker, fold) cells (Stage 0 addendum), so on most dates it holds no ordering
of the 84 names at all. This model predicts a distinct value for every name on
every date (0 of 420 constant cells, Hygiene 2026-09-21). It is a model that
can FAIL a ranking test, which the per-ticker one mostly cannot.

It runs in SHADOW (pipeline/pooled_shadow.py): real forecasts, intervals and
grades, written to separate tables no public endpoint reads, while the
per-ticker model keeps serving the site. The cutover is a later session.

THE MODEL, AND WHERE EACH CHOICE WAS MEASURED
---------------------------------------------
  * ONE XGBoost fitted on the whole panel — Stage 2b: pooling, not the loss,
    removes the per-ticker degeneracy.
  * The 15 scale-free FACTORS, cross-sectionally z-scored within each date,
    with NO ticker feature — Stage 0c: every STRONG the ticker feature produced
    was the persistent-identity landmine, and it is worth nothing
    cross-sectionally (Stage 2b).
  * MAE objective — Stage 2a/2b: a rank-IC objective overfits the selection
    and buys nothing.
  * The within-date STANDARDISED label (pipeline/label.py) — P6/Hygiene: it
    ends the tuner's gamma degeneracy and makes predictions fine-grained.
  * Nested Optuna search inside each purged fold, XGB_THREADS pinned.
  * MISSING STAYS MISSING. The NULLABLE features (sector_rel_* for thin
    sectors, earnings_surprise before vendor coverage, hurst warm-up) reach
    XGBoost as NaN, which it treats as missing with a learned default branch —
    `cross_sectional_zscore(keep_missing=True)`. Never 0.0.

`walk_forward` is the research harness `tools/stage2b_pooled.run_arm(panel,
"mae", "none", features=FACTORS)` restated as production code, and a test
holds the two to bit-identical predictions on the same panel — the stored
baselines were produced by run_arm, and a production path that drifted from
it silently would make every one of them incomparable.

THE PLATFORM. Windows and Linux give different predictions from identical
code and versions (XGBoost's subsampling draws differently under MSVC and GCC;
CLAUDE.md §7). Production trains on a GitHub Actions Ubuntu runner, so Linux
is the reference platform and `environment_hash` travels with every row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.baselines import FACTORS
from pipeline.determinism import xgb_params
from pipeline.evaluation import PurgedPanelWalkForward
from pipeline.label import (CS_MEAN, CS_SD, MOMENT_COLS, attach_causal_moments,
                            inverse_standardise, standardise_target)
from pipeline.model import EVAL_N_FOLDS, EVAL_TUNE_TRIALS
from pipeline.panel import SCALE_FREE, TARGET, cross_sectional_zscore
from pipeline.signals import HORIZON_SESSIONS

#: The shadow model's own identifier, recorded on every shadow row beside the
#: config, data and environment hashes. Bump it whenever anything below
#: changes what the model is — the per-ticker MODEL_VERSION does not cover it.
POOLED_MODEL_VERSION = "pooled-std-mae-noticker-v1"

FEATURES: list[str] = list(FACTORS)
OBJECTIVE = "mae"
#: Dates of history before the first test fold opens — the harness value every
#: pooled result since Stage 2b was measured at.
MIN_TRAIN_DATES = 500
N_FOLDS = EVAL_N_FOLDS
N_TRIALS = EVAL_TUNE_TRIALS
HORIZON = HORIZON_SESSIONS
RANDOM_STATE = 42

#: Raw-label copy kept beside the standardised target, so the inverse can be
#: scored against the quantity the dashboard publishes.
RAW_TARGET = "target_return_raw"


def prepare(panel: pd.DataFrame, keep_missing: bool = True) -> pd.DataFrame:
    """
    The panel as the model sees it: scale-free features z-scored within each
    date, the raw label kept as RAW_TARGET, the label standardised within each
    date with its moments attached.

    Within-date operations only, so applying them once before any split is
    safe: every input was observable on its own date (pipeline/panel.py).
    """
    out = panel.copy()
    out[RAW_TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out = cross_sectional_zscore(out, SCALE_FREE, keep_missing=keep_missing)
    out = standardise_target(out)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _fit(frame: pd.DataFrame, rows: np.ndarray, y: np.ndarray, params: dict):
    from xgboost import XGBRegressor

    model = XGBRegressor(**xgb_params(**params, random_state=RANDOM_STATE,
                                      enable_categorical=False))
    model.fit(frame.iloc[rows][FEATURES], y[rows])
    return model


def walk_forward(prepared: pd.DataFrame, n_trials: int | None = None,
                 n_folds: int | None = None, min_train: int | None = None,
                 horizon: int = HORIZON, verbose: bool = False,
                 fixed_params: dict | None = None
                 ) -> tuple[pd.DataFrame, list[dict]]:
    """
    Purged panel walk-forward, search nested inside each training fold.

    Returns out-of-sample predictions in STANDARDISED units (`y_true` is the
    standardised label; `y_raw` the raw one) with their fold, and per-fold
    diagnostics. Purge and embargo are both `horizon`, the legacy rule every
    pooled result in this project was scored under.
    """
    from pipeline.tuning import tune_pooled

    n_trials = N_TRIALS if n_trials is None else n_trials
    n_folds = N_FOLDS if n_folds is None else n_folds
    min_train = MIN_TRAIN_DATES if min_train is None else min_train
    if not all(c in prepared.columns for c in MOMENT_COLS):
        raise ValueError("walk_forward needs a prepared panel (pooled.prepare)")

    splitter = PurgedPanelWalkForward(n_folds=n_folds, horizon=horizon,
                                      embargo=horizon, min_train=min_train)
    dates = prepared["date"].to_numpy()
    y = pd.to_numeric(prepared[TARGET], errors="coerce").to_numpy(dtype=float)
    raw = prepared[RAW_TARGET].to_numpy(dtype=float) if RAW_TARGET in prepared \
        else np.full(len(prepared), np.nan)

    blocks: list[pd.DataFrame] = []
    folds: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dates)):
        tr = train_idx[np.isfinite(y[train_idx])]
        te = test_idx[np.isfinite(y[test_idx])]
        if len(tr) < 100 or len(te) == 0:
            continue
        t0 = time.time()
        params = fixed_params if fixed_params is not None else tune_pooled(
            prepared.iloc[tr], FEATURES, target=TARGET, horizon=horizon,
            n_trials=n_trials, tuning_objective=OBJECTIVE,
            enable_categorical=False)
        model = _fit(prepared, tr, y, params)
        pred = np.asarray(model.predict(prepared.iloc[te][FEATURES]), dtype=float)
        in_sample = np.asarray(model.predict(prepared.iloc[tr][FEATURES]), dtype=float)
        blocks.append(pd.DataFrame({
            "date": dates[te],
            "ticker": prepared.iloc[te]["ticker"].to_numpy(),
            "y_true": y[te],
            "y_pred": pred,
            "fold": fold,
            "y_raw": raw[te],
        }))
        folds.append({"fold": fold, "gamma": float(params.get("gamma", np.nan)),
                      "max_depth": int(params.get("max_depth", 0)),
                      "train_rows": int(tr.size), "test_rows": int(te.size),
                      "train_mae": float(np.mean(np.abs(y[tr] - in_sample))),
                      "params": dict(params),
                      "secs": round(time.time() - t0, 1)})
        if verbose:
            print(f"    fold {fold}: {tr.size:,} train rows, gamma "
                  f"{params.get('gamma', float('nan')):.3f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    if not blocks:
        raise ValueError("pooled walk-forward produced no fold; the panel is too "
                         "short or its labels are empty")
    return pd.concat(blocks, ignore_index=True), folds


@dataclass
class FittedPooled:
    """The model the daily job forecasts from, and what it was fitted on."""
    booster: bytes
    params: dict
    n_train_rows: int
    train_first_date: str
    train_last_date: str
    features: list[str] = field(default_factory=lambda: list(FEATURES))

    def model(self):
        from xgboost import XGBRegressor

        m = XGBRegressor()
        m.load_model(bytearray(self.booster))
        return m


def fit_final(prepared: pd.DataFrame, n_trials: int | None = None,
              horizon: int = HORIZON, params: dict | None = None) -> FittedPooled:
    """
    The production fit: search on every labelled row (purged inner CV, as in
    each outer fold), then fit on all of them.

    Only rows whose label has REALISED are labelled at all — the last
    `horizon` sessions carry NaN — so nothing here reads the future the
    forecast is about. The fit is never used to REPORT anything; every
    reported metric comes from `walk_forward`, which is F2's rule.
    """
    from pipeline.tuning import tune_pooled

    n_trials = N_TRIALS if n_trials is None else n_trials
    y = pd.to_numeric(prepared[TARGET], errors="coerce").to_numpy(dtype=float)
    rows = np.flatnonzero(np.isfinite(y))
    if rows.size < 1000:
        raise ValueError(f"only {rows.size} labelled rows; refusing to fit")
    params = params if params is not None else tune_pooled(
        prepared.iloc[rows], FEATURES, target=TARGET, horizon=horizon,
        n_trials=n_trials, tuning_objective=OBJECTIVE, enable_categorical=False)
    model = _fit(prepared, rows, y, params)
    raw = model.get_booster().save_raw(raw_format="ubj")
    dates = prepared["date"].to_numpy()[rows]
    return FittedPooled(booster=bytes(raw), params=dict(params),
                        n_train_rows=int(rows.size),
                        train_first_date=str(dates.min()),
                        train_last_date=str(dates.max()))


def predict_on(fitted: FittedPooled, prepared: pd.DataFrame,
               as_of: str | None = None) -> pd.DataFrame:
    """
    Standardised predictions for every ticker on `as_of` (default: the latest
    date in the panel). Reads ONLY that date's feature rows: the within-date
    z-score used nothing but that date's own cross-section.
    """
    as_of = as_of or str(prepared["date"].max())
    rows = prepared[prepared["date"].astype(str) == as_of]
    if rows.empty:
        raise ValueError(f"no rows on {as_of}")
    z = np.asarray(fitted.model().predict(rows[fitted.features]), dtype=float)
    return pd.DataFrame({"date": as_of, "ticker": rows["ticker"].to_numpy(),
                         "pred_z": z,
                         "close": pd.to_numeric(rows["close"], errors="coerce").to_numpy()})


def causal_frame(panel: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Per-date causal moment estimates of the RAW label — the only moments a
    live forecast may be inverted with (pipeline/label.py). Built from the raw
    panel, never from a standardised one: the moments of a z-score are 0 and 1.
    """
    if all(c in panel.columns for c in MOMENT_COLS):
        raise ValueError("causal_frame needs the RAW panel, not a standardised one")
    est = attach_causal_moments(panel[["date", "ticker", TARGET]], horizon=horizon)
    return (est[["date", CS_MEAN, CS_SD]].drop_duplicates("date")
            .rename(columns={CS_MEAN: "causal_mean", CS_SD: "causal_sd"}))


def invert(preds: pd.DataFrame, causal: pd.DataFrame,
           z_col: str = "y_pred") -> pd.DataFrame:
    """`preds` with `pred_return`: the standardised prediction turned back into
    a 30-session log return with the PAST-ONLY moments for its date."""
    out = preds.merge(causal, on="date", how="left")
    out["pred_return"] = inverse_standardise(out[z_col], out["causal_mean"],
                                             out["causal_sd"])
    return out
