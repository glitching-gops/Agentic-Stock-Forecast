"""
pipeline/series.py — Running a time-series forecaster inside the panel harness.

The harness speaks feature matrices: ``fit(X, y)`` then ``predict(X)``, where a
row is one (date, ticker) and the columns are indicators. A time-series
foundation model speaks something else entirely — hand it a history, get back a
forecast path, and there is no ``fit`` at all. Chronos-2 and TimesFM-2.5 cannot
be dropped into ``panel_walk_forward`` as they stand. This module is the
adapter that makes them fit, and it is built and proven BEFORE either of them
arrives, with forecasters whose answers are known in advance.

That ordering is the whole point. If the adapter is wrong, a foundation model's
result is uninterpretable — a poor number could be the model or the plumbing,
and there is no way to tell them apart after the fact. ``ZeroDrift`` exists so
there is a case whose output is analytically known: run through this adapter it
must reproduce ``baselines.ZeroForecast`` exactly, row for row. When that holds,
the adapter is not what is being measured any more.

────────────────────────────────────────────────────────────────────────────
THE PURGED FOLDS PROTECT NOTHING HERE. READ THIS BEFORE CHANGING ANYTHING.
────────────────────────────────────────────────────────────────────────────

Purging, the embargo and the training-set boundary all exist to stop a model
being FITTED on information that overlaps what it is scored on. A zero-shot
model is never fitted. ``fit`` is a no-op. Every one of those guarantees is
therefore inert, and the entire protection collapses onto one line: the history
handed to the forecaster must end at the as-of date.

A single off-by-one in that slice — ``<`` where ``<=`` was meant, an unsorted
index, a reindex that fills forward from a later row — hands the model the
answer, and it will look like a spectacular result rather than an error. This
is F1 with the meta-learner replaced by a 200M-parameter model, and it would be
considerably harder to spot: there would be no in-sample fit to re-run.

``_history_ending_at`` is the only place that slice is made. It slices by
position on a pre-sorted index, and the tests corrupt every value after the
as-of date and require the predictions to be bit-identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

# How many trailing observations to hand a forecaster. 2,048 is Chronos-2's
# context; TimesFM-2.5 accepts 16,384 and would take a ticker's whole history.
# Kept explicit rather than "as much as exists" because a model silently
# truncating its own input is a difference between runs that nothing records.
DEFAULT_CONTEXT = 2048

# A history shorter than this cannot support a 30-session view under any
# method: it is barely two horizons of data, so a trailing-return estimate
# would be computed from a single non-overlapping window.
MIN_CONTEXT = 90


def resolve_device(preferred: str | None = None) -> str:
    """
    Where a foundation model should run: explicit choice, env, then autodetect.

    Autodetect rather than a hardcoded default because the same code runs in
    two places with opposite answers. A GitHub runner has no GPU and must land
    on CPU without configuration; a workstation with a CUDA build should not
    have to be told. ``SERIES_DEVICE`` overrides both — set it to ``cpu`` to
    reproduce a CPU-measured table on a machine that has a GPU.

    Imported lazily. This module is reachable from ``pipeline.baselines`` and
    therefore from the API, where torch must never become a hard requirement.
    """
    choice = (preferred or os.getenv("SERIES_DEVICE") or "").strip().lower()
    if choice:
        return choice
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                           # noqa: BLE001
        return "cpu"


def configure_determinism() -> None:
    """
    Turn OFF TF32 so a GPU run reproduces the CPU-measured tables.

    Ada and Ampere silently execute float32 matmuls in TF32, which keeps the
    float32 exponent but truncates the mantissa from 23 bits to 10. That is
    invisible in a loss curve and very visible here: the quantity being
    forecast is around 1e-2, the comparison against the `zero` floor turns on
    the fifth decimal of MAE, and every number recorded so far was measured on
    CPU. A speedup that silently changes the third significant figure of the
    result is not a speedup, it is a different experiment.
    """
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:                                           # noqa: BLE001
        pass


@runtime_checkable
class SeriesForecaster(Protocol):
    """
    What a time-series model has to provide to run in this harness.

    ``forecast`` receives a whole date's cross-section at once — ``histories``
    maps ticker to that ticker's log relative-price series ending at the as-of
    date — and returns a predicted ``horizon``-session log return per ticker.

    Batched by date on purpose. Per-series inference over this panel is roughly
    180,000 calls at 95 tickers, which is not affordable on a two-core Actions
    runner; one call per date is ~1,900. It is also the shape Chronos-2's group
    attention wants, since the batch it conditions across is exactly a
    cross-section. A forecaster that has no batching simply loops inside.
    """

    name: str

    def forecast(self, histories: dict[str, np.ndarray],
                 horizon: int) -> dict[str, float]:
        ...


def _history_ending_at(series: pd.DataFrame, positions: dict[str, int],
                       as_of: str, context: int) -> dict[str, np.ndarray]:
    """
    Each ticker's observations up to AND INCLUDING `as_of`, most recent last.

    The single line the zero-shot guarantee rests on — see the module docstring.
    `positions` maps a date to its row position in `series`, which is sorted
    once by the caller; slicing positionally on a pre-sorted index avoids
    relying on label-based `.loc[:as_of]`, whose inclusivity depends on the
    index being monotonic and which silently returns everything if it is not.
    """
    end = positions.get(as_of)
    if end is None:
        return {}

    block = series.iloc[max(0, end + 1 - context): end + 1]

    histories: dict[str, np.ndarray] = {}
    for ticker in block.columns:
        values = block[ticker].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) >= MIN_CONTEXT:
            histories[ticker] = values
    return histories


@dataclass
class SeriesAdapter:
    """
    Wraps a ``SeriesForecaster`` so ``panel_walk_forward`` can run it unchanged.

    Give it ``feature_cols=["date", "ticker"]``. Those two columns are all it
    needs from the harness — they identify the row; the history comes from the
    series frame handed in at construction. Every other comparator in the
    project receives indicator columns and this one does not, which is exactly
    the difference the adapter exists to absorb.

    ``fit`` is a no-op because these models are zero-shot. It is not an
    oversight and it is not a placeholder for training that will arrive later:
    it is the property that makes the purged folds inert, and the reason the
    history slice carries the whole burden.
    """

    forecaster: SeriesForecaster
    series: pd.DataFrame                    # dates x tickers, log relative price
    horizon: int = 30
    context: int = DEFAULT_CONTEXT

    def __post_init__(self) -> None:
        if not self.series.empty and not self.series.index.is_monotonic_increasing:
            self.series = self.series.sort_index()
        self._positions = {d: i for i, d in enumerate(self.series.index)}

    @property
    def name(self) -> str:
        return getattr(self.forecaster, "name", type(self.forecaster).__name__)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeriesAdapter":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "date" not in X.columns or "ticker" not in X.columns:
            raise ValueError(
                "SeriesAdapter needs 'date' and 'ticker' columns; pass "
                "feature_cols=['date', 'ticker'] to panel_walk_forward."
            )

        out = np.zeros(len(X), dtype=float)
        dates = X["date"].to_numpy()
        tickers = X["ticker"].to_numpy()

        for as_of in pd.unique(dates):
            histories = _history_ending_at(self.series, self._positions,
                                           as_of, self.context)
            if not histories:
                continue

            predicted = self.forecaster.forecast(histories, self.horizon)

            rows = np.flatnonzero(dates == as_of)
            for row in rows:
                value = predicted.get(tickers[row])
                # A ticker the forecaster declined to score predicts no excess
                # return, which is the honest default: it is the same claim the
                # `zero` floor makes, so an abstention cannot flatter the
                # result relative to the comparator it is measured against.
                out[row] = float(value) if value is not None and np.isfinite(value) else 0.0

        return out


def adapter_factory(forecaster_cls, series: pd.DataFrame, horizon: int = 30,
                    context: int = DEFAULT_CONTEXT, **kwargs):
    """
    A ``model_factory`` for ``panel_walk_forward``.

    The harness calls the factory once per fold and expects a fresh estimator,
    so the forecaster is constructed per fold too. The series frame is shared —
    it is reference data, identical across folds, and the fold boundary is
    enforced by the as-of slice rather than by handing each fold a different
    frame. Handing folds different frames would be the more obvious design and
    a worse one: it would make the truncation depend on the caller getting the
    construction right, instead of on one tested function.
    """
    def factory() -> SeriesAdapter:
        return SeriesAdapter(forecaster=forecaster_cls(**kwargs), series=series,
                             horizon=horizon, context=context)
    return factory


# ── Forecasters whose answers are known in advance ────────────────────────────


class ZeroDrift:
    """
    Predicts no excess return, from the series path.

    THE CALIBRATION CASE. Run through ``SeriesAdapter`` this must reproduce
    ``baselines.ZeroForecast`` exactly — same rows, same predictions, same
    metrics. Any difference is the adapter, not the model, and is a defect to
    fix before a real forecaster goes anywhere near this code path.
    """

    name = "series_zero"

    def forecast(self, histories: dict[str, np.ndarray],
                 horizon: int) -> dict[str, float]:
        return {ticker: 0.0 for ticker in histories}


class LastReturn:
    """
    Predicts that the trailing `horizon`-session move repeats.

    The series momentum baseline, and a genuine one: a foundation model that
    cannot beat "what just happened happens again" on a relative-price series
    has not earned its dependency footprint. Distinct from
    ``baselines.SingleFactor('sector_rel_20d')`` in that it reads the same
    quantity the model reads, over the same window, so it isolates the model's
    contribution rather than the feature engineering's.
    """

    name = "series_last_return"

    def forecast(self, histories: dict[str, np.ndarray],
                 horizon: int) -> dict[str, float]:
        out = {}
        for ticker, values in histories.items():
            if len(values) > horizon:
                out[ticker] = float(values[-1] - values[-1 - horizon])
        return out


class HistoricalDrift:
    """
    Predicts the mean per-session drift of the series, scaled to the horizon.

    A random walk WITH drift, against ``ZeroDrift``'s driftless one. Worth
    separating: a ticker that has outperformed its benchmark steadily for years
    has a non-zero drift, and if that is all the signal there is, a foundation
    model reporting a positive result has found nothing a two-line estimator
    could not.
    """

    name = "series_drift"

    def forecast(self, histories: dict[str, np.ndarray],
                 horizon: int) -> dict[str, float]:
        out = {}
        for ticker, values in histories.items():
            if len(values) > 1:
                per_session = (values[-1] - values[0]) / (len(values) - 1)
                out[ticker] = float(per_session * horizon)
        return out


SERIES_BASELINES: dict[str, type] = {
    "series_zero": ZeroDrift,
    "series_last_return": LastReturn,
    "series_drift": HistoricalDrift,
}
