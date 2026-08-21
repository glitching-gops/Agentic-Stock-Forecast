"""
pipeline/chronos_forecaster.py — Chronos-2 as a comparator in the panel harness.

A ``SeriesForecaster`` (see ``pipeline/series.py``) backed by Amazon's Chronos-2,
a zero-shot time-series foundation model. It is deliberately the LAST thing
built in this part of Phase 2: the adapter it plugs into was proven first with
``ZeroDrift``, whose answer is known analytically, so a poor number here can be
read as the model rather than as the plumbing.

WHAT IT IS HANDED, AND WHY THAT IS THE WHOLE DESIGN
───────────────────────────────────────────────────
The series is ``log(close / benchmark_close)``. By the identity recorded in
``panel.relative_price_frame``, the forward 30-session difference of that series
IS ``target_excess_return`` — exactly, measured at 2.7e-15 on real NSE data. So
a univariate model given this one series predicts the excess return directly:

    prediction = median_forecast[t + horizon] - observed[t]

There is no second model for the benchmark and no differencing of two
independent errors. Forecasting the stock and the index separately and
subtracting is the obvious alternative and a much worse one: it would measure
the difference of two forecasts, each carrying its own bias, on a target whose
scale is smaller than either error.

WHY THE MEDIAN
──────────────
Chronos-2 returns 13 quantiles. The table this comparator joins reports MAE
against a ``zero`` floor, and MAE is minimised by the median, not the mean.
Taking the mean would also require the trapezoidal probability-mass weighting
in ``Chronos2Pipeline._get_prob_mass_per_quantile_level``, which is a second
approximation layered on a quantity we do not want. The median is read by
VALUE (``quantiles.index(0.5)``) and never by a hardcoded index — the quantile
vector is a property of the checkpoint, and a different one would silently turn
position 6 into the 40th percentile.

THE PURGED FOLDS STILL PROTECT NOTHING
──────────────────────────────────────
Everything in ``pipeline/series.py``'s module docstring applies here with more
force, because this is the model that would make an off-by-one look like a
discovery. ``fit`` is a no-op; the entire guarantee is that
``_history_ending_at`` slices at the as-of date. This module never touches the
series frame, never sees a date, and cannot widen that window — it only ever
receives arrays the adapter has already cut.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# The 28M checkpoint, published by the AutoGluon org rather than amazon — there
# is no `amazon/chronos-2-small`. Reported within ~1% of the 120M
# `amazon/chronos-2` at roughly half the inference cost, which is what makes a
# ~1,900-date panel affordable on a two-core runner at all.
DEFAULT_MODEL_ID = "autogluon/chronos-2-small"

# Measured on a 2-thread CPU, 30-session horizon, per date:
#   46 series x 2048 context -> 1.71 s   (~54 min over 1,900 dates)
#   95 series x 2048 context -> 3.31 s   (~105 min)
#   95 series x  512 context -> 0.93 s   (~29 min)
# Recorded here because cost is linear in both and the context actually used is
# otherwise invisible in the results table.

_PIPELINES: dict[str, Any] = {}


class ChronosUnavailable(ImportError):
    """Raised when the optional torch/chronos dependency is not installed."""


def load_pipeline(model_id: str = DEFAULT_MODEL_ID, torch_threads: int | None = None):
    """
    Loads a ``Chronos2Pipeline``, once per model id per process.

    Cached because ``panel_walk_forward`` builds a fresh estimator per fold, and
    ``adapter_factory`` a fresh forecaster with it. Re-reading 28M weights five
    times would be pure waste — and unlike a fitted model there is nothing
    fold-specific to reset, precisely because nothing is fitted.

    Imported lazily. ``pipeline/baselines.py`` is on the import path of the
    weekly job and, through it, of the API; torch must never become a hard
    requirement of either. It is installed only by the series-comparison
    workflow, from ``requirements-series.txt``.
    """
    if model_id in _PIPELINES:
        return _PIPELINES[model_id]

    try:
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:                                  # pragma: no cover
        raise ChronosUnavailable(
            "Chronos-2 needs torch and chronos-forecasting, which are NOT in "
            "requirements.txt on purpose: torch was removed in Phase 0 as the "
            "largest contributor to memory pressure on the free-tier Render "
            "instance, and neither the API nor the daily job calls this code. "
            "Install them with `pip install -r requirements-series.txt`."
        ) from exc

    threads = torch_threads or int(os.getenv("CHRONOS_THREADS", "0")) or None
    if threads:
        # Left at the torch default otherwise. Set explicitly on a runner,
        # because torch sizes its pool from the physical core count and a
        # shared two-core box oversubscribes badly.
        torch.set_num_threads(threads)

    logger.info(f"[Chronos] loading {model_id} (threads={torch.get_num_threads()})")
    _PIPELINES[model_id] = Chronos2Pipeline.from_pretrained(model_id)
    return _PIPELINES[model_id]


@dataclass
class Chronos2Forecaster:
    """
    Zero-shot Chronos-2 over the log relative-price series.

    ``cross_learning`` collapses the batch into a single attention group, so
    every ticker's forecast is conditioned on the rest of that date's
    cross-section. It is the in-context-learning behaviour Chronos-2 is sold on,
    and it costs nothing measurable — same tokens, different mask, timed at
    1.70 s against 1.71 s. It is nonetheless a genuinely different model, so it
    is a separate comparator rather than a default quietly flipped on.

    ``pipeline`` may be injected for tests. Nothing else in this class knows
    what a Chronos is: it maps arrays to a batch, reads one number out, and
    subtracts the last observation.
    """

    model_id: str = DEFAULT_MODEL_ID
    cross_learning: bool = False
    torch_threads: int | None = None
    pipeline: Any = None
    max_abs_prediction: float = field(default=2.0, repr=False)

    @property
    def name(self) -> str:
        stem = self.model_id.rsplit("/", 1)[-1].replace("-", "")
        return f"{stem}_xl" if self.cross_learning else stem

    def _pipeline(self):
        if self.pipeline is None:
            self.pipeline = load_pipeline(self.model_id, self.torch_threads)
        return self.pipeline

    def forecast(self, histories: dict[str, np.ndarray],
                 horizon: int) -> dict[str, float]:
        if not histories:
            return {}

        pipe = self._pipeline()

        # Insertion order is the contract between the batch sent and the list
        # returned: Chronos2Pipeline.predict yields one tensor per input,
        # positionally. Zipping the two without pinning the order first would
        # attach every forecast to the wrong ticker and still produce a
        # perfectly plausible-looking table.
        tickers = list(histories)
        batch = [np.asarray(histories[t], dtype=np.float32) for t in tickers]

        longest = max(len(h) for h in batch)
        limit = pipe.model_context_length
        if longest > limit:
            # Chronos truncates to its own context and warns. A run that
            # actually measured 8,192 while the table says 16,384 is a
            # difference nothing records, so it is an error here.
            raise ValueError(
                f"history of {longest} observations exceeds {self.model_id}'s "
                f"context of {limit}; lower SeriesAdapter(context=...) so the "
                f"window the table reports is the window the model saw"
            )

        quantiles = list(pipe.quantiles)
        if 0.5 not in quantiles:                                # pragma: no cover
            raise ValueError(
                f"{self.model_id} does not emit a median: {quantiles}")
        median = quantiles.index(0.5)

        forecasts = pipe.predict(
            batch,
            prediction_length=horizon,
            # One batch, so `cross_learning` conditions across the WHOLE
            # cross-section rather than across whichever names happened to land
            # in the same chunk.
            batch_size=max(len(batch), 1),
            cross_learning=self.cross_learning,
        )

        out: dict[str, float] = {}
        for ticker, history, path in zip(tickers, batch, forecasts):
            # (n_variates, n_quantiles, prediction_length); univariate input, so
            # variate 0. The LAST step is the 30-session-ahead level — the steps
            # before it are the path there and are not the target.
            level = float(np.asarray(path)[0, median, -1])
            move = level - float(history[-1])

            if not np.isfinite(move) or abs(move) > self.max_abs_prediction:
                # An excess return of e^2 over 30 sessions is not a forecast,
                # it is a defect. Declining is what the adapter turns into 0.0,
                # which is the `zero` floor's own claim and therefore cannot
                # flatter this comparator against the one it is measured on.
                logger.warning(
                    f"[Chronos] {ticker}: implausible prediction {move:+.4f}, "
                    f"declined")
                continue
            out[ticker] = move

        return out


# Registered variants. Kept OUT of `series.SERIES_BASELINES`, whose contract is
# that every member's answer is known in advance — that dict is the calibration
# set, and a foundation model is the thing being calibrated against it.
CHRONOS_VARIANTS: dict[str, dict] = {
    "chronos2small": {"cross_learning": False},
    "chronos2small_xl": {"cross_learning": True},
}
