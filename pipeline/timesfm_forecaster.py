"""
pipeline/timesfm_forecaster.py — TimesFM-2.5 as a comparator in the panel harness.

The second foundation model, and deliberately a different ARCHITECTURE rather
than a different size of the same one. Chronos-2 is a 28M encoder-only model
from Amazon; TimesFM-2.5 is a 200M decoder-only model from Google, trained on a
different corpus with a different objective. If two independent architectures
agree on this panel, that is evidence about the TARGET rather than about either
model — which is worth considerably more than a second null row.

It plugs into the same ``SeriesAdapter`` over the same log relative-price
series, on the same purged folds, over the same rows. See
``pipeline/chronos_forecaster.py`` for why the prediction is
``median_forecast[t+horizon] - observed[t]`` and why that quantity IS
``target_excess_return``; none of that reasoning changes here.

WHAT IS DIFFERENT FROM CHRONOS, AND WHY IT MATTERS
──────────────────────────────────────────────────
1. **The quantile vector is 9 values, not 13, and the median sits at index 5.**
   Chronos puts it at 6. Reading the median by POSITION rather than by value
   would silently return the 60th percentile here — a systematic upward bias
   that reads as skill in a rising market. This is not a hypothetical: it is
   the exact failure ``tests/test_phase2_chronos.py`` was written against, and
   the second checkpoint we tried is the one that triggers it.

   The layout is also offset. ``full_predictions[..., 0]`` is the point output
   and ``[..., 1:10]`` are the nine quantiles, so the median index is
   ``1 + quantiles.index(0.5)`` — not ``quantiles.index(0.5)``. The two agree
   with the checkpoint's own ``config.decode_index``, and this module refuses
   to run if they ever disagree.

2. **``truncate_negative`` clamps forecasts to zero.** It defaults to
   ``config.infer_is_positive``, which is True. The clamp fires when the
   *batch* minimum is non-negative — one scalar over the whole cross-section.
   Our series is ``log(close / benchmark_close)``, around -3 for an NSE name
   against its sector index, so it would not fire today. It would fire on a
   cross-section where every name happened to trade above its benchmark level,
   and it would do so silently, on some dates and not others. Forced off.

3. **``force_flip_invariance`` runs the decoder twice** — once on the input and
   once on its negation — and averages. It is on by default, it is part of what
   the checkpoint is, and it doubles inference cost. Exposed rather than
   hardcoded so the cost can be traded against the model as published.

THE PURGED FOLDS STILL PROTECT NOTHING
──────────────────────────────────────
Zero-shot, so ``fit`` is a no-op and the whole guarantee is that
``_history_ending_at`` slices at the as-of date. This module never sees a date.
``tools/check_chronos.py --model timesfm`` verifies it against live weights.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pipeline.series import configure_determinism, resolve_device

logger = logging.getLogger(__name__)

# The transformers PORT, not the `timesfm` PyPI package. That package is still
# at 2.0.2 and reaching 2.5 through it needs a git clone plus `pip install -e`,
# which is an unpinned dependency in a workflow. The port ships inside
# transformers, which chronos-forecasting already pins (>=4.41,<6), so this
# model costs no new dependency at all.
DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-transformers"

_MODELS: dict[str, Any] = {}


class TimesFMUnavailable(ImportError):
    """Raised when torch or a transformers new enough to carry the port is absent."""


def load_model(model_id: str = DEFAULT_MODEL_ID, torch_threads: int | None = None,
               device: str | None = None):
    """
    Loads the model once per id per process, in eval mode and float32.

    float32 rather than the checkpoint's stored dtype: this runs on a CPU
    runner, where bfloat16 matmuls are emulated and slower, and where a
    reduced-precision forecast of a quantity around 1e-2 would be measuring the
    dtype as much as the model.
    """
    # Keyed by (model, device) — a model-only key would return a CPU-resident
    # model to a caller that asked for CUDA, silently and slowly.
    device = resolve_device(device)
    key = (model_id, device)
    if key in _MODELS:
        return _MODELS[key]

    try:
        import torch
        from transformers import TimesFm2_5ModelForPrediction
    except ImportError as exc:                                  # pragma: no cover
        raise TimesFMUnavailable(
            "TimesFM-2.5 needs torch and transformers>=5. torch is NOT in "
            "requirements.txt on purpose: it was removed in Phase 0 as the "
            "largest contributor to memory pressure on the free-tier Render "
            "instance, and neither the API nor the daily job calls this code. "
            "Install with `pip install -r requirements-series.txt`."
        ) from exc

    threads = torch_threads or int(os.getenv("CHRONOS_THREADS", "0")) or None
    if threads:
        torch.set_num_threads(threads)

    configure_determinism()
    logger.info(f"[TimesFM] loading {model_id} on {device} "
                f"(threads={torch.get_num_threads()})")
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_id)
    # float32 on both devices. bfloat16 would be faster on an Ada GPU and
    # would also change the numbers, which would make every CPU-measured
    # table in this project incomparable with anything measured afterwards.
    model = model.to(device=device, dtype=torch.float32).eval()
    _MODELS[key] = model
    return model


def median_index(model) -> int:
    """
    Where the median lives in ``full_predictions``' last axis.

    ``full_predictions[..., 0]`` is the point output; ``[..., 1:]`` are
    ``config.quantiles`` in order. So the median is at ``1 + index(0.5)``, and
    the checkpoint states the same thing independently in ``decode_index``.
    Both are computed and required to agree: a checkpoint where they diverge is
    one whose layout is not the one this code was written for, and guessing
    which is right would be guessing at a systematic bias.
    """
    quantiles = list(model.config.quantiles)
    if 0.5 not in quantiles:                                    # pragma: no cover
        raise ValueError(f"{model.config.quantiles} contains no median")

    by_value = 1 + quantiles.index(0.5)
    declared = int(model.config.decode_index)
    if by_value != declared:                                    # pragma: no cover
        raise ValueError(
            f"median index disagreement: quantiles {quantiles} put it at "
            f"{by_value}, config.decode_index says {declared}. Refusing to "
            f"guess — one of them is a systematic bias in every prediction."
        )
    return by_value


@dataclass
class TimesFM25Forecaster:
    """
    Zero-shot TimesFM-2.5 over the log relative-price series.

    ``model`` may be injected for tests. As with Chronos, nothing in this class
    knows what a TimesFM is: it maps arrays to a batch, reads one number out,
    and subtracts the last observation.
    """

    model_id: str = DEFAULT_MODEL_ID
    force_flip_invariance: bool = True
    torch_threads: int | None = None
    device: str | None = None
    model: Any = None
    max_abs_prediction: float = field(default=2.0, repr=False)

    @property
    def name(self) -> str:
        stem = "timesfm25"
        return stem if self.force_flip_invariance else f"{stem}_noflip"

    def _model(self):
        if self.model is None:
            self.model = load_model(self.model_id, self.torch_threads, self.device)
        return self.model

    def forecast(self, histories: dict[str, np.ndarray],
                 horizon: int) -> dict[str, float]:
        if not histories:
            return {}

        import torch

        model = self._model()

        # Insertion order is the contract between the batch sent and the rows
        # returned — forward() yields one row per input, positionally, and
        # names nothing.
        tickers = list(histories)
        # Built directly on the model's device: forward() stacks these before
        # anything moves them, so CPU tensors against a CUDA model raise.
        dev = next(model.parameters()).device
        batch = [torch.tensor(np.asarray(histories[t], dtype=np.float32), device=dev)
                 for t in tickers]

        median = median_index(model)
        longest = max(len(h) for h in batch)
        limit = int(getattr(model, "context_len", 0) or 0)

        # TimesFM patches its input, so the context must be a multiple of
        # patch_length or the reshape inside the encoder fails outright:
        # "shape '[6, -1, 32]' is invalid for input of size 2400".
        #
        # Rounded UP, never down. `forward` keeps `ts[-forecast_context_len:]`,
        # so rounding down would quietly drop up to patch_length-1 of the
        # OLDEST observations from a window the results table claims was used.
        # Rounding up costs nothing: `_preprocess` left-pads to the requested
        # length and masks the padding, so the model sees the same real
        # observations either way.
        patch = int(getattr(model.config, "patch_length", 0) or 1)
        context_used = -(-longest // patch) * patch

        # One guard on the value actually sent, not on the raw history length.
        # Clamping to `limit` instead would be worse than useless: it can only
        # bind when the checkpoint's context is NOT a patch multiple, and the
        # clamped value is then not a patch multiple either — trading a clear
        # error for a confusing one. Both real checkpoints have patch-multiple
        # contexts (Chronos 8192, TimesFM 16384), so this only ever fires on a
        # history genuinely too long for the model.
        if limit and context_used > limit:
            raise ValueError(
                f"history of {longest} observations rounds to a {context_used}"
                f"-observation context, which exceeds {self.model_id}'s "
                f"{limit}; lower SeriesAdapter(context=...) so the window the "
                f"table reports is the window the model saw"
            )

        with torch.no_grad():
            out = model(
                past_values=batch,
                # Explicit rather than defaulted: forward() falls back to
                # self.context_len and truncates to it silently.
                forecast_context_len=context_used,
                # See the module docstring. Defaults to config.infer_is_positive
                # (True) and clamps the whole batch on one scalar.
                truncate_negative=False,
                force_flip_invariance=self.force_flip_invariance,
            )

        full = out.full_predictions.detach().to('cpu').numpy()   # (b, steps, 1+n_q)
        steps = full.shape[1]
        if steps < horizon:
            raise ValueError(
                f"{self.model_id} returned {steps} steps for a {horizon}-session "
                f"horizon; the checkpoint's horizon is shorter than the target"
            )

        result: dict[str, float] = {}
        for i, (ticker, history) in enumerate(zip(tickers, batch)):
            # Step horizon-1 is the 30th session ahead. The steps before it are
            # the path there and are not the target.
            level = float(full[i, horizon - 1, median])
            move = level - float(history[-1])

            if not np.isfinite(move) or abs(move) > self.max_abs_prediction:
                logger.warning(
                    f"[TimesFM] {ticker}: implausible prediction {move:+.4f}, "
                    f"declined")
                continue
            result[ticker] = move

        return result


# Registered variants. Kept OUT of `series.SERIES_BASELINES` for the same
# reason Chronos is: that dict is the calibration set whose answers are known
# in advance, and a model being calibrated cannot also be the calibration.
TIMESFM_VARIANTS: dict[str, dict] = {
    "timesfm25": {"force_flip_invariance": True},
}
