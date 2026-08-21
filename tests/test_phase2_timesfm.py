"""
tests/test_phase2_timesfm.py — Guards for the TimesFM-2.5 comparator.

The same three indices as Chronos — median quantile, last horizon step, last
observation — but on a checkpoint that lays them out differently, which is the
whole reason this file exists separately.

TimesFM emits NINE quantiles against Chronos' thirteen, and offsets them by one
because ``full_predictions[..., 0]`` is the point output. So its median sits at
index 5 where Chronos' sits at 6. Reading by position rather than by value
returns the 60th percentile here: a systematic upward bias, present in every
prediction, that reads as skill in a rising market and raises no error. The
Chronos suite was written against exactly that hypothetical; TimesFM is the
checkpoint that makes it real.

Two more differences carry their own tests. ``truncate_negative`` defaults to
``config.infer_is_positive`` and clamps forecasts at zero based on ONE scalar
taken over the whole batch, and ``forecast_context_len`` silently truncates the
history if it is left to default.

The stub's right answer is arithmetically known, for the reason given at the
top of tests/test_phase2_chronos.py: a test against 200M real parameters can
only check that a number came back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import series as S
from pipeline import timesfm_forecaster as T


# ── A stub whose right answer is known ────────────────────────────────────────


class StubConfig:
    def __init__(self, quantiles, decode_index, infer_is_positive=True,
                 patch_length=32):
        self.quantiles = quantiles
        self.decode_index = decode_index
        self.infer_is_positive = infer_is_positive
        self.patch_length = patch_length


class StubOutput:
    def __init__(self, full_predictions):
        self.full_predictions = full_predictions


class StubModel:
    """
    Stands in for ``TimesFm2_5ModelForPrediction``.

    ``levels`` is the value placed at the median column of the FINAL horizon
    step, per input position. Every other cell carries a decoy 100 apart per
    column and 1 apart per step, so a wrong column is off by ~100 and a wrong
    step by ~30 — neither can pass by coincidence.
    """

    def __init__(self, levels, quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                          0.8, 0.9),
                 decode_index=5, context_len=16384, steps=64, patch_length=32):
        self.levels = list(levels)
        self.config = StubConfig(quantiles, decode_index, patch_length=patch_length)
        self.context_len = context_len
        self.steps = steps
        self.calls: list[dict] = []

    def __call__(self, past_values, forecast_context_len=None,
                 truncate_negative=None, force_flip_invariance=None, **kwargs):
        self.calls.append({
            "n_inputs": len(past_values),
            "forecast_context_len": forecast_context_len,
            "truncate_negative": truncate_negative,
            "force_flip_invariance": force_flip_invariance,
            "lengths": [len(x) for x in past_values],
            "last_values": [float(x[-1]) for x in past_values],
        })
        patch = self.config.patch_length
        if forecast_context_len is not None and forecast_context_len % patch:
            raise RuntimeError(
                f"shape is invalid: context {forecast_context_len} is not a "
                f"multiple of patch_length {patch}")

        n_cols = 1 + len(self.config.quantiles)
        full = np.zeros((len(past_values), self.steps, n_cols))
        for i in range(len(past_values)):
            for c in range(n_cols):
                for st in range(self.steps):
                    full[i, st, c] = self.levels[i] + 100.0 * c + st
        return StubOutput(full)


class ExactStub(StubModel):
    """StubModel with the answer cell corrected for a known horizon."""

    def __init__(self, levels, horizon=30, **kwargs):
        super().__init__(levels, **kwargs)
        self.horizon = horizon

    def __call__(self, past_values, **kwargs):
        out = super().__call__(past_values, **kwargs)
        median = 1 + list(self.config.quantiles).index(0.5)
        if self.horizon <= self.steps:
            for i in range(len(past_values)):
                out.full_predictions[i, self.horizon - 1, median] = self.levels[i]
        return out


def histories(last_values, length=200):
    return {
        f"T{i:02d}.NS": np.concatenate([
            np.linspace(v - 1.0, v, length - 1), [v]]).astype(float)
        for i, v in enumerate(last_values)
    }


# ── The three indices ─────────────────────────────────────────────────────────


def test_the_prediction_is_a_move_not_a_level():
    hist = histories([-3.0, -2.0])
    stub = ExactStub(levels=[-3.0 + 0.05, -2.0 - 0.02])

    out = T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(+0.05)
    assert out["T01.NS"] == pytest.approx(-0.02)


def test_the_reference_is_the_last_observation_not_the_first():
    hist = {"T00.NS": np.linspace(-5.0, -3.0, 300)}
    stub = ExactStub(levels=[-3.0 + 0.10])

    out = T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(+0.10)


def test_the_median_column_is_offset_by_the_point_output():
    """
    full_predictions[..., 0] is the point forecast and [..., 1:] are the nine
    quantiles, so the median is at 1 + index(0.5) = 5, not index(0.5) = 4.
    Dropping the offset returns the 40th percentile of every prediction.
    """
    stub = ExactStub(levels=[-3.0])
    assert T.median_index(stub) == 5
    assert list(stub.config.quantiles).index(0.5) == 4


def test_a_chronos_style_position_would_read_the_wrong_quantile():
    """
    Chronos puts its median at 6 of 13. TimesFM's index 6 is the 0.6 quantile —
    a systematic upward bias in every prediction, with no error raised. This is
    the case tests/test_phase2_chronos.py was written against as a
    hypothetical; here it is the actual second checkpoint.
    """
    stub = ExactStub(levels=[-3.0])
    quantiles = list(stub.config.quantiles)

    assert T.median_index(stub) != 6
    assert quantiles[6 - 1] == pytest.approx(0.6)


def test_a_median_disagreement_between_layout_and_config_raises():
    """
    The quantile vector and decode_index state the median independently. If
    they diverge, one of them is a systematic bias in every prediction, and
    picking a winner would be guessing which.
    """
    stub = ExactStub(levels=[-3.0], decode_index=7)

    with pytest.raises(ValueError, match="disagreement"):
        T.median_index(stub)


def test_the_last_horizon_step_is_used_not_the_first():
    hist = histories([-3.0])
    stub = ExactStub(levels=[-3.0 + 0.07], horizon=30)

    out = T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)
    assert out["T00.NS"] == pytest.approx(+0.07)


def test_each_forecast_goes_to_the_ticker_that_produced_it():
    hist = histories([-3.0, -2.0, -1.0, -4.0])
    stub = ExactStub(levels=[-3.0 + 0.01, -2.0 + 0.02, -1.0 + 0.03, -4.0 + 0.04])

    out = T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(0.01)
    assert out["T01.NS"] == pytest.approx(0.02)
    assert out["T02.NS"] == pytest.approx(0.03)
    assert out["T03.NS"] == pytest.approx(0.04)


# ── The two TimesFM-specific traps ────────────────────────────────────────────


def test_negative_truncation_is_forced_off():
    """
    truncate_negative defaults to config.infer_is_positive, which is True, and
    clamps the forecast at zero when the BATCH minimum is non-negative — one
    scalar over the whole cross-section.

    Our series is log(close / benchmark_close), around -3 for an NSE name
    against its sector index, so it would not fire today. It would fire on a
    date where every name traded above its benchmark level, silently, on some
    dates and not others, producing a table that is right most of the time.
    """
    hist = histories([-3.0])
    stub = ExactStub(levels=[-3.0])

    T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)

    assert stub.calls[0]["truncate_negative"] is False


def test_the_context_length_is_passed_explicitly():
    """
    forward() does `ts[-forecast_context_len:]` and falls back to
    self.context_len. Left to default, a 2,048-observation window silently
    becomes whatever the checkpoint prefers, and the results table reports the
    number that was asked for rather than the one that was used.
    """
    hist = histories([-3.0, -2.0], length=768)          # already 24 * 32
    stub = ExactStub(levels=[-3.0, -2.0])

    T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)

    assert stub.calls[0]["forecast_context_len"] == 768


def test_the_context_is_rounded_up_to_a_patch_boundary_never_down():
    """
    TimesFM patches its input, so a context that is not a multiple of
    patch_length makes the encoder's reshape fail outright — which is at least
    loud. The quiet failure is rounding the other way: forward() keeps
    `ts[-forecast_context_len:]`, so rounding DOWN drops up to 31 of the oldest
    observations from a window the results table claims was used.

    Rounding up costs nothing. _preprocess left-pads to the requested length
    and masks the padding, so the model sees the same real observations.
    """
    hist = histories([-3.0], length=777)                # 24.28 patches
    stub = ExactStub(levels=[-3.0])

    T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)

    used = stub.calls[0]["forecast_context_len"]
    assert used % 32 == 0
    assert used == 800, used
    assert used >= 777, "rounding down would drop the oldest observations"


def test_the_rounded_context_never_exceeds_the_model():
    """
    The guard is on the value actually SENT, not on the raw history length.

    An earlier version clamped with min(context_used, limit) instead. That is
    worse than useless: it can only bind when the checkpoint's context is not a
    patch multiple, and the clamped value is then not a patch multiple either,
    so it trades a clear error for a confusing one. A mutation removing it
    survived, which is what exposed it.

    A 500-observation history rounds to 512: that fits a 512-context model. A
    470-observation history also rounds to 480, and 480 does not fit a
    470-context model even though the raw history does exactly. That gap is the
    whole point of guarding the rounded value rather than the raw length.
    """
    fits = ExactStub(levels=[-3.0], context_len=512)
    T.TimesFM25Forecaster(model=fits).forecast(
        histories([-3.0], length=500), horizon=30)
    assert fits.calls[0]["forecast_context_len"] == 512

    # 470 <= 470, so a guard on the RAW length would let this through.
    too_small = ExactStub(levels=[-3.0], context_len=470)
    with pytest.raises(ValueError, match="exceeds"):
        T.TimesFM25Forecaster(model=too_small).forecast(
            histories([-3.0], length=470), horizon=30)


def test_flip_invariance_reaches_the_model():
    hist = histories([-3.0])
    for flag in (True, False):
        stub = ExactStub(levels=[-3.0])
        T.TimesFM25Forecaster(model=stub,
                              force_flip_invariance=flag).forecast(hist, horizon=30)
        assert stub.calls[0]["force_flip_invariance"] is flag


def test_a_history_longer_than_the_model_context_raises():
    hist = histories([-3.0], length=900)
    stub = ExactStub(levels=[-3.0], context_len=512)

    with pytest.raises(ValueError, match="exceeds"):
        T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)


def test_a_horizon_longer_than_the_model_returns_raises():
    """
    Silently reading the last step the model DID return would answer a
    different question — a 20-session forecast reported as a 30-session one.
    """
    hist = histories([-3.0])
    stub = ExactStub(levels=[-3.0], steps=20)

    with pytest.raises(ValueError, match="horizon"):
        T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)


# ── Declining rather than inventing ───────────────────────────────────────────


def test_a_non_finite_forecast_is_declined():
    hist = histories([-3.0, -2.0])
    stub = ExactStub(levels=[np.nan, -2.0 + 0.02])

    out = T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)

    assert "T00.NS" not in out
    assert out["T01.NS"] == pytest.approx(+0.02)


def test_an_implausible_forecast_is_declined():
    hist = histories([-3.0, -2.0])
    stub = ExactStub(levels=[-3.0 + 5.0, -2.0 + 0.02])

    out = T.TimesFM25Forecaster(model=stub).forecast(hist, horizon=30)

    assert set(out) == {"T01.NS"}


def test_no_histories_makes_no_call():
    stub = ExactStub(levels=[])
    assert T.TimesFM25Forecaster(model=stub).forecast({}, horizon=30) == {}
    assert stub.calls == []


# ── Wiring ────────────────────────────────────────────────────────────────────


def test_timesfm_is_not_in_the_known_answer_set():
    assert not any("timesfm" in name for name in S.SERIES_BASELINES)


def test_importing_the_comparator_module_does_not_import_torch():
    """
    Same guard as Chronos. pipeline.baselines is imported by the weekly job and
    by the API; torch must stay inside load_model().
    """
    import subprocess
    import sys

    code = (
        "import sys; import pipeline.timesfm_forecaster, pipeline.baselines; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules "
        "if m.startswith('torch')); print('clean')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "clean" in proc.stdout


def test_timesfm_requires_the_series_comparators():
    from pipeline.baselines import compare_baselines

    with pytest.raises(ValueError, match="foundation-model"):
        compare_baselines(with_timesfm=True, with_series=False)


def test_timesfm_is_scored_when_asked_for(monkeypatch):
    import pipeline.baselines
    import pipeline.timesfm_forecaster
    from tests.test_phase2_baselines import make_panel

    panel = make_panel(n_dates=800, n_tickers=20, seed=131)
    rng = np.random.default_rng(131)
    panel["close"] = 100.0 * np.exp(rng.normal(0, 0.01, len(panel)).cumsum())
    panel["benchmark_close"] = 20000.0

    class FastStub:
        def __init__(self, force_flip_invariance=True, **kwargs):
            self.name = "stub"

        def forecast(self, histories, horizon):
            return {t: 0.0 for t in histories}

    monkeypatch.setattr(pipeline.baselines, "load_panel", lambda **k: panel)
    monkeypatch.setattr(pipeline.timesfm_forecaster, "TimesFM25Forecaster",
                        FastStub)

    comparison = pipeline.baselines.compare_baselines(
        with_pooled_xgb=False, with_timesfm=True)

    names = {r["name"] for r in comparison.results}
    assert set(T.TIMESFM_VARIANTS).issubset(names), names


# ── The real checkpoint ───────────────────────────────────────────────────────


timesfm_installed = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("torch") is None,
    reason="torch not installed (requirements-series.txt)",
)


@timesfm_installed
def test_the_real_checkpoint_lays_its_median_out_where_we_think():
    """
    Pins the layout the index arithmetic depends on against the actual weights,
    rather than against the stub that was written to match them.
    """
    model = T.load_model(T.DEFAULT_MODEL_ID)

    quantiles = list(model.config.quantiles)
    assert 0.5 in quantiles
    assert T.median_index(model) == 1 + quantiles.index(0.5)
    assert T.median_index(model) == int(model.config.decode_index)


@timesfm_installed
def test_the_real_model_forecasts_a_move_on_the_scale_of_the_target():
    """
    A sanity bound, not an accuracy claim. Returning levels instead of moves
    would land near -3; reading the wrong quantile decade would be similarly
    far out.
    """
    rng = np.random.default_rng(11)
    hist = {f"T{i:02d}.NS": np.log(500.0 / 20000.0)
            + np.cumsum(rng.normal(0.0, 0.012, 400)) for i in range(6)}

    out = T.TimesFM25Forecaster().forecast(hist, horizon=30)

    assert len(out) == 6
    assert all(np.isfinite(v) for v in out.values())
    assert max(abs(v) for v in out.values()) < 0.5


@timesfm_installed
def test_the_model_is_loaded_once_per_process():
    assert T.load_model(T.DEFAULT_MODEL_ID) is T.load_model(T.DEFAULT_MODEL_ID)
