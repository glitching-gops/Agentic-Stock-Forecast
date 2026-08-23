"""
tests/test_phase2_chronos.py — Guards for the Chronos-2 comparator.

``pipeline/chronos_forecaster.py`` does three things to a tensor and nothing
else: pick the median quantile, take the last horizon step, subtract the last
observation. Each of those is one index, and each wrong index produces a table
that still renders — a plausible daily IC, a plausible MAE, no exception
anywhere. That is what these tests are for.

Most of them run against a STUB pipeline whose output is chosen so the right
answer is arithmetically known. That is deliberate: a test that asserts against
the real 28M model can only check that a number came back, and would pass just
as happily on the 40th percentile of step 3 as on the median of step 30.

Two tests do exercise the real checkpoint and skip when it is not installed —
torch is not in requirements.txt on purpose, so CI for the daily and weekly
jobs must stay green without it.

The as-of guarantee itself is tested in tests/test_phase2_series.py; this
module only confirms Chronos cannot widen that window, which it cannot by
construction — it never sees a date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import chronos_forecaster as C
from pipeline import series as S


# ── A stub whose right answer is known ────────────────────────────────────────


class StubPipeline:
    """
    Stands in for ``Chronos2Pipeline``.

    ``levels`` is the value placed at the median quantile of the FINAL horizon
    step, per input position. Every other cell of the returned tensor is filled
    with a decoy that is far away and would be trivially visible if it were read
    by mistake: quantile q of step s gets ``level + 100*q_index + s``, so a
    wrong quantile index is off by ~100 and a wrong step index by ~30.
    """

    def __init__(self, levels, quantiles=None, context_length=8192):
        self.levels = list(levels)
        self.quantiles = quantiles or [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                       0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.model_context_length = context_length
        self.calls: list[dict] = []

    def predict(self, inputs, prediction_length, batch_size=256,
                cross_learning=False, **kwargs):
        self.calls.append({
            "n_inputs": len(inputs), "prediction_length": prediction_length,
            "batch_size": batch_size, "cross_learning": cross_learning,
            "lengths": [len(x) for x in inputs],
            "last_values": [float(x[-1]) for x in inputs],
        })

        median = self.quantiles.index(0.5)
        out = []
        for i in range(len(inputs)):
            grid = np.zeros((1, len(self.quantiles), prediction_length))
            for q in range(len(self.quantiles)):
                for s in range(prediction_length):
                    grid[0, q, s] = self.levels[i] + 100.0 * q + s
            # Undo the decoy offsets at the one cell that is the answer.
            grid[0, median, prediction_length - 1] = self.levels[i]
            out.append(grid)
        return out


def histories(last_values, length=200):
    """Flat series ending at each given value, keyed T00.NS, T01.NS, ..."""
    return {
        f"T{i:02d}.NS": np.concatenate([
            np.linspace(v - 1.0, v, length - 1), [v]]).astype(float)
        for i, v in enumerate(last_values)
    }


# ── The three indices ─────────────────────────────────────────────────────────


def test_the_prediction_is_a_move_not_a_level():
    """
    The series is log(close / benchmark_close), a LEVEL sitting around -3 for a
    typical NSE name against its sector index. The target is a 30-session
    DIFFERENCE of that level, in the region of +/-0.3.

    Returning the level would be a ~10x scale error visible in MAE immediately.
    The dangerous version is subtracting the wrong reference — history[0], or
    the first forecast step — which stays on the right scale and reads as a
    mediocre but believable comparator.
    """
    hist = histories([-3.0, -2.0], length=200)
    stub = StubPipeline(levels=[-3.0 + 0.05, -2.0 - 0.02])

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(+0.05)
    assert out["T01.NS"] == pytest.approx(-0.02)


def test_the_reference_is_the_last_observation_not_the_first():
    """A history that trends: subtracting history[0] would be off by its span."""
    hist = {"T00.NS": np.linspace(-5.0, -3.0, 300)}       # spans 2.0
    stub = StubPipeline(levels=[-3.0 + 0.10])

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(+0.10)
    assert out["T00.NS"] != pytest.approx(+2.10)


def test_the_median_quantile_is_used():
    """
    MAE is minimised by the median, and MAE against the `zero` floor is what
    the comparison table turns on. The stub puts a decoy 100 away at every
    other quantile, so reading the 1% tail or the 99% is unmissable.
    """
    hist = histories([-3.0])
    stub = StubPipeline(levels=[-3.0 + 0.07])

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(+0.07)


def test_the_real_checkpoints_disagree_about_where_the_median_lives():
    """
    Not hypothetical. Measured on the two published Chronos-2 checkpoints:

        autogluon/chronos-2-small   13 quantiles, median at index 6
        amazon/chronos-2            21 quantiles, median at index 10

    Index 6 on the 120M model is the 0.3 quantile. Had the median been read by
    position, swapping checkpoints would have applied a systematic downward
    bias to every prediction, with no error and a perfectly plausible table.
    """
    _require_torch("chronos")

    seen = {}
    for mid in ("autogluon/chronos-2-small", "amazon/chronos-2"):
        q = list(C.load_pipeline(mid).quantiles)
        seen[mid] = (len(q), q.index(0.5))

    assert seen["autogluon/chronos-2-small"] != seen["amazon/chronos-2"], seen
    small_median = seen["autogluon/chronos-2-small"][1]
    big_q = list(C.load_pipeline("amazon/chronos-2").quantiles)
    assert big_q[small_median] != 0.5, (
        "the checkpoints happen to agree; this test no longer proves anything")


def test_the_median_is_found_by_value_not_by_position():
    """
    Position 6 is the median of the 13 quantiles this checkpoint emits, and of
    nothing else. A checkpoint with a different quantile vector — the 9-quantile
    set the long-horizon unroller uses, for one — would make position 6 the 70th
    percentile, and every prediction would acquire a silent upward bias that
    looks like skill on a rising market.
    """
    hist = histories([-3.0])
    nine = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]     # median at 4
    stub = StubPipeline(levels=[-3.0 + 0.07], quantiles=nine)

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(+0.07)


def test_the_last_horizon_step_is_used_not_the_first():
    """
    Steps 0..28 are the path to the 30-session forecast, not the forecast. The
    stub offsets step s by +s, so reading step 0 lands 29 away.
    """
    hist = histories([-3.0])
    stub = StubPipeline(levels=[-3.0 + 0.07])

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(+0.07)
    assert stub.calls[0]["prediction_length"] == 30


def test_each_forecast_goes_to_the_ticker_that_produced_it():
    """
    predict() returns one tensor per input, positionally. Nothing in the return
    value names a ticker, so the mapping is the input order and only that. A
    shuffle here produces a complete, well-formed, entirely meaningless table.
    """
    hist = histories([-3.0, -2.0, -1.0, -4.0])
    stub = StubPipeline(levels=[-3.0 + 0.01, -2.0 + 0.02,
                                -1.0 + 0.03, -4.0 + 0.04])

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)

    assert out["T00.NS"] == pytest.approx(0.01)
    assert out["T01.NS"] == pytest.approx(0.02)
    assert out["T02.NS"] == pytest.approx(0.03)
    assert out["T03.NS"] == pytest.approx(0.04)
    assert stub.calls[0]["last_values"] == pytest.approx([-3.0, -2.0, -1.0, -4.0])


# ── Declining rather than inventing ───────────────────────────────────────────


def test_a_non_finite_forecast_is_declined():
    hist = histories([-3.0, -2.0])
    stub = StubPipeline(levels=[np.nan, -2.0 + 0.02])

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)

    assert "T00.NS" not in out
    assert out["T01.NS"] == pytest.approx(+0.02)


def test_an_implausible_forecast_is_declined():
    """
    e^2 of excess return over 30 sessions is a defect, not a forecast. It is
    dropped rather than clipped: clipping to the boundary would put a large
    wrong number into the ranking at the top of the book, where it does the
    most damage to a long-short spread.
    """
    hist = histories([-3.0, -2.0])
    stub = StubPipeline(levels=[-3.0 + 5.0, -2.0 + 0.02])

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)

    assert "T00.NS" not in out
    assert set(out) == {"T01.NS"}


def test_a_declined_ticker_predicts_no_excess_return():
    """
    The adapter turns an abstention into 0.0, which is the `zero` floor's own
    claim. That is the point: an abstention can never flatter this comparator
    against the baseline it is measured on.
    """
    hist = histories([-3.0])
    stub = StubPipeline(levels=[np.nan])
    forecaster = C.Chronos2Forecaster(pipeline=stub)

    frame = pd.DataFrame({"T00.NS": [-3.0] * 300},
                         index=[f"D{i}" for i in range(300)])
    adapter = S.SeriesAdapter(forecaster=forecaster, series=frame, horizon=30)
    rows = pd.DataFrame({"date": ["D299"], "ticker": ["T00.NS"]})

    assert adapter.predict(rows)[0] == 0.0


def test_no_histories_makes_no_call():
    stub = StubPipeline(levels=[])
    assert C.Chronos2Forecaster(pipeline=stub).forecast({}, horizon=30) == {}
    assert stub.calls == []


# ── Silent truncation is an error, not a warning ──────────────────────────────


def test_a_history_longer_than_the_model_context_raises():
    """
    Chronos truncates to its own context and carries on. A run that actually
    measured 512 observations while the results table says 2,048 is a
    difference nothing records — which is exactly the class of defect the
    experiment-tracking config_hash exists to prevent elsewhere.
    """
    hist = histories([-3.0], length=900)
    stub = StubPipeline(levels=[-3.0], context_length=512)

    with pytest.raises(ValueError, match="exceeds"):
        C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)


def test_a_history_exactly_at_the_context_limit_is_allowed():
    hist = histories([-3.0], length=512)
    stub = StubPipeline(levels=[-3.0 + 0.03], context_length=512)

    out = C.Chronos2Forecaster(pipeline=stub).forecast(hist, horizon=30)
    assert out["T00.NS"] == pytest.approx(0.03)


# ── The cross-section must not be split ───────────────────────────────────────


def test_the_whole_cross_section_goes_in_one_batch():
    """
    `cross_learning` conditions each forecast on the others in its BATCH. The
    batch size must therefore be DERIVED from the cross-section, not fixed.
    Chronos' own default is 256, which is fine today at 95 tickers and changes
    meaning silently the day the universe grows past it — some dates
    conditioned on the whole cross-section, some on a fragment, and nothing in
    the output telling them apart.

    So this runs a cross-section wider than that default on purpose. An earlier
    version used 40 names and asserted `batch_size >= n_inputs`, which the
    hardcoded 256 satisfies; it passed against the very defect it was written
    to catch.
    """
    n = 300
    assert n > 256, "must exceed Chronos' default batch_size to mean anything"
    hist = histories(list(np.linspace(-4.0, -2.0, n)), length=60)
    stub = StubPipeline(levels=list(np.linspace(-4.0, -2.0, n)))

    C.Chronos2Forecaster(pipeline=stub, cross_learning=True).forecast(
        hist, horizon=30)

    call = stub.calls[0]
    assert call["n_inputs"] == n
    assert call["batch_size"] >= n


def test_cross_learning_reaches_the_model():
    hist = histories([-3.0])
    for flag in (False, True):
        stub = StubPipeline(levels=[-3.0])
        C.Chronos2Forecaster(pipeline=stub, cross_learning=flag).forecast(
            hist, horizon=30)
        assert stub.calls[0]["cross_learning"] is flag


def test_the_two_variants_have_distinct_names():
    """
    They are different models and share a folds/rows table. Two rows under one
    name is not a comparison.
    """
    plain = C.Chronos2Forecaster(pipeline=StubPipeline([]))
    crossed = C.Chronos2Forecaster(pipeline=StubPipeline([]), cross_learning=True)

    assert plain.name != crossed.name

    # Derived from the checkpoint, not hardcoded: a model swap must rename
    # the table rows with it, or a 120M run is recorded under the 28M name
    # and the two become indistinguishable in experiment_runs.
    stem = C.DEFAULT_MODEL_ID.rsplit("/", 1)[-1].replace("-", "")
    assert set(C.CHRONOS_VARIANTS) == {stem, f"{stem}_xl"}
    assert C.CHRONOS_VARIANTS[stem]["cross_learning"] is False
    assert C.CHRONOS_VARIANTS[f"{stem}_xl"]["cross_learning"] is True


def test_chronos_is_not_in_the_known_answer_set():
    """
    SERIES_BASELINES' contract is that every member's answer is known in
    advance — it is the calibration set that licenses reading a foundation
    model's row as the model rather than the plumbing. A model being calibrated
    cannot also be the calibration.
    """
    assert not any("chronos" in name for name in S.SERIES_BASELINES)


# ── torch must not reach Render or the daily job ──────────────────────────────


def test_importing_the_comparator_module_does_not_import_torch():
    """
    torch was removed in Phase 0 as the largest contributor to memory pressure
    on a free-tier instance that had already been OOM-killed. This module is
    reachable from pipeline.baselines, which the weekly job and the API both
    import; the torch import must stay inside load_pipeline().
    """
    import subprocess
    import sys

    code = (
        "import sys; import pipeline.chronos_forecaster, pipeline.baselines; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules "
        "if m.startswith('torch')); "
        "assert 'chronos' not in sys.modules; print('clean')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "clean" in proc.stdout


def test_chronos_requires_the_series_comparators():
    """
    Chronos reads the relative-price series and nothing else, so with_series
    False cannot mean anything with it on. Raised rather than dropped: a run
    asked for a foundation model that quietly returned six linear comparators
    reads as "it did not help".
    """
    from pipeline.baselines import compare_baselines

    with pytest.raises(ValueError, match="foundation-model"):
        compare_baselines(with_chronos=True, with_series=False)


def test_chronos_is_scored_when_asked_for(monkeypatch):
    """
    The guard above must not be so eager that the rows never appear. Runs
    against a stub forecaster rather than the real weights: this test is about
    the wiring, and an hour of inference would only obscure it.
    """
    import numpy as np

    import pipeline.baselines
    import pipeline.chronos_forecaster
    from tests.test_phase2_baselines import make_panel

    panel = make_panel(n_dates=800, n_tickers=20, seed=101)
    rng = np.random.default_rng(101)
    panel["close"] = 100.0 * np.exp(rng.normal(0, 0.01, len(panel)).cumsum())
    panel["benchmark_close"] = 20000.0

    class FastStub:
        def __init__(self, cross_learning=False, **kwargs):
            self.cross_learning = cross_learning
            self.name = "stub"

        def forecast(self, histories, horizon):
            return {t: 0.0 for t in histories}

    monkeypatch.setattr(pipeline.baselines, "load_panel", lambda **k: panel)
    monkeypatch.setattr(pipeline.chronos_forecaster, "Chronos2Forecaster", FastStub)

    comparison = pipeline.baselines.compare_baselines(
        with_pooled_xgb=False, with_chronos=True)

    names = {r["name"] for r in comparison.results}
    assert set(C.CHRONOS_VARIANTS).issubset(names), names


def test_the_missing_dependency_message_names_the_requirements_file():
    import sys

    C._PIPELINES.clear()
    saved = {k: sys.modules.get(k) for k in ("torch", "chronos")}
    for k in saved:
        sys.modules[k] = None                    # forces ImportError on import
    try:
        with pytest.raises(C.ChronosUnavailable, match="requirements-series.txt"):
            C.load_pipeline("autogluon/chronos-2-small")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        C._PIPELINES.clear()


# ── The real checkpoint ───────────────────────────────────────────────────────


def _require_torch(*extra: str) -> None:
    """
    Skip unless torch — and any named extra — is genuinely usable.

    Called INSIDE each test, never at module scope. An earlier version probed
    at import time, which pytest runs during COLLECTION: with torch mid-
    reinstall that pulled a half-built torch and chronos into sys.modules
    before a single test ran, and unrelated files across the suite then failed
    together while passing in isolation. Collection must import nothing heavy.

    find_spec and a bare import are both too weak on their own: a partially
    removed package still has a spec and still imports, as an empty shell. So
    touch something real.
    """
    import importlib

    try:
        torch = importlib.import_module("torch")
        torch.__version__
        torch.tensor([0.0])
        for name in extra:
            importlib.import_module(name)
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f"torch/{extra} not usable: {str(exc)[:80]}")


def test_the_real_checkpoint_emits_a_median_and_the_expected_shape():
    """
    Pins the two facts the index arithmetic depends on, against the actual
    weights rather than against the stub that was written to match them.
    """
    _require_torch("chronos")
    pipe = C.load_pipeline(C.DEFAULT_MODEL_ID)

    assert 0.5 in pipe.quantiles
    assert pipe.model_context_length >= S.DEFAULT_CONTEXT

    rng = np.random.default_rng(0)
    hist = [np.cumsum(rng.normal(0, 0.01, 300)) for _ in range(3)]
    out = pipe.predict(hist, prediction_length=30)

    assert len(out) == 3
    assert tuple(np.asarray(out[0]).shape) == (1, len(pipe.quantiles), 30)


def test_the_real_model_forecasts_a_move_on_the_scale_of_the_target():
    """
    A sanity bound, not an accuracy claim. target_excess_return over 30
    sessions runs to roughly +/-0.3; a model returning levels instead of moves
    would land near -3, and one reading the wrong quantile decade would be
    similarly far out. Anything inside this bound is merely not obviously
    broken.
    """
    _require_torch("chronos")
    rng = np.random.default_rng(7)
    hist = {f"T{i:02d}.NS": np.log(500.0 / 20000.0)
            + np.cumsum(rng.normal(0.0, 0.012, 400)) for i in range(8)}

    out = C.Chronos2Forecaster().forecast(hist, horizon=30)

    assert len(out) == 8
    assert all(np.isfinite(v) for v in out.values())
    assert max(abs(v) for v in out.values()) < 0.5


def test_the_pipeline_is_loaded_once_per_process():
    """
    panel_walk_forward builds a fresh estimator per fold and adapter_factory a
    fresh forecaster with it. Re-reading the weights five times is pure waste,
    and unlike a fitted model there is nothing fold-specific to reset —
    precisely because nothing is fitted.
    """
    _require_torch("chronos")
    first = C.load_pipeline(C.DEFAULT_MODEL_ID)
    second = C.load_pipeline(C.DEFAULT_MODEL_ID)
    assert first is second


# ── The GPU must not change the answer ────────────────────────────────────────


def test_the_pipeline_cache_is_keyed_by_device():
    """
    A cache keyed on the model alone hands a CPU-resident pipeline to a caller
    that asked for CUDA. Nothing errors; the run is simply slow for no visible
    reason, which is the hardest kind of performance bug to notice.
    """
    _require_torch("chronos")

    C._PIPELINES.clear()
    cpu = C.load_pipeline(C.DEFAULT_MODEL_ID, device="cpu")
    assert ("cpu" in str(k) for k in C._PIPELINES)
    again = C.load_pipeline(C.DEFAULT_MODEL_ID, device="cpu")
    assert again is cpu

    import torch
    if torch.cuda.is_available():
        gpu = C.load_pipeline(C.DEFAULT_MODEL_ID, device="cuda")
        assert gpu is not cpu, "cuda request returned the cpu pipeline"


def test_cpu_and_gpu_agree_to_float32_precision():
    """
    The tables recorded so far were all measured on CPU. If moving to a GPU
    changed the third significant figure, none of them would be comparable with
    anything measured afterwards — so the speedup has to be verified as a
    speedup and not as a different experiment.

    Measured at 4.8e-07 against predictions of order 5e-3, which is float32
    rounding. TF32 is disabled in series.configure_determinism for this reason.
    """
    _require_torch("chronos")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    rng = np.random.default_rng(3)
    hist = {f"T{i:02d}.NS": np.log(500.0 / 20000.0)
            + np.cumsum(rng.normal(0.0, 0.012, 512)) for i in range(8)}

    cpu = C.Chronos2Forecaster(device="cpu").forecast(hist, horizon=30)
    gpu = C.Chronos2Forecaster(device="cuda").forecast(hist, horizon=30)

    assert set(cpu) == set(gpu)
    worst = max(abs(cpu[k] - gpu[k]) for k in cpu)
    assert worst < 1e-5, f"cpu and gpu disagree by {worst:.2e}"
