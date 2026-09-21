"""
The thread pin, and the contract it exists to enforce.

P6 measured a result crossing its own pre-registered t = 2.0 threshold on
`OMP_NUM_THREADS` alone — 20 threads gave +2.405, 6 gave +1.987, and zero of
162,535 predictions matched. The mechanism is not last-decimal noise: XGBoost's
`hist` reduces gradient histograms in parallel, floating-point addition is not
associative, so a trial's inner-CV score depends on how the work was divided
and Optuna then selects a different winner.

The centrepiece here is therefore BEHAVIOURAL and runs in subprocesses: the
same fit, under two different `OMP_NUM_THREADS`, must produce bit-identical
predictions. Asserting only that `xgb_params()` contains `n_jobs` would pass
against a build that ignored it — and it nearly did, because `n_jobs=4` does
NOT survive a 2-thread environment.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap

import pytest

from pipeline.determinism import (
    THREADS_ENV_VAR,
    XGB_THREADS,
    environment_fingerprint,
    threads_are_pinned,
    xgb_params,
    xgb_threads,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FIT = textwrap.dedent(
    """
    import hashlib, os, sys
    sys.path.insert(0, {repo!r})
    import numpy as np
    from xgboost import XGBRegressor
    from pipeline.determinism import xgb_params

    rng = np.random.default_rng(0)
    X = rng.normal(size=(8000, 12))
    y = X[:, 0] * 0.3 + X[:, 1] * 0.2 + rng.normal(size=8000)
    model = XGBRegressor(**xgb_params(
        n_estimators=120, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, gamma=1.0, random_state=42))
    model.fit(X, y)
    pred = model.predict(X[:2000])
    print(hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest())
    """
)


def _fit_digest(omp_threads: str) -> str:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = omp_threads
    env.pop(THREADS_ENV_VAR, None)
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.run(
        [sys.executable, "-c", _FIT.format(repo=REPO)],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_a_fit_is_bit_identical_across_thread_counts():
    """
    THE CONTRACT. Two subprocesses, two thread environments, one answer.

    Both values are at or above `XGB_THREADS`, which is the range the pin can
    actually cover — `threads_are_pinned()` is what reports the case below it,
    and the test for that is separate.
    """
    if (os.cpu_count() or 1) < 4:
        pytest.skip("needs at least 4 usable cores to vary the thread count")

    wide = _fit_digest("20")
    narrow = _fit_digest(str(XGB_THREADS))

    assert wide == narrow, (
        f"the same fit produced {wide[:16]} at OMP_NUM_THREADS=20 and "
        f"{narrow[:16]} at {XGB_THREADS}; the thread pin is not in force, and "
        f"every number this project records is machine-dependent again")


def test_the_thread_count_is_pinned_and_a_caller_cannot_take_it_back():
    params = xgb_params()
    assert params["n_jobs"] == XGB_THREADS
    assert params["tree_method"] == "hist"

    # `n_jobs` is the one knob a caller does not get to set. A hyperparameter
    # dict carrying it — which any saved tuner output might — must not win.
    assert xgb_params(n_jobs=17)["n_jobs"] == XGB_THREADS
    assert xgb_params(**{"n_jobs": 0, "gamma": 2.0})["n_jobs"] == XGB_THREADS
    # Everything else still passes through.
    assert xgb_params(gamma=2.0, max_depth=3)["gamma"] == 2.0
    assert xgb_params(max_depth=3)["max_depth"] == 3


def test_the_pin_is_low_enough_for_the_smallest_machine_that_runs_this():
    """
    Two, because OpenMP caps the team at what is available — so a pin ABOVE
    the smallest machine's core count silently degrades there and reproduces
    nothing. Measured: `n_jobs=4` gave one answer at OMP_NUM_THREADS=20 and a
    different one at 2. This repo's own CI has four (it is public), so two is
    set by the smallest machine a number might be produced on, not by CI.
    """
    assert XGB_THREADS <= 2, (
        f"XGB_THREADS={XGB_THREADS} exceeds a 2-core machine; the pin would "
        f"be capped there and the results would not match this machine's")
    assert XGB_THREADS >= 1


def test_every_xgboost_in_the_pipeline_is_built_through_the_pin():
    """
    The call sites, checked through the objects they actually produce rather
    than by reading the source. A site that forgot the pin returns an estimator
    whose `n_jobs` is None or the machine's core count.
    """
    from pipeline.baselines import _pooled_xgb_factory
    from pipeline.model import _model_factory

    for name, estimator in [
        ("baselines._pooled_xgb_factory", _pooled_xgb_factory()),
        ("model._model_factory", _model_factory()()),
        ("model._model_factory(params)", _model_factory({"max_depth": 3})()),
    ]:
        assert estimator.get_params()["n_jobs"] == XGB_THREADS, (
            f"{name} was built without the thread pin")
        assert estimator.get_params()["tree_method"] == "hist", (
            f"{name} was built without a pinned tree_method; the default is "
            f"'auto', which picks a different builder by data size")


def test_the_pin_reports_when_it_cannot_hold(monkeypatch):
    """
    `n_jobs` is a request, not a guarantee. The one case the pin cannot cover
    is a machine that permits fewer threads than it asks for, and that must be
    detectable rather than silent.
    """
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    ok, why = threads_are_pinned()
    assert ok, why

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    ok, why = threads_are_pinned()
    assert not ok and "OMP_NUM_THREADS" in why

    monkeypatch.setenv("OMP_NUM_THREADS", str(XGB_THREADS))
    assert threads_are_pinned()[0]

    # A non-numeric value is not a reason to claim the pin is broken.
    monkeypatch.setenv("OMP_NUM_THREADS", "banana")
    assert threads_are_pinned()[0]


def test_an_override_is_recorded_rather_than_silent(monkeypatch):
    """
    The escape hatch exists for exploration, and the fingerprint is what stops
    a number produced under it from being mistaken for a default-settings one.
    """
    assert environment_fingerprint()["xgb_threads_overridden"] is False

    monkeypatch.setenv(THREADS_ENV_VAR, "8")
    assert xgb_threads() == 8
    assert xgb_params()["n_jobs"] == 8
    fingerprint = environment_fingerprint()
    assert fingerprint["xgb_threads"] == 8
    assert fingerprint["xgb_threads_overridden"] is True

    # Garbage falls back to the pin rather than to the machine default.
    monkeypatch.setenv(THREADS_ENV_VAR, "not-a-number")
    assert xgb_threads() == XGB_THREADS
    monkeypatch.setenv(THREADS_ENV_VAR, "0")
    assert xgb_threads() == XGB_THREADS


def test_the_environment_hash_moves_with_the_thread_count(monkeypatch):
    """
    The third hash beside config_hash and data_hash. Before it, a thread-count
    change was indistinguishable from a code change in `experiment_runs`.
    """
    from pipeline.tracking import environment_hash

    base, fingerprint = environment_hash()
    assert fingerprint["xgb_threads"] == XGB_THREADS
    for key in ("xgboost", "optuna", "numpy", "pandas", "sklearn"):
        assert fingerprint[key] not in (None, ""), key

    monkeypatch.setenv(THREADS_ENV_VAR, "8")
    moved, _ = environment_hash()
    assert moved != base, (
        "the environment hash did not move when the thread count did, so a "
        "run at a different thread count is not attributable")
