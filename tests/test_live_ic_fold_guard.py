"""
The live gate must not grade a rank IC measured on one or two folds.

`pipeline.evaluation.compute_metrics` feeds `eval_rank_ic` and `eval_rank_ic_t`
to `agents.critic_agent.grade_evidence`. After Stage 0c it averaged the IC over
however many walk-forward folds carried an ordering and built the t-statistic
from ALL the rows, so an IC measured on one fold of five was graded as though
five folds backed it. Graded from the live table after the 2026-09-12 weekly
run: 19 tickers WEAK, 11 of them on a single fold (PNB.NS at IC +0.60, t
+4.78). Stage 0b had set MIN_FOLDS_FOR_ESTIMATE = 3 for the evidence track; it
never reached the live path.

Refusing the IC exposed a second defect. `_load_persisted_evaluation` read a
missing rank IC as "never evaluated" and discarded the whole row, conformal
calibration included, so the ticker's published price interval went with it.
24 live tickers were already in that state, 4 of them holding a Postgres NaN
rather than NULL.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agents.critic_agent import grade_evidence
from pipeline.evaluation import (
    MIN_FOLDS_FOR_ESTIMATE,
    compute_metrics,
    effective_sample_size,
    rank_ic,
)
from pipeline.model import MODEL_VERSION

ROWS_PER_FOLD = 300
HORIZON = 30


def _folds(n_scored: int, n_folds: int = 5, seed: int = 0):
    """``n_folds`` folds; the first ``n_scored`` carry a real ordering, the rest
    emit one constant each — the per-ticker tuner's training mean."""
    rng = np.random.default_rng(seed)
    yt, yp, fold = [], [], []
    for k in range(n_folds):
        y = rng.normal(0.0, 0.10, ROWS_PER_FOLD)
        p = (0.6 * y + rng.normal(0.0, 0.05, ROWS_PER_FOLD) if k < n_scored
             else np.full(ROWS_PER_FOLD, 0.004 * (k + 1)))
        yt.append(y)
        yp.append(p)
        fold.append(np.full(ROWS_PER_FOLD, k))
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(fold)


def _state(m: dict, hit_edge_pp: float) -> dict:
    """The gate's input as the daily path builds it. A NaN metric reaches the
    gate as None, because the write boundary stores NULL and the loader reads
    a non-finite value as missing. The hit-rate edge is pinned so the only
    thing that can differ between two states is the rank IC."""
    def finite(v):
        return v if v is not None and np.isfinite(v) else None
    return {"forecast_available": True,
            "eval_rank_ic": finite(m.get("rank_ic")),
            "eval_rank_ic_t": finite(m.get("rank_ic_t")),
            "eval_hit_rate": m["hit_rate"],
            "eval_baseline_hit_rate": m["hit_rate"] - hit_edge_pp,
            "eval_beats_naive": m["beats_naive_mae"]}


# ── the guard ─────────────────────────────────────────────────────────────────

def test_the_fold_threshold_is_one_constant_shared_with_the_evidence_track():
    from pipeline import evidence_shrinkage

    assert MIN_FOLDS_FOR_ESTIMATE == 3
    assert evidence_shrinkage.MIN_FOLDS_FOR_ESTIMATE == MIN_FOLDS_FOR_ESTIMATE, (
        "the live gate and the evidence track must refuse the same thin record")


def test_an_ic_from_fewer_than_three_scored_folds_is_not_measured():
    for n_scored in (1, 2):
        yt, yp, f = _folds(n_scored)
        # Not vacuous: the surviving fold carries a STRONG ordering, which is
        # exactly what the old code turned into a badge.
        assert rank_ic(yt[f == 0], yp[f == 0]) > 0.5

        m = compute_metrics(yt, yp, horizon=HORIZON, folds=f)
        assert m["n_folds_scored"] == n_scored
        assert math.isnan(m["rank_ic"]) and math.isnan(m["rank_ic_t"]), (
            f"an IC from {n_scored} scored fold(s) of 5 was reported as "
            f"{m['rank_ic']:+.3f} (t {m['rank_ic_t']:+.2f}); it is not a track "
            f"record and must not be measured")
        assert np.isfinite(m["hit_rate"]) and np.isfinite(m["mae"]), (
            "only the rank IC is refused; the other metrics are still measured")


def test_three_scored_folds_are_graded_on_their_own_rows():
    yt, yp, f = _folds(3)
    m = compute_metrics(yt, yp, horizon=HORIZON, folds=f)

    ic = float(np.mean([rank_ic(yt[f == k], yp[f == k]) for k in range(3)]))
    n_eff_scored = effective_sample_size(3 * ROWS_PER_FOLD, HORIZON)
    n_eff_all = effective_sample_size(5 * ROWS_PER_FOLD, HORIZON)

    assert m["n_folds_scored"] == 3
    assert m["rank_ic"] == pytest.approx(ic)
    assert m["rank_ic_t"] == pytest.approx(ic * math.sqrt(n_eff_scored - 1)), (
        "the t must be built from the rows the IC was measured on")
    assert m["rank_ic_t"] < ic * math.sqrt(n_eff_all - 1)
    assert m["n_effective"] == pytest.approx(round(n_eff_all, 1)), (
        "n_effective still describes the whole out-of-sample record")
    assert m["n_effective_ic"] == pytest.approx(round(n_eff_scored, 1))


def test_the_foldless_path_is_unchanged():
    """The baseline comparators score one block and pass no folds."""
    yt, yp, _ = _folds(3)
    m = compute_metrics(yt, yp, horizon=HORIZON)
    ic = rank_ic(yt, yp)
    assert m["rank_ic"] == pytest.approx(ic)
    assert m["rank_ic_t"] == pytest.approx(
        ic * math.sqrt(effective_sample_size(len(yt), HORIZON) - 1))
    assert m["n_folds_scored"] == 0


def test_a_one_fold_ic_can_no_longer_earn_a_badge():
    yt, yp, f = _folds(1)
    m = compute_metrics(yt, yp, horizon=HORIZON, folds=f)

    # What the gate was handed before the fix, from the SAME predictions.
    one_fold = rank_ic(yt[f == 0], yp[f == 0])
    before = dict(m, rank_ic=one_fold, rank_ic_t=one_fold * math.sqrt(
        effective_sample_size(len(yt), HORIZON) - 1))
    assert grade_evidence(_state(before, hit_edge_pp=5.0))[0] == "WEAK"

    grade, reasons = grade_evidence(_state(m, hit_edge_pp=5.0))
    assert grade == "INSUFFICIENT", (
        f"a one-fold IC still earned {grade}: {reasons}")
    assert not any("has not been through a weekly evaluation" in r
                   for r in reasons), (
        "the ticker WAS evaluated; the gate must say the IC was not measured, "
        "not that no evaluation exists")


# ── the loader and the write boundary ─────────────────────────────────────────

def _row(**overrides) -> pd.DataFrame:
    row = {
        "ticker": "TEST.NS", "eval_rank_ic": np.nan, "eval_rank_ic_t": np.nan,
        "eval_hit_rate": 58.2, "eval_baseline_hit_rate": 57.1,
        "eval_mae": 0.090, "eval_mae_naive": 0.095,
        "eval_n_oos": 1500, "eval_n_effective": 50.0, "eval_protocol": None,
        "conformal_quantile": 0.061, "conformal_coverage": 0.80,
        "conformal_n": 1500,
        "conformal_residuals": "[0.01, -0.02, 0.03, -0.04]",
        "model_version": MODEL_VERSION, "evaluated_at": "2026-09-12 07:39:25",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _load(frame: pd.DataFrame):
    import pipeline.model as model_mod

    original = model_mod.pd.read_sql
    model_mod.pd.read_sql = lambda *a, **kw: frame
    try:
        return model_mod._load_persisted_evaluation("TEST.NS")
    finally:
        model_mod.pd.read_sql = original


def test_a_ticker_with_no_rank_ic_keeps_its_evaluation_and_its_interval():
    # NaN is what pd.read_sql returns for both a NULL and a stored 'NaN'.
    loaded = _load(_row())
    assert loaded is not None, (
        "a ticker with a hit rate and a conformal calibration WAS evaluated; "
        "discarding the row drops its published price interval")
    assert loaded["rank_ic"] is None and loaded["rank_ic_t"] is None, (
        "a missing IC must reach the gate as None — as a NaN it would count as "
        "a check that ran and failed")
    assert loaded["hit_rate"] == pytest.approx(58.2)
    assert loaded["conformal_residuals"] is not None
    assert len(loaded["conformal_residuals"]) == 4


def test_a_daily_fit_placeholder_is_still_never_evaluated():
    placeholder = _row(eval_hit_rate=np.nan, eval_baseline_hit_rate=np.nan,
                       evaluated_at=None, conformal_residuals=None)
    assert _load(placeholder) is None


def test_a_nan_metric_is_written_as_null_not_as_a_postgres_nan():
    """Asserted at the function: SQLite turns a bound NaN into NULL on its own,
    so a round trip through the test database would launder the defect."""
    from pipeline.model import _evaluation_params

    params = _evaluation_params("TEST.NS", {
        "rank_ic": float("nan"), "rank_ic_t": np.float64("nan"),
        "hit_rate": 58.2, "majority_hit_rate": np.float64(57.1),
        "n_oos_predictions": 1500, "conformal_quantile": float("inf"),
    })
    assert params["ic"] is None and params["ic_t"] is None
    assert params["cq"] is None
    assert params["hit"] == pytest.approx(58.2)
    assert params["baseline"] == pytest.approx(57.1)
    assert params["n_oos"] == 1500
    assert not any(isinstance(v, np.generic) for v in params.values())
