"""
Regression tests for the non-leakage defects found in the Phase 0 audit.

One test (or group) per finding. Each fails if the defect returns.
"""

import inspect
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


# ── F5: the archived ensemble must stay off the live path ─────────────────────

def test_lstm_and_meta_learner_are_not_imported_by_live_modules():
    """
    The LSTM never checkpointed (NaN validation targets), and the Ridge
    meta-learner was fitted and scored on the same rows. Both are archived;
    nothing on the live path may import them.
    """
    live_dirs = ["pipeline", "agents", "api", "app", "data"]
    offenders = []

    paths = [p for d in live_dirs for p in (REPO / d).rglob("*.py")]
    paths += [REPO / "main.py", REPO / "scheduler.py"]

    for path in paths:
        if "archived" in path.parts or not path.exists():
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        if "lstm_model" in source or "meta_learner" in source:
            offenders.append(str(path.relative_to(REPO)))

    assert not offenders, (
        f"archived modules referenced from the live path: {offenders}. "
        f"They may only return via experiment E3."
    )


def test_archived_modules_still_exist_with_rationale():
    assert (REPO / "pipeline" / "archived" / "lstm_model.py").exists()
    assert (REPO / "pipeline" / "archived" / "meta_learner.py").exists()
    readme = (REPO / "pipeline" / "archived" / "README.md").read_text(encoding="utf-8")
    assert "val_loss" in readme and "meta_learner" in readme


# ── F6: labels must be backfilled ─────────────────────────────────────────────

def test_signal_writes_are_upserts_not_append_only():
    """
    The old code inserted only dates absent from the table, so the trailing 30
    rows kept `target = NULL` forever and the labelled set never grew.
    """
    from pipeline import signals

    source = inspect.getsource(signals._upsert_signals)
    assert "DELETE FROM signals" in source, (
        "signal writes must clear the recomputed range before reinserting, "
        "otherwise targets are never backfilled (F6)"
    )

    store_source = inspect.getsource(signals.compute_and_store)
    assert "isin(existing_dates)" not in store_source
    assert "_upsert_signals" in store_source


def test_labelled_row_count_is_queryable():
    """The monotonicity assertion in the pipeline depends on this helper."""
    from pipeline.signals import count_labelled_rows
    assert callable(count_labelled_rows)


# ── F7: sentiment must not be a model feature ─────────────────────────────────

def test_sentiment_is_not_in_the_feature_list():
    """
    `sentiment_score` was 0.0 for every training row and non-zero only for the
    row being predicted. It returns as a feature only when a dated news archive
    exists.
    """
    from pipeline.model import FEATURES
    assert "sentiment_score" not in FEATURES
    assert "sentiment" not in FEATURES


# ── F8: the target must be a return, not a price level ────────────────────────

def test_target_is_an_excess_return():
    from pipeline.model import TARGET
    from pipeline.signals import TARGET_COLS

    assert TARGET == "target_excess_return"
    assert "target_excess_return" in TARGET_COLS


def test_excess_return_target_is_computed_against_a_benchmark():
    """Target must be the stock's forward log return minus the benchmark's."""
    from pipeline.signals import HORIZON_SESSIONS

    n = 200
    close = pd.Series(np.linspace(100, 120, n))    # stock:     +20%
    bench = pd.Series(np.linspace(200, 300, n))    # benchmark: +50%

    log_c, log_b = np.log(close), np.log(bench)
    target_return = log_c.shift(-HORIZON_SESSIONS) - log_c
    benchmark_return = log_b.shift(-HORIZON_SESSIONS) - log_b
    excess = target_return - benchmark_return

    assert excess.notna().sum() == n - HORIZON_SESSIONS
    # The stock rises more slowly than its benchmark, so excess must be negative.
    assert excess.dropna().mean() < 0

    # And a stock outrunning its benchmark must show positive excess.
    fast = pd.Series(np.linspace(100, 200, n))
    log_f = np.log(fast)
    excess_fast = (log_f.shift(-HORIZON_SESSIONS) - log_f) - benchmark_return
    assert excess_fast.dropna().mean() > 0


# ── F10: no silent flat-forecast fallback ─────────────────────────────────────

def test_failed_forecast_is_distinguishable_from_a_flat_forecast():
    """
    The old fallback set forecast_price = current_price with mape = 100, which
    is indistinguishable from a genuine no-change prediction.
    """
    from agents.forecasting_agent import _failed_forecast

    failed = _failed_forecast("boom")
    assert failed["forecast_available"] is False
    assert failed["forecast_price"] is None
    assert failed["forecast_direction"] == "UNAVAILABLE"
    assert failed["forecast_error"] == "boom"


def test_warm_path_helper_is_gone():
    """The KeyError-swallowing warm path was removed rather than patched."""
    from agents import forecasting_agent
    assert not hasattr(forecasting_agent, "_generate_forecast_from_existing")


def test_feature_names_match_between_training_and_serving():
    """
    F10 was a name mismatch: the serving path wrote `sentiment` while FEATURES
    expected `sentiment_score`. Any such divergence must fail loudly.
    """
    from pipeline.model import FEATURES
    from pipeline.signals import FEATURE_COLS

    macro_features = {"usdinr", "india_vix", "nifty_5d_return",
                      "nifty_20d_return", "fii_net_flow", "dii_net_flow"}
    assert set(FEATURES) == set(FEATURE_COLS) | macro_features


# ── F11 / F12: adjustment consistency and no backward fill ────────────────────

def test_ohlcv_ingestion_overwrites_rather_than_appends():
    from pipeline import fetch

    source = inspect.getsource(fetch._replace_ticker_history)
    assert "DELETE FROM ohlcv" in source, (
        "appending unseen dates splices two adjustment bases together (F11)"
    )


def test_adjustment_break_detector_exists():
    from pipeline.fetch import detect_adjustment_breaks
    assert callable(detect_adjustment_breaks)


def test_no_backward_fill_anywhere_on_the_live_path():
    """bfill() imports future values into the past (F12)."""
    offenders = []
    for directory in ["pipeline", "agents", "data"]:
        for path in (REPO / directory).rglob("*.py"):
            if "archived" in path.parts:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if ".bfill(" in stripped or 'method="bfill"' in stripped:
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}")

    assert not offenders, f"backward fill found at {offenders}"


# ── F13: earnings surprise must lag the announcement ──────────────────────────

def test_earnings_surprise_lands_after_the_announcement_date():
    from pipeline.signals import compute_earnings_surprise

    source = inspect.getsource(compute_earnings_surprise)
    assert "s > row[\"announced\"]" in source or "s > row['announced']" in source, (
        "earnings surprise must attach to the first session strictly AFTER the "
        "announcement; Indian results are commonly declared post-close (F13)"
    )


# ── F4: universe must not be selected on model output ─────────────────────────

def test_performance_based_universe_selector_is_deleted():
    assert not (REPO / "tools" / "select_top_50.py").exists(), (
        "select_top_50.py ranked stocks by composite score and kept the top 5 "
        "per sector, selecting the universe on the model's own accuracy (F4)"
    )


def test_universe_rule_references_no_model_output():
    from data.universe import UniverseRule

    rule = UniverseRule()
    fingerprint = rule.fingerprint().lower()
    for forbidden in ["mape", "accuracy", "composite", "score", "verdict",
                      "forecast", "confidence"]:
        assert forbidden not in fingerprint, (
            f"universe rule references '{forbidden}', which is model output"
        )


def test_ingest_universe_is_separate_from_screened_universe():
    """
    The liquidity screen reads the ohlcv table, which is only populated for
    tickers already fetched. Ingesting over the SCREENED universe collapses it
    to whatever happens to be in the database — a 100-name universe silently
    became 5 names during Phase 0 development.
    """
    from data.universe import get_ingest_universe, get_universe

    assert callable(get_ingest_universe)
    assert callable(get_universe)

    ingest_src = inspect.getsource(get_ingest_universe)
    assert "get_index_members" in ingest_src
    assert "liquidity" not in ingest_src.lower().split('"""')[-1]


def test_pipeline_fetches_over_membership_then_screens():
    """Ingestion must run over index membership, screening after."""
    for path in [REPO / "scheduler.py", REPO / "main.py"]:
        source = path.read_text(encoding="utf-8")
        if "fetch_and_store" not in source:
            continue
        assert "get_ingest_universe" in source, (
            f"{path.name} must ingest over get_ingest_universe(), not the "
            f"already-screened universe"
        )


def test_universe_bias_is_reported_not_hidden():
    from data.universe import describe_universe_bias

    bias = describe_universe_bias()
    assert "survivorship_bias" in bias
    assert "note" in bias and bias["note"]


# ── F9: the evidence gate, not the LLM, sets the verdict ──────────────────────

def test_evidence_grade_is_deterministic_and_needs_no_llm():
    from agents.critic_agent import grade_evidence

    strong = {"forecast_available": True, "eval_rank_ic": 0.08,
              "eval_rank_ic_t": 2.6, "eval_hit_rate": 58.0,
              "eval_baseline_hit_rate": 52.0, "eval_beats_naive": True}
    assert grade_evidence(strong)[0] == "STRONG"

    weak = dict(strong, eval_rank_ic_t=0.4)
    assert grade_evidence(weak)[0] == "WEAK"

    nothing = {"forecast_available": True, "eval_rank_ic": 0.001,
               "eval_rank_ic_t": 0.1, "eval_hit_rate": 48.0,
               "eval_baseline_hit_rate": 56.0}
    assert grade_evidence(nothing)[0] == "INSUFFICIENT"


def test_grade_is_capped_when_magnitude_has_no_skill():
    """A rupee target is shown, so magnitude skill is required for STRONG."""
    from agents.critic_agent import grade_evidence

    state = {"forecast_available": True, "eval_rank_ic": 0.15,
             "eval_rank_ic_t": 3.0, "eval_hit_rate": 60.0,
             "eval_baseline_hit_rate": 51.0, "eval_beats_naive": False}
    grade, reasons = grade_evidence(state)
    assert grade == "WEAK"
    assert any("capped at WEAK" in r for r in reasons)


def test_grade_is_insufficient_when_no_forecast_exists():
    from agents.critic_agent import grade_evidence
    grade, _ = grade_evidence({"forecast_available": False,
                               "forecast_error": "no history"})
    assert grade == "INSUFFICIENT"


def test_composite_score_is_gated_by_evidence():
    """
    The old score gave 75 of 100 points from leaked metrics that barely varied.
    A model that failed its held-out checks must now score zero regardless of
    how large a move it predicts.
    """
    from agents.graph import compute_composite_score

    strong = compute_composite_score(0.05, "STRONG", 0.70)
    weak = compute_composite_score(0.05, "WEAK", 0.70)
    insufficient = compute_composite_score(0.05, "INSUFFICIENT", 0.70)

    assert strong > weak > insufficient
    assert insufficient == 0.0
    assert compute_composite_score(0.50, "INSUFFICIENT", 0.99) == 0.0
    assert compute_composite_score(-0.05, "STRONG", 0.30) == 0.0
    assert 0.0 <= strong <= 100.0


# ── F15: bound SQL parameters ─────────────────────────────────────────────────

def test_no_fstring_sql_interpolation_of_tickers():
    """`/api/admin/run/{ticker}` accepts a user-supplied ticker."""
    offenders = []
    for directory in ["pipeline", "agents", "api", "data"]:
        for path in (REPO / directory).rglob("*.py"):
            if "archived" in path.parts:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                stripped = line.strip()
                if stripped.startswith("#") or "ALTER TABLE" in stripped:
                    continue
                lowered = stripped.lower()
                looks_like_sql = any(k in lowered for k in
                                     ("select ", "insert into", "delete from", "update "))
                if looks_like_sql and ("f\"" in stripped or "f'" in stripped) \
                        and "{ticker}" in stripped:
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}")

    assert not offenders, f"f-string SQL with a ticker at {offenders}"


def test_admin_key_comparison_is_constant_time():
    from api import dependencies

    source = inspect.getsource(dependencies.verify_api_key)
    assert "compare_digest" in source


# ── Conformal calibration ─────────────────────────────────────────────────────

def test_conformal_intervals_achieve_nominal_coverage():
    from pipeline.conformal import check_coverage, fit_conformal

    rng = np.random.default_rng(7)
    y_pred = rng.normal(size=2000) * 0.02
    y_true = y_pred + rng.normal(size=2000) * 0.05

    calibration = fit_conformal(y_true, y_pred, coverage=0.80)
    assert calibration is not None

    y_pred_new = rng.normal(size=2000) * 0.02
    y_true_new = y_pred_new + rng.normal(size=2000) * 0.05

    coverage = check_coverage(calibration, y_true_new, y_pred_new)
    assert coverage["well_calibrated"], (
        f"realised coverage {coverage['realised_coverage']:.3f} is more than "
        f"5pp from the nominal 0.80"
    )


def test_conformal_refuses_to_calibrate_on_too_few_residuals():
    from pipeline.conformal import fit_conformal
    assert fit_conformal(np.array([0.1, 0.2]), np.array([0.1, 0.15])) is None


def test_probability_is_monotonic_in_the_prediction():
    from pipeline.conformal import fit_conformal

    rng = np.random.default_rng(8)
    y_pred = rng.normal(size=1000) * 0.02
    y_true = y_pred + rng.normal(size=1000) * 0.05
    calibration = fit_conformal(y_true, y_pred)

    probs = [calibration.prob_positive(p) for p in [-0.10, -0.02, 0.0, 0.02, 0.10]]
    assert probs == sorted(probs)
    assert calibration.prob_positive(0.0) == pytest.approx(0.5, abs=0.1)


def test_price_view_states_its_benchmark_assumption():
    """
    The implied rupee target only holds if the benchmark is flat. Shipping it
    without that caveat would re-introduce the overclaiming Phase 0 removes.
    """
    from pipeline.conformal import to_price_view

    view = to_price_view(1000.0, 0.02, None)
    assert "assumption" in view
    assert "flat" in view["assumption"].lower()
    assert view["random_walk_price"] == 1000.0


# ── Numpy scalars must never reach the database driver ────────────────────────
#
# The daily pipeline ran green for every ticker while writing nothing. Every
# insert died with `psycopg2.errors.InvalidSchemaName: schema "np" does not
# exist` because a numpy scalar reached psycopg2, which adapts it with the
# float adapter and renders it via repr(). numpy 2.x changed that repr from
# "42.75" to "np.float64(42.75)", so the value landed in the SQL text as a
# schema-qualified name.
#
# It hid because SQLite accepts a float subclass without calling repr(), and
# because the numpy values only reach the write for tickers that already have
# a persisted weekly evaluation. The run that exposed it failed on exactly the
# 33 tickers the weekly job had reached and succeeded on the other 62.

def test_numpy_scalar_passes_an_isinstance_float_check():
    """
    Documents why a type guard would not have caught this. np.float64
    subclasses float, so the driver accepts it and fails later at the SQL
    parser instead of raising a clean adaptation error.
    """
    assert isinstance(np.float64(42.75), float)


def test_to_native_unwraps_numpy_scalars_and_leaves_everything_else():
    from data.db import to_native, to_native_params

    assert type(to_native(np.float64(1.5))) is float
    assert type(to_native(np.int64(3))) is int
    assert type(to_native(np.bool_(True))) is bool

    for passthrough in [None, "RELIANCE.NS", 1.5, 3, True]:
        assert to_native(passthrough) is passthrough

    cleaned = to_native_params({"a": np.float64(1.5), "b": None, "c": "x"})
    assert not any(isinstance(v, np.generic) for v in cleaned.values())


def test_save_forecast_to_db_sends_no_numpy_scalars_to_the_driver():
    """
    End-to-end guard on the write path: build a state carrying numpy scalars
    exactly where the real one does (the persisted evaluation is read with
    pd.read_sql; the conformal interval and probability are computed in
    numpy) and assert nothing numpy survives into the bind parameters.
    """
    import agents.graph as graph_mod
    import data.db as db_mod
    import data.tickers as tickers_mod

    captured = []

    class _FakeConn:
        def execute(self, _statement, params=None):
            captured.append(params or {})

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeEngine:
        def connect(self):
            return _FakeConn()

    original = (db_mod.get_engine, tickers_mod.get_company,
                tickers_mod.get_sector, tickers_mod.get_benchmark_name)
    db_mod.get_engine = lambda: _FakeEngine()
    tickers_mod.get_company = lambda t: "Test Co"
    tickers_mod.get_sector = lambda t: "Test Sector"
    tickers_mod.get_benchmark_name = lambda t: "NIFTY 50"

    try:
        graph_mod.save_forecast_to_db({
            "ticker": "TEST.NS",
            "forecast_available": True,
            "current_price": 100.0,
            "forecast_price": np.float64(101.5),
            "forecast_direction": "OUTPERFORM",
            "forecast_change_pct": np.float64(1.5),
            "pred_excess_return": np.float64(0.015),
            "interval_low": np.float64(90.0),
            "interval_high": np.float64(112.0),
            "interval_coverage": np.float64(0.8),
            "prob_outperform": np.float64(0.61),
            "random_walk_price": np.float64(100.0),
            "benchmark_ticker": "^NSEI",
            "eval_rank_ic": np.float64(-0.116661),
            "eval_hit_rate": np.float64(42.7513),
            "eval_baseline_hit_rate": np.float64(51.6402),
            "eval_beats_naive": np.bool_(False),
            "evidence_grade": "WEAK",
            "model_version": "test",
        })
    finally:
        (db_mod.get_engine, tickers_mod.get_company,
         tickers_mod.get_sector, tickers_mod.get_benchmark_name) = original

    assert len(captured) == 2, "expected a forecasts insert and a leaderboard upsert"
    for params in captured:
        offenders = {k: type(v).__name__ for k, v in params.items()
                     if isinstance(v, np.generic)}
        assert not offenders, f"numpy scalars reached the driver: {offenders}"


def test_persisted_evaluation_loader_returns_no_numpy_scalars():
    """
    _load_persisted_evaluation reads with pd.read_sql, whose values are numpy
    dtypes, and its output flows through the agent state straight back into an
    INSERT. It must convert at that boundary.
    """
    import pipeline.model as model_mod

    frame = pd.DataFrame([{
        "ticker": "TEST.NS",
        "eval_rank_ic": 0.117, "eval_rank_ic_t": 0.92,
        "eval_hit_rate": 57.3, "eval_baseline_hit_rate": 54.7,
        "eval_mae": 0.041, "eval_mae_naive": 0.043,
        "eval_n_oos": 945, "eval_n_effective": 31,
        "eval_protocol": None, "conformal_quantile": 0.05,
        "conformal_coverage": 0.8, "conformal_n": 500,
        "conformal_residuals": None, "evaluated_at": "2026-08-15 17:01:32",
    }])
    # Round-trips through numpy dtypes exactly as pd.read_sql would.
    assert isinstance(frame.iloc[0]["eval_rank_ic"], np.generic)

    original = model_mod.pd.read_sql
    model_mod.pd.read_sql = lambda *a, **kw: frame
    try:
        loaded = model_mod._load_persisted_evaluation("TEST.NS")
    finally:
        model_mod.pd.read_sql = original

    assert loaded is not None
    offenders = {k: type(v).__name__ for k, v in loaded.items()
                 if isinstance(v, np.generic)}
    assert not offenders, f"loader leaked numpy scalars: {offenders}"


# ── Departed tickers must not outrank the live universe ───────────────────────

def test_prune_leaderboard_removes_only_tickers_outside_the_universe():
    """
    save_forecast_to_db upserts and never deletes, so a name that leaves the
    index keeps its last row — carrying a pre-Phase-0 composite score with no
    evidence gate, which outranks every gated score written today. The live
    leaderboard was still headed by IDEA.NS and SAIL.NS months after both left
    the NIFTY 100.
    """
    import agents.graph as graph_mod
    import data.db as db_mod

    engine = _leaderboard_fixture(
        ["RELIANCE.NS", "WIPRO.NS", "IDEA.NS", "SAIL.NS"]
    )

    original = db_mod.get_engine
    db_mod.get_engine = lambda: engine
    try:
        # 2 of 4 rows leave, which exceeds the default 25% guard, so this call
        # states its own tolerance rather than relying on the default.
        removed = graph_mod.prune_leaderboard(
            ["RELIANCE.NS", "WIPRO.NS"], max_fraction=1.0)
        # An empty universe must never trigger a delete — that would wipe the
        # whole leaderboard on a bad universe fetch.
        empty = graph_mod.prune_leaderboard([], max_fraction=1.0)
    finally:
        db_mod.get_engine = original

    assert removed == 2
    assert empty == 0
    assert _tickers(engine) == ["RELIANCE.NS", "WIPRO.NS"]


def test_prune_leaderboard_refuses_to_delete_most_of_the_table():
    """
    A short universe is not evidence of a mass delisting.

    get_universe() applies a liquidity floor and a listing-history floor over
    freshly fetched OHLCV, so a partial yfinance response or a half-written
    ohlcv table produces a SHORT universe rather than an empty one. Guarding
    only against the empty case leaves the far likelier failure wide open:
    every healthy name missing from a truncated universe looks exactly like a
    departure, and the prune would delete the live leaderboard on the strength
    of a bad download.
    """
    import agents.graph as graph_mod
    import data.db as db_mod

    universe = [f"T{i}.NS" for i in range(20)]
    engine = _leaderboard_fixture(universe)

    original = db_mod.get_engine
    db_mod.get_engine = lambda: engine
    try:
        # Only 3 of 20 names survive the screen — 85% of the table would go.
        removed = graph_mod.prune_leaderboard(universe[:3])
    finally:
        db_mod.get_engine = original

    assert removed == 0
    assert len(_tickers(engine)) == 20, "a bad universe must not empty the table"


def _leaderboard_fixture(tickers):
    """An in-memory leaderboard table holding one row per ticker."""
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite://")          # single shared connection
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE leaderboard "
                          "(ticker TEXT PRIMARY KEY, composite_score REAL)"))
        for i, ticker in enumerate(tickers):
            conn.execute(text("INSERT INTO leaderboard VALUES (:t, :s)"),
                         {"t": ticker, "s": float(i)})
        conn.commit()
    return engine


def _tickers(engine):
    from sqlalchemy import text

    with engine.connect() as conn:
        return sorted(r[0] for r in
                      conn.execute(text("SELECT ticker FROM leaderboard")))


def test_daily_job_prunes_the_leaderboard_after_forecasting():
    import scheduler

    source = inspect.getsource(scheduler.run_pipeline_job)
    assert "prune_leaderboard(universe)" in source


def test_leaderboard_total_counts_matches_not_page_size():
    """
    `total` must report how many rows matched, not how many were returned.
    Returning len(entries) made it a restatement of `limit`, hiding the
    difference between a leaderboard holding 5 rows and one holding 500.
    """
    source = (REPO / "api" / "routers" / "leaderboard.py").read_text(encoding="utf-8")

    assert "total=len(entries)" not in source
    assert "total_matching = len(df)" in source
    # The count must be taken before the page is sliced.
    assert source.index("total_matching = len(df)") < source.index("df.head(limit)")


# ── A composite of 0.0 must say which kind of zero it is ──────────────────────

def test_score_basis_separates_no_evidence_from_a_bearish_forecast():
    """
    85 of 95 leaderboard rows scored exactly 0.0 on 2026-08-15, for unrelated
    reasons that the number alone could not distinguish. compute_composite_score
    floors both of its components, so a stock the model actively predicted would
    UNDERPERFORM — with evidence in hand and real conviction — landed on the
    same 0.0 as a stock the weekly evaluation had never reached. Sorted by
    score they formed one undifferentiated block, so the single number a reader
    ranks on silently conflated "no view" with "negative view".
    """
    from agents.graph import classify_score_basis, compute_composite_score

    bearish = dict(pred_excess_return=-0.04, evidence_grade="WEAK",
                   prob_outperform=0.31)
    unevaluated = dict(pred_excess_return=0.06, evidence_grade="INSUFFICIENT",
                       prob_outperform=0.62)

    # The defect: both score zero and are indistinguishable by score alone.
    assert compute_composite_score(**bearish) == 0.0
    assert compute_composite_score(**unevaluated) == 0.0

    # The fix: the reason is recorded, so they are distinguishable.
    assert classify_score_basis(**bearish) == "NOT_LONG"
    assert classify_score_basis(**unevaluated) == "NO_EVIDENCE"

    assert classify_score_basis(pred_excess_return=None,
                                evidence_grade="WEAK",
                                prob_outperform=None) == "NO_FORECAST"

    ranked = dict(pred_excess_return=0.05, evidence_grade="STRONG",
                  prob_outperform=0.70)
    assert compute_composite_score(**ranked) > 0
    assert classify_score_basis(**ranked) == "RANKED"

    # Flags can drive a genuine long signal to zero; that is a fourth reason.
    assert classify_score_basis(**ranked, n_flags=20) == "FLAGGED_OUT"


def test_score_basis_is_persisted_to_both_tables():
    """A basis that never reaches the database explains nothing to a reader."""
    import agents.graph as graph_mod

    source = inspect.getsource(graph_mod.save_forecast_to_db)
    assert '"score_basis": score_basis' in source

    db_source = (REPO / "data" / "db.py").read_text(encoding="utf-8")
    assert '"score_basis"' in db_source, "column must exist in the schema migration"

    schema = (REPO / "api" / "schemas" / "leaderboard.py").read_text(encoding="utf-8")
    assert "score_basis" in schema, "the API must expose it or nothing changed"


# ── Sentiment must not report a reading it never took ─────────────────────────

def test_finbert_is_gone_from_the_live_path():
    """
    FinBERT needs torch, and torch was removed in Phase 0. The loader therefore
    raised on every production run ("name 'torch' is not defined"), scored
    nothing, and returned before a single headline was stored — while the
    dashboard went on rendering a sentiment gauge fed by a hardcoded 0.0.
    """
    source = (REPO / "pipeline" / "sentiment.py").read_text(encoding="utf-8")

    assert "get_finbert" not in source
    assert "ProsusAI/finbert" not in source
    assert "from transformers import" not in source

    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    declared = [line.split("#")[0].strip() for line in requirements.splitlines()]
    assert "transformers" not in declared, "dependency exists only for FinBERT"


def test_aggregate_sentiment_is_none_when_nothing_is_scored():
    """
    None means "no measurement"; 0.0 means "measured, and it balanced out".
    Returning 0.0 for the first is what let the UI show NEUTRAL for every stock
    in the universe, indefinitely, without a scorer running at all.
    """
    from sqlalchemy import create_engine, text

    import pipeline.sentiment as sent_mod

    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE sentiment (date TEXT, ticker TEXT, "
                          "headline TEXT, sentiment_label TEXT, "
                          "sentiment_score REAL)"))
        conn.execute(text("INSERT INTO sentiment VALUES "
                          "('2026-08-15', 'RELIANCE.NS', 'a', 'unscored', NULL)"))
        conn.execute(text("INSERT INTO sentiment VALUES "
                          "('2026-08-15', 'TCS.NS', 'b', 'positive', 0.9)"))
        conn.execute(text("INSERT INTO sentiment VALUES "
                          "('2026-08-15', 'TCS.NS', 'c', 'unscored', NULL)"))
        conn.commit()

    original = sent_mod.get_engine
    sent_mod.get_engine = lambda: engine
    try:
        unscored = sent_mod.get_aggregate_sentiment("RELIANCE.NS")
        mixed = sent_mod.get_aggregate_sentiment("TCS.NS")
        absent = sent_mod.get_aggregate_sentiment("NOSUCH.NS")
    finally:
        sent_mod.get_engine = original

    assert unscored is None
    assert absent is None
    # Unscored rows are ignored rather than counted as neutral, which would
    # otherwise drag a real reading toward zero.
    assert mixed == pytest.approx(0.9)


def test_sentiment_view_does_not_plot_a_gauge_without_a_reading():
    """A needle parked mid-dial reads as a measured neutral, not as missing."""
    source = (REPO / "app" / "components" / "sentiment_view.py").read_text(
        encoding="utf-8")

    assert "if sentiment_score is None:" in source
    assert "NOT SCORED" in source


# ── The evidence gate must need more than one weak correlation ────────────────

def _evidence_state(ic, ic_t, hit, baseline, beats_naive=True):
    return {
        "forecast_available": True,
        "eval_rank_ic": ic,
        "eval_rank_ic_t": ic_t,
        "eval_hit_rate": hit,
        "eval_baseline_hit_rate": baseline,
        "eval_beats_naive": beats_naive,
    }


def test_weak_evidence_requires_two_independent_checks():
    """
    WEAK used to need ONE of three checks, so the +0.02 rank-IC floor alone
    bought the badge. BEL.NS ranked 3rd on the live leaderboard with a rank IC
    of +0.049 while its hit rate sat 4.7pp BELOW the majority-class baseline,
    its IC t-statistic was +0.39, and its error was worse than a random walk's.
    """
    from agents.critic_agent import grade_evidence

    # BEL.NS, 2026-08-15. Passes the IC floor and nothing else.
    bel, _ = grade_evidence(_evidence_state(
        ic=0.049, ic_t=0.39, hit=54.9, baseline=59.5, beats_naive=False))
    assert bel == "INSUFFICIENT", "one weak correlation must not earn a badge"

    # CUMMINSIND.NS — same shape, 6.4pp below its baseline.
    cummins, _ = grade_evidence(_evidence_state(
        ic=0.083, ic_t=0.66, hit=50.6, baseline=57.0, beats_naive=False))
    assert cummins == "INSUFFICIENT"

    # DMART.NS — positive IC AND a 7.3pp hit-rate edge. Two checks, so WEAK.
    dmart, _ = grade_evidence(_evidence_state(
        ic=0.193, ic_t=1.4, hit=59.7, baseline=52.4))
    assert dmart == "WEAK"

    # Tying the baseline exactly is not an edge (ADANIENT.NS, +0.0pp).
    tied, _ = grade_evidence(_evidence_state(
        ic=0.092, ic_t=1.1, hit=54.9, baseline=54.9))
    assert tied == "INSUFFICIENT"


def test_strong_requires_every_check_to_have_actually_run():
    """
    Grading on `passed == checks` alone hands STRONG to a ticker whose single
    available metric happened to clear its floor — a statement about missing
    data, not about skill.
    """
    from agents.critic_agent import grade_evidence

    only_ic, _ = grade_evidence(_evidence_state(
        ic=0.30, ic_t=None, hit=None, baseline=None))
    assert only_ic != "STRONG"

    everything, _ = grade_evidence(_evidence_state(
        ic=0.30, ic_t=3.1, hit=61.0, baseline=52.0))
    assert everything == "STRONG"

    # ...unless the magnitude is worse than a random walk, which still caps it.
    capped, _ = grade_evidence(_evidence_state(
        ic=0.30, ic_t=3.1, hit=61.0, baseline=52.0, beats_naive=False))
    assert capped == "WEAK"


# ── The row guard must match what the splitter actually needs ─────────────────

def test_evaluation_row_guard_is_derived_from_the_splitter():
    """
    Every guard read `len(df) < 350` while PurgedWalkForward was built with
    min_train=500, and split() returns nothing when `n_samples - min_train <=
    0`. Tickers holding 350-500 rows therefore cleared the guard and produced
    zero predictions — 17 of 95 on 2026-08-15, reported as "no out-of-sample
    predictions" as though it were a property of the data.
    """
    from pipeline.evaluation import PurgedWalkForward
    from pipeline.model import (
        EVAL_MIN_TRAIN, EVAL_N_FOLDS, MIN_ROWS_FOR_EVALUATION,
        MIN_ROWS_FOR_FORECAST,
    )
    from pipeline.signals import HORIZON_SESSIONS

    splitter = PurgedWalkForward(n_folds=EVAL_N_FOLDS, horizon=HORIZON_SESSIONS,
                                 embargo=HORIZON_SESSIONS,
                                 min_train=EVAL_MIN_TRAIN)

    assert MIN_ROWS_FOR_EVALUATION > EVAL_MIN_TRAIN

    # The old dead zone: enough rows to pass the forecast floor, not enough to
    # yield a single fold.
    assert list(splitter.split(MIN_ROWS_FOR_FORECAST)) == []
    assert list(splitter.split(EVAL_MIN_TRAIN)) == []

    # At the derived guard, every fold covers a non-empty test window.
    folds = list(splitter.split(MIN_ROWS_FOR_EVALUATION))
    assert len(folds) == EVAL_N_FOLDS
    assert all(len(test) > 0 for _, test in folds)


def test_no_hardcoded_row_guard_survives_in_the_model_module():
    """The literal must be gone from the code, not merely from one call site."""
    source = (REPO / "pipeline" / "model.py").read_text(encoding="utf-8")

    offenders = [
        f"{lineno}: {line.strip()}"
        for lineno, line in enumerate(source.splitlines(), 1)
        if "len(df) < 350" in line and not line.strip().startswith("#")
    ]
    assert not offenders, (
        f"row guards must be derived from the splitter, not restated as "
        f"literals: {offenders}"
    )


# ── The weekly job must not depend on the daily job having run ────────────────

def test_weekly_job_ingests_its_own_data_before_evaluating():
    """
    The weekly job evaluated straight out of the `signals` table, which only
    the DAILY job writes. On 2026-08-15 it read a half-built table, refused 62
    of 95 tickers for "insufficient history" — INFY, TCS, RELIANCE, HDFCBANK
    among them — and reported an infrastructure race as a data-quality verdict.
    """
    import scheduler

    source = inspect.getsource(scheduler.run_weekly_evaluation_job)

    # Must be CALLED, not merely imported — the import alone satisfies a
    # substring check while the pipeline step is gone.
    assert "fetch_and_store(tickers=" in source, "weekly run must refresh OHLCV itself"
    assert "compute_and_store(tickers=" in source, (
        "weekly run must recompute signals itself"
    )

    # And it must do so BEFORE evaluating, or the ordering bug simply moves.
    assert source.index("compute_and_store(tickers=") < source.index(
        "evaluate_and_persist_universe("
    )
    assert source.index("fetch_and_store(tickers=") < source.index(
        "compute_and_store(tickers="
    )


# ── LLM budget: choose which tickers get a written narrative ──────────────────

def test_default_groq_model_has_the_larger_free_tier_budget():
    """
    openai/gpt-oss-20b allows 200k tokens/day; the daily job makes two calls
    per ticker across ~95 tickers and exhausted it partway through the
    2026-08-15 run. llama-3.1-8b-instant allows 500k/day on the same tier.
    """
    from agents.llm import DEFAULT_GROQ_MODEL

    assert DEFAULT_GROQ_MODEL == "llama-3.1-8b-instant"

    # And the model must be configured in exactly one place.
    for module in ["critic_agent.py", "forecasting_agent.py"]:
        source = (REPO / "agents" / module).read_text(encoding="utf-8")
        assert "gpt-oss" not in source, f"{module} still hardcodes a model name"


def test_narrative_is_written_only_for_tickers_that_can_rank():
    """
    Without a gate the token cap chose which stocks got a written narrative by
    arrival order — the tail of the alphabet logged "narrative generation
    failed". A row cannot rank without a persisted evaluation and a positive
    predicted excess return, and a narrative is only read beside a ranked row.
    """
    from agents.forecasting_agent import _deserves_a_written_narrative

    rankable = {"eval_evaluated_at": "2026-08-15 17:02:11",
                "pred_excess_return": 0.031}
    assert _deserves_a_written_narrative(rankable) is True

    # No evidence -> composite is gated to 0.0 regardless of the prediction.
    assert _deserves_a_written_narrative(
        {"eval_evaluated_at": None, "pred_excess_return": 0.031}) is False

    # Predicted underperformer -> both score components floor at zero.
    assert _deserves_a_written_narrative(
        {"eval_evaluated_at": "2026-08-15 17:02:11",
         "pred_excess_return": -0.02}) is False

    assert _deserves_a_written_narrative(
        {"eval_evaluated_at": "2026-08-15 17:02:11",
         "pred_excess_return": None}) is False


# ── A missing benchmark must never erase a ticker's labels ────────────────────

def test_missing_benchmark_skips_the_ticker_instead_of_nulling_its_target():
    """
    The excess-return target is `stock return - benchmark return`, so an index
    that resolves to nothing NULLs the target for every row of every stock
    mapped to it — and _upsert_signals DELETEs the recomputed range before
    reinserting, so writing that frame ERASES existing labels.

    On 2026-08-16 ^CNXAUTO, ^CNXINFRA and ^CNXREALTY came back unusable in a
    single run and 22 tickers went from ~2,390 labelled rows to 0. All three
    served full history again minutes later. Returning None keeps the old rows.
    """
    import pipeline.signals as sig

    dates = pd.date_range("2015-01-01", periods=400, freq="B").strftime("%Y-%m-%d")
    ohlcv = pd.DataFrame({
        "date": dates,
        "ticker": "MARUTI.NS",
        "open": np.linspace(100, 200, 400),
        "high": np.linspace(101, 202, 400),
        "low": np.linspace(99, 198, 400),
        "close": np.linspace(100, 200, 400),
        "adj_close": np.linspace(100, 200, 400),
        "volume": np.full(400, 1_000_000.0),
    })

    empty = pd.DataFrame(columns=["date", "benchmark_close"])
    original = sig.get_benchmark_series
    sig.get_benchmark_series = lambda *a, **k: empty
    try:
        frame = sig.compute_signals_frame("MARUTI.NS", ohlcv)
    finally:
        sig.get_benchmark_series = original

    assert frame is None, (
        "a frame whose target is null on every row must never reach the writer"
    )


def test_benchmark_fetch_retries_and_reports_an_unusable_response():
    """
    The old code raised only on an outright empty response, so a frame that
    arrived non-empty but cleaned down to nothing fell through the SUCCESS path
    and cached an empty result with no message at all — which is why the
    2026-08-16 log contains no benchmark error despite three indices failing.
    """
    import pipeline.signals as sig

    source = inspect.getsource(sig.get_benchmark_series)
    assert "BENCHMARK_FETCH_ATTEMPTS" in source, "a transient miss must be retried"
    assert "cleaned.empty" in source, (
        "a response that cleans down to nothing is a failure, not a result"
    )

    calls = {"n": 0}

    def _always_fails(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("network")

    original_dl, original_sleep = sig.yf.download, sig.time.sleep
    sig.yf.download = _always_fails
    sig.time.sleep = lambda *a, **k: None
    sig._benchmark_cache.pop("^CNXAUTO", None)
    try:
        out = sig.get_benchmark_series("^CNXAUTO")
    finally:
        sig.yf.download, sig.time.sleep = original_dl, original_sleep
        sig._benchmark_cache.pop("^CNXAUTO", None)

    assert out.empty
    assert calls["n"] == sig.BENCHMARK_FETCH_ATTEMPTS


def test_weekly_job_aborts_if_recomputing_signals_destroys_labels():
    """
    The daily job has carried this check since F6. The weekly job was given the
    same signal-writing power without it, and promptly wrote null targets over
    22 tickers and evaluated anyway.
    """
    import scheduler

    source = inspect.getsource(scheduler.run_weekly_evaluation_job)

    assert "count_labelled_rows()" in source
    assert "labelled_after < labelled_before" in source
    # The check must sit between the write and the evaluation to be worth
    # anything.
    assert source.index("compute_and_store(tickers=") < source.index(
        "labelled_after < labelled_before"
    ) < source.index("evaluate_and_persist_universe(")


# ── Conviction must not outvote the point forecast ────────────────────────────

def test_conviction_requires_the_point_forecast_to_agree():
    """
    The composite claimed to rank long candidates only, and did not. PNB.NS
    ranked THIRD on the live leaderboard on 2026-08-17 while forecasting a
    1.69% underperformance: signal floored to 0.00 as designed, but conviction
    independently collected 10.75 points from prob_outperform=0.567 and 0.38
    survived the flag deduction.

    The disagreement is real — prob_positive(-0.0169) is the share of
    calibration residuals above +0.0169, so >0.5 means the model is biased low
    — but a row whose point forecast and calibrated probability point opposite
    ways is not a ranking signal, and the score must not quietly side with
    whichever half scores higher.
    """
    from agents.graph import _score_parts, classify_score_basis, compute_composite_score

    # PNB.NS, exactly as served.
    signal, conviction = _score_parts(-0.0169213, 0.567196)
    assert signal == 0.0
    assert conviction == 0.0, "a predicted decline must earn no conviction points"
    assert compute_composite_score(-0.0169213, "WEAK", 0.567196, 1) == 0.0
    assert classify_score_basis(-0.0169213, "WEAK", 0.567196, 1) == "NOT_LONG"

    # A genuine long candidate is untouched (CANBK.NS).
    signal, conviction = _score_parts(0.0295344, 0.753968)
    assert signal > 0 and conviction == 40.0
    assert compute_composite_score(0.0295344, "WEAK", 0.753968, 1) == 23.86

    # The narrative gate and the score must now agree on what "rankable" means:
    # neither admits a non-positive predicted excess return.
    from agents.forecasting_agent import _deserves_a_written_narrative

    for pred in (-0.02, 0.0):
        assert compute_composite_score(pred, "STRONG", 0.99) == 0.0
        assert _deserves_a_written_narrative(
            {"eval_evaluated_at": "2026-08-16", "pred_excess_return": pred}) is False


# ── Rank must not invent an ordering the score cannot support ─────────────────

def test_tied_rows_share_a_rank():
    """
    93 of 95 rows share composite_score 0.0 under the 2-of-3 gate, and
    positional numbering handed them ranks 3 through 95 from whatever order
    pandas left them in — publishing "rank 47" as a fact about a stock the
    score cannot separate from 92 others.
    """
    from sqlalchemy import create_engine, text

    import api.routers.leaderboard as lb

    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE leaderboard (ticker TEXT, company TEXT, sector TEXT, "
            "composite_score REAL, score_basis TEXT, upside_pct REAL, "
            "critic_verdict TEXT, forecast_confidence TEXT, last_updated TEXT)"))
        rows = [("CANBK.NS", 23.86, "RANKED"), ("ADANIPOWER.NS", 13.64, "RANKED")]
        rows += [(f"Z{i}.NS", 0.0, "NO_EVIDENCE") for i in range(6)]
        for ticker, score, basis in rows:
            conn.execute(
                text("INSERT INTO leaderboard VALUES (:t, :t, 'S', :s, :b, 1.0, "
                     "'REJECTED', 'INSUFFICIENT', '2026-08-17')"),
                {"t": ticker, "s": score, "b": basis})
        conn.commit()

    original = lb.get_engine
    lb.get_engine = lambda: engine
    try:
        # Every argument is passed explicitly: calling the endpoint outside a
        # request leaves FastAPI's Query() defaults unresolved.
        response = lb.get_leaderboard(sector=None, verdict=None, evidence=None,
                                      sort_by="composite_score", limit=50)
    finally:
        lb.get_engine = original

    ranks = [e.rank for e in response.entries]
    assert response.total == 8
    assert ranks[:2] == [1, 2], "the two scoring rows rank normally"
    assert ranks[2:] == [3] * 6, (
        f"tied rows must share a rank, got {ranks[2:]}"
    )
    assert max(ranks) == 3, "no row may be numbered past the last real ordering"


# ── /api/stocks latency: the universe screen must not be N+1 ──────────────────

AS_OF = "2026-08-18"
TEST_RULE_KWARGS = dict(min_listing_days=10, liquidity_window=5,
                        liquidity_floor_inr=1000)


def _universe_fixture(liquid_fillers: int = 0):
    """
    An in-memory ohlcv + index_membership pair with one ticker per outcome.

    Dates run consecutively up to AS_OF so the liquidity window (as_of minus
    2 x liquidity_window days) captures the recent rows.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite://")          # single shared connection
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE ohlcv (date TEXT, ticker TEXT, "
                          "close REAL, volume REAL)"))
        conn.execute(text("CREATE TABLE index_membership (ticker TEXT, "
                          "index_name TEXT, effective_from TEXT, "
                          "effective_to TEXT, company TEXT, industry TEXT, "
                          "source TEXT)"))

        def member(ticker):
            conn.execute(
                text("INSERT INTO index_membership VALUES (:t, 'NIFTY100', "
                     "'2020-01-01', '9999-12-31', :t, 'X', 'test')"),
                {"t": ticker})

        def bars(ticker, n, close, volume):
            for i in range(n):
                day = (date.fromisoformat(AS_OF) - timedelta(days=i)).isoformat()
                conn.execute(
                    text("INSERT INTO ohlcv VALUES (:d, :t, :c, :v)"),
                    {"d": day, "t": ticker, "c": close, "v": volume})

        member("LIQUID.NS");   bars("LIQUID.NS",   12, 100.0, 100.0)
        member("ILLIQUID.NS"); bars("ILLIQUID.NS", 12,   1.0,   1.0)
        member("SHORT.NS");    bars("SHORT.NS",     5, 100.0, 100.0)
        member("NODATA.NS")    # in the index, never ingested

        for i in range(liquid_fillers):
            t = f"FILL{i}.NS"
            member(t); bars(t, 12, 100.0, 100.0)

        conn.commit()
    return engine


def _screen(engine, rule):
    """Runs get_universe against `engine`, counting SELECTs against ohlcv."""
    from sqlalchemy import event
    import data.universe as uni

    ohlcv_queries = []

    def record(conn, cursor, statement, params, context, executemany):
        if "ohlcv" in statement.lower():
            ohlcv_queries.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    original = uni.get_engine
    uni.get_engine = lambda: engine
    try:
        result = uni.get_universe(as_of=AS_OF, rule=rule)
    finally:
        uni.get_engine = original
        event.remove(engine, "before_cursor_execute", record)

    return result, ohlcv_queries


def test_universe_screen_decisions_are_unchanged():
    """
    The bulk-query rewrite must screen exactly as the per-ticker loop did:
    liquid + long history is kept, and each rejection reason still rejects.
    """
    from data.universe import UniverseRule

    rule = UniverseRule(**TEST_RULE_KWARGS)
    universe, _ = _screen(_universe_fixture(), rule)

    assert universe == ["LIQUID.NS"], (
        f"expected only the liquid, long-history name; got {universe}")


def test_universe_screen_does_not_query_per_ticker():
    """
    get_universe() ran two queries PER TICKER inside a Python loop — ~200
    sequential round trips to Supabase for a 100-name index, on every call.
    /api/stocks calls this on every request and was measured at 51 seconds
    against a warm Render instance, which is what made the dashboard slow to
    open. The query count must not scale with the size of the universe.
    """
    from data.universe import UniverseRule

    rule = UniverseRule(**TEST_RULE_KWARGS)

    small, small_queries = _screen(_universe_fixture(liquid_fillers=1), rule)
    large, large_queries = _screen(_universe_fixture(liquid_fillers=40), rule)

    assert len(small) == 2 and len(large) == 41, "fixture sanity"
    assert len(large_queries) == len(small_queries), (
        f"query count scales with universe size: {len(small_queries)} queries "
        f"for 5 tickers vs {len(large_queries)} for 44 — the N+1 is back")
    assert len(large_queries) <= 3, (
        f"the whole screen must take a constant handful of queries, "
        f"got {len(large_queries)}")
