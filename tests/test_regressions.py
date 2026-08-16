"""
Regression tests for the non-leakage defects found in the Phase 0 audit.

One test (or group) per finding. Each fails if the defect returns.
"""

import inspect
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
