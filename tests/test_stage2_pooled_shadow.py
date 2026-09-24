"""
Stage 2, Part 3 — the pooled model's production path, in shadow (2026-09-24).

  * `pipeline.pooled.walk_forward` IS the research harness `run_arm`, bit for
    bit — every stored pooled baseline was produced by run_arm;
  * daily inference reads only the forecast date's features, and the price-
    space inverse uses only PAST moments;
  * the Romano-Wolf producer is wired: a ticker `grade_panel_v3` grades STRONG
    stays STRONG in shadow storage, with its rejection recorded;
  * a tau2 ~ 0 panel is written as ONE statement, not 84 identical grades;
  * the weekly and daily shadow steps run end to end on SQLite;
  * public API responses are byte-identical with shadow rows present;
  * the conformal gate reads its pre-registered bands.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import pooled, pooled_shadow

PARAMS = {"n_estimators": 40, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 5, "gamma": 0.0,
          "reg_alpha": 0.0, "reg_lambda": 1.0, "tree_method": "hist"}


def synthetic(n_tickers=24, n_dates=420, seed=7, signal=0.03):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_dates).strftime("%Y-%m-%d")
    frames = []
    for i in range(n_tickers):
        x = rng.standard_normal((n_dates, len(pooled.FEATURES)))
        f = pd.DataFrame(x, columns=pooled.FEATURES)
        f["date"], f["ticker"] = dates, f"T{i:02d}.NS"
        f["close"] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_dates)))
        f["target_return"] = signal * x[:, 0] + rng.normal(0, 0.1, n_dates)
        f.loc[f.index[-30:], "target_return"] = np.nan
        frames.append(f)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]) \
        .reset_index(drop=True)


def test_walk_forward_is_the_research_harness_bit_for_bit():
    from pipeline.panel import SCALE_FREE, cross_sectional_zscore
    from tools.stage2b_pooled import run_arm

    raw = synthetic()
    ours, _ = pooled.walk_forward(pooled.prepare(raw, keep_missing=False),
                                  n_folds=3, min_train=200, n_trials=2)
    theirs, _ = run_arm(cross_sectional_zscore(raw, SCALE_FREE), "mae", "none",
                        n_trials=2, n_folds=3, min_train=200,
                        features=list(pooled.FEATURES), verbose=False)
    assert len(ours) == len(theirs)
    np.testing.assert_array_equal(ours["y_pred"].to_numpy(), theirs["y_pred"].to_numpy())
    np.testing.assert_array_equal(ours["y_true"].to_numpy(), theirs["y_true"].to_numpy())


def test_daily_inference_reads_only_the_forecast_dates_features():
    raw = synthetic()
    prepared = pooled.prepare(raw)
    fitted = pooled.fit_final(prepared, params=PARAMS)
    as_of = str(prepared["date"].max())
    a = pooled.predict_on(fitted, prepared, as_of)
    corrupted = raw.copy()
    other = corrupted["date"] != as_of
    corrupted.loc[other, pooled.FEATURES] = 999.0
    b = pooled.predict_on(fitted, pooled.prepare(corrupted), as_of)
    np.testing.assert_array_equal(a["pred_z"].to_numpy(), b["pred_z"].to_numpy())


def test_the_final_fit_never_trains_on_an_unrealised_label():
    raw = synthetic()
    fitted = pooled.fit_final(pooled.prepare(raw), params=PARAMS)
    last_labelled = raw.dropna(subset=["target_return"])["date"].max()
    assert fitted.train_last_date == last_labelled
    assert fitted.n_train_rows == int(raw["target_return"].notna().sum())


def test_the_price_space_inverse_uses_only_past_moments():
    raw = synthetic()
    dates = sorted(raw["date"].unique())
    as_of = dates[-1]
    base = pooled.causal_frame(raw).set_index("date").loc[as_of]
    # Corrupt every label that had NOT realised by as_of: a label at date d
    # realises at d + 30 sessions, so those after dates[-31] are the future.
    future = raw["date"] > dates[-31]
    tampered = raw.copy()
    tampered.loc[future, "target_return"] = 5.0
    after = pooled.causal_frame(tampered).set_index("date").loc[as_of]
    assert base["causal_mean"] == after["causal_mean"]
    assert base["causal_sd"] == after["causal_sd"]
    with pytest.raises(ValueError, match="RAW"):
        pooled.causal_frame(pooled.prepare(raw))


def _strong_panel():
    """One ticker whose prediction ranks its own return near-perfectly."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2019-01-01", periods=600).strftime("%Y-%m-%d")
    rows = []
    for d_i, d in enumerate(dates):
        y = rng.normal(size=30)
        pred = rng.normal(size=30)
        pred[0] = y[0] * 5 + rng.normal(0, 0.1)    # T00 carries real skill
        for i in range(30):
            rows.append((d, f"T{i:02d}.NS", min(d_i // 120, 4), y[i], pred[i]))
    return pd.DataFrame(rows, columns=["date", "ticker", "fold", "y_true", "y_pred"])


def test_romano_wolf_reaches_strong_through_the_shadow_wiring():
    from pipeline.evidence_panel import grade_panel_v3

    p = _strong_panel()
    g = grade_panel_v3(p["date"].to_numpy(), p["ticker"].to_numpy(), p["fold"].to_numpy(),
                       p["y_true"].to_numpy(), p["y_pred"].to_numpy(),
                       n_bootstrap=60, compute_auto_block=False)
    t00 = next(r for r in g.rows if r.ticker == "T00.NS")
    assert t00.grade == "STRONG" and t00.rw_rejected
    hashes = {"config_hash": "c", "data_hash": "d", "env_hash": "e"}
    rows, statement = pooled_shadow.evaluation_rows(g, "m1", hashes, "now")
    assert not statement["degenerate"]
    row = next(r for r in rows if r["ticker"] == "T00.NS")
    assert row["grade"] == "STRONG" and row["rw_rejected"] == 1
    assert row["rw_adjusted_p"] is not None and row["rw_adjusted_p"] < 0.10


def test_a_degenerate_panel_is_one_statement_not_84_grades():
    from pipeline.evidence_panel import grade_panel_v3

    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2019-01-01", periods=500).strftime("%Y-%m-%d")
    p = pd.DataFrame([(d, f"T{i:02d}.NS", min(k // 100, 4), rng.normal(), rng.normal())
                      for k, d in enumerate(dates) for i in range(25)],
                     columns=["date", "ticker", "fold", "y_true", "y_pred"])
    g = grade_panel_v3(p["date"].to_numpy(), p["ticker"].to_numpy(), p["fold"].to_numpy(),
                       p["y_true"].to_numpy(), p["y_pred"].to_numpy(),
                       n_bootstrap=40, compute_auto_block=False)
    # Whatever REML landed on, force the boundary: what is under test is the
    # SHADOW mapping of a degenerate panel, not where noise happens to fall.
    import dataclasses

    from pipeline.evidence_panel import DegeneracyVerdict
    g = dataclasses.replace(g, degeneracy=DegeneracyVerdict(
        True, 0.0, g.degeneracy.q_statistic, g.degeneracy.q_df,
        "tau2 at the boundary (forced)"))
    rows, statement = pooled_shadow.evaluation_rows(
        g, "m2", {"config_hash": "c", "data_hash": "d", "env_hash": "e"}, "now")
    assert statement["degenerate"] == 1 and statement["statement"]
    assert all(r["grade"] is None for r in rows)
    assert len({r["reason"] for r in rows}) == 1


def test_the_coverage_gate_reads_its_pre_registered_bands():
    assert pooled_shadow.COVERAGE_BAND_OVERALL == (0.75, 0.85)
    assert pooled_shadow.COVERAGE_BAND_FOLD == (0.70, 0.90)
    rng = np.random.default_rng(0)
    folds = np.repeat(np.arange(5), 2000)
    y = rng.normal(size=folds.size)
    ok = pooled_shadow.coverage_gate(y, np.zeros_like(y), folds)
    assert ok["status"] == pooled_shadow.GATE_PASS
    # a late fold far calmer than the calibration pool over-covers and fails
    y2 = y.copy()
    y2[folds == 4] *= 0.3
    bad = pooled_shadow.coverage_gate(y2, np.zeros_like(y2), folds)
    assert bad["status"] == pooled_shadow.GATE_FAIL
    assert any("fold 4" in r for r in bad["reasons"])


@pytest.fixture
def shadow_db(tmp_path, monkeypatch):
    """A real schema on a tmp SQLite with a small synthetic universe loaded
    into `signals` and `macro`, so load_panel and the shadow steps run as they
    do in production."""
    from sqlalchemy import create_engine

    from data import db
    from pipeline.signals import FEATURE_COLS

    url = f"sqlite:///{(tmp_path / 'shadow.sqlite').as_posix()}"
    eng = create_engine(url)
    monkeypatch.setattr(db, "get_engine", lambda: eng)
    db.init_db()
    raw = synthetic(n_tickers=24, n_dates=420)
    # load_panel reads the committed NSE calendar: keep only real sessions.
    from data import nse_calendar
    raw, _ = nse_calendar.drop_non_sessions(raw, nse_calendar.load())
    for c in FEATURE_COLS:
        if c not in raw.columns:
            raw[c] = 1.0
    raw["target_excess_return"] = raw["target_return"]
    raw.to_sql("signals", eng, index=False, if_exists="append")
    pd.DataFrame({"date": sorted(raw["date"].unique()), "usdinr": 80.0,
                  "india_vix": 14.0, "nifty_5d_return": 0.0,
                  "nifty_20d_return": 0.0}).to_sql("macro", eng, index=False,
                                                   if_exists="append")
    return eng, sorted(raw["ticker"].unique())


def test_weekly_then_daily_shadow_end_to_end(shadow_db, monkeypatch):
    eng, universe = shadow_db
    monkeypatch.setattr(pooled, "MIN_TRAIN_DATES", 200)
    monkeypatch.setattr(pooled, "N_FOLDS", 3)
    weekly = pooled_shadow.run_weekly_shadow(universe, eng, n_trials=1, bootstrap=30,
                                             verbose=False)
    assert weekly["folds"] >= 2 and weekly["constant_cells"] == 0
    assert weekly["coverage_status"] in (pooled_shadow.GATE_PASS, pooled_shadow.GATE_FAIL)
    models = pd.read_sql("SELECT * FROM shadow_models", eng)
    assert len(models) == 1 and models["pooled_version"].iloc[0] == pooled.POOLED_MODEL_VERSION
    for c in ("config_hash", "data_hash", "env_hash"):
        assert models[c].notna().all()
    ev = pd.read_sql("SELECT * FROM shadow_evaluations", eng)
    st = pd.read_sql("SELECT * FROM shadow_panel_statements", eng)
    assert len(ev) == len(universe) and len(st) == 1

    daily = pooled_shadow.run_daily_shadow(universe, eng)
    fc = pd.read_sql("SELECT * FROM shadow_forecasts", eng)
    assert daily["forecasts"] == len(universe) == len(fc)
    assert fc["pred_return"].notna().all() and fc["interval_low"].notna().all()
    assert (fc["interval_low"] < fc["implied_price"]).all()
    assert (fc["implied_price"] < fc["interval_high"]).all()
    # pooled v2: the band is q x THIS date's past-only spread, stored beside it
    cal = json.loads(models["calibration_json"].iloc[0])
    assert cal["method"] == "spread-normalised"
    # ...and the gate that judged it scored the same interval, not the constant one
    assert json.loads(models["coverage_json"].iloc[0])["method"] == "spread-normalised"
    assert weekly["coverage_method"] == "spread-normalised"
    assert fc["interval_spread"].notna().all() and (fc["interval_spread"] > 0).all()
    half = np.log(fc["interval_high"] / fc["current_price"]) - fc["pred_return"]
    np.testing.assert_allclose(half, cal["quantile"] * fc["interval_spread"], rtol=1e-9)
    # a re-run is a record, not a running total
    pooled_shadow.run_daily_shadow(universe, eng)
    assert len(pd.read_sql("SELECT * FROM shadow_forecasts", eng)) == len(fc)


def test_public_api_responses_are_byte_identical_with_shadow_rows(shadow_db, monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    eng, universe = shadow_db
    client = TestClient(app)
    paths = ["/api/stocks", "/api/forecasts", f"/api/signals/{universe[0]}",
             f"/api/forecasts/{universe[0]}"]

    import re

    # The wall-clock `last_updated` the list endpoint stamps on every response
    # is the only field allowed to differ between two calls; everything else
    # must be the same BYTES.
    def snap():
        return {p: (r.status_code, re.sub(rb'"last_updated":"[^"]*"', b"", r.content))
                for p in paths for r in [client.get(p)]}

    before = snap()
    pooled_shadow.init_shadow_tables(eng)
    with eng.begin() as c:
        from sqlalchemy import text
        c.execute(text("INSERT INTO shadow_forecasts (forecast_date, ticker, model_id, "
                       "pooled_version, created_at, pred_return, grade) VALUES "
                       "('2020-01-01', :t, 'm', 'v', 'now', 0.5, 'STRONG')"),
                  {"t": universe[0]})
    assert snap() == before


def test_the_shadow_inspection_endpoint_is_admin_gated(monkeypatch, shadow_db):
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    assert client.get("/api/admin/shadow/summary").status_code in (403, 422, 503)
    monkeypatch.setenv("ADMIN_API_KEY", "k")
    r = client.get("/api/admin/shadow/summary", headers={"X-API-Key": "k"})
    assert r.status_code == 200
    assert "model" in r.json()


def test_no_public_page_links_the_shadow_endpoint():
    import re
    from pathlib import Path

    web = Path(__file__).resolve().parents[1] / "web"
    hits = [p for p in web.rglob("*.ts*") if "node_modules" not in p.parts
            and re.search(r"admin/shadow|shadow_(forecasts|models|evaluations)",
                          p.read_text(encoding="utf-8", errors="ignore"))]
    assert not hits, hits
