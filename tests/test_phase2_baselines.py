"""
tests/test_phase2_baselines.py — Guards for the Phase 2 comparator harness.

Phase 2 exists to attack a null result, which makes it the phase most likely to
manufacture a positive one. Every test here pins a specific way that could
happen:

  - a pooled model reading ticker identity out of a price-denominated column
  - a z-score computed over a whole history rather than within a date
  - a panel split on row position, so the fold boundary lands on a different
    calendar date for every ticker
  - a constant prediction being credited with a portfolio it never chose

The last of those was not hypothetical. It was found by running the harness:
`zero`, `train_mean` and `majority` each reported alpha +0.00914 at t = +1.19,
which was the return of holding the alphabetically-first fifth of the universe.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import baselines as B
from pipeline import panel as P
from pipeline.evaluation import (
    PurgedPanelWalkForward,
    cross_sectional_report,
    panel_walk_forward,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_panel(n_dates: int = 800, n_tickers: int = 20, seed: int = 0,
               with_signal: bool = False) -> pd.DataFrame:
    """A synthetic panel with the same columns the real loader produces."""
    rng = np.random.default_rng(seed)
    dates = [f"{2015 + i // 252}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}-{i:04d}"
             for i in range(n_dates)]
    tickers = [f"T{i:02d}.NS" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for t in tickers:
            row = {"date": d, "ticker": t, "close": 100.0,
                   "benchmark_ticker": "^NSEI"}
            for col in P.FEATURES:
                row[col] = float(rng.normal())
            rows.append(row)

    df = pd.DataFrame(rows)
    if with_signal:
        # A genuine, recoverable cross-sectional signal, so a test can tell
        # "the harness found nothing" apart from "the harness cannot find
        # anything".
        df[P.TARGET] = 0.05 * df["roc_10"] + 0.01 * np.random.default_rng(seed + 1
                                                                          ).normal(size=len(df))
    else:
        df[P.TARGET] = rng.normal(size=len(df)) * 0.05

    # The most recent horizon of dates has no knowable label yet, as in production.
    df.loc[df["date"].isin(dates[-30:]), P.TARGET] = np.nan
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ── Feature taxonomy ──────────────────────────────────────────────────────────


def test_panel_features_match_the_production_model():
    """
    panel.FEATURES is maintained separately from model.FEATURES so the panel
    does not import XGBoost. Separately maintained lists drift; this is the
    only thing stopping them.
    """
    from pipeline.model import FEATURES as MODEL_FEATURES
    assert P.FEATURES == MODEL_FEATURES


def test_scale_free_and_price_scaled_partition_the_features():
    from pipeline.signals import FEATURE_COLS
    assert set(P.SCALE_FREE) | set(P.PRICE_SCALED) == set(FEATURE_COLS)
    assert not (set(P.SCALE_FREE) & set(P.PRICE_SCALED))


def test_obv_is_treated_as_price_scaled():
    """
    OBV is a cumulative sum from the first row of a ticker's history, so its
    level encodes how long that history is and how heavily the name trades.
    Pooled, it is a ticker label. Naming it explicitly because it is the one
    most likely to be waved through as "just another indicator".
    """
    assert "obv" in P.PRICE_SCALED
    assert "obv" not in B.FACTORS


def test_linear_factor_set_contains_no_price_denominated_column():
    for col in B.FACTORS:
        assert col in P.SCALE_FREE, f"{col} is denominated in price or volume"


def test_macro_columns_are_excluded_from_the_factor_set():
    """
    Every ticker sees the same USDINR on a given date, so after cross-sectional
    standardisation a macro column is identically zero and carries no ranking
    information. Including one would be harmless but misleading — it would
    appear in the loadings table with a coefficient that cannot mean anything.
    """
    for col in P.MACRO_COLS:
        assert col not in B.FACTORS


# ── Cross-sectional standardisation ───────────────────────────────────────────


def test_zscore_is_computed_within_a_date_not_across_time():
    """
    THE leakage test for this module. A z-score taken over a column's whole
    history uses a mean and standard deviation drawn partly from the future;
    one taken within a date does not.

    Appending later dates must therefore leave every earlier date's z-score
    bit-for-bit unchanged. If it does not, the transform is reading forward.
    """
    full = make_panel(n_dates=300, n_tickers=15, seed=3)
    early_dates = sorted(full["date"].unique())[:150]
    early = full[full["date"].isin(early_dates)].reset_index(drop=True)

    z_full = P.cross_sectional_zscore(full, ["roc_10"])
    z_early = P.cross_sectional_zscore(early, ["roc_10"])

    a = z_full[z_full["date"].isin(early_dates)].sort_values(["date", "ticker"])
    b = z_early.sort_values(["date", "ticker"])

    np.testing.assert_allclose(a["roc_10"].to_numpy(), b["roc_10"].to_numpy())


def test_zscore_removes_a_pure_level_difference_between_tickers():
    """A constant offset per ticker is exactly what a pooled model must not see."""
    df = make_panel(n_dates=60, n_tickers=20, seed=5)
    offset = {t: 1000.0 * i for i, t in enumerate(sorted(df["ticker"].unique()))}
    df["roc_10"] = df["roc_10"] + df["ticker"].map(offset)

    z = P.cross_sectional_zscore(df, ["roc_10"])
    per_date_std = z.groupby("date")["roc_10"].std()
    # Ordering survives; the 1000x level differences do not.
    assert z["roc_10"].abs().max() <= P.ZSCORE_CLIP + 1e-9
    assert per_date_std.between(0.5, 1.5).all()


def test_zscore_zeroes_a_date_too_thin_to_standardise():
    df = make_panel(n_dates=20, n_tickers=4, seed=7)
    z = P.cross_sectional_zscore(df, ["roc_10"], min_names=10)
    assert (z["roc_10"] == 0.0).all()


def test_zscore_clips_outliers():
    df = make_panel(n_dates=10, n_tickers=30, seed=9)
    df.loc[0, "roc_10"] = 1e6
    z = P.cross_sectional_zscore(df, ["roc_10"])
    assert z["roc_10"].max() <= P.ZSCORE_CLIP + 1e-9
    assert z["roc_10"].min() >= -P.ZSCORE_CLIP - 1e-9


# ── The panel splitter ────────────────────────────────────────────────────────


def test_panel_split_leaves_a_purge_gap_measured_in_dates():
    df = make_panel(n_dates=900, n_tickers=10, seed=11)
    grid = sorted(df["date"].unique())
    pos = {d: i for i, d in enumerate(grid)}
    splitter = PurgedPanelWalkForward(n_folds=4, horizon=30, embargo=30,
                                      min_train=500)

    folds = list(splitter.split(df["date"].to_numpy()))
    assert folds, "splitter produced no folds"

    for train_idx, test_idx in folds:
        last_train = max(pos[d] for d in df.iloc[train_idx]["date"])
        first_test = min(pos[d] for d in df.iloc[test_idx]["date"])
        assert first_test - last_train > 30 + 30, (
            f"purge gap is {first_test - last_train} dates, needs > 60"
        )


def test_panel_split_never_puts_a_training_date_after_a_test_date():
    df = make_panel(n_dates=800, n_tickers=8, seed=13)
    splitter = PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=500)
    for train_idx, test_idx in splitter.split(df["date"].to_numpy()):
        assert df.iloc[train_idx]["date"].max() < df.iloc[test_idx]["date"].min()


def test_panel_split_uses_one_date_boundary_for_every_ticker():
    """
    The reason this splitter exists rather than PurgedWalkForward. Tickers join
    the panel at different dates, so a boundary at row position 500 falls on a
    different calendar date for each of them; a date boundary does not.
    """
    df = make_panel(n_dates=800, n_tickers=6, seed=17)
    # Give one ticker a much shorter history, as a recent listing would have.
    late = sorted(df["date"].unique())[400]
    df = df[(df["ticker"] != "T00.NS") | (df["date"] >= late)].reset_index(drop=True)

    splitter = PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=500)
    for _, test_idx in splitter.split(df["date"].to_numpy()):
        block = df.iloc[test_idx]
        # Every ticker present in the fold spans the identical date range.
        spans = block.groupby("ticker")["date"].agg(["min", "max"])
        assert spans["min"].nunique() == 1 or len(spans) == 1
        assert spans["max"].nunique() == 1 or len(spans) == 1


def test_panel_split_yields_nothing_when_min_train_exceeds_the_grid():
    df = make_panel(n_dates=100, n_tickers=5, seed=19)
    splitter = PurgedPanelWalkForward(n_folds=5, horizon=30, min_train=500)
    assert list(splitter.split(df["date"].to_numpy())) == []


# ── The panel harness ─────────────────────────────────────────────────────────


class _Spy:
    """Records the rows and labels it was fitted on, so a test can inspect them."""

    seen: list[pd.DataFrame] = []
    labels: list[pd.Series] = []

    def fit(self, X, y):
        _Spy.seen.append(X.copy())
        _Spy.labels.append(pd.Series(y).copy())
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


def _spy_run(df, n_folds=3):
    _Spy.seen = []
    _Spy.labels = []
    res = panel_walk_forward(
        panel=df, feature_cols=["_row"], model_factory=_Spy,
        splitter=PurgedPanelWalkForward(n_folds=n_folds, horizon=30,
                                        min_train=500),
        name="spy",
    )
    return res, list(_Spy.seen)


def test_harness_never_fits_a_model_on_a_row_it_will_score():
    """
    Checked PER FOLD, which is the property that actually matters and is easy
    to get wrong in the other direction: fold 3 legitimately trains on rows
    fold 1 was scored on, because by fold 3's turn those rows are in the past.
    A test that forbade all reuse across folds would be forbidding the
    expanding window itself.
    """
    df = make_panel(n_dates=900, n_tickers=10, seed=23)
    df["_row"] = np.arange(len(df), dtype=float)

    res, blocks = _spy_run(df)
    assert res.n_folds_run > 0
    folds = sorted(res.predictions["fold"].unique())
    assert len(blocks) == len(folds)

    for block, fold in zip(blocks, folds):
        keys = res.predictions[res.predictions["fold"] == fold][["date", "ticker"]]
        scored = set(df.merge(keys, on=["date", "ticker"])["_row"].tolist())
        fitted = set(block["_row"].tolist())
        assert not (fitted & scored), f"fold {fold} scored a row it was fitted on"
        assert max(fitted) < min(scored), (
            f"fold {fold} trained on a row positioned after its own test window"
        )


def test_harness_uses_an_expanding_window():
    """
    Each fold trains on everything available up to its purge boundary, so the
    training set grows monotonically. Pinned because the alternative — a
    fixed-width rolling window — is a different experiment, and switching
    between them silently would move every metric with no visible cause.
    """
    df = make_panel(n_dates=900, n_tickers=10, seed=23)
    df["_row"] = np.arange(len(df), dtype=float)

    _, blocks = _spy_run(df)
    sizes = [len(b) for b in blocks]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), sizes
    for earlier, later in zip(blocks, blocks[1:]):
        assert set(earlier["_row"]).issubset(set(later["_row"]))


def test_harness_never_hands_a_model_an_unknowable_label():
    """
    The training half of the same rule. Scoring against a NaN is obviously
    wrong and is caught elsewhere; ASKING A MODEL TO FIT ONE is quieter,
    because several estimators here coerce NaN to zero on the way in and carry
    on. A model fitted on a fabricated zero label has been taught that the most
    recent 30 sessions were all flat.
    """
    df = make_panel(n_dates=900, n_tickers=10, seed=23)

    # A ticker whose history ENDS mid-panel — a delisting or a suspension. Its
    # own trailing 30 sessions are unlabelled like any other, but they sit in
    # the MIDDLE of the date grid rather than at the end, so they fall inside
    # the training window of every later fold. Without the interior case this
    # test cannot fail: the unlabelled tail of a live ticker is always beyond
    # the last training boundary anyway.
    grid = sorted(df["date"].unique())
    gone = df["ticker"] == "T00.NS"
    df = df[~(gone & df["date"].isin(grid[600:]))].reset_index(drop=True)
    df.loc[(df["ticker"] == "T00.NS") & df["date"].isin(grid[570:600]),
           P.TARGET] = np.nan
    df["_row"] = np.arange(len(df), dtype=float)
    assert df[P.TARGET].isna().sum() > 30, "fixture did not create interior gaps"

    res, _ = _spy_run(df)
    assert res.n_folds_run > 0
    assert _Spy.labels, "the spy was never fitted"
    for y in _Spy.labels:
        assert np.isfinite(y.to_numpy(dtype=float)).all(), (
            "a model was fitted on a row whose label is not yet knowable"
        )


def test_harness_drops_rows_whose_label_is_not_yet_knowable():
    df = make_panel(n_dates=900, n_tickers=10, seed=29)
    res = panel_walk_forward(
        panel=df, feature_cols=B.FACTORS, model_factory=B.ZeroForecast,
        splitter=PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=500),
        name="zero",
    )
    assert res.n_predictions > 0
    assert res.predictions["y_true"].notna().all()


def test_harness_recovers_a_signal_that_is_really_there():
    """
    A null-result harness that reports null on everything is not measuring, it
    is broken. This plants a linear cross-sectional signal and requires the
    linear comparator to find it.
    """
    df = make_panel(n_dates=900, n_tickers=25, seed=31, with_signal=True)
    df = P.cross_sectional_zscore(df, P.SCALE_FREE)
    res = panel_walk_forward(
        panel=df, feature_cols=B.FACTORS, model_factory=B.LinearFactorModel,
        splitter=PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=500),
        name="linear_factor",
    )
    assert res.metrics["daily_rank_ic"] > 0.20, res.metrics


def test_harness_reports_no_folds_rather_than_guessing():
    df = make_panel(n_dates=100, n_tickers=5, seed=37)
    res = panel_walk_forward(
        panel=df, feature_cols=B.FACTORS, model_factory=B.ZeroForecast,
        splitter=PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=500),
        name="zero",
    )
    assert res.n_folds_run == 0 and res.n_predictions == 0


def test_harness_rejects_a_frame_without_the_panel_columns():
    with pytest.raises(ValueError):
        panel_walk_forward(panel=pd.DataFrame({"date": ["2020-01-01"]}),
                           feature_cols=[], model_factory=B.ZeroForecast)


# ── Baselines ─────────────────────────────────────────────────────────────────


def test_zero_forecast_predicts_zero():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert (B.ZeroForecast().fit(X, pd.Series([1.0, 2.0, 3.0])).predict(X) == 0).all()


def test_majority_direction_is_fitted_on_training_data_only():
    """
    The difference between this and evaluation.majority_hit_rate, which reads
    the majority off the test set. A baseline that consults the answer is not a
    baseline.
    """
    y_train = pd.Series([-1.0] * 80 + [1.0] * 20)
    X = pd.DataFrame({"a": np.zeros(100)})
    model = B.MajorityDirection().fit(X, y_train)
    assert model.sign_ == -1.0
    assert (model.predict(pd.DataFrame({"a": np.zeros(5)})) == -1.0).all()


def test_train_mean_predicts_the_training_mean():
    y = pd.Series([0.02, 0.04, 0.06])
    model = B.TrainMeanForecast().fit(pd.DataFrame({"a": [0, 0, 0]}), y)
    assert model.mu_ == pytest.approx(0.04)


def test_single_factor_recovers_a_known_slope():
    rng = np.random.default_rng(41)
    x = rng.normal(size=500)
    X = pd.DataFrame({"sector_rel_20d": x})
    y = pd.Series(0.7 * x + 0.001)
    model = B.SingleFactor("sector_rel_20d").fit(X, y)
    assert model.slope_ == pytest.approx(0.7, abs=0.01)


def test_single_factor_preserves_a_negative_slope():
    """A reversal is a finding, not something to clamp away."""
    rng = np.random.default_rng(43)
    x = rng.normal(size=500)
    model = B.SingleFactor("lag5_ret").fit(pd.DataFrame({"lag5_ret": x}),
                                           pd.Series(-0.5 * x))
    assert model.slope_ < 0


def test_linear_factor_model_is_readable_after_fitting():
    df = make_panel(n_dates=50, n_tickers=25, seed=47, with_signal=True)
    df = P.cross_sectional_zscore(df, P.SCALE_FREE)
    model = B.LinearFactorModel().fit(df[B.FACTORS], df[P.TARGET].fillna(0.0))
    coefs = model.coefficients()
    assert set(coefs) == set(B.FACTORS)
    assert abs(coefs["roc_10"]) == max(abs(v) for v in coefs.values())


def test_every_registered_baseline_runs_through_the_harness():
    df = make_panel(n_dates=800, n_tickers=20, seed=53)
    df = P.cross_sectional_zscore(df, P.SCALE_FREE)
    splitter = PurgedPanelWalkForward(n_folds=2, horizon=30, min_train=500)
    for name, factory in B.BASELINES.items():
        res = panel_walk_forward(panel=df, feature_cols=B.FACTORS,
                                 model_factory=factory, splitter=splitter,
                                 name=name)
        assert res.n_predictions > 0, f"{name} produced nothing"


def test_a_constant_per_fold_predictor_has_no_daily_rank_ic():
    """
    Pins the difference between the two IC columns. TrainMeanForecast emits one
    constant per fold, so it holds no ranking information at all — yet on the
    first real panel run its POOLED rank IC came out at -0.007 because the
    constant differs between folds and the correlation picked up which fold a
    row belonged to. The daily IC, computed within each date, is correctly
    undefined. If this ever starts returning a number, the daily statistic has
    silently become the pooled one.
    """
    df = make_panel(n_dates=900, n_tickers=20, seed=67)
    df = P.cross_sectional_zscore(df, P.SCALE_FREE)
    res = panel_walk_forward(
        panel=df, feature_cols=B.FACTORS, model_factory=B.TrainMeanForecast,
        splitter=PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=500),
        name="train_mean",
    )
    assert res.n_folds_run > 1
    assert res.predictions.groupby("fold")["y_pred"].nunique().eq(1).all()
    assert np.isnan(res.metrics["daily_rank_ic"]), res.metrics["daily_rank_ic"]


def test_zero_forecast_mae_equals_the_naive_benchmark():
    """
    A self-consistency check on the harness rather than on a model. The naive
    MAE reported beside every result is mean(|y_true|), which IS the error of
    predicting zero excess return. If the `zero` row ever disagrees with its own
    naive column, one of the two is being computed over different rows.
    """
    df = make_panel(n_dates=900, n_tickers=20, seed=71)
    res = panel_walk_forward(
        panel=df, feature_cols=B.FACTORS, model_factory=B.ZeroForecast,
        splitter=PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=500),
        name="zero",
    )
    assert res.metrics["mae"] == pytest.approx(res.metrics["mae_naive_zero"])


# ── The degenerate-ranking defect ─────────────────────────────────────────────


def _constant_prediction_panel(n_dates: int = 200, n_tickers: int = 20,
                               alphabetical_alpha: float = 0.05) -> pd.DataFrame:
    """
    A panel where the alphabetically-first tickers really do outperform, and
    the prediction is a single constant. Any ranking result reported on this is
    an artifact of the sort, because there is nothing to sort by.
    """
    rows = []
    tickers = [f"T{i:02d}.NS" for i in range(n_tickers)]
    rng = np.random.default_rng(59)
    for d in range(n_dates):
        for i, t in enumerate(tickers):
            edge = alphabetical_alpha if i < n_tickers // 5 else 0.0
            rows.append({"date": f"D{d:04d}", "ticker": t,
                         "y_pred": 0.123, "y_true": edge + rng.normal() * 0.001})
    return pd.DataFrame(rows)


def test_a_constant_prediction_earns_no_alpha():
    """
    The defect this harness found on its first real run. `zero`, `train_mean`
    and `majority` each reported alpha +0.00914 at t = +1.19 with a long-short
    spread of +0.01744 — the return of the alphabetically-first fifth of the
    universe, credited to a predictor that expressed no preference at all.
    """
    report = cross_sectional_report(_constant_prediction_panel(), rebalance_every=1)
    assert report["n_rebalances"] == 0
    assert report["n_dates_no_ordering"] > 0
    assert "alpha_vs_equal_weight" not in report


def test_ranking_ties_are_not_broken_alphabetically():
    """
    Partial ties are rare for a continuous prediction and routine for a clipped
    or rounded one. When they happen the tiebreak must not correlate with the
    ticker's spelling, or a spurious edge reappears in a subtler form.
    """
    panel = _constant_prediction_panel(n_dates=400)
    # Two prediction levels, so there IS an ordering; ties fill each level and
    # straddle the quintile boundary.
    half = panel["ticker"] < "T10.NS"
    panel.loc[half, "y_pred"] = 0.2
    panel.loc[~half, "y_pred"] = 0.1

    report = cross_sectional_report(panel, rebalance_every=1)
    assert report["n_rebalances"] > 0
    # The top quintile is drawn from the tied 0.2 block. If the tiebreak were
    # alphabetical it would be exactly T00-T03, which carry the planted edge.
    assert report["alpha_vs_equal_weight"] < 0.04, report


def test_a_real_ordering_is_still_scored():
    """The tie guard must not suppress a genuine ranking."""
    rng = np.random.default_rng(61)
    rows = []
    for d in range(200):
        for i in range(20):
            pred = rng.normal()
            rows.append({"date": f"D{d:04d}", "ticker": f"T{i:02d}.NS",
                         "y_pred": pred, "y_true": pred * 0.05 + rng.normal() * 0.001})
    report = cross_sectional_report(pd.DataFrame(rows), rebalance_every=1)
    assert report["n_rebalances"] == 200
    assert report["mean_rank_ic"] > 0.5
    assert report["alpha_vs_equal_weight"] > 0


# ── Macro alignment ───────────────────────────────────────────────────────────


def test_macro_forward_fill_does_not_bleed_between_tickers():
    """
    The panel is a long frame, so a forward fill applied AFTER the merge would
    carry one ticker's value into the next ticker's row rather than forward in
    time. _attach_macro fills on the date-indexed macro frame instead.
    """
    from sqlalchemy import create_engine

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE macro (date TEXT, usdinr REAL, india_vix REAL, "
            "nifty_5d_return REAL, nifty_20d_return REAL, "
            "fii_net_flow REAL, dii_net_flow REAL)")
        # A value on day 2 only. Day 1 precedes every observation, so it must
        # stay empty for BOTH tickers however the frame happens to be ordered.
        conn.exec_driver_sql(
            "INSERT INTO macro VALUES ('2020-01-02', 75.0, 20.0, 0.0, 0.0, 0.0, 0.0)")

    # Ordered by TICKER, not by date. Nothing guarantees _attach_macro receives
    # a date-sorted frame, and this ordering is what separates a fill down the
    # date axis from a fill down the rows: a row-wise ffill reaching B.NS's
    # first row finds A.NS's LAST row above it, which is a later date.
    panel = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        "ticker": ["A.NS", "A.NS", "B.NS", "B.NS"],
    })
    out = P._attach_macro(panel, engine).set_index(["ticker", "date"])

    assert out.loc[("A.NS", "2020-01-02"), "usdinr"] == 75.0
    assert out.loc[("B.NS", "2020-01-02"), "usdinr"] == 75.0
    # Neither ticker may carry a value on the day before the macro observation.
    assert pd.isna(out.loc[("A.NS", "2020-01-01"), "usdinr"])
    assert pd.isna(out.loc[("B.NS", "2020-01-01"), "usdinr"]), (
        "B.NS inherited a value from A.NS's later row: the fill ran down the "
        "long frame instead of down the date axis"
    )


def test_macro_gap_before_the_first_observation_is_not_backfilled():
    """Backfilling a leading gap imports the future (audit finding F12)."""
    from sqlalchemy import create_engine

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE macro (date TEXT, usdinr REAL, india_vix REAL, "
            "nifty_5d_return REAL, nifty_20d_return REAL, "
            "fii_net_flow REAL, dii_net_flow REAL)")
        conn.exec_driver_sql(
            "INSERT INTO macro VALUES ('2020-06-01', 75.0, 20.0, 0.0, 0.0, 0.0, 0.0)")

    panel = pd.DataFrame({"date": ["2020-01-01", "2020-06-01"],
                          "ticker": ["A.NS", "A.NS"]})
    out = P._attach_macro(panel, engine)
    assert pd.isna(out.loc[out["date"] == "2020-01-01", "usdinr"]).all()
    assert out.loc[out["date"] == "2020-06-01", "usdinr"].iloc[0] == 75.0


def test_compare_baselines_refuses_a_panel_too_thin_to_rank(monkeypatch):
    """
    The refusal lives in compare_baselines, not in the CLI, so the weekly job
    inherits it rather than reimplementing it. Below the breadth threshold
    every feature is zeroed by design and no date can be ranked, so the table
    that would be produced is a grid of constants in the exact shape of a
    result — and unlike a CLI run, nobody is watching when the weekly job
    records one.
    """
    import pipeline.baselines

    thin = make_panel(n_dates=800, n_tickers=4, seed=73)
    monkeypatch.setattr(pipeline.baselines, "load_panel", lambda **k: thin)

    comparison = pipeline.baselines.compare_baselines()
    assert not comparison.ranked
    assert "too thin" in comparison.note
    assert comparison.to_metrics()["comparators"] == []


def test_compare_baselines_scores_a_panel_with_enough_breadth(monkeypatch):
    """The refusal must not be so eager that it never measures anything."""
    import pipeline.baselines

    wide = make_panel(n_dates=800, n_tickers=25, seed=79)
    monkeypatch.setattr(pipeline.baselines, "load_panel", lambda **k: wide)

    comparison = pipeline.baselines.compare_baselines(with_pooled_xgb=False)
    assert comparison.ranked
    assert {r["name"] for r in comparison.results} == set(B.BASELINES)
    assert comparison.best() is not None


def test_series_comparators_are_skipped_without_a_benchmark_level(monkeypatch):
    """
    A SeriesAdapter handed an all-NaN series does not fail. It declines every
    ticker and predicts 0.0 — the correct default for an abstention — which
    would appear in the table as three extra rows indistinguishable from
    `zero`. Rows written before benchmark_close was persisted have perfectly
    good targets and no reconstructible series, so the comparison must skip
    those comparators and say why rather than report them as measured.
    """
    import pipeline.baselines

    wide = make_panel(n_dates=800, n_tickers=20, seed=83)
    wide["close"] = 100.0
    wide["benchmark_close"] = np.nan
    monkeypatch.setattr(pipeline.baselines, "load_panel", lambda **k: wide)

    comparison = pipeline.baselines.compare_baselines(with_pooled_xgb=False)
    names = {r["name"] for r in comparison.results}
    assert names == set(B.BASELINES)
    assert not names & set(__import__("pipeline.series", fromlist=["x"]).SERIES_BASELINES)
    assert "benchmark_close" in comparison.note


def test_series_comparators_are_scored_when_the_level_is_present(monkeypatch):
    """The guard must not be so eager that the series row never appears."""
    import pipeline.baselines
    from pipeline.series import SERIES_BASELINES

    n = 800
    panel = make_panel(n_dates=n, n_tickers=20, seed=89)
    rng = np.random.default_rng(89)
    panel["close"] = 100.0 * np.exp(rng.normal(0, 0.01, len(panel)).cumsum())
    panel["benchmark_close"] = 20000.0
    monkeypatch.setattr(pipeline.baselines, "load_panel", lambda **k: panel)

    comparison = pipeline.baselines.compare_baselines(with_pooled_xgb=False)
    names = {r["name"] for r in comparison.results}
    assert set(SERIES_BASELINES).issubset(names), names
    assert comparison.note == ""


# ── The weekly job regenerates the comparison ─────────────────────────────────


def _stub_weekly(monkeypatch, **overrides):
    """
    Stubs every expensive or networked step of the weekly job.

    Returns the dict that finish_run was called with, so a test can assert on
    what the run actually recorded rather than on what it logged.
    """
    import data.tickers
    import data.universe
    import pipeline.fetch
    import pipeline.model
    import pipeline.signals
    import pipeline.tracking
    import pipeline.validation
    import scheduler

    recorded: dict = {}

    def _finish(run_id, status, gate=None, metrics=None, notes=None, engine=None):
        recorded.update({"run_id": run_id, "status": status,
                         "metrics": metrics or {}, "notes": notes})

    monkeypatch.setattr(data.universe, "sync_current_membership", lambda: None)
    monkeypatch.setattr(data.tickers, "refresh_metadata", lambda: None)
    monkeypatch.setattr(data.universe, "get_ingest_universe", lambda: ["A.NS"])
    monkeypatch.setattr(data.universe, "get_universe",
                        overrides.get("get_universe", lambda: ["A.NS", "B.NS"]))
    monkeypatch.setattr(pipeline.fetch, "fetch_and_store", lambda **k: None)
    monkeypatch.setattr(pipeline.signals, "compute_and_store",
                        lambda **k: pipeline.signals.SignalsReport(
                            10, ["A.NS", "B.NS"], [], []))
    monkeypatch.setattr(pipeline.signals, "count_labelled_rows", lambda *a: 100)
    monkeypatch.setattr(pipeline.validation, "run_gate",
                        lambda *a, **k: pipeline.validation.GateReport(
                            pipeline.validation.PASS, []))
    monkeypatch.setattr(pipeline.tracking, "start_run", lambda *a, **k: "run-1")
    monkeypatch.setattr(pipeline.tracking, "finish_run", _finish)
    monkeypatch.setattr(pipeline.model, "evaluate_and_persist_universe",
                        overrides.get("evaluate_and_persist_universe",
                                      lambda **k: {"A.NS": {}, "B.NS": {}}))

    if "compare_baselines" in overrides:
        import pipeline.baselines
        monkeypatch.setattr(pipeline.baselines, "compare_baselines",
                            overrides["compare_baselines"])

    del scheduler
    return recorded


def test_weekly_job_records_the_baseline_comparison(monkeypatch):
    """
    The comparison is regenerated by the job, not by a person remembering to
    run a tool. tools/report_performance.py exists because the README's
    hand-typed metrics were wrong and stayed wrong; a comparison table that
    only refreshes when someone runs a CLI decays the same way.
    """
    import pipeline.baselines
    import scheduler

    comparison = pipeline.baselines.BaselineComparison(
        coverage={"tickers": 2, "dates": 900, "labelled_rows": 1800,
                  "median_names_per_date": 2.0},
        results=[{"name": "zero", "daily_rank_ic": float("nan"), "mae": 0.069,
                  "n_oos": 100, "beats_naive_mae": False},
                 {"name": "linear_factor", "daily_rank_ic": 0.02, "mae": 0.070,
                  "n_oos": 100, "beats_naive_mae": False}],
    )
    recorded = _stub_weekly(monkeypatch,
                            compare_baselines=lambda **k: comparison)

    scheduler.run_weekly_evaluation_job()

    assert recorded["status"] == "OK"
    names = [c["name"] for c in recorded["metrics"]["baselines"]["comparators"]]
    assert names == ["zero", "linear_factor"]


def test_weekly_job_scores_the_baselines_over_the_screened_universe(monkeypatch):
    """
    The comparison must cover the tickers the run actually evaluated, not
    whatever happens to be sitting in the signals table. A panel built from a
    wider set would be measuring a different universe from the leaderboard it
    is supposed to describe.
    """
    import pipeline.baselines
    import scheduler

    seen: dict = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        return pipeline.baselines.BaselineComparison(note="stub")

    _stub_weekly(monkeypatch, compare_baselines=_spy,
                 get_universe=lambda: ["A.NS", "B.NS", "C.NS"])
    scheduler.run_weekly_evaluation_job()

    assert seen.get("tickers") == ["A.NS", "B.NS", "C.NS"]


def test_a_failed_baseline_comparison_does_not_fail_the_week(monkeypatch):
    """
    The comparison reads; it never writes to signals, forecasts or
    model_metadata. The "a job that writes nothing must raise" rule exists to
    stop a run PUBLISHING nothing while reporting success — this step publishes
    nothing by design, so failing an hour of persisted evaluation over it would
    trade a real result for a measurement.
    """
    import scheduler

    def _boom(**k):
        raise RuntimeError("panel exploded")

    recorded = _stub_weekly(monkeypatch, compare_baselines=_boom)
    scheduler.run_weekly_evaluation_job()

    assert recorded["status"] == "OK"
    assert "panel exploded" in recorded["metrics"]["baselines"]["note"]


def test_a_failed_baseline_comparison_is_still_recorded(monkeypatch):
    """
    Non-fatal must not mean invisible. A step that can fail without failing the
    run has to leave a trace, or its absence reads as "it was fine".
    """
    import scheduler

    def _boom(**k):
        raise RuntimeError("panel exploded")

    recorded = _stub_weekly(monkeypatch, compare_baselines=_boom)
    scheduler.run_weekly_evaluation_job()

    assert recorded["metrics"]["baselines"]["comparators"] == []
    assert recorded["metrics"]["baselines"]["note"]


def test_a_refused_comparison_is_recorded_with_its_reason(monkeypatch):
    """A panel too thin to rank must record why, not an empty table."""
    import pipeline.baselines
    import scheduler

    refusal = pipeline.baselines.BaselineComparison(
        coverage={"tickers": 3, "median_names_per_date": 3.0},
        note="panel too thin to rank: the median date holds 3 names",
    )
    recorded = _stub_weekly(monkeypatch, compare_baselines=lambda **k: refusal)
    scheduler.run_weekly_evaluation_job()

    assert recorded["status"] == "OK"
    assert "too thin" in recorded["metrics"]["baselines"]["note"]
    assert recorded["metrics"]["baselines"]["comparators"] == []


# ── What gets written must be valid JSON ──────────────────────────────────────


def test_undefined_metrics_are_written_as_null_not_nan():
    """
    json.dumps serialises float('nan') as a bare NaN token. Python's own
    json.loads accepts it as an extension, so it round-trips locally and looks
    correct — while every strict parser rejects it: JavaScript's JSON.parse,
    jq, and a Postgres ::jsonb cast alike.

    This is not an edge case here. A comparator with no ordering has an
    undefined rank IC BY DESIGN, so any table containing `zero` or `majority`
    produces NaN on every single weekly run.
    """
    from pipeline.tracking import json_safe

    payload = {"comparators": [{"name": "zero", "daily_rank_ic": float("nan"),
                                "alpha_t": float("inf"), "n_oos": 100}]}
    encoded = json.dumps(json_safe(payload), allow_nan=False)
    assert '"daily_rank_ic": null' in encoded
    assert '"alpha_t": null' in encoded
    assert '"n_oos": 100' in encoded


def test_finish_run_writes_json_a_strict_parser_can_read():
    """
    Wiring, not the helper. json_safe existing is worth nothing if the write
    boundary does not call it — and since finish_run swallows its own
    exceptions so that tracking can never fail a run, a bad encode would be
    invisible apart from one printed line.
    """
    from sqlalchemy import create_engine, text

    from pipeline.tracking import finish_run

    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE experiment_runs (run_id TEXT PRIMARY KEY, "
            "finished_at TEXT, status TEXT, gate_status TEXT, "
            "gate_report TEXT, metrics TEXT, notes TEXT)")
        conn.exec_driver_sql("INSERT INTO experiment_runs (run_id) VALUES ('r1')")

    finish_run("r1", "OK", metrics={
        "baselines": {"comparators": [{"name": "zero",
                                       "daily_rank_ic": float("nan")}]}},
        engine=engine)

    with engine.connect() as conn:
        stored = conn.execute(
            text("SELECT metrics FROM experiment_runs WHERE run_id = 'r1'")
        ).scalar()

    assert stored and "NaN" not in stored, stored
    parsed = json.loads(stored)
    assert parsed["baselines"]["comparators"][0]["daily_rank_ic"] is None


def test_json_safe_leaves_finite_numbers_alone():
    from pipeline.tracking import json_safe

    assert json_safe({"a": 0.0, "b": [1.5, -2.5], "c": "x", "d": True}) == \
        {"a": 0.0, "b": [1.5, -2.5], "c": "x", "d": True}


def test_a_baseline_comparison_survives_a_round_trip_through_finish_run():
    """
    End to end: the real to_metrics() output, encoded the way finish_run
    encodes it, must be readable by a strict parser.
    """
    from pipeline.baselines import BaselineComparison
    from pipeline.tracking import json_safe

    comparison = BaselineComparison(
        coverage={"tickers": 46, "dates": 2420, "labelled_rows": 106858,
                  "median_names_per_date": 45.0},
        results=[
            {"name": "zero", "n_oos": 86161, "folds": 5,
             "daily_rank_ic": float("nan"), "rebalance_ic_t": float("nan"),
             "hit_rate": 48.11, "majority_hit_rate": 51.89, "mae": 0.06907,
             "mae_naive_zero": 0.06907, "beats_naive_mae": False,
             "alpha_vs_equal_weight": float("nan"), "alpha_t": float("nan"),
             "n_rebalances": 0},
            {"name": "linear_factor", "n_oos": 86161, "folds": 5,
             "daily_rank_ic": 0.0215, "rebalance_ic_t": 0.91,
             "hit_rate": 51.60, "majority_hit_rate": 51.89, "mae": 0.06938,
             "mae_naive_zero": 0.06907, "beats_naive_mae": False,
             "alpha_vs_equal_weight": 0.00023, "alpha_t": 0.05,
             "n_rebalances": 63},
        ],
        loadings={"prox_52w": 0.00561},
    )

    encoded = json.dumps(json_safe(comparison.to_metrics()),
                         default=str, allow_nan=False)
    restored = json.loads(encoded)

    assert restored["panel_tickers"] == 46
    assert restored["comparators"][0]["daily_rank_ic"] is None
    assert restored["comparators"][1]["daily_rank_ic"] == 0.0215
    assert restored["loadings"]["prox_52w"] == 0.00561


# ── Retargeting the horizon ───────────────────────────────────────────────────


def _price_panel_with_gaps(n_dates=700, n_tickers=14, horizon=30, seed=5):
    """
    A panel where tickers do NOT share every date.

    The gaps are the point. `relative_price_frame` pivots onto the UNION of all
    dates, so a ticker absent from a date another one trades gets a placeholder
    row there. Shifting the wide frame steps that placeholder; shifting within
    the ticker does not. A fixture where everyone trades every day cannot tell
    the two apart.
    """
    rng = np.random.default_rng(seed)
    dates = [f"D{i:05d}" for i in range(n_dates)]
    bench = 20000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, n_dates)))

    rows = []
    for k in range(n_tickers):
        close = 500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.016, n_dates)))
        log_rel = np.log(close / bench)
        keep = np.ones(n_dates, dtype=bool)
        if k % 3 == 1:                       # a late listing
            keep[: 40 + k] = False
        if k % 3 == 2:                       # an interior suspension
            keep[200 + k: 200 + k + 7] = False

        idx = np.flatnonzero(keep)
        # The label is the ticker's OWN h-session forward move, which is what
        # pipeline/signals.py computes.
        target = np.full(len(idx), np.nan)
        if len(idx) > horizon:
            target[:-horizon] = log_rel[idx][horizon:] - log_rel[idx][:-horizon]

        for j, i in enumerate(idx):
            row = {"date": dates[i], "ticker": f"T{k:02d}.NS",
                   "close": close[i], "benchmark_close": bench[i],
                   "benchmark_ticker": "^NSEI", P.TARGET: target[j]}
            for c in P.FEATURE_COLS:
                row[c] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def test_retargeting_at_the_stored_horizon_reproduces_the_stored_label():
    """
    THE CALIBRATION CASE. Retargeting is exact by the relative-price identity,
    so asking for the horizon the label already uses must return the label
    itself. Anything else means the derivation is wrong, and every other
    horizon in a sweep is wrong with it.
    """
    panel = _price_panel_with_gaps()
    back = P.retarget_horizon(panel, 30)

    a = pd.to_numeric(panel[P.TARGET], errors="coerce")
    b = pd.to_numeric(back[P.TARGET], errors="coerce")
    both = a.notna() & b.notna()

    assert both.sum() > 5_000
    assert np.abs(a[both] - b[both]).max() < 1e-9
    assert b.notna().sum() == a.notna().sum()


def test_the_shift_is_within_each_ticker_not_across_the_date_grid():
    """
    The wide frame's index is the UNION of every ticker's dates. Shifting it
    steps rows that belong to OTHER tickers, so a name with a late listing or a
    suspension silently gets a longer horizon than the table claims — and only
    for the names whose history is most irregular.

    Measured on real data before the fix: the identity broke by 4.8e-02 against
    a stored label that reproduces at 3.3e-15 once the shift is per ticker.
    """
    panel = _price_panel_with_gaps()
    back = P.retarget_horizon(panel, 30)

    gapped = panel[panel["ticker"] == "T02.NS"]          # the suspension case
    assert len(gapped) < panel["date"].nunique(), "fixture must have a gap"

    a = pd.to_numeric(panel[P.TARGET], errors="coerce")
    b = pd.to_numeric(back[P.TARGET], errors="coerce")
    per_ticker = panel.assign(_a=a, _b=b).groupby("ticker").apply(
        lambda g: np.abs(g["_a"] - g["_b"]).max(), include_groups=False)

    assert per_ticker.max() < 1e-9, (
        f"worst ticker {per_ticker.idxmax()} off by {per_ticker.max():.2e}")


def test_a_shorter_horizon_produces_a_smaller_target():
    """A 5-session move is smaller than a 30-session one; roughly sqrt(6)x."""
    panel = _price_panel_with_gaps()
    sd = {h: pd.to_numeric(P.retarget_horizon(panel, h)[P.TARGET],
                           errors="coerce").std() for h in (5, 10, 20, 30)}
    assert sd[5] < sd[10] < sd[20] < sd[30]


def test_a_shorter_horizon_labels_more_rows():
    """Only the trailing h rows per ticker are unlabelled, so smaller h labels more."""
    panel = _price_panel_with_gaps()
    n = {h: pd.to_numeric(P.retarget_horizon(panel, h)[P.TARGET],
                          errors="coerce").notna().sum() for h in (5, 30)}
    assert n[5] > n[30]


def test_retargeting_refuses_without_a_benchmark_level():
    panel = _price_panel_with_gaps()
    panel["benchmark_close"] = np.nan
    with pytest.raises(ValueError, match="benchmark_close"):
        P.retarget_horizon(panel, 10)


def test_retargeting_refuses_on_thin_benchmark_coverage():
    """
    Retargeting a partly-covered panel would drop the uncovered tickers, so a
    sweep would compare horizons over DIFFERENT universes — the exact confound
    a sweep exists to avoid.
    """
    panel = _price_panel_with_gaps()
    half = panel["ticker"].isin(panel["ticker"].unique()[:7])
    panel.loc[half, "benchmark_close"] = np.nan
    with pytest.raises(ValueError, match="below"):
        P.retarget_horizon(panel, 10)


def test_the_horizon_reaches_the_splitter_and_the_rebalance_schedule(monkeypatch):
    """
    A sweep that retargeted the label but left the embargo and the rebalance
    frequency at 30 would purge six times too much at h=5 and count overlapping
    windows as independent.
    """
    import pipeline.baselines

    panel = _price_panel_with_gaps(n_dates=800, n_tickers=20, seed=11)
    monkeypatch.setattr(pipeline.baselines, "load_panel", lambda **k: panel)

    seen = {}
    real = pipeline.baselines.panel_walk_forward

    def spy(**kw):
        seen["rebalance_every"] = kw["rebalance_every"]
        seen["horizon"] = kw["splitter"].horizon
        seen["embargo"] = kw["splitter"].embargo
        return real(**kw)

    monkeypatch.setattr(pipeline.baselines, "panel_walk_forward", spy)
    pipeline.baselines.compare_baselines(
        with_pooled_xgb=False, with_series=False, horizon=5)

    assert seen["rebalance_every"] == 5
    assert seen["horizon"] == 5
    assert seen["embargo"] == 5
