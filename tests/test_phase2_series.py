"""
tests/test_phase2_series.py — Guards for the series-forecaster adapter.

The adapter exists so that Chronos-2 and TimesFM-2.5 can run in the same
harness as everything else. It is tested before either of them arrives, with
forecasters whose answers are known in advance, because a wrong adapter makes a
foundation model's result uninterpretable: a poor number could be the model or
the plumbing and nothing distinguishes them afterwards.

The centrepiece is ``test_a_forecaster_cannot_see_past_its_as_of_date``.

A zero-shot model is never fitted, so ``fit`` is a no-op and every guarantee the
purged folds provide is inert — purging, the embargo and the training boundary
all constrain what a model is FITTED on, and nothing is fitted. The whole
protection is one slice: the history handed to the forecaster must end at the
as-of date. That test corrupts every value after it and requires the
predictions to come back bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import baselines as B
from pipeline import panel as P
from pipeline import series as S
from pipeline.evaluation import PurgedPanelWalkForward, panel_walk_forward


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_price_panel(n_dates: int = 900, n_tickers: int = 12, seed: int = 0,
                     horizon: int = 30) -> pd.DataFrame:
    """
    A panel built from real price paths, with the target derived exactly the way
    pipeline/signals.py derives it.

    Deriving rather than inventing the target matters: these tests assert an
    identity between the label and the relative-price series, and a target
    generated independently of the prices could satisfy it only by accident.
    """
    rng = np.random.default_rng(seed)
    dates = [f"D{i:05d}" for i in range(n_dates)]
    tickers = [f"T{i:02d}.NS" for i in range(n_tickers)]

    bench = 20000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, n_dates)))

    rows = []
    for t in tickers:
        close = 500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.016, n_dates)))
        log_close, log_bench = np.log(close), np.log(bench)

        target_return = np.full(n_dates, np.nan)
        bench_return = np.full(n_dates, np.nan)
        target_return[:-horizon] = log_close[horizon:] - log_close[:-horizon]
        bench_return[:-horizon] = log_bench[horizon:] - log_bench[:-horizon]

        block = pd.DataFrame({
            "date": dates, "ticker": t, "close": close,
            "benchmark_close": bench, "benchmark_ticker": "^NSEI",
            "target_return": target_return,
            "benchmark_return": bench_return,
            P.TARGET: target_return - bench_return,
        })
        for col in P.FEATURES:
            block[col] = rng.normal(size=n_dates)
        rows.append(block)

    return pd.concat(rows, ignore_index=True).sort_values(
        ["date", "ticker"]).reset_index(drop=True)


def _splitter(n_folds: int = 3) -> PurgedPanelWalkForward:
    return PurgedPanelWalkForward(n_folds=n_folds, horizon=30, min_train=500)


# ── The identity the whole design rests on ────────────────────────────────────


def test_the_relative_series_forward_return_is_the_label():
    """
    log(rel[t+h]) - log(rel[t]) == target_excess_return, exactly.

    This is why a univariate time-series model can predict a benchmark-relative
    target at all. Forecasting the stock and the index separately and
    subtracting compounds two independent errors into a quantity smaller than
    either; forecasting this one series does not. If the identity ever stops
    holding, every series forecaster is silently predicting a different
    quantity from the one it is scored on.
    """
    panel = make_price_panel(n_dates=400, n_tickers=6, seed=1)
    wide = P.relative_price_frame(panel)

    forward = wide.shift(-30) - wide
    # pivot_table drops rows that are entirely NaN, so the label frame is
    # shorter than the series frame; align them before comparing.
    expected = panel.pivot_table(index="date", columns="ticker",
                                 values=P.TARGET, aggfunc="last")
    expected = expected.reindex(index=forward.index, columns=forward.columns)

    both = forward.notna() & expected.notna()
    assert both.to_numpy().sum() > 1000, "fixture produced nothing to compare"
    np.testing.assert_allclose(
        forward.to_numpy()[both.to_numpy()],
        expected.to_numpy()[both.to_numpy()],
        atol=1e-12,
    )


def test_relative_series_is_empty_without_a_benchmark_level():
    """
    A row written before benchmark_close was persisted has a perfectly good
    target and no way to reconstruct the series it came from. That must read as
    a gap, not as a zero.
    """
    panel = make_price_panel(n_dates=200, n_tickers=4, seed=3)
    panel["benchmark_close"] = np.nan
    wide = P.relative_price_frame(panel)
    assert wide.isna().all().all()


def test_panel_coverage_reports_benchmark_level_separately_from_labels():
    panel = make_price_panel(n_dates=200, n_tickers=4, seed=5)
    panel.loc[panel["ticker"] == "T00.NS", "benchmark_close"] = np.nan
    cov = P.panel_coverage(panel)
    assert cov["rows_with_benchmark_close"] == 600
    assert cov["labelled_rows"] > cov["rows_with_benchmark_close"]


# ── The guarantee that replaces the purged folds ──────────────────────────────


class _Recorder:
    """Records the history it was handed, so a test can inspect the slice."""

    name = "recorder"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def forecast(self, histories, horizon):
        for ticker, values in histories.items():
            self.calls.append((ticker, len(values)))
        return {t: 0.0 for t in histories}


def test_a_forecaster_cannot_see_past_its_as_of_date():
    """
    THE test for this module.

    A zero-shot model is never fitted, so purging, the embargo and the training
    boundary are all inert — they constrain what a model is FITTED on and
    nothing is fitted. The entire protection is that the history ends at the
    as-of date.

    Corrupting every observation after that date must change nothing. If it
    does, the model is reading its own answer, and the result would look like a
    breakthrough rather than a bug: F1 with the meta-learner replaced by a
    200M-parameter model, and no in-sample fit to re-run as evidence.
    """
    panel = make_price_panel(n_dates=900, n_tickers=8, seed=7)
    clean = P.relative_price_frame(panel)

    rng = np.random.default_rng(11)
    corrupted = clean.copy()
    after = corrupted.index > corrupted.index[600]
    corrupted.loc[after, :] += rng.normal(0, 5.0, size=(after.sum(),
                                                        corrupted.shape[1]))
    assert not np.allclose(clean.to_numpy()[after], corrupted.to_numpy()[after])

    rows = panel[panel["date"] <= clean.index[600]][["date", "ticker"]]
    a = S.SeriesAdapter(S.LastReturn(), clean).predict(rows)
    b = S.SeriesAdapter(S.LastReturn(), corrupted).predict(rows)

    assert np.isfinite(a).all() and np.abs(a).sum() > 0, "nothing was predicted"
    np.testing.assert_array_equal(a, b)


def test_the_history_ends_on_the_as_of_date_not_before_it():
    """
    Inclusive, not exclusive. An off-by-one in the safe direction is still a
    defect: it silently shortens every context by one observation and makes the
    trailing-return forecasters measure a window they did not intend.
    """
    panel = make_price_panel(n_dates=300, n_tickers=3, seed=13)
    wide = P.relative_price_frame(panel)
    positions = {d: i for i, d in enumerate(wide.index)}
    as_of = wide.index[200]

    # Above MIN_CONTEXT, or _history_ending_at declines every ticker and the
    # assertions below run over an empty dict — which is how this test passed
    # against a slice that stopped one row short.
    histories = S._history_ending_at(wide, positions, as_of, context=150)
    assert len(histories) == 3, histories.keys()
    for ticker, values in histories.items():
        assert len(values) == 150
        assert values[-1] == pytest.approx(wide.loc[as_of, ticker])


def test_the_context_window_is_respected():
    panel = make_price_panel(n_dates=600, n_tickers=3, seed=17)
    wide = P.relative_price_frame(panel)
    positions = {d: i for i, d in enumerate(wide.index)}

    histories = S._history_ending_at(wide, positions, wide.index[500],
                                     context=128)
    assert histories and all(len(v) == 128 for v in histories.values())


def test_a_ticker_without_enough_history_is_declined_not_guessed():
    panel = make_price_panel(n_dates=300, n_tickers=3, seed=19)
    wide = P.relative_price_frame(panel)
    positions = {d: i for i, d in enumerate(wide.index)}

    early = S._history_ending_at(wide, positions, wide.index[10], context=2048)
    assert early == {}


def test_an_abstention_predicts_no_excess_return():
    """
    A ticker the forecaster declines to score gets 0.0 — the same claim the
    `zero` floor makes. Any other default would let abstaining flatter a model
    against the comparator it is measured against.
    """
    panel = make_price_panel(n_dates=300, n_tickers=3, seed=23)
    wide = P.relative_price_frame(panel)

    class _Abstains:
        name = "abstains"

        def forecast(self, histories, horizon):
            return {}

    rows = panel[panel["date"] == wide.index[200]][["date", "ticker"]]
    out = S.SeriesAdapter(_Abstains(), wide).predict(rows)
    assert (out == 0.0).all()


# ── The calibration case ──────────────────────────────────────────────────────


def test_zero_drift_through_the_adapter_matches_the_zero_baseline_exactly():
    """
    THE CALIBRATION CASE, and the reason this module was built before any real
    forecaster. ZeroDrift routed through the adapter and ZeroForecast routed
    through the feature path are the same claim, so they must produce the same
    rows and the same metrics. A difference is the adapter, not the model — and
    once a 120M-parameter model is on this code path, that distinction can no
    longer be drawn.
    """
    panel = make_price_panel(n_dates=900, n_tickers=12, seed=29)
    wide = P.relative_price_frame(panel)
    splitter = _splitter()

    through_features = panel_walk_forward(
        panel=panel, feature_cols=B.FACTORS, model_factory=B.ZeroForecast,
        splitter=splitter, name="zero")

    through_series = panel_walk_forward(
        panel=panel, feature_cols=["date", "ticker"],
        model_factory=S.adapter_factory(S.ZeroDrift, wide),
        splitter=splitter, name="series_zero")

    assert through_series.n_predictions == through_features.n_predictions > 0
    pd.testing.assert_frame_equal(
        through_series.predictions.sort_values(["date", "ticker"]
                                               ).reset_index(drop=True),
        through_features.predictions.sort_values(["date", "ticker"]
                                                 ).reset_index(drop=True),
    )
    assert through_series.metrics["mae"] == pytest.approx(
        through_features.metrics["mae"])


# ── The trivial forecasters ───────────────────────────────────────────────────


def test_last_return_recovers_a_known_trailing_move():
    dates = [f"D{i:04d}" for i in range(200)]
    # A series rising by exactly 0.001 per session: any 30-session trailing
    # move is 0.030.
    wide = pd.DataFrame({"A.NS": np.arange(200) * 0.001}, index=dates)
    rows = pd.DataFrame({"date": [dates[150]], "ticker": ["A.NS"]})

    out = S.SeriesAdapter(S.LastReturn(), wide, horizon=30).predict(rows)
    assert out[0] == pytest.approx(0.030)


def test_historical_drift_recovers_a_known_drift():
    dates = [f"D{i:04d}" for i in range(200)]
    wide = pd.DataFrame({"A.NS": np.arange(200) * 0.002}, index=dates)
    rows = pd.DataFrame({"date": [dates[150]], "ticker": ["A.NS"]})

    out = S.SeriesAdapter(S.HistoricalDrift(), wide, horizon=30).predict(rows)
    assert out[0] == pytest.approx(0.060)


def test_last_return_reads_the_relative_series_not_the_stock():
    """
    A stock that rises exactly as fast as its benchmark has a flat relative
    series and therefore no predicted excess return, however large its own move
    was. This is what makes the forecast benchmark-relative rather than
    directional.
    """
    n = 200
    dates = [f"D{i:04d}" for i in range(n)]
    close = 100.0 * np.exp(np.arange(n) * 0.004)
    bench = 20000.0 * np.exp(np.arange(n) * 0.004)

    panel = pd.DataFrame({"date": dates, "ticker": "A.NS", "close": close,
                          "benchmark_close": bench})
    wide = P.relative_price_frame(panel)
    rows = pd.DataFrame({"date": [dates[150]], "ticker": ["A.NS"]})

    out = S.SeriesAdapter(S.LastReturn(), wide, horizon=30).predict(rows)
    assert out[0] == pytest.approx(0.0, abs=1e-12)


def test_every_series_baseline_runs_through_the_harness():
    panel = make_price_panel(n_dates=900, n_tickers=12, seed=31)
    wide = P.relative_price_frame(panel)

    for name, cls in S.SERIES_BASELINES.items():
        res = panel_walk_forward(
            panel=panel, feature_cols=["date", "ticker"],
            model_factory=S.adapter_factory(cls, wide),
            splitter=_splitter(2), name=name)
        assert res.n_predictions > 0, f"{name} produced nothing"


# ── Adapter contract ──────────────────────────────────────────────────────────


def test_the_adapter_refuses_a_frame_without_date_and_ticker():
    wide = pd.DataFrame({"A.NS": [1.0, 2.0]}, index=["D0", "D1"])
    with pytest.raises(ValueError, match="date"):
        S.SeriesAdapter(S.ZeroDrift(), wide).predict(pd.DataFrame({"rsi": [1.0]}))


def test_the_adapter_sorts_an_unsorted_series_frame():
    """
    The positional slice is only correct on a sorted index. An unsorted frame
    must be sorted rather than trusted — silently slicing an unsorted index is
    exactly how a history ends up containing later dates.
    """
    dates = [f"D{i:04d}" for i in range(200)]
    wide = pd.DataFrame({"A.NS": np.arange(200) * 0.001}, index=dates)
    shuffled = wide.sample(frac=1.0, random_state=0)
    assert not shuffled.index.is_monotonic_increasing

    rows = pd.DataFrame({"date": [dates[150]], "ticker": ["A.NS"]})
    out = S.SeriesAdapter(S.LastReturn(), shuffled, horizon=30).predict(rows)
    assert out[0] == pytest.approx(0.030)


def test_the_forecaster_is_called_once_per_date_not_once_per_row():
    """
    Batching by date is not a micro-optimisation. Per-series inference over
    this panel is ~180,000 calls at 95 tickers, which is not affordable on a
    two-core runner; one call per date is ~1,900. It is also the shape
    Chronos-2's group attention wants, since the batch it conditions across is
    a cross-section.
    """
    panel = make_price_panel(n_dates=700, n_tickers=6, seed=37)
    wide = P.relative_price_frame(panel)

    recorder = _Recorder()
    rows = panel[panel["date"].isin(wide.index[600:610])][["date", "ticker"]]
    assert len(rows) == 60

    calls: list[int] = []

    class _Counting(_Recorder):
        def forecast(self, histories, horizon):
            calls.append(len(histories))
            return super().forecast(histories, horizon)

    S.SeriesAdapter(_Counting(), wide).predict(rows)
    assert len(calls) == 10 and all(n == 6 for n in calls), calls
    del recorder


def test_fit_is_a_no_op_and_returns_the_adapter():
    """
    Zero-shot. Pinned because the day someone adds training here, every
    statement in this module's docstring about where the protection lives stops
    being true — and the purged folds would then matter again.
    """
    wide = pd.DataFrame({"A.NS": [1.0, 2.0]}, index=["D0", "D1"])
    adapter = S.SeriesAdapter(S.ZeroDrift(), wide)
    before = adapter.__dict__.copy()
    assert adapter.fit(pd.DataFrame({"a": [1.0]}), pd.Series([0.1])) is adapter
    assert adapter.__dict__.keys() == before.keys()


# ── A zero-volume session must not delete a different session ─────────────────


def _signals_frame(volume: np.ndarray, n: int = 400):
    """Builds a signals frame offline, with the benchmark and earnings stubbed."""
    from pipeline import signals

    sessions = pd.bdate_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    prices = np.linspace(100.0, 180.0, n)
    ohlcv = pd.DataFrame({
        "ticker": "TEST.NS", "date": sessions, "open": prices,
        "high": prices * 1.01, "low": prices * 0.99, "close": prices,
        "adj_close": prices, "volume": volume,
    })
    bench = pd.DataFrame({"date": sessions,
                          "benchmark_close": np.linspace(20000.0, 26000.0, n)})

    original = (signals.get_benchmark_series, signals.get_benchmark,
                signals.compute_earnings_surprise)
    signals.get_benchmark_series = lambda *a, **k: bench
    signals.get_benchmark = lambda t: ("^NSEI", False)
    signals.compute_earnings_surprise = lambda t, df: df.assign(earnings_surprise=0.0)
    try:
        return signals.compute_signals_frame("TEST.NS", ohlcv), sessions
    finally:
        (signals.get_benchmark_series, signals.get_benchmark,
         signals.compute_earnings_surprise) = original


def test_a_zero_volume_session_does_not_delete_the_row_ten_sessions_later():
    """
    vroc_10 = volume.pct_change(10) divides by the volume ten rows back, so a
    single zero-volume day produced inf ten rows LATER, which became NaN and
    then lost that row to dropna(subset=FEATURE_COLS).

    RELIANCE.NS carried five zero-volume sessions and was missing exactly five
    interior rows because of it — healthy sessions, normal OHLC, deleted by a
    defect in a neighbour a fortnight earlier, with nothing in any log. 43 of
    46 tickers in the local database were affected.
    """
    volume = np.full(400, 1_000_000.0)
    volume[100] = 0.0

    frame, sessions = _signals_frame(volume)
    assert frame is not None

    victim = sessions[110]
    assert victim in set(frame["date"]), (
        f"{victim} was deleted because volume was zero ten sessions earlier"
    )


def test_the_stored_grid_has_no_interior_gaps_after_a_zero_volume_day():
    """
    The lost row is the smaller half. The larger half is that the stored
    session grid develops invisible holes while target_excess_return is still
    computed on the full ohlcv sequence, so a row-stepped 30-session horizon
    silently measures 31 sessions across the hole.
    """
    volume = np.full(400, 1_000_000.0)
    volume[[100, 150, 200]] = 0.0

    frame, sessions = _signals_frame(volume)
    stored = list(frame["date"])
    lo, hi = stored[0], stored[-1]
    expected = [d for d in sessions if lo <= d <= hi]

    assert stored == expected, (
        f"{len(expected) - len(stored)} interior sessions missing from the "
        f"stored grid"
    )


def test_the_written_frame_carries_the_benchmark_level():
    """
    benchmark_return is a label — it looks 30 sessions ahead — so nothing may
    read it as an input. Without the LEVEL there is no relative-price series at
    all, and every series forecaster silently falls back to predicting zero.
    """
    frame, _ = _signals_frame(np.full(400, 1_000_000.0))
    assert "benchmark_close" in frame.columns
    assert frame["benchmark_close"].notna().all()


def test_the_identity_holds_through_the_real_signals_code_path():
    """
    The synthetic panel proves the arithmetic. This proves it survives the
    round trip through compute_signals_frame — the adjustment basis, the
    benchmark merge, the ffill and the dropna — which is where an identity like
    this normally dies. Measured live on RELIANCE/WIPRO/HDFCBANK the residual
    is 2.7e-15 against a target ranging +/-0.3.
    """
    from pipeline.signals import HORIZON_SESSIONS

    frame, _ = _signals_frame(np.full(400, 1_000_000.0))
    rel = np.log(frame["close"].astype(float)
                 / frame["benchmark_close"].astype(float))
    forward = rel.shift(-HORIZON_SESSIONS) - rel

    both = forward.notna() & frame["target_excess_return"].notna()
    assert both.sum() > 100, "nothing to compare"
    np.testing.assert_allclose(
        forward[both].to_numpy(),
        frame.loc[both, "target_excess_return"].to_numpy(),
        atol=1e-12,
    )


def test_vroc_is_finite_and_neutral_when_the_base_volume_was_zero():
    """
    A rate of change against zero is undefined, not infinite. 0.0 is the honest
    reading and it keeps the row; the alternative that was there before turned
    an undefined value into a deletion.
    """
    volume = np.full(400, 1_000_000.0)
    volume[100] = 0.0

    frame, sessions = _signals_frame(volume)
    row = frame[frame["date"] == sessions[110]]
    assert len(row) == 1
    assert np.isfinite(row["vroc_10"].iloc[0])
    assert row["vroc_10"].iloc[0] == pytest.approx(0.0)


def test_a_genuine_volume_collapse_is_still_reported():
    """
    The fix must not flatten real information. The zero-volume row ITSELF has a
    well-defined -100% rate of change against a non-zero base, and that is a
    true statement about the session.
    """
    volume = np.full(400, 1_000_000.0)
    volume[100] = 0.0

    frame, sessions = _signals_frame(volume)
    row = frame[frame["date"] == sessions[100]]
    assert len(row) == 1
    assert row["vroc_10"].iloc[0] == pytest.approx(-1.0)


# ── The gate notices a gapped grid ────────────────────────────────────────────


def _gap_fixture(missing_dates: list[str]):
    """An ohlcv table and a signals table that omits `missing_dates`."""
    from sqlalchemy import create_engine

    sessions = pd.bdate_range("2024-01-01", periods=120).strftime("%Y-%m-%d")
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE ohlcv (ticker TEXT, date TEXT, close REAL)")
        conn.exec_driver_sql(
            "CREATE TABLE signals (ticker TEXT, date TEXT, close REAL)")
        for d in sessions:
            conn.exec_driver_sql(
                f"INSERT INTO ohlcv VALUES ('T.NS', '{d}', 100.0)")
            if d not in missing_dates:
                conn.exec_driver_sql(
                    f"INSERT INTO signals VALUES ('T.NS', '{d}', 100.0)")
    return engine, list(sessions)


def test_the_gate_warns_when_the_session_grid_has_interior_gaps():
    from pipeline.validation import WARN, check_sessions_are_contiguous

    _, sessions = _gap_fixture([])
    engine, _ = _gap_fixture([sessions[40], sessions[70]])

    check = check_sessions_are_contiguous(engine, ["T.NS"])
    assert check.status == WARN
    assert "T.NS(-2)" in check.detail


def test_the_gate_passes_on_a_contiguous_grid():
    from pipeline.validation import PASS, check_sessions_are_contiguous

    engine, _ = _gap_fixture([])
    check = check_sessions_are_contiguous(engine, ["T.NS"])
    assert check.status == PASS


def test_the_gate_ignores_rows_outside_the_stored_range():
    """
    A ticker whose signals start later than its ohlcv is not gapped — the
    leading rows are dropped legitimately, because the indicators have no
    lookback yet. Only holes INSIDE the stored range are a defect.
    """
    from pipeline.validation import PASS, check_sessions_are_contiguous
    from sqlalchemy import create_engine

    sessions = pd.bdate_range("2024-01-01", periods=120).strftime("%Y-%m-%d")
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE ohlcv (ticker TEXT, date TEXT, close REAL)")
        conn.exec_driver_sql(
            "CREATE TABLE signals (ticker TEXT, date TEXT, close REAL)")
        for i, d in enumerate(sessions):
            conn.exec_driver_sql(f"INSERT INTO ohlcv VALUES ('T.NS','{d}',100.0)")
            if i >= 50:
                conn.exec_driver_sql(
                    f"INSERT INTO signals VALUES ('T.NS','{d}',100.0)")

    assert check_sessions_are_contiguous(engine, ["T.NS"]).status == PASS


def test_the_contiguity_check_is_wired_into_the_gate():
    from pipeline.validation import CHECKS, check_sessions_are_contiguous

    assert check_sessions_are_contiguous in CHECKS
