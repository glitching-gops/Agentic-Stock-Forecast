"""
The Kronos comparator: the synthetic relative candle, and the as-of boundary.

Most of these run against a STUB predictor whose right answer is arithmetically
known, with decoys at every row and column the real answer does not live in. A
test against the real checkpoint can only check that a number came back, and
would pass just as happily on the 3rd session's `high` as on the 30th session's
`close` — which is the whole failure mode, because either renders a complete
and plausible results table.

For a zero-shot model the purged folds protect nothing: `fit` is a no-op, so
every guarantee collapses onto the history slice ending at the as-of date. The
corruption test below is therefore the most important one in this file.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import pipeline.kronos_forecaster as kf


HORIZON = 5          # short, so a stub's decoys are easy to place and read
CONTEXT = 120        # above MIN_CONTEXT, small enough to build by hand


# ── A stub whose answer is known before the model runs ───────────────────────

class _StubPredictor:
    """
    Returns a forecast whose terminal relative close is exactly
    ``anchor * exp(step)`` for the ticker at input position ``i``, where
    ``step = (i + 1) / 100``.

    Everything else is a decoy. Non-terminal rows are multiplied by 100, and
    open/high/low carry a different multiplier again, so reading the wrong row
    or the wrong column produces a number nowhere near the right one — and the
    assertions below are exact rather than approximate.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list,
                      pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=True):
        self.calls.append({
            "n": len(df_list), "pred_len": pred_len,
            "history_len": [len(d) for d in df_list],
            "last_close": [float(d["close"].iloc[-1]) for d in df_list],
            "x_last": [pd.Timestamp(x.iloc[-1]) for x in x_timestamp_list],
            "sample_count": sample_count,
        })

        out = []
        for i, (df, y) in enumerate(zip(df_list, y_timestamp_list)):
            anchor = float(df["close"].iloc[-1])
            terminal = anchor * np.exp((i + 1) / 100.0)

            close = np.full(pred_len, terminal * 100.0)   # decoy on every row
            close[-1] = terminal                          # the one true value
            frame = pd.DataFrame({
                "open":   close * 7.0,                    # decoy column
                "high":   close * 11.0,
                "low":    close * 13.0,
                "close":  close,
                "volume": np.ones(pred_len),
                "amount": np.ones(pred_len),
            }, index=pd.Index(list(y)[:pred_len]))
            out.append(frame)
        return out


@pytest.fixture
def stub(monkeypatch):
    predictor = _StubPredictor()
    monkeypatch.setattr(kf, "load_predictor",
                        lambda *a, **k: predictor)
    # torch is imported inside forecast_candles purely to seed. Keep the test
    # suite free of it — CI for the daily and weekly jobs must stay green
    # without torch installed at all. It RECORDS the seeds rather than
    # discarding them: a stub that swallows the argument makes the per-date
    # seed untestable, and the first version of this file did exactly that.
    predictor.seeds = []
    monkeypatch.setitem(
        __import__("sys").modules, "torch",
        type("torch", (), {"manual_seed": staticmethod(predictor.seeds.append)}))
    return predictor


def _frames(n_dates=CONTEXT + HORIZON, tickers=("AAA.NS", "BBB.NS", "CCC.NS"),
            seed=0):
    """Well-formed relative candles: high >= max(open, close) >= min >= low."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    frames = {}
    close = pd.DataFrame(
        {t: 0.5 * np.exp(np.cumsum(rng.normal(0, 0.01, n_dates)))
         for t in tickers}, index=dates)
    open_ = close.shift(1).fillna(close.iloc[0])
    frames["open"] = open_
    frames["close"] = close
    frames["high"] = pd.concat([open_, close]).groupby(level=0).max() * 1.01
    frames["low"] = pd.concat([open_, close]).groupby(level=0).min() * 0.99
    frames["volume"] = pd.DataFrame(
        rng.integers(1e5, 1e6, (n_dates, len(tickers))).astype(float),
        index=dates, columns=list(tickers))
    return {k: v.reindex(index=dates, columns=list(tickers))
            for k, v in frames.items()}


def _X(frames, as_of):
    return pd.DataFrame({"date": [as_of] * len(frames["close"].columns),
                         "ticker": list(frames["close"].columns)})


def _adapter(frames, **kwargs):
    return kf.KronosAdapter(
        forecaster=kf.KronosForecaster(context=CONTEXT, **kwargs),
        frames=frames, horizon=HORIZON, context=CONTEXT)


# ── The three indices that each render a plausible table when wrong ──────────

def test_the_prediction_is_the_terminal_close_against_the_anchor(stub):
    """
    predicted = log(forecast close at t+horizon) - log(observed close at t).

    The stub makes the right answer (i+1)/100 exactly. Every decoy is orders of
    magnitude away: a wrong ROW carries a factor of 100 (log 100 = 4.6), a wrong
    COLUMN a factor of 7, 11 or 13.
    """
    frames = _frames()
    as_of = frames["close"].index[CONTEXT - 1]

    out = _adapter(frames).predict(_X(frames, as_of))

    assert out == pytest.approx([0.01, 0.02, 0.03], abs=1e-12)


def test_the_anchor_is_the_as_of_close_not_the_first_row(stub):
    """
    A prediction anchored on the START of the context window measures the whole
    window's drift plus the forecast, and is wrong by a random amount that looks
    like a return.
    """
    frames = _frames()
    as_of = frames["close"].index[CONTEXT - 1]
    _adapter(frames).predict(_X(frames, as_of))

    observed = stub.calls[-1]["last_close"]
    expected = [float(frames["close"][t].iloc[CONTEXT - 1])
                for t in frames["close"].columns]
    assert observed == pytest.approx(expected)


def test_the_model_is_asked_for_exactly_the_horizon(stub):
    """`pred_len` must be the horizon; iloc[-1] is then t+horizon by definition."""
    frames = _frames()
    _adapter(frames).predict(_X(frames, frames["close"].index[CONTEXT - 1]))
    assert stub.calls[-1]["pred_len"] == HORIZON


# ── The zero-shot guarantee: the ONLY thing protecting this comparator ───────

def test_nothing_after_the_as_of_date_can_reach_the_model(stub):
    """
    Corrupt every value after the as-of date in all five frames; the prediction
    must be bit-identical.

    `fit` is a no-op, so purging, the embargo and the training boundary are all
    inert here. An off-by-one in the slice hands the model its own answer and
    reads as a breakthrough — F1 with a 102M-parameter model in place of the
    meta-learner, and no in-sample fit to re-run as evidence.
    """
    frames = _frames()
    as_of = frames["close"].index[CONTEXT - 1]
    clean = _adapter(frames).predict(_X(frames, as_of))

    poisoned = {}
    for col, frame in frames.items():
        f = frame.copy()
        f.iloc[CONTEXT:] = 1e6
        poisoned[col] = f

    after = _adapter(poisoned).predict(_X(poisoned, as_of))

    assert np.array_equal(clean, after), (
        "values after the as-of date changed the forecast — the history slice "
        "is leaking the future")
    assert stub.calls[-1]["x_last"][0] == pd.Timestamp(as_of), (
        "the last observation handed to the model must BE the as-of date")


def test_the_context_window_ends_at_the_as_of_date_and_is_full_length(stub):
    frames = _frames()
    as_of = frames["close"].index[CONTEXT + 2]
    _adapter(frames).predict(_X(frames, as_of))

    call = stub.calls[-1]
    assert call["history_len"] == [CONTEXT] * 3
    assert call["x_last"][0] == pd.Timestamp(as_of)


# ── The synthetic relative candle ────────────────────────────────────────────

def test_the_relative_candle_reproduces_the_excess_return_identity():
    """
    log(rel_close[t+h]) - log(rel_close[t]) IS the excess return, exactly.

    That identity is what lets a price model score `target_excess_return`
    directly instead of differencing two independent forecast errors. It holds
    only if the denominator is the SAME DAY'S benchmark close — a previous-day
    denominator breaks it by a small amount nothing downstream would catch.
    """
    rng = np.random.default_rng(7)
    n, h = 200, 30
    dates = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    bench = 18000 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))

    ohlcv = pd.DataFrame({"date": dates, "ticker": "AAA.NS",
                          "open": close * 0.998, "high": close * 1.01,
                          "low": close * 0.99, "close": close,
                          "volume": 1e5})
    rel = kf.relative_candles(ohlcv, pd.Series(bench, index=ohlcv.index))

    identity = (np.log(rel["close"].to_numpy()[h:])
                - np.log(rel["close"].to_numpy()[:-h]))
    excess = ((np.log(close[h:]) - np.log(close[:-h]))
              - (np.log(bench[h:]) - np.log(bench[:-h])))

    assert np.abs(identity - excess).max() < 1e-12


def test_dividing_by_one_positive_scalar_keeps_the_bar_well_formed():
    """
    high >= max(open, close) >= min(open, close) >= low must survive the
    transform, or Kronos is handed candles that cannot occur in its corpus.
    """
    rng = np.random.default_rng(3)
    n = 50
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    ohlcv = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=n), "ticker": "AAA.NS",
        "open": open_, "close": close,
        "high": np.maximum(open_, close) * 1.004,
        "low": np.minimum(open_, close) * 0.996, "volume": 1e5})
    bench = pd.Series(20000 * np.exp(np.cumsum(rng.normal(0, 0.007, n))),
                      index=ohlcv.index)

    rel = kf.relative_candles(ohlcv, bench)

    assert (rel["high"] >= rel[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (rel["low"] <= rel[["open", "close"]].min(axis=1) + 1e-12).all()
    assert (rel[["open", "high", "low", "close"]] > 0).all().all()


def test_a_non_positive_benchmark_level_is_refused_not_divided_by():
    """A zero or negative index level is not a price; dividing produces a
    'candle' that is finite, wrong, and invisible downstream."""
    ohlcv = pd.DataFrame({"date": pd.bdate_range("2022-01-03", periods=3),
                          "ticker": "AAA.NS", "open": [10.0, 11, 12],
                          "high": [11.0, 12, 13], "low": [9.0, 10, 11],
                          "close": [10.5, 11.5, 12.5], "volume": [1e5] * 3})
    rel = kf.relative_candles(ohlcv, pd.Series([100.0, 0.0, -5.0],
                                               index=ohlcv.index))

    assert np.isfinite(rel["close"].iloc[0])
    assert rel["close"].iloc[1:].isna().all(), (
        "a non-positive benchmark level must produce NaN, not a number")


def test_columns_are_masked_jointly_so_a_candle_is_never_built_from_two_days(stub):
    """
    `series._history_ending_at` drops non-finite values PER TICKER, which is
    right for one series and wrong for five aligned ones: a hole in `high` that
    `low` does not share would shift one column against the other and build a
    bar from different sessions, with nothing downstream able to detect it.

    A ticker with any hole in its window is therefore skipped entirely, and
    counted — an abstention lands as 0.0, which is exactly the `zero` floor's
    prediction, so an uncounted one is indistinguishable from a genuine null.
    """
    frames = _frames()
    frames["high"] = frames["high"].copy()
    frames["high"].iloc[CONTEXT // 2, 0] = np.nan          # AAA.NS only

    adapter = _adapter(frames)
    out = adapter.predict(_X(frames, frames["close"].index[CONTEXT - 1]))

    assert adapter.forecaster.skipped_short == 1
    assert stub.calls[-1]["n"] == 2, "the holed ticker must not be sent"
    assert stub.calls[-1]["history_len"] == [CONTEXT, CONTEXT], (
        "the survivors' context must not be truncated to the holed ticker's")
    assert out[0] == 0.0, "an abstention predicts no excess return"
    assert out[1] != 0.0 and out[2] != 0.0


# ── Configuration that must fail loudly rather than quietly ──────────────────

def test_finiteness_is_checked_separately_from_positivity(stub):
    """
    Two guards covering the same failure are one untestable guard.

    The price columns are checked for BOTH finiteness and positivity, and the
    two overlap on NaN (`nan > 0` is False), so a NaN test alone leaves the
    finiteness mask free to be deleted with the suite still green — which is
    exactly what a mutation of `&=` to `|=` proved.

    Two cases separate them. `+inf` in a price column PASSES `> 0` and is
    caught only by finiteness. `volume` carries no positivity check at all —
    a halted session legitimately trades zero — so a NaN there is caught only
    by finiteness either.
    """
    for col, bad in (("high", np.inf), ("volume", np.nan)):
        frames = _frames()
        frames[col] = frames[col].copy()
        frames[col].iloc[CONTEXT // 2, 0] = bad

        adapter = _adapter(frames)
        out = adapter.predict(_X(frames, frames["close"].index[CONTEXT - 1]))

        assert adapter.forecaster.skipped_short == 1, (
            f"{bad} in {col!r} must disqualify that ticker's window")
        assert out[0] == 0.0 and out[1] != 0.0


def test_a_zero_price_is_disqualified_although_it_is_finite(stub):
    """
    The case only the positivity mask catches.

    A zero or negative price is finite, so the finiteness mask lets it through,
    and it is real: vendors emit zero closes on bad rows. Handed to the model it
    is standardised into an ordinary-looking value; landed on the anchor it makes
    log(0). Without this test the positivity mask could be deleted with the suite
    still green — two guards covering the same failure are one untestable guard.
    """
    frames = _frames()
    frames["low"] = frames["low"].copy()
    frames["low"].iloc[CONTEXT // 3, 1] = 0.0          # BBB.NS, finite, not a price

    adapter = _adapter(frames)
    out = adapter.predict(_X(frames, frames["close"].index[CONTEXT - 1]))

    assert adapter.forecaster.skipped_short == 1
    assert out[1] == 0.0 and out[0] != 0.0


def test_zero_volume_does_not_disqualify_a_session(stub):
    """
    The counterweight. A halted session trades zero and that is a real
    observation — refusing it would silently drop tickers for being illiquid,
    which is a different experiment from the one the table claims.
    """
    frames = _frames()
    frames["volume"] = frames["volume"].copy()
    frames["volume"].iloc[CONTEXT // 2, 0] = 0.0

    adapter = _adapter(frames)
    out = adapter.predict(_X(frames, frames["close"].index[CONTEXT - 1]))

    assert adapter.forecaster.skipped_short == 0
    assert out[0] != 0.0


def test_a_context_above_the_checkpoints_cap_is_refused():
    """
    Kronos-base caps at 512. Asking for 2048 would be silently truncated by the
    predictor's own buffer while the results table claimed `@2048`.
    """
    with pytest.raises(ValueError, match="caps at context 512"):
        kf.load_predictor("NeoQuasar/Kronos-base", context=2048)


def test_an_unknown_checkpoint_is_refused_rather_than_paired_by_guess():
    """A mismatched tokenizer decodes to plausible numbers, not to an error."""
    with pytest.raises(ValueError, match="unknown Kronos checkpoint"):
        kf.load_predictor("NeoQuasar/Kronos-large", context=512)


def test_every_checkpoint_declares_its_own_tokenizer_and_cap():
    assert kf.TOKENIZERS["NeoQuasar/Kronos-mini"] == \
        ("NeoQuasar/Kronos-Tokenizer-2k", 2048)
    assert kf.TOKENIZERS["NeoQuasar/Kronos-small"][1] == 512
    assert kf.TOKENIZERS["NeoQuasar/Kronos-base"] == \
        ("NeoQuasar/Kronos-Tokenizer-base", 512)


def test_the_comparator_name_is_derived_from_the_checkpoint_and_context():
    """
    A hardcoded name records a 102M run under the 4.1M name and makes the two
    indistinguishable in `experiment_runs` afterwards.
    """
    assert kf.KronosForecaster(model_id="NeoQuasar/Kronos-base",
                               context=512).name == "kronos_base@512"
    assert kf.KronosForecaster(model_id="NeoQuasar/Kronos-mini",
                               context=2048).name == "kronos_mini@2048"


def test_misaligned_frames_are_refused():
    frames = _frames()
    frames["high"] = frames["high"].iloc[1:]
    with pytest.raises(ValueError, match="not aligned"):
        _adapter(frames)


def test_a_missing_frame_is_refused():
    frames = _frames()
    del frames["volume"]
    with pytest.raises(ValueError, match="needs frames for"):
        _adapter(frames)


# ── Sampling ─────────────────────────────────────────────────────────────────

def test_the_seed_is_per_date_so_a_subset_reproduces_the_whole(stub):
    """
    Seeding once per RUN would make a date's forecast depend on how many dates
    were scored before it: re-scoring one date would not reproduce its value
    from the full run, and a sweep over seeds could not be compared date by
    date. The seed is `self.seed + dates_scored`, so date k always gets seed
    `seed + k` regardless of what else ran.
    """
    frames = _frames()
    forecaster = kf.KronosForecaster(context=CONTEXT, seed=41)
    adapter = kf.KronosAdapter(forecaster=forecaster, frames=frames,
                               horizon=HORIZON, context=CONTEXT)

    two_dates = pd.concat([_X(frames, frames["close"].index[CONTEXT - 1]),
                           _X(frames, frames["close"].index[CONTEXT])])
    adapter.predict(two_dates)

    assert forecaster.dates_scored == 2, "one model call per date"
    assert stub.seeds == [41, 42], (
        f"date k must be seeded with seed+k regardless of what else ran; "
        f"got {stub.seeds}")


def test_sampling_is_chunked_so_memory_does_not_scale_with_sample_count(stub):
    """
    The library samples by REPEATING the input batch, so memory scales with
    `sample_count` — 2.68 GiB at one sample for 90 series at context 512 caps a
    6 GB card around two. Chunking turns a wall into a time budget.
    """
    frames = _frames()
    adapter = _adapter(frames, sample_count=7, max_batch_samples=2)
    adapter.predict(_X(frames, frames["close"].index[CONTEXT - 1]))

    sizes = [c["sample_count"] for c in stub.calls]
    assert sizes == [2, 2, 2, 1], f"chunks were {sizes}"
    assert sum(sizes) == 7, "every requested sample must actually be drawn"
    assert max(sizes) <= 2, "a chunk above the cap defeats the point"


def test_a_short_final_chunk_is_not_overweighted(stub):
    """
    `sample_count=3` at a cap of 2 gives chunks of 2 and 1. A plain mean over
    CHUNKS weights the single draw as heavily as the pair — a 3:1 error in the
    wrong direction that produces a perfectly ordinary-looking number.
    """
    assert kf.KronosForecaster(sample_count=3, max_batch_samples=2)._sample_chunks()         == [2, 1]
    assert kf.KronosForecaster(sample_count=4, max_batch_samples=2)._sample_chunks()         == [2, 2]
    assert kf.KronosForecaster(sample_count=1, max_batch_samples=8)._sample_chunks()         == [1]


def test_paths_are_averaged_in_LOG_space_not_price_space(monkeypatch):
    """
    The mean of log-prices is not the log of the mean.

    `predict_batch` averages decoded PRICES internally, so doing the same across
    chunks would apply a Jensen bias that GROWS with dispersion — precisely the
    quantity sampling is meant to reduce. Measured dispersion here is large
    (predicted SD 0.19 at one sample against a target SD of 0.10), so the bias
    is not academic.

    This stub returns a different terminal price on each chunk, so the two
    conventions give different answers and the assertion can tell them apart.
    """
    class _Varying:
        def __init__(self):
            self.n = 0

        def predict_batch(self, df_list, x_ts, y_ts, pred_len, T=1.0, top_p=0.9,
                          sample_count=1, verbose=True):
            self.n += 1
            step = 0.4 * self.n                     # 0.4 then 0.8
            out = []
            for df, y in zip(df_list, y_ts):
                terminal = float(df["close"].iloc[-1]) * np.exp(step)
                close = np.full(pred_len, terminal * 100.0)
                close[-1] = terminal
                out.append(pd.DataFrame(
                    {"open": close, "high": close, "low": close, "close": close,
                     "volume": close, "amount": close},
                    index=pd.Index(list(y)[:pred_len])))
            return out

    varying = _Varying()
    monkeypatch.setattr(kf, "load_predictor", lambda *a, **k: varying)
    monkeypatch.setitem(__import__("sys").modules, "torch",
                        type("torch", (), {"manual_seed": staticmethod(lambda s: None)}))

    frames = _frames()
    out = _adapter(frames, sample_count=2, max_batch_samples=1).predict(
        _X(frames, frames["close"].index[CONTEXT - 1]))

    log_space = (0.4 + 0.8) / 2                                   # 0.6
    price_space = float(np.log((np.exp(0.4) + np.exp(0.8)) / 2))   # 0.6199...

    assert out[0] == pytest.approx(log_space, abs=1e-12)
    assert abs(price_space - log_space) > 0.01, "the two must be separable"

    # UNEVEN chunks, end to end. Equal chunks cannot tell a weighted average
    # from an unweighted one, so the test above passes either way.
    varying.n = 0
    out = _adapter(frames, sample_count=3, max_batch_samples=2).predict(
        _X(frames, frames["close"].index[CONTEXT - 1]))

    weighted = (0.4 * 2 + 0.8 * 1) / 3          # 0.5333...
    unweighted = (0.4 + 0.8) / 2                # 0.6, the chunk-mean error
    assert out[0] == pytest.approx(weighted, abs=1e-12), (
        f"got {out[0]:.6f}; a plain mean over chunks would give {unweighted}")


# ── The dependency boundary ──────────────────────────────────────────────────

def test_importing_this_module_does_not_import_torch():
    """
    torch is not in requirements.txt — it was removed in Phase 0 as the largest
    contributor to memory pressure on an instance that had already been
    OOM-killed. `pipeline.baselines` is imported by the weekly job AND the API,
    so nothing it can reach may pull torch in at import time.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import pipeline.baselines, pipeline.kronos_forecaster; "
         "assert 'torch' not in sys.modules, sorted(k for k in sys.modules "
         "if k.startswith('torch')); print('clean')"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


# ── The scoring grid: fewer dates must not mean different folds ──────────────

def test_score_dates_narrows_what_is_measured_not_what_is_trained_on():
    """
    `score_dates` exists so a model costing 643 s per cross-section can sit in
    the same table as a ridge costing a second for its whole run. That is only
    legitimate if it changes the EVALUATION SAMPLE and nothing else: the folds,
    the purge and the embargo must be identical, or the expensive row and the
    cheap rows stop being comparable and the difference reads as a result.
    """
    from pipeline.evaluation import (PurgedPanelWalkForward, oos_dates,
                                     panel_walk_forward)

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2018-01-01", periods=700).strftime("%Y-%m-%d")
    tickers = [f"T{i}.NS" for i in range(8)]
    panel = pd.DataFrame([
        {"date": d, "ticker": t, "f1": rng.normal(),
         "target_excess_return": rng.normal(0, 0.07)}
        for d in dates for t in tickers])

    splitter = PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=400)

    class _Constant:
        seen_train: list[int] = []

        def fit(self, X, y):
            _Constant.seen_train.append(len(X))
            return self

        def predict(self, X):
            return np.asarray(X["f1"], dtype=float)

    full = panel_walk_forward(panel, ["f1", "date", "ticker"],
                             _Constant, splitter, name="c")
    train_sizes_full = list(_Constant.seen_train)
    _Constant.seen_train = []

    grid = oos_dates(panel, splitter)[::30]
    restricted = panel_walk_forward(panel, ["f1", "date", "ticker"],
                                    _Constant, splitter, name="c",
                                    rebalance_every=1, score_dates=set(grid))
    train_sizes_restricted = list(_Constant.seen_train)

    assert train_sizes_full == train_sizes_restricted, (
        "the training sets moved — this is no longer the same experiment")
    assert full.n_folds_run == restricted.n_folds_run

    assert set(restricted.predictions["date"]) == set(grid)
    assert len(restricted.predictions) < len(full.predictions)

    # And the rebalance count survives: the restricted run scores every one of
    # its dates, which are exactly the dates the full run would have sampled.
    assert (restricted.cross_sectional["n_rebalances"]
            == full.cross_sectional["n_rebalances"])


def test_sub_sampling_an_already_sub_sampled_grid_is_refused():
    """
    `score_dates` plus the default `rebalance_every=30` would take every 30th
    of the 64 rebalance dates — two rebalances, a t-statistic on n=2, and a
    table that renders.
    """
    from pipeline.evaluation import PurgedPanelWalkForward, panel_walk_forward

    panel = pd.DataFrame({"date": ["2020-01-01"], "ticker": ["A.NS"],
                          "target_excess_return": [0.01]})
    with pytest.raises(ValueError, match="rebalance_every=1"):
        panel_walk_forward(panel, ["date", "ticker"], lambda: None,
                           PurgedPanelWalkForward(), score_dates={"2020-01-01"})


def test_the_grid_is_built_from_LABELLED_rows_only():
    """
    The last `horizon` sessions carry no forward label yet, and
    `panel_walk_forward` drops them from both sides. If `oos_dates` counted
    them the grid would be longer, so `[::30]` would land on DIFFERENT dates
    than the full run's rebalances — and the expensive comparator would be
    scored on a grid the cheap ones never used, while the table still claimed
    one set of folds.
    """
    from pipeline.evaluation import PurgedPanelWalkForward, oos_dates

    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2018-01-01", periods=700).strftime("%Y-%m-%d")
    tickers = ["A.NS", "B.NS", "C.NS", "D.NS"]
    panel = pd.DataFrame([
        {"date": d, "ticker": t, "target_excess_return": rng.normal(0, 0.07)}
        for d in dates for t in tickers])

    # The real shape: the newest 30 sessions cannot have a 30-session label.
    unlabelled = set(dates[-30:])
    panel.loc[panel["date"].isin(unlabelled), "target_excess_return"] = np.nan

    splitter = PurgedPanelWalkForward(n_folds=3, horizon=30, min_train=400)
    grid = oos_dates(panel, splitter)

    assert not (set(grid) & unlabelled), (
        "dates with no label reached the scoring grid; the rebalance dates "
        "would then differ between comparators")


# ── The Kaggle split: the round trip must not change the number ──────────────

def test_the_kaggle_round_trip_reproduces_the_local_prediction(stub, tmp_path):
    """
    The one thing the export/score split has to guarantee.

    Moving the compute to a notebook is not a compute problem, it is a
    COMPARABILITY problem: a second path that differences against a different
    anchor, reads a different row, or averages in price space produces a
    complete and plausible number that cannot go in the same table. So the
    notebook returns raw terminal prices and `score_kronos` does the arithmetic
    — and this asserts the two paths agree to the last bit on identical inputs.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_kronos", pathlib.Path("tools/score_kronos.py"))
    score_kronos = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_kronos)

    frames = _frames()
    as_of = frames["close"].index[CONTEXT - 1]
    tickers = list(frames["close"].columns)

    # (a) THE LOCAL PATH.
    local = _adapter(frames, sample_count=2, max_batch_samples=1).predict(
        _X(frames, as_of))

    # (b) THE NOTEBOOK PATH, replayed. The stub's terminal close for input
    # position i is anchor * exp((i+1)/100) — the same value the notebook would
    # have written into `terminal`, identical across both samples because the
    # stub is deterministic.
    end = CONTEXT - 1
    anchors = np.array([float(frames["close"][t].iloc[end]) for t in tickers])
    terminal = np.stack([anchors * np.exp((np.arange(len(tickers)) + 1) / 100.0)] * 2,
                        axis=1)

    # float64, matching the exporter — see its comment on the anchor.
    candles = np.stack([frames[c].to_numpy(dtype=np.float64)
                        for c in kf.INPUT_COLS], axis=-1)
    package = {
        "candles": candles,
        "tickers": np.array(tickers, dtype=object),
        "row_date": np.array([str(as_of)] * len(tickers), dtype=object),
        "row_ticker": np.arange(len(tickers), dtype=np.int16),
        "row_end": np.full(len(tickers), end, dtype=np.int32),
        "row_fold": np.zeros(len(tickers), dtype=np.int8),
        "row_target": np.zeros(len(tickers), dtype=np.float32),
        "meta": np.array([json.dumps({"input_cols": kf.INPUT_COLS})],
                         dtype=object),
    }
    np.savez(tmp_path / "pred.npz", terminal=terminal,
             row_index=np.arange(len(tickers), dtype=np.int32),
             run=np.array([json.dumps({"model": "NeoQuasar/Kronos-base",
                                       "context": CONTEXT, "samples": 2,
                                       "seed": 0, "seconds": 1.0})],
                          dtype=object))

    frame, _ = score_kronos.load_run(str(tmp_path / "pred.npz"), package)

    assert list(frame["ticker"]) == tickers, "the row mapping is positional"
    np.testing.assert_allclose(frame["y_pred"].to_numpy(), local, rtol=0, atol=1e-12)


def test_the_scorer_averages_paths_in_LOG_space(tmp_path):
    """
    The round-trip test above cannot catch this: its stub is deterministic, so
    every sampled path is identical and the log-space mean equals the
    price-space one. Real samples differ — that is the entire point of drawing
    more than one — and `predict_batch` averages decoded PRICES inside a chunk,
    so repeating that across chunks applies a Jensen bias that GROWS with
    dispersion. Measured dispersion here is large, so it is not academic.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_kronos", pathlib.Path("tools/score_kronos.py"))
    score_kronos = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_kronos)

    anchor = 0.5
    paths = np.array([[anchor * np.exp(0.4), anchor * np.exp(0.8)]])

    candles = np.zeros((1, 1, len(kf.INPUT_COLS)), dtype=np.float64)
    candles[0, 0, kf.INPUT_COLS.index("close")] = anchor
    package = {
        "candles": candles,
        "tickers": np.array(["A.NS"], dtype=object),
        "row_date": np.array(["2024-01-02"], dtype=object),
        "row_ticker": np.zeros(1, dtype=np.int16),
        "row_end": np.zeros(1, dtype=np.int32),
        "row_fold": np.zeros(1, dtype=np.int8),
        "row_target": np.zeros(1, dtype=np.float32),
        "meta": np.array([json.dumps({"input_cols": kf.INPUT_COLS})],
                         dtype=object),
    }
    np.savez(tmp_path / "p.npz", terminal=paths,
             row_index=np.zeros(1, dtype=np.int32),
             run=np.array([json.dumps({"model": "m", "context": 512,
                                       "samples": 2, "seed": 0,
                                       "seconds": 1.0})], dtype=object))

    frame, _ = score_kronos.load_run(str(tmp_path / "p.npz"), package)

    log_space = (0.4 + 0.8) / 2                                    # 0.6
    price_space = float(np.log((np.exp(0.4) + np.exp(0.8)) / 2))    # 0.6199...
    assert abs(price_space - log_space) > 0.01, "the two must be separable"
    assert frame["y_pred"].iloc[0] == pytest.approx(log_space, abs=1e-12)


def test_the_scorer_reads_the_close_column_by_NAME_from_the_package():
    """
    `candles` is a dense (dates, tickers, 5) array, so the close is at an
    INDEX. The package carries `input_cols` and the scorer looks the position
    up there rather than hardcoding 3 — a column reorder in INPUT_COLS would
    otherwise silently anchor every prediction on `low`.
    """
    source = pathlib.Path("tools/score_kronos.py").read_text(encoding="utf-8")
    assert 'input_cols.index("close")' in source
    assert kf.INPUT_COLS.index("close") == 3, (
        "if this moved, the guard above is what keeps the scorer correct")


def test_a_row_the_notebook_never_reached_is_excluded_not_scored_as_zero(tmp_path):
    """
    A truncated run must not look like a weak result.

    The notebook initialises `terminal` to NaN and fills what it reaches, so an
    all-NaN row is one it never got to — a wall-clock timeout, a --limit-dates
    smoke test. Filling those with 0.0 puts the `zero` floor's own prediction
    into the sample: a real run that covered 1 of 63 rebalances reported MAE
    0.06705 against a floor of 0.06532, a plausible near-floor null that was 98%
    the zero prediction. Scored over the rows it actually reached, the same run
    was +209.9% worse than the floor.

    This is the ChronosProbe landmine in a new place, and the fix is the same:
    a cached or partial artifact must cover what is being scored, or say so.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score_kronos", pathlib.Path("tools/score_kronos.py"))
    score_kronos = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_kronos)

    anchor = 0.5
    # Row 0 was reached; rows 1 and 2 were not. Row 3 was reached but decoded
    # to a non-positive price — an ABSTENTION, which is part of the sample.
    terminal = np.array([[anchor * np.exp(0.05)],
                         [np.nan],
                         [np.nan],
                         [-1.0]])

    candles = np.zeros((1, 4, len(kf.INPUT_COLS)), dtype=np.float64)
    candles[0, :, kf.INPUT_COLS.index("close")] = anchor
    package = {
        "candles": candles,
        "tickers": np.array(["A.NS", "B.NS", "C.NS", "D.NS"], dtype=object),
        "row_date": np.array(["2024-01-02"] * 4, dtype=object),
        "row_ticker": np.arange(4, dtype=np.int16),
        "row_end": np.zeros(4, dtype=np.int32),
        "row_fold": np.zeros(4, dtype=np.int8),
        "row_target": np.zeros(4, dtype=np.float32),
        "meta": np.array([json.dumps({"input_cols": kf.INPUT_COLS})],
                         dtype=object),
    }
    np.savez(tmp_path / "p.npz", terminal=terminal,
             row_index=np.arange(4, dtype=np.int32),
             run=np.array([json.dumps({"model": "m", "context": 512,
                                       "samples": 1, "seed": 0,
                                       "seconds": 1.0})], dtype=object))

    frame, run = score_kronos.load_run(str(tmp_path / "p.npz"), package)

    assert run["not_scored"] == 2, "rows never attempted must be dropped"
    assert run["rows_scored"] == 2
    assert list(frame["ticker"]) == ["A.NS", "D.NS"]

    # The reached row keeps its forecast; the abstention predicts nothing.
    assert frame["y_pred"].iloc[0] == pytest.approx(0.05, abs=1e-12)
    assert frame["y_pred"].iloc[1] == 0.0
    assert run["abstentions"] == 1, (
        "a decode to a non-positive price IS part of the sample and abstains; "
        "it must not be confused with a row the run never reached")
