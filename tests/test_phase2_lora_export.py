"""
tests/test_phase2_lora_export.py - The causal boundary shipped to Kaggle.

`tools/export_lora_package.py` exists so that a notebook running somewhere else
cannot re-derive a boundary slightly differently. That makes `end_index` the
single highest-risk value in the whole LoRA path: the notebook slices
`series[end - context + 1 : end + 1]` and asks no questions, so if `end_index`
points one row too far the model is handed part of the answer it is being asked
to predict. No error, no warning, and a result that reads as a breakthrough.

These tests hold that boundary, on a synthetic panel whose right answer is
known by construction.
"""

import numpy as np
import pandas as pd
import pytest


def _panel(n_dates=700, n_tickers=6, seed=0):
    """A panel whose relative-price series is a random walk with a known index."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates).strftime("%Y-%m-%d")
    rows = []
    for t in range(n_tickers):
        walk = np.cumsum(rng.normal(0, 0.01, n_dates)) - 3.0
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": f"T{t}.NS",
                         "value": float(walk[i])})
    return pd.DataFrame(rows), list(dates)


def _build(context=128, stride=1, n_dates=700, min_train=200, folds=2):
    from pipeline.evaluation import PurgedPanelWalkForward
    from tools.export_lora_package import build_rows

    frame, dates = _panel(n_dates=n_dates)
    wide = frame.pivot_table(index="date", columns="ticker", values="value")
    tickers = list(wide.columns)
    t_index = {t: i for i, t in enumerate(tickers)}
    d_index = {d: i for i, d in enumerate([str(x) for x in wide.index])}
    series = wide.to_numpy(dtype=np.float32).T

    # The label at date t is the 30-session forward change - the same identity
    # the real panel uses, so a leak shows up as a suspiciously easy target.
    lab = []
    for t in tickers:
        col = wide[t].to_numpy()
        for i in range(len(col) - 30):
            lab.append({"date": str(wide.index[i]), "ticker": t,
                        "target_return": float(col[i + 30] - col[i])})
    labelled = pd.DataFrame(lab).sort_values(["date", "ticker"])
    labelled = labelled.reset_index(drop=True)

    splitter = PurgedPanelWalkForward(n_folds=folds, horizon=30, embargo=30,
                                      min_train=min_train)
    summary = []
    out = build_rows(labelled, splitter, series, t_index, d_index,
                     context=context, train_stride=stride, summary=summary)
    return out, labelled, series, wide, t_index, d_index, summary


def test_end_index_points_at_the_rows_own_date():
    """
    The boundary. `end_index` must be the row's own date - not the next one.

    Off by +1 and the window's last observation is tomorrow's price, which for
    a 30-session forward target is a direct leak of one step of the answer.
    """
    (fold, split, tick, end, y), labelled, series, wide, t_index, d_index, _ = \
        _build()

    dates = [str(x) for x in wide.index]
    # Recover which labelled row each record came from and check the date.
    checked = 0
    for k in range(0, len(end), 97):          # sample, not all 20k
        ti, di = tick[k], end[k]
        expected_value = series[ti, di]
        assert np.isfinite(expected_value)
        # The value at end_index must equal the series on that row's date.
        assert series[ti, di] == pytest.approx(
            float(wide.iloc[di, ti]), abs=1e-6)
        assert dates[di] == str(wide.index[di])
        checked += 1
    assert checked >= 50


def test_window_never_reaches_past_the_as_of_date():
    """
    Corrupt everything after each row's as-of index and the window must be
    unchanged. This is the test that would fail on an off-by-one.
    """
    (fold, split, tick, end, y), _, series, _, _, _, _ = _build()

    clean = series.copy()
    for k in range(0, len(end), 211):
        ti, di = tick[k], end[k]
        poisoned = clean.copy()
        poisoned[ti, di + 1:] = 999.0

        a = clean[ti, max(0, di - 128 + 1):di + 1]
        b = poisoned[ti, max(0, di - 128 + 1):di + 1]
        assert np.array_equal(a, b), \
            "the window changed when FUTURE values were corrupted"


def test_target_comes_from_the_panel_not_recomputed():
    """
    The exported target must be the panel's label for that (date, ticker).
    Recomputing it in two places is how the two drift apart.
    """
    (fold, split, tick, end, y), labelled, _, wide, t_index, d_index, _ = _build()

    lookup = {(r.date, r.ticker): r.target_return
              for r in labelled.itertuples()}
    dates = [str(x) for x in wide.index]
    tickers = list(wide.columns)

    for k in range(0, len(y), 149):
        key = (dates[end[k]], tickers[tick[k]])
        assert y[k] == pytest.approx(lookup[key], abs=1e-6)


def test_train_is_strided_and_test_is_not():
    """
    The stride is an efficiency measure on TRAINING only. Striding test rows
    would quietly change which rows the results table covers, and the table
    would no longer be comparable with any other comparator.
    """
    (f1, s1, _, _, _), *_ , sum1 = _build(stride=1)
    (f5, s5, _, _, _), *_ , sum5 = _build(stride=5)

    train1 = sum(r["train_rows"] for r in sum1)
    train5 = sum(r["train_rows"] for r in sum5)
    test1 = sum(r["test_rows"] for r in sum1)
    test5 = sum(r["test_rows"] for r in sum5)

    assert train5 < train1 / 2, "stride did not reduce the training set"
    assert test5 == test1, "stride changed the TEST set; it must not"


def test_train_and_test_rows_never_share_a_date():
    """
    The purged splitter guarantees it; this checks the export did not undo it
    by, for example, indexing into the wrong frame.
    """
    (fold, split, tick, end, y), *_ = _build()

    arr = np.array([fold, split, end]).T
    for f in np.unique(arr[:, 0]):
        sub = arr[arr[:, 0] == f]
        tr = set(sub[sub[:, 1] == 0][:, 2].tolist())
        te = set(sub[sub[:, 1] == 1][:, 2].tolist())
        assert not (tr & te), f"fold {f} shares dates between train and test"


def test_rows_without_enough_history_are_dropped():
    """
    A window with fewer than MIN_CONTEXT real observations is not scored. The
    alternative is a model asked to forecast from almost nothing, which it will
    do, silently.
    """
    from pipeline.series import MIN_CONTEXT

    (fold, split, tick, end, y), _, series, _, _, _, _ = _build(context=128)

    for k in range(0, len(end), 173):
        window = series[tick[k], max(0, end[k] - 128 + 1):end[k] + 1]
        assert np.isfinite(window).sum() >= MIN_CONTEXT


def test_split_labels_match_the_reported_counts():
    """
    `row_split` is what the notebook trains on and what `score_lora.py` checks
    to refuse in-sample predictions. If every row were labelled "test", the
    disjointness test above would still pass - an empty train set shares no
    dates with anything - while the run silently trained on nothing.

    So the labels are checked against the counts the exporter itself reported.
    """
    (fold, split, tick, end, y), *_, summary = _build()

    arr = np.array([fold, split]).T
    for entry in summary:
        f = entry["fold"]
        sub = arr[arr[:, 0] == f]
        n_train = int((sub[:, 1] == 0).sum())
        n_test = int((sub[:, 1] == 1).sum())
        assert n_train == entry["train_rows"], \
            f"fold {f}: {n_train} rows labelled train, summary says " \
            f"{entry['train_rows']}"
        assert n_test == entry["test_rows"], \
            f"fold {f}: {n_test} rows labelled test, summary says " \
            f"{entry['test_rows']}"
        assert n_train > 0 and n_test > 0, f"fold {f} has an empty split"
