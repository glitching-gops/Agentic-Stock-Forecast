"""
The Kaggle round trip for Chronos-2 and TimesFM-2.5.

Everything here guards a failure that produces a COMPLETE, PLAUSIBLE TABLE
rather than an error. The package declares which rows are scorable, what each
row's anchor is, which fold it belongs to and what the two floors predicted for
it; a notebook fills in one number per row. Every one of those declarations has
a wrong version that renders perfectly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.export_series_package import (SHIPPED_MODULES,  # noqa: E402
                                         collect_sources)
from tools.series_kaggle import install_sources             # noqa: E402


# ── The shipped source is the tested source ───────────────────────────────

def test_the_notebook_runs_our_code_rather_than_a_copy_of_it():
    """
    The package carries pipeline/series.py and the two forecasters as source
    text. That is the whole reason this path is allowed to call `.forecast()`
    on Kaggle at all, where `kronos_kaggle.py` is forbidden from computing
    anything: the arithmetic that could be wrong is OURS, hash-verified, rather
    than a notebook's approximation of it.

    What it protects is specific and measured. The median quantile is at index
    10 on amazon/chronos-2, 6 on autogluon/chronos-2-small and 5 on TimesFM
    (offset by one, because column 0 is the point output). Index 6 on the 120M
    checkpoint is the 0.3 quantile - a systematic downward bias on every
    prediction, no error, no warning, a table that renders.
    """
    sources = collect_sources(str(REPO))

    assert set(sources) == set(SHIPPED_MODULES)
    for rel, entry in sources.items():
        on_disk = (REPO / rel).read_text(encoding="utf-8")
        assert entry["text"] == on_disk, f"{rel} was not shipped verbatim"
        assert entry["sha256"] == hashlib.sha256(
            on_disk.encode("utf-8")).hexdigest()

    # The forecasters must be reachable from the shipped set alone. Both import
    # pipeline.series, so shipping either without it puts an ImportError in
    # front of an hour of GPU time.
    assert "pipeline/series.py" in sources


def test_a_tampered_package_refuses_to_run(tmp_path):
    """
    A hash that is recorded but never checked is a comment. The check has to
    ABORT, because the failure it guards - a package built from a different
    revision than the table it will be compared against - produces numbers that
    look exactly as comparable as real ones.
    """
    sources = collect_sources(str(REPO))
    sources["pipeline/series.py"]["text"] += "\n# smuggled in\n"

    package = {"sources": np.array([json.dumps(sources)], dtype=object)}

    with pytest.raises(SystemExit, match="does not match its own recorded"):
        install_sources(package, str(tmp_path / "shipped"))


def test_the_shipped_modules_import_with_no_repository_present(tmp_path):
    """
    They are written to a bare directory and imported from there, so anything
    they reach for outside the shipped set fails on Kaggle and nowhere else -
    after the dataset is uploaded and the GPU quota is spent.

    Run in a SUBPROCESS with the repo off sys.path. In-process, `pipeline` is
    already imported and the test would pass against a module that could not
    possibly load on its own.
    """
    where = tmp_path / "shipped"
    install_sources({"sources": np.array([json.dumps(collect_sources(str(REPO)))],
                                         dtype=object)}, str(where))
    sys.path.remove(str(where))

    script = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from pipeline.series import _history_ending_at, configure_determinism\n"
        "from pipeline.chronos_forecaster import Chronos2Forecaster\n"
        "from pipeline.timesfm_forecaster import TimesFM25Forecaster\n"
        "print('ok')\n" % str(where)
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout

    # And torch must NOT have been dragged in: the whole point of importing
    # these lazily is that the package stays installable before the GPU is.
    check = script.replace("print('ok')",
                           "print('torch' in sys.modules)")
    proc = subprocess.run([sys.executable, "-c", check], cwd=str(tmp_path),
                          capture_output=True, text=True)
    assert proc.stdout.strip().endswith("False"), \
        "importing the shipped forecasters pulled in torch at module level"


# ── The package's declarations ────────────────────────────────────────────

def _synthetic_package(tmp_path, n_dates=140, n_tickers=4, holes=True):
    """
    A package shaped like the real one, with a known right answer.

    Longer than MIN_CONTEXT (90) on purpose: at 40 dates `_history_ending_at`
    returns {} for every ticker and the comparison loop below never runs. That
    is the vacuous-test failure this suite has already hit once, when a
    context=50 fixture was checked against MIN_CONTEXT=90.
    """
    rng = np.random.default_rng(0)
    dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_dates)]
    tickers = [f"T{i}.NS" for i in range(n_tickers)]

    series = np.cumsum(rng.normal(0, 0.01, size=(n_dates, n_tickers)), axis=0) + 5.0
    if holes:
        # A hole in one ticker, which is what makes "the anchor is the last
        # FINITE observation" different from "the value at row_end".
        series[n_dates - 1, 1] = np.nan
        series[:5, 2] = np.nan

    return dates, tickers, series


def test_the_anchor_is_the_last_finite_observation_not_the_row_at_the_index():
    """
    The scorer differences the notebook's forecast against `row_anchor`. The
    forecasters difference against `history[-1]`, which is the last FINITE
    value after `_history_ending_at` drops non-finite ones - not the value
    sitting at `row_end`.

    On a ticker whose most recent session is missing those are different
    numbers, and the difference is a whole session's return applied to that
    row alone. Nothing downstream can detect it: the prediction stays the right
    order of magnitude and only the fifth decimal of MAE moves.
    """
    from pipeline.series import _history_ending_at
    from tools.export_series_package import scorable_rows

    dates, tickers, series = _synthetic_package(None)
    frame = pd.DataFrame(series, index=dates, columns=tickers)
    positions = {d: i for i, d in enumerate(dates)}

    as_of = dates[-1]
    end = positions[as_of]
    target = {(as_of, t): 0.01 for t in tickers}

    rows = scorable_rows(series, positions, tickers, [as_of], target,
                         {as_of: 0}, min_context=1)
    assert rows, "nothing was scorable, so this test proves nothing"

    histories = _history_ending_at(frame, positions, as_of, context=len(dates))
    checked = 0
    for row in rows:
        ticker = row["ticker"]
        if ticker not in histories:
            continue
        assert row["anchor"] == float(histories[ticker][-1]), \
            f"{ticker}: the package anchor is not what the forecaster subtracts"
        checked += 1
    assert checked >= 2

    # The ticker with a missing final session is where `values[end]` and "last
    # finite" disagree - without it this test passes against either rule.
    hole = next(r for r in rows if r["ticker"] == tickers[1])
    assert not np.isfinite(series[end, 1])
    assert hole["anchor"] == float(series[end - 1, 1])

    # ELIGIBILITY COUNTS FINITE OBSERVATIONS, NOT ROWS. tickers[2] has its
    # first five sessions missing, so at a threshold set between the two counts
    # it must be REFUSED - a rule that counted rows would admit it and hand the
    # model a window five observations shorter than the table claims.
    #
    # Set exactly at the boundary: `end + 1` rows exist, five of them empty.
    threshold = end + 1 - 2
    tight = scorable_rows(series, positions, tickers, [as_of], target,
                          {as_of: 0}, min_context=threshold)
    admitted = {r["ticker"] for r in tight}
    assert tickers[0] in admitted, "a full-history ticker must still qualify"
    assert tickers[2] not in admitted, \
        "a ticker with holes was admitted by counting rows instead of observations"
    assert all(r["avail"] >= threshold for r in tight)


def test_eligibility_does_not_move_with_the_model_context():
    """
    THE CORRECTION THE KRONOS PAIR NEEDED.

    Kronos requires a full window, so base@512 scored 90 tickers and mini@2048
    scored 81 - the two rows were measured over different universes and the
    "context" comparison confounded coverage with context. A univariate
    foundation model left-pads, so eligibility here is fixed at MIN_CONTEXT
    over the history available at the as-of date and does not move with what
    the notebook asks for.

    The consequence under test: the rows a package declares are a function of
    the package, so chronos@512, chronos@2048 and timesfm@16384 are scored on
    identical rows and their difference is the context alone.
    """
    from pipeline.series import MIN_CONTEXT, _history_ending_at

    n_dates = MIN_CONTEXT + 60
    dates = [f"d{i:04d}" for i in range(n_dates)]
    tickers = ["A.NS", "B.NS"]
    series = np.ones((n_dates, 2)) * 5.0
    series[: n_dates - MIN_CONTEXT + 10, 1] = np.nan   # B is a young listing

    frame = pd.DataFrame(series, index=dates, columns=tickers)
    positions = {d: i for i, d in enumerate(dates)}
    as_of = dates[-1]

    eligible_by_context = {}
    for context in (MIN_CONTEXT, 512, 2048, 16384):
        histories = _history_ending_at(frame, positions, as_of, context)
        eligible_by_context[context] = set(histories)

    # A.NS qualifies at every context. B.NS has too little history and
    # qualifies at NONE of them - what must not happen is qualifying at some.
    assert len({frozenset(v) for v in eligible_by_context.values()}) == 1, \
        f"the eligible set moved with the context: {eligible_by_context}"
    assert "A.NS" in eligible_by_context[512]

    # The window itself DOES grow with the context - otherwise the flag would
    # be doing nothing and this test would pass against a constant.
    short = _history_ending_at(frame, positions, as_of, MIN_CONTEXT)
    long = _history_ending_at(frame, positions, as_of, 512)
    assert len(long["A.NS"]) > len(short["A.NS"])


# ── The scorer ────────────────────────────────────────────────────────────

def _package_npz(tmp_path, n_rows=120, n_dates=6):
    """A minimal package the scorer can read."""
    rng = np.random.default_rng(1)
    dates = [f"2026-0{1 + i}-01" for i in range(n_dates)]
    tickers = [f"T{i}.NS" for i in range(n_rows // n_dates)]

    row_date, row_ticker, row_fold = [], [], []
    for k, d in enumerate(dates):
        for j in range(len(tickers)):
            row_date.append(d)
            row_ticker.append(j)
            row_fold.append(k // 2)

    n = len(row_date)
    target = rng.normal(0.02, 0.09, n)

    # THE BETA FLOOR MUST ACTUALLY RANK, or every test below is vacuous. A
    # random beta_market has an IC near zero, which makes "beats beta_market on
    # reb_IC" indistinguishable from "has a positive IC" and lets a mutant that
    # drops the floor entirely go unnoticed. On the real panel beta_market
    # scores reb_IC +0.046 at t +1.51, so it is given a real edge here.
    beta = 0.5 * target + rng.normal(0, 0.05, n)

    path = tmp_path / "pkg.npz"
    np.savez_compressed(
        path,
        series=np.zeros((n_dates, len(tickers))),
        tickers=np.array(tickers, dtype=object),
        dates=np.array(dates, dtype=object),
        row_date=np.array(row_date, dtype=object),
        row_ticker=np.asarray(row_ticker, dtype=np.int16),
        row_end=np.zeros(n, dtype=np.int32),
        row_fold=np.asarray(row_fold, dtype=np.int8),
        row_target=target,
        row_anchor=np.ones(n),
        row_avail=np.full(n, 500, dtype=np.int32),
        row_market=np.full(n, 0.02),
        row_beta_market=beta,
        sources=np.array([json.dumps({})], dtype=object),
        meta=np.array([json.dumps({
            "target": "target_return", "horizon": 30, "folds": 5,
            "min_train": 500, "min_context": 90, "n_rebalances": n_dates,
            "n_oos_dates": 180, "n_tickers": len(tickers),
            "floors": ["market", "beta_market"],
            "avail_min": 500, "avail_median": 500, "avail_max": 500,
        })], dtype=object),
    )
    return path, n


def _predictions_npz(tmp_path, pred, name="chronos2"):
    path = tmp_path / f"{name}.npz"
    np.savez_compressed(
        path, pred=pred,
        used_context=np.zeros(1, dtype=np.int32),
        run=np.array([json.dumps({
            "name": name, "context": 2048, "seconds": 60.0,
        })], dtype=object))
    return path


def test_a_run_that_never_reached_a_row_is_not_a_run_that_declined_it(tmp_path):
    """
    A partial run must not read as a weak result. The notebook initialises
    every prediction to NaN and fills what it reaches, so an all-NaN row is one
    a wall-clock timeout or a --limit-dates smoke test never got to.

    Filling those with 0.0 puts the `zero` forecast's OWN prediction into the
    sample. Measured on the Kronos path: a run covering 1 of 63 rebalances
    reported MAE 0.06705 against a floor of 0.06532 - a plausible near-floor
    null that was 98% the zero prediction. Scored over the rows it actually
    reached, the same run was +209.9% worse than the floor.
    """
    from tools.score_series import load_run

    pkg_path, n = _package_npz(tmp_path)
    package = np.load(pkg_path, allow_pickle=True)

    pred = np.full(n, np.nan)
    pred[:20] = 0.05                      # one date's worth actually reached
    frame, run = load_run(str(_predictions_npz(tmp_path, pred)), package)

    assert run["rows_scored"] == 20
    assert run["not_scored"] == n - 20
    assert run["coverage"] == pytest.approx(20 / n)
    assert len(frame) == 20
    assert (frame["y_pred"] == 0.05).all(), \
        "an unreached row must be dropped, never filled with the floor's claim"


def test_predictions_from_a_different_package_are_refused(tmp_path):
    """
    The mapping from a prediction back to its (date, ticker) is POSITIONAL and
    nothing else. A predictions file built from a different export lines up
    every forecast against the wrong row and produces a complete, well-formed,
    entirely meaningless table with no error anywhere.
    """
    from tools.score_series import load_run

    pkg_path, n = _package_npz(tmp_path)
    package = np.load(pkg_path, allow_pickle=True)

    wrong = _predictions_npz(tmp_path, np.zeros(n - 5), name="mismatched")
    with pytest.raises(SystemExit, match="different packages"):
        load_run(str(wrong), package)


def test_clearing_the_floor_needs_both_halves(tmp_path):
    """
    `zero` is not the floor on an absolute-return target - 57.67% of the labels
    are positive and 32.8% of their variance is shared, so it is beaten by
    drift and beta before a model opens its eyes.

    `clears_floor` is MAE below `market` AND reb_IC above `beta_market`. Either
    alone is passable by something that has learned nothing about companies: a
    constant beats `market` on MAE by predicting the drift, and a beta sort
    scores a positive IC in a rising market.
    """
    from tools.score_series import load_run, report

    pkg_path, n = _package_npz(tmp_path)
    package = np.load(pkg_path, allow_pickle=True)
    target = package["row_target"].astype(float)
    beta = package["row_beta_market"].astype(float)

    # A perfect forecast clears both halves.
    frame, _ = load_run(str(_predictions_npz(tmp_path, target.copy(),
                                             name="oracle")), package)
    perfect = report(frame, "oracle", float(target.std()))
    assert perfect["beats_market"] and perfect["beats_beta_ic"]
    assert perfect["clears_floor"] is True

    # ORDERING WITHOUT MAGNITUDE. Ten times the target ranks PERFECTLY - rank
    # IC is scale-invariant - while its MAE is an order of magnitude worse than
    # predicting the market mean. Exactly one half passes, so this is the case
    # that separates `and` from `or`, and it is not hypothetical: TimesFM's
    # better MAE was measured to be shrinkage rather than information, and
    # Kronos over-dispersed by 2.7x with a reb_IC of -0.02.
    frame, _ = load_run(str(_predictions_npz(tmp_path, target * 10,
                                             name="loud")), package)
    loud = report(frame, "perfect ordering, hopeless magnitude", float(target.std()))
    assert loud["beats_beta_ic"] is True
    assert loud["beats_market"] is False
    assert loud["clears_floor"] is False, "one half is not the floor"

    # THE ORDERING FLOOR IS beta_market, NOT ZERO. The beta floor's own
    # predictions have a clearly POSITIVE rank IC here, as they do on the real
    # panel (+0.046 at t +1.51) - so a gate reading `reb_IC > 0` would pass
    # them. It has to read `> beta_market`, or a comparator that has learned
    # nothing beyond which names are high-beta grades as evidence.
    frame, _ = load_run(str(_predictions_npz(tmp_path, beta.copy(),
                                             name="asfloor")), package)
    asfloor = report(frame, "beta_market as a comparator", float(target.std()))
    assert asfloor["beta_market_ic"] > 0.05, \
        "the floor has no ordering in this fixture, so it cannot be failed"
    assert asfloor["beats_beta_ic"] is False
    assert asfloor["clears_floor"] is False

    # MAE IS GRADED AGAINST `market`, NOT AGAINST ZERO. The market forecast is
    # a constant at the panel's drift, so it beats zero on MAE by construction
    # on a target whose mean is positive. A comparator that merely predicts
    # that drift therefore looks like a winner against zero and is exactly
    # level with the floor it should be measured on.
    frame, _ = load_run(str(_predictions_npz(tmp_path,
                                             package["row_market"].astype(float).copy(),
                                             name="asmarket")), package)
    asmarket = report(frame, "market as a comparator", float(target.std()))
    zero_mae = float(np.abs(target).mean())
    assert asmarket["mae"] < zero_mae, \
        "the fixture's market forecast does not beat zero, so this is vacuous"
    assert asmarket["beats_market"] is False, \
        "the market forecast cannot beat itself; grading against zero would say it does"

    # And the per-fold breakdown is always produced, never only the pooled
    # number: both prior positive results in this project ran +0.0879 -> -0.0307
    # across folds while their pooled t-statistics read +3.32 and +2.37.
    assert len(perfect["per_fold"]) > 1
