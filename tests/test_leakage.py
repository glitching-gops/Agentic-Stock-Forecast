"""
Regression tests for the leakage defects found in the Phase 0 audit.

Each test corresponds to a finding and fails if that defect is reintroduced.
These are the tests whose absence let F1-F3 ship and reach a public dashboard.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation import (
    PurgedWalkForward,
    assert_no_leakage,
    compute_metrics,
    effective_sample_size,
    majority_hit_rate,
)

HORIZON = 30


# ── F3: purging and embargo ───────────────────────────────────────────────────

def test_folds_leave_a_purge_gap_of_at_least_the_horizon():
    """Training must end at least `horizon + embargo` rows before the test window."""
    splitter = PurgedWalkForward(n_folds=5, horizon=HORIZON,
                                 embargo=HORIZON, min_train=250)
    n = 1200
    folds = list(splitter.split(n))
    assert folds, "splitter produced no folds"

    for train_idx, test_idx in folds:
        gap = int(test_idx.min()) - int(train_idx.max()) - 1
        assert gap >= HORIZON + HORIZON, (
            f"gap of {gap} rows is smaller than horizon+embargo "
            f"({HORIZON + HORIZON}); a 30-session label would straddle the split"
        )


def test_train_indices_always_precede_test_indices():
    splitter = PurgedWalkForward(n_folds=6, horizon=HORIZON, min_train=300)
    for train_idx, test_idx in splitter.split(1500):
        assert train_idx.max() < test_idx.min()


def test_assert_no_leakage_catches_a_contiguous_split():
    """
    The exact defect from pipeline/model.py: an 85/15 contiguous split whose
    last training label lands inside the test window.
    """
    dates = [f"d{i:04d}" for i in range(500)]
    split = int(500 * 0.85)
    train, test = dates[:split], dates[split:]

    with pytest.raises(AssertionError, match="Label leakage"):
        assert_no_leakage(train, test, HORIZON, dates)


def test_assert_no_leakage_accepts_a_purged_split():
    dates = [f"d{i:04d}" for i in range(500)]
    train = dates[:300]
    test = dates[300 + HORIZON + HORIZON:]
    assert_no_leakage(train, test, HORIZON, dates)   # must not raise


# ── F2: nested tuning ─────────────────────────────────────────────────────────

def test_tuner_never_sees_the_rows_its_own_fold_is_scored_on():
    """
    F2 was tuning on the full labelled set and then reporting a slice of it as
    held out. The contract is per fold: the tuner receives that fold's training
    slice only, separated from that fold's test slice by the purge gap.

    Reusing an EARLIER fold's test rows as a LATER fold's training rows is
    correct rolling-origin behaviour and is not leakage — each model is still
    fitted only on data preceding its own test window.
    """
    from pipeline.evaluation import walk_forward
    from sklearn.linear_model import Ridge

    n = 900
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.normal(size=n))
    dates = [f"d{i:04d}" for i in range(n)]

    train_rows_per_call: list[set] = []

    def spying_tuner(X_train, y_train):
        train_rows_per_call.append(set(X_train.index))
        return {}

    result = walk_forward(
        X=X, y=y, dates=dates,
        model_factory=lambda: Ridge(alpha=1.0),
        splitter=PurgedWalkForward(n_folds=4, horizon=HORIZON, min_train=300),
        tuner=spying_tuner,
    )

    assert result.n_folds_run > 0
    assert len(train_rows_per_call) == result.n_folds_run, "tuner call count mismatch"

    for position, fold_id in enumerate(sorted(result.predictions["fold"].unique())):
        fold_tests = result.predictions[result.predictions["fold"] == fold_id]
        test_rows = {int(d[1:]) for d in fold_tests["date"]}
        train_rows = train_rows_per_call[position]

        overlap = train_rows & test_rows
        assert not overlap, (
            f"fold {fold_id}: tuner saw {len(overlap)} of the rows this fold is "
            f"scored on"
        )

        gap = min(test_rows) - max(train_rows) - 1
        assert gap >= HORIZON, (
            f"fold {fold_id}: only {gap} rows between the tuner's training data "
            f"and its test window; a {HORIZON}-session label spans the gap"
        )


# ── Stage 2b: the POOLED training path ────────────────────────────────────────
#
# The per-ticker tests above do not generalise to it automatically. Pooling
# introduces a failure mode they cannot see: one ticker's future rows reaching
# another ticker's training fold through the pooling itself, which no
# single-series purge check would notice because within each series the
# boundary still looks correct.


def _synthetic_panel(n_dates: int = 900, n_tickers: int = 12,
                     seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = np.repeat([f"d{i:04d}" for i in range(n_dates)], n_tickers)
    tickers = np.tile([f"T{j:02d}.NS" for j in range(n_tickers)], n_dates)
    n = dates.size
    return pd.DataFrame({
        "date": dates, "ticker": tickers,
        "a": rng.normal(size=n), "b": rng.normal(size=n),
        "target_return": rng.normal(size=n),
    })


def test_the_pooled_tuner_never_sees_any_tickers_rows_from_its_own_test_window():
    """
    The pooled counterpart of the F2 contract.

    A per-ticker purge that holds for every series individually can still be
    violated in a pooled fit if the boundary is computed per series: tickers
    join the panel on different dates, so a row-position boundary lands on a
    different CALENDAR date for each of them, and a training row for ticker A
    can then postdate the test window that ticker B is scored on. The fix is
    that ``PurgedPanelWalkForward`` splits the shared DATE grid, and this
    asserts the consequence directly — across the whole cross-section, not
    ticker by ticker.
    """
    from pipeline.evaluation import PurgedPanelWalkForward

    panel = _synthetic_panel()
    dates = panel["date"].to_numpy()
    splitter = PurgedPanelWalkForward(n_folds=4, horizon=HORIZON,
                                      embargo=HORIZON, min_train=300)

    folds = list(splitter.split(dates))
    assert folds, "the splitter yielded nothing; the test would be vacuous"

    for fold, (train_idx, test_idx) in enumerate(folds):
        train_dates = set(dates[train_idx])
        test_dates = set(dates[test_idx])

        assert not (train_dates & test_dates), (
            f"fold {fold}: {len(train_dates & test_dates)} dates appear in both "
            f"the pooled training and test sets")

        # The purge, measured on the shared grid rather than per series.
        grid = np.unique(dates)
        last_train = np.searchsorted(grid, max(train_dates))
        first_test = np.searchsorted(grid, min(test_dates))
        gap = first_test - last_train - 1
        assert gap >= HORIZON, (
            f"fold {fold}: only {gap} grid dates between the pooled training "
            f"data and its test window; a {HORIZON}-session label spans it")

        # And the pooled-specific one: no ticker's training row may postdate
        # ANY ticker's test row.
        assert max(train_dates) < min(test_dates), (
            f"fold {fold}: a training row dated {max(train_dates)} sits at or "
            f"after the test window opening at {min(test_dates)} — pooled "
            f"across tickers, that is one name's future in another's past")


def test_the_pooled_search_is_handed_only_its_own_training_slice():
    """
    ``tune_pooled`` must be structurally incapable of seeing outer test rows,
    the same contract ``tune`` has. Asserted by spying on what the scorer is
    actually given, not by reading the call site.
    """
    import pipeline.tuning as tuning
    from pipeline.evaluation import PurgedPanelWalkForward

    panel = _synthetic_panel(n_dates=800)
    dates = panel["date"].to_numpy()
    splitter = PurgedPanelWalkForward(n_folds=3, horizon=HORIZON,
                                      embargo=HORIZON, min_train=400)
    train_idx, test_idx = list(splitter.split(dates))[0]
    test_dates = set(dates[test_idx])

    seen: list[set] = []
    real = tuning.purged_panel_cv_score

    def spy(frame, *a, **kw):
        seen.append(set(frame["date"]))
        return real(frame, *a, **kw)

    tuning.purged_panel_cv_score = spy
    try:
        tuning.tune_pooled(panel.iloc[train_idx], ["a", "b"],
                           horizon=HORIZON, n_trials=2)
    finally:
        tuning.purged_panel_cv_score = real

    assert seen, "the scorer was never called; the spy proves nothing"
    for call in seen:
        assert not (call & test_dates), (
            f"the pooled search was handed {len(call & test_dates)} dates from "
            f"the fold it is later scored on")


def test_the_pooled_inner_cv_purges_too():
    """
    Nesting is not automatic. The inner CV that scores each trial splits the
    training slice again, and if IT were contiguous the search would be
    selected on leaked labels even though the outer fold is clean — which is
    exactly F3 one level down.
    """
    from pipeline.evaluation import PurgedPanelWalkForward

    panel = _synthetic_panel(n_dates=800)
    inner = PurgedPanelWalkForward(
        n_folds=3, horizon=HORIZON, embargo=HORIZON,
        min_train=max(2 * HORIZON, panel["date"].nunique() // 2))

    splits = list(inner.split(panel["date"].to_numpy()))
    assert splits, "no inner folds; the test would be vacuous"
    grid = np.unique(panel["date"].to_numpy())
    for train_idx, test_idx in splits:
        d = panel["date"].to_numpy()
        last_train = np.searchsorted(grid, d[train_idx].max())
        first_test = np.searchsorted(grid, d[test_idx].min())
        assert first_test - last_train - 1 >= HORIZON


def test_the_ticker_feature_needs_no_encoding_fitted_across_a_fold():
    """
    XGBoost 3.x reads a pandas category natively, so the ticker feature carries
    NO fitted statistic — there is no target or frequency encoding that could
    be computed with sight of held-out rows. Pinned because switching to an
    encoding would silently reintroduce exactly that, in the one part of this
    pipeline that has been leakage-tested hardest.
    """
    import xgboost
    from tools.stage2b_pooled import TICKER_COL, with_ticker

    assert int(xgboost.__version__.split(".")[0]) >= 2, (
        "native categorical support is assumed; below XGBoost 1.5 this path "
        "would need an encoding, and that encoding would need fold discipline")

    panel = _synthetic_panel(n_dates=60, n_tickers=5)
    tagged = with_ticker(panel, "identity")
    assert str(tagged[TICKER_COL].dtype) == "category"
    # Categories come from the ticker column alone — nothing derived from the
    # target, which is what makes the feature fold-independent.
    assert list(tagged[TICKER_COL].cat.categories) == sorted(panel["ticker"].unique())
    assert (tagged[TICKER_COL].astype(str).to_numpy() == panel["ticker"].to_numpy()).all()


def test_the_placebo_preserves_frequency_and_breaks_identity():
    """
    The within-date shuffle must keep every date's label multiset intact — a
    placebo that also changed the cross-section's composition would be testing
    two things at once.
    """
    from tools.stage2b_pooled import TICKER_COL, with_ticker

    panel = _synthetic_panel(n_dates=40, n_tickers=8)
    placebo = with_ticker(panel, "shuffled")

    for date, g in placebo.groupby("date"):
        assert sorted(g[TICKER_COL].astype(str)) == sorted(
            panel[panel["date"] == date]["ticker"]), (
            f"{date}: the placebo changed which names are present")

    matches = (placebo[TICKER_COL].astype(str).to_numpy()
               == panel["ticker"].to_numpy()).mean()
    assert matches < 0.5, (
        "the placebo left most rows carrying their own ticker; it is not "
        "breaking the identity-to-return link")


# ── Overlapping-label t-statistic correction ──────────────────────────────────

def test_effective_sample_size_discounts_overlap():
    assert effective_sample_size(3000, 30) == pytest.approx(100.0)
    assert effective_sample_size(100, 1) == pytest.approx(100.0)
    assert effective_sample_size(5, 30) == pytest.approx(1.0)


def test_t_statistic_uses_effective_sample_size():
    """
    A t-statistic computed on raw row count would be ~sqrt(30) times larger.
    This is what turned a rank IC of 0.18 into an apparent t of 7.9.
    """
    rng = np.random.default_rng(1)
    n = 2000
    y_true = rng.normal(size=n)
    y_pred = y_true * 0.2 + rng.normal(size=n)

    m = compute_metrics(y_true, y_pred, horizon=30)
    naive_t = m["rank_ic"] * np.sqrt(n - 1)

    assert m["rank_ic_t"] < naive_t / 4, (
        "t-statistic does not appear to discount overlapping labels"
    )
    assert m["n_effective"] == pytest.approx(n / 30, rel=0.01)


# ── Baseline reporting ────────────────────────────────────────────────────────

def test_metrics_always_carry_a_majority_baseline():
    """
    A directional accuracy figure without its baseline is uninterpretable —
    the previous system reported 85% while the majority baseline was ~59%.
    """
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=500)
    y_pred = rng.normal(size=500)

    m = compute_metrics(y_true, y_pred, horizon=30)
    assert "majority_hit_rate" in m
    assert "mae_naive_zero" in m
    assert "beats_naive_mae" in m


def test_majority_baseline_is_never_below_fifty_percent():
    for p_up in [0.1, 0.3, 0.5, 0.7, 0.95]:
        rng = np.random.default_rng(3)
        y = np.where(rng.random(1000) < p_up, 1.0, -1.0)
        assert majority_hit_rate(y) >= 50.0


def test_a_zero_skill_model_reports_near_zero_ic():
    """Sanity check that the harness cannot manufacture skill from noise."""
    rng = np.random.default_rng(4)
    y_true = rng.normal(size=3000)
    y_pred = rng.normal(size=3000)

    m = compute_metrics(y_true, y_pred, horizon=30)
    assert abs(m["rank_ic"]) < 0.06
    assert abs(m["rank_ic_t"]) < 2.0


# ── Stage 1: the reversal features are point-in-time ──────────────────────────

def test_reversal_features_at_a_date_are_blind_to_that_date_and_after():
    """
    A reversal feature at t may read closes up to t-1 and nothing else — its
    window skips session t, and its beta is the one known at t-1. So corrupting
    every close from a date D onward must leave every feature at dates <= D
    exactly as it was. This is stronger than the purge, which only separates
    folds: a feature that read session t would carry t's own return into the
    row whose 30-session label starts there.
    """
    from pipeline.reversal import REVERSAL_COLS, reversal_features

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=150).strftime("%Y-%m-%d")
    market = rng.normal(0.0, 0.01, len(dates))
    panel = pd.concat([
        pd.DataFrame({"date": dates, "ticker": f"L{i:02d}",
                      "close": 100 * np.exp(np.cumsum(
                          (0.6 + i / 20) * market + rng.normal(0, 0.02, len(dates))))})
        for i in range(15)], ignore_index=True)
    cut = dates[110]

    before = reversal_features(panel)
    shocked = panel.copy()
    later = shocked["date"] >= cut
    shocked.loc[later, "close"] *= rng.uniform(0.3, 3.0, int(later.sum()))
    after = reversal_features(shocked)

    key = ["date", "ticker"]
    a = before[before["date"] <= cut].set_index(key)[REVERSAL_COLS]
    b = after[after["date"] <= cut].set_index(key)[REVERSAL_COLS]
    assert a.notna().any().all(), "not vacuous: every column is defined somewhere"
    pd.testing.assert_frame_equal(a, b)


def test_sue_features_at_a_date_are_blind_to_every_filing_not_yet_public():
    """
    A SUE feature at session t may use only results public by t's close: a
    filing disseminated before 15:00 IST on t, or on any earlier day. So
    corrupting every EPS disclosed at or after a cut instant (the cut day at
    15:00) must leave every feature at sessions up to the cut day exactly as
    it was. It must also move the very next session, so nothing is lagged more
    than the rule says.

    The fixture places the cases a random panel would rarely hit:
    * filings at 16:00 and at exactly 15:00:00 on the cut day (public once
      the closing window has opened);
    * a filing at 14:00 on it (public before the close, and not corrupted);
    * an OLD quarter disclosed after the cut, which must not enter any
      earlier quarter's year-ago or sigma term.
    """
    from pipeline.earnings import (EVENT_COLS, SUE_FFILL, announcements,
                                   event_features)

    rng = np.random.default_rng(12)
    grid = list(pd.bdate_range("2019-01-01", "2023-12-29").strftime("%Y-%m-%d"))
    ends = pd.date_range("2016-03-31", "2023-09-30", freq="QE")
    rows = []
    for i in range(12):
        eps = 10 + rng.normal(0, 1, len(ends)).cumsum()
        hours = rng.choice([11, 14, 16, 19], len(ends))
        for e, x, h in zip(ends, eps, hours):
            rows.append((f"S{i:02d}", e, "standalone", x,
                         e + pd.Timedelta(days=int(rng.integers(20, 55)), hours=int(h))))
    table = pd.DataFrame(rows, columns=["symbol", "period_end", "basis", "eps", "disclosed"])
    actions = pd.DataFrame(columns=["symbol", "ex_date", "kind", "factor", "subject"])
    tickers = [f"S{i:02d}.NS" for i in range(12)]
    cols = EVENT_COLS + [SUE_FFILL]

    cut_day = grid[800]
    cut = pd.Timestamp(cut_day + " 15:00")          # the closing window opens
    qe = max(e for e in ends if e + pd.Timedelta(days=20) <= pd.Timestamp(cut_day))
    at = lambda sym, e: (table["symbol"] == sym) & (table["period_end"] == e)   # noqa: E731
    for sym in ("S00", "S01", "S02", "S03"):
        table.loc[at(sym, qe), "disclosed"] = pd.Timestamp(cut_day + " 16:00")
    table.loc[at("S04", qe), "disclosed"] = pd.Timestamp(cut_day + " 14:00")
    table.loc[at("S06", qe), "disclosed"] = cut                 # exactly 15:00:00
    table.loc[at("S05", qe - pd.offsets.QuarterEnd(5)), "disclosed"] = cut + pd.Timedelta(days=1)
    nxt = table["period_end"] > qe
    table.loc[nxt, "disclosed"] = table.loc[nxt, "disclosed"].clip(lower=cut + pd.Timedelta(days=2))

    before = event_features(announcements(table, actions), grid, tickers)
    shocked = table.copy()
    later = shocked["disclosed"] >= cut
    shocked.loc[later, "eps"] = rng.normal(0, 50, int(later.sum()))
    after = event_features(announcements(shocked, actions), grid, tickers)

    key = ["date", "ticker"]
    a = before[before["date"] <= cut_day].set_index(key)[cols]
    b = after[after["date"] <= cut_day].set_index(key)[cols]
    assert (a["sue_missing"] == 0).any(), "not vacuous: SUEs are defined before the cut"
    assert a.loc[(cut_day, "S04.NS"), "sue_age"] == 0, "the 14:00 filing is usable that day"
    pd.testing.assert_frame_equal(a, b)
    moved = before[before["date"] > cut_day].set_index(key)[cols]         .compare(after[after["date"] > cut_day].set_index(key)[cols])
    assert moved.index.get_level_values("date").min() == grid[801], (
        "a 16:00 filing on the cut day is usable at the very next session")


# ── P6: the horizon-parameterised purge and embargo ───────────────────────────
#
# This is a change to the leakage-critical core, so it gets its own tests
# rather than leaning on the 30-session ones above. Three things can go wrong
# and only the third is visible from outside:
#
#   1. the derivation returns the wrong width;
#   2. `run_arm` computes the right width and then splits on the old constant;
#   3. the OUTER split widens and the nested search does not, so every
#      hyperparameter is chosen across a boundary the outer fold refuses to
#      trust — F3 one level down, and it reads as a merely optimistic number.

P6_HORIZONS = (5, 10, 20, 30)


def test_the_purge_is_never_narrower_than_the_label_it_must_span():
    from pipeline.evaluation import (
        POLITIS_WHITE_FLOOR_SESSIONS,
        horizon_purge_embargo,
    )

    for h in P6_HORIZONS + (63, 90, 200):
        assert horizon_purge_embargo(h) >= h, (
            f"a purge of {horizon_purge_embargo(h)} cannot span an {h}-session "
            f"label; training labels would reach into the test window")

    # And never narrower than the panel's own measured serial dependence,
    # which is the half of the rule that does not follow from arithmetic.
    for h in P6_HORIZONS:
        assert horizon_purge_embargo(h) >= POLITIS_WHITE_FLOOR_SESSIONS

    # Above the floor the label width takes over, so the rule is a max and not
    # a constant. A mutant returning the floor unconditionally fails here...
    assert horizon_purge_embargo(200) == 200
    # ...and one returning the horizon unconditionally fails here.
    assert horizon_purge_embargo(5) == POLITIS_WHITE_FLOOR_SESSIONS

    # The legacy rule every pre-P6 result was measured under stays reachable,
    # and is what the h=30 regression pin runs.
    for h in P6_HORIZONS:
        assert horizon_purge_embargo(h, floor=h) == h


def test_the_purge_derivation_refuses_a_nonsensical_width():
    from pipeline.evaluation import horizon_purge_embargo

    for bad in (0, -1, -30):
        with pytest.raises(ValueError):
            horizon_purge_embargo(bad)
        with pytest.raises(ValueError):
            horizon_purge_embargo(30, floor=bad)


def test_run_arm_refuses_a_purge_narrower_than_its_own_label():
    """The guard that makes the leak unconstructable rather than merely
    unlikely: a caller cannot ask for a 30-session label purged at 5."""
    from tools.stage2b_pooled import run_arm

    panel = _synthetic_panel(n_dates=400, n_tickers=6)
    with pytest.raises(ValueError, match="narrower than"):
        run_arm(panel, "mae", "none", features=["a", "b"],
                horizon=30, purge=5, fixed_params={"n_estimators": 2})


def _record_run_arm(panel, horizon, purge, monkeypatch, min_train=200):
    """
    Runs `run_arm` with the nested search replaced by a recorder, and returns
    what the training slice and the test block actually were, per fold.

    The search is STUBBED rather than skipped via `fixed_params`, because
    `fixed_params` is the one path that never calls `tune_pooled` — and the
    horizon the SEARCH is purged at is exactly what needs observing.
    """
    import tools.stage2b_pooled as sp

    seen = []

    def fake_tune(frame, features, target="target_return", horizon=30,
                  n_trials=10, tuning_objective="mae", enable_categorical=False):
        seen.append({"train_dates": np.unique(frame["date"].to_numpy()),
                     "search_horizon": horizon})
        return {"n_estimators": 2, "max_depth": 2, "tree_method": "hist"}

    monkeypatch.setattr(sp, "tune_pooled", fake_tune)
    preds, _ = sp.run_arm(panel, "mae", "none", n_trials=1, min_train=min_train,
                          features=["a", "b"], horizon=horizon, purge=purge,
                          verbose=False)
    return preds, seen


@pytest.mark.parametrize("horizon", P6_HORIZONS)
def test_run_arm_purges_and_embargoes_at_the_width_it_was_given(horizon, monkeypatch):
    """
    Measured through `run_arm` itself, on the real training slices it hands the
    search and the real test blocks it scores.

    Asserting on a splitter the test constructs for itself would pass happily
    against a `run_arm` that had gone back to the hardcoded 30 — the mutant
    that matters most here, because under the DECIDING rule at h=5 that mutant
    uses a 60-date gap where 126 was asked for, and nothing in the output says
    so.
    """
    from pipeline.evaluation import horizon_purge_embargo

    panel = _synthetic_panel(n_dates=900, n_tickers=8)
    grid = np.unique(panel["date"].to_numpy())

    for purge in (horizon, horizon_purge_embargo(horizon)):
        preds, seen = _record_run_arm(panel, horizon, purge, monkeypatch)
        assert seen, "no fold ran; the test would be vacuous"
        assert len(seen) == preds["fold"].nunique()

        for rec, (fold, block) in zip(seen, preds.groupby("fold", sort=True)):
            last_train = int(np.searchsorted(grid, rec["train_dates"].max()))
            first_test = int(np.searchsorted(grid, block["date"].min()))
            gap = first_test - last_train - 1

            assert gap >= 2 * purge, (
                f"h={horizon}, purge={purge}, fold {fold}: gap of {gap} grid "
                f"dates is below the {2 * purge} the purge and embargo require")
            assert gap >= horizon, (
                f"h={horizon}, fold {fold}: a training label spanning {horizon} "
                f"sessions reaches into the test window across a {gap}-date gap")
            # The nested search is purged at the SAME width as the outer split.
            assert rec["search_horizon"] == purge, (
                f"the outer fold split at {purge} while the inner search was "
                f"purged at {rec['search_horizon']}; every hyperparameter would "
                f"be chosen across a boundary the outer fold refuses to trust")
            # And the search was handed training rows only.
            assert rec["train_dates"].max() < block["date"].min()


def test_the_legacy_rule_reproduces_the_pre_p6_split_exactly():
    """
    The regression pin, at the altitude a unit test can reach: parameterising
    the splitter must change nothing when the parameters are set to what they
    used to be hardcoded at. The full pin is the run itself, against
    `stage2b_pooled_oos.npz` at drift <= 1e-9.
    """
    from pipeline.evaluation import PurgedPanelWalkForward

    panel = _synthetic_panel(n_dates=1200, n_tickers=8)
    dates = panel["date"].to_numpy()

    was = PurgedPanelWalkForward(n_folds=5, horizon=HORIZON, embargo=HORIZON,
                                 min_train=500)
    now = PurgedPanelWalkForward(n_folds=5, horizon=30, embargo=30, min_train=500)

    old = list(was.split(dates))
    new = list(now.split(dates))
    assert old and len(old) == len(new)
    for (a_tr, a_te), (b_tr, b_te) in zip(old, new):
        assert np.array_equal(a_tr, b_tr)
        assert np.array_equal(a_te, b_te)


def test_a_shorter_horizon_does_not_shrink_the_embargo_below_the_measured_floor():
    """
    The empirical half of the rule, and the one a reader is most likely to
    "simplify" away. Stage 0c measured this panel's dependence at 35.8-62.5
    sessions against a 30-session label, so it is a property of the panel and
    not of the label: a 5-session label does not make it five times shorter.
    """
    from pipeline.evaluation import (
        POLITIS_WHITE_FLOOR_SESSIONS,
        horizon_purge_embargo,
    )

    widths = [horizon_purge_embargo(h) for h in (5, 10, 20, 30)]
    assert len(set(widths)) == 1, (
        f"the purge tracked the horizon ({widths}); below the measured floor "
        f"it must not")
    assert widths[0] == POLITIS_WHITE_FLOOR_SESSIONS
