"""
tests/test_phase3_features.py — the news scorer, news features and the regime.

The guards here are ordered by how quietly each would fail.

1. THE AS-OF BOUNDARY. A row dated t may use articles published on or before t.
   An off-by-one hands the model tomorrow's news and reads as a breakthrough —
   the same failure `series._history_ending_at` carries, tested the same way:
   corrupt everything after the boundary and require bit-identical output.

2. NULL vs ZERO. A window we searched and found empty is a measurement. A
   window we never searched is not. Collapsing them is how a blocked fetch for
   all 95 tickers became a market-wide silent day, and how a dead FinBERT
   loader displayed a confident NEUTRAL for months.

3. LEVELS. A ticker's average news volume is a near-constant per-ticker
   attribute, measured in CLAUDE.md §7 as worth a rebalance t of +0.77 from
   pure noise. Counts must be expressed against the ticker's own baseline.

4. LABEL ORDER READ, NOT ASSUMED. Three series checkpoints put their median
   quantile at three different indices and a hardcoded position silently
   returned the 30th percentile of every prediction. A permuted sentiment label
   map fails identically: every score keeps its magnitude and flips its sign.

5. MARKET-WIDE COLUMNS ARE ZERO AFTER WITHIN-DATE STANDARDISATION. That is how
   `fii_net_flow` spent years in FEATURES contributing nothing, so the regime
   is allowed to reach the panel only as an interaction.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from pipeline.news_features import (
    NEWS_COLS,
    NEWS_WINDOW_SESSIONS,
    build_news_features,
    load_observed_months,
)
from pipeline.regime import (
    REGIME_INTERACTIONS,
    REGIME_STATE,
    build_regime_features,
    compute_market_state,
    describe_regime,
    rolling_beta,
)


@pytest.fixture
def engine(monkeypatch):
    import data.db as db

    eng = create_engine("sqlite://")
    monkeypatch.setattr(db, "_ENGINE", eng, raising=False)
    db.init_db()
    return eng


SCORER = "test/scorer@abc123"

#: cross_sectional_zscore ZEROES any date carrying fewer than MIN_NAMES_PER_DATE
#: (10) names — a z-score over six numbers manufactures outliers rather than
#: removing them. A synthetic panel below that breadth makes every standardised
#: column identically zero, which silently turns any assertion about
#: cross-sectional variance into a tautology. The first version of this file
#: used three tickers and hit exactly that.
WIDE = tuple(f"T{i:02d}.NS" for i in range(12))

_NEXT_ID = [0]


def _sessions(n=120, start="2024-01-01"):
    return [d.strftime("%Y-%m-%d")
            for d in pd.bdate_range(start=start, periods=n)]


def _panel(dates, tickers=("AAA.NS",), close=100.0):
    rows = []
    rng = np.random.default_rng(0)
    for t in tickers:
        price = close
        for d in dates:
            price *= float(np.exp(rng.normal(0, 0.01)))
            rows.append({"date": d, "ticker": t, "close": price})
    return pd.DataFrame(rows)


def _seed_news(engine, rows, scorer=SCORER, observed=None):
    """
    rows: (published_at, ticker, score). `observed` defaults to every month.

    Ids come from a module counter rather than `enumerate`, because several
    tests seed twice against one engine and a per-call index collides on the
    article_id primary key — the second batch then silently vanishes into
    ON CONFLICT, or raises, depending on the write path.
    """
    now = datetime.now(timezone.utc).isoformat()
    with engine.connect() as conn:
        for day, ticker, score in rows:
            _NEXT_ID[0] += 1
            aid = f"art{_NEXT_ID[0]}"
            conn.execute(text(
                "INSERT INTO news_articles (article_id, published_at, title, "
                "url, source, provider, first_seen) VALUES "
                "(:a,:p,:t,'u','s','stub',:n)"),
                {"a": aid, "p": day, "t": f"headline {aid}", "n": now})
            conn.execute(text(
                "INSERT INTO news_mentions (article_id, ticker, matched_by, "
                "first_seen) VALUES (:a,:t,'alias',:n)"),
                {"a": aid, "t": ticker, "n": now})
            conn.execute(text(
                "INSERT INTO news_scores (article_id, scorer_id, label, score, "
                "confidence, scored_at) VALUES (:a,:s,'neutral',:v,0.9,:n)"),
                {"a": aid, "s": scorer, "v": float(score), "n": now})

        months = observed if observed is not None else sorted(
            {(t, d[:7]) for d, t, _ in rows})
        for ticker, month in months:
            start = f"{month}-01"
            end = str(pd.Period(month, freq="M").end_time.date())
            # DO NOTHING because several tests seed twice against one engine
            # and a month recorded by the first call is still recorded.
            conn.execute(text(
                "INSERT INTO news_coverage (ticker, window_start, window_end, "
                "provider, status, n_articles, saturated, attempted_at) VALUES "
                "(:t,:s,:e,'stub','ok',1,0,:n) "
                "ON CONFLICT (ticker, window_start, window_end, provider) "
                "DO NOTHING"),
                {"t": ticker, "s": start, "e": end, "n": now})
        conn.commit()


# ── 1. The as-of boundary ─────────────────────────────────────────────────────

def test_a_row_cannot_see_an_article_published_after_it(engine):
    """
    THE GUARANTEE THE WHOLE FEATURE RESTS ON, tested the way the series adapter's
    is: corrupt everything after the boundary and require the output at the
    boundary to be bit-identical. An off-by-one here hands the model tomorrow's
    news, produces a plausible table, and raises nothing.
    """
    dates = _sessions(90)
    panel = _panel(dates)
    cut = dates[44]

    _seed_news(engine, [(d, "AAA.NS", +0.5) for d in dates[:45]])
    before = build_news_features(panel, engine)

    # Now add wildly different news AFTER the cut.
    _seed_news(engine, [(d, "AAA.NS", -0.9) for d in dates[45:]])
    after = build_news_features(panel, engine)

    a = before[before["date"] <= cut].set_index("date")[NEWS_COLS]
    b = after[after["date"] <= cut].set_index("date")[NEWS_COLS]
    pd.testing.assert_frame_equal(a, b), "future articles moved a past row"

    # And the future news must actually have changed something, or the test
    # would pass against a build that ignores news entirely.
    late = after[after["date"] > cut]["news_sent_mean"].dropna()
    assert len(late) and late.min() < 0, "the corrupted tail was never read"


def test_an_article_published_ON_the_date_is_usable_that_day(engine):
    """
    The boundary is inclusive. A trailing window that excluded the current
    session would silently discard the freshest article on every row — the
    opposite error, equally invisible.
    """
    dates = _sessions(40)
    panel = _panel(dates)
    _seed_news(engine, [(dates[10], "AAA.NS", +0.8)])

    out = build_news_features(panel, engine).set_index("date")
    assert out.loc[dates[10], "news_sent_mean"] == pytest.approx(0.8)
    assert pd.isna(out.loc[dates[9], "news_sent_mean"]), (
        "an article dated t must not be visible at t-1")


# ── 2. Null versus zero ───────────────────────────────────────────────────────

def test_an_unsearched_month_is_null_and_an_empty_one_is_not(engine):
    """
    Two months with no articles. One we searched and found empty; one we never
    searched. The first is a measurement, the second is not, and a feature that
    reports 0.0 for both is the sentiment-gauge defect in a new place.
    """
    dates = _sessions(160, start="2024-01-01")
    panel = _panel(dates)
    months = sorted({d[:7] for d in dates})

    # Real news through the first four months, so the ticker's own trailing
    # baseline exists — `news_count_excess` is undefined before that by design,
    # and testing it earlier would be testing the warm-up, not the rule.
    early = [d for d in dates if d[:7] in months[:4]]
    _seed_news(engine, [(d, "AAA.NS", +0.2) for d in early])

    quiet, unseen = months[4], months[5]
    # We SEARCHED the fifth month and it held nothing. That is a measurement.
    _seed_news(engine, [], observed=[("AAA.NS", quiet)])
    # The sixth month we never searched at all.

    out = build_news_features(panel, engine).set_index("date")
    in_quiet = [d for d in dates if d[:7] == quiet]
    in_unseen = [d for d in dates if d[:7] == unseen]

    assert out.loc[in_quiet, "news_count_excess"].notna().any(), (
        "a month we searched and found empty HAS a measurement — its count is "
        "zero, and zero is a reading")
    assert out.loc[in_unseen, NEWS_COLS].isna().all().all(), (
        "a month we never searched has NO measurement and must be NULL")


def test_the_observed_flag_says_whether_we_LOOKED_not_whether_we_FOUND():
    """
    `news_observed` is what tells the ridge that a filled-in 0.0 is a
    placeholder rather than a neutral reading. It must therefore track COVERAGE
    — did we search this window — and not whether any article turned up.

    Keying it on `news_sent_mean` would collapse "we looked and it was quiet"
    into "we never looked", which is the exact distinction `news_coverage` was
    added to preserve, thrown away one layer downstream.
    """
    import inspect

    from pipeline.baselines import compare_baselines

    source = inspect.getsource(compare_baselines)
    assert 'panel["news_observed"] = panel["news_count_excess"].notna()' in source, (
        "news_observed must be derived from the COUNT column, which exists "
        "for any searched window, not from the sentiment column, which is "
        "null whenever a searched window happened to be quiet")


def test_a_blocked_window_is_not_treated_as_searched(engine):
    """`news_coverage` rows that are not `ok` are periods we could not see."""
    dates = _sessions(40)
    panel = _panel(dates)
    now = datetime.now(timezone.utc).isoformat()
    with engine.connect() as conn:
        conn.execute(text(
            "INSERT INTO news_coverage (ticker, window_start, window_end, "
            "provider, status, n_articles, saturated, attempted_at) VALUES "
            "('AAA.NS','2024-01-01','2024-01-31','stub','blocked',0,0,:n)"),
            {"n": now})
        conn.commit()

    assert load_observed_months(engine) == set()
    out = build_news_features(panel, engine)
    assert out[NEWS_COLS].isna().all().all()


# ── 3. Levels ─────────────────────────────────────────────────────────────────

def test_volume_is_expressed_against_the_tickers_own_baseline(engine):
    """
    THE +0.77-FROM-NOTHING LANDMINE. A raw count is a near-constant per-ticker
    attribute: a tree partitions on it, identifies the ticker, and learns which
    names paid in the training window. Two tickers with constant but DIFFERENT
    news volume must end up with the same excess, because neither is doing
    anything unusual.
    """
    dates = _sessions(140)
    panel = _panel(dates, tickers=("AAA.NS", "BBB.NS"))

    rows = []
    for d in dates:
        rows += [(d, "AAA.NS", 0.1)] * 1        # quiet name, steady
        rows += [(d, "BBB.NS", 0.1)] * 8        # loud name, equally steady
    _seed_news(engine, rows)

    out = build_news_features(panel, engine)
    tail = out[out["date"] >= dates[-10]]
    a = tail[tail["ticker"] == "AAA.NS"]["news_count_excess"].mean()
    b = tail[tail["ticker"] == "BBB.NS"]["news_count_excess"].mean()

    assert abs(a - b) < 0.2, (
        f"a name with 8x the coverage scored {b:+.2f} against {a:+.2f} — the "
        f"feature is reading ticker identity, not news")


def test_a_volume_spike_does_move_the_feature(engine):
    """The complement: the baseline must not flatten a genuine surge to nothing."""
    dates = _sessions(140)
    panel = _panel(dates)

    rows = [(d, "AAA.NS", 0.1) for d in dates[:-5]]
    for d in dates[-5:]:
        rows += [(d, "AAA.NS", 0.1)] * 20
    _seed_news(engine, rows)

    out = build_news_features(panel, engine).set_index("date")
    assert out.loc[dates[-1], "news_count_excess"] > out.loc[dates[-30], "news_count_excess"]


# ── 4. One scorer only ────────────────────────────────────────────────────────

def test_features_use_exactly_one_scorer(engine):
    """
    Mixing checkpoints puts two different quantities under one column name,
    which is F7 in a new costume. The default is whichever scorer holds the
    most rows, so a half-finished re-score cannot contaminate a feature.
    """
    dates = _sessions(40)
    panel = _panel(dates)
    _seed_news(engine, [(dates[5], "AAA.NS", +0.9)], scorer="old@1")
    _seed_news(engine, [(dates[5], "AAA.NS", -0.9)], scorer="new@2")

    only_old = build_news_features(panel, engine, scorer_id="old@1").set_index("date")
    only_new = build_news_features(panel, engine, scorer_id="new@2").set_index("date")
    assert only_old.loc[dates[5], "news_sent_mean"] > 0
    assert only_new.loc[dates[5], "news_sent_mean"] < 0


# ── 5. The regime ─────────────────────────────────────────────────────────────

def test_only_interactions_reach_the_panel(engine):
    """
    A bare market-wide column is identically ZERO after cross_sectional_zscore
    and would be dead weight in exactly the way fii_net_flow was. The builder
    must not hand one back.
    """
    dates = _sessions(200)
    panel = _panel(dates, tickers=WIDE)
    out = build_regime_features(panel, engine)

    assert set(out.columns) == {"date", "ticker"} | set(REGIME_INTERACTIONS)
    for col in REGIME_STATE:
        assert col not in out.columns, (
            f"{col} is market-wide; every ticker shares it, so it standardises "
            f"to zero within each date and cannot carry ranking information")


def test_a_market_wide_column_really_does_standardise_to_zero(engine):
    """
    The premise of the rule above, measured rather than asserted — because it
    is the reason two columns sat in FEATURES for years doing nothing.
    """
    from pipeline.panel import cross_sectional_zscore

    dates = _sessions(200)
    panel = _panel(dates, tickers=WIDE)
    state = compute_market_state(panel, engine)
    merged = panel.merge(state, on="date", how="left")

    z = cross_sectional_zscore(merged, ["regime_vol"])
    assert float(np.nanmax(np.abs(z["regime_vol"]))) == pytest.approx(0.0, abs=1e-9)

    # ...whereas the interaction does not, which is the entire point.
    inter = build_regime_features(panel, engine)
    zi = cross_sectional_zscore(panel.merge(inter, on=["date", "ticker"]),
                                ["beta_x_regime_vol"])
    assert float(np.nanstd(zi["beta_x_regime_vol"])) > 0.1


def test_beta_is_trailing_and_never_sees_the_future(engine):
    """
    A fold-wide beta used as a FEATURE would be F2 in miniature — a quantity
    estimated partly from the test window. Corrupting the tail must not move an
    earlier row.
    """
    dates = _sessions(200)
    panel = _panel(dates, tickers=WIDE)
    cut = dates[150]

    before = rolling_beta(panel)
    shocked = panel.copy()
    tail = shocked["date"] > cut
    shocked.loc[tail, "close"] = shocked.loc[tail, "close"] * 5.0
    after = rolling_beta(shocked)

    a = before[before["date"] <= cut].set_index(["date", "ticker"])["beta"]
    b = after[after["date"] <= cut].set_index(["date", "ticker"])["beta"]
    pd.testing.assert_series_equal(a, b)


def test_regime_percentiles_use_only_history_up_to_the_date(engine):
    """
    "Volatility is in its 90th percentile" must be a statement a reader could
    have made AT THE TIME. Ranking against the full sample would quietly need
    the rest of the panel.
    """
    dates = _sessions(200)
    panel = _panel(dates, tickers=WIDE)
    state = compute_market_state(panel, engine)

    mid = dates[120]
    a = describe_regime(state, as_of=mid)
    b = describe_regime(state.copy(), as_of=mid)
    assert a == b

    full = describe_regime(state)
    assert a["date"] == mid and full["date"] != mid


# ── 6. The comparator wiring ──────────────────────────────────────────────────

def test_the_news_comparator_receives_its_columns_through_the_constructor():
    """
    `LinearFactorModel` reads `self.columns`, NOT `X`. Passing extra columns
    through `feature_cols` alone silently does nothing — that produced a
    `linear_factor+val` row identical to `linear_factor` to five decimal places,
    which read as "valuation does not help" rather than "valuation was never
    supplied".
    """
    from pipeline.baselines import (
        FACTORS, NewsAugmentedFactor, RegimeAugmentedFactor,
        baseline_feature_columns,
    )

    news = NewsAugmentedFactor()
    assert set(NEWS_COLS).issubset(news.columns)
    assert "news_observed" in news.columns
    assert set(FACTORS).issubset(news.columns)

    regime = RegimeAugmentedFactor()
    assert set(REGIME_INTERACTIONS).issubset(regime.columns)

    # And the harness must hand it the same frame in production.
    assert set(NEWS_COLS).issubset(baseline_feature_columns("news_factor"))
    assert set(REGIME_INTERACTIONS).issubset(baseline_feature_columns("regime_factor"))


def test_the_observed_indicator_exists_because_fillna_means_neutral():
    """
    `LinearFactorModel.fit` fills NaN with 0.0, and for a SIGNED sentiment score
    0.0 is "measured as neutral" rather than "no measurement". Without an
    explicit indicator the early panel — ~1 article per ticker-month before
    2022 — would teach the model that the market was permanently neutral.
    """
    from pipeline.baselines import NewsAugmentedFactor

    assert "news_observed" in NewsAugmentedFactor().columns, (
        "a signed feature filled with 0.0 needs a companion flag saying "
        "whether the zero is a reading or a placeholder")


# ── 7. The scorer ─────────────────────────────────────────────────────────────

def test_the_label_order_is_read_from_the_checkpoint_not_assumed():
    """
    Three series checkpoints put their median quantile at three different
    indices; a hardcoded position returned the 30th percentile of every
    prediction with no error anywhere. A permuted sentiment label map fails the
    same way — every score keeps its magnitude and flips its sign.
    """
    from pipeline.news_scoring import ScorerUnavailable, label_map

    class _Cfg:
        id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

    class _Model:
        config = _Cfg()

    assert label_map(_Model()) == {0: "negative", 1: "neutral", 2: "positive"}

    class _Wrong:
        class config:
            id2label = {0: "LABEL_0", 1: "LABEL_1"}

    with pytest.raises(ScorerUnavailable):
        label_map(_Wrong())


def test_importing_the_scorer_does_not_import_torch():
    """
    torch must never reach requirements.txt, and `pipeline.baselines` and the
    API can transitively import this module. The import lives inside
    `load_scorer` for that reason.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import pipeline.news_scoring, sys; "
         "print('torch' in sys.modules or 'transformers' in sys.modules)"],
        capture_output=True, text=True)
    assert result.stdout.strip() == "False", result.stdout + result.stderr


# ── 8. The relevance audit ────────────────────────────────────────────────────

def test_the_relevance_audit_counts_the_four_cells_correctly():
    """
    The audit is a MEASUREMENT tool, so its arithmetic is the thing to pin. A
    precision figure that is quietly wrong is worse than none: it would be
    quoted as evidence that the alias filter is fine.

    Measured on the real archive 2026-09-04: precision 0.680, recall 0.723 over
    100 hand-labelled articles, with 13 of 16 false positives coming from two
    tickers whose company name is a common word (TRENT, LTM).
    """
    import sys, os
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    from tools.audit_news_relevance import score

    rows = [
        {"kept_by_filter": True,  "label": 1},   # TP
        {"kept_by_filter": True,  "label": 1},   # TP
        {"kept_by_filter": True,  "label": 0},   # FP
        {"kept_by_filter": False, "label": 1},   # FN
        {"kept_by_filter": False, "label": 0},   # TN
        {"kept_by_filter": False, "label": None},  # unlabelled: ignored
    ]
    m = score(rows)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["n_labelled"] == 5, "an unlabelled row must not count as a zero"
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_an_unlabelled_worksheet_refuses_rather_than_reporting_zero():
    """
    Reporting precision 0.000 on an unlabelled sheet would read as "the filter
    keeps nothing right" instead of "nobody has labelled it yet" — the same
    None-vs-0.0 confusion this whole phase is about.
    """
    import sys, os
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    from tools.audit_news_relevance import score

    with pytest.raises(SystemExit):
        score([{"kept_by_filter": True, "label": None}])
