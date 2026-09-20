"""
P6 — the horizon sweep. The parts that are wrong silently.

The purge and embargo derivation is leakage-critical and lives in
`tests/test_leakage.py` beside the other boundary contracts, because that is
where someone looking for a boundary rule will look. What is here is everything
else that renders a plausible table while being wrong:

  - a label rebuilt by stepping the shared date grid instead of each ticker's
    own sessions;
  - a book rebalanced every 30 sessions while the label is 5, so the windows
    overlap by 25 of 30 and every t-statistic downstream inflates;
  - a Sharpe annualised through `portfolio.REBALANCES_PER_YEAR`, which is
    pinned at 252/30 and would understate an h=5 book's drag six-fold;
  - R5 quietly applied to, or withheld from, the wrong arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.panel import TARGET
from pipeline.portfolio import REBALANCES_PER_YEAR
from tools.p6_horizon import (
    ARM_FEATURES,
    FIVE,
    PLACEBO_COLS,
    R5_ARMS,
    _sharpe,
    _t,
    book,
    net_of_cost,
    panel_at,
    spread_per_ic,
    verdicts_b,
)

HORIZONS = (5, 10, 20, 30)


def _price_panel(n_dates: int = 300, n_tickers: int = 20, seed: int = 3,
                 ragged: bool = False) -> pd.DataFrame:
    """
    A panel carrying the columns `retarget_horizon` reads: date, ticker, close
    and the stored 30-session label.

    `ragged` drops a middle slab of one ticker's rows, which is what makes the
    per-ticker-shift contract testable at all. On a rectangular panel a shift
    across the shared union grid and a shift within each ticker are the same
    operation, so a test built on one could not tell them apart.
    """
    rng = np.random.default_rng(seed)
    dates = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_dates)]
    rows = []
    for j in range(n_tickers):
        px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_dates)))
        for i, d in enumerate(dates):
            if ragged and j == 0 and 100 <= i < 140:
                continue
            rows.append({"date": d, "ticker": f"T{j:02d}.NS", "close": px[i]})
    out = pd.DataFrame(rows)
    out[TARGET] = np.nan
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ── the label ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_relabelled_target_is_the_h_session_forward_log_return(horizon):
    """
    The identity `panel.retarget_horizon` rests on, asserted at the altitude
    P6 uses it: log(close[t+h]) - log(close[t]), per ticker.
    """
    panel = _price_panel()
    out = panel_at(panel, horizon)

    one = out[out["ticker"] == "T05.NS"].sort_values("date").reset_index(drop=True)
    lp = np.log(one["close"].to_numpy(dtype=float))
    expected = np.full(len(one), np.nan)
    expected[:-horizon] = lp[horizon:] - lp[:-horizon]

    got = pd.to_numeric(one[TARGET], errors="coerce").to_numpy(dtype=float)
    both = np.isfinite(got) & np.isfinite(expected)
    assert both.sum() == len(one) - horizon
    assert np.max(np.abs(got[both] - expected[both])) < 1e-12


def test_a_ticker_with_a_gap_is_shifted_by_its_own_sessions_not_the_grids():
    """
    The vroc_10 landmine inverted. The wide frame's index is the UNION of every
    ticker's dates, so shifting it by h steps h rows of that union — which is
    MORE than h sessions for any name with a hole, and the label then silently
    measures a longer horizon for exactly the names whose history is most
    irregular.
    """
    panel = _price_panel(ragged=True)
    out = panel_at(panel, FIVE)

    gapped = out[out["ticker"] == "T00.NS"].sort_values("date").reset_index(drop=True)
    lp = np.log(gapped["close"].to_numpy(dtype=float))
    got = pd.to_numeric(gapped[TARGET], errors="coerce").to_numpy(dtype=float)

    # Its own sessions are what count, gap included: row i is scored against
    # row i+5 OF THIS TICKER, not against whatever the union grid holds there.
    for i in range(len(gapped) - FIVE):
        assert abs(got[i] - (lp[i + FIVE] - lp[i])) < 1e-12, (
            f"row {i} of the gapped ticker was stepped across the shared grid")

    clean = out[out["ticker"] == "T01.NS"]
    assert pd.to_numeric(clean[TARGET], errors="coerce").notna().sum() == \
        len(clean) - FIVE


@pytest.mark.parametrize("horizon", HORIZONS)
def test_relabelling_costs_exactly_h_labels_per_ticker(horizon):
    """A shorter horizon labels MORE rows, and by a known amount. This is what
    makes the OOS row counts across the grid readable rather than mysterious."""
    panel = _price_panel()
    out = panel_at(panel, horizon)
    per_ticker = out.groupby("ticker")[TARGET].apply(lambda s: s.notna().sum())
    sizes = panel.groupby("ticker").size()
    assert (per_ticker == sizes - horizon).all()


# ── the books ─────────────────────────────────────────────────────────────────


def _pred_frame(n_dates: int = 240, n_tickers: int = 30, seed: int = 5):
    rng = np.random.default_rng(seed)
    dates = np.repeat([f"d{i:04d}" for i in range(n_dates)], n_tickers)
    tickers = np.tile([f"T{j:02d}.NS" for j in range(n_tickers)], n_dates)
    n = dates.size
    return pd.DataFrame({"date": dates, "ticker": tickers,
                         "y_pred": rng.normal(size=n),
                         "y_true": rng.normal(size=n) * 0.05})


@pytest.mark.parametrize("horizon", HORIZONS)
def test_the_book_rebalances_at_the_horizon_so_its_windows_do_not_overlap(horizon):
    """
    `rebalance_every` must track h. Left at 30 while the label is 5, successive
    books would hold windows overlapping by 25 of 30 sessions, and the plain t
    on their difference — which R4 takes — would inflate by roughly sqrt(6).
    """
    preds = _pred_frame()
    b = book(preds, horizon)
    grid = sorted(preds["date"].unique())
    pos = [grid.index(d) for d in b["date"]]

    assert len(b) == len(grid[::horizon]), (
        f"expected {len(grid[::horizon])} rebalances at h={horizon}, got {len(b)}")
    assert all(y - x == horizon for x, y in zip(pos, pos[1:])), (
        "successive rebalances are not h grid-dates apart, so their return "
        "windows overlap")


def test_a_shorter_horizon_buys_more_independent_windows():
    """The whole reason P6 sweeps downward, pinned so a regression in the
    rebalance rule shows up as a count rather than as a quiet t-statistic."""
    preds = _pred_frame()
    counts = {h: len(book(preds, h)) for h in HORIZONS}
    assert counts[5] > counts[10] > counts[20] > counts[30]
    assert counts[5] == pytest.approx(counts[30] * 6, rel=0.05)


def test_the_net_of_cost_comparison_charges_the_round_trip_on_real_turnover():
    """R4 must be a cost difference, not a cost assumption: an arm that holds
    the same names as the baseline pays the same, and one that churns pays
    more."""
    preds = _pred_frame()
    same = preds.copy()
    churned = preds.copy()
    rng = np.random.default_rng(11)
    churned["y_pred"] = rng.normal(size=len(churned))

    flat = net_of_cost(preds, same, FIVE)
    assert flat["diff"] == pytest.approx(0.0, abs=1e-12)
    assert flat["turn_a"] == pytest.approx(flat["turn_b"])
    assert flat["round_trip"] > 0

    moved = net_of_cost(preds, churned, FIVE)
    assert moved["net_b"] < moved["gross_b"], "a book with turnover paid nothing"
    assert moved["n_rebalances"] == flat["n_rebalances"]


@pytest.mark.parametrize("horizon", HORIZONS)
def test_cell_metrics_counts_rebalances_at_the_horizon_it_is_scoring(horizon):
    """
    `reb_t` is the only t in the sweep table that claims to rest on
    NON-OVERLAPPING windows, and `cell_metrics` is where it is formed. Left at
    the module's 30 while the label is 5 it would report 63 windows instead of
    387 — a number that renders perfectly, is wrong by 6x, and would be read
    across the grid as a horizon effect.

    Caught nothing until it was written: the first pass of this file tested
    `book()` and left `cell_metrics` covered only by proxy, and the mutant
    reverting it survived.
    """
    from tools.stage2b_pooled import cell_metrics

    preds = _pred_frame().assign(fold=0)
    grid = sorted(preds["date"].unique())

    m = cell_metrics(preds, rebalance_every=horizon)
    assert m["n_rebalances"] == len(grid[::horizon]), (
        f"h={horizon}: scored {m['n_rebalances']} rebalances against "
        f"{len(grid[::horizon])} non-overlapping windows on this grid")

    # And the default is still the 30 every pre-P6 table was built on.
    assert cell_metrics(preds)["n_rebalances"] == len(grid[::30])


# ── annualisation ─────────────────────────────────────────────────────────────

def test_the_sharpe_is_annualised_from_this_horizons_own_rebalance_count():
    """
    `portfolio.REBALANCES_PER_YEAR` is a module constant pinned at 252/30, and
    the 2026-09-05 pre-registration flagged exactly this: annualising an h=5
    book through it understates the drag six-fold. P6 does not fix the constant
    — it is production code — so this test is the thing standing between the
    report and that number.
    """
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.005, 0.02])
    per = float(r.mean() / r.std(ddof=1))

    for h in HORIZONS:
        assert _sharpe(r, h) == pytest.approx(per * np.sqrt(252.0 / h))

    assert _sharpe(r, 30) == pytest.approx(per * np.sqrt(REBALANCES_PER_YEAR))
    assert _sharpe(r, 5) > _sharpe(r, 30)
    # The mutant that matters: the constant used at every horizon.
    assert _sharpe(r, 5) != pytest.approx(per * np.sqrt(REBALANCES_PER_YEAR))


def test_a_degenerate_series_earns_no_statistic():
    assert np.isnan(_t([0.01, 0.01, 0.01, 0.01]))
    assert np.isnan(_t([0.01, 0.02]))
    assert np.isnan(_sharpe([0.01, 0.01, 0.01], 5))


# ── the economic bar ──────────────────────────────────────────────────────────

def test_the_spread_per_unit_of_ic_is_re_estimated_at_each_horizon():
    """
    P4's `break_even_ic` takes spread-per-IC as an argument and says it must be
    estimated from the data. It is also not transferable across horizons: a
    5-session return is smaller than a 30-session one, so the same ordering
    buys a smaller spread and the break-even rises from both directions at
    once.
    """
    rng = np.random.default_rng(17)
    n_dates, n_tickers = 400, 40
    dates = np.repeat([f"d{i:04d}" for i in range(n_dates)], n_tickers)
    tickers = np.tile([f"T{j:02d}.NS" for j in range(n_tickers)], n_dates)
    # A random walk, so an h-session return scales with sqrt(h) by construction.
    steps = rng.normal(0, 0.01, size=(n_dates, n_tickers))
    cum = np.cumsum(steps, axis=0)

    per_h = {}
    for h in (5, 30):
        y = np.full((n_dates, n_tickers), np.nan)
        y[:-h] = cum[h:] - cum[:-h]
        frame = pd.DataFrame({"date": dates, "ticker": tickers,
                              "y_true": y.reshape(-1)}).dropna()
        per_h[h] = spread_per_ic(frame, h, seed=1)

    assert all(np.isfinite(v) and v > 0 for v in per_h.values())
    assert per_h[5] < per_h[30], (
        f"spread per unit of IC did not shrink with the horizon "
        f"({per_h}); the h=5 break-even would then be understated")


# ── the verdicts ──────────────────────────────────────────────────────────────

def _fake_b(sue_t: float, sue_diff: float, placebo_max: float,
            r4_diff: float, timing_diff: float) -> dict:
    return {
        "pairs": {
            "sue_vs_baseline": {"diff": sue_diff, "t": sue_t},
            "delivery_vs_baseline": {"diff": sue_diff, "t": sue_t},
            "sue_vs_timing": {"diff": timing_diff, "t": timing_diff * 100},
        },
        "placebo": {"sue": [{"diff": placebo_max}], "delivery": [{"diff": placebo_max}]},
        "r4": {"sue": {"diff": r4_diff}, "delivery": {"diff": r4_diff}},
    }


def test_r5_is_applied_to_the_sue_arm_and_withheld_from_delivery():
    """
    Pilot 3's landmine: `sue_age` and `sue_missing` persist at +0.40 and +0.47
    within-date rank and reproduced every grade the SUE arm moved, so the
    surprise needs an arm carrying its auxiliary columns WITHOUT it. Delivery's
    two columns are one construction and Pilot 2 already ran the level arm that
    plays that role, so applying R5 there would be inventing a rule.
    """
    v = verdicts_b(_fake_b(sue_t=3.0, sue_diff=0.02, placebo_max=0.01,
                           r4_diff=0.001, timing_diff=-0.004))
    assert v["sue"]["r5"] is False
    assert v["delivery"]["r5"] is None
    assert R5_ARMS == ("sue",)

    # R5 failing is enough on its own to deny the SUE arm SIGNAL...
    assert v["sue"]["signal"] is False
    assert any("R5" in f for f in v["sue"]["failed"])
    # ...while delivery, which has no R5, clears on the other three.
    assert v["delivery"]["signal"] is True


@pytest.mark.parametrize("rule,kwargs", [
    ("R3", dict(sue_t=1.9, sue_diff=0.02, placebo_max=0.01, r4_diff=0.001,
                timing_diff=0.004)),
    ("R2", dict(sue_t=3.0, sue_diff=0.02, placebo_max=0.05, r4_diff=0.001,
                timing_diff=0.004)),
    ("R4", dict(sue_t=3.0, sue_diff=0.02, placebo_max=0.01, r4_diff=-0.001,
                timing_diff=0.004)),
    ("R5", dict(sue_t=3.0, sue_diff=0.02, placebo_max=0.01, r4_diff=0.001,
                timing_diff=-0.004)),
])
def test_any_one_failing_rule_denies_signal(rule, kwargs):
    v = verdicts_b(_fake_b(**kwargs))["sue"]
    assert v["signal"] is False
    assert any(rule in f for f in v["failed"]), v["failed"]


def test_all_four_rules_passing_is_signal():
    v = verdicts_b(_fake_b(sue_t=3.0, sue_diff=0.02, placebo_max=0.01,
                           r4_diff=0.001, timing_diff=0.004))["sue"]
    assert v == {"r2": True, "r3": True, "r4": True, "r5": True,
                 "signal": True, "failed": [], "placebo_max": 0.01}


def test_r3_is_a_signed_threshold_not_a_magnitude():
    """A strongly NEGATIVE paired t is a measurement, not a pass. Pilot 3's R5
    came back at t -2.81 and an abs() there would have called it signal."""
    v = verdicts_b(_fake_b(sue_t=-3.0, sue_diff=-0.02, placebo_max=-0.05,
                           r4_diff=0.001, timing_diff=0.004))["sue"]
    assert v["r3"] is False


# ── the arms ──────────────────────────────────────────────────────────────────

def test_the_arms_differ_only_in_the_columns_under_test():
    """Each arm is FACTORS plus its own columns, and the timing arm is the SUE
    arm minus exactly the surprise — which is what makes R5 a one-column
    comparison rather than a two-model one."""
    from pipeline.baselines import FACTORS
    from pipeline.earnings import SUE_AGE, SUE_EVT, SUE_MISSING

    base = set(ARM_FEATURES["baseline"])
    assert base == set(FACTORS)
    assert set(ARM_FEATURES["sue"]) - base == {SUE_EVT, SUE_AGE, SUE_MISSING}
    assert set(ARM_FEATURES["timing"]) - base == {SUE_AGE, SUE_MISSING}
    assert set(ARM_FEATURES["sue"]) - set(ARM_FEATURES["timing"]) == {SUE_EVT}
    assert set(ARM_FEATURES["delivery"]) - base == set(PLACEBO_COLS["delivery"])

    # The placebo permutes exactly the columns the arm added, and no others.
    for arm in ("sue", "delivery"):
        assert set(PLACEBO_COLS[arm]) == set(ARM_FEATURES[arm]) - base


# ── Part C: the scale diagnostic ──────────────────────────────────────────────

def test_standardising_the_target_preserves_every_within_date_ranking():
    """
    The diagnostic only means anything if refitting on the standardised label
    changes the LOSS SCALE and nothing about what a correct answer is. A
    within-date z-score is a positive affine map inside each date, so every
    per-date rank IC of a fixed ordering is untouched — which is what lets the
    refitted model be scored against the REAL label without a second argument
    about comparability.
    """
    from pipeline.evaluation import rank_ic
    from tools.p6_horizon import standardise_target

    rng = np.random.default_rng(23)
    n_dates, n_tickers = 40, 25
    panel = pd.DataFrame({
        "date": np.repeat([f"d{i:03d}" for i in range(n_dates)], n_tickers),
        "ticker": np.tile([f"T{j:02d}.NS" for j in range(n_tickers)], n_dates),
    })
    # Deliberately different scale and centre per date, which is exactly what
    # a horizon change does to the whole panel at once.
    scale = np.repeat(rng.uniform(0.01, 0.3, n_dates), n_tickers)
    centre = np.repeat(rng.normal(0, 0.05, n_dates), n_tickers)
    panel[TARGET] = centre + scale * rng.normal(size=len(panel))
    pred = rng.normal(size=len(panel))

    std = standardise_target(panel)
    for d in panel["date"].unique():
        m = (panel["date"] == d).to_numpy()
        raw_ic = rank_ic(panel[TARGET].to_numpy()[m], pred[m])
        std_ic = rank_ic(std[TARGET].to_numpy()[m], pred[m])
        assert raw_ic == pytest.approx(std_ic), (
            f"date {d}: standardising moved the rank IC {raw_ic} -> {std_ic}")

    # It is a within-DATE operation, so each date is centred on its own.
    means = std.groupby("date")[TARGET].mean()
    sds = std.groupby("date")[TARGET].std()
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(sds.to_numpy(), 1.0, atol=1e-12)

    # And the scale it removes is real: the raw label's per-date sd varies by
    # more than an order of magnitude here, which is the confound Part C
    # exists to remove.
    raw_sds = panel.groupby("date")[TARGET].std()
    assert raw_sds.max() / raw_sds.min() > 5


def test_standardising_a_degenerate_date_does_not_invent_an_ordering():
    """A date whose label is constant has no ordering, and dividing by a zero
    standard deviation must produce NaN rather than a fabricated one."""
    from tools.p6_horizon import standardise_target

    panel = pd.DataFrame({
        "date": ["d0"] * 4 + ["d1"] * 4,
        "ticker": [f"T{j}" for j in range(4)] * 2,
        TARGET: [0.1, 0.1, 0.1, 0.1, -0.02, 0.01, 0.03, 0.05],
    })
    out = standardise_target(panel)
    flat = out[out["date"] == "d0"][TARGET]
    assert flat.isna().all(), "a constant date was given a fabricated ordering"
    assert out[out["date"] == "d1"][TARGET].notna().all()
