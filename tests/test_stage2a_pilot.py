"""
Stage 2a — the spot-check and pilot harnesses.

These tools decide which cells get measured and which tickers get retuned, and
a selection rule that can be steered by the outcome is the failure mode the
pre-registration exists to prevent. So the selection and stratification rules
are pinned here, along with the degeneracy metrics and the reproduction guard
that makes the "before" column the same measurement Stage 0 recorded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluation import PurgedWalkForward
from pipeline.model import EVAL_MIN_TRAIN, EVAL_N_FOLDS
from pipeline.signals import HORIZON_SESSIONS
from tools.stage2a_gamma_spotcheck import (
    Cell,
    cell_verdict,
    fold_slices,
    mode_share,
    select_cells,
    unique_fraction,
)
from tools.stage2a_pilot import STRATA, stratify, verify_against_cache


# ── the degeneracy metrics ────────────────────────────────────────────────────


def test_mode_share_and_unique_fraction_on_known_arrays():
    constant = np.full(10, 0.037)
    assert mode_share(constant) == 1.0
    assert unique_fraction(constant) == pytest.approx(0.1)

    half = np.array([1.0] * 5 + [2.0, 3.0, 4.0, 5.0, 6.0])
    assert mode_share(half) == pytest.approx(0.5)
    assert unique_fraction(half) == pytest.approx(0.6)


def test_the_metrics_use_exact_equality_not_a_tolerance():
    """
    A tree that makes no splits returns the same float BIT FOR BIT, so exact
    equality is the sharpest available test and a tolerance would only blur it
    — two genuinely different predictions a float apart would be counted as
    one, and a cell that did split would be filed as degenerate.
    """
    values = np.array([0.05, 0.05 + 1e-12, 0.05, 0.05])
    assert mode_share(values) == pytest.approx(0.75)
    assert unique_fraction(values) == pytest.approx(0.5)


# ── the pre-registered cell-selection rule ────────────────────────────────────


def _grid(n_tickers: int = 12, n_folds: int = 5,
          degenerate: set[tuple[int, int]] | None = None) -> list[Cell]:
    names = [f"{chr(ord('A') + i)}.NS" for i in range(n_tickers)]
    cells = []
    for i, name in enumerate(names):
        for k in range(n_folds):
            deg = degenerate is None or (i, k) in degenerate
            cells.append(Cell(name, k, 1.0 if deg else 0.4, 100))
    return cells


def test_a_nearly_constant_cell_is_not_a_fully_constant_one():
    """
    The population is cells at mode-share EXACTLY 1.000.

    The Stage 0 addendum measured 16 cells sitting in (0.90, 1.00) — a real
    population, small but non-empty. A selection rule written with a threshold
    instead of equality would sweep them in, and the "as-tuned prediction is a
    constant" premise that makes the whole spot-check readable would be false
    for those cells. Alphabetically first, so a loose rule would pick it.
    """
    cells = ([Cell("A.NS", 0, 0.995, 100)]
             + [Cell(f"{c}.NS", 0, 1.0, 100) for c in "BCD"])
    chosen = select_cells(cells)
    assert "A.NS" not in {c.ticker for c in chosen}
    assert sorted(c.ticker for c in chosen) == ["B.NS", "C.NS"]


def test_selection_takes_two_per_fold_all_distinct_tickers_alphabetically():
    chosen = select_cells(_grid())
    assert len(chosen) == 10
    assert len({c.ticker for c in chosen}) == 10, "one cell per ticker"
    assert sorted({c.fold for c in chosen}) == [0, 1, 2, 3, 4]
    # Fold 0 must take the alphabetically first two, and no later fold may
    # reuse them.
    fold0 = sorted(c.ticker for c in chosen if c.fold == 0)
    assert fold0 == ["A.NS", "B.NS"]
    assert "A.NS" not in {c.ticker for c in chosen if c.fold != 0}


def test_selection_never_reaches_a_cell_that_was_not_fully_constant():
    """The population is defined by the cached prediction's constancy, which is
    fixed before any refit happens — so nothing the refit shows can move it."""
    only_fold_2 = {(i, 2) for i in range(6)}
    chosen = select_cells(_grid(degenerate=only_fold_2))
    assert {c.fold for c in chosen} == {2}
    assert all(c.mode_share == 1.0 for c in chosen)


def test_selection_returns_fewer_cells_rather_than_reusing_a_ticker():
    """
    Breadth beats count. With only six degenerate tickers the later folds run
    out, and the rule takes the shortfall instead of scoring one ticker twice —
    two cells from the same name are not two independent witnesses.
    """
    chosen = select_cells(_grid(n_tickers=6))
    assert len({c.ticker for c in chosen}) == len(chosen) == 6


def test_selection_is_deterministic():
    cells = _grid()
    assert select_cells(cells) == select_cells(list(reversed(cells)))


# ── the verdict ───────────────────────────────────────────────────────────────


def _rows(ics, mode_shares, gammas=(0.0, 0.5, 1.0, 2.0)):
    return [{"ticker": "X.NS", "fold": 0, "gamma": g, "rank_ic": ic,
             "mode_share": ms, "mae": 0.1}
            for g, ic, ms in zip(gammas, ics, mode_shares)]


def test_a_cell_supports_the_hypothesis_only_when_the_ordering_improves():
    as_tuned = {"as_tuned_rank_ic": 0.20, "as_tuned_mae": 0.1}

    split_and_better = cell_verdict(
        _rows([0.30, 0.25, np.nan, np.nan], [0.2, 0.5, 1.0, 1.0]), as_tuned)
    assert split_and_better["supports_h1"]

    # Splits, but every split ordering is worse than the as-tuned one.
    split_but_worse = cell_verdict(
        _rows([0.05, 0.02, np.nan, np.nan], [0.2, 0.5, 1.0, 1.0]), as_tuned)
    assert not split_but_worse["supports_h1"]

    # Constant everywhere: nothing to improve on.
    never_splits = cell_verdict(
        _rows([np.nan] * 4, [1.0] * 4), as_tuned)
    assert not never_splits["splits_anywhere"]
    assert not never_splits["supports_h1"]


def test_an_undefined_baseline_makes_any_finite_ic_count_as_improvement():
    """
    A known, deliberate weakness, pinned so it cannot be forgotten.

    Every fully-constant cell has an UNDEFINED as-tuned rank IC, so "improves on
    its own as-tuned value" is satisfied by any finite IC — including a NEGATIVE
    one. The verdict is therefore a test of "did it start expressing an
    ordering", not "did it start being right", and the report must carry the
    count of cells whose best IC is actually positive alongside it.
    """
    undefined = {"as_tuned_rank_ic": float("nan"), "as_tuned_mae": 0.1}
    negative = cell_verdict(_rows([-0.30, np.nan, np.nan, np.nan],
                                  [0.2, 1.0, 1.0, 1.0]), undefined)
    assert negative["supports_h1"], "documented behaviour, not an endorsement"
    assert negative["best_rank_ic"] < 0


# ── the harness reproduces evaluate_ticker's folds ────────────────────────────


def test_the_pilot_splits_on_exactly_the_evaluation_protocol():
    """
    Not a reimplementation of the splitter — the splitter itself, at the
    evaluation constants. Asserted against an independently constructed one so
    that a drifting constant fails here rather than silently rescoring folds.
    """
    n = 1200
    reference = list(PurgedWalkForward(
        n_folds=EVAL_N_FOLDS, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=EVAL_MIN_TRAIN).split(n))
    ours = fold_slices(n)
    assert len(ours) == len(reference) == EVAL_N_FOLDS
    for (a_tr, a_te), (b_tr, b_te) in zip(ours, reference):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)


# ── the stratification ────────────────────────────────────────────────────────


def _cells_with(counts: dict[str, int], n_folds: int = 5) -> list[Cell]:
    return [Cell(t, k, 1.0 if k < c else 0.3, 100)
            for t, c in counts.items() for k in range(n_folds)]


def test_strata_are_disjoint_and_cover_the_boundaries():
    counts = {f"T{i:02d}.NS": i % 6 for i in range(24)}
    strata = stratify(_cells_with(counts))

    for name, members in strata.items():
        lo, hi, size = STRATA[name]
        assert len(members) <= size
        for t in members:
            assert lo <= counts[t] <= hi, f"{t} has {counts[t]} constant folds"

    picked = [t for members in strata.values() for t in members]
    assert len(picked) == len(set(picked)), "a ticker may sit in one stratum only"


def test_a_fully_degenerate_ticker_is_never_filed_as_a_control():
    strata = stratify(_cells_with({"A.NS": 5, "B.NS": 0}))
    assert strata["degenerate"] == ["A.NS"]
    assert strata["control"] == ["B.NS"]


def test_stratification_is_alphabetical_and_deterministic():
    counts = {f"{c}.NS": 5 for c in "ZYXWVU"}
    strata = stratify(_cells_with(counts))
    assert strata["degenerate"] == ["U.NS", "V.NS", "W.NS", "X.NS", "Y.NS"]


# ── the reproduction guard ────────────────────────────────────────────────────


def _cache(tmp_path, pred: np.ndarray, truth: np.ndarray, folds: np.ndarray):
    path = tmp_path / "cache.npz"
    np.savez(path, tickers=np.array(["A.NS"], dtype=object),
             offsets=np.array([0, pred.size]), dates=np.arange(pred.size).astype(object),
             y_true=truth, y_pred=pred, fold=folds.astype(np.int32))
    return str(path)


def test_the_before_arm_must_reproduce_the_stage_0_cache(tmp_path):
    truth = np.linspace(-0.1, 0.1, 40)
    pred = np.full(40, 0.02)
    folds = np.repeat([0, 1], 20)
    path = _cache(tmp_path, pred, truth, folds)

    faithful = [{"fold": k,
                 "mae": float(np.mean(np.abs(truth[folds == k] - pred[folds == k]))),
                 "mode_share": 1.0} for k in (0, 1)]
    assert verify_against_cache(faithful, "A.NS", path) == pytest.approx(0.0, abs=1e-12)

    drifted = [dict(r) for r in faithful]
    drifted[1]["mae"] += 1e-4
    assert verify_against_cache(drifted, "A.NS", path) > 1e-9


def test_step_a_refuses_to_sweep_a_cell_it_cannot_reproduce(monkeypatch):
    """
    The stop condition, exercised.

    If the as-tuned refit does not land on the cached prediction, every gamma
    below it is being compared against a baseline that was never measured — so
    the tool must raise rather than print a table. ``tune`` is stubbed out
    because what is under test is the guard, not the search.
    """
    import tools.stage2a_gamma_spotcheck as spot

    n = 700
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=n) for c in spot.FEATURES})
    df[spot.TARGET] = rng.normal(scale=0.1, size=n)

    monkeypatch.setattr(spot, "tune", lambda *a, **kw: {"n_estimators": 5,
                                                        "max_depth": 2,
                                                        "tree_method": "hist"})
    cell = Cell("A.NS", 0, 1.0, 0)

    _, test_idx = fold_slices(n)[0]
    wrong = np.full(len(test_idx), 0.123456)
    with pytest.raises(RuntimeError, match="drifts"):
        spot.sweep_cell(cell, df, wrong)

    too_short = np.full(len(test_idx) - 1, 0.123456)
    with pytest.raises(RuntimeError, match="fold boundary moved"):
        spot.sweep_cell(cell, df, too_short)


def test_the_reproduction_tolerance_is_tight_enough_to_mean_something():
    """A tolerance loose enough to pass a different fit is not a guard."""
    from tools.stage2a_gamma_spotcheck import REPRODUCTION_TOL
    assert REPRODUCTION_TOL <= 1e-9


def test_the_guard_notices_a_constancy_change_even_when_mae_agrees(tmp_path):
    """
    MAE alone cannot separate "the same fit" from "a different fit that happens
    to miss by the same average amount". Constancy is the second, independent
    witness — and it is the quantity this whole stage is about, so a guard blind
    to it would be checking the wrong thing.
    """
    truth = np.linspace(-0.1, 0.1, 40)
    pred = np.full(40, 0.02)
    folds = np.repeat([0, 1], 20)
    path = _cache(tmp_path, pred, truth, folds)

    same_mae_different_shape = [
        {"fold": k,
         "mae": float(np.mean(np.abs(truth[folds == k] - pred[folds == k]))),
         "mode_share": 0.25}
        for k in (0, 1)
    ]
    assert verify_against_cache(same_mae_different_shape, "A.NS", path) > 0.5


# ── shadow discipline ─────────────────────────────────────────────────────────


def test_the_pilot_writes_nothing_that_is_served_or_reused():
    """
    The pilot must not touch the production hyperparameter cache, the
    evaluation persistence, or any table. Asserted at the source because the
    claim is about calls that are ABSENT.
    """
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1]
              / "tools" / "stage2a_pilot.py").read_text(encoding="utf-8")
    for forbidden in ("save_params", "tune_and_cache", "_persist_evaluation",
                      "evaluate_and_persist", "get_engine", "to_sql",
                      "store_grading"):
        assert forbidden not in source, (
            f"tools/stage2a_pilot.py calls {forbidden} — the pilot is sandboxed "
            f"and may write only its report"
        )
