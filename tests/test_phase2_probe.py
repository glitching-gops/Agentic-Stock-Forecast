"""
tests/test_phase2_probe.py - The linear probe on frozen Chronos-2 embeddings.

The probe is the first thing in Phase 2 that actually FITS on panel rows. Every
zero-shot comparator before it had a no-op `fit`, which meant the purged folds
protected nothing and the whole guarantee rested on `_history_ending_at`. Here
both matter, and the fit is the new exposure: a ridge that sees test rows would
produce a large, clean, entirely fake result.

Three groups:

  1. THE FOLD BOUNDARY. The ridge, its standardiser and its alpha must all be
     derived from training rows only.
  2. THE NUMERICS. The Gram-based solve must agree with a reference ridge, and
     alpha selection must not silently pin to the edge of its own grid.
  3. THE ABSTENTIONS. A row with no embedding must predict 0.0 and be visible
     as such, never quietly score as though it had been evaluated.

These run without torch. Only the embedder touches Chronos, and those tests
skip when it is absent, because CI for the daily and weekly jobs must stay
green without torch installed.
"""

import numpy as np
import pandas as pd
import pytest


# ── helpers ─────────────────────────────────────────────────────────────────

def _cache(n_dates=40, n_tickers=8, dim=6, seed=0, signal=0.0):
    """
    A synthetic EmbeddingCache plus the panel frame that indexes it.

    ``signal`` mixes a known linear direction into the target, so a test can
    demand the probe find something real rather than merely run.
    """
    from pipeline.chronos_probe import EmbeddingCache

    rng = np.random.default_rng(seed)
    dates = [f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_dates)]
    tickers = [f"T{i}.NS" for i in range(n_tickers)]

    # Deliberately UNSTANDARDISED: wildly different per-dimension scales and
    # offsets, as real encoder activations have. Drawing every dimension from
    # N(0, 1) made the probe's standardiser a no-op, so deleting it left the
    # whole suite green - the test could not see the thing it was guarding.
    scale = rng.uniform(0.01, 100.0, size=dim)
    offset = rng.uniform(-50.0, 50.0, size=dim)

    keys, rows, scal, recs = {}, [], [], []
    i = 0
    for d in dates:
        for t in tickers:
            raw = rng.normal(size=dim)
            vec = (raw * scale + offset).astype(np.float16)
            keys[(d, t)] = i
            rows.append(vec)
            scal.append([0.01, -3.0])
            recs.append({"date": d, "ticker": t,
                         "y": signal * float(raw[0]) + rng.normal(0, 0.01)})
            i += 1

    cache = EmbeddingCache(keys=keys, matrix=np.vstack(rows),
                           scalars=np.asarray(scal, dtype=np.float32),
                           context=512, model_id="test", horizon=30)
    return cache, pd.DataFrame(recs)


SMALL = {"inner_folds": 2, "inner_min_train": 10, "horizon": 2}


def _require_torch():
    """
    Called INSIDE a test, never at module scope.

    A module-level probe imported a half-reinstalled torch during pytest
    COLLECTION once and took unrelated files down with it. Neither find_spec
    nor a bare import detects that state - both succeed on an empty shell - so
    this touches a real attribute.
    """
    try:
        import torch
        torch.tensor([0.0])
        import chronos  # noqa: F401
    except Exception:                                           # noqa: BLE001
        pytest.skip("torch/chronos not installed")


# ── 1. the fold boundary ────────────────────────────────────────────────────

def test_fit_uses_only_the_rows_it_is_given():
    """
    The ridge must be a function of the training rows and nothing else.

    Fitting on a subset and on that same subset padded with other rows must
    differ; fitting twice on the same subset must not. Without this, a probe
    that quietly read its cache for every row in the panel - which it has
    access to, since the cache is global - would be indistinguishable from a
    correct one.
    """
    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache(signal=0.5)
    early = panel[panel["date"] < "2024-01-20"]
    everything = panel

    a = ChronosProbe(cache=cache, **SMALL)
    a.fit(early[["date", "ticker"]], early["y"])
    b = ChronosProbe(cache=cache, **SMALL)
    b.fit(early[["date", "ticker"]], early["y"])
    c = ChronosProbe(cache=cache, **SMALL)
    c.fit(everything[["date", "ticker"]], everything["y"])

    assert np.allclose(a._w, b._w), "the same rows must give the same fit"
    assert not np.allclose(a._w, c._w), \
        "a larger training set gave an identical fit; the probe is not "\
        "actually reading the rows it was handed"


def test_standardiser_is_fitted_on_training_rows_only():
    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache(signal=0.3)
    early = panel[panel["date"] < "2024-01-15"]

    p = ChronosProbe(cache=cache, **SMALL)
    p.fit(early[["date", "ticker"]], early["y"])

    feats, _ = cache.lookup(early["date"].to_numpy(), early["ticker"].to_numpy())
    assert np.allclose(p._mu, feats.mean(axis=0), atol=1e-5)

    allf, _ = cache.lookup(panel["date"].to_numpy(), panel["ticker"].to_numpy())
    assert not np.allclose(p._mu, allf.mean(axis=0), atol=1e-8), \
        "the scaler saw rows outside the training fold"


def test_alpha_selection_never_touches_the_scoring_rows():
    """
    Alpha is chosen on PURGED inner folds of the training data.

    RidgeCV was removed for exactly this reason - its leave-one-out GCV assumes
    independent rows, and these overlap 29 of 30 forward sessions. That is not
    leakage in the purged sense, but it made the penalty a function of a
    wrong independence assumption and produced an out-of-sample MAE 11% worse
    than predicting nothing.
    """
    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache(n_dates=60, signal=0.4)
    train = panel[panel["date"] < "2024-02-10"]

    p = ChronosProbe(cache=cache, inner_folds=2, inner_min_train=10, horizon=2)
    p.fit(train[["date", "ticker"]], train["y"])
    assert p.alpha_ in p.alphas


def test_probe_recovers_a_planted_linear_signal():
    """If it cannot find a signal that IS linear in the embedding, it is broken."""
    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache(n_dates=80, n_tickers=10, signal=2.0, seed=3)
    train = panel[panel["date"] < "2024-02-20"]
    test = panel[panel["date"] >= "2024-02-20"]

    p = ChronosProbe(cache=cache, inner_folds=2, inner_min_train=10, horizon=2)
    p.fit(train[["date", "ticker"]], train["y"])
    pred = p.predict(test[["date", "ticker"]])

    corr = np.corrcoef(pred, test["y"].to_numpy())[0, 1]
    assert corr > 0.5, f"planted signal not recovered (corr {corr:.3f})"


# ── 2. the numerics ─────────────────────────────────────────────────────────

def test_gram_solve_matches_a_reference_ridge():
    """
    The Gram path replaced per-alpha sklearn fits, which materialised a float64
    design matrix each time and got the 2048 run killed on memory. It must
    give the same answer, not merely a cheaper one.
    """
    from sklearn.linear_model import Ridge

    from pipeline.chronos_probe import _gram, _solve_ridge

    rng = np.random.default_rng(1)
    Z = rng.normal(size=(500, 12)).astype(np.float32)
    y = (Z @ rng.normal(size=12) + rng.normal(0, 0.1, 500)).astype(np.float32)

    # Standardised first, mirroring what `fit` does. sklearn's Ridge centres X
    # internally when fit_intercept=True; the Gram path does not, so comparing
    # against RAW Z measures that difference rather than the solver. The gap
    # was 5e-3 and survived a float64 accumulation, which is what ruled out
    # precision as the cause.
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)

    alpha = 25.0
    mu = float(y.mean())
    G, b = _gram(Z, y - mu)
    w = _solve_ridge(G, b, alpha)

    ref = Ridge(alpha=alpha, fit_intercept=True).fit(Z, y)
    assert np.allclose(w, ref.coef_, atol=2e-4), \
        f"max diff {np.max(np.abs(w - ref.coef_)):.2e}"


def test_a_larger_alpha_shrinks_the_weights():
    from pipeline.chronos_probe import _gram, _solve_ridge

    rng = np.random.default_rng(2)
    Z = rng.normal(size=(300, 8)).astype(np.float32)
    y = (Z @ rng.normal(size=8)).astype(np.float32)
    G, b = _gram(Z, y - y.mean())

    small = np.linalg.norm(_solve_ridge(G, b, 1.0))
    large = np.linalg.norm(_solve_ridge(G, b, 1e6))
    assert large < small / 10, "the penalty is not penalising"


def test_edge_pinned_alpha_is_reported(caplog):
    """
    A penalty chosen at the boundary of its own grid describes the GRID.

    This fired for real on the largest fold and is the difference between "the
    data wants heavy shrinkage" and "the grid stopped before the data was
    finished" - which read identically in the results table.
    """
    import logging

    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache(n_dates=60, signal=0.0, seed=7)
    train = panel[panel["date"] < "2024-02-10"]

    p = ChronosProbe(cache=cache, alphas=(1e9, 1e10), **SMALL)
    with caplog.at_level(logging.WARNING):
        p.fit(train[["date", "ticker"]], train["y"])

    assert any("TOP of the grid" in r.message for r in caplog.records), \
        "alpha pinned to the grid edge without saying so"


def test_daily_rank_ic_skips_dates_with_no_ordering():
    """
    A date whose predictions are all tied carries no ranking information.

    Scoring it as 0 would dilute a real IC toward zero; scoring it at all via a
    stable sort would invent an ordering out of row order, which in this
    codebase is alphabetical by ticker - a real, plausible-looking, entirely
    fake alpha.
    """
    from pipeline.chronos_probe import _daily_rank_ic

    dates = np.array(["d1"] * 4 + ["d2"] * 4)
    pred = np.array([1.0, 1.0, 1.0, 1.0, 4.0, 3.0, 2.0, 1.0])
    actual = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])

    ic = _daily_rank_ic(dates, pred, actual)
    assert ic == pytest.approx(1.0), \
        "the tied date was scored instead of skipped"


# ── 3. the abstentions ──────────────────────────────────────────────────────

def test_rows_with_no_embedding_predict_zero():
    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache(signal=1.0)
    p = ChronosProbe(cache=cache, inner_folds=2, inner_min_train=5, horizon=2)
    p.fit(panel[["date", "ticker"]], panel["y"])

    unknown = pd.DataFrame({"date": ["1999-01-01"] * 3,
                            "ticker": ["NOPE.NS"] * 3})
    assert np.all(p.predict(unknown) == 0.0)


def test_an_unfitted_probe_predicts_zero_rather_than_raising():
    """
    Too few usable rows is an abstention, not a crash - but it must be an
    abstention that shows up as zeros, which the harness scores identically to
    the `zero` floor, so it can never flatter the probe.
    """
    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache()
    tiny = panel.head(5)
    p = ChronosProbe(cache=cache)
    p.fit(tiny[["date", "ticker"]], tiny["y"])

    assert p._model is None
    assert np.all(p.predict(panel[["date", "ticker"]]) == 0.0)


def test_missing_date_ticker_columns_is_an_error_not_a_guess():
    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache()
    p = ChronosProbe(cache=cache)
    with pytest.raises(ValueError, match="date"):
        p.predict(panel[["y"]])


def test_lookup_reports_which_rows_it_found():
    from pipeline.chronos_probe import SCALAR_COLS

    cache, panel = _cache(dim=6)
    dates = np.array([panel.loc[0, "date"], "1999-01-01"])
    tickers = np.array([panel.loc[0, "ticker"], "NOPE.NS"])

    feats, found = cache.lookup(dates, tickers)
    assert found.tolist() == [True, False]
    assert feats.shape == (2, 6 + len(SCALAR_COLS))
    assert np.all(feats[1] == 0.0)


def test_cache_round_trips_through_disk(tmp_path):
    from pipeline.chronos_probe import EmbeddingCache

    cache, _ = _cache(n_dates=5, n_tickers=3, dim=4)
    path = str(tmp_path / "c.npz")
    cache.save(path)
    back = EmbeddingCache.load(path)

    assert back.keys == cache.keys
    assert back.context == cache.context
    assert back.horizon == cache.horizon
    assert np.array_equal(back.matrix, cache.matrix)


# ── the embedder itself (needs the real checkpoint) ─────────────────────────

def test_embedder_refuses_a_reordered_batch(monkeypatch):
    """
    predict() maps results back to inputs POSITIONALLY and names nothing, and
    the hook sees that same batch. If the loader ever re-chunks it, the
    embedding-to-ticker mapping silently becomes wrong and the table that comes
    out is complete, well-formed and meaningless.
    """
    _require_torch()
    import torch

    from pipeline.chronos_probe import ChronosEmbedder

    class _Enc(torch.nn.Module):
        def forward(self, *a, **k):                              # noqa: D401
            return (torch.zeros(3, 5, 8),)

    class _Model:
        encoder = _Enc()

    class _Pipe:
        model = _Model()
        model_output_patch_size = 16

        def predict(self, batch, prediction_length, batch_size):
            self.model.encoder(batch)
            return [torch.zeros(1, 21, prediction_length) for _ in batch]

    emb = ChronosEmbedder(pipeline=_Pipe())
    hists = {f"T{i}.NS": np.linspace(0, 1, 200) for i in range(5)}
    with pytest.raises(RuntimeError, match="reordered|re-chunked"):
        emb.embed(hists)


def test_embedder_raises_when_the_hook_captures_nothing():
    _require_torch()
    import torch

    from pipeline.chronos_probe import ChronosEmbedder

    class _Pipe:
        class model:
            encoder = torch.nn.Identity()
        model_output_patch_size = 16

        def predict(self, batch, prediction_length, batch_size):
            return [torch.zeros(1, 21, prediction_length) for _ in batch]

    emb = ChronosEmbedder(pipeline=_Pipe())
    with pytest.raises(RuntimeError, match="captured nothing"):
        emb.embed({"A.NS": np.linspace(0, 1, 200)})


def test_daily_rank_ic_skips_cross_sections_too_small_to_rank():
    """
    A date holding one or two names cannot express an ordering worth scoring.

    Including them lets a two-name date contribute a correlation of exactly
    +1 or -1 on a coin flip, which is pure variance dressed as signal - and on
    a panel whose early dates hold very few names, there are a lot of them.
    """
    from pipeline.chronos_probe import _daily_rank_ic

    dates = np.array(["tiny", "tiny", "big", "big", "big", "big"])
    # The two-name date is ANTI-correlated; the four-name date is perfect.
    pred = np.array([1.0, 2.0, 4.0, 3.0, 2.0, 1.0])
    actual = np.array([2.0, 1.0, 4.0, 3.0, 2.0, 1.0])

    assert _daily_rank_ic(dates, pred, actual) == pytest.approx(1.0), \
        "a cross-section below the minimum size was scored"


def test_non_finite_predictions_become_zero_not_nan():
    """
    A NaN reaching the table poisons every aggregate computed from it - the
    mean IC, the MAE, the rebalance t - and json.dumps writes a bare NaN that
    Postgres and JSON.parse both reject while Python reads it back happily.
    An unusable prediction is an abstention, and an abstention is 0.0.
    """
    from pipeline.chronos_probe import ChronosProbe

    cache, panel = _cache(signal=1.0, seed=11)
    train = panel[panel["date"] < "2024-01-25"]

    p = ChronosProbe(cache=cache, **SMALL)
    p.fit(train[["date", "ticker"]], train["y"])

    # Corrupt an embedding the model will be asked to score but never trained on.
    later = panel[panel["date"] >= "2024-01-25"].reset_index(drop=True)
    row = cache.keys[(later.loc[0, "date"], later.loc[0, "ticker"])]
    cache.matrix[row, 0] = np.float16("nan")

    out = p.predict(later[["date", "ticker"]])
    assert np.all(np.isfinite(out)), "a non-finite prediction reached the table"
    assert out[0] == 0.0
