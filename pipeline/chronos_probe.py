"""
pipeline/chronos_probe.py - A linear probe on frozen Chronos-2 embeddings.

Phase 2, the fine-tuning stage. This is the CHEAP question asked before the
expensive one.

Five zero-shot foundation-model configurations have been scored on this panel
and none cleared the pre-registered reb_t > 2. But a zero-shot failure has two
very different explanations, and the results table cannot tell them apart:

  1. The model's REPRESENTATION of the series carries no information about the
     30-session excess return. Fine-tuning is then close to hopeless - LoRA
     adapts a representation, it does not conjure one.
  2. The representation carries information but the model's FORECAST HEAD,
     trained on a generic corpus to minimise a quantile loss over arbitrary
     series, does not express it as a 30-step-ahead relative-price move.

A linear probe separates them. Freeze the model, take the encoder state the
forecast head reads, and fit a ridge from that vector to the actual target. If
the probe finds nothing, explanation 1 holds and the expensive path is not
worth taking - which is a real result, not a failure to try.

WHY THIS IS AFFORDABLE
----------------------
The model is FROZEN, so its embedding for a given (date, ticker) does not
depend on which fold that row lands in. Compute the embeddings once for the
whole panel, cache them, and every fold's ridge fit is then a matter of
seconds. The purged-fold discipline is preserved exactly, because the fold
boundary constrains which rows the RIDGE sees, and the ridge is the only thing
being fitted.

That is also the one thing that must not be got wrong. A zero-shot model is
never fitted, so `SeriesAdapter.fit` is a no-op and the folds protect nothing
but the as-of slice. Here there IS a fit, so the folds are load-bearing again:
`ChronosProbe.fit` must see training rows only, which it does because the
harness hands it exactly those.

HOW THE EMBEDDING IS TAKEN
--------------------------
By a forward hook on the encoder, driven through the ordinary `predict` path
rather than by reimplementing the preprocessing. Chronos-2 left-pads, patches,
normalises per series and appends a [REG] token before the encoder sees
anything; reproducing that by hand would be a second implementation to keep in
step with the first, and any drift between them would show up as a probe that
scores differently from the comparator it is supposed to sit beside.

The captured state has shape (batch, context_patches + 1 + output_patches,
d_model). The last `output_patches` positions are what the forecast head reads.
At a 30-session horizon with a 16-session output patch there are two of them,
and BOTH are kept: sessions 1-16 live in the first, 17-32 in the second, and
discarding either throws away half the model's picture of the horizon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pipeline.chronos_forecaster import DEFAULT_MODEL_ID, load_pipeline
from pipeline.series import MIN_CONTEXT, _history_ending_at, resolve_device

logger = logging.getLogger(__name__)

# Scalar features carried alongside the embedding. Chronos normalises each
# series before the encoder, so the hidden state is scale-free by construction
# while the target is not: a 30-session excess return of 0.07 means something
# different for a name whose relative price moves 0.5% a day and one that moves
# 2%. Without a scale the ridge is asked to predict a magnitude from a vector
# that had its magnitude divided out.
SCALAR_COLS = ["ctx_std", "ctx_last"]

# Rows per float64 block when forming the Gram matrix. Bounds the temporary at
# roughly 250 MB for a 1,538-dimensional embedding.
_GRAM_CHUNK = 20_000


def _scalars(history: np.ndarray) -> tuple[float, float]:
    diffs = np.diff(np.asarray(history, dtype=float))
    sd = float(np.std(diffs)) if diffs.size else 0.0
    return (sd if np.isfinite(sd) else 0.0, float(history[-1]))



def _daily_rank_ic(dates: np.ndarray, pred: np.ndarray,
                   actual: np.ndarray) -> float:
    """
    Mean per-date Spearman correlation.

    Per-date, not pooled: a pooled correlation over every (date, ticker) row at
    once can be moved by knowing which months were good, which is not ordering
    skill. Dates whose predictions are entirely tied carry no ordering and are
    skipped rather than scored - the same rule the main harness applies.
    """
    total, n = 0.0, 0
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 3:
            continue
        pr = pd.Series(pred[m]).rank().to_numpy()
        ar = pd.Series(actual[m]).rank().to_numpy()
        sd = pr.std() * ar.std()
        # Zero spread in either ranking IS the no-ordering case: fully tied
        # values all take the same rank. An explicit `np.all(p == p[0])` test
        # stood here too and was pure duplication - removing it left every test
        # green, which is the same trap the fundamentals work hit. Two guards
        # covering one failure are one untestable guard.
        if sd <= 0:
            continue
        total += float(np.mean((pr - pr.mean()) * (ar - ar.mean())) / sd)
        n += 1
    return total / n if n else float("nan")


def _gram(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ``Z.T @ Z`` and ``Z.T @ y``, the only two things a ridge needs from the data.

    Everything downstream is a 1,538 x 1,538 solve, so the sample size stops
    mattering the moment these exist. That is what makes a ten-value alpha grid
    over three inner folds affordable: thirty ridge fits become three matrix
    products and thirty small solves.

    It is also what makes it FIT IN MEMORY. Refitting sklearn's Ridge per alpha
    materialised a float64 copy of the design matrix each time - on the largest
    fold that is 171,723 x 1,538 x 8 bytes, or 2.1 GB per copy, and the run was
    killed part-way through scoring. The Gram matrix is 19 MB regardless.

    Accumulated in float64 but CHUNKED, which is the point. Casting the whole
    design matrix to float64 is the 2.1 GB allocation that got the 2048 run
    killed; casting 20,000 rows at a time costs 246 MB and gives the same
    answer. Accumulating in float32 instead was measured against a reference
    ridge at 5e-3 on coefficients of order 1 - a 0.3% error that no ranking
    statistic would notice, but there is no reason to accept it when the exact
    version is bounded too.
    """
    n, p = Z.shape
    G = np.zeros((p, p), dtype=np.float64)
    b = np.zeros(p, dtype=np.float64)
    for start in range(0, n, _GRAM_CHUNK):
        block = Z[start:start + _GRAM_CHUNK].astype(np.float64)
        G += block.T @ block
        b += block.T @ y[start:start + _GRAM_CHUNK].astype(np.float64)
    return G, b


def _solve_ridge(G: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge weights for one penalty. y is centred, so there is no intercept."""
    p = G.shape[0]
    return np.linalg.solve(G + alpha * np.eye(p, dtype=np.float64), b)


@dataclass
class ChronosEmbedder:
    """Captures encoder states for one cross-section at a time."""

    model_id: str = DEFAULT_MODEL_ID
    horizon: int = 30
    device: str | None = None
    pipeline: Any = None
    _dim: int | None = field(default=None, init=False, repr=False)

    def _pipe(self):
        if self.pipeline is None:
            self.pipeline = load_pipeline(self.model_id,
                                          device=resolve_device(self.device))
        return self.pipeline

    @property
    def dim(self) -> int | None:
        return self._dim

    def embed(self, histories: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """One cross-section in, one embedding per ticker out."""
        if not histories:
            return {}

        import torch

        pipe = self._pipe()
        # Insertion order is the contract: predict() maps results back to
        # inputs positionally and names nothing, and the hook sees that same
        # batch. See the landmine in CLAUDE.md.
        names = list(histories.keys())
        batch = [np.asarray(histories[n], dtype=float) for n in names]

        captured: list[Any] = []

        def hook(_mod, _inp, out):
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            captured.append(h.detach())

        handle = pipe.model.encoder.register_forward_hook(hook)
        try:
            with torch.no_grad():
                pipe.predict(batch, prediction_length=self.horizon,
                             batch_size=max(len(batch), 1))
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(
                "the encoder hook captured nothing; Chronos-2's internals have "
                "moved and the probe would otherwise silently embed garbage")

        # The FIRST call is the only one conditioned purely on real history.
        # Long-horizon unrolling re-enters the encoder with the model's own
        # generated values appended, and an embedding taken from that is partly
        # a picture of the model's guess rather than of the data.
        first = captured[0]
        if first.shape[0] != len(batch):
            raise RuntimeError(
                f"encoder batch was {first.shape[0]} rows for {len(batch)} "
                f"inputs; the loader reordered or re-chunked the batch, so the "
                f"embedding-to-ticker mapping cannot be trusted")

        n_out = max(1, -(-self.horizon // pipe.model_output_patch_size))
        vec = first[:, -n_out:, :].reshape(first.shape[0], -1)
        arr = vec.to("cpu", dtype=torch.float32).numpy()
        self._dim = int(arr.shape[1])

        return {n: arr[i] for i, n in enumerate(names)}


def build_embedding_cache(series: pd.DataFrame, dates: list[str],
                          context: int, model_id: str = DEFAULT_MODEL_ID,
                          horizon: int = 30, device: str | None = None,
                          on_date=None) -> "EmbeddingCache":
    """
    Embed every (date, ticker) once. Frozen model, so this is fold-independent.

    Stored float16: these are activations feeding a ridge that standardises its
    inputs anyway, and float32 for a full panel is 1.5 GB of RAM for no gain in
    a fifth-decimal-place statistic.
    """
    positions = {d: i for i, d in enumerate(series.index)}
    embedder = ChronosEmbedder(model_id=model_id, horizon=horizon, device=device)

    keys: list[tuple[str, str]] = []
    rows: list[np.ndarray] = []
    scalars: list[tuple[float, float]] = []

    for n, as_of in enumerate(dates):
        histories = _history_ending_at(series, positions, as_of, context)
        if not histories:
            continue
        vectors = embedder.embed(histories)
        for ticker, vec in vectors.items():
            keys.append((str(as_of), ticker))
            rows.append(vec.astype(np.float16))
            scalars.append(_scalars(histories[ticker]))
        if on_date is not None:
            on_date(n + 1, len(dates), len(keys))

    if not rows:
        raise RuntimeError(
            f"no date produced an embedding; the panel may hold fewer than "
            f"MIN_CONTEXT={MIN_CONTEXT} observations per ticker")

    return EmbeddingCache(
        keys={k: i for i, k in enumerate(keys)},
        matrix=np.vstack(rows),
        scalars=np.asarray(scalars, dtype=np.float32),
        context=context, model_id=model_id, horizon=horizon,
    )


@dataclass
class EmbeddingCache:
    keys: dict[tuple[str, str], int]
    matrix: np.ndarray            # (n_rows, d) float16
    scalars: np.ndarray           # (n_rows, len(SCALAR_COLS)) float32
    context: int
    model_id: str
    horizon: int

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) + int(self.scalars.shape[1])

    def lookup(self, dates, tickers) -> tuple[np.ndarray, np.ndarray]:
        """Rows for these (date, ticker) pairs, and a mask of which were found."""
        idx = np.full(len(dates), -1, dtype=np.int64)
        for i, (d, t) in enumerate(zip(dates, tickers)):
            j = self.keys.get((str(d), t))
            if j is not None:
                idx[i] = j
        found = idx >= 0
        out = np.zeros((len(dates), self.dim), dtype=np.float32)
        if found.any():
            take = idx[found]
            out[found] = np.hstack([
                self.matrix[take].astype(np.float32),
                self.scalars[take],
            ])
        return out, found

    def save(self, path: str) -> None:
        ks = np.array([f"{d}\t{t}" for (d, t) in self.keys.keys()], dtype=object)
        np.savez_compressed(
            path, keys=ks, matrix=self.matrix, scalars=self.scalars,
            meta=np.array([self.context, self.horizon], dtype=np.int64),
            model_id=np.array([self.model_id], dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "EmbeddingCache":
        z = np.load(path, allow_pickle=True)
        keys = {}
        for i, s in enumerate(z["keys"]):
            d, t = str(s).split("\t")
            keys[(d, t)] = i
        meta = z["meta"]
        return cls(keys=keys, matrix=z["matrix"], scalars=z["scalars"],
                   context=int(meta[0]), model_id=str(z["model_id"][0]),
                   horizon=int(meta[1]))


@dataclass
class ChronosProbe:
    """
    Ridge from a frozen Chronos-2 embedding to the 30-session excess return.

    Satisfies ``fit``/``predict`` with ``feature_cols=["date", "ticker"]``, so
    it rides the same harness, folds and rows as every other comparator.

    Unlike ``SeriesAdapter`` this one genuinely FITS, which puts the purged
    folds back in charge: the embedding for a row is fixed, but which rows the
    ridge may learn from is not, and that is the harness's job.
    """

    cache: EmbeddingCache
    # Reaches 1e7 because the inner selection genuinely wants a heavy penalty
    # on 1,538 correlated dimensions; a grid that stops at 1e5 would pin the
    # choice to its own upper edge and hide that.
    alphas: tuple = (1.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10)
    # What the inner split optimises. "mse" is the conventional choice and the
    # one ridge itself minimises; "ic" targets cross-sectional ORDERING, which
    # a forecast can get right while being poorly scaled. They disagree here,
    # so both are reported rather than one being chosen quietly.
    select_metric: str = "mse"
    horizon: int = 30
    inner_folds: int = 3
    inner_min_train: int = 250
    name: str = "chronos2_probe"
    alpha_: float | None = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _w: Any = field(default=None, init=False, repr=False)
    _y_mu: float = field(default=0.0, init=False, repr=False)
    _mu: Any = field(default=None, init=False, repr=False)
    _sd: Any = field(default=None, init=False, repr=False)

    @staticmethod
    def _cols(X: pd.DataFrame):
        if "date" not in X.columns or "ticker" not in X.columns:
            raise ValueError(
                "ChronosProbe needs 'date' and 'ticker' columns; pass "
                "feature_cols=['date', 'ticker'] to panel_walk_forward.")
        return X["date"].to_numpy(), X["ticker"].to_numpy()

    def _select_alpha(self, Z: np.ndarray, y: np.ndarray,
                      dates: np.ndarray) -> float:
        """
        Choose the ridge penalty on PURGED, DATE-BLOCKED inner folds.

        Not ``RidgeCV``. Its generalised cross-validation is leave-one-out,
        which assumes independent rows, and these rows are about as dependent
        as rows get: consecutive dates share 29 of their 30 forward sessions,
        and every name in a cross-section moves with the market on the same
        day. LOO therefore reads ~38,000 training rows as ~38,000 independent
        observations when the panel holds perhaps sixty independent windows,
        and answers that almost no regularisation is needed.

        Measured: RidgeCV picked alpha = 1 on the first fold, and the resulting
        probe scored an out-of-sample MAE 11% WORSE than predicting zero. That
        is not the model failing to find signal, it is the penalty being chosen
        by a procedure whose assumption the data violates.

        The inner split reuses the same purged splitter as the outer one, so
        the embargo that protects the outer test fold protects the inner one
        too.
        """
        from pipeline.evaluation import PurgedPanelWalkForward

        inner = PurgedPanelWalkForward(n_folds=self.inner_folds,
                                       horizon=self.horizon,
                                       embargo=self.horizon,
                                       min_train=self.inner_min_train)
        splits = list(inner.split(dates))
        if not splits:
            # Too few dates to purge an inner fold. Take the strongest penalty
            # rather than the weakest: with no way to measure, the choice that
            # cannot manufacture a signal is the honest default.
            logger.warning("[Probe] no inner fold available; using alpha=%g",
                           max(self.alphas))
            return float(max(self.alphas))

        # One Gram per inner fold; every alpha is then a small solve against it.
        prepared = []
        for tr, te in splits:
            y_mu = float(y[tr].mean())
            G, b = _gram(Z[tr], y[tr] - y_mu)
            prepared.append((G, b, y_mu, te))

        best, best_score = None, np.inf
        for alpha in self.alphas:
            scores = []
            for G, b, y_mu, te in prepared:
                w = _solve_ridge(G, b, float(alpha))
                pred = Z[te] @ w.astype(np.float32) + y_mu
                if self.select_metric == "ic":
                    # NEGATED so lower is better, keeping one comparison below.
                    scores.append(-_daily_rank_ic(dates[te], pred, y[te]))
                else:
                    scores.append(float(np.mean((pred - y[te]) ** 2)))
            score = float(np.mean(scores)) if scores else np.inf
            if np.isfinite(score) and score < best_score:
                best, best_score = float(alpha), score

        if best is not None and best >= max(self.alphas) * 0.999:
            # Pinned to the edge means the grid, not the data, chose the
            # penalty - and the number it produced describes the grid.
            logger.warning(
                "[Probe] alpha selection hit the TOP of the grid (%g). The "
                "penalty is grid-limited, so widen it before reading the "
                "result.", best)
        return best if best is not None else float(max(self.alphas))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ChronosProbe":
        dates, tickers = self._cols(X)
        feats, found = self.cache.lookup(dates, tickers)
        target = np.asarray(y, dtype=float)
        usable = found & np.isfinite(target)
        if usable.sum() < 100:
            logger.warning("[Probe] only %d usable training rows; abstaining",
                           int(usable.sum()))
            self._model = None
            return self

        Z = feats[usable]
        # Standardised on the TRAINING fold only. Fitting the scaler on all
        # rows would leak the test fold's distribution into the transform - a
        # small leak, and exactly the kind this project keeps finding.
        self._mu = Z.mean(axis=0)
        self._sd = Z.std(axis=0)
        self._sd[self._sd < 1e-8] = 1.0
        Z = (Z - self._mu) / self._sd
        target = target[usable]

        self.alpha_ = self._select_alpha(
            Z, target, np.asarray(dates, dtype=str)[usable])

        self._y_mu = float(target.mean())
        G, b = _gram(Z, target - self._y_mu)
        self._w = _solve_ridge(G, b, self.alpha_).astype(np.float32)
        self._model = True
        logger.info("[Probe] fold: %d rows, alpha %g", len(target), self.alpha_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dates, tickers = self._cols(X)
        out = np.zeros(len(X), dtype=float)
        if self._model is None:
            return out

        feats, found = self.cache.lookup(dates, tickers)
        if found.any():
            Z = (feats[found] - self._mu) / self._sd
            pred = Z @ self._w + self._y_mu
            out[found] = np.where(np.isfinite(pred), pred, 0.0)
        # A row with no cached embedding predicts no excess return - the same
        # claim the `zero` floor makes, so an abstention cannot flatter the
        # probe relative to what it is measured against.
        return out


def probe_factory(cache: EmbeddingCache, name: str = "chronos2_probe",
                  select_metric: str = "mse"):
    """A ``model_factory`` for ``panel_walk_forward``: fresh ridge per fold."""
    def factory() -> ChronosProbe:
        return ChronosProbe(cache=cache, name=name,
                            select_metric=select_metric)
    return factory
