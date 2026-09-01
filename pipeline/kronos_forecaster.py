"""
pipeline/kronos_forecaster.py — Kronos, the finance-pretrained comparator.

The third foundation model, and the first one pretrained on financial data
specifically: 12 billion K-line records from 45 global exchanges, against
Chronos-2's and TimesFM-2.5's general time-series corpora. That difference is
the whole reason to spend compute on it. Two general models already agree that
30-session excess return on this panel is not forecastable zero-shot; a model
trained on candlesticks is the one remaining zero-shot argument that the corpus,
rather than the target, was the limitation.

Four things make it materially unlike the two comparators already in the table,
and each one is a place a plausible-looking wrong answer could come from.

────────────────────────────────────────────────────────────────────────────
1. IT IS MULTIVARIATE, AND OUR TARGET IS A UNIVARIATE RATIO.
────────────────────────────────────────────────────────────────────────────

``pipeline.series`` rests on the relative-price identity: the 30-session forward
difference of ``log(close / benchmark_close)`` IS ``target_excess_return``,
exactly, measured at 2.7e-15 on real NSE data. A univariate model handed that
one series predicts the target directly instead of forecasting the stock and
the index separately and differencing two independent errors.

Kronos wants ``[open, high, low, close, volume]``. The reconciliation is a
SYNTHETIC RELATIVE CANDLESTICK — divide all four prices by the SAME-DAY
benchmark close:

    rel_open = open / benchmark_close   ...   rel_close = close / benchmark_close

``rel_close`` is then exactly the identity's series, so the target is untouched.
Dividing all four by one positive scalar is monotone, so
``high >= max(open, close) >= min(open, close) >= low`` still holds and the bar
stays well-formed. Volume passes through raw — it is the stock's own, an
observation at t, and Kronos normalises each series column-wise anyway.

Two alternatives were rejected. Degenerate bars (``open = high = low = close``)
are far off the pretraining distribution, and a null from them would be
uninterpretable — indistinguishable from the model having nothing to say.
Forecasting the stock and the benchmark separately and differencing is the exact
thing the identity exists to avoid.

────────────────────────────────────────────────────────────────────────────
2. IT SAMPLES. THERE IS NO GREEDY PATH.
────────────────────────────────────────────────────────────────────────────

``sample_from_logits`` is called with ``sample_logits=True`` hardcoded at both
call sites, so every forecast is a draw from ``torch.multinomial``. Chronos and
TimesFM are deterministic forward passes; this is not, and two consequences
follow that do not apply to them.

A single ``reb_t`` is ONE DRAW. Quoting it as the result would repeat the
valuation error in a new form — there the number moved with an arbitrary
``min_train``; here it moves with the seed, at a fixed setting. ``seed`` is
therefore explicit and required, and ``tools/run_kronos.py`` sweeps it.

``sample_count`` averages N sampled paths in PRICE space
(``preds.mean(axis=1)`` after decode). Averaging shrinks toward the conditional
mean, and on a near-random-walk target MAE rewards shrinkage directly with no
extra information — that is already measured here, it is why TimesFM beat both
Chronos variants on MAE while carrying the rank IC closest to zero. So MAE
across ``sample_count`` is confounded BY CONSTRUCTION. Read ``reb_IC``, which is
scale-invariant and does not move under shrinkage.

────────────────────────────────────────────────────────────────────────────
3. CONTEXT CAPS AT 512 ABOVE 4.1M PARAMETERS.
────────────────────────────────────────────────────────────────────────────

    Kronos-mini    Kronos-Tokenizer-2k     2048     4.1M
    Kronos-small   Kronos-Tokenizer-base    512    24.7M
    Kronos-base    Kronos-Tokenizer-base    512   102.3M

512 is exactly the context at which everything in this project is null:
``chronos2`` scored reb_t +1.86 at 2048 and +0.88 at 512 on the same rows. So
``Kronos-base`` enters at the setting we already know kills the signal and
CANNOT be given more, while ``Kronos-mini`` reaches 2048 at 4.1M parameters.
Neither alone is interpretable, which is why both are scored: it is the same
context-versus-architecture separation the 120M Chronos swap was made to enable.

The tokenizer pairing is not a preference. Each checkpoint was trained against
one, so ``TOKENIZERS`` maps them and ``load_predictor`` refuses an unknown
model id rather than guessing — a mismatched tokenizer decodes to numbers, not
to an error.

────────────────────────────────────────────────────────────────────────────
4. CONTAMINATION IS A FIRST-ORDER THREAT, AND THE AUTHORS SAY SO.
────────────────────────────────────────────────────────────────────────────

Kronos was pretrained on 45 exchanges and published 2 Aug 2025, so most of this
panel sits inside its training window; the project page warns in as many words
that the model has likely seen the historical periods a quant team would want to
evaluate on. Read against this project's own history that is a sharp warning:
BOTH prior positive results here — valuation at +3.32 and LoRA at +2.37 — were
carried entirely by the EARLIEST folds. A Kronos result concentrated in early
folds is therefore presumed memorisation until the most recent fold says
otherwise, and ``tools/run_kronos.py`` prints the per-fold breakdown for that
reason rather than as a nicety.

One partial mitigation falls out of the design above: what Kronos could have
memorised is PRICES, and what it is shown here is ``price / benchmark_close``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.series import (
    DEFAULT_CONTEXT,
    MIN_CONTEXT,
    _block_ending_at,
    configure_determinism,
    resolve_device,
)

logger = logging.getLogger(__name__)

# Checkpoint -> the tokenizer it was trained against, and its context cap.
# Derived rather than assumed: a mismatched tokenizer produces numbers, not an
# error, and a context above the cap is silently truncated by the predictor's
# own buffer rather than refused.
TOKENIZERS: dict[str, tuple[str, int]] = {
    "NeoQuasar/Kronos-mini":  ("NeoQuasar/Kronos-Tokenizer-2k",   2048),
    "NeoQuasar/Kronos-small": ("NeoQuasar/Kronos-Tokenizer-base",  512),
    "NeoQuasar/Kronos-base":  ("NeoQuasar/Kronos-Tokenizer-base",  512),
}

DEFAULT_MODEL = "NeoQuasar/Kronos-base"

# The upstream commit `pipeline/vendor/kronos/` was taken from. Named here so
# the Kaggle notebook can pin its own clone to the same source and so a drift
# between the two is a checkable fact rather than an assumption. See VENDOR.md.
VENDORED_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"

# The five columns handed to the model. `amount` is deliberately NOT supplied:
# the predictor derives it as volume * mean(prices) when absent, and supplying
# our own would be inventing a turnover figure we do not have.
PRICE_COLS = ["open", "high", "low", "close"]
INPUT_COLS = PRICE_COLS + ["volume"]

_CACHE: dict[tuple, object] = {}


def load_predictor(model_id: str = DEFAULT_MODEL, context: int = 512,
                   device: str | None = None):
    """
    A cached ``KronosPredictor``.

    Keyed by (model, tokenizer, context, device). Device is in the key for the
    reason recorded in CLAUDE.md: a model-only key hands a CPU-resident model to
    a caller that asked for CUDA and the run is merely slow, with nothing in the
    output to see. Context is in the key because ``max_context`` is a constructor
    argument here, not a per-call one — two contexts are two predictors.

    torch is imported INSIDE this function. `pipeline.baselines` imports this
    module lazily and the weekly job and the API both import `baselines`; torch
    must not reach either. See the landmine in CLAUDE.md.
    """
    if model_id not in TOKENIZERS:
        raise ValueError(
            f"unknown Kronos checkpoint {model_id!r}. Known: "
            f"{sorted(TOKENIZERS)}. Each was trained against a specific "
            f"tokenizer and a mismatched pairing decodes to plausible numbers "
            f"rather than to an error, so this refuses rather than guessing."
        )

    tokenizer_id, cap = TOKENIZERS[model_id]
    if context > cap:
        raise ValueError(
            f"{model_id} caps at context {cap}; {context} was requested. The "
            f"predictor would silently truncate to {cap} and the results table "
            f"would claim a context the model never saw."
        )
    if context < MIN_CONTEXT:
        raise ValueError(f"context {context} is below MIN_CONTEXT={MIN_CONTEXT}")

    device = resolve_device(device)
    key = (model_id, tokenizer_id, context, device)
    if key in _CACHE:
        return _CACHE[key]

    configure_determinism()

    from pipeline.vendor.kronos import Kronos, KronosPredictor, KronosTokenizer

    started = time.time()
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    predictor = KronosPredictor(model, tokenizer, device=device,
                                max_context=context)
    logger.info("[Kronos] loaded %s (%s) on %s in %.1fs", model_id,
                tokenizer_id, device, time.time() - started)

    _CACHE[key] = predictor
    return predictor


def relative_candles(ohlcv: pd.DataFrame,
                     benchmark_close: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    The synthetic relative candlestick: every price divided by the same-day
    benchmark close.

    ``ohlcv`` is long — date, ticker, open, high, low, close, volume — and
    ``benchmark_close`` is indexed the same way. Returns a long frame carrying
    the same columns in relative units, with ``volume`` untouched.

    ONE denominator per row, and it must be the SAME day's. Using the previous
    day's close, or the benchmark's own open against the stock's open, would
    break the identity that makes ``rel_close`` the target's series — and it
    would break it by a small amount that no assertion downstream would catch.
    """
    out = ohlcv.copy()
    bench = pd.to_numeric(
        benchmark_close if isinstance(benchmark_close, pd.Series)
        else benchmark_close.iloc[:, 0], errors="coerce")
    bench = bench.reindex(out.index)

    # A non-positive benchmark level is not a price. Dividing by it produces a
    # negative or infinite "candle" that survives every downstream check and
    # standardises to something finite inside the model.
    bench = bench.where(bench > 0)

    for col in PRICE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce") / bench
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    return out


def usable_mask(blocks: dict[str, pd.DataFrame]) -> np.ndarray:
    """
    Which (row, ticker) cells of an aligned slice may be handed to the model.

    JOINT, not per-column. `series._history_ending_at` drops non-finite values
    per ticker, which is right for one series and wrong for five aligned ones:
    a hole in `high` that `low` does not share shifts one column against the
    other and builds a bar from two different sessions, with nothing downstream
    able to detect it.

    Two checks, and neither subsumes the other. Finiteness catches `+inf`,
    which passes `> 0`, and it is the only check on `volume` — a halted session
    legitimately trades zero. Positivity catches a zero or negative price,
    which is finite and is not a price; vendors emit those.

    Shared with `tools/export_kronos_package.py` so the rows shipped to a
    notebook are exactly the rows this forecaster would have scored at home.
    """
    close = blocks["close"]
    usable = np.ones((len(close), len(close.columns)), dtype=bool)
    for col in INPUT_COLS:
        usable &= np.isfinite(blocks[col].to_numpy(dtype=float))

    positive = np.ones_like(usable)
    for col in PRICE_COLS:
        positive &= blocks[col].to_numpy(dtype=float) > 0
    return usable & positive


def candle_frames(relative: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Long relative candles -> one wide ``dates x tickers`` frame per column.

    Wide-per-field is what lets the as-of slice be made ONCE per field by
    ``series._block_ending_at`` — the same function, on the same shared date
    index, at the same position — rather than by a second implementation of the
    boundary. Every frame is reindexed onto the union of dates so the five
    slices are positionally identical by construction.
    """
    frames: dict[str, pd.DataFrame] = {}
    for col in INPUT_COLS:
        wide = relative.pivot_table(index="date", columns="ticker", values=col,
                                    aggfunc="last")
        frames[col] = wide.sort_index()

    dates = frames["close"].index
    tickers = frames["close"].columns
    return {c: f.reindex(index=dates, columns=tickers) for c, f in frames.items()}


def load_candles(panel: pd.DataFrame, engine=None,
                 target: str | None = None) -> dict[str, pd.DataFrame]:
    """
    The five aligned candle frames the multivariate path needs.

    The panel comes from `signals`, which carries `close` and `benchmark_close`
    but not open/high/low — those live in `ohlcv`. So this is the one place the
    two tables meet, and it joins on (date, ticker) with an INNER join.

    WHICH CANDLE IS DECIDED BY THE TARGET, never chosen by the caller, for the
    same reason `panel.price_frame` derives its series from the target name.
    Kronos is scored on the forward move of `close` in whatever basis it was
    handed, so a relative candle scored against an absolute label measures a
    quantity nobody asked for — and it fails silently: the decode succeeds, the
    prices are plausible, and the only symptom is a comparator that will not
    beat the floor.

      absolute target   the RAW candle. `close` is the identity's series and no
                        index is involved, so the synthetic construction below
                        is not merely unnecessary, it would be wrong.
      excess target     the synthetic relative candle, every price over the
                        SAME-DAY benchmark_close.

    The benchmark level is taken from the PANEL, not re-fetched, whenever it is
    needed. `signals` is where `target_excess_return` was computed from, so
    using any other source for the denominator would risk the identity holding
    on one series and not on the one actually scored.
    """
    from sqlalchemy import text

    from data.db import get_engine
    from pipeline.panel import EXCESS_TARGET, TARGET

    target = target or TARGET
    needs_benchmark = target == EXCESS_TARGET

    if needs_benchmark and "benchmark_close" not in panel.columns:
        raise ValueError(
            "the panel carries no benchmark_close, so there is no relative "
            "candle to build. Recompute signals — see panel.price_frame."
        )

    engine = engine or get_engine()
    ohlcv = pd.read_sql(
        text("SELECT date, ticker, open, high, low, close, volume FROM ohlcv"),
        engine)
    ohlcv["date"] = ohlcv["date"].astype(str)

    if not needs_benchmark:
        # Restricted to the panel's own (date, ticker) rows so the scored grid
        # is identical either way. Without this the absolute path would quietly
        # score a WIDER set of rows than the relative one and the two would not
        # be comparable — the confound `data_hash` exists to expose after the
        # fact, reintroduced here before it.
        keys = panel[["date", "ticker"]].copy()
        keys["date"] = keys["date"].astype(str)
        merged = ohlcv.merge(keys, on=["date", "ticker"], how="inner")
        if merged.empty:
            raise ValueError("no (date, ticker) rows are present in BOTH ohlcv "
                             "and the panel; nothing can be scored")
        return candle_frames(merged)

    level = panel[["date", "ticker", "benchmark_close"]].copy()
    level["date"] = level["date"].astype(str)

    merged = ohlcv.merge(level, on=["date", "ticker"], how="inner")
    if merged.empty:
        raise ValueError("no (date, ticker) rows are present in BOTH ohlcv and "
                         "the panel; nothing can be scored")

    relative = relative_candles(merged, merged["benchmark_close"])
    return candle_frames(relative)


@dataclass
class KronosForecaster:
    """
    A ``SeriesForecaster``-shaped comparator over synthetic relative candles.

    It does NOT satisfy the univariate ``SeriesForecaster`` protocol — that one
    receives ``histories: dict[ticker, 1-D array]`` of log relative price, and
    this needs five aligned arrays per ticker. It is driven by
    ``KronosAdapter`` below instead, which is the multivariate sibling of
    ``series.SeriesAdapter`` and reuses its as-of slice rather than its shape.
    """

    model_id: str = DEFAULT_MODEL
    context: int = 512
    seed: int = 0
    sample_count: int = 1
    temperature: float = 1.0
    top_p: float = 0.9
    device: str | None = None

    # How many sampled paths may share one batch. Memory scales with this
    # because the library samples by repeating the input; 2.68 GiB at 1 sample
    # for 90 series at context 512 puts a 6 GB card near its limit at 2.
    max_batch_samples: int = 2

    # Counted rather than logged away: an abstention lands as 0.0, which is
    # exactly the `zero` floor's prediction, so a comparator that quietly
    # declined most of the panel would be indistinguishable from a genuine null.
    skipped_short: int = field(default=0, init=False)
    skipped_bad_forecast: int = field(default=0, init=False)
    dates_scored: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        """
        DERIVED from the checkpoint, never hardcoded.

        Switching checkpoints must rename the row with it, or `experiment_runs`
        records a 102M run under the 4.1M name and the two become
        indistinguishable after the fact — the same reason `CHRONOS_VARIANTS`
        derives its keys.
        """
        short = self.model_id.rsplit("/", 1)[-1].replace("Kronos-", "kronos_")
        return f"{short}@{self.context}"

    def forecast_candles(self, blocks: dict[str, pd.DataFrame],
                         dates: pd.Index, horizon: int) -> dict[str, float]:
        """
        One date's cross-section -> predicted `horizon`-session log excess return.

        ``blocks`` maps each of open/high/low/close/volume to the SAME
        ``rows x tickers`` slice ending at the as-of date. ``dates`` is that
        slice's index, needed because Kronos ingests calendar features.
        """
        import torch

        predictor = load_predictor(self.model_id, self.context, self.device)

        close = blocks["close"]
        tickers = list(close.columns)

        # ── Joint finiteness, not per-column ────────────────────────────────
        #
        # `series._history_ending_at` drops non-finite values per ticker, which
        # is right for one series and WRONG here: if `high` has a hole where
        # `low` does not, dropping independently shifts one column against the
        # other and builds a candle from two different days. Nothing downstream
        # could detect it. So a row is kept only if every column is finite for
        # that ticker, and the columns stay aligned by construction.
        usable = usable_mask(blocks)

        dfs, x_stamps, y_stamps, kept, anchors = [], [], [], [], []
        future = pd.Series(pd.bdate_range(
            pd.Timestamp(dates[-1]) + pd.Timedelta(days=1), periods=horizon))

        for j, ticker in enumerate(tickers):
            mask = usable[:, j]
            # THE FULL CONTEXT, OR NOTHING. `predict_batch` requires every
            # series in the call to share one history length, and truncating
            # the cross-section to its shortest member would silently cut every
            # other ticker's context to whatever the youngest listing has —
            # changing the experiment while the table still says `@512`.
            if mask.sum() < len(close):
                self.skipped_short += 1
                continue

            frame = pd.DataFrame(
                {col: blocks[col].to_numpy(dtype=float)[mask, j]
                 for col in INPUT_COLS})
            dfs.append(frame)
            x_stamps.append(pd.Series(pd.to_datetime(dates[mask])))
            y_stamps.append(future)
            kept.append(ticker)
            anchors.append(float(frame["close"].iloc[-1]))

        if not dfs:
            return {}

        # THE SEED IS PER DATE, NOT PER RUN. Seeding once at construction would
        # make a date's forecast depend on how many dates were scored before it,
        # so re-scoring a subset would not reproduce the whole — and the sweep
        # over seeds could not be compared date by date.
        torch.manual_seed(self.seed + self.dates_scored)
        self.dates_scored += 1

        # SAMPLING IN CHUNKS, AVERAGED IN LOG SPACE.
        #
        # `sample_count` is not a refinement here, it is what makes the output a
        # forecast at all. Chronos and TimesFM return a median quantile from one
        # forward pass; Kronos returns a DRAW from a 30-step autoregressive
        # sampler, and a single draw is a path, not an estimate. Measured on the
        # real panel at sample_count=1: predicted cross-sectional SD 0.192 for
        # base@512 and 0.421 for mini@2048, against a TARGET dispersion of
        # 0.102 — one name at a predicted +247% excess return.
        #
        # But the library expands the BATCH to sample: `x.repeat(1, N, 1, 1)`,
        # so memory scales with N and 2.68 GiB at N=1 caps a 6 GB card around
        # N=2. Chunking keeps memory flat and makes cost linear in time
        # instead, which is a budget rather than a wall.
        #
        # The average is taken in LOG space, on the quantity actually scored.
        # `predict_batch` averages decoded PRICES internally, and the mean of
        # log-prices is not the log of the mean — averaging in price space
        # would apply a Jensen bias that grows with the dispersion, which is
        # exactly the thing being reduced.
        chunks = self._sample_chunks()
        totals = np.zeros(len(dfs), dtype=float)
        for chunk in chunks:
            batch = predictor.predict_batch(
                dfs, x_stamps, y_stamps, pred_len=horizon,
                T=self.temperature, top_p=self.top_p,
                sample_count=chunk, verbose=False)
            terminal = np.array([float(f["close"].iloc[-1]) for f in batch])
            with np.errstate(divide="ignore", invalid="ignore"):
                totals += np.where(terminal > 0, np.log(terminal), np.nan) * chunk
        mean_log = totals / float(self.sample_count)

        out = mean_log

        predictions: dict[str, float] = {}
        for ticker, log_terminal, anchor in zip(kept, out, anchors):
            # THE LAST ROW IS t+horizon, AND THE ANCHOR IS t.
            # `pred_len=horizon` rows are returned, so `iloc[-1]` is the
            # horizon-th session ahead — read above, where the chunk is
            # reduced. An off-by-one there forecasts 29 or 31 sessions against
            # a 30-session label and still renders a full, plausible table.
            if not (np.isfinite(log_terminal) and anchor > 0):
                # Sampling can decode to a non-positive price once the per-series
                # standardisation is undone. log() of that is not a return.
                self.skipped_bad_forecast += 1
                continue
            predictions[ticker] = float(log_terminal - np.log(anchor))

        return predictions

    def _sample_chunks(self) -> list[int]:
        """
        `sample_count` split into batches of at most `max_batch_samples`.

        Sizes are returned rather than a count so the weighted average above
        stays correct when the last chunk is short — a plain mean over chunks
        would silently overweight it.
        """
        if self.sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        cap = max(1, self.max_batch_samples)
        whole, remainder = divmod(self.sample_count, cap)
        return [cap] * whole + ([remainder] if remainder else [])


@dataclass
class KronosAdapter:
    """
    The multivariate sibling of ``series.SeriesAdapter``.

    Same contract with the harness — ``feature_cols=["date", "ticker"]``, a
    no-op ``fit``, one model call per date — and the same as-of guarantee, made
    by the same ``_block_ending_at``. What differs is that five frames are
    sliced instead of one, at identical positions on a shared index.

    ``fit`` is a no-op because this model is zero-shot, so the purged folds
    protect nothing and the entire guarantee collapses onto the slice. See the
    warning at the top of ``pipeline/series.py``; it applies here unchanged and
    with a larger model.
    """

    forecaster: KronosForecaster
    frames: dict[str, pd.DataFrame]
    horizon: int = 30
    context: int = DEFAULT_CONTEXT

    def __post_init__(self) -> None:
        missing = set(INPUT_COLS) - set(self.frames)
        if missing:
            raise ValueError(f"KronosAdapter needs frames for {sorted(missing)}")

        index = self.frames["close"].index
        columns = self.frames["close"].columns
        for col, frame in self.frames.items():
            if not frame.index.equals(index) or not frame.columns.equals(columns):
                raise ValueError(
                    f"frame {col!r} is not aligned with 'close'. The five slices "
                    f"are taken by POSITION on a shared index; misalignment "
                    f"would build a candle from different days without failing."
                )
        if not index.is_monotonic_increasing:
            raise ValueError("frames must be sorted by date")

        self._positions = {d: i for i, d in enumerate(index)}

    @property
    def name(self) -> str:
        return self.forecaster.name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KronosAdapter":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "date" not in X.columns or "ticker" not in X.columns:
            raise ValueError(
                "KronosAdapter needs 'date' and 'ticker' columns; pass "
                "feature_cols=['date', 'ticker'] to panel_walk_forward.")

        out = np.zeros(len(X), dtype=float)
        dates = X["date"].to_numpy()
        tickers = X["ticker"].to_numpy()

        for as_of in pd.unique(dates):
            blocks: dict[str, pd.DataFrame] = {}
            for col in INPUT_COLS:
                block = _block_ending_at(self.frames[col], self._positions,
                                         as_of, self.context)
                if block is None:
                    break
                blocks[col] = block
            if len(blocks) != len(INPUT_COLS):
                continue

            predicted = self.forecaster.forecast_candles(
                blocks, blocks["close"].index, self.horizon)

            for row in np.flatnonzero(dates == as_of):
                value = predicted.get(tickers[row])
                # An abstention predicts no excess return — the same claim the
                # `zero` floor makes, so it cannot flatter this comparator
                # against the floor it is measured against.
                out[row] = (float(value)
                            if value is not None and np.isfinite(value) else 0.0)

        return out


def adapter_factory(frames: dict[str, pd.DataFrame], horizon: int = 30,
                    context: int = 512, **kwargs):
    """A ``model_factory`` for ``panel_walk_forward``. One forecaster per fold."""
    def factory() -> KronosAdapter:
        return KronosAdapter(forecaster=KronosForecaster(context=context, **kwargs),
                             frames=frames, horizon=horizon, context=context)
    return factory


#: The pre-P1 name, when the only candle was the relative one. Kept so a Phase 2
#: script that asked for a RELATIVE candle still gets one after the default
#: moved, rather than silently receiving a raw candle under the old call.
def load_relative_candles(panel: pd.DataFrame, engine=None) -> dict[str, pd.DataFrame]:
    from pipeline.panel import EXCESS_TARGET

    return load_candles(panel, engine=engine, target=EXCESS_TARGET)
