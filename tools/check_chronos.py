"""
tools/check_chronos.py — Is the Chronos-2 comparator actually working?

    python tools/check_chronos.py              # synthetic series, no database
    python tools/check_chronos.py --real       # also run on the stored panel

Answers a specific question: not "does it import" but "is the thing it computes
the thing we think it computes". Seven checks, each with a known answer.

The one that matters most is CALIBRATION and AS-OF.

For a zero-shot model the purged folds protect nothing — purging, the embargo
and the training boundary all constrain what a model is FITTED on, and nothing
here is fitted. The whole guarantee is that the history handed over ends at the
as-of date. An off-by-one there hands the model its own answer and would read
as a breakthrough rather than a bug, so this checks it against the live weights
by corrupting every value after the as-of date and requiring bit-identical
output.

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def synthetic_series(n_dates: int = 900, n_tickers: int = 12, seed: int = 0):
    """log(close / benchmark_close), the exact quantity the adapter is fed."""
    rng = np.random.default_rng(seed)
    dates = [f"D{i:05d}" for i in range(n_dates)]
    bench = 20000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, n_dates)))
    cols = {}
    for i in range(n_tickers):
        close = 500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.016, n_dates)))
        cols[f"T{i:02d}.NS"] = np.log(close / bench)
    return pd.DataFrame(cols, index=dates)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true",
                    help="also run on the stored panel (needs DATABASE_URL and "
                         "a benchmark_close column that has been populated)")
    ap.add_argument("--context", type=int, default=None,
                    help="trailing observations to hand the model")
    args = ap.parse_args()

    print("\n1. DEPENDENCIES")
    try:
        import torch
        from chronos import Chronos2Pipeline                    # noqa: F401
    except ImportError as exc:
        check("torch and chronos-forecasting are installed", False, str(exc))
        print("\n  pip install -r requirements-series.txt")
        return 1
    check("torch and chronos-forecasting are installed", True,
          f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    check("this is a CPU build (a GPU wheel on a CPU box is 2.5GB wasted)",
          "+cpu" in torch.__version__ or torch.cuda.is_available(),
          torch.__version__)

    from pipeline import series as S
    from pipeline.baselines import ZeroForecast
    from pipeline.chronos_forecaster import (
        DEFAULT_MODEL_ID, Chronos2Forecaster, load_pipeline)

    context = args.context or S.DEFAULT_CONTEXT

    print("\n2. THE CHECKPOINT")
    print(f"  loading {DEFAULT_MODEL_ID} ...")
    t0 = time.time()
    pipe = load_pipeline(DEFAULT_MODEL_ID)
    params = sum(p.numel() for p in pipe.model.parameters()) / 1e6
    check("weights load from the local HuggingFace cache", True,
          f"{params:.1f}M params in {time.time() - t0:.1f}s")
    check("the model emits a median quantile", 0.5 in pipe.quantiles,
          f"{len(pipe.quantiles)} quantiles, median at index "
          f"{pipe.quantiles.index(0.5) if 0.5 in pipe.quantiles else '?'}")
    check(f"the model context ({pipe.model_context_length}) covers ours "
          f"({context})", pipe.model_context_length >= context)

    series = synthetic_series()
    if args.real:
        from pipeline.panel import load_panel, relative_price_frame
        panel = load_panel()
        real = relative_price_frame(panel)
        n_ok = int(real.notna().any().sum())
        if n_ok >= 2:
            series = real
            print(f"\n  using the STORED panel: {real.shape[0]:,} dates x "
                  f"{n_ok} tickers with a benchmark level")
        else:
            print(f"\n  stored panel has no benchmark level on any ticker; "
                  f"falling back to a synthetic series. Recompute signals.")

    as_of = series.index[-1]
    rows = pd.DataFrame({"date": as_of, "ticker": list(series.columns)})

    print("\n3. CALIBRATION — a forecaster whose answer is known")
    zero_through_adapter = S.SeriesAdapter(
        forecaster=S.ZeroDrift(), series=series, horizon=30,
        context=context).predict(rows)
    zero_direct = ZeroForecast().predict(rows)
    check("ZeroDrift through the adapter reproduces the `zero` baseline "
          "exactly", np.array_equal(zero_through_adapter, zero_direct),
          "if this fails, nothing below it can be attributed to the model")

    print("\n4. THE AS-OF GUARANTEE — the only thing protecting a zero-shot model")
    adapter = S.SeriesAdapter(forecaster=Chronos2Forecaster(), series=series,
                              horizon=30, context=context)
    mid = series.index[len(series) // 2]
    mid_rows = pd.DataFrame({"date": mid, "ticker": list(series.columns)})
    clean = adapter.predict(mid_rows)

    poisoned = series.copy()
    after = poisoned.index > mid
    poisoned.loc[after, :] = poisoned.loc[after, :] + 99.0
    poisoned_out = S.SeriesAdapter(forecaster=Chronos2Forecaster(),
                                   series=poisoned, horizon=30,
                                   context=context).predict(mid_rows)

    check("corrupting every value AFTER the as-of date changes nothing",
          np.array_equal(clean, poisoned_out),
          f"max drift {np.abs(clean - poisoned_out).max():.3e}")

    print("\n5. DETERMINISM")
    again = S.SeriesAdapter(forecaster=Chronos2Forecaster(), series=series,
                            horizon=30, context=context).predict(mid_rows)
    check("the same input gives the same output", np.array_equal(clean, again))

    print("\n6. SCALE — is it predicting a RETURN or a LEVEL?")
    live = adapter.predict(rows)
    finite = live[np.isfinite(live)]
    biggest = float(np.abs(finite).max()) if len(finite) else float("nan")
    check("predictions sit on the target's scale, not the series' level",
          biggest < 0.5,
          f"max |prediction| {biggest:.4f}; target_excess_return runs to "
          f"~0.3, the SERIES level to ~-3")
    print(f"       sample: " + ", ".join(
        f"{t} {v:+.4f}" for t, v in list(zip(rows['ticker'], live))[:6]))

    print("\n7. COST")
    n = len(series.columns)
    t0 = time.time()
    adapter.predict(rows)
    per_date = time.time() - t0
    print(f"       {n} tickers x {context} context: {per_date:.2f}s per date "
          f"on {torch.get_num_threads()} threads")
    print(f"       1,900 dates -> {per_date * 1900 / 60:.0f} min "
          f"(a GitHub runner has 2 threads; scale accordingly)")

    failed = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 70}")
    if failed:
        print(f"{len(failed)} of {len(_results)} checks FAILED:")
        for _, name, detail in failed:
            print(f"  - {name}  {detail}")
        return 1
    print(f"All {len(_results)} checks passed. "
          f"{DEFAULT_MODEL_ID} is wired correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
