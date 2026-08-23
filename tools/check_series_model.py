"""
tools/check_series_model.py — Is a foundation-model comparator actually working?

    python tools/check_series_model.py                  # both, synthetic series
    python tools/check_series_model.py --model timesfm
    python tools/check_series_model.py --model chronos --real

Answers a specific question: not "does it import" but "is the thing it computes
the thing we think it computes". Every check has a known answer.

The two that matter most are CALIBRATION and AS-OF.

For a zero-shot model the purged folds protect nothing — purging, the embargo
and the training boundary all constrain what a model is FITTED on, and nothing
here is fitted. The whole guarantee is that the history handed over ends at the
as-of date. An off-by-one there hands the model its own answer and would read
as a breakthrough rather than a bug, so this checks it against the live weights
by corrupting every value after the as-of date and requiring bit-identical
output.

Exit status is 0 only if every check passes.

(Was tools/check_chronos.py. Renamed when TimesFM-2.5 arrived — the checks are
the same and only the loader differs.)
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


def describe_chronos():
    from pipeline.chronos_forecaster import (
        DEFAULT_MODEL_ID, Chronos2Forecaster, load_pipeline)
    pipe = load_pipeline(DEFAULT_MODEL_ID)
    params = sum(p.numel() for p in pipe.model.parameters()) / 1e6
    quantiles = list(pipe.quantiles)
    return {
        "id": DEFAULT_MODEL_ID, "factory": Chronos2Forecaster,
        "params": params, "context": pipe.model_context_length,
        "quantiles": quantiles,
        "median_at": quantiles.index(0.5) if 0.5 in quantiles else None,
        "has_median": 0.5 in quantiles,
    }


def describe_timesfm():
    from pipeline.timesfm_forecaster import (
        DEFAULT_MODEL_ID, TimesFM25Forecaster, load_model, median_index)
    model = load_model(DEFAULT_MODEL_ID)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    quantiles = list(model.config.quantiles)
    return {
        "id": DEFAULT_MODEL_ID, "factory": TimesFM25Forecaster,
        "params": params, "context": int(getattr(model, "context_len", 0) or 0),
        "quantiles": quantiles,
        # Raises if the layout and config.decode_index disagree, which is
        # itself the check.
        "median_at": median_index(model),
        "has_median": 0.5 in quantiles,
    }


LOADERS = {"chronos": describe_chronos, "timesfm": describe_timesfm}


def run_one(which: str, series: pd.DataFrame, context: int) -> None:
    print(f"\n{'=' * 70}\n{which.upper()}\n{'=' * 70}")
    from pipeline import series as S
    from pipeline.baselines import ZeroForecast
    import torch

    print("  loading ...")
    t0 = time.time()
    info = LOADERS[which]()
    check(f"{info['id']} loads from the local HuggingFace cache", True,
          f"{info['params']:.1f}M params in {time.time() - t0:.1f}s")
    check("the model exposes a median quantile", info["has_median"],
          f"{len(info['quantiles'])} quantiles, median read at index "
          f"{info['median_at']}")
    if info["context"]:
        check(f"the model context ({info['context']}) covers ours ({context})",
              info["context"] >= context)

    factory = info["factory"]
    as_of = series.index[-1]
    rows = pd.DataFrame({"date": as_of, "ticker": list(series.columns)})

    # CALIBRATION is model-independent but must be re-run per series: it proves
    # the adapter over THIS frame, which is what the model result is read
    # against.
    zero_through_adapter = S.SeriesAdapter(
        forecaster=S.ZeroDrift(), series=series, horizon=30,
        context=context).predict(rows)
    check("ZeroDrift through the adapter reproduces the `zero` baseline exactly",
          np.array_equal(zero_through_adapter, ZeroForecast().predict(rows)),
          "if this fails, nothing below it can be attributed to the model")

    mid = series.index[len(series) // 2]
    mid_rows = pd.DataFrame({"date": mid, "ticker": list(series.columns)})
    clean = S.SeriesAdapter(forecaster=factory(), series=series, horizon=30,
                            context=context).predict(mid_rows)

    poisoned = series.copy()
    after = poisoned.index > mid
    poisoned.loc[after, :] = poisoned.loc[after, :] + 99.0
    poisoned_out = S.SeriesAdapter(forecaster=factory(), series=poisoned,
                                   horizon=30, context=context).predict(mid_rows)
    check("corrupting every value AFTER the as-of date changes nothing",
          np.array_equal(clean, poisoned_out),
          f"max drift {np.abs(clean - poisoned_out).max():.3e}")

    again = S.SeriesAdapter(forecaster=factory(), series=series, horizon=30,
                            context=context).predict(mid_rows)
    check("the same input gives the same output", np.array_equal(clean, again))

    adapter = S.SeriesAdapter(forecaster=factory(), series=series, horizon=30,
                              context=context)
    t0 = time.time()
    live = adapter.predict(rows)
    per_date = time.time() - t0

    finite = live[np.isfinite(live)]
    biggest = float(np.abs(finite).max()) if len(finite) else float("nan")
    check("predictions sit on the target's scale, not the series' level",
          biggest < 0.5,
          f"max |prediction| {biggest:.4f}; target runs to ~0.3, the SERIES "
          f"level to ~-3")
    print("       sample: " + ", ".join(
        f"{t} {v:+.4f}" for t, v in list(zip(rows["ticker"], live))[:6]))
    where = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
             else f"{torch.get_num_threads()} CPU threads")
    print(f"       cost: {len(series.columns)} tickers x {context} context "
          f"= {per_date:.2f}s per date on {where}")
    print(f"             1,900 dates -> {per_date * 1900 / 60:.0f} min here. "
          f"The workflow runner has 2 CPU threads and no GPU.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["chronos", "timesfm", "both"],
                    default="both")
    ap.add_argument("--real", action="store_true",
                    help="use the stored panel (needs DATABASE_URL and a "
                         "benchmark_close column that has been populated)")
    ap.add_argument("--context", type=int, default=None)
    args = ap.parse_args()

    print("\nDEPENDENCIES")
    try:
        import torch
    except ImportError as exc:
        check("torch is installed", False, str(exc))
        print("\n  pip install -r requirements-series.txt")
        return 1
    check("torch is installed", True,
          f"{torch.__version__}, {torch.get_num_threads()} threads")
    # What this actually checks is that the wheel matches the hardware. A CUDA
    # wheel on a box with no GPU is 2.5GB downloaded for nothing; a CPU wheel
    # on a box with one leaves an 18x speedup unused. Labelling it "this is a
    # CPU build" was wrong the moment CUDA arrived.
    cuda = torch.cuda.is_available()
    check("the torch build matches the hardware",
          ("+cpu" in torch.__version__) or cuda,
          f"{torch.__version__} | cuda {cuda}"
          + (f" | {torch.cuda.get_device_name(0)}" if cuda else ""))

    from pipeline import series as S
    context = args.context or S.DEFAULT_CONTEXT

    series = synthetic_series()
    if args.real:
        from pipeline.panel import load_panel, relative_price_frame
        real = relative_price_frame(load_panel())
        n_ok = int(real.notna().any().sum())
        if n_ok >= 2:
            series = real
            print(f"\n  using the STORED panel: {real.shape[0]:,} dates x "
                  f"{n_ok} tickers with a benchmark level")
        else:
            print("\n  stored panel has no benchmark level on any ticker; "
                  "using a synthetic series instead. Recompute signals.")

    wanted = ["chronos", "timesfm"] if args.model == "both" else [args.model]
    for which in wanted:
        try:
            run_one(which, series, context)
        except ImportError as exc:
            check(f"{which} is available", False, str(exc)[:160])

    failed = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 70}")
    if failed:
        print(f"{len(failed)} of {len(_results)} checks FAILED:")
        for _, name, detail in failed:
            print(f"  - {name}  {detail}")
        return 1
    print(f"All {len(_results)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
