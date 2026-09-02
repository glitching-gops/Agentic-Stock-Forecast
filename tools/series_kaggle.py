"""
tools/series_kaggle.py - The Kaggle side for Chronos-2 and TimesFM-2.5.

Attach the package from `tools/export_series_package.py` as a Kaggle dataset,
then in a GPU notebook:

    !pip install -q "chronos-forecasting>=2.0" "transformers==5.5.4"
    !python -c "import transformers as t; print(t.__version__, \\
        hasattr(t,'TimesFm2_5ModelForPrediction'))"
    !python series_kaggle.py --package /kaggle/input/<ds>/series_panel.npz \\
        --model chronos --model-id amazon/chronos-2 --context 2048 \\
        --out /kaggle/working/chronos2_2048.npz

TRANSFORMERS IS PINNED, NOT RANGED, AND THAT IS THE FIX FOR A REAL FAILURE
--------------------------------------------------------------------------
TimesFM-2.5 ships inside transformers as `TimesFm2_5ModelForPrediction`, a
class that landed in 5.x. `pip install "transformers>=5.0,<6"` on a Kaggle
image that already carries a 4.x can resolve by leaving it exactly where it is,
and the first thing that then fails is the fourth of five configurations, an
hour of attention later. `preflight()` below now checks this in two seconds.

DO NOT `pip install -r requirements-series.txt` ON KAGGLE
---------------------------------------------------------
That file opens with `--extra-index-url https://download.pytorch.org/whl/cpu`,
which is correct for a GitHub Actions runner and catastrophic here: it
DOWNGRADES Kaggle's CUDA torch to a CPU build, `resolve_device()` then returns
"cpu" without complaint because that is genuinely all torch can see, and the
run is merely slow with nothing anywhere saying why. The two lines above are
the only installs needed; torch is already present.

The device is asserted rather than assumed - see the check in `main`.

HOW THIS DIFFERS FROM kronos_kaggle.py, AND WHY THE DIFFERENCE IS SAFE
----------------------------------------------------------------------
`kronos_kaggle.py` returns raw terminal prices and computes NOTHING, because
Kronos ships no pip package, its notebook had to hand-roll the model call, and
every arithmetic step there has a wrong version that renders a plausible table.

This script does the opposite, and it is a stronger guarantee rather than a
weaker one: the package carries `pipeline/series.py`,
`pipeline/chronos_forecaster.py` and `pipeline/timesfm_forecaster.py` as source
text with a sha256 each. They are written out and imported here, so the window
slice, the median-quantile lookup, the TF32 settings, TimesFM's
`truncate_negative` override and its patch-multiple context rounding are THE
TESTED IMPLEMENTATIONS - not a notebook's copy of them.

That is what the hash check below is for. If it fails, the package and the
repository have diverged and the run must not proceed, because the one thing
worse than a wrong number here is a wrong number that looks comparable.

Both facts that make this affordable: the three modules import only numpy,
pandas and the standard library at module level (torch is imported inside
`load_pipeline`), and `_history_ending_at` takes a DataFrame, which is
reconstructed from the package's own matrix. So the slice this script hands the
model is produced by the same function the local harness uses, on the same
data, at the same as-of index.

THE WINDOW IS STILL GIVEN, NOT DERIVED
--------------------------------------
`row_end` is the position in the shared date index that each row is as-of. For
a zero-shot model the purged folds protect nothing - there is no fit - so that
single slice IS the entire causal guarantee. An off-by-one hands the model its
own answer and would read as a breakthrough.

PARTIAL RUNS
------------
`pred` is initialised to NaN and filled as dates complete. An all-NaN row is
one the run never reached - a wall-clock timeout, a `--limit-dates` smoke test.
`score_series.py` DROPS those rather than filling them with 0.0, because 0.0 is
the `zero` forecast's own claim and a mostly-unreached run would otherwise
report a plausible near-floor null that is 98% the floor's own prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd


def install_sources(package, where: str = "/kaggle/working/_shipped") -> None:
    """Writes the shipped modules out and puts them on the path, hash-checked."""
    sources = json.loads(str(package["sources"][0]))
    os.makedirs(where, exist_ok=True)

    for rel, entry in sources.items():
        text = entry["text"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(
                f"REFUSING TO RUN: {rel} in the package does not match its own "
                f"recorded sha256. The package is corrupt; re-export it."
            )
        path = os.path.join(where, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"  shipped {rel}  sha256 {digest[:12]}")

    init = os.path.join(where, "pipeline", "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
    if where not in sys.path:
        sys.path.insert(0, where)


# The exact version this repository's results were produced against. Pinned in
# the remediation text below rather than left as a range, because a range is
# what failed: `pip install "transformers>=5.0,<6"` on a Kaggle image that
# already carries a 4.x can resolve to leaving it exactly where it is.
KNOWN_GOOD_TRANSFORMERS = "5.5.4"


def preflight(model: str) -> None:
    """
    Fail in two seconds rather than after the package load and a model download.

    THE COST OF NOT DOING THIS IS MEASURED, not hypothetical. A run of five
    configurations reached the fourth before `TimesFm2_5ModelForPrediction`
    turned out to be missing, and the exception it raised then told the reader
    to `pip install -r requirements-series.txt` - which on Kaggle is the WORST
    available advice, because that file opens with the CPU torch index and
    would replace the CUDA build the first three configurations had just used.
    That message is correct locally and in CI, which is exactly why the
    Kaggle-specific one belongs here instead of there.
    """
    import importlib

    if model == "timesfm":
        transformers = importlib.import_module("transformers")
        version = getattr(transformers, "__version__", "unknown")
        has_class = hasattr(transformers, "TimesFm2_5ModelForPrediction")
        print(f"  transformers {version}  "
              f"TimesFm2_5ModelForPrediction: {has_class}")
        if not has_class:
            raise SystemExit(
                f"\nREFUSING TO RUN: transformers {version} does not export "
                f"TimesFm2_5ModelForPrediction.\n"
                f"\nTimesFM-2.5 ships INSIDE transformers and the class landed "
                f"in 5.x. A `>=5.0,<6` range is not enough on a Kaggle image "
                f"that already carries a 4.x - the resolver is free to leave it "
                f"alone. Pin it, then confirm:\n"
                f"\n    !pip install -q --upgrade "
                f"'transformers=={KNOWN_GOOD_TRANSFORMERS}'\n"
                f"    !python -c \"import transformers as t; "
                f"print(t.__version__, hasattr(t,'TimesFm2_5ModelForPrediction'))\"\n"
                f"\nDO NOT install requirements-series.txt to fix this. It "
                f"pins the CPU torch index and would silently replace this "
                f"notebook's CUDA build.\n"
            )
        return

    if model == "chronos":
        try:
            chronos = importlib.import_module("chronos")
        except ImportError as exc:
            raise SystemExit(
                f"\nREFUSING TO RUN: chronos-forecasting is not importable "
                f"({exc}).\n\n    !pip install -q 'chronos-forecasting>=2.0'\n"
                f"\nDO NOT install requirements-series.txt to fix this - it "
                f"pins the CPU torch index.\n"
            ) from exc
        print(f"  chronos-forecasting "
              f"{getattr(chronos, '__version__', 'unknown')}")


def build_forecaster(args):
    if args.model == "chronos":
        from pipeline.chronos_forecaster import Chronos2Forecaster
        return Chronos2Forecaster(
            model_id=args.model_id or "amazon/chronos-2",
            cross_learning=args.cross_learning,
            device=args.device,
        )
    if args.model == "timesfm":
        from pipeline.timesfm_forecaster import TimesFM25Forecaster
        return TimesFM25Forecaster(
            model_id=args.model_id or "google/timesfm-2.5-200m-transformers",
            device=args.device,
        )
    raise SystemExit(f"unknown model {args.model!r}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", choices=["chronos", "timesfm"], required=True)
    ap.add_argument("--model-id", default=None,
                    help="checkpoint. Defaults to the biggest: amazon/chronos-2 "
                         "(120M) or google/timesfm-2.5-200m-transformers (231M)")
    ap.add_argument("--context", type=int, default=2048,
                    help="trailing observations handed to the model. Chronos' "
                         "cost is QUADRATIC in this - measured 14.8x for a 4x "
                         "context - because attention is quadratic in patches.")
    ap.add_argument("--cross-learning", action="store_true",
                    help="Chronos only: condition each forecast on the rest of "
                         "the cross-section. Measured NEGATIVE at both 512 and "
                         "2048 on the excess target; re-measured here, not "
                         "assumed.")
    ap.add_argument("--device", default=None, help="cuda / cpu; autodetected")
    ap.add_argument("--out", default="/kaggle/working/series_predictions.npz")
    ap.add_argument("--limit-dates", type=int, default=None,
                    help="smoke test. Leaves the rest NaN, which the scorer "
                         "drops rather than counting as abstentions.")
    ap.add_argument("--checkpoint-every", type=int, default=5,
                    help="dates between saves. A Kaggle session that hits its "
                         "wall clock keeps everything up to the last save.")
    ap.add_argument("--shipped-dir", default="/kaggle/working/_shipped")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="proceed without CUDA. Off by default: the usual cause "
                         "on Kaggle is a CPU torch wheel installed over the "
                         "CUDA one, which is silent and merely slow.")
    args = ap.parse_args()

    # BEFORE the package load and before any weights are fetched. See preflight.
    preflight(args.model)

    package = np.load(args.package, allow_pickle=True)
    meta = json.loads(str(package["meta"][0]))
    print(f"  package: target {meta['target']}, horizon {meta['horizon']}, "
          f"{meta['n_rebalances']} rebalances, {meta['n_tickers']} tickers")
    print(f"  history available per row: min {meta['avail_min']}, "
          f"median {meta['avail_median']}, max {meta['avail_max']}")
    if args.context > meta["avail_max"]:
        print(f"  NOTE: --context {args.context} exceeds the longest history "
              f"in the package ({meta['avail_max']}); no row can use it all.")

    install_sources(package, args.shipped_dir)

    from pipeline.series import (_history_ending_at, configure_determinism,
                                 resolve_device)

    # ASSERTED, NOT ASSUMED. `requirements-series.txt` pins the CPU torch
    # index, so installing it here replaces Kaggle's CUDA build; the run then
    # completes correctly and slowly with nothing recording why, and the cost
    # table in CLAUDE.md silently stops describing it. This is the one failure
    # on this path that costs a whole GPU session and leaves no trace.
    device = resolve_device(args.device)
    print(f"  device: {device}")
    if device != "cuda" and not args.allow_cpu:
        import torch
        raise SystemExit(
            f"REFUSING TO RUN ON {device.upper()}. torch is "
            f"{torch.__version__}; cuda.is_available() is "
            f"{torch.cuda.is_available()}. If the notebook has a GPU attached, "
            f"a CPU torch wheel has been installed over the CUDA one - do not "
            f"install requirements-series.txt here. Pass --allow-cpu to "
            f"proceed deliberately."
        )
    args.device = device

    # TF32 OFF. Ada and Ampere run float32 matmuls in TF32 by default, which
    # keeps the exponent and cuts the mantissa from 23 bits to 10. The forecast
    # is ~1e-2 and the comparison against the floor turns on the fifth decimal
    # of MAE, so leaving it on would quietly make this run incomparable with
    # every CPU-measured number it is meant to sit beside.
    configure_determinism()

    dates = [str(d) for d in package["dates"]]
    tickers = [str(t) for t in package["tickers"]]
    frame = pd.DataFrame(np.asarray(package["series"], dtype=float),
                         index=dates, columns=tickers)
    positions = {d: i for i, d in enumerate(dates)}

    row_date = [str(d) for d in package["row_date"]]
    row_ticker = package["row_ticker"].astype(int)
    horizon = int(meta["horizon"])

    # Row lookup per (date, ticker) so a forecaster that declines a ticker
    # simply leaves that row NaN rather than shifting every later row by one.
    slot: dict[tuple[str, str], int] = {}
    for i, (d, j) in enumerate(zip(row_date, row_ticker)):
        slot[(d, tickers[j])] = i

    grid = list(dict.fromkeys(row_date))
    if args.limit_dates:
        grid = grid[: args.limit_dates]

    forecaster = build_forecaster(args)
    print(f"  forecaster: {forecaster.name} @ context {args.context}")

    pred = np.full(len(row_date), np.nan, dtype=np.float64)
    used_context = np.zeros(len(grid), dtype=np.int32)
    started = time.time()

    def save(done: int) -> None:
        run = {
            "model": args.model,
            "model_id": args.model_id or "",
            "name": forecaster.name,
            "context": args.context,
            "cross_learning": bool(args.cross_learning),
            "horizon": horizon,
            "dates_done": done,
            "dates_total": len(grid),
            "seconds": round(time.time() - started, 1),
        }
        np.savez_compressed(args.out, pred=pred,
                            used_context=used_context,
                            run=np.array([json.dumps(run)], dtype=object))

    for k, as_of in enumerate(grid):
        histories = _history_ending_at(frame, positions, as_of, args.context)
        if not histories:
            continue
        used_context[k] = max(len(h) for h in histories.values())

        out = forecaster.forecast(histories, horizon)
        for ticker, move in out.items():
            i = slot.get((as_of, ticker))
            if i is not None and np.isfinite(move):
                pred[i] = float(move)

        done = k + 1
        rate = (time.time() - started) / done
        if done % args.checkpoint_every == 0 or done == len(grid):
            save(done)
            print(f"  {done}/{len(grid)} dates  {rate:.1f}s/date  "
                  f"eta {(len(grid) - done) * rate / 60:.0f} min", flush=True)

    save(len(grid))
    filled = int(np.isfinite(pred).sum())
    print(f"\n  wrote {args.out}")
    print(f"    {filled} of {len(pred)} rows predicted "
          f"({filled / max(len(pred), 1):.1%} of the grid)")
    print(f"    {time.time() - started:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
