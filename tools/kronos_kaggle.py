"""
tools/kronos_kaggle.py - The Kaggle side. Slice, decode, return prices.

Paste this into a Kaggle GPU notebook cell (T4 x2 or P100) after attaching the
package from `tools/export_kronos_package.py` as a dataset, then run:

    !python kronos_kaggle.py --package /kaggle/input/<dataset>/kronos_512.npz \\
        --model NeoQuasar/Kronos-base --samples 8 --seed 0 \\
        --out /kaggle/working/kronos_base_512_s0.npz

THIS SCRIPT DELIBERATELY COMPUTES NOTHING IT COULD GET WRONG
------------------------------------------------------------
It returns the TERMINAL RELATIVE CLOSE of each sampled path, per row, per
sample. It does not difference against an anchor, does not average, does not
take a logarithm, does not decide which row of the forecast is t+horizon, and
does not choose which rows are scorable. All of that happens in
`tools/score_kronos.py`, at home, in the same process and through the same
`cross_sectional_report` as every other row of the results table.

That is a deliberate line, not laziness. Each of those steps has a wrong
version that produces a complete and plausible number:

    an off-by-one row     forecasts 29 or 31 sessions against a 30-day label
    the wrong column      returns a `high` where a `close` was meant
    the wrong anchor      measures the window's drift as though it were skill
    a price-space mean    applies a Jensen bias that grows with dispersion

A notebook is the worst place for any of them, because nothing there is under
test and the output looks the same either way.

THE WINDOW IS GIVEN, NOT DERIVED
--------------------------------
`row_end` is the position in the shared date index that each row is as-of. This
script slices `candles[end - context + 1 : end + 1]` and nothing else. For a
zero-shot model the purged folds protect nothing — there is no fit — so that
single slice IS the entire causal guarantee, and it is not the notebook's to
make. An off-by-one there hands the model its own answer and would read as a
breakthrough.

REPRODUCIBILITY
---------------
Kronos samples: `sample_from_logits` is called with `sample_logits=True`
hardcoded, so there is no greedy path. The seed is set PER ROW-BATCH from
`--seed` plus the batch index, so re-running one batch reproduces its values
independently of how many ran before it. Determinism also needs TF32 off — Ada
and Ampere run float32 matmuls in TF32 by default, which keeps the exponent and
cuts the mantissa from 23 bits to 10, and the comparison against the `zero`
floor turns on the fifth decimal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# Pinned to the commit `pipeline/vendor/kronos/` was taken from, so the notebook
# and the local checkout run the same model code. Kronos ships no pip package;
# an unpinned clone is a dependency that can change under a recorded result.
UPSTREAM = "https://github.com/shiyu-coder/Kronos"


def ensure_kronos(commit: str, where: str = "/kaggle/working/kronos_src",
                  local_src: str | None = None):
    """
    Put the pinned Kronos source on `sys.path`.

    `local_src` skips the clone and uses a directory already holding the model
    package — how this script is smoke-tested against the repo's own vendored
    copy before a Kaggle session is spent on it. The clone path is what runs on
    Kaggle, and it is PINNED: an unpinned clone is a dependency that can change
    under a recorded result, which is the same reason the `timesfm` PyPI
    package is refused in requirements-series.txt.
    """
    if local_src:
        if local_src not in sys.path:
            sys.path.insert(0, local_src)
        print(f"  kronos source taken from {local_src} (clone skipped)")
        return

    if not os.path.isdir(where):
        subprocess.run(["git", "clone", "--quiet", UPSTREAM, where], check=True)
    subprocess.run(["git", "-C", where, "checkout", "--quiet", commit],
                   check=True)
    head = subprocess.run(["git", "-C", where, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != commit:
        raise SystemExit(f"clone is at {head}, expected {commit}")
    if where not in sys.path:
        sys.path.insert(0, where)
    print(f"  kronos source pinned at {head[:12]}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", default="NeoQuasar/Kronos-base")
    ap.add_argument("--samples", type=int, default=8,
                    help="sampled paths per row. NOT optional: one draw from a "
                         "30-step autoregressive sampler is a path, not an "
                         "estimate. Measured at 1 sample the predicted "
                         "cross-sectional SD is ~2x the target's.")
    ap.add_argument("--batch-samples", type=int, default=2,
                    help="paths per forward batch. MEMORY scales with this: "
                         "measured 2.68 GiB for 90 series at context 512 with "
                         "one path, so 4 needs ~11 GiB and 8 needs ~21 GiB - "
                         "past a 16 GB T4. And exceeding VRAM does not fail "
                         "cleanly, it CRAWLS: at the limit on a 6 GB card the "
                         "cost went 90.8 s/date to 1261 s/date for twice the "
                         "paths. Raise it only while s/date stays roughly "
                         "linear in it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", default="/kaggle/working/kronos_predictions.npz")
    ap.add_argument("--limit-dates", type=int, default=None,
                    help="smoke test: score only the first N rebalance dates")
    ap.add_argument("--kronos-src", default=None,
                    help="directory containing the `model` package. Skips the "
                         "pinned clone; used to smoke-test this script locally "
                         "against the repo's vendored copy.")
    args = ap.parse_args()

    pkg = np.load(args.package, allow_pickle=True)
    meta = json.loads(str(pkg["meta"][0]))
    context = int(meta["context"])
    horizon = int(meta["horizon"])
    input_cols = list(meta["input_cols"])

    ensure_kronos(meta["vendored_commit"], local_src=args.kronos_src)
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    # TF32 OFF. See the module docstring — it is not a rounding nicety, it is
    # the difference between a number comparable with the CPU-measured tables
    # and one that merely looks like it.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer_id, cap = meta["tokenizers"][args.model]
    if context > int(cap):
        raise SystemExit(f"{args.model} caps at context {cap}, package is {context}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, device=device,
                                max_context=context)
    print(f"  {args.model} on {device} | context {context} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    candles = pkg["candles"]                       # (dates, tickers, 5)
    dates = [str(d) for d in pkg["dates"]]
    row_date = np.array([str(d) for d in pkg["row_date"]])
    row_ticker = pkg["row_ticker"].astype(int)
    row_end = pkg["row_end"].astype(int)

    unique_dates = list(dict.fromkeys(row_date.tolist()))
    if args.limit_dates:
        unique_dates = unique_dates[:args.limit_dates]

    chunks = ([args.batch_samples] * (args.samples // args.batch_samples)
              + ([args.samples % args.batch_samples]
                 if args.samples % args.batch_samples else []))
    print(f"  {len(row_date):,} rows | {len(unique_dates)} rebalance dates | "
          f"{args.samples} samples in chunks {chunks}")

    # (n_rows, n_samples) of terminal relative closes. Every sample is kept
    # rather than averaged here, so the averaging convention — weighted, and in
    # log space — stays a decision made at home where it is under test.
    terminal = np.full((len(row_date), args.samples), np.nan, dtype=np.float64)

    started = time.time()
    for step, as_of in enumerate(unique_dates):
        picked = np.flatnonzero(row_date == as_of)
        if len(picked) == 0:
            continue

        end = int(row_end[picked[0]])
        window = slice(end - context + 1, end + 1)
        x_stamp = pd.Series(pd.to_datetime(dates[window]))
        y_stamp = pd.Series(pd.bdate_range(
            pd.Timestamp(dates[end]) + pd.Timedelta(days=1), periods=horizon))

        dfs, xs, ys = [], [], []
        for row in picked:
            block = candles[window, int(row_ticker[row]), :]
            dfs.append(pd.DataFrame(block.astype(np.float64), columns=input_cols))
            xs.append(x_stamp)
            ys.append(y_stamp)

        torch.manual_seed(args.seed * 100_000 + step)

        filled = 0
        for chunk in chunks:
            out = predictor.predict_batch(dfs, xs, ys, pred_len=horizon,
                                          T=args.temperature, top_p=args.top_p,
                                          sample_count=chunk, verbose=False)
            # predict_batch averages the `chunk` paths internally, so one call
            # yields one averaged path. Recording it `chunk` times keeps the
            # weighting implicit and exact when the last chunk is short.
            for k, frame in enumerate(out):
                terminal[picked[k], filled:filled + chunk] = \
                    float(frame["close"].iloc[-1])
            filled += chunk

        if step % 5 == 0 or step == len(unique_dates) - 1:
            done = step + 1
            rate = (time.time() - started) / done
            print(f"  {done}/{len(unique_dates)} dates | {rate:.1f} s/date | "
                  f"eta {rate * (len(unique_dates) - done) / 60:.0f} min",
                  flush=True)

    np.savez_compressed(
        args.out,
        terminal=terminal,
        row_index=np.arange(len(row_date), dtype=np.int32),
        run=np.array([json.dumps({
            "model": args.model, "context": context, "samples": args.samples,
            "batch_samples": args.batch_samples, "seed": args.seed,
            "temperature": args.temperature, "top_p": args.top_p,
            "chunks": chunks, "device": device,
            "seconds": round(time.time() - started, 1),
            "dates_scored": len(unique_dates),
            "package_meta": meta,
        })], dtype=object),
    )
    print(f"\n  wrote {args.out} in {(time.time() - started) / 60:.1f} min")
    print("  Download it and run: python tools/score_kronos.py "
          f"--predictions <file> --package {os.path.basename(args.package)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
