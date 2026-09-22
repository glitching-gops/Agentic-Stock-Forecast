"""
tools/baseline_repro.py — does the stored h=30 baseline still reproduce, EXACTLY?

The stored baseline is the `new_pinned` arm of `hygiene_repin_oos.npz`: pooled
x MAE, no ticker feature, the within-date standardised label, XGB_THREADS
pinned, h=30, legacy purge, on `panel_cache.parquet`. This re-runs that arm
through the same `tools.hygiene_repin.run_one` and compares row for row.

The bar is drift EXACTLY 0.0, not the 1e-9 `REPRO_TOL` the Stage 1 tools use:
the point of a lock file is that nothing moved, and "moved by less than a
nanometre" is a statement that something did.

Used three ways, and each is its own attribution step:

    after the lock          same panel, pinned libraries -> must be 0.0
    after the phantom fix   rebuilt panel -> the delta IS the fix
    twice in a row          the new stored baseline reproduces itself

    python tools/baseline_repro.py                          # vs hygiene_repin_oos.npz
    python tools/baseline_repro.py --panel-cache X.parquet --against Y.npz --arm A
    python tools/baseline_repro.py --save out.npz            # keep this run's predictions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.determinism import environment_fingerprint  # noqa: E402
from pipeline.label import standardise_target  # noqa: E402
from tools.hygiene_repin import COLS, PANEL_CACHE, PRED_OUT, run_one  # noqa: E402
from tools.stage0c_close import _from_arm  # noqa: E402
from tools.stage1_reversal import as_data, reproduction, sha256_file  # noqa: E402
from tools.stage2b_pooled import load_cached_panel  # noqa: E402

STORED_ARM = "new_pinned"


def installed_versions() -> dict:
    """Every distribution in the running interpreter, name -> version."""
    from importlib.metadata import distributions
    return {d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in distributions()}


def run(panel_cache: str, trials: int):
    std_panel = standardise_target(load_cached_panel(panel_cache))
    return run_one(std_panel, trials, "baseline_repro", verbose=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-cache", default=PANEL_CACHE)
    ap.add_argument("--against", default=PRED_OUT)
    ap.add_argument("--arm", default=STORED_ARM)
    ap.add_argument("--trials", type=int, default=None,
                    help="default: pipeline.model.EVAL_TUNE_TRIALS, as stored")
    ap.add_argument("--save", default=None,
                    help="write this run's predictions as arm --save-arm")
    ap.add_argument("--save-arm", default=STORED_ARM)
    ap.add_argument("--json", default=None)
    ap.add_argument("--score", action="store_true",
                    help="also score the run: DK cs IC, rebalance IC, grades")
    ap.add_argument("--bootstrap", type=int, default=None,
                    help="grading bootstrap draws (default BOOTSTRAP_B)")
    args = ap.parse_args()

    from pipeline.model import EVAL_TUNE_TRIALS
    trials = EVAL_TUNE_TRIALS if args.trials is None else args.trials

    started = time.time()
    env = environment_fingerprint()
    print(f"python {sys.version.split()[0]} on {sys.platform}; {env}", flush=True)
    print(f"panel {args.panel_cache} sha256 {sha256_file(args.panel_cache)}", flush=True)

    preds, folds = run(args.panel_cache, trials)
    rep = reproduction(_from_arm(args.against, args.arm), as_data(preds))
    exact = (rep["only_stored"] == 0 and rep["only_rerun"] == 0
             and rep["max_pred_drift"] == 0.0 and rep["max_truth_drift"] == 0.0)
    rep["exact"] = exact
    rep["gammas"] = [round(f["gamma"], 3) for f in folds]
    rep["runtime_min"] = round((time.time() - started) / 60, 1)
    rep["python"] = sys.version.split()[0]
    rep["platform"] = sys.platform
    rep["environment"] = env

    print(f"rows stored {rep['rows_stored']:,} / re-run {rep['rows_rerun']:,}, "
          f"unmatched {rep['only_stored']}/{rep['only_rerun']}", flush=True)
    print(f"max prediction drift {rep['max_pred_drift']:.3e}, "
          f"max label drift {rep['max_truth_drift']:.3e}, gammas {rep['gammas']}",
          flush=True)
    print(f"EXACT REPRODUCTION: {'YES' if exact else 'NO'} "
          f"({rep['runtime_min']} min)", flush=True)

    if args.score:
        from pipeline.evidence_panel import BOOTSTRAP_B
        from tools.hygiene_repin import score_arm

        rep["metrics"] = score_arm(preds, folds,
                                   BOOTSTRAP_B if args.bootstrap is None else args.bootstrap,
                                   True)
        m = rep["metrics"]
        print(f"cs IC {m['dk_cs_ic']:+.5f} (DK t {m['dk_t']:+.2f}), reb IC "
              f"{m['reb_ic']:+.4f} (t {m['reb_t']:+.2f}), S/W/I "
              f"{m['strong']}/{m['weak']}/{m['insufficient']}, rows {m['n_rows']:,}",
              flush=True)

    if args.save:
        np.savez_compressed(args.save, **{f"{args.save_arm}__{c}": preds[c].to_numpy()
                                          for c in COLS})
        print(f"saved {args.save}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({**rep, "versions": installed_versions()}, f, indent=1,
                      default=float)
    sys.exit(0 if exact else 1)


if __name__ == "__main__":
    main()
