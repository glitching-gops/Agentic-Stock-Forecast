"""
tools/platform_digest.py — does THIS machine produce the reference platform's
pooled-model numbers, bit for bit?

WHY. Identical code, identical library versions and identical threads give
DIFFERENT predictions on Windows and Linux (CLAUDE.md §7): XGBoost's row and
column subsampling draws a different sample from the same seed under MSVC's
and GCC's standard libraries. "Linux" is not one platform either — WSL and a
GitHub Actions runner can ship different C++ runtimes. The expectation is that
they agree, because the distributions are template code compiled INTO the
manylinux wheel rather than read from the runtime library; this checks it
instead of assuming it.

It fits the production pooled path (`pipeline.pooled.walk_forward` and
`fit_final`, with subsampling below 1.0 because that is the part that
diverges) on a FIXED synthetic panel — no database, no network, no snapshot —
and prints one digest. The snapshot itself cannot travel to a CI runner; this
can, and it exercises the same code on the same library build.

    python tools/platform_digest.py                 # print the digest
    python tools/platform_digest.py --check         # compare to the recorded one

The recorded digest is the reference platform's (tools/platform_digest.json).
A mismatch on the CI runner means the runner, where production trains, is the
reference — see docs/dashboard-switch-preregistration.md §3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RECORD = Path(__file__).with_suffix(".json")

#: Fixed hyperparameters with subsampling BELOW 1.0 — the divergent path —
#: so the digest does not depend on the search landing there by chance.
PARAMS = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 4,
          "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10,
          "gamma": 0.1, "reg_alpha": 0.5, "reg_lambda": 5.0, "tree_method": "hist"}


def synthetic_panel(n_tickers: int = 60, n_dates: int = 520, seed: int = 20260924
                    ) -> pd.DataFrame:
    from pipeline.pooled import FEATURES

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates).strftime("%Y-%m-%d")
    rows = []
    for i in range(n_tickers):
        x = rng.standard_normal((n_dates, len(FEATURES)))
        y = 0.02 * x[:, 0] - 0.01 * x[:, 3] + 0.1 * rng.standard_normal(n_dates)
        f = pd.DataFrame(x, columns=FEATURES)
        f["date"], f["ticker"] = dates, f"T{i:03d}.NS"
        f["close"] = 100.0
        f["target_return"] = y
        f.loc[f.index[-30:], "target_return"] = np.nan
        # Some missing values in a NULLABLE feature, as production now has.
        f.loc[rng.random(n_dates) < 0.05, "earnings_surprise"] = np.nan
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def digest() -> dict:
    from pipeline import pooled
    from pipeline.determinism import environment_fingerprint

    prepared = pooled.prepare(synthetic_panel())
    preds, _ = pooled.walk_forward(prepared, n_folds=3, min_train=250,
                                   fixed_params=PARAMS)
    searched, folds = pooled.walk_forward(prepared, n_folds=2, min_train=350,
                                          n_trials=2)
    fitted = pooled.fit_final(prepared, params=PARAMS)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(preds["y_pred"].to_numpy(dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(searched["y_pred"].to_numpy(dtype=np.float64)).tobytes())
    h.update(fitted.booster)
    return {"digest": h.hexdigest(),
            "fixed_params_prediction_sum": float(preds["y_pred"].sum()),
            "searched_gammas": [round(f["gamma"], 6) for f in folds],
            "environment": environment_fingerprint()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="write this machine's digest as the reference")
    args = ap.parse_args()
    d = digest()
    print(json.dumps(d, indent=1))
    if args.record:
        RECORD.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
        print(f"recorded {RECORD}")
    if args.check:
        ref = json.loads(RECORD.read_text(encoding="utf-8"))
        same = ref["digest"] == d["digest"]
        print(f"reference ({ref['environment'].get('platform')}): {ref['digest']}")
        print(f"this machine ({d['environment'].get('platform')}): {d['digest']}")
        print(f"PLATFORM DIGEST MATCHES THE REFERENCE: {'YES' if same else 'NO'}")
        sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
