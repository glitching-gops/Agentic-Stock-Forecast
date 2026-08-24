"""
tools/run_probe.py - Score a linear probe on frozen Chronos-2 embeddings.

    python tools/run_probe.py --context 512                 # validate the path
    python tools/run_probe.py --context 2048 --json p.json  # the real run
    python tools/run_probe.py --cache emb512.npz            # reuse a cache

THE QUESTION. Five zero-shot foundation-model configurations have been scored
on this panel and none cleared reb_t > 2. That result cannot distinguish "the
representation holds nothing" from "the representation holds something the
generic forecast head does not express as a 30-step relative-price move". A
ridge on the frozen encoder state separates the two, and it is the cheap
experiment that decides whether LoRA is worth days of compute.

WHAT IS AND IS NOT FITTED. The model is frozen; only the ridge is fitted, and
only on each fold's training rows. That makes the embeddings fold-independent,
so they are computed ONCE for the whole panel and cached - the expensive part
runs a single time and every fold's fit is then seconds.

It also puts the purged folds back in charge. For a zero-shot comparator the
folds protect nothing (there is no fit) and the entire guarantee rests on the
as-of slice. Here there IS a fit, so both matter: `_history_ending_at` keeps
the embedding causal, and the harness keeps the ridge from seeing test rows.

READ THE OUTPUT AGAINST `chronos2` AT THE SAME CONTEXT, not against the whole
table - the probe and the zero-shot forecaster differ in exactly one thing,
which is whether a fitted linear head replaces the pretrained one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pipeline.baselines import FACTORS, _pooled_xgb_factory      # noqa: E402
from pipeline.evaluation import (PurgedPanelWalkForward,         # noqa: E402
                                 panel_walk_forward)
from pipeline.panel import (MIN_NAMES_PER_DATE, TARGET,          # noqa: E402
                            load_panel, panel_coverage,
                            relative_price_frame, usable_dates)
from pipeline.series import DEFAULT_CONTEXT                      # noqa: E402
from pipeline.signals import HORIZON_SESSIONS                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--cache", type=str, default=None,
                    help="path to read/write the embedding cache (.npz). "
                         "Reused if it exists and matches the context.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-dates", type=int, default=None,
                    help="embed only the LAST N usable dates (smoke test; the "
                         "numbers are not results). The last rather than the "
                         "first: the panel opens in 2016 when no ticker yet "
                         "holds MIN_CONTEXT observations, so the earliest "
                         "dates embed nothing at all")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout, force=True)

    from pipeline.chronos_probe import (EmbeddingCache,
                                        build_embedding_cache, probe_factory)

    panel = load_panel()
    if panel.empty:
        print("  REFUSED: no signals rows", file=sys.stderr)
        return 1
    cov = panel_coverage(panel)
    print(f"panel {cov['rows']:,} rows | {cov['tickers']} tickers | "
          f"{cov['dates']:,} dates | {cov['first_date']} -> {cov['last_date']}")

    series = relative_price_frame(panel)
    dates = usable_dates(panel)
    if args.max_dates:
        dates = dates[-args.max_dates:]
        print(f"  SMOKE TEST: only the last {len(dates)} dates are embedded, "
              f"so the scores below are not results.")

    cache = None
    if args.cache and os.path.exists(args.cache):
        cache = EmbeddingCache.load(args.cache)
        if cache.context != args.context:
            print(f"  cache at context {cache.context} does not match the "
                  f"requested {args.context}; rebuilding", file=sys.stderr)
            cache = None
        else:
            print(f"  reused cache: {len(cache.keys):,} rows, dim {cache.dim}")

    if cache is None:
        print(f"\nEmbedding {len(dates):,} dates at context {args.context} ...")
        started = time.time()

        def progress(n, total, rows):
            if n % 50 and n != total:
                return
            per = (time.time() - started) / max(n, 1)
            print(f"  {n}/{total} dates  {rows:,} rows  "
                  f"{per:.2f}s/date  ~{per * (total - n) / 60:.0f} min left",
                  flush=True)

        cache = build_embedding_cache(
            series=series, dates=dates, context=args.context,
            horizon=HORIZON_SESSIONS, device=args.device, on_date=progress)
        took = time.time() - started
        print(f"  embedded {len(cache.keys):,} rows, dim {cache.dim}, "
              f"in {took / 60:.1f} min")
        if args.cache:
            cache.save(args.cache)
            print(f"  wrote {args.cache} "
                  f"({os.path.getsize(args.cache) / 2**20:.0f} MB)")

    # The cache must cover what is being scored. A ChronosProbe handed rows it
    # has no embedding for does not fail - it abstains and predicts 0.0, which
    # lands in the table as an MAE exactly equal to the `zero` floor and a
    # blank IC. That is indistinguishable from a genuine null result, and it is
    # how a run that embedded nothing would be read as "the probe found no
    # signal". So the panel is restricted to the dates actually embedded, and
    # what that costs is stated rather than absorbed.
    cached_dates = {d for (d, _t) in cache.keys}
    before = len(panel)
    panel = panel[panel["date"].astype(str).isin(cached_dates)].reset_index(drop=True)
    dropped = before - len(panel)
    if dropped:
        print(f"\n  panel restricted to embedded dates: {before:,} -> "
              f"{len(panel):,} rows ({dropped:,} dropped)")
        if not args.max_dates:
            print(f"  WARNING: the cache did not cover {dropped:,} rows of the "
                  f"panel. Rebuild it, or the comparison below is against a "
                  f"different sample than the published table.", file=sys.stderr)

    scored_cov = panel_coverage(panel)
    if scored_cov["median_names_per_date"] < MIN_NAMES_PER_DATE:
        print(f"\n  REFUSED: the embedded panel's median date holds "
              f"{scored_cov['median_names_per_date']:.0f} names, below the "
              f"{MIN_NAMES_PER_DATE} needed to rank a cross-section.",
              file=sys.stderr)
        return 1

    splitter = PurgedPanelWalkForward(
        n_folds=args.folds, horizon=HORIZON_SESSIONS,
        embargo=HORIZON_SESSIONS, min_train=args.min_train)

    print("\nScoring ...")
    runs = [
        ("probe_mse", probe_factory(cache, "probe_mse", "mse"),
         ["date", "ticker"]),
        ("probe_ic", probe_factory(cache, "probe_ic", "ic"),
         ["date", "ticker"]),
        # Scored beside the probe on identical folds and rows. pooled_xgb is
        # the incumbent to beat; without it in the same run the probe's number
        # would have to be compared against a table built on a different panel.
        ("pooled_xgb", _pooled_xgb_factory, FACTORS),
    ]

    results = []
    for name, factory, cols in runs:
        t0 = time.time()
        r = panel_walk_forward(panel=panel, feature_cols=cols,
                               model_factory=factory, splitter=splitter,
                               name=name, target=TARGET,
                               rebalance_every=HORIZON_SESSIONS)
        xs = r.cross_sectional
        row = {"name": name,
               "daily_rank_ic": r.metrics.get("daily_rank_ic"),
               "rebalance_ic": xs.get("mean_rank_ic"),
               "rebalance_ic_t": xs.get("rank_ic_t"),
               "n_rebalances": xs.get("n_rebalances"),
               "mae": r.metrics.get("mae"),
               "mae_naive_zero": r.metrics.get("mae_naive_zero"),
               "n_oos": r.metrics.get("n_oos"),
               "seconds": round(time.time() - t0, 1)}
        results.append(row)

    def f(v, spec="+.4f"):
        return "     -" if v is None or not np.isfinite(v) else format(v, spec)

    print(f"\n{'comparator':<18s} {'reb_IC':>9s} {'reb_t':>7s} {'dailyIC':>9s} "
          f"{'MAE':>9s} {'vs floor':>9s} {'n_reb':>6s} {'secs':>7s}")
    print("-" * 82)
    floor = results[0].get("mae_naive_zero")
    for r in results:
        vs = (f"{100 * (r['mae'] / floor - 1):+.1f}%"
              if floor and r.get("mae") else "-")
        print(f"{r['name']:<18s} {f(r['rebalance_ic']):>9s} "
              f"{f(r['rebalance_ic_t'], '+.2f'):>7s} "
              f"{f(r['daily_rank_ic']):>9s} {f(r['mae'], '.5f'):>9s} "
              f"{vs:>9s} {r['n_rebalances']:>6d} {r['seconds']:>7.1f}")
    print(f"\n  MAE floor (predict zero): {floor:.5f}" if floor else "")
    print("  reb_t is the pre-registered criterion. Below 2 is not evidence.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"coverage": cov, "context": args.context,
                       "embedding_dim": cache.dim,
                       "cached_rows": len(cache.keys),
                       "results": results}, fh, indent=2, default=str)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
