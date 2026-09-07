# Scripts

Utility and migration scripts. These are run manually as needed, not as part of the main application.

## migrate_to_supabase.py
One-time migration script to transfer data from local SQLite to Supabase PostgreSQL.
Run once during initial deployment setup.

## data_check.py
Utility script to check data sanity in the database.

## verify_endpoints.py
Utility script to verify the health of the FastAPI endpoints.

## verify_stage1.py & verify_stage2.py
Validation scripts used during initial pipeline execution and universe scaling to verify system outputs.

## stage2a_gamma_spotcheck.py & stage2a_pilot.py

Stage 2a — the tuner-objective pilot. Diagnostic, sandboxed, and **not** wired
into the daily or weekly job.

`stage2a_gamma_spotcheck.py` (Step A) takes ten fully-constant (ticker, fold)
cells from the Stage 0 out-of-sample cache, holds every hyperparameter at
whatever the nested Optuna search chose for that exact cell, and sweeps `gamma`
alone. It refuses to report anything unless the as-tuned refit reproduces the
cached prediction — the same role `tau = 1.00` played in the Stage 0 addendum.

`stage2a_pilot.py` (Step B) runs both tuning objectives over a stratified
sample of 14 tickers and reports degeneracy, within-fold rank IC, MAE and the
train/out-of-sample gap side by side. It writes a markdown report and a CSV and
nothing else: no hyperparameter cache, no `model_metadata`, no table.

```bash
python tools/stage2a_gamma_spotcheck.py --markdown docs/stage2a-step-a.md
python tools/stage2a_pilot.py --markdown docs/stage2a-step-b.md
```

Method and both decision rules are fixed in `docs/stage2a-preregistration.md`,
written before either was run on real data.

## stage2b_pooled.py & stage2b_panel_diagnostic.py

Stage 2b — the pooled cross-sectional model. Diagnostic, sandboxed, and not
wired into either job.

`stage2b_pooled.py` runs a nested purged hyperparameter search over the WHOLE
panel rather than one ticker at a time, under both tuning objectives and with
the ticker categorical present, absent and placebo-shuffled. It reports the
degeneracy rate, held-out MAE, cross-sectional and time-series rank IC, and the
train/out-of-sample gap for each cell, plus the measured standard error of the
objective under a persistence-preserving null — which is the quantity the whole
pooling hypothesis rests on.

`stage2b_panel_diagnostic.py` re-runs Stage 0's empirical-Bayes panel
statistics on the pooled model's held-out predictions. It REUSES
`pipeline/evidence_shrinkage.py` rather than reimplementing it, and that module
lives on `stage0-evidence-grading`, which is not merged here — so it imports
lazily and tells you how to materialise the file. That copy is gitignored on
this branch and must never be committed into it.

```bash
python tools/stage2b_pooled.py --build-cache          # once, ~40 s
python tools/stage2b_pooled.py --markdown stage2b_report.md
git show stage0-evidence-grading:pipeline/evidence_shrinkage.py > pipeline/evidence_shrinkage.py
python tools/stage2b_panel_diagnostic.py
```

Method and decision rule are fixed in `docs/stage2b-preregistration.md`,
written before any pooled training run.

## Phase 0 changes

- `select_top_50.py` — **deleted.** It ranked stocks by composite score and kept
  the top 5 per sector, which selected the universe on the model's own reported
  accuracy (audit finding F4). The universe now comes from `data/universe.py`,
  which applies a point-in-time rule referencing no model output.
- `update_tickers.py` — **deleted.** It rewrote the hard-coded `TICKERS` dict in
  `data/tickers.py`. That dict no longer defines the universe; `tickers.py` now
  holds metadata only and reads it from the `index_membership` table.
- The remaining tools take their ticker list from `data.universe.get_universe()`.

### Recovering historical index membership

`data/universe.py` records membership from the first `sync_current_membership()`
call onward, so history before that date is unknown and evaluations covering it
are survivorship-biased. `backfill_membership_from_wayback()` reconstructs
earlier membership from Internet Archive snapshots of the NSE constituent CSV.
It is a manual tool because archive.org's CDX endpoint is frequently
unavailable — it returns 0 and reports the failure rather than writing partial
history that could be mistaken for complete. Re-run it periodically:

```bash
python -c "from data.universe import backfill_membership_from_wayback as b; print(b())"
```
