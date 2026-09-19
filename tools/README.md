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

## stage0c_close.py

Stage 0c — the closing session of the evidence-grading track. Runs the
corrected layer (`pipeline/evidence_panel.py`) over every variant whose
held-out predictions already exist. Nothing is retrained.

Four corrections land at once, and every one of them is expected to REDUCE the
graded count: cross-sectional rank-demeaning of both sides, a date-level
circular block bootstrap that re-runs the whole empirical-Bayes pipeline per
replicate, Romano-Wolf stepdown across the 84 simultaneous tests, and REML with
an explicit tau2 ~ 0 detector that emits ONE panel statement instead of 84
duplicated ones.

Two placebos run by default and they are the point of the tool. Predictions
permuted **within each date** preserve the demeaning geometry exactly and
destroy only the name-to-outcome link; per-ticker constants carry no
information at all. If either grades names, the layer is manufacturing grades
rather than revealing them.

```bash
pip install -r requirements-evidence.txt      # arch + linearmodels, lazily imported
python tools/stage0c_close.py --markdown stage0c_report.md
python tools/stage0c_close.py --block 63 --markdown stage0c_report_block63.md
```

Method and decision rule are fixed in `docs/stage0c-preregistration.md`; the
closing document is `docs/stage0-closing.md`.

## stage0b_regrade.py

Stage 0b — the audit and fix of the grading methodology's IC.

`pipeline/evidence_shrinkage.py` (and `pipeline/evaluation.compute_metrics`,
which feeds the LIVE gate) computed each ticker's rank IC by concatenating
every walk-forward fold into one series and correlating once. Pooling across
groups conflates between-group with within-group variation, and on this panel
the folds' prediction levels run against their realised returns at rho -0.600 —
so the pooled figure came out negative while every fold's internal ranking was
positive.

The shrinkage module is FIXED: the point estimate is the mean of the
within-fold rank ICs, and the bootstrap resamples within folds and takes the
larger of the within-fold and between-fold variance components. The live gate
is NOT fixed — changing it changes `forecast_confidence`, which is a production
change — but this tool measures exactly what correcting it would do.

```bash
python tools/stage0b_regrade.py --markdown stage0b_regrade.md
```

Method and decision rule are fixed in `docs/stage0b-preregistration.md`;
the results are in `docs/stage0b-findings.md`.

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

## stage1_reversal.py

Stage 1, Pilot 1 — multi-lookback residual reversal. Sandboxed and wired into
neither job.

It adds `pipeline/reversal.py`'s four residualised (`rev_resid_{1,5,10,20}`)
and four raw (`rev_raw_{1,5,10,20}`) skip-one lookback returns to the pooled ×
MAE, no-ticker model, and grades all three arms through the Stage 0c harness
(`grade_panel_v3`, B = 1000). The baseline arm is REUSED from
`stage2b_pooled_oos.npz`, and is also re-run: it must reproduce the stored
predictions exactly before the other two arms are read. On top of that:
- a nine-draw within-date placebo on the residual arm;
- paired per-date cross-sectional IC tests with Driscoll-Kraay SEs at 30 lags;
- a feature-level IC table against Da, Liu & Schaumburg (2014).

```bash
python tools/stage1_reversal.py --smoke --markdown stage1_smoke.md   # minutes
python tools/stage1_reversal.py --markdown stage1_report.md          # ~1 h
```

Both need `panel_cache.parquet` and `stage2b_pooled_oos.npz`, whose sha256
the run checks against `docs/stage1-preregistration.md`. Point `--panel-cache`
and `--stage2b-npz` at them if they live outside this checkout.

## backfill_delivery.py, stage1b_delivery.py & stage1b_phantom_check.py

Stage 1, Pilot 2 — NSE delivery %. `backfill_delivery.py` fetches NSE's
daily `MTO_<ddmmyyyy>.DAT` for every panel session into `delivery_cache.npz`.
It throttles to 3 requests a second, resumes where it left off, and aborts on
a failure RATE, not a count. `--report` prints coverage; `--validate`
re-checks MTO against `sec_bhavdata_full` and the UDiFF bhavcopy.

`stage1b_delivery.py` runs the pre-registered arms: S1, R3 deciding, and the
column-permuting placebo. `stage1b_phantom_check.py` re-scores R3 on the
stored predictions with the four phantom 2026 sessions removed; it trains
nothing.

```bash
python tools/backfill_delivery.py            # ~2,440 requests, ~15 min
python tools/stage1b_delivery.py --markdown stage1b_report.md
python tools/stage1b_phantom_check.py --markdown phantom_check.md   # seconds
```

## backfill_results.py & stage1c_sue.py

Stage 1, Pilot 3 — seasonal-random-walk SUE from NSE's own results filings.
`backfill_results.py` reads four things:
- the legacy results list (to the December-2024 quarter);
- the Integrated Filing list (from March 2025);
- NSE's corporate actions (bonuses, splits, demergers);
- every first-disclosed quarterly filing from 2013, parsed for basic EPS.

It writes all of it to `results_cache.npz`, at no more than 3 requests a
second, resumably. `--reparse` re-reads documents an older parser left
without an EPS. Read `--report` first. It shows:
- coverage by year;
- how many old HTML pages reconciled with net profit ÷ shares, and at which
  row shift;
- XBRL EPS against its own implied EPS;
- disclosure timing against the 15:00 cutoff;
- NSE's splits against `corporate_actions`.

`stage1c_sue.py --design-only` prints the outcome-blind design inputs. The
full run adds R4 (net of the 0.2225% round trip) and the raw PEAD sort book
to Pilot 2's harness.

```bash
python tools/backfill_results.py                       # ~7,700 documents, ~45 min
python tools/backfill_results.py --report --markdown results_coverage.md
python tools/stage1c_sue.py --design-only
python tools/stage1c_sue.py --markdown stage1c_report.md                # ~1.5 h
```

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

## Evidence-Grading Redesign — Stage 0

**Separate track from the project's Phase 0-6 roadmap; neither renumbers the
other.**

### run_evidence_grading.py

Runs the Stage 0 shadow grading layer (`pipeline/evidence_shrinkage.py`) and
prints the old-grade against new-grade crosstab plus the panel diagnostics
`mu_hat` and `tau2_hat` — which are the finding, not an intermediate number.

Empirical-Bayes partial pooling (James-Stein / Efron-Morris shrinkage,
DerSimonian-Laird between-ticker variance, block bootstrap at the 30-session
label horizon, Benjamini-Hochberg FDR across the panel) replaces the live
gate's per-ticker frequentist checks, which at `n_effective ≈ 64` per name
demand a rank IC of ~0.25 to reach t = 2 — an effect size that does not exist
in monthly cross-sectional equity prediction.

**Shadow only.** Writes `evidence_grades_v2` and nothing the public API serves;
`forecast_confidence` and the old gate are untouched. No Render redeploy.

```bash
python tools/run_evidence_grading.py --rebuild --store --block-sweep
python tools/run_evidence_grading.py --no-rebuild        # re-grade a cache
```

The first form re-runs the per-ticker walk-forward (~65 min for 84 tickers)
because `evaluate_and_persist_ticker` does not persist its out-of-sample
predictions — see `docs/stage0-evidence-grading.md`. Method, decision table and
reproduction commands live there; the pre-registration is
`docs/stage0-preregistration.md`.
