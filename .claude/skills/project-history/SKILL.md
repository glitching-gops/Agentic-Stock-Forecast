---
name: project-history
description: >
  The closed history of this project: the Phase 0 leakage audit (findings
  F1-F15, what each defect was and how it was fixed) and the post-audit
  production incidents. Load this when a finding ID such as F1, F6, F11 or F13
  is referenced anywhere in the code, tests or CLAUDE.md and you need to know
  what it was; when you need the rationale behind a guard or an archived
  module; or when asked what the audit found. Every defect here is FIXED and
  CLOSED - the rules that are still operationally live were distilled into
  CLAUDE.md section 7 (Landmines), which stays loaded at all times.
---

# Project history - the closed record

Moved out of CLAUDE.md so it loads on demand rather than in every session.
Nothing here is an open issue. If a rule here still governs day-to-day work it
also appears in CLAUDE.md section 7, which is the authoritative live copy.

## 3. The Phase 0 audit

The previous version reported **~4.3% MAPE and ~85% directional accuracy**.
Those were artifacts. Three leakage defects compounded, over six functional
bugs and several hygiene issues.

### Leakage (the headline defects)

| ID | Finding | Status |
|---|---|---|
| **F1** | Ridge meta-learner was **fitted and scored on the same rows**. Re-running the procedure reproduces the headline numbers from pure in-sample fit. | Fixed — meta-learner archived; `tests/test_leakage.py` |
| **F2** | Optuna received the **full labelled set**, including the 15% later reported as held out. | Fixed — nested tuning inside each training fold |
| **F3** | Cross-validation used **contiguous folds with no purging or embargo**. | Fixed — `PurgedWalkForward(n_folds, horizon, embargo, min_train)` |

### Selection and target

| ID | Finding | Status |
|---|---|---|
| **F4** | Universe was **selected on the model's own composite score** (top 5 per sector), so every averaged metric was biased upward. | Fixed — `data/universe.py` rule references no model output |
| **F8** | Target was a **price level**, not a return. | Fixed — `TARGET = "target_excess_return"` |
| **F7** | `sentiment_score` was **0.0 for every training row** and non-zero only for the row being predicted — a train/serve mismatch at exactly the row that matters. | Fixed — removed from `FEATURES`; returns only with a dated news archive |

### Functional bugs (each confirmed by execution)

| ID | Finding | Status |
|---|---|---|
| **F5** | LSTM **never wrote a checkpoint** (NaN validation targets), so the advertised ensemble was never running. | Fixed — archived; import-guard test |
| **F6** | Training labels were **never backfilled** — the labelled set froze at the day the DB was first populated. | Fixed — upsert writes; monotonicity guard in both jobs |
| **F10** | Warm-path forecast raised `KeyError` and **persisted today's price as the forecast** with `mape = 100`. | Fixed — warm path removed entirely |
| **F11** | OHLCV ingestion appended unseen dates, **splicing two corporate-action adjustment bases together**. | Fixed — `DELETE FROM ohlcv` then rewrite |
| **F12** | Backward-fill in macro data introduced lookahead. | Fixed — leading gaps left as gaps |
| **F13** | Earnings surprise attached to the announcement date; **Indian results are commonly declared post-close**. | Fixed — attaches to first session strictly after |

### Scoring and security

| ID | Finding | Status |
|---|---|---|
| **F9** | ~75 of the composite score's 100 points came from **leaked in-sample metrics** that barely varied across stocks; the verdict was set by deterministic branches on `mape`/`dir_acc`. | Fixed — evidence-gated composite; LLM may only downgrade |
| **F15** | f-string SQL interpolation of user-supplied tickers, non-timing-safe API key compare, open CORS. | Fixed — bound params, `secrets.compare_digest`, `ALLOWED_ORIGINS` |

*(F14 is not referenced anywhere in the codebase.)*

---

## 4. Post-Phase-0 remediation (complete)

Issues that only surfaced once the pipeline ran at full scale.

### Two incidents worth remembering

**Weekly job evaluated only 33/95 (2026-08-15).** Not a timeout — it finished in
14 minutes. The weekly job *read* the `signals` table but never *built* it; only
the daily job wrote it. An infrastructure race was reported as a data-quality
verdict ("insufficient history" for INFY, TCS, RELIANCE, HDFCBANK). Fixed by
making the weekly job self-sufficient.

**Weekly run destroyed labels for 22 tickers (2026-08-16).** Three sector
indices were transiently unavailable, producing a NULL target on every row.
`_upsert_signals` is DELETE-range-then-reinsert, so a NULL-target frame is
*destructive* — roughly 2,390 labels erased per ticker. It was silent because
`get_benchmark_series` only raised on an *empty* response; a frame that cleaned
down to nothing fell through the success path. Claimed fixed at three layers:
retry, treat "cleans to nothing" as failure, and refuse to write a null-target
frame. **Only the first two were actually implemented** — see the next entry.

**Two scheduled runs published nothing and reported success (2026-08-17/18).**
The daily cron fired correctly both days; both runs finished in ~13 min against
a normal ~39, and every forecast in the database stayed stamped with the 17 Aug
*manual* dispatch. Two defects compounded:

1. The third remediation layer above never existed. `_upsert_signals` guarded
   `if df.empty` and nothing else, so a benchmark outage still wrote null
   targets over good labels. Measured live: RELIANCE / TVSMOTOR / ITC /
   JSWSTEEL at 600 of 600 rows null (`^CNXENERGY`, `^CNXAUTO`, `^CNXFMCG`,
   `^CNXMETAL`), against the correct 29 trailing nulls elsewhere.
2. The F6 guard then correctly aborted — and `scheduler.py` contained no
   `sys.exit`, no `raise`, and a catch-all `except Exception`, so the process
   exited 0 and Actions marked both runs green.

Fixed at both layers: a decrease check at the write boundary in
`_upsert_signals` (`LabelLossRefused`), a frame-level skip for a benchmark that
downloads but does not *align*, and `PipelineAbort` on every abort path in
both jobs. `compute_and_store` now returns a `SignalsReport` so the caller can
see what was skipped and refused.

**Lesson:** any job given `compute_and_store` must also carry the F6
monotonicity guard — and a job that publishes nothing must exit non-zero, or
the guard's only visible effect is a faster green run.

---
