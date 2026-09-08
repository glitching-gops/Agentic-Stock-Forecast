# Stage 0b pre-registration — audit and fix the grading methodology's IC

**Written 2026-09-08, before the audit or the fix was run on real data.**

Branch `stage0b-grading-fix`, off `stage2b-pooled-model` with
`stage0-evidence-grading` merged in — the first branch that holds both the
shrinkage module and the pooled model's held-out predictions. Not merged into
`main`.

---

## What triggered this

Stage 2b's panel diagnostic swapped in a model that emits **zero** constant
predictions and `mu_hat` barely moved: **−0.05988 → −0.052**, with the identical
**ρ = −0.600** between fold-level prediction levels and fold-level realised
returns that the Stage 0 addendum measured on the degenerate model. Same
model, same data, the per-ticker IC reads **−0.0512 pooled over folds** and
**+0.0120 within them**.

So Stage 0's headline number — and the addendum's entire degeneracy-threshold
sweep built on top of it — was measuring a property of the walk-forward
protocol's fold structure, not model skill.

**The mechanism.** `grade_evidence` and `pipeline/evidence_shrinkage.py`
concatenate every fold's predictions into one series and correlate once.
Pooling across groups conflates between-group with within-group variation: when
the folds' own average prediction levels run opposite to their own average
realised returns, the pooled correlation comes out negative even where every
fold's internal ranking is positive. `_mean_daily_rank_ic` in
`pipeline/evaluation.py` already avoids this at the panel level, and its
docstring already names the failure.

## The decision rule, fixed in advance

> This session succeeds if it definitively determines (a) which IC/correlation
> calculations in this codebase use the buggy pooled-across-folds pattern versus
> the correct within-fold-then-averaged pattern, explicitly including whether
> the project's original headline comparator table in `pipeline/baselines.py` is
> affected, and (b) what the corrected per-ticker grading methodology actually
> shows across every already-computed model variant. It does not succeed merely
> by producing more STRONG or WEAK grades than before — a corrected methodology
> that confirms the panel still carries little measurable per-ticker skill is
> exactly as successful an outcome as one that reveals more.

## Order of work, fixed so the audit cannot be shaped by the fix

1. **Audit first, code second.** Every rank/Pearson/IC-like statistic computed
   over data spanning more than one purge-embargo fold is enumerated and
   classified — within-fold-then-averaged, or pooled-then-correlated — before a
   line of the fix is written. The tools built *during this investigation*
   (`stage0_degeneracy_sweep`, `stage2a_*`, `stage2b_*`) are audited too: they
   could have inherited the pattern by copying from the module now known to
   carry it.
2. **If `pipeline/baselines.py`'s headline comparator table is affected, that is
   reported before anything else and before the fix proceeds.** It is a bigger
   finding than this session's plan and must not be folded into the rest.
3. **The synthetic proof is built before the real re-grade is run**, so the fix
   is demonstrated against an analytically known answer rather than judged by
   whether the real numbers improve.

## The fix, and what it is not

A **clean replacement**, not an alternative behind a flag. There is no
legitimate reason to keep a calculation known to be wrong as a live option.

**The variance estimator is in scope, not just the point estimate.** The
existing moving-block bootstrap in `evidence_shrinkage.py` was built to estimate
the sampling variance of the *pooled* statistic. Once the point estimate becomes
within-fold-then-averaged, its sampling behaviour is different, and the old
bootstrap is presumed invalid until shown otherwise. It will be checked
explicitly and changed if it does not estimate the new statistic's variance.

## The synthetic proof

Five folds; a genuinely positive within-fold prediction/target correlation of
roughly **+0.3** in every one; fold-level mean predictions and fold-level mean
targets deliberately **anti-correlated** across the five, mirroring the −0.600
Stage 2b measured. Three numbers reported side by side: the analytically correct
within-fold answer, what the old code returns, and what the new code returns.
Pinned as a permanent regression test.

## Predictions on record

Recorded so being wrong is visible. Stage 0's pre-registration got two of three
wrong and they stayed on the record.

1. **The `baselines.py` headline table is CLEAN.** `cross_sectional_report`
   scores per date and `_mean_daily_rank_ic` averages per date, so the
   comparator numbers this project has been measured against since Phase 2
   should be unaffected. If this is wrong it is the most important finding in
   the session.
2. **`mu_hat` flips sign under the fix**, to roughly the +0.01 to +0.13 range
   the within-fold measurements have shown, and its magnitude falls.
3. **The corrected grading still produces 0 STRONG**, because the corrected
   per-ticker IC is estimated on ~64 effective observations where a t of 2
   demands an IC of 0.25, and the within-fold ICs measured so far are an order
   of magnitude below that. Removing an artifact is not the same as finding
   signal.

Prediction 3 is the one that matters: the expected outcome is a **trustworthy
null**, not a result.

## Non-goals

- No retraining, re-tuning, or new data. Only the grading calculation changes,
  applied to predictions that already exist.
- The fold-0-peaks / fold-4-negative pattern seen across six Stage 2b arms —
  including the pure-noise placebo — is **out of scope**. It is a real and
  separate question about non-stationarity in this window and deserves its own
  investigation.
- Nothing written to `model_metadata`, `forecast_confidence`, the API or the
  frontend. Nothing merged to `main`.
- **A rise in STRONG or WEAK counts is not success.** Correctness is.
