# Stage 0b — the grading IC was pooled across folds, and it is fixed

> Closes a loop across four sessions. Stage 2b found that swapping in a model
> with **zero** constant predictions moved Stage 0's `mu_hat` only from −0.05988
> to −0.052, with the identical ρ = −0.600 between fold-level prediction levels
> and fold-level realised returns. That is not a model property. It is what a
> correlation computed over pooled folds reads.
>
> Branch `stage0b-grading-fix`, off `stage2b-pooled-model` with
> `stage0-evidence-grading` merged in. Not merged to `main`.
> Pre-registered before the audit or the fix ran:
> [`stage0b-preregistration.md`](stage0b-preregistration.md).

---

## 1. The audit — every IC computed over more than one fold

The order was fixed in advance: classify everything before writing a line of
the fix.

### CLEAN — within-group, then averaged

| where | what it computes | why it is safe |
|---|---|---|
| `evaluation.cross_sectional_report` | rank IC **per rebalance date**, then the mean and its t | This is `reb_IC` / `reb_t`. Every date belongs to one fold, so grouping by date groups by fold a fortiori. |
| `evaluation._mean_daily_rank_ic` | rank IC **per date**, averaged | `daily_IC`. Its own docstring names the pooled failure. |
| `baselines.compare_baselines` | consumes both of the above | **The headline comparator table is clean.** See below. |
| `portfolio.break_even_ic` | fed `xs["mean_rank_ic"]` from `cross_sectional_report` | the break-even 0.00512 and everything derived from it stand |
| `neutralise.residual_report` | rank IC per rebalance date, then mean and t | P5's residual numbers stand |
| `chronos_probe._daily_rank_ic` | per date, averaged inside each fold, then across folds | P2's probe stands |
| `tuning.purged_cv_rank_ic_score` | rank IC **inside each inner fold**, averaged | Stage 2a's objective was built correctly |
| `tuning.per_date_rank_ic` | per date, averaged | Stage 2b's pooled objective was built correctly |
| `tools/stage2a_pilot.py`, `stage2a_gamma_spotcheck.py` | one row **is** one (ticker, fold) cell | within-fold by construction |
| `tools/stage2b_pooled.py`, `stage2b_sweep.py` | `ts_rank_ic` grouped by (ticker, fold); `cs_rank_ic` per date within fold; `reb_ic` from `cross_sectional_report` | Stage 2b's tables stand |

### BUGGY — pooled, then correlated

| where | what it feeds | status |
|---|---|---|
| `evidence_shrinkage.block_bootstrap_ic` | Stage 0's `mu_hat`, the addendum's whole τ sweep, Stage 2b's panel diagnostic | **FIXED here** |
| `evaluation.compute_metrics` called from per-ticker `walk_forward` | `eval_rank_ic` / `eval_rank_ic_t` → `model_metadata` → **the LIVE `grade_evidence` gate** → `forecast_confidence` | **NOT fixed — measured instead.** See §5. |

### Deliberate, and correctly labelled

`tools/stage0_degeneracy_sweep.wholesale_within_fold` and
`tools/run_evidence_grading.fold_diagnostic` compute **both** statistics on
purpose, side by side. They are the diagnostics that found the −0.070 / +0.126
gap in the first place. They are not defects.

`tools/run_baselines.py` prints the pooled figure in its own `pooledIC` column
with a legend reading *"Moved by market timing AND by fold identity … Trust
daily_IC."* It was already documented as untrustworthy.

## 2. **The original headline table is NOT affected**

This was the audit's most important question and the answer is unambiguous.

Every number quoted in CLAUDE.md's comparator tables — `daily_IC`, `reb_IC`,
`reb_t`, `alpha_t`, MAE — comes from the within-date-then-averaged path. The
pooled figure exists in `compare_baselines`' in-memory dict as `rank_ic`, and:

- it is **not** in `BaselineComparison.to_metrics()`, so it was never persisted
  to `experiment_runs.metrics`;
- it is **not** what `clears_floor` uses (`rebalance_ic` is);
- it is **not** what `best()` ranks on (`daily_rank_ic` is);
- it **is** printed by `tools/run_baselines.py`, in a separate column, under a
  legend telling the reader not to trust it.

**So `pooled_xgb` +0.0389 / t +2.42, `beta_market` +0.0464, the P2/P3/P4/P5
tables and the break-even 0.00512 all stand.** A regression test now pins that,
so the pooled column cannot quietly become the reported one.

The blast radius is the **evidence-grading track only**: Stage 0, the Stage 0
addendum's τ sweep, Stage 2b's panel diagnostic, and the live per-ticker gate.

## 3. The fix, and the synthetic proof

`within_fold_rank_ic` replaces the pooled correlation outright — no flag, no
alternative path. Five folds, each built with a within-fold rank IC of **+0.30**,
whose fold-level means are anti-correlated at ρ = **−1.000**:

| | value |
|---|---|
| **analytically correct** (by construction) | **+0.3000** |
| OLD — pooled, then correlated | **−0.9492** |
| NEW — within-fold, then averaged | **+0.2708** |

Per-fold ICs `+0.209, +0.334, +0.255, +0.252, +0.305`; fold prediction levels
`−1.00 … +1.00` against realised levels `+0.99 … −1.00`. The control test
removes the fold levels and the two statistics then agree to 0.05, which is what
makes this a test of *pooling* rather than of the fixture. Pinned permanently in
`tests/test_stage0b_within_fold_ic.py`.

A second test asserts the corrected statistic agrees with
`_mean_daily_rank_ic` **to 1e-12** on the same grouped data — one definition of
the quantity, not two.

## 4. The variance estimator had to change TWICE

The old moving-block bootstrap drew blocks from the whole concatenated series
and computed a pooled correlation on each resample. That estimates the variance
of the *pooled* statistic, so it was replaced with a **within-fold** bootstrap:
blocks drawn inside each fold, fold ICs averaged exactly as the point estimate
averages them.

**And that first replacement was wrong, in the direction that flatters a
result.** Resampling inside folds says nothing about how much the ICs differ
*between* periods — which is part of the uncertainty of a statistic that is a
mean over periods. Measured on the 84-ticker panel:

| | median `sigma2` |
|---|---|
| within-fold bootstrap alone | 0.00555 |
| between-fold `var(fold ICs) / K` | 0.00797 |
| **ratio** | **1.5×** |

Ignoring the between-fold component graded **12 tickers STRONG**. With it, that
falls to 3. The estimator now takes the **larger of the two**: the between-fold
sample variance is unbiased for the total (each fold IC already carries its own
sampling noise) but very noisy at K = 5, and the bootstrap is the floor under
it so a ticker whose few fold ICs happen to agree cannot buy a near-zero
variance from the coincidence.

Two further consequences, both improvements:

- **`respect_fold_gaps` is gone**, with `contiguous_segments` and
  `_block_start_pool` behind it. They existed so a block could not span the seam
  a dropped fold leaves; the within-fold draw cannot reach a seam at all, so the
  property is structural now rather than optional.
- **A ticker scored on fewer than 3 folds is refused.** A mean over two numbers
  is not a track record, and this binds hardest on the model that most needs it
  — see §6.

## 5. The LIVE gate: measured, and deliberately NOT changed

`agents/critic_agent.grade_evidence` reads `eval_rank_ic` / `eval_rank_ic_t`,
which `pipeline/evaluation.compute_metrics` computes the pooled way. Correcting
it would change `forecast_confidence` on the next weekly run, which is a
production change and out of this session's scope. Running the **real** grader
on both statistics over all 84 tickers:

| grade | on the pooled IC (live today) | on the within-fold IC |
|---|---|---|
| INSUFFICIENT | 82 | 69 |
| WEAK | 2 | 14 |
| STRONG | 0 | 1 |

**16 of 84 tickers change grade** — 14 INSUFFICIENT → WEAK, HINDUNILVR.NS
WEAK → STRONG, BOSCHLTD.NS WEAK → INSUFFICIENT.

That is a real, live defect awaiting a decision. It is also the *unshrunk* gate,
with no FDR control and no minimum-fold rule, so it is more permissive than the
Stage 0 layer by construction — §6 is the better-behaved measurement.

## 6. The re-grade — every variant, corrected statistic, nothing retrained

**Per-ticker IC, old statistic against new.** The first row reproduces the
Stage 0 addendum exactly (−0.0700 pooled / +0.1262 within), which is what makes
the rest readable.

| variant | tickers | pooled-over-folds | positive | within-fold | positive |
|---|---|---|---|---|---|
| per-ticker × mae (production) | 84 | **−0.0700** | 21/84 | **+0.1262** | 51/84 |
| per-ticker × rank_ic (Stage 2a, 14) | 14 | +0.0075 | 8/14 | **+0.1458** | 12/14 |
| pooled × mae | 84 | −0.0512 | 21/84 | **+0.0120** | 46/84 |
| pooled × rank_ic | 84 | −0.0563 | 21/84 | **+0.0490** | 58/84 |
| pooled × mae, no ticker | 84 | −0.0262 | 31/84 | **+0.0127** | 47/84 |
| pooled × rank_ic, no ticker | 84 | −0.0279 | 35/84 | **+0.0092** | 50/84 |

**Every variant flips sign.** That is the bug, in six independent places.

**The grading.**

| variant | n_usable | `mu_hat` | sd | z | `tau2_hat` | STRONG | WEAK | INSUFF |
|---|---|---|---|---|---|---|---|---|
| *Stage 0 as reported (pooled)* | 63 | *−0.05988* | 0.01229 | −4.87 | *0.00221* | 0 | 0 | 84 |
| per-ticker × mae (production) | **10** | +0.15028 | 0.03256 | +4.62 | 0.00460 | **10** | 0 | 74 |
| per-ticker × rank_ic (Stage 2a) | **13** | +0.12866 | 0.02631 | +4.89 | 0.00893 | 6 | 5 | 3 |
| **pooled × mae** | **84** | **+0.01575** | 0.01019 | **+1.55** | 0.00018 | **0** | **0** | **84** |
| pooled × rank_ic | 84 | +0.05640 | 0.00985 | +5.72 | 0.00615 | 3 | 23 | 58 |
| pooled × mae, no ticker | 84 | +0.02102 | 0.01044 | +2.01 | 0.00123 | 0 | 3 | 81 |
| pooled × rank_ic, no ticker | 84 | +0.01282 | 0.00984 | +1.30 | 0.00339 | 0 | 4 | 80 |

### Read the `n_usable` column before the STRONG column

**The two per-ticker rows are survivorship and must not be quoted.** Only **10
of 84** production tickers have three folds carrying an ordering at all; the
other 74 are refused. Those ten are selected on the model having *chosen* to
split — the Stage 0 addendum's caveat on its own +0.1444, now visible in a
column instead of buried in a mean. The Stage 2a row is worse: 13 usable of 14,
because that pilot's whole purpose was to force splits.

**The `pooled × rank_ic` row's 3 STRONG / 23 WEAK is the ticker feature.** Its
no-ticker sibling scores +0.0128 at z +1.30 with **0 STRONG**. Stage 2b measured
the same pair cross-sectionally at +0.0204 with the ticker against +0.0161
without — indistinguishable. So knowing which company it is helps the
**time-series** quantity the gate grades and does nothing for the
**cross-sectional** quantity the product needs. The gate is rewarding exactly
the channel the tradeable measurement says is worthless.

**The best cross-sectional arm grades zero.** `pooled × mae` — reb_t +1.09, the
strongest thing in Stage 2b — returns **0 STRONG, 0 WEAK, `tau2` 0.00018**, and
that is the one row here with no survivorship and no identity channel.

### On the Stage 2a row's provenance

Stage 2a never persisted its held-out predictions, and the signals table has
advanced three sessions since it ran, so the pilot's own stop condition refuses
a reproduction at a drift of 2.53e-04 — correctly. That row is therefore a
**fresh run of the same arm on current data**, not Stage 2a's committed
numbers. `tools/stage2a_pilot.py` now persists predictions so this cannot
recur. (Two attempts also died on `psycopg2 SSL error: unexpected eof` — the
session-pooler drop in §7 — and the recorded remedy applied: read the exported
package, not the database.)

## 7. What is now trustworthy, and what still is not

**Trustworthy.** The direction. Six variants all flip from negative to positive,
the synthetic proof is analytic, and the `baselines.py` audit says the
cross-sectional track was never affected. `mu_hat = −0.05988` should never be
quoted again in any form.

**Not trustworthy — and unchanged by this session.**

1. **The gate grades the wrong QUANTITY.** A per-ticker time-series rank IC
   answers "within this stock, did the dates it ranked higher pay more". Every
   economic statement this project has — `reb_IC`, the break-even 0.00512, the
   portfolio simulator, the deflated Sharpe — is **cross-sectional**. A market
   or regime factor moving all names together produces a positive time-series
   IC for every ticker at once, which is exactly the pattern in the table. That
   is a *design* mismatch, not a calculation error, and Stage 0b did not touch
   it.
2. **`mu_hat`'s standard error is still overstated.** The precision-weighted SE
   treats 84 tickers as independent observations when the panel holds five
   periods and one market. The addendum recorded this; it is why z = +5.72 on
   `pooled × rank_ic` is not a credible number in either direction.
3. **The live gate still pools.** See §5. Sixteen published grades hang on it.
4. **The fold-0-peaks / fold-4-negative pattern** across all six Stage 2b arms,
   including the pure-noise placebo, is untouched and out of scope. Until it is
   explained, any per-fold statistic on this panel — including the corrected
   one — carries an unquantified period effect.

## 8. Recommendation

**Do not read the corrected table as evidence of skill.** The only row without
survivorship or an identity channel is `pooled × mae`, and it grades **0 STRONG,
0 WEAK** at `tau2` 0.00018 — the panel making one statement about itself.

Two things follow, in order.

1. **Fix the live gate, or stop publishing its grades.** It is measurably wrong
   today and 16 of 84 rows move when it is corrected. That is a one-line change
   to `compute_metrics` plus a `MODEL_VERSION` bump, and it is a production
   decision rather than a shadow one.
2. **Then change what the gate grades.** A per-ticker time-series IC is not the
   quantity this product is about and is not what any of its economic bars
   measure. Grading the cross-sectional contribution instead would make the
   evidence layer and the portfolio layer answer the same question for the first
   time — and on the evidence here, that answer is still no.
