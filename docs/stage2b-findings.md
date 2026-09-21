> **CORRECTION NOTICE — 2026-09-21. The numbers below are not reproducible as
> written, and two of the reasons are defects rather than noise.** Nothing here
> is rewritten; see `docs/hygiene-findings.md` for what changed and by how
> much.
>
> 1. **They were produced at an unpinned thread count.** Nothing in the
>    repository pinned XGBoost's `n_jobs`, so every fit ran at whatever this
>    workstation defaulted to (20). Measured: the same code with the same seeds
>    gives different hyperparameters and a different model at a different
>    thread count — zero of 162,535 predictions matching, a maximum difference
>    of 6.2 prediction standard deviations. Re-pinned at `XGB_THREADS = 2`, the
>    30-session baseline moves from cs IC −0.00101 to −0.00423 and from
>    1 STRONG / 1 WEAK / 82 to 2 STRONG / 3 WEAK / 79. **Every verdict here
>    survives — a null that wobbles into another null is still a null — but the
>    digits do not.**
> 2. **They were produced on the RAW label**, which is no longer the pipeline's
>    default training target. `gamma` is denominated in the loss, so the fixed
>    `[0, 5]` search range meant something different at every label scale;
>    the within-date standardised label (`pipeline/label.py`) is the default
>    now, and under it the same architecture emits 0 of 420 constant cells
>    instead of 7, and 84 distinct predictions per date instead of 8.
>
> 3. **Any QUANTILE or BOOK figure here — long-short spread, top-quintile
>    return, alpha, net-of-cost — was computed with a tie guard that refused
>    only a wholly tied cross-section.** The pooled model's predictions are
>    discrete, and 48 of the 64 h=30 books had legs that were MAJORITY chosen
>    by the ticker tie-break rather than by the model. Those columns are
>    withdrawn rather than corrected. **The rank-IC figures are unaffected and
>    reproduce to the digit**, because `rank_ic` averages ranks over ties.
>
> The session that measured all of this changed no conclusion in this
> document.

<!-- stage0c-correction-notice -->
> ## ⚠ CORRECTION NOTICE — partially superseded 2026-09-08
>
> **Section 8's panel diagnostic table is computed on the pooled-across-folds
> IC that Stage 0b then fixed.** Its `mu_hat` figures (-0.052, -0.055, -0.026)
> and the "84 ANTI_SIGNAL" row are artifacts of that statistic, not
> measurements of the pooled model.
>
> The section's ARGUMENT, however, is what opened Stage 0b and it stands
> exactly as written: `mu_hat` barely moved when the constants went away, and
> rho(fold level, fold return) was -0.600 in both. That was the observation that
> located the bug.
>
> **Everything else in this document is unaffected.** The 2x2, the placebo
> arms, the min_train sweep, the effective-sample-size measurement and the
> cross-sectional `reb_IC`/`reb_t` figures all come from
> `cross_sectional_report` and `per_date_rank_ic`, which were audited clean in
> Stage 0b and are within-group-then-averaged.
>
> See [`stage0-closing.md`](stage0-closing.md).

# Stage 2b — pooling removes the degeneracy completely, and finds nothing underneath it

> The first slice of the full Stage 2 (pooled model consolidation), triggered by
> Stage 2a's finding that a per-ticker model choosing nine hyperparameters off
> ~10 effective observations is over-specified whatever metric scores it.
>
> Branch `stage2b-pooled-model`, off `stage2a-tuner-objective-pilot`, **not
> merged**. Separate track from Phase 0-6. Pre-registered before any pooled
> training run: [`stage2b-preregistration.md`](stage2b-preregistration.md).
> Tools: `tools/stage2b_pooled.py`, `tools/stage2b_sweep.py`,
> `tools/stage2b_panel_diagnostic.py`.
>
> **Shadow only.** No hyperparameter cache, no `model_metadata`, no forecast,
> nothing the API or the web app serves. **NO RENDER REDEPLOY.**

---

## 0. What `pooled_xgb` already was

The investigation gate, answered before any new training code was written.

`pipeline/baselines.py::_pooled_xgb_factory` is **already one XGBoost fitted on
the whole pooled panel**, driven through `panel_walk_forward` on the 15
scale-free `FACTORS` columns. It is not an ensemble of per-ticker models and not
a scoring-time pooling. What it lacks is exactly two things:

- it is **UNTUNED**, deliberately — the docstring says so, because a searched
  tree could not be read beside an unsearched ridge without deflating it first;
- it carries **no ticker identity**.

So this session is an audit and an extension, not a rewrite. What Stage 2b adds
is the nested pooled hyperparameter search under both objectives, the ticker
categorical, and the degeneracy / train-gap diagnostics Stage 2a used to catch
overfitting. No duplicate pooled path was created, and
`pipeline/model.py` is untouched.

## 1. Pooling composes with the purge, by construction

`PurgedPanelWalkForward` splits the **shared date grid**, not row positions. A
training row on grid date `i` carries a label spanning `[i, i + horizon]`, so
training ends at `t - horizon - embargo` for the whole cross-section at once.
The pooled-specific failure — one ticker's future reaching another ticker's
training fold *through the pooling* — is therefore unconstructable rather than
merely absent, and `tests/test_leakage.py` now asserts it directly
(`max(train_dates) < min(test_dates)` across the panel, plus the grid-measured
purge gap, plus the same two checks on the inner CV that scores each trial).

Nothing had to change to make this so. The splitter was already correct; what
was missing was a test saying so at panel altitude.

The ticker feature needs **no encoding**: XGBoost 3.2 reads a pandas category
natively (`enable_categorical=True`), so there is no fitted statistic that could
be computed with sight of held-out rows. A test pins the version assumption.

## 2. The effective sample size behind one hyperparameter decision

Measured, not assumed — and measured at the **inner-CV altitude**, because that
is where a trial is selected, not where the winner is reported.

| | per-ticker (Stage 2a) | pooled (Stage 2b) |
|---|---|---|
| training rows in the scored block | 270 | **16,968** |
| dates | — | 202 (84 names/date) |
| n_eff, rows / horizon | 9.0 | 566 |
| n_eff, **dates** / horizon | 9.0 | **6.7** |
| **null SE of the rank-IC objective** | **0.0579** | **0.0078** |

400 moving-block (30-session) surrogate predictions, so the null keeps a real
prediction's persistence; an iid surrogate would understate both spreads and
flatter the pooled arm most.

**The objective's standard error falls 7.4×** — which is the quantity H2 rests
on, and it is real. But note the row that matters for interpretation:
**pooling buys BREADTH WITHIN A DATE, not independent time windows.** The
count of non-overlapping windows actually goes *down* (6.7 against 9.0). The
cross-sectional IC is a within-date statistic, so it benefits; anything that
depends on independent periods does not, and that is the ceiling this run
eventually runs into.

## 3. The 2×2, and the four controls that make it readable

Each cell on its own protocol. The per-ticker × mae cell is the production
baseline read from the Stage 0 out-of-sample cache, not recomputed.

| cell | constant cells | degeneracy | reb IC | **reb t** | cross-sec IC | time-series IC | OOS MAE | train MAE | gap |
|---|---|---|---|---|---|---|---|---|---|
| **per-ticker × mae** (production) | 316/420 | **75%** | — | — | −0.0143 | +0.1363 | 0.09418 | — | — |
| **per-ticker × rank_ic** (Stage 2a, 14 tickers) | 13/70 | 19% | — | — | — | +0.1298 | 0.11600 | 0.07981 | **+0.03619** |
| **pooled × mae** | 6/420 | **1%** | +0.0258 | **+1.09** | +0.0221 | +0.0125 | 0.08919 | 0.08488 | +0.00431 |
| **pooled × rank_ic** | 0/420 | **0%** | +0.0204 | +0.86 | +0.0121 | +0.0490 | 0.09001 | 0.08299 | +0.00702 |
| pooled × mae, no ticker | 7/420 | 2% | −0.0015 | −0.09 | −0.0013 | +0.0141 | 0.08882 | 0.08586 | +0.00297 |
| pooled × rank_ic, no ticker | 0/420 | 0% | +0.0161 | +0.76 | +0.0149 | +0.0092 | 0.08992 | 0.08395 | +0.00597 |
| pooled × rank_ic, **placebo ticker** | 0/420 | 0% | +0.0186 | +0.88 | +0.0122 | +0.0053 | 0.08958 | 0.08422 | +0.00536 |
| pooled × mae, **placebo ticker** | 3/420 | 1% | −0.0053 | −0.31 | −0.0009 | −0.0030 | 0.08866 | 0.08609 | +0.00258 |
| *untuned `pooled_xgb`, for reference* | 0/420 | 0% | +0.0213 | **+1.14** | +0.0092 | +0.0199 | 0.09293 | 0.07958 | +0.01335 |

64 non-overlapping rebalances, 0 dates without an ordering, in every pooled row.

**`reb IC` and `cross-sec IC` are different samples and only the first carries a
t.** `cross-sec IC` averages every out-of-sample date; consecutive dates share
29 of their 30 forward sessions, so a t on it would be inflated roughly 5×.
`reb IC` is the mean over the 64 non-overlapping rebalance dates. CLAUDE.md
records these two as having carried opposite signs before, so they get separate
columns.

The per-ticker × rank_ic row is Stage 2a's pilot and covers **14 tickers, not
84**. Its MAE and gap are comparable in kind; a cross-sectional IC over a
14-name cross-section is not comparable to one over 84 and is deliberately left
blank rather than filled with a number that invites the comparison.

## 4. What the 2×2 says

**Prediction 1 was right, and by more than it claimed.** Pooling alone removes
the degeneracy: **316/420 constant cells become 6/420 under the unchanged MAE
objective**, and 0/420 under rank IC. The untuned `pooled_xgb` that has been in
`baselines.py` all along already emits **zero** constants. So the degeneracy was
never a property of the objective — it is a property of the **altitude**. At
~400 rows the total loss reduction any split can offer is smaller than a
`gamma` of 2 demands; at ~100,000 rows the same `gamma` is easily paid. The
selected gammas under pooling are still high (0.17 to 4.93) and the trees split
anyway.

**Prediction 2 — the pre-registered decision rule — narrowly FAILS.** Against
pooled × mae on identical rows, pooled × rank_ic gives:

| clause | threshold | measured | verdict |
|---|---|---|---|
| OOS MAE no more than +2% worse | ≤ +2.0% | **+0.9%** | met |
| train/OOS gap grows no more than 1.5× | ≤ 1.5× | **1.63×** | **not met** |

Both had to hold. So the Stage 2a overfitting signature is **attenuated but not
gone**: the MAE regression collapses from +6.7% to +0.9%, which is the
sample-size effect working, while the gap still widens by more than the
pre-registered allowance. The honest reading is that H2 is **mostly right and
not entirely** — estimation noise was the dominant cause of Stage 2a's failure,
and something smaller survives it.

**And the objective change buys nothing anyway.** Tuning on the cross-sectional
rank IC produces a *worse* out-of-sample cross-sectional rank IC than tuning on
MAE (+0.0121 against +0.0221), and a worse rebalance IC (+0.0204 against
+0.0258). Optimising the metric you are graded by does not help here.

**Tuning itself buys nothing.** The untuned `pooled_xgb` scores the best
rebalance t in the table (**+1.14**), ahead of both searched arms. Twenty-five
years of hyperparameter folklore against 64 independent windows.

## 5. The ticker feature is the §7 landmine, and the placebo says so from both sides

Under MAE the ticker feature is the **entire** cross-sectional result:

| pooled × mae | reb IC | reb t |
|---|---|---|
| with the real ticker | **+0.0258** | +1.09 |
| with the ticker removed | −0.0015 | −0.09 |
| with the ticker **placebo-shuffled** | −0.0053 | −0.31 |

The placebo — labels permuted within each date, so cardinality and per-date
frequency are identical and only the name↔return link is broken — scores the
same as having no ticker at all. So the +0.0258 is not "an 84-level categorical
gives the tree something to split on". It is specifically **knowing which
company it is**.

**That is precisely the landmine CLAUDE.md §7 records**: two random per-ticker
constants — carrying zero information about returns — scored `pooled_xgb` at a
mean rebalance **t of +0.77** over 24 draws (sd 0.49, max +1.77), because the
tree identifies the name and learns which names paid in the training window,
and over a 4.4-year panel that persists into the test folds.

**So the within-date placebo run here is the WRONG null for this question, and
the right one is already on file.** A shuffled label cannot reproduce the effect
by construction, because the effect *is* persistent identity. Measured against
the recorded null instead, pooled × mae's **+1.09 sits below the +0.77 mean plus
one sd**, and well inside the observed maximum of +1.77. It is not evidence of
anything.

Under the rank-IC objective the picture is different and equally deflating: the
real ticker (+0.0204), no ticker (+0.0161) and the **pure-noise placebo**
(+0.0186) are indistinguishable. **A column of noise captures more than half of
what the "real" feature is worth**, which is what noise looks like.

## 6. The early-fold pattern, for the sixth time — and now in the placebo too

Cross-sectional IC by fold:

| cell | fold 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| per-ticker × mae | +0.0954 | −0.1300 | −0.0713 | +0.0167 | +0.0175 |
| pooled × mae | **+0.1062** | +0.0240 | −0.0386 | +0.0123 | +0.0066 |
| pooled × rank_ic | **+0.1034** | +0.0240 | −0.0394 | −0.0201 | −0.0073 |
| pooled × mae, no ticker | +0.0348 | −0.0009 | −0.0113 | −0.0124 | −0.0165 |
| pooled × rank_ic, no ticker | +0.0698 | −0.0102 | +0.0375 | −0.0046 | −0.0182 |
| pooled × rank_ic, placebo | **+0.0704** | −0.0083 | +0.0259 | −0.0077 | −0.0192 |
| pooled × mae, placebo | +0.0325 | −0.0083 | −0.0115 | −0.0001 | −0.0168 |

Every arm peaks in fold 0 and every arm is **negative in fold 4**. Valuation
(+3.32), LoRA (+2.37), `pooled_xgb` (+2.42), P3's `regime_factor` and P5's
`linear_factor` were all carried by the earliest fold; this is the sixth
instance and the first where it also appears in an arm whose only extra feature
is **pure noise** (+0.0704 in fold 0 for the rank-IC placebo). That is worth
recording on its own: fold 0's apparent IC is partly a property of the fold, not
of the model in it.

## 7. The min_train sweep — prediction 3 holds, nothing clears anything

Standing policy: a result measured at one hyperparameter setting is not a
result. The non-overlapping rebalance IC and its t, across the grid:

| | 380 | 440 | **500** | 560 | 620 |
|---|---|---|---|---|---|
| pooled × mae, reb IC | +0.0202 | +0.0109 | **+0.0258** | +0.0191 | +0.0253 |
| pooled × mae, **reb t** | +0.99 | +0.47 | **+1.09** | +0.78 | +1.03 |
| pooled × rank_ic, reb IC | +0.0228 | +0.0134 | **+0.0204** | +0.0124 | +0.0234 |
| pooled × rank_ic, **reb t** | +1.12 | +0.58 | **+0.86** | +0.50 | +0.95 |
| pooled × mae, constant cells | 8 | 5 | 6 | 11 | 12 |
| pooled × rank_ic, constant cells | 0 | 5 | 0 | 3 | 3 |
| rebalances | 68 | 66 | 64 | 62 | 60 |

**The largest t anywhere in the grid is +1.12.** The pre-registered Phase 2 bar
is a positive rebalance IC with **t > 2**, and nothing comes close. There is no
shape either — the profile alternates rather than varying smoothly, which is
what noise looks like, not an effect.

The rebalance ICs do sit above the break-even of **0.0051** at zero market
impact, and below the 0.0166 required at 25bp. But P4 already recorded that
"clearing the bar is not the hard part; clearing it with something other than
beta is" — and here the whole of the MAE arm's ordering comes from the ticker
feature, which is the persistent-identity channel, not a company view.

## 8. THE PANEL DIAGNOSTIC OVERTURNS THE STAGE 0 ADDENDUM'S ATTRIBUTION

Stage 0's statistics, recomputed on each model's own held-out predictions
through the unmodified `pipeline/evidence_shrinkage.py`. The first row
reproduces Stage 0 **exactly** — `mu_hat` −0.05988189, `tau2` 0.00220935,
n_usable 63 — which is what makes the rows below it readable.

| model | n_usable | `mu_hat` | sd | z | `tau2_hat` | STRONG | WEAK | ANTI | INSUFF |
|---|---|---|---|---|---|---|---|---|---|
| per-ticker × mae (Stage 0) | 63 | **−0.05988** | 0.01229 | −4.87 | 0.00221 | 0 | 0 | 0 | 84 |
| pooled × mae | **84** | **−0.05219** | 0.01007 | −5.19 | 0.00267 | 0 | 0 | 0 | 84 |
| pooled × rank_ic | **84** | **−0.05456** | 0.00961 | −5.68 | 0.00034 | 0 | 0 | **84** | 0 |
| pooled × mae, no ticker | **84** | −0.02574 | 0.00997 | −2.58 | 0.00098 | 0 | 0 | 0 | 84 |

**`mu_hat` does not move.** The pooled model emits **zero constant cells**, and
the panel statistic Stage 0 measured at −0.05988 comes back at **−0.05219**.

The Stage 0 addendum attributed that number to the degeneracy — *"three
quarters of fold-level fits emit one repeated number, so the pooled rank IC
ranks rows substantially by WHICH FOLD THEY CAME FROM — five constants against
five period returns."* **That attribution is wrong, and this is the measurement
that says so.** Removing every constant leaves the number essentially where it
was.

The mechanism is the fold **level**, and constancy was only its most extreme
form. Measured on the pooled model's own predictions:

| | per-ticker IC pooled over folds | per-ticker IC within folds | ρ(fold prediction level, fold realised return) |
|---|---|---|---|
| per-ticker × mae (Stage 0) | −0.0700 | +0.1262 | **−0.600** |
| **pooled × mae** | **−0.0512** (21/84 positive) | **+0.0120** (46/84) | **−0.600** |
| pooled × rank_ic | −0.0563 (21/84) | +0.0490 (58/84) | −0.300 |
| pooled × mae, no ticker | −0.0262 (31/84) | +0.0127 (47/84) | −0.300 |

Fold prediction levels run `+0.0217, −0.0103, +0.0303, +0.0210, +0.0253`
against realised fold returns of `−0.0078, +0.0572, +0.0204, +0.0243, +0.0145`.
The fold the model was least optimistic about is the fold that paid most. **Five
folds, one market, and a rank correlation of −0.600 computed over five points.**

**So Stage 0's `mu_hat` is not a measurement of any model — it is a property of
the walk-forward protocol on this panel.** Any fold-wise refitted model, degenerate
or not, inherits it, because every fold produces its own prediction level and
those five levels happen to run against the five period returns. `mu_hat` would
have come back near −0.05 for a model with no constants at all, which is
precisely what has now been measured.

**And the ANTI_SIGNAL row must not be read as a finding.** `pooled × rank_ic`
grades all 84 tickers ANTI_SIGNAL at `tau2_hat` = 0.00034 — near the degenerate
limit `PanelGrading.degenerate` exists to flag. At that `tau2` the shrinkage
weight is ~1 for every ticker, so every posterior mean IS the grand mean and the
whole board carries one identical grade. It is the panel making a single
statement about a protocol artifact and printing it 84 times.

The one row that does move is **pooled × mae without the ticker feature**, at
−0.0258 against −0.0522 with it — so half of the apparent "anti-signal" is the
ticker feature's persistent-identity channel running the wrong way once pooled
across folds.

## 9. What Stage 2b changes, and what it does not

**Answered.** The degeneracy is an **altitude** problem, not an objective
problem. It is gone — 75% of cells to 1% — under the unchanged production
objective, and the untuned `pooled_xgb` that has been in `baselines.py` since
Phase 2 never had it at all.

**Mostly answered.** H2 was mostly right: pooling cuts the objective's null
standard error 7.4× and shrinks Stage 2a's MAE regression from +6.7% to +0.9%.
It is not entirely right: the train/OOS gap still widens 1.63× against a
pre-registered allowance of 1.5×, so the pre-registered rule **fails**, narrowly
and on one clause of two.

**Overturned.** The Stage 0 addendum attributed `mu_hat = -0.05988` to the
fold-level constants. With zero constants the same statistic reads **-0.052**,
and ρ(fold prediction level, fold realised return) is **-0.600** in both. The
number is a property of the walk-forward protocol on this panel, inherited by
any fold-wise refitted model. It was never a measurement of a model.

**Not established, and it is the point.** That any of this forecasts anything.
The largest rebalance t across a ten-cell sweep is **+1.12**; the untuned
comparator beats both searched arms; the objective change makes the metric it
optimises *worse*; and every arm — including one whose only extra column is pure
noise — is carried by fold 0 and negative in fold 4.

## 10. Recommendation

**Neither pooled variant should go into a shadow-mode evidence-grading rerun as
it stands, and the reason is not that they are weak — it is that the grader
would be reading a protocol artifact.**

Section 8 is decisive on this. `grade_evidence` and Stage 0's shrinkage layer
both consume a **per-ticker rank IC pooled over folds**, and that quantity
carries a −0.05 bias on this panel for reasons that have nothing to do with the
model. Feeding it a non-degenerate model does not fix it: `pooled × rank_ic`
grades **all 84 tickers ANTI_SIGNAL** at a `tau2` of 0.0003, which is the panel
making one statement about the protocol and printing it 84 times.

The fix is in the grader's input, and it is cheap: **score each ticker WITHIN
its fold and average, rather than pooling across folds.** That is the same
correction `_mean_daily_rank_ic` already applies at the panel level and the same
one the Stage 0 addendum's within-fold analysis used. Until that lands, a
grading rerun on any model measures the folds.

**What is worth carrying forward from Stage 2b is the pooled model itself, not
either objective.** It removes a defect that has contaminated every per-ticker
number this project has produced, at no cost in MAE (0.09418 → 0.08919, 5.3%
better). Whether it forecasts anything remains unanswered, and Stage 2b's own
evidence says it does not.
