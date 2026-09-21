# P6 — the horizon sweep: findings

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

**Run 2026-09-20, against `docs/p6-preregistration.md`** (sha256
`7814bfc9…c73f4`, written and hashed at 19:18:19, before anything ran on real
data). `tools/p6_horizon.py`, B = 1000, 10 trials, nine placebo retrains per
Part B arm, DK lags 30, Politis-White floor 63.

The pre-registration is a dated addendum to P6's original terms of 2026-09-05,
which lived only in the gitignored CLAUDE.md; §1 there reproduces them.

## The answer, in three lines

**No horizon in {5, 10, 20, 30} reaches the pre-registered bar.** The largest
t is +1.02, there is no profile, and an arbitrary purge choice moves the
statistic as far as the horizon does.

**The two shortest horizons did not produce a model capable of failing it.**
At h=5 the MAE-tuned pooled model emits five distinct numbers across 162,535
rows — one constant per fold — because `gamma` is denominated in the loss and
the label's dispersion falls with the horizon. That is §4, and it is the
finding.

**Fixed, the horizon still is not the missing axis.** Refit on a within-date
standardised label the degeneracy disappears completely and the
cross-sectional IC is flat at +0.009 to +0.018 across all four horizons —
largest at h=20, not at h=5. Neither SUE nor delivery % adds anything at five
sessions. One cell reached t +2.41 and did not survive the skeptic pass; what
killed it is new, and is in §7.2.

---

## 1. A0 — the regression pin

The 30-session baseline is unchanged. Re-run through the horizon-parameterised
`run_arm` with `horizon=30, purge=30` — the legacy rule — on the STORED label:

| | |
|---|---|
| rows stored / re-run | 160,435 / 160,435 |
| unmatched either way | 0 / 0 |
| max prediction drift | **0.0e+00** |
| max label drift | **0.0e+00** |
| per-fold gammas | 1.792, 0.290, 4.934, 4.934, 4.934 |

Those are the same five gammas Pilot 1 recorded, and the same 160,435 rows
Pilots 1 and 3 reproduced. **Parameterising the splitter changed nothing when
the parameters are set to what they used to be hardcoded at.** Prediction PA1
held, at drift exactly zero.

---

## 2. Why the sweep relabels at h=30 too

The sweep's four cells all take their label from `panel.retarget_horizon`,
including h=30, so the grid shares one label construction. That is not the same
as the stored label: the stored 30-session label was computed from
full-precision closes, and `retarget_horizon` rebuilds it from closes stored to
three decimals. **They differ by up to 5.0e-06** on a label of order 0.1 —
5e-05 relative.

That is far smaller than anything measured here, and "far smaller than the
effect" is precisely the argument that retired three results in this project,
so it is not used as one. The pin runs on the stored label because it is the
only way to reach 1e-9; the sweep runs on the rebuilt one because four cells
sharing one construction is worth more than one cell matching history. The same
5e-06 forced the Pilot 2 phantom check to widen its tolerance from 1e-9 to
1e-5, and it is the same cause.

---

## 3. Part A — the sweep

All eight cells, both purge rules. `cs IC` is the mean per-date cross-sectional
rank IC; its SE is a Driscoll-Kraay SE at 30 lags computed directly on that
series, which is more conservative than the kernel-bandwidth version
`grade_panel_v3` reports (Pilot 1 quoted 0.00811 for the same arm where this
column reads 0.01201). One estimator is used throughout, so the columns are
internally comparable.

| h | rule | purge | OOS rows | **dates with an ordering** | **cs IC (DK SE)** | **t** | reb IC | reb t | n reb | constant cells | S / W / I | mu_hat | z | tau2 | PW auto | break-even IC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | legacy | 5 | 162,535 | **383 / 1,935** | +0.00256 (0.01286) | **+0.20** | +0.0082 | +0.58 | 76 / 387 | 340/420 | 0 / 0 / 84 | n/a | n/a | 0.00000 | 11.5 | 0.0152 |
| 5 | **deciding** | 63 | 162,535 | **0 / 1,935** | n/a | **n/a** | n/a | n/a | **0 / 387** | **420/420** | 0 / 0 / 84 | n/a | n/a | 0.00000 | n/a | 0.0152 |
| 10 | legacy | 10 | 162,115 | 1,154 / 1,930 | +0.01436 (0.01537) | **+0.93** | +0.0104 | +0.63 | 115 / 193 | 170/420 | 0 / 0 / 84 | −0.01633 | −1.01 | 0.00000 | 26.9 | 0.0117 |
| 10 | **deciding** | 63 | 162,115 | 388 / 1,930 | +0.01367 (0.01395) | **+0.98** | +0.0200 | +1.18 | 39 / 193 | 339/420 | 0 / 0 / 84 | n/a | n/a | 0.00000 | 14.5 | 0.0117 |
| 20 | legacy | 20 | 161,275 | 1,532 / 1,920 | −0.00333 (0.01182) | **−0.28** | −0.0054 | −0.34 | 76 / 96 | 96/420 | 0 / 2 / 82 | −0.00862 | −0.65 | 0.00375 | 47.6 | 0.0103 |
| 20 | **deciding** | 63 | 161,275 | 1,552 / 1,920 | +0.01479 (0.01448) | **+1.02** | +0.0171 | +0.96 | 78 / 96 | 88/420 | **1** / 3 / 80 | −0.00208 | −0.16 | 0.00337 | 5.5 | 0.0103 |
| 30 | legacy | 30 | 160,435 | 1,910 / 1,910 | −0.00101 (0.01201) | **−0.08** | −0.0014 | −0.09 | 64 / 64 | **7/420** | 1 / 1 / 82 | −0.01728 | −1.24 | 0.00435 | 35.8 | 0.0061 |
| 30 | **deciding** | 63 | 160,435 | 1,910 / 1,910 | +0.00899 (0.01332) | **+0.67** | +0.0113 | +0.61 | 64 / 64 | 5/420 | 0 / 1 / 83 | −0.01813 | −1.33 | 0.00356 | 48.8 | 0.0061 |

**A1: no horizon reaches t ≥ +2.0, on either rule. The largest is +1.02.**
A4's min_train sweep was therefore not triggered, as pre-registered.

**The h=30 legacy cell reproduces the established baseline**, on a label
rebuilt from scratch: cs IC −0.00101 against Pilot 1's −0.00101, reb IC
−0.0014 against −0.0014, reb t −0.09 against −0.09, 7/420 constant cells
against 7/420, 1 STRONG / 1 WEAK / 82, mu_hat −0.01728 against −0.01727, tau2
0.00435 against 0.00435, and a Politis-White block of **35.8** against the
35.8 Stage 0c recorded for this architecture. The 5e-06 label difference of §2
moves nothing.

### 3.1 The profile, judged as rule A2 requires

| | h=5 | h=10 | h=20 | h=30 |
|---|---|---|---|---|
| cs IC, legacy | +0.0026 | +0.0144 | −0.0033 | −0.0010 |
| cs IC, deciding | n/a | +0.0137 | +0.0148 | +0.0090 |
| reb IC, legacy | +0.0082 | +0.0104 | −0.0054 | −0.0014 |
| reb IC, deciding | n/a | +0.0200 | +0.0171 | +0.0113 |

**No monotone trend in either direction on either rule** (PA3 held), and no
single peak that survives the other rule: the legacy profile peaks at h=10 and
goes negative at h=20, while the deciding profile is flat from h=10 to h=30.
Nothing here is a shape.

**And the purge width moves the statistic as much as the horizon does.** At
h=20 the cs IC is −0.0033 on the legacy rule and +0.0148 on the deciding rule —
a swing of 0.018 from changing nothing but an arbitrary methodological choice,
against a total spread of −0.003 to +0.015 across the entire sweep. Whatever
is being measured here, its sensitivity to the purge is the same size as its
sensitivity to the thing under test. That is what noise looks like, and it is
the most economical summary of the table.

It also refutes half of PA4. The two rules agree on every VERDICT — none
signals — but the deciding rule's IC is the **stronger** of the two at h=20 and
h=30, not the weaker, even though it trains on 66 fewer dates in fold 0.

### 3.2 The grades are not consistent across cells

| cell | graded STRONG or WEAK |
|---|---|
| h=20, legacy | SOLARINDS.NS, TATACONSUM.NS |
| h=20, deciding | BAJAJHLDNG.NS, M&M.NS, VBL.NS, ZYDUSLIFE.NS |
| h=30, legacy | ICICIBANK.NS, NESTLEIND.NS |
| h=30, deciding | ICICIBANK.NS |

Eight cells, nine grades, and **exactly one name appears twice** — ICICIBANK,
in the two h=30 cells, which score the same rows. The h=20 rules, also scoring
the same rows, share none. Stage 0c's nine-draw placebo established that a
STRONG count of 1 is what an alpha-0.10 familywise procedure hands out on
noise; PA5 held at every cell, and the non-overlap is the second witness.

### 3.3 The early-fold shape, for the eighth time

Cross-sectional IC by fold, deciding rule:

| h | fold 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 20 | **+0.0375** | +0.0004 | +0.0371 | −0.0159 | n/a |
| 30 | **+0.0406** | +0.0058 | +0.0252 | −0.0132 | −0.0154 |

Fold 0 is the largest positive cell at both horizons and fold 4 is negative at
both. This project has now recorded that shape in valuation, LoRA, pooled_xgb,
the news comparator, P5's residuals, Stage 2b's placebo and both Stage 1
reversal arms. It is a property of the early panel, not of anything tested in
it.

### 3.4 The economic bar rises steeply as the horizon shortens

Re-estimated per horizon from that horizon's own rows, by planting edges of
known size, exactly as P4's `break_even_ic` requires. The turnover term is
held at P4's 0.80 here, so these are comparable across the grid and across
P4; a per-cell figure at each arm's OWN measured turnover is in the skeptic
pass of §7.

| h | spread per unit of IC | break-even rank IC | rebalances per year |
|---|---|---|---|
| 5 | 0.117 | **0.0152** | 50.4 |
| 10 | 0.152 | 0.0117 | 25.2 |
| 20 | 0.173 | 0.0103 | 12.6 |
| 30 | 0.292 | **0.0061** | 8.4 |

The h=30 figure of 0.0061 sits next to P4's independently measured 0.00512,
which is a useful agreement given the two are estimated by different methods.
**The bar is 2.5x higher at h=5 than at h=30**, from both directions at once:
six times the rebalances, and a spread per unit of IC that falls with the
return's own size. So even the largest cs IC in the table, +0.0148 at h=20,
sits only 1.4x its own break-even before any market impact — and it is not
distinguishable from zero.

---

## 4. The thing that actually happened: the tuner switches the model off as the label shrinks

This is the finding, and it was not predicted by anything in the
pre-registration.

**Count the distinct prediction values each cell produced, over its whole
out-of-sample set:**

| cell | OOS rows | **distinct predicted values** | constant (ticker, fold) cells |
|---|---|---|---|
| h=5, deciding | 162,535 | **5** | 420 / 420 |
| h=5, legacy | 162,535 | **9** | 340 / 420 |
| h=10, deciding | 162,115 | **9** | 339 / 420 |
| h=10, legacy | 162,115 | 1,198 | 170 / 420 |
| h=20, legacy | 161,275 | 5,397 | 96 / 420 |
| h=20, deciding | 161,275 | 34,980 | 88 / 420 |
| h=30, legacy | 160,435 | 16,774 | **7 / 420** |
| h=30, deciding | 160,435 | **63,920** | 5 / 420 |

**Five distinct numbers across 162,535 rows is one constant per fold.** At h=5
under the deciding rule the model is not a weak predictor — it is not a
predictor. Every cross-sectional rank IC it could produce is undefined, 0 of
387 rebalance dates carry an ordering, and every statistic in that row is NaN.
That is not a null. It is a non-measurement, and it would have been reported as
a null by any table that showed only the IC.

### 4.1 The mechanism, measured

`gamma` is the minimum loss reduction a split must buy, **in the units of the
loss.** For an MAE objective those units are the label's own scale, and the
label's scale falls with the horizon, roughly as sqrt(h) on a near-random-walk
target:

| h | label sd | mean abs label | within-date sd |
|---|---|---|---|
| 5 | 0.04760 | 0.03342 | 0.03859 |
| 10 | 0.06782 | 0.04808 | 0.05467 |
| 20 | 0.09732 | 0.06976 | 0.07770 |
| 30 | 0.11964 | 0.08718 | 0.09536 |

**The tuner searches `gamma` over a fixed [0, 5] at every horizon.** So a gamma
of 3 that was merely restrictive at h=30 is 2.5x more binding at h=5 — and the
search scores on MAE, for which a constant is near-optimal, so it walks to the
top of the range and stays there. The selected gammas say it outright:

| cell | per-fold gamma |
|---|---|
| h=30, legacy (the A0 pin) | 1.792, 0.290, 4.934, 4.934, 4.934 |
| h=5, legacy | 1.943, 1.943, 3.188, 0.917, 0.290 |
| h=5, deciding | 4.934, 4.744, 3.188, 3.876, … |

And the degeneracy is **monotone in the horizon** on the legacy rule — 7, 96,
170, 340 constant cells at h = 30, 20, 10, 5 — which is what a scale mechanism
predicts and a coincidence does not.

### 4.2 It is the Stage 2a/2b landmine on a third axis

Stage 0 found 316 of 420 cells emitting a constant and blamed the objective.
Stage 2a changed the objective and showed that was not the fix: it bought the
splits with a +6.7% MAE regression and a 2.5x wider train/test gap. Stage 2b
found the real axis — **altitude**: pool the rows and the same gamma is easily
paid, because the total loss reduction available scales with the sample. The
recorded landmine reads "a constant-prediction model is a symptom of altitude,
not of the loss."

P6 adds the third axis: **label SCALE.** Pooling is already in force here and
the row count has not changed — what shrank is the loss each row can offer. So
the landmine needs an amendment, and it is the reusable output of this session:

> **`gamma` is denominated in the loss, so a fixed [0, 5] search range means a
> different thing at every label scale.** Shorten the horizon — or move to any
> smaller-magnitude target — and the same search silently turns the model off.
> A change to the target's scale must be accompanied by a rescaled `gamma`
> range, a standardised label, or a scale-free objective.

### 4.3 What it does to the sweep

It makes h=5 (both rules) and h=10 (deciding) **uninterpretable as tests of the
hypothesis they were run to test.** "No signal at h=5" and "no model at h=5"
produce the same table, and Part A alone cannot tell them apart. That is the
same problem Pilot 3's announcement check exists to rule out, arriving from the
other direction: there, a drift null from a surprise the market ignored would
have been uninterpretable; here, a horizon null from a model that never split
is uninterpretable.

**So A1's "no horizon reaches t ≥ +2.0" is true and is not the whole answer.**
At h=20 and h=30 it is a real null from a real model. At h=5 and h=10 it is
mostly an artefact of the tuner. Part C is the controlled version, and it was
added for exactly this reason — after Part A, descriptively, deciding nothing.

---

## 5. The purge floor, judged after the fact

The pre-registration fixed purge = embargo = `max(h, 63)` and justified 63 on
Stage 0c's Politis-White measurements, arguing the panel's dependence is a
property of the panel and so does not shrink with the label. Prediction PA6
said the recomputed figure would stay in the 30–60 band at every horizon and
not track h.

**PA6 is wrong on both clauses.** Recomputed on each cell's own realised IC
series:

| h | legacy | deciding |
|---|---|---|
| 5 | **11.5** | n/a (no ordering) |
| 10 | 26.9 | 14.5 |
| 20 | 47.6 | **5.5** |
| 30 | **35.8** | 48.8 |

Two things are true at once and neither was anticipated:

1. **The dependence does track the horizon**, at least at the short end. 11.5
   at h=5 and 26.9 at h=10 are nowhere near 63, and they are ordered with h.
   The label overlap is clearly part of what Stage 0c was measuring, not merely
   a floor beneath a panel-wide constant.
2. **The estimator is unstable on these samples.** At h=20 the same rows give
   47.6 under one purge rule and 5.5 under the other; at h=30 they give 35.8
   and 48.8. A statistic that moves by 9x when only the purge changes is not
   resolving a parameter.

**What that costs.** The floor of 63 is above every figure in the table, so it
is conservative everywhere — which is the right direction to be wrong in. But
it is 5.5x the measured dependence at h=5, and it is what took the h=5 cell
from one ordered fold (legacy, 490 training dates in fold 0) to none (deciding,
374). The pre-registered rule made the h=5 cell unmeasurable, and it did so on
the strength of a number measured at a different horizon.

**It is not switched after the fact.** The deciding rule was hashed before the
run and both rules were run at every horizon precisely so this could be seen
rather than argued about. The honest statement is that at h=30 and h=20 the
choice is defensible and changes no verdict, and at h=5 and h=10 the
pre-registered floor was the wrong call — which is a fact about the
pre-registration, recorded, not a licence to read the legacy column instead.
A future sweep should derive the floor per horizon from the label structure
being tested, and accept that it cannot be derived from the run it constrains.

---

## 6. A defect the degeneracy exposed: the tie guard is binary when it should measure breadth

`rebalance_books` skips a date whose predictions are **entirely** tied, and
breaks partial ties on a hash of the ticker. That guard exists because a stable
sort on tied predictions once turned `zero`, `train_mean` and `majority` into
real-looking portfolios earning +0.00914 alpha at t +1.19 — the return of
holding the alphabetically-first fifth of the universe. The landmine reads: **a
prediction with no ordering must earn no ranking result.**

**It does not catch a prediction with almost no ordering.** Measured on this
sweep's own stored predictions:

| cell | dates with 1 distinct prediction | dates with 2–3 | books formed | **mean turnover** | mean net / rebalance |
|---|---|---|---|---|---|
| h=5, deciding | 1,935 / 1,935 | 0 | **0** | n/a | n/a |
| h=5, legacy | 1,552 / 1,935 | 117 | 76 | **0.105** | +0.00110 |
| h=10, deciding | 1,542 / 1,930 | 388 | 39 | **0.034** | −0.00016 |
| h=30, deciding | **0** / 1,910 | 62 | 64 | 0.485 | +0.00737 |

A date on which 84 names take **two** distinct predicted values is not fully
tied, so it passes the guard — and then the quintile boundaries fall inside a
block of 40-odd identical numbers, where the ticker hash decides who is in the
book. The result is a nearly static portfolio: turnover 0.105 at h=5 and
**0.034** at h=10, against 0.485 for the real h=30 model. Those books are not
strategies. They are one arbitrary fifth of the universe, held.

**Why it matters here.** R4 charges cost on turnover and compares an arm's net
return to the baseline's. At h=5 the baseline's book is a hash-ordered
constant, so an R4 comparison against it measures which arbitrary fifth each
arm happened to freeze on. The pre-registered Part B at h=5 inherits that, and
it is one of the two reasons its verdicts are reported but not relied on.

**RECORDED, NOT FIXED.** `rebalance_books` and `cross_sectional_report` are in
`pipeline/evaluation.py`, which the weekly job imports, and P6 is research. The
fix is a breadth test rather than a binary one — refuse a date with fewer than
some minimum of distinct predicted values, in the same place
`MIN_CROSS_SECTION` already refuses a thin cross-section — and it would change
`n_rebalances` on historical tables, so it needs a deliberate decision rather
than a quiet edit. **No table in this project before P6 is affected**, because
no pre-P6 pooled arm was ever this degenerate: the h=30 baseline has zero fully
tied dates and 7 constant cells of 420.

The landmine should read: *a prediction with no MEANINGFUL ordering must earn
no ranking result, and "no meaningful ordering" is a count of distinct values,
not a test for all-equal.*

---

## 7. Part C — the scale diagnostic, and what it found instead

Post hoc, added after Part A, pre-registered as nothing. Each horizon refitted
on a WITHIN-DATE STANDARDISED label — a positive affine map inside each date,
so it changes no cross-sectional ranking and no per-date rank IC, only the
scale `gamma` is measured against — and scored against the REAL h-session
label.

| h | constant cells | no-ordering dates | n reb | **cs IC (DK SE)** | **t** | reb IC | reb t |
|---|---|---|---|---|---|---|---|
| 5 | **0 / 420** | 0 | 387 | +0.01459 (0.00607) | **+2.41** | +0.0137 | +1.92 |
| 10 | **0 / 420** | 0 | 193 | +0.00896 (0.00905) | +0.99 | +0.0122 | +1.10 |
| 20 | **0 / 420** | 0 | 96 | +0.01758 (0.01229) | +1.43 | +0.0148 | +0.92 |
| 30 | **0 / 420** | 0 | 64 | +0.01524 (0.01398) | +1.09 | +0.0292 | +1.40 |

**The mechanism is confirmed outright. Standardising the label removes every
constant cell at every horizon** — 0 of 420 everywhere, against 420 of 420 at
h=5 on the raw label. The degeneracy of §4 was the label's scale and nothing
else, and the h=5 cell that produced five distinct numbers across 162,535 rows
now produces a fully ordered cross-section on all 1,935 dates.

### 7.1 Read across the grid, as rule A3 requires, there is no horizon effect

The pre-registered rule from 2026-09-05 is explicit: **reb_IC and cs IC are
comparable across the grid; t is not**, because n changes and a shorter horizon
earns a larger t from an identical effect. Applied here:

| | h=5 | h=10 | h=20 | h=30 |
|---|---|---|---|---|
| **cs IC (comparable)** | +0.0146 | +0.0090 | +0.0176 | +0.0152 |
| **reb IC (comparable)** | +0.0137 | +0.0122 | +0.0148 | +0.0292 |
| t (not comparable) | +2.41 | +0.99 | +1.43 | +1.09 |

**The IC is flat.** Four horizons spanning a 6x range of label width produce
+0.009 to +0.018, with the LARGEST at h=20, not h=5. The t column orders
differently only because the DK SE shrinks with the label overlap — 0.00607 at
h=5 against 0.01398 at h=30, on nearly the same number of dates.

So Part C does **not** say there is signal at five sessions. It says the
standardised model earns a small positive cross-sectional IC at EVERY horizon,
of a size that does not vary with the horizon. **That is a statement about the
objective, not about the horizon** — and it means P6's original question is
answered in the negative twice over: the horizon is not the missing axis on the
raw label, and it is not the missing axis on the standardised one either.

### 7.2 The h=5 cell's t of +2.41 does not survive, and what kills it is new

`tools/p6_scale_followup.py` runs the standing attack list. The first attack
was not on the list.

**THE RESULT MOVES ACROSS THE PRE-REGISTERED THRESHOLD ON THE NUMBER OF
OPENMP THREADS.** The same code, the same panel, the same seeds, the same
`random_state=42`, the same Optuna `SEED` — run twice, differing only in
`OMP_NUM_THREADS`:

| threads | selected gammas, per fold | cs IC | DK SE | **t** |
|---|---|---|---|---|
| 20 | 4.744, 3.188, 3.188, 0.172, 0.172 | +0.014593 | 0.006068 | **+2.405** |
| 6 | 3.188, 3.188, 0.172, 0.172, 0.172 | +0.012050 | 0.006063 | **+1.987** |

And the two models are not near-copies:

| | |
|---|---|
| identical predictions | **0 of 162,535 (0.0%)** |
| max absolute difference | 0.435, which is **6.2 prediction sd** |
| mean absolute difference | 0.0136 |
| correlation | 0.947 |

**The search lands on different hyperparameters.** XGBoost's `hist` method
accumulates histograms in parallel, and floating-point addition is not
associative, so the inner-CV score of a trial depends on how the work was
divided. Optuna then picks a different winner — fold 0 takes gamma 4.744 at 20
threads and 3.188 at 6 — and a different winner is a different model.

**A result that crosses t = 2.0 on an environment variable is not a result.**
This is the TF32 landmine in a new place: there, leaving TF32 on would have
"quietly invalidated every CPU-measured table" because the comparison turned on
the fifth decimal of MAE. Here nothing was configured wrongly at all — the
default is simply not pinned.

**What it costs the rest of this project.** Every number in every table here
was produced on this machine at its default 20 threads, so they are mutually
comparable and the A0 pin's drift of exactly 0.0 is real. But **they are
reproducible only at 20 threads**, and nothing in the repository says so. Any
future run on a GitHub runner (2 cores), on Kaggle, or on a different
workstation will not reproduce them, and the failure will look like a code
change rather than an environment one. That is recorded in CLAUDE.md §7 and it
needs a fix — pinning `OMP_NUM_THREADS` and `n_jobs` in the harness, or
recording them in `config_hash` — before the next result is quoted.

### 7.3 The rest of the attack list

With the caveat that the cell under attack is already a coin-flip against its
own threshold:

- **Per fold:** +0.0292, −0.0026, +0.0016, +0.0032, +0.0290. Two folds carry
  essentially all of it and three are flat. Unusually, this is NOT the
  early-fold shape — fold 4 is as large as fold 0 — so it is lumpy rather than
  the recorded artifact.

**The min_train sweep, the attack that retired valuation and `pooled_xgb`:**

| min_train | 380 | 420 | 460 | **500** | 540 | 580 |
|---|---|---|---|---|---|---|
| cs IC | +0.0103 | +0.0099 | +0.0151 | +0.0121 | +0.0161 | +0.0157 |
| **t** | +1.83 | +1.75 | **+2.43** | +1.99 | **+2.44** | **+2.53** |
| constant cells | 0/420 | 0/420 | 0/420 | 0/420 | 0/420 | 0/420 |

**Three of six clear +2.0, and none is negative.** That is a materially better
showing than valuation's, which spiked to +3.32 at one setting between
neighbours of +1.30 and +1.18 — the IC here stays inside 0.0099 to 0.0161 at
every setting. It fails the pre-registered "majority of six" bar, and it is not
the collapse that retired the earlier results. Recorded as it is.

**The within-date target permutation, retrained nine times** — the corrected
placebo pointed at the TARGET, because what is under test is whether the model
can rank names at all, not whether an added column helps:

| seed | cs IC | t |
|---|---|---|
| 20260920 | −0.00297 | −0.57 |
| 20260921 | +0.00083 | +0.16 |
| 20260922 | −0.00237 | −0.37 |
| 20260923 | +0.00714 | +1.38 |
| 20260924 | +0.00386 | +0.76 |
| 20260925 | **−0.01339** | **−2.75** |
| 20260926 | +0.00739 | +1.54 |
| 20260927 | −0.01280 | −2.51 |
| 20260928 | −0.00498 | −0.99 |
| **the real fit** | **+0.01205** | **+1.99** |

Two things here, pulling opposite ways. The real fit's IC **beats all nine**,
by a clear margin — +0.01205 against a maximum of +0.00739 — which is a genuine
edge over a null that destroys only the name-to-outcome link and leaves every
feature, date effect and fold boundary intact. And a draw carrying no
information by construction reached **t −2.75**, with a second at −2.51, so
**the t threshold is not calibrated for this statistic on this panel** and the
real fit's +1.99 sits comfortably inside the range the placebo itself produces.
The IC comparison carries the weight; the t does not.

**Money**, at the arm's own measured turnover of 0.56 rather than P4's assumed
0.80:

| | |
|---|---|
| gross per rebalance | +0.00171 (t +1.71) |
| **net of the 0.2225% round trip** | **+0.00047 (t +0.47)** |
| annualised net Sharpe, at 252/5 rebalances a year | **+0.17** |
| break-even rank IC here | 0.0106, against a measured +0.01205 |

It clears its own break-even — narrowly — and then the cost takes **73% of the
gross**, leaving +0.047% per rebalance that is not distinguishable from zero.

### 7.4 The verdict on Part C

| attack | result |
|---|---|
| headline t >= 2.0 | **FAIL** (+1.99 at 6 threads, +2.41 at 20) |
| holds at a majority of min_train settings | **FAIL** (3 of 6) |
| beats every target-permuted retrain | PASS (max +0.00739) |
| not carried by fold 0 alone | PASS |
| clears its own break-even IC | PASS |
| **SURVIVES** | **NO** |

**So Part C does not produce a result. It produces a hypothesis and a
measurement debt.** The hypothesis: once the label scale is fixed, the pooled
model earns a small cross-sectional edge — around +0.012 to +0.018 — that beats
its own placebo at every horizon and is worth close to nothing after costs.
That deserves its own pre-registration, on a pinned thread count, before it is
quoted anywhere.

---

## 8. Part B — SUE and delivery % at five sessions

### 8.1 The pre-registered run is vacuous, not null

Run exactly as hashed: raw label, deciding purge, four arms, eighteen placebo
retrains. Every one of them is fully degenerate.

| arm | constant cells | dates with an ordering | rebalances | cs IC |
|---|---|---|---|---|
| baseline@5 | 420 / 420 | 0 / 1,935 | 0 / 387 | undefined |
| sue@5 | 420 / 420 | 0 / 1,935 | 0 / 387 | undefined |
| timing@5 | 420 / 420 | 0 / 1,935 | 0 / 387 | undefined |
| delivery@5 | 420 / 420 | 0 / 1,935 | 0 / 387 | undefined |

All eighteen placebo retrains likewise. Every verdict the harness printed reads
FAIL, and **not one of them means anything**: R3 is `NaN < 2.0`, R2 compares
`NaN` to `NaN`, R4 has no book to charge. This is §4's degeneracy arriving in
the place it does the most damage — the arms differ from the baseline by three
columns and two columns, and a model that emits one number per fold cannot
express the difference.

**It is reported in full and relied on for nothing.** The pre-registered
protocol was executed as written; what it produced is a non-measurement, and
calling it a null would be the error the whole document is about.

One part of it IS interpretable, because it never touches the model — **the raw
sort books, at h=5:**

| feature | rebalances | gross | t | net | t | turnover | net Sharpe |
|---|---|---|---|---|---|---|---|
| `sue_evt` alone | 369 | +0.0001 | +0.10 | −0.0004 | −0.52 | 0.23 | −0.19 |
| `deliv_abn_l1` alone | 387 | +0.0008 | +1.05 | −0.0008 | −1.03 | 0.72 | −0.37 |

The literal PEAD test at five sessions earns **+0.01% per rebalance gross, at
t +0.10**, and is negative after cost. At thirty sessions it was −0.52% gross
and −0.69% net. So the surprise does not drift at five sessions either — it is
not that the 30-session window was looking in the wrong place.

Prediction PB4 held on the letter (positive gross, negative net) and not in
spirit: the gross is indistinguishable from zero rather than a real edge eaten
by costs.

### 8.2 The interpretable re-run, on a standardised label

Post hoc, and the only version in which the comparison exists. Identical in
every other respect: same folds, same purge, same rows, same nine placebo
seeds per arm, scored against the same real 5-session label.

| arm | cs IC (DK SE) | t | reb IC | reb t | constant cells | S / W / I |
|---|---|---|---|---|---|---|
| baseline@5 | +0.01459 (0.00607) | +2.41 | +0.0137 | +1.92 | 0 / 420 | 2 / 10 / 72 |
| sue@5 | +0.01449 (0.00577) | +2.51 | +0.0138 | +1.91 | 0 / 420 | 0 / 10 / 74 |
| timing@5 | +0.01494 (0.00610) | +2.45 | +0.0140 | +1.93 | 0 / 420 | 0 / 8 / 76 |
| delivery@5 | +0.01470 (0.00654) | +2.25 | +0.0151 | +2.04 | 0 / 420 | 0 / 13 / 71 |

**The four arms are the same arm.** They span +0.01449 to +0.01494 — a range of
0.00045, against a baseline of +0.01459. Paired per date:

| comparison | dates | Δ cs IC | DK SE | **t** |
|---|---|---|---|---|
| sue@5 − baseline@5 **[R3]** | 1,935 | **−0.00010** | 0.00246 | **−0.04** |
| delivery@5 − baseline@5 **[R3]** | 1,935 | **+0.00011** | 0.00292 | **+0.04** |
| timing@5 − baseline@5 | 1,935 | +0.00035 | 0.00266 | +0.13 |
| sue@5 − timing@5 **[R5]** | 1,935 | −0.00045 | 0.00192 | −0.23 |

| rule | sue@5 | delivery@5 |
|---|---|---|
| **R3** (deciding) | **FAIL**, t −0.04 | **FAIL**, t +0.04 |
| **R2** (placebo) | **FAIL**, −0.00010 against a placebo max of +0.00094 | **FAIL**, +0.00011 against +0.00072 |
| **R4** (net of cost) | pass, +0.00080 per rebalance (t +1.54) | pass, +0.00064 (t +1.07) |
| **R5** (surprise over timing) | **FAIL**, −0.00045 | n/a |
| **SIGNAL** | **NO** | **NO** |

**Neither feature adds anything at five sessions.** Both paired gains are
smaller in magnitude than the smallest of their own nine placebo draws, and R5
is negative again exactly as it was at thirty sessions — the surprise still
does not improve on its own timing and missingness columns.

R4 passes for both, and it should not be read as support: the arms' books earn
+0.0008 and +0.0006 per rebalance more than the baseline's at t +1.54 and
+1.07, on arms whose ranking is statistically identical to the baseline's. It
is the one rule of the four that is not a comparison against a null.

**The grades are noise, and the placebos say so directly.** The four arms grade
8 to 13 WEAK; the eighteen placebo retrains — whose added columns carry no
name-to-value link at all — grade **3 to 12 WEAK and 0 to 1 STRONG**. The arms
sit inside that band.

### 8.3 Predictions, scored

| # | prediction | held |
|---|---|---|
| PB1 | neither arm reaches R3's +2.0 | **yes** (−0.04 and +0.04) |
| PB2 | the SUE gain is LARGER than its −0.0055 at 30 sessions, still failing | **yes**, −0.0001 against −0.0055 — but because it is nearer zero, not because the surprise recovered |
| PB3 | R5 fails again, negative | **yes**, −0.00045 |
| PB4 | the raw `sue_evt` book is positive gross, negative net | yes on the letter; the gross is +0.0001 at t +0.10, which is zero |
| PB5 | delivery moves less than SUE on every statistic | **no** — they are indistinguishable, and delivery's reb IC is the larger |

---

## 9. What this means

### 9.1 The horizon is not the missing axis

P6 was scoped to separate "five phases of null are a fact about the panel" from
"five phases of null are a fact about one horizon". At the two horizons where
the model actually fitted something — 20 and 30 sessions — the answer is
unambiguous: nothing, at |t| ≤ 1.02, with no profile, and with an arbitrary
purge choice moving the statistic as far as the horizon does. At 5 and 10
sessions the question was not answered by Part A at all, because the tuner
switched the model off; Part C is what answers it.

**Stage 1's closing document put this on the record before the sweep ran:** if
nothing appears at any horizon, the reading that public, dated information is
priced within days is "wrong — or at least incomplete". It is worth being
precise about which. Pilot 3's decay is still measured and still real: the SUE
moves the announcement return at per-season t +5.08 and predicts nothing 30
sessions later. What P6 adds is that the information does not reappear at 5,
10 or 20 sessions either. **So the decay is not the whole explanation for the
Stage 1 nulls — it explains why a 30-session target misses the earnings
surprise, and it does not explain why a 5-session target finds nothing else.**

### 9.2 The accumulated evidence, stated once

| axis | what has been tried | result |
|---|---|---|
| features | 24 technical, 6 macro, valuation, news sentiment, reversal, delivery %, SUE | nothing clears reb_t > 2 under a sweep |
| models | ridge, XGBoost per-ticker and pooled, Chronos-2, TimesFM-2.5, Kronos, LoRA, a frozen-embedding probe | same |
| targets | 30-session excess return, 30-session absolute return | same |
| horizons | 5, 10, 20, 30 | same |
| grading | per-ticker frequentist, empirical Bayes, within-fold, cross-sectionally demeaned, Romano-Wolf | every layer graded FEWER names, not more |
| trials | ~131 before P6, ~141 after | the two results that ever cleared the bar (valuation +3.32, LoRA +2.37) were both killed by a sweep |

**On this panel — 84 large and mid-cap NSE names — the honest reading is that
there is no gradeable cross-sectional signal in the public information this
project can obtain, at any horizon it has tested.** That is a conclusion about
this panel, and the panel's composition is not incidental to it: the universe
was frozen on DATA QUALITY alone (≥ 2,400 sessions of history), which selects
precisely for large, liquid, heavily covered names — the part of any market
where public information is most efficiently priced. The design that made the
measurement trustworthy is the same design that makes the answer negative.

### 9.3 What is NOT concluded

- Not that Indian equities are unforecastable. Nothing here has been measured
  outside the frozen 84.
- Not that the instrument is broken. It reproduces at drift 0.0, it detects
  planted edges (P4: net Sharpe strictly increasing in planted IC), its
  placebos behave (Stage 0c: 1 STRONG in 9 noise draws), and it measured a real
  effect when one existed (Pilot 3's announcement return, t +5.08).
- Not that nothing in the project works. The conformal intervals measured
  **80.1% coverage against a nominal 80%** — the uncertainty quantification is
  calibrated even though the point forecast carries no edge.
- Not that h=5 and h=10 are settled by Part A. They are settled, if at all, by
  Part C.

---

## 10. What to do next, and what not to

**What not to do: a fifth feature on the same universe at the same horizons.**
Three Stage 1 sources and four horizons have now produced the same answer, and
the marginal information from a fourth source tested the same way is close to
zero. That is the variation treadmill, and the project's own history says where
it ends.

Three things are worth doing, in this order, and the first is not optional.

### 10.1 Pay the measurement debt first — it is cheap and it is blocking

`gamma` is denominated in the loss, so any work at a shorter horizon, or on any
smaller-magnitude target, is currently measuring the tuner rather than the
market. Until that is fixed, **no short-horizon result from this codebase means
anything**, positive or negative. Three candidate fixes, all in
`pipeline/tuning.py`:

- standardise the label within each date before fitting (what Part C does, and
  the cheapest);
- scale the `gamma` search range by the training label's dispersion;
- move the pooled objective to something scale-free.

This is a production-path change and P6 is research, so it is recorded rather
than made. It should be its own small, pre-registered pilot with the A0-style
pin against the 30-session baseline, because option 1 and option 2 both change
every stored comparator number.

The tie-breadth defect of §6 belongs in the same pilot, for the same reason.

### 10.2 Change the universe, not the feature

The strongest remaining hypothesis is not about features or horizons — it is
about **which names**. The universe was frozen on data quality alone (≥ 2,400
sessions), which selects for large, liquid, heavily covered companies. The
briefing that scoped Stage 1 said it directly, in §C2: analyst coverage is thin
below the large-cap tier, which is exactly where post-earnings drift and other
public-information effects survive in the literature. This panel is the part of
the market where they should NOT survive, and they do not.

**Everything needed to test that already exists.** `pipeline/earnings.py` parses
point-in-time EPS from NSE's own filings with second-resolution disclosure
times, for any NSE symbol, not just these 84. Re-pointing it at a mid- or
small-cap universe reuses the ingestion, the SUE construction, the leakage
tests and the four-rule harness. The new work is universe selection and its
data-quality problem — survivorship, thinner history, wider spreads, and a cost
model that can no longer assume 0.2225% — not a new scraper.

**It is also the honest test of the current conclusion.** "This panel carries no
gradeable public-information signal" and "public information is priced
everywhere in India" are different claims, and only the second would be
refuted by a mid-cap result. Right now the project has evidence for the first
and none either way for the second.

### 10.3 Or change what is predicted

The one thing in this project that demonstrably works is the **conformal
interval: 80.1% measured coverage against a nominal 80%.** The uncertainty
quantification is calibrated even though the point forecast carries no edge.
That is a real, shippable product — a calibrated 30-session price band per name,
with the honest statement that the direction is not predictable — and it needs
no one to beat the market's pricing of public news.

It is also the only route to something useful *today* rather than after another
research cycle.

### 10.4 The question this leaves

Three routes, and they are not substitutes: 10.1 is prerequisite maintenance,
10.2 is another research cycle with a genuinely different hypothesis, and 10.3
is the only one that ships. **Is the goal a research answer or a working
product?** Because if it is the product, 10.3 is available now and 10.2 is a
month; if it is the answer, 10.1 then 10.2, and 10.3 can wait.

---

## Recorded, not fixed

- **Nothing pins `OMP_NUM_THREADS`.** §7.2 measures what that costs: the same
  fit crosses the pre-registered threshold when the thread count changes, and
  0 of 162,535 predictions match. Every table in this project is reproducible
  only at this machine's 20 threads, and no file says so. It is the single
  highest-value fix on this list, because it is cheap and it silently
  invalidates cross-machine comparison.
- **The tie guard is binary, not a breadth test.** §6, with the measured
  turnovers.
- **The pre-registered Part B produced no measurement.** §8.1. Re-running it
  after the `gamma` fix would cost one afternoon and would make its verdicts
  mean something.
- **`gamma` is not rescaled with the label.** The fix is one of: search `gamma`
  on a relative scale, standardise the label before fitting, or switch the
  pooled objective to something scale-free. All three are production-shaped
  changes to `pipeline/tuning.py` and P6 is research, so none was made.
- **Grading used the 30-session break-even at every horizon.**
  `grade_panel_v3`'s `break_even` gates STRONG, and it was passed P4's
  0.00512363994209475 — measured at h=30 — in all eight cells, because that is
  what Stage 0c's `grade()` does and changing it mid-sweep would have made the
  grade columns incomparable with every earlier table. The per-horizon
  break-even is reported in its own column and is HIGHER at short horizons, so
  the 30-session figure makes STRONG *easier* to earn there, not harder. Any
  STRONG at h < 30 must be read against its own horizon's bar.
- **`portfolio.REBALANCES_PER_YEAR` is still pinned at 252/30.** The 2026-09-05
  pre-registration flagged it; P6 avoided it rather than fixing it, and every
  Sharpe here is annualised explicitly as `mean / sd * sqrt(252 / h)`. A test
  pins that (`test_the_sharpe_is_annualised_from_this_horizons_own_rebalance_count`).
- **The phantom sessions, re-checked at every horizon.** Zero TRAINING rows are
  affected at any horizon under the deciding rule — all four phantoms sit in
  fold 4's test window and the purge keeps training clear. The share of TEST
  windows spanning one FALLS with the horizon: 1.03% at h=5, 2.07% at h=10,
  4.11% at h=20, 5.23% at h=30.

  **That is the opposite of what §6 of the pre-registration predicted.** It
  argued h=5 would be proportionally more distorted. Per WINDOW that is true —
  an affected window is short by 1 session of 5 rather than 1 of 30 — but the
  aggregate exposure runs the other way, because a short window has five times
  fewer chances to span one of four isolated dates. The caveat was half right
  and stated the wrong half as the conclusion.
