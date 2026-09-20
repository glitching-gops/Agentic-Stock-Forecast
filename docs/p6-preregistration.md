# P6 — the horizon sweep: pre-registration, and a dated addendum to it

**Status: this file is an ADDENDUM, not a replacement.** P6 was pre-registered
on 2026-09-05, in CLAUDE.md section 5 ("P6 — THE HORIZON SWEEP: SCOPED AND
PRE-REGISTERED, NOT YET RUN"). CLAUDE.md is gitignored, so that text has never
been in the repository; §1 below reproduces it so the original terms are on the
record here too, and §2 onward is what 2026-09-20 adds. Nothing in §1 is
softened. Where §2 departs from §1 it says so in the open.

Written and hashed **before** anything below was run on real data. The only
measurements taken before this file was hashed are the outcome-blind ones in
§2.2 and §2.3 — fold counts, date counts and row counts, which read whether a
label exists and never what it is, exactly as `design_inputs` does in Pilots 2
and 3.

---

## 1. What P6 already specified (2026-09-05), reproduced

**The question.** Every number this project has produced is at a 30-session
horizon. Five phases of null could be a fact about the panel, or a fact about
that one horizon, and nothing measured so far separates them.

**The grid: 5, 10, 20, 30 sessions. Downward, on power.** The panel holds
~2,400 dates, so non-overlapping rebalances go 480 / 240 / 120 / 64 — up to
7.5x the independent windows we have ever had. Short-horizon cross-sectional
effects are also where equity predictability more usually lives. Longer
horizons were rejected: h=90 leaves ~26 rebalances and could not support
inference.

**Most of it is already built.** `compare_baselines(horizon=...)` already
retargets the label, sets the splitter's purge AND embargo to the horizon, and
sets `rebalance_every` to it. `panel.retarget_horizon` is exact and free — the
h-session forward difference of `price_frame` IS the h-session label — so
nothing is refetched and no approximation enters.

Four rules, fixed before any number was seen:

1. **JUDGE THE PROFILE, NOT THE MAXIMUM.** A real effect varies smoothly with
   horizon: monotone or single-peaked across adjacent horizons counts, an
   isolated spike between flat neighbours does not. This is the P5 lesson made
   into a rule — a pre-registered regime split there produced **t +5.21** at
   one cell whose neighbours on the same rows read +1.78 and +1.21.
2. **reb_IC IS THE COMPARABLE STATISTIC ACROSS THE GRID. t IS NOT.** n changes
   7.5x from h=30 to h=5, and t = IC / (sd/sqrt(n)), so a shorter horizon earns
   a larger t from an IDENTICAL effect. t is reported per horizon as a
   within-horizon significance test and is never read across the grid as a
   shape. Reading the t column left-to-right would manufacture a "shorter is
   better" conclusion out of sample size alone.
3. **Each horizon is additionally swept across `min_train`**, as standing
   policy. A result at one (horizon, min_train) cell is not a result.
4. **Trials accumulate: P4 N=40 -> P5 N=103 -> P6 ~N=131.** Any headline is
   deflated at the cumulative count.

**The economic bar moves with the horizon, and that is the likeliest outcome.**
A year holds 252/h rebalances — 50.4 at h=5 against 8.4 at h=30 — so the cost
drag is ~6x. And the long-short spread per unit of rank IC SHRINKS with h,
because a 5-session return is smaller than a 30-session one, which raises the
break-even from both directions at once. P4's measured break-even was rank IC
0.0051 at zero impact for h=30. **A positive short-horizon IC that does not
clear its OWN re-estimated break-even is not a result**, and `break_even_ic`
must be re-estimated per horizon from that horizon's own data rather than
scaled by assumption.

**One defect to fix first.** `portfolio.REBALANCES_PER_YEAR` is a module
constant equal to `252 / HORIZON_SESSIONS` = 8.4. Every annualisation and the
whole break-even calculation read it, so a sweep would annualise an h=5 book at
8.4 rebalances a year and understate its drag **six-fold**.

**What deliberately does not change.** `news_features.NEWS_WINDOW_SESSIONS`
stays at 30 rather than tracking h. It was set to 30 for ARTICLE DENSITY — at
~1 article per ticker-month before 2022 a 5-session news window is empty for
most rows — not to match the horizon, and shrinking it would confound "news at
a shorter horizon" with "news on a sparser window".

**Scope: the cheap comparators only.** No foundation models. P2 is closed on
both targets over two architectures, four contexts and an ensemble, and
re-opening it at four horizons is GPU-hours to lower a null.

**This is research, not a production change.** `HORIZON_SESSIONS` stays 30
unless something clears the bar above. Changing it bumps `MODEL_VERSION`,
redefines every stored label, invalidates every persisted evaluation and every
conformal calibration, and orphans `forecast_outcomes`.

---

## 2. What this addendum adds, and where it departs from §1

### 2.1 The architecture is fixed, and it is the Stage 1 baseline

§1 said "the cheap comparators". This session narrows that, deliberately, to
**pooled × MAE with no ticker feature** — the arm every Stage 1 pilot was
measured against, whose held-out predictions are frozen in
`stage2b_pooled_oos.npz` under the key `pooled_mae_noticker`.

The reason is that P6's job has changed since it was scoped. It was written as
"is the null horizon-specific". Pilot 3 turned it into the direct test of a
specific mechanism — that public, dated information is priced within days — and
that test is only readable if the horizon is the ONLY thing that moves. Adding
comparators would put a second axis on it.

**FULL RETRAIN AND RELABEL AT EACH HORIZON. Not a re-score.** At each h the
label is rebuilt by `panel.retarget_horizon`, the splitter is rebuilt at h, the
nested Optuna search is re-run inside every training fold, and the model is
refitted. Scoring stored 30-session predictions against shorter realised
windows would hold the model's hyperparameters and fitted trees at a horizon
they were chosen for, and would answer a question nobody asked.

### 2.2 The purge and the embargo — and the one place §1 and this session's brief cannot both be satisfied

At each horizon h the purge and the embargo are both

    max(h, POLITIS_WHITE_FLOOR_SESSIONS)        # pipeline/evaluation.py

with `POLITIS_WHITE_FLOOR_SESSIONS = 63`. The gap between the last training
date and the first test date is therefore twice that.

**Why a floor at all.** The purge must be at least h — that is arithmetic: a
training row's label spans `[i, i+h]`, so anything closer than h to the test
window overlaps it. The floor is the other half, and it is empirical. Stage 0c
measured the Politis-White (2004, with the Patton-Politis-White 2009
correction) automatic block length on this panel's own date-level IC series:

| variant | automatic block |
|---|---|
| per-ticker × mae | **62.5** |
| pooled × mae | 54.3 |
| pooled × rank_ic | 50.9 |
| **pooled × mae, no ticker** (the architecture swept here) | **35.8** |
| pooled × rank_ic, no ticker | 42.4 |
| PLACEBO, shuffled within date | 1.9 |

Every real variant exceeds the 30-session label and the placebo does not, so
the dependence is a property of the panel rather than of the label overlap.
**It therefore does not shrink when a shorter horizon is tested.** A 5-session
label does not make the market's own serial dependence five times shorter.

**63 is the ceiling of the WIDEST real variant (62.5), not of the 35.8 that
matches the architecture being swept.** Taking 35.8 would be choosing the
measurement that costs the least training data.

**The figure is taken from Stage 0c's record rather than recomputed, and there
is a specific reason.** The automatic length is estimated from a MODEL's
realised IC series, so it cannot exist before that model has been fitted and
scored — and a floor derived from the run it is meant to constrain is not a
constraint. It is instead recomputed AFTER each run and reported beside 63, as
a check on whether 63 was adequate for this new label structure. That is the
arrangement Stage 0c itself used when it reported the automatic length beside
the 30 it had actually used.

**THE DEPARTURE, STATED PLAINLY.** This session's brief asks for two things
that cannot both hold:

- purge = embargo = `max(h, Politis-White floor)` at every horizon; and
- that the horizon-parameterised code reproduce the established 30-session
  numbers EXACTLY at h=30.

At h=30 the first rule gives a gap of 2 × 63 = 126 dates. Every existing result
in this project, `stage2b_pooled_oos.npz` included, was measured at 2 × 30 =
60. A 126-date gap trains on 66 fewer dates in fold 0 and cannot reproduce a
60-date gap's predictions. No implementation can make it.

So the two requirements are separated rather than quietly reconciled:

- **The LEGACY rule, purge = embargo = h**, is what the regression pin runs. At
  h=30 it must reproduce `pooled_mae_noticker` row for row at drift ≤ 1e-9.
  That pins the CODE: parameterising the splitter changed nothing when the
  parameters are set to what they used to be hardcoded at.
- **The DECIDING rule, purge = embargo = max(h, 63)**, is what every verdict
  below is read from.

Both are run at all four horizons, so the sweep is reported twice over and the
cost of the wider purge is visible rather than assumed. If the two rules
disagree about whether a horizon carries signal, **the deciding rule wins and
the disagreement is itself reported as the finding.**

**The inner search is purged at the same width as the outer split.** Nesting is
not automatic: a search left at 30 inside a fold split at 63 would choose every
hyperparameter across a boundary the outer fold refuses to trust — F3 one level
down, and invisible from outside, because the reported number would merely be
optimistic rather than wrong-shaped.

### 2.3 What is held fixed across the grid, and why

| quantity | value | why it does not move with h |
|---|---|---|
| bootstrap block, `grade_panel_v3` | 30 sessions | the same argument as the purge floor: measured dependence is 35.8–62.5, so 30 already understates it at EVERY horizon. Shrinking it to 5 would narrow every SE at exactly the horizon where a spurious t is most likely. |
| Driscoll-Kraay lags | 30 | as above. Reported at the default lag rule too, which decides nothing. |
| `min_train` | 500 dates | the harness default every prior result used. Swept only if a horizon signals (rule 3 of §1). |
| folds | 5 | so fold composition is not a second moving part. |
| `EVENT_WINDOW` (SUE) | 30 sessions | a FEATURE definition — "this name reported within the last 30 sessions" — not a label horizon. Shrinking it with h would confound "SUE at a shorter horizon" with "SUE on a narrower event window", which is the `NEWS_WINDOW_SESSIONS` argument from §1. |
| `rebalance_every`, for books and reb_IC | **h** | this one DOES move, and must: it is what makes successive windows non-overlapping. |

**Measured before this file was hashed, outcome-blind.** All five folds survive
at every horizon under both rules, so the wide purge is not a fold-count
constraint. What it costs is training dates in the early folds and one inner CV
fold in fold 0:

| h | rule | purge | folds | train dates by fold | OOS dates | rebalances | inner CV folds |
|---|---|---|---|---|---|---|---|
| 5 | legacy | 5 | 5 | 490 / 878 / 1266 / 1654 / 2042 | 1,935 | 387 | 3,3,3,3,3 |
| 5 | deciding | 63 | 5 | 374 / 762 / 1150 / 1538 / 1926 | 1,935 | 387 | **2**,3,3,3,3 |
| 10 | legacy | 10 | 5 | 480 / 868 / 1256 / 1644 / 2032 | 1,930 | 193 | 3,3,3,3,3 |
| 10 | deciding | 63 | 5 | 374 / 762 / 1150 / 1538 / 1926 | 1,930 | 193 | **2**,3,3,3,3 |
| 20 | legacy | 20 | 5 | 460 / 848 / 1236 / 1624 / 2012 | 1,920 | 96 | 3,3,3,3,3 |
| 20 | deciding | 63 | 5 | 374 / 762 / 1150 / 1538 / 1926 | 1,920 | 96 | **2**,3,3,3,3 |
| 30 | legacy | 30 | 5 | 440 / 828 / 1216 / 1604 / 1992 | 1,910 | 63 | 3,3,3,3,3 |
| 30 | deciding | 63 | 5 | 374 / 762 / 1150 / 1538 / 1926 | 1,910 | 63 | **2**,3,3,3,3 |

The test windows are identical across rules at a given h, so the two rules are
scored on the same rows and differ only in what the model was allowed to learn
from. **Fold 0 loses one of three inner folds under the deciding rule**, which
means its hyperparameters are chosen on two scored inner blocks rather than
three — recorded here so it is not discovered afterwards as an explanation.

### 2.4 The economic bar

Per-rebalance net returns only. `portfolio.REBALANCES_PER_YEAR` is still the
30-session constant §1 flagged, and it is **not fixed in this session** — it is
production code and P6 is research. It is avoided instead: nothing below quotes
an annualised return or a `BookResult` aggregate that reads it, and where a
Sharpe is reported it is computed explicitly as `mean / sd * sqrt(252 / h)`.
The defect stays on the record, unfixed.

---

## 3. Part A — the sweep. Hypotheses and deciding rules

**H-A.** If this panel's 30-session null is an artefact of the horizon rather
than a property of the panel, a shorter horizon will show cross-sectional
predictive information that 30 sessions does not.

**The competing hypothesis, which Pilot 3 makes concrete.** Public, dated
information is priced within about two sessions on these 84 names. Under that
reading a shorter horizon captures more of the reaction and less of the
subsequent noise, so if anything is there at all it appears at h=5 or h=10 and
fades toward h=30.

**A0 — the stop condition.** At h=30 under the LEGACY rule, the re-run must
reproduce `stage2b_pooled_oos.npz::pooled_mae_noticker` with zero unmatched
rows either way and max drift ≤ 1e-9 in both `y_pred` and `y_true`. **If it
does not, the run stops and no other horizon is read.**

**A1 — THE DECIDING RULE.** At each horizon, on the deciding purge rule: the
mean per-date cross-sectional rank IC with a Driscoll-Kraay SE at 30 lags. A
horizon carries signal iff **t ≥ +2.0**. This is the same statistic and the
same threshold that decided all three Stage 1 pilots.

**A2 — the profile rule (§1 rule 1).** A horizon that passes A1 while both
adjacent horizons sit below +1.0 is reported as an isolated spike and does NOT
count, pending A4.

**A3 — reading across the grid (§1 rule 2).** reb_IC is compared across
horizons; t is not. Every t belongs to its own horizon.

**A4 — the min_train sweep (§1 rule 3).** Triggered only if some horizon passes
A1. That horizon is then re-run at min_train 380/420/460/500/540/580 and must
keep t ≥ +2.0 at a majority of the six.

**A5 — the grade panel.** `grade_panel_v3` at Stage 0c's settings (block 30,
alpha 0.10, Romano-Wolf, REML, B = 1000) at every horizon. Stage 0c's nine-draw
within-date placebo established that a STRONG count of 1 is what an alpha-0.10
familywise procedure hands out on noise, so **a grade distribution is read as
evidence only at STRONG ≥ 3**. Grades are reported at every horizon regardless,
and they decide nothing on their own.

**A6 — deflation.** ~131 trials preceded this session. The four sweep cells and
the three arms of Part B add ~10 more.

### Predictions (Part A)

| # | prediction |
|---|---|
| **PA1** | A0 passes, at drift exactly 0.0. |
| **PA2** | No horizon reaches t ≥ +2.0 on A1. |
| **PA3** | The cross-sectional IC profile across 5/10/20/30 has no monotone trend — the four values do not order with h in either direction. |
| **PA4** | The deciding rule (purge 63) and the legacy rule (purge h) agree on the verdict at all four horizons, and the deciding rule's IC is the weaker of the two more often than not, because it trains on less. |
| **PA5** | Every horizon grades STRONG ≤ 2. |
| **PA6** | Recomputed Politis-White stays in the 30–60 band at every horizon and does not track h — so the fixed floor of 63 is vindicated rather than merely assumed. |

---

## 4. Part B — SUE and delivery % at 5 sessions

The direct test of Pilot 3's diagnostic. Its SUE lines up with the ANNOUNCEMENT
return at t +5.08 over 40 reporting seasons and predicts nothing over the
following 30 sessions. If that decay is the whole story, a 5-session label is
where the surviving part of it would show.

**B0 — the comparator is the 5-session baseline from Part A**, on the deciding
purge rule, on the same folds and the same rows. The existing 30-session
baseline is not used and must not be: a 5-session arm against a 30-session
baseline confounds the horizon change with the feature's own effect, and the
result would not be interpretable as either.

**The arms**, each independently — never combined, so attribution stays clean:

| arm | columns |
|---|---|
| (a) baseline@5 | FACTORS |
| (b) sue@5 | FACTORS + `sue_evt`, `sue_age`, `sue_missing` |
| (c) timing@5 | FACTORS + `sue_age`, `sue_missing` — the identity arm |
| (d) delivery@5 | FACTORS + `deliv_abn_l1`, `deliv_abn5_l1` |

**The ingestion is untouched.** `pipeline/earnings.py` and
`pipeline/delivery.py` are reused exactly as built and tested in Pilots 2 and
3. This session changes the horizon, the label and the purge, and nothing else.

**B1 — R3, THE DECIDING RULE.** Paired per-date cross-sectional rank IC, (arm −
baseline@5), Driscoll-Kraay SE at 30 lags. Signal iff **t ≥ +2.0**.

**B2 — R2, the corrected placebo.** Nine retrains per arm with that arm's new
columns permuted JOINTLY within each date — never a prediction shuffle, which
is the error Pilot 1 recorded. The arm's IC gain must exceed all nine.

**B3 — R4, net of cost.** Long-short top-minus-bottom quintile books over the
NON-OVERLAPPING 5-session rebalances, charged the 0.2225% round trip on
name-by-name turnover. The arm's net return per rebalance must exceed the
baseline's. This bar is materially harder at h=5 than at h=30: a year holds
50.4 rebalances instead of 8.4.

**B4 — R5, the surprise beyond its own timing columns. SUE ONLY.** (b) − (c)
must be positive. Carried forward because Pilot 3 measured `sue_age` and
`sue_missing` persisting at +0.40 and +0.47 within-date rank over 250 sessions,
and reproducing every grade the SUE arm moved. A within-date column permutation
cannot catch that, because it destroys identity and information together.

**SIGNAL for the SUE arm iff R3 ∧ R2 ∧ R4 ∧ R5.**
**SIGNAL for the delivery arm iff R3 ∧ R2 ∧ R4.** Delivery has no R5: its two
columns are one construction, and Pilot 2 already ran the level arm that plays
that role.

**The raw sort books at h=5**, descriptive: `sue_evt` alone and `deliv_abn_l1`
alone, long the top quintile and short the bottom, net of the round trip. This
is the literal PEAD test at the horizon the diagnostic points at.

### Predictions (Part B)

| # | prediction |
|---|---|
| **PB1** | Neither arm reaches R3's +2.0. |
| **PB2** | The SUE arm's gain over baseline@5 is LARGER than its −0.0055 at 30 sessions — the decay is real, so a shorter horizon should recover some of it — while still failing +2.0. |
| **PB3** | R5 fails again: (b) − (c) is negative, as it was at 30 sessions. |
| **PB4** | The raw `sue_evt` sort book at h=5 is positive GROSS and negative NET, because 50.4 rebalances a year at ~78% turnover is about 8.7% of round-trip cost. |
| **PB5** | The delivery arm moves less than the SUE arm on every statistic, as it did at 30 sessions, because Pilot 2 found no reaction to test. |

---

## 5. Non-goals

- No change to the live gate, `model_metadata`, `forecast_confidence`, the API
  or the frontend. Shadow only, as every prior pilot.
- `HORIZON_SESSIONS` stays 30. Nothing here is a production change, and no
  merge to `main` happens without review.
- Reversal is not re-tested at another horizon.
- Bulk/block deals — the one originally-planned Stage 1 source never attempted
  — is deprioritised in favour of this, not abandoned. `docs/stage1-closing.md`
  records why.
- No foundation models (§1).
- `pipeline/earnings.py` and `pipeline/delivery.py` are not touched.

## 6. Caveats carried in before the run

- **Every SE here is likely too small.** Politis-White puts this panel's
  dependence at 35.8–62.5 sessions against a block of 30.
- **~131 prior trials.** A single t of +2.0 on this panel is not what it would
  be on a fresh one.
- **2013–2017 SUE coverage is 73–87%**, so the early folds carry more imputed
  rows. Unchanged from Pilot 3.
- **The panel carries four phantom 2026 holiday sessions.** They sit in fold
  4's test window; at h=5 a label window spanning one measures 4 real sessions
  rather than 5, which is a proportionally LARGER distortion than at h=30. The
  SUE features already exclude them (`event_features(..., phantoms=...)`); the
  LABEL does not, at any horizon, and did not at 30 either. Recorded, not
  fixed, and re-checked after the run the way Pilot 2's was.
