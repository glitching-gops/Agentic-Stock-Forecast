# The evidence-grading track, closed

> **This is the single document to read for this whole track.** It supersedes
> the numbers in `stage0-evidence-grading.md`, `stage0-addendum-degeneracy-sweep.md`,
> `stage0b-findings.md` and section 8 of `stage2b-findings.md`, each of which
> now carries a correction notice pointing here. Those documents are kept
> unedited: annotate, never erase — the standard this project set when it
> retracted the ~4.3% MAPE and ~85% directional-accuracy figures.
>
> Six sessions: Stage 0 → Stage 0 addendum → Stage 2a → Stage 2b → Stage 0b →
> Stage 0c. Separate numbering from the project's Phase 0-6 roadmap; neither
> renumbers the other.
>
> Pre-registrations, all written before their own runs:
> [`stage0-preregistration.md`](stage0-preregistration.md),
> [`stage2a-preregistration.md`](stage2a-preregistration.md),
> [`stage2b-preregistration.md`](stage2b-preregistration.md),
> [`stage0b-preregistration.md`](stage0b-preregistration.md),
> [`stage0c-preregistration.md`](stage0c-preregistration.md).

---

## 1. What the track set out to do, and what it actually found

**The premise.** An external audit argued the evidence gate was
*underpowered*: it grades each ticker alone against `|t| >= 2` on
`n_eff = 1909/30 = 63.6` observations, and at n = 64 a t of 2.0 demands a rank
IC of **0.25** against the 0.02-0.05 real cross-sectional signals carry. The
proposed fix was partial pooling, and the prediction was that it would
legitimately grade more names.

**It graded fewer — and then the whole quantity turned out to be wrong.** Three
distinct defects were found, each hiding the next:

| # | defect | found in | consequence |
|---|---|---|---|
| 1 | 316 of 420 (ticker, fold) fits emit a CONSTANT — the trees make no splits | Stage 0 | 75% of the panel had no ordering to grade |
| 2 | the per-ticker IC POOLED across walk-forward folds and correlated once | Stage 0b | `mu_hat = -0.05988` was five fold levels against five period returns, not skill |
| 3 | the graded quantity was a RAW time-series IC, and its SE assumed 84 independent tickers | Stage 0c | a common market factor inflated every name at once; z = +5.72 was not credible |

Defect 1 was diagnosed by Stage 2a (the tuner scores MAE, under which a constant
is near-optimal) and dissolved by Stage 2b (it is an ALTITUDE problem: pooling
across 84 tickers removes it under the unchanged objective, and the untuned
`pooled_xgb` never had it). Defect 2 was found because Stage 2b graded a model
with **zero** constants and `mu_hat` barely moved. Defect 3 is what Stage 0c
closes.

## 2. The corrected methodology, and why each piece is there

**Cross-sectional rank-demeaning, both sides.** Within each date, ranks across
the names present, mapped to [-1, 1] (Gu-Kelly-Xiu / Kelly-Pruitt-Su). Applied
to predictions *and* targets: demeaning the target alone leaves the market in
the prediction, so a model forecasting nothing but the market level still scores
against the residual through its own drift. Dates below
`MIN_CROSS_SECTION = 20` are dropped and counted, never imputed and never
centred against another date's cross-section.

The centering matrix `C = I - (1/N)11'` is idempotent with rank N-1 and induces
an exact pairwise correlation of `-1/(N-1) = -0.01205` at N = 84, plus one lost
degree of freedom per date — Pearson (1897), Chayes (1960), Aitchison (1986).
It needs no separate correction because the date-level bootstrap resamples whole
cross-sections, and that is **tested** rather than asserted.

**A date-level circular block bootstrap, re-running the whole pipeline.** Blocks
of 30 sessions (the label horizon), drawn WITHIN each fold so the walk-forward
ordering and the purge/embargo survive. Circular rather than moving-block, which
under-samples the series ends — and here the ends are the fold boundaries. Each
replicate recomputes every ticker's fold-averaged IC, every `sigma2_i`, `tau2`,
`mu_hat` and every posterior. The spread across replicates is the standard
error, and it contains within-fold, between-fold and cross-sectional dependence
at once.

**Therefore Stage 0b's additive `max(within, between)` term is switched OFF
where the bootstrap is in force.** It survives only as the fallback below
`MIN_FOLDS_FOR_ESTIMATE = 3`, and every row records which path it took.

**Romano-Wolf stepdown for multiplicity.** Grading 84 tickers is 84 simultaneous
tests. Benjamini-Hochberg is not safe here — it needs PRDS, and this panel has a
common market factor (positive dependence) *and* the `-1/(N-1)` demeaning
induces (negative). Romano-Wolf assumes nothing about dependence: it reads it off
the same bootstrap. BH and Benjamini-Yekutieli are reported beside it as the
optimistic and conservative bounds. Harvey-Liu-Zhu (2016) argue t > 3.0 for a
new factor given the testing already done in the literature; this project has run
103+ configurations on this one panel.

**REML for `tau2`, HKSJ for intervals, and a degeneracy detector.** DL is
negatively biased at small K (Langan et al. 2019); HKSJ outperforms the standard
interval (IntHout, Ioannidis & Borm 2014) with a caveat at five or fewer very
unequal units — **which is this panel's regime**, so the interval is reported as
approximate. The detector fires when `tau2` sits at the zero boundary or
Cochran's Q does not exceed its df; when it does, the random-effects model has
collapsed to a fixed-effect one, every posterior IS the grand mean, and the layer
emits **one** panel statement instead of 84 identical ones.

**Two placebos, and they are the point.** Predictions permuted *within each
date* preserve the demeaning geometry exactly and destroy only the
name-to-outcome link; per-ticker constants carry no information at all. If
either grades names, the corrected layer is manufacturing grades rather than
revealing them.

## 3. Every superseded number, and what replaced it

| number | where it was published | what it actually was | replaced by |
|---|---|---|---|
| `mu_hat = -0.05988` | Stage 0 headline | five fold-level prediction levels correlated against five period returns (ρ = −0.600), not skill | per-variant `mu_hat` in §4, computed within-fold and rank-demeaned |
| `tau2_hat = 0.00221` | Stage 0 | the DL estimate of the between-ticker variance of that broken statistic | REML `tau2` per variant, with a zero-boundary detector |
| the whole τ sweep (τ = 1.00 … 0.10) | Stage 0 addendum | the same broken statistic, swept over an exclusion threshold. The τ = 1.00 cell "reproducing Stage 0 to the last digit" was a check of the code path, not of the number | nothing — the sweep's *premise* (that excluding degenerate cells would fix `mu_hat`) was refuted by Stage 2b |
| within-fold `+0.1444`, t +6.94 | Stage 0 addendum | a RAW per-ticker time-series IC on survivor cells, inflated by a common market factor and selected on the model having chosen to split | after demeaning and Romano-Wolf: §4 |
| "`mu_hat` is dominated by fold-level constants" | Stage 0 addendum | wrong attribution: the fold LEVEL was the mechanism, constancy only its extreme form | Stage 2b measured `mu_hat = -0.052` on a model with ZERO constants |
| `mu_hat = +0.15028`, 10 STRONG | Stage 0b, per-ticker × mae | measured on 10 of 84 names, selected by the model's own choice to split | `n_usable` is now 84/84 after demeaning; see §4 |
| z = +5.72 | Stage 0b, pooled × rank_ic | a precision-weighted SE that treated 84 tickers as independent | bootstrap z in §4 |
| 84 identical ANTI_SIGNAL grades | Stage 2b panel diagnostic | one panel statement at τ² = 0.0003, printed once per company | the τ² ≈ 0 detector emits ONE statement |

**What was audited and STANDS, unchanged:** `pipeline/baselines.py`'s headline
comparator table. `cross_sectional_report` and `_mean_daily_rank_ic` are
within-group-then-averaged, the pooled figure is never persisted and never used
by `clears_floor` or `best()`, and three regression tests now pin that. So
`pooled_xgb` +0.0389 / t +2.42, `beta_market` +0.0464, every P2–P5 table and the
break-even 0.00512 are unaffected — **and nothing on the published `/research`
page is invalidated by this track.**

## 4. The corrected numbers

`B = 1000` replicates, each a full empirical-Bayes re-run; block = 30 sessions;
Romano-Wolf at α = 0.10. Runtime ≈ 250 s per variant, ~34 minutes for all eight.
**Nothing retrained** — every set of predictions already existed.

| variant | n_usable | `mu_hat` | boot SE | naive SE | **z (boot)** | τ² REML | τ² DL | τ²≈0 | STRONG | WEAK |
|---|---|---|---|---|---|---|---|---|---|---|
| per-ticker × mae (production) | **84/84** | +0.01449 | 0.00977 | 0.01140 | **+1.48** | 0.00508 | 0.00506 | no | **1** | 14 |
| per-ticker × rank_ic (Stage 2a) | **0/14** | n/a | n/a | n/a | n/a | 0 | 0 | **YES** | **0** | 0 |
| pooled × mae | **84/84** | +0.00983 | 0.01204 | 0.01294 | **+0.82** | 0.00797 | 0.00814 | no | **3** | 9 |
| pooled × rank_ic | **84/84** | +0.02433 | 0.01246 | 0.01133 | **+1.95** | 0.00516 | 0.00517 | no | **4** | 12 |
| pooled × mae, no ticker | **84/84** | −0.01727 | 0.01390 | 0.01115 | **−1.24** | 0.00435 | 0.00423 | no | **1** | 1 |
| pooled × rank_ic, no ticker | **84/84** | −0.02453 | 0.01412 | 0.00936 | **−1.74** | 0.00211 | 0.00197 | no | **0** | 0 |
| **PLACEBO: shuffled within date** | 84/84 | +0.00311 | 0.00309 | 0.00250 | +1.01 | 0.00004 | 0.00003 | no | **0** | 0 |
| **PLACEBO: per-ticker constants** | **0/84** | n/a | n/a | n/a | n/a | 0 | 0 | **YES** | **0** | 0 |

### Against Stage 0b, on the same predictions

| variant | n_usable | `mu_hat` | z | STRONG |
|---|---|---|---|---|
| per-ticker × mae | 10 → **84** | +0.15028 → **+0.01449** | +4.62 → **+1.48** | 10 → **1** |
| per-ticker × rank_ic | 13 → **0** | +0.12866 → n/a | +4.89 → n/a | 6 → **0** |
| pooled × mae | 84 → 84 | +0.01575 → **+0.00983** | +1.55 → **+0.82** | 0 → 3 |
| pooled × rank_ic | 84 → 84 | +0.05640 → **+0.02433** | **+5.72 → +1.95** | 3 → 4 |
| pooled × mae, no ticker | 84 → 84 | +0.02102 → **−0.01727** | +2.01 → **−1.24** | 0 → 1 |
| pooled × rank_ic, no ticker | 84 → 84 | +0.01282 → **−0.02453** | +1.30 → **−1.74** | 0 → 0 |

**Three of the four predicted reductions happened, and one prediction was wrong
in an informative way.**

- **The +0.150 / 10-STRONG figure is gone.** `n_usable` went 10 → 84, not by
  loosening anything but because demeaning gives a name whose prediction is
  constant in time a cross-sectional rank that still moves. `mu_hat` fell to
  +0.0145 and STRONG to 1.
- **z = +5.72 collapsed to +1.95** — and **not for the reason expected.** The
  bootstrap SE (0.01246) is within 10% of the naive one (0.01133); on two
  variants it is *smaller*. **The z fell because the POINT ESTIMATE fell**, from
  +0.0564 to +0.0243, when the common market factor was demeaned out. The naive
  SE was not the main problem; the quantity was.
- **The τ²≈0 detector fired on two variants** — both structurally ungradeable
  ones — and emitted one panel statement each instead of 84 rows.
- **Prediction 3 was wrong.** Some full-population variants still show 1–4
  STRONG. §5 is about whether those survive scrutiny.

### The two quantities, still not the same thing

| variant | per-ticker `mu_hat` (the badge) | per-date cross-sectional IC (the book) | DK SE |
|---|---|---|---|
| per-ticker × mae | **+0.0145** | **−0.0148** | 0.0123 |
| pooled × mae | +0.0098 | +0.0223 | 0.0114 |
| pooled × rank_ic | +0.0243 | +0.0124 | 0.0113 |
| pooled × mae, no ticker | −0.0173 | −0.0010 | 0.0081 |

**For the production model the two carry OPPOSITE SIGNS.** The badge quantity
says +0.0145; the quantity a long-short book earns, and the one the 0.00512
break-even is denominated in, says −0.0148. That is the Stage 0b finding
sharpened: the gate and the portfolio have never been answering the same
question, and on the live model they disagree in direction.

### Bootstrap SE against Driscoll-Kraay

These estimate **different estimands** and are reported as such: the bootstrap
SE is for `mu_hat` (the shrunk per-ticker mean); the Driscoll-Kraay SE is for
the per-date cross-sectional mean, computed by `linearmodels.PanelOLS` with a
Bartlett kernel on the (ticker, date) panel — no resampling anywhere, which is
what makes it independent. Both land at **0.008–0.014** over the same ~63
independent windows. Agreement on magnitude is a genuine sanity check; it is
not agreement on a number, and the report never treats it as one.

### Block length: 30 against Politis-White

| variant | used | automatic |
|---|---|---|
| per-ticker × mae | 30 | **62.5** |
| pooled × mae | 30 | 54.3 |
| pooled × rank_ic | 30 | 50.9 |
| pooled × mae, no ticker | 30 | 35.8 |
| pooled × rank_ic, no ticker | 30 | 42.4 |
| PLACEBO shuffled | 30 | **1.9** |

**The automatic length is 1.2× to 2.1× the label horizon on every real variant
and 1.9 on the placebo.** That gap is the finding: the data carry dependence
*beyond* the 30-session label overlap, and the placebo — which destroys the
cross-sectional ranking but leaves the target's own persistence — does not. A
block of 30 therefore **understates** the uncertainty here. It was not silently
replaced: 30 is the known, defensible horizon, and the honest reading is that
every SE in the table is, if anything, too small.

The bootstrap distributions of `mu_hat` are close to symmetric (skew −0.33 to
+0.15) with no multimodality and no single-date dominance.

## 5. The placebo distribution — the only thing that makes §4 readable

Prediction 3 was wrong: some full-population variants still show 1–4 STRONG. So
the question became whether the corrected layer produces those on **noise**.
Nine within-date permutation draws (the main run's placebo plus eight seeded
replications, B = 1000 each):

| base | seeds | `mu_hat` range | τ² range | RW rejections | **STRONG** | WEAK |
|---|---|---|---|---|---|---|
| per-ticker × mae | 11, 22, 33, 44 | −0.0030 … +0.0046 | 0.00001 … 0.00017 | 0, 0, 0, 0 | **0, 0, 0, 0** | 3, 2, 0, 0 |
| pooled × rank_ic | 11, 22, 33, 44 | −0.0028 … +0.0050 | 0.00000 … 0.00008 | **1**, 0, 0, 0 | **1, 0, 0, 0** | 2, 0, 0, 0 |
| (main run placebo) | — | +0.0031 | 0.00004 | 1 | 0 | 0 |

**Three things separate the real variants from the null, and one does not.**

| quantity | null (9 draws) | real variants | separates? |
|---|---|---|---|
| τ² | ≤ **0.00017** | 0.0021 – 0.0080 | **YES — 12× to 47×** |
| \|`mu_hat`\| | ≤ 0.0050 | 0.0098 – 0.0245 | **YES — 2× to 5×** |
| WEAK count | 0 – 3 | 0 – 14 | partly; a WEAK is cheap |
| **STRONG count** | 0 – 1 | 0 – 4 | **only above 1** |

**Romano-Wolf is calibrated, not liberal.** One of nine null draws produced at
least one rejection — 11% against a nominal α = 0.10. That is what familywise
control is supposed to look like, and it means a single STRONG is exactly what
noise delivers about a tenth of the time.

**So a count of 1 is not evidence. A count of 3–4 is above anything nine null
draws produced.** That leaves `pooled × mae` (3) and `pooled × rank_ic` (4)
as the only two cells not explained by chance — and immediately raises the
question of what they are made of.

### And what they are made of is the ticker feature

| arm | STRONG | its no-ticker sibling |
|---|---|---|
| pooled × mae | **3** | **1** |
| pooled × rank_ic | **4** | **0** |

**Every STRONG grade this layer produces disappears, or nearly does, when the
model is denied the ticker's identity.** That is the CLAUDE.md §7 landmine
exactly: two random per-ticker constants, carrying zero information about
returns, scored `pooled_xgb` at a mean rebalance t of **+0.77** (sd 0.49, max
+1.77) because the tree identifies the name and learns which names paid in the
training window, and over a 4.4-year panel that persists into the test folds.

Stage 2b already measured that the ticker feature is worth nothing
**cross-sectionally** (+0.0204 with it against +0.0161 without,
indistinguishable). It is worth something to a **per-ticker** statistic, which
is the quantity this gate grades. The two facts are consistent and together they
are the answer: the gate rewards the one channel the tradeable measurement says
is worthless.

**Note also that τ² on every null draw is ≤ 0.00017** — near the boundary
where every posterior collapses onto the grand mean — while the pre-registered
detector uses an absolute tolerance of 1e-6 and did not fire. The detector is
calibrated as pre-registered and is NOT retuned here after seeing the numbers;
it is recorded as a limitation: a scale-free rule (τ² relative to the median
σᵢ², or the shrinkage weight itself) would fire on these draws and an absolute
one does not.


## 6. What the corrected pipeline still cannot resolve

**1. The fold-0-peaks / fold-4-negative pattern. This is the top open item.**
Across six Stage 2b arms — *including one whose only extra column is pure
noise* — the cross-sectional IC peaks in fold 0 (+0.033 to +0.106) and is
negative in fold 4. Valuation (+3.32), LoRA (+2.37), `pooled_xgb` (+2.42), P3's
`regime_factor` and P5's `linear_factor` were all carried by the earliest fold
too. Deliberately not investigated here. Until it is explained, **every per-fold
statistic on this panel — including the corrected one — carries an unquantified
period effect**, because the estimate is a mean over five folds and one of them
behaves differently for reasons nobody has identified.

**2. ~63 date-clusters is borderline, and 5 folds is not asymptotic at all.**
Clustering is on the date dimension (~63 independent 30-session windows), never
on the fold dimension. Applied practice treats fewer than ~30–40 clusters as the
danger zone; Cameron & Miller (2015) note there is no firm threshold. At ~63 this
panel is borderline rather than comfortable, and the HKSJ interval refers to t on
83 df while the panel holds about five genuinely independent periods — the exact
regime IntHout et al. flag as needing caution. **Every standard error in this
layer is approximate and is labelled so in the output.**

**3. The automatic block length is 1.2×–2.1× the horizon.** See §4. The
dependence exceeds the label overlap, so the reported SEs are, if anything, too
small.

**4. Informative missingness is no longer binding — but the mechanism remains.**
Demeaning took `n_usable` from 10/84 to 84/84 on the production model, which
removes the biased complete-case comparison Stage 0b had to warn about. The
underlying fact has not changed: a degenerate fold yields an undefined
correlation, so missingness depends on the model's own behaviour (MNAR, not
MCAR). No inverse-probability weighting is attempted — at N = 84 the missingness
model cannot be credibly estimated and would add false precision.

**5. A 14-name panel cannot be graded at all.** The Stage 2a pilot returns
`n_usable = 0/14` because every date falls below `MIN_CROSS_SECTION = 20`. That
is correct behaviour, not a failure: a cross-sectional demeaning over 14 names is
not a cross-sectional demeaning. It does mean the Stage 2a arm is structurally
outside this comparison.

## 7. The live gate, changed but NOT deployed

`pipeline/evaluation.compute_metrics` now takes fold labels and returns the
within-fold average; `agents/critic_agent.grade_evidence` now requires
`eval_rw_significant` — the panel-level Romano-Wolf rejection — before it will
award STRONG. **Absent means not established**: a ticker whose panel grading has
not run is capped at WEAK rather than promoted on unadjusted evidence, so the
change can only remove a STRONG, never create one.

Running the **real** grader on the old and new inputs over all 84 tickers:

| grade | pooled IC, unadjusted (live today) | within-fold, demeaned, Romano-Wolf |
|---|---|---|
| STRONG | 0 | **0** |
| WEAK | 2 | **1** |
| INSUFFICIENT | 82 | **83** |

**One ticker moves: BOSCHLTD.NS, WEAK → INSUFFICIENT** (pooled IC +0.0883,
demeaned within-fold IC −0.0976).

That is far smaller than the 16 tickers Stage 0b measured, and the reason is
worth stating: Stage 0b measured the within-fold fix ALONE, which raised many
tickers' ICs. Stage 0c adds the demeaning, which removes the common market
factor those raised ICs were largely made of. **The two corrections very nearly
cancel in the grade distribution and do not cancel at all in what the number
means.**

The change is prepared and **not deployed**. No push, no Render redeploy.

## 8. What a Render redeploy would change on the live dashboard

Stated plainly for the hand-back, and it is less than it sounds:

- **One badge changes**: BOSCHLTD.NS drops WEAK → INSUFFICIENT. Nothing gains a
  badge; nothing reaches STRONG.
- **No STRONG badge exists today and none would appear.** The published board
  has never carried one.
- **`/research` is unaffected.** Every figure on it comes from
  `cross_sectional_report`, audited clean, and none of the evidence-grading
  numbers were ever published there.
- The dashboard would need one copy change to be honest about the τ² ≈ 0 case:
  when the panel is degenerate the badges are ONE panel-level statement repeated,
  not 84 per-company findings, and the surface should say so rather than render
  84 identical chips.

## 9. The bottom line

**Does this panel carry real, gradeable per-ticker evidence? No.**

The corrected layer produces STRONG grades on exactly two of six variants, and
both of them lose those grades when the model is denied the ticker's identity —
which is the channel this project measured, in a different session, as earning a
rebalance t of +0.77 from pure noise. On the arms that cannot see which company
it is, the count is 0 and 1, and 1 is what a correctly calibrated familywise
procedure hands out on noise about a tenth of the time.

**The live gate ends at 0 STRONG, 1 WEAK, 83 INSUFFICIENT.** No published badge
has ever read STRONG, and after this change none would.

**Is the "no" trustworthy now? Substantially, and with three named caveats.**

What makes it more trustworthy than any previous "no" on this track:

- the graded quantity is finally the right shape — within-fold, and
  cross-sectionally demeaned so a common market factor cannot inflate every name
  at once;
- its uncertainty comes from resampling the whole pipeline at the date level,
  cross-checked by a closed-form Driscoll-Kraay SE that lands in the same place;
- its multiplicity is controlled by a procedure that assumes nothing about
  dependence, and that procedure is **verified calibrated on nine null draws**;
- and the whole layer is verified not to manufacture grades: a within-date
  permutation, which preserves the demeaning geometry exactly, returns τ² ≤
  0.00017, |`mu_hat`| ≤ 0.005 and essentially no grades.

What still stands between this and a fully trustworthy null:

1. **The fold-0/fold-4 pattern is unexplained**, and it appears in a pure-noise
   arm. Until it is, every per-fold statistic here carries an unquantified
   period effect.
2. **~63 clusters and 5 folds.** Every SE in this layer is approximate and
   labelled so; the automatic block length says the dependence exceeds the label
   horizon, so they are if anything too small.
3. **The gate and the book still measure different things.** For the production
   model they now disagree in *sign* (+0.0145 per-ticker against −0.0148
   cross-sectional). Stage 0c reports both and refuses to conflate them, but it
   does not resolve which one a per-company badge should be denominated in.
   **That is a product question, not a statistical one**, and it is the first
   thing to settle if this layer is ever taken further.

**What the track bought.** Not a signal. It bought a measurement that is
correct, an audit proving the project's headline comparator table was never
affected, three defects found and fixed, a placebo that says the layer does not
invent grades, and a documented, reproducible answer to a question that had been
answered three times before with numbers that were artifacts.
