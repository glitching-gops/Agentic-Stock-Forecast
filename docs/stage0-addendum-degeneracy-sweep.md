<!-- stage0c-correction-notice -->
> ## ⚠ CORRECTION NOTICE — superseded 2026-09-08
>
> **The entire tau sweep in this document is computed on a broken statistic.**
>
> Every row of the tau grid — including the `tau = 1.00` cell that reproduces
> Stage 0's `mu_hat = -0.05988` "to the last digit" — was produced by a
> per-ticker IC that pooled across walk-forward folds and correlated once. That
> reproduction was a genuine check of the code path and a false reassurance
> about the number: both sides of the comparison carried the same defect.
>
> **And this document's central attribution is WRONG.** It concluded that
> `mu_hat` is "dominated by fold-level constants" and flips sign when they are
> excluded. Stage 2b then graded a model that emits **zero** constant
> predictions and `mu_hat` moved only to **-0.052**, with the identical
> rho = -0.600. The fold LEVEL was the mechanism; constancy was only its most
> extreme form, and removing the constants was never going to fix it.
>
> **What still stands:** that the degeneracy distribution is bimodal (316 cells
> at mode-share exactly 1.000, only 16 in (0.90, 1.00)); that the constant IS
> the training mean, a median 0.043 sd from the preceding period's mean; and
> the root-cause reading that the tuner optimises a metric under which a
> constant is near-optimal. Stage 2a and Stage 2b confirmed all three.
>
> **What does not:** every `mu_hat`, `tau2_hat`, z-statistic and grade count in
> the sweep table, and the within-fold `+0.1444` — which Stage 0c shows was
> measured on a raw time-series IC that a common market factor inflates, and on
> survivor cells selected by the model's own choice to split.
>
> See [`stage0-closing.md`](stage0-closing.md).

# Stage 0 Addendum — degeneracy sensitivity sweep

> **This is an ADDENDUM, not an amendment.** Stage 0's committed numbers —
> `mu_hat = -0.05988`, `tau2_hat = 0.00221`, the crosstab, the pre-registration
> — are untouched. Nothing here recomputes them in place, rewrites
> `evidence_grades_v2`, or moves a pre-registered threshold. It is a diagnostic
> that sits beside them.
>
> Separate track from the project's Phase 0-6 roadmap. See
> [`stage0-evidence-grading.md`](stage0-evidence-grading.md) and
> [`stage0-preregistration.md`](stage0-preregistration.md).

Tool: `tools/stage0_degeneracy_sweep.py`. Tests:
`tests/test_stage0_degeneracy_sweep.py`.

---

## The gap this fills

Stage 0 found that **316 of 420 (ticker, fold) cells emit a constant
prediction**, and that 21 tickers do so in all five folds. The grader's v2
guard refuses exactly those 21. That leaves **211 degenerate cells spread
across the other 63 tickers** — tickers that ARE graded and that still feed
`mu_hat = -0.05988`. This asks how much of that number survives once partial
degeneracy is excluded too.

## Method

- **Two independent degeneracy metrics**, checked for agreement rather than
  assumed: `mode_share` (fraction of a fold's predictions equal to its modal
  value) and `unique_fraction` (distinct values / row count, scale-free).
- **A sweep, not a cutoff.** τ ∈ {1.00, 0.90, 0.75, 0.50, 0.30, 0.10}; a fold
  is dropped when its mode-share *exceeds* τ, so **τ = 1.00 drops nothing** and
  must reproduce Stage 0 exactly.
- **The bootstrap is told about the seam.** Dropping fold 2 of 5 leaves rows
  either side months apart, so `block_bootstrap_ic(respect_fold_gaps=True)`
  refuses to draw a block across that join. The flag is strictly additive and
  bit-identical when there is no gap — pinned by a test, because Stage 0's
  numbers are committed and must not move underneath them.
- **The effective ticker count is reported at every τ.** `TRUST_MIN_TICKERS =
  42` (half the frozen universe) is declared up front; cells below it are
  flagged `UNTRUSTED`.

## Do the two metrics agree?

Spearman(`mode_share`, `unique_fraction`) = **−0.858** over all 420 cells, and
the cells each one flags overlap at **96.5–99.8%** at every τ. They are
measuring the same thing.

## The sweep

| τ | folds dropped | tickers gone | n_eff | folds/ticker | `mu_hat` | sd | z | `tau2_hat` | STRONG | WEAK | trust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1.00** | 0 | 0 | **63** | 5.00 | **−0.05988** | 0.01229 | −4.87 | 0.00221 | 0 | 0 | ok |
| 0.90 | 332 | 26 | 58 | 1.52 | **+0.04111** | 0.02038 | +2.02 | 0.02001 | 1 | 8 | ok |
| 0.75 | 347 | 32 | 52 | 1.40 | **+0.05953** | 0.02259 | +2.63 | 0.02132 | 1 | 8 | ok |
| 0.50 | 382 | 51 | 33 | 1.15 | +0.06644 | 0.02885 | +2.30 | 0.03937 | 0 | 8 | **UNTRUSTED** |
| 0.30 | 396 | 62 | 22 | 1.09 | +0.08141 | 0.03681 | +2.21 | 0.02363 | 0 | 4 | **UNTRUSTED** |
| 0.10 | 417 | 81 | 3 | 1.00 | +0.13288 | 0.11746 | +1.13 | 0.00000 | 0 | 0 | **UNTRUSTED** |

**τ = 1.00 reproduces Stage 0 exactly** — `mu_hat` and `tau2_hat` to the last
digit, `n_effective` 63. The sanity check passes, so the rows below it are
measuring the data and not a reimplementation.

A robustness spot-check at **block 60** reproduces the whole shape
(−0.0577 → +0.0461 → +0.0661 → …), so none of this is a block-length artifact.

## The flip is ABRUPT, and it is not partial degeneracy

**The entire sign change happens in the first step, τ 1.00 → 0.90.** That step
drops 332 of 420 folds at once — because the degeneracy distribution is
**bimodal, not a continuum**: 316 cells sit at mode-share exactly 1.000, only
**16** lie anywhere in (0.90, 1.00), and the remaining 85 are spread thinly
across the whole rest of the range.

So the premise the addendum was commissioned on turns out to be wrong in a
useful way. There is essentially **no "partially degenerate" population**. The
211 ungated cells are almost all *fully* constant — the v2 guard's binary test
was the right shape, it was simply applied at the ticker level when the problem
lives at the cell level.

## Reconciliation with the −0.070 → +0.126 figure

Those two operations are different and the sweep sits between them:

| | value | over |
|---|---|---|
| per-ticker IC **pooled** over folds | **−0.0700** | 84 tickers |
| per-ticker IC **averaged within** folds | **+0.1262** | 63 tickers |
| sweep, unweighted mean `hat_ic` at τ = 0.75 (trusted) | **+0.0565** | 52 tickers |
| sweep, unweighted mean `hat_ic` at τ = 0.10 (untrusted, n = 3) | +0.1473 | 3 tickers |

**The trusted end of the sweep falls well short of +0.126** and only the
untrusted τ = 0.10 cell overshoots it. That is the expected relationship:
wholesale within-fold scoring removes the fold level from *every* ticker at
once, while exclusion keeps the level inside whatever folds survive. `+0.126`
is an upper bound on what removing the level can do; `+0.06` is what a
defensible exclusion rule actually buys.

## The finding underneath: a positive within-fold IC

Scoring only non-degenerate cells, each inside its own fold — so the
fold-level artifact is removed **by construction** rather than swept for:

| fold | cells | mean IC | sd | positive |
|---|---|---|---|---|
| 0 | 9 | +0.2087 | 0.1161 | 8/9 |
| 1 | 43 | +0.1123 | 0.1672 | 31/43 |
| 2 | 15 | +0.0998 | 0.1817 | 11/15 |
| 3 | 11 | +0.1237 | 0.2087 | 10/11 |
| 4 | 26 | +0.1773 | 0.2576 | 20/26 |

Treating the **five fold means as the units** — which is the right altitude,
because every ticker in a fold trades the same market over the same months —
gives mean **+0.1444**, sd 0.0465, **t +6.94, p 0.0023**, and all five folds
carry the same sign (P = 1/32 under a symmetric null).

**Both z-statistics in the sweep table are overstated**, and this is why. The
precision-weighted standard error treats 52–63 tickers as independent
observations when the panel holds five periods. The −4.87 at τ = 1.00 and the
+2.63 at τ = 0.75 are the same error in opposite directions — the
`effective_sample_size` family of mistake, at the cross-section instead of
along time.

### Why this is not yet a result

1. **Selection.** These 104 cells are the survivors of 420, selected on the
   model having made any split at all. Whether that is independent of whether
   the model was *right* is untested, and it is the biggest threat.
2. **It is the wrong quantity.** This is a per-ticker **time-series** rank IC —
   "within this stock, dates the model ranked higher did have higher forward
   returns". Every other null in this project (`pooled_xgb` reb_IC +0.0389,
   `beta_market` +0.0464, the break-even 0.0051) is **cross-sectional**. A
   positive time-series IC does not contradict the cross-sectional null; they
   are different claims and only the cross-sectional one is tradeable as a
   book.
3. **A market factor moves all names together.** If the models' predictions
   track anything regime-like, every ticker in a fold gets a positive
   time-series IC at once — which is exactly the pattern observed. P5 measured
   beta's *cross-sectional* channel at R² 0.065; the time-series channel has
   never been tested.
4. **Deflation.** 6 τ cells, on a panel carrying ~131 prior trials.

## Root cause of the degeneracy — CONTEXT, not a finding

| fold | cells | constant | % | mean rows | mean target sd |
|---|---|---|---|---|---|
| 0 | 84 | 75 | 89.3 | 387 | 0.14631 |
| 1 | 84 | 41 | 48.8 | 387 | 0.11272 |
| 2 | 84 | 69 | 82.1 | 387 | 0.10170 |
| 3 | 84 | 73 | 86.9 | 387 | 0.09498 |
| 4 | 84 | 58 | 69.0 | 361 | 0.08398 |

Degeneracy is high in **every** fold and does not track training size (fold 3
has ~4× fold 0's training data and is 86.9% constant) or target dispersion
(which falls monotonically while degeneracy does not).

**The mechanism is measurable and it is the tuner.** For degenerate cells, the
constant sits a median of **0.0051 from the mean of the preceding period —
0.043 of a standard deviation — and 97.5% of them are within a quarter of an
sd.** The constant *is* the training mean: the trees are making **no splits at
all**.

That is a coherent story rather than a mystery. `pipeline/tuning.py` searches
`gamma` over [0, 5] and `min_child_weight` over [5, 40], and scores candidates
on **mean absolute error** of a 30-session return whose dispersion is ~0.10.
The total loss reduction any split can offer on that target is tiny, so a gamma
much above zero makes every split unprofitable — and MAE positively *rewards*
the resulting constant, because on a near-random-walk target the training mean
is a strong MAE predictor.

**So the tuner optimises a metric under which a constant is near-optimal, while
the gate grades a rank IC that a constant cannot express at all.** That
mismatch is Stage 2's territory and nothing here attempts to fix it.

## What this changes

Stage 0's headline conclusion — **0 STRONG, 0 WEAK, the constraint is signal
not measurement** — stands at the pre-registered configuration and is not
revised.

What the addendum adds is that `mu_hat = -0.05988` is **not a measurement of
skill at all**: it is dominated by fold-level constants, it flips sign the
moment they are excluded, and its standard error is overstated in both
directions. It should be read as a diagnostic of a broken model, not as
evidence that the models are backwards.

And underneath it there is a positive within-fold time-series IC that Stage 0's
headline was masking, which is heavily caveated but is the first thing on this
panel that is not obviously zero.
