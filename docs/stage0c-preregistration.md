# Stage 0c pre-registration — closing the evidence-grading track

**Written 2026-09-08, before any of this ran on real data.**

Branch `stage0c-close-evidence-track`, off `stage0b-grading-fix`. The last
session on this track. Nothing is merged to `main` and nothing is deployed;
both decisions are handed back.

---

## The decision rule, fixed in advance

> Stage 0c closes the evidence-grading track. It succeeds if the graded
> quantity is computed correctly (within-fold, rank-demeaned), its uncertainty
> is estimated credibly (date-level block bootstrap with the empirical-Bayes
> pipeline re-run per replicate, cross-checked against Driscoll-Kraay), its
> multiplicity is controlled (Romano-Wolf stepdown), its heterogeneity is
> estimated with REML and its degenerate tau²≈0 case detected and reported once
> rather than 84 times. It does NOT succeed by producing more STRONG or WEAK
> grades. Every change in this session is expected to reduce the graded count;
> a smaller number of defensible grades is the intended outcome. If the
> corrected pipeline grades zero tickers STRONG, that is a complete and
> successful result to be reported as such.

## Why each change is expected to REDUCE the count

Recorded here so a falling count cannot later be mistaken for a regression.

| change | direction | mechanism |
|---|---|---|
| Romano-Wolf stepdown | **fewer STRONG** | 84 simultaneous tests; a nominal per-ticker t > 2 stops being sufficient |
| date-level bootstrap re-running EB per replicate | **fewer STRONG** | the SE absorbs cross-sectional dependence the precision-weighted formula ignored; the implausible z = +5.72 should collapse |
| tau² ≈ 0 detector | **84 grades → 1 statement** | at the zero boundary every posterior IS the grand mean, so the panel is making one claim, not 84 |
| common-subset comparison | **removes a flattering number** | `per-ticker × mae`'s `mu_hat = +0.150` was measured on 10 of 84 names |
| rank-demeaning both sides | either | removes the common market factor a per-ticker time-series IC was partly reading |

## What is being fixed, and why it is a fix rather than a preference

Stage 0b left two problems explicitly open. Stage 0c closes both.

1. **The graded quantity was a raw per-ticker time-series IC.** A market factor
   moving every name together produces a positive time-series IC for every
   ticker at once — which is exactly the pattern Stage 0b's table showed. The
   fix is cross-sectional rank-demeaning of **both** predictions and targets
   within each date (Gu-Kelly-Xiu / Kelly-Pruitt-Su convention: rank
   period-by-period, map to [−1, 1]), so what remains is each name's
   contribution *relative to its own cross-section*.

2. **`mu_hat`'s standard error treated 84 tickers as independent.** They are
   not: one market, ~64 independent 30-session windows. The fix is a
   **date-level circular block bootstrap** that re-runs the entire
   empirical-Bayes pipeline inside every replicate — resample dates within
   fold, recompute every ticker's fold-averaged IC, every σ²ᵢ, τ², `mu_hat` and
   every posterior. The spread across replicates is the standard error, and it
   contains within-fold, between-fold and cross-sectional-dependence
   uncertainty at once.

**Because the bootstrap already contains the between-fold component, the
additive `max(within, between)` term from Stage 0b is switched OFF wherever the
bootstrap is in force.** Keeping both would double-count. It survives only as
the fallback for tickers below `MIN_FOLDS_FOR_ESTIMATE = 3`, and the switch
between the two paths is logged rather than implicit.

## Constants fixed before the run

| constant | value | why |
|---|---|---|
| `MIN_CROSS_SECTION` | 20 | a date with fewer names cannot support a cross-sectional demeaning; mirrors the minimum-firms convention in Fama-MacBeth |
| `BLOCK_LENGTH_SESSIONS` | 30 | the known label horizon, cross-checked against Politis-White `optimal_block_length` |
| `BOOTSTRAP_B` | 1000 | replicates, each a full EB re-run |
| `MIN_FOLDS_FOR_ESTIMATE` | 3 | unchanged from Stage 0b |
| `FDR_Q` | 0.10 | unchanged |
| `STRONG_POSTERIOR_THRESHOLD` | 0.90 | unchanged |

**No threshold is loosened in this session.** `MIN_RANK_IC`, `MIN_IC_TSTAT` and
`MIN_HIT_RATE_EDGE_PP` are untouched.

## The sum-to-zero constraint, and why it needs no separate correction

Cross-sectional demeaning applies the centering matrix `C = I_N − (1/N)11ᵀ`,
which is idempotent with rank `N − 1`. It induces an exact pairwise correlation
of `−1/(N−1)` — **−0.01205 at N = 84** — and removes one degree of freedom per
date. The date-level bootstrap absorbs this by construction, because it
resamples whole cross-sections rather than individual names. It is nonetheless
**tested**, not assumed: a test builds i.i.d. cross-sections, demeans them, and
requires the empirical mean off-diagonal correlation to match `−1/(N−1)`.

Lineage: Pearson (1897) on spurious correlation from indices; Chayes (1960) on
constant-sum correlation; Aitchison (1986) on compositional data.

## Predictions on record

Stage 0's pre-registration got two of three wrong and they stayed on the record;
Stage 0b's prediction that STRONG would stay at zero was wrong too. Same rules.

1. **`per-ticker × mae`'s 10 STRONG does not survive.** It rests on 10 of 84
   names and a z computed as though tickers were independent.
2. **The tau² ≈ 0 detector fires on at least one pooled variant.**
   `pooled × mae` already reported τ² = 0.00018 in Stage 0b, which is at the
   boundary in all but name.
3. **The corrected pipeline grades ZERO tickers STRONG on any variant whose
   `n_usable` is 84.** The variants that grade names are the ones that grade
   few names.

If prediction 3 is wrong — if a full-population variant survives Romano-Wolf —
that is the first real per-ticker evidence this project has produced, and it
must then be checked against the cross-sectional bar before anyone acts on it.

## Non-goals

- **The fold-0-peaks / fold-4-negative pattern is NOT investigated.** Six
  instances including a pure-noise placebo. Recorded as the top open item.
- No new data, no retraining, no re-tuning, no change to model, target or
  horizon.
- No FarmTest, no rpy2.
- **The previously committed Stage 0 / addendum / 2a / 2b / 0b documents are not
  rewritten.** Superseded numbers get a prepended correction notice pointing at
  the closing document. Annotate, never erase — the standard this project set
  when it retracted the ~4.3% MAPE and ~85% directional-accuracy figures.
- Nothing is merged to `main`, nothing is deployed, and the `/research` page is
  not edited. Findings about it are reported for the hand-back.
