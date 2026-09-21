# Stage 1, Pilot 1 — multi-lookback residual reversal: findings

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

**Run 2026-09-13, against `docs/stage1-preregistration.md` (sha256
`2cad8b73…3d5e` at run time).** Branch `stage1-residual-reversal`, off
`main` at `61ad22c`. `tools/stage1_reversal.py`, B = 1000, 10 trials, nine
placebo draws, DK lags 30. Runtime 29.1 min. A separate track from Phase 0-6
and from the Stage 0-0c evidence track.

## The answer

| rule | verdict | what decided it |
|---|---|---|
| S1 — baseline reproduces | **PASS** | 160,435 of 160,435 rows, max drift **0.0e+00** |
| R1 — moves tickers out of INSUFFICIENT | **PASS**, both arms | residual: **ABB.NS** only; raw: ABB, BEL, VBL |
| R2 — survives the nine-draw placebo | **PASS**, by one ticker | residual arm 2 STRONG / 2 RW / tau2 0.00466 against a placebo max of 1 / 1 / 0.00027 |
| R3 — SIGNAL (R1, R2 and a paired cross-sectional t ≥ 2) | **NO** | Δ cross-sectional IC over the baseline **+0.0002, t +0.07** |
| R4 — does residualising matter? | **COMPARABLE** | residual − raw +0.0049, t +0.75 |
| R6 — min_train sweep (triggered by R1 and R2) | nothing | every cell within abs(reb t) < 1 |

**Multi-lookback reversal moves one ticker's grade and does not move the
tradeable measurement at all. Whether the residual is taken against the market
makes no measurable difference.** By the pre-registered rules that makes it not
signal. The failing clause is R3, the one written to catch exactly this: a
per-ticker grade that the cross-sectional measurement does not support.

## S1 — the baseline reproduces exactly

The stored Stage 2b arm `pooled_mae_noticker` was re-run through the same
`run_arm` on the same frozen panel. It chose the same per-fold gammas (1.792,
0.290, 4.934, 4.934, 4.934) and returned identical predictions on identical
rows, at drift 0.0. So everything below is a comparison against the baseline
that Stage 0c graded, not against a near relative of it. The re-graded baseline
matches Stage 0c to the last digit: mu_hat −0.01727, z −1.24, tau2 0.00435,
1 STRONG / 1 WEAK / 82.

## The three arms

| arm | mu_hat | boot SE | z | tau2 | STRONG | WEAK | INSUFF | RW | BH | BY | per-date cs IC (DK SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) baseline | −0.01727 | 0.01390 | −1.24 | 0.00435 | 1 | 1 | 82 | 1 | 3 | 1 | −0.00101 (0.00811) |
| (b) residual reversal | **−0.02564** | 0.01300 | **−1.97** | 0.00466 | **2** | 0 | 82 | 2 | 2 | 1 | −0.00081 (0.00821) |
| (c) raw reversal | −0.00726 | 0.01164 | −0.62 | 0.00572 | **2** | 3 | 79 | 2 | 3 | 2 | −0.00568 (0.00707) |

Note the residual arm's panel mean. Its average demeaned per-ticker IC moves
AWAY from zero, −0.0173 → −0.0256, while its STRONG count rises. The two
extra columns make the average ticker slightly worse ranked, and one ticker
better.

Stage 2b's cell metrics, on 64 non-overlapping rebalances:

| arm | constant cells | reb IC | reb t | cs IC | OOS MAE | gap | cs IC by fold |
|---|---|---|---|---|---|---|---|
| (a) | 7/420 | −0.0014 | −0.09 | −0.0013 | 0.08882 | +0.00297 | +0.035, −0.001, −0.011, −0.012, −0.017 |
| (b) | 6/420 | −0.0027 | −0.16 | −0.0010 | 0.08901 | +0.00329 | +0.029, +0.006, −0.011, −0.012, −0.016 |
| (c) | 9/420 | −0.0039 | −0.26 | −0.0058 | 0.08906 | +0.00313 | +0.010, +0.001, −0.011, −0.012, −0.016 |

Both reversal arms are marginally WORSE than the baseline on MAE and on reb
t. Every arm is positive only in fold 0, the early-fold shape for the seventh
time on this panel.

## Which tickers moved, and why ABB is not evidence

| ticker | (a) grade, θ, RW p | (b) residual | (c) raw |
|---|---|---|---|
| ICICIBANK.NS | **STRONG**, +0.096, 0.055 | STRONG, +0.108, 0.019 | STRONG, +0.144, 0.002 |
| ABB.NS | INSUFFICIENT, +0.055, 1.000 (P(θ>0) 0.869) | **STRONG**, +0.093, 0.089 | **STRONG**, +0.141, 0.003 |
| NESTLEIND.NS | WEAK, +0.077, 0.306 | INSUFFICIENT, +0.050 | WEAK, +0.068 |
| BEL.NS | INSUFFICIENT, +0.019 | INSUFFICIENT, +0.054 | WEAK, +0.098 |
| VBL.NS | INSUFFICIENT, +0.052 | INSUFFICIENT, +0.031 | WEAK, +0.082 |

**Of the residual arm's two STRONGs, one was already STRONG in the baseline.**
The net movement is ABB.NS into STRONG and NESTLEIND.NS out of WEAK: one
name up, one down.

**ABB was a baseline near-miss** (P(θ>0) 0.869). Both reversal arms push it
over the Romano-Wolf line, and the RAW arm pushes it FURTHER (adjusted p
0.003, against 0.089). Whatever moves it is the lookback information, not the
residualising. Its per-fold time-series IC in the residual arm runs +0.43,
+0.01, +0.06, +0.49, −0.28: two of five folds carry it, and the most recent
fold is negative.

## The placebo, and what it cannot rule out

| seed | STRONG | WEAK | RW | tau2 | mu_hat |
|---|---|---|---|---|---|
| 20260913-21 (nine draws) | 0-1 | 0-2 | 0-1 | 0.00000-0.00027 | −0.0037 to +0.0024 |
| **max** | **1** | 2 | **1** | **0.00027** | |
| residual arm | **2** | 0 | **2** | **0.00466** | −0.02564 |

Two draws gave one STRONG each, so across the nine the null handed out 2
STRONGs, in line with Stage 0c's calibration. The residual arm clears the
maximum on all three counts, and R2 passes as written.

**R2 WAS THE WRONG NULL FOR AN ADDED FEATURE, and this is the pilot's most
reusable lesson.** Shuffling an arm's predictions within each date asks
whether the ARM beats noise. It destroys the baseline's own ordering along
with the new features', so it credits the arm with every grade the baseline
already had. Here that is ICICIBANK's STRONG and a tau2 that every real model
on this panel exceeds by 15-20x: the baseline's 0.00435 and the raw arm's
0.00572 would both clear the tau2 bar too. The null for "does adding X help"
is to shuffle the NEW COLUMNS within each date and retrain, nine times, holding
everything else fixed. This was not pre-registered here, so it was not run, and
the verdict does not need it: R3 fails on its own. It should be the placebo in
every later Stage 1 pre-registration.

## The tradeable quantity

Per-date cross-sectional rank IC, paired on identical rows (1,910 dates):

| comparison | mean first | mean second | Δ | DK SE (30 lags) | **t** | t, default 7 lags |
|---|---|---|---|---|---|---|
| (b) residual − (a) baseline | −0.00101 | −0.00081 | +0.00020 | 0.00309 | **+0.07** | +0.09 |
| (c) raw − (a) baseline | −0.00101 | −0.00568 | −0.00467 | 0.00619 | **−0.75** | −1.16 |
| (b) residual − (c) raw | −0.00568 | −0.00081 | +0.00487 | 0.00652 | **+0.75** | +1.10 |

The baseline's cross-sectional IC is −0.001; the residual arm's is −0.0008.
**What a long-short book earns did not move.** The default-lag column shows
why the 30-lag amendment mattered: every t is ~1.5x larger at the default rule,
though none crosses 2 either way.

## Residual against raw, and Da-Liu-Schaumburg

Each feature's own per-date cross-sectional IC against the target, over arm
(a)'s 1,910 out-of-sample dates, DK at 30 lags:

| k | raw IC | raw t | residual IC | residual t | abs(resid) / abs(raw) |
|---|---|---|---|---|---|
| 1 | +0.0058 | +1.57 | +0.0070 | +1.84 | 1.20 |
| 5 | +0.0066 | +0.81 | +0.0082 | +1.05 | 1.24 |
| 10 | +0.0079 | +0.67 | +0.0118 | +1.08 | 1.49 |
| 20 | +0.0173 | +1.09 | +0.0202 | +1.30 | 1.17 |

**All eight point in the REVERSAL direction and none is significant** (max t
+1.84). That runs against the Chui, Ranganathan, Rohit & Veeraraghavan (2023)
expectation of momentum among liquid Indian names, but the sign rests on t
below 2 and is not a finding. Residualising helps each feature by
**1.2-1.5x**. DLS report about **4x** on alpha (6.8x on t).

That the DLS gap does not replicate was predicted, for two reasons:
- Within-date z-scoring already removes the market's common LEVEL from any raw
  return, so a market-only residual can remove only
  `(beta_i − mean beta) · m_k`.
- DLS residualise against three factors, not one.

So the pilot tests a much narrower adjustment than theirs, and the narrow
adjustment is nearly free.

## The predictions, scored

| prediction | held | note |
|---|---|---|
| P1 arm (a) reproduces | yes | drift 0.0 |
| P2 no tradeable improvement | yes | +0.0002, t +0.07 |
| P3 residual and raw comparable within 0.003 | **no** | comparable by t (+0.75), but the gap is 0.0049 |
| P4 at most 1 STRONG per arm, and R2 fails | **no** | 2 STRONG each; R2 passes, by one ticker |
| P5 every feature's IC below 0.02 | **no** | `rev_resid_20` +0.0202 |

The P5 breach was visible in the smoke run, before the real run started:
feature ICs do not depend on training. It was not acted on, and it stays on
the record.

## Caveats

- **Every SE here is likely too small.** Politis-White puts the panel's
  dependence at 35.8-62.5 sessions, beyond the 30-session label and block.
- **Trials accumulate.** This pilot adds 2 arms, 9 placebo draws and 12 sweep
  cells to the ~131 prior configurations on this panel. ABB's adjusted p of
  0.089 is adjusted across 84 tickers within one arm, not across that history.
- **The residual is market-only**, and the baseline already carries `lag1_ret`,
  `lag5_ret`, `roc_10` and three sector-relative returns, so the added
  information is small by construction. That was stated before the run.

## What this means for the next session

The price-only short-horizon information set looks exhausted on this panel.
Adding one of the best-documented anomalies in the literature, in both its raw
and its residualised form, moves nothing a book earns. That argues for the
briefing's next step, NSE delivery percentage, because it is genuinely NEW
information rather than another transform of the same closes. Two changes carry
over from here:
1. The placebo must be the permuted-new-columns retrain described above, not a
   prediction shuffle.
2. R3's paired cross-sectional test stays the deciding clause.
