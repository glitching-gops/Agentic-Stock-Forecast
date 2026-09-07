# Stage 2a — the tuner objective IS the cause, and changing it is not the fix

> A narrow, falsifiable test of one hypothesis that fell out of the Stage 0
> addendum. **Not** Stage 1 (new data) and **not** the full Stage 2
> (learning-to-rank loss, triple-barrier labels, pooled-model consolidation,
> horizon sweep). Branch `stage2a-tuner-objective-pilot`, off `main`.
>
> Pre-registered before either step ran on real data:
> [`stage2a-preregistration.md`](stage2a-preregistration.md).
> Tools: `tools/stage2a_gamma_spotcheck.py`, `tools/stage2a_pilot.py`.
> Tests: `tests/test_stage2a_tuning_objective.py`, `tests/test_stage2a_pilot.py`.
>
> **Shadow only.** Nothing here writes a hyperparameter cache, a
> `model_metadata` row, a forecast, or anything the API or the web app serves.
> The existing MAE tuning path is untouched, default, and pinned by a test.

---

## The hypothesis

The Stage 0 addendum measured that **316 of 420 (ticker, fold) fits emit a
constant prediction**, that the constant sits a median of **0.043 standard
deviations** from the preceding period's mean — so the trees make no splits at
all — and that `pipeline/tuning.py` scores candidates on **MAE** of a target
whose dispersion is ~0.10, under which a constant is frequently near-optimal.

> **H1.** The MAE objective is what drives the degeneracy. Lowering `gamma` at
> otherwise-identical hyperparameters should restore splits and improve the
> within-fold rank IC.

---

## Step A — the hypothesis is CONFIRMED, and sharper than stated

Ten fully-constant cells, chosen by the pre-registered outcome-blind rule (two
per fold, ten distinct tickers, alphabetically first within each fold). Every
other hyperparameter held at whatever the nested Optuna search chose for that
exact cell.

**The refit reproduces the Stage 0 cache at drift `0.0e+00` on all ten cells**,
so what follows is measuring the data and not a second implementation. That is
the same role `tau = 1.00` played in the addendum, and it is a stop condition.

| ticker | fold | as-tuned γ | γ=0 uniq | γ=0 IC | γ=0.5 uniq | γ=0.5 IC | γ=1 uniq | γ=1 IC | γ=2 uniq |
|---|---|---|---|---|---|---|---|---|---|
| ABB.NS | 0 | 1.792 | 0.995 | +0.4352 | 0.057 | +0.4232 | 0.005 | +0.4580 | 0.003 |
| ADANIENT.NS | 0 | 1.943 | 1.000 | −0.0344 | 0.003 | — | 0.003 | — | 0.003 |
| AMBUJACEM.NS | 1 | 1.792 | 0.954 | +0.0310 | 0.008 | +0.3101 | 0.003 | — | 0.003 |
| APOLLOHOSP.NS | 1 | 2.280 | 0.974 | +0.0948 | 0.003 | — | 0.003 | — | 0.003 |
| ADANIENSOL.NS | 2 | 4.934 | 0.979 | +0.0966 | 0.251 | +0.2417 | 0.067 | +0.3088 | 0.031 |
| ADANIPOWER.NS | 2 | 4.934 | 0.985 | +0.3196 | 0.302 | +0.2190 | 0.083 | +0.2340 | 0.021 |
| ASIANPAINT.NS | 3 | 1.943 | 1.000 | +0.3308 | 0.013 | +0.4366 | 0.003 | — | 0.003 |
| BAJAJ-AUTO.NS | 3 | 1.943 | 1.000 | −0.3187 | 0.010 | −0.1701 | 0.003 | — | 0.003 |
| AXISBANK.NS | 4 | 1.792 | 1.000 | +0.3403 | 0.008 | +0.1336 | 0.003 | — | 0.003 |
| BAJAJFINSV.NS | 4 | 4.934 | 0.942 | +0.4785 | 0.191 | +0.2508 | 0.025 | +0.0614 | 0.003 |

Aggregated over the ten cells:

| γ | cells still constant | mean unique fraction | IC defined | mean IC | positive |
|---|---|---|---|---|---|
| **0.0** | **0 / 10** | 0.983 | 10 | +0.177 | 8 / 10 |
| 0.5 | 2 / 10 | 0.085 | 8 | +0.231 | 7 / 8 |
| 1.0 | 6 / 10 | 0.020 | 4 | +0.266 | 4 / 4 |
| 2.0 | 8 / 10 | 0.007 | 2 | +0.243 | 2 / 2 |

**Do not read the mean-IC column downward as a trend.** n changes with γ because
the cells that still split are a different, self-selected set — the same class
of error as reading a `t` column across a horizon grid whose sample size moves.

**10 of 10 cells support H1** against a pre-registered threshold of 6. Under the
stricter reading — the split ordering must also be *positive*, not merely
defined — it is **8 of 10**, ADANIENT (−0.034) and BAJAJ-AUTO (−0.319/−0.170)
being the exceptions. Either way the gate is cleared, so the run proceeded to
Step B.

### The sharper version: the search OFFERS low gamma and MAE REJECTS it

`tune()` runs a seeded TPE at `n_trials=10`, which is inside Optuna's random
startup phase — so the ten proposed configurations are **identical for every
ticker and every fold**, whatever the objective values come back as. Measured
directly, the ten proposed gammas are:

```
0.290, 0.917, 2.280, 4.744, 0.172, 3.876, 1.943, 4.934, 1.792, 3.188
```

**Three of the ten are below 1.0.** Not one of the ten sampled cells picked any
of them — every single as-tuned gamma is 1.792, 1.943, 2.280 or 4.934.

That reframes H1. The problem is **not** that the `[0, 5]` search range fails to
offer a splitting configuration; it offers three, and MAE turns all three down.
The binding constraint is the objective, which is what Step B changes.

---

## Step B — the fix works, and FAILS its pre-registered success criterion

`pipeline/tuning.py` gains `tuning_objective`, defaulting to the unchanged
`"mae"`. The alternative `"rank_ic"` scores the **mean of within-fold Spearman
correlations, never pooled** — pooling is the artifact the addendum diagnosed at
the evaluation stage and it must not be reintroduced at the tuning stage — and
gives a fold whose predictions are constant a penalty of **2.0**, strictly worse
than the +1.0 that a perfectly *inverted* predictor would score, so Optuna must
actively steer away from a constant rather than merely decline to reward it.

Fourteen tickers, stratified alphabetically within stratum, both arms run
through the same harness on the same folds. The MAE arm reproduces the Stage 0
cache at drift `0.0e+00` for all fourteen.

> **The requested control group does not exist.** Exactly **one** ticker of 84
> (CGPOWER.NS) is non-constant in all five folds. The control stratum is the
> least-degenerate population available — at most one constant fold — and is
> underpowered at n = 4.

| stratum | n | cells | constant before | constant after | IC before | IC after | MAE before | MAE after | train/OOS gap before | after |
|---|---|---|---|---|---|---|---|---|---|---|
| **degenerate** | 5 | 25 | 25 (100%) | **7 (28%)** | undefined | +0.0545 | 0.07589 | **0.08282** | −0.00076 | **+0.01254** |
| **partial** | 5 | 25 | 17 (68%) | **5 (20%)** | +0.3430 | +0.1838 | 0.10163 | **0.10299** | +0.01803 | **+0.02964** |
| **control** | 4 | 20 | 3 (15%) | **1 (5%)** | +0.1300 | +0.1443 | 0.15865 | **0.17373** | +0.02923 | **+0.07394** |

Whole pilot: constant cells **45/70 → 13/70**, mean selected gamma **2.482 →
0.403**, mean out-of-sample MAE **0.10873 → 0.11600 (+6.7%)**.

### The IC columns above are NOT comparable, and the matched read is worse

A constant cell has an undefined IC, so the "before" mean is taken over 25 cells
and the "after" over 57 — a comparison whose sample changed underneath it, which
is the landmine that retired the valuation lag sweep. Held to matched cells:

| | cells | IC before | IC after | improved | MAE before | MAE after |
|---|---|---|---|---|---|---|
| **non-constant in BOTH arms** | 25 | **+0.1982** | **+0.1859** | **9 / 25** | 0.14245 | 0.15404 |
| newly split (constant before) | 32 | undefined | **+0.0859** | — | 0.09467 | 0.09713 |
| still constant after | 13 | undefined | undefined | — | — | — |

**Where an ordering already existed, the rank-IC objective did not improve it** —
+0.198 → +0.186, better in 9 of 25 cells, which is what a coin flip looks like.
**Where it created one, the ordering is weak**: +0.0859, positive in 18 of 32,
and its `t` of +1.83 treats 32 cells as independent when they come from 14
tickers over 5 periods and one market, so the real number is smaller.

### The overfitting signature is unambiguous

**Training MAE falls in 55 of 70 cells while out-of-sample MAE rises in 39 of
70.** The train/out-of-sample gap goes **+0.01452 → +0.03619**, a 2.5× widening,
and it is *worst in the control stratum* — the tickers that were already
splitting fine (+0.029 → +0.074, with ADANIENSOL at +0.027 → +0.099).

### Against the pre-registered criterion

> "…succeeds as a useful pilot if the degeneracy rate drops materially among
> previously-degenerate pilot tickers **without materially worse MAE**, and
> **without degrading the previously-clean control tickers**."

| clause | verdict |
|---|---|
| degeneracy drops materially among previously-degenerate tickers | **met** — 100% → 28% |
| without materially worse MAE | **NOT met** — +9.1% on that stratum, +6.7% overall |
| without degrading the previously-clean control tickers | **NOT met** — control MAE +9.5%, gap 2.5× |

**Step B does not clear its own bar.** Two of three clauses fail, and the
STRONG/WEAK count was never the criterion.

---

## Why it fails, measured rather than guessed

The inner CV that scores each trial is a 3-fold purged walk-forward over the
outer training slice. At the 30-session label horizon its independent-observation
count per inner fold is:

| outer fold | inner train rows | inner test rows | **n_eff per inner fold** |
|---|---|---|---|
| 0 | 440 | 98 | **3.3** |
| 1 | 827 | 184 | 6.1 |
| 2 | 1,214 | 270 | 9.0 |
| 3 | 1,601 | 356 | 11.9 |
| 4 | 1,988 | 442 | 14.7 |

**The objective is selecting the best of ten configurations on a rank IC
estimated from as few as 3.3 independent observations.** MAE is a far
lower-variance statistic at that sample size, which is why swapping to rank IC
does not merely change *which* configuration wins — it makes the choice mostly
noise, and the noise is spent buying splits.

This project has recorded that exact failure before. Phase 2's linear probe:
*"Selecting alpha on ranking IC made it WORSE at both contexts (+0.0211 to
−0.0033 at 512). Optimising a noisy inner ranking metric overfits the selection
itself."* Stage 2a reproduces it with XGBoost hyperparameters in place of a
ridge penalty.

**So the honest reading is that MAE was doing double duty.** It was scoring
accuracy *and* acting as the model-selection regulariser that kept a
nine-dimensional search over ~10 effective observations from running away.
Removing it converts "no splits" into "overfit splits", and the addendum's
diagnosis — that the tuner and the grader optimise different things — is correct
about the cause and incomplete about the remedy.

---

## What this does and does not change

**Stands.** The tuner objective is the cause of the degeneracy. Step A settles
that at drift 0.0 on ten cells with a sweep, not a point.

**Stands.** The Stage 0 addendum's `mu_hat = -0.05988` remains a diagnostic of a
broken model rather than a measurement of skill. Nothing here revises it, and
nothing here touches `stage0-evidence-grading`.

**New, and it is a caution.** The addendum's within-fold IC of +0.1444 was
measured on cells that *chose* to split under MAE. When splits are induced
deliberately, the newly-split cells score **+0.0859** — 60% of it — and only 18
of 32 are positive. That is consistent with the addendum's own first caveat
(survivorship among 104 of 420 cells) being a real effect and not a formality.

**Not established.** That a rank-IC tuning objective helps. On this pilot it
costs 6.7% of MAE, widens the train/test gap 2.5×, and does not improve the
ordering of any cell that already had one.

## Recommendation

**Do not roll this out to the 84-ticker panel as it stands.** The right next
experiment is not "same objective, more tickers" — it is to supply the
regularisation MAE was implicitly providing, and to test that at pilot scale
first. Three candidates, in increasing order of how much they concede:

1. **A composite objective** — rank IC with an MAE guardrail (reject any
   configuration whose inner MAE is materially worse than the training mean's),
   so the selection cannot buy an ordering at any accuracy cost.
2. **Narrow the search space instead of the objective.** Step A showed γ = 0
   splits every cell; `[0, 5]` is almost entirely inside the no-split regime and
   most of the range is doing nothing but hiding this.
3. **Accept the altitude.** A per-ticker model choosing among nine
   hyperparameters on ~10 effective observations is over-specified regardless of
   the metric. This is the argument for the *pooled* cross-sectional model the
   full Stage 2 proposes, where the cross-section supplies the sample size the
   time series cannot.
