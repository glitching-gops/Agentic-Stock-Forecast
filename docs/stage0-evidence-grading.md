# Evidence-Grading Redesign — Stage 0: partial pooling, in shadow

> **Numbering.** This is Stage 0 of a five-stage (0-4) evidence-grading
> redesign. It is a **separate track** from the project's own **Phase 0-6**
> roadmap (Correctness, Infrastructure, Forecasting, Agents, Evaluation,
> Research, Horizon sweep) documented in `CLAUDE.md` and the README. Neither
> scheme renumbers the other.

Pre-registration: [`stage0-preregistration.md`](stage0-preregistration.md),
committed before the layer was run on a single real ticker.

---

## What this is

The live gate (`agents/critic_agent.py::grade_evidence`) grades each of 84
tickers on its own held-out track record: rank IC ≥ 0.02, signed IC t ≥ 2.0,
hit rate ≥ 1pp over the majority baseline, 2 of 3 to earn WEAK. Three names
clear it. The three checks pass at rates 0.385 / 0.042 / 0.042, which give an
**expected 3.12** names under independence — the yield is what chance produces.

The reason is not a bug, it is arithmetic. Each ticker's t-statistic is built
from `n_effective = n_oos / horizon ≈ 63.6` independent observations, because
consecutive 30-session labels share 29 of their 30 sessions. At n = 64,
`t ≈ IC·√(n−1)`, so:

| what you want | rank IC you need at n = 64 |
|---|---|
| an expected t of 2.0 | **0.25** |
| 80% power at α = 0.05 | **0.34** |
| what real cross-sectional equity signals carry | **0.02 – 0.05** |

Detecting an IC of 0.04 at t > 2 needs ~2,500 independent observations. No
per-ticker frequentist test can grade most names honestly at 64.

That is the classic "many units, each with too few observations" problem, and
its standard fix is **partial pooling**: let the panel lend each ticker
precision instead of asking each ticker to speak alone.

## The method

Closed-form empirical Bayes — no MCMC, no PyMC. Four lines of algebra over 84
pairs of numbers, each verifiable by hand.

1. **Per-ticker block bootstrap.** Moving-block bootstrap of each ticker's own
   out-of-sample `(y_true, y_pred)` series, block length **30 sessions** — the
   label horizon, the purge width and the embargo width, i.e. the number of
   sessions two neighbouring observations share. 2,000 resamples, seeded per
   ticker. Gives `hat_ic_i` and its sampling variance `sigma2_i`.
   *The out-of-sample series is contiguous — `PurgedWalkForward` opens each
   test window where the previous one closes and carves the purge out of the
   training end — so no resample crosses a purge gap.*
2. **Flat pooling** to a precision-weighted (fixed-effect) grand mean
   `mu_hat`, with variance `var_mu = 1/Σw`. Sector-hierarchical pooling is an
   explicit non-goal for this stage.
3. **Between-ticker variance** `tau2` by **DerSimonian-Laird** method of
   moments, clipped at zero.
4. **James-Stein / Efron-Morris shrinkage**:
   `B_i = sigma2_i/(sigma2_i+tau2)`, `theta_i = B_i·mu_hat + (1−B_i)·hat_ic_i`.
5. **Posterior variance** `sigma2_i·tau2/(sigma2_i+tau2) + B_i²·var_mu`. The
   second term is **Morris' correction** and is load-bearing — see below.
6. **P(skill)** `= Φ(theta_i/√v_i)`, Normal approximation. A decision
   statistic, not a calibrated probability.
7. **Benjamini-Hochberg FDR** at `q = 0.10`, ONE family of two-sided p-values
   over the measured tickers; the sign of `theta_i` then routes to the positive
   or the anti-signal branch.
8. **The economic bar** is `pipeline.portfolio.break_even_ic`, reused not
   reimplemented, at the recorded panel `spread_per_ic = 0.3474` and turnover
   0.80: **0.00512** at zero impact.

### The grade

| BH significant | sign(θ) | P(skill) ≥ 0.90 | θ ≥ break-even | grade |
|---|---|---|---|---|
| yes | negative | — | — | **ANTI_SIGNAL** |
| yes | positive | yes | yes | **STRONG** |
| yes | positive | yes | no | **WEAK** |
| yes | positive | no | — | **WEAK** |
| no | — | yes | — | **WEAK** |
| no | — | no | — | **INSUFFICIENT** |

A ticker is **INSUFFICIENT with a stated reason**, and **excluded from the BH
family**, when it cannot be measured at all — counting an untested null into the
family makes every real p-value harder to reject for a bookkeeping reason.
Three cases qualify:

- fewer than 90 out-of-sample rows (three blocks);
- a prediction constant over the whole series, so rank IC is undefined;
- **a prediction constant within *every* walk-forward fold** — added in grader
  `v2` after the first run, see "The artifact" below. Such a ticker holds no
  ordering anywhere, and its pooled rank IC is entirely the arrangement of its
  fold-level constants against the realised period returns.

Every run also prints a **fold diagnostic** (section 2 of the report) measuring
how much of the graded quantity is that fold-level channel rather than a
ranking. It qualifies every number below it.

## Two ways this could have lied, both guarded

**1. `tau2 = 0` collapsing the posterior.** The classical Efron-Morris
posterior variance is `sigma2·tau2/(sigma2+tau2)`, which is **exactly zero**
when `tau2` is zero — the single most likely outcome on this panel. Every
theta would collapse onto `mu_hat` with infinite confidence, every p-value
would hit 0 or 1, every BH test would reject, and the run would report a full
board of STRONG grades out of a degenerate limit. Morris' `B_i²·var_mu` term
supplies the correct limit: at `tau2 = 0` the posterior variance IS the
variance of the grand mean. Mutation-verified.

**2. A block shorter than the label overlap.** It would understate `sigma2`,
make every posterior over-confident, and produce grades from a bootstrap that
broke the dependence it exists to respect. `--block-sweep` reports the answer
at blocks 10/20/30/45/60 beside the pre-registered 30.

**And one honest limitation that is not a bug.** When `tau2 = 0` the shrinkage
weight is 1 for every ticker, so **every measured name receives an identical
posterior and an identical grade**. That is one statement about the universe
printed N times, not N per-ticker findings, and the whole board can only move
together. `PanelGrading.degenerate` flags it, the report prints a banner, and
the sentence is appended to every stored `grade_v2_reason` so a reader of the
table cannot miss it.

## Shadow mode

Nothing in Stage 0 writes to `forecast_current`, `forecasts`, `model_metadata`
or any column the public API or the Next.js frontend serves.
`forecast_confidence` is untouched and the old gate runs unmodified as the
baseline. Grades land in a new table, `evidence_grades_v2`, **append-only** by
`(run_id, ticker)` for the same reason `forecast_outcomes` is: a past reading
must not be quietly improved.

The DDL lives in `pipeline/evidence_shrinkage.py::init_evidence_tables`,
following `data.universe.init_universe_tables` rather than `data.db.init_db` —
`data/db.py` is imported by the API, so putting a research table there would
make a shadow-mode change a reason to redeploy Render. **No Render redeploy is
needed for this branch.**

`tests/test_stage0_evidence_shrinkage.py::test_stage0_writes_nothing_the_public_api_serves`
asserts this rather than intending it.

## Reproducing a run

```powershell
# python is not on PATH in this environment
$py = "C:\Users\venuw\AppData\Local\Programs\Python\Python313\python.exe"

# Full run: regenerate every ticker's track record (~65 min), grade, report,
# sweep the block length, and write the shadow table.
& $py tools/run_evidence_grading.py --rebuild --store --block-sweep `
      --verify-against-persisted --csv evidence_grades_v2.csv

# Re-grade an existing cache in seconds (no model fitting).
& $py tools/run_evidence_grading.py --no-rebuild --block-sweep

# Tests
& $py -m pytest tests/test_stage0_evidence_shrinkage.py -q
```

The report prints, in order: the panel diagnostics (`mu_hat`, `tau2_hat`,
Cochran's Q, the break-even bar), the **old-grade against new-grade crosstab**,
the ten largest posteriors in both tails, the block-length robustness table,
and the pre-registered reading of the result.

## Why it costs an hour, and the one-line follow-up that would remove it

**The per-ticker walk-forward's predictions are not persisted anywhere.**
`pipeline.model.evaluate_and_persist_ticker` writes
`WalkForwardResult.metrics` into `model_metadata` and the conformal residuals
alongside, then drops `WalkForwardResult.predictions` when it returns. The only
surviving trace of a ticker's track record is four scalars, and four scalars
cannot be bootstrapped — so "consume the weekly job's existing output" is not
literally possible today.

The tool therefore **re-runs `pipeline.model.evaluate_ticker` unmodified**.
This is safe rather than a hidden re-derivation: Optuna's TPE sampler is seeded
and XGBoost's `random_state` is fixed, so the same input frame yields
bit-identical predictions (verified in-process, and re-verified on every run by
`--verify-against-persisted`). It costs ~47 s a ticker.

**The follow-up, for the user to approve:** persisting `result.predictions` in
`evaluate_and_persist_ticker` would let this step read the weekly job's own
output for free and would halve the weekly CI budget. It is not done here
because `pipeline/model.py` is explicitly out of Stage 0's scope.

## Orchestration

A new `continue-on-error` step at the end of
`.github/workflows/weekly-evaluation.yml`, after the per-ticker evaluation has
persisted and after the tuned-parameter cache has been saved. It publishes
nothing, so — like the baseline comparison — a defect in it must cost a
recorded skip rather than an hour of persisted evaluation. Nothing is added to
the daily pipeline.

## Dependencies

None added. Benjamini-Hochberg is six lines of numpy rather than a dependency
on `statsmodels`, because `requirements.txt` is installed by Render, by the
daily pipeline and by the weekly evaluation, and this project has a standing
rule against growing that file for anything the live path does not need (the
same rule that keeps torch out of it). The test suite cross-checks the
implementation against `statsmodels.stats.multitest.multipletests` when that
package happens to be importable, and against the published Benjamini-Hochberg
(1995) worked example when it is not.

## References

- James & Stein (1961), *Estimation with quadratic loss*.
- Efron & Morris (1973, 1975), *Stein's estimation rule and its competitors*;
  *Data analysis using Stein's estimator and its generalizations*.
- Morris (1983), *Parametric empirical Bayes inference: theory and
  applications* — the posterior-variance correction for an estimated mean.
- DerSimonian & Laird (1986), *Meta-analysis in clinical trials* — the
  method-of-moments between-study variance.
- Benjamini & Hochberg (1995), *Controlling the false discovery rate*.

---

# The run — 2026-09-06

84 tickers, 160,296 out-of-sample rows, `rebuild-absolute-return-v3`,
`config_hash 01196443967ef0a0`, `data_hash f45ee0d30fe2b909`.
Shadow table run_ids: `d692631cd1fa47e5` (grader v1, pre-guard) and
`3dffcfe1242148fb` (`stage0-eb-shrinkage-v2`, the reported one).

## The crosstab

Old and new are measured on the **same regenerated rows**, so the difference is
the method and not the week's extra data.

| old \ new | INSUFFICIENT | WEAK | STRONG | ANTI_SIGNAL |
|---|---|---|---|---|
| WEAK | 2 | 0 | 0 | 0 |
| INSUFFICIENT | 82 | 0 | 0 | 0 |

**old: 2 WEAK / 82 INSUFFICIENT → new: 0 STRONG / 0 WEAK / 84 INSUFFICIENT.**
(The live board carries 3 WEAK from the 2026-09-05 weekly run; the third moved
on the week of extra data, which is exactly why the crosstab does not use it.)

## The panel diagnostics

| | |
|---|---|
| tickers measured | **63** (21 refused as unmeasurable) |
| `mu_hat` — precision-weighted grand mean IC | **−0.05988**, sd 0.01229 |
| `mu_hat` — random-effects (diagnostic) | −0.05993 |
| `tau2_hat` — between-ticker variance | **0.00221** (sd 0.047) |
| Cochran's Q | 76.38 against E[Q] = 62 |
| break-even rank IC | 0.00512 |

`tau2_hat` is **positive**, so the panel is not degenerate: there is real
between-ticker variation. And `mu_hat` is negative at roughly five standard
errors. **Neither of those means what it appears to mean** — see the artifact
below.

## Block-length robustness

The pre-registered block is 30. The neighbours are diagnostics, not a menu.

| block | `mu_hat` | `tau2_hat` | mean σ | STRONG | WEAK | ANTI | INSUF |
|---|---|---|---|---|---|---|---|
| 10 | −0.05977 | 0.00735 | 0.0679 | 0 | 2 | 8 | 74 |
| 20 | −0.05975 | 0.00407 | 0.0885 | 0 | 0 | 0 | 84 |
| **30** | **−0.05988** | **0.00221** | **0.0988** | **0** | **0** | **0** | **84** |
| 45 | −0.05815 | 0.00129 | 0.1040 | 0 | 0 | 0 | 84 |
| 60 | −0.05774 | 0.00110 | 0.1053 | 0 | 0 | 0 | 84 |

**Zero STRONG at every block length.** The graded counts at block 10 — 2 WEAK
and 8 ANTI_SIGNAL — are the second pre-registered failure mode caught in the
act: a block a third of the label overlap understates σ by ~30% and
manufactures ten grades out of it. σ is still rising at block 60, so the
pre-registered 30 is the *least* conservative choice in the credible range, and
it already grades nothing.

## THE ARTIFACT: the graded quantity is barely a ranking

This is the most useful thing Stage 0 produced, and it was not what the stage
was looking for.

| | |
|---|---|
| (ticker, fold) cells | 420 |
| ... emitting a **constant** prediction | **316 (75.2%)** |
| ... out-of-sample rows inside one | 120,746 / 160,296 (**75.3%**) |
| tickers by number of constant folds | 0:1, 1:3, 2:6, 3:16, 4:37, **5:21** |
| per-ticker IC **pooled** over folds | mean **−0.0700**, 21/84 positive |
| per-ticker IC **within** folds | mean **+0.1262**, 51/84 positive |
| fold prediction level vs realised return | ρ = **−0.600** over **5** folds |

Three quarters of all (ticker, fold) cells produce a **single repeated number**,
and **21 of 84 tickers do so in every one of their five folds**. Those tickers
contain no ordering information anywhere, yet their pooled rank IC is large and
confident — because a pooled IC over the whole out-of-sample series can rank
rows by *which fold they came from*. With five fold-level constants running
opposite to five realised period returns (ρ = −0.60, n = 5), that channel alone
produces the panel's negative sign.

Removing the fold level **flips** the panel mean from −0.070 to +0.126.

This is `CLAUDE.md`'s pooled-IC landmine — `TrainMeanForecast` scored a pooled
IC of −0.007 while holding no ranking information at all — measured on the live
per-ticker gate, where it turns out to dominate.

**It cost the first run its only non-null grade.** Grader v1 graded WIPRO.NS
**ANTI_SIGNAL** on a pooled IC of −0.3914 whose within-fold IC is **undefined in
all five of its folds**. The v2 guard refuses a ticker whose prediction is
constant within *every* fold, exactly as the live gate's signed-t fix did: it
can only remove a grade, never create one, and it moves no pre-registered
threshold.

**So `mu_hat = −0.0599` is not evidence that these models are reliably
backwards on companies.** It is largely five constants lining up against five
period returns. The honest reading of the sign is that it is uninterpretable at
this altitude, not that it is negative.

## The pre-registered reading

> "If, after shrinkage and FDR control, the number of STRONG/WEAK names does not
> rise materially from the current 3, the honest conclusion is that the
> constraint is signal, not measurement — proceed to Stage 1 (new data
> sources), not to further loosening of this grading scheme."

**STRONG + WEAK went from 2 to 0.** It did not rise. Stage 0's own rule says
the constraint is **signal, not measurement**, and the next move is Stage 1.

Two of the pre-registration's three predictions were wrong and are left on the
record: `tau2_hat` came back **positive**, not zero, and `mu_hat` came back at
−0.060 rather than near zero. The prediction that mattered — **zero STRONG** —
held at every block length.

## What Stage 0 bought

Not more graded names; fewer. What it bought is:

1. **A quantified answer to the power question.** The measurement was not the
   binding constraint. Partial pooling, an honest bootstrap and FDR control
   across the panel produce *fewer* graded names, not more.
2. **A measured defect in the quantity both gates grade** — 75% constant cells,
   a pooled/within-fold sign flip, and a per-ticker IC substantially driven by a
   five-point nuisance correlation. Any later stage that keeps reporting a
   pooled per-ticker rank IC inherits this.
3. **A reusable, tested layer** that will grade any future track record on the
   same panel with shrinkage, FDR and an economic bar — which is what Stage 1's
   new data sources will need to be judged against.

## Open, and for the user to decide

- **Persist `WalkForwardResult.predictions`** in `evaluate_and_persist_ticker`
  (5 lines, `pipeline/model.py`, out of Stage 0's scope) and this step stops
  costing a second full walk-forward every week.
- **The constant-prediction rate is a model finding, not a grading one.** 75% of
  fold-level fits emitting one repeated number says something about the
  per-ticker XGBoost configuration — depth, `min_child_weight`, or simply too
  little signal for a tree to split on. It belongs to Stage 2, and it is
  recorded here because Stage 0 is what found it.
