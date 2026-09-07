# Stage 2a pre-registration — the tuner-objective pilot

**Written 2026-09-07, before any real-data run of Step A.**

Branch `stage2a-tuner-objective-pilot`, off `main`. This is a narrow
falsification test of one hypothesis, not Stage 1 (new data) and not the full
Stage 2 (learning-to-rank loss, triple-barrier labels, pooled-model
consolidation, horizon sweep).

Separate numbering track from the project's own Phase 0-6 roadmap, and a
successor to the Stage 0 addendum. Nothing here touches
`stage0-evidence-grading`.

---

## The hypothesis under test

The Stage 0 addendum measured that **316 of 420 (ticker, fold) model fits emit a
constant prediction**, that the constant sits a median of **0.043 standard
deviations** from the preceding period's mean — i.e. the trees make no splits at
all — and that `pipeline/tuning.py` searches `gamma` over [0, 5] while scoring
candidates on **MAE** of a target with dispersion ~0.10, under which a constant
is frequently near-optimal.

> **H1.** The MAE tuning objective, over a wide `gamma` range, is what drives the
> degeneracy. Lowering `gamma` at otherwise-identical hyperparameters should
> restore splits and improve the within-fold rank IC.

## The decision rule, fixed in advance

> This pilot succeeds as a falsification test if Step A shows that lowering
> gamma reduces constancy and improves within-fold rank IC across most sampled
> degenerate cells. If it does not, stop before Step B and report that the
> tuning-objective hypothesis is not supported (or is incomplete), rather than
> proceeding to a pilot retune anyway. If Step B runs, it succeeds as a useful
> pilot if the degeneracy rate drops materially among previously-degenerate
> pilot tickers without materially worse MAE, and without degrading the
> previously-clean control tickers — not merely if the STRONG/WEAK count rises.

**"Most sampled cells" is fixed at ≥ 6 of the 10** before the run, so that
"most" cannot be reinterpreted afterwards. Both halves must hold in the same
cell for it to count: constancy must fall (unique fraction rises above its
degenerate value) **and** the within-fold rank IC must improve on the cell's own
as-tuned value.

## How the cells are chosen — declared before they are seen

Cherry-picking which cells to report is the failure mode this section exists to
prevent, so the rule is mechanical and outcome-blind:

1. Score every one of the 420 (ticker, fold) cells in `evidence_oos.npz` by
   `mode_share` — the fraction of a fold's predictions equal to its modal value.
   Cells at **exactly 1.000** are the fully-constant population.
2. For each fold 0…4 in order, take the **first two tickers alphabetically**
   among that fold's fully-constant cells, skipping any ticker already selected.
3. That yields **10 cells, 10 distinct tickers, 2 per fold** — spanning every
   fold, clustered in none.

No cell is dropped after the fact. Every selected cell appears in the reported
table whatever it shows.

## The gamma grid

`{0, 0.5, 1, 2}` — a sweep, not a single point, per the standing rule that a
result measured at one hyperparameter setting is not a result. Every other
hyperparameter is held at whatever the nested Optuna search chose for that
exact cell, re-derived by re-running the seeded, unmodified `tune()` on that
fold's own training slice.

**Reproduction check, and it is a stop condition.** Refitting at the as-tuned
parameters must reproduce the cached out-of-sample predictions for that cell to
within 1e-9. If it does not, the refit harness is not measuring the same thing
the Stage 0 cache recorded, and the run stops for debugging before any gamma is
varied. This is the same role τ = 1.00 played in the addendum.

## What is measured, per (cell, gamma)

- `unique_fraction` — distinct predicted values / rows. The constancy measure.
- `mode_share` — the second, independent constancy measure carried over from
  the addendum, reported so the two can disagree visibly.
- **within-fold rank IC** — Spearman between prediction and realised return
  inside that one fold. Never pooled across folds; pooling is the artifact the
  addendum diagnosed and it must not be reintroduced here.
- **MAE** — reported for every cell and every gamma, always beside the rank IC.
  A reduction in degeneracy bought with materially worse MAE is a finding
  against the fix, not a footnote.

## Predictions on record

Recorded so that being wrong is visible rather than reinterpretable. The Stage 0
pre-registration got two of three wrong and they stayed on the record; the same
applies here.

1. **Constancy falls sharply at `gamma = 0`** in most sampled cells — this is
   the direct consequence of the mechanism the addendum measured.
2. **Within-fold rank IC improves in a majority of cells but by less than the
   addendum's +0.144**, because the addendum's figure is measured on cells
   selected for having split, and this one is not.
3. **MAE gets slightly worse at `gamma = 0`.** If MAE were not worse, the MAE
   objective would not have chosen the constant in the first place, so a
   *better* MAE at gamma 0 would be evidence the mechanism is misdiagnosed.

Prediction 3 is the important one: it means the expected outcome of this pilot
is a **trade**, not a free win, and Step B must be judged on the size of the
trade rather than on the direction of the rank IC alone.

## Non-goals

- No change to `pipeline/model.py`'s inference path, `agents/**`, `api/**` or
  `web/**`.
- The existing MAE tuning path stays intact, default and selectable. Step B adds
  an alternative scoring path; it does not replace one.
- No run across the full 84-ticker panel this session.
- Nothing is written to the production hyperparameter cache, to
  `model_metadata`'s production columns, or to anything the live API serves.
- **A falling degeneracy rate is not success on its own.** Neither is a rising
  STRONG/WEAK count — the addendum's own honest read already declined to treat
  its +0.144 within-fold IC as a validated result, and nothing here is entitled
  to more credit than that.
