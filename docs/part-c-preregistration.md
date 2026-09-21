# Part C — the scale diagnostic, pre-registered properly

**Written and hashed before the clean re-run.** Part C was added to P6 AFTER
the degeneracy was observed and was never pre-registered, yet it carries the
headline claim in `docs/p6-findings.md` §7 — that the null is real and the
cross-sectional IC is flat across a 6× range of label width. Everything else in
this project is held to a pre-registration; this is the one loose thread in the
closing argument, and this file closes it.

The original Part C figures are on the record and are what this re-run is
judged against:

| h | original cs IC | original t | constant cells |
|---|---|---|---|
| 5 | +0.01459 | **+2.41** | 0 / 420 |
| 10 | +0.00896 | +0.99 | 0 / 420 |
| 20 | +0.01758 | +1.43 | 0 / 420 |
| 30 | +0.01524 | +1.09 | 0 / 420 |

## The hypothesis

**H-C. No horizon effect survives once label-scale degeneracy is removed.**

The competing hypothesis is the one Part C was built to rule out: that the
h=5 and h=10 nulls in P6's Part A were artefacts of a model that never split,
and that a model which does split finds something there.

## What changes from the original run, and what does not

**Changed, and both are the reason for re-running:**

- The within-date standardised label is now the pipeline's default training
  target, from one implementation in `pipeline/label.py`, rather than a
  transform applied inside one tool.
- The thread count is pinned at `XGB_THREADS = 2`
  (`pipeline/determinism.py`). The original Part C ran at this machine's
  default of 20, and P6 measured that the selected hyperparameters — and
  therefore the model — depend on it.

**Unchanged:** the grid {5, 10, 20, 30}, the purge rule
`max(h, 63)`, `min_train = 500`, five folds, ten trials, the bootstrap block
and Driscoll-Kraay lags at 30, and scoring against the REAL h-session label
rather than the standardised one.

## The deciding rule

**C1 — READ THE PROFILE ON THE IC, NOT ON THE MAXIMUM, AND NOT ON THE t.**
This is P6's own rule A3, pre-registered on 2026-09-05 and reproduced here
because it is what decides this run: `n` changes across the grid, so a shorter
horizon earns a larger t from an identical effect. **The cross-sectional IC is
the comparable statistic; the t belongs to its own horizon only.**

A horizon effect is claimed only if the IC profile has a shape — monotone
across the grid, or a single peak whose neighbours are materially lower. Four
values inside a narrow band, in any order, is no effect regardless of what any
individual t reads.

**C2 — the `min_train` sweep.** Any horizon whose t reaches +2.0 is re-run at
`min_train` 380/420/460/500/540/580 and must hold t ≥ +2.0 at a **majority**
of the six. A result at one setting is not a result; this rule retired the
valuation finding and `pooled_xgb`.

**C3 — reproduction.** The original figures above are the comparator. They are
NOT expected to reproduce exactly, because the thread count changed and P6
measured that this moves the selected hyperparameters. **If they do not
reproduce, that is reported prominently as a finding about the fragility, not
reconciled away.** What must survive is the SHAPE and the VERDICT, not the
digits.

**C4 — the h=5 cell specifically.** It read t +2.41 originally and +1.987 at
six threads. Under the pin it reads whatever it reads, and it is then put
through C2. Its previous skeptic pass (`tools/p6_scale_followup.py`) returned
SURVIVES = NO on 3 of 6 `min_train` settings; the question is whether that
holds at the pinned thread count.

## Predictions

| # | prediction |
|---|---|
| **PC1** | The degeneracy stays gone: 0 of 420 constant cells at every horizon. |
| **PC2** | The IC profile is flat again — four values inside roughly 0.005 to 0.020, with no monotone trend and no isolated peak. |
| **PC3** | The individual figures do NOT reproduce to three decimals, because the thread count changed. |
| **PC4** | No horizon survives C2. In particular h=5 fails the majority-of-six sweep again. |
| **PC5** | The verdict is unchanged: no horizon effect. |

## What would falsify the closing argument

A monotone or single-peaked IC profile across the grid, with the peak's
horizon surviving the `min_train` sweep at a majority of settings. That would
mean the horizon IS an axis and P6's conclusion was wrong. It is not expected;
it is written down so that it can happen.
