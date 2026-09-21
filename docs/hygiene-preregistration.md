# 2026-09-21 — hygiene session: what is expected, before anything is run

**This session is not expected to produce signal. It is expected to change
stored numbers.** Written and hashed before any re-run on real data, for the
same reason every other pre-registration in this project exists: so that what
comes out cannot be narrated afterwards as what was wanted.

## What is being changed, and why it is a defect fix rather than a new idea

P6 (`docs/p6-findings.md`) established two things about the pipeline, neither
of them about the market:

1. **`gamma` is denominated in the loss.** The label's dispersion falls from
   0.1196 at h=30 to 0.0476 at h=5 while `tune_pooled` searches `gamma` over a
   fixed `[0, 5]`, so at short horizons every split looks unprofitable and the
   model emits one constant per fold — five distinct predicted values across
   162,535 rows at h=5, and 340 of 420 constant cells.
2. **Nothing pinned the thread count.** The same fit moved from t +2.405 to
   t +1.987 on `OMP_NUM_THREADS` alone, with zero of 162,535 predictions
   matching, because Optuna selects a different winner when the parallel
   reduction order changes.

Both are measurement defects. Fixing them changes numbers that were already
not measuring what they claimed to.

## The predictions

| # | prediction |
|---|---|
| **H1** | No arm in this session clears the project's standing bar (paired cross-sectional t ≥ +2.0) in a form that survives a `min_train` sweep. |
| **H2** | The standardised label removes the degeneracy at every horizon — 0 of 420 constant cells — as Part C already measured post hoc. |
| **H3** | The thread pin changes the 30-session baseline's numbers, because it changes the thread count this machine was using. The A0 drift against the committed predictions will therefore be NON-ZERO, and that is the expected outcome, not a failure. |
| **H4** | The nulls survive both changes. A null that wobbles into another null is still a null. |
| **H5** | Conformal coverage after the standardise-and-invert round trip stays within 5 percentage points of the 80% nominal. If it does not, the round trip is not shippable and the session stops and reports rather than proceeding. |
| **H6** | The tie-guard fix leaves every rank-IC verdict in this project unchanged, because `rank_ic` averages ranks over ties and was never contaminated. Only the quantile and book columns move. |

## The standing rule for this session

**Any apparent new signal is to be investigated as a defect before it is
reported as a finding.** This project has twice promoted a number that a sweep
later retired, and both times the number arrived while something else was being
changed. A positive result from a hygiene session is a red flag about the
hygiene, not a discovery — the specific thing to suspect is that standardising
the target has leaked a cross-sectional statistic of the FUTURE window into the
training label.

That risk is real and is why the inverse has two forms
(`pipeline/label.py`): the realised moments are exact and legitimate for
scoring history, and are **never** available when forecasting. A live forecast
inverted with realised moments would read as a large improvement in both MAE
and coverage, and would be F1 in a new place.

## Non-goals, restated

- The universe does not change. That is the next session, and entangling it
  with a label change would make its result unattributable between the two.
- No features, data sources or model architectures are added.
- The daily-pipeline connection retry is NOT added. The 2026-09-14 failure was
  a provider-side Supabase pooler `EAUTHQUERY` timeout; the missing retry is
  real technical debt and is recorded, not fixed.
- The live gate is not touched and nothing is deployed without review.
