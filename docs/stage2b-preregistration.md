# Stage 2b pre-registration — the pooled cross-sectional model

**Written 2026-09-07, before any pooled training run.**

Branch `stage2b-pooled-model`, off `stage2a-tuner-objective-pilot`. The first
slice of the full Stage 2 (pooled model consolidation), triggered by Stage 2a's
finding that a per-ticker model choosing nine hyperparameters off ~10 effective
observations per fold is over-specified whatever metric scores it.

Separate numbering track from the project's Phase 0-6 roadmap. Nothing here
touches `stage0-evidence-grading`, and nothing is merged into `main`.

---

## The hypothesis under test

> **H2.** The Stage 2a overfitting was an ESTIMATION-NOISE problem, not an
> objective problem. Pooling training across all 84 tickers raises the effective
> sample size behind each hyperparameter decision by orders of magnitude, so a
> rank-IC objective should work at pooled scale without the MAE regression and
> train/OOS gap blowup the per-ticker pilot showed.

## The decision rule, fixed in advance

> This step succeeds as a test of the sample-size hypothesis if pooled×rank_ic
> shows materially reduced degeneracy relative to per-ticker×rank_ic's
> overfitting signature — specifically, without the OOS MAE regression and
> train/OOS gap blowup seen in the Stage 2a pilot — because pooling supplies the
> effective sample size that per-ticker rank-IC scoring lacked. If pooled×rank_ic
> still shows that overfitting signature at pooled scale, that falsifies
> 'insufficient sample size' as the root cause and points to something else —
> the objective itself, not merely its estimation noise. Separately, pooled×mae
> vs. per-ticker×mae tests whether pooling alone (no objective change) already
> reduces degeneracy, which would indicate the noisy-estimation problem was
> present even under MAE scoring, just less visible.

**"Materially" is fixed before the run** so it cannot be reinterpreted
afterwards. Stage 2a measured, on its 14-ticker pilot, an OOS MAE regression of
**+6.7%** and a train/OOS gap widening from **+0.0145 to +0.0362 (2.5×)**.
pooled×rank_ic avoids that signature if, against pooled×mae on identical rows:

- OOS MAE is no more than **+2%** worse, and
- the train/OOS gap grows by no more than **1.5×**.

Both must hold. Either one failing means the signature survived pooling.

## What is measured, for every cell of the 2×2

- **Degeneracy** — `mode_share` and `unique_fraction` per (ticker, fold) cell,
  the same two metrics used in the Stage 0 addendum and Stage 2a.
- **Held-out MAE**, on the model's own out-of-sample rows.
- **Cross-sectional rank IC** — per-date Spearman across tickers, averaged.
  This is the primary quantity: it is what `reb_ic`, the break-even IC and the
  whole portfolio apparatus already use, and it is what the product claims to do
  (rank names against each other), which the per-ticker time-series IC is not.
- **Time-series within-ticker IC**, as a secondary diagnostic, so the numbers
  stay comparable with Stage 2a's.
- **Train/OOS gap** — training MAE beside held-out MAE. A rank IC bought with
  under-regularised splits shows up here and nowhere else.

## The objective for the pooled search

**Cross-sectional per-date rank IC, averaged over the dates in a fold, then
averaged over folds.** Never pooled across concatenated dates or folds — the
pooled correlation reads fold identity and month identity as skill, which is the
artifact the Stage 0 addendum diagnosed and `_mean_daily_rank_ic` already exists
to avoid. A date with no ordering scores the degenerate penalty rather than
being skipped, so the search must steer away from constants rather than merely
decline to reward them.

## Predictions on record

Recorded so being wrong is visible. Stage 2a got one of its three wrong (MAE
improved at gamma 0 in four of ten cells) and it stayed on the record.

1. **Pooling alone (pooled×mae) removes most of the degeneracy**, without any
   objective change — because MAE estimated on ~10⁵ pooled rows resolves a split
   that MAE on ~400 rows cannot.
2. **pooled×rank_ic avoids the Stage 2a overfitting signature** and passes the
   thresholds above.
3. **Neither pooled cell clears the break-even rank IC of 0.0051** by a margin
   that survives a `min_train` sweep. Removing an artifact is not the same as
   finding signal, and five phases of null are not overturned by a tuner change.

Prediction 3 is the one that matters for what this is worth: the expected
outcome is a better-behaved instrument, not a result.

## The hazard this run has to control for

Adding **ticker identity** as a feature to a pooled model is exactly the
construct CLAUDE.md §7 records as earning a positive t-statistic from nothing:
two random per-ticker constants scored `pooled_xgb` at a mean rebalance t of
**+0.77** over 24 draws, because the tree identifies the name and learns which
names paid in the training window, and that persists into the test folds.

So every pooled cell is run **with and without** the ticker feature, and a
partial placebo — ticker labels permuted **within each date**, preserving
cardinality and frequency while breaking the identity↔return link — is run
alongside. The placebo is partial and is reported as such: it destroys the
label's persistence across dates, which the real feature has, so it bounds the
null from below rather than reproducing it.

**No relabeling can placebo-test pure identity**, because any relabeling of 84
tickers onto 84 categories is a bijection and carries identical information.
That is a limitation of the construct, not of this run, and it is why the
with/without comparison carries the weight.

## Non-goals

- No new data sources. No triple-barrier labels, sample-uniqueness weighting, or
  new horizons. No ridge/random-features alternative. XGBoost only.
- The existing per-ticker path stays intact, unchanged and selectable; the
  per-ticker cells of the 2×2 are REUSED from the production baseline and the
  Stage 2a pilot, not recomputed.
- Nothing is written to `model_metadata`, the tuned-parameter cache,
  `forecast_confidence`, the API or the frontend.
- **A falling degeneracy rate is not success on its own**, and neither is a
  rising rank IC that does not clear the break-even and survive a sweep.
