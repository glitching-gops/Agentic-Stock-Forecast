# Stage 1, Pilot 1 — multi-lookback residual reversal: pre-registration

**Written 2026-09-13, before any reversal feature was computed on real data and
before any arm was trained or graded.** Branch `stage1-residual-reversal`, off
`main` at `61ad22c`. One attributable change: four residualised and four raw
lookback-return features, added to the pooled × MAE, no-ticker model and graded
through the Stage 0c harness. No new data source, no new dependency, no model,
label or horizon change, no ticker-identity feature.

A SEPARATE track from the project's Phase 0-6 roadmap and from the Stage 0-0c
evidence-grading track. It renumbers nothing.

## The success statement, verbatim

> "This pilot succeeds if it definitively determines whether multi-lookback
> residual reversal features, added to the pooled × mae (no-ticker)
> architecture and graded through the Stage 0c evidence-grading harness, move
> any tickers out of INSUFFICIENT — and, if so, whether a permutation placebo
> (mirroring Stage 0c's 9-draw calibration) confirms the movement survives the
> same noise-calibration check that validated Romano-Wolf in Stage 0c. It also
> tests, via the raw-reversal comparison arm, whether residualizing against the
> market specifically matters on this panel, as the literature suggests, or
> whether raw reversal performs comparably. It does not succeed merely by
> producing more STRONG or WEAK grades — a result that fails the placebo check,
> or that shows no difference between the residual and raw arms, is exactly as
> informative as one that clears every bar."

## The baseline this is anchored to

**The panel harness's own grade of the baseline arm, not the live gate's.** The
three arms are graded by `pipeline.evidence_panel.grade_panel_v3`, which
already applies the three-fold minimum, the cross-sectional demeaning and
Romano-Wolf. Its grade of arm (a), from `stage0c_report.md`:

| arm (a) `pooled_mae_noticker` | value |
|---|---|
| n_usable | 84/84 |
| mu_hat (boot SE) | −0.01727 (0.01390), z −1.24 |
| tau2 REML | 0.00435 |
| grades | **1 STRONG / 1 WEAK / 82 INSUFFICIENT** |
| Romano-Wolf / BH / BY rejections | 1 / 3 / 1 |
| per-date cross-sectional IC (DK SE) | −0.00101 (0.00811) |
| Stage 2b: reb IC, reb t, OOS MAE | −0.0015, −0.09, 0.08882 |

The live gate's 0 / 1 / 83 is a different quantity from a different code path
and is NOT this pilot's yardstick.

**Frozen inputs.**

| input | identity |
|---|---|
| `panel_cache.parquet` | sha256 `990C0A07D5391E9B6B09942626583B74C5B15321D0BD51391C5A515F810E0ECB`; 204,900 rows, 84 tickers, 2,440 dates, 2016-10-28 → 2026-09-07 |
| `stage2b_pooled_oos.npz` | sha256 `A23B124D2DEB03B88B1D6DA91165FCC4C9F9965A488DF774268F429F78AEF507` |
| arm (a) arrays in it | sha256 `67cc72bee795ea9b7a77ad7c5e49de070b08ff774ade43f09a0613a6e04012e1` over (date, ticker, y_true, y_pred, fold); 160,435 rows, 1,910 dates, 5 folds |

## The features

For k ∈ {1, 5, 10, 20} sessions and a skip of ONE session:

    r_k(i, t)   = log close(i, t−1) − log close(i, t−1−k)
    m_k(t)      = the same window's sum of the equal-weighted universe daily
                  log return
    beta(i, t−1) = `pipeline.regime.rolling_beta` (trailing 252 sessions,
                  min 60), LAGGED ONE SESSION
    rev_raw_k   = −r_k
    rev_resid_k = −(r_k − beta(i, t−1) · m_k)

So every feature at date t is a function of closes up to t−1 only. The beta is
lagged too, so not even its 1/252 weight on session t reaches the feature.

**One definition of "the market".** `rolling_beta` and `compute_market_state`
each compute the equal-weighted universe return inline today. Both are moved
onto a single `regime.market_log_returns` helper, which the residual also uses,
so three definitions cannot drift apart. That refactor must leave
`rolling_beta` numerically identical, and a test pins it.
`pipeline/neutralise.py` is the project's other residualisation. It is NOT
reused: it regresses ACROSS names within a date, which is the right operator
for neutralising a target and the wrong one for a time-series residual return.

**Preprocessing identical to every other pooled feature.** The eight columns are
cross-sectionally z-scored within each date by the same
`cross_sectional_zscore`, which clips at ±3, zeroes dates with fewer than 10
names and fills missing values with 0. The sign is irrelevant to the tree; the
negation keeps the feature-level IC reading as "reversal".

## The arms

All three use `tools/stage2b_pooled.run_arm` on the cached panel: MAE
objective, ticker mode `none`, `EVAL_TUNE_TRIALS` = 10, 5 purged folds,
`min_train` = 500 dates, horizon, purge and embargo 30. The seeded search is
unchanged.

| arm | features |
|---|---|
| (a) baseline | `FACTORS` (15) — REUSED from `stage2b_pooled_oos.npz`, not retrained |
| (b) residual reversal | `FACTORS` + `rev_resid_{1,5,10,20}` (19) |
| (c) raw reversal | `FACTORS` + `rev_raw_{1,5,10,20}` (19) |

**What the baseline already contains, stated before any number.** `FACTORS`
already carries `lag1_ret`, `lag5_ret` (price change over 1 and 5 sessions,
INCLUDING session t), `roc_10`, and `sector_rel_{5,10,20}d`, which are returns
relative to a sector index. So arm (c) adds little that is new, apart from the
skipped session and the 20-session raw window. And within-date z-scoring already
removes the common market LEVEL from any raw return. The market-only residual
can therefore remove only `(beta_i − mean beta) · m_k`, a far narrower
adjustment than Da, Liu & Schaumburg's three-factor residual.

## Stop condition, checked before (b) and (c) are read

**S1. Arm (a) must reproduce.** `run_arm` is re-run for arm (a) on the frozen
panel. It must return the stored arm-(a) predictions: the same (date, ticker,
fold) rows and max |Δ y_pred| ≤ 1e-9. If it does not, the code or the inputs
have moved underneath the stored baseline, and (b) and (c) would not be
comparable with it. The pilot then STOPS and reports rather than grading.

## Decision rules

Every arm is graded by `grade_panel_v3` at B = 1000, block 30, bootstrap seed
20260908, α = 0.10 and break-even 0.00512 — Stage 0c's settings exactly.

**R1 — movement.** Arm (b) or (c) "moves tickers out of INSUFFICIENT" iff the
set of tickers graded STRONG or WEAK in that arm, and INSUFFICIENT in arm (a),
is non-empty. The set is reported, not only its size.

**R2 — the placebo.** The residual arm's predictions are permuted WITHIN EACH
DATE nine times, with seeds 20260913 to 20260921, and each draw is graded
identically. The movement SURVIVES only if arm (b) beats the maximum over the
nine draws on all three of:
- its STRONG count;
- its Romano-Wolf rejection count;
- its tau2 REML.

For reference, Stage 0c's nine draws gave at most 1 STRONG and tau2 ≤ 0.00017.

**DK lags for every paired test below: 30, the label horizon.** An amendment,
made 2026-09-13 after the first draft (sha256 `00E77145…A99D1F`, written
18:54:39 +05:30) and before any data was run. `driscoll_kraay_se`'s default
Newey-West rule chooses about 7 lags over ~1,910 dates, but consecutive dates
share 29 of their 30 forward sessions, so 7 lags would understate every SE here.
The default-lag figure is reported beside it as a sensitivity check and decides
nothing.

**R3 — the tradeable quantity.** Paired per-date cross-sectional rank IC,
(b) − (a), with the Driscoll-Kraay SE from `evidence_panel.driscoll_kraay_se`.
**A result is reported as SIGNAL only if R1, R2 and a paired DK t ≥ +2.0 all
hold.** Anything less is reported as not-signal, naming the clause that failed.
A STRONG count that the cross-sectional measurement does not support is the
ticker-feature pattern from Stage 0c, and it is not signal.

**R4 — does residualising matter?** The paired per-date cross-sectional IC,
(b) − (c), with its DK t:
- t ≥ +2: residual better.
- t ≤ −2: raw better.
- |t| < 2: COMPARABLE.

The grade counts of (b) and (c) are reported beside it, descriptively.

**R5 — against Da-Liu-Schaumburg, descriptive.** For each k, the per-date
cross-sectional rank IC of `rev_resid_k` and `rev_raw_k` against the target over
arm (a)'s out-of-sample dates, with DK SEs, and the ratio of their magnitudes.
DLS report 1.34%/month (t 9.28) for residual reversal against 0.33% (t 1.37)
for raw, about 4× on alpha and 6.8× on t. Their sample is US and three-factor, and
the comparison here is market-only on a z-scored panel. It is reported, not
tested.

**R6 — conditional sweep.** If R1 and R2 both hold for any arm, that arm and
arm (a) are re-run at `min_train` ∈ {380, 420, 460, 500, 540, 580}, reporting reb
IC, reb t and cross-sectional IC per setting. A lone spike is not a result (P5).
The grading is not repeated per setting.

Also reported for every arm, descriptively: Stage 2b's `cell_metrics` —
constant cells, reb IC and reb t over about 64 non-overlapping rebalances, OOS
MAE and the train/OOS gap — and the per-fold cross-sectional IC, because the
early-fold artifact has appeared six times.

## Predictions, scored in the findings whichever way they fall

- **P1.** Arm (a) reproduces at drift 0 (S1 passes).
- **P2.** Neither reversal arm improves the tradeable quantity: |Δ cs IC| for
  (b) − (a) < 0.005 with DK |t| < 2.
- **P3.** Residual and raw are COMPARABLE (R4, |t| < 2), with |Δ cs IC| for
  (b) − (c) < 0.003. The DLS gap does NOT replicate, for the z-scoring and
  single-factor reasons above.
- **P4.** Each reversal arm grades at most 1 STRONG, inside the placebo range,
  so R2 fails.
- **P5.** Every one of the eight features has |mean per-date IC| < 0.02. No sign
  is predicted. Chui, Ranganathan, Rohit & Veeraraghavan (2023) find momentum
  rather than reversal among LIQUID Indian names, which would make the reversal
  features' IC negative; that is noted, not assumed.

## Standing caveats carried in

- **Every SE here is likely too small.** Politis-White puts the panel's serial
  dependence at 35.8-62.5 sessions, beyond the 30-session label horizon and the
  30-session bootstrap block. So a t near 2 is weaker than it looks.
- **Trials accumulate.** This pilot adds three arms and nine placebo draws to
  the ~131 prior configurations on this panel.
- **`eval_rw_significant` is still unwired, deliberately and separately.** It
  needs a schema migration and a Render redeploy, and it changes no live grade
  today. The pilot is unaffected, because `grade_panel_v3` computes Romano-Wolf
  itself.

## Explicit non-goals

No NSE delivery %, SUE, bulk/block deals, analyst revisions or FII/DII. No new
external data. No ticker identity. No horizon or label change. No merge, no
Render redeploy, and no `git add` / `commit` / `push` by Claude.
