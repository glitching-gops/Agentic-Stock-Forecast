# Stage 2 — the dashboard switch — pre-registration

Written 2026-09-24, **before any measurement on the reference platform**. Its
sha256 is printed by `tools/stage2_measure.py` at the start of every run and
recorded in every output file; the tool refuses to run without this file.
Nothing below may be edited after the first reference run. Anything learned
afterwards goes in `docs/dashboard-switch-findings.md`, dated.

Builds on `main` at `bd1b028` plus this session's uncommitted changes (Part 0,
Part 1, the pooled production path), all of which are in place before the
first reference run.

## 1. Purpose — correctness, not accuracy

The public dashboard serves the per-ticker model, which emits ONE constant
prediction in 316 of 420 (ticker, fold) cells (Stage 0 addendum). On most dates
it holds no ordering of the 84 names at all. The pooled, standardised-label
model predicts a distinct value for every name on every date.

**The switch is a correctness upgrade, not an accuracy claim.** Every panel
test on this universe is null, and **the expected outcome is that the pooled
model stays null** here too. The reason to switch is that the live model mostly
does not rank at all, not that the pooled one ranks well. Any copy drafted for
the site must say so.

## 2. The frozen snapshot

Every number in this session is built from one snapshot, read once, read-only,
by `tools/stage2_snapshot.py build`:

| | |
|---|---|
| content hash | `76fe43a52bdcf2950c583763c8bf148752d67fdcdba5e70c7aeaa634b6ec3e6d` |
| cutoff | 2026-09-23 (the last session in `ohlcv`) |
| ohlcv | 209,609 rows, the frozen 84, sha256 `2a30b7f1…c5f9c1c9` |
| membership | 100 rows, sha256 `0b88a01b…944c5d0c` |
| macro | 2,616 rows, sha256 `7102e4d7…1df9c3` |
| signals (stored) | 205,384 rows, sha256 `80cd9269…6f813c1` |
| earnings (vendor tables) | 2,074 rows, all 84 tickers, sha256 `6448ea7f…b9fe19` |

**Vendor defect tags carried into every table built from it** (CLAUDE.md
rule 8):

1. **2025-03-18**: NSE traded; Yahoo serves a flat zero-volume bar for 83 of 84
   names. Every return into or out of that date is fiction. Kept as a session.
2. **49 tickers' stored Yahoo-index levels end 2026-09-15** (their writes were
   refused by the sector outage). Only the OLD arm of Part 0 reads them, and it
   forward-fills them exactly as the pre-v4 code did.

`earnings_surprise` was repaired for all 84 tickers by the 2026-09-23 daily
run before the snapshot was taken, so it carries no tag.

## 3. Platform

- **Reference platform for this session: Linux** — WSL Ubuntu (glibc 2.43),
  CPython 3.12.0, the locked environment (`tools/check_locked_env.py`, 108
  pins), `XGB_THREADS = 2`.
- **Production is a GitHub Actions `ubuntu-latest` runner.** WSL and the runner
  may ship different C++ runtimes. The hypothesis is that they agree, because
  XGBoost's random distributions are template code compiled INTO the manylinux
  wheel, not taken from the runtime `libstdc++`. That is tested, not assumed:
  `tools/platform_digest.py` fits the pooled path on a fixed synthetic panel
  with subsampling below 1.0, and prints a digest. The WSL digest is recorded
  in the findings; `.github/workflows/platform-digest.yml` produces the
  runner's on the first push.
- **If the digests differ, the CI runner is the reference**, because that is
  where production trains, and the backtest criteria below must be
  re-established there before any cutover. Shadowing does not wait on it.

## 4. Part 0 — the sector benchmark, and its expected delta

Old arm: sector momentum and the excess label against Yahoo's NSE sector
indices, as stored (dead ones forward-filled). New arm: the leave-one-out
equal-weighted sector-peer mean (`pipeline/sector_benchmark.py`), with
`MIN_SECTOR_PEERS = 2` and the 11 thin-sector names' `sector_rel_*` NULL.
Both panels built by `tools/stage2_snapshot.py panel` from the snapshot above,
through the production signals code; both carry the Part 1 changes. The target
the model trains on (`target_return`) is identical in both, so the difference
is the three `sector_rel_*` features and nothing else.

**Prediction: the change in the pooled baseline's per-date cross-sectional
rank IC (new − old, paired by date, Driscoll-Kraay SE at 30 lags) is small:
|Δ| ≤ 0.005 and |t| < 2.0.** It is a plumbing repair. If |t| ≥ 2.0, that is a
finding to investigate as a defect first — never a win to claim.

## 5. The reference baseline and its reproduction

pooled × MAE, no ticker feature, within-date standardised label, 15 FACTORS
cross-sectionally z-scored with NULLs kept as missing, h = 30, legacy purge,
5 folds, `min_train` 500 dates, 10 nested trials — `pipeline/pooled.py`, which
a test holds bit-identical to the research harness `run_arm`.

**It must reproduce itself exactly twice on the reference platform: drift
exactly 0.0 on predictions and labels, every row matched.** If it does not,
nothing else in §6-§7 is read.

## 6. Pooled vs per-ticker, on the same snapshot and platform

The per-ticker model is re-measured with the production `evaluate_ticker`,
unmodified, on the NEW-arm panel (what it would train on after Part 0), on the
same platform. Compared on the (date, ticker) rows BOTH produce out of sample:

| measure | definition |
|---|---|
| degeneracy | constant (ticker, fold) cells; median distinct predictions per date |
| cross-sectional IC | per-date Spearman of prediction vs raw `target_return`, ≥ 20 names, mean with Driscoll-Kraay SE (30 lags) |
| MAE, price space | mean abs error of the predicted 30-session LOG return (= log of the price ratio) vs realised. Pooled: z-score inverted with the CAUSAL moments, never the realised ones |
| conformal coverage | §7, both models, pooled-residual expanding calibration |
| grades | `grade_panel_v3`, B = 1000, block 30, seed 20260908, alpha 0.10, for both; plus the live gate's own grade for the per-ticker model |
| Romano-Wolf | rejections at alpha 0.10, from the same bootstrap |

**Predictions.**

1. Pooled: 0 constant cells; median distinct predictions per date = the names
   present. Per-ticker: at least half its cells constant.
2. Pooled cross-sectional IC: |DK t| < 2.0 (null). Per-ticker: |DK t| < 2.0.
3. MAE: the pooled model is within +5% of the per-ticker model's MAE on the
   common rows. The per-ticker model mostly predicts its training mean, which
   is a strong MAE baseline, and the causal inverse adds its own error
   (Hygiene measured +2.2%).
4. Grades: pooled 0 STRONG. tau2 likely at the zero boundary, in which case the
   shadow path writes ONE panel statement, not 84 grades. Per-ticker, live
   gate: 0 STRONG (no ticker has `eval_rw_significant`).

## 7. The conformal gate

Coverage of the nominal 80% interval, **in price space** (the interval is
monotone in the log return, so covering the realised price and covering the
realised log return are the same event), after the causal inverse, measured by
EXPANDING calibration: for each fold k, calibrate on every fold before k and
check on k (`conformal.expanding_fold_coverage`). Constants in
`pipeline/pooled_shadow.py`.

| band | tolerance | justification |
|---|---|---|
| overall (pooled over every checked row) | **[0.75, 0.85]** | the ±5pp rule `conformal.check_coverage` already calls `well_calibrated`; inside it "80%" is an honest description |
| per checked fold | **[0.70, 0.90]** | a fold holds ~13 non-overlapping 30-session windows and a market shock moves every name's residual together, so a fold is noisy; outside ±10pp a reader relying on "80%" is materially misled for that whole period |

**PASS requires both: overall in band AND every checked fold in band.**

**Expected risk, stated before measuring:** the Hygiene session measured
0.8166 at the first checkable fold and 0.87-0.91 later, because the panel's
target dispersion falls across folds. A per-fold FAIL in the late folds is a
real possibility.

**If it fails:** report and stop. The calibration is NOT tuned after seeing
the result — not in this session. The failure BLOCKS THE CUTOVER. The shadow
path still ships, with the pre-registered calibration unchanged and
`coverage_status = 'FAIL'` on every shadow model and forecast row, so the
defect is visible at the point of reading. The remedies, to be pre-registered
as a separate step:

- recency-weighted or rolling-window calibration;
- **volatility-normalised nonconformity scores** — residual ÷ the causal sd, the
  interval rescaled by the current causal sd — recommended, because the
  failure mode is dispersion drift and this scales the band with it;
- adaptive conformal inference (Gibbs & Candès, 2021).

## 8. Cutover criteria

### Backtest — from the Linux walk-forward on this snapshot

- **B1** Reproduction: §5 exact, twice.
- **B2** Degeneracy: pooled 0 constant cells.
- **B3** Correctness, not accuracy: pooled cross-sectional IC not significantly
  NEGATIVE (DK t > −2.0). No positive claim is made either way.
- **B4** MAE within +5% of the per-ticker model on the common rows.
- **B5** Conformal gate PASS (§7).
- **B6** Grades produced through the Romano-Wolf-wired path, and, if tau2 is at
  the boundary, published as one panel statement.
- **B7** Platform: the runner's digest equals the WSL digest, or the backtest
  is re-established on the runner.

### Operational — from the shadow cycles in production

- **O1** Two consecutive weekly runs complete the shadow step: a model row, an
  evaluation per ticker (or one panel statement), a coverage status, all with
  `pooled_version`, config, data and environment hashes; the environment is
  Linux.
- **O2** Every daily run in between writes a shadow forecast for every name
  whose signals were written, finite, with an interval and a grade or statement.
- **O3** The weekly job's total runtime stays under 80% of its 300-minute
  timeout, and the shadow step's memory fits the runner.
- **O4** Public API responses unchanged by the shadow tables (tested in the
  suite; spot-checked live).
- **O5** The user's explicit go-ahead.

### What two shadow cycles can and cannot show

The horizon is 30 sessions. A forecast written in shadow week one does not
resolve for about six weeks. **Two cycles verify that the shadow path runs,
writes, grades, calibrates and reproduces in production. They cannot measure
live predictive performance or live coverage** — that evidence is the Linux
walk-forward backtest (B1-B6). The two kinds of evidence are never to be
conflated: no shadow-cycle figure will be reported as a performance or
coverage result.

## 9. Non-goals

No cutover, no change to what the public site shows beyond what Part 0 itself
changes (the benchmark name, and `sector_rel_*` NULL for 11 names); no change
to `subsample`/`colsample_bytree`; no lock move; no fix to `macro.py`'s
holiday-inclusive Nifty returns, 2025-03-18, the unlocked torch/transformers
extras, or the daily connection retry; no `experiment_runs` cleanup SQL; no
universe work, learning-to-rank or triple-barrier labels; no calibration tuning
after measuring.
