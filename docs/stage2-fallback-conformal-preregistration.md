# Stage 2, continued — the thin-sector fallback and the conformal fix: pre-registration

Written 2026-09-24, BEFORE any Part B panel is built or any Part C coverage is
computed, and hashed; the sha256 of this file is recorded in every output the
measurement tool writes and in the findings. Two parts, with separate
hypotheses and separate deciding rules, because either can pass or fail on its
own.

Both models are null and are expected to stay null. This session is a
correctness upgrade, not an accuracy claim.

## 0. Fixed inputs

- **Snapshot:** `76fe43a52bdcf2950c583763c8bf148752d67fdcdba5e70c7aeaa634b6ec3e6d`
  (cutoff 2026-09-23). NOT rebuilt: Yahoo re-adjusts prices between rebuilds,
  so a new snapshot would make every delta unattributable. The panel is built
  from this snapshot through the production signals code
  (`tools/stage2_snapshot.py panel --arm new`), the float32 round trip applied
  as before.
- **Platform:** WSL Ubuntu, CPython 3.12.0, the locked environment,
  `XGB_THREADS = 2`. The CI runner reproduced the WSL platform digest
  `ab65d091…136a77` bit for bit on the push of `6c52836` (run 35953248538), so
  WSL numbers stand for the runner.
- **Model:** pooled × MAE, no ticker feature, within-date standardised label,
  causal-moment inverse, `min_train = 500`, 10 nested trials, 5 folds — exactly
  the Stage 2 reference. No subsample, colsample, lock or search change.
- **Comparators already on disk (pass 2 of the last session):**
  `stage2/ref_a.npz` (v4, the 11 thin names NULL), `stage2/old_a.npz` (the
  Yahoo indices), `stage2/diag/fallback.npz` and `stage2/diag/placebo_{0,1,2}.npz`.
- **Statistic for Part B:** per-date cross-sectional Spearman rank IC of the
  inverted prediction against the raw 30-session log return, ≥ 20 names a
  date, paired by date, Driscoll-Kraay at 30 lags (`tools/stage2_measure.py`).

## 1. What is NOT blind here, stated first

Part B's arm was already measured, post hoc, in the last session:
`tools/stage2_diagnose.py`'s `fallback` panel gave the 11 thin names
market-relative momentum (+0.0125, t +0.85; Δ vs old +0.0018, t +0.62), and
three placebo panels put NULLs on 11 random names (Δ vs fallback −0.0024,
+0.0025, −0.0029). **So Part B's expected numbers are known.** What is not
known is whether the PRODUCTION code, rebuilt through the signals path,
produces that arm — the diagnostic built it by patching a panel after the
fact, without the void-window mask and with the flag column rewritten by hand.
Part B is therefore a confirmation that the production change implements the
arm that was diagnosed, plus a re-run of the fingerprint comparison on
production-built panels. It is not a discovery test, and its numbers must not
be read as one.

Part C is blind: no spread-normalised coverage has been computed on any panel.

## 2. Part B — market-relative momentum for the thin-sector names

### 2.1 The change

- For a name whose sector has fewer than `MIN_SECTOR_PEERS + 1 = 3` labelled
  constituents (or no usable label — "Unknown" stays missing, never a sector),
  `sector_rel_{5,10,20}d` are computed against the SAME benchmark its excess
  label already falls back to: the leave-one-out equal-weighted mean daily log
  return of every other universe name, from `regime.market_log_returns`,
  summed over the peers directly (never total minus own), cumulated into a
  level, with the same void-window rule. Stored `benchmark_ticker =
  'EW-LOO:MARKET'`, `benchmark_sector_specific = 0`.
- The feature and the label read ONE `Benchmark` object, so they cannot
  disagree. Any place they differ is reported and aligned.
- `sector_rel_missing` stops meaning "missing". Its new meaning, "the
  reference is the market", is exactly `1 − benchmark_sector_specific`, which
  is already stored from the same object. The flag is RETIRED rather than
  renamed (one fact, one column; two columns for one fact can drift). It was
  never a model input; a test asserts that no such flag, nor
  `benchmark_sector_specific`, enters any feature matrix.
- `MODEL_VERSION` → `absolute-return-sector-loo-market-v5`;
  `POOLED_MODEL_VERSION` → `pooled-std-mae-noticker-v2`.

### 2.2 Hypotheses and deciding rules

- **B0, reproduction (a precondition for reading the rest).** The production-
  built panel equals `stage2/diag/panel_fallback.parquet` on every FACTORS
  column and both labels for every (date, ticker) row, to float equality. If
  it does, the Part B pooled predictions must be bit-identical to
  `stage2/diag/fallback.npz`. If it does not, the differing rows are
  reported and the new numbers stand on their own. Two independent runs of
  the Part B model must reproduce each other exactly (drift 0.0).
- **B1, the fingerprint is structurally gone.** Beyond the warm-up rows every
  ticker shares, the share of rows with any `sector_rel_*` NULL is the same
  for the 11 thin names as for the other 73, to within 0.5 percentage points.
- **B2, the fingerprint comparison, repeated.** From the Part B panel, four
  arms are rebuilt through `tools/stage2_diagnose.py` (NULLs placed on
  `sector_rel_*` after the fact): the REAL 11 (which reconstructs v4) and 11
  RANDOM names from the 73, three draws, seed 20260924 as before. Prediction:
  each random-11 arm's Δ vs the Part B model lies within ±0.005; the real-11
  arm's Δ is positive and larger than every random draw (last session:
  +0.0109). The real-11 arm reproducing v4's ref_a predictions exactly is a
  consistency check, not a rule.
- **B3, Part 0's original bound, now expected to hold.** Δ(Part B − old
  Yahoo arm) satisfies |Δ| ≤ 0.005 and |t| < 2 (expected ≈ +0.0018).
- **B4, the delta against the NULL version.** Δ(Part B − v4 ref_a) is
  reported; expected ≈ −0.011 (the fingerprint's worth, removed). Descriptive.
- **B5, still null.** The Part B pooled cs IC has |DK t| < 2; expected ≈
  +0.0125.

**Part B PASSES iff B1, B2 (random draws inside ±0.005) and B3 and B5 hold.**
On failure: report and stop. No other representation of the thin names is
tried in this session. The code change lands either way — it is the user's
design decision — and the findings record whether it removed the channel.

## 3. Part C — spread-normalised conformal intervals (pooled model only)

### 3.1 The failure being fixed

The last session's gate (docs/dashboard-switch-preregistration.md §7):
pooled overall 0.8684 against [0.75, 0.85], fold 4 0.9033 against [0.70,
0.90]. The mechanism: the cross-sectional dispersion of 30-session returns
falls across the folds (0.106 → 0.080 over the P5 folds), so a constant
half-width calibrated on the wilder early folds over-covers the calmer later
ones.

### 3.2 The method, fixed now

- **Spread estimate, past-only.** `s(t)` = the mean, over the 21 grid dates
  ending 30 dates before `t`, of the realised cross-sectional standard
  deviation of the 30-session log-return label (`label.causal_moments(…,
  lookback=21, min_dates=21, horizon=30)`). The label dated `t − 30` spans
  `[t − 30, t]` and is realised at `t`'s close; nothing later is read. 21 is
  one trading month: short enough to be the CURRENT spread, long enough not
  to be one date's noise. It is chosen here and not swept.
- **Why not the inverse's own `causal_sd`** (252-date average, same lag):
  it trails the fold-level drift by roughly half a year, which is the very
  drift that failed the gate. The point inverse keeps using it unchanged —
  changing the prediction would be a second change, moving MAE and IC.
- **Nonconformity score:** `e = (y − ŷ) / s(t)`, `y` the realised 30-session
  log return, `ŷ` the causally inverted prediction.
- **Calibration:** split-conformal on |e| with the finite-sample rank
  `ceil((n + 1)(1 − α))`, α = 0.20 — the existing `fit_conformal` rule
  applied to the normalised scores.
- **Interval:** `ŷ ± q · s(t)` in log-return space; in price space
  `[P · exp(ŷ − q s), P · exp(ŷ + q s)]`, `P` the close on `t`. Coverage is
  measured in price space; the map is monotone, so it is the same event as
  log-return coverage, and a test holds the two equal.
- **Measurement:** expanding by fold, exactly as the failed gate — for fold
  k, calibrate on every row of folds < k, check on fold k; overall = the
  pooled share over every checked row (folds 1–4). Rows whose `s(t)` is
  undefined are excluded from BOTH methods and counted.
- **Production:** the weekly shadow step calibrates `q` on every
  out-of-sample row; the daily step prices each forecast with `q · s(as_of)`,
  and `s(as_of)` is stored on the shadow forecast row. The shadow path uses
  this method whether the gate passes or fails — choosing the code path on
  the measured outcome is what the rule forbids; `coverage_status` records
  the outcome and a FAIL keeps the cutover blocked.

### 3.3 The deciding rule — the SAME bands as the failed gate, not widened

**PASS iff overall coverage ∈ [0.75, 0.85] AND every checked fold ∈ [0.70,
0.90].** Loosening the bands after failing them would be the post-hoc tuning
this project refuses. On FAIL: report and stop; no second method in this
session. The next remedy is adaptive conformal inference (Gibbs & Candès
2021), pre-registered as its own step.

### 3.4 Predictions (directional, not deciding)

- Overall coverage moves from 0.868 toward 0.80, and the largest fold
  deviation from 0.80 shrinks from 0.103.
- Mean interval width tracks dispersion: across folds 1–4 the new method's
  mean width is ordered with the fold's realised cross-sectional dispersion
  (Spearman > 0), where the constant method's width is fixed within a fold.
  The expected picture is narrower bands in calm folds and wider in volatile
  ones, not uniformly narrower bands.
- Reported, per fold and overall, before (constant q, same rows) and after:
  coverage, mean width in log-return units, mean width as % of price.

### 3.5 Remedies considered and not chosen

- **Rolling calibration window** (calibrate only on the most recent N
  dates). It discards most of the pool: each fold holds ~13 non-overlapping
  30-session windows and one market shock moves every name at once, so a
  short window's 80th percentile is a noisy estimate. It also corrects a
  regime only after a full window of mis-covered outcomes has realised, and
  it adds a window length to choose.
- **Adaptive conformal inference** (Gibbs & Candès 2021): adjusts α online
  from realised misses. Its guarantee is long-run average coverage, not
  per-fold coverage, which is what the gate checks; and here a miss is known
  only 30 sessions later, with overlapping windows, so the feedback is
  delayed and heavily autocorrelated. Kept as the next step if this fails.
- **Per-name volatility scaling** (each stock's own trailing volatility).
  The measured failure is temporal drift of the whole cross-section, not
  cross-name heterogeneity; per-name scaling reallocates coverage between
  names, which is a separate question with its own test.

### 3.6 Scope

The pooled model only. The live per-ticker model is replaced at cutover, so
its intervals are not fixed. **Recorded for the cutover session:** the site
currently publishes per-ticker intervals labelled 80% whose measured per-fold
coverage runs from 0.721 to 0.912 (overall 0.859) on this snapshot; the
cutover copy must say that the band changes, and why.

## 4. Testing, before any measurement is read

The flag absent from every feature matrix; the leave-one-out market mean
never containing the stock; the fallback feature and the excess label
reading the same reference; the past-only spread (a leakage test that fails
when handed the current date's spread); coverage against a hand-checked case;
the fingerprint arms reproducing; the full suite; mutation checks on the new
guards.
