# Stage 2 — the dashboard switch — findings

> **FOLLOW-UP, 2026-09-24 (the same day): both blockers below are resolved,
> pre-registered, in `docs/stage2-fallback-conformal-findings.md`.** The 11
> thin-sector names now take market-relative momentum instead of NULL
> (MODEL_VERSION v5). The fingerprint is gone: Δ vs the Yahoo arm is +0.0009
> (t +0.31) and the pooled cs IC is +0.0116 (t +0.79). The spread-normalised
> interval passes the UNCHANGED conformal gate: 0.825 overall, folds 0.848 /
> 0.820 / 0.775 / 0.859. The CI runner reproduced the WSL digest. Nothing
> below is rewritten.

2026-09-24. Measured against `docs/dashboard-switch-preregistration.md`
(sha256 `07bb38a254a649c0231895d61d6693a3c10fb436b7885276568c4b116cbc016b`,
written and hashed before the first reference run), on the frozen snapshot
`76fe43a5…3e6d` (cutoff 2026-09-23), on the reference platform: WSL Ubuntu,
CPython 3.12.0, the locked environment, `XGB_THREADS = 2`. Every figure below
carries that provenance in its own output file under `stage2/` (gitignored).

**Headline.** Parts 0–2 are done and Part 3 is built, and the pooled model is
ready to SHADOW. It is not ready to be CUT OVER: the conformal gate fails,
exactly as the pre-registration warned it might, and so does the live
per-ticker model's. Two predictions failed and one of them — Part 0's delta —
turned out to be the missingness-fingerprint landmine rather than the
benchmark, which leaves one design decision for the user.

## 0. What was measured twice, and why

The reference runs were made **twice**. After the first pass, the Part 1 audit
of the rebuilt panel found a fourth silent defect (§2: `earnings_surprise`
carried forward for years once the vendor stopped recording a ticker). Part 1
is meant to precede the baseline, so it was fixed, both panels were rebuilt
from the same snapshot on the reference platform, and everything was re-run.
**Pass 2 is the result.** Pass 1 is kept in `stage2/pass1/` and quoted where it
changes the reading — it does, in §1.

## 1. Part 0 — the sector benchmark

`pipeline/sector_benchmark.py`: each stock is benchmarked against the
equal-weighted mean daily log return of the OTHER names in its sector within
the frozen universe (leave-one-out; summed over the peers directly, so the
stock is never in its own benchmark — a test moves its price and requires its
benchmark bit-identical). `regime.market_log_returns` is the one definition of
daily returns it reads.

**Thin sectors.** Measured on the frozen 84 (20-session log returns, nine
sectors of ≥ 5 names), the mean of m random peers shares a median of 0.58 /
0.74 / 0.85 / 0.92 of its variance with the full sector mean at m = 1 / 2 / 3 /
4 (lowest sector at m = 2: 0.65). One peer is a pairwise spread, so
`MIN_SECTOR_PEERS = 2`: a sector needs three names. **11 of 84 tickers lose the
signal** (7 sectors of 1–2 names): ADANIPORTS, ASIANPAINT, BHARTIARTL, DLF,
INDHOTEL, INDIGO, LT, PIDILITIND, SOLARINDS, TITAN, TRENT — 13.1% of rows.
Their three `sector_rel_*` features are NULL with `sector_rel_missing = 1`;
their EXCESS LABEL falls back to the leave-one-out universe mean, flagged
`benchmark_sector_specific = 0`, so the label stays complete (202,973 of
202,973) and the write guard does not refuse them. 3 peers would have lost
the same 11; 4 would lose 15.

**"Unknown" is missing, never a sector**, and `fetch_nse_constituents` now
raises when NSE's CSV lacks `company_name` or `industry`, as it already did for
`symbol`. Labels are read from each ticker's LATEST membership row, open or
closed, so a name leaving the NIFTY 100 does not silently become "Unknown".

**The dependency is improved, not removed.** Sector labels are NSE's own
(`source = 'nse-archives'`): loud on death (the download raises), loud on a
column rename (the new schema check), and SILENT on reclassification —
`industry` is written once when a ticker joins and never updated, so the label
applied to 2016 is today's. Recorded, not claimed away.

**All 84 write again** — simulated against the stored label counts through
the write guard's own arithmetic (`tools/stage2_snapshot.py guard`): 84 of 84
under the new benchmark, where the live table has refused 49 since
2026-09-15. The live confirmation is the first daily run after merge.

**`sessions_are_contiguous` closes for the right reason.** The check now flags
surplus rows as well as gaps. On the stored table today, 49 tickers carry 3–4
SURPLUS rows each (phantom sessions their refused writes kept) — the old
one-directional check passed exactly because the difference had inverted.
After the v4 rewrite: 0 gapped, 0 surplus, all 84.

**MODEL_VERSION** `rebuild-absolute-return-v3` → `absolute-return-sector-loo-v4`.

### The measured delta — the prediction FAILED, and the cause is not the benchmark

Pooled × MAE, no ticker, standardised label; per-date cross-sectional rank IC,
paired by date, Driscoll-Kraay at 30 lags. Pre-registered: |Δ| ≤ 0.005, |t| < 2.

| | pass 1 | **pass 2** |
|---|---|---|
| old: Yahoo indices | +0.0183 (t +1.45) | +0.0107 (t +0.72) |
| new: peer benchmark, thin names NULL | +0.0207 (t +1.50) | +0.0233 (t +1.53) |
| **Δ new − old** | **+0.0024 (t +0.40)** held | **+0.0126 (t +2.31)** FAILED |

Investigated as a defect first, as the pre-registration requires.
**These diagnostics are POST HOC** (`tools/stage2_diagnose.py`):

- **No look-ahead.** The new features on 2019-03-15, 2022-06-15 and 2025-01-10,
  for five tickers, are identical with every later price removed from the input.
- **By fold:** Δ is +0.021 (t +2.24) and +0.039 (t +2.99) in folds 0–1, then
  +0.006, −0.005, +0.001. The early-fold shape, again.
- **Restricted to the 73 names that HAVE a sector benchmark, Δ = +0.0022
  (t +0.72)** — inside the prediction. The excess lives in the 11 NULL names.
- **fallback** (the 11 get market-relative momentum instead of NULL):
  +0.0125 (t +0.85), Δ vs old **+0.0018 (t +0.62)**.
- **placebo** (fallback, then the NULLs moved onto 11 RANDOM names, 3 draws):
  Δ vs fallback −0.0024, +0.0025, −0.0029 (|t| ≤ 1.06).
- NULLs on the REAL 11 vs fallback: **+0.0109 (t +2.06)**.
- **min_train sweep** (Δ new − old): +0.0062 / +0.0104 / +0.0117 / **+0.0126** /
  +0.0081 / +0.0087 at 380 / 420 / 460 / **500** / 540 / 580, t +1.07 to +2.31 —
  it crosses 2 only at the default cell.

**Reading.** The benchmark change itself moves the baseline by +0.0018
(t +0.62), inside the prediction. What failed the prediction is the
REPRESENTATION of the 11 thin-sector names: NULL in three features on every
date is a persistent GROUP marker, and the tree learns which group paid in the
early folds — exactly the recorded "a missingness flag is a company
fingerprint" landmine. Random 11-name markers carry nothing; these 11
(consumer, quality and infrastructure names) had a group return the early
folds rewarded. **It is not skill and must not be quoted as an improvement.**
And the fact that the same comparison read +0.0024 in pass 1 says the
between-arm figure is inside the noise that a hyperparameter search flipping
one fold's gamma produces (pass 2's arms chose 0.172 and 3.188 in fold 2).

**This is a design conflict for the user, not a fix to make after seeing the
number.** The explicit NULL is what the prompt asked for and what the silent-
neutral rule requires; it also hands the pooled model an 11-name identity
channel. The shadow path ships as pre-registered (NULL). See §7 for the
recommendation.

## 2. Part 1 — the silent-neutral audit

Every signal column, over the stored table and over the rebuilt v4 panel
(`pipeline/silent_neutral.py`; full table in `stage2/silent_neutral_audit.csv`).
Indicators: modal share and value, share at a neutral point (0, 0.5, 1, 50),
tickers constant across their history, dates constant across tickers, longest
run of an identical value.

| column | stored: flagged indicator | verdict | v4 |
|---|---|---|---|
| earnings_surprise | 36.7% of rows exactly 0.0 (88–94% in 2016–20) | **FAILURE: "not observed" stored as "no surprise"** | NULL before the vendor's first announcement |
| earnings_surprise | run of 1,954 identical values (BAJAJHLDNG) | **FAILURE: stale value carried as current** | NULL after 85 sessions without an announcement |
| earnings_surprise | any vendor failure wrote 0.0 over the whole history | **FAILURE (the lxml path, still open per ticker)** | refused: the ticker is SKIPPED, its stored signals kept |
| hurst | 0.0 for EVERY ticker on 13 dates in late 2016; 0.5 when incomputable | **FAILURE: partial-window estimator + neutral fill** | NULL until every lag has a full window |
| sector_rel_* | 0.0 when the benchmark was unavailable; dead indices forward-filled | **FAILURE** (Part 0) | peer benchmark; NULL + indicator for thin sectors |
| benchmark_close | runs of up to 41 identical values | **FAILURE: dead index forward-filled** | longest run 2 |
| vroc_10, lag1_ret | 4 dates constant across tickers | genuine given the data: the four 2026 phantom sessions the 49 refused tickers still hold | 0 (rewritten) |
| vroc_10, lag1_ret, roc_10 | small share at −1 / 0 | genuine: 2025-03-18's flat bar (a tagged vendor defect) and real zero moves | unchanged |
| stoch_k, williams_r, prox_52w | ≤ 0.1% at their bounds | genuine: bounded oscillators touch their bounds | unchanged |
| hurst | 0.2–0.6% at 0.0 after warm-up | genuine: the estimator's lower clip | unchanged |
| usdinr, india_vix, nifty_* | every date constant across tickers | genuine: market-wide by construction (never in FACTORS) | unchanged |

**Fixed the way `require_earnings_parser()` was:** refuse rather than write a
neutral value. A configuration failure (no HTML parser) fails before the loop;
a per-ticker vendor failure SKIPS that ticker — it keeps what it has and is
named in the run's `skipped` list — instead of replacing its whole history
with zeros through the DELETE-range rewrite. `EARNINGS_STALE_SESSIONS = 85`
is measured: 1,902 consecutive announcement pairs, median gap 91 calendar
days, 95th percentile 112; 85 sessions ≈ 120 days.

**NULLs reach the models as missing.** `signals.NULLABLE_FEATURES` rows are
never dropped; `load_panel` and `load_features_for_ticker` keep them NaN;
`cross_sectional_zscore(keep_missing=True)` keeps them NaN for the pooled
model, which XGBoost reads with a learned default branch. The ridge
comparators keep the old fill (they cannot read NaN).

**The standing check** (`validation.check_no_silent_neutral_signals`, WARN):
over the last 120 sessions, any ticker constant at a neutral value, or any date
constant across every ticker, for every scale-free feature. On the v4 panel:
quiet. On today's stored table: it flags the three 2026 phantom dates the
refused tickers still hold (true positive; clears on the rewrite). **Replayed
on the 2026-09-22 state it names all five zeroed `earnings_surprise` tickers**
— on the run that wrote them, not weeks later.

## 3. Part 2 — the Linux reference

| | |
|---|---|
| snapshot | `76fe43a52bdcf2950c583763c8bf148752d67fdcdba5e70c7aeaa634b6ec3e6d` |
| new-arm panel (built on WSL) | sha256 `95a29cb3…45d0f75`, 205,493 rows, 84 tickers |
| old-arm panel | sha256 `30eeb02f…40fc32f` |
| reference predictions | sha256 `1f298ab5…63532567`, 161,028 rows, gammas 4.744 / 3.188 / 0.172 / 0.172 / 0.172 |
| **reproduction** | **A = B = C, drift exactly 0.0 on predictions and labels, every row matched** (pass 1 likewise) |

**CI vs WSL — informational until a commit is pushed.** `tools/platform_digest.py`
fits the production pooled path on a fixed synthetic panel with subsampling
< 1.0: WSL `ab65d091…136a77`, reproduced exactly; Windows `61842f00…e9720f`
(different, as the landmine says). The runner's digest comes from
`.github/workflows/platform-digest.yml` on the first push. If it differs, the
runner is the reference and B1–B6 are re-established there.

### Pooled vs per-ticker — same snapshot, same platform, 160,973 common rows

| | pooled (shadow) | per-ticker (live) |
|---|---|---|
| constant (ticker, fold) cells | **0 / 420** | **315 / 420** |
| cross-sectional rank IC (DK t) | +0.0232 (+1.52) | −0.0167 (−0.82) |
| … the same with the thin-sector NULLs as market fallback | +0.0125 (+0.85) | — |
| MAE, 30-session log return (price space) | **0.09048** | 0.09473 |
| conformal, overall (band 0.75–0.85) | 0.8684 **FAIL** | 0.8590 **FAIL** |
| conformal, folds 1–4 (band 0.70–0.90) | 0.826 / 0.880 / 0.867 / **0.903** | **0.721** / **0.912** / 0.895 / **0.912** |
| grades, `grade_panel_v3` (B = 1000) | 1 STRONG (WIPRO) / 3 WEAK / 80 | 0 / 4 WEAK / 80 |
| Romano-Wolf rejections, alpha 0.10 | 1 | 0 |
| tau2 (REML) / panel statement? | 0.00251 / no | 0.00280 / no |
| live gate's own grade | — | 84 INSUFFICIENT |

### The pre-registered predictions

| # | prediction | outcome |
|---|---|---|
| §4 | Part 0 |Δ| ≤ 0.005, |t| < 2 | **FAILED** in pass 2 (+0.0126, t +2.31); held in pass 1 (+0.0024, t +0.40); the excess is the NULL fingerprint (§1) |
| §5 | exact reproduction, twice | **HELD** (three runs, both passes) |
| 6.1 | pooled 0 constant cells; per-ticker ≥ half | **HELD** (0 / 420 and 315 / 420) |
| 6.2 | both cross-sectional ICs |t| < 2 | **HELD** (+1.52, −0.82) |
| 6.3 | pooled MAE within +5% of per-ticker | **HELD** — 4.5% LOWER |
| 6.4 | pooled 0 STRONG; tau2 likely at the boundary | **FAILED**: 1 STRONG (WIPRO) and tau2 0.0025, so grades are per ticker. Stage 0c's nine-draw placebo produced 1 STRONG in 9 on noise: one STRONG is inside the null band and is not evidence |
| §7 | conformal gate | **FAIL** (below) |

## 4. The conformal gate — FAIL, and the calibration was not touched

Pooled, in price space after the CAUSAL inverse, expanding calibration:
overall **0.8684** against [0.75, 0.85]; fold 4 **0.9033** against [0.70,
0.90]. The half-width falls from 0.170 (calibrated through fold 0) to 0.145
(through fold 3) while the target's dispersion falls faster, so every later
fold OVER-covers — the pre-registered risk, measured. The per-ticker model
fails the same gate worse per fold (0.721 then 0.912), so the live "80.1%" is
an aggregate hiding two opposite errors.

**Per the pre-registration: report and stop.** Nothing was recalibrated. The
failure blocks the cutover; the shadow path ships with the calibration
unchanged and writes `coverage_status = 'FAIL'` on every shadow model and
forecast row. The remedies, each to be pre-registered as its own step:

1. **Volatility-normalised nonconformity scores — recommended.** Score the
   residual divided by the causal cross-sectional sd, and rescale the band by
   the current causal sd. The failure mode is dispersion drift, and this makes
   the band move with dispersion instead of carrying the wild early period's
   width into a calm late one.
2. Recency-weighted or rolling-window calibration (simple; trades a narrower
   calibration pool for recency).
3. Adaptive conformal inference (Gibbs & Candès, 2021): online adjustment of
   the miscoverage level; strongest guarantee under drift, most machinery.

## 5. Part 3 — the shadow path

`pipeline/pooled.py` (the model), `pipeline/pooled_shadow.py` (storage, the
gate, the weekly and daily steps), wired into `scheduler.py` as NON-FATAL
steps recorded in `experiment_runs.metrics["pooled_shadow"]`:

- **weekly**, after the per-ticker evaluation persists: walk-forward with the
  nested search, `grade_panel_v3` (demeaned within-fold IC, date-level
  bootstrap B = 1000, REML/HKSJ, Romano-Wolf, the tau2 ≈ 0 detector), the
  coverage gate, the conformal calibration, the final fit → `shadow_models`,
  `shadow_evaluations`, `shadow_panel_statements`;
- **daily**, after the live forecasts: predict every name from the latest
  shadow model, invert with the CAUSAL moments, price with the shadow
  calibration → `shadow_forecasts`.

Every shadow row carries `pooled_version` (`pooled-std-mae-noticker-v1`),
`config_hash`, `data_hash` and `env_hash`. **Romano-Wolf is wired:** the grade
is `grade_panel_v3`'s, whose STRONG requires the rejection; `rw_rejected` and
the adjusted p are on every evaluation row (a test drives a synthetic ticker
to STRONG through it). **A degenerate panel is one statement**, the rows carry
grade NULL (tested). **The inverse is past-only** (tested by corrupting every
unrealised label). **Isolation:** the shadow tables' DDL lives in the module,
not `data/db.py`; no public endpoint reads them; public responses are
byte-identical with shadow rows present (tested, excluding only the wall-clock
`last_updated` stamp); inspection is `GET /api/admin/shadow/summary` and
`/forecasts`, API-key-gated and linked from nowhere (tested).

**Runtime and memory, measured** (`tools/stage2_runtime.py` under
`/usr/bin/time -v taskset -c 0-3`, i.e. four cores as on a public-repo
`ubuntu-latest` runner, full production settings: 10 nested trials, B = 1000,
the snapshot panel loaded into a throwaway SQLite with the real schema):

| step | wall time | of which | peak memory |
|---|---|---|---|
| weekly shadow | **3 min 35 s** | walk-forward 102 s, grading 67 s, final fit + write ~37 s | **1.24 GB** |
| daily shadow | **3 s** | one panel load, one predict | (within the above) |

The last weekly run took 64 min in all (49 min evaluation + 14 min Stage 0
shadow grading) against a 300-minute timeout. The thread pin (merged
2026-09-21, not yet exercised by a weekly run) is expected to lengthen the
per-ticker evaluation — CLAUDE.md measured 2.75x on a large fit — so the
honest projection is ~135 min + ~10 min for this step with a 2-3x allowance
for a slower runner CPU: under 60% of the budget, and 1.24 GB of 16 GB. **It
fits; no bootstrap draw was cut.** O3 is confirmed on the first real run.

**Schema:** four NEW tables, created by `init_shadow_tables()` from the jobs
themselves (no `data/db.py` DDL), plus ONE new `signals` column
(`sector_rel_missing`, the Part 0 indicator) in `data/db.py`'s safe-migration
list. **Render redeploy: yes** — for the `data/db.py` column, `data/tickers.py`
(benchmark display names), `data/universe.py` (the schema check) and the new
admin endpoints, not for the shadow tables themselves.

## 6. The cutover checklist

| criterion | status |
|---|---|
| B1 reproduction | **met** |
| B2 pooled 0 constant cells | **met** |
| B3 pooled cs IC not significantly negative | **met** (t +1.52; +0.85 without the fingerprint) |
| B4 MAE within +5% | **met** (−4.5%) |
| B5 conformal gate | **NOT MET** — blocks the cutover |
| B6 grades through the Romano-Wolf-wired path | **met**; tau2 not at the boundary, so no panel statement was needed |
| B7 runner digest = WSL digest | **waiting** on the first push |
| O1 two weekly shadow cycles | **waiting** (first: the Saturday after merge) |
| O2 dailies write 84 shadow forecasts | **waiting** |
| O3 weekly runtime < 80% of timeout, memory fits | measured on a 4-core Linux equivalent (§5); confirmed on the first real run |
| O4 public API unchanged | tested; spot-check after deploy |
| O5 the user's go-ahead | — |

Two shadow cycles will show the path runs, writes, grades and reproduces in
production. **They cannot show live predictive performance or live coverage**
— a forecast resolves 30 sessions later — and no shadow-cycle figure is to be
read as either. That evidence is §3–§4.

## 7. The read

**Ready to shadow: yes.** It is a model that ranks every name on every date,
reproduces exactly on the reference platform, prices with a past-only inverse,
grades through the corrected layer and writes nothing the public reads.

**Against eventually cutting over, as things stand:**

1. **The conformal gate fails**, for both models. Cutting over to a band that
   reads "80%" while covering 87–90% in the recent regime is publishing a
   wrong number. Fix first, pre-registered (§4, recommendation 1).
2. **The NULL representation gives the pooled model an 11-name identity
   channel** (§1). Recommendation: give thin-sector names market-relative
   momentum in the feature, with `sector_rel_missing` / `benchmark_sector_
   specific` recording that the reference is the market — the `fallback` arm,
   which is a real measurement against a different, recorded reference rather
   than a neutral sentinel, and which carried no fingerprint (placebos flat).
   A decision for the user, and a pre-registered change if taken; it restarts
   the shadow cycles.
3. None of this argues against the switch's PURPOSE. The per-ticker model
   holds a constant in 315 of 420 cells, its cross-sectional IC is negative,
   its MAE is 4.7% higher and its coverage is miscalibrated in both directions.
   The pooled model is null too — as predicted — but it is a null that can be
   measured.

## 8. Recorded, deliberately not fixed

`pipeline/macro.py`'s holiday-inclusive Nifty returns (a per-ticker feature;
changing it mid-shadow would muddy the comparison); 2025-03-18's flat bars
(tagged); the unlocked torch/transformers extras; the daily job's connection
retry; the `experiment_runs` cleanup SQL. The v4 bump also splits the
grade-predictiveness cohort `tools/score_grade_predictiveness.py` waits on:
its default stays v3, which holds the forecasts published 2026-09-02 onward.
