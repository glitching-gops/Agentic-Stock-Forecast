# Stage 2, continued — the thin-sector fallback and the conformal fix — findings

2026-09-24. Measured against `docs/stage2-fallback-conformal-preregistration.md`
(sha256 `8abd1ebf53f47897c2ec604270ac7e605f5382122b376e91c6bed0530a809e86`,
written and hashed at 04:33 UTC, before any Part B panel was built or any Part C
coverage computed). Every run used the frozen snapshot `76fe43a5…3e6d` (cutoff
2026-09-23, re-verified), on WSL Ubuntu, CPython 3.12.0, the locked environment
(108 pins), with `XGB_THREADS = 2`. Each output under `stage2/v5/` (gitignored)
carries the pre-registration's hash, the snapshot and the platform.
`tools/stage2_fallback_report.py` applies the deciding rules.

**Headline.** Both parts PASS their pre-registered rules. **Part B:** the
11-name fingerprint is gone. The benchmark change now moves the baseline
+0.0009 (t +0.31), which is Part 0's original bound, met. The pooled
cross-sectional IC is +0.0116 (t +0.79), still null. **Part C:** the
spread-normalised interval covers 0.825 overall, and 0.848 / 0.820 / 0.775 /
0.859 by fold. That is inside the UNCHANGED bands, where the constant-width
interval on the same rows covered 0.869 and failed. Neither result is an
accuracy claim: the model is exactly as null as before, and its band now says
"80%" more truthfully.

**The CI runner reproduces the reference platform.** On the push of
`6c52836`, `platform-digest.yml` (run 35953248538) printed
`ab65d091…136a77`, bit for bit the WSL digest. So these WSL numbers stand for
the runner, and cutover criterion B7 is met.

## 1. Part B — market-relative momentum for the 11 thin-sector names

**The change.** A name whose sector has fewer than three constituents computes
`sector_rel_{5,10,20}d` against the leave-one-out equal-weighted market. This
is the SAME `Benchmark` object its excess label already used:
`EW-LOO:MARKET`, built from `regime.market_log_returns`, summed over the peers
directly, with the same void-window rule. The same holds for an unlabelled
("Unknown") name.

**Alignment with the excess label: no difference found.** Before v5 the label
fell back and the feature went NULL. Both now read one object, so they cannot
disagree. A test holds `sector_rel_w = Δ_w close − Δ_w benchmark_close` and
`benchmark_return = Δ_30 log benchmark_close` on the same stored column.

**The flag is RETIRED, not renamed.** Its new meaning would be "the reference
is the market". That is exactly `benchmark_sector_specific = 0`, which is
already stored from the same object, and two columns for one fact can drift.
`sector_rel_missing` is no longer written. The `signals` column stays, so a
fresh database matches the live one, and rows from v5 on hold NULL there.
Neither column was ever a model input. A test asserts that neither the flag,
`benchmark_sector_specific` nor `benchmark_ticker` appears in any feature list,
or in the feature names of a fitted pooled booster that was handed all three.

`MODEL_VERSION` → **`absolute-return-sector-loo-market-v5`**;
`POOLED_MODEL_VERSION` → **`pooled-std-mae-noticker-v2`**.

### Results against the pre-registration

| rule | prediction | measured | |
|---|---|---|---|
| B0 | panel equals the post-hoc fallback panel | **not identical**: differs only on the 11 names' three `sector_rel_*` columns (details below); every other FACTOR and both labels are identical on all 205,493 rows | reported |
| B0 | two v5 runs reproduce | **exact**, drift 0.0 | held |
| B1 | the 11's NULL share = the 73's, ±0.5pp | **0.000 vs 0.000** | **held** |
| B2 | random-11 NULL arms within ±0.005 of v5 | −0.0008 (t −0.29), +0.0027 (t +0.93), +0.0011 (t +0.34) | **held** |
| B2 | real-11 NULL arm above every random draw | **+0.0117 (t +2.27)** | **held** |
| B3 | v5 − old Yahoo arm: \|Δ\| ≤ 0.005, \|t\| < 2 | **+0.0009 (t +0.31)** | **held** |
| B4 | v5 − v4 (NULL version) ≈ −0.011, descriptive | −0.0117 (t −2.27) | as expected |
| B5 | v5 cs IC \|t\| < 2 (expected ≈ +0.0125) | **+0.0116 (t +0.79)** | **held** |

**Part B: PASS.**

**B0: why production differs from last session's diagnostic.**
`tools/stage2_diagnose.py` patched a finished panel. It recomputed the momentum
from the float32-rounded `close` and `benchmark_close`, starting at each
ticker's first STORED row. Production computes from full-precision prices over
the whole price history, and rounds afterwards. So on all 26,917 thin-name rows
the values differ by at most 2.1e-7. On the first 5 / 10 / 20 stored rows of
each of the 11 (55 / 110 / 220 cells), production has a value where the
diagnostic had NaN. The diagnostic arm itself therefore carried a small version
of the fingerprint it was built to remove. Production is the correct
construction. Its predictions differ from the diagnostic's (max drift 0.48), so
the numbers above are new measurements, and they land where the diagnostic
said: +0.0116 against +0.0125, and Δ vs old +0.0009 against +0.0018.

**The consistency check is exact.** Putting the NULLs back on the real 11,
starting from the v5 panel, reproduces last session's v4 reference predictions
**bit for bit** (drift 0.0 on all 161,028 rows). The whole v4-to-v5 difference
therefore sits in those 11 names' three columns, and in nothing else the
rebuild could have touched.

**By fold, v5 − v4:** −0.0135, **−0.0304 (t −3.04)**, −0.0120, +0.0014,
−0.0036. The v4 excess sat in folds 0–1, as recorded, and it is gone.

**What B2 does and does not show.** It is the same comparison as last
session, re-run on production-built panels. It was not blind: §1 of the
pre-registration says so. It confirms that the production change implements
the arm that was diagnosed. It is not new evidence that the arm is right.

## 2. Part C — the spread-normalised interval (pooled model only)

**The method, as pre-registered.**
- **Score and interval.** Score `e = (y − ŷ) / s(t)`. Split-conformal on |e|
  at rank `ceil((n+1)·0.8)`. Interval `ŷ ± q·s(t)`.
- **The spread.** `s(t)` is the mean realised cross-sectional sd of the
  30-session label over the 21 grid dates ending 30 dates before t
  (`pooled.spread_frame`, through `label.causal_moments`). It is past-only.
  One test corrupts every label that has not yet realised and requires `s(t)`
  unchanged. Mutants that read the spread on the forecast date, or one session
  early, both fail it.
- **The point inverse** keeps its 252-date causal moments, unchanged.
- **Scoring.** Expanding by fold, in price space. No row lacked a spread.

| fold | realised dispersion | constant: coverage | width (log) | width (% of price) | **spread-normalised: coverage** | width (log) | width (% of price) |
|---|---|---|---|---|---|---|---|
| 1 | 0.109 | 0.824 | 0.339 | 35.4 | **0.848** | 0.366 | 38.2 |
| 2 | 0.095 | 0.881 | 0.328 | 33.5 | **0.820** | 0.290 | 29.6 |
| 3 | 0.083 | 0.868 | 0.305 | 31.8 | **0.775** | 0.248 | 25.9 |
| 4 | 0.083 | **0.904** | 0.291 | 29.3 | **0.859** | 0.256 | 25.8 |
| **overall** (128,352 rows) | | **0.869 FAIL** | | | **0.825 PASS** | | |

Bands: overall [0.75, 0.85], each fold [0.70, 0.90], unchanged from the
failed gate. **Part C: PASS.**

- **Constant method on these rows:** 0.869 and 0.904 in fold 4. This is the
  original failure, reproduced on the v5 model (v4's own was 0.868 / 0.903).
- **Width tracks dispersion.** Across folds, width and realised dispersion
  have Spearman +0.80. The band is WIDER in the dispersed fold 1 (38.2% of
  price vs 35.4%) and narrower in the calm folds 3–4 (about 26% vs 30–32%).
  That is the pre-registered picture, not a uniformly narrower band.
- **Date by date the tracking is weak.** Across the dates of the checked
  folds, the correlation of `s(t)` with the date's realised dispersion is only
  **+0.22**. The spread estimate follows the regime. It does not forecast the
  next 30 sessions' dispersion. Fold 3 (0.775) and fold 4 (0.859) still sit
  4–6pp either side of 0.80: inside the bands, but not on target.
- **For reference only:** the same method on the v4 (NULL) model gives
  0.823, with folds 0.848 / 0.818 / 0.773 / 0.857. So the pass does not
  depend on Part B.

**Skepticism, stated plainly.** This is one pre-registered method at one
pre-registered lookback, and no other lookback was run. Running some would
turn a pass into a search. The honest precision is limited:
- Each fold holds about 13 non-overlapping 30-session windows.
- One market shock moves every name's residual at once.
- A fold's coverage is therefore an estimate with a few points of noise.
The pass says the band no longer carries the early period's width into the
late one. It does not say the band is exactly 80% in any given month.

**Production.** The weekly shadow step now works as follows:
- It calibrates `q` in spread units on every out-of-sample row, and stores it
  with `method = 'spread-normalised'` and the 21-date lookback in
  `calibration_json`.
- It judges the interval with the same gate. `coverage_json.method` records
  which interval was judged; a mutant that scored the constant interval
  survived until that was asserted.

The daily step prices each forecast at `q × s(as_of)` and stores `s(as_of)`
in the new `shadow_forecasts.interval_spread` column. That column is added to
an existing table by `init_shadow_tables`, which is tested. With no spread,
the step writes no interval, never one at a made-up width.

**Recorded for the cutover session: the live per-ticker model's intervals
were NOT fixed.** The site publishes per-ticker bands labelled 80% whose
measured coverage on this snapshot is 0.859 overall and 0.721 / 0.912 / 0.895 /
0.912 by fold. The cutover copy must say that the band changes and why: the
old band was right on average and wrong in both directions period by period.

## 3. The cutover checklist, updated

| criterion | status |
|---|---|
| B1 reproduction | **met** (v5: drift 0.0) |
| B2 pooled 0 constant cells | met |
| B3 pooled cs IC not significantly negative | **met** (+0.0116, t +0.79; the fingerprint is gone, so no qualifier is needed now) |
| B4 MAE within +5% of per-ticker | met on v4 (−4.5%); not re-measured for v5, whose predictions differ only through the 11 names' three features |
| B5 conformal gate | **MET** (0.825; every fold inside) |
| B6 grades through the Romano-Wolf-wired path | met on v4; re-graded by the first v5 weekly shadow run |
| B7 runner digest = WSL digest | **MET** (run 35953248538) |
| O1 two weekly shadow cycles | **restarts**: the first is the first weekly run after this lands |
| O2 dailies write 84 shadow forecasts, now with `interval_spread` | waiting |
| O3 weekly runtime and memory | measured before (3 min 35 s, 1.24 GB); `spread_frame` adds one groupby |
| O4 public API unchanged | tested |
| O5 the user's go-ahead | — |

Backtest criteria B1–B7 are now all met. **The cutover waits only on the
operational ones.** Two shadow cycles show that the path runs. They cannot
show live coverage, because a forecast resolves 30 sessions later.

## 4. Tests

The full suite ran on the final tree: **888 pass, 0 fail, 35 skipped** (the
torch tests and the two opt-in reproductions). One boundary test was added
afterwards; it passes. New coverage is `tests/test_stage2_fallback_conformal.py`,
plus assertions added to the pooled-shadow end-to-end test.

**Mutants on the new guards: 17 of 17 caught.** One survivor was closed first:
a weekly gate that scored the constant interval.

Last session's 18 mutants were re-run on this tree:
- 16 are caught;
- 1 no longer applies, because the NULL-fill code it mutated is gone;
- 1 survived: `>` for `>=` at `MIN_SECTOR_PEERS`. The v4 thin-sector test had
  been the one holding that boundary. A three-name-sector test now holds it,
  and the mutant is caught.

## 5. Not done, deliberately

- A second conformal method (none was needed).
- Any lookback other than 21.
- The per-ticker model's intervals.
- `macro.py`'s holiday-inclusive Nifty returns.
- 2025-03-18's flat bars.
- The torch/transformers extras.
- The daily connection retry.
- Universe work.

The v5 bump splits the grade-predictiveness cohort again.
`tools/score_grade_predictiveness.py` keeps defaulting to v3.
