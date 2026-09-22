# Stage 1 re-run — pre-registration

Written 2026-09-21, before any Stage 1 arm was re-run under the conditions
below. Its sha256 is recorded in the commit that adds it and printed by
`tools/stage1_rerun.py` at the start of every real run; the tool refuses to run
without this file on disk.

## 1. Why re-run at all

The three Stage 1 pilots (reversal 2026-09-13, delivery % 2026-09-13, SUE
2026-09-19) were measured on a model that no longer exists. Since then:

- **the label** is the within-date standardised one (`pipeline/label.py`). On
  the old raw label the pooled × MAE model emitted a median of **8 distinct
  predictions per date across 84 names**; standardised, 84. A model that can
  only express eight ranks can only reward a feature that moves a name across
  one of seven boundaries, so the old nulls may partly be a statement about
  resolution rather than about the features;
- **threads** are pinned at `XGB_THREADS = 2` (the old runs used 20);
- **libraries** are locked (`requirements*.txt`, hashed, 2026-09-21);
- **the panel** is rebuilt without the four 2026 sessions NSE did not trade
  (2026-01-15, 05-01, 05-28, 06-26), from NSE's own calendar.

## 2. Hypothesis, and the expected outcome

**H0 (expected): the three Stage 1 nulls were produced by the features, not by
the old label's low ranking resolution. All three remain null.**

The alternative is that the resolution confound hid a real gain, which would
show up as an arm clearing the deciding rule below where it did not before.

## 3. Design — identical for every arm

- h = 30 sessions, legacy purge (30), 5 purged folds, `EVAL_MIN_TRAIN_DATES`.
- pooled × MAE objective, **no ticker feature**, `EVAL_TUNE_TRIALS` = 10.
- within-date standardised label; `XGB_THREADS` = 2; the locked libraries.
- the calendar-clean panel `panel_cache_clean.parquet` built by
  `tools/phantom_rebuild.py`, sha256
  `c0a26d4593917946bbb7fc2a9a314ab9ccc363828b5569aad8272e2883c93a75`.
- the §4 baseline's prediction arrays (`arm_arrays_sha256`, the Stage 1
  convention, since an `.npz` file's own hash moves with its zip timestamps):
  `bed813ac312b4165443211f375837708b19baecb0803ac5626ea67aabda17948`.
  Its measured figures, known before this was written and stated so nothing is
  hidden: cs IC +0.01142 (DK t +0.81), rebalance IC +0.0294 (t +1.41),
  0 STRONG / 0 WEAK / 84 INSUFFICIENT, 160,104 rows.
- the inputs the features read: `delivery_cache.npz`
  `f8dbe738cefb37afafaaf237b6b4775e16c27e5a0581b16e7e7d9b9fdfeb55dd`,
  `results_cache.npz`
  `7d6ebaf74a59710e659afeefb9cc1fc39d135335ea94e084207bf6ad1359fdd8`.
- **Baseline:** FACTORS only, the §4 baseline `baseline_clean_oos.npz` arm
  `new_pinned`. **S1, the stop condition:** before any feature arm is read, the
  baseline is re-run inside the tool on each pilot's panel and must reproduce
  the stored baseline at drift **exactly 0.0**. If it does not, nothing else is
  read.
- **Reference platform: Windows**, the platform the stored baseline was
  produced on. XGBoost's row/column subsampling differs between platforms at
  the same seed (measured 2026-09-21), so every arm runs on the same one.
- Features, ingestion and feature construction are REUSED unchanged:
  `tools.stage1_reversal.attach_reversal`, `tools.stage1b_delivery`'s
  `delivery_features` / `attach_delivery`, `tools.stage1c_sue`'s
  `sue_announcements` / `raw_features` / `attach_sue`.

### Arms

| pilot | arm | features |
|---|---|---|
| reversal | `resid_reversal` (the hypothesis) | FACTORS + 4 residual reversal columns, k = 1, 5, 10, 20, skip one |
| reversal | `raw_reversal` | FACTORS + the same 4, not residualised |
| delivery | `abnormal` (the hypothesis) | FACTORS + `deliv_abn_l1`, `deliv_abn5_l1` |
| delivery | `level` | FACTORS + `deliv_pct_l1` |
| SUE | `sue` (the hypothesis) | FACTORS + `sue_evt`, `sue_age`, `sue_missing` |
| SUE | `timing` | FACTORS + `sue_age`, `sue_missing` |

Pilot 3's descriptive `naive` arm (forward-filled SUE) is not re-run; it decided
nothing.

## 4. The rules

- **R3 — deciding.** The paired cross-sectional rank-IC gain of the hypothesis
  arm over the baseline, per date, with a Driscoll-Kraay SE (30 lags). A gain
  counts only at **t ≥ +2.0**.
- **R2 — the placebo.** The hypothesis arm's new columns are permuted JOINTLY
  within each date and the model is RETRAINED, nine seeds per pilot. Predictions
  are never shuffled. R2 passes only if the real gain beats the maximum of the
  nine placebo gains. Seeds: reversal 20260921-20260929, delivery
  20260914-20260922 (Pilot 2's), SUE 20260920-20260928 (Pilot 3's).
- **R5 — SUE only.** The surprise must add over timing and coverage alone:
  `sue` − `timing` paired gain > 0.
- **Resolution check, every arm.** Distinct predicted values per date (median
  and minimum over dates) and constant (ticker, fold) cells. It confirms the old
  confound is gone rather than assuming it.
- Grades (STRONG / WEAK / INSUFFICIENT) through `grade_panel_v3` at Stage 0c's
  settings, reported for every arm and never decisive.
- A signal requires R3 AND R2 (and R5 for SUE). Only then is the pre-registered
  `min_train` sweep (380-580) run.

## 5. Multiple testing, stated in advance

Roughly **150 trials** have now been run on this panel. Under pure noise the
expected maximum |t| over that many draws is about **3.2**. **A single arm
crossing t = 2 is not, on its own, a finding.** It would first be investigated
as a defect (leakage through the rebuilt panel, the label transform, a feature
join), then read against the 3.2 deflation, and only then described.

## 6. What would change the Stage 1 conclusion

Stated before the run:

- **It stands** if every hypothesis arm fails R3 (t < +2.0), whatever the grades
  do. The asterisk on `docs/stage1-closing.md` is then removed: the nulls were
  the features.
- **It is reopened, not overturned,** if a hypothesis arm clears R3 AND R2 (AND
  R5 for SUE) AND survives the `min_train` sweep with neighbouring cells of the
  same sign AND the defect checks in §5 find nothing. Even then it is a
  hypothesis for the universe-change session, deflated at ~150 trials, not a
  result.
- **A grade-only movement changes nothing.** Pilot 2 showed that retraining with
  any two extra noise columns moves the same grades.

## 7. Predictions

- P1: S1 passes at drift exactly 0.0 on all three pilot panels.
- P2: every hypothesis arm fails R3.
- P3: every hypothesis arm fails R2 or R3.
- P4: SUE fails R5.
- P5: every arm's median distinct predictions per date is ≥ 80 of 84, and 0
  constant cells — the old resolution confound is gone in every arm, not only
  the baseline.
