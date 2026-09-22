# Stage 1 — closing the new-data track

> **CORRECTION NOTICE — 2026-09-21. The numbers below are not reproducible as
> written, and two of the reasons are defects rather than noise.** Nothing here
> is rewritten; see `docs/hygiene-findings.md` for what changed and by how
> much.
>
> 1. **They were produced at an unpinned thread count.** Nothing in the
>    repository pinned XGBoost's `n_jobs`, so every fit ran at whatever this
>    workstation defaulted to (20). Measured: the same code with the same seeds
>    gives different hyperparameters and a different model at a different
>    thread count — zero of 162,535 predictions matching, a maximum difference
>    of 6.2 prediction standard deviations. Re-pinned at `XGB_THREADS = 2`, the
>    30-session baseline moves from cs IC −0.00101 to −0.00423 and from
>    1 STRONG / 1 WEAK / 82 to 2 STRONG / 3 WEAK / 79. **Every verdict here
>    survives — a null that wobbles into another null is still a null — but the
>    digits do not.**
> 2. **They were produced on the RAW label**, which is no longer the pipeline's
>    default training target. `gamma` is denominated in the loss, so the fixed
>    `[0, 5]` search range meant something different at every label scale;
>    the within-date standardised label (`pipeline/label.py`) is the default
>    now, and under it the same architecture emits 0 of 420 constant cells
>    instead of 7, and 84 distinct predictions per date instead of 8.
>
> 3. **Any QUANTILE or BOOK figure here — long-short spread, top-quintile
>    return, alpha, net-of-cost — was computed with a tie guard that refused
>    only a wholly tied cross-section.** The pooled model's predictions are
>    discrete, and 48 of the 64 h=30 books had legs that were MAJORITY chosen
>    by the ticker tie-break rather than by the model. Those columns are
>    withdrawn rather than corrected. **The rank-IC figures are unaffected and
>    reproduce to the digit**, because `rank_ic` averages ranks over ties.
>
> The session that measured all of this changed no conclusion in this
> document.

**The single document to hand a reader for Stage 1.** It carries the three
pilots, what each one failed on, and the one reading that ties them together.
The per-pilot detail stays where it was written: `docs/stage1-findings.md`,
`docs/stage1b-findings.md`, `docs/stage1c-findings.md`, each beside the
pre-registration it was run against.

Written 2026-09-20, after Pilot 3. Stage 1 is closed on the terms below and
reopens only if something in §5 changes.

---

## 1. What Stage 1 asked

Every Phase 0–6 null on this panel was measured on price-derived features. The
obvious remaining explanation was that the panel had never been shown anything
else. Stage 1 was three attempts to show it something else, each pre-registered
before it ran, each graded through the same Stage 0c harness
(`grade_panel_v3`), each added to the same architecture — pooled × MAE, no
ticker feature — so that the FEATURE was the only thing that moved.

Three pilots ran. All three are nulls. They are not the same null, and that is
what Stage 1 bought.

---

## 2. The three pilots

| | Pilot 1 — reversal | Pilot 2 — delivery % | Pilot 3 — SUE |
|---|---|---|---|
| ran | 2026-09-13 | 2026-09-13 | 2026-09-19 |
| source | price only | NSE `MTO_<ddmmyyyy>.DAT` | NSE results filings |
| new data? | no | **yes** | **yes** |
| **R3, the deciding rule** | **+0.0002, t +0.07** | **−0.00066, t −0.19** | **−0.0055, t −0.56** |
| verdict | NOT SIGNAL | NOT SIGNAL | NOT SIGNAL |

### Pilot 1 — multi-lookback residual reversal. Failed on R3.

Four skip-one lookback returns (k = 1, 5, 10, 20), residualised on
`regime.rolling_beta` lagged one session, plus the same four raw.

- S1 passed: the baseline reproduced at drift **0.0**.
- R1 passed — the residual arm moved ABB.NS out of INSUFFICIENT — and R2
  passed by one ticker against a nine-draw placebo.
- **R3 failed: the paired cross-sectional IC gain is +0.0002 at t +0.07.**
- R4: residualising against the market made no measurable difference
  (residual − raw +0.0049, t +0.75).
- R6: every min_train cell within |reb t| < 1.

**What it moved was one ticker's grade and nothing tradeable.** Each feature
alone is 1.2–1.5× its raw twin, against Da-Liu-Schaumburg's ~4×, and no
feature exceeds t +1.84.

**Its durable output was a correction to the method, not a result.** Pilot 1's
first placebo shuffled the arm's PREDICTIONS within each date, which destroys
the baseline's own ordering too and therefore credits the arm with grades the
baseline already had. **The null for "does X help" is to shuffle X's COLUMNS
within each date and retrain.** That correction is what made Pilots 2 and 3
readable, and it is now standing policy.

### Pilot 2 — NSE delivery %. Failed on R3 and R2.

Abnormal delivery (t−1 against its own trailing 60 sessions), plus a raw-level
arm as the identity-risk control.

- S1 passed at drift **0.0**.
- **R3 failed: −0.00066, t −0.19.**
- **R2 failed: the arm beat 8 of 9 column-permuted retrains, not all 9.**
- The raw level gains +0.0005 at t +0.08. No delivery column's own
  cross-sectional IC reaches |t| 0.6.

**The data are genuinely new and genuinely useless here.** The backfill covers
2,436 of 2,440 panel sessions at 97.7–100% a year, and on UDiFF-era dates MTO
equals `sec_bhavdata_full`'s deliverable quantity exactly, so this is not a
coverage failure.

**And the corrected placebo settled Pilot 1 retrospectively.** The abnormal arm
makes the same grade moves both reversal arms made — ABB.NS into STRONG,
NESTLEIND.NS out of WEAK — and **so do 6 of 9 retrains whose added columns are
pure noise.** That grade movement is what retraining with ANY two extra columns
produces.

**Checked afterwards (2026-09-19):** the panel's four phantom 2026 holiday
sessions all sit in fold 4's TEST window, and no fold's training labels reach
past 2025-01-03, so no fitted model ever saw one. R3 re-scored on the stored
predictions reads t −0.19 as run, −0.19 with the labels recounted over 30 real
sessions, and −0.18 truncated before the first phantom — the last of which is
exact rather than approximate. **The verdict stands.**

### Pilot 3 — seasonal-random-walk SUE. Failed on all four rules.

SUE = (EPS_q − EPS_{q−4}) / σ(EPS_q − EPS_{q−4}), from EPS **as first
disclosed** in NSE's own filings, entering as a 30-session event window plus
days-since and a missingness flag.

| rule | verdict | figure |
|---|---|---|
| S1 — the baseline reproduces | PASS | 160,435 of 160,435 rows, drift 0.0 |
| **R3 — the deciding rule** | **FAIL** | **−0.00550, t −0.56** |
| R2 — the column-permuting placebo | FAIL | below all nine retrains (−0.0028 … +0.0087) |
| R4 — net of the 0.2225% round trip | FAIL | 0.0075 per rebalance below the baseline |
| R5 — the surprise beyond its own timing columns | FAIL | −0.00581, **t −2.81** |

**R5 is new in Pilot 3 and it was decisive.** The outcome-blind design inputs
showed `sue_age` and `sue_missing` persisting at +0.40 and +0.47 within-date
rank over 250 sessions, so an arm carrying ONLY those two columns was added
before the pre-registration was hashed. It reproduced **every** grade the SUE
arm moved — PIDILITIND, TATACONSUM and TRENT into WEAK — with no surprise in it
at all.

Without that arm, three new WEAK grades would have read as the earnings
surprise at work. **A within-date column permutation cannot catch this, because
it destroys identity and information together.** That is now a landmine: any
added feature carrying a persistent missingness flag or a days-since column
needs an arm that holds those columns WITHOUT the feature.

**The ingestion is the durable part of Pilot 3**, and it is reusable by
anything that needs point-in-time Indian fundamentals: EPS for 7,201 of 7,660
first disclosures since 2013, second-resolution dissemination times, old HTML
reconciled against net profit ÷ shares, 95.9% of XBRL EPS within 15% of their
own implied EPS, and splits taken from NSE after yfinance was measured wrong on
TECHM 2015 and BAJFINANCE 2016.

---

## 3. The unifying finding, and exactly how far it goes

Pilot 3 is the informative one, because it is the only pilot where the
measurement can be checked against something the market visibly did.

A post-hoc, descriptive check (`tools/stage1c_announcement_check.py`) scored
each SUE against its own ANNOUNCEMENT return — last close before the filing was
public, to the close of the first session it was usable, net of the
equal-weighted panel, with every window spanning a bonus or split dropped:

| SUE quintile | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| mean announcement return | −0.36% | −0.28% | +0.05% | +0.27% | +0.45% |

- 2,805 announcements. Per reporting season the Spearman correlation averages
  **+0.095 over 40 seasons at t +5.08**, positive in 31 of 40, monotone by
  quintile, **+0.81% top minus bottom in about two sessions**.
- Over the following 30 sessions the same quantity predicts **nothing**: among
  names carrying a live event, `sue_evt`'s own per-date IC is **−0.021**.
- The raw PEAD sort book is negative before costs and **−0.69% per rebalance
  net, t −1.55**.

**So the three nulls can be told apart:**

| pilot | what it establishes |
|---|---|
| reversal | price alone carries nothing at this horizon |
| delivery % | new data, and no measurable reaction to test |
| **SUE** | **new data the market DOES react to — correctly measured — and prices within about two sessions** |

**The reading this suggests: on these 84 names, information from public, dated
sources is priced within days, and a 30-session target cannot see it by
construction.** That would explain all three pilots at once, and it explains
why a fourth source tested the same way should be expected to repeat it.

### What that reading is NOT

**It is a hypothesis suggested by one diagnostic on one feature. It is not a
proven property of this panel.** Stated plainly, because this project has twice
promoted a directional reading from a small number of cells and had to retract
it:

- The decay is measured on **one** feature. Reversal and delivery % have no
  announcement-return analogue, so nothing has confirmed the same shape for
  them — delivery % has no event to date in the first place.
- The announcement check is **post hoc and descriptive**. It was written after
  the run, decides nothing, and was never pre-registered.
- "Priced within ~2 sessions" is read off the median gap between disclosure and
  the first usable session. It is a statement about where the return LANDS, not
  a measurement of how fast information decays.
- **Nothing here has yet been measured at any horizon other than 30 sessions.**
  The claim that a shorter target would see what the 30-session one cannot is,
  so far, an inference from a decay curve with two points on it.

**The horizon sweep is the direct test of it, and its outcome is genuinely
open.** If the reading is right, something should appear at 5 or 10 sessions
and fade toward 30. If nothing appears at any horizon, the reading is wrong —
or at least incomplete — and the honest conclusion moves somewhere less
comfortable. That is written down here, before the sweep runs, so neither
outcome can be narrated as the expected one afterwards.

---

## 4. Why the horizon sweep comes next, and not a fourth data source

Bulk and block deals — the one originally-planned Stage 1 source never
attempted — is **deprioritised, not abandoned.** The reasoning:

1. **The three nulls now agree on a mechanism, and the mechanism is testable
   on data already in hand.** A fourth source tests the same hypothesis a
   fourth time at the same horizon; the sweep tests the explanation for why
   the first three failed. It costs no new scraper, no new access question and
   no new parser.
2. **The one pilot that found real information found it fully priced.** SUE is
   the most informative public signal available on this universe — a
   pre-scheduled, legally-timestamped disclosure — and it survives about two
   sessions. Bulk/block deals are disclosed on the same SEBI clock, to the same
   audience, in the same market. There is no reason to expect them to outlive
   an earnings surprise, and every reason built on the §3 reading to expect
   them not to.
3. **A fourth null at 30 sessions would not distinguish the two explanations
   that matter** — "this panel carries no gradeable public information" versus
   "30 sessions is the wrong window". The sweep does distinguish them, which is
   the only thing that changes what to do next.
4. **The horizon is the one axis never varied.** Every number this project has
   produced, across six phases and four evidence-grading stages, is at 30
   sessions. That is a large untested assumption to leave standing while adding
   a fifth feature under it.
5. **It is cheap and exact.** `panel.retarget_horizon` rebuilds the label from
   the price-series identity, so no data is refetched and no approximation
   enters.

**What would bring bulk/block deals back.** If the sweep shows the panel
carries cross-sectional information at a shorter horizon, then the question
"which sources feed it" reopens immediately, and bulk/block deals are the next
one to try — they are flow rather than fundamentals, so they are the least
correlated with what Pilots 2 and 3 already tested.

---

## 5. What Stage 1 leaves on the record

**Method, now standing:**

- **Permute the added COLUMNS within each date and retrain. Never shuffle
  predictions.** (Pilot 1's correction, Pilot 2's confirmation.)
- **Any added feature with a persistent missingness flag or a days-since column
  needs an arm carrying those columns WITHOUT it.** (Pilot 3's R5.)
- **The null band for an added feature widens with the persistence of what is
  added.** Pilot 3's nine-draw placebo spread (−0.0028 … +0.0087) is roughly
  three times Pilot 2's (−0.0036 … +0.0008), on three added columns instead of
  two. A future pilot should expect its R2 bar to move the same way.
- **R3 — the paired per-date cross-sectional IC with a Driscoll-Kraay SE — is
  the deciding rule.** All three pilots produced grade movements that R3
  refused, and in every case the placebo agreed with R3.

**Open, and recorded rather than fixed:**

- The four phantom 2026 holiday sessions are still in `panel_cache.parquet`.
  They do not touch Pilot 2's verdict and did not reach any fitted model, but
  the fix belongs in ingestion and redefines labels near those dates, so it
  needs a `MODEL_VERSION` decision.
- 2013–2017 EPS coverage is 73–87%, so the early folds carry more imputed rows.
- SIEMENS, TMPV, HINDUNILVR and VEDL lose their SUE for about two years after
  their 2025–2026 demergers.
- Every SE in Stage 1 is likely too small: Politis-White puts this panel's
  dependence at 35.8–62.5 sessions against a block of 30.

**Assets built, which outlive the nulls:** a 2,436-session delivery-% archive,
a 7,201-quarter point-in-time EPS archive with second-resolution disclosure
times, and a four-rule harness that can test any new column on the same folds
and the same floors as everything else in this project.

**Next: P6, the horizon sweep.** Pre-registered at
`docs/p6-preregistration.md`.

---

## Addendum — 2026-09-21: re-run under the new conditions. The asterisk is removed.

**Pre-registered** in `docs/stage1-rerun-preregistration.md` (sha256
`2469804d41f9c4918212d99c1c889eef4cca1682892dd06ae6d7013011369a72`, LF, fixed
before any arm ran), **run by** `tools/stage1_rerun.py`. The correction notice at
the top of this document said the verdicts were measured on a model that emitted
~8 distinct predictions per date across 84 names, and so could partly be a fact
about resolution rather than about the features. That is now tested, not argued.

**The conditions changed; the pilots did not.** Same features, same ingestion,
same construction (imported unchanged from the three pilot tools), same h=30,
pooled × MAE, no ticker feature. What changed: the within-date standardised
label, `XGB_THREADS` = 2, the locked libraries, and the calendar-clean panel
(`panel_cache_clean.parquet`, the four 2026 phantom sessions removed). Windows,
the platform the stored baseline was produced on.

**S1 held three times.** The baseline re-run on each pilot's panel reproduces
the stored post-fix baseline (`baseline_clean_oos.npz`) at drift **exactly
0.0**, 160,104 of 160,104 rows.

**The resolution confound is gone in every arm, measured.**

| | old label (as run) | new label |
|---|---|---|
| median distinct predictions per date, 84 names | **8 - 12** (min 2-4) | **84** (min 83-84) |
| constant (ticker, fold) cells of 420 | 3 - 9 | **0** |

**All three stay null on R3.** Hypothesis arms, paired per-date cross-sectional
IC gain over the baseline, Driscoll-Kraay SE at 30 lags:

| pilot | hypothesis arm | old R3 | **new R3** | new R2 (placebos beaten) | verdict |
|---|---|---|---|---|---|
| 1 reversal | residual reversal, k = 1/5/10/20 | +0.0002, t +0.07 | **−0.00186, t −0.84** | 9 of 9 | NOT SIGNAL |
| 2 delivery % | abnormal delivery | −0.00066, t −0.19 | **−0.00343, t −2.06** | 4 of 9 | NOT SIGNAL |
| 3 SUE | SRW surprise, event window | −0.0055, t −0.56 | **−0.00061, t −0.16** | 9 of 9 | NOT SIGNAL |

The other arms: raw reversal −0.00252 (t −1.66); delivery level +0.00544
(t +1.61); SUE timing-only −0.00189 (t −0.51). **R5, the surprise over timing
alone: +0.00128, t +0.36** — positive, which was pre-registered as a FAILED
prediction (P4 expected it to fail) and does not matter, because R3 fails
first. R4, descriptive: the SUE book trails the baseline book by 0.30% per
rebalance net of the round trip (t −1.33).

**Grades moved, and nothing else did.** Baselines and the reversal arms grade
0 / 0 / 84. Delivery arms each lift one name to WEAK (WIPRO, CIPLA); the SUE arm
three (CIPLA, MOTHERSON, WIPRO) — and **the timing-only arm, which carries no
surprise at all, lifts six, one of them STRONG.** That is Pilot 3's
identity-fingerprint finding again, on the new label: the grade movement comes
from `sue_age` and `sue_missing`, not from the surprise.

**Read against the placebo, with one thing the old runs could not show.**
Every one of the 27 placebo retrains LOST cross-sectional IC against the
baseline (−0.0016 to −0.0043). On the new label, adding two to four columns of
within-date noise to the pooled model costs it ~0.003 of IC. So R2 now separates
"less harmful than noise" from "worse than noise": reversal and SUE beat all nine
of their placebos and are STILL below the baseline. Delivery's abnormal arm is
inside its placebo band (4 of 9). **Its t −2.06 is a NEGATIVE deviation inside
the placebo range, and at ~150 trials on this panel a best |t| near 3.2 is
expected from noise.** Not a finding in either direction.

**The pre-registered predictions:** P1 (S1 exact) held; P2 (every hypothesis
arm fails R3) held; P3 held; **P4 (SUE fails R5) FAILED** — R5 is +0.00128;
P5 (84 distinct per date, 0 constant cells in every arm) held.

**Conclusion: the three Stage 1 nulls were the features.** Remove the
resolution asterisk from this document. The track stays closed on the terms of
§5; nothing in this addendum reopens it. Detail in
`docs/pre-universe-hygiene-findings.md` §3.
