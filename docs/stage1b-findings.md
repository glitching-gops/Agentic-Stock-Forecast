# Stage 1, Pilot 2 — NSE delivery %: findings

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

**Run 2026-09-13 21:11-21:38 IST, against `docs/stage1b-preregistration.md`
(sha256 `d61f707c…f8ce`, written 21:07:10, before the run).** Short-lived
branch `stage1b-delivery`, the first external scraper under the default-to-
`main` policy. `pipeline/delivery.py`, `tools/backfill_delivery.py`,
`tools/stage1b_delivery.py`. B = 1000, 10 trials, nine retrained placebos, DK
lags 30. The delivery cache sha256 begins `f8dbe738cefb37af`.

## The answer

| rule | verdict | what decided it |
|---|---|---|
| S1 — the baseline reproduces | **PASS** | 160,435 of 160,435 rows, max drift **0.0e+00** |
| **R3 — THE DECIDING RULE** | **FAIL** | abnormal − baseline cross-sectional IC **−0.00066, t −0.19** |
| R2 — the corrected placebo | **FAIL** | the arm's gain beats 8 of 9 permuted-column retrains, not all 9 (max +0.00082) |
| **SIGNAL** | **NO** | |

**Delivery % is genuinely new information, and on this panel it predicts
nothing.** The honest read is "no signal here either", NOT "the ingestion needs
fixing before this is a fair test", for four reasons:
- **The ingestion is validated independently of any outcome.** On both UDiFF-era
  dates checked, the deliverable quantity equals `sec_bhavdata_full`'s on 100% of
  EQ symbols.
- **Coverage is 97.7-100% of panel rows** in every year.
- **Every file is checked** against the date it claims to cover.
- **Renames are mapped by date.** 13,047 rows came from nine tickers' former
  symbols.

Each delivery column's own cross-sectional IC is indistinguishable from zero.

## The ingestion, which is the durable part

- **Source.** NSE's daily `MTO_<ddmmyyyy>.DAT` (Security-Wise Delivery
  Position), read straight from nsearchives.nseindia.com at no more than 3
  requests a second, with `requests` (already a dependency).
- **Why not a library.** jugaad-data 0.35.5 and `nse` 4.0.1 are both
  maintained, but both fetch `sec_bhavdata_full`, which the archive answers
  with 404 before mid-2024. jugaad's other route is the website API, which needs
  landing-page cookies, and the landing page answered this machine's scripted
  client with 403. `nse` would also pin `httpx==0.28.1`.
- **UDiFF (circular 62424).** The July-2024 UDiFF CM bhavcopy carries no
  delivery column. The MTO format is unchanged from 2016 to 2026.
  `tools/backfill_delivery.py --validate` checks all of this on demand:

| date | MTO EQ | sec_full EQ | UDiFF EQ | deliv qty identical | % gap | UDiFF delivery cols | universe present |
|---|---|---|---|---|---|---|---|
| 2026-09-11 | 2,636 | 2,637 | 2,637 | 100.00% | 0.000 | none | 84/84 |
| 2024-07-08 | 1,909 | 1,909 | 1,909 | 100.00% | 0.000 | none | 82/84 * |

\* The two not found are TMPV and LTM, which traded as TATAMOTORS and LTIM on
that date. The check looks for today's symbols; the ingestion maps by date.

- **Backfill.** 2,440 requests, 2,436 `ok` and 4 `not_found`. No blocks, no
  errors, and the rate-based abort never came close to firing.
- **Point-in-time.** The file for session T is published after T's close, so
  every feature at t is built from t-1 and earlier. A leakage test corrupts
  every value from a date onward and requires every feature up to it unchanged.
  It also requires the feature one session later to move, so the lag is exactly
  one.
- **Missing values** stay missing and are never forward-filled. That includes
  the ADANIPOWER, ADANIENSOL and CGPOWER gaps (91.5-92.3% coverage), where on
  sampled dates the MTO file has no row for the symbol in any series. Their
  cause is not verified: the 2021-2024 gaps predate `sec_bhavdata_full`'s reach.

## The three arms

| arm | mu_hat | z | tau2 | STRONG / WEAK / INSUFF | RW | per-date cs IC (DK SE) | reb t | OOS MAE |
|---|---|---|---|---|---|---|---|---|
| (a) baseline | −0.01727 | −1.24 | 0.00435 | 1 / 1 / 82 | 1 | −0.00101 (0.00811) | −0.09 | 0.08882 |
| (b) abnormal | −0.02485 | −1.78 | 0.00461 | 2 / 0 / 82 | 2 | −0.00167 (0.00865) | +0.05 | 0.08866 |
| (c) level | −0.03626 | **−2.64** | 0.00305 | 1 / 0 / 83 | 1 | −0.00052 (0.00885) | −0.24 | 0.08866 |

| paired comparison | Δ cs IC | DK SE (30 lags) | **t** |
|---|---|---|---|
| (b) abnormal − (a) | −0.00066 | 0.00355 | **−0.19** |
| (c) level − (a) | +0.00049 | 0.00622 | **+0.08** |
| (b) abnormal − (c) level | −0.00115 | 0.00578 | −0.20 |

Each delivery column's own per-date cross-sectional IC against the target
(DK, 30 lags): level **+0.0095 (t +0.58)**, abnormal **−0.0004 (t −0.07)**,
5-session abnormal **−0.0015 (t −0.17)**.

Both delivery arms shave OOS MAE by 0.00016 (−0.2%). That is within noise, and
MAE is not the measure being judged.

## THE CORRECTED PLACEBO, AND WHAT IT SAYS ABOUT THE REVERSAL PILOT

| seed | Δ cs IC vs (a) | t | STRONG | WEAK | RW | tau2 |
|---|---|---|---|---|---|---|
| 20260914 | −0.00096 | −0.27 | 2 | 0 | 2 | 0.00466 |
| 20260915 | −0.00073 | −0.21 | 2 | 0 | 2 | 0.00472 |
| 20260916 | −0.00104 | −0.30 | 2 | 0 | 2 | 0.00466 |
| 20260917 | −0.00356 | −0.97 | 0 | 2 | 0 | 0.00357 |
| 20260918 | +0.00082 | +0.21 | 2 | 0 | 2 | 0.00488 |
| 20260919 | −0.00356 | −0.97 | 0 | 2 | 0 | 0.00357 |
| 20260920 | −0.00356 | −0.97 | 0 | 2 | 0 | 0.00357 |
| 20260921 | −0.00101 | −0.29 | 2 | 0 | 2 | 0.00469 |
| 20260922 | −0.00086 | −0.24 | 2 | 0 | 2 | 0.00466 |
| **arm (b)** | **−0.00066** | **−0.19** | **2** | 0 | **2** | 0.00461 |

**Arm (b) is indistinguishable from its own noise-column retrains, on every
statistic.** Its STRONG count, Romano-Wolf count and tau2 match six of the
nine placebos exactly or nearly.

**That is the verdict the reversal pilot could not reach.** There, ABB.NS
moved from INSUFFICIENT to STRONG and NESTLEIND.NS fell out of WEAK in both
reversal arms. That cleared a prediction-shuffle placebo, and only R3 stopped it
being read as signal. Here the abnormal-delivery arm makes the IDENTICAL move,
ABB.NS in and NESTLEIND.NS out, and so do six of nine retrains whose two added
columns carry no information at all. **So that grade movement is what
retraining the pooled model with any two extra columns produces.** It carries
no information about the columns added. A placebo that permutes the new
columns and retrains can see this; a placebo that shuffles predictions cannot,
because it never retrains. Pilot 1's findings already named its placebo as the
wrong null; this is the measurement that shows how wrong.

Three draws (20260917, 20260919, 20260920) returned identical figures to one
another. That is consistent with the seeded search settling on the same
configuration and the trees never splitting on the permuted columns. It is
recorded, not investigated: placebo predictions are not saved.

## The predictions, scored

| prediction | held | note |
|---|---|---|
| P1 arm (a) reproduces | yes | drift 0.0 |
| P2 no signal | yes | −0.00066, t −0.19 |
| P3 the level arm moves more grades than the abnormal arm | **no** | the level arm moved none, and lost NESTLEIND |
| P4 every column's own abs(IC) < 0.02 | yes | max +0.0095 |
| P5 the placebo gains within abs(Δ) < 0.005 | yes | −0.0036 to +0.0008 |

The level arm's mu_hat of −0.0363 (z −2.64) is the most negative of the three.
The persistent per-company level made the average ticker's demeaned ranking
WORSE, not better. That is the identity channel showing up as harm, not help.

## Recorded, not fixed

- **The panel carries four phantom sessions.** On 2026-01-15, 05-01, 05-28 and
  06-26 every close equals the previous session's, and NSE has no delivery file
  for those dates. So a row-stepped 30-session label that spans one measures 29
  real sessions. It is now a CLAUDE.md landmine. The fix belongs in ingestion
  and needs a `MODEL_VERSION` decision. **Its effect on THIS verdict was
  measured on 2026-09-19 and is nil; see the addendum at the end.**
- **MTO's traded-quantity denominator shifted after mid-2024.** It was 0.889 of
  `sec_bhavdata_full`'s on 2026-09-11 and 1.000 on 2024-07-08. The
  abnormal transform absorbs a slow level shift; the raw level may not be
  comparable across it.
- **Every SE here is likely too small** (Politis-White 35.8-62.5 sessions), and
  trials keep accumulating.

## What this means for Stage 1

Two pilots, two clean nulls:
- the price-only information set, exhausted by the reversal pilot;
- the first genuinely new data source, delivery %.

Neither moves what a book earns at a 30-session horizon on this 84-name panel.
Reaching a third source (SUE, bulk/block deals, analyst revisions) would be the
same test again, and should use this pilot's harness unchanged: the stored
baseline's S1, the column-permutation placebo, and R3 deciding.

The more useful question may now be the horizon itself. P6, the 5/10/20/30-
session sweep, is already pre-registered and unrun, and delivery % is a far
more natural short-horizon quantity than a 30-session one.

## Addendum, 2026-09-19: the phantom sessions do not touch the verdict

All four phantom sessions (2026-01-15, 05-01, 05-28, 06-26) fall inside fold
4's scored test window, so the verdict above was provisional until this was
checked. The check is `tools/stage1b_phantom_check.py`. Its rule was written
and hashed before it ran (sha256 `6ba71d22…9a84f`): the verdict changes iff
R3's t reaches +2.0 once the phantom sessions are removed.

**No phantom session reaches any fitted model.** Every fold's training
labels end by 2025-01-03, eleven months before the first phantom:

| fold | train last | train labels reach | test window |
|---|---|---|---|
| 0 | 2018-08-02 | 2018-09-18 | 2018-11-05 → 2020-06-09 |
| 1 | 2020-03-06 | 2020-04-24 | 2020-06-10 → 2021-12-27 |
| 2 | 2021-09-29 | 2021-11-12 | 2021-12-28 → 2023-07-20 |
| 3 | 2023-04-25 | 2023-06-07 | 2023-07-21 → 2025-02-13 |
| 4 | 2024-11-21 | 2025-01-03 | 2025-02-14 → 2026-09-07 |

So the defect can reach only the SCORING of fold 4: 8,563 scored rows on 102
dates have a label window that spans a phantom session, or sit on one. On
those rows the correctly counted label moves by a median of 0.0102.

| variant | scored dates | (b) abnormal − (a) | DK SE | **t** | (c) level − (a) | t |
|---|---|---|---|---|---|---|
| V0, as run | 1,910 | −0.00066 | 0.00355 | **−0.19** | +0.00049 | +0.08 |
| V1, labels recounted over 30 real sessions, phantom rows dropped | 1,906 | −0.00066 | 0.00356 | **−0.19** | +0.00051 | +0.08 |
| V2, truncated before 2025-12-03 (no label or lookback can touch one) | 1,748 | −0.00071 | 0.00388 | **−0.18** | −0.00018 | −0.03 |

**The verdict stands, and is now final: R3 fails under every variant.**

- **The label recount is checked against the stored labels first.** It
  reproduces them on all 151,872 untouched rows, to within 5.0e-06, which is
  the precision of a close stored to three decimals.
- **The spec's 1e-9 tolerance was mis-set.** It fired the check's own stop
  condition before any R3 figure was computed. It was widened to 1e-5, which
  is recorded here as a deviation from the written spec.
- **V2 is exact, not an approximation.** Because no model trained on a
  phantom session, the stored predictions for rows before 2025-12-03 are
  exactly what a corrected panel would produce.
- **R2 was not re-scored.** The placebo predictions were never saved, and R3
  decides.
