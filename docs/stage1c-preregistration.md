# Stage 1, Pilot 3 — SRW SUE (post-earnings drift): pre-registration

**Written 2026-09-19, before any model trained on or was graded with these
columns.** It was written after the results backfill and after the
outcome-blind design inputs below, none of which read a target or a
prediction. Branch `stage1c-sue`, a short-lived exception to the
default-to-`main` policy, because this is a new external source: NSE's
quarterly results filings. There is one attributable change: an
earnings-surprise event feature, added to the pooled × MAE no-ticker model.

It is the third Stage 1 pilot. Reversal and delivery % were both null under
R3. Per the briefing's C2 reasoning, the surprise is measured against a
seasonal random walk, not analyst consensus. Consensus is thin or absent
below the large-cap tier, so a consensus surprise would be undefined for a
material share of the 84 names. SRW needs only reported EPS. The briefing
document itself was not available in this repository. Its sections A5, B2,
B3 and C2 are applied as the session prompt summarised them.

## The hypothesis, falsifiable

> A company's standardised unexpected earnings, live for 30 sessions after
> the result is public, carries CROSS-SECTIONAL information about the next 30
> sessions' return that the 15 technical `FACTORS` do not. If it does, adding
> it to pooled × MAE (no ticker) raises the paired per-date cross-sectional
> rank IC over the baseline at a Driscoll-Kraay t ≥ 2.0 (30 lags). That gain
> exceeds every one of nine retrained placebos with the SUE columns permuted
> within each date. And it survives the 0.2225% round-trip cost in a traded
> book.

## The data, as ingested and validated

- **Source: NSE's own filings,** read directly at no more than 3 requests a
  second (`pipeline/earnings.py`, `tools/backfill_results.py`).
  - The legacy results list, 2005 to the December-2024 quarter, has
    second-resolution dissemination times. "Old" filings link an HTML page;
    "New" ones link XBRL.
  - The Integrated Filing list covers March 2025 onward, with XBRL and an
    Original/Revised flag.
  - NSE's corporate actions supply bonuses, splits and demergers.
  - History is attached to the current symbol, so TMPV carries Tata Motors'
    filings and LTM carries LTI's.
- **Why not a library or a paid feed.**
  - BseIndiaApi-style wrappers and BSE's own announcement API cap a query at
    12 months and return announcement PDFs, not structured EPS. Every EPS
    would need PDF parsing.
  - Paid structured feeds (Prowess, Capitaline, CMOTS) are not available to
    this project, and a vendor's history cannot be re-audited against the
    exchange's own timestamps.
  - NSE's lists need only the cookies a listing page sets. The documents on
    nsearchives need none.
- **What the backfill fetched.**
  - List calls: 252 of 252 OK.
  - 11,170 filings listed.
  - 7,660 first disclosures from the quarter ended March 2013: 2,418 old
    HTML, 4,236 legacy XBRL, 1,006 Integrated XBRL.
  - 7,385 documents fetched; 275 answered 404, all old HTML pages.
  - Basic EPS parsed for 7,201.
  - Coverage by period-end year: 84.6%, 86.5%, 86.7%, 84.1% and 73.3% for
    2013-2017, and 97.6-100% from 2018.
- **The old HTML pages are validated, not trusted.**
  - In NSE's bank template every value from "Face Value" down sits one row
    below its label. So an old-format EPS is accepted only if it agrees within
    15% with net profit / (paid-up capital / face value), at a row shift of 0
    or +1.
  - 2,004 pages reconciled: 2,002 at shift 0 and 2 at +1. 141 reconciled at
    neither and yield no EPS.
- **XBRL.**
  - EPS comes from the quarter's own non-dimensional context.
  - Older instances reference an undefined `OneD`, NSE's convention for the
    current quarter. Wherever a file does define it, it is exactly the quarter.
    2,604 EPS came through this route.
  - Some files stamp the year-to-date `FourD` context with the quarter's
    dates. There OneD wins (1,017 EPS).
  - 20 Integrated filings define OneD as a half-year. They are refused.
  - Of 5,197 XBRL EPS, 5,151 carry the tags for an implied EPS, and 95.9% of
    those agree within 15%.
- **Splits and bonuses: NSE primary, yfinance filling what NSE lacks.**
  - 84 of 88 NSE bonuses and splits since 2012 match `corporate_actions`
    (yfinance) within 5 days and 1%.
  - Every disagreement was resolved against the EPS series itself. NSE is
    right on TECHM 2015 (×4, bonus 1:1 plus split 10→5; EPS fell from 29.74
    to 8.50) and on BAJFINANCE 2016 (×10). yfinance lacks the bonus half of
    BAJAJFINSV 2022 and BAJFINANCE 2025.
  - A yfinance split with no NSE action within 5 days is added. That is only
    MOTHERSON 2015-07-23, where no EPS is parsed anyway.
- **Structural breaks.**
  - NSE demergers: ABB 2019-12-20, HINDUNILVR 2025-12-05, ITC 2025-01-06,
    MOTHERSON 2022-01-14, RELIANCE 2023-07-20, SIEMENS 2025-04-07,
    TMPV 2025-10-14, VEDL 2026-04-30.
  - Mergers: HDFCBANK 2023-07-01, LTM 2022-11-14.

## The SUE, exactly

For each company, each basis (consolidated, standalone), and each quarter q
disclosed at time d_q:

- **E_k is basic EPS as FIRST disclosed**, never a later restatement.
- **Share basis.** E_k is restated to the share basis at d_q:
  E_k × S(d_k) / S(d_q), where S(x) is the cumulative share factor from NSE's
  bonuses and face-value splits with an ex-date on or before x.
- **Point-in-time.** A quarter k ≠ q enters only if d_k < d_q.
- **Seasonal difference.** D_j = E_j − E_(j−4), where j−4 is the same quarter
  a year earlier. D_j is void if a merger or demerger falls in
  (period_end_(j−4), period_end_j].
  - Mergers are HDFCBANK 2023-07-01 and LTM 2022-11-14 (hand-listed, since
    no action on the acquirer's symbol records them).
  - Demergers come from NSE's corporate actions.
- **SUE_q = D_q / sd(D_(q−1), …, D_(q−8)),** requiring at least 4 of the 8.
- **One announcement per (company, quarter).** It uses the consolidated SUE
  where one is defined, else the standalone SUE, disclosed when that filing
  was. If neither basis defines a SUE, the announcement still exists (its age
  resets) with the SUE missing.

## Point-in-time, and the mixed-frequency merge

- **Session grid.** The panel's own session grid, minus the four phantom
  2026 sessions. A phantom row carries the previous real session's state, and
  no announcement is usable on one.
- **Usable from.** A filing disseminated on a trading day before 15:00 IST,
  when NSE's closing-price window opens, is usable at that session's close.
  Otherwise it is usable at the next session.
- **The timestamp** is the LATER of NSE's two records (broadcast,
  dissemination). So a result is never treated as public earlier than NSE
  says it was.

| column | definition |
|---|---|
| `sue_evt` | the latest announcement's SUE while fewer than 30 sessions have passed since it became usable, 0.0 after; missing if that announcement has no SUE or there has been none |
| `sue_age` | sessions since the latest announcement became usable, capped at 63; missing if none |
| `sue_missing` | 1 where the latest announcement has no SUE, or none exists |
| `sue_ffill` | the latest DEFINED SUE carried forward with no window: the naive construction, arm (c) only |

- **Imputation.** Missing `sue_evt`, `sue_age` and `sue_ffill` take that
  date's cross-sectional median. The missingness itself is carried by
  `sue_missing`.
- **Nothing is forward-filled** except in `sue_ffill`, which exists to be
  compared against.
- **Standardisation.** All columns are z-scored within date like every other
  pooled feature.

## Outcome-blind design inputs

All measured on the panel's grid, reading no target and no prediction
(`tools/stage1c_sue.py --design-only`):

- **Announcements:** 4,101, of which 3,201 carry a SUE (2,263 consolidated,
  938 standalone) and 900 have none.
- **Timing.** Of those disseminated inside the panel's span, 2,059 were after
  15:00 on a trading day, 692 intraday, and 361 on a non-trading day.
- **Share of panel rows with a defined SUE:** 2016 0.84, 2017 0.83,
  2018 0.78, 2019 0.70, 2020 0.80, 2021 0.90, 2022 0.97, 2023 0.90,
  2024 0.97, 2025 0.98, 2026 0.96.
- **Rows with a LIVE event** (fewer than 30 sessions old): 41%.
- **SUE distribution,** quantiles 1/10/50/90/99%: −5.39, −1.18, +0.46,
  +2.65, +9.61. The within-date z-score clips at 3.
- **Within-date rank persistence at 250 sessions:** `sue_evt` +0.017,
  `sue_ffill` +0.113, **`sue_age` +0.397, `sue_missing` +0.469.**

**The last two are the §7 landmine's shape, and that is why arm (d) exists.**
- **Why they persist.** Companies report at a consistent point in each
  season, and the names with thin filing histories are the same names year
  after year. So days-since and the missingness flag are partly a per-company
  fingerprint, as persistent as delivery's raw level (+0.447).
- **Why the placebo cannot see it.** A tree can use a fingerprint to
  recognise the company. The within-date permutation destroys identity as
  well as information, so R2 cannot separate the two. Arm (d) carries both
  columns WITHOUT the surprise, and R5 requires the surprise to add over it.

## The arms

All use `tools/stage2b_pooled.run_arm` on the frozen panel: MAE objective, no
ticker, 10 trials, 5 purged folds, `min_train` 500, and horizon, purge and
embargo all 30.

| arm | features |
|---|---|
| (a) baseline | `FACTORS` — REUSED from `stage2b_pooled_oos.npz` (sha256 of arrays `67cc72be…12e1`) |
| (b) sue | `FACTORS` + `sue_evt`, `sue_age`, `sue_missing` — **the hypothesis** |
| (c) naive | `FACTORS` + `sue_ffill` — the forward-filled construction, descriptive |
| (d) timing | `FACTORS` + `sue_age`, `sue_missing` — everything but the surprise: the identity-risk arm |

**S1 — stop condition.** Arm (a) is re-run and must reproduce the stored
predictions at drift ≤ 1e-9 on identical rows. If it does not, stop.

## The decision rules

- **R3 — THE DECIDING RULE.** The paired per-date cross-sectional rank IC,
  (b) − (a), on identical rows, with the Driscoll-Kraay SE at **30 lags**.
  R3 holds iff **t ≥ +2.0**. No other check substitutes for it.
- **R2 — THE CORRECTED PLACEBO.** The three event columns are permuted
  JOINTLY within each date, and arm (b) is RETRAINED nine times, with seeds
  20260920 to 20260928. Predictions are NEVER shuffled. R2 holds iff arm
  (b)'s paired gain over (a) EXCEEDS ALL NINE placebo gains.
- **R4 — NET OF COST.** Each arm's long-short top-minus-bottom quintile book
  is traded on its own predictions over the 64 non-overlapping rebalances.
  - The cost is 0.2225% round trip on actual name-by-name turnover
    (`portfolio.simulate`).
  - R4 holds iff (b)'s mean net return per rebalance EXCEEDS (a)'s.
  - An IC gain that the arm's extra turnover eats is not a result.
- **R5 — THE SURPRISE ITSELF.** The paired per-date cross-sectional IC,
  (b) − (d). R5 holds iff it is positive. Without it, a gain could be
  timing or coverage acting as company identity.
- **SIGNAL iff R3 and R2 and R4 and R5.** Anything less is reported as
  not-signal, naming the clause that failed.
- **R6.** If SIGNAL, arms (a) and (b) are re-run at `min_train` ∈ {380, 420,
  460, 500, 540, 580}. A lone spike is not a result.

**Descriptive only; they decide nothing:**
- **The raw PEAD book.** A long-short quintile sort on `sue_evt` alone, over
  arm (a)'s rows, gross and net of the same cost. This is the direct answer to
  the briefing's open question, whether PEAD survives Indian costs. A pure
  sort is not what the pipeline trades, so it cannot make SIGNAL.
- `grade_panel_v3` grades for the three arms (B = 1000, block 30).
- Arms (c) and (d) against (a), and (b) against (c).
- Each SUE column's own per-date IC (DK, 30 lags).
- `sue_evt`'s IC among the names with a live event only (at least 10 on a
  date).

## Predictions, scored in the findings whichever way they fall

- **P1.** Arm (a) reproduces (S1 passes).
- **P2.** No signal: arm (b)'s gain over (a) has abs(Δ) < 0.005 and
  abs(t) < 2.
- **P3.** The raw PEAD book's net-of-cost return per rebalance has t < 2.
- **P4.** `sue_evt`'s own abs(IC) < 0.02.
- **P5.** All nine placebo gains lie within abs(Δ) < 0.005.

These predict a null for a stated reason. PEAD is documented as weakest in
the largest, most liquid names, and these 84 are the top of the NSE.

## Caveats carried in

- **Every SE is likely too small** (Politis-White 35.8-62.5 sessions), and
  trials keep accumulating: the reversal and delivery pilots' arms, draws and
  sweeps, plus this pilot's, on top of ~131 prior configurations.
- **The timestamp is the results FILING's, not the board-meeting outcome
  announcement.** A company can announce by PDF minutes before its structured
  filing. The filing is the later of the two, so this can only delay a
  feature, never leak one.
- **The consolidated-then-standalone rule mixes bases across companies**, and
  within a company across the 2019 start of mandatory quarterly consolidated
  results. Each SUE is internally consistent: it never mixes bases within its
  own differences.
- **SRW SUE ignores analysts entirely,** by design. A null here says nothing
  about consensus surprise in the large-cap tier where it is defined.

## Non-goals

- No database write. The cache is a local, gitignored `.npz`.
- No production wiring, no Render redeploy, and no ticker identity.
- Nothing is wired into `evidence_panel.py` or `evidence_shrinkage.py`.
- Claude runs no `git add` / `commit` / `push`.
