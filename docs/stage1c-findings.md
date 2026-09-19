# Stage 1, Pilot 3 — SRW SUE (post-earnings drift): findings

**Run 2026-09-19 23:38 to 2026-09-20 01:19 IST, against
`docs/stage1c-preregistration.md`.** The pre-registration's sha256 is
`02bed013…cef9`; it was written and hashed at 23:34:12, before the run. The
results cache hashes to `7d6ebaf7…fdd8`.

The branch is `stage1c-sue`: a new external source, so a short-lived branch.
The code is `pipeline/earnings.py`, `tools/backfill_results.py`,
`tools/stage1c_sue.py` and `tools/stage1c_announcement_check.py`. The run used
B = 1000, 10 trials, nine retrained placebos and DK lags of 30.

## The answer

| rule | verdict | what decided it |
|---|---|---|
| S1 — the baseline reproduces | **PASS** | 160,435 of 160,435 rows, max drift **0.0e+00** |
| **R3 — THE DECIDING RULE** | **FAIL** | sue − baseline cross-sectional IC **−0.00550, t −0.56** |
| R2 — the corrected placebo | **FAIL** | the arm's gain is BELOW all nine permuted-column retrains (−0.0028 to +0.0087) |
| R4 — net of cost | **FAIL** | the arm's long-short book earns 0.0075 per rebalance LESS than the baseline's, net (t −1.58) |
| R5 — the surprise itself | **FAIL** | sue − timing −0.00581, **t −2.81**: adding the surprise to timing and coverage makes the ordering worse |
| **SIGNAL** | **NO** | |

**Earnings surprise is measured correctly, the market prices it on the day,
and nothing is left to drift into the next 30 sessions.** The honest read
is the briefing's large-cap null. Post-earnings drift does not survive in a
universe this liquid, gross or net of cost. It is not an implementation
issue: the instrument check below is what separates the two readings.

## The instrument check: the market does react to this SUE

This is descriptive and post hoc; it was added after the run and decides
nothing (`tools/stage1c_announcement_check.py`). A drift null from a SUE the
market ignores would be uninterpretable. So the check scores each SUE
against its own ANNOUNCEMENT return:
- the move from the last close before the filing was public to the close of
  the first session it was usable (median 2 sessions);
- taken net of the equal-weighted panel;
- with every window spanning a bonus or split dropped.

| SUE quintile | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| mean announcement return | −0.36% | −0.28% | +0.05% | +0.27% | +0.45% |

- **The measure is right.** 2,805 announcements. Spearman(SUE, announcement
  return) is **+0.093**, and per reporting season it averages +0.095 over 40
  seasons at **t +5.08**, positive in 31 of 40. The quintiles are monotone, at
  **+0.81%** top minus bottom in about two sessions.
- **So the SUE carries the information it should, and the market moves on
  it when it is released.** Only 2016 disagrees (−0.21), and 2016 has the
  fewest defined SUEs and the thinnest old-format coverage.
- **After that move, the same quantity predicts nothing, or slightly the
  reverse.** Among names with a live event, `sue_evt`'s own per-date IC against
  the next 30 sessions is **−0.021 (t −1.30)**. The raw PEAD book below is
  negative too.

## The ingestion

- **Source: NSE's own filings,** with no library and no paid feed. The
  pre-registration records why.
  - Legacy results list: 2005 to the December-2024 quarter, second-resolution
    dissemination times.
  - Integrated Filing list: from March 2025.
  - NSE's corporate actions.
  - 252 of 252 list calls OK.
  - 7,660 first disclosures from March 2013; 7,385 documents fetched, 275
    old HTML pages 404.
  - Basic EPS for **7,201**. Coverage is 73-87% for 2013-2017 and 97.6-100%
    from 2018.
- **Three silent traps in NSE's own data, each now pinned by a test.**
  1. **The bank template's old HTML is misaligned.** Every value from "Face
     Value" down sits one row below its label, so read by label the "Basic
     EPS" is the diluted figure. Every old-format EPS must now reconcile with
     net profit ÷ (paid-up ÷ face value): 2,002 pages at shift 0, 2 at +1, and
     141 at neither, which yield no EPS.
  2. **Older XBRL references an undefined `OneD` context** (2,604 EPS read by
     NSE's convention). Wherever a file defines OneD, it is the quarter.
  3. **Some XBRL stamps the year-to-date `FourD` context with the quarter's
     dates** (ABB, Dec-2022: 14.41 quarter against 47.96 YTD). OneD wins
     there (1,017 EPS).

  Of the XBRL EPS, 95.9% agree with their own implied EPS within 15%.
- **Splits and bonuses: NSE is primary, and neither source is complete.**
  - NSE is right where yfinance is wrong on TECHM 2015 (×4; the EPS falls from
    29.74 to 8.50) and on BAJFINANCE 2016 (×10).
  - yfinance lacks the bonus half of BAJAJFINSV 2022 and BAJFINANCE 2025.
  - NSE lacks MOTHERSON's 2015 bonus, which yfinance supplies.
  - A combined NSE subject ("Bonus 1:1/Face Value Split …") must multiply both
    parts. My first parser did not, and the cross-check caught it.
- **Point-in-time.** A filing is usable at a session's close only if
  disseminated before 15:00 IST that trading day, using the LATER of NSE's two
  timestamps.
  - Inside the panel's span: 2,059 filings after 15:00, 692 intraday, 361 on
    non-trading days.
  - A quarter enters another's SUE only if disclosed first, and only as first
    disclosed.
  - The leakage test places filings at 14:00, at exactly 15:00:00 and at
    16:00 on the cut day, plus an old quarter disclosed after it. It catches
    all three timing mutants on its own.

## The four arms

| arm | per-date cs IC (DK SE) | reb t | STRONG / WEAK / INSUFF | tau2 | OOS MAE |
|---|---|---|---|---|---|
| (a) baseline | −0.00101 (0.00811) | −0.09 | 1 / 1 / 82 | 0.00435 | 0.08882 |
| (b) sue | −0.00651 (0.00670) | −0.74 | 0 / 4 / 80 | 0.00506 | 0.08871 |
| (c) naive, forward-filled | −0.00087 (0.00890) | −0.17 | 0 / 0 / 84 | 0.00311 | 0.08892 |
| (d) timing, no surprise | −0.00070 (0.00666) | −0.26 | 1 / 4 / 79 | 0.00556 | 0.08885 |

| paired comparison | Δ cs IC | DK SE | **t** |
|---|---|---|---|
| (b) sue − (a) **[R3]** | −0.00550 | 0.00976 | **−0.56** |
| (c) naive − (a) | +0.00014 | 0.00702 | +0.02 |
| (d) timing − (a) | +0.00031 | 0.00959 | +0.03 |
| (b) sue − (d) timing **[R5]** | −0.00581 | 0.00207 | **−2.81** |
| (b) sue − (c) naive | −0.00564 | 0.00968 | −0.58 |

**The grade movement is the identity channel, and arm (d) is what shows it.**
- Arm (b) moves PIDILITIND, TATACONSUM and TRENT into WEAK. So does arm (d),
  which has no surprise at all, only days-since and the missingness flag.
  Those two columns persist at +0.40 and +0.47 within-date rank over 250
  sessions.
- One of the nine noise retrains produced 5 WEAK.
- Without arm (d), three new WEAK grades would have looked like the
  surprise at work.

**R5's t of −2.81 is the one large statistic here, and it is negative.**
- Arms (b) and (d) differ by a single column, `sue_evt`. Their predictions
  are highly correlated, so the paired SE (0.00207) is a fifth of the others.
  A small, consistent difference therefore earns a large t.
- Its sign matches the negative live-name IC above: the model learned a
  positive surprise effect in training that does not hold out of sample.
- It is one statistic among 10+ on a panel carrying well over 130 trials,
  so it is read as "the surprise does not help", not as a tradeable reversal.

## Net of cost

These are long-short top-minus-bottom quintile books over the 64
non-overlapping rebalances, charged the 0.2225% round trip on name-by-name
turnover.

| book | gross / rebalance | net / rebalance | turnover |
|---|---|---|---|
| (a) baseline | +0.50% | +0.40% | 0.46 |
| (b) sue | −0.26% | −0.35% | 0.40 |
| raw PEAD sort on `sue_evt` alone | −0.52% (t −1.16) | **−0.69% (t −1.55)** | 0.78 |

**The briefing's open question, whether PEAD survives Indian costs, has its
answer on this universe: there is no PEAD to survive them.**
- The raw sort is negative before costs.
- The cost drag is about 0.17% per rebalance, because the book turns over
  78% each time: an event-driven book churns more than a model book.
- The annualised net Sharpe is −0.58.

## The corrected placebo

| seed | Δ vs (a) | t | STRONG | WEAK |
|---|---|---|---|---|
| 20260920 | +0.00674 | +0.66 | 2 | 2 |
| 20260921 | +0.00314 | +0.30 | 0 | 1 |
| 20260922 | −0.00169 | −0.17 | 0 | 0 |
| 20260923 | +0.00257 | +0.24 | 0 | 2 |
| 20260924 | +0.00102 | +0.10 | 0 | 2 |
| 20260925 | −0.00278 | −0.27 | 0 | 0 |
| 20260926 | +0.00252 | +0.25 | 1 | 1 |
| 20260927 | +0.00873 | +0.85 | 0 | 5 |
| 20260928 | +0.00578 | +0.56 | 0 | 2 |
| **arm (b)** | **−0.00550** | **−0.56** | **0** | **4** |

The placebo spread (−0.0028 to +0.0087) is roughly three times delivery's
(−0.0036 to +0.0008). Three added columns, two of them persistent, give the
retrained trees more to move with. **So the null band for an added feature
widens with the persistence of what is added.** A future pilot should expect
its R2 bar to move the same way.

## The predictions, scored

| prediction | held | note |
|---|---|---|
| P1 arm (a) reproduces | yes | drift 0.0 |
| P2 no signal: abs(Δ) < 0.005 and abs(t) < 2 | **no** | abs(Δ) 0.0055: a null, but a slightly larger and NEGATIVE one than predicted |
| P3 the raw PEAD book's net return has t < 2 | yes | t −1.55 |
| P4 `sue_evt`'s own abs(IC) < 0.02 | yes | −0.0057 (live names only: −0.021, descriptive) |
| P5 the placebo gains within abs(Δ) < 0.005 | **no** | max +0.0087; the null band is wider than predicted |

## Recorded, not fixed

- **2013-2017 EPS coverage is 73-87%.** Of that era's shortfall, 275 pages
  are 404s and 141 are unvalidated. SUE therefore defines 70-84% of panel
  rows in 2016-2020, against 90-98% after, so the early folds carry more
  imputed rows.
- **20 Integrated filings define `OneD` as a half-year** and are refused.
- **SIEMENS, TMPV, HINDUNILVR and VEDL** lose their SUE for about two years
  after their 2025-2026 demergers.
- **The 2026-09-14 daily run failed.** See CLAUDE.md; this is unrelated to
  this pilot.

## What this means for Stage 1

Three pilots, three nulls, and they can now be told apart:
- **reversal:** price only;
- **delivery %:** new data, no reaction to test;
- **SUE:** new data the market DOES react to, and prices at once.

The third is the most informative of the three. Its information is real and
measured correctly, and on these 84 names it is fully priced in about two
sessions. **At a 30-session horizon, what this panel can still learn from
public, dated information may simply be nothing.** A fourth data source
tested the same way is likely to repeat this. P6, the pre-registered horizon
sweep, asks the question this result points at: whether anything survives
at 5 or 10 sessions.
