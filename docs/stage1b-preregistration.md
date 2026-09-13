# Stage 1, Pilot 2 — NSE delivery %: pre-registration

**Written 2026-09-13, before any model trained on or was graded with these
columns.** It was written after the delivery backfill and after the
OUTCOME-BLIND design inputs below, which read no target and no prediction.
Branch `stage1b-delivery`, a short-lived exception to the default-to-`main`
policy, because this is the project's first scrape of a new external source.
One attributable change: yesterday's NSE delivery percentage, in two transforms
and as a raw level, added to the pooled × MAE, no-ticker model.

A separate track from Phase 0-6 and from the Stage 0-0c evidence track.

## The hypothesis, falsifiable

> Abnormal delivery, meaning yesterday's delivery % relative to its own
> trailing 60-session mean, carries CROSS-SECTIONAL information about the next
> 30 sessions' return that the 15 technical `FACTORS` do not. If it does, then
> adding it to pooled × MAE (no ticker) raises the paired per-date
> cross-sectional rank IC over the baseline at a Driscoll-Kraay t ≥ 2.0 (30
> lags), and that gain exceeds every one of nine retrained placebos in which the
> delivery columns are permuted within each date.

Delivery % is genuinely NEW information, not another transform of the closes
the reversal pilot exhausted. It is the share of traded quantity that settled
by delivery rather than being squared off intraday, and it is published by the
exchange, not derived from price.

## The data, as ingested and validated

- **Source: NSE's daily `MTO_<ddmmyyyy>.DAT`,** the Security-Wise Delivery
  Position, fetched directly from nsearchives.nseindia.com, throttled to at most
  3 requests a second. No new dependency.
  - Both maintained libraries were read. jugaad-data 0.35.5 and `nse` 4.0.1 each
    fetch `sec_bhavdata_full`, which the archive holds only from mid-2024 (404
    for 2016 and 2019), so neither covers this panel.
  - The archive serves MTO files without cookies. www.nseindia.com answers a
    scripted client with 403, so a 403 triggers one cookie warm-up and a retry,
    and is otherwise recorded as BLOCKED.
- **UDiFF (NSE circular 62424, July 2024).** The UDiFF CM bhavcopy carries NO
  delivery column. The MTO format is identical from 2016-10-28 to 2026-09-11.
  On 2026-09-11 and 2024-07-08, MTO's deliverable quantity equals
  `sec_bhavdata_full`'s on 100% of EQ symbols, the percentages differ by 0.000,
  and the UDiFF closes agree with `sec_bhavdata_full`'s on 100%.
- **Backfill.** 2,436 of the panel's 2,440 sessions were fetched `ok`. EQ rows
  for the 118 symbols in the universe's rename histories were cached in
  `delivery_cache.npz`. Nine tickers were matched under a FORMER symbol on
  13,047 rows, via NSE's `symbolchange.csv` (for example LTI→LTIM→LTM and
  TATAMOTORS→TMPV).
- **The four sessions with no file are PHANTOM sessions in the panel.** On
  2026-01-15, 05-01, 05-28 and 06-26, every close in the panel equals the
  previous session's exactly, so the price source is carrying exchange holidays
  as trading days. That is a panel defect, recorded and not fixed here:
  `panel_cache.parquet` is frozen and hashed so that the baseline reproduces.
  Delivery on those dates is missing by construction.
- **Coverage of panel (date, ticker) rows:** 97.7-100% every year.
  ADANIPOWER.NS (91.5%), ADANIENSOL.NS (92.3%) and CGPOWER.NS (92.3%) have
  multi-month gaps. On sampled dates the MTO file carries no row for those
  symbols in any series. They stay missing.

## Point-in-time, and missing values

The file for session T is published after T's close, so every feature at t is
built from delivery of t-1 and earlier, on the panel's own session grid. A
missing session stays NaN. Rolling means use the observations that exist,
subject to their minimums, and nothing is forward-filled. After within-date
z-scoring a missing value becomes the date's mean (0), the panel-wide
convention for "no information", not a stale figure. A leakage test corrupts
every delivery value from a date onward and requires every feature up to that
date unchanged, and requires the feature one session later to move.

## The features, and why the transform

| column | definition |
|---|---|
| `deliv_pct_l1` | delivery % of session t-1 (the raw level) |
| `deliv_abn_l1` | `deliv_pct_l1` minus its own trailing 60-session mean (at least 40 observations) |
| `deliv_abn5_l1` | the mean over sessions t-5..t-1 minus the same 60-session mean |

**Outcome-blind design inputs.** Coverage of panel rows: level 99.5%, abn
97.8%, abn5 97.2%. **Within-date rank persistence at 250 sessions: level
+0.447, abn +0.004, abn5 +0.008.**

The raw level is substantially a per-company constant: large, widely held
names settle a steadier share by delivery. That is the §7 landmine, where a
persistent per-ticker feature lets a tree recognise WHICH COMPANY it is, and it
earned `pooled_xgb` a mean rebalance t of +0.77 from pure noise. The trailing-
mean transform removes that persistence. So the transform is the hypothesis,
and the level runs only as a comparison arm that measures the identity channel.

## The arms

All use `tools/stage2b_pooled.run_arm` on the frozen panel: MAE objective, no
ticker, 10 trials, 5 purged folds, `min_train` 500, horizon, purge and embargo
30. The delivery columns are z-scored within date like every other feature.

| arm | features |
|---|---|
| (a) baseline | `FACTORS` — REUSED from `stage2b_pooled_oos.npz` (sha256 of arrays `67cc72be…12e1`) |
| (b) abnormal | `FACTORS` + `deliv_abn_l1`, `deliv_abn5_l1` — **the hypothesis** |
| (c) level | `FACTORS` + `deliv_pct_l1` — the identity-risk comparison, descriptive |

**S1 — stop condition.** Arm (a) is re-run and must reproduce the stored
predictions at drift ≤ 1e-9 on identical rows. If it does not, stop.

## The decision rules

**R3 — THE DECIDING RULE.** Paired per-date cross-sectional rank IC, (b) − (a),
on identical rows, with the Driscoll-Kraay SE from
`evidence_panel.driscoll_kraay_se` at **30 lags**. R3 holds iff **t ≥ +2.0**.
No other check substitutes for it. This is the rule that correctly read the
reversal pilot as null while its grade counts looked promising.

**R2 — THE CORRECTED PLACEBO.** The two abnormal columns are permuted JOINTLY
within each date, and arm (b) is RETRAINED, nine times with seeds 20260914 to
20260922. Every other input is identical, so each draw destroys exactly the
name-to-delivery link. Predictions are NEVER shuffled: pilot 1's
prediction-shuffle credited its arm with the baseline's own STRONG. R2 holds
iff arm (b)'s paired gain over (a) EXCEEDS ALL NINE placebo gains, computed the
same way.

**SIGNAL iff R3 and R2.** Anything less is reported as not-signal, naming the
clause that failed.

**Descriptive only; they decide nothing:**
- the three arms' `grade_panel_v3` grades (B = 1000, block 30, seed 20260908);
- which tickers move out of INSUFFICIENT;
- arm (c) against (a) and against (b);
- each delivery column's own per-date IC (DK, 30 lags);
- Stage 2b's cell metrics.

**R6.** If SIGNAL, arms (a) and (b) are re-run at `min_train` ∈ {380, 420, 460,
500, 540, 580}. A lone spike is not a result.

## Predictions, scored in the findings whichever way they fall

- **P1.** Arm (a) reproduces (S1 passes).
- **P2.** No signal: arm (b)'s gain over (a) has abs(Δ) < 0.005 and abs(t) < 2.
- **P3.** The level arm moves more tickers out of INSUFFICIENT than the
  abnormal arm does, with no cross-sectional gain (abs(t) < 2): the identity
  channel at work.
- **P4.** Every delivery column's own abs(IC) < 0.02.
- **P5.** All nine placebo gains lie within abs(Δ) < 0.005.

## Caveats carried in

- **Every SE is likely too small.** Politis-White puts the panel's dependence
  at 35.8-62.5 sessions.
- **Trials accumulate:** the reversal pilot's arms, draws and sweep cells, plus
  this pilot's, on top of the ~131 prior configurations.
- **The quantity MTO uses as its denominator is its own.** On 2026-09-11 it was
  0.889 of `sec_bhavdata_full`'s traded quantity, and on 2024-07-08 it was
  1.000. So the denominator's definition shifted after mid-2024, plausibly with
  NSE's newer settlement segments. The delivery % is internally consistent
  throughout, and the trailing-mean transform absorbs a slow level shift, but
  the raw level's post-2024 values may not be comparable with earlier ones.
- **The four phantom panel sessions** above.

## Non-goals

No database write: the cache is a local, gitignored `.npz`. No production
wiring, no Render redeploy, no other new source, and no ticker identity.
Claude runs no `git add` / `commit` / `push`.
