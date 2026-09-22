# Pre-universe hygiene — the lock, the phantom sessions, the Stage 1 re-run

2026-09-21. Hygiene, not discovery. Run on `main` at `96c2ffc`, after the
consolidation (`main` = `origin/main`, no other branch, one worktree).

The order was fixed in advance so that each change is attributable on its own:
**CLAUDE.md → lock (then exact reproduction) → phantom fix (then measured
delta) → pre-register → Stage 1 re-run.** No step's numbers were read before
the step before it had passed.

---

## 1. The dependency lock

### Tool: `uv pip compile`, not `pip-compile`

Both can write a fully pinned, hashed `requirements.txt` that plain pip installs,
and both treat an existing lock as the preference on a re-compile. The
difference that decided it:

- **uv resolves `--universal`**: ONE lock valid on every platform, with
  environment markers where a dependency is platform-specific
  (`colorama ; sys_platform == 'win32'`, `nvidia-nccl-cu12 ; sys_platform ==
  'linux'`). `pip-compile` resolves for the interpreter it runs in. Development
  here is Windows on Python 3.13; CI and Render are Linux on 3.12. A
  `pip-compile` lock made here is a Windows-3.13 lock, and making a Linux one
  means running pip-tools inside Linux 3.12 and keeping two files.
- uv was already installed; pip-tools was not.
- The output is an ordinary requirements file. **Nothing that installs it needs
  uv** — CI and Render keep using pip.

### Pinned to what produced the stored numbers, not to what a fresh resolve picks

The first compile ran with this machine's `pip freeze` as a constraint; a second
compile without it kept every pin, so the committed command reproduces the
lock. Every locked version equals the installed one — except
`nvidia-nccl-cu12`, a Linux-only XGBoost dependency this Windows machine never
had, which both pip and the lock take from PyPI.

**Three exact pins in the old `requirements.txt` did not match what ran
locally:** `requests==2.32.3` (local 2.33.1), `httpx==0.27.0` (0.28.1),
`psycopg2-binary==2.9.9` (2.9.12). They were added in May with no recorded
reason, and `psycopg2-binary` 2.9.9 **ships no wheel for Python 3.13**, so the
old pin cannot install on the development interpreter at all. The lock takes the
local versions. All three are I/O. **This is the one change the lock makes to
what production runs** — CI and Render were on the old pins.

| lock | pins | installed by |
|---|---|---|
| `requirements.txt` | 88 | Render, daily, weekly, series workflow |
| `requirements-evidence.txt` | 18 | weekly (adds arch, linearmodels, statsmodels ...) |
| `requirements-research.txt` | 1 | research machines only (pyarrow) |

**`pyarrow` was an undeclared dependency.** Building an environment from the
locks alone, the first research tool failed: every one reads
`panel_cache.parquet`, and no requirements file named a parquet engine. The
numbers in `docs/` were produced with whatever pyarrow happened to be installed
— and pandas 3 backs its default string dtype with pyarrow when present, so it
is not only I/O. It now has its own lock, which Render never installs.

### Platform checks

- **Linux, CPython 3.12.0, pip** (a throwaway WSL venv): the serving lock, then
  the evidence lock, then the research lock install in hash-checking mode with
  `pip check` clean after each. The serving set carries **no arch,
  linearmodels or torch**. **`ta` 0.11.0 has no wheel and builds from its
  sdist** without trouble. The Linux environment is exactly the lock (96 of 98
  pins apply; the other two are Windows-only).
- **Windows, CPython 3.13.15, a fresh venv built only from the locks:** installs
  and matches the lock (97 of 98).
- The development interpreter itself matches all 107 pins.

### The exact-reproduction check — PASSED, and a finding it did not expect

In the fresh Windows venv built only from the locks, the h=30 standardised
baseline (`hygiene_repin_oos.npz`, `new_pinned`) reproduces **at drift exactly
0.0 on all 160,435 rows**, labels included (6.2 min). The lock changed nothing
that matters.

**On Linux the same locked code does NOT reproduce it:** max drift 0.64,
correlation 0.988 in folds 0-3 and 0.878 in fold 4, where the inner search picks
gamma 0.172 instead of 3.188. Linux reproduces ITSELF exactly, run to run.
Located step by step rather than assumed:

| checked on both platforms | identical? |
|---|---|
| the standardised panel's feature and target arrays | yes |
| Optuna's ten proposed configurations (seed 42) | yes |
| an XGBoost fit, `subsample` = `colsample_bytree` = 1.0 | yes |
| the same fit with `subsample` 0.8 | **no** |
| the same fit with `colsample_bytree` 0.7 | **no** |

XGBoost's row and column subsampling draws from the C++ standard library, whose
generator is specified but whose distributions are not; MSVC and GCC draw a
different sample from the same seed. **No lock can fix that.** Every stored
number in this repository is a Windows number; CI and Render are Linux. See the
platform landmine in CLAUDE.md §7 and §6 below.

---

## 2. The phantom sessions

### Root cause: Yahoo's holiday bar, stored verbatim

Read-only, from the production `ohlcv` table: on 2026-01-15 all 116 tickers
carry open = high = low = close = the previous close and volume 0 (RELIANCE.NS:
1458.8 four times, volume 0, between two ordinary bars). Not a generated
business-day calendar, not a forward fill in our code — **Yahoo returns a bar
for an NSE holiday and `pipeline/fetch.py` stored it.**

### The calendar: NSE's own records

- **Past dates: NSE's delivery archive.** NSE publishes
  `MTO_<ddmmyyyy>.DAT` for every session it held and no other date — the
  source `pipeline/delivery.py` already reads. 1,484 dates probed (every
  weekend, every listed holiday, every weekday not already confirmed) plus the
  2,436 sessions the Stage 1b backfill had confirmed.
- **Future dates: NSE's published holiday list**,
  `/api/holiday-master?type=trading&year=Y`, CM segment — answers without
  cookies, for every year back to 2016.
- **Staying current:** `tools/sync_nse_calendar.py` rebuilds
  `data/nse_calendar.json`; run it each December when NSE publishes the next
  year, and after any ad hoc closure. Between syncs, `live_calendar()` merges
  the current year's live list at run time, so an ad hoc closure announced after
  the commit is still honoured. A date in a year the calendar does not cover
  raises `CalendarNotCurrent` rather than guessing, and
  `test_the_calendar_covers_the_current_year` fails on 1 January until the file
  is updated.

**The list alone would have been wrong.** It shows five Muhurat evenings as
holidays — 2016-10-30, 2019-10-27, 2020-11-14, 2023-11-12, 2025-10-21 — and NSE
traded on every one. Result: **156 weekday closures, 11 weekend sessions**
(Muhurat, budget days, disaster-recovery drills), 2016-2026, **no weekday
closure missing from NSE's list.**

### Where the fix lives

1. `pipeline/fetch.py` drops every non-session before writing. Because each run
   replaces the 10-year window, the first run after the merge also removes the
   phantom rows already stored.
2. `pipeline/signals._upsert_signals` counts labels over NSE sessions only. Without
   it, the first clean recompute is a "decrease" of 4-5 labelled rows for every
   ticker and the label guard refuses all 84. It excuses exactly those rows:
   a real session's lost label still refuses the write (tested).
3. `pipeline/panel.load_panel` drops non-sessions too, so no panel depends on
   every ticker having been rewritten since.
4. The gate gains `no_market_wide_flat_bars` (WARN): any recent date with a
   flat zero-volume bar across ≥ 90% of ≥ 10 tickers, labelled as either a
   missed holiday or a real session Yahoo has no prices for.

### The all-years audit

Every date in ten years of `ohlcv` where ≥ 50% of tickers carry the holiday bar:

| date | NSE traded? | what it is |
|---|---|---|
| 2025-03-18 | **yes** | **a real session Yahoo serves as a flat bar** (117 tickers, 99%) |
| 2026-01-15 | no | municipal election, Maharashtra |
| 2026-05-01 | no | Maharashtra Day |
| 2026-05-28 | no | Bakri Id |
| 2026-06-26 | no | Muharram |
| 2026-09-14 | no | **new** — Ganesh Chaturthi, after the panel cache was built |

**No phantom before 2026, so none reached a training window** (fold training
labels reach 2025-01-03 at the latest): the "no model saw them" conclusion
stands, now over every year rather than 2026 alone. **2025-03-18 is a different
defect** — a traded day whose prices are fiction — and it is in fold 4's test
window. It stays a session; fixing it needs a second price source (NSE's own
bhavcopy), which was out of scope.

### The measured delta — the fix alone

A fresh `load_panel()` today would differ from the 2026-09-07 panel in far more
than the calendar, so the panel was rebuilt TWICE from ONE read-only snapshot
(`tools/phantom_rebuild.py`), through the production signals code, with the
benchmark and earnings inputs frozen to the stored panel's: CONTROL (rows as
stored) and CLEAN (non-sessions removed). **CLEAN − CONTROL is the fix and
nothing else.**

| baseline (h=30, standardised, 2 threads, locked, Windows) | cs IC | DK t | reb IC | reb t | S/W/I | fold-4 gamma |
|---|---|---|---|---|---|---|
| stored, 2026-09-07 panel (the §1 reproduction) | +0.01041 | +0.73 | +0.0287 | +1.35 | 0/0/84 | 3.188 |
| CONTROL: rebuilt, phantoms kept | +0.01209 | +0.85 | +0.0280 | +1.33 | 0/0/84 | 0.172 |
| **CLEAN: rebuilt, phantoms removed** | **+0.01142** | **+0.81** | **+0.0294** | **+1.41** | **0/0/84** | 0.172 |

Paired per date, Driscoll-Kraay, 30 lags:

| comparison | what it isolates | Δ cs IC | t |
|---|---|---|---|
| **CLEAN − CONTROL** | **the calendar fix alone** | **−0.00063** | **−0.61** |
| CONTROL − stored | rebuilding from today's data | +0.00167 | +1.20 |
| CLEAN − stored | both | +0.00097 | +0.55 |

**The fix moves the baseline by noise**, as the Stage 1b truncation check
predicted (t −0.18 against −0.19). It is spread over folds 1-4 rather than
confined to fold 4, for a mechanical reason: four fewer dates re-cut every fold
boundary (387 dates a fold instead of 388), so only fold 0 is untouched.

**The larger mover is the rebuild itself** — Yahoo re-scales `adj_close` history
at every dividend ex-date, and the stored panel also mixes signals written on
different days — and it flipped fold 4's selected gamma. Recorded as a
landmine: compare two panels only when both come from one snapshot.

**CLEAN is the new stored baseline**: `baseline_clean_oos.npz`, arm
`new_pinned`, arrays sha256 `bed813ac…`, on `panel_cache_clean.parquet`
(`c0a26d45…`). **It reproduces itself exactly**: the Stage 1 re-run's S1
re-fits it on each of the three pilot panels and gets drift 0.0 on 160,104 of
160,104 rows, three times. (A standalone twice-in-a-row job was stopped by
Claude Code when the machine ran low on memory, and was not restarted.)

---

## 3. The Stage 1 re-run

Pre-registered in `docs/stage1-rerun-preregistration.md` (sha256
`2469804d…9a72`, written and hashed at 18:18 before any arm ran), run by
`tools/stage1_rerun.py`: 36 fits (3 baselines, 6 arms, 27 placebo retrains) in
parallel processes at two threads each, then `grade_panel_v3` at B = 1000.

**S1: exact, three times** (drift 0.0, 160,104 rows, on each pilot's panel).

**Resolution, the confound the re-run exists to remove:** old label 8-12
distinct predictions per date (min 2-4), 3-9 constant cells; new label **84 of
84** (min 83), **0 of 420** constant cells, in every arm and every placebo.

| pilot | arm | old gain, t | **new gain, t** | new S/W/I | placebos beaten |
|---|---|---|---|---|---|
| reversal | **resid_reversal** | +0.0002, +0.07 | **−0.00186, −0.84** | 0/0/84 | 9 of 9 |
| reversal | raw_reversal | — | −0.00252, −1.66 | 0/0/84 | — |
| delivery | **abnormal** | −0.00066, −0.19 | **−0.00343, −2.06** | 0/1/83 | 4 of 9 |
| delivery | level | +0.0005, +0.08 | +0.00544, +1.61 | 0/1/83 | — |
| SUE | **sue** | −0.0055, −0.56 | **−0.00061, −0.16** | 0/3/81 | 9 of 9 |
| SUE | timing | — | −0.00189, −0.51 | 1/5/78 | — |
| | baseline | | cs IC +0.01142 | 0/0/84 | |

**R3 fails for every hypothesis arm: all three remain NOT SIGNAL.** SUE's R5
(surprise over timing) is +0.00128, t +0.36 — positive, which fails the
pre-registered prediction P4, and decides nothing because R3 fails first. R4,
descriptive: the SUE book trails the baseline by 0.30% per rebalance net of cost
(t −1.33).

**New, and only visible on the new label:** all 27 placebo retrains lose IC
against the baseline (−0.0016 to −0.0043). Adding a few within-date-noise
columns to the pooled model COSTS it about 0.003 of IC, so a "passed" R2 now
means "less harmful than noise", not "helpful". Delivery's abnormal arm at
t −2.06 sits inside its own placebo band and is negative; at ~150 trials a best
|t| near 3.2 is noise's expectation.

**The grades repeat Pilot 3's identity lesson:** the timing-only SUE arm,
carrying no surprise, grades six names (one STRONG); the arm with the surprise
grades three.

**Conclusion: the Stage 1 nulls were the features, not the label's
resolution.** `docs/stage1-closing.md` carries a dated addendum removing the
asterisk.

---

## 4. What this means for the dashboard switch (the next session)

- **The lock changes what Render installs**: `requests` 2.33.1, `httpx` 0.28.1,
  `psycopg2-binary` 2.9.12 instead of the old pins, and every transitive version
  frozen. Render must redeploy, and its build command should end with the lock
  check (§5).
- **The pooled model will run on Linux.** Its numbers will not match any stored
  Windows figure bit for bit. Establish the production baseline ON Linux (WSL
  reproduces itself exactly, ~4x faster) before comparing anything to it.
- **The first daily run after merge rewrites `ohlcv` and `signals` without the
  five phantom dates**, and the weekly run after it re-measures every per-ticker
  evaluation on labels that no longer step over them. Together with the thread
  pin, expect the badges to move once.

## 5. Remaining debt before a universe comparison

- **The platform.** Pick ONE reference platform for the universe comparison and
  run both arms on it.
- **The snapshot.** Build both universes' panels from one read at one time;
  Yahoo's history moves.
- **2025-03-18**, a traded day with fictional prices, until a second price
  source exists.
- **`pipeline/macro.py`** steps `nifty_5d_return` / `nifty_20d_return` over a
  grid that includes NSE holidays. Not in FACTORS; a per-ticker model feature.
- `requirements-scoring.txt` and `requirements-series.txt` are not locked
  (torch, transformers, chronos). The workflows install them before the lock so
  they cannot move a locked package, but their own versions float.
