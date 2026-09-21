# Hygiene — the standardised label, the thread pin, and Part C re-run clean

**Run 2026-09-21 against `docs/hygiene-preregistration.md`** (sha256
`134ba407…6202`) and `docs/part-c-preregistration.md` (sha256
`783f969b…b6f7`), both written and hashed before anything ran on real data.

Not a discovery session. Three defects P6 exposed are closed, every stored
comparator is re-pinned at the new scale, and Part C — which carried P6's
headline claim and had never been pre-registered — is re-run under a
pre-registration.

**Nothing here produced signal, and that was the pre-registered expectation.**

## The answer, in five lines

1. **The standardised label removes the degeneracy completely** — 0 of 420
   constant cells at every horizon — and it does something that was not
   predicted: the predictions stop being coarse. The median date went from
   **8 distinct predicted values across 84 names to 84**, and the quintile
   books from 48 of 64 majority-tie-break to **none**.
2. **The thread pin moved the 30-session baseline**, exactly as H3 said it
   would: drift 7.0e-02 against the committed predictions, and the graded
   count moved 1/1/82 → 2/3/79 on the thread count ALONE.
3. **Conformal coverage survives the round trip.** Through the causal inverse
   — the only one available live — it reads 0.8166 at the first checkable
   fold against 0.80 nominal, and tracks the raw label's own coverage to
   within 3 percentage points at every fold.
4. **Part C reproduces its shape and its verdict, and not its digits.** The
   IC profile is flat again, largest at h=20 and smallest at h=10, every value
   within 0.0021 of the original. No horizon effect.
5. **One prediction failed, and it is the one that matters most.** The h=5
   cell now SURVIVES its `min_train` sweep — five of six settings at
   t ≥ +2.0 — where both pre-registrations said it would not. Investigated as a
   defect first: the transform is clean (identical rows, no leak across the
   boundary, both tested and mutation-verified). What survives is a small,
   FLAT positive IC of about +0.014 that is not a horizon effect, sits inside
   the range a no-information placebo reaches, deflates to nothing at ~150
   trials, and earns **+0.027% per rebalance net of cost**. A hypothesis for
   the universe session, not a result. §4.

---

## 1. The standardised label, and the full re-pin

`pipeline/label.py` holds the transform and both inverses, and
`tools/stage2b_pooled.run_arm` — the training entry point every Stage 1 pilot
and P6 goes through — applies it by DEFAULT. There is one opt-out in the
codebase and it exists for one purpose: the A0 regression pin, which has to
reproduce predictions frozen under the old label.

### The comparator table, old against new

All three at h=30, legacy purge, identical folds and rows. The middle row is
the point of splitting them: it isolates what the THREAD PIN alone moved,
before the label changes anything.

| arm | label | threads | cs IC (DK SE) | t | reb IC | reb t | constant cells | S / W / I | mu_hat | tau2 | OOS MAE | per-fold gamma |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **committed** (in the repo today) | raw | 20, unpinned | −0.00101 (0.01201) | −0.08 | −0.0014 | −0.09 | 7/420 | 1 / 1 / 82 | −0.01728 | 0.00435 | 0.08882 | 1.792, 0.290, 4.934, 4.934, 4.934 |
| **old_pinned** | raw | **2, pinned** | −0.00423 (0.01048) | −0.40 | −0.0041 | −0.28 | 6/420 | **2 / 3 / 79** | −0.01001 | 0.00541 | 0.08890 | 1.943, 0.290, 4.744, 4.934, 4.744 |
| **new_pinned** | **standardised** | 2, pinned | +0.01041 (0.01420) | +0.73 | +0.0287 | +1.35 | **0/420** | **0 / 0 / 84** | −0.03654 | 0.00247 | 0.76074 | 3.188 × 5 |

Three things to read off it.

**The thread pin alone changed the graded count.** Same label, same code, same
seeds — 1 STRONG / 1 WEAK becomes 2 STRONG / 3 WEAK, and the cross-sectional
IC goes from −0.00101 to −0.00423. Four of the five selected gammas moved. That
is the blast radius of §2, measured rather than argued.

**The standardised label grades FEWER names, not more: 0 / 0 / 84.** Every
grading change this project has made has gone the same way, and this one does
too. It is also the sanity check that matters for a hygiene session — a label
change that suddenly graded more names would be the red flag §0 of the
pre-registration told us to hunt for.

**The OOS MAE column is not comparable across the last row** and must never be
quoted as an improvement or a regression. 0.76074 is in units of a z-score;
0.08890 is in log-return units. The two measure the same thing on axes that
differ by a factor of eight.

### What was not predicted: the predictions stop being coarse

| arm | distinct predicted values per date (median, of 84 names) | books majority tie-break | books tradeable |
|---|---|---|---|
| old_pinned (raw) | **8** | 48 of 64 | 16 |
| new_pinned (standardised) | **84** | **0 of 64** | **64** |

The coarseness P6 found — the median date carrying nine distinct values across
84 names, one date with 52 names tied at the top — was the SAME defect wearing
a second face. A high `gamma` buys few splits, few splits means few leaves, and
few leaves means names share a predicted value. Standardising the label lets
the trees split, and the cross-section separates completely.

So the quintile books at h=30 are real for the first time. Sixty-four of
sixty-four, against sixteen.

### The A0 re-pin

| | |
|---|---|
| rows stored / re-run | 160,435 / 160,435 |
| unmatched either way | 0 / 0 |
| max label drift | 0.0e+00 |
| **max prediction drift** | **7.009e-02** |

**Non-zero, and that is prediction H3 holding rather than a failure.** The
committed predictions were produced at 20 threads; these are at 2. The labels
are identical, the rows are identical, and the model is not.

**Reproducible twice in a row on this machine: drift 0.0e+00, identical.**
That is the property the pin was added for, and it is now true of a re-run, of
a second process, and — measured in `tests/test_determinism.py` — of a
different thread environment.

### Is `[0, 5]` still the right `gamma` range?

Investigated rather than assumed, because the standardisation does not settle
it on its own.

| label | pooled sd | mean abs | what `gamma` is compared against |
|---|---|---|---|
| raw | 0.11964 | 0.08718 | a loss reduction of order 0.09 |
| standardised | 0.99403 | 0.75197 | a loss reduction of order 0.75 |

**`[0, 5]` stays, and the evidence is behavioural rather than dimensional.**
The range is unchanged and the selected gamma came back 3.188 on all five folds
— comfortably inside the range, not pinned at either end — while producing
**0 of 420 constant cells**. Under the raw label a gamma of 3.188 switched the
model off; under the standardised one the same value splits normally. The range
was never wrong in itself; what was wrong was the scale it was being compared
against, and fixing the scale fixed the range.

Two caveats kept on the record. The label's POOLED sd is 0.994 rather than
exactly 1 — dates too thin or too flat to standardise drop out — so the scale
is approximately, not exactly, unit. And a search that settles on the same
value at every fold is worth watching: it is what a flat objective looks like,
and if a future run shows it again with a DIFFERENT value, the range is the
first thing to re-examine.

---

## 2. Conformal coverage after the round trip

A standardised label means the model predicts **relative cross-sectional
position**, so a price band only exists after inverting — and the inverse needs
that date's cross-sectional mean and standard deviation, **neither of which is
knowable at prediction time**. They are properties of the h-session window that
has not happened yet: the mean is the market's move over it, the sd is the
cross-section's dispersion over it.

So `pipeline/label.py` provides two inverses and they are not interchangeable:

- **realised moments** — exact, correct for scoring history, and **never
  available live**. Inverting a live forecast with them would be F1 in a new
  place: a quantity from the future used to dress up a prediction of it, and it
  would read as a large improvement in both MAE and coverage.
- **causal moments** — a trailing estimate lagged by the horizon and then
  averaged over 252 dates, which is the only thing a forecast can use. It is an
  approximation, and the conformal layer is what turns its error into a
  measured interval instead of an assumed one.

Coverage is measured **split-conformal**: calibrated on the early folds and
checked on a fold it never saw. Checking on the calibration pool reports the
quantile's own definition back and would look like a pass.

| arm | inverse | half-width | **realised coverage (fold 4)** | MAE |
|---|---|---|---|---|
| old label (raw) | direct | 0.14160 | 0.9006 | 0.08890 |
| new label (z) | realised | 0.11506 | 0.8735 | 0.07229 |
| new label (z) | **causal** | 0.14614 | **0.9060** | 0.09087 |

### Per check-fold, calibrated on everything before it

| arm / inverse | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|
| old (raw), direct | 0.7868 | 0.8890 | 0.8776 | 0.9006 |
| new (z), realised | 0.7881 | 0.8705 | 0.8671 | 0.8735 |
| new (z), **causal** | **0.8166** | 0.8849 | 0.8706 | 0.9060 |

**Coverage does not degrade. The round trip is not what moves it.**

The single figure to read is fold 1 — the first fold with enough history behind
it to calibrate on, and the one whose regime most resembles its calibration
set. There the causal round trip reads **0.8166 against 0.80 nominal**, closer
to nominal than the raw label's own 0.7868.

**The over-coverage at folds 2-4 is a property of the panel, not of this
change**, and the per-fold table is what shows it: the raw label over-covers by
just as much, at every fold, with no standardisation anywhere near it. The
cause is on the record — this panel's target dispersion falls monotonically
across the folds (0.108, 0.104, 0.100, 0.088, 0.077), so a half-width
calibrated on the early, wilder folds is too wide for the later, calmer ones.
An interval that is too wide is conservative rather than wrong, and it was true
before this session.

**What the round trip does cost, stated plainly:**

- **MAE rises 2.2%**, 0.08890 → 0.09087. That is the causal moment estimate's
  error, and it is the honest price of predicting a relative position and
  converting back.
- **The interval widens 3.2%**, 0.14160 → 0.14614, for the same reason.
- The gap between the two inverses — 0.11506 against 0.14614 — is how much of
  the interval is the moment estimate rather than the model. **A fifth of the
  published half-width would disappear if the future's dispersion were known**,
  which is a useful measure of how much of this problem is the market's
  volatility rather than the model's ignorance.

**H5 holds and nothing is stopped.** Coverage stays within 5 points of nominal
at the fold where that question is answerable, and the deterioration at later
folds is pre-existing and conservative.

### What this does NOT say, and the scope decision behind it

**The 80.1% figure on the record is the PER-TICKER path's, and that path is
untouched by this session.** `pipeline/model.py` trains one ticker's time
series at a time; a within-date z-score is a cross-sectional operation and
needs the whole panel's labels at each date, which `load_features_for_ticker`
does not load. Standardising there would couple every ticker's target to the
whole universe, bump `MODEL_VERSION`, invalidate every persisted evaluation and
conformal calibration, and send all 84 names to INSUFFICIENT until the weekly
job re-ran.

**It was not done, and the reason is not only cost.** The per-ticker
degeneracy — 316 of 420 constant cells — was diagnosed by Stage 2b as an
ALTITUDE problem, not a scale one, and Stage 2a measured that changing the
objective did not fix it. The defect this session closes is specific to the
POOLED path, so applying its fix to the per-ticker path would be a
production-grade disruption for a problem it does not address.

**Consequence: the per-ticker conformal layer is unchanged, so its 80.1%
coverage is unchanged.** The table above measures what a published interval
WOULD cost if the pooled, standardised prediction ever became the thing served
— which is the next session's question, not this one's.

---

## 3. The thread pin, and what else turned out to be unpinned

`pipeline/determinism.py` is one module, imported by all seven `XGBRegressor`
construction sites in the codebase. It pins `n_jobs` and `tree_method`, and
`n_jobs` is deliberately **not overridable by a caller** — a hyperparameter
dict carrying one does not win.

### The obvious fix was the wrong one, and it was measured

`n_jobs=4` — the natural choice on a 20-core workstation — does **not**
reproduce across machines. OpenMP caps the team at whatever the environment
permits, so a pin ABOVE the smallest machine's core count silently degrades
there:

| OMP_NUM_THREADS | n_jobs | prediction digest |
|---|---|---|
| 2 | unpinned | `5f8eba01…` |
| 20 | unpinned | `de5191e7…` ← differs |
| 2 | 4 | `5f8eba01…` |
| 20 | 4 | `64ff5a4f…` ← still differs |
| 2 / 8 / 20 | **2** | `5f8eba01…` — all three identical |
| 2 / 8 / 20 | 1 | `f3f00ce3…` — all three identical |

**So the rule is: pin at or below the smallest core count anything will run
on.** Two is that floor, and it is **not** this repo's CI: the repository is
public, and its hosted runners have four vCPUs. Two is set by the smallest
machine a recorded number might plausibly be produced on — a private-repo
runner, a two-core VM, a laptop.

**What that means in production, stated plainly.** Before the pin the
scheduled jobs fitted at the runner's four threads; after it they fit at two.
So **the first weekly run after this merges will move the persisted per-ticker
evaluations**, in the same way the 30-session baseline moved when the pin was
applied to it here — a one-off step, attributable through
`tracking.environment_hash`, and the price of every later run being
reproducible. Expect the live grade distribution to shift on that run and do
not read the shift as a model change.

**The cost, measured on a 160,000-row fit:** 0.8 s at 20 threads, 1.2 s at 4,
**2.2 s at 2**, 6.0 s at 1. A 2.75× slowdown against the workstation default,
paid for a number that means the same thing on two machines.

`threads_are_pinned()` reports the one case the pin cannot cover — a machine or
an `OMP_NUM_THREADS` below the pin — because that is detectable rather than
mysterious.

### The blast radius, stated plainly

**Every result committed to this repository before today was
machine-dependent.** They were produced on this workstation at its default of
20 threads and nothing recorded that. A re-run on a GitHub runner, on Kaggle,
or on any other machine would not have reproduced them, and the failure would
have looked like a code change.

**The nulls survive it. A null that wobbles into another null is still a
null.** The 30-session baseline moved from cs IC −0.00101 to −0.00423 and from
reb t −0.09 to −0.28; both are nulls, and no verdict in Stage 1 or P6 turns on
the difference.

**But the fragility bites positives specifically, and that is not a
coincidence.** The single most positive cell the sweep produced — Part C at
h=5 — is the one that moved across its own threshold, +2.405 at 20 threads
against +1.987 at 6, with zero of 162,535 predictions matching and a maximum
difference of 6.2 prediction standard deviations. A statistic near a decision
boundary is exactly where a small perturbation changes the decision, so
unpinned threads are most dangerous precisely where the stakes are highest.

The grade counts show the same shape at h=30: the thread pin alone moved
1 STRONG / 1 WEAK / 82 to **2 STRONG / 3 WEAK / 79**. Grades are the output
closest to a threshold, and they are what moved.

### What else was found unpinned, and what was done about it

| source | status |
|---|---|
| `n_jobs` on all seven `XGBRegressor` sites | **pinned** at `XGB_THREADS` |
| `tree_method` — the default is `"auto"`, which picks a different builder BY DATA SIZE, so a small fold could be fitted differently from its neighbours | **pinned** to `"hist"`; six of seven sites already passed it and "every site except one" is how the fold-guard landmine happened |
| Optuna's sampler | already seeded (`SEED = 42`) |
| `random_state` on every fit | already 42 |
| **library versions** — `requirements.txt` uses `>=`, so xgboost, optuna, numpy, pandas and scikit-learn all float | **NOT pinned.** Recorded instead: `environment_fingerprint()` captures all five, and `tracking.environment_hash()` is a third hash beside `config_hash` and `data_hash`. Tightening `requirements.txt` to `==` is a real change to what installs on Render and in CI, and it belongs in its own session. |
| a single-core machine | cannot be covered — `XGB_THREADS = 2` would be capped to 1. `threads_are_pinned()` reports it. |

The environment fingerprint rides in `experiment_runs.metrics`, not in a new
column: adding one is a `data/db.py` migration, which means a Render redeploy,
and the JSON was already there.

---

## 4. Part C, pre-registered and re-run clean

Run under `docs/part-c-preregistration.md`, hashed before the run, with the
standardised label as the pipeline default and the thread count pinned. Grid,
purge rule (`max(h, 63)`), `min_train`, folds, trials, block and DK lags all
unchanged; scored against the REAL h-session label.

### Against the original figures

| h | original cs IC | **clean cs IC** | Δ | original t | clean t | constant cells |
|---|---|---|---|---|---|---|
| 5 | +0.01459 | **+0.01406** | −0.00053 | +2.41 | +2.26 | 0 / 420 |
| 10 | +0.00896 | **+0.00875** | −0.00021 | +0.99 | +0.91 | 0 / 420 |
| 20 | +0.01758 | **+0.01852** | +0.00094 | +1.43 | +1.53 | 0 / 420 |
| 30 | +0.01524 | **+0.01727** | +0.00203 | +1.09 | +1.24 | 0 / 420 |

**The shape reproduces exactly and the digits do not, which is what PC3
predicted.** Every value lands within 0.0021 of its original; the ordering is
identical — h=20 largest, h=10 smallest — and the degeneracy stays gone at
every horizon. The thread count moved the selected hyperparameters, as P6
measured, and the conclusion did not move with them.

### C1, the deciding rule — read on the IC profile

| | h=5 | h=10 | h=20 | h=30 |
|---|---|---|---|---|
| **cs IC (the comparable statistic)** | +0.0141 | +0.0088 | +0.0185 | +0.0173 |

**Flat: four values inside 0.009 to 0.019, no monotone trend, no single
peak — and the largest two at the LONGEST horizons.** C1 is met for "no horizon
effect". H-C holds, and P6's closing argument now rests on a pre-registered run
rather than a post-hoc one.

### C4, the h=5 cell — and the one prediction this session got wrong

It reads **t +2.26** under the pin (+2.41 unpinned at 20 threads, +1.987 at 6).
C2 put it through the `min_train` sweep, and this time it holds:

| min_train | 380 | 420 | 460 | **500** | 540 | 580 |
|---|---|---|---|---|---|---|
| cs IC | +0.0132 | +0.0101 | +0.0169 | +0.0141 | +0.0189 | +0.0144 |
| **t** | **+2.39** | +1.78 | **+2.60** | **+2.26** | **+2.99** | **+2.19** |

**Five of six at t ≥ +2.0, and every one positive.** It also beats all nine
target-permuted retrains on IC (+0.01406 against a maximum of +0.00726), is not
carried by fold 0 alone (folds: +0.031, −0.004, +0.006, +0.006, +0.031 — fold 4
as large as fold 0), and clears its own break-even at its own turnover (0.0106).
`tools/p6_scale_followup.py` returns **SURVIVES = YES**.

**Prediction PC4 was wrong, and so was H1 in the hygiene pre-registration.**
Both said this cell would fail the sweep again. On the record, not reconciled.

### Investigated as a defect first, as the pre-registration required

A positive result from a hygiene session is a red flag about the hygiene, and
the pre-registration named the specific suspect: that standardising the label
leaks a cross-sectional statistic of a future window into training. Checked,
in order:

1. **Did the transform change which rows are scored?** No. The standardised and
   raw-label arms score the same 160,435 rows on the same 1,910 dates — zero
   rows in either that are not in the other.
2. **Does a date's z-score depend on any other date?** No.
   `test_a_standardised_label_depends_on_its_own_date_alone` corrupts every
   label from a cut date onward and requires every earlier z-score to be
   bit-identical. Mutated to use panel-wide moments, it fails.
3. **Can a fold's own test window reach its model?** No.
   `test_run_arm_on_the_standardised_label_is_blind_to_its_own_test_window`
   corrupts every label from the last fold's first test date onward, re-runs,
   and requires that fold's predictions to be bit-identical. They are.

**So the transform is clean, and what survives is real in the narrow sense —
and small in every sense that matters:**

- **It is not a horizon effect.** C1 decides that, and the same small IC sits
  at every horizon. The h=5 cell clears t = 2 because its standard error is the
  smallest — 0.00622 against 0.01397 at h=30 on nearly the same number of dates
  — which is exactly what P6's rule A3 says a t across the grid measures.
- **The t threshold is not calibrated here.** In nine target-permuted
  retrains, which carry no information by construction, two reached **t −2.64**
  and **−2.22**. A t of +2.26 sits inside the range the null itself produces.
- **Deflated for the search, it is nothing.** This panel has carried roughly
  150 trials. The expected maximum |t| from pure noise over that many is about
  sqrt(2 ln 150) ≈ **3.2**, and the best cell in the sweep reads +2.99.
- **It does not pay.** Net of the 0.2225% round trip the book earns **+0.027%
  per rebalance at t +0.28**, an annualised net Sharpe of **+0.10**. The cost
  takes 82% of a gross edge that was +0.151% at t +1.55.

**What it IS:** the standardised pooled model carries a small positive
cross-sectional IC, roughly +0.009 to +0.019, that is stable across four
horizons and six `min_train` settings and beats its own target placebo. That is
the most robust positive this panel has produced, and it is worth nothing after
costs. **It is a hypothesis for the next session's pre-registration, and it is
precisely the right thing to carry into a less liquid universe** — where the
same model, the same label and the same folds would say whether an edge of this
size grows once the names are less efficiently priced, or stays at zero.

---

## 5. The two recorded defects, cleared

### 5.1 The tie guard now measures breadth

**What it was.** `evaluation.rebalance_books` refused a date only when the
whole cross-section was tied, and broke partial ties on a hash of the ticker.
P6 found that a date on which 84 names take two distinct values passes that
test, and then both quantile boundaries fall inside a block of ~40 identical
numbers where the hash decides membership.

**Measuring it found something worse than the note said.** The pooled model's
predictions are discrete, and the median date carried **nine distinct values
across 84 names**; one sampled date had 52 names tied at the top value, so a
top quintile of 16 was drawn from that block by hash — 94% of the leg chosen by
a hash of its spelling, then reported as a traded book with a net return.

**The fix is a fraction, not a boolean.** For each leg, everything strictly
beyond the cut is the model's choice and everything at it is arbitrary; a leg
that is MAJORITY arbitrary is refused. `MAX_ARBITRARY_LEG_FRACTION = 0.5`, and
the fraction is carried on `RebalanceBook.arbitrary_fraction` so a refusal is
auditable rather than a bare flag.

**A first cut of it was wrong and the tests caught it.** Requiring the cut to
be strict — the k-th prediction greater than the (k+1)-th — refuses a tied
block that EXACTLY fills a leg, which has no arbitrary choice in it at all. It
would have thrown away determined books and moved every historical number for
nothing.

**The separation that matters.** A date whose legs are arbitrary still RANKS —
`rank_ic` averages ranks over ties, which is correct — so its IC stays in the
sample and only the quantile columns are withheld. Refusing the whole date
would change which dates `mean_rank_ic` averages over, which is the
sweep-that-changes-the-row-count error one level down.

**Measured consequence: every rank-IC verdict in this project is unchanged.**
Re-scored on P6's own stored predictions, the reb_ICs come back to the digit —
h=30 legacy −0.00145 / t −0.09, h=20 deciding +0.01708 / +0.96, h=10 deciding
+0.01998 / +1.18 — exactly what `docs/p6-findings.md` reports. **What moves is
the quantile and book columns**, and there the fix quarantines them:

| cell | books formed | majority tie-break | tradeable | mean arbitrary fraction |
|---|---|---|---|---|
| h=30, raw label | 64 | **48** | 16 | 0.737 |
| h=10, raw label | 39 | **39** | 0 | 0.955 |
| h=30, **standardised** | 64 | **0** | **64** | 0.000 |

So R4-style net-of-cost comparisons on the raw-label pooled arms were
contaminated, and the standardised label removes the problem at its source.

### 5.2 `REBALANCES_PER_YEAR` — flagged by P6, never addressed, now fixed

**It was never addressed.** P6's 2026-09-05 pre-registration called it "one
defect to fix first"; P6 then worked around it by annualising by hand in
`tools/p6_horizon._sharpe` and left the constant alone. Nothing shipped the
wrong number, but the next caller would have got it.

`REBALANCES_PER_YEAR = 252 / HORIZON_SESSIONS = 8.4` was a module constant, and
every annualisation read it regardless of the book's own horizon. An h=5 book
rebalances **50.4** times a year, so its cost drag was understated **six-fold**
and its Sharpe scaled by sqrt(8.4) instead of sqrt(50.4).

**The fix:** `rebalances_per_year(rebalance_every)`, `BookResult` carries the
width it actually traded at, and `metrics()` annualises by that. The module
constant survives as the 30-session default, because several callers
legitimately want it by name. Measured through the fix, the annual cost drag at
0.80 turnover is **9.39% at h=5 against 1.51% at h=30** — the 6× that was
missing.

### 5.3 Render

**A redeploy IS needed to keep Render on the same code — but not because either
defect fix touches it.** The tie guard (`pipeline/evaluation.py`) and the
annualisation fix (`pipeline/portfolio.py`) are never reached from the API.
What IS reached is the thread pin: `api/routers/admin.py` lazily imports
`agents.graph`, which imports `agents.forecasting_agent`, which imports
`pipeline.model` — and `pipeline.model` now builds every `XGBRegressor` through
`pipeline.determinism`. So the admin "run the graph" endpoint imports changed
code.

**Functionally it changes one thing on Render: the thread count of a fit when
that endpoint is hit.** No schema change, no new column, no endpoint, and the
per-ticker label is untouched, so nothing a reader of the API sees changes
shape. Redeploy after merge so the service and the scheduled jobs run the same
code; it is not urgent, because what the site shows comes from the GitHub
Actions jobs, not from Render computing anything. The environment fingerprint
went into `experiment_runs.metrics`, which already exists, precisely so that
this did not also need a `data/db.py` migration.

**What DOES change on the next weekly run**, without any deploy: the
`metrics["baselines"]` block gains an `environment` entry, and any book with
majority-tie-break legs stops being counted as traded. `compare_baselines` was
deliberately left on the RAW label — its floors (`market`, `train_mean`) are
defined in return units, and standardising would make its MAE column
meaningless.

---

## 6. What was NOT re-pinned, and why

**The Stage 1 feature arms were not re-run under the new label.** Reversal,
delivery % and SUE are PAIRED comparisons against a baseline, and each was
decided by R3 at |t| ≤ 0.56 against a +2.0 bar. The baselines they pair against
are re-pinned above; re-running every arm and its nine placebos is roughly
three hours per pilot to move nulls that are nowhere near their threshold. Each
document carries a correction notice saying so, rather than an unstated
assumption that its digits still hold.

**`compare_baselines` stays on the raw label.** Its floors — `market`,
`train_mean`, `zero` — are defined in return units, and a z-scored target would
make its MAE column meaningless.

**The per-ticker production path is untouched** (see §2). Its 80.1% conformal
coverage is therefore unchanged.

## 7. Predictions, scored

| # | prediction | held |
|---|---|---|
| H1 | nothing clears the bar in a form that survives a `min_train` sweep | **no** — Part C's h=5 cell survives C2 (§4) |
| H2 | 0 of 420 constant cells at every horizon | **yes** |
| H3 | the thread pin makes A0 drift non-zero | **yes**, 7.0e-02 |
| H4 | the nulls survive both changes | **yes** — every Stage 1 and P6 verdict stands |
| H5 | conformal coverage stays within 5 points of nominal through the round trip | **yes** at the first checkable fold (0.8166); the later-fold over-coverage is pre-existing |
| H6 | the tie guard leaves every rank-IC verdict unchanged | **yes**, to the digit |
| PC1 | 0/420 constant cells | **yes** |
| PC2 | the IC profile is flat | **yes** |
| PC3 | the figures do not reproduce to three decimals | **yes**, within 0.0021 |
| PC4 | h=5 fails C2 again | **no** — 5 of 6 settings clear +2.0 |
| PC5 | no horizon effect | **yes** |

## 8. Tests

**794 pass, 0 fail**, up from 765 at the start of the session. New:

- `tests/test_determinism.py` — 7 tests. The centrepiece runs the same fit in
  two subprocesses under different `OMP_NUM_THREADS` and requires bit-identical
  predictions; the rest pin that `n_jobs` cannot be overridden by a caller,
  that the pin is low enough for a 2-core machine, that every `XGBRegressor`
  construction site is built through it, and that the case it cannot cover is
  reported.
- `tests/test_label.py` — 12 tests. Exact round trip with realised moments;
  the causal inverse blind to the window it inverts and approximate by
  construction; thin and constant cross-sections refused; and an undefined
  z-score never infinite.
- `tests/test_leakage.py` — the default transform refuses a panel it would
  empty, and two contracts for the standardised label: a z-score depends on its
  own date alone, and a fold's model is blind to its own test window.
- `tests/test_phase2_baselines.py` and `tests/test_phase4_portfolio.py` — the
  tie guard on both legs, a tied block that exactly fills a leg, arbitrary
  books withheld from trading and counted separately, and annualisation by the
  width a book actually traded at.

**Two tests were changed rather than added, and both changes are the point.**
`test_ranking_ties_are_not_broken_alphabetically` asserted the old, weaker
contract — that a hash-decided leg be random rather than absent — and is
superseded by one asserting no book at all. And the `BookResult` field
allowlist grew by two counters, with added checks that no field is a holding.

**Mutation-verified: 20 mutants across the new guards, all caught.** Two
survived the first pass and each changed something:

- **"Only the top leg is checked" survived** — no test had an arbitrary SHORT
  leg under a clean long leg. A test now does.
- **The z-score's infinity guard looked redundant and is load-bearing.**
  Deleting it left the suite green, so the obvious conclusion was the
  redundant-guard landmine. Writing the test to prove it redundant proved the
  opposite: twelve copies of 1/3 have a mean that is not exactly 1/3, so the
  numerator is ~−5.5e-17 over a standard deviation of exactly 0.0, and **every
  name gets −inf** — a fabricated, extreme, identical ordering. The guard
  stayed; the test is new.

**And one defect was found by making the transform the default**, before any
number was read: on a cross-section thinner than `MIN_NAMES_PER_DATE` it
empties the label, every fold fails its row guard, and `run_arm` returned
nothing — indistinguishable from a splitter that produced no folds. It now
refuses loudly.

**Reproducible twice on this machine:** the re-pin was run, `run_arm` was then
refactored to standardise by default, and the re-pin was run again — drift
0.0e+00 against the first, and 0.0e+00 between its own two internal runs.

## 9. Is the pipeline ready for a universe change?

**Yes, on the three things this session could fix — and with two pieces of
debt that would contaminate a universe comparison if left, both small.**

What is now true:

- **The label no longer switches the model off.** 0/420 constant cells at every
  horizon, and 84 distinct predictions per date instead of 8.
- **A number means the same thing on two machines**, and the environment it
  was produced in travels with it.
- **Books are the model's books**, and annualisation uses the width they
  actually traded at.
- **The inverse is causal**, and the leakage contract for it is tested from
  both ends.

What would still contaminate a universe comparison:

1. **Library versions float.** `requirements.txt` uses `>=`. A universe run
   done on one day and its baseline re-run on another could straddle an
   xgboost or pandas release, and `environment_hash` would only tell you
   afterwards. **Pin exact versions before the universe session starts** — or
   run both arms in one sitting and check the hash matches.
2. **The phantom 2026 holiday sessions are still in the panel.** Harmless to
   everything measured so far (zero training rows touch them), but a new
   universe means a new panel build, and it should not inherit them. Fix in
   ingestion before the new panel is built, not after.

And one thing that is a question rather than debt: **the per-ticker production
path still trains on the raw label.** Nothing in the universe experiment needs
it changed, but if the pooled standardised model ever becomes what is served,
the conformal layer in §2 is the measurement that says what its interval would
cost.
