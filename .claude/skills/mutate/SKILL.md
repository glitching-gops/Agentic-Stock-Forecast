---
name: mutate
description: Mutation-verify a guard or test - revert the fix, confirm the test fails with a meaningful message, restore. Use after adding any new guard, invariant or regression test, and when auditing whether an existing test has teeth.
---

# Mutate

The project's stated norm, from CLAUDE.md section 8:

> New guards should be **mutation-verified**: revert the fix, confirm the test
> fails with a meaningful message, restore. Several tests in this suite were
> found toothless that way.

Roughly 100 mutations have been run across the Phase 2 suites. It is not
ceremony - it has repeatedly found real problems, listed at the bottom.

## Interpreter

```
$PROJECT_PYTHON   # or C:/Users/venuw/AppData/Local/Programs/Python/Python313/python.exe
```

## The loop

For each guard under test:

### 1. Name the mutation precisely

State what you are breaking and what you expect to fail. "Break the guard" is
not a mutation; "invert the `<` in the coverage check so a partial fetch is
accepted" is.

Cover the classes that have actually bitten here:

| class | example |
|---|---|
| boundary | `>=` -> `>`, `500` -> `499` |
| inversion | negate a condition, swap if/else bodies |
| deletion | remove the guard line entirely |
| constant | change a threshold, an index, a version string |
| identity | replace a computed value with a hardcoded one |

### 2. Apply exactly one mutation

One at a time. Two mutations can mask each other, and you learn nothing about
which test caught what.

**String replacement is a landmine here.** `str.replace(a, b, 1)` has twice hit
the wrong occurrence - the daily job instead of the weekly one. Use
`rpartition`, or anchor on a longer unique string. Verify the diff before
running anything:

```bash
git diff
```

### 3. Run the narrowest suite that should catch it

```bash
"$PROJECT_PYTHON" -m pytest tests/test_phase2_baselines.py -q
```

Full suite is ~79 s; a single file is usually 2-5 s.

### 4. Judge the failure, not just its existence

A mutation is caught only if:

- a test **fails** (not errors on an unrelated import, not skips), **and**
- the failure message would tell a reader **what broke**, not merely that an
  assertion was false.

A test that fails with `assert False` has technically caught it and has still
told you nothing. Note that as a weak test.

**A surviving mutation is a finding.** It means either the test is toothless or
the guard is redundant. Both have happened here - do not assume the test is at
fault. Investigate which before writing more test code.

### 5. Restore, always

```bash
git checkout -- <file>
git diff        # must be empty
```

Restore whether the mutation was caught or not, and verify it. A left-behind
mutation is far worse than an unverified guard.

### 6. Re-run the suite clean

```bash
"$PROJECT_PYTHON" -m pytest tests/ -q
```

Confirm you are back to green before reporting.

## Report

A table: mutation, what you expected, what actually happened, caught or
survived. Then, for each survivor, your diagnosis - weak test, or redundant
guard.

## Traps this suite has already hit

Check whether your new test has any of these shapes.

- **A vacuous test.** One used `context=50` against `MIN_CONTEXT=90`, so
  `_history_ending_at` returned `{}` and the assertion loop never ran. Two
  surviving mutations exposed it. Assert that the loop body executed.
- **A test that passed against the exact defect it was written to catch.** The
  `batch_size` guard used 40 names and asserted `batch_size >= n_inputs`, which
  a hardcoded 256 satisfies. It now runs 300.
- **Two guards covering one failure are one UNTESTABLE guard.** `first_seen`
  was preserved in both Python and a SQL `COALESCE`. Each mutation-tested alone
  left the suite green, because the other silently covered for it. The
  redundancy read as belt-and-braces and was a hole in the tests. One mechanism.
- **A redundant guard found by a survivor.** The probe had an explicit tie
  check fully subsumed by the `sd <= 0` test below it. Removed, not tested harder.
- **A defect in the CODE, not the test.** A surviving mutant exposed a
  `min(context_used, limit)` clamp that could only bind when the checkpoint's
  context was not a patch multiple - and produced a non-multiple context when
  it did. The test was fine; the code was wrong.
- **A module-scope probe runs at COLLECTION, not at test time.** A
  `_chronos_usable()` helper at module scope imported a half-reinstalled torch
  into `sys.modules` before a single test ran, and unrelated files across the
  suite failed together while each passed in isolation. Guards for optional
  heavy dependencies go INSIDE the test. Neither `find_spec` nor a bare
  `import` detects a mid-reinstall package - touch a real attribute.
- **Never grep source for a literal line.** One test asserted
  `"total_matching = len(df)" in source` and failed the moment the count moved
  into SQL with behaviour unchanged. Assert through the endpoint.
- **Carrying your own `CREATE TABLE`.** `tests/test_phase2_fundamentals.py`
  runs `data.db.init_db()` against in-memory SQLite instead, so a schema change
  that breaks the writer breaks the tests too. A test with its own DDL passes
  happily while the real schema is wrong - and writing it that way found a live
  defect, `load_fundamentals` not selecting `first_seen`.
