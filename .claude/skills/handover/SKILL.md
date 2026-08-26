---
name: handover
description: Prepare a commit for the user to run themselves - stage the work, run the suite, check the Render import path and the repo's landmines, then print the exact git commands. Never commits or pushes.
disable-model-invocation: true
---

# Handover

Working agreement 1 of CLAUDE.md, stated by the user verbatim:

> "everytime you want me to commit and push a change, give me proper commands
> to avoid confusion."

**The user runs every git commit and push.** This skill exists to make that
handover complete and checked, not to automate it away.

## Absolute constraint

**Never run `git commit`. Never run `git push`. Never run `git tag`.**

Not with `--dry-run`, not "just to check the message". Staging (`git add`) is
allowed and is the point. If you catch yourself constructing a commit command
to execute rather than to print, stop.

## Interpreter

Python is not on PATH. Use `$PROJECT_PYTHON`, or
`C:/Users/venuw/AppData/Local/Programs/Python/Python313/python.exe`.

## Steps

### 1. Survey what changed

```bash
git status --short
git diff --stat
git diff --cached --stat
```

Separate: modified tracked files, new untracked files, and anything staged
already. Name every file you intend to include. If something in the working
tree is unrelated to this piece of work, say so and leave it out - a commit
that sweeps in an unrelated file is the user's problem to untangle later.

### 2. Run the suite

```bash
"$PROJECT_PYTHON" -m pytest tests/ -q
```

~79 s, 311 tests at last count. Report the real result. If tests fail, say so
with the output and **stop** - do not print commit commands for a red suite
unless the user explicitly asks for them anyway.

### 3. Check the Render import path

The measured import surface of the live API is exactly:

- `api/**`
- `data/db.py`, `data/tickers.py`, `data/universe.py`
- `requirements.txt` (Render installs it on deploy)

```bash
git status --short | grep -E '(^| )(api/|data/db\.py|data/tickers\.py|data/universe\.py|requirements\.txt)'
```

If anything matches, the handover **must** carry an explicit line telling the
user to redeploy Render manually. Render does not auto-redeploy in this
workflow; missing it once meant the live API served stale code across three
rounds of debugging a phantom bug.

If nothing matches, say so explicitly too - "no Render redeploy needed" is
useful information, and its absence reads as an oversight.

### 4. Check the landmines this diff could have tripped

Only the ones the diff actually touches:

- **`SECTOR_INDICES` changed?** The benchmark mapping is half the label -
  `target_excess_return` is the stock's return MINUS its benchmark's, so
  editing it silently redefines every historical target. `MODEL_VERSION` in
  `pipeline/model.py` must be bumped in the same commit, or a run before and
  after is indistinguishable in `experiment_runs` and stale `eval_*` metrics
  keep backing new forecasts. Also re-run
  `"$PROJECT_PYTHON" tools/audit_benchmarks.py --apply-check`.
- **A new guard added?** It should be mutation-verified. Offer `/mutate`.
- **A path that writes signals?** Confirm the F6 monotonicity guard is intact -
  `_upsert_signals` is DELETE-range + reinsert, and a frame with NULL targets
  destroys labels.
- **`torch` in `requirements.txt`?** Never. It belongs in
  `requirements-series.txt` only.
- **New generated artifacts?** `*.npz`, `*.db`, `logs/`, `tuned_params/` are
  gitignored. Confirm nothing large slipped into the staged set:
  `git diff --cached --stat | tail -1`.

### 5. Print the commands

Give one fenced block the user can paste, with real filenames - never `<file>`
placeholders, never `git add -A` unless every changed file genuinely belongs.

```bash
git add path/one.py path/two.py
git commit -m "type: what changed and why"
git push origin main
```

Commit message: the repo's existing style is a conventional-commit prefix and a
lowercase summary - `feat:`, `fix:`, `audit:`, `test:`, `docs:`. Look at
`git log --oneline -10` and match it.

**`CLAUDE.md` and `PLAN.md` are gitignored by the user's explicit instruction**
("keep claude.md ignored for now"). Never include them, and never suggest
un-ignoring them.

### 6. State the follow-ups

After the command block, in plain sentences:

- Whether a Render redeploy is required (from step 3), and that it is manual.
- Whether a GitHub Actions workflow will pick the change up on its next cron
  run, and when that is - daily weekdays 18:30 IST, weekly Saturday 08:30 IST.
- Anything you deliberately left out of the commit, and why.
