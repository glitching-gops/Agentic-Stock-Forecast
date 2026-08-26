#!/usr/bin/env bash
#
# PostToolUse(Edit|Write) on pipeline/*.py - module-load invariant guard.
#
# WHY THIS EXISTS
#   Two guards in this repo protect results that took days to establish, and
#   both fail SILENTLY until something imports the module:
#
#     1. pipeline/baselines.py raises ImportError at module load if FACTORS
#        picks up a column that is not in panel.SCALE_FREE. Pooled across
#        tickers, a price-denominated column IS ticker identity, so a linear
#        model on it fits the name rather than the signal.
#
#     2. torch must never reach the shared import path. Render, the daily
#        pipeline and the weekly evaluation all install requirements.txt and
#        none of them call the foundation-model code; torch was removed in
#        Phase 0 as the largest contributor to memory pressure on an instance
#        that had already been OOM-killed. chronos_forecaster imports it
#        INSIDE load_pipeline precisely so pipeline.baselines cannot pull it in.
#
#   Measured cost: 1.75 s. The full suite is 79 s, which is far too slow for a
#   per-edit hook - this is the subset that is cheap enough to always run.
#
# EXIT 2 is deliberate: it feeds stderr back to the model as a blocking error,
# so a broken invariant gets fixed in the same turn rather than surfacing in CI.

set -u

payload=$(cat)

path=$(printf '%s' "$payload" \
  | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  | head -1 \
  | tr '\\' '/' \
  | sed 's|//*|/|g')

[ -n "$path" ] || exit 0

case "$path" in
  */pipeline/*.py|pipeline/*.py) ;;
  *) exit 0 ;;
esac

# Repo root is two levels up from this script (.claude/hooks/x.sh).
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root" || exit 0

# Resolve an interpreter. python is NOT on PATH on this machine, which is why
# PROJECT_PYTHON is set in .claude/settings.local.json. If none resolves, SAY
# SO - a guard that quietly skips itself reads as a guard that passed.
py=""
if [ -n "${PROJECT_PYTHON:-}" ] && [ -x "${PROJECT_PYTHON}" ]; then
  py="${PROJECT_PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  py=python3
elif command -v python >/dev/null 2>&1; then
  py=python
fi

if [ -z "$py" ]; then
  cat <<'JSON'
{
  "systemMessage": "pipeline invariant check SKIPPED - no Python interpreter found. Set PROJECT_PYTHON in .claude/settings.local.json.",
  "suppressOutput": true
}
JSON
  exit 0
fi

out=$("$py" -c "
import sys
import pipeline.baselines
import pipeline.panel
assert 'torch' not in sys.modules, (
    'torch reached the shared import path via pipeline.baselines/panel. '
    'Render, the daily job and the weekly job all import these and none of '
    'them need torch; it was removed in Phase 0 as the largest contributor '
    'to the OOM. Keep the import inside load_pipeline.'
)
" 2>&1)
status=$?

if [ $status -ne 0 ]; then
  echo "pipeline invariant check FAILED after editing $path" >&2
  echo "$out" >&2
  exit 2
fi

exit 0
