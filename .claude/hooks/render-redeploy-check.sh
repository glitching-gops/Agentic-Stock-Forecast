#!/usr/bin/env bash
#
# PostToolUse(Edit|Write) - Render redeploy reminder.
#
# WHY THIS EXISTS
#   Render does not auto-redeploy in this workflow. When an edit lands on
#   something the live API imports, the deployed instance keeps serving the
#   OLD code with no error anywhere. That has already cost a debugging round:
#   the API silently served stale code across three rounds while a phantom bug
#   was chased. See CLAUDE.md section 2, working agreement 2.
#
# THE TRIGGER SET IS MEASURED, NOT GUESSED
#   Filtering every import in api/ + main.py yields exactly three non-api
#   modules: data.db, data.tickers, data.universe. requirements.txt is
#   included because Render installs it on deploy. Nothing else in data/ or
#   pipeline/ is reachable from a request, so nothing else belongs here.
#   Re-derive after any refactor with:
#     grep -rhE '^[[:space:]]*(from|import)[[:space:]]' api/ main.py | sort -u
#
# NOTE: jq is NOT installed in this environment. The path is pulled out of the
# hook's stdin JSON with sed. Any rewrite that reaches for jq will silently
# match nothing, and this guard becomes decoration that reads as protection.

set -u

payload=$(cat)

# JSON -> path, then normalise Windows separators so one pattern set matches
# both an absolute "C:\Users\...\api\main.py" and a relative "api/main.py".
# JSON escapes each backslash, so the captured text holds doubles; tr turns
# every one into a slash and the following sed collapses the resulting runs.
path=$(printf '%s' "$payload" \
  | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  | head -1 \
  | tr '\\' '/' \
  | sed 's|//*|/|g')

[ -n "$path" ] || exit 0

case "$path" in
  */api/*|api/*|*/data/db.py|data/db.py|*/data/tickers.py|data/tickers.py|*/data/universe.py|data/universe.py|*/requirements.txt|requirements.txt) ;;
  *) exit 0 ;;
esac

# Trim to a repo-relative path for a readable message.
rel=${path##*/Agentic-Stock-Forecast/}

cat <<JSON
{
  "systemMessage": "RENDER: $rel is on the live API's import path - the deployed instance serves stale code until you redeploy manually.",
  "suppressOutput": true,
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "RENDER REDEPLOY REQUIRED: you just edited $rel, which the live FastAPI service on Render imports. Render does NOT auto-redeploy in this user's workflow. Before you finish this turn, tell the user explicitly that a manual Render redeploy is needed for this change to take effect. Do not omit it because the change looks small - the failure mode is silent, and it has already cost a debugging round."
  }
}
JSON
