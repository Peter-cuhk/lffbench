#!/bin/bash
# Run one RPent (Harness VLA framework) agent on an LFF-Bench robot_server session.
# usage: scripts/rpent_player.sh <run log dir>/session.json [--model gpt-6-astra] [--effort medium] [--max-turns 400] ...
#   Start a v2 session first (the RPent robot `lffbench` speaks interface v2 only), e.g.:
#   LFF_STYLE=v2 scripts/start_robot_session.sh l1_bias_place 3000 full F2 10330 direct 3 rpent rgb 1
# RPent checkout with the LFF-Bench robot (`lffbench`, planner `api`): integrations/rpent/setup.sh puts it in
# third_party/rpent with its venv in third_party/rpent/.venv (override: RPENT_DIR / RPENT_VENV, e.g. in env.local.sh).
# API key / endpoint: OPENAI_API_KEY / OPENAI_BASE_URL from the environment or from .env (LFF_ENV_FILE).
# Writes into the session's log dir: rpent_transcript.jsonl (every model request/response, images as file refs),
# final_report.md, rpent_result.json, rpent_player.log and rpent/ (RPent's own run.log / transcript / step artifacts).
set -e
[ -f "$1" ] || { echo "usage: $0 <session.json> [player args]"; exit 2; }
SESSION=$(readlink -f "$1"); shift
LOG=$(dirname "$SESSION")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/env.local.sh" ] && source "$ROOT/env.local.sh"
RPENT=${RPENT_DIR:-$ROOT/third_party/rpent}
VENV=${RPENT_VENV:-$RPENT/.venv}
ENVF=${LFF_ENV_FILE:-$ROOT/.env}
[ -f "$ENVF" ] && { set -a; source "$ENVF"; set +a; }
cd "$RPENT"
PYTHONPATH="$RPENT" "$VENV/bin/python" -m robots.lffbench.player --session "$SESSION" "$@" 2>&1 | tee -a "$LOG/rpent_player.log"
exit "${PIPESTATUS[0]}"
