#!/bin/bash
# Drive one robot_server session with the GPT-Policy harness (frameworks/gpt-policy, simulated
# lffbench_sim adapter + OpenAI Responses provider, GPT-6 Astra, reasoning effort medium).
#
# usage: scripts/gptpolicy_player.sh <LOG>/session.json [extra gpt-policy-lffbench options]
#   e.g. LFF_STYLE=v2 scripts/start_robot_session.sh l1_bias_place 3000 full F2 10340 direct 3 gptpolicy rgb 1
#        scripts/gptpolicy_player.sh runs/claude/gptpolicy_l1_bias_place_s3000_..._p10340/session.json
# The robot_server interface (v1 / v2) is detected from the first status reply.
#
# Writes into the session's LOG directory: gptpolicy_transcript.jsonl (every model request/response,
# images as file paths + sha256, every robot-server call), final_report.md, gptpolicy_player.log and
# GPT-Policy's own recording gptpolicy_run_*/. The robot server writes events.jsonl and rollout.mp4.
# The server is stopped at the end (it has finished the episode, and rollout.mp4 is closed by then).
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <LOG>/session.json [options]"; exit 2; }
SESSION=$(readlink -f "$1"); shift
LOG=$(dirname "$SESSION")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/env.local.sh" ] && source "$ROOT/env.local.sh"
# GPT-Policy checkout + venv from integrations/gpt-policy/setup.sh (override: GPTPOLICY_VENV)
VENV=${GPTPOLICY_VENV:-$ROOT/third_party/gpt-policy/.venv}
ENVF=${LFF_ENV_FILE:-$ROOT/.env}
[ -f "$ENVF" ] && { set -a; source "$ENVF"; set +a; }   # OPENAI_API_KEY / OPENAI_BASE_URL / SSL_CERT_FILE (never echoed)
export PYTHONUNBUFFERED=1
stop_server() {   # also on abnormal exit; SIGTERM makes the server close rollout.mp4 first
  if [ "${GPTPOLICY_KEEP_SERVER:-0}" != "1" ] && [ -f "$LOG/server.pid" ]; then
    kill "$(cat "$LOG/server.pid")" 2>/dev/null || true
  fi
}
trap stop_server EXIT
trap 'exit 143' TERM INT HUP
cd /tmp   # neutral working directory; the model has no file access anyway
"$VENV/bin/gpt-policy-lffbench" --session "$SESSION" --model "${GPTPOLICY_MODEL:-gpt-6-astra}" \
  --effort "${GPTPOLICY_EFFORT:-medium}" "$@" 2>&1 | tee -a "$LOG/gptpolicy_player.log"
RC=${PIPESTATUS[0]}
echo "gptpolicy_player.sh: gpt-policy-lffbench exit code $RC" >> "$LOG/gptpolicy_player.log"
exit $RC
