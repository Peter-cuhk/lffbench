#!/bin/bash
# Start a robot server + isolated sandbox for one task instance.
# usage: scripts/start_robot_session.sh <task> <seed> <memory full|none> <feedback F0|F1|F2> <port> [instruction direct|indirect] [k] [tag] [perception depth|rgb] [video 0|1]
# The sandbox gets an opaque random name (no task / condition in the path); the readable run name is only in LOG.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK=$1; SEED=$2; MEM=$3; FB=$4; PORT=$5; INSTR=${6:-direct}; K=${7:-}; TAG=${8:-}; PERC=${9:-depth}; VID=${10:-0}
RUN=${TAG:+${TAG}_}${TASK}_s${SEED}_${MEM}_${FB}_${INSTR}_k${K:-d}_$(date +%m%d-%H%M%S)_p${PORT}
SANDBOX=${LFF_SANDBOX_ROOT:-/tmp/robotlab}/$(python3 -c 'import secrets;print(secrets.token_hex(6))')
LOG=${LFF_RUNS_DIR:-$ROOT/runs/claude}/$RUN
mkdir -p "$SANDBOX" "$LOG"
cd "$ROOT" && source run_env.sh && export OMP_NUM_THREADS=1
KARG=""; [ -n "$K" ] && KARG="--k $K"
VARG=""; [ "$VID" = "1" ] && VARG="--video $LOG/rollout.mp4"
nohup python -m lffbench.agent.robot_server --task "$TASK" --seed "$SEED" --memory "$MEM" --feedback "$FB" \
  --instruction "$INSTR" --port "$PORT" --sandbox "$SANDBOX" --log-dir "$LOG" --perception "$PERC" --style "${LFF_STYLE:-v1}" ${LFF_EXTRA_ARGS:-} $KARG $VARG > "$LOG/server.log" 2>&1 &
echo $! > "$LOG/server.pid"
for i in $(seq 1 120); do [ -f "$LOG/READY" ] && break; sleep 1; done
[ -f "$LOG/READY" ] || { echo "server failed"; tail -20 "$LOG/server.log"; exit 1; }
echo "{\"run\": \"$RUN\", \"sandbox\": \"$SANDBOX\", \"port\": $PORT}" > "$LOG/session.json"
echo "SANDBOX=$SANDBOX"
echo "LOG=$LOG"
