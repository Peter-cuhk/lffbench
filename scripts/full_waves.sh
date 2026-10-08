#!/bin/bash
# Run the remaining waves of the full GPT-6 run: wait until at most MAX_LEFT (default 4) runs of the batch are still going, stop the
# robot servers of ended runs (keeps memory bounded), then launch the next wave on all three harnesses.
# usage: scripts/full_waves.sh <batch> "<wave2 task:seed ...>" "<wave3 ...>" ...
cd "$(dirname "$0")/.."
B=$1; shift
reap() {
  python3 - "$B" <<'PY'
import json, os, signal, sys, subprocess
b = sys.argv[1]
m = json.load(open(f"runs/gpt/{b}/manifest.json"))
st = subprocess.run(["python3", "scripts/gpt_batch.py", "status", "--batch", b], capture_output=True, text=True).stdout.splitlines()
running = 0
for r, line in zip(m["runs"], st):
    if line.rstrip().endswith("running"):
        running += 1
        continue
    try:
        pid = int(open(os.path.join(r["log"], "server.pid")).read())
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
print(running)
PY
}
wait_below() { while true; do n=$(reap | tail -1); [ "$n" -le "$1" ] && break; sleep 120; done; }
port=${PORT0:-9900}
for W in "$@"; do
  wait_below ${MAX_LEFT:-4}
  echo "$(date -u +%H:%M) launching wave: $W"
  for h in gpt_player rpent gptpolicy; do
    python3 scripts/gpt_batch.py start --batch $B --harness $h --model gpt-6-astra --effort medium --perception rgb \
      --style ${STYLE:-v1} --video --k 3 --port0 $port --tasks $W | grep -c '"run"'
    port=$((port + 30))
  done
done
wait_below 0
echo "$(date -u +%H:%M) all waves done"
