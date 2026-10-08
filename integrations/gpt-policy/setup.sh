#!/bin/bash
# Optional: the GPT-Policy harness (https://github.com/cheng-haha/GPT-Policy) driving LFF-Bench sessions.
# Clones GPT-Policy at the pinned commit into third_party/gpt-policy, applies lffbench.patch (adds the lffbench_sim
# adapter, the LFF-Bench protocol, an OpenAI Responses provider and the `gpt-policy-lffbench` entry point) and makes
# third_party/gpt-policy/.venv (Python 3.10, exact package versions from requirements.lock.txt).
# GPT-Policy's LICENSE says it is published for review only, so its source is not copied into this repo.
# Then: scripts/gptpolicy_player.sh <session.json>   (see README "RPent / GPT-Policy").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
URL=https://github.com/cheng-haha/GPT-Policy.git
COMMIT=3a74ed4b45d25ff9af4939c9209292e725712f50
DST=${GPTPOLICY_DIR:-$ROOT/third_party/gpt-policy}
command -v uv >/dev/null || { echo "needs uv (https://docs.astral.sh/uv/)"; exit 1; }
if [ ! -d "$DST/.git" ]; then
  git clone -q "$URL" "$DST"
  git -C "$DST" checkout -q "$COMMIT"
  git -C "$DST" apply "$HERE/lffbench.patch"
fi
uv venv -q --python 3.10 "$DST/.venv"
uv pip install -q --python "$DST/.venv/bin/python" -r "$HERE/requirements.lock.txt"
uv pip install -q --python "$DST/.venv/bin/python" --no-deps -e "$DST"
"$DST/.venv/bin/gpt-policy-lffbench" --help >/dev/null && echo "gpt-policy ok"
