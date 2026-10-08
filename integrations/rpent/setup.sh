#!/bin/bash
# Optional: the RPent harness (Harness VLA, https://github.com/RLinf/RPent) driving LFF-Bench sessions.
# Clones RPent at the pinned commit into third_party/rpent, applies lffbench.patch (adds robots/lffbench + a small
# api_loop.py change) and makes third_party/rpent/.venv (Python 3.12, exact package versions from requirements.lock.txt).
# Then: scripts/rpent_player.sh <session.json>   (see README "RPent / GPT-Policy").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
URL=https://github.com/RLinf/RPent.git
COMMIT=068cd64f4f175179a13c67d429a4d00ca56fb196
DST=${RPENT_DIR:-$ROOT/third_party/rpent}
command -v uv >/dev/null || { echo "needs uv (https://docs.astral.sh/uv/)"; exit 1; }
if [ ! -d "$DST/.git" ]; then
  git clone -q "$URL" "$DST"
  git -C "$DST" checkout -q "$COMMIT"
  git -C "$DST" apply "$HERE/lffbench.patch"
fi
uv venv -q --python 3.12 "$DST/.venv"
uv pip install -q --python "$DST/.venv/bin/python" -r "$HERE/requirements.lock.txt"
uv pip install -q --python "$DST/.venv/bin/python" --no-deps -e "$DST"
cd "$DST" && PYTHONPATH="$DST" "$DST/.venv/bin/python" -c "import robots.lffbench.player, rpent; print('rpent ok')"
