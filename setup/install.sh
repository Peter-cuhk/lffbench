#!/bin/bash
# One-time setup of LFF-Bench on a Linux machine:   bash setup/install.sh
#   .venv/                  Python 3.10 venv (uv downloads Python 3.10 if the machine has none)
#   third_party/LIBERO/     upstream LIBERO at a pinned commit (≈640 MB: robot / table / object assets)
#   .libero/config.yaml     LIBERO's path config (otherwise its first import asks questions on stdin)
# Needs: git, uv (https://docs.astral.sh/uv/), and for CPU rendering the OSMesa library (Ubuntu: apt install libosmesa6).
# The agent's shell jail is separate and needs root: sudo bash jail/setup_jail.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LIBERO_URL=https://github.com/Lifelong-Robot-Learning/LIBERO.git
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
LIB=third_party/LIBERO
command -v uv >/dev/null || { echo "needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

if [ ! -d "$LIB/libero/libero/assets" ]; then
  echo "[1/4] LIBERO @ ${LIBERO_COMMIT:0:7} -> $LIB"
  rm -rf "$LIB"; mkdir -p "$LIB"
  git -C "$LIB" init -q
  git -C "$LIB" remote add origin "$LIBERO_URL"
  git -C "$LIB" fetch -q --depth 1 origin "$LIBERO_COMMIT"
  git -C "$LIB" checkout -q FETCH_HEAD
else
  echo "[1/4] LIBERO already in $LIB"
fi

echo "[2/4] venv .venv (Python 3.10)"
[ -x .venv/bin/python ] || uv venv -q --python 3.10 .venv
uv pip install -q --python .venv/bin/python -r setup/requirements.txt -c setup/requirements.lock.txt
uv pip install -q --python .venv/bin/python --no-deps robosuite==1.4.1   # LIBERO itself: on PYTHONPATH (run_env.sh)

echo "[3/4] .libero/config.yaml"
B="$ROOT/$LIB/libero/libero"
mkdir -p .libero
cat > .libero/config.yaml <<CFG
benchmark_root: $B
bddl_files: $B/bddl_files
init_states: $B/init_files
datasets: $B/../datasets
assets: $B/assets
CFG

echo "[4/4] import check"
set +u; source run_env.sh; set -u
python -c "import mujoco, robosuite, libero.libero, openai; print('ok: mujoco', mujoco.__version__, '| robosuite', robosuite.__version__)" 2>&1 | { grep -v WARNING || true; }
test "${PIPESTATUS[0]}" = 0
echo
echo "Done. Next:"
echo "  sudo bash jail/setup_jail.sh                 # the agent's shell jail (LA interface)"
echo "  cp .env.example .env && edit .env            # OPENAI_API_KEY / OPENAI_BASE_URL"
echo "  source run_env.sh && python scripts/check_setup.py --api"
