#!/usr/bin/env bash
# Launch the Qwen3-VL OpenAI-compatible server on the local PPU.
# Usage: bash scripts/qwen_vl_server.sh [--port 8765] [--host 127.0.0.1] [other qwen_vl_server.py args]
# Uses the system python3.12 (torch 2.6 PPU build) + a private overlay dir that only
# holds transformers==4.57.6 / tokenizers==0.22.1 / huggingface_hub==0.36.0; the
# overlay is put on PYTHONPATH for this process only.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PPU runtime libs (libhggcrt1.so etc.); without this torch spins printing load errors.
source /usr/local/PPU_SDK/envsetup.sh cuda >/dev/null
[ -n "${QWEN_OVERLAY:-}" ] && export PYTHONPATH="${QWEN_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
exec /usr/local/bin/python3.12 "${HERE}/qwen_vl_server.py" "$@"
