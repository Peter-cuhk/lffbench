# source this: environment for LFF-Bench (CPU MuJoCo + OSMesa by default)
LFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# machine-specific overrides (not in git): LFF_VENV, LIBERO_CONFIG_PATH, LFF_EXTRA_PYTHONPATH (LIBERO source), RPENT_DIR, ...
[ -f "$LFF_ROOT/env.local.sh" ] && source "$LFF_ROOT/env.local.sh"
source "${LFF_VENV:-$LFF_ROOT/.venv}/bin/activate"
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$LFF_ROOT/.libero}
# LIBERO is used from source (its setup.py does not install an importable `libero` package)
export PYTHONPATH=$LFF_ROOT:${LFF_EXTRA_PYTHONPATH:-$LFF_ROOT/third_party/LIBERO}
