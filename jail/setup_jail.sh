#!/bin/bash
# Install the jail the agent's `shell` tool runs in (LA interface, --style la). Needs root:
#   sudo bash jail/setup_jail.sh
#
#   /opt/lffjail/jail_inner.sh   jail entry (copy of jail/jail_inner.sh); robot_server.py and gpt_player.py call it
#                                through `unshare --user --map-root-user --mount --net`
#   /opt/lffjail/work            mount point of the episode workspace inside the jail
#   /opt/lffjail/py              Python for the agent's own scripts: numpy, scipy, Pillow, OpenCV
#
# The path is fixed: the agent sees /opt/lffjail/work as its working directory, and the jail hides /home /root /tmp /mnt,
# so everything it needs must live under /opt (or /usr). For the same reason the base interpreter of /opt/lffjail/py
# must be a system Python (/usr/bin/python3 or /usr/local/bin/python3), not a conda / pyenv / uv one under $HOME.
# Override with JAIL_PYTHON=/usr/bin/python3.12 if needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
J=/opt/lffjail
[ "$(id -u)" = 0 ] || { echo "run as root: sudo bash $0"; exit 1; }

PY=${JAIL_PYTHON:-$(command -v python3 || true)}
[ -n "$PY" ] || { echo "no python3 found; install it (apt install python3 python3-venv) or set JAIL_PYTHON"; exit 1; }
REAL=$(readlink -f "$PY")
case "$REAL" in
  /usr/*|/opt/*|/bin/*) ;;
  *) echo "$PY resolves to $REAL, which is hidden inside the jail; use a system python (JAIL_PYTHON=/usr/bin/python3)"; exit 1 ;;
esac
for t in unshare setpriv mount timeout; do
  command -v $t >/dev/null || { echo "missing $t (util-linux / coreutils)"; exit 1; }
done

mkdir -p "$J/work"
install -m 755 "$HERE/jail_inner.sh" "$J/jail_inner.sh"

if [ ! -x "$J/py/bin/python" ]; then
  if "$PY" -m venv "$J/py" 2>/dev/null; then :; else
    echo "'$PY -m venv' failed (Debian/Ubuntu: apt install python3-venv); trying uv"
    UV=${UV:-$(command -v uv || true)}
    [ -n "$UV" ] || { echo "no uv either; install python3-venv or set UV=/path/to/uv"; exit 1; }
    "$UV" venv --python "$REAL" "$J/py"
  fi
fi
if [ -x "$J/py/bin/pip" ]; then
  "$J/py/bin/pip" install -q --upgrade numpy scipy pillow opencv-python-headless
else
  UV=${UV:-$(command -v uv)}
  "$UV" pip install -q --python "$J/py/bin/python" numpy scipy pillow opencv-python-headless
fi
chmod -R a+rX "$J"
"$J/py/bin/python" -c "import numpy, scipy, PIL, cv2; print('jail python ok:', numpy.__version__, scipy.__version__, PIL.__version__, cv2.__version__)"
echo "installed $J. Check it as the user that will run the benchmark: python scripts/check_setup.py"
