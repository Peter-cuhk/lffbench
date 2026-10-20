#!/bin/bash
# Install the jail for native agent harnesses (Claude Code, Codex; docs/15 §3.4). Needs root:
#   sudo bash jail/setup_harness_jail.sh
#
#   /opt/lffjail/hjail/inner.sh    jail entry (copy of jail/harness_jail_inner.sh), started by lffbench/harnesses/jail.py
#                                  through `unshare --user --map-root-user --mount --pid --fork --kill-child --mount-proc`
#   /opt/lffjail/hjail/mcp_fwd.py  stdio <-> Unix socket relay the jailed harness starts as its MCP server
#   /opt/lffjail/hjail/stage/N     mount points used while the jail is being built (empty)
# Everything lives under /opt because the jail hides /mnt /home /root /tmp. The relay runs on the system python3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
J=/opt/lffjail/hjail
[ "$(id -u)" = 0 ] || { echo "run as root: sudo bash $0"; exit 1; }
for t in unshare setpriv mount; do
  command -v $t >/dev/null || { echo "missing $t (util-linux)"; exit 1; }
done
[ -x /usr/bin/python3 ] || { echo "/usr/bin/python3 is needed inside the jail for the MCP relay"; exit 1; }
mkdir -p "$J"
for i in $(seq 0 15); do mkdir -p "$J/stage/$i"; done
install -m 755 "$HERE/harness_jail_inner.sh" "$J/inner.sh"
install -m 644 "$HERE/mcp_fwd.py" "$J/mcp_fwd.py"
echo "installed $J"
