#!/bin/bash
# Runs as namespace-root inside `unshare --user --map-root-user --mount --net` (see lffbench/agent/robot_server.py SHELL_TEMPLATE, scripts/gpt_player.py run_shell).
# $1 = the episode workspace (bound read-write at /opt/lffjail/work); the rest = the command to run.
# Inside: no network (empty net namespace), host data hidden (/mnt /home /root /tmp /proc /run/secrets
# /etc/dsw are empty tmpfs), system dirs read-only, all capabilities dropped and locked, clean environment.
set -e
WS=$1; shift
mount --make-rprivate /
mount --bind "$WS" /opt/lffjail/work
for d in /usr /etc /opt /bin /lib /lib64 /sbin /var /srv; do
  [ -d "$d" ] && [ ! -L "$d" ] && { mount --rbind "$d" "$d" && mount -o remount,bind,ro "$d"; } 2>/dev/null || true
done
for d in /mnt /home /root /tmp /run/secrets /etc/dsw /proc; do
  [ -d "$d" ] && mount -t tmpfs -o size=256m,mode=1777 none "$d" 2>/dev/null || true
done
cd /opt/lffjail/work
exec setpriv --inh-caps=-all --bounding-set=-all --securebits=+noroot,+noroot_locked,+no_setuid_fixup,+no_setuid_fixup_locked,+keep_caps_locked --no-new-privs \
  env -i PATH=/opt/lffjail/py/bin:/usr/bin:/bin HOME=/opt/lffjail/work MPLCONFIGDIR=/tmp LANG=C.UTF-8 "$@"
