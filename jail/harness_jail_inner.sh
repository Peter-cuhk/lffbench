#!/bin/bash
# Jail for a native agent harness process (Claude Code, Codex), docs/15 §3.4. Started by lffbench/harnesses/jail.py as
#   setpriv --reuid=65534 --regid=65534 --clear-groups -- unshare --user --map-root-user --mount --fork --kill-child \
#     /opt/lffjail/hjail/inner.sh ...
# i.e. as namespace-root that is the unprivileged host user nobody. Unlike jail_inner.sh (the agent's own shell) it
# keeps the network, because the harness must reach its model API. What it loses is every host path: /mnt /home /root
# /tmp /run/secrets /etc/dsw become empty tmpfs and system dirs read-only. Only the directories given with --rw (the
# one-time work and state dirs, the socket dir of the MCP relay; the MCP server itself runs outside) are mounted back at
# their original paths; --ro does the same read-only. The harness binary runs from its copy under /opt/lffjail/hjail/bin.
#
#   inner.sh --keep K1,K2,... --proc static|host [--ro DIR]... [--rw DIR]... --cwd DIR [--exe BINARY] -- CMD ARGS...
# --proc static: /proc becomes a read-only tmpfs with static copies of host-wide files (CPU, memory, kernel version) and
#   this process's own self/ entries, no other pid; --exe makes /proc/self/exe point to the harness binary.
# --proc host: the host procfs stays (needed by Bun executables such as Claude Code; a fresh procfs cannot be mounted in
#   this container because its /proc has locked submounts). As uid 65534 the harness can then read other processes'
#   command lines, but not their environ, fd, cwd or root.
# The command gets exactly the variables named in --keep (taken from this script's environment), nothing else.
set -e
STAGE=/opt/lffjail/hjail/stage
keep=; cwd=/; ro=(); rw=(); exe=; proc=static
while [ $# -gt 0 ]; do
  case "$1" in
    --keep) keep=$2; shift 2 ;;
    --proc) proc=$2; shift 2 ;;
    --ro) ro+=("$2"); shift 2 ;;
    --rw) rw+=("$2"); shift 2 ;;
    --cwd) cwd=$2; shift 2 ;;
    --exe) exe=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "harness_jail_inner: bad argument $1" >&2; exit 97 ;;
  esac
done
[ $# -gt 0 ] || { echo "harness_jail_inner: no command" >&2; exit 97; }
case "$proc" in static|host) ;; *) echo "harness_jail_inner: --proc static|host" >&2; exit 97 ;; esac
mount --make-rprivate /
for d in /usr /etc /opt /bin /lib /lib64 /sbin /var /srv; do
  [ -d "$d" ] && [ ! -L "$d" ] && { mount --rbind "$d" "$d" && mount -o remount,bind,ro "$d"; } 2>/dev/null || true
done
# Stage the directories to keep (their host paths are still visible here), then hide the host trees.
srcs=(); modes=(); i=0
for s in "${ro[@]}"; do mount --bind "$s" "$STAGE/$i"; srcs+=("$s"); modes+=(ro); i=$((i+1)); done
for s in "${rw[@]}"; do mount --bind "$s" "$STAGE/$i"; srcs+=("$s"); modes+=(rw); i=$((i+1)); done
for d in /mnt /home /root /tmp /run/secrets /etc/dsw; do
  [ -d "$d" ] && mount -t tmpfs -o size=256m,mode=1777 none "$d"
done
for j in "${!srcs[@]}"; do
  s=${srcs[$j]}
  [ -d "$s" ] || mkdir -p "$s"
  mount --bind "$STAGE/$j" "$s"
  [ "${modes[$j]}" = ro ] && mount -o remount,bind,ro "$s"
  umount -l "$STAGE/$j"
done
# Last, because mount / umount read /proc/self/mountinfo.
if [ "$proc" = static ]; then
  P=$(mktemp -d /tmp/.lffproc.XXXXXX)
  mkdir -p "$P/self/fd" "$P/self/ns" "$P/sys/kernel" "$P/sys/vm" "$P/sys/fs"
  for f in version stat cpuinfo meminfo loadavg uptime; do cat /proc/$f > "$P/$f" 2>/dev/null || true; done
  for f in stat statm status cgroup uid_map gid_map setgroups limits smaps_rollup mountinfo; do
    cat /proc/self/$f > "$P/self/$f" 2>/dev/null || true
  done
  cp "$P/self/mountinfo" "$P/mounts"
  for f in kernel/overflowuid kernel/overflowgid kernel/pid_max vm/overcommit_memory vm/mmap_min_addr fs/file-max fs/nr_open; do
    cat /proc/sys/$f > "$P/sys/$f" 2>/dev/null || true
  done
  [ -n "$exe" ] && ln -s "$exe" "$P/self/exe"
  mount -t tmpfs -o size=4m,mode=755 none /proc
  cp -a "$P/." /proc/
  rm -rf "$P"
  mount -o remount,ro /proc
fi
cd "$cwd"
envargs=()
IFS=, read -ra ks <<< "$keep"
for k in "${ks[@]}"; do
  [ -n "$k" ] && [ -n "${!k+x}" ] && envargs+=("$k=${!k}")
done
exec setpriv --inh-caps=-all --bounding-set=-all --securebits=+noroot,+noroot_locked,+no_setuid_fixup,+no_setuid_fixup_locked,+keep_caps_locked --no-new-privs \
  env -i "${envargs[@]}" "$@"
