"""Jail for native harness processes (docs/15 §3.4): the harness sees no host path at all, instead of only "having no
tool that reads files".

Without it, Claude Code / Codex run as root on the host with /mnt visible (task code with the hidden-variable rules,
task cards, earlier runs' logs, .secrets); isolation then rests on every built-in tool being switched off. With it
(adapter option jail=True, `harness_run.py --jail`, `api_batch.py start --jail`):
  - the harness runs as the unprivileged host user nobody (uid 65534) in new user + mount namespaces
    (jail/harness_jail_inner.sh, installed under /opt/lffjail/hjail by jail/setup_harness_jail.sh): /mnt /home /root
    /tmp /run/secrets /etc/dsw are empty tmpfs, system dirs read-only, all capabilities dropped; the network stays
    (model API);
  - /proc: "static" (Codex) = a read-only tmpfs with static copies of host-wide files (CPU, memory, kernel version) and
    the process's own self/ entries, no other pid; "host" (Claude Code, a Bun executable that aborts without a real
    procfs; a fresh procfs cannot be mounted in this container because its /proc has locked submounts) = the host
    procfs, where as uid 65534 the harness can list processes and read their command lines but not their environ, fd,
    cwd or root (checked in tests/harness_selfcheck/jail_probe.py);
  - visible again, at their original paths: the one-time work dir, the one-time state dir and the socket dir of the
    MCP relay (chowned to uid 65534); the harness binary runs from a verified copy of its release dir under
    /opt/lffjail/hjail/bin (the original lives under 0700 dirs in /mnt);
  - the MCP server (scripts/mcp_robot_server.py) runs outside the jail; the harness starts the relay
    /opt/lffjail/hjail/mcp_fwd.py, which connects to a Unix socket served here: every connection gets its own MCP
    server process with stdin / stdout on the socket and the environment {PATH} only.
The harness command line (except argv[0] = the copy), its environment (exact whitelist) and the paths it sees are
unchanged; only the MCP server entry of its config points to the relay.
"""
import hashlib
import os
import secrets
import shutil
import socket
import subprocess
import threading

JAIL_DIR = "/opt/lffjail/hjail"
INNER = os.path.join(JAIL_DIR, "inner.sh")
RELAY = os.path.join(JAIL_DIR, "mcp_fwd.py")
BIN_DIR = os.path.join(JAIL_DIR, "bin")
RELAY_PYTHON = "/usr/bin/python3"
SOCK_ROOT = "/tmp/lffsock"
JAIL_UID = 65534
MAX_BINDS = 16


class JailError(RuntimeError):
    pass


def installed_problems():
    p = []
    for f in (INNER, RELAY):
        if not os.path.exists(f):
            p.append(f"{f} missing (sudo bash jail/setup_harness_jail.sh)")
    if not os.path.isdir(os.path.join(JAIL_DIR, "stage", str(MAX_BINDS - 1))):
        p.append(f"{JAIL_DIR}/stage/0..{MAX_BINDS - 1} missing (sudo bash jail/setup_harness_jail.sh)")
    if not os.access(RELAY_PYTHON, os.X_OK):
        p.append(f"{RELAY_PYTHON} missing")
    for t in ("unshare", "setpriv"):
        if not shutil.which(t):
            p.append(f"{t} not found")
    return p


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _outermost(dirs):
    """Drop directories that lie inside another one of the list (one bind mount covers them)."""
    ds = sorted({os.path.realpath(d) for d in dirs if d}, key=len)
    out = []
    for d in ds:
        if not any(d == o or d.startswith(o.rstrip("/") + "/") for o in out):
            out.append(d)
    return out


def _chown_tree(root, uid):
    os.chown(root, uid, uid)
    for dp, dn, fn in os.walk(root):
        for n in dn + fn:
            try:
                os.lchown(os.path.join(dp, n), uid, uid)
            except FileNotFoundError:
                pass


def binary_copy(binary):
    """A verified copy of the binary's release dir under /opt/lffjail/hjail/bin (readable by uid 65534); returns
    (path of the binary in the copy, info). The release dir is the binary's dir, or its parent if that is a bin/."""
    real = os.path.realpath(binary)
    if not os.path.isfile(real):
        raise JailError(f"harness binary {binary} -> {real} not found")
    sha = sha256_file(real)
    rel_root = os.path.dirname(real)
    if os.path.basename(rel_root) == "bin":
        rel_root = os.path.dirname(rel_root)
    dst_root = os.path.join(BIN_DIR, f"{os.path.basename(rel_root)}-{sha[:12]}")
    dst = os.path.join(dst_root, os.path.relpath(real, rel_root))
    if not (os.path.isfile(dst) and sha256_file(dst) == sha):
        os.makedirs(BIN_DIR, mode=0o755, exist_ok=True)
        tmp = dst_root + ".tmp-" + secrets.token_hex(3)
        shutil.copytree(rel_root, tmp, symlinks=True)
        for dp, dn, fn in os.walk(tmp):
            os.chmod(dp, 0o755)
        if os.path.exists(dst_root):
            shutil.rmtree(dst_root)
        os.rename(tmp, dst_root)
        if sha256_file(dst) != sha:
            raise JailError(f"copy of {real} does not match its sha256")
    return dst, dict(original=real, copy=dst, sha256=sha)


class _Relay:
    """Serves one Unix socket; each connection gets a fresh MCP server process attached to it."""

    def __init__(self, sock_path, argv, env, stderr_path, uid):
        self.path, self.argv, self.env, self.stderr_path = sock_path, argv, env, stderr_path
        self.procs = []
        self.closed = False
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(sock_path)
        os.chown(sock_path, uid, uid)
        os.chmod(sock_path, 0o600)
        self.srv.listen(8)
        self.t = threading.Thread(target=self._serve, daemon=True)
        self.t.start()

    def _serve(self):
        while not self.closed:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            try:
                err = open(self.stderr_path, "ab") if self.stderr_path else subprocess.DEVNULL
                p = subprocess.Popen(self.argv, stdin=conn.fileno(), stdout=conn.fileno(), stderr=err, env=self.env,
                                     cwd="/", start_new_session=True)
                self.procs.append(p)
                if err is not subprocess.DEVNULL:
                    err.close()
            finally:
                conn.close()

    def close(self):
        self.closed = True
        try:
            self.srv.close()
        except OSError:
            pass
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


class HarnessJail:
    """One per adapter instance. mcp_command() registers a real MCP server command and returns the relay command for
    the harness config; wrap() turns the harness argv into the jailed one; close() stops relays and removes sockets."""

    def __init__(self, binary, proc="static", uid=JAIL_UID, stderr_dir=None):
        probs = installed_problems()
        if probs:
            raise JailError("; ".join(probs))
        if proc not in ("static", "host"):
            raise JailError(f"proc must be static or host, not {proc!r}")
        self.binary_real = os.path.realpath(binary)
        self.binary_jailed, self.binary_info = binary_copy(binary)
        self.proc, self.uid = proc, uid
        self.sock_dir = None
        self.relays = []
        self.stderr_dir = stderr_dir

    def info(self):
        return dict(enabled=True, uid=self.uid, proc=self.proc, binary=self.binary_info, inner=INNER,
                    inner_sha256=sha256_file(INNER), relay=RELAY)

    def _sock_dir(self):
        if self.sock_dir is None:
            os.makedirs(SOCK_ROOT, mode=0o711, exist_ok=True)
            os.chmod(SOCK_ROOT, 0o711)
            d = os.path.join(SOCK_ROOT, secrets.token_hex(6))
            os.mkdir(d, 0o700)
            os.chown(d, self.uid, self.uid)
            self.sock_dir = d
        return self.sock_dir

    def mcp_command(self, real_argv, env=None, label="mcp"):
        """-> (command, args) for the harness's MCP config: the relay on a new socket served by real_argv."""
        d = self._sock_dir()
        path = os.path.join(d, f"{label}-{len(self.relays)}.sock")
        err = os.path.join(self.stderr_dir, f"jail_{label}_stderr.log") if self.stderr_dir else None
        self.relays.append(_Relay(path, list(real_argv), dict(env or {"PATH": "/usr/bin:/bin"}), err, self.uid))
        return RELAY_PYTHON, [RELAY, path]

    def wrap(self, argv, env, cwd, rw_dirs):
        """The jailed argv. Popen it with the same env: the jail forwards exactly those variables. Chowns the
        directories that are mounted back (one-time dirs only) to the jail uid."""
        rw = _outermost(list(rw_dirs) + ([self.sock_dir] if self.sock_dir else []) + ([cwd] if cwd else []))
        if len(rw) > MAX_BINDS:
            raise JailError(f"too many bind mounts ({len(rw)} > {MAX_BINDS})")
        for d in rw:
            if not os.path.isdir(d):
                raise JailError(f"bind source {d} is not a directory")
            if not d.startswith(("/tmp/lffrun/", "/tmp/lffstate/", SOCK_ROOT + "/")):
                raise JailError(f"refusing to mount {d} into the jail (only one-time dirs)")
            _chown_tree(d, self.uid)
        cmd = [self.binary_jailed if os.path.realpath(argv[0]) == self.binary_real else argv[0]] + list(argv[1:])
        out = ["setpriv", f"--reuid={self.uid}", f"--regid={self.uid}", "--clear-groups", "--",
               "unshare", "--user", "--map-root-user", "--mount", "--fork", "--kill-child",
               INNER, "--keep", ",".join(sorted(env)), "--proc", self.proc]
        for d in rw:
            out += ["--rw", d]
        return out + ["--cwd", cwd or "/", "--exe", self.binary_jailed, "--"] + cmd

    def close(self):
        for r in self.relays:
            r.close()
        self.relays = []
        if self.sock_dir and self.sock_dir.startswith(SOCK_ROOT + "/"):
            shutil.rmtree(self.sock_dir, ignore_errors=True)
        self.sock_dir = None
