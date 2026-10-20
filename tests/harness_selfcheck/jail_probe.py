"""Probe what a process inside the native-harness jail (lffbench/harnesses/jail.py, docs/15 §3.4) can see. Free, no
model call. For each harness binary: start a bash probe through HarnessJail.wrap() with that harness's /proc mode and
check, then run the binary's --version inside the jail.

  source run_env.sh && python tests/harness_selfcheck/jail_probe.py [--out report.json]

Pass criteria: /mnt /home /root empty; /tmp holds only the mounted one-time dirs; /usr and the binary copy read-only;
the work dir writable; environment exactly the given whitelist; uid outside = 65534; the MCP relay round-trips;
proc=static: no pid visible; proc=host: other processes' environ / root / fd / cwd unreadable; --version works.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lffbench.harnesses.base import new_run_dir  # noqa: E402
from lffbench.harnesses.jail import HarnessJail  # noqa: E402

PROBE = r'''
emit() { printf '%s\t%s\n' "$1" "$2"; }
emit mnt "$(ls -A /mnt | tr '\n' ' ')"
emit home "$(ls -A /home | tr '\n' ' ')"
emit root "$(ls -A /root 2>&1 | tr '\n' ' ')"
emit tmp "$(cd /tmp && find . -mindepth 1 -maxdepth 2 | sort | tr '\n' ' ')"
emit usr_write "$(touch /usr/.x 2>/dev/null && echo WRITABLE || echo ro)"
emit bin_write "$(touch "$(dirname "$BIN")/.x" 2>/dev/null && echo WRITABLE || echo ro)"
emit cwd_write "$(touch ./.x 2>/dev/null && echo ok && rm -f ./.x || echo FAIL)"
emit env "$(env -0 | tr '\0' '\n' | cut -d= -f1 | grep -v -E '^(PWD|SHLVL|_)$' | sort | tr '\n' ' ')"
emit uid_map "$(cat /proc/self/uid_map 2>/dev/null | tr -s ' ' | tr '\n' ';')"
emit pids "$(ls /proc | grep -c -E '^[0-9]+$')"
emit relay "$(echo relay-roundtrip | timeout 5 "$RELAY_CMD" "$RELAY_SCRIPT" "$RELAY_SOCK" 2>&1)"
emit secrets "$(ls /mnt/cpfs/workspace/agentic-robotics 2>&1 | head -1)"
P=$(ls /proc | grep -E '^[0-9]+$' | while read p; do [ "$p" != "$$" ] && [ "$p" != self ] && echo $p; done | head -1)
emit other_pid "$P"
if [ -n "$P" ]; then
  emit other_environ "$(cat /proc/$P/environ 2>&1 >/dev/null | head -c 60 || true)"
  emit other_root "$(ls /proc/$P/root/ 2>&1 >/dev/null | head -c 60 || true)"
  emit other_fd "$(ls /proc/$P/fd 2>&1 >/dev/null | head -c 60 || true)"
  emit other_cwd "$(ls /proc/$P/cwd/ 2>&1 >/dev/null | head -c 60 || true)"
fi
emit version "$("$BIN" --version 2>&1 | grep -v '^WARNING' | head -1)"
'''


def run_probe(binary, proc):
    j = HarnessJail(binary, proc=proc)
    work = new_run_dir("/tmp/lffrun")
    state = new_run_dir("/tmp/lffstate")
    try:
        cmd, args = j.mcp_command(["/bin/cat"], label="probe")
        env = dict(PATH="/usr/bin:/bin", HOME=state, LANG="C.UTF-8", BIN=j.binary_jailed, RELAY_CMD=cmd,
                   RELAY_SCRIPT=args[0], RELAY_SOCK=args[1], PROBE_CANARY="x")
        argv = j.wrap(["/bin/bash", "-c", PROBE], env, work, [state])
        r = subprocess.run(argv, env=env, cwd=work, capture_output=True, text=True, timeout=120)
        out = dict(line.split("\t", 1) for line in r.stdout.splitlines() if "\t" in line)
        out["_rc"], out["_stderr"] = r.returncode, r.stderr[-500:]
        return out, j.info(), os.path.basename(work), os.path.basename(state), os.path.basename(j.sock_dir)
    finally:
        j.close()
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(state, ignore_errors=True)


def judge(o, proc, work, state, sock, expect_env):
    c = {}
    c["mnt empty"] = o.get("mnt", "x").strip() == ""
    c["home empty"] = o.get("home", "x").strip() == ""
    c["root empty"] = o.get("root", "x").strip() == ""
    want_tmp = {"./lffrun", f"./lffrun/{work}", "./lffstate", f"./lffstate/{state}", "./lffsock", f"./lffsock/{sock}"}
    c["tmp only one-time dirs"] = set(o.get("tmp", "").split()) == want_tmp
    c["/usr read-only"] = o.get("usr_write") == "ro"
    c["binary copy read-only"] = o.get("bin_write") == "ro"
    c["work dir writable"] = o.get("cwd_write") == "ok"
    c["env = whitelist"] = set(o.get("env", "").split()) == set(expect_env)
    um = o.get("uid_map", "")
    c["uid 65534 outside"] = (um.strip(" ;").split(" ")[1:2] == ["65534"]) if um.strip() else proc == "static"
    c["relay round-trip"] = o.get("relay") == "relay-roundtrip"
    c["repo invisible"] = "No such file" in o.get("secrets", "")
    if proc == "static":
        c["no pid visible"] = o.get("pids") == "0"
    else:
        for k in ("other_environ", "other_root", "other_fd", "other_cwd"):
            c[f"{k} denied"] = "Permission denied" in o.get(k, "") or not o.get("other_pid")
    c["--version runs"] = bool(o.get("version")) and "crash" not in o.get("version", "").lower() \
        and "panic" not in o.get("version", "").lower()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from lffbench.harnesses.claude_code import claude_bin
    from lffbench.harnesses.codex import codex_bin
    report, ok = {}, True
    for name, b, proc in (("claude_code", claude_bin(), "host"), ("codex", codex_bin(), "static")):
        o, info, work, state, sock = run_probe(b, proc)
        env_keys = ["PATH", "HOME", "LANG", "BIN", "RELAY_CMD", "RELAY_SCRIPT", "RELAY_SOCK", "PROBE_CANARY"]
        c = judge(o, proc, work, state, sock, env_keys)
        report[name] = dict(proc=proc, checks=c, observed=o, jail=info)
        print(f"== {name} (proc={proc}, binary copy {info['binary']['copy']})")
        for k, v in c.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
            ok &= bool(v)
        if not all(c.values()):
            print("  observed:", json.dumps(o, ensure_ascii=False)[:1500])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
