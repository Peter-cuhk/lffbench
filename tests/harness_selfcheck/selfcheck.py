"""Offline self-check of the harness adapters (docs/15 §4 items 1-11 + "oracle through every access path").

No real model is called: a local fake Anthropic / OpenAI-Responses server (lffbench/harnesses/fakeapi.py) answers
from a script, so Claude Code / Codex are driven turn by turn; the robot side is a real robot_server
(l1_height_bias seed 3000 by default, style la, k=2, --hide-k) on a free port.

  source run_env.sh
  python tests/harness_selfcheck/selfcheck.py --out /tmp/<scratch>/hs_report [--adapters claude_code,claude_code_full,codex]
         [--port0 15520] [--write-pins] [--skip-oracle] [--keep]

Checks per native adapter (pass criteria as in docs/15 §4 / research note 02 §4.1):
  1 version + harness.json     2 tool list exactly as declared     3 memory canaries never reach a request
  4 first message byte-identical to the reference driver; tool schemas identical; harness prefix listed
  5 out-of-bounds reads / network / robot port fail     6 image bytes (md5) unchanged
  7 tool-call budget -> automatic finish, attempt switch, episode over     8 harness stream / MCP log / events.jsonl agree
  9 no state carried over (same config dirs reused, second run sees nothing of the first)
  10 process env = whitelist; MCP server env has no host-only variables     11 work dir path opaque, no task / seed / paths
  + fail-closed launch: clean-state guard, pin match, tampered pin refused, extra tool refused
Oracle: the same scripted command sequence through HTTP (api_player), MCP via Claude Code, MCP via Codex on three
fresh servers; events.jsonl must agree line by line once timestamps are removed, and so must the verdicts.
Writes <out>/selfcheck.json and <out>/selfcheck.md.
"""
import argparse
import base64
import copy
import glob
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lffbench.harnesses import EpisodeSpec, HarnessRefused, get_adapter, render_first_message  # noqa: E402
from lffbench.harnesses import base as HB  # noqa: E402
from lffbench.harnesses import contract as C  # noqa: E402
from lffbench.harnesses import pins as P  # noqa: E402
from lffbench.harnesses.fakeapi import FakeAnthropic, FakeResponses  # noqa: E402
import harness_run  # noqa: E402

TASK, SEED = "l1_height_bias", 3000
AVOID = [(9300, 9320), (11800, 11830), (13000, 13999)]
TIME_KEYS = {"t", "t_recv", "t_send", "server_s", "think_s", "wall_s", "dt", "elapsed_s", "time"}
MODELS = dict(claude_code=("claude-sonnet-5-5", "high"), claude_code_full=("claude-sonnet-5-5", "high"),
              codex=("gpt-6-astra", "high"))
EXTRA_PINS = dict(claude_code=[("claude-opus-5-5", "high")], claude_code_full=[("claude-opus-5-5", "high")],
                  codex=[("gpt-6-astra", "medium")])

# oracle for l1_height_bias seed 3000: attempt 1 trusts the visible block height (misses the grasp), attempt 2
# grasps 4 cm lower and places the tile (found once by hand on this instance; any fixed sequence would do)
ORACLE = ([("mcp", "shell", {"command": "ls obs"}),
           ("mcp", "view_images", {"paths": ["obs/s000_agentview.jpg"]})]
          + [("mcp", "robot", {"command": c}) for c in [
              "move_eef x=0.594 y=-0.112 z=0.2 gripper=open", "move_eef z=0.115", "move_eef gripper=close",
              "move_eef z=0.25", "move_eef x=0.644 y=0.193", "move_eef z=0.08", "move_eef gripper=open",
              'finish reason="attempt 1 done"',
              "move_eef x=0.594 y=-0.112 z=0.2 gripper=open", "move_eef z=0.075", "move_eef gripper=close",
              "move_eef z=0.25", "move_eef x=0.644 y=0.193", "move_eef z=0.06", "move_eef gripper=open",
              "move_eef z=0.2", 'finish reason="placed"']]
          + [("text", "Report: attempt 1 missed the grasp; attempt 2 grasped 4 cm lower and placed the tile.")])


# ------------------------------------------------------------------------------------------------ infrastructure
def port_free(p):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p))
            return True
        except OSError:
            return False


class Ports:
    def __init__(self, p0):
        self.p = p0

    def next(self):
        while True:
            self.p += 1
            if any(a <= self.p <= b for a, b in AVOID):
                continue
            if port_free(self.p):
                return self.p


class Servers:
    """robot_server sessions started with scripts/start_robot_session.sh into a scratch runs dir."""

    def __init__(self, runs_dir, ports):
        self.runs_dir, self.ports, self.logs = runs_dir, ports, []

    def start(self, tag, k=2, hide_k=True, mem="full", extra=""):
        port = self.ports.next()
        env = dict(os.environ, LFF_RUNS_DIR=self.runs_dir, LFF_STYLE="la",
                   LFF_EXTRA_ARGS=(("--hide-k " if hide_k else "") + extra).strip())
        out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "start_robot_session.sh"), TASK, str(SEED), mem,
                              "F1", str(port), "direct", str(k), tag, "rgb", "0"], capture_output=True, text=True,
                             env=env)
        kv = dict(x.split("=", 1) for x in out.stdout.splitlines() if x.startswith(("SANDBOX=", "LOG=")))
        if "LOG" not in kv:
            raise RuntimeError("robot server failed: " + out.stdout[-800:] + out.stderr[-800:])
        self.logs.append((kv["LOG"], kv["SANDBOX"]))
        return kv["LOG"], kv["SANDBOX"], port

    def stop_all(self, remove=True):
        for log, sb in self.logs:
            pf = os.path.join(log, "server.pid")
            if os.path.exists(pf):
                try:
                    os.kill(int(open(pf).read()), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
            if remove and sb.startswith("/tmp/robotlab/") and os.path.isdir(sb):
                shutil.rmtree(sb, ignore_errors=True)


def events(log):
    return [json.loads(x) for x in open(os.path.join(log, "events.jsonl"))]


def strip_times(o):
    if isinstance(o, dict):
        return {k: strip_times(v) for k, v in o.items() if k not in TIME_KEYS}
    if isinstance(o, list):
        return [strip_times(v) for v in o]
    return o


def jread(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


# ------------------------------------------------------------------------------------------------ scripts
def js_call(name, args):
    return ("const r = await tools.mcp__lff__%s(%s);\n"
            "for (const c of (r.content || [])) { if (c.type === 'image') image(c); else text(c.text); }"
            % (name, json.dumps(args)))


def translate(steps, api):
    """Harness-neutral steps -> fake-API script (one step per turn)."""
    out = []
    for st in steps:
        if st[0] == "mcp":
            if api == "claude":
                out.append([("tool", "mcp__lff__" + st[1], st[2])])
            elif api == "ref":
                out.append([("tool", st[1], st[2])])
            else:
                out.append([("exec", js_call(st[1], st[2]))])
        elif st[0] == "builtin" and api == "claude":
            out.append([("tool", st[1], st[2])])
        elif st[0] == "js" and api == "codex":
            out.append([("exec", st[1])])
        elif st[0] == "text":
            out.append([("text", st[1])])
    return out


def fake_for(adapter_name, script, rec):
    return (FakeResponses if adapter_name == "codex" else FakeAnthropic)(script, record_dir=rec).start()


def api_of(adapter_name):
    return "codex" if adapter_name == "codex" else "claude"


JAIL = False  # --jail: every adapter runs in the native-harness jail (docs/15 §3.4)
CODEX_AUTH = "apikey"  # --codex-auth chatgpt: codex authenticates with a copy of the real ChatGPT login (requests still
CODEX_AUTH_HOME = None  # go only to the fake server); its source CODEX_HOME is taken before the canary env is set up


def adapter_kw(name, fake, allow_unpinned=True, pins_dir=None):
    if name == "codex" and CODEX_AUTH == "chatgpt":
        return dict(auth="chatgpt", auth_home=CODEX_AUTH_HOME, base_url=fake.base_url, timeout_s=600,
                    allow_unpinned=allow_unpinned, pins_dir=pins_dir, jail=JAIL)
    if name == "codex":
        return dict(api_env={"OPENAI_API_KEY": "sk-fake-not-a-key"}, base_url=fake.base_url, timeout_s=600,
                    allow_unpinned=allow_unpinned, pins_dir=pins_dir, jail=JAIL)
    return dict(api_env={"ANTHROPIC_API_KEY": "sk-fake-not-a-key", "ANTHROPIC_BASE_URL": fake.base_url},
                timeout_s=600, allow_unpinned=allow_unpinned, pins_dir=pins_dir, jail=JAIL)


def run_native(name, log, script_steps, rec, allow_unpinned=True, pins_dir=None, reuse_state=None,
               unsafe=False, skip_preflight=False, keep=False, tamper=None, cont=False):
    """One harness_run.run() of adapter `name` on session `log` against a scripted fake API. cont: resume the CLI
    when it exits before the episode is over (off elsewhere: the scripts end with a reply mid-episode)."""
    fake = fake_for(name, translate(script_steps, api_of(name)), rec)
    try:
        kw = adapter_kw(name, fake, allow_unpinned, pins_dir)
        if tamper:
            kw.update(tamper)
        m, e = MODELS[name]
        a = harness_run.parser().parse_args(["--harness", name, "--session", os.path.join(log, "session.json"),
                                             "--model", m, "--effort", e])
        a.reuse_state, a.skip_preflight, a.keep_run_dir, a.no_continue = reuse_state, skip_preflight, keep, not cont
        if unsafe:
            orig = HB.HarnessAdapter.unsafe_skip_clean_check
            HB.HarnessAdapter.unsafe_skip_clean_check = True
        try:
            code, hj = harness_run.run(a, adapter_kw=kw)
        finally:
            if unsafe:
                HB.HarnessAdapter.unsafe_skip_clean_check = orig
        reqs = (fake.message_requests() if name != "codex" else fake.model_requests())
        return code, hj, reqs
    finally:
        fake.stop()


# ------------------------------------------------------------------------------------------------ request helpers
def first_user_blocks(name, body):
    if name == "codex":
        return [c.get("text") for it in body.get("input") or [] if it.get("type") == "message"
                and it.get("role") == "user" for c in it.get("content") or []]
    m = (body.get("messages") or [{}])[0]
    c = m.get("content")
    return [c] if isinstance(c, str) else [b.get("text") for b in c or [] if b.get("type") == "text"]


def images_in(name, body):
    """md5 of every image in the request."""
    out = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("type") == "image" and isinstance(o.get("source"), dict) and o["source"].get("data"):
                out.append(hashlib.md5(base64.b64decode(o["source"]["data"])).hexdigest())
            elif o.get("type") == "input_image" and str(o.get("image_url", "")).startswith("data:"):
                out.append(hashlib.md5(base64.b64decode(o["image_url"].split(",", 1)[1])).hexdigest())
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(body)
    return out


def tool_outputs(name, body):
    """Text of every tool result in the request (in order)."""
    out = []
    if name == "codex":
        for it in body.get("input") or []:
            if it.get("type") in ("custom_tool_call_output", "function_call_output"):
                o = it.get("output")
                out.append("\n".join(p.get("text", "") for p in o) if isinstance(o, list) else str(o))
        return out
    for m in body.get("messages") or []:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_result":
                c = b.get("content")
                out.append(c if isinstance(c, str) else "\n".join(x.get("text", "") for x in c or []
                                                                   if x.get("type") == "text"))
    return out


def json_in(o, key):
    """The first JSON object in a tool output (Codex wraps outputs in 'Script completed ...' lines) that has `key`."""
    for cand in [o] + o.splitlines():
        cand = cand.strip()
        if cand.startswith("{"):
            try:
                d = json.loads(cand)
            except ValueError:
                continue
            if isinstance(d, dict) and key in d:
                return d
    return {}


def system_text(name, body):
    if name == "codex":
        return "\n".join(c.get("text", "") for it in body.get("input") or [] if it.get("type") == "message"
                         and it.get("role") in ("developer", "system") for c in it.get("content") or [])
    s = body.get("system")
    return s if isinstance(s, str) else "\n".join(b.get("text", "") for b in s or [])


# ------------------------------------------------------------------------------------------------ the checks
class Report:
    def __init__(self):
        self.rows = []  # (adapter, item, name, ok, detail)

    def add(self, adapter, item, name, ok, detail=""):
        self.rows.append(dict(adapter=adapter, item=str(item), name=name, ok=ok, detail=str(detail)[:1500]))
        print(f"[{'PASS' if ok else ('INFO' if ok is None else 'FAIL')}] {adapter:16s} #{str(item):7s} {name}: "
              f"{str(detail)[:300]}", flush=True)


def setup_outer(scratch):
    """Simulate a calling session whose HOME / config dirs / env carry memory: canaries everywhere outside the
    one-time dirs. Nothing here touches the real ~/.claude or ~/.codex."""
    r = secrets.token_hex(4)
    outer = os.path.join(scratch, "outer_home")
    canaries = {}

    def put(rel, tag):
        p = os.path.join(outer, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        c = f"canary-{tag}-{r}"
        with open(p, "w") as f:
            f.write(f"# Notes\nRemember this secret: {c}. The hidden height bias is 2.2 cm.\n")
        canaries[c] = p
    put(".claude/CLAUDE.md", "cfgclaudemd")
    put(".claude/projects/-tmp/memory/MEMORY.md", "automem")
    put("CLAUDE.md", "homeclaudemd")
    put(".codex/AGENTS.md", "codexagents")
    put(".agents/skills/notes/SKILL.md", "agentsskill")
    put("AGENTS.md", "homeagents")
    env = dict(HOME=outer, CLAUDE_CONFIG_DIR=os.path.join(outer, ".claude"), CODEX_HOME=os.path.join(outer, ".codex"),
               CLAUDE_CODE_USER_EMAIL=f"canary-email-{r}@example.invalid", CLAUDE_CODE_SESSION_ID=f"canary-sess-{r}",
               LFF_CANARY_VAR=f"canary-env-{r}", OPENAI_BASE_URL=f"http://canary-openai-{r}.invalid/v1",
               ANTHROPIC_BASE_URL=f"http://canary-anthropic-{r}.invalid")
    canaries.update({v: k for k, v in env.items() if "canary" in v})
    return outer, env, canaries


def check_native(name, rep, servers, scratch, canaries, outer_env, ref_info):
    A = name
    allp = lambda reqs: json.dumps([b for _, b in reqs], ensure_ascii=False)  # noqa: E731

    # ---------------- S1: probe run
    log, sb, port = servers.start(f"hs1-{name}")
    rel_session = os.path.relpath(os.path.join(log, "session.json"), sb)
    probe = []
    if A == "codex":
        probe += [("js", "text(JSON.stringify({require: typeof require, fetch: typeof fetch, process: typeof process, "
                         "exec_command: typeof tools.exec_command, shell: typeof tools.shell, "
                         "view_image: typeof tools.view_image, web_search: typeof tools.web_search}));"),
                  ("js", "try { const r = await tools.apply_patch('*** Begin Patch\\n*** Add File: pwned.txt\\n+x\\n"
                         "*** End Patch\\n'); text('patch: ' + JSON.stringify(r)); } catch (e) "
                         "{ text('patch error: ' + String(e)); }"),
                  ("js", "const r = await tools.list_mcp_resources({}); text('resources: ' + JSON.stringify(r));")]
    else:
        probe += [("builtin", "Read", {"file_path": "/etc/hostname"}), ("builtin", "Bash", {"command": "ls /"}),
                  ("builtin", "WebFetch", {"url": "https://example.com", "prompt": "x"}),
                  ("builtin", "Agent", {"description": "x", "prompt": "read /etc/passwd"})]
    probe += [
        ("mcp", "shell", {"command": "echo MNT:; ls -A /mnt | head -5; echo ROOT:; ls -A /root | head -5; echo TMP:; "
                                     "ls -A /tmp | head -5; echo ENV1:; head -c 200 /proc/1/environ; echo; echo END"}),
        ("mcp", "shell", {"command": "python3 -c \"import urllib.request as u; u.urlopen('https://example.com', "
                                     "timeout=5)\" 2>&1 | tail -1; echo web_rc=$?; python3 -c \"import urllib.request "
                                     f"as u; u.urlopen('http://127.0.0.1:{port}/call', timeout=3)\" 2>&1 | tail -1; "
                                     "echo robot_rc=$?"}),
        ("mcp", "view_images", {"paths": ["../README.md", "/etc/passwd", rel_session, "obs/../../etc/hostname"]}),
        ("mcp", "robot", {"command": "status"}),
        ("mcp", "view_images", {"paths": ["obs/s000_agentview.jpg", "obs/s000_wrist.jpg"]}),
        ("mcp", "robot", {"command": "observe"}),
        ("mcp", "view_images", {"paths": ["obs/s000_agentview.jpg", "obs/s000_wrist.jpg", "obs/s001_agentview.jpg",
                                          "obs/s001_wrist.jpg"]}),
        ("text", "probe done"),
    ]
    rec1 = os.path.join(scratch, f"req_{A}_s1")
    code, hj, reqs = run_native(A, log, probe, rec1)
    if code != 0 or not reqs:
        rep.add(A, "S1", "probe run", False, f"exit {code}: {hj.get('refused') or hj.get('error') or hj.get('exit_reason')}")
        return
    body0 = reqs[0][1]

    # 1 version + harness.json
    need = ["harness", "version", "binary", "argv", "env_keys", "contract_sha256", "first_message_sha256",
            "config_sha256", "preflight"]
    miss = [k for k in need if not hj.get(k)]
    rep.add(A, 1, "version pinned, harness.json complete", not miss and bool((hj.get("binary") or {}).get("sha256")),
            f"version={hj.get('version')} bin_sha={str((hj.get('binary') or {}).get('sha256'))[:12]} missing={miss} "
            f"preflight_fp={str((hj.get('preflight') or {}).get('fingerprint_sha256'))[:12]}")

    # 2 tool list
    if A == "codex":
        pre = hj.get("preflight") or {}
        ok = not pre.get("problems") and pre.get("all_tools") is not None
        rep.add(A, 2, "tool list = declared residual set (top-level + ALL_TOOLS)", ok,
                f"top={pre.get('top_tools')} ALL_TOOLS={pre.get('all_tools')}")
    else:
        names = [t.get("name") for t in body0.get("tools") or []]
        init = (hj.get("init") or {}).get("tools")
        rep.add(A, 2, "tool list = exactly 3 MCP tools (request + init event)",
                names == ["mcp__lff__robot", "mcp__lff__shell", "mcp__lff__view_images"] and init == names,
                f"request={names} init={init}")

    # 3 memory canaries (outer HOME / config / env)
    text = allp(reqs)
    found = [c for c in canaries if c in text]
    extra = ""
    if A != "codex":
        mp = (hj.get("init") or {}).get("memory_paths")
        extra = f" init.memory_paths={mp}"
    rep.add(A, 3, "outer memory / instruction / env canaries absent from all requests", not found,
            f"{len(canaries)} canaries, found={found}{extra}")

    # 4 first message byte-identical to the reference driver; tool schemas
    fm = open(os.path.join(log, "first_message.txt")).read()
    blocks = first_user_blocks(A, body0)
    ref_instr, ref_first_user = ref_info.get("instructions"), ref_info.get("first_user")
    same_ref = (ref_instr is not None and fm == ref_instr + "\n\n" + ref_first_user)
    prefix = blocks[:blocks.index(fm)] if fm in blocks else blocks
    tools_ok, tdet = True, ""
    ours, _ = C.tool_schemas()
    if A == "codex":
        pf = json.load(open(os.path.join(log, "harness_preflight.json")))["fingerprint"]
        at = {t["name"]: t for t in pf.get("all_tools") or [] if isinstance(t, dict)}
        for t in ours:
            d = (at.get("mcp__lff__" + t["name"]) or {}).get("description", "")
            if not d.startswith(t["description"]):
                tools_ok, tdet = False, tdet + f" {t['name']}: description differs;"
        tdet += " (Codex shows our description + a TypeScript declaration generated from the schema)"
        dev_chars = pf.get("developer_chars")
        prefix_note = f"harness preset: {dev_chars} chars of developer messages; user input = exactly our message"
    else:
        rt = {t["name"]: t for t in body0.get("tools") or []}
        for t in ours:
            r = rt.get("mcp__lff__" + t["name"]) or {}
            if r.get("description") != t["description"] or r.get("input_schema") != t["parameters"]:
                tools_ok = False
                tdet += f" {t['name']}: desc_eq={r.get('description') == t['description']} " \
                        f"schema_eq={r.get('input_schema') == t['parameters']} got={json.dumps(r.get('input_schema'))[:200]};"
        prefix_note = f"harness prefix blocks before our message: {[p[:90] for p in prefix]}; system prompt " \
                      f"{len(system_text(A, body0))} chars"
    rep.add(A, 4, "first message = reference instructions + first user turn (byte-identical); tool schemas equal",
            (fm in blocks) and same_ref and tools_ok,
            f"in_request={fm in blocks} equals_api_player={same_ref} tools_equal={tools_ok}{tdet}; {prefix_note}")

    # 5 out-of-bounds
    last = reqs[-1][1]
    outs = tool_outputs(A, last)
    det = []
    ok5 = True
    if A == "codex":
        o0 = next((o for o in outs if '"require"' in o), "")
        try:
            d = json.loads(o0[o0.index("{"):o0.rindex("}") + 1])
        except ValueError:
            d = {}
        bad = {k: v for k, v in d.items() if v != "undefined"}
        ok5 &= bool(d) and not bad
        det.append(f"js typeof: {d}")
        op = next((o for o in outs if "patch" in o), "")
        ok5 &= ("blocked" in op or "rejected" in op or "error" in op.lower())
        det.append("apply_patch: " + op[-160:].replace("\n", " "))
        ores = next((o for o in outs if "resources:" in o), "")
        det.append(ores[-120:].replace("\n", " "))
        ok5 &= not os.path.exists(os.path.join(hj.get("cwd") or "/nonexistent", "pwned.txt"))
    else:
        bi = [o for o in outs if "No such tool available" in o or "not available" in o.lower()]
        ok5 &= len(bi) >= 4
        det.append(f"builtin rejections={len(bi)}/4")
    o_sh = next((o for o in outs if "MNT:" in o), "")
    o_sh_d = json_in(o_sh, "output").get("output", o_sh)
    mnt = o_sh_d.split("MNT:")[1].split("ROOT:")[0].strip() if "MNT:" in o_sh_d else "?"
    root_ = o_sh_d.split("ROOT:")[1].split("TMP:")[0].strip() if "ROOT:" in o_sh_d else "?"
    env1 = o_sh_d.split("ENV1:")[1].split("END")[0].strip() if "ENV1:" in o_sh_d else "?"
    ok5 &= mnt == "" and root_ == "" and "=" not in env1
    det.append(f"jail: /mnt=[{mnt[:60]}] /root=[{root_[:60]}] /proc/1/environ=[{env1[:80]}]")
    o_net = next((o for o in outs if "web_rc" in o), "")
    o_net_d = json_in(o_net, "output").get("output", o_net)
    net_fail = ("Error" in o_net_d.split("web_rc")[0] or "error" in o_net_d.split("web_rc")[0].lower())
    robot_fail = ("rror" in o_net_d.split("web_rc")[1].split("robot_rc")[0]) if "robot_rc" in o_net_d else False
    ok5 &= net_fail and robot_fail
    det.append("net: " + o_net_d.replace("\n", " | ")[:300])
    o_vi = next((o for o in outs if "not_viewable" in o), "")
    vi = json_in(o_vi, "attached")
    ok5 &= vi.get("attached") == [] and len(vi.get("not_viewable") or []) == 4
    det.append(f"view_images out-of-workspace: {vi}")
    rep.add(A, 5, "out-of-bounds file / network / robot-port access fails", ok5, "; ".join(det))

    # 6 image bytes
    want = [hashlib.md5(open(os.path.join(sb, p), "rb").read()).hexdigest() for p in
            ["obs/s000_agentview.jpg", "obs/s000_wrist.jpg", "obs/s001_agentview.jpg", "obs/s001_wrist.jpg"]]
    got = images_in(A, last)
    rep.add(A, 6, "image bytes reach the model unchanged (md5)", all(w in got for w in want),
            f"expected 4 distinct files (2 viewed twice); request has {len(got)} images; all md5 found="
            f"{all(w in got for w in want)}" + (" (Codex: forwarded because the scripted JS calls image())"
                                                 if A == "codex" else ""))

    # 8 three logs
    traj = jread(os.path.join(log, "trajectory.jsonl"))
    t_calls = [tc["function_name"] for r in traj if r.get("kind") == "step" for tc in r.get("tool_calls") or []
               if tc["function_name"] in ("robot", "shell", "view_images")]
    mcp = [d for d in jread(os.path.join(log, "mcp_transcript.jsonl")) if d.get("ev") == "tool"]
    m_calls = [d["name"] for d in mcp]
    ev = events(log)
    http = [e["cmd"] for e in ev if e.get("ev") == "http"]
    m_robot = [d["arguments"].get("command", "").split()[0] for d in mcp if d["name"] == "robot"]
    ok8 = t_calls == m_calls and http == ["status"] + m_robot
    rep.add(A, 8, "harness stream / mcp_transcript / events.jsonl one-to-one", ok8,
            f"stream={t_calls} mcp={m_calls} events.http={http} (first status = adapter's first-message call)")

    # 10 env
    obs = hj.get("env_observed_keys")
    wl = hj.get("env_keys")
    mstart = next((d for d in jread(os.path.join(log, "mcp_transcript.jsonl")) if d.get("ev") == "start"), {})
    menv = mstart.get("env_keys") or []
    msha = mstart.get("env_value_sha8") or {}
    # the calling environment carries canary values (HOME, CLAUDE_CONFIG_DIR, CODEX_HOME, *_BASE_URL, email, session
    # id, ...); one of them reached the MCP server if any variable there has the same value hash
    canary_sha = {hashlib.sha256(v.encode()).hexdigest()[:8]: k for k, v in outer_env.items()}
    host_only = sorted(f"{k}<-{canary_sha[h]}" for k, h in msha.items() if h in canary_sha)
    added = sorted(set(menv) - set(wl or []))  # set by the harness itself for its MCP child (its env was the whitelist)
    # any other calling-session variable with the same (non-trivial) value; 1-char constants such as CLAUDECODE=1 are
    # set by the harness itself and coincide with the calling Claude Code session by construction
    host_only += sorted(f"{k}<-host" for k in added if len(os.environ.get(k, "")) > 1
                        and msha.get(k) == hashlib.sha256(os.environ[k].encode()).hexdigest()[:8])
    rep.add(A, 10, "process env = whitelist exactly; MCP server env has no host-only variables",
            obs is not None and set(obs) == set(wl) and not host_only,
            f"whitelist={wl} observed_equal={obs is not None and set(obs) == set(wl)} canary values in MCP env="
            f"{host_only}; added by the harness for its MCP child: {added}")

    # 11 opaque path
    st = system_text(A, body0)
    cwd = hj.get("cwd") or ""
    opaque = bool(re.fullmatch(r"/tmp/lffrun/[0-9a-f]{12}", cwd))
    text11 = text.replace(rel_session, "")  # the probe's own view_images argument is echoed back; not a leak
    leaks = [s for s in (TASK, f"s{SEED}", os.path.basename(log), log, sb, "robotlab", "hs1-") if s in text11]
    shown = sorted(set(re.findall(r"/tmp/lffrun/[0-9a-f]{12}", text)))
    rep.add(A, 11, "work dir path opaque; no task / seed / run / sandbox path in any request", opaque and not leaks,
            f"cwd={cwd} paths_in_requests={shown} cwd_in_system_prompt={cwd in st} leaks={leaks}")

    # adapter audit of the probe run (violations only; harmless residuals are notes)
    au = json.load(open(os.path.join(log, "harness_audit.json")))
    rep.add(A, "audit", "adapter audit of the probe run has no violations", au.get("ok") is True,
            f"violations={au.get('violations')} notes={au.get('notes')}")

    # ---------------- S2: budget (max_tool_calls 4, k=2)
    log2, sb2, port2 = servers.start(f"hs2-{name}", extra="--max-tool-calls 4")
    steps = [("mcp", "shell", {"command": f"echo call{i}"}) for i in range(8)]
    steps += [("mcp", "robot", {"command": "status"}), ("mcp", "robot", {"command": "observe"}),
              ("mcp", "shell", {"command": "echo after"}), ("text", "done")]
    code, hj2, reqs2 = run_native(A, log2, steps, os.path.join(scratch, f"req_{A}_s2"))
    ev2 = events(log2)
    dones = [e for e in ev2 if e.get("ev") == "done"]
    mcp2 = [d for d in jread(os.path.join(log2, "mcp_transcript.jsonl")) if d.get("ev") == "tool"]
    auto = [i for i, d in enumerate(mcp2) if any("budget of 4 tool calls" in o for o in d.get("output") or [])]
    over = [d for d in mcp2 if d["name"] == "robot" and any("episode is over" in o.lower() or "over" in o.lower()
                                                             for o in d.get("output") or [])]
    seen = any("budget of 4 tool calls" in o for _, b in reqs2[-1:] for o in tool_outputs(A, b))
    ok7 = (len(dones) == 2 and all("budget" in (d.get("reason") or "") for d in dones) and auto == [3, 7]
           and bool(over) and seen and mcp2[-1].get("finished") is True)
    rep.add(A, 7, "tool-call budget: auto finish at the cap, counter resets, episode over afterwards", ok7,
            f"done events={[(d['attempt'], d.get('reason')) for d in dones]} auto-finish after call #={auto} "
            f"robot-after-end={[d['output'][0][:80] for d in over][:2]} model saw the note={seen}")

    # ---------------- S3: no state carried over (second run reuses the first run's config dirs)
    log3, sb3, port3 = servers.start(f"hs3-{name}")
    cz = f"state-canary-{secrets.token_hex(4)}"
    r1_steps = [("mcp", "shell", {"command": f"echo {cz}"}), ("text", f"Remember {cz}.")]
    code1, hj3a, reqs3a = run_native(A, log3, r1_steps, os.path.join(scratch, f"req_{A}_s3a"), keep=True)
    state = hj3a.get("state_dirs") or {}
    left = {k: HB.snapshot_tree(v) for k, v in state.items() if k != "state" and os.path.isdir(v)}
    threads, thread1 = None, None
    if A == "codex":
        from lffbench.harnesses.codex import Codex
        thread1 = Codex.thread_id(os.path.join(log3, "codex_events.jsonl"))
        for f in left.get("codex_home", []):
            if re.match(r"state_\d+\.sqlite$", f):
                threads = Codex.sqlite_rows(os.path.join(state["codex_home"], f), "threads")
    code2, hj3b, reqs3b = run_native(A, log3, [("text", "second run")], os.path.join(scratch, f"req_{A}_s3b"),
                                     reuse_state=state, unsafe=True)
    leaked = cz in allp(reqs3b)
    for v in list(state.values()) + [hj3a.get("cwd") or ""]:
        if v.startswith(("/tmp/lffstate/", "/tmp/lffrun/")):
            shutil.rmtree(v, ignore_errors=True)
    transcripts = [p for v in left.values() for p in v if p.endswith(".jsonl") and "mcp-logs" not in p]
    if A == "codex":  # not --ephemeral (exec resume): run 1's own session stays in its one-time CODEX_HOME
        transcripts = [p for p in transcripts if not (thread1 and p.startswith("sessions/")
                                                      and p.endswith(f"-{thread1}.jsonl"))]
    ok9 = (code1 == 0 and code2 == 0 and not leaked and not transcripts
           and (threads in (None, 0) or (A == "codex" and threads == 1)))
    rep.add(A, 9, "no state: a second run in the SAME config dirs sees nothing of the first", ok9,
            f"run1 exit={code1} run2 exit={code2} canary in run2 requests={leaked} files left after run1={left} "
            f"threads rows={threads}")

    # ---------------- S3b: early exit before the episode is over -> resumed with the reference driver's nudge
    if A == "codex":  # the adapters with can_resume (Claude Code: not yet)
        log5, sb5, port5 = servers.start(f"hs5-{name}", extra="--max-tool-calls 4")
        steps = [("mcp", "robot", {"command": "finish reason=\"probe\""}), ("mcp", "shell", {"command": "echo a"}),
                 ("text", "I stopped after the one authorized attempt."), ("mcp", "shell", {"command": "echo b"}),
                 ("text", "stop 2"), ("text", "stop 3"), ("text", "stop 4"), ("text", "stop 5")]
        code, hj5, reqs5 = run_native(A, log5, steps, os.path.join(scratch, f"req_{A}_s3b"), cont=True)
        cont = hj5.get("continuation") or []
        resumes = [c for c in cont if c.get("ev") == "resume"]
        want = ("Continue: issue the next robot command with the `robot` tool. You are in attempt 2 of 2; the episode "
                "is NOT over.")
        mcp5 = [d for d in jread(os.path.join(log5, "mcp_transcript.jsonl")) if d.get("ev") == "tool"]
        shell_calls = [d.get("calls_in_attempt") for d in mcp5 if d["name"] == "shell"]
        users = [first_user_blocks(A, b) for _, b in reqs5]
        nudged = [i for i, u in enumerate(users) if u and u[-1] == want]
        b0, b1 = reqs5[0][1], reqs5[nudged[0]][1] if nudged else {}
        preset = (A != "codex" or (json.dumps(b0.get("input", [])[:4], sort_keys=True)
                                   == json.dumps(b1.get("input", [])[:4], sort_keys=True)
                                   and b0.get("model") == b1.get("model") and b0.get("reasoning") == b1.get("reasoning")))
        history = bool(nudged) and "I stopped after the one authorized attempt." in json.dumps(b1)
        au5 = json.load(open(os.path.join(log5, "harness_audit.json")))
        ok12 = (code == 0 and len(resumes) == 4 and all(r["message"] == want for r in resumes)
                and [c["ev"] for c in cont][-1] == "abort" and shell_calls == [1, 2] and users[-1].count(want) == 4
                and preset and history and au5.get("ok") is True)
        rep.add(A, 12, "CLI exits mid-episode -> resumed with the reference nudge (history kept, same preset, MCP "
                "budget carried over, run ends after 4 tool-less replies in a row)", ok12,
                f"exit={code} continuation={[(c['ev'], c.get('attempt')) for c in cont]} "
                f"nudge ok={all(r['message'] == want for r in resumes)} requests ending in the nudge={nudged} "
                f"nudges in the last request={users[-1].count(want) if users else 0} "
                f"shell calls_in_attempt={shell_calls} preset same={preset} history kept={history} "
                f"audit={au5.get('violations')}")

    # ---------------- S4: fail-closed launch guards + what the harness would read without them
    m, e = MODELS[A]
    ep = EpisodeSpec.from_session(os.path.join(log3, "session.json"), m, e, "")
    fm3, _ = render_first_message(ep)
    fake = fake_for(A, [], None)
    ad = get_adapter(A, **adapter_kw(A, fake))
    spec = ad.prepare(ep, fm3)
    target = "AGENTS.md" if A == "codex" else "CLAUDE.md"
    with open(os.path.join(spec.cwd, target), "w") as f:
        f.write("canary-guard")
    try:
        ad.launch(spec, ep)
        refused = False
    except HarnessRefused as ex:
        refused, msg = True, str(ex)
    ad.cleanup(spec)
    fake.stop()
    rep.add(A, "FC-a", "launch refused when the work dir is not empty", refused, msg if refused else "launched!")

    # what the harness reads if the one-time dirs were NOT clean (guard bypassed, probing only)
    probes = {}
    rr = secrets.token_hex(4)
    fake = fake_for(A, [], os.path.join(scratch, f"req_{A}_s4b"))
    ad = get_adapter(A, **adapter_kw(A, fake))
    ad.unsafe_skip_clean_check = True
    spec = ad.prepare(ep, fm3)

    def put(path, tag):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        c = f"canary-{tag}-{rr}"
        open(path, "w").write(f"Remember: {c}\n")
        probes[c] = path
    if A == "codex":
        put(os.path.join(spec.cwd, "AGENTS.md"), "workagents")
        put(os.path.join(spec.state_dirs["codex_home"], "AGENTS.md"), "codexhomeagents")
        put(os.path.join(spec.state_dirs["home"], ".agents", "skills", "x", "SKILL.md"), "homeagentsskill")
    else:
        put(os.path.join(spec.cwd, "CLAUDE.md"), "workclaudemd")
        put(os.path.join(spec.state_dirs["config"], "CLAUDE.md"), "cfgclaudemd")
        slug = spec.cwd.replace("/", "-")
        put(os.path.join(spec.state_dirs["config"], "projects", slug, "memory", "MEMORY.md"), "automem")
    h = ad.launch(spec, ep, skip_preflight=True)
    ad.wait(h, timeout=300)
    reqs4 = fake.message_requests() if A != "codex" else fake.model_requests()
    t4 = allp(reqs4)
    seen4 = {os.path.relpath(p, spec.state_dirs["state"]) if p.startswith(spec.state_dirs["state"])
             else "work/" + os.path.basename(p): (c in t4) for c, p in probes.items()}
    ad.cleanup(spec)
    fake.stop()
    rep.add(A, "FC-b", "canaries inside the one-time dirs (guard bypassed): which ones the harness reads", None,
            f"{seen4}  -> this is why the launch guard requires empty dirs")
    return dict(log=log, reqs=len(reqs))


def check_pins(name, rep, servers, scratch, write):
    """Pin the fingerprint (after the checks passed) and test the fail-closed launch against it."""
    A = name
    log, sb, port = servers.start(f"hs5-{name}")
    m, e = MODELS[A]
    out = {}
    for model, effort in [(m, e)] + EXTRA_PINS.get(A, []):
        ep = EpisodeSpec.from_session(os.path.join(log, "session.json"), model, effort, "")
        fm, _ = render_first_message(ep)
        fake = fake_for(A, [], None)
        ad = get_adapter(A, **adapter_kw(A, fake))
        fps = []
        for _ in range(2):  # stability: two preflights on two different work dirs
            spec = ad.prepare(ep, fm)
            fps.append(ad.preflight(spec, ep))
            ad.cleanup(spec)
        fake.stop()
        stable = fps[0]["fingerprint_sha256"] == fps[1]["fingerprint_sha256"]
        rep.add(A, "FC-c", f"preflight fingerprint stable across work dirs ({model}/{effort})", stable,
                f"{fps[0]['fingerprint_sha256'][:16]} vs {fps[1]['fingerprint_sha256'][:16]}")
        if write and stable:
            k = ad.pin_key(ep)
            summ = {k2: fps[0].get(k2) for k2 in ("system_chars", "developer_chars", "tools", "top_tools", "all_tools",
                                                  "effort_in_request", "model_in_request", "sandbox")
                    if fps[0].get(k2) is not None}
            P.write(k, fps[0]["fingerprint_sha256"], summ)
            out[k] = fps[0]["fingerprint_sha256"]
    # pinned launch must match (when a pin exists); a tampered pin and an extra tool must be refused
    ep = EpisodeSpec.from_session(os.path.join(log, "session.json"), m, e, "")
    fm, _ = render_first_message(ep)
    fake = fake_for(A, [], None)
    ad = get_adapter(A, **adapter_kw(A, fake, allow_unpinned=False))
    spec = ad.prepare(ep, fm)
    if P.load(ad.pin_key(ep)) is None:
        rep.add(A, "FC-d", "pinned launch: fingerprint matches the pin", None, "no pin yet (run with --write-pins)")
    else:
        try:
            pre = ad.preflight(spec, ep)
            rep.add(A, "FC-d", "pinned launch: fingerprint matches the pin", pre.get("pin_status") == "match",
                    pre.get("pin_status"))
        except HarnessRefused as ex:
            rep.add(A, "FC-d", "pinned launch: fingerprint matches the pin", False, str(ex))
    ad.cleanup(spec)
    tmp_pins = os.path.join(scratch, f"pins_tampered_{A}")
    shutil.rmtree(tmp_pins, ignore_errors=True)
    k = ad.pin_key(ep)
    P.write(k, "0" * 64, {}, pins_dir=tmp_pins)
    ad2 = get_adapter(A, **adapter_kw(A, fake, allow_unpinned=False, pins_dir=tmp_pins))
    spec = ad2.prepare(ep, fm)
    try:
        ad2.preflight(spec, ep)
        rep.add(A, "FC-e", "tampered pin -> launch refused", False, "not refused")
    except HarnessRefused as ex:
        rep.add(A, "FC-e", "tampered pin -> launch refused", True, str(ex)[:200])
    ad2.cleanup(spec)
    shutil.rmtree(tmp_pins, ignore_errors=True)
    # an extra built-in tool slipped into the config -> refused before any real request
    if A == "codex":
        ad3 = get_adapter(A, **adapter_kw(A, fake, allow_unpinned=True))
        import lffbench.harnesses.codex as CX
        saved = list(CX.DISABLE)
        CX.DISABLE.remove("shell_tool")
        CX.DISABLE.remove("unified_exec")
    else:
        ad3 = get_adapter(A, **dict(adapter_kw(A, fake, allow_unpinned=True), extra_args=["--tools", "Read"]))
    spec = ad3.prepare(ep, fm)
    try:
        ad3.preflight(spec, ep)
        rep.add(A, "FC-f", "extra built-in tool in the config -> launch refused", False, "not refused")
    except HarnessRefused as ex:
        rep.add(A, "FC-f", "extra built-in tool in the config -> launch refused", True, str(ex)[:220])
    finally:
        if A == "codex":
            CX.DISABLE[:] = saved
    ad3.cleanup(spec)
    fake.stop()
    return out


def oracle_paths(rep, servers, scratch, adapters):
    """The same scripted command sequence through every access path; events.jsonl and verdicts must agree."""
    paths = {}
    # HTTP: the reference driver against a fake Anthropic server
    log, sb, port = servers.start("hs6-http")
    fake = FakeAnthropic(translate(ORACLE, "ref"), record_dir=os.path.join(scratch, "req_oracle_http")).start()
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(ANTHROPIC_BASE_URL=fake.base_url, ANTHROPIC_API_KEY="sk-fake-not-a-key")
    a = harness_run.parser().parse_args(["--harness", "ours", "--session", os.path.join(log, "session.json"),
                                         "--provider", "anthropic", "--model", "claude-sonnet-5-5", "--effort", "high"])
    code, hj = harness_run.run(a, adapter_kw=dict(provider="anthropic", env=env, timeout_s=600))
    fake.stop()
    paths["http (api_player)"] = (log, code, hj)
    start = next((d for d in jread(os.path.join(log, "transcript.jsonl")) if d.get("ev") == "start"), {})
    first_user = next((d.get("text") for d in jread(os.path.join(log, "transcript.jsonl")) if d.get("ev") == "user"), None)
    ref_info = dict(instructions=start.get("instructions"), first_user=first_user,
                    first_message_sha256=hj.get("first_message_sha256"), contract=hj.get("contract_sha256"))
    for A in [x for x in ("claude_code", "codex") if x in adapters]:
        log, sb, port = servers.start(f"hs6-{A}")
        code, hj, reqs = run_native(A, log, ORACLE, os.path.join(scratch, f"req_oracle_{A}"))
        paths[f"mcp ({A})"] = (log, code, hj)
    base_name = "http (api_player)"
    base_ev = [strip_times(e) for e in events(paths[base_name][0])]
    base_done = [(e["attempt"], e["success"], e.get("outcome", {}).get("failure")) for e in events(paths[base_name][0])
                 if e.get("ev") == "done"]
    for nm, (log, code, hj) in paths.items():
        ev = [strip_times(e) for e in events(log)]
        done = [(e["attempt"], e["success"], e.get("outcome", {}).get("failure")) for e in events(log)
                if e.get("ev") == "done"]
        diff = next((i for i, (x, y) in enumerate(zip(base_ev, ev)) if x != y), None)
        same = len(ev) == len(base_ev) and diff is None
        det = f"exit={code} events={len(ev)} verdicts={done} contract={str(hj.get('contract_sha256'))[:12]} " \
              f"first_msg={str(hj.get('first_message_sha256'))[:12]}"
        if not same:
            i = diff if diff is not None else min(len(ev), len(base_ev))
            det += f" first diff at line {i}: {json.dumps(base_ev[i] if i < len(base_ev) else None)[:300]} vs " \
                   f"{json.dumps(ev[i] if i < len(ev) else None)[:300]}"
        ok = same and done == base_done and hj.get("contract_sha256") == paths[base_name][2].get("contract_sha256") \
            and hj.get("first_message_sha256") == paths[base_name][2].get("first_message_sha256")
        rep.add(nm, "oracle", f"events.jsonl (no timestamps) + verdicts + contract + first message == {base_name}",
                ok if nm != base_name else (bool(done) and code == 0), det)
    return ref_info


def write_md(rep, path, meta):
    lines = [f"# harness self-check {meta['time']}", "", f"versions: {meta.get('versions')}", "",
             "| adapter | # | check | result | detail |", "|---|---|---|---|---|"]
    for r in rep.rows:
        res = "PASS" if r["ok"] else ("info" if r["ok"] is None else "**FAIL**")
        det = r["detail"].replace("|", "/").replace("\n", " ")[:400]
        lines.append(f"| {r['adapter']} | {r['item']} | {r['name']} | {res} | {det} |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="report + scratch dir (deleted parts: servers, sandboxes)")
    ap.add_argument("--adapters", default="claude_code,claude_code_full,codex")
    ap.add_argument("--port0", type=int, default=15520)
    ap.add_argument("--write-pins", action="store_true", help="pin the fingerprints of adapters whose checks passed")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep run logs and captured requests")
    ap.add_argument("--jail", action="store_true", help="run every native adapter in the native-harness jail")
    ap.add_argument("--codex-auth", default="apikey", choices=["apikey", "chatgpt"],
                    help="codex: fake API key, or a copy of the real ChatGPT login (LFF_CODEX_AUTH_HOME / CODEX_HOME); "
                         "requests still go only to the fake server; pins are per auth mode")
    a = ap.parse_args()
    global JAIL, CODEX_AUTH, CODEX_AUTH_HOME
    JAIL = a.jail
    CODEX_AUTH = a.codex_auth
    CODEX_AUTH_HOME = os.path.abspath(os.environ.get("LFF_CODEX_AUTH_HOME") or os.environ.get("CODEX_HOME")
                                      or os.path.expanduser("~/.codex"))
    adapters = [x for x in a.adapters.split(",") if x]
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    scratch = os.path.join(out, "scratch")
    os.makedirs(scratch, exist_ok=True)
    runs = os.path.join(scratch, "runs")
    servers = Servers(runs, Ports(a.port0))
    rep = Report()
    saved_env = dict(os.environ)
    meta = dict(time=time.strftime("%F %T"), adapters=adapters, jail=JAIL, codex_auth=CODEX_AUTH)
    try:
        # static: capability declarations (a class with a missing item cannot even be defined)
        try:
            class Bad(HB.HarnessAdapter):  # noqa: F841
                name = "bad"
                capabilities = dict(only_benchmark_tools=HB.Cap("yes", "launch"))
            rep.add("static", "cap", "incomplete capability declaration rejected", False, "class was accepted")
        except HB.CapabilityError as e:
            rep.add("static", "cap", "incomplete capability declaration rejected", True, str(e)[:160])
        vers = {}
        for A in adapters:
            ad = get_adapter(A)
            vers[A] = ad.version().get("version")
        meta["versions"] = vers
        ref_info = {}
        if not a.skip_oracle:
            ref_info = oracle_paths(rep, servers, scratch, adapters)
        outer, outer_env, canaries = setup_outer(scratch)
        os.environ.update(outer_env)  # the calling environment now carries memory everywhere
        results = {}
        for A in adapters:
            try:
                results[A] = check_native(A, rep, servers, scratch, canaries, outer_env, ref_info)
            except Exception:
                rep.add(A, "-", "check run crashed", False, traceback.format_exc()[-1200:])
        os.environ.clear()
        os.environ.update(saved_env)
        for A in adapters:
            passed = all(r["ok"] is not False for r in rep.rows if r["adapter"] == A)
            try:
                meta.setdefault("pins", {})[A] = check_pins(A, rep, servers, scratch, a.write_pins and passed)
            except Exception:
                rep.add(A, "FC", "pin checks crashed", False, traceback.format_exc()[-1200:])
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        servers.stop_all(remove=True)
        for d in glob.glob("/tmp/lffstate/*") + glob.glob("/tmp/lffrun/*"):
            pass  # adapters remove their own dirs; nothing else of ours lives there
        n_fail = sum(1 for r in rep.rows if r["ok"] is False)
        meta["n_fail"], meta["n_pass"] = n_fail, sum(1 for r in rep.rows if r["ok"] is True)
        json.dump(dict(meta=meta, rows=rep.rows), open(os.path.join(out, "selfcheck.json"), "w"), indent=1)
        write_md(rep, os.path.join(out, "selfcheck.md"), meta)
        if not a.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        print(f"\n{meta['n_pass']} passed, {n_fail} failed; report {out}/selfcheck.md", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
