"""Claude Code CLI as a native harness (docs/15 §3.3; flags measured in research_notes/harness接入调研/02 §2.1).

Two variants:
  claude_code        arm B : -p --bare (system prompt ~400 chars, no CLAUDE.md / auto-memory / hooks / plugins)
  claude_code_full   arm B': same but without --bare: Claude Code's full system prompt; still a one-time config dir and
                     HOME, an empty opaque work dir, --tools "" and only our MCP server
Both: process env is exactly the whitelist (env -i equivalent; plus the calling HTTP(S) proxy, base.proxy_env),
HOME / CLAUDE_CONFIG_DIR / TMPDIR are one-time dirs
under /tmp/lffstate/<hex>, the work dir is /tmp/lffrun/<12 hex> (its path is written into the system prompt), the MCP
server is scripts/mcp_robot_server.py on the run's session.

Fail-closed launch: before the real run the same argv / env is started once against a local fake Anthropic server
(free) with a schema-only MCP server; the first request's tool list, system prompt, harness-added message prefix and
request parameters are normalised and hashed, and the launch is refused unless the hash equals the pinned one
(pins/claude_code*.json, written by tests/harness_selfcheck after the self-check passed). During the real run the
stream's init event is checked again (tool list); on a mismatch the process is killed.
"""
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time

from . import contract as C
from . import pins
from .base import (NO_PROXY_KEYS, PROXY_KEYS, Cap, HarnessAdapter, HarnessRefused, LaunchSpec, binary_info,
                   new_run_dir, proxy_env, snapshot_tree)
from .fakeapi import FakeAnthropic
from .trajectory import TrajWriter, strip_mcp

ROOT = C.ROOT
MCP_SCRIPT = os.path.join(ROOT, "scripts", "mcp_robot_server.py")
MCP_TOOLS = ["mcp__lff__robot", "mcp__lff__shell", "mcp__lff__view_images"]
BASE_PATH = "/usr/bin:/bin"


def claude_bin():
    return os.environ.get("LFF_CLAUDE_BIN") or shutil.which("claude") or "claude"


def _norm(s, spec):
    """Remove what legitimately differs between runs from preset prompt text: paths of the one-time dirs, dates."""
    for k, v in sorted(dict(spec.state_dirs, cwd=spec.cwd).items(), key=lambda kv: -len(kv[1])):
        s = s.replace(v, f"<{k.upper()}>")
    s = re.sub(r"/tmp/lffrun/[0-9a-f]{12}", "<RUNDIR>", s)
    s = re.sub(r"-tmp-lffrun-[0-9a-f]{12}", "<RUNDIR_SLUG>", s)          # auto-memory dir name derived from the cwd
    s = re.sub(r"(cc_version=\d+\.\d+\.\d+)\.[0-9a-f]+", r"\1.<X>", s)  # billing header suffix varies per request
    s = re.sub(r"/tmp/lffstate/[0-9a-f]{12}", "<STATEDIR>", s)
    s = re.sub(r"\b20\d\d-\d\d-\d\d\b", "<DATE>", s)
    s = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>", s)
    return s


def request_fingerprint(body, headers, spec, first_message):
    """Normalised preset part of the first request + the checks that do not need a pin."""
    problems = []
    system = body.get("system")
    if isinstance(system, str):
        system = [dict(type="text", text=system)]
    sys_blocks = [dict(text=_norm(b.get("text", ""), spec), cache=bool(b.get("cache_control"))) for b in system or []]
    tools = [dict(name=t.get("name"), description=t.get("description"), input_schema=t.get("input_schema"))
             for t in body.get("tools") or []]
    if [t["name"] for t in tools] != MCP_TOOLS:
        problems.append(f"tool list is {[t['name'] for t in tools]}, expected {MCP_TOOLS}")
    msgs = body.get("messages") or []
    first = msgs[0] if msgs else {}
    content = first.get("content")
    if isinstance(content, str):
        content = [dict(type="text", text=content)]
    prefix, found = [], False
    for b in content or []:
        if b.get("type") == "text" and b.get("text") == first_message:
            found = True
            break
        prefix.append(_norm(b.get("text", "") if b.get("type") == "text" else json.dumps(b, sort_keys=True), spec))
    if not found:
        problems.append("the first user message does not contain our first message byte-for-byte as one block")
    params = {k: body.get(k) for k in ("model", "max_tokens", "thinking", "output_config", "context_management",
                                       "temperature", "top_p", "top_k", "tool_choice")}
    fp = dict(system=sys_blocks, tools=tools, user_prefix=prefix, params=params,
              anthropic_beta=sorted((headers.get("anthropic-beta") or headers.get("Anthropic-Beta") or "").split(",")),
              system_chars=sum(len(b.get("text", "")) for b in system or []))
    return fp, problems


class ClaudeCode(HarnessAdapter):
    name = "claude_code"
    bare = True
    uses_mcp = True
    clean_dirs = ("home", "config", "tmp")
    ancestor_files = ("CLAUDE.md", "CLAUDE.local.md", ".claude", "AGENTS.md")
    capabilities = dict(
        only_benchmark_tools=Cap("yes", "preflight", '--tools "" + --strict-mcp-config: request tools = 3 MCP tools'),
        images_reach_model=Cap("yes", "probe", "MCP image content forwarded as image blocks, md5 equal"),
        non_persistent_session=Cap("yes", "probe", "--no-session-persistence; one-time config dir removed after run"),
        pinned_version=Cap("yes", "launch", "claude --version + binary sha256 in harness.json; pin per version"),
        machine_readable_events=Cap("yes", "stream", "--output-format stream-json --verbose"),
        tool_calls_logged=Cap("yes", "stream", "tool_use / tool_result in stream + mcp_transcript + events.jsonl"),
        model_and_effort=Cap("yes", "preflight", "--model / --effort; request output_config.effort checked"),
        persistent_memory=Cap("absent", "probe", "--bare: no auto-memory; canaries never reach the request"),
        instruction_file_injection=Cap("absent", "probe", "--bare: CLAUDE.md discovery off; canaries absent"),
        web_access=Cap("absent", "preflight", "no WebFetch / WebSearch tool in the request"),
        host_file_read=Cap("absent", "probe", "no Read tool; MCP shell is jailed; view_images confined"),
        builtin_shell=Cap("absent", "preflight", "no Bash tool in the request"),
        subagent_delegation=Cap("absent", "preflight", "no Agent / Task tool in the request"),
        background_execution=Cap("absent", "preflight", "no Bash / Monitor tool in the request"),
    )

    def __init__(self, api_env=None, max_budget_usd=None, timeout_s=10800, pins_dir=None, allow_unpinned=False,
                 python=None, extra_args=None, jail=False):
        """api_env: the only API variables passed to the harness, e.g. {"ANTHROPIC_API_KEY": ..., optional
        "ANTHROPIC_BASE_URL": ...}. Never inherited from the calling environment. jail: run the CLI in the
        native-harness jail (lffbench/harnesses/jail.py; /proc stays the host's: Bun needs it)."""
        if jail:
            from .jail import HarnessJail
            self.jail = HarnessJail(claude_bin(), proc="host")
        self.api_env = dict(api_env or {})
        self.proxy = proxy_env()
        self.max_budget_usd = max_budget_usd
        self.timeout_s = timeout_s
        self.pins_dir = pins_dir
        self.allow_unpinned = allow_unpinned
        self.python = python or sys.executable
        self.extra_args = list(extra_args or [])
        self._version = None

    # ------------------------------------------------------------------------------------------------ version
    def version(self):
        if self._version is None:
            b = claude_bin()
            out = subprocess.run([b, "--version"], env=dict(PATH=BASE_PATH, HOME="/nonexistent"), capture_output=True,
                                 text=True, timeout=60)
            v = (out.stdout.strip().split() or ["?"])[0]
            self._version = dict(version=v, binary=binary_info(b))
        return self._version

    def pin_key(self, ep):
        v = self.version()
        return pins.key(self.name, v["version"], v["binary"]["sha256"], ep.model, ep.effort)

    # ------------------------------------------------------------------------------------------------ prepare
    def mcp_config(self, session_path=None, describe=False):
        """describe: for config_sha256 only (jailed, no relay is set up for it)."""
        args = ["--session", session_path] if session_path else ["--schema-only"]
        if describe or self.jail is None:
            cmd, a = ("<jail relay>" if self.jail is not None else self.python), [MCP_SCRIPT] + args
        else:
            cmd, a = self.mcp_server_command(self.python, MCP_SCRIPT, args)
        return {"mcpServers": {"lff": {"type": "stdio", "command": cmd, "args": a, "env": {"PATH": BASE_PATH}}}}

    def argv(self, spec_paths, ep, first_message):
        a = [claude_bin(), "-p", first_message]
        if self.bare:
            a.append("--bare")
        a += ["--tools", "", "--strict-mcp-config", "--mcp-config", spec_paths["mcp_config"],
              "--allowedTools", ",".join(MCP_TOOLS), "--permission-mode", "dontAsk", "--disable-slash-commands",
              "--no-session-persistence", "--output-format", "stream-json", "--verbose",
              "--model", ep.model, "--effort", ep.effort]
        if self.max_budget_usd:
            a += ["--max-budget-usd", str(self.max_budget_usd)]
        return a + self.extra_args

    def env(self, state):
        e = dict(PATH=BASE_PATH, HOME=state["home"], CLAUDE_CONFIG_DIR=state["config"], TMPDIR=state["tmp"],
                 LANG="C.UTF-8", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1",
                 DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1", MCP_TIMEOUT="60000", MCP_TOOL_TIMEOUT="900000")
        e.update(self.proxy)
        e.update(self.api_env)
        return e

    def _state_dirs(self, hexname=None):
        root = os.path.join("/tmp/lffstate", hexname) if hexname else new_run_dir("/tmp/lffstate")
        os.makedirs(root, exist_ok=True)
        state = dict(state=root, home=os.path.join(root, "home"), config=os.path.join(root, "config"),
                     tmp=os.path.join(root, "tmp"))
        for k in ("home", "config", "tmp"):
            os.makedirs(state[k], exist_ok=True)
        return state

    def prepare(self, ep, first_message, reuse_state=None):
        work = new_run_dir("/tmp/lffrun")
        state = reuse_state or self._state_dirs(os.path.basename(work))
        if self.jail is not None:
            self.jail.stderr_dir = ep.log_dir
        mcp_path = os.path.join(state["state"], "mcp.json")
        with open(mcp_path, "w") as f:
            json.dump(self.mcp_config(ep.session_path), f)
        argv = self.argv(dict(mcp_config=mcp_path), ep, first_message)
        spec = LaunchSpec(argv=argv, env=self.env(state), cwd=work, run_dir=work, state_dirs=state,
                          first_message=first_message, config_files=dict(mcp_config=mcp_path),
                          stdout_path=os.path.join(ep.log_dir, "claude_stream.jsonl"),
                          stderr_path=os.path.join(ep.log_dir, "claude_stderr.txt"),
                          redact_argv={2: f"<first message: sha256={C.sha256_text(first_message)}>"},
                          timeout_s=self.timeout_s)
        return spec

    def config_sha256(self, spec, ep):
        """Hash of everything configurable (argv without the first message, MCP config, env values except keys /
        one-time dirs)."""
        argv = [("<MCP_CONFIG>" if a == spec.config_files["mcp_config"] else a) for a in self.logged_argv(spec)[3:]]
        env = {k: (v if k not in self.api_env else "<api>") for k, v in spec.env.items()
               if k not in ("HOME", "CLAUDE_CONFIG_DIR", "TMPDIR") + PROXY_KEYS + NO_PROXY_KEYS}
        return C.sha256_json(dict(argv=argv, mcp=self.mcp_config("<SESSION>", describe=True), env=env))

    # ------------------------------------------------------------------------------------------------ preflight
    def preflight(self, spec, ep, record_dir=None):
        """Run the identical command against a fake API (free); refuse on any mismatch with the pin."""
        t0 = time.time()
        fake = FakeAnthropic([[("text", "preflight ok")]], record_dir=record_dir).start()
        pre_state = self._state_dirs()
        try:
            mcp_path = os.path.join(pre_state["state"], "mcp.json")
            with open(mcp_path, "w") as f:
                json.dump(self.mcp_config(None), f)
            argv = self.argv(dict(mcp_config=mcp_path), ep, spec.first_message)
            env = self.env(pre_state)
            env.update(ANTHROPIC_BASE_URL=fake.base_url, ANTHROPIC_API_KEY="sk-preflight-not-a-key")
            r = subprocess.run(self.jailed_argv(argv, env, spec.cwd, list(pre_state.values())), env=env, cwd=spec.cwd,
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
            reqs = fake.message_requests()
            if not reqs:
                raise HarnessRefused(f"preflight: harness sent no request (rc={r.returncode}): "
                                     f"{r.stderr.decode('utf-8', 'replace')[-500:]}")
            n, body = reqs[0]
            headers = {}
            if record_dir:
                headers = json.load(open(os.path.join(record_dir, f"req_{n:03d}.json"))).get("headers", {})
            fp, problems = request_fingerprint(body, headers, spec, spec.first_message)
            init = next((json.loads(x) for x in r.stdout.decode("utf-8", "replace").splitlines()
                         if x.startswith('{"type":"system"') and '"init"' in x), {})
            if init.get("tools") != MCP_TOOLS:
                problems.append(f"init event tools {init.get('tools')}")
            if self.bare and "memory_paths" in init:
                problems.append("init event has memory_paths under --bare")
            if [s.get("status") for s in init.get("mcp_servers", [])] != ["connected"]:
                problems.append(f"MCP servers {init.get('mcp_servers')}")
            if os.listdir(spec.cwd):
                problems.append(f"work dir not empty after preflight: {os.listdir(spec.cwd)}")
            fp_sha = C.sha256_json(fp)
            out = dict(fingerprint_sha256=fp_sha, system_chars=fp["system_chars"], n_requests=len(reqs),
                       tools=[t["name"] for t in fp["tools"]], effort_in_request=(fp["params"].get("output_config")
                                                                                   or {}).get("effort"),
                       model_in_request=fp["params"].get("model"), init_tools=init.get("tools"),
                       memory_paths=init.get("memory_paths"), dt_s=round(time.time() - t0, 1), problems=problems)
            if out["effort_in_request"] != ep.effort:
                problems.append(f"effort in request {out['effort_in_request']!r} != {ep.effort!r}")
            pin = pins.load(self.pin_key(ep), self.pins_dir)
            out["pin"] = pin and dict(key=pin["key"], fingerprint_sha256=pin["fingerprint_sha256"])
            out["fingerprint"] = fp
            if problems:
                raise HarnessRefused("preflight: " + "; ".join(problems))
            if pin is None:
                if not self.allow_unpinned:
                    raise HarnessRefused(f"preflight: no pin for {self.pin_key(ep)} (run tests/harness_selfcheck "
                                         f"with --write-pins after the self-check passes)")
                out["pin_status"] = "unpinned (allowed)"
            elif pin["fingerprint_sha256"] != fp_sha:
                raise HarnessRefused(f"preflight: preset prompt / tools fingerprint {fp_sha[:12]} != pinned "
                                     f"{pin['fingerprint_sha256'][:12]} ({pin['key']})")
            else:
                out["pin_status"] = "match"
            return out
        finally:
            fake.stop()
            shutil.rmtree(pre_state["state"], ignore_errors=True)

    def launch(self, spec, ep, skip_preflight=False):
        try:
            h = super().launch(spec, ep, skip_preflight)
        except HarnessRefused:
            raise
        # second line of defence: the init event of the real run
        deadline = time.time() + 120
        while time.time() < deadline and h.proc.poll() is None:
            init = self._init_event(spec)
            if init is not None:
                h.info["init"] = {k: init.get(k) for k in ("tools", "mcp_servers", "model", "claude_code_version",
                                                           "permissionMode", "memory_paths", "apiKeySource")}
                if init.get("tools") != MCP_TOOLS:
                    self.kill(h)
                    h.exit_reason = "refused: init tools " + json.dumps(init.get("tools"))
                    raise HarnessRefused(h.exit_reason)
                break
            time.sleep(0.2)
        return h

    @staticmethod
    def _init_event(spec):
        try:
            with open(spec.stdout_path) as f:
                for line in f:
                    if line.startswith("{"):
                        d = json.loads(line)
                        if d.get("type") == "system" and d.get("subtype") == "init":
                            return d
        except (OSError, ValueError):
            return None
        return None

    # ------------------------------------------------------------------------------------------------ collect
    def collect(self, h):
        spec, ep = h.spec, h.ep
        tw = TrajWriter(os.path.join(ep.log_dir, "trajectory.jsonl"))
        init, result, by_id, compactions, other = None, None, {}, 0, []
        lines = open(spec.stdout_path, encoding="utf-8", errors="replace").read().splitlines()
        for line in lines:
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            t = d.get("type")
            if t == "system" and d.get("subtype") == "init":
                init = d
            elif t == "system":
                if "compact" in str(d.get("subtype", "")):
                    compactions += 1
                other.append(d.get("subtype"))
            elif t == "assistant":
                m = d.get("message") or {}
                mid = m.get("id") or d.get("uuid")
                st = by_id.get(mid)
                if st is None:
                    u = m.get("usage") or {}
                    st = tw.add_step("agent", timestamp=d.get("timestamp"), model_name=m.get("model"),
                                     metrics=dict(prompt_tokens=u.get("input_tokens"),
                                                  completion_tokens=u.get("output_tokens"),
                                                  cached_tokens=u.get("cache_read_input_tokens")))
                    by_id[mid] = st
                for b in m.get("content") or []:
                    if b.get("type") == "text":
                        st["message"] += b.get("text", "")
                    elif b.get("type") == "thinking":
                        st["reasoning_content"] += b.get("thinking", "")
                    elif b.get("type") == "tool_use":
                        st["tool_calls"].append(dict(tool_call_id=b.get("id"), function_name=strip_mcp(b.get("name")),
                                                     native_name=b.get("name"), arguments=b.get("input")))
            elif t == "user":
                for b in (d.get("message") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        c = b.get("content")
                        texts, imgs = [], 0
                        for x in (c if isinstance(c, list) else [dict(type="text", text=c)]):
                            if x.get("type") == "text":
                                texts.append(x.get("text") or "")
                            elif x.get("type") == "image":
                                imgs += 1
                        tw.add_result(b.get("tool_use_id"), "\n".join(texts), bool(b.get("is_error")), imgs)
            elif t == "result":
                result = d
        v = self.version()
        tw.set_header(session_id=(init or {}).get("session_id"), source=os.path.basename(spec.stdout_path),
                      agent=dict(name=self.name, version=v["version"], model_name=ep.model, effort=ep.effort),
                      tools=(init or {}).get("tools"))
        u = (result or {}).get("usage") or {}
        tw.set_final(h.exit_reason, native_result={k: (result or {}).get(k) for k in
                                                   ("subtype", "is_error", "num_turns", "duration_ms",
                                                    "duration_api_ms", "total_cost_usd", "usage", "modelUsage")},
                     total_prompt_tokens=(u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                     + (u.get("cache_creation_input_tokens") or 0),
                     total_completion_tokens=u.get("output_tokens"), total_cached_tokens=u.get("cache_read_input_tokens"),
                     total_cost_usd=(result or {}).get("total_cost_usd"), compactions=compactions)
        tw.write()
        h.info.update(init_full=init, result=result, compactions=compactions, other_system=other, traj=tw)
        return tw

    # ------------------------------------------------------------------------------------------------ audit
    def audit(self, h):
        spec, ep = h.spec, h.ep
        init = h.info.get("init_full") or {}
        tw = h.info.get("traj")
        v, problems, notes = [], [], []
        if init.get("tools") != MCP_TOOLS:
            v.append(f"init tools {init.get('tools')}")
        if self.bare and init.get("memory_paths"):
            v.append(f"memory_paths {init.get('memory_paths')}")
        if not self.bare and init.get("memory_paths"):
            mp = json.dumps(init.get("memory_paths"))
            if spec.state_dirs["state"] not in mp and spec.cwd not in mp:
                v.append(f"memory_paths outside the one-time dirs: {mp}")
            notes.append("full variant: memory_paths point into the one-time dirs (empty at start)")
        names = [tc.get("native_name") for st in (tw.steps if tw else []) for tc in st["tool_calls"]]
        bad = sorted({n for n in names if n not in MCP_TOOLS})
        if bad:
            notes.append(f"model asked for unavailable tools (all rejected by the harness): {bad}")
        if h.env_observed_keys is not None and set(h.env_observed_keys) != set(spec.env):
            v.append(f"process env keys {h.env_observed_keys} != whitelist {sorted(spec.env)}")
        left = dict(config=snapshot_tree(spec.state_dirs["config"]), home=snapshot_tree(spec.state_dirs["home"]),
                    work=snapshot_tree(spec.cwd))
        mcp_logs = [p for p in left["home"] if "/mcp-logs-" in p]
        transcripts = [p for p in left["config"] + left["home"] if p.endswith(".jsonl") and p not in mcp_logs]
        if transcripts:
            v.append(f"session transcripts written: {transcripts}")
        if mcp_logs:  # MCP connection log of the CLI in the one-time HOME (removed after the run, never read back)
            notes.append(f"Claude Code wrote its MCP log into the one-time HOME: {mcp_logs}")
        if left["work"]:
            v.append(f"files in the work dir: {left['work']}")
        mcp_env = None
        mt = os.path.join(ep.log_dir, "mcp_transcript.jsonl")
        if os.path.exists(mt):
            for line in open(mt):
                d = json.loads(line)
                if d.get("ev") == "start" and "env_keys" in d:
                    mcp_env = d["env_keys"]
        out = dict(harness=self.name, violations=v, problems=problems, notes=notes, files_left=left,
                   mcp_server_env_keys=mcp_env, compactions=h.info.get("compactions"),
                   init={k: init.get(k) for k in ("tools", "mcp_servers", "model", "claude_code_version",
                                                   "permissionMode", "memory_paths", "apiKeySource")},
                   tool_calls=len(names), unavailable_tool_requests=bad, ok=not v)
        with open(os.path.join(ep.log_dir, "harness_audit.json"), "w") as f:
            json.dump(out, f, indent=1)
        return out


class ClaudeCodeFull(ClaudeCode):
    name = "claude_code_full"
    bare = False
    capabilities = dict(ClaudeCode.capabilities,
                        persistent_memory=Cap("partial", "probe", "auto-memory is on in the harness, but its dirs are "
                                              "one-time and empty; canaries outside them never reach the request"),
                        instruction_file_injection=Cap("partial", "probe", "CLAUDE.md discovery is on, but work / "
                                                       "config / home are empty one-time dirs; canaries outside absent"))
