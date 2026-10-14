"""Codex CLI as a native harness (arm F; flags measured in research_notes/harness接入调研/02 §2.3, Codex 0.156).

Launch: process env exactly the whitelist (env -i equivalent), a brand-new empty CODEX_HOME and HOME (one-time, under
/tmp/lffstate/<hex>), work dir /tmp/lffrun/<12 hex>, `codex exec --json --ephemeral --skip-git-repo-check
--ignore-user-config --ignore-rules -s read-only`, ~20 features disabled, project docs / permissions / environment
context / collaboration-mode / skills instructions off, web search off, and only our MCP server with
default_tools_approval_mode="approve" (otherwise exec mode silently denies MCP calls).

The model API is a custom Responses provider "lffapi" (base_url + env_key), so the preflight can point exactly the
same config at a local fake server by changing only base_url.

Known, accepted differences (reported, not fixable by config in 0.156):
  - code mode: the model sees one `exec` tool (JavaScript in a V8 isolate without fs / network); our tools are nested
    and listed only in ALL_TOOLS; MCP images reach the model only if the model's JS calls image(...)
  - apply_patch (blocked by the read-only sandbox), clock, 3 MCP-resource tools (our server has no resources),
    request_user_input, and 6 collaboration.* sub-agent tools cannot be removed; a sub-agent call is an audit violation
  - --json has tokens but no dollar cost
Fail-closed launch: the preflight checks the top-level tools, the exec nested-tool list (ALL_TOOLS, printed by a
scripted exec call), the developer (preset) messages and request parameters against the pin.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time

from . import contract as C
from . import pins
from .base import Cap, HarnessAdapter, HarnessRefused, LaunchSpec, binary_info, new_run_dir, snapshot_tree
from .fakeapi import FakeResponses
from .trajectory import TrajWriter

ROOT = C.ROOT
MCP_SCRIPT = os.path.join(ROOT, "scripts", "mcp_robot_server.py")
BASE_PATH = "/usr/bin:/bin"
DISABLE = ["shell_tool", "unified_exec", "view_image", "multi_agent", "multi_agent_v2", "apps", "browser_use",
           "browser_use_external", "computer_use", "image_generation", "plugins", "hooks", "skill_search",
           "tool_suggest", "goals", "sleep_tool", "in_app_browser", "memories", "skill_mcp_dependency_install",
           "realtime_conversation"]
# what remains in 0.156 after the lock-down (measured): anything else is refused at launch
EXPECTED_TOP = ["collaboration.followup_task", "collaboration.interrupt_agent", "collaboration.list_agents",
                "collaboration.send_message", "collaboration.spawn_agent", "collaboration.wait_agent",
                "functions.exec", "functions.request_user_input", "functions.request_user_input_async",
                "functions.wait"]
EXPECTED_ALL_TOOLS = ["apply_patch", "clock__curr_time", "list_mcp_resource_templates", "list_mcp_resources",
                      "mcp__lff__robot", "mcp__lff__shell", "mcp__lff__view_images", "read_mcp_resource"]
MCP_TOOLS = ["robot", "shell", "view_images"]
OK_ITEMS = ("agent_message", "reasoning", "mcp_tool_call", "error")
PREFLIGHT_JS = "text(JSON.stringify(ALL_TOOLS));"


def codex_bin():
    return os.environ.get("LFF_CODEX_BIN") or shutil.which("codex") or "codex"


def toml_str(s):
    return json.dumps(s)  # a JSON string literal is a valid TOML basic string


def _norm(s, cwd):
    s = s.replace(cwd, "<CWD>")
    s = re.sub(r"/tmp/lffrun/[0-9a-f]{12}", "<RUNDIR>", s)
    s = re.sub(r"/tmp/lffstate/[0-9a-f]{12}", "<STATEDIR>", s)
    s = re.sub(r"\b20\d\d-\d\d-\d\d\b", "<DATE>", s)
    s = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<UUID>", s)
    return s


def _walk_tools(ts, prefix=""):
    out = []
    for t in ts or []:
        if t.get("type") == "namespace":
            out += _walk_tools(t.get("tools"), prefix + t.get("name", "") + ".")
        else:
            out.append(dict(name=prefix + (t.get("name") or t.get("type") or "?"), type=t.get("type"),
                            description=t.get("description"), parameters=t.get("parameters"), format=t.get("format")))
    return out


def request_fingerprint(body, cwd, first_message):
    problems = []
    items = body.get("input") or []
    top = _walk_tools(body.get("tools"))
    dev, users = [], []
    for it in items:
        if it.get("type") == "additional_tools":
            top += _walk_tools(it.get("tools"))
        elif it.get("type") == "message" and it.get("role") in ("developer", "system"):
            dev.append(_norm("\n".join(c.get("text", "") for c in it.get("content") or []), cwd))
        elif it.get("type") == "message" and it.get("role") == "user":
            users.append(it)
    names = sorted(t["name"] for t in top)
    if names != EXPECTED_TOP:
        problems.append(f"top-level tools {names} != expected {EXPECTED_TOP}")
    texts = [c.get("text") for u in users for c in u.get("content") or []]
    if texts != [first_message]:
        problems.append(f"user input is not exactly our first message ({len(texts)} user text blocks)")
    for t in top:
        if t.get("description"):
            t["description"] = _norm(t["description"], cwd)
    params = {k: body.get(k) for k in ("model", "reasoning", "parallel_tool_calls", "store", "include", "text",
                                       "tool_choice", "instructions", "max_output_tokens", "temperature")}
    meta = body.get("client_metadata") or {}
    try:
        turn_meta = json.loads(meta.get("x-codex-turn-metadata") or "{}")
    except ValueError:
        turn_meta = {}
    sandbox = {k: turn_meta.get(k) for k in ("sandbox", "sandbox_mode")}
    if sandbox.get("sandbox_mode") != "read-only":
        problems.append(f"sandbox {sandbox}")
    fp = dict(tools=sorted(top, key=lambda t: t["name"]), developer=dev, params=params, sandbox=sandbox,
              developer_chars=sum(len(d) for d in dev))
    return fp, problems


def all_tools_from(body):
    """The ALL_TOOLS list printed by the preflight exec call (in the second request)."""
    for it in body.get("input") or []:
        if it.get("type") == "custom_tool_call_output":
            out = it.get("output")
            parts = out if isinstance(out, list) else [dict(text=out)]
            for p in parts:
                t = (p.get("text") or "").strip()
                if t.startswith("["):
                    try:
                        return json.loads(t)
                    except ValueError:
                        pass
    return None


class Codex(HarnessAdapter):
    name = "codex"
    uses_mcp = True
    clean_dirs = ("home", "codex_home", "tmp")
    ancestor_files = ("AGENTS.md", "AGENTS.override.md", ".codex")
    capabilities = dict(
        only_benchmark_tools=Cap("partial", "preflight", "our 3 MCP tools are reachable only via exec(JS); residual "
                                 "apply_patch(blocked) / clock / resource×3 / request_user_input / collaboration×6"),
        images_reach_model=Cap("partial", "probe", "only if the model's JS calls image(); bytes unchanged when it does"),
        non_persistent_session=Cap("yes", "probe", "--ephemeral; threads table empty; CODEX_HOME removed after run"),
        pinned_version=Cap("yes", "launch", "codex --version + binary sha256 in harness.json; pin per version"),
        machine_readable_events=Cap("yes", "stream", "exec --json (items + usage, no cost)"),
        tool_calls_logged=Cap("yes", "stream", "mcp_tool_call items + mcp_transcript + events.jsonl"),
        model_and_effort=Cap("yes", "preflight", "-m / model_reasoning_effort; request reasoning.effort checked"),
        persistent_memory=Cap("absent", "probe", "memories feature off; empty one-time CODEX_HOME; canaries absent"),
        instruction_file_injection=Cap("absent", "probe", "project_doc_max_bytes=0 + empty CODEX_HOME (AGENTS.md)"),
        web_access=Cap("absent", "probe", "web_search disabled; exec isolate has no fetch"),
        host_file_read=Cap("absent", "probe", "shell / unified_exec / view_image off; exec has no require; "
                           "read-only sandbox"),
        builtin_shell=Cap("absent", "preflight", "shell_tool and unified_exec disabled (not in the tool list)"),
        subagent_delegation=Cap("partial", "preflight", "collaboration.* tools cannot be disabled in 0.156; any use "
                                "is an audit violation"),
        background_execution=Cap("partial", "preflight", "functions.wait / long exec yields exist; audited"),
    )

    def __init__(self, api_env=None, base_url=None, timeout_s=10800, pins_dir=None, allow_unpinned=False,
                 python=None, extra_args=None):
        """api_env: {"OPENAI_API_KEY": ...} (the provider's env_key); base_url: Responses endpoint (default official)."""
        self.api_env = dict(api_env or {})
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout_s = timeout_s
        self.pins_dir = pins_dir
        self.allow_unpinned = allow_unpinned
        self.python = python or sys.executable
        self.extra_args = list(extra_args or [])
        self._version = None

    def version(self):
        if self._version is None:
            b = codex_bin()
            out = subprocess.run([b, "--version"], env=dict(PATH=BASE_PATH, HOME="/nonexistent",
                                                             CODEX_HOME="/nonexistent"),
                                 capture_output=True, text=True, timeout=60)
            v = (out.stdout.strip().split() or ["?"])[-1]
            self._version = dict(version=v, binary=binary_info(b))
        return self._version

    def pin_key(self, ep):
        v = self.version()
        return pins.key(self.name, v["version"], v["binary"]["sha256"], ep.model, ep.effort)

    # ------------------------------------------------------------------------------------------------ prepare
    def config_args(self, ep, session_path, base_url):
        mcp_args = [MCP_SCRIPT, "--session", session_path] if session_path else [MCP_SCRIPT, "--schema-only"]
        c = ["model_reasoning_effort=" + toml_str(ep.effort),
             "project_doc_max_bytes=0", "include_permissions_instructions=false", "include_environment_context=false",
             "include_collaboration_mode_instructions=false", "skills.include_instructions=false",
             "skills.bundled.enabled=false", 'web_search="disabled"',
             'model_provider="lffapi"', 'model_providers.lffapi.name="lffapi"',
             "model_providers.lffapi.base_url=" + toml_str(base_url), 'model_providers.lffapi.wire_api="responses"',
             'model_providers.lffapi.env_key="OPENAI_API_KEY"',
             "mcp_servers.lff.command=" + toml_str(self.python),
             "mcp_servers.lff.args=[" + ",".join(toml_str(x) for x in mcp_args) + "]",
             "mcp_servers.lff.env={PATH=" + toml_str(BASE_PATH) + "}",
             'mcp_servers.lff.enabled_tools=["robot","shell","view_images"]',
             'mcp_servers.lff.default_tools_approval_mode="approve"', "mcp_servers.lff.required=true",
             "mcp_servers.lff.startup_timeout_sec=60", "mcp_servers.lff.tool_timeout_sec=900"]
        return c

    def argv(self, ep, session_path, base_url, first_message):
        a = [codex_bin(), "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
             "--ignore-rules", "-s", "read-only", "-m", ep.model]
        for f in DISABLE:
            a += ["--disable", f]
        for c in self.config_args(ep, session_path, base_url):
            a += ["-c", c]
        return a + self.extra_args + [first_message]

    def env(self, state):
        e = dict(PATH=BASE_PATH, HOME=state["home"], CODEX_HOME=state["codex_home"], TMPDIR=state["tmp"],
                 LANG="C.UTF-8")
        e.update(self.api_env)
        return e

    def _state_dirs(self, hexname=None):
        root = os.path.join("/tmp/lffstate", hexname) if hexname else new_run_dir("/tmp/lffstate")
        os.makedirs(root, exist_ok=True)
        state = dict(state=root, home=os.path.join(root, "home"), codex_home=os.path.join(root, "codex_home"),
                     tmp=os.path.join(root, "tmp"))
        for k in ("home", "codex_home", "tmp"):
            os.makedirs(state[k], exist_ok=True)
        return state

    def prepare(self, ep, first_message, reuse_state=None):
        work = new_run_dir("/tmp/lffrun")
        state = reuse_state or self._state_dirs(os.path.basename(work))
        if not reuse_state and os.listdir(state["codex_home"]):
            raise HarnessRefused("CODEX_HOME is not empty")
        argv = self.argv(ep, ep.session_path, self.base_url, first_message)
        return LaunchSpec(argv=argv, env=self.env(state), cwd=work, run_dir=work, state_dirs=state,
                          first_message=first_message, stdout_path=os.path.join(ep.log_dir, "codex_events.jsonl"),
                          stderr_path=os.path.join(ep.log_dir, "codex_stderr.txt"),
                          redact_argv={len(argv) - 1: f"<first message: sha256={C.sha256_text(first_message)}>"},
                          timeout_s=self.timeout_s)

    def config_sha256(self, spec, ep):
        """Hash of everything configurable (argv without the first message, env values except keys / one-time dirs)."""
        argv = [a.replace(ep.session_path, "<SESSION>") for a in self.logged_argv(spec)[1:-1]]
        env = {k: (v if k not in self.api_env else "<api>") for k, v in spec.env.items()
               if k not in ("HOME", "CODEX_HOME", "TMPDIR")}
        return C.sha256_json(dict(argv=argv, env=env))

    # ------------------------------------------------------------------------------------------------ preflight
    def preflight(self, spec, ep, record_dir=None):
        t0 = time.time()
        fake = FakeResponses([[("exec", PREFLIGHT_JS)], [("text", "preflight ok")]], record_dir=record_dir).start()
        pre = self._state_dirs()
        try:
            argv = self.argv(ep, None, fake.base_url, spec.first_message)
            env = self.env(pre)
            env["OPENAI_API_KEY"] = "sk-preflight-not-a-key"
            r = subprocess.run(argv, env=env, cwd=spec.cwd, stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
            reqs = fake.model_requests()
            if len(reqs) < 2:
                raise HarnessRefused(f"preflight: {len(reqs)} model requests (rc={r.returncode}): "
                                     f"{r.stderr.decode('utf-8', 'replace')[-800:]}")
            fp, problems = request_fingerprint(reqs[0][1], spec.cwd, spec.first_message)
            at = all_tools_from(reqs[1][1])
            if at is None:
                problems.append("exec preflight did not print ALL_TOOLS")
                at = []
            at_names = sorted(t.get("name") if isinstance(t, dict) else str(t) for t in at)
            if at_names != EXPECTED_ALL_TOOLS:
                problems.append(f"ALL_TOOLS {at_names} != expected {EXPECTED_ALL_TOOLS}")
            fp["all_tools"] = [({k: (_norm(v, spec.cwd) if isinstance(v, str) else v) for k, v in t.items()}
                                if isinstance(t, dict) else t) for t in sorted(at, key=lambda t: json.dumps(t))]
            effort = (fp["params"].get("reasoning") or {}).get("effort")
            if effort != ep.effort:
                problems.append(f"reasoning.effort in request {effort!r} != {ep.effort!r}")
            if os.listdir(spec.cwd):
                problems.append(f"work dir not empty after preflight: {os.listdir(spec.cwd)}")
            fp_sha = C.sha256_json(fp)
            out = dict(fingerprint_sha256=fp_sha, developer_chars=fp["developer_chars"], n_requests=len(reqs),
                       top_tools=[t["name"] for t in fp["tools"]], all_tools=at_names, effort_in_request=effort,
                       model_in_request=fp["params"].get("model"), sandbox=fp["sandbox"], dt_s=round(time.time() - t0, 1),
                       problems=problems)
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
            shutil.rmtree(pre["state"], ignore_errors=True)

    # ------------------------------------------------------------------------------------------------ collect
    def collect(self, h):
        spec, ep = h.spec, h.ep
        tw = TrajWriter(os.path.join(ep.log_dir, "trajectory.jsonl"))
        thread, usage, items, errors = None, dict(input=0, cached=0, output=0, reasoning=0), [], []
        for line in open(spec.stdout_path, encoding="utf-8", errors="replace"):
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            t = d.get("type")
            if t == "thread.started":
                thread = d.get("thread_id")
            elif t == "turn.completed":
                u = d.get("usage") or {}
                usage["input"] += u.get("input_tokens") or 0
                usage["cached"] += u.get("cached_input_tokens") or 0
                usage["output"] += u.get("output_tokens") or 0
                usage["reasoning"] += u.get("reasoning_output_tokens") or 0
            elif t in ("error", "turn.failed"):
                errors.append(d)
            elif t == "item.completed":
                it = d.get("item") or {}
                items.append(it.get("type"))
                if it.get("type") == "agent_message":
                    tw.add_step("agent", message=it.get("text", ""))
                elif it.get("type") == "reasoning":
                    tw.add_step("agent", reasoning=it.get("text", ""))
                else:
                    name = it.get("tool") if it.get("type") == "mcp_tool_call" else it.get("type")
                    args = it.get("arguments") if it.get("type") == "mcp_tool_call" else {
                        k: v for k, v in it.items() if k not in ("id", "type", "status")}
                    st = tw.add_step("agent", tool_calls=[dict(tool_call_id=it.get("id"), function_name=name,
                                                               native_name=f"{it.get('server', '')}.{name}"
                                                               if it.get("type") == "mcp_tool_call" else name,
                                                               item_type=it.get("type"), arguments=args)])
                    res = it.get("result") or {}
                    texts = [c.get("text", "") for c in res.get("content") or [] if c.get("type") == "text"]
                    imgs = sum(1 for c in res.get("content") or [] if c.get("type") == "image")
                    tw.add_result(it.get("id"), "\n".join(texts) if texts else json.dumps(it.get("error")),
                                  bool(it.get("error")) or it.get("status") == "failed", imgs)
                    del st
        v = self.version()
        tw.set_header(session_id=thread, source=os.path.basename(spec.stdout_path),
                      agent=dict(name=self.name, version=v["version"], model_name=ep.model, effort=ep.effort),
                      tools=EXPECTED_ALL_TOOLS)
        tw.set_final(h.exit_reason, native_result=dict(errors=errors[-5:], item_types=sorted(set(items))),
                     total_prompt_tokens=usage["input"], total_completion_tokens=usage["output"],
                     total_cached_tokens=usage["cached"], total_cost_usd=None, reasoning_tokens=usage["reasoning"])
        tw.write()
        h.info.update(items=items, traj=tw, thread=thread, errors=errors)
        return tw

    # ------------------------------------------------------------------------------------------------ audit
    @staticmethod
    def sqlite_rows(path, table):
        """Row count of `table`, read from a copy (the db may be in WAL mode and still open)."""
        import tempfile
        try:
            with tempfile.TemporaryDirectory() as td:
                for suf in ("", "-wal", "-shm"):
                    if os.path.exists(path + suf):
                        shutil.copy(path + suf, os.path.join(td, "db.sqlite" + suf))
                con = sqlite3.connect(os.path.join(td, "db.sqlite"))
                n = con.execute(f"select count(*) from {table}").fetchone()[0]
                con.close()
                return n
        except (sqlite3.Error, OSError) as e:
            return f"n/a ({e})"

    def audit(self, h):
        spec, ep = h.spec, h.ep
        v, notes = [], []
        items = h.info.get("items") or []
        bad = sorted({i for i in items if i not in OK_ITEMS})
        if bad:
            v.append(f"non-MCP items in the event stream: {bad}")
        tw = h.info.get("traj")
        residual = []
        for st in (tw.steps if tw else []):
            for tc in st["tool_calls"]:
                if tc.get("item_type") != "mcp_tool_call":
                    continue
                if tc["native_name"].startswith("lff.") and tc["function_name"] in MCP_TOOLS:
                    continue
                if tc["function_name"] in ("list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource"):
                    residual.append(tc["function_name"])  # our server has no resources: always empty
                else:
                    v.append(f"unexpected MCP call {tc['native_name']}")
        if residual:
            notes.append(f"residual MCP-resource tool calls (empty, our server has no resources): {residual}")
        if h.env_observed_keys is not None and set(h.env_observed_keys) != set(spec.env):
            v.append(f"process env keys {h.env_observed_keys} != whitelist {sorted(spec.env)}")
        ch = spec.state_dirs["codex_home"]
        left = dict(codex_home=snapshot_tree(ch), home=snapshot_tree(spec.state_dirs["home"]),
                    work=snapshot_tree(spec.cwd))
        threads = None
        for f in left["codex_home"]:
            if re.match(r"state_\d+\.sqlite$", f):
                threads = self.sqlite_rows(os.path.join(ch, f), "threads")
        if isinstance(threads, int) and threads > 0:
            v.append(f"{threads} thread(s) persisted in CODEX_HOME")
        sessions = [f for f in left["codex_home"] if f.startswith(("sessions/", "archived_sessions/"))]
        if sessions:
            v.append(f"session files written: {sessions}")
        if left["work"]:
            v.append(f"files in the work dir: {left['work']}")
        notes.append("image forwarding (model calls image()) is not visible in --json; needs the model proxy (P1)")
        mcp_env = None
        mt = os.path.join(ep.log_dir, "mcp_transcript.jsonl")
        if os.path.exists(mt):
            for line in open(mt):
                d = json.loads(line)
                if d.get("ev") == "start" and "env_keys" in d:
                    mcp_env = d["env_keys"]
        out = dict(harness=self.name, violations=v, notes=notes, files_left=left, threads_rows=threads,
                   mcp_server_env_keys=mcp_env, item_types=sorted(set(items)), errors=h.info.get("errors"), ok=not v)
        with open(os.path.join(ep.log_dir, "harness_audit.json"), "w") as f:
            json.dump(out, f, indent=1)
        return out
