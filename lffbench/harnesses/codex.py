"""Codex CLI as a native harness (arm F; flags measured in research_notes/harness接入调研/02 §2.3, Codex 0.156).

Launch: process env exactly the whitelist (env -i equivalent), a brand-new empty CODEX_HOME and HOME (one-time, under
/tmp/lffstate/<hex>), work dir /tmp/lffrun/<12 hex>, `codex exec --json --skip-git-repo-check --ignore-user-config
--ignore-rules -s read-only`, ~20 features disabled, project docs / permissions / environment context /
collaboration-mode / skills instructions off, web search off, and only our MCP server with
default_tools_approval_mode="approve" (otherwise exec mode silently denies MCP calls).

Early exit: `codex exec` ends when the model replies without a tool call, also when the episode is not over (measured
2026-10-19, cx_pilot6b: gpt-6-astra stopped after a failed first attempt, "I stopped after the one authorized
attempt", although the finish reply said attempt 2 of 3 starts now). scripts/harness_run.py then sends the reference
driver's nudge as a user turn with `codex exec resume <thread>` (resume_argv), at most api_player.MAX_NO_TOOL times in
a row. Hence no --ephemeral: the session is kept in the one-time CODEX_HOME (memories off; removed after the run).
Measured with 0.163 against the fake server: the resumed request has the same developer messages, tools, model and
parameters as the first, then the earlier turns and the nudge as a user message.

The model API is a custom Responses provider "lffapi", so the preflight can point exactly the same config at a local
fake server by changing only base_url. Two ways to authenticate (auth=):
  apikey   env_key OPENAI_API_KEY (api_env), default endpoint the official API
  chatgpt  a ChatGPT subscription login (no API key; same as RPent's codex planner reusing `codex login`): the
           provider gets requires_openai_auth=true (requests carry the login's bearer + chatgpt-account-id), default
           endpoint the ChatGPT Codex backend; a copy of auth.json from the source CODEX_HOME is put into the one-time
           CODEX_HOME (the only file allowed there at launch). If Codex refreshes the login during a run, the refreshed
           auth.json is written back to the source (refresh tokens rotate), unless the source changed meanwhile.
           Launch is refused when the source login is close to its refresh (parallel copies would each rotate it).
Measured with 0.163 against the fake server: same body keys and headers in both modes plus chatgpt-account-id; the
preset differs (exec tool description), so pins are per auth mode (pin name codex-chatgpt).
The calling environment's HTTP(S) proxy is passed through (base.proxy_env; NO_PROXY = loopback only).

Known, accepted differences (reported, not fixable by config in 0.156):
  - code mode: the model sees one `exec` tool (JavaScript in a V8 isolate without fs / network); our tools are nested
    and listed only in ALL_TOOLS; MCP images reach the model only if the model's JS calls image(...)
  - apply_patch (blocked by the read-only sandbox), clock, 3 MCP-resource tools (our server has no resources),
    request_user_input, and 6 collaboration.* sub-agent tools cannot be removed; a sub-agent call is an audit violation
  - --json has tokens but no dollar cost
Fail-closed launch: the preflight checks the top-level tools, the exec nested-tool list (ALL_TOOLS, printed by a
scripted exec call), the developer (preset) messages and request parameters against the pin.
"""
import base64
import datetime
import fcntl
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
from .base import (NO_PROXY_KEYS, PROXY_KEYS, Cap, HarnessAdapter, HarnessRefused, LaunchSpec, binary_info,
                   new_run_dir, proxy_env, snapshot_tree)
from .fakeapi import FakeResponses
from .trajectory import TrajWriter

ROOT = C.ROOT
MCP_SCRIPT = os.path.join(ROOT, "scripts", "mcp_robot_server.py")
BASE_PATH = "/usr/bin:/bin"
OFFICIAL_API = "https://api.openai.com/v1"
CHATGPT_BACKEND = "https://chatgpt.com/backend-api/codex"
AUTH_FILE = "auth.json"
AUTH_LOCK = "/tmp/lffstate/.codex_auth.lock"
MAX_LOGIN_AGE_H = 8 * 24     # codex-rs TOKEN_REFRESH_INTERVAL (8 days, from the source, not measured here): a login
                             # older than this is refreshed at start, by every parallel copy
MIN_ACCESS_LEFT_H = 24       # an access token expiring during a run makes the run's copy refresh it


def _jwt_claims(tok):
    try:
        p = tok.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(p + "=" * (-len(p) % 4)))
    except (IndexError, ValueError, AttributeError):
        return {}


def chatgpt_login_status(home):
    """-> dict(ok, problem, last_refresh_h, access_token_valid_h) of the ChatGPT login in home/auth.json. No secret
    leaves this function."""
    src = os.path.join(home, AUTH_FILE)
    try:
        with open(src) as f:
            a = json.load(f)
    except (OSError, ValueError) as e:
        return dict(ok=False, problem=f"cannot read {src}: {type(e).__name__}")
    tok = a.get("tokens") or {}
    if a.get("auth_mode") != "chatgpt" or not tok.get("access_token") or not tok.get("refresh_token"):
        return dict(ok=False, problem=f"{src} is not a ChatGPT login (auth_mode={a.get('auth_mode')!r})")
    now = time.time()
    try:
        lr = datetime.datetime.fromisoformat(str(a.get("last_refresh")).replace("Z", "+00:00")).timestamp()
        age_h = (now - lr) / 3600
    except ValueError:
        age_h = float("inf")
    left_h = (float(_jwt_claims(tok["access_token"]).get("exp") or 0) - now) / 3600
    out = dict(ok=True, problem=None, last_refresh_h=round(age_h, 1), access_token_valid_h=round(left_h, 1))
    if age_h >= MAX_LOGIN_AGE_H or left_h < MIN_ACCESS_LEFT_H:
        out.update(ok=False, problem=f"the ChatGPT login in {home} is due for a refresh (last refresh {age_h:.0f} h ago, "
                                     f"access token valid {left_h:.0f} h): refresh it once outside the benchmark (use "
                                     f"codex with CODEX_HOME={home}), so that parallel runs do not each rotate it")
    return out
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
    can_resume = True
    clean_dirs = ("home", "codex_home", "tmp")
    ancestor_files = ("AGENTS.md", "AGENTS.override.md", ".codex")
    capabilities = dict(
        only_benchmark_tools=Cap("partial", "preflight", "our 3 MCP tools are reachable only via exec(JS); residual "
                                 "apply_patch(blocked) / clock / resource×3 / request_user_input / collaboration×6"),
        images_reach_model=Cap("partial", "probe", "only if the model's JS calls image(); bytes unchanged when it does"),
        non_persistent_session=Cap("yes", "probe", "only this run's thread, in the one-time CODEX_HOME (for exec "
                                   "resume), removed after the run; memories off"),
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
                 python=None, extra_args=None, jail=False, auth="apikey", auth_home=None):
        """api_env: {"OPENAI_API_KEY": ...} (the provider's env_key, auth="apikey"); base_url: Responses endpoint
        (default: official API / ChatGPT Codex backend by auth). auth="chatgpt": a ChatGPT login copied from
        auth_home (default LFF_CODEX_AUTH_HOME, else the calling CODEX_HOME, else ~/.codex), no API key.
        jail: run the CLI in the native-harness jail (lffbench/harnesses/jail.py, static /proc)."""
        if auth not in ("apikey", "chatgpt"):
            raise ValueError(f"auth must be apikey or chatgpt, not {auth!r}")
        if jail:
            from .jail import HarnessJail
            self.jail = HarnessJail(codex_bin(), proc="static")
        self.auth = auth
        self.auth_home = None
        self._auth_copies = {}  # one-time CODEX_HOME -> sha256 of the auth.json copied into it
        if auth == "chatgpt":
            self.auth_home = os.path.abspath(auth_home or os.environ.get("LFF_CODEX_AUTH_HOME")
                                             or os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex"))
            self.clean_allowed = {"codex_home": (AUTH_FILE,)}
        self.api_env = dict(api_env or {}) if auth == "apikey" else {}
        self.base_url = base_url or (OFFICIAL_API if auth == "apikey" else CHATGPT_BACKEND)
        self.proxy = proxy_env()
        self.timeout_s = timeout_s
        self.pins_dir = pins_dir
        self.allow_unpinned = allow_unpinned
        self.python = python or sys.executable
        self.extra_args = list(extra_args or [])
        self._version = None

    # ------------------------------------------------------------------------------------------------ auth
    def auth_info(self):
        if self.auth == "apikey":
            return dict(mode="apikey", keys=sorted(self.api_env))
        st = chatgpt_login_status(self.auth_home)
        return dict(mode="chatgpt", source=self.auth_home, **{k: v for k, v in st.items() if k != "ok"})

    def _install_auth(self, codex_home):
        """auth="chatgpt": copy the source login into a one-time CODEX_HOME (refused if it is due for a refresh)."""
        if self.auth != "chatgpt":
            return
        st = chatgpt_login_status(self.auth_home)
        if not st["ok"]:
            raise HarnessRefused("ChatGPT login: " + st["problem"])
        dst = os.path.join(codex_home, AUTH_FILE)
        with open(os.path.join(self.auth_home, AUTH_FILE), "rb") as f:
            data = f.read()
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self._auth_copies[codex_home] = C.sha256_file(dst)

    def _sync_auth_back(self, codex_home):
        """After a run: if Codex refreshed the login in the run's copy, write it back to the source (refresh tokens
        rotate, the source's old one may no longer work), unless the source changed since the copy. -> note or None."""
        copied = self._auth_copies.pop(codex_home, None)
        p = os.path.join(codex_home, AUTH_FILE)
        if copied is None or not os.path.exists(p) or C.sha256_file(p) == copied:
            return None
        try:
            with open(p, "rb") as f:
                data = f.read()
            new = json.loads(data)
        except (OSError, ValueError):
            return "auth.json in the run's CODEX_HOME changed and is unreadable: not written back"
        if new.get("auth_mode") != "chatgpt" or not (new.get("tokens") or {}).get("refresh_token"):
            return "auth.json in the run's CODEX_HOME changed but holds no ChatGPT login: not written back"
        src = os.path.join(self.auth_home, AUTH_FILE)
        os.makedirs(os.path.dirname(AUTH_LOCK), exist_ok=True)
        with open(AUTH_LOCK, "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if C.sha256_file(src) != copied:
                return "the run refreshed its login copy, but the source auth.json changed meanwhile: not written back"
            tmp = src + ".lff-tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, src)
        return "the run refreshed the ChatGPT login: refreshed auth.json written back to the source"

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
        name = self.name if self.auth == "apikey" else f"{self.name}-{self.auth}"
        return pins.key(name, v["version"], v["binary"]["sha256"], ep.model, ep.effort)

    # ------------------------------------------------------------------------------------------------ prepare
    def config_args(self, ep, session_path, base_url):
        mcp_cmd, mcp_args = self.mcp_server_command(self.python, MCP_SCRIPT, ["--session", session_path] if session_path
                                                    else ["--schema-only"])
        c = ["model_reasoning_effort=" + toml_str(ep.effort),
             "project_doc_max_bytes=0", "include_permissions_instructions=false", "include_environment_context=false",
             "include_collaboration_mode_instructions=false", "skills.include_instructions=false",
             "skills.bundled.enabled=false", 'web_search="disabled"',
             'model_provider="lffapi"', 'model_providers.lffapi.name="lffapi"',
             "model_providers.lffapi.base_url=" + toml_str(base_url), 'model_providers.lffapi.wire_api="responses"',
             ('model_providers.lffapi.env_key="OPENAI_API_KEY"' if self.auth == "apikey"
              else "model_providers.lffapi.requires_openai_auth=true"),
             "mcp_servers.lff.command=" + toml_str(mcp_cmd),
             "mcp_servers.lff.args=[" + ",".join(toml_str(x) for x in mcp_args) + "]",
             "mcp_servers.lff.env={PATH=" + toml_str(BASE_PATH) + "}",
             'mcp_servers.lff.enabled_tools=["robot","shell","view_images"]',
             'mcp_servers.lff.default_tools_approval_mode="approve"', "mcp_servers.lff.required=true",
             "mcp_servers.lff.startup_timeout_sec=60", "mcp_servers.lff.tool_timeout_sec=900"]
        return c

    def argv(self, ep, session_path, base_url, first_message):
        a = [codex_bin(), "exec", "--json", "--skip-git-repo-check", "--ignore-user-config",
             "--ignore-rules", "-s", "read-only", "-m", ep.model]
        for f in DISABLE:
            a += ["--disable", f]
        for c in self.config_args(ep, session_path, base_url):
            a += ["-c", c]
        return a + self.extra_args + [first_message]

    @staticmethod
    def thread_id(stdout_path):
        t = None
        for line in open(stdout_path, encoding="utf-8", errors="replace"):
            if line.startswith("{") and '"thread.started"' in line:
                t = json.loads(line).get("thread_id") or t
        return t

    def resume_argv(self, h, message):
        """`codex exec resume <thread> <message>` with the launch's flags and config; resume has no -s, so the
        sandbox goes in as sandbox_mode."""
        thread = self.thread_id(h.spec.stdout_path)
        if not thread:
            raise HarnessRefused("resume: no thread id in the event stream")
        a = list(h.spec.argv[:-1])  # without the first message
        i = a.index("exec")
        rest = a[i + 1:]
        j = rest.index("-s")
        mode = rest[j + 1]
        del rest[j:j + 2]
        return a[:i + 1] + ["resume"] + rest + ["-c", "sandbox_mode=" + toml_str(mode), thread, message]

    def env(self, state):
        e = dict(PATH=BASE_PATH, HOME=state["home"], CODEX_HOME=state["codex_home"], TMPDIR=state["tmp"],
                 LANG="C.UTF-8")
        e.update(self.proxy)
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
        if self.jail is not None:
            self.jail.stderr_dir = ep.log_dir
        if not reuse_state and os.listdir(state["codex_home"]):
            raise HarnessRefused("CODEX_HOME is not empty")
        argv = self.argv(ep, ep.session_path, self.base_url, first_message)
        return LaunchSpec(argv=argv, env=self.env(state), cwd=work, run_dir=work, state_dirs=state,
                          first_message=first_message, stdout_path=os.path.join(ep.log_dir, "codex_events.jsonl"),
                          stderr_path=os.path.join(ep.log_dir, "codex_stderr.txt"),
                          redact_argv={len(argv) - 1: f"<first message: sha256={C.sha256_text(first_message)}>"},
                          timeout_s=self.timeout_s)

    def launch(self, spec, ep, skip_preflight=False):
        self._install_auth(spec.state_dirs["codex_home"])
        return super().launch(spec, ep, skip_preflight=skip_preflight)

    def cleanup(self, spec):
        self._sync_auth_back(spec.state_dirs.get("codex_home", ""))
        super().cleanup(spec)

    def config_sha256(self, spec, ep):
        """Hash of everything configurable (argv without the first message, env values except keys / one-time dirs)."""
        argv = [re.sub(r"/tmp/lffsock/[0-9a-f]{12}/[^\"\],]+", "<JAIL_SOCKET>", a.replace(ep.session_path, "<SESSION>"))
                for a in self.logged_argv(spec)[1:-1]]
        env = {k: (v if k not in self.api_env else "<api>") for k, v in spec.env.items()
               if k not in ("HOME", "CODEX_HOME", "TMPDIR") + PROXY_KEYS + NO_PROXY_KEYS}
        return C.sha256_json(dict(argv=argv, env=env))

    # ------------------------------------------------------------------------------------------------ preflight
    def preflight(self, spec, ep, record_dir=None):
        t0 = time.time()
        fake = FakeResponses([[("exec", PREFLIGHT_JS)], [("text", "preflight ok")]], record_dir=record_dir).start()
        pre = self._state_dirs()
        try:
            self._install_auth(pre["codex_home"])
            argv = self.argv(ep, None, fake.base_url, spec.first_message)
            env = self.env(pre)
            if self.auth == "apikey":
                env["OPENAI_API_KEY"] = "sk-preflight-not-a-key"
            r = subprocess.run(self.jailed_argv(argv, env, spec.cwd, list(pre.values())), env=env, cwd=spec.cwd,
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=180)
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
            self._sync_auth_back(pre["codex_home"])
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
        note = self._sync_auth_back(ch)
        if note:
            notes.append(note)
        left = dict(codex_home=snapshot_tree(ch), home=snapshot_tree(spec.state_dirs["home"]),
                    work=snapshot_tree(spec.cwd))
        threads = None
        for f in left["codex_home"]:
            if re.match(r"state_\d+\.sqlite$", f):
                threads = self.sqlite_rows(os.path.join(ch, f), "threads")
        if isinstance(threads, int) and threads > 1:
            v.append(f"{threads} threads persisted in CODEX_HOME (only this run's one is expected)")
        thread = h.info.get("thread")
        sessions = [f for f in left["codex_home"] if f.startswith(("sessions/", "archived_sessions/"))]
        foreign = [f for f in sessions if not (thread and f.startswith("sessions/") and f.endswith(f"-{thread}.jsonl"))]
        if foreign:
            v.append(f"session files other than this run's thread: {foreign}")
        if h.info.get("resumes"):
            notes.append(f"resumed {len(h.info['resumes'])}x with the reference driver's nudge (CLI exited before the "
                         f"episode was over)")
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
