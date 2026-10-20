"""Harness adapter interface (docs/15 §3).

An adapter runs one episode (one robot_server session) with one agent harness and leaves behind, in the run's log
dir: harness.json (what was launched), trajectory.jsonl (unified trajectory), harness_audit.json (checks + violations).
Budgets, automatic finish and attempt switching are NOT the adapter's business: robot_server and the MCP server /
reference driver enforce them identically for every harness.

  name           stable harness name (the "harness" column of a table)
  capabilities   CapabilityDecl: every item of REQUIRED and FORBIDDEN must be declared, with its evidence source;
                 a missing item makes the class definition fail (so the import fails)
  version()      the actual installed version (probed before launch)
  prepare(ep, first_message) -> LaunchSpec: one-time dirs, configs, argv / env
  launch(spec)   fail-closed preflight, then start the process (env exactly the whitelist) -> RunHandle
  collect(h)     native event stream -> trajectory.jsonl (ATIF-like), usage, exit reason
  audit(h)       tool list, preset prompt, memory, state left behind, violations
"""
import dataclasses
import json
import os
import secrets
import shutil
import signal
import subprocess
import time

from . import contract as C

EVIDENCE = ("launch", "stream", "server", "probe", "preflight")

REQUIRED = {
    "only_benchmark_tools": "the model can call exactly robot / shell / view_images (ours) and nothing else of use",
    "images_reach_model": "view_images returns the image bytes to the model unchanged",
    "non_persistent_session": "nothing of the session is written to disk for later runs",
    "pinned_version": "the harness version (and binary hash) is recorded and pinned",
    "machine_readable_events": "a machine-readable event stream of the whole run",
    "tool_calls_logged": "every tool call is logged (harness stream + MCP / driver transcript + events.jsonl)",
    "model_and_effort": "model and reasoning effort can be set and are visible in the request",
}
FORBIDDEN = {
    "persistent_memory": "harness memory across runs (C4: auto-memory, CLAUDE.md / AGENTS.md, memories db)",
    "instruction_file_injection": "instruction files picked up from the work / config / home dir",
    "web_access": "web search / fetch",
    "host_file_read": "reading host files outside the episode workspace",
    "builtin_shell": "a shell of the harness itself (outside our jailed shell tool)",
    "subagent_delegation": "spawning sub-agents",
    "background_execution": "background tasks that outlive a tool call",
}


@dataclasses.dataclass(frozen=True)
class Cap:
    status: str          # REQUIRED items: "yes" | "partial" | "no"; FORBIDDEN items: "absent" | "present" | "partial"
    evidence: str        # one of EVIDENCE
    note: str = ""


PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")
NO_PROXY_KEYS = ("NO_PROXY", "no_proxy")
NO_PROXY_LOCAL = "127.0.0.1,localhost,::1"


def proxy_env(environ=None):
    """The calling environment's HTTP(S) proxy, passed to a native harness so it can reach the model API on machines
    that only get out through a proxy. NO_PROXY is loopback only (the preflight's fake API), never the host's list.
    Not part of config_sha256 (machine plumbing, not harness configuration)."""
    environ = os.environ if environ is None else environ
    e = {k: environ[k] for k in PROXY_KEYS if environ.get(k)}
    if e:
        e.update({k: NO_PROXY_LOCAL for k in NO_PROXY_KEYS})
    return e


class CapabilityError(TypeError):
    pass


def check_capabilities(name, caps):
    if not isinstance(caps, dict):
        raise CapabilityError(f"{name}: capabilities must be a dict")
    missing = [k for k in list(REQUIRED) + list(FORBIDDEN) if k not in caps]
    extra = [k for k in caps if k not in REQUIRED and k not in FORBIDDEN]
    if missing or extra:
        raise CapabilityError(f"{name}: capability declaration incomplete: missing {missing}, unknown {extra}")
    for k, c in caps.items():
        if not isinstance(c, Cap) or c.evidence not in EVIDENCE:
            raise CapabilityError(f"{name}: capability {k!r} needs a Cap(status, evidence in {EVIDENCE})")
        ok = ("yes", "partial", "no") if k in REQUIRED else ("absent", "present", "partial")
        if c.status not in ok:
            raise CapabilityError(f"{name}: capability {k!r} status must be one of {ok}")


class HarnessRefused(RuntimeError):
    """Fail-closed: the launch preflight found a tool list / preset prompt that is not the pinned one."""


@dataclasses.dataclass
class EpisodeSpec:
    session_path: str
    log_dir: str
    sandbox: str
    port: int
    token: str
    max_tool_calls: int
    model: str
    effort: str
    provider: str = ""
    task: str = ""
    seed: int = 0
    extra: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_session(cls, session_path, model, effort, provider="", **extra):
        session_path = os.path.abspath(session_path)
        log_dir = os.path.dirname(session_path)
        sess = json.load(open(session_path))
        iface = json.load(open(os.path.join(log_dir, "agent_iface.json")))
        st = C.session_start(log_dir)
        return cls(session_path=session_path, log_dir=log_dir, sandbox=sess["sandbox"], port=int(sess["port"]),
                   token=iface["token"], max_tool_calls=int(iface["max_tool_calls"]), model=model, effort=effort,
                   provider=provider, task=st["task"], seed=int(st["seed"]), extra=extra)


@dataclasses.dataclass
class LaunchSpec:
    argv: list                       # full argv (the first message may be one element)
    env: dict                        # the complete environment of the harness process (nothing else is inherited)
    cwd: str
    run_dir: str = ""                # one-time dir tree, removed after audit
    state_dirs: dict = dataclasses.field(default_factory=dict)   # name -> path (home, config, ...)
    first_message: str = ""
    config_files: dict = dataclasses.field(default_factory=dict)  # name -> path of files the adapter wrote
    stdout_path: str = ""
    stderr_path: str = ""
    redact_argv: dict = dataclasses.field(default_factory=dict)   # argv index -> replacement for the log
    preflight: dict = dataclasses.field(default_factory=dict)
    timeout_s: float = 10800


@dataclasses.dataclass
class RunHandle:
    spec: LaunchSpec
    ep: EpisodeSpec
    proc: object = None
    pid: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    exit_code: int = None
    exit_reason: str = ""
    env_observed_keys: list = None   # read from /proc/<pid>/environ right after the start
    info: dict = dataclasses.field(default_factory=dict)


def env_keys_of(pid):
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read()
    except OSError:
        return None
    return sorted({kv.split(b"=", 1)[0].decode("utf-8", "replace") for kv in raw.split(b"\0") if kv})


def new_run_dir(root="/tmp/lffrun"):
    """/tmp/lffrun/<12 hex>: the opaque work dir (its path is shown to the model by some harnesses)."""
    os.makedirs(root, exist_ok=True)
    while True:
        d = os.path.join(root, secrets.token_hex(6))
        try:
            os.mkdir(d, 0o700)
            return d
        except FileExistsError:
            continue


def binary_info(path):
    real = os.path.realpath(path)
    return dict(path=path, realpath=real, sha256=C.sha256_file(real) if os.path.exists(real) else None)


class HarnessAdapter:
    name = None
    capabilities = None
    uses_mcp = False
    clean_dirs = ()                  # keys of spec.state_dirs that must be empty at launch
    clean_allowed = {}               # key of spec.state_dirs -> file names the adapter itself put there (e.g. auth.json)
    ancestor_files = ()              # instruction files that must not exist in the work dir or any ancestor
    unsafe_skip_clean_check = False  # tests/harness_selfcheck only (probing what the harness would read)
    jail = None                      # lffbench/harnesses/jail.HarnessJail when the adapter runs jailed (docs/15 §3.4)
    can_resume = False               # resume(): the CLI can continue its session with one more user message

    def clean_state_problems(self, spec):
        """Fail-closed: the work dir and the one-time dirs must be empty, and no instruction file may sit in an
        ancestor of the work dir (harnesses walk up the tree looking for CLAUDE.md / AGENTS.md)."""
        p = []
        if spec.cwd and os.path.isdir(spec.cwd) and os.listdir(spec.cwd):
            p.append(f"work dir not empty: {sorted(os.listdir(spec.cwd))[:5]}")
        for k in self.clean_dirs:
            d = spec.state_dirs.get(k)
            left = [n for n in os.listdir(d) if n not in self.clean_allowed.get(k, ())] if d and os.path.isdir(d) else []
            if left:
                p.append(f"{k} dir not empty: {sorted(left)[:5]}")
        d = spec.cwd
        while d and d != "/":
            d = os.path.dirname(d)
            for f in self.ancestor_files:
                if os.path.exists(os.path.join(d, f)):
                    p.append(f"instruction file in an ancestor of the work dir: {os.path.join(d, f)}")
        return p

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.name is None:
            raise CapabilityError(f"{cls.__name__}: no name")
        check_capabilities(cls.name, cls.capabilities)

    # -------------------------------------------------------------------------------------------- interface
    def version(self):
        raise NotImplementedError

    def prepare(self, ep, first_message):
        raise NotImplementedError

    def preflight(self, spec, ep):
        """Fail-closed launch check; returns a dict for harness.json, raises HarnessRefused."""
        return dict(skipped=True)

    def collect(self, h):
        raise NotImplementedError

    def audit(self, h):
        raise NotImplementedError

    # -------------------------------------------------------------------------------------------- shared
    def mcp_server_command(self, python, script, args):
        """(command, args) of the MCP server entry in the harness config: our server, or, jailed, the relay to it."""
        if self.jail is not None:
            return self.jail.mcp_command([python, script] + list(args), env={"PATH": "/usr/bin:/bin"})
        return python, [script] + list(args)

    def jailed_argv(self, argv, env, cwd, dirs):
        """argv as started: unchanged, or wrapped into the jail with `dirs` (one-time dirs) mounted back."""
        return self.jail.wrap(argv, env, cwd, dirs) if self.jail is not None else list(argv)

    def jail_info(self):
        return self.jail.info() if self.jail is not None else dict(enabled=False)

    def auth_info(self):
        """How the harness authenticates to the model API (for harness.json; never a secret)."""
        return dict(mode="api_env", keys=sorted(getattr(self, "api_env", {}) or {}))

    def launch(self, spec, ep, skip_preflight=False):
        if not self.unsafe_skip_clean_check:
            probs = self.clean_state_problems(spec)
            if probs:
                spec.preflight = dict(problems=probs)
                raise HarnessRefused("clean-state check: " + "; ".join(probs))
        spec.preflight = dict(skipped=True, reason="skip_preflight") if skip_preflight else self.preflight(spec, ep)
        out = open(spec.stdout_path, "wb")
        err = open(spec.stderr_path, "wb")
        h = RunHandle(spec=spec, ep=ep, t_start=time.time())
        argv = self.jailed_argv(spec.argv, spec.env, spec.cwd, list(spec.state_dirs.values()) + [spec.run_dir])
        h.proc = subprocess.Popen(argv, env=spec.env, cwd=spec.cwd, stdin=subprocess.DEVNULL, stdout=out,
                                  stderr=err, start_new_session=True)
        h.pid = h.proc.pid
        for _ in range(20):  # the exec has happened once /proc/<pid>/environ differs from ours
            keys = env_keys_of(h.pid)
            if keys is not None and set(keys) == set(spec.env):
                break
            time.sleep(0.05)
        h.env_observed_keys = env_keys_of(h.pid)
        return h

    def resume_argv(self, h, message):
        """argv that continues the run's session with one more user message (adapters with can_resume)."""
        raise NotImplementedError

    def resume(self, h, message):
        """Continue the run's session with one more user message, after the CLI exited before the episode was over
        (scripts/harness_run.py sends the reference driver's nudge). Same env, work dir and one-time dirs; the output
        is appended to the same stdout / stderr files."""
        spec = h.spec
        argv = self.resume_argv(h, message)
        out, err = open(spec.stdout_path, "ab"), open(spec.stderr_path, "ab")
        jargv = self.jailed_argv(argv, spec.env, spec.cwd, list(spec.state_dirs.values()) + [spec.run_dir])
        h.proc = subprocess.Popen(jargv, env=spec.env, cwd=spec.cwd, stdin=subprocess.DEVNULL, stdout=out,
                                  stderr=err, start_new_session=True)
        h.pid, h.exit_code, h.exit_reason = h.proc.pid, None, ""
        keys = None
        for _ in range(20):
            keys = env_keys_of(h.pid)
            if keys is not None and set(keys) == set(spec.env):
                break
            time.sleep(0.05)
        h.info.setdefault("resumes", []).append(dict(t=time.time(), pid=h.pid, message=message,
                                                     argv=[("<message>" if x == message else x) for x in argv],
                                                     env_observed_keys=keys))
        return h

    def wait(self, h, timeout=None):
        try:
            h.exit_code = h.proc.wait(timeout=timeout or h.spec.timeout_s)
            h.exit_reason = "exited" if h.exit_code == 0 else f"exit code {h.exit_code}"
        except subprocess.TimeoutExpired:
            self.kill(h)
            h.exit_reason = "timeout"
        h.t_end = time.time()
        return h

    @staticmethod
    def kill(h):
        try:
            os.killpg(h.proc.pid, signal.SIGTERM)
            h.exit_code = h.proc.wait(timeout=20)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(h.proc.pid, signal.SIGKILL)
            h.exit_code = h.proc.wait()

    def logged_argv(self, spec):
        return [spec.redact_argv.get(i, a) for i, a in enumerate(spec.argv)]

    def cleanup(self, spec):
        if spec.run_dir and spec.run_dir.startswith(("/tmp/lffrun/", "/tmp/lffstate/")) and os.path.isdir(spec.run_dir):
            shutil.rmtree(spec.run_dir, ignore_errors=True)
        for d in spec.state_dirs.values():
            if d.startswith(("/tmp/lffrun/", "/tmp/lffstate/")) and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        if self.jail is not None:
            self.jail.close()


def snapshot_tree(root):
    """Relative paths of all files under root (to list what a harness left behind)."""
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            out.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(out)


def first_message_text(readme, status_json):
    """What a native harness gets as its first user message: the reference driver's instructions (preamble + README,
    byte-identical) followed by its first user turn (the status report). The reference driver sends the same two
    strings as system prompt and first user message; native harnesses keep their own system prompt."""
    from .contract import tool_schemas
    _, preamble = tool_schemas()
    return preamble + readme + "\n\n" + "Begin. Current status (robot command `status`):\n" + status_json


def reference_instructions(readme):
    from .contract import tool_schemas
    _, preamble = tool_schemas()
    return preamble + readme
