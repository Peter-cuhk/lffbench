"""The reference driver scripts/api_player.py wrapped as an adapter ("ours", the main table's harness).

Nothing about the driver changes: it is started exactly as api_batch.py always started it (--style la), it makes its
own first `status` call, keeps its own transcript.jsonl and enforces the tool-call budget itself. The adapter only
adds harness.json, trajectory.jsonl (converted from transcript.jsonl) and harness_audit.json, like for every harness.
"""
import json
import os
import sys

from . import contract as C
from .base import Cap, HarnessAdapter, LaunchSpec, reference_instructions
from .trajectory import TrajWriter

ROOT = C.ROOT
PLAYER = os.path.join(ROOT, "scripts", "api_player.py")
DRIVER_FILES = ["scripts/api_player.py", "lffbench/agent/context.py", "lffbench/agent/backends/base.py",
                "lffbench/agent/backends/anthropic_messages.py", "lffbench/agent/backends/openai_responses.py",
                "lffbench/agent/backends/gemini.py", "lffbench/agent/backends/chat_completions.py"]


class Reference(HarnessAdapter):
    name = "ours"
    capabilities = dict(
        only_benchmark_tools=Cap("yes", "launch", "the driver sends exactly TOOL_LA / SHELL_TOOL / VIEW_TOOL_LA"),
        images_reach_model=Cap("yes", "probe", "FileImage sends the JPEG bytes the server wrote"),
        non_persistent_session=Cap("yes", "launch", "no state outside the run's log dir"),
        pinned_version=Cap("yes", "launch", "sha256 of the driver + backend sources"),
        machine_readable_events=Cap("yes", "stream", "transcript.jsonl"),
        tool_calls_logged=Cap("yes", "stream", "transcript.jsonl tool events + events.jsonl"),
        model_and_effort=Cap("yes", "launch", "--model / --effort mapped per provider (docs/12 §2)"),
        persistent_memory=Cap("absent", "launch", "strictly append-only context within the run only"),
        instruction_file_injection=Cap("absent", "launch", "only README.md of the sandbox"),
        web_access=Cap("absent", "launch", "no such tool"),
        host_file_read=Cap("absent", "probe", "shell jailed, view_images confined to the workspace"),
        builtin_shell=Cap("absent", "launch", "only the jailed shell tool"),
        subagent_delegation=Cap("absent", "launch", "no such tool"),
        background_execution=Cap("absent", "launch", "no such tool"),
    )

    def __init__(self, provider="anthropic", env=None, timeout_s=10800, python=None, **_):
        self.provider = provider
        self.base_env = dict(env if env is not None else os.environ)
        self.timeout_s = timeout_s
        self.python = python or sys.executable

    def version(self):
        per = {f: C.sha256_file(os.path.join(ROOT, f)) for f in DRIVER_FILES if os.path.exists(os.path.join(ROOT, f))}
        sha = C.sha256_json(per)
        return dict(version=f"api_player@{sha[:12]}", sources_sha256=sha, binary=dict(path=PLAYER, sha256=per.get(
            "scripts/api_player.py")))

    def prepare(self, ep, first_message=None):
        argv = [self.python, PLAYER, "--session", ep.session_path, "--provider", ep.provider or self.provider,
                "--model", ep.model, "--effort", ep.effort, "--style", "la"]
        return LaunchSpec(argv=argv, env=self.base_env, cwd=ROOT, first_message="",
                          stdout_path=os.path.join(ep.log_dir, "api_player.log"),
                          stderr_path=os.path.join(ep.log_dir, "api_player.log"), timeout_s=self.timeout_s)

    def config_sha256(self, spec, ep):
        return C.sha256_json(dict(argv=[a.replace(ep.session_path, "<SESSION>") for a in spec.argv[1:]]))

    def launch(self, spec, ep, skip_preflight=True):
        # stdout and stderr go to the same api_player.log (as before the adapter layer)
        import subprocess
        import time
        from .base import RunHandle, env_keys_of
        spec.preflight = dict(skipped=True, reason="reference driver: tools and instructions are our own code")
        log = open(spec.stdout_path, "ab")
        h = RunHandle(spec=spec, ep=ep, t_start=time.time())
        h.proc = subprocess.Popen(spec.argv, env=spec.env, cwd=spec.cwd, stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        h.pid = h.proc.pid
        time.sleep(0.2)
        h.env_observed_keys = env_keys_of(h.pid)
        return h

    def collect(self, h):
        ep = h.ep
        tw = TrajWriter(os.path.join(ep.log_dir, "trajectory.jsonl"))
        p = os.path.join(ep.log_dir, "transcript.jsonl")
        start, end, first_user, tot = None, None, None, dict(inp=0, out=0, cached=0)
        last_agent = None
        for line in (open(p) if os.path.exists(p) else []):
            d = json.loads(line)
            e = d.get("ev")
            if e == "start":
                start = d
            elif e == "user":
                if first_user is None:
                    first_user = d.get("text")
                tw.add_step("user", message=d.get("text", ""), timestamp=d.get("t"))
            elif e == "assistant":
                u = d.get("usage") or {}
                tot["inp"] += u.get("input_tokens", 0)
                tot["out"] += u.get("output_tokens", 0)
                tot["cached"] += u.get("cached_tokens", 0)
                last_agent = tw.add_step("agent", message=d.get("text") or "", reasoning=d.get("reasoning_summary") or "",
                                         timestamp=d.get("t"), tool_calls=[dict(tool_call_id=c["id"],
                                                                                function_name=c["name"],
                                                                                native_name=c["name"],
                                                                                arguments=c["arguments"])
                                                                           for c in d.get("tool_calls") or []],
                                         metrics=dict(prompt_tokens=u.get("input_tokens"),
                                                      completion_tokens=u.get("output_tokens"),
                                                      cached_tokens=u.get("cached_tokens")))
            elif e == "tool" and last_agent is not None:
                pending = [tc for tc in last_agent["tool_calls"]
                           if not any(r["source_call_id"] == tc["tool_call_id"]
                                      for r in last_agent["observation"]["results"])]
                if pending:
                    tw.add_result(pending[0]["tool_call_id"], d.get("output", ""), False, len(d.get("images") or []))
            elif e in ("end", "abort"):
                end = d
        tw.set_header(session_id=os.path.basename(ep.log_dir), source="transcript.jsonl",
                      agent=dict(name=self.name, version=self.version()["version"], model_name=ep.model,
                                 effort=ep.effort, provider=ep.provider or self.provider),
                      tools=[t["name"] for t in (start or {}).get("tools") or []])
        tw.set_final(h.exit_reason if not end or end.get("ev") == "end" else "abort: " + str(end.get("reason")),
                     native_result=end, total_prompt_tokens=tot["inp"], total_completion_tokens=tot["out"],
                     total_cached_tokens=tot["cached"])
        tw.write()
        h.info.update(traj=tw, start=start, first_user=first_user)
        if start and first_user is not None:  # the same string a native harness gets as its first message
            readme = open(os.path.join(ep.sandbox, "README.md")).read()
            h.info["instructions_match_readme"] = start.get("instructions") == reference_instructions(readme)
            # the same composition as base.first_message_text: identical sha256 for the same instance
            h.spec.first_message = start.get("instructions", "") + "\n\n" + first_user
        return tw

    def audit(self, h):
        ep = h.ep
        start = h.info.get("start") or {}
        tools, _ = C.tool_schemas()
        v = []
        if start and start.get("tools") != tools:
            v.append("driver tools differ from the contract's tool schemas")
        if h.info.get("instructions_match_readme") is False:
            v.append("driver instructions != preamble + README")
        out = dict(harness=self.name, violations=v, ok=not v, notes=["reference driver: no preflight needed"])
        with open(os.path.join(ep.log_dir, "harness_audit.json"), "w") as f:
            json.dump(out, f, indent=1)
        return out
