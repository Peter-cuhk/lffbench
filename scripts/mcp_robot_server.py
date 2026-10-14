"""MCP server (stdio) exposing one robot_server session to a native agent harness (Claude Code, Codex, Gemini CLI, ...).

The tools are exactly the unified harness's LA tools (scripts/api_player.py): `robot`, `shell` (jailed, in the episode
workspace) and `view_images` (returned as MCP image content), with the same per-attempt tool-call budget: when it is
used up the attempt is finished automatically and the result is appended to that call's output. So a native harness
and the API harness see the same README, tools and limits; only the agent loop around them differs.

  python scripts/mcp_robot_server.py --session runs/<...>/session.json
  python scripts/mcp_robot_server.py --schema-only     (launch preflight: same tools/list, no session, every call fails)
Every call is logged to <log dir>/mcp_transcript.jsonl (the first line lists the names of the environment variables the
harness passed to this process, for the harness self-check). Nothing but JSON-RPC goes to stdout.
"""
import argparse
import base64
import json
import os
import sys
import time
import traceback

ENV_AT_START = dict(os.environ)  # what the harness passed (before our own imports touch the environment)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_player import (SHELL_TOOL, TOOL_LA, VIEW_TOOL_LA, call_server, parse_command, run_shell,  # noqa: E402
                        workspace_image)

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
TOOLS = [TOOL_LA, SHELL_TOOL, VIEW_TOOL_LA]


class Session:
    def __init__(self, session_path):
        sess = json.load(open(session_path))
        self.sb, self.port = sess["sandbox"], sess["port"]
        self.log_dir = os.path.dirname(os.path.abspath(session_path))
        iface = json.load(open(os.path.join(self.log_dir, "agent_iface.json")))
        self.token, self.max_tool_calls = iface["token"], int(iface["max_tool_calls"])
        self.calls_in_attempt, self.finished = 0, False
        self.tlog = open(os.path.join(self.log_dir, "mcp_transcript.jsonl"), "a")

    def log(self, d):
        d["t"] = time.time()
        self.tlog.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.tlog.flush()

    def robot(self, cmdline):
        cmd, args = parse_command(cmdline)
        if cmd is None:
            return json.dumps(dict(ok=False, error=args))
        raw = call_server(self.port, self.token, cmd, args)
        try:
            rep = json.loads(raw)
        except ValueError:  # help text
            return raw
        msg = (rep.get("message", "") + " " + rep.get("error", "")) if isinstance(rep, dict) else ""
        if isinstance(rep, dict) and isinstance(rep.get("attempt_ended"), dict):
            msg += " " + rep["attempt_ended"].get("message", "")
        if "episode is over" in msg or "No attempts left" in msg or "Stop now" in msg:
            self.finished = True
        if isinstance(rep, dict) and (rep.get("ended") or isinstance(rep.get("attempt_ended"), dict)):
            self.calls_in_attempt = 0
        return json.dumps(rep)

    def call(self, name, args):
        """-> (MCP content list, is_error)"""
        if not self.finished:
            self.calls_in_attempt += 1
        if name == "robot":
            content, err = [dict(type="text", text=self.robot(str(args.get("command", ""))))], False
        elif name == "shell":
            code, out, dt = run_shell(self.sb, str(args.get("command", "")))
            content, err = [dict(type="text", text=json.dumps(dict(exit_code=code, output=out)))], False
        elif name == "view_images":
            paths = [str(x) for x in (args.get("paths") or [])][:8]
            full = [(x, workspace_image(self.sb, x)) for x in paths]
            ok = [x for x, f in full if f]
            out = dict(ok=bool(ok), attached=ok)
            if len(ok) < len(paths):
                out["not_viewable"] = [x for x, f in full if not f]
            content, err = [dict(type="text", text=json.dumps(out))], False
            for x, f in full:
                if f:
                    mime = "image/png" if f.lower().endswith(".png") else "image/jpeg"
                    content.append(dict(type="image", mimeType=mime, data=base64.b64encode(open(f, "rb").read()).decode()))
        else:
            return [dict(type="text", text=f"unknown tool {name!r}")], True
        if not self.finished and self.calls_in_attempt >= self.max_tool_calls:
            raw = call_server(self.port, self.token, "finish", dict(reason="tool-call budget used up (automatic)"))
            try:
                rep = json.loads(raw)
            except ValueError:
                rep = dict(raw=raw)
            msg = rep.get("message", "") + " " + rep.get("error", "")
            if "episode is over" in msg or "No attempts left" in msg or "Stop now" in msg:
                self.finished = True
            self.calls_in_attempt = 0
            content.append(dict(type="text", text=(f"The budget of {self.max_tool_calls} tool calls for this attempt was "
                                                   f"used up, so the attempt was finished automatically. Result: "
                                                   + json.dumps(rep))))
        return content, err


class SchemaOnly:
    """No robot session: the harness preflight only needs tools/list (byte-identical to a real session)."""
    finished = False

    def log(self, d):
        pass

    def call(self, name, args):
        return [dict(type="text", text="schema-only preflight server: no robot session")], True


def serve(sess):
    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send(dict(jsonrpc="2.0", id=None, error=dict(code=-32700, message="parse error")))
            continue
        mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        if mid is None:  # notification (notifications/initialized, cancelled, ...)
            continue
        try:
            if method == "initialize":
                want = params.get("protocolVersion")
                result = dict(protocolVersion=want if want in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                              capabilities=dict(tools=dict(listChanged=False)),
                              serverInfo=dict(name="lffbench-robot", version="1.0"))
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = dict(tools=[dict(name=t["name"], description=t["description"], inputSchema=t["parameters"])
                                     for t in TOOLS])
            elif method == "tools/call":
                name, args = params.get("name"), params.get("arguments") or {}
                t0 = time.time()
                content, err = sess.call(name, args)
                sess.log(dict(ev="tool", name=name, arguments=args, dt=round(time.time() - t0, 2), is_error=err,
                              output=[c["text"] for c in content if c["type"] == "text"],
                              images=sum(1 for c in content if c["type"] == "image"), finished=sess.finished))
                result = dict(content=content, isError=err)
            else:
                send(dict(jsonrpc="2.0", id=mid, error=dict(code=-32601, message=f"method not found: {method}")))
                continue
            send(dict(jsonrpc="2.0", id=mid, result=result))
        except Exception as e:
            sess.log(dict(ev="error", method=method, error=traceback.format_exc()[-2000:]))
            send(dict(jsonrpc="2.0", id=mid, error=dict(code=-32603, message=f"{type(e).__name__}: {e}")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="session.json written by start_robot_session.sh (la style)")
    ap.add_argument("--schema-only", action="store_true", help="serve the tool list only (harness launch preflight)")
    a = ap.parse_args()
    if a.schema_only:
        serve(SchemaOnly())
        return
    if not a.session:
        ap.error("--session is required (or --schema-only)")
    sess = Session(a.session)
    import hashlib  # names + 8-hex sha256 of each value (enough to tell inherited from fresh values, reveals nothing)
    sess.log(dict(ev="start", session=a.session, pid=os.getpid(), env_keys=sorted(ENV_AT_START),
                  env_value_sha8={k: hashlib.sha256(v.encode()).hexdigest()[:8] for k, v in ENV_AT_START.items()}))
    serve(sess)


if __name__ == "__main__":
    main()
