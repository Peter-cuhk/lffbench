"""Drive one robot_server session with a model through its provider's official API (the unified harness).

Every provider gets exactly the same input: the sandbox README as its instructions and the same tools; only the
wire format differs (lffbench/agent/backends/). After every command the JSON report is returned as the tool output.
Nothing else is sent: no files, no memory, no paths outside the sandbox.

  python scripts/api_player.py --session runs/<...>/session.json --provider anthropic --model claude-opus-5-5 --effort high --style la

Providers (key / endpoint from the environment, e.g. `set -a; source .env; set +a`):
  openai     OpenAI Responses API          OPENAI_API_KEY [OPENAI_BASE_URL]
  anthropic  Anthropic Messages API        ANTHROPIC_API_KEY
  gemini     Gemini API (google-genai)     GEMINI_API_KEY
  chat       any OpenAI-compatible Chat Completions endpoint (vLLM / SGLang for open models, xAI, ...)
             LFF_CHAT_BASE_URL, LFF_CHAT_API_KEY
`--effort` is the provider's reasoning knob: OpenAI reasoning.effort, Anthropic output_config.effort, Gemini
thinking_level, chat: reasoning_effort (omitted when "none").

--style la (LIBERO-Agent-style workspace, the paper's interface): the sandbox is the agent's episode workspace. Besides
`robot` the model gets `shell` (a command run in the workspace inside a jail: no network, robot server unreachable,
host data hidden, see /opt/lffjail/jail_inner.sh) and `view_images` (any image file in the workspace, including ones
it made itself). Tool calls per attempt are capped (agent_iface.json: max_tool_calls); at the cap the attempt is
finished for it.
Logs in the run's log dir: transcript.jsonl (every request / reply / tool call), final_report.md.
"""
import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if not os.environ.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"  # the venv's own CA bundle lacks the proxy's issuer

from lffbench.agent.backends import make_backend  # noqa: E402
from lffbench.agent.context import Conversation  # noqa: E402

PROVIDERS = ("openai", "anthropic", "gemini", "chat")

TOOL = dict(
    name="robot",
    description=("Run one robot command. `command` is exactly what you would type after `./robot` in the "
                 "instructions (e.g. `observe`). Returns the command's JSON report."),
    parameters=dict(type="object", properties=dict(command=dict(type="string", description="the robot command")),
                    required=["command"], additionalProperties=False),
)

VIEW_TOOL = dict(
    name="view_images",
    description=("Look at camera images the robot has written. `paths` are image paths exactly as printed by robot "
                 "commands (e.g. obs/a1_s003_wrist.jpg); any image of this episode can be viewed again. The images "
                 "are attached right after this call."),
    parameters=dict(type="object", properties=dict(paths=dict(type="array", items=dict(type="string"),
                                                              description="image paths to view")),
                    required=["paths"], additionalProperties=False),
)

TOOL_LA = dict(
    name="robot",
    description=("Run one robot command. `command` is exactly one of the robot commands in the instructions, e.g. "
                 "`status` or `move_eef z=1.0 gripper=open`. Returns the command's JSON report, including the paths "
                 "of the new observation files."),
    parameters=TOOL["parameters"],
)

SHELL_TOOL = dict(
    name="shell",
    description=("Run a shell command (bash) in your workspace directory, e.g. `ls obs` or `python3 analyse.py`. "
                 "python3 has numpy, scipy, Pillow and OpenCV. No network; programs cannot control the robot. "
                 "Time limit 60 s per call. Returns the exit code and the combined stdout/stderr (long output is "
                 "truncated in the middle)."),
    parameters=dict(type="object", properties=dict(command=dict(type="string", description="the bash command")),
                    required=["command"], additionalProperties=False),
)

VIEW_TOOL_LA = dict(
    name="view_images",
    description=("Look at image files in your workspace (.jpg / .jpeg / .png), e.g. obs/a1_s003_wrist.jpg or an image "
                 "you made yourself; paths are relative to the workspace. At most 8 per call. The images are attached "
                 "right after this call."),
    parameters=dict(type="object", properties=dict(paths=dict(type="array", items=dict(type="string"),
                                                              description="image paths to view")),
                    required=["paths"], additionalProperties=False),
)

PREAMBLE_LA = ("You control the robot described below and work in the workspace described below. Tools: `robot` runs "
               "one robot command per call (the argument is exactly a command from the instructions, e.g. `status`); "
               "`shell` runs a shell command in your workspace; `view_images` shows you image files from the "
               "workspace. When a robot command tells you to stop or that the episode is over, stop calling tools and "
               "reply with a short report: what you did (key commands and parameter values), what you observed, any "
               "feedback you got, and how you changed your approach.\n\n")

PREAMBLE = ("You control the robot described below. You act only through the `robot` tool, one command per call. "
            "When a command tells you to stop or that the episode is over, stop calling the tool and reply with a "
            "short report: what you did (key commands and parameter values), what you observed, any feedback you "
            "got, and how you changed your approach.\n\n")

JAIL = "/opt/lffjail/jail_inner.sh"
SHELL_TIMEOUT = 60
SHELL_MAX_CHARS = 12000


def run_shell(workspace, command):
    """Run `command` in the jailed workspace; returns (exit code, output text, seconds)."""
    t0 = time.time()
    argv = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--", JAIL, workspace,
            "timeout", "-k", "5", str(SHELL_TIMEOUT), "bash", "-c", command]
    try:
        r = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=SHELL_TIMEOUT + 30)
        code, out = r.returncode, r.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        code, out = 124, (e.stdout or b"").decode("utf-8", "replace")
    if code == 124:
        out += f"\n[killed: time limit of {SHELL_TIMEOUT} s]"
    if len(out) > SHELL_MAX_CHARS:
        h = SHELL_MAX_CHARS // 2
        out = out[:h] + f"\n[... {len(out) - SHELL_MAX_CHARS} characters truncated ...]\n" + out[-h:]
    return code, out, time.time() - t0


def workspace_image(workspace, rel):
    """Resolve a model-given image path inside the workspace (no escaping via .. or symlinks); None if not allowed."""
    if not isinstance(rel, str) or not rel or rel.startswith("/"):
        return None
    root = os.path.realpath(workspace)
    full = os.path.realpath(os.path.join(root, rel))
    if not full.startswith(root + os.sep) or not os.path.isfile(full):
        return None
    if os.path.splitext(full)[1].lower() not in (".jpg", ".jpeg", ".png") or os.path.getsize(full) > 8 * 2 ** 20:
        return None
    return full


class FileImage:
    """The exact JPEG bytes the server wrote (same bytes a Claude Code player reads)."""

    def __init__(self, path):
        self.path = path
        base = os.path.basename(path)
        self.camera = "wrist" if "wrist" in base else ("agentview" if "agentview" in base else "camera")
        self.label = base

    def data_url(self, quality=None):
        mime = "image/png" if self.path.lower().endswith(".png") else "image/jpeg"
        with open(self.path, "rb") as f:
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def parse_command(s):
    s = s.strip()
    s = s.removeprefix("./robot").strip() if s.startswith("./robot") else (s[len("robot "):] if s.startswith("robot ") else s)
    toks = shlex.split(s.strip())
    if not toks:
        return "help", {}
    cmd, args = toks[0], {}
    for a in toks[1:]:
        if "=" not in a:
            return None, f"bad argument {a!r}: use key=value"
        k, v = a.split("=", 1)
        try:
            args[k] = json.loads(v)
        except ValueError:
            args[k] = v
    return cmd, args


def call_server(port, token, cmd, args):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/call", data=json.dumps({"cmd": cmd, "args": args}).encode(),
                                 headers={"Content-Type": "application/json", "X-Token": token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=900) as r:
        return r.read().decode()


MAX_NO_TOOL = 3  # nudges in a row for replies without a tool call; the next such reply ends the run


def continue_nudge(port, token, stx=None):
    """The user message sent when the model replies without a tool call before the episode is over. Native harnesses
    send the same text when their CLI exits early (scripts/harness_run.py). stx: a status reply already fetched."""
    try:  # tell it where it is: models sometimes believe the episode ended after a failed attempt
        if stx is None:
            stx = json.loads(call_server(port, token, "status", {}))
        where = (f" You are in attempt {stx.get('attempt')} of {stx.get('max_attempts')}; the episode is NOT over."
                 if stx.get("attempt") else "")
    except Exception:
        where = ""
    return "Continue: issue the next robot command with the `robot` tool." + where


def build_backend(a, v2):
    """The model backend for --provider; image pruning only where the provider allows editing earlier turns."""
    if a.provider == "openai":
        return make_backend("openai", model=a.model, reasoning_effort=a.effort, reasoning_summary="auto",
                            max_live_images=None if v2 else a.max_live_images, cache_breakpoints="none", max_retries=1)
    if a.provider == "anthropic":  # append-only history (thinking blocks are bound to it): never prune
        return make_backend("anthropic", model=a.model, effort=a.effort, max_live_images=None)
    if a.provider == "gemini":
        return make_backend("gemini", model=a.model, thinking_level=a.effort,
                            max_live_images=None if v2 else a.max_live_images)
    extra = None if a.effort in ("", "none", None) else dict(reasoning_effort=a.effort)
    return make_backend("chat", model=a.model, max_tokens=a.max_output_tokens, temperature=None, extra_body=extra,
                        max_live_images=None if v2 else a.max_live_images)


def context_overflow(err):
    """The request no longer fits: token limit (OpenAI / Anthropic / Gemini / Kimi wordings) or too many images."""
    e = err.lower()
    return (("context" in e and ("length" in e or "window" in e or "too long" in e)) or "prompt is too long" in e
            or ("exceed" in e and "token" in e) or "too many images" in e or "maximum number of images" in e)


def live_images(conv):
    return sum(1 for it in conv.items if it["kind"] == "user" and it.get("live") for p in it["parts"]
               if p["type"] == "image")


def main(argv=None, default_provider=None, transcript_name="transcript.jsonl"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session.json written by start_robot_session.sh")
    ap.add_argument("--provider", default=default_provider, choices=PROVIDERS, required=default_provider is None)
    ap.add_argument("--model", default="gpt-6-astra" if default_provider == "openai" else None,
                    required=default_provider is None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--max-live-images", type=int, default=16)
    ap.add_argument("--max-output-tokens", type=int, default=32768, help="chat provider only")
    ap.add_argument("--api-retries", type=int, default=8,
                    help="retries of a failed API call before giving up (back-off 5 s doubling to 120 s: ~9 min)")
    ap.add_argument("--max-request-images", type=int, default=500,
                    help="when the conversation holds more viewed images than this, switch to the context-full "
                         "policy (Claude through a gateway failed at 601 images, 2026-10-15)")
    ap.add_argument("--style", default="v1", choices=["v1", "v2", "la"],
                    help="v2: images are not attached automatically; the model views them on demand with "
                         "view_images, and no history is pruned (the agent manages its own context)")
    a = ap.parse_args(argv)

    sess = json.load(open(a.session))
    sb, port = sess["sandbox"], sess["port"]
    log_dir = os.path.dirname(os.path.abspath(a.session))
    la = a.style == "la"
    if la:  # no robot CLI in a la workspace: the token is only in the log dir
        iface = json.load(open(os.path.join(log_dir, "agent_iface.json")))
        token, max_tool_calls = iface["token"], int(iface["max_tool_calls"])
    else:
        token = open(os.path.join(sb, "robot")).read().split('TOKEN = "', 1)[1].split('"', 1)[0]
    readme = open(os.path.join(sb, "README.md")).read()
    tlog = open(os.path.join(log_dir, transcript_name), "a")

    def log(d):
        d["t"] = time.time()
        tlog.write(json.dumps(d, ensure_ascii=False) + "\n")
        tlog.flush()

    v2 = a.style in ("v2", "la")  # la = v2 robot interface + workspace tools
    tools = [TOOL_LA, SHELL_TOOL, VIEW_TOOL_LA] if la else ([TOOL, VIEW_TOOL] if v2 else [TOOL])
    backend = build_backend(a, v2)
    conv = Conversation((PREAMBLE_LA if la else PREAMBLE) + readme)
    log(dict(ev="start", provider=a.provider, model=a.model, effort=a.effort, style=a.style, backend=backend.config(),
             instructions=conv.system, tools=tools))

    def images_part(paths, title):
        parts = [dict(type="text", text=title)]
        for p in paths:
            parts.append(dict(type="image", image=FileImage(os.path.join(sb, p))))
        return parts

    st = json.loads(call_server(port, token, "status", {}))
    if v2:  # images stay files; the model decides what to look at
        first = [dict(type="text", text=("Begin. Current status (robot command `status`):\n" if la else
                                         "Begin. Current status (`./robot status`):\n") + json.dumps(st))]
    else:
        first = [dict(type="text", text="Begin. Current status (`./robot status`):\n" +
                      json.dumps({k: v for k, v in st.items() if k != "images"}))]
        first += images_part(st.get("images", []), "Current camera images (agentview, wrist):")[1:]
    conv.add_user(first, live=True)
    log(dict(ev="user", text=first[0]["text"], images=st.get("images", [])))

    finished, no_tool, report = False, 0, None
    calls_in_attempt = 0  # la: tool calls (robot + shell + view_images) in the current attempt
    usage_tot = dict(input_tokens=0, cached_tokens=0, cache_write_tokens=0, output_tokens=0, reasoning_tokens=0)
    for turn in range(a.max_turns):
        resp, err = None, None
        if v2 and backend.max_live_images is None and live_images(conv) > a.max_request_images:
            backend.max_live_images = 32  # same context-full policy as below, before the request limit is hit
            log(dict(ev="context_overflow", reason="image count", n_images=live_images(conv), keep_images=32))
        for r in range(a.api_retries + 1):
            t0 = time.time()
            try:
                out = backend.act(conv, tools)
            except Exception as e:  # flaky endpoint: back off and retry the identical request
                err = f"{type(e).__name__}: {str(e)[:300]}"
                if v2 and context_overflow(err):
                    # the context is full (every provider): from now on keep only the newest viewed images
                    backend.max_live_images = 32 if backend.max_live_images is None else max(backend.max_live_images // 2, 4)
                    log(dict(ev="context_overflow", reason=err[:120], keep_images=backend.max_live_images))
                log(dict(ev="api_error", turn=turn, retry=r, error=err, dt=time.time() - t0))
                time.sleep(min(120, 5 * 2 ** r))
                continue
            if out.status == "refusal":  # the provider declined: no fallback model (it would change the model under test)
                resp, err = out, out.error
                break
            if out.status == "incomplete":  # truncated / malformed turn: not appended, ask again
                err = out.error
                log(dict(ev="api_error", turn=turn, retry=r, error=err, dt=time.time() - t0, usage=out.usage))
                continue
            resp = out
            break
        if resp is None:
            log(dict(ev="abort", reason="api failed", error=err))
            print("ABORT api failed:", err, flush=True)
            sys.exit(3)
        for k in usage_tot:
            usage_tot[k] += resp.usage.get(k, 0)
        if resp.status == "refusal":
            log(dict(ev="refusal", turn=turn, latency_s=resp.latency_s, usage=resp.usage, error=resp.error,
                     text=resp.text))
            log(dict(ev="abort", reason="refusal", error=resp.error))
            print("ABORT refusal:", resp.error, flush=True)
            break
        conv.add_assistant(resp.text, resp.tool_calls, raw=resp.raw, backend=backend.name)
        log(dict(ev="assistant", turn=turn, latency_s=resp.latency_s, usage=resp.usage, text=resp.text,
                 reasoning_summary=resp.reasoning_summary,
                 reasoning_summary_status="recorded" if resp.reasoning_summary else "not_returned",
                 status=resp.status, error=resp.error,
                 tool_calls=[dict(id=c.id, name=c.name, arguments=c.arguments) for c in resp.tool_calls],
                 request_stats=resp.request_stats))
        if finished:
            report = resp.text
            break
        if not resp.tool_calls:
            no_tool += 1
            nudge = continue_nudge(port, token)
            if no_tool > MAX_NO_TOOL:
                log(dict(ev="abort", reason="no tool call 4 times"))
                break
            conv.add_user([dict(type="text", text=nudge)], live=False)
            log(dict(ev="user", text=nudge))
            continue
        no_tool = 0
        for c in resp.tool_calls:
            if la and not finished:
                calls_in_attempt += 1
            if la and c.name == "shell":
                code, out, dt = run_shell(sb, str(c.arguments.get("command", "")))
                out_text = json.dumps(dict(exit_code=code, output=out))
                conv.add_tool_output(c.id, c.name, out_text)
                log(dict(ev="tool", turn=turn, command="shell: " + str(c.arguments.get("command", "")), output=out_text,
                         shell_s=round(dt, 2)))
                continue
            if la and c.name == "view_images":
                paths = [str(x) for x in (c.arguments.get("paths") or [])][:8]
                full = [(x, workspace_image(sb, x)) for x in paths]
                ok_paths = [x for x, f in full if f]
                out = dict(ok=bool(ok_paths), attached=ok_paths)
                if len(ok_paths) < len(paths):
                    out["not_viewable"] = [x for x, f in full if not f]
                conv.add_tool_output(c.id, c.name, json.dumps(out))
                log(dict(ev="tool", turn=turn, command="view_images " + " ".join(paths), output=json.dumps(out), images=ok_paths))
                if ok_paths:
                    parts = [dict(type="text", text="Requested images:")]
                    for x, f in full:
                        if f:
                            parts.append(dict(type="image", image=FileImage(f)))
                    conv.add_user(parts, live=True)
                continue
            if c.name == "view_images":
                paths = [str(x) for x in (c.arguments.get("paths") or [])][:8]
                ok_paths = [x for x in paths if x.startswith("obs/") and "/.." not in x and ".." not in x
                            and os.path.exists(os.path.join(sb, x))]
                bad = [x for x in paths if x not in ok_paths]
                out = dict(ok=bool(ok_paths), attached=ok_paths)
                if bad:
                    out["not_found"] = bad
                conv.add_tool_output(c.id, c.name, json.dumps(out))
                log(dict(ev="tool", turn=turn, command="view_images " + " ".join(paths), output=json.dumps(out), images=ok_paths))
                if ok_paths:
                    conv.add_user(images_part(ok_paths, "Requested images:"), live=True)
                continue
            cmdline = str(c.arguments.get("command", ""))
            cmd, args = parse_command(cmdline)
            if cmd is None:
                out_text, paths = json.dumps(dict(ok=False, error=args)), []
            else:
                raw = call_server(port, token, cmd, args)
                try:
                    rep = json.loads(raw)
                    if v2:  # keep the image paths in the reply: the model views them itself if it wants
                        paths = []
                        out_text = json.dumps(rep)
                    else:
                        paths = rep.pop("images", []) if isinstance(rep, dict) else []
                        out_text = json.dumps(rep)
                except ValueError:  # help text
                    rep, paths, out_text = {}, [], raw
                msg = (rep.get("message", "") + " " + rep.get("error", "")) if isinstance(rep, dict) else ""
                if isinstance(rep, dict) and isinstance(rep.get("attempt_ended"), dict):
                    msg += " " + rep["attempt_ended"].get("message", "")
                if "episode is over" in msg or "No attempts left" in msg or "Stop now" in msg:
                    finished = True
                if isinstance(rep, dict) and (rep.get("ended") or isinstance(rep.get("attempt_ended"), dict)):
                    calls_in_attempt = 0
            conv.add_tool_output(c.id, c.name, out_text)
            log(dict(ev="tool", turn=turn, command=cmdline, output=out_text, images=paths))
            if paths and not v2:
                conv.add_user(images_part(paths, f"Camera images after `{cmdline}` (agentview, wrist):"), live=True)
        if la and not finished and calls_in_attempt >= max_tool_calls:
            raw = call_server(port, token, "finish", dict(reason="tool-call budget used up (automatic)"))
            try:
                rep = json.loads(raw)
            except ValueError:
                rep = dict(raw=raw)
            msg = rep.get("message", "") + " " + rep.get("error", "")
            if "episode is over" in msg or "No attempts left" in msg or "Stop now" in msg:
                finished = True
            calls_in_attempt = 0
            note = (f"The budget of {max_tool_calls} tool calls for this attempt was used up, so the attempt was "
                    f"finished automatically. Result: " + json.dumps(rep))
            conv.add_user([dict(type="text", text=note)], live=False)
            log(dict(ev="user", text=note, auto_finish=True))
        if finished:
            ask = "Stop now and reply with your short report (no more tool calls)."
            conv.add_user([dict(type="text", text=ask)], live=False)
            log(dict(ev="user", text=ask))
    if report:
        with open(os.path.join(log_dir, "final_report.md"), "w") as f:
            f.write(report)
    log(dict(ev="end", finished=finished, usage_total=usage_tot))
    print(json.dumps(dict(finished=finished, usage=usage_tot, report=(report or "")[:200])), flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
