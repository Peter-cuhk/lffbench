"""Batch bookkeeping for Claude-subagent policy runs.

  start   python scripts/claude_batch.py start --batch exp1 --model sonnet --tasks t1:s1,s2 t2:s1 --k 3 --port0 9000
          -> starts one server per (task, seed, condition) run and writes runs/claude/<batch>/manifest.json
             conditions: harness = memory full + feedback (default F1) + k attempts in one agent;
                         noharness = k independent single-attempt agents (memory none, k=1)
  usage   python scripts/claude_batch.py usage --batch exp1 --run <run> --agent <agentId> --tokens N --ms N --tools N
  audit   python scripts/claude_batch.py audit --batch exp1      (runs audit_subagent on every recorded agent)
  table   python scripts/claude_batch.py table --batch exp1 [--batch exp2 ...]
  stop    python scripts/claude_batch.py stop --batch exp1
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "claude")
RUNS_DIR_DEFAULT = RUNS
TRANSCRIPTS = os.environ.get("CLAUDE_TASKS_DIR", "")  # the Claude Code session's tasks/ dir (<agentId>.output files)

PLAYER_PROMPT = """You are controlling a simulated robot arm. Everything you need is in the directory {sb}

Rules (strict):
- First read {sb}/README.md — it explains the robot, the coordinate frames, the cameras, the commands and your task.
- Control the robot only by running shell commands of the form:
  cd {sb} && ./robot <command> key=value ...
  Run one robot command per shell call.
- Look at camera images only with your file-reading tool on the image paths the robot prints (they are inside that directory's obs/ folder).
- Do not read, list, search or execute anything else on this machine: no other files or directories, no other programs, no scripts of your own, no web access. Only README.md, the obs/ images and ./robot.
- Keep going until a robot command tells you to stop or that the episode is over.

When you are told to stop, reply with a short report: what you did (key commands and parameter values), what you observed, any feedback you got, and how you changed your approach."""


PLAYER_PROMPT_LA = """You are controlling a simulated robot arm. Your workspace is the directory {sb}

Rules (strict):
- First read {sb}/README.md: it explains the robot, the commands, the observation files and your task.
- Every shell call must be exactly one of these forms, with nothing chained before or after it:
  cd {sb} && ./robot <command> key=value ...
  cd {sb} && ./shell '<bash command>'
  cd {sb} && ./shell <<'EOF'
  <multi-line bash command, e.g. a python3 heredoc>
  EOF
  ./robot runs one robot command. ./shell runs your command inside the workspace (ls, python3 with numpy / scipy /
  Pillow / OpenCV, writing your own notes or scripts, ...); it has no network and cannot control the robot.
- Use your file-reading tool to look at images and other files, and your file-writing tools if you want, but only on
  files inside {sb}.
- Do not read, list, search or execute anything else on this machine, do not use any other tool, no web access.
- Keep going until a robot command tells you to stop or that the episode is over.

When you are told to stop, reply with a short report: what you did (key commands and parameter values), what you
observed, any feedback you got, and how you changed your approach."""


THINK_ALOUD = """

Think aloud (for the experiment log): before every ./robot command, first write one to three plain sentences as
normal text (not inside a tool call): what you currently see or know, what you conclude from it, and what the next
command is meant to achieve. After each attempt ends, also write what you think went wrong (or right) and what you
will change. Keep these notes short and factual."""


def manifest_path(batch):
    return os.path.join(RUNS, batch, "manifest.json")


def load(batch):
    with open(manifest_path(batch)) as f:
        return json.load(f)


def save(batch, m):
    os.makedirs(os.path.dirname(manifest_path(batch)), exist_ok=True)
    with open(manifest_path(batch), "w") as f:
        json.dump(m, f, indent=1)


def port_free(port):
    import socket
    with socket.socket() as so:
        try:
            so.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def start_one(task, seed, mem, k, port, tag, perception="depth", video=False, style="v1", extra="", feedback="F1"):
    env = dict(os.environ, LFF_RUNS_DIR=os.environ.get("LFF_RUNS_DIR", RUNS_DIR_DEFAULT), LFF_STYLE=style, LFF_EXTRA_ARGS=extra)
    out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "start_robot_session.sh"), task, str(seed), mem, feedback,
                          str(port), "direct", str(k), tag, perception, "1" if video else "0"], capture_output=True, text=True,
                         env=env)
    kv = dict(line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line and line.split("=")[0] in ("SANDBOX", "LOG"))
    if "SANDBOX" not in kv:
        raise RuntimeError(out.stdout + out.stderr)
    return kv["SANDBOX"], kv["LOG"]


def cmd_start(a):
    m = load(a.batch) if os.path.exists(manifest_path(a.batch)) else dict(batch=a.batch, runs=[])
    port = a.port0
    used = {r["port"] for r in m["runs"]}
    for spec in a.tasks:
        task, seeds = spec.split(":")
        for seed in seeds.split(","):
            conds = [("harness", "full", a.k, 0)] + [("noharness", "none", 1, i) for i in range(a.k)]
            for cond, mem, k, rep in conds:
                if a.only and cond != a.only:
                    continue
                while port in used or not port_free(port):
                    port += 1
                tag = f"{a.batch}-{a.model}-{cond}{rep if cond == 'noharness' else ''}" + ("-nominal" if a.nominal else "")
                extra = ("--agent-iface cli" if a.style == "la" else "") + (" --nominal" if a.nominal else "")
                os.environ["LFF_RUNS_DIR"] = os.path.join(RUNS, a.batch)
                sb, log = start_one(task, int(seed), mem, k, port, tag, a.perception, a.video, a.style, extra.strip(), a.feedback)
                used.add(port)
                r = dict(run=os.path.basename(log), task=task, seed=int(seed), model=a.model, cond=cond, rep=rep, k=k,
                         port=port, sandbox=sb, log=log, perception=a.perception, video=a.video, style=a.style,
                         nominal=a.nominal, feedback=a.feedback, think_aloud=a.think_aloud,
                         prompt=(PLAYER_PROMPT_LA if a.style == "la" else PLAYER_PROMPT).format(sb=sb)
                         + (THINK_ALOUD if a.think_aloud else ""))
                m["runs"].append(r)
                print(json.dumps({k_: r[k_] for k_ in ("run", "sandbox")}), flush=True)
    save(a.batch, m)


def cmd_usage(a):
    m = load(a.batch)
    for r in m["runs"]:
        if r["run"] == a.run or r["sandbox"] == a.run:
            r.update(agent=a.agent, tokens=a.tokens, ms=a.ms, tool_uses=a.tools)
    save(a.batch, m)


def cmd_audit(a):
    m = load(a.batch)
    for r in m["runs"]:
        if not r.get("agent"):
            continue
        tr = os.path.join(TRANSCRIPTS, f"{r['agent']}.output")
        if not os.path.exists(tr):
            r["audit"] = "transcript missing"
            continue
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "audit_subagent.py"), tr, r["sandbox"],
                              "--out", os.path.join(r["log"], "audit.json")] + (["--la"] if r.get("style") == "la" else []),
                             capture_output=True, text=True)
        res = json.loads(out.stdout)
        r["audit"] = dict(n_calls=res["n_calls"], n_violations=res["n_violations"])
        if res["n_violations"]:
            print(r["run"], res["violations"][:3])
    save(a.batch, m)
    print("audited", sum(1 for r in m["runs"] if r.get("audit")))


def run_stats(r):
    ev = [json.loads(l) for l in open(os.path.join(r["log"], "events.jsonl"))]
    dones = [e for e in ev if e["ev"] == "done"]
    http = [e for e in ev if e["ev"] == "http"]
    calls = [e for e in ev if e["ev"] == "call"]
    first = next((d["attempt"] for d in dones if d["success"]), None)
    think = [h["think_s"] for h in http if h.get("think_s") is not None]
    srv = [h["server_s"] for h in http]
    t0 = http[0]["t_recv"] if http else None
    t1 = http[-1]["t_send"] if http else None
    return dict(attempts=len(dones), first_success=first, n_calls=len(calls), n_http=len(http),
                think_s=sum(think), server_s=sum(srv), wall_s=(t1 - t0) if t0 else None,
                think_per_cmd=(sum(think) / len(think)) if think else None,
                outcomes=[(d["attempt"], d["success"], d.get("feedback", "")) for d in dones])


def cmd_table(a):
    rows = []
    for b in a.batch:
        for r in load(b)["runs"]:
            try:
                st = run_stats(r)
            except FileNotFoundError:
                continue
            rows.append(dict(r, **st))
    import collections
    # group: (task, model) -> harness runs / noharness groups by seed
    g = collections.defaultdict(lambda: dict(h=[], n=collections.defaultdict(list)))
    for x in rows:
        key = (x["task"], x["model"])
        if x["cond"] == "harness":
            g[key]["h"].append(x)
        else:
            g[key]["n"][x["seed"]].append(x)
    lines = ["| task | model | n | harness @1 | harness @k | no-harness @1 | no-harness @k (k indep.) | think s/cmd | cmds/attempt | wall min/run (harness) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    tot = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for (task, model), d in sorted(g.items()):
        H = [x for x in d["h"] if x["attempts"] > 0]
        K = max([x["k"] for x in H] or [3])
        h1 = sum(1 for x in H if x["first_success"] == 1)
        hk = sum(1 for x in H if x["first_success"] is not None)
        seeds = [s for s, xs in d["n"].items() if all(x["attempts"] > 0 for x in xs)]
        n1 = sum(1 for s in seeds if sorted(d["n"][s], key=lambda x: x["rep"])[0]["first_success"])
        nk = sum(1 for s in seeds if any(x["first_success"] for x in d["n"][s]))
        allx = H + [x for s in seeds for x in d["n"][s]]
        th = [x["think_per_cmd"] for x in allx if x["think_per_cmd"]]
        cpa = [x["n_calls"] / max(x["attempts"], 1) for x in allx]
        wall = [x["wall_s"] / 60 for x in H if x["wall_s"]]
        f = lambda v: f"{sum(v) / len(v):.1f}" if v else "—"
        lines.append(f"| {task} | {model} | {len(H)}/{len(seeds)} | {h1}/{len(H)} | {hk}/{len(H)} | {n1}/{len(seeds)} | "
                     f"{nk}/{len(seeds)} | {f(th)} | {f(cpa)} | {f(wall)} |")
        t = tot[model]
        t[0] += h1; t[1] += hk; t[2] += len(H); t[3] += nk; t[4] += len(seeds)
    for model, t in tot.items():
        lines.append(f"| **all** | {model} | | {t[0]}/{t[2]} | {t[1]}/{t[2]} | | {t[3]}/{t[4]} | | | |")
    print("\n".join(lines))


def cmd_restart(a):
    """Re-run interrupted runs from scratch: stop the old server, start a fresh one (new sandbox, new port),
    keep the old entry under 'superseded'."""
    m = load(a.batch)
    used = {r["port"] for r in m["runs"]}
    port = a.port0
    for i, r in enumerate(list(m["runs"])):
        if r["run"] not in a.runs:
            continue
        try:
            os.kill(int(open(os.path.join(r["log"], "server.pid")).read()), 15)
        except (ProcessLookupError, FileNotFoundError):
            pass
        while port in used or not port_free(port):
            port += 1
        mem = "full" if r["cond"] == "harness" else "none"
        tag = f"{a.batch}-{r['model']}-{r['cond']}{r['rep'] if r['cond'] == 'noharness' else ''}"
        if r.get("style") == "la":
            os.environ["LFF_RUNS_DIR"] = os.path.dirname(r["log"])
        extra = ("--agent-iface cli" if r.get("style") == "la" else "") + (" --nominal" if r.get("nominal") else "")
        sb, log = start_one(r["task"], r["seed"], mem, r["k"], port, tag, r.get("perception", "depth"), r.get("video", False),
                            r.get("style", "v1"), extra.strip(), r.get("feedback", "F2"))
        used.add(port)
        new = {k: v for k, v in r.items() if k not in ("agent", "tokens", "ms", "tool_uses", "audit")}
        new.update(run=os.path.basename(log), port=port, sandbox=sb, log=log,
                   prompt=(PLAYER_PROMPT_LA if r.get("style") == "la" else PLAYER_PROMPT).format(sb=sb))
        m.setdefault("superseded", []).append(dict(r, reason="interrupted (rate limit)"))
        m["runs"][i] = new
        print(json.dumps(dict(run=new["run"], sandbox=sb)))
    save(a.batch, m)


def cmd_stop(a):
    for r in load(a.batch)["runs"]:
        pid = os.path.join(r["log"], "server.pid")
        if os.path.exists(pid):
            try:
                os.kill(int(open(pid).read()), 15)
            except ProcessLookupError:
                pass


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("start"); s.add_argument("--batch", required=True); s.add_argument("--model", required=True)
    s.add_argument("--tasks", nargs="+", required=True); s.add_argument("--k", type=int, default=3)
    s.add_argument("--port0", type=int, default=9000); s.add_argument("--only", choices=["harness", "noharness"])
    s.add_argument("--perception", default="depth", choices=["depth", "rgb", "rgbd"]); s.add_argument("--video", action="store_true")
    s.add_argument("--style", default="v1", choices=["v1", "v2", "la"])
    s.add_argument("--nominal", action="store_true", help="L1 control: hidden calibration error removed")
    s.add_argument("--think-aloud", action="store_true",
                   help="append a request to write 1-3 plain sentences before every robot command (the transcript keeps no "
                        "thinking text, so this is the only way to record the agent's reasoning); changes the prompt")
    s.add_argument("--feedback", default="F1", choices=["F0", "F1", "F2"],
                   help="after a failed attempt: F1 = success/failure only (default since 2026-10-08); F2 = measured error "
                        "(privileged; neither LIBERO-Agent nor RoboDojo gives it), ablation only")
    u = sub.add_parser("usage"); u.add_argument("--batch", required=True); u.add_argument("--run", required=True)
    u.add_argument("--agent", required=True); u.add_argument("--tokens", type=int); u.add_argument("--ms", type=int)
    u.add_argument("--tools", type=int)
    au = sub.add_parser("audit"); au.add_argument("--batch", required=True)
    t = sub.add_parser("table"); t.add_argument("--batch", nargs="+", required=True)
    st = sub.add_parser("stop"); st.add_argument("--batch", required=True)
    rs = sub.add_parser("restart"); rs.add_argument("--batch", required=True); rs.add_argument("--runs", nargs="+", required=True)
    rs.add_argument("--port0", type=int, default=9300)
    a = ap.parse_args()
    dict(start=cmd_start, usage=cmd_usage, audit=cmd_audit, table=cmd_table, stop=cmd_stop, restart=cmd_restart)[a.cmd](a)


if __name__ == "__main__":
    main()
