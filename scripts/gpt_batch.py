"""Batch of GPT-player runs on robot_server sessions (same sessions / README / images as the Claude players).

  start  python scripts/gpt_batch.py start --batch exp2 --model gpt-6-astra --perception rgb --video \
             --tasks l1_bias_place:1000,1001 l5_slide_to_target:1000,1001 --port0 9710
         -> one server + one gpt_player process per (task, seed); manifest at runs/gpt/<batch>/manifest.json
  status python scripts/gpt_batch.py status --batch exp2
  table  python scripts/gpt_batch.py table --batch exp2
  stop   python scripts/gpt_batch.py stop --batch exp2       (stops the robot servers and any player still running)
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from claude_batch import run_stats  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "gpt")
# OPENAI_API_KEY / OPENAI_BASE_URL: taken from this file if it exists (KEY=value lines), else from the environment
ENV_FILE = os.environ.get("LFF_ENV_FILE") or os.path.join(ROOT, ".env")


def mpath(batch):
    return os.path.join(RUNS, batch, "manifest.json")


def load(batch):
    return json.load(open(mpath(batch)))


def save(batch, m):
    os.makedirs(os.path.dirname(mpath(batch)), exist_ok=True)
    json.dump(m, open(mpath(batch), "w"), indent=1)


def port_free(port):
    import socket
    with socket.socket() as so:
        try:
            so.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def cmd_start(a):
    m = load(a.batch) if os.path.exists(mpath(a.batch)) else dict(batch=a.batch, runs=[])
    used = {r["port"] for r in m["runs"]}
    port = a.port0
    env = dict(os.environ, LFF_RUNS_DIR=os.path.join(RUNS, a.batch), LFF_STYLE=a.style,
               LFF_EXTRA_ARGS="--nominal" if a.nominal else "")
    for spec in a.tasks:
        task, seeds = spec.split(":")
        for seed in seeds.split(","):
            while port in used or not port_free(port):
                port += 1
            out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "start_robot_session.sh"), task, seed, "full", a.feedback,
                                  str(port), "direct", str(a.k), f"{a.batch}-{a.harness}" + ("-nominal" if a.nominal else ""), a.perception,
                                  "1" if a.video else "0"], capture_output=True, text=True, env=env)
            kv = dict(l.split("=", 1) for l in out.stdout.splitlines() if l.startswith(("SANDBOX=", "LOG=")))
            if "LOG" not in kv:
                print("server failed", task, seed, out.stdout[-500:], out.stderr[-500:])
                continue
            used.add(port)
            log = kv["LOG"]
            if a.harness == "gpt_player":
                player = (f"cd {ROOT} && source run_env.sh >/dev/null 2>&1 && "
                          f"{{ [ ! -f {ENV_FILE} ] || {{ set -a && source {ENV_FILE} && set +a; }}; }} && "
                          f"exec python scripts/gpt_player.py --session {log}/session.json --model {a.model} "
                          f"--effort {a.effort} --style {a.style} > {log}/gpt_player.log 2>&1")
            elif a.harness == "rpent":
                player = (f"bash {ROOT}/scripts/rpent_player.sh {log}/session.json --model {a.model} --effort {a.effort} "
                          f"> {log}/rpent_player.out 2>&1")
            else:  # gptpolicy
                player = (f"GPTPOLICY_MODEL={a.model} GPTPOLICY_EFFORT={a.effort} GPTPOLICY_KEEP_SERVER=1 "
                          f"bash {ROOT}/scripts/gptpolicy_player.sh {log}/session.json > {log}/gptpolicy_player.out 2>&1")
            # detach stdio: otherwise a player that does not exec keeps the caller's stdout pipe open and the
            # caller (e.g. `gpt_batch.py start ... | grep`) blocks until the run ends
            p = subprocess.Popen(["bash", "-c", player], start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            r = dict(run=os.path.basename(log), task=task, seed=int(seed), model=a.model, effort=a.effort, cond="harness",
                     rep=0, k=a.k, port=port, sandbox=kv["SANDBOX"], log=log, perception=a.perception, player_pid=p.pid,
                     harness=a.harness, style=a.style, nominal=a.nominal)
            m["runs"].append(r)
            print(json.dumps(dict(run=r["run"], pid=p.pid)), flush=True)
    save(a.batch, m)


def cmd_status(a):
    for r in load(a.batch)["runs"]:
        tr = os.path.join(r["log"], "gpt_transcript.jsonl")
        errs = 0
        end = None
        if os.path.exists(tr):
            for l in open(tr):
                d = json.loads(l)
                errs += d["ev"] == "api_error"
                if d["ev"] in ("end", "abort"):
                    end = d
        ev = [json.loads(l) for l in open(os.path.join(r["log"], "events.jsonl"))]
        n = sum(1 for e in ev if e["ev"] == "call")
        dones = [e for e in ev if e["ev"] == "done"]
        alive = True
        try:
            os.kill(r["player_pid"], 0)
        except (ProcessLookupError, KeyError, PermissionError):
            alive = False
        if end:
            state = "END" if end["ev"] == "end" else "ABORT " + end.get("reason", "")
        else:
            state = "running" if alive else ("END" if dones and (dones[-1]["success"] or len(dones) >= r["k"]) else "ABORT player exited")
        print(f"{r.get('harness', 'gpt_player'):10s} {r['task']:24s} s{r['seed']}  cmds={n:3d}  api_err={errs}  "
              f"attempts={[('S' if d['success'] else 'F') for d in dones]}  {state}")


def cmd_table(a):
    import collections
    rows = []
    for b in a.batch:
        for r in load(b)["runs"]:
            try:
                st = run_stats(r)
            except FileNotFoundError:
                continue
            rows.append(dict(r, **st, first_grasp=first_grasp(r), harness=r.get("harness", "gpt_player")))
    lines = ["| harness | task | seed | attempts | first success | cmds | close_gripper (held / total) | think s/cmd | wall min |",
             "|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: (x["task"], x["harness"], x["seed"])):
        lines.append(f"| {x['harness']} | {x['task']} | {x['seed']} | {x['attempts']} | {x['first_success'] or '—'} | "
                     f"{x['n_calls']} | {x['first_grasp'][1]}/{x['first_grasp'][0]} | "
                     f"{(x['think_per_cmd'] or 0):.1f} | {(x['wall_s'] or 0) / 60:.1f} |")
    by = collections.defaultdict(list)
    for x in rows:
        by[x["harness"]].append(x)
    lines.append("")
    for h, xs in sorted(by.items()):
        n = len(xs)
        g = [sum(x["first_grasp"][i] for x in xs) for i in (0, 1)]
        lines.append(f"{h}: @1 = {sum(1 for x in xs if x['first_success'] == 1)}/{n}, "
                     f"@k = {sum(1 for x in xs if x['first_success'])}/{n}, close_gripper held {g[1]}/{g[0]}")
    print("\n".join(lines))


def cmd_stop(a):
    for r in load(a.batch)["runs"]:
        pids = [r.get("player_pid")]
        pf = os.path.join(r["log"], "server.pid")
        if os.path.exists(pf):
            pids.append(int(open(pf).read()))
        for pid in pids:
            if pid:
                try:
                    os.kill(pid, 15)  # SIGTERM: the server closes rollout.mp4 first
                except (ProcessLookupError, PermissionError):
                    pass


def first_grasp(r):
    """(close_gripper calls, of which ended holding an object) over the whole run."""
    tot = held = 0
    for l in open(os.path.join(r["log"], "events.jsonl")):
        d = json.loads(l)
        if d.get("ev") == "call" and d.get("cmd") == "close_gripper":
            tot += 1
            held += bool(d.get("report", {}).get("holding_object"))
    return tot, held


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("start")
    s.add_argument("--batch", required=True); s.add_argument("--model", default="gpt-6-astra")
    s.add_argument("--effort", default="medium"); s.add_argument("--tasks", nargs="+", required=True)
    s.add_argument("--k", type=int, default=3); s.add_argument("--port0", type=int, default=9710)
    s.add_argument("--perception", default="rgb", choices=["rgb", "depth", "rgbd"]); s.add_argument("--video", action="store_true")
    s.add_argument("--harness", default="gpt_player", choices=["gpt_player", "rpent", "gptpolicy"])
    s.add_argument("--style", default="v1", choices=["v1", "v2", "la"])
    s.add_argument("--nominal", action="store_true", help="L1 control: hidden calibration error removed")
    s.add_argument("--feedback", default="F1", choices=["F0", "F1", "F2"],
                   help="F1 = success/failure only (default since 2026-10-08); F2 = measured error, ablation only")
    st = sub.add_parser("status"); st.add_argument("--batch", required=True)
    t = sub.add_parser("table"); t.add_argument("--batch", nargs="+", required=True)
    sp = sub.add_parser("stop"); sp.add_argument("--batch", required=True)
    a = ap.parse_args()
    dict(start=cmd_start, status=cmd_status, table=cmd_table, stop=cmd_stop)[a.cmd](a)


if __name__ == "__main__":
    main()
