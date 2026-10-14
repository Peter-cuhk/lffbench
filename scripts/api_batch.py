"""Batch of runs with a concurrency limit: the runner for the paper's main table (reference driver scripts/api_player.py)
and, with --harness, for native agent harnesses (side table, docs/15).

  start   python scripts/api_batch.py start --batch m1_opus --provider anthropic --model claude-opus-5-5 --effort high \
              --tasks l1_height_bias:3000,3001 lh_bridge:3000 --k 3 --concurrency 6 --video
          -> writes runs/api/<batch>/manifest.json (one entry per run, status "pending") and starts a background
             runner that keeps at most --concurrency runs going (robot server + api_player each) and stops each
             server when its player exits.
  status  python scripts/api_batch.py status --batch m1_opus
  table   python scripts/api_batch.py table --batch m1_opus [--batch ...]     (same table as claude_batch + tokens)
  stop    python scripts/api_batch.py stop --batch m1_opus                    (runner, players, servers; runs -> "stopped")
  retry   python scripts/api_batch.py retry --batch m1_opus [--runs <run> ...] (default: every failed / stopped / timeout run;
          old logs stay, listed under "superseded"; starts the runner again)
  contract python scripts/api_batch.py contract --batch b1 [--batch b2 ...]  (contract_sha256 per instance and condition;
          flags instances whose hash differs between batches: those cells must not share a table)

--harness (default ours): which agent harness drives each run, through scripts/harness_run.py (lffbench/harnesses):
  ours              scripts/api_player.py, unchanged (--provider as before)
  claude_code       Claude Code CLI --bare, only our MCP tools, one-time config dirs (arm B; provider anthropic)
  claude_code_full  Claude Code CLI without --bare (full system prompt), otherwise the same (arm B')
  codex             Codex CLI exec --ephemeral, empty CODEX_HOME, only our MCP tools reachable (arm F; provider openai)
Native harnesses are refused at launch (status "refused") unless their preset prompt / tool list matches the pin
written by tests/harness_selfcheck (--allow-unpinned to launch anyway; the preflight still runs). Every run gets
harness.json, contract.json, trajectory.jsonl and harness_audit.json in its log dir.

Conditions (same as claude_batch): harness = memory full, k attempts in one agent; noharness = independent single-attempt
agents (memory none, k=1), --indep-reps of them per instance (default k, for pass@k^iid). Protocol of 2026-10-11 (default):
all independent runs first, the history runs only after every independent run has finished (--no-barrier to mix them);
history runs with --hide-k (README and status say 1 attempt, byte-identical to the independent run; --no-hide-k to show k).
API keys come from LFF_ENV_FILE (default <repo>/.env, KEY=value lines) or the environment.
"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import claude_batch  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "api")
ENV_FILE = os.environ.get("LFF_ENV_FILE") or os.path.join(ROOT, ".env")
ACTIVE = ("starting", "running")
RETRYABLE = ("failed", "stopped", "timeout", "server_failed")
HARNESS_PROVIDER = dict(claude_code="anthropic", claude_code_full="anthropic", codex="openai")


def bdir(batch):
    return os.path.join(RUNS, batch)


def mpath(batch):
    return os.path.join(bdir(batch), "manifest.json")


def load(batch):
    with open(mpath(batch)) as f:
        return json.load(f)


def save(batch, m):
    os.makedirs(bdir(batch), exist_ok=True)
    tmp = mpath(batch) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1)
    os.replace(tmp, mpath(batch))


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, TypeError, ValueError):
        return False


def runner_alive(batch):
    p = os.path.join(bdir(batch), "runner.pid")
    return os.path.exists(p) and alive(open(p).read().strip())


def launch_runner(batch):
    log = open(os.path.join(bdir(batch), "runner.log"), "a")
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_runner", "--batch", batch], cwd=ROOT,
                         start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    with open(os.path.join(bdir(batch), "runner.pid"), "w") as f:
        f.write(str(p.pid))
    return p.pid


# ------------------------------------------------------------------------------------------------------------ start
def cmd_start(a):
    if runner_alive(a.batch):
        sys.exit(f"batch {a.batch}: runner already running; stop it first or use another batch name")
    m = load(a.batch) if os.path.exists(mpath(a.batch)) else dict(batch=a.batch, runs=[], superseded=[])
    m.update(concurrency=a.concurrency, barrier=not a.no_barrier, port0=a.port0, run_timeout_s=a.run_timeout)
    reps = a.k if a.indep_reps is None else a.indep_reps
    if a.harness != "ours":
        need = HARNESS_PROVIDER[a.harness]
        if a.provider and a.provider != need:
            sys.exit(f"--harness {a.harness} drives {need} models only (got --provider {a.provider})")
        a.provider = need
    elif not a.provider:
        sys.exit("--provider is required with --harness ours")
    conds = [c.strip() for c in a.conds.split(",") if c.strip()]
    n0 = len(m["runs"])
    for cond in conds:  # the order of --conds is the order of the phases
        for spec in a.tasks:
            task, seeds = spec.split(":")
            for seed in seeds.split(","):
                plan = [("harness", "full", a.k, 0)] if cond == "harness" else \
                       [("noharness", "none", 1, i) for i in range(reps)]
                for c, mem, k, rep in plan:
                    m["runs"].append(dict(id=f"{len(m['runs']):04d}", task=task, seed=int(seed), provider=a.provider,
                                          agent_harness=a.harness, api_base_url=a.api_base_url,
                                          max_budget_usd=a.max_budget_usd, allow_unpinned=a.allow_unpinned,
                                          model=a.model, effort=a.effort, cond=c, rep=rep, k=k, mem=mem,
                                          hide_k=(c == "harness" and not a.no_hide_k), perception=a.perception,
                                          video=a.video, style="la", feedback=a.feedback, phase=conds.index(cond),
                                          nominal=a.nominal, status="pending"))
    save(a.batch, m)
    print(f"{len(m['runs']) - n0} runs planned ({', '.join(conds)}); manifest {mpath(a.batch)}")
    print("runner pid", launch_runner(a.batch))


# ------------------------------------------------------------------------------------------------------------ runner
class Runner:
    def __init__(self, batch):
        self.batch = batch
        self.lock = threading.Lock()
        self.m = load(batch)
        self.used_ports = set()
        self.stop = False

    def update(self, r, **kw):
        with self.lock:
            r.update(kw)
            save(self.batch, self.m)

    def next_run(self):
        with self.lock:
            runs = self.m["runs"]
            pending = [r for r in runs if r["status"] == "pending"]
            if not pending:
                return None
            if self.m.get("barrier", True):
                phase = min(r["phase"] for r in pending)
                if any(r["status"] in ACTIVE and r["phase"] < phase for r in runs):
                    return "wait"
                pending = [r for r in pending if r["phase"] == phase]
            r = pending[0]
            r["status"] = "starting"
            save(self.batch, self.m)
            return r

    def port(self):
        with self.lock:
            p = int(self.m.get("port0", 14000))
            while p in self.used_ports or not claude_batch.port_free(p):
                p += 1
            self.used_ports.add(p)
            return p

    def run_one(self, r):
        port = self.port()
        model_tag = re.sub(r"[^A-Za-z0-9.-]+", "-", r["model"])[:40]
        ah = r.get("agent_harness", "ours")
        if ah != "ours":
            model_tag += "-" + ah.replace("_", "")
        tag = (f"{self.batch}-{model_tag}-{r['cond']}{r['rep'] if r['cond'] == 'noharness' else ''}"
               + ("-hidek" if r["hide_k"] else "") + ("-nominal" if r.get("nominal") else ""))
        extra = (" --hide-k" if r["hide_k"] else "") + (" --nominal" if r.get("nominal") else "")
        t0 = time.time()
        try:
            sb, log = claude_batch.start_one(r["task"], r["seed"], r["mem"], r["k"], port, tag, r["perception"],
                                             r["video"], r["style"], extra.strip(), r["feedback"])
        except Exception as e:  # server did not come up
            self.update(r, status="server_failed", error=str(e)[-500:], port=port)
            with self.lock:
                self.used_ports.discard(port)
            return
        self.update(r, status="running", port=port, sandbox=sb, log=log, run=os.path.basename(log), t_start=t0)
        opts = ""
        if r.get("api_base_url"):
            opts += f" --api-base-url {shlex.quote(r['api_base_url'])}"
        if r.get("max_budget_usd"):
            opts += f" --max-budget-usd {r['max_budget_usd']}"
        if r.get("allow_unpinned"):
            opts += " --allow-unpinned"
        # every harness (also ours: api_player.py with the same arguments as before, its output still in
        # api_player.log) goes through the adapter layer, which adds harness.json / contract.json / trajectory.jsonl
        player = (f"cd {ROOT} && source run_env.sh >/dev/null 2>&1 && "
                  f"{{ [ ! -f {ENV_FILE} ] || {{ set -a && source {ENV_FILE} && set +a; }}; }} && "
                  f"exec python scripts/harness_run.py --harness {ah} --session {log}/session.json "
                  f"--provider {r['provider']} --model {shlex.quote(r['model'])} --effort {r['effort']} "
                  f"--timeout {int(self.m.get('run_timeout_s', 10800))}{opts} > {log}/harness_run.log 2>&1")
        p = subprocess.Popen(["bash", "-c", player], start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.update(r, player_pid=p.pid)
        status = None
        try:
            code = p.wait(timeout=self.m.get("run_timeout_s", 10800))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            code, status = p.wait(), "timeout"
        self.stop_server(log)
        if status is None:
            status = "stopped" if self.stop else ("done" if code == 0 else "refused" if code == 2 else "failed")
        self.update(r, status=status, exit_code=code, t_end=time.time(), wall_s=round(time.time() - t0, 1))
        with self.lock:
            self.used_ports.discard(port)

    @staticmethod
    def stop_server(log):
        pf = os.path.join(log, "server.pid")
        if not os.path.exists(pf):
            return
        pid = int(open(pf).read())
        try:
            os.kill(pid, signal.SIGTERM)  # the server closes rollout.mp4 first
        except ProcessLookupError:
            return
        for _ in range(120):
            if not alive(pid):
                return
            time.sleep(0.5)

    def worker(self):
        while not self.stop:
            r = self.next_run()
            if r is None:
                return
            if r == "wait":
                time.sleep(5)
                continue
            try:
                self.run_one(r)
            except Exception as e:  # keep the other workers going
                self.update(r, status="failed", error=f"runner: {type(e).__name__}: {e}"[:500])

    def main(self):
        os.environ["LFF_RUNS_DIR"] = bdir(self.batch)  # claude_batch.start_one puts the run logs here
        for r in self.m["runs"]:  # a previous runner died: its half-started runs go back to pending
            if r["status"] in ACTIVE:
                r["status"] = "pending"
        save(self.batch, self.m)

        def on_term(*_):
            self.stop = True
            with self.lock:
                for r in self.m["runs"]:
                    if r["status"] == "running" and r.get("player_pid"):
                        try:
                            os.killpg(r["player_pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
        signal.signal(signal.SIGTERM, on_term)
        ths = [threading.Thread(target=self.worker, daemon=True) for _ in range(int(self.m.get("concurrency", 4)))]
        for t in ths:
            t.start()
            time.sleep(2)  # stagger server start-ups
        for t in ths:
            t.join()
        print(time.strftime("%F %T"), "runner finished", flush=True)


def cmd_runner(a):
    Runner(a.batch).main()


# ------------------------------------------------------------------------------------------------------------ views
def transcript_usage(log):
    tot = dict(calls=0, input_tokens=0, cached_tokens=0, output_tokens=0, api_errors=0, end=None)
    p = os.path.join(log or "", "transcript.jsonl")
    if not os.path.exists(p):  # native harness: totals from the unified trajectory
        sys.path.insert(0, ROOT)
        from lffbench.harnesses.trajectory import usage_summary
        u = usage_summary(log)
        if u:
            tot.update(calls=u.get("calls", 0), input_tokens=u.get("total_prompt_tokens") or 0,
                       cached_tokens=u.get("total_cached_tokens") or 0,
                       output_tokens=u.get("total_completion_tokens") or 0, end=u.get("exit_reason"),
                       cost_usd=u.get("total_cost_usd"))
        return tot
    for line in open(p):
        d = json.loads(line)
        if d["ev"] == "assistant":
            tot["calls"] += 1
            for k in ("input_tokens", "cached_tokens", "output_tokens"):
                tot[k] += (d.get("usage") or {}).get(k, 0)
        elif d["ev"] == "api_error":
            tot["api_errors"] += 1
        elif d["ev"] in ("end", "abort"):
            tot["end"] = d["ev"] if d["ev"] == "end" else "abort: " + d.get("reason", "")
    return tot


def cmd_status(a):
    m = load(a.batch)
    counts = {}
    for r in m["runs"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        att = ""
        if r.get("log") and os.path.exists(os.path.join(r["log"], "events.jsonl")):
            ev = [json.loads(line) for line in open(os.path.join(r["log"], "events.jsonl"))]
            att = "".join("S" if e["success"] else "F" for e in ev if e["ev"] == "done")
        u = transcript_usage(r.get("log"))
        print(f"{r['id']} {r['status']:13s} {r.get('agent_harness', 'ours'):16s} "
              f"{r['cond']:9s}{r['rep'] if r['cond'] == 'noharness' else '':1} "
              f"{r['task']:24s} s{r['seed']}  {att:4s} calls={u['calls']:3d} api_err={u['api_errors']} "
              f"in={u['input_tokens'] / 1e6:.2f}M out={u['output_tokens'] / 1e3:.0f}k {u['end'] or ''}")
    print("runner", "alive" if runner_alive(a.batch) else "not running", "|", counts)


def cmd_table(a):
    def done_runs(b):
        m = load(b)
        runs = [dict(r) for r in m["runs"] if r.get("log") and r["status"] == "done"]
        for r in runs:  # the harness is part of the table's "model" column (claude_batch groups by task, model)
            if r.get("agent_harness", "ours") != "ours":
                r["model"] = f"{r['model']} @ {r['agent_harness']}"
        return dict(m, runs=runs)
    claude_batch.load = done_runs  # same manifest fields (task, seed, model, cond, rep, k, log): reuse its table
    rows = [r for b in a.batch for r in done_runs(b)["runs"]]
    claude_batch.cmd_table(argparse.Namespace(batch=a.batch))
    by = {}
    for r in rows:
        u = transcript_usage(r["log"])
        t = by.setdefault((r.get("agent_harness", "ours"), r["provider"], r["model"], r["effort"]), [0, 0, 0, 0, 0])
        t[0] += 1
        t[1] += u["calls"]
        t[2] += u["input_tokens"]
        t[3] += u["cached_tokens"]
        t[4] += u["output_tokens"]
    print("\n| harness | provider | model | effort | runs done | model calls | input tokens (M) | of which cached (M) | output tokens (M) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (h, p, mo, e), t in sorted(by.items()):
        print(f"| {h} | {p} | {mo} | {e} | {t[0]} | {t[1]} | {t[2] / 1e6:.1f} | {t[3] / 1e6:.1f} | {t[4] / 1e6:.2f} |")


def cmd_contract(a):
    """contract_sha256 per (task, seed, cond) and batch; an instance whose hash differs between batches is flagged."""
    import collections
    seen = collections.defaultdict(dict)
    for b in a.batch:
        for r in load(b)["runs"]:
            p = os.path.join(r.get("log") or "", "contract.json")
            if r.get("log") and os.path.exists(p):
                sha = json.load(open(p))["contract_sha256"]
                seen[(r["task"], r["seed"], r["cond"])].setdefault(b, set()).add(sha)
    bad = 0
    print("| task | seed | cond | " + " | ".join(a.batch) + " | same |")
    print("|---|---|---|" + "---|" * len(a.batch) + "---|")
    for (task, seed, cond), per in sorted(seen.items()):
        shas = set().union(*per.values())
        same = len(shas) == 1
        bad += not same
        cells = [",".join(sorted(x[:12] for x in per.get(b, []))) or "-" for b in a.batch]
        print(f"| {task} | {seed} | {cond} | " + " | ".join(cells) + f" | {'yes' if same else '**NO**'} |")
    print(f"\n{len(seen)} instance/condition rows, {bad} with differing contracts")


def cmd_stop(a):
    pf = os.path.join(bdir(a.batch), "runner.pid")
    if os.path.exists(pf) and alive(open(pf).read().strip()):
        os.kill(int(open(pf).read().strip()), signal.SIGTERM)  # the runner stops its players and servers
        for _ in range(240):
            if not runner_alive(a.batch):
                break
            time.sleep(0.5)
    m = load(a.batch)
    for r in m["runs"]:  # anything left over (runner killed hard)
        if r["status"] in ACTIVE:
            if r.get("player_pid"):
                try:
                    os.killpg(r["player_pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if r.get("log"):
                Runner.stop_server(r["log"])
            r["status"] = "stopped"
    save(a.batch, m)
    print("stopped")


def cmd_retry(a):
    if runner_alive(a.batch):
        sys.exit("runner still running: stop it first")
    m = load(a.batch)
    n = 0
    for r in m["runs"]:
        if (a.runs and (r.get("run") in a.runs or r["id"] in a.runs)) or (not a.runs and r["status"] in RETRYABLE):
            if r.get("log"):
                m.setdefault("superseded", []).append(dict(r))
            for k in ("port", "sandbox", "log", "run", "player_pid", "exit_code", "t_start", "t_end", "wall_s", "error"):
                r.pop(k, None)
            r["status"] = "pending"
            n += 1
    save(a.batch, m)
    print(n, "runs reset to pending; runner pid", launch_runner(a.batch))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--batch", required=True)
    s.add_argument("--harness", default="ours", choices=["ours", "claude_code", "claude_code_full", "codex"],
                   help="agent harness (docs/15); native harnesses imply the provider")
    s.add_argument("--provider", default=None, choices=["openai", "anthropic", "gemini", "chat"],
                   help="required for --harness ours")
    s.add_argument("--api-base-url", default=None, help="native harnesses: model API endpoint (see harness_run.py)")
    s.add_argument("--max-budget-usd", type=float, default=None, help="claude_code*: per-run dollar cap")
    s.add_argument("--allow-unpinned", action="store_true", help="native harnesses: launch without a pin")
    s.add_argument("--model", required=True)
    s.add_argument("--effort", default="high")
    s.add_argument("--tasks", nargs="+", required=True, help="task:seed,seed ...")
    s.add_argument("--k", type=int, default=3)
    s.add_argument("--indep-reps", type=int, default=None, help="independent runs per instance (default k)")
    s.add_argument("--conds", default="noharness,harness", help="conditions in phase order")
    s.add_argument("--no-barrier", action="store_true", help="do not wait for one phase to finish before the next")
    s.add_argument("--no-hide-k", action="store_true", help="history runs: README / status show all k attempts")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--port0", type=int, default=14000)
    s.add_argument("--run-timeout", type=int, default=10800, help="seconds per run before the player is killed")
    s.add_argument("--perception", default="rgb", choices=["rgb", "rgbd"])
    s.add_argument("--video", action="store_true")
    s.add_argument("--feedback", default="F1", choices=["F0", "F1", "F2"])
    s.add_argument("--nominal", action="store_true", help="L1 control: hidden calibration error removed")
    r = sub.add_parser("_runner")
    r.add_argument("--batch", required=True)
    for name in ("status", "stop"):
        sub.add_parser(name).add_argument("--batch", required=True)
    t = sub.add_parser("table")
    t.add_argument("--batch", nargs="+", required=True)
    ct = sub.add_parser("contract")
    ct.add_argument("--batch", nargs="+", required=True)
    rt = sub.add_parser("retry")
    rt.add_argument("--batch", required=True)
    rt.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args()
    dict(start=cmd_start, _runner=cmd_runner, status=cmd_status, table=cmd_table, stop=cmd_stop, retry=cmd_retry,
         contract=cmd_contract)[a.cmd](a)


if __name__ == "__main__":
    main()
