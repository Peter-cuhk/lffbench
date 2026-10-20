"""api_batch.py --memfree followup (2026-10-19): memory-free retries are planned only after the history run's attempt 1
failed, one fresh agent at a time, until a success or k attempts in all; attempt 1 of the history run is the chain's
first attempt in the table. No simulator, no model: manifests and server logs (events.jsonl) are written by hand."""
import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import api_batch  # noqa: E402
import claude_batch  # noqa: E402


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api_batch, "RUNS", str(tmp_path))
    monkeypatch.setattr(api_batch, "launch_runner", lambda batch: 0)
    monkeypatch.setattr(api_batch, "runner_alive", lambda batch: False)
    return tmp_path


def write_log(tmp, name, outcomes, protocol="cross"):
    """A run log whose server log has one `done` per attempt (outcomes: 'FFS' etc.)."""
    d = tmp / "logs" / name
    d.mkdir(parents=True)
    ev = [dict(ev="session_start", protocol=protocol, t=0.0)]
    for i, o in enumerate(outcomes, 1):
        ev.append(dict(ev="http", attempt=i, cmd="move", t=i, t_recv=i, t_send=i + 0.1, server_s=0.1, think_s=1.0))
        ev.append(dict(ev="call", attempt=i, cmd="move", step=1, t=i))
        ev.append(dict(ev="done", attempt=i, step=1, success=(o == "S"), t=i + 0.5))
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    return str(d)


def start(batch, tasks, **kw):
    ns = dict(batch=batch, harness="ours", provider="chat", api_base_url=None, max_budget_usd=None,
              allow_unpinned=False, jail=False, model="m", effort="high", tasks=tasks, k=3, indep_reps=None,
              conds=None, no_barrier=False, no_hide_k=False, concurrency=1, port0=14000, run_timeout=60,
              perception="rgb", video=False, feedback="F1", nominal=False, memfree="followup", seeds_per_task=None,
              codex_auth="chatgpt", codex_catalog="generic", codex_provider="openai", claude_auth=None)  # the CLI defaults
    ns.update(kw)
    with redirect_stdout(io.StringIO()):
        api_batch.cmd_start(argparse.Namespace(**ns))
    return api_batch.load(batch)


def finish(m, r, log):
    r.update(status="done", log=log)
    return api_batch.plan_followup(m, r)


def test_start_plans_history_runs_only(runs_dir):
    m = start("b", ["t1:3000,3001,3002", "t2:3005,3006"], seeds_per_task=1)
    assert [(r["task"], r["seed"], r["cond"], r["followup"], r["hide_k"]) for r in m["runs"]] == \
        [("t1", 3000, "harness", True, True), ("t2", 3005, "harness", True, True)]
    up = start("u", ["t1:3000"], memfree="upfront")
    assert [r["cond"] for r in up["runs"]] == ["noharness"] * 3 + ["harness"]
    assert not any(r.get("followup") for r in up["runs"])
    with pytest.raises(SystemExit):
        start("x", ["t1:3000"], no_hide_k=True)
    with pytest.raises(SystemExit):
        start("y", ["t1:3000"], conds="noharness,harness")


def test_chain(runs_dir):
    m = start("b", ["t1:3000", "t2:3000", "t3:3000"])
    h1, h2, h3 = m["runs"]
    assert finish(m, h1, write_log(runs_dir, "h1", "S")) is None          # attempt 1 solved: nothing more
    r1 = finish(m, h2, write_log(runs_dir, "h2", "FFS"))                  # history run solved later: chain starts
    assert (r1["cond"], r1["rep"], r1["k"], r1["mem"], r1["hide_k"], r1["chain"]) == \
        ("noharness", 1, 1, "none", False, h2["id"])
    assert r1["task"] == "t2" and r1["seed"] == 3000 and r1["status"] == "pending"
    assert api_batch.plan_followup(m, h2) is None                         # idempotent
    r2 = finish(m, r1, write_log(runs_dir, "r1", "F"))
    assert (r2["rep"], r2["chain"]) == (2, h2["id"])
    assert finish(m, r2, write_log(runs_dir, "r2", "F")) is None          # k = 3 attempts in all
    r3 = finish(m, h3, write_log(runs_dir, "h3", "FFF"))
    assert finish(m, r3, write_log(runs_dir, "r3", "S")) is None          # first success ends the chain
    failed = dict(h3, status="failed")
    assert api_batch.plan_followup(m, failed) is None
    assert len({r["id"] for r in m["runs"]}) == len(m["runs"])


def test_no_chain_when_the_scene_persists(runs_dir):
    # protocol "within" (scene not reset between attempts): a fresh agent would start from the initial scene while the
    # history run's attempt 2 starts from the failed one -> no memory-free retries for these tasks (user 2026-10-20)
    m = start("b", ["t1:3000"])
    assert finish(m, m["runs"][0], write_log(runs_dir, "w1", "FFF", protocol="within")) is None
    assert [r["cond"] for r in m["runs"]] == ["harness"]


def test_table_rows(runs_dir):
    m = start("b", ["t1:3000", "t2:3000", "t3:3000", "t4:3000"])
    h1, h2, h3, h4 = m["runs"]
    finish(m, h1, write_log(runs_dir, "h1", "S"))
    r = finish(m, h2, write_log(runs_dir, "h2", "FS"))
    r = finish(m, r, write_log(runs_dir, "h2r1", "F"))
    finish(m, r, write_log(runs_dir, "h2r2", "F"))
    r = finish(m, h3, write_log(runs_dir, "h3", "FFF"))
    finish(m, r, write_log(runs_dir, "h3r1", "S"))
    finish(m, h4, write_log(runs_dir, "h4", "FFF"))                       # chain not finished yet
    api_batch.save("b", m)
    assert [x["rep"] for x in api_batch.memfree_chain(m, h1)] == [0]
    assert [x["rep"] for x in api_batch.memfree_chain(m, h2)] == [0, 1, 2]
    assert [x["rep"] for x in api_batch.memfree_chain(m, h3)] == [0, 1]
    assert api_batch.memfree_chain(m, h4) == []
    st = claude_batch.run_stats(dict(api_batch.memfree_chain(m, h2)[0]))
    assert (st["attempts"], st["first_success"], st["n_calls"]) == (1, None, 1)
    out = io.StringIO()
    with redirect_stdout(out):
        api_batch.cmd_table(argparse.Namespace(batch=["b"]))
    row = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[1:-1]] for line in out.getvalue().splitlines()
           if line.startswith("| t")}
    # columns: task, model, n, harness @1, harness @k, no-harness @1, no-harness @k
    assert row["t1"][3:7] == ["1/1", "1/1", "1/1", "1/1"]
    assert row["t2"][3:7] == ["0/1", "1/1", "0/1", "0/1"]
    assert row["t3"][3:7] == ["0/1", "0/1", "0/1", "1/1"]
    assert row["t4"][2] == "1/0"                                           # history run counted, chain not yet


def test_retry_drops_stale_chain(runs_dir):
    m = start("b", ["t1:3000"])
    h = m["runs"][0]
    r1 = finish(m, h, write_log(runs_dir, "h", "FFF"))
    r2 = finish(m, r1, write_log(runs_dir, "r1", "F"))
    finish(m, r2, write_log(runs_dir, "r2", "F"))
    api_batch.save("b", m)
    with redirect_stdout(io.StringIO()):
        api_batch.cmd_retry(argparse.Namespace(batch="b", runs=[r1["id"]]))
    m = api_batch.load("b")
    assert [(r["id"], r["status"]) for r in m["runs"]] == [(h["id"], "done"), (r1["id"], "pending")]
    assert any(x.get("superseded_because") for x in m["superseded"])
    with redirect_stdout(io.StringIO()):
        api_batch.cmd_retry(argparse.Namespace(batch="b", runs=[h["id"]]))
    m = api_batch.load("b")
    assert [(r["id"], r["status"]) for r in m["runs"]] == [(h["id"], "pending")]


def test_followup_command_and_runner(runs_dir):
    m = start("b", ["t1:3000", "t2:3000"], memfree="none")
    h1, h2 = m["runs"]
    assert not h1["followup"]
    h1.update(status="done", log=write_log(runs_dir, "h1", "FFS"))
    h2.update(status="running")
    api_batch.save("b", m)
    with redirect_stdout(io.StringIO()):
        api_batch.cmd_followup(argparse.Namespace(batch="b"))
    m = api_batch.load("b")
    assert [(r["cond"], r.get("chain")) for r in m["runs"]] == [("harness", None), ("harness", None), ("noharness", h1["id"])]
    m["runs"][2]["status"] = "done"
    m["runs"][2]["log"] = write_log(runs_dir, "r1", "F")
    api_batch.save("b", m)
    runner = api_batch.Runner("b")
    assert runner.next_run() == "wait"   # nothing pending, but the running history run may add a retry
