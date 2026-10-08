"""Protocol logic (memory modes, feedback levels, cross vs within, early stop, metrics) with a fake 1-D task and
a fake executor: no simulator, runs in seconds."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from lffbench.agent.backends.base import Backend, BackendResponse
from lffbench.agent.context import (AttemptRecord, PromptBuilder, StepRecord, ToolCall, pick_keyframes,
                                    prune_live_images, select_history)
from lffbench.agent.harness import AgentHarness, HarnessConfig
from lffbench.agent.images import ImageRef
from lffbench.agent.metrics import aggregate, paired_delta
from lffbench.agent.runlog import RunLogger
from lffbench.agent.tools import ToolResult
from lffbench.task_base import LFFTask


class FakeTask(LFFTask):
    """Slide a puck with one 'push(speed)': it travels gain * speed. Hidden variable: gain (per seed)."""
    name = "fake_slide"
    max_attempts = 4
    instruction = "Push the puck so that it stops at 0.10 m."

    def __init__(self, protocol="cross"):  # no LIBERO env
        self.protocol = protocol
        self.env = SimpleNamespace(objects_dict={"puck_1": None})
        self.resets = 0
        self.pos = 0.0

    def sample_instance(self, seed):
        return dict(seed=seed, gain=0.2 + 0.05 * (seed % 5), _secret=1)

    def reset_instance(self, inst, recorder=None):
        self.resets += 1
        self.pos = 0.0

    def outcome(self, inst):
        err = self.pos - 0.10
        ok = abs(err) <= 0.01
        return dict(success=ok, along_err=err, overshoot=err > 0.05,
                    detail=f"The puck stopped {100 * abs(err):.1f} cm {'past' if err > 0 else 'short of'} the target.")


class FakeExecutor:
    def __init__(self, harness, inst):
        self.task, self.inst = harness.task, inst
        self.sink = harness._sink
        self.sk = SimpleNamespace(hold=lambda n: None)

    def render(self, cam, res=None):
        return np.zeros((res or 32, res or 32, 3), np.uint8)

    def observe(self, label=""):
        return [self.sink(ImageRef(camera=c, array=self.render(c, 32), label=label)) for c in ("agentview", "wrist")]

    def proprio(self):
        return dict(puck=round(self.task.pos, 4))

    def call(self, name, args):
        args = json.loads(args) if isinstance(args, str) else dict(args)
        if name == "push":
            self.task.pos += self.inst["gain"] * args["speed"]
            return ToolResult(name, args, True, dict(ok=True, tool="push", touched_object=True))
        if name == "done":
            return ToolResult(name, args, True, dict(ok=True, ended=True), ended=True)
        return ToolResult(name, args, False, dict(ok=False, error="unknown tool"))


class ParsingBackend(Backend):
    """Model stand-in that reads the F2 feedback text from the conversation (like an LLM would)."""
    name = "fake"

    def __init__(self):
        self.seen_texts = []
        self.speed = None

    def begin_run(self, info):
        self.speed = 0.5

    def begin_attempt(self, ainfo):
        self.calls = 0
        self.history = ainfo.history

    def act(self, conv, tools):
        self.calls += 1
        texts = [p["text"] for it in conv.items if it["kind"] == "user" for p in it["parts"] if p["type"] == "text"]
        self.seen_texts.append(texts)
        if self.calls == 1:
            # proportional correction from the last feedback measurement (if any)
            fb = [t for t in texts if t.startswith("Outcome feedback")]
            if fb:
                last = fb[-1]
                cm = float(last.split("stopped ")[1].split(" cm")[0]) / 100
                past = " past " in last
                travelled = 0.10 + (cm if past else -cm)
                self.speed = self.speed * 0.10 / max(travelled, 1e-3)
            return BackendResponse(tool_calls=[ToolCall("c1", "push", dict(speed=self.speed))], text="plan",
                                   usage=dict(input_tokens=100, cached_tokens=50, output_tokens=10, reasoning_tokens=5),
                                   latency_s=0.01)
        return BackendResponse(tool_calls=[ToolCall("c2", "done", dict(reason="pushed"))], latency_s=0.01)


def run(tmp_path, memory="full", feedback="F2", protocol="cross", seeds=(1, 2, 3), donors=None, k=None,
        early_stop="auto"):
    task = FakeTask(protocol)
    be = ParsingBackend()
    log = RunLogger(str(tmp_path / f"{memory}_{feedback}_{protocol}"))
    cfg = HarnessConfig(memory=memory, feedback=feedback, k=k, image_res=32, early_stop=early_stop, model_for_cost="gpt-6-astra")
    h = AgentHarness(task, be, cfg, log, executor_factory=FakeExecutor, cam_axes={})
    out = [h.run_instance(s, donor=(donors or {}).get(s)) for s in seeds]
    log.close()
    return task, be, out, log.run_dir


def events(run_dir):
    with open(os.path.join(run_dir, "events.jsonl")) as f:
        return [json.loads(l) for l in f]


def test_full_memory_learns(tmp_path):
    task, be, out, d = run(tmp_path, "full")
    for summ, recs in out:
        assert summ["status"] == "ok" and summ["first_success"] in (1, 2, 3)
        assert task.resets >= len(recs)  # cross: reset before every attempt
    agg = aggregate([s for s, _ in out], 4)
    assert agg["succ_at"][-1] == 1.0
    ev = events(d)
    assert {e["type"] for e in ev} >= {"run_start", "llm_call", "tool_call", "observation", "attempt_end", "run_end"}
    rs = next(e for e in ev if e["type"] == "run_start")
    assert "_secret" not in rs["inst"]  # private keys are not logged
    call = next(e for e in ev if e["type"] == "llm_call")
    assert call["usage"]["input_tokens"] == 100 and call["cost_usd"] > 0 and call["latency_s"] >= 0


def test_none_memory_is_independent(tmp_path):
    _, be, out, d = run(tmp_path, "none")
    for texts in be.seen_texts:
        assert not any("earlier attempts" in t or t.startswith("Outcome feedback") or "attempt" in t.lower()
                       for t in texts)
    # identical prompts for every attempt of a run (pure i.i.d. retries)
    ev = events(d)
    ends = [e for e in ev if e["type"] == "attempt_end"]
    assert all(e["history_shown"] == [] for e in ends)


def test_last_and_feedback_levels(tmp_path):
    _, be, out, d = run(tmp_path, "last", "F1", seeds=(4,))
    for e in events(d):
        if e["type"] == "attempt_end" and e["attempt"] > 1:
            assert len(e["history_shown"]) == 1 and e["history_shown"][0][1] == e["attempt"] - 1
            assert e["feedback_shown"] in ("Attempt failed.", "Attempt succeeded.")
    flat = [t for texts in be.seen_texts for t in texts]
    assert not any("cm" in t for t in flat if t.startswith("Outcome feedback"))


def test_f0_no_feedback_and_no_early_stop(tmp_path):
    _, be, out, d = run(tmp_path, "full", "F0", seeds=(1,), k=3)
    summ, recs = out[0]
    assert summ["n_attempts"] == 3  # F0: no early stop (would leak success)
    flat = [t for texts in be.seen_texts for t in texts]
    assert not any(t.startswith("Outcome feedback") for t in flat)


def test_mismatched_uses_donor(tmp_path):
    _, _, donor_out, _ = run(tmp_path, "full", seeds=(7,))
    donor = donor_out[0][1]
    _, be, out, d = run(tmp_path, "mismatched", seeds=(3,), donors={3: donor})
    for e in events(d):
        if e["type"] == "attempt_end" and e["attempt"] > 1:
            assert all(s == 7 for s, _ in e["history_shown"])
            assert len(e["history_shown"]) == min(e["attempt"] - 1, len(donor))
    with pytest.raises(ValueError):
        select_history("mismatched", [1], None)


def test_within_protocol(tmp_path):
    task, be, out, d = run(tmp_path, "full", protocol="within", seeds=(2,))
    summ, recs = out[0]
    assert task.resets == 1  # one continuous episode
    flat = [t for texts in be.seen_texts[1:] for t in texts]
    if len(recs) > 1:
        assert any("Continue from the current state" in t for t in flat)


def test_select_and_prune_and_keyframes():
    own = [AttemptRecord(seed=1, attempt=i) for i in (1, 2, 3)]
    assert select_history("none", own) == []
    assert select_history("full", own) == own
    assert select_history("last", own) == own[-1:]
    donor = [AttemptRecord(seed=9, attempt=i) for i in (1, 2)]
    assert select_history("mismatched", own, donor) == donor
    steps = [StepRecord(step=i, name="move_to" if i % 3 else "push", args={}, report={}, ok=True,
                        frame=ImageRef("agentview")) for i in range(1, 9)]
    kf = pick_keyframes(steps, ImageRef("agentview"), ImageRef("agentview"), 4)
    assert len(kf) == 4 and kf[0][0] == "start of attempt" and kf[-1][0].startswith("end of attempt")
    assert all("push" in c for c, _ in kf[1:3])  # informative frames preferred


def test_history_prefix_is_append_only():
    """The history item of attempt j+1 starts with the history item of attempt j (minus its end marker), so
    the request prefix is cache-friendly across attempts."""
    P = PromptBuilder(FakeTask(), "cross", 4, "full", "F2", 32, {})

    def rec(i):
        return AttemptRecord(seed=1, attempt=i, steps=[StepRecord(1, "push", dict(speed=0.1 * i), dict(ok=True), True)],
                             keyframes=[("end", ImageRef("agentview"))], feedback=f"Attempt failed. err {i}")

    def texts(parts):
        return [p.get("text", "<img>") for p in parts]
    h1 = texts(P.history_parts("T", [rec(1)]))
    h2 = texts(P.history_parts("T", [rec(1), rec(2)]))
    assert h2[:len(h1) - 1] == h1[:-1]


def test_paired_delta():
    w = [dict(seed=s, status="ok", first_success=f) for s, f in zip(range(10), [1, 2, None, 3, 1, 2, 2, None, 1, 4])]
    i = [dict(seed=s, status="ok", first_success=f) for s, f in zip(range(10), [1, None, None, None, 1, None, 2, None, None, None])]
    r = paired_delta(w, i, k=5, n_boot=2000)
    assert r["n"] == 10 and abs(r["delta_icl"] - 0.5) < 1e-9 and r["ci95"][0] > 0
    assert r["discordant"] == dict(with_only=5, iid_only=0)
