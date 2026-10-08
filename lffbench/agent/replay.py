"""Rebuild AttemptRecords from a finished run directory (events.jsonl + images/), e.g. to use a full-memory
run as the donor pool of a mismatched-memory run without re-running (and re-paying for) it."""
import json
import os

from .context import AttemptRecord, StepRecord
from .images import ImageRef


def load_events(run_dir):
    with open(os.path.join(run_dir, "events.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(run_dir, task=None, feedback="F2"):
    """{seed: [AttemptRecord]} with keyframes loaded from disk. Feedback text is re-rendered at the
    requested level with task.feedback() when a task is given (else the logged F2 text / '')."""
    ev = load_events(run_dir)
    insts, steps, out = {}, {}, {}
    for e in ev:
        t = e["type"]
        if t == "run_start":
            insts[e["seed"]] = e["inst"]
            out[e["seed"]] = []
        elif t == "tool_call":
            steps.setdefault((e["seed"], e["attempt"]), []).append(
                StepRecord(step=e["step"], name=e["name"], args=e["args"], report=e["report"], ok=e["ok"]))
        elif t == "attempt_end":
            seed, a = e["seed"], e["attempt"]
            kfs = []
            for cap, rel in e.get("keyframes") or []:
                p = os.path.join(run_dir, rel)
                with open(p, "rb") as f:
                    kfs.append((cap, ImageRef(camera="agentview", path=p, jpeg=f.read())))
            o = e["outcome"]
            if task is not None:
                fb = task.feedback(insts.get(seed, {}), o, feedback)
            else:
                fb = {"F0": "", "F1": "Attempt succeeded." if e["success"] else "Attempt failed."}.get(
                    feedback, e.get("feedback_f2", ""))
            out.setdefault(seed, []).append(AttemptRecord(
                seed=seed, attempt=a, steps=steps.get((seed, a), []), keyframes=kfs, outcome=o,
                success=bool(e["success"]), feedback=fb, done_reason=e.get("done_reason"),
                diagnosis=e.get("diagnosis"), ended_by=e.get("ended_by", ""), backend_meta=e.get("backend_meta") or {}))
    return out


def derange(target_seeds, donor_seeds):
    """Map each target seed to a donor seed != itself (cyclic shift over the sorted donor pool)."""
    pool = sorted(donor_seeds)
    if not pool:
        raise ValueError("empty donor pool")
    m = {}
    for i, s in enumerate(sorted(target_seeds)):
        j = (pool.index(s) + 1) % len(pool) if s in pool else i % len(pool)
        if pool[j] == s:
            raise ValueError(f"cannot find a donor different from seed {s} (pool {pool})")
        m[s] = pool[j]
    return m
