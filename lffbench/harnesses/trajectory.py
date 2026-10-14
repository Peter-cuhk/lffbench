"""Unified trajectory (minimal, field names follow Harbor's ATIF where there is one).

<log_dir>/trajectory.jsonl, one JSON object per line:
  line 1   {"kind": "header", "schema": "lff-traj-0", "atif_like": "ATIF v1.x field names", "session_id",
            "agent": {"name", "version", "model_name", "effort"}, "tools": [...], "source": "<native log file>"}
  steps    {"kind": "step", "step_id", "timestamp", "source": "user"|"agent"|"system", "message",
            "reasoning_content", "tool_calls": [{"tool_call_id", "function_name", "arguments"}],
            "observation": {"results": [{"source_call_id", "content", "is_error", "images"}]},
            "metrics": {"prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd"}}
  last     {"kind": "final", "final_metrics": {"total_prompt_tokens", "total_completion_tokens",
            "total_cached_tokens", "total_cost_usd", "total_steps"}, "exit_reason", "native_result"}
Not full ATIF: no per-step model name, no subagent trajectories, images only counted (bytes stay in the workspace).
Tool names are reported without the MCP prefix (robot / shell / view_images), so all harnesses line up.
"""
import json
import os

SCHEMA = "lff-traj-0"


def strip_mcp(name):
    for p in ("mcp__lff__", "lff."):
        if name and name.startswith(p):
            return name[len(p):]
    return name


class TrajWriter:
    def __init__(self, path):
        self.path = path
        self.steps = []
        self.header = None
        self.final = None

    def set_header(self, **kw):
        self.header = dict(kind="header", schema=SCHEMA, atif_like="field names follow Harbor ATIF", **kw)

    def add_step(self, source, message="", reasoning="", tool_calls=None, timestamp=None, metrics=None, **extra):
        st = dict(kind="step", step_id=len(self.steps) + 1, timestamp=timestamp, source=source, message=message or "",
                  reasoning_content=reasoning or "", tool_calls=tool_calls or [],
                  observation=dict(results=[]), metrics=metrics or {})
        st.update(extra)
        self.steps.append(st)
        return st

    def add_result(self, call_id, content, is_error=False, images=0):
        for st in reversed(self.steps):
            if any(tc["tool_call_id"] == call_id for tc in st["tool_calls"]):
                st["observation"]["results"].append(dict(source_call_id=call_id, content=content, is_error=is_error,
                                                         images=images))
                return True
        return False

    def set_final(self, exit_reason, native_result=None, **metrics):
        fm = dict(total_prompt_tokens=0, total_completion_tokens=0, total_cached_tokens=0, total_cost_usd=None,
                  total_steps=len(self.steps))
        fm.update({k: v for k, v in metrics.items() if v is not None})
        self.final = dict(kind="final", final_metrics=fm, exit_reason=exit_reason, native_result=native_result)

    def write(self):
        with open(self.path, "w") as f:
            for d in [self.header] + self.steps + [self.final]:
                if d is not None:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return self.path

    def tool_call_names(self):
        return [tc["function_name"] for st in self.steps for tc in st["tool_calls"]]


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def usage_summary(log_dir):
    """Token / cost totals from trajectory.jsonl (None if missing)."""
    p = os.path.join(log_dir or "", "trajectory.jsonl")
    if not os.path.exists(p):
        return None
    rows = read(p)
    fin = next((r for r in rows if r.get("kind") == "final"), None)
    steps = [r for r in rows if r.get("kind") == "step" and r.get("source") == "agent"]
    out = dict(calls=len(steps), exit_reason=(fin or {}).get("exit_reason"))
    out.update((fin or {}).get("final_metrics") or {})
    return out
