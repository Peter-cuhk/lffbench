"""Information contract of one run (docs/15 §1 layer 2) and its hash `contract_sha256`.

The contract is everything the agent is given that must be identical across harnesses and models for the same
instance and condition: the README text, the three tool schemas, the image spec, the feedback level, --hide-k, the
budgets, the memory condition, the task and seed, and the code that builds the scene and judges it. Two runs whose
`contract_sha256` differ must not share a table cell.

What is NOT in the contract on purpose: the harness (it is the thing being compared), the model, the port, the
sandbox / log paths, timestamps.

  contract = build_contract(log_dir)          # reads session.json, agent_iface.json, events.jsonl, README.md
  contract_sha256(contract)                   # sha256 of the canonical JSON
  write_contract(log_dir)                     # -> <log_dir>/contract.json, returns (contract, sha)
"""
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the lffbench repo
CONTRACT_VERSION = 1
IMAGE_SPEC = dict(cameras=["agentview", "wrist"], width=512, height=512, format="JPEG", jpeg_quality=90,
                  delivery="files in the workspace, viewed with view_images (bytes unchanged)")
# code that builds the scene, runs the robot and judges the result (task modules are added per task)
SERVER_CODE = ["lffbench/agent/robot_server.py", "lffbench/agent/tools.py", "lffbench/agent/camera.py",
               "lffbench/agent/context.py", "lffbench/agent/images.py", "lffbench/envs.py", "lffbench/skills.py",
               "lffbench/task_base.py", "lffbench/bddl.py", "lffbench/objects.py"]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_json(obj):
    return sha256_text(canonical(obj))


def tool_schemas():
    """The three LA tools exactly as the reference driver and the MCP server define them."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from api_player import SHELL_TOOL, TOOL_LA, VIEW_TOOL_LA, PREAMBLE_LA  # noqa: E402
    return [TOOL_LA, SHELL_TOOL, VIEW_TOOL_LA], PREAMBLE_LA


def task_module_files(task):
    """Source files of the task class `task` (found by its `name = "<task>"` line; .bak files are ignored)."""
    pat = re.compile(r"""^\s*name\s*=\s*["']%s["']""" % re.escape(task), re.M)
    hits = []
    for p in sorted(glob.glob(os.path.join(ROOT, "lffbench", "tasks", "*.py"))):
        try:
            if pat.search(open(p, encoding="utf-8").read()):
                hits.append(os.path.relpath(p, ROOT))
        except OSError:
            pass
    arm = os.path.join(ROOT, "lffbench", "arm_specs", f"{task}.py")
    if os.path.exists(arm):
        hits.append(os.path.relpath(arm, ROOT))
    return hits


def server_code_sha(task):
    files = SERVER_CODE + task_module_files(task)
    per = {f: (sha256_file(os.path.join(ROOT, f)) if os.path.exists(os.path.join(ROOT, f)) else None) for f in files}
    return sha256_json(per), per


def session_start(log_dir):
    with open(os.path.join(log_dir, "events.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("ev") == "session_start":
                return d
    raise RuntimeError(f"no session_start event in {log_dir}/events.jsonl")


def readme_budget(readme):
    """The per-attempt budgets as the README states them (what the agent is told)."""
    out = {}
    for key, pat in (("robot_commands", r"at most (\d+) robot commands"),
                     ("sim_seconds", r"at most ([\d.]+) s of simulated"),
                     ("tool_calls", r"at most (\d+) tool calls")):
        m = re.search(pat, readme)
        out[key] = float(m.group(1)) if m and "." in m.group(1) else (int(m.group(1)) if m else None)
    return out


def build_contract(log_dir):
    log_dir = os.path.abspath(log_dir)
    sess = json.load(open(os.path.join(log_dir, "session.json")))
    iface = json.load(open(os.path.join(log_dir, "agent_iface.json")))
    st = session_start(log_dir)
    readme = open(os.path.join(sess["sandbox"], "README.md"), encoding="utf-8").read()
    tools, preamble = tool_schemas()
    code_sha, code_files = server_code_sha(st["task"])
    return dict(
        contract_version=CONTRACT_VERSION,
        task=st["task"], seed=st["seed"],
        readme_sha256=sha256_text(readme),
        preamble_sha256=sha256_text(preamble),
        tools=tools,
        image_spec=dict(IMAGE_SPEC, perception=st.get("perception", iface.get("perception"))),
        feedback=st.get("feedback"), instruction=st.get("instruction"),
        memory=st.get("memory"), k=st.get("k"), k_shown=st.get("k_shown"), hide_k=st.get("k_shown") != st.get("k"),
        protocol=st.get("protocol"), arm=st.get("arm"), dossier_sha256=sha256_text(st.get("dossier") or ""),
        budget=dict(readme_budget(readme), max_tool_calls=iface.get("max_tool_calls"),
                    tool_call_budget_enforced_by="mcp_robot_server / api_player (per attempt)"),
        agent_iface=iface.get("agent_iface"),
        server_code_sha256=code_sha, server_code_files=code_files,
    )


def contract_sha256(contract):
    return sha256_json(contract)


def write_contract(log_dir):
    c = build_contract(log_dir)
    sha = contract_sha256(c)
    with open(os.path.join(log_dir, "contract.json"), "w") as f:
        json.dump(dict(contract_sha256=sha, contract=c), f, indent=1, ensure_ascii=False)
    return c, sha


if __name__ == "__main__":
    for d in sys.argv[1:]:
        c, sha = write_contract(d)
        print(sha, d)
