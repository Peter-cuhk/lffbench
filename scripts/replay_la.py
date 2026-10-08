"""Replay the robot commands of a logged agent attempt on a fresh task instance (current task code) and print the
outcome next to the logged one -- e.g. to check a rule change against old runs without re-running the agent.

usage: python scripts/replay_la.py <run_dir> <attempt> [--seg] [--fb]
LA `move_eef` calls mirror robot_server._handle_v2 (missing axes default to the current pose, move_to then gripper);
v1 tool calls (move_to / open_gripper / close_gripper / push, world frame) are passed to the ToolExecutor as logged.
--seg prints the per-command board / cube motion (tasks whose Skills keep `segments`), --fb the F2 text.
"""
import importlib
import json
import os
import pkgutil
import sys

import numpy as np

from lffbench.agent.tools import WS_HI, WS_LO, ToolExecutor
from lffbench import tasks as task_pkg
from lffbench.task_base import TASKS

for _m in pkgutil.iter_modules(task_pkg.__path__):
    importlib.import_module(f"lffbench.tasks.{_m.name}")


def main():
    run_dir, attempt = sys.argv[1], int(sys.argv[2])
    ev = [json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl")) if l.strip()]
    start = next(e for e in ev if e["ev"] == "session_start")
    task = TASKS[start["task"]](cam_res=128)
    inst = task.sample_instance(start["seed"])
    task.reset_instance(inst)
    sim = task.env.sim
    off = np.array(sim.data.body_xpos[sim.model.body_name2id("robot0_base")], float).copy()
    ex = ToolExecutor(task, inst, image_res=128, cams=("agentview",))
    logged = None
    for e in ev:
        if e.get("attempt") != attempt:
            continue
        if e["ev"] == "done":
            logged = e
            break
        if e["ev"] == "call" and e["cmd"] in ("move_to", "open_gripper", "close_gripper", "push"):  # v1 tools
            ex.call(e["cmd"], e["args"])
            print(f"  s{e['step']:>2} {e['cmd']} {e['args']}  [seg {len(getattr(task.sk, 'segments', []))}]")
            continue
        if e["ev"] != "call" or e["cmd"] != "move_eef":
            continue
        args = e["args"]
        p = ex.proprio()
        cur = np.asarray(p["eef_pos"], float) - off
        tgt = [float(args.get(k, cur[i])) for i, k in enumerate("xyz")]
        yaw = float(args.get("yaw", p["eef_yaw_deg"]))
        speed = None if args.get("speed") in (None, "") else float(args["speed"])
        moving = any(k in args for k in ("x", "y", "z", "yaw"))
        tgt_w = np.asarray(tgt) + off
        if moving and not (np.all(tgt_w >= WS_LO) and np.all(tgt_w <= WS_HI)):
            print(f"  s{e['step']:>2} rejected")
            continue
        if moving:
            mv = dict(x=float(tgt_w[0]), y=float(tgt_w[1]), z=float(tgt_w[2]), yaw=yaw)
            if speed is not None:
                mv["speed"] = speed
            ex.call("move_to", mv)
        if args.get("gripper"):
            ex.call("open_gripper" if args["gripper"] == "open" else "close_gripper", {})
        print(f"  s{e['step']:>2} {' '.join(f'{k}={v}' for k, v in args.items())}  [seg {len(getattr(task.sk, 'segments', []))}]")
    if "--seg" in sys.argv and hasattr(task.sk, "segments"):
        for g in task.sk.segments:
            if g["path"] > 0.0005 or g["touch"]:
                print(f"     seg@{g['step0']:>4} {g['kind']:<5} steps={g['steps']:>3} path={100*g['path']:.2f}cm "
                      f"driven={100*g['driven']:.2f}cm touch={g['touch']:>3} board_dxy={g['board_dxy']} "
                      f"cube_dxy={g['cube_dxy']} cube_from_start={g['cube_from_start']}")
    out = task.outcome(inst)
    keys = ("moved_toward_robot", "n_pulls", "pulls_cm", "pre_pull_cube_cm", "stray_cm", "n_pushes", "max_advance")
    print("REPLAY :", "S" if out["success"] else "F", out.get("failure"), {k: out[k] for k in keys if k in out})
    if "--fb" in sys.argv:
        print("F2     :", task.feedback(inst, out, "F2"))
    if logged:
        print("LOGGED :", "S" if logged["success"] else "F", logged["outcome"].get("failure"))


if __name__ == "__main__":
    main()
