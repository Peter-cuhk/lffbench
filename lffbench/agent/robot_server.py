"""Robot server for driving an LFF-Bench task from an isolated agent (e.g. a Claude Code subagent).

The server owns the simulator, the task instance and its hidden variables. The agent only gets a
sandbox directory containing a tiny `robot` CLI (talks to this server over localhost HTTP), a README
with the instructions, and the observation images the server writes into `<sandbox>/obs/`.
Nothing about the benchmark (code paths, hidden variables, success rule, scripted policies) is ever
written into the sandbox.

Attempt protocol (same semantics as harness.py):
  cross   `done` -> outcome is measured; if failed and attempts remain, the scene is reset to the same
          initial state + hidden variables and the next attempt starts.
  within  `done` -> outcome is measured; if failed and done-calls remain, continue from the current state.
memory=none: every attempt is meant for a fresh agent; after `done` the server tells the agent to stop
(no feedback) and waits for the orchestrator to start a new agent (`./robot status` shows the state).

Run:  python -m lffbench.agent.robot_server --task l5_slide_to_target --seed 1001 --sandbox <dir> --port 8765
"""
import argparse
import importlib
import json
import os
import pkgutil
import shutil
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import signal
import textwrap

import numpy as np

from .. import tasks as task_pkg
from ..task_base import TASKS
from . import camera as C
from .context import PromptBuilder
from .tools import ACTION_TOOLS, WS_HI, WS_LO, ToolExecutor, _json_default

for _m in pkgutil.iter_modules(task_pkg.__path__):
    importlib.import_module(f"lffbench.tasks.{_m.name}")

LFF_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHELL_TEMPLATE = r'''#!/bin/bash
# Run a shell command inside this workspace, in the episode jail (no network, host data hidden):
#   ./shell '<command>'        or        ./shell <<'EOF' ... EOF   (command read from stdin)
WS="$(cd "$(dirname "$0")" && pwd)"
if [ $# -gt 0 ]; then CMD="$*"; else CMD="$(cat)"; fi
exec unshare --user --map-root-user --mount --net -- /opt/lffjail/jail_inner.sh "$WS" timeout -k 5 60 bash -c "$CMD"
'''

CLI_TEMPLATE = r'''#!/usr/bin/env python3
"""Robot command-line interface. Usage:  ./robot <command> [key=value ...]   (./robot help)"""
import json, sys, urllib.request
PORT = __PORT__
TOKEN = "__TOKEN__"
def main():
    if len(sys.argv) < 2:
        sys.argv.append("help")
    cmd, args = sys.argv[1], {}
    for a in sys.argv[2:]:
        if "=" not in a:
            print(f"bad argument {a!r}: use key=value"); sys.exit(2)
        k, v = a.split("=", 1)
        try:
            args[k] = json.loads(v)
        except ValueError:
            args[k] = v
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/call", data=json.dumps({"cmd": cmd, "args": args}).encode(),
                                 headers={"Content-Type": "application/json", "X-Token": TOKEN})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=600) as r:
        print(r.read().decode())
main()
'''


def strip_axes(text):
    """v2: keep the robot-centric words of the F2 feedback ("to the robot's left", "forward") but drop the explicit
    coordinate-axis labels ("(+y)", "+x, ", "+yaw"), so the feedback does not hand over the axis convention that the
    v2 instructions deliberately leave out."""
    import re
    t = re.sub(r"\s?\((?:[+-]?[xyz]|[+-]yaw)\)", "", text)          # " (+y)", " (x)", " (-yaw)"
    t = re.sub(r"\(([+-][xyz]|[+-]yaw), ", "(", t)                     # "(+x, away from the robot)"
    t = re.sub(r", [+-](?:[xyz]|yaw)\)", ")", t)                       # "(... robot's left, +yaw)"
    t = re.sub(r"\s{2,}", " ", t)
    return t


class VideoRec:
    """Records agentview | wrist while the simulator steps (one frame every `every` control steps, played at
    `fps`, i.e. sped up by every * fps / 20). A header shows the attempt, the current robot command and the
    report of the previous one; gripper commands and attempt outcomes get short freeze frames."""

    BAND = 56

    def __init__(self, path, env, res=320, every=3, fps=20, title=""):
        import imageio
        from . import camera as _C
        self.env, self.res, self.every, self.fps, self.title = env, res, every, fps, title
        self.cams = [_C.mj_cam("agentview"), _C.mj_cam("wrist")]
        self.w = imageio.get_writer(path, fps=fps, quality=7, macro_block_size=8)
        self.n = 0
        self.lines = ["", "", ""]
        self.last = None
        orig = env.step

        def step(a):
            out = orig(a)
            self.n += 1
            if self.w is not None and self.n % self.every == 0:
                self.grab()
            return out

        env.step = step

    def _font(self, size):
        from PIL import ImageFont
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    def grab(self):
        from PIL import Image, ImageDraw
        from .. import envs as _E
        img = np.concatenate([_E.render(self.env, c, self.res) for c in self.cams], axis=1)
        im = Image.new("RGB", (2 * self.res, self.res + self.BAND), (0, 0, 0))
        im.paste(Image.fromarray(img), (0, self.BAND))
        d = ImageDraw.Draw(im)
        hdr = [self.title] + self.lines[:2]
        for i, s_ in enumerate(hdr):
            d.text((5, 3 + 17 * i), s_[:120], fill=(255, 255, 255) if i else (255, 220, 120), font=self._font(13))
        self.last = np.asarray(im)
        self.w.append_data(self.last)

    def hold(self, secs):
        if self.w is None:
            return
        self.grab()
        for _ in range(int(secs * self.fps)):
            self.w.append_data(self.last)

    def card(self, lines, secs=3.0, color=(20, 20, 20)):
        if self.w is None:
            return
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (2 * self.res, self.res + self.BAND), color)
        d = ImageDraw.Draw(im)
        y = 14
        for s_ in lines:
            for part in textwrap.wrap(s_, 62) or [""]:
                d.text((14, y), part, fill=(255, 255, 255), font=self._font(15))
                y += 19
            y += 5
        fr = np.asarray(im)
        for _ in range(int(secs * self.fps)):
            self.w.append_data(fr)

    def close(self):
        if self.w is not None:
            self.w.close()
            self.w = None


def _short(rep):
    keys = ("touched_object", "holding_object", "gripper_width_m", "eef_pos", "eef_yaw_deg", "error")
    return ", ".join(f"{k}={rep[k]}" for k in keys if k in rep)


class RobotSession:
    def __init__(self, a):
        self.a = a
        self.task = TASKS[a.task](cam_res=a.image_res)
        if a.max_steps is None:  # long procedural tasks declare a larger per-attempt budget
            a.max_steps = int(getattr(self.task, "agent_max_steps", 40))
        self.inst = self.task.sample_instance(a.seed)
        if getattr(a, "nominal", False):
            # control condition: same scene, hidden calibration error removed (L1 tasks only) -> plain pick-and-place
            hidden = [k for k in ("bias", "yaw_bias") if k in self.inst]
            if not hidden:
                raise SystemExit(f"--nominal is not supported for {a.task} (no bias / yaw_bias in the instance)")
            for k in [k for k in self.inst if "bias" in k]:  # bias, bias_r, yaw_bias, yaw_bias_deg, ...
                v = self.inst[k]
                if isinstance(v, (list, tuple)):
                    self.inst[k] = [0.0] * len(v)
                elif isinstance(v, (int, float)):
                    self.inst[k] = 0.0
            self.inst["nominal"] = True
        self.protocol = getattr(self.task, "protocol", "cross")
        self.k = a.k or self.task.max_attempts
        self.sandbox = os.path.abspath(a.sandbox)
        self.obs_dir = os.path.join(self.sandbox, "obs")
        self.log_dir = os.path.abspath(a.log_dir)
        os.makedirs(self.obs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "images"), exist_ok=True)
        self.events = open(os.path.join(self.log_dir, "events.jsonl"), "a")
        import secrets
        self.token = secrets.token_hex(8)
        self.n_actions = 0  # robot commands in the current attempt (step budget)
        self.attempt = 1
        self.step = 0
        self.finished = False
        self.waiting_new_agent = False
        self.history = []
        self.last_send = None
        cam_axes = {c: C.describe_axes(self.task.env, c, a.image_res) for c in C.CAMERAS}
        names = getattr(self.task, "agent_object_names", None) or list(self.task.env.objects_dict)
        memory = "none" if a.memory == "none" else "full"
        self.pb = PromptBuilder(self.task, self.protocol, self.k, memory, a.feedback, a.image_res, cam_axes,
                                object_names=names, cams=("agentview", "wrist"), depth_tool=a.perception == "depth")
        self.video = None
        if a.video:
            self.video = VideoRec(a.video, self.task.env, every=a.video_every, fps=a.video_fps,
                                  title=f"{a.task}  seed={a.seed}  perception={a.perception}  (sped up x{a.video_every * a.video_fps // 20})")
            self.video.card([f"Task: {a.task}   seed {a.seed}   memory={a.memory}  feedback={a.feedback}  k={self.k}",
                             "Instruction: " + self.instruction(),
                             "Left: agentview   Right: wrist camera.  Header: current robot command and the report of "
                             "the previous one."], 3)
        self._start_attempt(reset=True)
        # la: agent-facing positions are in the robot-base frame (LIBERO-Agent: "Coordinates are expressed in the
        # robot-base frame and measured in metres"); the LIBERO base is axis-aligned with the world, so a translation
        self.base_off = np.zeros(3)
        if a.style == "la":
            sim = self.task.env.sim
            bid = sim.model.body_name2id("robot0_base")
            assert np.allclose(sim.data.body_xmat[bid].reshape(3, 3), np.eye(3), atol=1e-6), "robot base is rotated"
            self.base_off = np.array(sim.data.body_xpos[bid], float).copy()
        self._write_sandbox()
        self.log(dict(ev="session_start", task=a.task, seed=a.seed, k=self.k, protocol=self.protocol,
                      memory=a.memory, feedback=a.feedback, instruction=a.instruction, perception=a.perception,
                      inst={k: v for k, v in self.inst.items() if not k.startswith("_")}))

    # ------------------------------------------------------------------ helpers
    def log(self, d):
        d["t"] = time.time()
        self.events.write(json.dumps(d, default=_json_default) + "\n")
        self.events.flush()

    def instruction(self):
        hook = getattr(self.task, "agent_instruction", None)
        if hook is not None:
            return hook(self.inst, self.a.instruction)
        if self.a.instruction == "indirect" and getattr(self.task, "instruction_indirect", ""):
            return self.task.instruction_indirect
        return self.task.instruction

    def feedback_text(self, out):
        import inspect
        fn = self.task.feedback
        if "indirect" in inspect.signature(fn).parameters:
            fb = fn(self.inst, out, self.a.feedback, indirect=self.a.instruction == "indirect")
        else:
            fb = fn(self.inst, out, self.a.feedback)
        return strip_axes(fb) if self.a.style in ("v2", "la") else fb

    def _start_attempt(self, reset):
        self.n_actions = 0
        if reset:
            self.task.reset_instance(self.inst)
        self.ex = ToolExecutor(self.task, self.inst, image_res=self.a.image_res, cams=("agentview", "wrist"))
        self.step = 0
        self.sim0 = self.task.sk.n_steps  # v2: simulated-time budget is counted per attempt

    def _save_obs(self, tag):
        """Same bytes as the GPT harness sends (JPEG q90). Under memory=none the file names carry no
        attempt number, so a fresh agent cannot tell it is a retry."""
        from PIL import Image
        paths = []
        for ref in self.ex.observe(label=tag):
            name = (f"s{self.step:03d}_{ref.camera}.jpg" if self.a.memory == "none"
                    else f"a{self.attempt}_s{self.step:03d}_{ref.camera}.jpg")
            Image.fromarray(ref.array).save(os.path.join(self.obs_dir, name), quality=90)
            shutil.copy(os.path.join(self.obs_dir, name), os.path.join(self.log_dir, "images", f"a{self.attempt}_{name}"))
            paths.append(f"obs/{name}")
        if self.a.style == "la":
            self.last_files = self._write_la_files(f"a{self.attempt}_s{self.step:03d}")
        return paths

    def readme(self):
        if self.a.style == "la":
            return self.readme_la()
        if self.a.style == "v2":
            return self.readme_v2()
        sp = self.pb.system_prompt()
        sp = sp.replace("You act only by calling the provided tools, exactly one tool call per turn. Every action tool "
                        "returns an execution report; after each action you also receive new camera images and the "
                        "robot's proprioceptive state.",
                        "You act only through the `./robot` command in this directory (one robot command at a time). "
                        "Every action command prints an execution report and the paths of new camera images taken "
                        "after the action; look at them with your file-reading tool.")
        sp = sp.replace("Finish by calling done(reason)", "Finish by running `./robot done reason=\"...\"`")
        sp = sp.replace("Each attempt ends when you call done; you will then see a summary of the attempt (key images, the "
                        "actions you executed and outcome feedback).",
                        "Each attempt ends when you run `./robot done`; that command prints the outcome feedback (if any) "
                        "and the images of the reset scene for the next attempt.")
        sp = sp.replace("Each attempt ends when you call done; you will then see a summary of the attempt (key images, the "
                        "actions you executed).",
                        "Each attempt ends when you run `./robot done`; that command prints the images of the reset "
                        "scene for the next attempt.")
        sp = sp.replace("before your first tool call", "before your first robot command")
        tools = "\n".join([
            "  ./robot help                                  this text",
            "  ./robot status                                task, current images and proprioception",
            "  ./robot observe                               take new images (agentview + wrist) + proprioception",
            "  ./robot move_to x=.. y=.. z=.. yaw=.. [speed=..]   yaw in degrees; speed m/s (omit = fastest)",
            "  ./robot open_gripper | ./robot close_gripper",
            "  ./robot pixel_to_world camera=agentview|wrist u=.. v=..   (pixel -> 3-D point via depth)",
            "  ./robot done reason=\"...\"                     end the attempt / segment",
        ] if self.a.perception == "depth" else [
            "  ./robot help                                  this text",
            "  ./robot status                                task, current images and proprioception",
            "  ./robot observe                               take new images (agentview + wrist) + proprioception",
            "  ./robot move_to x=.. y=.. z=.. yaw=.. [speed=..]   yaw in degrees; speed m/s (omit = fastest)",
            "  ./robot open_gripper | ./robot close_gripper",
            "  ./robot done reason=\"...\"                     end the attempt / segment",
        ])
        return ("# Robot task\n\n" + sp + "\n\n## Commands\n" + tools +
                "\n\nArguments are key=value; numbers are plain (e.g. x=0.12). Quote strings with spaces.\n"
                "Images are written to obs/ inside this directory. After every action, look at the new images "
                "before deciding on the next command.\n"
                f"Each attempt allows at most {self.a.max_steps} robot commands (help/status not counted); after that "
                "the attempt ends automatically.\n\n## TASK\n" + self.instruction() + "\n")

    def _write_sandbox(self):
        os.makedirs(self.sandbox, exist_ok=True)
        if self.a.style == "la":
            with open(os.path.join(self.log_dir, "agent_iface.json"), "w") as f:
                json.dump(dict(token=self.token, port=self.a.port, workspace=self.sandbox, agent_iface=self.a.agent_iface,
                               max_tool_calls=self.a.max_tool_calls, perception=self.a.perception), f)
            with open(os.path.join(self.sandbox, "README.md"), "w") as f:
                f.write(self.readme())
            if self.a.agent_iface == "cli":  # coding-agent harness (Claude Code): ./robot + ./shell in the workspace
                for name, body in (("robot", CLI_TEMPLATE.replace("__PORT__", str(self.a.port)).replace("__TOKEN__", self.token)),
                                   ("shell", SHELL_TEMPLATE)):
                    with open(os.path.join(self.sandbox, name), "w") as f:
                        f.write(body)
                    os.chmod(os.path.join(self.sandbox, name), 0o755)
            return
        cli = os.path.join(self.sandbox, "robot")
        with open(cli, "w") as f:
            f.write(CLI_TEMPLATE.replace("__PORT__", str(self.a.port)).replace("__TOKEN__", self.token))
        os.chmod(cli, 0o755)
        with open(os.path.join(self.sandbox, "README.md"), "w") as f:
            f.write(self.readme())

    def status(self):
        if self.a.memory == "none":
            return dict(task=self.instruction(), finished=self.finished)
        st = dict(task=self.instruction(), protocol=self.protocol, attempt=self.attempt, max_attempts=self.k,
                  finished=self.finished)
        if self.waiting_new_agent:
            st["note"] = "The previous agent finished its attempt. A new attempt is ready; the scene has been reset."
        return st

    # ------------------------------------------------------------------ dispatch
    def handle(self, cmd, args):
        if self.a.style in ("v2", "la"):
            return self.handle_v2(cmd, args)
        if cmd == "help":
            return dict(text=self.readme())
        if cmd == "status":
            st = self.status()
            st["images"] = self._save_obs("status")
            st.update(self.ex.proprio())
            self.waiting_new_agent = False
            return st
        if self.finished:
            return dict(ok=False, error="The episode is over. Stop now.")
        if self.waiting_new_agent:
            return dict(ok=False, error="This attempt has ended. Stop now.")
        if cmd in ("observe", "pixel_to_world", "move_to", "open_gripper", "close_gripper", "push"):
            self.n_actions += 1
            if self.n_actions > self.a.max_steps:
                rep = self._done("step budget exhausted (automatic)")
                rep["note"] = f"The step budget of {self.a.max_steps} commands for this attempt was used up."
                return rep
        if cmd == "observe":
            self.step += 1
            rep = dict(ok=True, images=self._save_obs("observe"))
            rep.update(self.ex.proprio())
            self.log(dict(ev="call", attempt=self.attempt, step=self.step, cmd=cmd, args=args, report=rep))
            return rep
        if cmd == "done":
            return self._done(args.get("reason", ""))
        allowed = tuple(t for t in ACTION_TOOLS if t != "push" or self.a.allow_push)
        if self.a.perception == "depth":
            allowed += ("pixel_to_world",)
        if cmd not in allowed:
            return dict(ok=False, error=f"unknown command {cmd!r}; run ./robot help")
        self.step += 1
        if self.video is not None:
            argtxt = " ".join(f"{k}={v}" for k, v in args.items())
            self.video.lines[0] = f"attempt {self.attempt}/{self.k}   command #{self.n_actions}: {cmd} {argtxt}"
        res = self.ex.call(cmd, args)
        rep = dict(res.report)
        if self.video is not None:
            self.video.lines[1] = f"  -> {_short(rep)}"
            if cmd in ("close_gripper", "open_gripper"):
                self.video.hold(1.2)
        if res.ok and cmd in ACTION_TOOLS:
            rep["images"] = self._save_obs(cmd)
        self.log(dict(ev="call", attempt=self.attempt, step=self.step, cmd=cmd, args=args, report=rep,
                      wall_s=res.wall_s, sim_steps=res.sim_steps))
        return rep

    def _done(self, reason):
        out = self.task.outcome(self.inst)
        succ = bool(out["success"])
        fb = self.feedback_text(out)
        self.history.append(dict(attempt=self.attempt, success=succ, feedback=fb, reason=reason))
        self.log(dict(ev="done", attempt=self.attempt, step=self.step, reason=reason, success=succ, feedback=fb,
                      outcome={k: v for k, v in out.items() if k != "feedback"}))
        last = self.attempt >= self.k or succ or bool(out.get("terminal"))
        if self.video is not None:
            self.video.lines = [f"attempt {self.attempt}/{self.k}: done", f"  reason: {reason}", ""]
            self.video.hold(1.0)
            self.video.card([f"Attempt {self.attempt}: {'SUCCESS' if succ else 'FAILED'}",
                             f"Commands used in this attempt: {self.n_actions}", "Agent's reason: " + str(reason)[:300],
                             "Feedback: " + (fb if self.a.memory != "none" and self.a.feedback != "F0" else "(not shown)")],
                            3.5, (20, 50, 20) if succ else (60, 20, 20))
            if last:
                self.video.close()
            else:
                self.video.lines = ["", "", ""]
        if self.a.memory == "none":
            # independent attempts: no feedback to the agent; orchestrator starts a fresh agent
            if last:
                self.finished = True
            else:
                self.attempt += 1
                self._start_attempt(reset=self.protocol == "cross")
                self.waiting_new_agent = True
            return dict(ok=True, ended=True, message="Attempt recorded. Stop now and report what you did.")
        rep = dict(ok=True, ended=True)
        if self.a.feedback != "F0":
            rep["feedback"] = fb
        if last and (succ or self.a.feedback != "F0"):
            self.finished = True
            rep["message"] = "The episode is over. Stop now and report."
            return rep
        if self.attempt >= self.k or out.get("terminal"):
            self.finished = True
            rep["message"] = ("The failure cannot be undone, so the episode is over. Stop now and report."
                              if out.get("terminal") else "No attempts left. The episode is over. Stop now and report.")
            return rep
        self.attempt += 1
        if self.protocol == "cross":
            self._start_attempt(reset=True)
            rep["message"] = (f"Attempt {self.attempt} of {self.k} starts now. The scene has been reset to exactly the "
                              f"same initial state.")
        else:
            self._start_attempt(reset=False)
            rep["message"] = (f"Continue from the current state (segment {self.attempt}; you may call done at most "
                              f"{self.k} times in total).")
        self.step += 1
        rep["images"] = self._save_obs("attempt_start")
        rep.update(self._la_files())
        if self.a.style in ("v2", "la"):
            rep.update(self._state_v2())
        else:
            rep.update(self.ex.proprio())
        return rep


V2_DROP = ("holding", "touched", "events", "sim_time_s", "tool", "reached", "yaw_used_deg", "target_clipped_to")


class _V2:
    """Interface v2 (2026-10-07, aligned with LIBERO-Agent / RoboDojo-L3 at the user's request):
    - priors: only the task instruction, the attempt rules, the budgets and the command definitions (no axes,
      table height, camera orientation or robot geometry);
    - observations: two RGB images published as files after every command + end-effector state (position,
      yaw, gripper width) + how far the arm stopped from the commanded target + budget left; no
      held/touched flags and no per-step task events;
    - action: move_eef = absolute target (omitted coordinates keep their value; gripper after the motion;
      optional speed); unreachable targets are rejected and the arm does not move;
    - an attempt ends with `finish` (alias `done`) or when a budget runs out; the environment judges it."""


def _readme_v2(self):
    sec = self.sim_budget()
    proto = (f"You have up to {self.k} attempts at this task. Before every attempt the scene is reset to exactly the "
             f"same initial state." if self.protocol == "cross" else
             f"Everything happens in one continuous scene. You may end an attempt up to {self.k} times; after a "
             f"failed attempt you continue from the current state.")
    return "\n".join([
        "# Robot task", "",
        "You control a robot arm with a two-finger gripper in a physics simulation, only through the `./robot` "
        "command in this directory (one command at a time).", "",
        "## Task", self.instruction(), "",
        "## Attempts", proto + " An attempt ends when you run `./robot finish reason=\"...\"` or when its budget "
        "is used up. " + _fb_sentence(self.a.feedback) + " The episode ends at the first success or after the last "
        "attempt.", "",
        "## Budget (per attempt)",
        f"- at most {self.a.max_steps} robot commands (status and help are free);",
        f"- at most {sec:.0f} s of simulated robot motion.",
        "Every reply shows what is left.", "",
        "## Observations",
        "Every command writes two new camera images into obs/ and prints their paths: 'agentview' (a fixed camera "
        "looking at the table) and 'wrist' (a camera on the gripper). It also reports the end-effector state: "
        "eef_pos = [x, y, z] of the gripper's grasp point in metres, eef_yaw_deg = rotation of the gripper about "
        "the vertical axis in degrees, gripper_width_m = opening of the fingers in metres, and after a motion how "
        "far the gripper stopped from your target.", "",
        "## Commands",
        "  ./robot status                       task, attempt, budget, current state and images",
        "  ./robot observe                      new images and state without moving",
        "  ./robot move_eef [x=..] [y=..] [z=..] [yaw=..] [gripper=open|close] [speed=..]",
        "        move the grasp point to an absolute target in the same frame as eef_pos (metres; yaw in degrees);",
        "        omitted coordinates keep their current value; the gripper command is executed after the motion;",
        "        speed in m/s (omit = fastest); a target the arm cannot reach is rejected and the arm does not move.",
        "  ./robot finish reason=\"...\"          end the current attempt",
        "",
        "Arguments are key=value; numbers are plain (e.g. x=0.12). Quote strings with spaces.", ""])


def _sim_budget(self):
    return 3.0 * self.a.max_steps  # seconds of simulated motion per attempt


def _budget_left(self):
    used = (self.task.sk.n_steps - self.sim0) * 0.05
    return dict(commands=max(self.a.max_steps - self.n_actions, 0), sim_seconds=round(max(self.sim_budget() - used, 0.0), 1))


def _state_v2(self):
    p = self.ex.proprio()
    pos = p["eef_pos"]
    if self.a.style == "la":
        pos = [round(float(v), 4) for v in np.asarray(pos, float) - self.base_off]
    st = dict(eef_pos=pos, eef_yaw_deg=p["eef_yaw_deg"], gripper_width_m=p["gripper_width_m"])
    if self.a.style == "la":  # read-only joint angles, as LIBERO-Agent (all levels) and RoboDojo give them
        st["joint_pos_rad"] = [round(float(v), 4) for v in self._believed_joints()]
    st["budget_left"] = self._budget_left()
    return st


def _clean_v2(rep):
    return {k: v for k, v in rep.items() if not k.startswith(V2_DROP)}


def _handle_v2(self, cmd, args):
    if cmd == "help":
        return dict(text=self.readme())
    if cmd == "status":
        st = dict(task=self.instruction(), attempt=self.attempt, max_attempts=self.k, finished=self.finished)
        st["images"] = self._save_obs("status")
        st.update(self._la_files())
        st.update(self._state_v2())
        return st
    if self.finished:
        return dict(ok=False, error="The episode is over. Stop now.")
    if cmd in ("finish", "done"):
        return self._done(args.get("reason", ""))
    if cmd not in ("observe", "move_eef"):
        return dict(ok=False, error=f"unknown command {cmd!r}; run ./robot help")
    self.n_actions += 1
    if self.n_actions > self.a.max_steps:
        rep = self._done("command budget used up (automatic)")
        rep["note"] = f"The budget of {self.a.max_steps} commands for this attempt was used up."
        return rep
    self.step += 1
    if cmd == "observe":
        rep = dict(ok=True, images=self._save_obs("observe"))
        rep.update(self._la_files())
        rep.update(self._state_v2())
        self.log(dict(ev="call", attempt=self.attempt, step=self.step, cmd=cmd, args=args, report=rep))
        return rep
    # ---------------- move_eef
    p = self.ex.proprio()
    unknown = set(args) - {"x", "y", "z", "yaw", "gripper", "speed"}
    if unknown:
        rep = dict(ok=False, error=f"unknown argument(s) {sorted(unknown)}; use x, y, z, yaw, gripper, speed")
        self.log(dict(ev="call", attempt=self.attempt, step=self.step, cmd=cmd, args=args, report=rep))
        return rep
    off = self.base_off  # zero except la (robot-base frame)
    try:
        cur = np.asarray(p["eef_pos"], float) - off
        tgt = [float(args.get(k, cur[i])) for i, k in enumerate("xyz")]
        yaw = float(args.get("yaw", p["eef_yaw_deg"]))
        speed = None if args.get("speed") in (None, "") else float(args["speed"])
    except (TypeError, ValueError):
        rep = dict(ok=False, error="x, y, z, yaw and speed must be numbers")
        self.log(dict(ev="call", attempt=self.attempt, step=self.step, cmd=cmd, args=args, report=rep))
        return rep
    grip = args.get("gripper")
    if grip not in (None, "open", "close"):
        rep = dict(ok=False, error="gripper must be open or close")
        self.log(dict(ev="call", attempt=self.attempt, step=self.step, cmd=cmd, args=args, report=rep))
        return rep
    moving = any(k in args for k in ("x", "y", "z", "yaw"))
    tgt_w = np.asarray(tgt) + off  # simulator (world) frame
    if moving and not (np.all(tgt_w >= WS_LO) and np.all(tgt_w <= WS_HI)):
        rep = dict(ok=False, error="target rejected: the arm cannot reach it; the arm did not move")
        rep.update(self._state_v2())
        self.log(dict(ev="call", attempt=self.attempt, step=self.step, cmd=cmd, args=args, report=rep))
        return rep
    if self.video is not None:
        argtxt = " ".join(f"{k}={v}" for k, v in args.items())
        self.video.lines[0] = f"attempt {self.attempt}/{self.k}   command #{self.n_actions}: {cmd} {argtxt}"
    left = self._budget_left()["sim_seconds"]
    raw = {}
    if moving:
        mv = dict(x=float(tgt_w[0]), y=float(tgt_w[1]), z=float(tgt_w[2]), yaw=yaw)
        if speed is not None:
            mv["speed"] = speed
        res = self.ex.call("move_to", mv)
        raw.update(res.report)
    if grip is not None:
        res = self.ex.call("open_gripper" if grip == "open" else "close_gripper", {})
        raw.update(res.report)
    rep = dict(ok=bool(raw.get("ok", True)))
    if raw.get("error"):
        rep["error"] = raw["error"]
    rep.update(self._state_v2())
    if moving:
        err = np.asarray(tgt) - np.asarray(rep["eef_pos"])
        dyaw = (yaw - rep["eef_yaw_deg"] + 90.0) % 180.0 - 90.0
        rep["stopped_from_target"] = dict(cm=round(100 * float(np.linalg.norm(err)), 1), deg=round(abs(dyaw), 1))
    if self.video is not None:
        self.video.lines[1] = f"  -> eef={rep['eef_pos']} width={rep['gripper_width_m']} {rep.get('stopped_from_target', '')}"
        if grip is not None:
            self.video.hold(1.2)
    rep["images"] = self._save_obs(cmd)
    rep.update(self._la_files())
    self.log(dict(ev="call", attempt=self.attempt, step=self.step, cmd=cmd, args=args, report=rep, raw=raw))
    if rep["budget_left"]["sim_seconds"] <= 0 and left > 0:
        fin = self._done("motion budget used up (automatic)")
        fin["note"] = f"The {self.sim_budget():.0f} s motion budget for this attempt was used up."
        rep["attempt_ended"] = fin
    return rep


def _la_files(self):
    """la: paths of the non-image observation files written with the last images (empty for other styles)."""
    return dict(self.last_files) if self.a.style == "la" and getattr(self, "last_files", None) else {}


def _believed_pose(self):
    """4x4 pose of the grasp point as the ROBOT believes it (the frame of eef_pos / move_eef): the true pose shifted by
    the task's hidden calibration errors (Skills.bias, and YawSkills.yaw_bias about the vertical axis). A real robot
    computes its wrist-camera pose from its own (miscalibrated) kinematics, so the wrist calibration uses this pose."""
    sk = self.task.sk
    beta = float(getattr(sk, "yaw_bias", 0.0))
    c, s_ = np.cos(-beta), np.sin(-beta)
    Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    T_true = np.eye(4)
    T_true[:3, :3], T_true[:3, 3] = sk.eef_mat(), sk.eef_pos()
    T_b = np.eye(4)
    T_b[:3, :3], T_b[:3, 3] = Rz @ sk.eef_mat(), sk.eef_pos() - np.asarray(sk.bias, float)
    return T_b, T_true


def _believed_joints(self, iters=30):
    """The 7 arm joint angles as the robot's encoders report them. For tasks with a hidden calibration error (L1) these
    are the angles consistent with the robot's own belief: standard forward kinematics of them gives exactly the
    reported (believed) grasp pose, as on a real arm whose encoder offsets are miscalibrated; true joint angles would
    hand the hidden error to anyone who runs the public Franka kinematics. Without a calibration error they are the true
    angles. Solved by damped least squares on a scratch copy of the simulator state (2-3 iterations)."""
    import mujoco
    env, sk = self.task.env, self.task.sk
    rob = env.robots[0]
    idx, dof, site = np.array(rob._ref_joint_pos_indexes), np.array(rob._ref_joint_vel_indexes), rob.eef_site_id
    m, d0 = env.sim.model._model, env.sim.data._data
    T_b, _ = self._believed_pose()
    if np.allclose(T_b[:3, 3], d0.site_xpos[site]) and np.allclose(T_b[:3, :3], d0.site_xmat[site].reshape(3, 3)):
        return d0.qpos[idx].copy()
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        ep = T_b[:3, 3] - d.site_xpos[site]
        Re = T_b[:3, :3] @ d.site_xmat[site].reshape(3, 3).T
        er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        if np.linalg.norm(ep) < 1e-7 and np.linalg.norm(er) < 1e-7:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, dof], jr[:, dof]])
        d.qpos[idx] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.concatenate([ep, er]))
    return d.qpos[idx].copy()


def _write_la_files(self, prefix):
    """la observation files besides the two JPEGs: <prefix>_state.json always; with --perception rgbd also metric
    depth for both cameras (<prefix>_<cam>_depth.npy, float32 metres along the optical axis, same upright pixels as
    the JPEG) and <prefix>_camera.json (K and cam_to_world in the eef_pos frame, OpenCV camera axes).
    The fixed agentview camera is calibrated in the true frame; the wrist camera pose is computed from the robot's
    believed gripper pose (so neither file reveals the hidden calibration error of the L1 tasks)."""
    out = {}
    st = self._state_v2()
    st = dict(eef_pos=st["eef_pos"], eef_yaw_deg=st["eef_yaw_deg"], gripper_width_m=st["gripper_width_m"],
              joint_pos_rad=st["joint_pos_rad"], attempt=self.attempt, step=self.step)
    with open(os.path.join(self.obs_dir, f"{prefix}_state.json"), "w") as f:
        json.dump(st, f, default=_json_default)
    out["state_file"] = f"obs/{prefix}_state.json"
    if self.a.perception == "rgbd":
        res = self.a.image_res
        cams, depth_files = {}, []
        T_b, T_true = self._believed_pose()
        for cam in ("agentview", "wrist"):
            d = C.render_depth(self.task.env, cam, res).astype(np.float32)
            np.save(os.path.join(self.obs_dir, f"{prefix}_{cam}_depth.npy"), d)
            depth_files.append(f"obs/{prefix}_{cam}_depth.npy")
            E = C.cam_to_world(self.task.env, cam)
            if cam == "wrist":
                E = T_b @ np.linalg.inv(T_true) @ E
            E = E.copy()
            E[:3, 3] -= self.base_off  # robot-base frame
            cams[cam] = dict(K=np.round(C.intrinsics(self.task.env, cam, res), 6).tolist(),
                             cam_to_base=np.round(E, 6).tolist(), width=res, height=res)
        with open(os.path.join(self.obs_dir, f"{prefix}_camera.json"), "w") as f:
            json.dump(cams, f)
        out["depth_files"] = depth_files
        out["camera_file"] = f"obs/{prefix}_camera.json"
    return out



def _fb_sentence(level):
    """README sentence for the feedback level. F1 (default since 2026-10-08): success / failure only, like
    LIBERO-Agent's finish_episode outcome; F2 (measured error, e.g. "2.5 cm past the centre") is privileged
    simulator information that neither LIBERO-Agent nor RoboDojo gives, kept only as an ablation."""
    return {"F0": "The environment then checks the result.",
            "F1": "The environment then checks the result and tells you only whether the attempt succeeded.",
            "F2": "The environment then checks the result and tells you whether the attempt succeeded and, if not, "
                  "what went wrong."}[level]

def _readme_la(self):
    """LIBERO-Agent-style instructions: the agent works in an episode workspace (observations are files, it can run
    programs on them); the robot itself is driven only through the harness's `robot` tool."""
    sec = self.sim_budget()
    rgbd = self.a.perception == "rgbd"
    cli = self.a.agent_iface == "cli"
    pre = "./robot " if cli else ""
    proto = (f"You have up to {self.k} attempts at this task. Before every attempt the scene is reset to exactly the "
             f"same initial state." if self.protocol == "cross" else
             f"Everything happens in one continuous scene. You may end an attempt up to {self.k} times; after a "
             f"failed attempt you continue from the current state.")
    files = [
        "- obs/a<attempt>_s<step>_agentview.jpg and obs/a<attempt>_s<step>_wrist.jpg: 512x512 RGB images from "
        "'agentview' (a fixed camera looking at the table) and 'wrist' (a camera on the gripper);",
        "- obs/a<attempt>_s<step>_state.json: the end-effector state: eef_pos = [x, y, z] of the gripper's grasp "
        "point (midway between the two fingers, about 1 cm above the fingertips), eef_yaw_deg = rotation of the gripper about the vertical axis in degrees, gripper_width_m = opening "
        "of the fingers in metres, joint_pos_rad = the 7 arm joint angles (radians, base to wrist; read-only).",
    ]
    if rgbd:
        files += [
            "- obs/a<attempt>_s<step>_agentview_depth.npy and obs/a<attempt>_s<step>_wrist_depth.npy: metric depth "
            "(numpy float32 array, 512x512, metres along each camera's optical axis), pixel-aligned with the images "
            "(row 0 = top of the image);",
            "- obs/a<attempt>_s<step>_camera.json: for each camera, K (3x3 intrinsics of the 512x512 image) and "
            "cam_to_base (4x4 pose of the camera in the robot-base frame; camera axes: x right, y down, z forward).",
        ]
    return "\n".join([
        "# Robot task", "",
        "You control a Franka Emika Panda robot arm (7 joints) with a two-finger parallel gripper in a physics "
        "simulation.", "",
        "## Task", self.instruction(), "",
        "## Attempts", proto + f" An attempt ends when you run the robot command `{pre}finish reason=\"...\"` or when its "
        "budget is used up. " + _fb_sentence(self.a.feedback) + " The episode ends at the first success or after the "
        "last attempt.", "",
        "## Budget (per attempt)",
        f"- at most {self.a.max_steps} robot commands (status and help are free);",
        f"- at most {sec:.0f} s of simulated robot motion;",
        *([] if cli else [f"- at most {self.a.max_tool_calls} tool calls in total (robot, shell and view_image together)."]),
        "Every robot reply shows the robot budgets that are left.", "",
        "## Coordinates",
        "Coordinates are expressed in the robot-base frame and measured in metres; angles are in degrees.", "",
        "## Workspace and observations",
        "This directory is your workspace for the whole episode. Every robot command writes its observation into "
        "obs/ and prints the paths of the new files:",
        *files,
        "Robot replies also include the end-effector state and, after a motion, how far the gripper stopped from your "
        "target. " + ("You can look at any image with your file-reading tool, run shell commands and programs in this "
                      "workspace with `./shell '<command>'` (python3 with numpy, scipy, Pillow and OpenCV; no network; "
                      "programs cannot control the robot), and keep your own files and notes in this directory; they "
                      "persist across attempts." if cli else
                      "You can look at any image with view_image, run shell commands and programs here (python3 with "
                      "numpy, scipy, Pillow and OpenCV; no network; programs cannot control the robot), and keep your "
                      "own files and notes in this directory; they persist across attempts."), "",
        "## Robot commands " + ("(run from this directory as `./robot <command>`, one command at a time)" if cli
                                else "(argument of the `robot` tool)"),
        f"  {pre}status                       task, attempt, budget, current state and observation files",
        f"  {pre}observe                      new observation without moving",
        f"  {pre}move_eef [x=..] [y=..] [z=..] [yaw=..] [gripper=open|close] [speed=..]",
        "        move the grasp point to an absolute target (robot-base frame, metres; yaw in degrees);",
        "        omitted coordinates keep their current value; the gripper command is executed after the motion;",
        "        speed in m/s (omit = fastest); a target the arm cannot reach is rejected and the arm does not move.",
        f"  {pre}finish reason=\"...\"          end the current attempt",
        "",
        "Arguments are key=value; numbers are plain (e.g. x=0.12). Quote strings with spaces.", ""])


for _n, _f in (("_believed_joints", _believed_joints), ("readme_la", _readme_la), ("_la_files", _la_files), ("_believed_pose", _believed_pose),
               ("_write_la_files", _write_la_files), ("readme_v2", _readme_v2), ("sim_budget", _sim_budget), ("_budget_left", _budget_left),
               ("_state_v2", _state_v2), ("handle_v2", _handle_v2)):
    setattr(RobotSession, _n, _f)
RobotSession._clean_v2 = staticmethod(_clean_v2)


def serve(a):
    sess = RobotSession(a)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_POST(self):
            t_recv = time.time()
            cmd = ""
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                cmd = body.get("cmd", "")
                if self.headers.get("X-Token") != sess.token:
                    rep = dict(ok=False, error="unauthorised")
                else:
                    rep = sess.handle(cmd, body.get("args", {}) or {})
            except Exception as e:  # never leak a traceback (it contains benchmark paths) to the agent
                sess.log(dict(ev="error", error=traceback.format_exc()))
                rep = dict(ok=False, error=f"internal error: {type(e).__name__}")
            data = json.dumps(rep, default=_json_default, indent=None).encode()
            if "text" in rep:
                data = rep["text"].encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
            t_send = time.time()
            # think time = gap between the previous reply and this request (model inference + its own tool overhead)
            sess.log(dict(ev="http", cmd=cmd, attempt=sess.attempt, t_recv=t_recv, t_send=t_send,
                          server_s=t_send - t_recv, think_s=(t_recv - sess.last_send) if sess.last_send else None))
            sess.last_send = t_send

    def _term(*_):
        if sess.video is not None:
            sess.video.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, _term)
    srv = HTTPServer(("127.0.0.1", a.port), H)
    print(f"robot server ready on 127.0.0.1:{a.port} sandbox={sess.sandbox} log={sess.log_dir}", flush=True)
    with open(os.path.join(sess.log_dir, "READY"), "w") as f:
        f.write(str(os.getpid()))
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--memory", default="full", choices=["full", "none"])
    ap.add_argument("--feedback", default="F1", choices=["F0", "F1", "F2"])
    ap.add_argument("--instruction", default="direct", choices=["direct", "indirect"])
    ap.add_argument("--image-res", type=int, default=512)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--sandbox", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="robot commands per attempt (default: the task's agent_max_steps, else 40)")
    ap.add_argument("--allow-push", action="store_true", help="expose the composite push primitive (off by default)")
    ap.add_argument("--perception", default="depth", choices=["depth", "rgb", "rgbd"],
                    help="rgb: images only (no pixel_to_world / depth); depth: adds the depth back-projection tool "
                         "(v1); rgbd (la): also writes metric depth and camera calibration files")
    ap.add_argument("--style", default="v1", choices=["v1", "v2", "la"],
                    help="v2: LIBERO-Agent / RoboDojo-aligned interface (see _V2); la: v2 + LIBERO-Agent workspace "
                         "(observation files, programs in the workspace, robot only through the harness tool)")
    ap.add_argument("--nominal", action="store_true",
                    help="control: remove the hidden calibration error of an L1 task (bias / yaw_bias = 0)")
    ap.add_argument("--agent-iface", default="tools", choices=["tools", "cli"],
                    help="la: tools = harness tools (robot / shell / view_images, our GPT driver); cli = ./robot and "
                         "./shell in the workspace for a coding agent such as Claude Code")
    ap.add_argument("--max-tool-calls", type=int, default=100,
                    help="la: tool calls per attempt (robot + shell + view_image), enforced by the harness")
    ap.add_argument("--video", default="", help="record agentview|wrist mp4 to this path")
    ap.add_argument("--video-every", type=int, default=3, help="one frame every N control steps (20 Hz)")
    ap.add_argument("--video-fps", type=int, default=20)
    serve(ap.parse_args())


if __name__ == "__main__":
    main()
