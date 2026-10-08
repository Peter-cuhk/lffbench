"""Backend-neutral conversation, attempt records, memory selection and prompt text.

A conversation is a list of items (plain dicts):
  {"kind": "user", "parts": [{"type": "text", "text": str} | {"type": "image", "image": ImageRef}], "live": bool}
  {"kind": "assistant", "text": str|None, "tool_calls": [ToolCall], "raw": list|None, "backend": str}
  {"kind": "tool_output", "call_id": str, "name": str, "text": str}
The system prompt is kept separately (Conversation.system). Backends convert this into their own wire
format (OpenAI Responses input items, Chat Completions messages, ...).

"live" user items are the current attempt's observations; old live images can be dropped to bound the
request size. History (earlier attempts) is packed into the attempt's first user item and never pruned,
so the request prefix stays byte-stable within an attempt and grows append-only across attempts
(prompt-cache friendly).
"""
import copy
import json
import math
from dataclasses import dataclass, field
from typing import Optional

from ..envs import TABLE_Z

MEMORY_MODES = ("none", "full", "last", "mismatched")
FEEDBACK_LEVELS = ("F0", "F1", "F2")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""


@dataclass
class StepRecord:
    step: int
    name: str
    args: dict
    report: dict
    ok: bool
    wall_s: float = 0.0
    sim_steps: int = 0
    frame: object = None  # ImageRef of the agentview after this step (for keyframes)


@dataclass
class AttemptRecord:
    seed: int
    attempt: int
    steps: list = field(default_factory=list)
    keyframes: list = field(default_factory=list)  # [(caption, ImageRef)]
    outcome: dict = field(default_factory=dict)
    success: bool = False
    feedback: str = ""
    done_reason: Optional[str] = None
    diagnosis: Optional[str] = None
    ended_by: str = ""
    backend_meta: dict = field(default_factory=dict)

    def action_signature(self, nd=2):
        """Rounded action sequence (to compare consecutive attempts: 'same mistake again')."""
        sig = []
        for s in self.steps:
            if s.name in ("get_observation", "pixel_to_world", "locate", "done"):
                continue
            sig.append((s.name, tuple(sorted((k, round(v, nd) if isinstance(v, (int, float)) and v is not None else v)
                                             for k, v in s.args.items()))))
        return tuple(sig)


def select_history(mode, own, donor=None):
    """Which earlier attempts the agent sees at the start of the next attempt."""
    if mode == "none":
        return []
    if mode == "full":
        return list(own)
    if mode == "last":
        return list(own[-1:])
    if mode == "mismatched":
        if donor is None:
            raise ValueError("mismatched memory needs donor attempts from another instance")
        return list(donor[:len(own)])
    raise ValueError(mode)


class Conversation:
    def __init__(self, system):
        self.system = system
        self.items = []

    def add_user(self, parts, live=True):
        self.items.append(dict(kind="user", parts=list(parts), live=live))

    def add_assistant(self, text, tool_calls, raw=None, backend=""):
        self.items.append(dict(kind="assistant", text=text, tool_calls=list(tool_calls), raw=raw, backend=backend))

    def add_tool_output(self, call_id, name, text):
        self.items.append(dict(kind="tool_output", call_id=call_id, name=name, text=text))

    def n_images(self):
        return sum(1 for it in self.items if it["kind"] == "user" for p in it["parts"] if p["type"] == "image")


def prune_live_images(items, max_live):
    """Copy of `items` keeping only the newest `max_live` live images. Hysteresis: when the budget is
    exceeded, prune down to half of it so the request prefix stays stable for several steps."""
    if max_live is None or max_live < 0:
        return items
    live = [(i, j) for i, it in enumerate(items) if it["kind"] == "user" and it.get("live")
            for j, p in enumerate(it["parts"]) if p["type"] == "image"]
    if len(live) <= max_live:
        return items
    keep_n = max(max_live // 2, 1)
    drop = set(live[:len(live) - keep_n])
    out = []
    for i, it in enumerate(items):
        if it["kind"] == "user" and any((i, j) in drop for j in range(len(it["parts"]))):
            it = dict(it)
            parts = []
            for j, p in enumerate(it["parts"]):
                if (i, j) in drop:
                    parts.append(dict(type="text", text=f"[{p['image'].camera} image omitted to save context]"))
                else:
                    parts.append(p)
            it["parts"] = parts
        out.append(it)
    return out


# ---------------------------------------------------------------------------------------------- text

def fmt_num(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1e4 else f"{v:.4g}"
    return json.dumps(v)


def fmt_call(name, args):
    return f"{name}(" + ", ".join(f"{k}={fmt_num(v)}" for k, v in args.items()) + ")"


def fmt_report(name, rep):
    if not rep.get("ok", False):
        return "ERROR: " + str(rep.get("error"))
    bits = []
    if "reached" in rep:
        bits.append("reached" if rep["reached"] else "NOT reached")
    if "sweep_reached" in rep:
        bits.append("sweep completed" if rep["sweep_reached"] else "sweep NOT completed")
    if "eef_pos" in rep and name in ("move_to", "push", "open_gripper", "close_gripper"):
        bits.append("eef (" + ", ".join(f"{x:.3f}" for x in rep["eef_pos"]) + ")")
    if name in ("open_gripper", "close_gripper"):
        bits.append(f"width {rep.get('gripper_width_m', 0):.3f} m")
    if rep.get("holding_object") not in (None, False):
        bits.append(f"holding {rep['holding_object']}" if isinstance(rep["holding_object"], str) else "holding an object")
    if rep.get("touched_object") or rep.get("touched_objects"):
        t = rep.get("touched_objects")
        bits.append("touched " + ", ".join(t) if t else "touched an object")
    if name == "pixel_to_world":
        bits.append("world (" + ", ".join(f"{x:.3f}" for x in rep["world_xyz"]) + ")")
    if name == "locate":
        bits.append("pos (" + ", ".join(f"{x:.3f}" for x in rep["position"]) + ")")
    if "target_clipped_to" in rep or "start_clipped_to" in rep:
        bits.append("target clipped to workspace")
    if rep.get("events"):
        bits.append("EVENT: " + " ".join(rep["events"]))
    return "; ".join(bits) if bits else "ok"


def summarize_actions(steps):
    lines = []
    for s in steps:
        if s.name == "done":
            lines.append(f"{s.step}. done(reason={json.dumps(s.args.get('reason', ''))})")
        else:
            lines.append(f"{s.step}. {fmt_call(s.name, s.args)} -> {fmt_report(s.name, s.report)}")
    return "\n".join(lines) if lines else "(no actions)"


def _text(t):
    return dict(type="text", text=t)


def _img(ref):
    return dict(type="image", image=ref)


class PromptBuilder:
    """All text the agent sees. One instance per (task, config)."""

    def __init__(self, task, protocol, k, memory, feedback, image_res, cam_axes, privileged_locate=False,
                 object_names=(), grasp_report="bool", cams=("agentview", "wrist"), depth_tool=True):
        self.task, self.protocol, self.k = task, protocol, k
        self.depth_tool = depth_tool
        self.memory, self.feedback = memory, feedback
        self.image_res, self.cam_axes = image_res, cam_axes
        self.privileged_locate, self.object_names = privileged_locate, list(object_names)
        self.grasp_report, self.cams = grasp_report, tuple(cams)

    # ----------------------------------------------------------------------------- system prompt
    def system_prompt(self):
        ax = self.cam_axes.get("agentview") or {}
        wx = self.cam_axes.get("wrist") or {}
        lines = [
            "You control a Franka Panda robot arm with a parallel-jaw gripper in a tabletop physics simulation. "
            "You act only by calling the provided tools, exactly one tool call per turn. Every action tool returns an "
            "execution report; after each action you also receive new camera images and the robot's proprioceptive "
            "state.",
            "",
            "Coordinates (world frame, metres): the table top is at z = %.2f; the robot base is at x = -0.66, y = 0; "
            "+x points away from the robot, +y to the robot's left, +z up. Positions given to move_to / push are for "
            "the gripper's grasp point (midway between the fingertips). The gripper always points straight down; yaw "
            "(degrees) turns it about the vertical axis: at yaw 0 the fingers close along the world y axis, at yaw 90 "
            "along the world x axis; yaw is taken modulo 180 degrees because the gripper is symmetric." % TABLE_Z,
            "",
            "Direction words in every text you receive (task, reports, feedback) are robot-centric: forward = +x (away "
            "from the robot), backward = -x (toward the robot), left = +y (the robot's left), right = -y (the robot's "
            "right), up = +z. They never mean left/right in an image.",
            "",
            "Cameras (%dx%d images): 'agentview' looks at the table from the side opposite the robot, so the robot "
            "appears at the top of that image; image-right is %s and image-down is %s in world coordinates (so the "
            "robot's left appears on the right side of that image). 'wrist' is "
            "mounted on the gripper and looks down between the fingers (at yaw 0, image-right is %s and image-down is "
            "%s).%s"
            % (self.image_res, self.image_res, ax.get("right", "?"), ax.get("down", "?"), wx.get("right", "?"),
               wx.get("down", "?"),
               " For pixel_to_world, u is the column counted from the left edge and v the row counted from the top "
               "edge of the image as you received it." if self.depth_tool else ""),
            "",
            "Proprioception (end-effector position, yaw, gripper width%s) comes from the robot's own sensors."
            % (", whether an object is held" if self.grasp_report != "none" else ""),
        ]
        if self.privileged_locate:
            lines += ["", "The locate tool is available and returns exact object positions. Objects: "
                      + ", ".join(self.object_names) + "."]
        else:
            lines += ["", "Object positions are not given to you: localise objects from the images, e.g. by calling "
                          "pixel_to_world on the pixel where you see them." if self.depth_tool else
                      "Object positions are not given to you and there is no depth sensor: the cameras give colour "
                      "images only. Localise objects from the images."]
        lines += ["", self._protocol_text(), "", self._feedback_text()]
        if self.memory != "none":
            lines += ["", "At the start of every attempt after the first, before your first tool call, write one to "
                          "three sentences assessing the earlier attempt(s): what happened and what you will keep or "
                          "change."]
        lines += ["", "Finish by calling done(reason) when you believe the task is complete. Be precise: small errors "
                      "in position or speed matter."]
        return "\n".join(lines)

    def _protocol_text(self):
        if self.memory == "none":
            if self.protocol == "within":
                return "Work on the task until you believe it is complete, then call done."
            return "You have one attempt at the task; it ends when you call done."
        if self.protocol == "within":
            if self.feedback == "F0":
                return (f"Everything happens in one continuous episode. After you call done you may be asked to "
                        f"continue from the current state; you may call done at most {self.k} times.")
            return (f"Everything happens in one continuous episode. When you call done, the task is checked; if it is "
                    f"not complete you are told so and continue from the current state. You may call done at most "
                    f"{self.k} times.")
        return (f"You get up to {self.k} attempts at the same task. Before every attempt the scene is reset to exactly "
                f"the same initial state, with the same objects and the same physical properties. Each attempt ends "
                f"when you call done; you will then see a summary of the attempt (key images, the actions you "
                f"executed{'' if self.feedback == 'F0' else ' and outcome feedback'}).")

    def _feedback_text(self):
        if self.memory == "none":
            return ""
        if self.feedback == "F0":
            return ("You will not be told whether an attempt succeeded; judge it from the images. "
                    + ("Attempts continue until the budget is used up." if self.protocol == "cross" else ""))
        if self.feedback == "F1":
            return "After each attempt you are told only whether it succeeded."
        return "After each attempt you are told whether it succeeded and, if not, a measurement of what went wrong."

    # ----------------------------------------------------------------------------- per attempt
    def history_parts(self, instruction, history):
        """First user item of an attempt (not pruned): task + packed earlier attempts. The header carries no
        count so that the item grows append-only from one attempt to the next (prompt-cache friendly)."""
        parts = [_text("TASK: " + instruction)]
        if history:
            parts.append(_text("\n=== Your earlier attempts ==="))
            for i, rec in enumerate(history, 1):
                parts += self.history_block(rec, i)
            parts.append(_text("=== End of earlier attempts ==="))
        return parts

    def current_parts(self, attempt, obs_images, proprio, continuing=False):
        """Second user item of an attempt (live): attempt counter + current observation."""
        parts = []
        if self.memory != "none":
            if self.protocol == "within" and continuing:
                if self.feedback == "F0":
                    parts.append(_text(f"You called done. Continue from the current state (this is segment "
                                       f"{attempt}; you may call done at most {self.k} times in total)."))
                else:
                    parts.append(_text(f"You called done, but the task is not complete. Continue from the current "
                                       f"state (this is segment {attempt}; you may call done at most {self.k} times "
                                       f"in total)."))
            elif self.protocol == "cross":
                parts.append(_text(f"This is attempt {attempt} of at most {self.k}. The scene has been reset."))
        parts += self.observation_parts("Current observation", obs_images, proprio)
        return parts

    def history_block(self, rec, idx):
        head = f"\n--- Attempt {idx} ---" if self.protocol == "cross" else f"\n--- Segment {idx} (until done call {idx}) ---"
        parts = [_text(head + "\nActions executed:\n" + summarize_actions(rec.steps))]
        fb = (rec.feedback or "").strip()
        if self.feedback != "F0" and fb:
            parts.append(_text("Outcome feedback: " + fb))
        if rec.keyframes:
            parts.append(_text("Key images (" + ", ".join(c for c, _ in rec.keyframes) + "):"))
            for cap, ref in rec.keyframes:
                parts.append(_text(f"[{cap}]"))
                parts.append(_img(ref))
        return parts

    def observation_parts(self, title, images, proprio):
        parts = [_text(f"\n{title}:")]
        for ref in images:
            parts.append(_text(f"{ref.camera} camera:"))
            parts.append(_img(ref))
        parts.append(_text("Proprioception: " + json.dumps(proprio)))
        return parts

    def step_observation(self, step, tool_name, images, proprio):
        return self.observation_parts(f"Observation after step {step} ({tool_name})", images, proprio)

    @staticmethod
    def nudge():
        return [_text("Please continue by calling exactly one tool (call done if you are finished).")]


def pick_keyframes(steps, initial, final, n=4):
    """<= n agentview keyframes: start, up to n-2 informative intermediate frames, end (after settling)."""
    out = [("start of attempt", initial)] if initial is not None and n >= 2 else []
    budget = n - len(out) - (1 if final is not None else 0)
    cands = [s for s in steps if s.frame is not None]
    if budget > 0 and cands:
        events = [s for s in cands if s.name in ("push", "close_gripper", "open_gripper")
                  or s.report.get("touched_object") or s.report.get("touched_objects")]
        pool = events if len(events) >= budget else cands
        if len(pool) > budget:
            idx = [round(i * (len(pool) - 1) / max(budget - 1, 1)) for i in range(budget)] if budget > 1 \
                else [len(pool) - 1]
            pool = [pool[i] for i in sorted(set(idx))]
        for s in sorted(pool, key=lambda s: s.step)[:budget]:
            out.append((f"after step {s.step} ({s.name})", s.frame))
    if final is not None:
        out.append(("end of attempt, after objects settled", final))
    return out[:n]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def deep_strip_images(obj):
    """JSON-friendly copy of conversation items (images -> paths)."""
    obj = copy.copy(obj)
    if isinstance(obj, dict):
        return {k: (v.path if k == "image" else deep_strip_images(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_strip_images(v) for v in obj]
    return obj
