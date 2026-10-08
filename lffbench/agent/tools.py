"""Agent-facing tool interface on top of `Skills`.

Every action tool calls the task's `Skills` object, so the task's hidden calibration `bias` keeps
acting on every commanded position. Proprioception is reported in the robot's own (possibly
miscalibrated) frame: reported eef = true eef - bias, i.e. the robot "believes" it reached the
commanded point; only the cameras show the truth. Camera-derived quantities (pixel_to_world) and
the privileged `locate` tool are in the true world frame.

Tool schemas are plain JSON schema (OpenAI strict-mode compatible: every property listed in
`required`, optional ones nullable, no additional properties).
"""
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .. import envs
from . import camera as C
from .images import ImageRef

TABLE_Z = envs.TABLE_Z
# Workspace limits for commanded grasp-point positions (world frame, metres). Targets outside are
# clipped and the report says so.
WS_LO = np.array([-0.55, -0.45, TABLE_Z + 0.005])
WS_HI = np.array([0.35, 0.45, TABLE_Z + 0.50])
SPEED_RANGE = (0.02, 1.0)
PUSH_DIST_RANGE = (0.0, 0.40)
RUNUP_RANGE = (0.02, 0.30)

ACTION_TOOLS = ("move_to", "open_gripper", "close_gripper", "push")


def _num(desc):
    return {"type": "number", "description": desc}


def _opt_num(desc):
    return {"type": ["number", "null"], "description": desc}


def _fn(name, desc, props=None):
    props = props or {}
    return dict(name=name, description=desc,
                parameters={"type": "object", "properties": props, "required": list(props),
                            "additionalProperties": False})


def tool_specs(privileged_locate=False, object_names=None, image_res=512):
    """Backend-neutral tool list: [{name, description, parameters}]."""
    specs = [
        _fn("move_to",
            "Move the gripper's grasp point (midway between the fingertips) in a straight line to the world "
            "position (x, y, z) with the gripper pointing straight down and rotated to `yaw`. The gripper "
            "opening is not changed. Returns whether the target was reached, the end-effector position, gripper "
            "width, whether an object is held and whether the arm touched an object during the motion.",
            {"x": _num("target x, metres (world frame; +x away from the robot)"),
             "y": _num("target y, metres (world frame; +y to the robot's left)"),
             "z": _num(f"target z, metres (table top is z = {TABLE_Z:.2f})"),
             "yaw": _num("gripper rotation about the vertical axis in DEGREES; 0 = fingers close along the world "
                         "y axis, 90 = fingers close along the world x axis. Taken modulo 180 (the two-finger "
                         "gripper is symmetric): the controller uses the equivalent angle closest to the current "
                         "yaw within +-113 deg; the report gives the yaw actually used"),
             "speed": _opt_num("straight-line speed in m/s (0.02-1.0); null = as fast as the controller allows "
                               "(about 0.6 m/s)")}),
        _fn("open_gripper", "Open the gripper fully (the arm holds its position)."),
        _fn("close_gripper", "Close the gripper (the arm holds its position). Closing on an object grasps it."),
        _fn("push",
            "Straight-line push with the closed gripper. The gripper closes, moves above the point `runup` metres "
            "behind (x, y) (opposite to the push direction), descends to height z, sweeps along (dir_x, dir_y) "
            "through (x, y) and stops `distance` metres past (x, y), moving at `speed` m/s, then lifts 12 cm. The "
            "gripper is rotated automatically to a fixed orientation relative to the push direction.",
            {"x": _num("x of the point where the push should start acting (e.g. the near face of the object), m"),
             "y": _num("y of that point, m"),
             "dir_x": _num("x component of the push direction (normalised internally)"),
             "dir_y": _num("y component of the push direction"),
             "distance": _num("how far past (x, y) the gripper travels, metres (0-0.40)"),
             "speed": _num("sweep speed, m/s (0.02-1.0; the controller saturates near 0.6)"),
             "z": _opt_num(f"height of the grasp point during the sweep, m; null = {TABLE_Z + 0.02:.2f} "
                           "(2 cm above the table)"),
             "runup": _opt_num("distance behind (x, y) where the sweep starts, m (0.02-0.30); null = 0.06. A longer "
                               "run-up lets the gripper reach the commanded speed before contact.")}),
        _fn("get_observation",
            "Return the current camera images (agentview and wrist) and the robot's proprioceptive state."),
        _fn("pixel_to_world",
            f"Back-project an image pixel to a 3-D world point using the camera's depth map. (u, v) are pixel "
            f"coordinates in the {image_res}x{image_res} image exactly as you received it: u = column from the "
            f"left edge, v = row from the top edge. Returns the world point (x, y, z) of the surface seen at that "
            f"pixel and its height above the table.",
            {"camera": {"type": "string", "enum": ["agentview", "wrist"], "description": "which image"},
             "u": _num("column (0 = left edge)"), "v": _num("row (0 = top edge)")}),
        _fn("done",
            "Finish the current attempt. Call this when you believe the task is complete (or you want to stop). "
            "Give a short reason: what you did and whether you think it worked.",
            {"reason": {"type": "string", "description": "short explanation"}}),
    ]
    if privileged_locate:
        names = list(object_names or [])
        prop = {"type": "string", "description": "object name"}
        if names:
            prop["enum"] = names
        specs.insert(-1, _fn("locate",
                             "PRIVILEGED (ablation): return the true world position of an object's origin and its "
                             "axis-aligned bounding box.", {"object_name": prop}))
    return specs


@dataclass
class ToolResult:
    name: str
    args: dict
    ok: bool
    report: dict
    images: list = field(default_factory=list)
    ended: bool = False
    wall_s: float = 0.0
    sim_steps: int = 0

    def text(self):
        return json.dumps(self.report, separators=(", ", ": "), default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def _r(v, nd=4):
    return [round(float(x), nd) for x in v]


class ContactTracker:
    """Collects the objects the robot touches. Contacts are checked after every MuJoCo substep (a control
    step is 25 substeps; a fast push touches the object for less than one control step, so checking only at
    control-step boundaries misses it -- measured). Implemented by wrapping `env.sim.step` once per env; the
    wrapper only reads contacts and dispatches to the currently active tracker."""

    def __init__(self, env):
        self.env = env
        m = env.sim.model
        self.is_robot = np.zeros(m.ngeom, bool)
        self.is_robot[envs.robot_geom_ids(env)] = True
        self.names = []
        self.obj_of = np.full(m.ngeom, -1, int)
        for name in list(env.objects_dict) + list(getattr(env, "fixtures_dict", {}) or {}):
            try:
                ids = envs.obj_geom_ids(env, name)
            except Exception:  # fixtures without contact geoms
                continue
            self.obj_of[ids] = len(self.names)
            self.names.append(name)
        self.touched = set()
        sim = env.sim
        if not getattr(sim, "_lff_step_wrapped", False):
            orig = sim.step

            def step(*a, **k):
                orig(*a, **k)
                t = getattr(sim, "_lff_tracker", None)
                if t is not None:
                    t.check()
            sim.step = step
            sim._lff_step_wrapped = True
        sim._lff_tracker = self

    def reset(self):
        self.touched = set()

    def check(self):
        d = self.env.sim.data
        n = d.ncon
        if not n:
            return
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        hit = np.concatenate([self.obj_of[g2[self.is_robot[g1]]], self.obj_of[g1[self.is_robot[g2]]]])
        for i in set(hit[hit >= 0].tolist()):
            self.touched.add(self.names[i])


def parse_args(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    raw = raw.strip()
    return json.loads(raw) if raw else {}


class ToolExecutor:
    """Executes agent tool calls on a task whose instance has just been reset (task.sk is live)."""

    def __init__(self, task, inst, image_res=512, cams=("agentview", "wrist"), privileged_locate=False,
                 grasp_report="bool", jpeg_quality=90, image_sink=None):
        self.task = task
        self.inst = inst
        self.env = task.env
        self.sk = task.sk  # keep our own reference: the scripted backend may swap task.sk for a proxy
        self.image_res = int(image_res)
        self.cams = tuple(cams)
        self.privileged_locate = privileged_locate
        self.grasp_report = grasp_report  # "bool" | "name" | "none"
        self.jpeg_quality = jpeg_quality
        self.image_sink = image_sink  # callable(ImageRef) -> path (saves the image)
        self.objects = list(self.env.objects_dict)
        self.tracker = ContactTracker(self.env)  # task recorders / hooks on Skills are left untouched
        self._depth_cache = {}
        self.n_calls = 0

    # ------------------------------------------------------------------ state
    def eef_yaw_deg(self):
        R = self.sk.eef_mat() @ self.sk.R0.T
        return math.degrees(math.atan2(R[1, 0], R[0, 0]))

    def held(self):
        names = [o for o in self.objects if self.sk.holding(o)]
        return names

    def _holding_field(self, names):
        if self.grasp_report == "name":
            return names[0] if names else None
        if self.grasp_report == "none":
            return None
        return bool(names)

    def proprio(self):
        believed = self.sk.eef_pos() - self.sk.bias
        p = dict(eef_pos=_r(believed), eef_yaw_deg=round(self.eef_yaw_deg(), 1),
                 gripper="closed" if self.sk.grip > 0 else "open",
                 gripper_width_m=round(self.sk.gripper_width(), 4))
        if self.grasp_report != "none":
            p["holding_object"] = self._holding_field(self.held())
        return p

    def render(self, cam, res=None):
        res = res or self.image_res
        return envs.render(self.env, C.mj_cam(cam), res)

    def observe(self, label=""):
        imgs = []
        for cam in self.cams:
            ref = ImageRef(camera=cam, array=self.render(cam), label=f"{label} {cam}".strip())
            if self.image_sink is not None:
                self.image_sink(ref)
            imgs.append(ref)
        self._depth_cache = {}
        return imgs

    # ------------------------------------------------------------------ dispatch
    def call(self, name, raw_args):
        t0 = time.time()
        n0 = self.sk.n_steps
        self.n_calls += 1
        try:
            args = parse_args(raw_args)
        except Exception as e:
            return ToolResult(name, {"_raw": str(raw_args)}, False, dict(ok=False, error=f"arguments are not valid JSON: {e}"),
                              wall_s=time.time() - t0)
        fn = getattr(self, f"_t_{name}", None)
        if fn is None or (name == "locate" and not self.privileged_locate):
            return ToolResult(name, args, False, dict(ok=False, error=f"unknown tool {name!r}"), wall_s=time.time() - t0)
        try:
            res = fn(**args)
        except TypeError as e:
            res = ToolResult(name, args, False, dict(ok=False, error=f"bad arguments: {e}"))
        except ValueError as e:
            res = ToolResult(name, args, False, dict(ok=False, error=str(e)))
        res.args = args
        res.wall_s = time.time() - t0
        res.sim_steps = self.sk.n_steps - n0
        if res.sim_steps:
            self._depth_cache = {}
        hook = getattr(self.task, "on_agent_tool", None)
        if hook is not None:
            hook(self.inst, name, args, res.report)
        drain = getattr(self.task, "drain_events", None)  # task-side events (e.g. L6: an item slipped)
        if drain is not None:
            ev = drain()
            if ev:
                res.report["events"] = list(ev)
        return res

    # ------------------------------------------------------------------ helpers
    def _clip_xyz(self, xyz):
        xyz = np.asarray(xyz, float)
        if not np.all(np.isfinite(xyz)):
            raise ValueError("non-finite coordinates")
        c = np.clip(xyz, WS_LO, WS_HI)
        return c, (None if np.allclose(c, xyz) else _r(c))

    @staticmethod
    def _clip(v, lo_hi, what):
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"{what} must be finite")
        return float(np.clip(v, *lo_hi))

    def _action_report(self, name, extra=None):
        names = self.held()
        rep = dict(ok=True, tool=name)
        if extra:
            rep.update(extra)
        rep.update(self.proprio())
        if self.grasp_report == "name":
            rep["touched_objects"] = sorted(self.tracker.touched)
        elif self.grasp_report == "bool":
            rep["touched_object"] = bool(self.tracker.touched)
        rep["sim_time_s"] = round(self.sk.n_steps * 0.05, 2)
        return rep

    # ------------------------------------------------------------------ tools
    def _t_move_to(self, x, y, z, yaw, speed=None):
        tgt, clipped = self._clip_xyz([x, y, z])
        spd = None if speed is None else self._clip(speed, SPEED_RANGE, "speed")
        self.tracker.reset()
        rep = self.sk.move_to(tgt, yaw=math.radians(float(yaw)), speed=spd)  # Skills wraps yaw mod pi
        extra = dict(reached=bool(rep["reached"]), yaw_used_deg=round(math.degrees(self.sk.yaw), 1))
        if clipped:
            extra["target_clipped_to"] = clipped
        return ToolResult("move_to", {}, True, self._action_report("move_to", extra))

    def _t_open_gripper(self):
        self.tracker.reset()
        self.sk.set_gripper(False)
        return ToolResult("open_gripper", {}, True, self._action_report("open_gripper"))

    def _t_close_gripper(self):
        self.tracker.reset()
        self.sk.set_gripper(True)
        return ToolResult("close_gripper", {}, True, self._action_report("close_gripper"))

    def _t_push(self, x, y, dir_x, dir_y, distance, speed, z=None, runup=None):
        d = np.array([float(dir_x), float(dir_y)])
        if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-6:
            raise ValueError("push direction must be a non-zero vector")
        start, clipped = self._clip_xyz([x, y, TABLE_Z + 0.02 if z is None else z])
        dist = self._clip(distance, PUSH_DIST_RANGE, "distance")
        spd = self._clip(speed, SPEED_RANGE, "speed")
        pre = 0.06 if runup is None else self._clip(runup, RUNUP_RANGE, "runup")
        self.tracker.reset()
        rep = self.sk.push(start[:2], d, dist, spd, z=float(start[2]), pre=pre)
        extra = dict(sweep_reached=bool(rep["reached"]))
        if clipped:
            extra["start_clipped_to"] = clipped
        return ToolResult("push", {}, True, self._action_report("push", extra))

    def _t_get_observation(self):
        imgs = self.observe(label="requested")
        return ToolResult("get_observation", {}, True, dict(ok=True, images=[i.camera for i in imgs], **self.proprio()),
                          images=imgs)

    def _t_pixel_to_world(self, camera, u, v):
        cam = str(camera)
        if cam not in C.CAMERAS:
            raise ValueError(f"camera must be one of {sorted(C.CAMERAS)}")
        if cam not in self._depth_cache:
            self._depth_cache[cam] = C.render_depth(self.env, cam, self.image_res)
        r = C.pixel_to_world(self.env, cam, float(u), float(v), self.image_res, depth=self._depth_cache[cam])
        rep = dict(ok=True, camera=cam, u=float(u), v=float(v), world_xyz=_r(r["xyz"]),
                   height_above_table_m=round(float(r["xyz"][2] - TABLE_Z), 4), depth_m=round(r["depth"], 4))
        if not r["valid"]:
            rep["warning"] = "pixel shows the far background; the point is unreliable"
        return ToolResult("pixel_to_world", {}, True, rep)

    def _t_locate(self, object_name):
        if object_name not in self.env.objects_dict:
            raise ValueError(f"unknown object {object_name!r}; known: {self.objects}")
        lo, hi = self._aabb(object_name)
        p = envs.obj_pos(self.env, object_name)
        return ToolResult("locate", {}, True, dict(ok=True, object=object_name, position=_r(p), bbox_min=_r(lo),
                                                   bbox_max=_r(hi)))

    def _aabb(self, name):
        m, d = self.env.sim.model, self.env.sim.data
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in envs.obj_geom_ids(self.env, name):
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            h = np.abs(R) @ m.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
        return lo, hi

    def _t_done(self, reason=""):
        return ToolResult("done", {}, True, dict(ok=True, ended=True), ended=True)
