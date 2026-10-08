"""L1-C yaw_bias: put a long bar into a narrow holder, with an arm whose wrist rotation is miscalibrated.

Category system_identification (level L1, action fine-tuning): the method (top-down pick across the bar, carry,
line the bar up with the holder, release just above the holder) and the scene understanding are right; only the
robot's own wrist rotation is off.

Hidden variable: a fixed wrist-yaw calibration offset beta, |beta| ~ U[15, 30] deg with a random sign: every
commanded gripper yaw theta is executed as theta + beta. The robot reports its yaw in its own (miscalibrated) frame,
so the agent's proprioception says "yaw = theta" (implemented by shifting the robot's notion of yaw 0, `Skills.R0`,
by beta: the agent harness computes the reported yaw from R0, so beta never shows up in a reading). At reset the
wrist is turned to the robot's own yaw 0 (true yaw beta), so the first reading is 0.

Why the prior fails: the bar is 14 cm long and 1.8 cm wide; the holder's opening is 3.2 cm wide (16 cm long).
Commanded to grasp square across the bar, the jaws actually come down rotated by beta; closing turns the bar until it
lies square in the jaws, so the bar ends up rotated by beta relative to where the robot thinks it is. Lined up
"exactly" with the holder, the bar is then beta off the holder's axis; its footprint across the opening
(14 sin beta + 1.8 cos beta >= 5.4 cm for beta >= 15 deg) is wider than the holder (4.4 cm outside), so it comes
down across the walls and stays on top of the holder at an angle -- clearly visible from both cameras. The
alignment tolerance is about +-5 deg (geometry; measured: see the task card).

What has to be learned: F2 reports by how many degrees, and in which sense, the bar's long axis was rotated from the
holder's long axis when the gripper opened (and how far the jaws were turned from square to the bar when they
closed). That angle is beta itself, so after one failure beta can be estimated and subtracted from every commanded
yaw.
"""
import re

import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import bin_xml, box_xml, register_generated
from ..skills import Skills, _rotz
from ..task_base import LFFTask, register_task

BAR, HOLDER = "bar_1", "holder_1"
DISTRACTORS = ("cream_cheese_1", "butter_1")
FREE_OBJECTS = (BAR,) + DISTRACTORS
NAMES = {BAR: "blue bar", HOLDER: "holder", "cream_cheese_1": "cream cheese", "butter_1": "butter"}
DESCR = {BAR: "long blue stick", HOLDER: "narrow grey trough", "cream_cheese_1": "flat blue box",
         "butter_1": "small orange box"}

BAR_HALF = (0.070, 0.009, 0.009)  # 14 x 1.8 x 1.8 cm, long side along the bar's x axis
HOLDER_IN = (0.080, 0.016)  # inner half extents of the holder's opening: 16 x 3.2 cm
WALL_T, FLOOR_T, WALL_H = 0.006, 0.006, 0.030
HOLDER_HALF_H = (WALL_H + FLOOR_T) / 2  # body origin at mid height
RIM_Z = envs.TABLE_Z + FLOOR_T + WALL_H
MASS = {BAR: 0.070, "cream_cheese_1": 0.227, "butter_1": 0.250}
# Bar friction 0.5 with geom priority, so it is used for all bar contacts (also against the finger pads, mu 2). With
# the pads' mu 2 the jaws pinch a misaligned bar at their corners and it stays skewed in the hand (measured: turned
# 3-13 deg of beta 15-25 deg while closing); with 0.5 the closing jaws square it up (turned = beta, measured).
BAR_MU = 0.5

SITE_ABOVE_TABLE = 0.0105  # grasp point 1.05 cm above the table: fingertips ~1 mm above it
Z_CARRY = envs.TABLE_Z + 0.15
RELEASE_GAP = 0.012  # bar bottom this far above the holder's rim at release
BETA_MIN, BETA_MAX = np.deg2rad(15), np.deg2rad(30)
AXIS_MAX = np.deg2rad(45)  # bar / holder long axes within +-45 deg of the forward (x) direction
HOME_XY = np.array([-0.21, 0.0])
REHOME_STEPS = 40


def wrap_half(a):
    """angle modulo pi into [-pi/2, pi/2) (the bar, the holder and the jaws are 180-deg symmetric)"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def _yaw_of_xmat(xmat):
    return float(np.arctan2(xmat[1, 0], xmat[0, 0]))


def _seg_dist(p, a, b):
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(a + u * ab - p))


def _q(axis, a):
    h = a / 2
    return {"z": np.array([np.cos(h), 0, 0, np.sin(h)])}[axis]


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def dir_words(dx, dy):
    """Robot-frame description: +x = forward (away from the robot), +y = the robot's left."""
    parts = []
    if abs(dy) >= 0.0005:
        parts.append(f"{100 * abs(dy):.1f} cm to the robot's {'left (+y)' if dy > 0 else 'right (-y)'}")
    if abs(dx) >= 0.0005:
        parts.append(f"{100 * abs(dx):.1f} cm {'forward (+x, away from the robot)' if dx > 0 else 'backward (-x, toward the robot)'}")
    return " and ".join(parts) if parts else "exactly at the position"


def rot_words(a):
    """Sense of a rotation about the vertical axis. Positive = counter-clockwise seen from above = from forward (+x)
    towards the robot's left (+y) = increasing yaw."""
    d = np.degrees(a)
    if abs(d) < 0.5:
        return "not rotated (within 0.5 deg)"
    if d > 0:
        return f"rotated {d:.0f} deg counter-clockwise seen from above (from forward towards the robot's left, +yaw)"
    return f"rotated {-d:.0f} deg clockwise seen from above (from forward towards the robot's right, -yaw)"


class YawSkills(Skills):
    """Skills with a miscalibrated wrist: commanded yaw theta is executed as theta + yaw_bias. Implemented by shifting
    the robot's notion of yaw 0 (`R0`, which Skills uses to build every orientation target and the agent harness uses
    to report the yaw) by yaw_bias; `R0_true` is the real top-down orientation at yaw 0. Also logs what the outcome
    measurement needs, whoever drives the skills (scripted policy or agent)."""

    def __init__(self, env, yaw_bias=0.0, recorder=None):
        super().__init__(env, bias=None, recorder=recorder)
        self.yaw_bias = float(yaw_bias)
        self.R0_true = self.R0.copy()
        self.R0 = _rotz(self.yaw_bias) @ self.R0_true
        self.events = []
        m = env.sim.model
        self._finger_geoms = {
            side: [m.geom_name2id(g) for g in env.robots[0].gripper.important_geoms[side]]
            for side in ("left_finger", "right_finger")}
        self._grip_geoms = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith("gripper0_")]
        self._obj_geoms = {n: envs.obj_geom_ids(env, n) for n in FREE_OBJECTS}
        self.bar_z0 = float(envs.obj_pos(env, BAR)[2])
        self.bar_zmax = self.bar_z0
        self.bar_ref_yaw = self.bar_yaw()
        self.bar_ref = envs.obj_pos(env, BAR)  # bar pose before the fingers last touched it

    # ------------------------------------------------------------------ state
    def true_yaw(self):
        return _yaw_of_xmat(self.eef_mat() @ self.R0_true.T)

    def reported_yaw(self):
        """Yaw as the miscalibrated robot believes it to be. Anything shown to the agent must use this."""
        return _yaw_of_xmat(self.eef_mat() @ self.R0.T)

    def bar_yaw(self):
        return _yaw_of_xmat(self.env.sim.data.body_xmat[envs.body_id(self.env, BAR)].reshape(3, 3))

    def touching(self):
        return [n for n, gs in self._obj_geoms.items() if envs.contacts_between(self.env, self._grip_geoms, gs)]

    def holding_bar(self):
        bg = self._obj_geoms[BAR]
        return all(envs.contacts_between(self.env, gs, bg) for gs in self._finger_geoms.values())

    def holding(self, obj_name):
        """The agent harness reports `holding_object` through this; same finger-contact test as the outcome."""
        if obj_name == BAR:
            return self.holding_bar()
        return super().holding(obj_name)

    def wrap_yaw(self, yaw):
        """Like Skills.wrap_yaw (the gripper is symmetric under pi), but the window |yaw| <= YAW_LIMIT is applied
        to the TRUE wrist yaw (commanded + yaw_bias): the joint-7 limit is a property of the real wrist. Without
        this, a commanded -113 deg with yaw_bias -30 deg drives joint 7 into its limit (measured: stops at true
        -137 deg, reported -107 deg)."""
        cands = [yaw + k * np.pi for k in range(-3, 4)]
        ok = [c for c in cands if abs(c + self.yaw_bias) <= self.YAW_LIMIT] or cands
        return float(min(ok, key=lambda c: abs(c - self.yaw)))

    def move_to(self, target, *args, **kwargs):
        rep = super().move_to(target, *args, **kwargs)
        rep["yaw_reported_deg"] = round(float(np.degrees(self.reported_yaw())), 1)
        return rep

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.bar_zmax = max(self.bar_zmax, float(envs.obj_pos(self.env, BAR)[2]))
        if self.grip < 0 and not envs.contacts_between(self.env, self._grip_geoms, self._obj_geoms[BAR]):
            self.bar_ref_yaw = self.bar_yaw()
            self.bar_ref = envs.obj_pos(self.env, BAR)

    def set_gripper(self, close, steps=15):
        if close and self.grip < 0:
            self.events.append(dict(kind="close", eef=self.eef_pos().tolist(), true_yaw=self.true_yaw(),
                                    bar=envs.obj_pos(self.env, BAR).tolist(), bar_ref_yaw=self.bar_ref_yaw,
                                    bar_ref=np.asarray(self.bar_ref).tolist(), touching=self.touching(),
                                    step=self.n_steps))
        elif not close and self.grip > 0 and self.holding_bar():
            self.events.append(dict(kind="release", eef=self.eef_pos().tolist(), true_yaw=self.true_yaw(),
                                    bar=envs.obj_pos(self.env, BAR).tolist(), bar_yaw=self.bar_yaw(),
                                    step=self.n_steps))
        rep = super().set_gripper(close, steps)
        if close and self.events and self.events[-1]["kind"] == "close":
            self.events[-1]["bar_yaw_after"] = self.bar_yaw()
        return rep


@register_task
class YawBias(LFFTask):
    name = "l1_yaw_bias"
    level = "L1"
    category = "system_identification"
    capabilities = ("Perceive", "Utilize")
    protocol = "cross"
    instruction = "Pick up the blue bar and put it into the holder."
    instruction_indirect = "Pick up the long blue stick and drop it into the narrow grey trough so that it lies inside."
    pick_lift_min = 0.03

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        bar = register_generated("LffYbBar", box_xml("lff_yb_bar", BAR_HALF, (0.15, 0.35, 0.85, 1.0), density=1500))
        holder = register_generated("LffYbHolder", bin_xml("lff_yb_holder", HOLDER_IN, WALL_H, wall_t=WALL_T,
                                                            floor_t=FLOOR_T, rgba=(0.55, 0.56, 0.60, 1.0),
                                                            density=2000), free=False)
        objs = [(BAR, bar, "bar_region", (-0.21, -0.26, -0.19, -0.24)),
                ("cream_cheese_1", "cream_cheese", "cheese_region", (-0.21, 0.24, -0.19, 0.26)),
                ("butter_1", "butter", "butter_region", (0.15, -0.26, 0.17, -0.24))]
        fx = [(HOLDER, holder, "holder_region", (0.09, -0.01, 0.11, 0.01))]
        return write_bddl(self.name, self.instruction, objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        self._base_quat = {n: envs.obj_quat(self.env, n) for n in FREE_OBJECTS}
        m = self.env.sim.model
        self._holder_bid = m.body_name2id(self.env.fixtures_dict[HOLDER].root_body)

    def _place(self, name, xy, yaw, z0=envs.TABLE_Z):
        q = _qmul(_q("z", yaw), self._base_quat[name])
        envs.set_obj_pose(self.env, name, [xy[0], xy[1], z0 + 0.3], quat_wxyz=q)
        self.env.sim.forward()
        qa, _ = envs.free_joint_addr(self.env, name)
        self.env.sim.data.qpos[qa + 2] -= envs.obj_min_z(self.env, name) - (z0 + 0.0005)
        self.env.sim.forward()

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        while True:
            b = np.array([rng.uniform(-0.16, 0.04), rng.uniform(-0.20, 0.20)])
            h = np.array([rng.uniform(-0.14, 0.06), rng.uniform(-0.22, 0.22)])
            if np.linalg.norm(h - b) < 0.22:
                continue
            placed = [b, h]
            dis = {}
            for name in DISTRACTORS:
                for _ in range(300):
                    p = np.array([rng.uniform(-0.22, 0.12), rng.uniform(-0.27, 0.27)])
                    if any(np.linalg.norm(p - o) < 0.15 for o in placed):
                        continue
                    if any(_seg_dist(p, a, c) < 0.10 for a, c in ((b, h), (HOME_XY, b))):
                        continue
                    break
                else:
                    break
                placed.append(p)
                dis[name] = dict(xy=p.round(5).tolist(), yaw=float(rng.uniform(-np.pi, np.pi)))
            if len(dis) == len(DISTRACTORS):
                break
        bar_yaw = float(rng.uniform(-AXIS_MAX, AXIS_MAX))
        holder_yaw = float(rng.uniform(-AXIS_MAX, AXIS_MAX))
        beta = float(rng.choice([-1.0, 1.0]) * rng.uniform(BETA_MIN, BETA_MAX))
        return dict(seed=int(seed), bar_xy=b.round(5).tolist(), bar_yaw=bar_yaw, holder_xy=h.round(5).tolist(),
                    holder_yaw=holder_yaw, distractors=dis, yaw_bias=beta, yaw_bias_deg=float(np.degrees(beta)))

    def apply_instance(self, inst):
        m = self.env.sim.model
        m.body_pos[self._holder_bid] = [inst["holder_xy"][0], inst["holder_xy"][1], envs.TABLE_Z + HOLDER_HALF_H]
        m.body_quat[self._holder_bid] = _q("z", inst["holder_yaw"])
        for name, mass in MASS.items():
            envs.scale_mass(self.env, name, mass)
        self.env.sim.forward()
        self._place(BAR, inst["bar_xy"], inst["bar_yaw"])
        envs.set_friction(self.env, BAR, BAR_MU, priority=3)
        for name, v in inst["distractors"].items():
            self._place(name, v["xy"], v["yaw"])

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = YawSkills(self.env, yaw_bias=inst["yaw_bias"], recorder=recorder)
        self.sk.hold(REHOME_STEPS)  # turn the wrist to the robot's own yaw 0 (true yaw = beta)
        self.sk.n_steps = 0
        self.sk.events = []
        self.start_pos = {n: envs.obj_pos(self.env, n) for n in FREE_OBJECTS}
        return self.sk

    # ------------------------------------------------------------ outcome
    def holder_frame(self, xy):
        """xy (world) -> (along, across) in the holder frame, relative to the holder centre"""
        c = self.env.sim.model.body_pos[self._holder_bid][:2].copy()
        a = self.holder_yaw()
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        return R @ (np.asarray(xy[:2]) - c)

    def holder_yaw(self):
        q = self.env.sim.model.body_quat[self._holder_bid]
        return float(2 * np.arctan2(q[3], q[0]))

    def in_holder(self):
        p = envs.obj_pos(self.env, BAR)
        u, v = self.holder_frame(p)
        return bool(abs(u) < HOLDER_IN[0] and abs(v) < HOLDER_IN[1] and p[2] < RIM_Z
                    and envs.obj_upright_cos(self.env, BAR) > 0.95)

    def outcome(self, inst):
        envs.settle(self.env, 40)
        sk = self.sk
        p = envs.obj_pos(self.env, BAR)
        held = sk.holding_bar()
        success = bool(self.in_holder() and not held)
        picked = sk.bar_zmax > sk.bar_z0 + self.pick_lift_min
        closes = [e for e in sk.events if e["kind"] == "close"]
        last_close = closes[-1] if closes else None
        releases = [e for e in sk.events if e["kind"] == "release" and last_close and e["step"] >= last_close["step"]]
        hy = self.holder_yaw()
        end_mis = wrap_half(sk.bar_yaw() - hy)
        hc = self.env.sim.model.body_pos[self._holder_bid][:2]
        out = dict(success=success, picked=bool(picked), held_at_end=bool(held), bar_end=p.round(4).tolist(),
                   bar_end_misalign_deg=round(float(np.degrees(end_mis)), 1), grasp_misalign=None,
                   bar_turned_in_grasp=None, release_misalign=None, release_offset=None, touched_at_close=None)
        grasp_txt = ""
        if last_close is not None:
            gm = wrap_half(last_close["true_yaw"] - last_close["bar_ref_yaw"])
            turned = wrap_half(last_close.get("bar_yaw_after", last_close["bar_ref_yaw"]) - last_close["bar_ref_yaw"])
            out["grasp_misalign"] = round(gm, 4)
            out["bar_turned_in_grasp"] = round(turned, 4)
            out["touched_at_close"] = last_close["touching"]
            go = np.array(last_close["eef"][:2]) - np.array(last_close.get("bar_ref", last_close["bar"])[:2])
            a = last_close["bar_ref_yaw"]
            along, across = go @ [np.cos(a), np.sin(a)], go @ [-np.sin(a), np.cos(a)]
            over_bar = bool(abs(along) <= BAR_HALF[0] and abs(across) <= BAR_HALF[1] + 0.02)
            out["grasp_xy_offset"] = go.round(4).tolist()
            out["grasp_over_bar"] = over_bar
            if over_bar:
                grasp_txt = f"When the jaws closed on the blue bar, they were {rot_words(gm)} from square across the bar"
                if np.linalg.norm(go) >= 0.015:
                    grasp_txt += f" (the grasp point was {dir_words(*go)} of the bar's centre)"
            else:
                grasp_txt = (f"When the jaws closed, the grasp point was {dir_words(*go)} of the blue bar's centre, i.e. "
                             f"not over the bar, and the jaws were {rot_words(gm)} from square across the bar")
            if abs(np.degrees(turned)) >= 2:
                grasp_txt += f", and the closing fingers turned the bar by {abs(np.degrees(turned)):.0f} deg"
        if releases:
            r = releases[-1]
            rm = wrap_half(r["bar_yaw"] - hy)
            out["release_misalign"] = round(rm, 4)
            out["release_offset"] = (np.array(r["bar"][:2]) - hc).round(4).tolist()
        if success:
            out["failure"] = None
            detail = "The blue bar is inside the holder."
            if out["release_misalign"] is not None:
                detail += (f" When the gripper opened, the bar's long axis was {rot_words(out['release_misalign'])} "
                           f"from the holder's long axis.")
        elif not picked:
            out["failure"] = "missed_grasp"
            if last_close is None:
                detail = "The gripper never closed; the blue bar was not picked up."
            else:
                detail = f"{grasp_txt}, but the bar was not picked up. It is still on the table."
        elif held:
            out["failure"] = "not_released"
            detail = f"{grasp_txt}. The bar was picked up but it is still in the gripper."
        elif releases:
            out["failure"] = "missed_place"
            if p[2] > RIM_Z - 0.002:
                where = (f"The bar came down across the holder's walls instead of dropping in; it is now lying on top "
                         f"of the holder, its long axis {rot_words(end_mis)} from the holder's long axis.")
            else:
                where = (f"The bar did not end up inside the holder (its centre is now "
                         f"{dir_words(*(p[:2] - hc))} of the holder's centre).")
            detail = (f"{grasp_txt}. When the gripper opened above the holder, the bar's long axis was "
                      f"{rot_words(out['release_misalign'])} from the holder's long axis, and the bar's centre was "
                      f"{dir_words(*out['release_offset'])} of the holder's centre. {where}")
        else:
            out["failure"] = "dropped"
            detail = (f"{grasp_txt}. The bar was lifted but slipped out of the fingers before the gripper opened; it is "
                      f"now {dir_words(*(p[:2] - hc))} of the holder's centre.")
        out["detail"] = detail
        ind = detail
        for k in sorted(NAMES, key=lambda n: -len(NAMES[n])):
            ind = ind.replace(NAMES[k], DESCR[k])
        ind = re.sub(r"\bbar\b", "stick", ind)  # the bare noun would give the direct name away
        out["detail_indirect"] = ind
        return out

    def feedback(self, inst, out, level="F2", indirect=False):
        if indirect and level == "F2":
            out = dict(out, detail=out.get("detail_indirect", out["detail"]))
        return super().feedback(inst, out, level)

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def execute(self, inst, params):
        """Top-down pick square across the bar at its centre, carry, line the bar up with the holder and release it
        just above the holder's rim. params['dyaw'] (rad) is the policy's own yaw correction, added to every
        commanded yaw."""
        sk = self.sk
        dy = float(params.get("dyaw", 0.0))
        b = envs.obj_pos(self.env, BAR)  # perfect perception of the current scene
        psi = sk.bar_yaw()
        hc = self.env.sim.model.body_pos[self._holder_bid][:2].copy()
        phi = self.holder_yaw()
        z_grasp = envs.TABLE_Z + SITE_ABOVE_TABLE
        z_rel = RIM_Z + RELEASE_GAP + SITE_ABOVE_TABLE
        sk.set_gripper(False, steps=8)
        sk.move_to([b[0], b[1], Z_CARRY], yaw=wrap_half(psi) + dy)
        sk.move_to([b[0], b[1], z_grasp + 0.06], tol=0.002)
        sk.move_to([b[0], b[1], z_grasp], tol=0.003, max_steps=80)
        sk.set_gripper(True, steps=15)
        sk.move_to([b[0], b[1], Z_CARRY], speed=0.3)
        if not sk.holding_bar():  # missed: a real controller sees the jaws closed on nothing and stops
            return
        sk.move_to([hc[0], hc[1], Z_CARRY], yaw=wrap_half(phi) + dy, tol=0.003)
        sk.move_to([hc[0], hc[1], z_rel], tol=0.003, max_steps=100)
        sk.hold(5)
        sk.set_gripper(False, steps=12)
        sk.move_to([hc[0], hc[1], Z_CARRY], speed=0.3)

    def default_params(self, inst):
        return dict(dyaw=0.0)

    def oracle_params(self, inst):
        return dict(dyaw=-float(inst["yaw_bias"]))

    def adapt_params(self, inst, history):
        """Scripted F2 learner: the reported angle between the bar and the holder when the gripper opened is the
        part of the wrist offset not yet corrected; accumulate it into the correction subtracted from every
        commanded yaw. If the bar was never released, use the jaw angle at the grasp instead."""
        last = history[-1]
        dy = float(last["params"]["dyaw"])
        o = last["outcome"]
        if o.get("release_misalign") is not None:
            dy -= o["release_misalign"]
        elif o.get("grasp_misalign") is not None:
            dy -= o["grasp_misalign"]
        return dict(dyaw=round(dy, 4))

    blind_sigma = np.deg2rad(4.0)

    def blind_params(self, inst, attempt, rng):
        """Retry the same plan with a small random perturbation of the commanded yaws; no use of history."""
        return dict(dyaw=round(float(rng.normal(0.0, self.blind_sigma)), 4))
