"""L3-B too_wide_push: a box that is slightly too wide for the gripper must be pushed into a square on the table.

Why the prior fails: the box looks like an ordinary graspable box (narrow side 8.3-8.9 cm, long side 11-13 cm,
4 cm high), but the Panda's fingers open to only 7.9 cm. A top-down grasp across the narrow side therefore puts
both fingers on top of the box; closing them grips nothing. Across the long side it is worse, and retrying the
same grasp with small variations fails the same way. A difference of 4-10 mm between the box and the maximum
opening is not readable from the camera images, and the agent is not told the gripper's opening.

What must be learned from the failure: this object cannot be picked up at all -> change the manipulation mode:
close the gripper and push the box (there is no push primitive: the agent closes the gripper, lowers it beside
the box and moves it along the table with move_to), keeping the box from turning, until it lies completely
inside the square.

Layout (randomised per instance): box size, position and yaw (long side roughly along x or along y, +-10 deg),
target square (18 x 18 cm inner, axis-aligned) 16-22 cm away along one world axis (forward, backward, left or
right) with a lateral offset of up to 3 cm.

Protocol: cross (reset to the same instance before every attempt), k = 5.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import box_xml, register_generated
from ..skills import DT, _rot_err
from ..task_base import LFFTask, register_task
from .l5_slide import square_frame_xml

BOX_HALF_MAX = (0.07, 0.05, 0.02)  # registered size (x = long, y = narrow, z); shrunk per instance
BOX_HALF_Z = 0.02  # 4 cm high
BOX_MASS = 0.40
BOX_MU = 0.6  # box sliding friction (priority over the table's 1.0 and the fingers')
GRIP_OPEN = 0.079  # measured distance between the finger pads when fully open
SQ_IN = 0.09  # target square: 18 cm inner side
SQ_LINE = 0.008
Z_GRASP = envs.TABLE_Z + 0.022  # site height of a normal top-down grasp of a 4 cm box (pads 1.7-3.4 cm above table)
Z_PUSH = envs.TABLE_Z + 0.02  # site height while pushing: finger tips ~1 cm above the table
PUSH_SPEED = 0.08
FRONT_GAP = 0.03  # where a push stroke starts: this far behind the box face


def _wrap_pi(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


DIRS = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}


@register_task
class TooWidePush(LFFTask):
    name = "l3_too_wide_push"
    level = "L3"
    category = "strategy_switching"
    capabilities = ("Perceive", "Reason", "Plan")
    protocol = "cross"
    instruction = "Put the brown box completely inside the green square marked on the table."
    instruction_indirect = "Put the cardboard-coloured package completely inside the green outlined area on the table."

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        self._mon = None
        self.inst = None
        orig_step = self.env.step

        def step(action):  # per-control-step event monitor (also sees agent-driven attempts)
            ret = orig_step(action)
            if self._mon is not None:
                self._monitor_step()
            return ret

        self.env.step = step

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        box = register_generated("LffWideBox", box_xml("lff_wide_box", BOX_HALF_MAX, (0.62, 0.45, 0.28, 1.0),
                                                       density=500))
        sq = register_generated("LffWideSquare", square_frame_xml("lff_wide_square", inner_half=SQ_IN, line_w=SQ_LINE),
                                free=False)
        objs = [("box_1", box, "box_region", (-0.11, -0.21, -0.09, -0.19))]
        fx = [("square_1", sq, "square_region", (-0.01, 0.19, 0.01, 0.21))]
        return write_bddl(self.name, self.instruction, objs, fx)

    def _box_geoms(self):
        if not hasattr(self, "_bg"):
            m = self.env.sim.model
            bid = envs.body_id(self.env, "box_1")
            self._bg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        return self._bg

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        while True:
            narrow = float(rng.uniform(0.083, 0.089))
            long_ = float(rng.uniform(0.11, 0.13))
            long_axis = str(rng.choice(["x", "y"]))
            yaw = (0.0 if long_axis == "x" else np.pi / 2) + float(rng.uniform(-np.radians(10), np.radians(10)))
            dname = str(rng.choice(list(DIRS)))
            u = np.array(DIRS[dname])
            # half extent of the box along the push direction (box roughly axis aligned)
            ext = 0.5 * (long_ if (long_axis == "x") == (dname in ("forward", "backward")) else narrow)
            dist = float(rng.uniform(SQ_IN + ext + 0.01, SQ_IN + ext + 0.07))
            lat = float(rng.uniform(-0.03, 0.03))
            box_xy = np.array([rng.uniform(-0.20, 0.04), rng.uniform(-0.20, 0.20)])
            sq_xy = box_xy + u * dist + np.array([-u[1], u[0]]) * lat
            # reach: pushing backward needs the hand ~5 cm beyond the box's far face; the arm's reach at table height
            # ends around x = 0.05-0.08 (seed 1015 of the first version: hand stalled at x = 0.09, box never moved)
            reach_ok = dname != "backward" or box_xy[0] + ext <= -0.01
            if -0.22 <= sq_xy[0] <= 0.06 and -0.20 <= sq_xy[1] <= 0.20 and reach_ok:
                return dict(seed=int(seed), box_half=[long_ / 2, narrow / 2], box_yaw=float(yaw), box_xy=box_xy.tolist(),
                            square_xy=sq_xy.tolist(), direction=dname, long_axis=long_axis)

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        for g in self._box_geoms():
            m.geom_size[g, 0], m.geom_size[g, 1] = inst["box_half"]
            m.geom_size[g, 2] = BOX_HALF_Z
        envs.scale_mass(self.env, "box_1", BOX_MASS)
        envs.set_friction(self.env, "box_1", BOX_MU)
        sq = self.env.fixtures_dict["square_1"]
        for g in range(m.ngeom):
            if (m.geom_id2name(g) or "").startswith("square_1_"):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        bid = m.body_name2id(sq.root_body)
        m.body_pos[bid] = [inst["square_xy"][0], inst["square_xy"][1], envs.TABLE_Z + 0.0002]
        m.body_quat[bid] = [1.0, 0.0, 0.0, 0.0]
        bx, by = inst["box_xy"]
        envs.set_obj_pose(self.env, "box_1", [bx, by, envs.TABLE_Z + BOX_HALF_Z + 0.0005], yaw=inst["box_yaw"])

    def reset_instance(self, inst, recorder=None):
        self._mon = None
        sk = super().reset_instance(inst, recorder=recorder)
        self._start_monitor()
        return sk

    # ------------------------------------------------------------ geometry / perception stand-in
    def observe(self):
        """What a perfect perceiver reads off the image: box centre, yaw and footprint, square centre."""
        b = envs.obj_pos(self.env, "box_1")
        return dict(box_xy=b[:2].copy(), box_yaw=_yaw_of(envs.obj_quat(self.env, "box_1")),
                    square_xy=np.array(self.inst["square_xy"], float), half=np.array(self.inst["box_half"], float))

    def corners3d(self):
        """the 8 corners of the box (world)"""
        d = self.env.sim.data
        bid = envs.body_id(self.env, "box_1")
        R = d.body_xmat[bid].reshape(3, 3)
        h = np.array([self.inst["box_half"][0], self.inst["box_half"][1], BOX_HALF_Z])
        return np.array([d.body_xpos[bid] + R @ (h * [sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1)
                         for sz in (-1, 1)])

    def corners(self, o=None):
        """footprint corners (xy) of the box in its current pose (also when it is tilted or on its side)"""
        return self.corners3d()[:, :2]

    def protrusion(self, o=None):
        """How far the box footprint sticks out beyond each edge of the square (m, >0 = outside)."""
        o = o or self.observe()
        rel = self.corners(o) - o["square_xy"]
        return {"forward": float(rel[:, 0].max() - SQ_IN), "backward": float(-rel[:, 0].min() - SQ_IN),
                "left": float(rel[:, 1].max() - SQ_IN), "right": float(-rel[:, 1].min() - SQ_IN)}

    def extent_along(self, u, o=None):
        """half extent of the box footprint along the unit xy vector u"""
        o = o or self.observe()
        c, s = np.cos(o["box_yaw"]), np.sin(o["box_yaw"])
        hx, hy = o["half"]
        return float(hx * abs(u @ [c, s]) + hy * abs(u @ [-s, c]))

    # ------------------------------------------------------------ contacts / event monitor
    def _groups(self):
        if not hasattr(self, "_gg"):
            m = self.env.sim.model
            fg = {g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith("gripper0_finger")}
            robot = set(envs.robot_geom_ids(self.env))
            self._gg = (fg, robot, set(envs.obj_geom_ids(self.env, "box_1")))
        return self._gg

    def _finger_landings(self):
        """{'box' | 'table' | 'other': n_fingers} for fingers resting on a top surface (contact normal vertical)."""
        m, d = self.env.sim.model, self.env.sim.data
        fg, robot, box = self._groups()
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in fg and c.geom2 not in robot:
                f, other = c.geom1, c.geom2
            elif c.geom2 in fg and c.geom1 not in robot:
                f, other = c.geom2, c.geom1
            else:
                continue
            if abs(c.frame[2]) < 0.7:
                continue
            lab = "box" if other in box else ("table" if (m.geom_id2name(other) or "").startswith("table") else "other")
            name = m.geom_id2name(f) or ""
            out.setdefault(lab, set()).add("1" if "finger1" in name else "2")
        return out

    def _closing_axis(self):
        m, d = self.env.sim.model, self.env.sim.data
        jid = m.joint_name2id(self.env.robots[0].gripper.joints[0])
        a = d.xaxis[jid][:2]
        return a / max(np.linalg.norm(a), 1e-9)

    def _start_monitor(self):
        b = envs.obj_pos(self.env, "box_1")
        self._mon = dict(rest_z=float(b[2]), start_xy=b[:2].copy(), start_yaw=_yaw_of(envs.obj_quat(self.env, "box_1")),
                         max_dz=0.0, top_hit=None, prev=set(), was_held=False)

    def _monitor_step(self):
        mon = self._mon
        zmin = float(self.corners3d()[:, 2].min())
        mon["max_dz"] = max(mon["max_dz"], zmin - envs.TABLE_Z)  # whole box above the table
        hits = self._finger_landings()
        open_ = self.sk is not None and self.sk.gripper_width() > 0.03
        if "box" in hits and "box" not in mon["prev"] and open_:  # an open gripper landing on the box top = a grasp
            o = self.observe()
            ca = self._closing_axis()
            e = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
            mon["top_hit"] = dict(span=2 * self.extent_along(ca, o), n_fingers=len(hits["box"]),
                                  offset=float(np.linalg.norm(e[:2] - o["box_xy"])))
        if "box" in hits and len(hits["box"]) == 2 and mon["top_hit"] is not None and open_:
            mon["top_hit"]["n_fingers"] = 2
        mon["prev"] = set(hits)

    # ------------------------------------------------------------ outcome / F2
    def outcome(self, inst):
        """Success = the whole box footprint inside the square, box resting on the table, released."""
        envs.settle(self.env, 30)
        env, mon = self.env, self._mon or {}
        o = self.observe()
        b = envs.obj_pos(env, "box_1")
        prot = self.protrusion(o)
        inside = max(prot.values()) <= 0.0
        upright = abs(envs.obj_upright_cos(env, "box_1")) > 0.9
        held = bool(self.sk.holding("box_1")) if self.sk is not None else False
        zmin = float(self.corners3d()[:, 2].min())
        resting = abs(zmin - envs.TABLE_Z) < 0.01
        success = bool(inside and resting and not held)
        rel = o["box_xy"] - o["square_xy"]
        moved = float(np.linalg.norm(o["box_xy"] - mon.get("start_xy", o["box_xy"])))
        turned = float(np.degrees(_wrap_pi(o["box_yaw"] - mon.get("start_yaw", o["box_yaw"])))) if upright else 0.0
        lifted = mon.get("max_dz", 0.0) > 0.01
        top = mon.get("top_hit")
        parts, mode = [], "success"

        def where():
            return (f"its centre is {100 * abs(rel[0]):.1f} cm {'forward' if rel[0] > 0 else 'backward'} and "
                    f"{100 * abs(rel[1]):.1f} cm {'left' if rel[1] > 0 else 'right'} of the square's centre")

        def sticking():
            return ", ".join(f"{100 * v:.1f} cm beyond the square's {k} edge" for k, v in
                             sorted(prot.items(), key=lambda kv: -kv[1]) if v > 0.0)

        if success:
            parts.append("The box lies completely inside the square.")
        elif b[2] < envs.TABLE_Z - 0.05:
            mode = "fell_off"
            parts.append("The box fell off the table.")
        elif held:
            mode = "held"
            parts.append(f"The box is still held by the gripper; {where()}.")
        elif not resting:
            mode = "not_resting"
            parts.append(f"The box is not resting on the table (it is propped up or tilted); {where()}.")
        else:
            mode = "outside"
            if top is not None and not lifted:
                mode = "too_wide"
                if top["n_fingers"] >= 2 or top["span"] > GRIP_OPEN:
                    parts.append(f"The fingers came down on top of the box instead of on either side of it: along the "
                                 f"gripper's closing direction the box measures {100 * top['span']:.1f} cm, wider than "
                                 f"the gripper's maximum opening ({100 * GRIP_OPEN:.1f} cm). The box was not grasped.")
                else:
                    parts.append(f"A finger came down on top of the box (the gripper was {100 * top['offset']:.1f} cm "
                                 f"from the box centre). The box was not grasped.")
            if moved < 0.01:
                mode = "not_moved" if mode == "outside" else mode
                parts.append(f"The box did not move; it sticks out {sticking()}.")
            else:
                parts.append(f"The box was moved {100 * moved:.1f} cm but is not completely inside the square: it "
                             f"sticks out {sticking()}; {where()}.")
                if abs(turned) >= 5:
                    rot = "counter-clockwise" if turned > 0 else "clockwise"
                    parts.append(f"It turned by {abs(turned):.0f} deg ({rot} seen from above).")
        if not success and not upright and mode not in ("fell_off",):
            parts.append("The box is no longer lying on its large bottom face.")
        out = dict(success=success, mode=mode, inside=bool(inside), protrusion=prot, rel=[float(rel[0]), float(rel[1])],
                   moved=moved, turned_deg=turned, lifted=bool(lifted), held=held, upright=bool(upright),
                   top_hit_span=None if top is None else float(top["span"]),
                   terminal=bool(b[2] < envs.TABLE_Z - 0.05), detail=" ".join(parts))
        self._start_monitor()
        return out

    # ------------------------------------------------------------ primitives for the scripted policies
    def _yaw_near(self, yaw):
        cands = [yaw + k * np.pi for k in (-2, -1, 0, 1, 2)]
        cands = [c for c in cands if abs(c) <= np.pi / 2 + 0.4] or cands
        return float(min(cands, key=lambda c: abs(c - self.sk.yaw)))

    def _align(self, tol=0.03, max_steps=60):
        sk = self.sk
        for _ in range(max_steps):
            R = sk._R() @ sk.eef_mat().T
            if np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) < tol:
                break
            sk.hold(1)

    def _guarded_descent(self, xyz, speed=0.12, max_steps=60):
        """descent that stops when a finger lands on a top surface above the target (contact sensing)"""
        sk = self.sk
        p0 = sk.eef_pos()
        tgt = np.asarray(xyz, float)
        dist = max(np.linalg.norm(tgt - p0), 1e-6)
        R = sk._R()
        for k in range(max_steps):
            ref = p0 + (tgt - p0) * min(1.0, (k + 1) * speed * DT / dist)
            sk._act(ref - sk.eef_pos(), _rot_err(R, sk.eef_mat()), sk.grip)
            if self._finger_landings() and sk.eef_pos()[2] > tgt[2] + 0.005:
                break
            if np.linalg.norm(tgt - sk.eef_pos()) < 0.003:
                break
        sk.hold(4)

    def _finger_front(self, direction):
        m, d = self.env.sim.model, self.env.sim.data
        e = self.sk.eef_pos()
        best = 0.0
        for g in range(m.ngeom):
            if not (m.geom_id2name(g) or "").startswith("gripper0_finger") or m.geom_type[g] != 7:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - e
            best = max(best, float((w[:, :2] @ direction).max()))
        return best

    def grasp_and_place(self, dx=0.0, dy=0.0, dyaw=0.0, across="narrow"):
        """Top-down grasp across the box's narrow side at its centre (+ offsets); if it is held, put it in the square."""
        sk, o = self.sk, self.observe()
        gx, gy = o["box_xy"][0] + dx, o["box_xy"][1] + dy
        # yaw 0 closes along world y; to close across the narrow side (box local y) use the box yaw
        yaw = self._yaw_near(o["box_yaw"] + (0.0 if across == "narrow" else np.pi / 2) + dyaw)
        sk.set_gripper(False, steps=8)
        sk.move_to([gx, gy, Z_GRASP + 0.10], yaw=yaw)
        self._align()
        self._guarded_descent([gx, gy, Z_GRASP])
        sk.set_gripper(True, steps=15)
        z0 = envs.obj_pos(self.env, "box_1")[2]
        sk.move_to([gx, gy, Z_GRASP + 0.12], speed=0.25)
        if not (sk.holding("box_1") and envs.obj_pos(self.env, "box_1")[2] > z0 + 0.06):
            sk.set_gripper(False, steps=8)
            return
        sx, sy = o["square_xy"]
        sk.move_to([sx, sy, Z_GRASP + 0.12])
        sk.move_to([sx, sy, Z_GRASP + 0.005])
        sk.set_gripper(False, steps=12)
        sk.move_to([sx, sy, Z_GRASP + 0.12])

    def push_stroke(self, u, dist):
        """Closed gripper behind the box face opposite to u (fingers side by side across the push direction), sweep
        along u so that the box moves `dist`, then lift away. Built from move_to + close only."""
        sk, o = self.sk, self.observe()
        u = np.asarray(u, float) / np.linalg.norm(u)
        c = o["box_xy"]
        ext = self.extent_along(u, o)
        yaw = self._yaw_near(float(np.arctan2(u[1], u[0])))  # yaw a: closing axis along (-sin a, cos a), i.e. across u
        sk.set_gripper(True, steps=6)
        face = c - u * ext
        p_hi = face - u * 0.08
        sk.move_to([p_hi[0], p_hi[1], Z_PUSH + 0.10], yaw=yaw)
        self._align()
        front = self._finger_front(u)
        start = face - u * (front + FRONT_GAP)
        stop = face + u * (dist - front)
        sk.move_to([start[0], start[1], Z_PUSH + 0.10])
        sk.move_to([start[0], start[1], Z_PUSH], tol=0.003)
        sk.move_to([stop[0], stop[1], Z_PUSH], speed=PUSH_SPEED, tol=0.004, max_steps=400)
        sk.hold(3)
        back = stop - u * 0.02
        sk.move_to([back[0], back[1], Z_PUSH], speed=0.1, max_steps=40)
        sk.move_to([back[0], back[1], Z_PUSH + 0.10], speed=0.25)

    def push_to(self, goal_xy, max_strokes=4, tol=0.008):
        """Closed-loop push along the box's own axes (each stroke perpendicular to a face keeps the box from turning):
        re-observe after every stroke, push along the axis with the larger remaining error."""
        for _ in range(max_strokes):
            o = self.observe()
            d = np.asarray(goal_xy, float) - o["box_xy"]
            c, s = np.cos(o["box_yaw"]), np.sin(o["box_yaw"])
            axes = [np.array([c, s]), np.array([-s, c])]
            comps = [float(d @ a) for a in axes]
            i = int(np.argmax(np.abs(comps)))
            if abs(comps[i]) < tol:
                break
            u = axes[i] * np.sign(comps[i])
            self.push_stroke(u, abs(comps[i]))

    def execute(self, inst, params):
        if params["mode"] == "push":
            goal = np.array(inst["square_xy"]) + np.asarray(params.get("goal_offset", (0.0, 0.0)))
            self.push_to(goal, max_strokes=params.get("strokes", 4))
        else:
            self.grasp_and_place(params.get("dx", 0.0), params.get("dy", 0.0), params.get("dyaw", 0.0),
                                 params.get("across", "narrow"))

    # ------------------------------------------------------------ scripted references
    def default_params(self, inst):
        return dict(mode="grasp")

    def oracle_params(self, inst):
        return dict(mode="push")

    def blind_params(self, inst, attempt, rng):
        return dict(mode="grasp", dx=float(rng.normal(0, 0.008)), dy=float(rng.normal(0, 0.008)),
                    dyaw=float(rng.normal(0, 0.15)))

    def adapt_params(self, inst, history):
        """too_wide (fingers on top of the box, span > opening) -> the box cannot be picked up: push it instead.
        A push that leaves the box outside -> push again, aiming past the reported miss (box centre offset)."""
        out, prm = history[-1]["outcome"], history[-1]["params"]
        if prm["mode"] != "push":
            if out["mode"] in ("too_wide", "not_moved"):
                return dict(mode="push")
            return dict(prm)
        if out["mode"] == "outside":
            off = np.asarray(prm.get("goal_offset", (0.0, 0.0))) - 0.5 * np.asarray(out["rel"])
            return dict(mode="push", goal_offset=off.tolist())
        return dict(prm)
