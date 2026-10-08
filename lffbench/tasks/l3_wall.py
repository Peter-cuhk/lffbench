"""L3-A wall_flush: a flat long box lies flush against a low fixed wall; put it on the plate.

Why the prior fails: the only graspable dimension of the box is its 6 cm width (the Panda opens to
~7.9 cm; the box is 16 cm long). Grasping across the width needs one finger between the box and the
wall, but the box touches the wall (gap 0-3 mm) along its whole length, so that finger lands on top of
the wall (4.5 cm high; the box is 3.5 cm high) and the gripper closes above the box. Retrying the same
grasp with small variations fails the same way.

What must be learned from the failure: this grasp is infeasible in this state -> change the manipulation
mode: first slide the box along the wall until it clears the wall's end (non-prehensile pre-grasp
manipulation), then grasp it across its width.

Layout: the wall runs along y (left-right as seen from the robot). Randomised per instance: wall length
(24-30 cm), wall position, whether the wall is on the far (+x) or near (-x) side of the box, where the
box sits along the wall (always fully alongside it, >= 2 cm of wall beyond each end), box-wall gap
(0-3 mm, below camera resolution), plate position (on the open side of the box).

Protocols: `l3_wall_flush` (cross: reset to the same instance before every attempt) and
`l3_wall_flush_within` (one episode; each attempt starts from wherever the previous one left things).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import box_xml, register_generated
from ..skills import DT, _rot_err
from ..task_base import LFFTask, register_task

BOX_HALF = np.array([0.03, 0.08, 0.0175])  # 6 (x) x 16 (y) x 3.5 cm; long axis along y
WALL_HALF_T = 0.015  # 3 cm thick (x)
WALL_HALF_H = 0.0225  # 4.5 cm high
WALL_HALF_L_MAX = 0.16  # registered half length (y); shrunk per instance
WALL_MU = 2.0  # wall sliding friction (MuJoCo uses the max of the two geoms: finger-wall and box-wall contacts get 2.0)
BOX_DENSITY = 1500.0  # -> 0.50 kg
GRIP_OPEN = 0.079  # measured distance between the finger pads when fully open
FINGER_NEEDS = 0.012  # free space a finger needs beside the box (pad + finger shell), measured
Z_GRASP = envs.TABLE_Z + 0.022  # gripper-site height for grasping (pads cover 1.7-3.4 cm above the table)
Z_PUSH = envs.TABLE_Z + 0.022  # site height while pushing: finger tips 1.2 cm above the table, hand ~7 mm above the wall
PUSH_SPEED = 0.08
FINGER_BELOW_SITE = 0.0096  # finger tips are this far below the gripper site (measured)
PLACE_Z = envs.TABLE_Z + 0.048  # site height at release over the plate (box bottom ~8 mm above the rim)


def _wrap_half_pi(a):
    """wrap to [-pi/2, pi/2): box and parallel gripper are both symmetric under 180 deg"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def _yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return _wrap_half_pi(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@register_task
class WallFlush(LFFTask):
    name = "l3_wall_flush"
    level = "L3"
    category = "strategy_switching"
    capabilities = ("Perceive", "Reason", "Plan")
    protocol = "cross"
    instruction = "Put the long box on the plate."
    instruction_indirect = "Put the flat blue block onto the round dish."

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        self._mon = None
        # Per-control-step event monitor. It wraps env.step so that it also sees attempts driven by the agent
        # harness (whose tools call Skills / env.step directly), not only the scripted `execute`.
        orig_step = self.env.step

        def step(action):
            ret = orig_step(action)
            if self._mon is not None:
                self._monitor_step()
            return ret

        self.env.step = step

    def reset_instance(self, inst, recorder=None):
        self._mon = None
        sk = super().reset_instance(inst, recorder=recorder)
        self._start_monitor()
        return sk

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        # 0.5 kg box and a grippy wall top (mu 2): with mu 1 / 0.13 kg an *unguarded* press onto the wall top (plain
        # move_to down, as an agent would do) let the finger slide ~5 cm off the wall and, on the wall-far side,
        # flipped the box onto its 3.5 cm side (10/60 instances) -- an accidental, graspable state (see task card).
        box = register_generated("LffFlushBox", box_xml("lff_flush_box", BOX_HALF, (0.12, 0.30, 0.80, 1.0),
                                                        density=BOX_DENSITY))
        wall = register_generated("LffFlushWall", box_xml("lff_flush_wall", (WALL_HALF_T, WALL_HALF_L_MAX, WALL_HALF_H),
                                                          (0.62, 0.50, 0.36, 1.0), density=1000,
                                                          friction=(WALL_MU, 0.005, 0.0001)), free=False)
        # regions only need to be valid and far apart; the layout is overridden in apply_instance
        objs = [("box_1", box, "box_region", (-0.21, -0.26, -0.19, -0.24)),
                ("plate_1", "plate", "plate_region", (-0.01, 0.24, 0.01, 0.26))]
        fx = [("wall_1", wall, "wall_region", (0.19, -0.01, 0.21, 0.01))]
        return write_bddl(self.name, self.instruction, objs, fx)

    def _wall(self):
        """(root body id, geom ids) of the wall fixture"""
        if not hasattr(self, "_wall_cache"):
            m = self.env.sim.model
            bid = m.body_name2id(self.env.fixtures_dict["wall_1"].root_body)

            def under(b):
                while b > 0:
                    if b == bid:
                        return True
                    b = m.body_parentid[b]
                return False

            self._wall_cache = (bid, [g for g in range(m.ngeom) if under(m.geom_bodyid[g])])
        return self._wall_cache

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        side = int(rng.choice([-1, 1]))  # +1: wall on the far (+x) side of the box, -1: near (-x) side
        wall_hl = float(rng.uniform(0.12, 0.15))
        wall_y = float(rng.uniform(-0.06, 0.06))
        u_max = wall_hl - BOX_HALF[1] - 0.02  # >= 2 cm of wall beyond each end of the box
        box_y = wall_y + float(rng.uniform(-u_max, u_max))
        gap = float(rng.uniform(0.0, 0.003))
        box_x = float(rng.uniform(-0.12, -0.04)) if side > 0 else float(rng.uniform(-0.20, -0.12))
        wall_x = box_x + side * (WALL_HALF_T + BOX_HALF[0] + gap)
        plate_xy = [box_x - side * float(rng.uniform(0.16, 0.18)), box_y + float(rng.uniform(-0.06, 0.06))]
        return dict(seed=int(seed), side=side, wall_hl=wall_hl, wall_xy=[wall_x, wall_y], gap=gap,
                    box_xy=[box_x, box_y], plate_xy=plate_xy)

    def apply_instance(self, inst):
        m = self.env.sim.model
        bid, geoms = self._wall()
        for g in geoms:  # shrink the (collision + visual) wall boxes to this instance's length
            m.geom_size[g, 1] = inst["wall_hl"]
        m.body_pos[bid] = [inst["wall_xy"][0], inst["wall_xy"][1], envs.TABLE_Z + WALL_HALF_H]
        m.body_quat[bid] = [1, 0, 0, 0]
        bx, by = inst["box_xy"]
        envs.set_obj_pose(self.env, "box_1", [bx, by, envs.TABLE_Z + BOX_HALF[2] + 0.0005])
        self._move_keep_orientation("plate_1", inst["plate_xy"])

    def _move_keep_orientation(self, name, xy, clearance=0.002):
        """Move a LIBERO mesh object to xy keeping the orientation LIBERO's sampler gave it at reset (that
        orientation is the upright one; see task card), lowest collision point `clearance` above the table."""
        qa, da = envs.free_joint_addr(self.env, name)
        d = self.env.sim.data
        d.qpos[qa:qa + 2] = xy
        d.qvel[da:da + 6] = 0.0
        self.env.sim.forward()
        d.qpos[qa + 2] -= envs.obj_min_z(self.env, name) - (envs.TABLE_Z + clearance)
        self.env.sim.forward()

    # ------------------------------------------------------------ perception stand-in
    def observe(self):
        """What a perfect perceiver reads off the camera image: box centre / yaw, the wall's footprint and the
        plate centre. (Not the box-wall gap: 0-3 mm is below the camera resolution.)"""
        bid, geoms = self._wall()
        m = self.env.sim.model
        wx, wy = m.body_pos[bid][:2]
        hl = m.geom_size[geoms[0], 1]
        b = envs.obj_pos(self.env, "box_1")
        return dict(box_xy=b[:2].copy(), box_yaw=_yaw_of(envs.obj_quat(self.env, "box_1")),
                    wall_x=float(wx), wall_y=(float(wy - hl), float(wy + hl)),
                    plate_xy=envs.obj_pos(self.env, "plate_1")[:2].copy())

    def _gap_now(self, o=None):
        o = o or self.observe()
        return float(abs(o["box_xy"][0] - o["wall_x"]) - WALL_HALF_T - BOX_HALF[0])

    def alongside_wall(self, o=None):
        """box (centre) still beside the wall and too close to it for a finger"""
        o = o or self.observe()
        return bool(o["wall_y"][0] < o["box_xy"][1] < o["wall_y"][1] and self._gap_now(o) < FINGER_NEEDS)

    # ------------------------------------------------------------ contacts
    def _finger_landings(self):
        """Surfaces the finger tips are resting on: contacts between a finger and a non-robot geom whose normal is
        mostly vertical (a tip landing on a top surface). Returns {label: contact z}, label in 'wall' | 'box' |
        'plate' | 'table' | 'other'. Side contacts (a finger brushing the box's side, the push stroke) are ignored."""
        m, d = self.env.sim.model, self.env.sim.data
        if not hasattr(self, "_geom_groups"):
            fg = {g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith("gripper0_finger")}
            robot = set(envs.robot_geom_ids(self.env))
            self._geom_groups = (fg, robot, set(self._wall()[1]), set(envs.obj_geom_ids(self.env, "box_1")),
                                 set(envs.obj_geom_ids(self.env, "plate_1")))
        fg, robot, wall, box, plate = self._geom_groups
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in fg and c.geom2 not in robot:
                other = c.geom2
            elif c.geom2 in fg and c.geom1 not in robot:
                other = c.geom1
            else:
                continue
            if abs(c.frame[2]) < 0.7:
                continue
            if other in wall:
                lab = "wall"
            elif other in box:
                lab = "box"
            elif other in plate:
                lab = "plate"
            else:
                lab = "table" if (m.geom_id2name(other) or "").startswith("table") else "other"
            out.setdefault(lab, float(c.pos[2]))
        return out

    def _finger_landing(self):
        hits = self._finger_landings()
        for lab in ("wall", "box", "plate", "table", "other"):
            if lab in hits:
                return lab
        return None

    # ------------------------------------------------------------ event monitor (what happened in this attempt)
    def _closing_axis(self):
        m, d = self.env.sim.model, self.env.sim.data
        jid = m.joint_name2id(self.env.robots[0].gripper.joints[0])
        a = d.xaxis[jid][:2]
        return a / max(np.linalg.norm(a), 1e-9)

    def _snapshot(self, contact_z):
        """state at the moment a finger lands on something"""
        o = self.observe()
        e = self.sk.eef_pos() if self.sk is not None else self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        bx = self.env.sim.data.body_xmat[envs.body_id(self.env, "box_1")].reshape(3, 3)[:2, 0]  # box short axis
        cosang = abs(float(self._closing_axis() @ (bx / max(np.linalg.norm(bx), 1e-9))))
        yaw_err = float(np.arccos(np.clip(cosang, 0.0, 1.0)))
        wy0, wy1 = o["wall_y"]
        return dict(contact_z=float(contact_z - envs.TABLE_Z), eef=[float(e[0]), float(e[1]), float(e[2])],
                    box_xy=[float(o["box_xy"][0]), float(o["box_xy"][1])], gap=self._gap_now(o),
                    alongside=self.alongside_wall(o), yaw_err=yaw_err,
                    span=float(2 * (BOX_HALF[0] * np.cos(yaw_err) + BOX_HALF[1] * np.sin(yaw_err))),
                    box_offset=float(np.linalg.norm(np.array(e[:2]) - o["box_xy"])),
                    # signed wall length beside the gripper towards each end (< 0: gripper is past that end)
                    wall_left={1: float(wy1 - e[1]), -1: float(e[1] - wy0)})

    def _start_monitor(self):
        b = envs.obj_pos(self.env, "box_1")
        self._mon = dict(start_xy=b[:2].copy(), rest_z=float(b[2]), max_dz=0.0, wall_hit=None, box_top_hit=None,
                         prev=set())

    def _monitor_step(self):
        mon = self._mon
        hits = self._finger_landings()
        mon["max_dz"] = max(mon["max_dz"], float(envs.obj_pos(self.env, "box_1")[2] - mon["rest_z"]))
        for lab, key in (("wall", "wall_hit"), ("box", "box_top_hit")):
            if lab in hits and lab not in mon["prev"]:  # onset of a landing; keep the latest one
                mon[key] = self._snapshot(hits[lab])
        mon["prev"] = set(hits)

    def _finger_front(self, direction):
        """how far the (closed) fingers stick out from the gripper site along `direction` (xy unit vector)"""
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

    # ------------------------------------------------------------ primitives used by the scripted policies
    def _yaw_near(self, yaw):
        """The parallel gripper (and the box) are symmetric under 180 deg, so pick yaw + k*pi closest to the current
        commanded yaw: the OSC's axis-angle error is ill-defined at 180 deg and a half-turn request can flip the
        hand over (measured: in the within protocol it twisted the arm into a state it never recovered from)."""
        cands = [yaw + k * np.pi for k in (-1, 0, 1)]
        cands = [c for c in cands if abs(c) <= np.pi / 2 + 0.4] or cands
        return float(min(cands, key=lambda c: abs(c - self.sk.yaw)))

    def _align(self, tol=0.03, max_steps=60):
        """Skills.move_to stops on position only; hold until the commanded yaw has actually been reached."""
        sk = self.sk
        for _ in range(max_steps):
            R = sk._R() @ sk.eef_mat().T
            if np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) < tol:
                break
            sk.hold(1)

    def _guarded_descent(self, xyz, speed=0.12, max_steps=60):
        """Straight-line descent that stops as soon as a finger lands on a surface above the target (the way a
        grasp approach with contact sensing behaves). Without it, the OSC keeps pressing the finger into the
        wall top and the compliant arm slides ~2 cm sideways off it. Returns what the finger landed on."""
        sk = self.sk
        p0 = sk.eef_pos()
        tgt = np.asarray(xyz, float) + sk.bias
        dist = max(np.linalg.norm(tgt - p0), 1e-6)
        R = sk._R()
        landed = None
        for k in range(max_steps):
            ref = p0 + (tgt - p0) * min(1.0, (k + 1) * speed * DT / dist)
            sk._act(ref - sk.eef_pos(), _rot_err(R, sk.eef_mat()), sk.grip)
            hit = self._finger_landing()
            if hit is not None and sk.eef_pos()[2] > tgt[2] + 0.005:
                landed = hit
                break
            if np.linalg.norm(tgt - sk.eef_pos()) < 0.003:
                break
        sk.hold(4)
        return landed

    def push_along_wall(self, end, margin):
        """Closed gripper (fingers side by side across the box) behind the box's trailing end face; slide the
        box along y (towards the wall's +y end if end=+1, -y end if end=-1) until the box centre is `margin`
        beyond that end of the wall."""
        sk, o = self.sk, self.observe()
        bx, by = o["box_xy"]
        wall_end = o["wall_y"][1] if end > 0 else o["wall_y"][0]
        dist = float(np.clip(end * (wall_end + end * margin - by), 0.0, 0.32))
        yaw = self._yaw_near(o["box_yaw"] + np.pi / 2)  # fingers close along the box's short axis
        u = np.array([0.0, float(end)])
        sk.set_gripper(True, steps=6)
        sk.move_to([bx, by - end * (BOX_HALF[1] + 0.06), Z_PUSH + 0.10], yaw=yaw)
        self._align()
        front = self._finger_front(u)
        face = by - end * BOX_HALF[1]
        start_y = face - end * (front + 0.035)
        stop_y = face + end * (dist - front)
        sk.move_to([bx, start_y, Z_PUSH + 0.10])
        sk.move_to([bx, start_y, Z_PUSH], tol=0.003)
        sk.move_to([bx, stop_y, Z_PUSH], speed=PUSH_SPEED, tol=0.005, max_steps=300)
        sk.hold(3)
        sk.move_to([bx, stop_y, Z_PUSH + 0.10], speed=0.2)

    def grasp_and_place(self, dx=0.0, dy=0.0, dyaw=0.0):
        """Top-down grasp across the box's width at the observed box centre (+ offsets), lift; if the box is
        held, carry it to the plate and release it there."""
        sk, o = self.sk, self.observe()
        gx, gy = o["box_xy"][0] + dx, o["box_xy"][1] + dy
        yaw = self._yaw_near(o["box_yaw"] + np.pi / 2 + dyaw)
        sk.set_gripper(False, steps=8)
        sk.move_to([gx, gy, Z_GRASP + 0.10], yaw=yaw)
        self._align()
        self._guarded_descent([gx, gy, Z_GRASP])
        sk.set_gripper(True, steps=15)
        z0 = envs.obj_pos(self.env, "box_1")[2]
        sk.move_to([gx, gy, Z_GRASP + 0.15], speed=0.25)
        if not (sk.holding("box_1") and envs.obj_pos(self.env, "box_1")[2] > z0 + 0.08):
            sk.set_gripper(False, steps=8)
            return
        px, py = o["plate_xy"]
        sk.move_to([px, py, PLACE_Z + 0.10])
        sk.move_to([px, py, PLACE_Z], tol=0.004)
        sk.set_gripper(False, steps=12)
        sk.move_to([px, py, PLACE_Z + 0.10])

    def execute(self, inst, params):
        if params["mode"] == "push_grasp":
            self.push_along_wall(params["end"], params["margin"])
        self.grasp_and_place(params.get("dx", 0.0), params.get("dy", 0.0), params.get("dyaw", 0.0))

    # ------------------------------------------------------------ outcome / F2 feedback
    def outcome(self, inst):
        """Success = box resting flat on the plate, released. The F2 text is built from the event monitor, so it
        describes agent-driven attempts as well as scripted ones. Ends the current monitoring segment."""
        envs.settle(self.env, 30)
        env, mon = self.env, self._mon or {}
        b = envs.obj_pos(env, "box_1")
        p = envs.obj_pos(env, "plate_1")
        dxy = float(np.linalg.norm(b[:2] - p[:2]))
        flat = abs(envs.obj_upright_cos(env, "box_1")) > 0.9
        held = bool(self.sk.holding("box_1")) if self.sk is not None else False
        on_plate = envs.contacts_between(env, envs.obj_geom_ids(env, "box_1"), envs.obj_geom_ids(env, "plate_1"))
        success = bool(dxy < 0.05 and envs.obj_min_z(env, "box_1") > envs.TABLE_Z + 0.004 and flat and not held
                       and on_plate)
        lifted = mon.get("max_dz", 0.0) > 0.03
        start = mon.get("start_xy", b[:2])
        moved_y = float(b[1] - start[1])
        pushed = (not lifted) and abs(moved_y) > 0.02
        wall_hit, top_hit = mon.get("wall_hit"), mon.get("box_top_hit")
        side_txt = "far (+x)" if inst["side"] > 0 else "near (-x)"
        mode, parts, wall_left = "success", [], None
        if success:
            parts.append("The box is resting on the plate.")
        else:
            if pushed:
                parts.append(f"The box was pushed {100 * abs(moved_y):.1f} cm along the wall (towards "
                             f"{'+y' if moved_y > 0 else '-y'}).")
            if lifted:
                if held:
                    mode = "still_held"
                    parts.append("The box is still in the gripper; it was not released on the plate.")
                elif on_plate:
                    mode = "placed_off"
                    parts.append(f"The box was put on the plate but is not resting on it properly ({100 * dxy:.1f} cm "
                                 f"from the plate centre{'' if flat else ', tilted'}).")
                else:
                    mode = "placed_off"
                    parts.append(f"The box was lifted but ended on the table, {100 * dxy:.1f} cm from the plate "
                                 f"centre.")
            elif wall_hit is not None:
                mode = "wall_blocked"
                parts.append(
                    f"The gripper could not get around the box: the finger on the wall side ({side_txt} side) came down "
                    f"on top of the wall, {100 * wall_hit['contact_z']:.1f} cm above the table, which is higher than "
                    f"the top of the box ({200 * BOX_HALF[2]:.1f} cm), so the box was not grasped or lifted. Where the "
                    f"gripper came down, the box is flush against the wall (gap {1000 * max(wall_hit['gap'], 0):.0f} mm).")
                if pushed:
                    e = 1 if moved_y > 0 else -1
                    wall_left = wall_hit["wall_left"][e]
                    if wall_left > 0:
                        parts.append(f"The wall still continues {100 * wall_left:.1f} cm beyond the grasp point in the "
                                     f"{'+y' if e > 0 else '-y'} direction.")
                    else:
                        parts.append(f"The grasp point was only {100 * -wall_left:.1f} cm past the "
                                     f"{'+y' if e > 0 else '-y'} end of the wall, so the finger still came down on it.")
            elif top_hit is not None:
                mode = "box_top"
                if top_hit["yaw_err"] > np.radians(30):
                    parts.append(f"The fingers came down on top of the box: along the gripper's closing direction the "
                                 f"box measures {100 * top_hit['span']:.0f} cm, more than the {100 * GRIP_OPEN:.1f} cm "
                                 f"maximum opening. The box was not lifted.")
                else:
                    parts.append(f"A finger came down on top of the box (the gripper was {100 * top_hit['box_offset']:.1f}"
                                 f" cm from the box centre), so the box was not grasped or lifted.")
            else:
                mode = "not_lifted"
                parts.append(f"The box was not lifted; it is {100 * dxy:.1f} cm from the plate centre.")
            if b[2] < envs.TABLE_Z - 0.05:
                mode = "fell_off"
                parts.append("The box fell off the table.")
            elif not flat and mode != "placed_off":
                parts.append("The box is no longer lying flat.")
        out = dict(success=success, mode=mode, plate_dist=dxy, box_xy=[float(b[0]), float(b[1])], lifted=bool(lifted),
                   pushed_y=moved_y, alongside_wall=self.alongside_wall(), wall_left=wall_left,
                   terminal=bool(b[2] < envs.TABLE_Z - 0.05), detail=" ".join(parts))
        self._start_monitor()  # next segment (within protocol) starts here
        return out

    # ------------------------------------------------------------ scripted references
    def _nearer_end(self):
        o = self.observe()
        by = o["box_xy"][1]
        return 1 if (o["wall_y"][1] - by) < (by - o["wall_y"][0]) else -1

    def default_params(self, inst):
        return dict(mode="grasp")

    def oracle_params(self, inst):
        return dict(mode="push_grasp", end=self._nearer_end(), margin=0.045)

    def blind_params(self, inst, attempt, rng):
        return dict(mode="grasp", dx=float(rng.normal(0, 0.008)), dy=float(rng.normal(0, 0.008)),
                    dyaw=float(rng.normal(0, 0.12)))

    def adapt_params(self, inst, history):
        """Uses the F2 measurements of the previous attempt:
        wall_blocked / box_top on a direct grasp -> switch mode: slide the box out past the nearer wall end
                                                    (wall ends read off the image), then grasp;
        wall_blocked after a push                -> push further by the reported remaining wall length + 2 cm;
        anything else (missed, dropped, ...)     -> keep the strategy (the state is re-observed)."""
        out, prm = history[-1]["outcome"], history[-1]["params"]
        if prm["mode"] != "push_grasp":
            if out["mode"] in ("wall_blocked", "box_top"):
                return dict(mode="push_grasp", end=self._nearer_end(), margin=0.045)
            return dict(prm)
        if out["mode"] == "wall_blocked":
            return dict(mode="push_grasp", end=prm["end"],
                        margin=float(prm["margin"] + max(out["wall_left"] or 0.0, 0.0) + 0.02))
        return dict(prm)

    # ------------------------------------------------------------ protocols
    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        if self.protocol != "within":
            return super().run_scripted(kind, inst, k=k, seed=seed, on_attempt=on_attempt)
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        self.reset_instance(inst)
        for a in range(k):
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":
                params = self.default_params(inst) if a == 0 else self.adapt_params(inst, history)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            else:
                raise ValueError(kind)
            if params["mode"] == "push_grasp" and not self.alongside_wall():
                params = dict(params, mode="grasp")  # already pushed clear in an earlier attempt of this episode
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history


@register_task
class WallFlushWithin(WallFlush):
    name = "l3_wall_flush_within"
    protocol = "within"
