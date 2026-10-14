"""LH lid_box: put three blocks on their marks, then open a box, put a red cube in, close it -- the lid falls open onto
the marks behind the box and shoves the blocks standing there off their marks.

Family: long-horizon self-disturbance ("state_restoration"): the agent's own normal later-stage action (opening the
lid) physically breaks earlier progress (blocks on marks); it has to notice and restore it before finishing (within an
attempt), or learn to open the box first (across attempts; the final-state judge does not enforce an order).

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * a flat wooden box (fixture, 12 x 16 cm, walls 3.2 cm, lid 0.6 cm) whose lid is hinged along its far edge; a black
    ball knob sticks out of the middle of the lid's front edge. The lid is heavy (0.3 kg) and has no damping: once it
    is lifted past vertical it falls backward and comes to rest on a stop (hidden angle) or on whatever is behind the
    box;
  * three 6 cm cubes ("blocks", blue / green / yellow; light, 54 g, and slippery, mu 0.25) in a row in front of the box
    together with a small red cube (2.2 cm);
  * three square marks (7.6 cm, outline + light fill) in the blocks' colours: two right behind the box (their near
    edge 0.2-1.7 cm from the box's back wall), one beside the box.

Hidden variable: the lid's opening stop (0 = closed, 90 = upright, 180 = flat behind the box): "far", uniform in FAR
(138-165 deg), in 13 of every 20 consecutive seeds, else "near", uniform in NEAR (100-113 deg). Before the lid is opened
nothing shows it. A far lid lands on the top edge of the blocks standing behind the box and its weight shoves them
away from the robot ("forward") off their marks (measured: 4-10 cm); a near lid comes to rest above them. Stops of
~115-135 deg are left out on purpose: there the lid only grazes the blocks (1-3 cm, right at the tolerance).

Why the natural order fails: the instruction lists the blocks first (as a user would), so the natural plan places the
blocks, opens the lid (it falls back onto them), puts the cube in, closes the lid and finishes without looking at the
marks again. What must be learned: re-check every stage before finishing and put knocked blocks back after closing the
lid (within the attempt), or open the box first / place the blocks behind it last (across attempts).

Lid mechanics for agents (`LidSkills`, measured, see the task card): grasping the ball knob from above (fingers on its
left and right, the default yaw) makes a pivot -- the ball turns freely between the fingers about the hinge axis. While
the knob is held, the arm is compliant along the hinge constraint: a commanded straight move is projected onto the
knob's circle around the hinge (at most 1 cm per control step; ~0.07 m/s along the arc in the agent self-test), so
"up 10 cm, then back over the hinge, release" opens the lid and "up, then forward, release" closes it; a move that
makes no more progress ends early (like KeySkills' stalling wrist in l7_key_turn). Without the compliance, all 6
typical straight-move sequences tried (prototype: this lid with a 2 cm stem, 2026-10-09) pulled the knob out of the
fingers and the lid fell shut; with an earlier, lighter lid only a 3-segment approximation of the arc opened it.

Protocol: cross (reset to the same instance before every attempt), k = 3. Outcome: final-state judge (all blocks on
their marks, red cube in the box, lid closed, nothing held, scene static) plus an event log of every stage predicate
(achieved / lost / restored, with step and cause) for the server log; F1 gives the agent only success / failure.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, box_xml, register_generated
from ..skills import DT, Skills, _rot_err
from ..task_base import LFFTask, register_task

# ---------------------------------------------------------------- geometry / physics
BOX = "box_1"
BOX_MODEL = "lff_lh_lid_box"
BOX_D, BOX_W = 0.12, 0.16  # outer depth (x) and width (y)
WALL_H, WALL_T, FLOOR_T = 0.032, 0.008, 0.006
LID_T, LID_MASS = 0.006, 0.30
KNOB_R, STEM_L = 0.013, 0.032
HINGE_DAMP, HINGE_FRIC = 0.002, 0.002
IN_X = BOX_D / 2 - WALL_T  # inner half sizes of the cavity
IN_Y = BOX_W / 2 - WALL_T

BLOCK = 0.06  # cube edge
BLOCK_DENS = 250.0  # -> 54 g
BLOCK_MU = 0.25
CUBE_E = 0.022  # red cube edge
MARK_HALF = 0.038
MARK_LINE = 0.004
TOL = 0.020  # block centre within 2 cm (per axis) of its mark's centre
FLAT_COS = np.cos(np.radians(15.0))
CUBE_MARGIN = 0.004
LID_CLOSED = np.radians(4.0)
# hidden stop (deg): in a fraction P_FAR of the instances the lid opens far back, uniform in FAR (it lands on blocks
# standing behind the box and shoves them 4-10 cm), otherwise it stops near upright, uniform in NEAR (it comes to rest
# above them). Far / near is balanced per block of 20 consecutive seeds (13 / 7), see sample_instance.
# Bimodal on purpose: stops between ~115 and ~135 deg make the lid graze the blocks and push them 1-3 cm, i.e. right
# at the tolerance, which is neither a clean disturbance nor a clean no-disturbance (measured, see the task card).
P_FAR = 0.65
FAR = (138.0, 165.0)
NEAR = (100.0, 113.0)
GAP_LO, GAP_HI = 0.002, 0.017  # near edge of a behind-mark to the box's back wall (block near face +0.8 cm further)

COLORS = ("blue", "green", "yellow")
RGBA = dict(blue=(0.15, 0.30, 0.85, 1.0), green=(0.10, 0.60, 0.20, 1.0), yellow=(0.92, 0.76, 0.10, 1.0))
BLOCKS = {c: f"{c}_block_1" for c in COLORS}
MARKS = {c: f"{c}_mark_1" for c in COLORS}
CUBE = "red_cube_1"

Z_CARRY = 0.13  # scripted transit height of the grasp point above the table when carrying a block
Z_KNOB = WALL_H + LID_T / 2  # knob centre above the table with the lid closed
SMAX = 0.010  # max projected displacement command per control step while the knob is held
STALL = 12


def box_lid_xml(name, body_rgba=(0.55, 0.38, 0.22, 1.0), lid_rgba=(0.70, 0.52, 0.32, 1.0), knob_rgba=(0.10, 0.10, 0.11, 1.0)):
    hx, hy = BOX_D / 2, BOX_W / 2
    parts = [((0, 0, FLOOR_T / 2), (hx, hy, FLOOR_T / 2)),
             ((hx - WALL_T / 2, 0, WALL_H / 2), (WALL_T / 2, hy, WALL_H / 2)),
             ((-hx + WALL_T / 2, 0, WALL_H / 2), (WALL_T / 2, hy, WALL_H / 2)),
             ((0, hy - WALL_T / 2, WALL_H / 2), (hx - WALL_T, WALL_T / 2, WALL_H / 2)),
             ((0, -hy + WALL_T / 2, WALL_H / 2), (hx - WALL_T, WALL_T / 2, WALL_H / 2))]
    g = []
    # box: contype 2 / conaffinity 1 and lid: contype 4 / conaffinity 1 -> the lid never collides with its own box (the
    # hinge limit holds it closed), but both collide with the robot, the table and the free objects (contype 1)
    for k, (pos, s) in enumerate(parts):
        g.append(f'<geom name="{name}_w{k}" type="box" pos="{_fmt(pos)}" size="{_fmt(s)}" rgba="{_fmt(body_rgba)}" '
                 f'friction="0.8 0.005 0.0001" {_SOLID} contype="2" conaffinity="1" group="0"/>')
        g.append(f'<geom type="box" pos="{_fmt(pos)}" size="{_fmt(s)}" rgba="{_fmt(body_rgba)}" contype="0" '
                 f'conaffinity="0" group="1"/>')
    m, L = LID_MASS, BOX_D
    lid_c = (-L / 2, 0, LID_T / 2)
    lid_s = (L / 2, hy, LID_T / 2)
    stem = (-L - STEM_L / 2, 0, LID_T / 2)
    ball = (-L - STEM_L - KNOB_R * 0.7, 0, LID_T / 2)
    lid = [f'<inertial pos="{_fmt(lid_c)}" mass="{m}" diaginertia="{m * (BOX_W ** 2 + LID_T ** 2) / 12:.6f} '
           f'{m * (L ** 2 + LID_T ** 2) / 12:.6f} {m * (BOX_W ** 2 + L ** 2) / 12:.6f}"/>',
           # placeholder range: the opening stop is set per instance (apply_instance)
           f'<joint name="{name}_hinge" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="0 2.8" '
           f'damping="{HINGE_DAMP}" frictionloss="{HINGE_FRIC}" armature="0.0005"/>',
           f'<geom name="{name}_lid" type="box" pos="{_fmt(lid_c)}" size="{_fmt(lid_s)}" rgba="{_fmt(lid_rgba)}" mass="0" '
           f'friction="0.8 0.005 0.0001" contype="4" conaffinity="1" group="0"/>',
           f'<geom type="box" pos="{_fmt(lid_c)}" size="{_fmt(lid_s)}" rgba="{_fmt(lid_rgba)}" contype="0" conaffinity="0" group="1"/>',
           f'<geom name="{name}_stem" type="cylinder" pos="{_fmt(stem)}" quat="0.7071 0 0.7071 0" size="0.0045 {STEM_L / 2:.4f}" '
           f'rgba="{_fmt(knob_rgba)}" mass="0" friction="1 0.005 0.0001" contype="4" conaffinity="1" group="0"/>',
           f'<geom type="cylinder" pos="{_fmt(stem)}" quat="0.7071 0 0.7071 0" size="0.0045 {STEM_L / 2:.4f}" '
           f'rgba="{_fmt(knob_rgba)}" contype="0" conaffinity="0" group="1"/>',
           f'<geom name="{name}_knob" type="sphere" pos="{_fmt(ball)}" size="{KNOB_R:.4f}" rgba="{_fmt(knob_rgba)}" mass="0" '
           f'friction="1 0.005 0.0001" contype="4" conaffinity="1" group="0"/>',
           f'<geom type="sphere" pos="{_fmt(ball)}" size="{KNOB_R:.4f}" rgba="{_fmt(knob_rgba)}" contype="0" conaffinity="0" group="1"/>']
    shell = "\n        ".join(g)
    lid_xml = "\n          ".join(lid)
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        {shell}
        <body name="lid" pos="{hx:.4f} 0 {WALL_H:.4f}">
          {lid_xml}
        </body>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {WALL_H:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{hx:.4f} {hy:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>"""


def mark_xml(name, rgba):
    """Flat, visual-only square drawn on the table: a light fill plus an outline in the block's colour."""
    fill = tuple(0.55 + 0.45 * c for c in rgba[:3]) + (1.0,)
    o, i, w = MARK_HALF, MARK_HALF - MARK_LINE, MARK_LINE
    parts = [((0, 0, 0.0004), (o, o, 0.0004), fill),
             ((0, i + w / 2, 0.0010), (o, w / 2, 0.0002), rgba),
             ((0, -i - w / 2, 0.0010), (o, w / 2, 0.0002), rgba),
             ((i + w / 2, 0, 0.0010), (w / 2, o, 0.0002), rgba),
             ((-i - w / 2, 0, 0.0010), (w / 2, o, 0.0002), rgba)]
    g = "\n".join(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(c)}" '
                  f'contype="0" conaffinity="0" group="1" />' for p, sz, c in parts)
    return _wrap(name, g, 0.0012, o)


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


def _lr(v):
    return f"{100 * abs(v):.1f} cm to the {'left' if v > 0 else 'right'}"


class LidSkills(Skills):
    """Skills for the lid box: a per-step stage monitor, and a hinge-compliant arm while the knob is held (see the
    module docstring)."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task
        self.latched = False
        self.rad0 = None
        self.y0 = None
        self.knob_steps = 0  # control steps driven with the knob held (compliant mode)
        self.knob_grasps = 0

    def holding_knob(self):
        """both fingers (pad or finger body) touch the knob ball or its stem"""
        g = self.env.robots[0].gripper
        return bool(self.env._check_grasp(gripper=[g.important_geoms["left_finger"], g.important_geoms["right_finger"]],
                                          object_geoms=self.task.knob_geom_names))

    def _act(self, dpos, drot, grip):
        if self.latched:
            e = self.eef_pos()
            if self.grip < 0 or np.linalg.norm(e - self.task.knob_xyz()) > 0.035:
                self.latched = False  # opened, or the knob slipped out of the fingers
            else:
                hx, hz = self.task.hinge_xz()
                r = np.array([e[0] - hx, e[2] - hz])
                rn = max(float(np.linalg.norm(r)), 1e-6)
                tng = np.array([-r[1], r[0]]) / rn
                s = float(np.clip(dpos[0] * tng[0] + dpos[2] * tng[1], -SMAX, SMAX))
                corr = (self.rad0 - rn) * r / rn
                dpos = np.array([s * tng[0] + corr[0], self.y0 - e[1], s * tng[1] + corr[1]])
                self.knob_steps += 1
        super()._act(dpos, drot, grip)
        self.task.monitor_step()

    def set_gripper(self, close, steps=15):
        rep = super().set_gripper(close, steps)
        if close and self.holding_knob():
            e = self.eef_pos()
            hx, hz = self.task.hinge_xz()
            self.latched = True
            self.rad0 = float(np.hypot(e[0] - hx, e[2] - hz))
            self.y0 = float(e[1])
            self.knob_grasps += 1
        elif not close:
            self.latched = False
        return rep

    def move_to(self, target, yaw=None, speed=None, tol=0.004, max_steps=250, apply_bias=True, wrap_yaw=True):
        """Skills.move_to plus an early stop when the knob is held and the arm has stopped making progress."""
        tgt = np.asarray(target, float) + (self.bias if apply_bias else 0.0)
        if yaw is not None:
            self.yaw = self.wrap_yaw(float(yaw)) if wrap_yaw else float(yaw)
        R = self._R()
        p0 = self.eef_pos()
        dist = np.linalg.norm(tgt - p0)
        direction = (tgt - p0) / max(dist, 1e-9)
        s = 0.0
        hist = []
        stalled = False
        for k in range(max_steps):
            if speed is not None:
                s = min(dist, s + speed * DT)
                ref = p0 + direction * s + direction * min(dist - s, speed * DT * 1.5)
            else:
                ref = tgt
            self._act(ref - self.eef_pos(), _rot_err(R, self.eef_mat()), self.grip)
            cur = self.eef_pos()
            if np.linalg.norm(tgt - cur) < tol and (speed is None or s >= dist):
                break
            hist.append(cur)
            if (self.latched and len(hist) > STALL and (speed is None or s >= dist)
                    and np.linalg.norm(hist[-1] - hist[-1 - STALL]) < 0.002):
                stalled = True
                break
        rep = dict(prim="move_to", reached=bool(np.linalg.norm(tgt - self.eef_pos()) < 0.02),
                   pos=self.eef_pos().round(4).tolist(), steps=k + 1, stalled=stalled)
        self.log.append(rep)
        return rep


@register_task
class LidBox(LFFTask):
    name = "lh_lid_box"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Plan", "Reason", "Utilize")
    protocol = "cross"
    max_attempts = 3
    agent_max_steps = 110  # careful oracle ~45-55 commands + restoring two blocks (~14) + looking around
    instruction = ("Put each of the three coloured blocks on the square mark of the same colour on the table. Then open "
                   "the lid of the wooden box (lift it by the black knob on its front edge), put the red cube inside "
                   "the box, and close the lid.")
    instruction_indirect = ("Set the blue, green and yellow cubes onto the outlined squares of matching colour. Then open "
                            "the flat wooden case (its lid lifts by the black ball at the front), place the small red "
                            "cube in it and shut the lid.")
    agent_object_names = ["blue block", "green block", "yellow block", "red cube", "wooden box", "lid knob",
                          "blue mark", "green mark", "yellow mark"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        box = register_generated("LffLhLidBox", box_lid_xml(BOX_MODEL), free=False)
        b = BLOCK / 2
        objs, fx = [], [(BOX, box, "box_region", (-0.18, -0.01, -0.16, 0.01))]
        for i, c in enumerate(COLORS):
            blk = register_generated(f"LffLhBlock{c.title()}",
                                     box_xml(f"lff_lh_block_{c}", (b, b, b), RGBA[c], density=BLOCK_DENS,
                                             friction=(BLOCK_MU, 0.005, 0.0001)).replace(_SOLID, ""))
            mark = register_generated(f"LffLhMark{c.title()}", mark_xml(f"lff_lh_mark_{c}", RGBA[c]), free=False)
            objs.append((BLOCKS[c], blk, f"{c}_block_region", (-0.31, -0.22 + 0.15 * i, -0.29, -0.20 + 0.15 * i)))
            fx.append((MARKS[c], mark, f"{c}_mark_region", (0.0, -0.22 + 0.15 * i, 0.02, -0.20 + 0.15 * i)))
        cube = register_generated("LffLhRedCube", box_xml("lff_lh_red_cube", (CUBE_E / 2,) * 3, (0.85, 0.10, 0.08, 1.0),
                                                          density=500))
        objs.append((CUBE, cube, "red_cube_region", (-0.31, 0.24, -0.29, 0.26)))
        return write_bddl(self.name, "put the blocks on their marks and the red cube in the box", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        pre = f"{BOX}_{BOX_MODEL}"
        self.jid = m.joint_name2id(pre + "_hinge")
        self.qa = m.jnt_qposadr[self.jid]
        self.dof = m.jnt_dofadr[self.jid]
        self.box_bid = m.body_name2id(self.env.fixtures_dict[BOX].root_body)
        self.knob_geom_name = pre + "_knob"
        self.knob_geom_names = [pre + "_knob", pre + "_stem"]
        self.knob_gid = m.geom_name2id(self.knob_geom_name)
        self.lid_gids = {m.geom_name2id(pre + s) for s in ("_lid", "_stem", "_knob")}
        self.robot_gids = set(envs.robot_geom_ids(self.env))
        self.mark_bid = {c: m.body_name2id(self.env.fixtures_dict[MARKS[c]].root_body) for c in COLORS}
        self.block_gids = {c: set(envs.obj_geom_ids(self.env, BLOCKS[c])) for c in COLORS}
        self.inst = None
        self.mon = None

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        bx, by = float(rng.uniform(-0.19, -0.15)), float(rng.uniform(-0.03, 0.03))
        hinge_x = bx + BOX_D / 2
        order = [str(c) for c in rng.permutation(COLORS)]  # [behind-left, behind-right, side]
        side = 1.0 if rng.random() < 0.5 else -1.0
        marks = {}
        gaps = {}
        for c, sy in zip(order[:2], (1.0, -1.0)):
            g = float(rng.uniform(GAP_LO, GAP_HI))
            gaps[c] = g
            marks[c] = [hinge_x + g + MARK_HALF, by + sy * 0.05 + float(rng.uniform(-0.005, 0.005))]
        marks[order[2]] = [bx + float(rng.uniform(-0.03, 0.03)), by + side * (BOX_W / 2 + 0.025 + MARK_HALF)]
        # start row in front of the box: four slots outside the knob's lane, shuffled (3 blocks + red cube)
        slots = [-0.22, -0.11, 0.11, 0.22]  # 11 cm apart: room for the open fingers next to a 6 cm block
        items = [BLOCKS[c] for c in COLORS] + [CUBE]
        perm = rng.permutation(len(items))
        start = {items[i]: [float(-0.31 + rng.uniform(-0.01, 0.01)), float(slots[j] + rng.uniform(-0.01, 0.01))]
                 for j, i in enumerate(perm)}
        rng.random()  # (draw kept so the other parameters stay the same)
        # balanced blocks like l7_key_turn: exactly round(20 * P_FAR) of every 20 consecutive seeds open far
        far = bool((int(seed) * 7) % 20 < round(20 * P_FAR))
        stop = float(rng.uniform(*(FAR if far else NEAR)))
        return dict(seed=int(seed), box_xy=[bx, by], stop_deg=stop, lid_far=far, marks=marks,
                    behind=order[:2], side_color=order[2], side="left" if side > 0 else "right", gaps=gaps, start=start,
                    start_yaw={k: float(rng.uniform(-0.3, 0.3)) for k in items})

    def apply_instance(self, inst):
        self.inst = inst
        m, d = self.env.sim.model, self.env.sim.data
        bx, by = inst["box_xy"]
        m.body_pos[self.box_bid] = [bx, by, envs.TABLE_Z]
        m.body_quat[self.box_bid] = [1.0, 0.0, 0.0, 0.0]
        m.jnt_range[self.jid] = [0.0, np.radians(inst["stop_deg"])]
        m.jnt_solref[self.jid] = [0.01, 1.0]
        d.qpos[self.qa] = 0.0
        d.qvel[self.dof] = 0.0
        for c in COLORS:
            x, y = inst["marks"][c]
            m.body_pos[self.mark_bid[c]] = [x, y, envs.TABLE_Z + 0.0002]
            m.body_quat[self.mark_bid[c]] = [1.0, 0.0, 0.0, 0.0]
        for name, xy in inst["start"].items():
            envs.place_upright(self.env, name, xy, yaw=inst["start_yaw"][name])
        for c in COLORS:
            envs.set_friction(self.env, BLOCKS[c], BLOCK_MU)

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.mon = None
        self.sk = LidSkills(self.env, self, recorder=recorder)
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ state helpers
    def lid_q(self):
        return float(self.env.sim.data.qpos[self.qa])

    def lid_deg(self):
        return float(np.degrees(self.lid_q()))

    def knob_xyz(self):
        return self.env.sim.data.geom_xpos[self.knob_gid].copy()

    def hinge_xz(self):
        return self.inst["box_xy"][0] + BOX_D / 2, envs.TABLE_Z + WALL_H

    def held(self, name):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.objects_dict[name]))

    def _flat(self, name):
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, name)].reshape(3, 3)
        return bool(np.max(np.abs(R[2])) >= FLAT_COS)

    def block_offset(self, c):
        p = envs.obj_pos(self.env, BLOCKS[c])
        return p[:2] - np.asarray(self.inst["marks"][c])

    def on_mark(self, c, tol=TOL):
        name = BLOCKS[c]
        p = envs.obj_pos(self.env, name)
        off = p[:2] - np.asarray(self.inst["marks"][c])
        return bool(np.max(np.abs(off)) <= tol and p[2] < envs.TABLE_Z + BLOCK / 2 + 0.01 and self._flat(name)
                    and not self.held(name))

    def cube_in_box(self):
        p = envs.obj_pos(self.env, CUBE)
        rel = p[:2] - np.asarray(self.inst["box_xy"])
        return bool(abs(rel[0]) <= IN_X - CUBE_MARGIN and abs(rel[1]) <= IN_Y - CUBE_MARGIN
                    and p[2] < envs.TABLE_Z + WALL_H and not self.held(CUBE))

    def lid_closed(self):
        return self.lid_q() <= LID_CLOSED

    def stage_values(self):
        v = {f"{c}_on_mark": self.on_mark(c) for c in COLORS}
        v["cube_in_box"] = self.cube_in_box()
        v["lid_closed"] = self.lid_closed()
        return v

    # ------------------------------------------------------------ per-step monitor
    def _init_monitor(self):
        self.mon = dict(state={f"{c}_on_mark": False for c in COLORS} | {"cube_in_box": False}, ever=set(),
                        lost=set(), events=[], last_touch={c: None for c in COLORS}, lid_hits=set(),
                        lid_open_max=0.0, lid_opened=False, lid_open_steps=[], lid_close_steps=[], step=0,
                        lid_prev=0.0, lid_moving_step=-10 ** 9)

    def _touches(self):
        """Per block: was it touched by the lid / the robot in this control step?"""
        d = self.env.sim.data
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            for cc in COLORS:
                g = self.block_gids[cc]
                other = c.geom2 if c.geom1 in g else (c.geom1 if c.geom2 in g else None)
                if other is None:
                    continue
                if other in self.lid_gids:
                    out[cc] = "lid"
                elif other in self.robot_gids and out.get(cc) != "lid":
                    out[cc] = "robot"
        return out

    def monitor_step(self):
        mon = self.mon
        if mon is None:
            return
        mon["step"] += 1
        step = self.sk.n_steps if self.sk is not None else mon["step"]
        for c, who in self._touches().items():
            mon["last_touch"][c] = (who, step)
            if who == "lid":
                mon["lid_hits"].add(c)
        q = self.lid_deg()
        if q > 95.0 and abs(q - mon["lid_prev"]) > 0.5:
            mon["lid_moving_step"] = step  # the open lid is swinging (its contacts can last less than a control step)
        mon["lid_prev"] = q
        mon["lid_open_max"] = max(mon["lid_open_max"], q)
        if not mon["lid_opened"] and q > 90.0:
            mon["lid_opened"] = True
            mon["lid_open_steps"].append(step)
            mon["events"].append(dict(step=step, pred="lid", ev="opened_past_vertical"))
        elif mon["lid_opened"] and q <= np.degrees(LID_CLOSED):
            mon["lid_opened"] = False
            mon["lid_close_steps"].append(step)
            mon["events"].append(dict(step=step, pred="lid", ev="closed"))
        if mon["step"] % 2:
            return  # predicates every other control step (0.1 s)
        for c in COLORS:
            key = f"{c}_on_mark"
            was = mon["state"][key]
            now = self.on_mark(c, TOL if not was else TOL + 0.005)  # hysteresis against jitter at the border
            if now != was:
                self._event(key, now, step, c)
        was = mon["state"]["cube_in_box"]
        now = self.cube_in_box()
        if now != was:
            self._event("cube_in_box", now, step, None)

    def _event(self, key, now, step, color):
        mon = self.mon
        mon["state"][key] = now
        if now:
            ev = "restored" if key in mon["lost"] else "achieved"
            mon["ever"].add(key)
            if key in mon["lost"]:
                mon["lost"].discard(key)
            mon["events"].append(dict(step=step, pred=key, ev=ev))
        else:
            cause = "other"
            if color is not None:
                if self.held(BLOCKS[color]):
                    cause = "picked_up"
                else:
                    lt = mon["last_touch"][color]
                    if lt is not None and step - lt[1] <= 60:
                        cause = lt[0]
                    if cause == "other" and step - mon["lid_moving_step"] <= 20 and color in self.inst["behind"]:
                        cause = "lid"  # lost while the open lid was swinging down onto the blocks behind the box
            elif self.held(CUBE):
                cause = "picked_up"
            mon["lost"].add(key)
            ev = dict(step=step, pred=key, ev="lost", cause=cause, lid_deg=round(self.lid_deg(), 1))
            if color is not None:
                ev["offset_cm"] = (100 * self.block_offset(color)).round(1).tolist()
            mon["events"].append(ev)

    # ------------------------------------------------------------ outcome
    def _static(self):
        d = self.env.sim.data
        for name in [BLOCKS[c] for c in COLORS] + [CUBE]:
            _, da = envs.free_joint_addr(self.env, name)
            if np.linalg.norm(d.qvel[da:da + 3]) > 0.02 or np.linalg.norm(d.qvel[da + 3:da + 6]) > 0.5:
                return False
        return abs(float(d.qvel[self.dof])) < 0.2

    def outcome(self, inst):
        envs.settle(self.env, 30)
        mon = self.mon
        vals = self.stage_values()
        static = self._static()
        held = [n for n in [BLOCKS[c] for c in COLORS] + [CUBE] if self.held(n)]
        success = bool(all(vals.values()) and static and not held)
        regress = [e for e in mon["events"] if e["ev"] == "lost"]
        restored = [e for e in mon["events"] if e["ev"] == "restored"]
        lid_knocks = [e for e in regress if e.get("cause") == "lid"]
        parts = []
        if success:
            failure = None
            parts.append("All three blocks are on their marks, the red cube is in the box and the lid is closed.")
        else:
            off = [c for c in COLORS if not vals[f"{c}_on_mark"]]
            failure = ("block_off_mark" if off else "cube_not_in_box" if not vals["cube_in_box"] else
                       "lid_open" if not vals["lid_closed"] else "held" if held else "not_static")
            for c in off:
                o = self.block_offset(c)
                name = BLOCKS[c]
                if self.held(name):
                    parts.append(f"The {c} block is still held by the gripper.")
                    continue
                txt = (f"The {c} block is not on the {c} mark: its centre is {_fb(o[0])} and {_lr(o[1])} of the mark's "
                       f"centre")
                if not self._flat(name):
                    txt += " and it is tipped over"
                why = [e for e in regress if e["pred"] == f"{c}_on_mark"]
                if f"{c}_on_mark" not in mon["ever"]:
                    txt += " (it was never on its mark)."
                elif why and why[-1].get("cause") == "lid":
                    txt += (f" (it had been on its mark; the lid pushed it off when it fell open to "
                            f"{why[-1]['lid_deg']:.0f} degrees).")
                elif why and why[-1].get("cause") == "robot":
                    txt += " (it had been on its mark; the robot knocked it off)."
                else:
                    txt += " (it had been on its mark earlier)."
                parts.append(txt)
            if not vals["cube_in_box"]:
                if self.held(CUBE):
                    parts.append("The red cube is still held by the gripper.")
                else:
                    p = envs.obj_pos(self.env, CUBE)
                    rel = p[:2] - np.asarray(inst["box_xy"])
                    parts.append(f"The red cube is not in the box: its centre is {_fb(rel[0])} and {_lr(rel[1])} of the "
                                 f"box's centre.")
            if not vals["lid_closed"]:
                parts.append(f"The lid is not closed (open {self.lid_deg():.0f} degrees).")
            if not static and all(vals.values()):
                parts.append("Something was still moving at the end.")
        if mon["lid_open_max"] < 90.0:
            parts.append("The lid was never opened past upright.")
        out = dict(success=success, failure=failure, stages=vals, static=bool(static), held=held,
                   events=mon["events"], n_regressions=len(regress), n_restored=len(restored),
                   n_lid_knocks=len(lid_knocks), lid_hits=sorted(mon["lid_hits"]),
                   lid_open_max=round(mon["lid_open_max"], 1), lid_final_deg=round(self.lid_deg(), 1),
                   knob_grasps=self.sk.knob_grasps, knob_steps=self.sk.knob_steps,
                   block_offsets_cm={c: (100 * self.block_offset(c)).round(1).tolist() for c in COLORS},
                   detail=" ".join(parts))
        return out

    # ------------------------------------------------------------ scripted policies (perfect perception)
    def _grasp_yaw(self, name):
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, name)].reshape(3, 3)
        i = int(np.argmin(np.abs(R[2])))  # the most horizontal body axis (works whichever face the cube rests on)
        yaw = float(np.arctan2(R[1, i], R[0, i]))
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)  # a cube: grasp yaw mod 90 deg

    def pick_place(self, name, dest_xy, z_rel, z_carry=Z_CARRY, noise=None, dest_yaw=0.0):
        """Top-down pick of `name` at its centre, carry at `z_carry` (grasp point above the table), release with the
        grasp point `z_rel` above the table at `dest_xy`, lift."""
        sk = self.sk
        Z = envs.TABLE_Z
        p = envs.obj_pos(self.env, name)
        yaw = self._grasp_yaw(name)
        zg = p[2] - Z
        sk.set_gripper(False, steps=8)
        sk.move_to([p[0], p[1], Z + max(z_carry, zg + 0.06)], yaw=yaw)
        sk.move_to([p[0], p[1], Z + zg], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([p[0], p[1], Z + z_carry], speed=0.3)
        if not self.held(name):
            sk.set_gripper(False, steps=8)
            return False
        d = np.asarray(dest_xy, float) + (np.asarray(noise, float) if noise is not None else 0.0)
        sk.move_to([d[0], d[1], Z + z_carry], yaw=dest_yaw)
        sk.move_to([d[0], d[1], Z + z_rel], tol=0.003)
        sk.set_gripper(False, steps=12)
        sk.move_to([d[0], d[1], Z + z_carry])
        return True

    def place_block(self, c, noise=None):
        return self.pick_place(BLOCKS[c], self.inst["marks"][c], BLOCK / 2 + 0.004, noise=noise)

    def put_cube(self, noise=None):
        bx, by = self.inst["box_xy"]
        return self.pick_place(CUBE, [bx, by], FLOOR_T + CUBE_E / 2 + 0.008, z_carry=0.12, noise=noise)

    def _grasp_knob(self):
        sk = self.sk
        k = self.knob_xyz()
        sk.set_gripper(False, steps=8)
        sk.move_to([k[0], k[1], k[2] + 0.08], yaw=0.0)
        sk.move_to([k[0], k[1], k[2]], tol=0.003)
        sk.set_gripper(True, steps=15)
        return k

    def open_lid(self, back=0.03, up=0.16):
        """grasp the knob; straight up 10 cm; straight to `back` m behind the hinge, `up` m above it; release; lift"""
        sk = self.sk
        k = self._grasp_knob()
        hx, hz = self.hinge_xz()
        sk.move_to([k[0], k[1], k[2] + 0.10])
        sk.move_to([hx + back, k[1], hz + up])
        sk.set_gripper(False, steps=10)
        e = sk.eef_pos()
        sk.move_to([e[0] - 0.03, e[1], e[2] + 0.06])
        sk.hold(20)

    def close_lid(self):
        """grasp the knob where it is; straight up 10 cm; straight forward to 8 cm in front of the hinge; release"""
        sk = self.sk
        k = self._grasp_knob()
        hx, hz = self.hinge_xz()
        sk.move_to([k[0], k[1], k[2] + 0.10])
        sk.move_to([hx - 0.08, k[1], k[2] + 0.10])
        sk.set_gripper(False, steps=10)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], e[2] + 0.06])
        sk.hold(20)

    def restore(self, rounds=2):
        """look at every stage and fix what is wrong (lid open -> close it; cube out -> put it in; block off -> put it
        back), up to `rounds` passes"""
        for _ in range(rounds):
            v = self.stage_values()
            if all(v.values()):
                return True
            if not v["cube_in_box"]:
                if self.lid_q() < np.radians(95):
                    self.open_lid()
                self.put_cube()
                self.close_lid()
            elif not v["lid_closed"]:
                self.close_lid()
            for c in COLORS:
                if not self.on_mark(c):
                    self.place_block(c)
        return all(self.stage_values().values())

    def execute(self, inst, params):
        order = params.get("order") or list(COLORS)
        noise = params.get("noise") or {}
        if params["plan"] == "blocks_first":
            for c in order:
                self.place_block(c, noise.get(c))
            self.open_lid(**params.get("open", {}))
            self.put_cube(noise.get("cube"))
            self.close_lid()
        else:  # "lid_first": open, cube in, close, then the blocks
            self.open_lid(**params.get("open", {}))
            self.put_cube(noise.get("cube"))
            self.close_lid()
            for c in order:
                self.place_block(c, noise.get(c))
        if params.get("check"):
            self.restore()
        self.sk.hold(10)

    def default_params(self, inst):
        return dict(plan="blocks_first", check=False)

    def oracle_params(self, inst):
        return dict(plan="lid_first", check=True)

    def adapt_params(self, inst, history):
        """Used from attempt 2 on, i.e. after a failed attempt (attempt 1 is the natural order plus a final check and
        repair, see run_scripted): open the box, put the cube in and close it first, then place the blocks, check."""
        return dict(plan="lid_first", check=True)

    def blind_params(self, inst, attempt, rng):
        """the natural order again with small random variations (block order, placement, release point), no check"""
        return dict(plan="blocks_first", check=False, order=[str(c) for c in rng.permutation(COLORS)],
                    noise={c: rng.normal(0.0, 0.004, 2).tolist() for c in COLORS},
                    open=dict(back=float(rng.uniform(0.0, 0.05)), up=float(rng.uniform(0.14, 0.18))))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":
                params = dict(plan="blocks_first", check=True) if a == 0 else self.adapt_params(inst, history)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "switch":  # extra control (not in validate's table): cross-attempt learning only, no re-check
                params = self.default_params(inst) if a == 0 else dict(plan="lid_first", check=False)
            else:
                raise ValueError(kind)
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history
