"""LH false_bottom_lid: open a lidded wooden box, put a tall red block and two blue cubes in, close the lid -- in half
of the boxes a hidden false bottom makes the box too shallow for the block standing upright, so the lid comes down on
the block and stays open; the block must be laid on its side.

Family: long-horizon common sense + self-disturbance ("state_restoration"): the agent's own normal action (placing the
block upright, as it was found) makes the last stage (lid closed) impossible; it has to notice the blocked lid, take
the block out again, lay it down and close the lid (within an attempt), or lay the block down from the start in this
box (across attempts).

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * a wooden box (fixture): outer 20.6 cm (forward-back) x 13.6 cm (left-right) x 9 cm high, walls 0.8 cm; the
    inside is pure black and matte (walls' inner faces, floor and false bottom). The lid (0.6 cm, wood, 0.15 kg) is
    hinged along the box's left or right long edge; a black ball knob sticks out of the middle of the opposite edge.
    Opened past upright the lid falls over to the hinge side and hangs on its stop (182-192 deg) just below
    horizontal beside the box, below every transit;
  * a red block 4 x 4 x 7.5 cm standing upright and two 4 cm blue cubes, in a row in front of the box (robot side),
    on 3 of 4 slots 16 cm apart.

Hidden variable: the box's inner depth. With a false bottom (10 of every 20 consecutive seeds) the floor is 4.5 cm
below the rim instead of 8.4 cm. The outside is identical and the inside is dark: from above (wrist camera) and from
the far camera the floor height is not readable (measured: renders of the open, empty box with and without the false
bottom are pixel-identical in both cameras). The Panda hand (21 cm along the finger axis) cannot go below the rim,
so the fingertips cannot touch the false bottom either. Upright the block is 7.5 cm tall: it fits the deep box (top 0.9 cm
below the rim) but stands 3 cm above the rim on the false bottom; lying down (4 cm) it fits both.

Why the natural action fails: the block is found standing; carried top-down by a gripper that can only turn about the
vertical, it is naturally put in standing. On a false bottom the closing lid then comes down on the block's top and
stays open at ~28-41 deg (measured; clearly visible in both cameras). What must be
learned: take the block out, lay it on its side (tip it over on the table by pushing its upper part with the closed
fingers, then pick it up lying) and put it in lying (within the attempt), or lay it down before putting it in (across
attempts).

Lid mechanics for agents (inherited from lh_lid_box's `LidSkills`, generalized to a hinge axis along x): while the
knob is held, a commanded straight move is projected onto the knob's circle around the hinge (at most 1 cm per control
step), so "up, then over the hinge, release" opens the lid and "up, then away from the hinge, release" closes it.

Protocol: cross (reset to the same instance before every attempt), k = 3. Outcome: final-state judge (lid closed, all
three items inside the box below the rim, nothing held, scene static) plus an event log for the server log
(per-item in_box achieved / lost / restored, lid opened / closed / blocked, red block laid down / stood up); F1 gives
the agent only success / failure.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task
from .lh_lid_box import SMAX, LidSkills, _fb, _lr

# ---------------------------------------------------------------- geometry / physics (box frame: hinge along local y
# at the local +x edge; the box is turned by +-90 deg so that the hinge runs forward-back on the left or right side)
BOX = "box_1"
BOX_MODEL = "lff_fbl_box"
S_IN, L_IN = 0.120, 0.190  # inner cavity across the hinge (local x -> world y) and along it (local y -> world x);
#                             roomy enough that closed fingers (~4 cm out from a held 4 cm item) stay off the walls
WALL_T, WALL_H, FLOOR_T = 0.008, 0.090, 0.006
FB_DEPTH = 0.045  # inner depth with the false bottom (floor 4.5 cm above the table)
LX, LY = S_IN + 2 * WALL_T, L_IN + 2 * WALL_T
LID_T, LID_MASS = 0.006, 0.15
KNOB_R, STEM_L = 0.013, 0.032
HINGE_DAMP, HINGE_FRIC = 0.002, 0.002
LID_SOLREF, LID_SOLIMP = "0.005 1", "0.99 0.998 0.001"
BACKOFF = 0.003  # m per control step: a hand pressing on a lid that rests on an item yields
STOP = (182.0, 192.0)  # lid opening stop (deg, uniform): opened, the lid hangs just below horizontal beside the box
#                        (free end 6.5-7 cm up), below every transit; at < 180 deg its free end and knob rise to 12-14 cm
#                        and a carried item sweeps into them (measured)
WOOD = (0.55, 0.38, 0.22, 1.0)
LID_RGBA = (0.70, 0.52, 0.32, 1.0)
DARK = (0.0, 0.0, 0.0, 1.0)  # pure black, matte (no specular): every lit shade is 0, so the floor height is invisible
KNOB_RGBA = (0.10, 0.10, 0.11, 1.0)

RED = "red_block_1"
BLUES = ("blue_cube_1", "blue_cube_2")
ITEMS = (RED,) + BLUES
RED_W, RED_H = 0.040, 0.075
CUBE_E = 0.040
RED_DENS, CUBE_DENS = 600.0, 500.0  # 72 g, 32 g
ITEM_MU = 0.8
P_FB = 0.5

IN_TOL = 0.004  # an item's footprint corners may reach this far past the inner walls (an item leaning on a wall
#                 touches it; measured: a cube dropped into the deep box ended tilted 52 deg against the front wall)
RIM_TOL = 0.004  # item top at most this far above the rim
LID_CLOSED = np.radians(5.0)
DEBOUNCE = 6  # predicate checks (every 0.1 s)
UPRIGHT_COS, LYING_COS = np.cos(np.radians(25.0)), np.cos(np.radians(65.0))

Z_CARRY = 0.16  # scripted transit height of the grasp point above the table when carrying an item (Sonnet: 11-20 cm)
Z_HAND_MIN = WALL_H - 0.024 + 0.006  # lowest grasp point over the box: the Panda hand (2.4 cm above the grasp point, 21 cm
#                                       along the finger axis) must stay above the rim
# scripted in-box slots along the box's long axis (u, forward-back, box centre = 0), all at v = 0 (centre line).
# The lying red block lies across the box (axis left-right) and is held with the fingers closing forward-back, so its
# slot is 1 cm further in: fingers 4 cm out from its centre stay 1 cm off the back wall (measured: at the upright
# slot the back finger caught the wall and dropped the block)
SLOT_U = {"red_upright": -0.0575, "red_lying": -0.0475, "blue_cube_1": 0.005, "blue_cube_2": 0.055}
START_Y = (-0.24, -0.08, 0.08, 0.24)
START_X = -0.34


def fb_box_xml(name):
    hx, hy = LX / 2, LY / 2
    col = [((0, 0, FLOOR_T / 2), (hx, hy, FLOOR_T / 2)),
           ((hx - WALL_T / 2, 0, WALL_H / 2), (WALL_T / 2, hy, WALL_H / 2)),
           ((-hx + WALL_T / 2, 0, WALL_H / 2), (WALL_T / 2, hy, WALL_H / 2)),
           ((0, hy - WALL_T / 2, WALL_H / 2), (hx - WALL_T, WALL_T / 2, WALL_H / 2)),
           ((0, -hy + WALL_T / 2, WALL_H / 2), (hx - WALL_T, WALL_T / 2, WALL_H / 2))]
    g = []
    # box: contype 2 / conaffinity 1, lid: contype 4 / conaffinity 1 -> the lid never collides with its own box (the
    # hinge limit holds it closed) but both collide with the robot, the table and the free items (as in lh_lid_box)
    for k, (pos, s) in enumerate(col):
        g.append(f'<geom name="{name}_w{k}" type="box" pos="{_fmt(pos)}" size="{_fmt(s)}" rgba="{_fmt(WOOD)}" '
                 f'friction="0.8 0.005 0.0001" {_SOLID} contype="2" conaffinity="1" group="0"/>')
    fb_top = WALL_H - FB_DEPTH
    fbh = (fb_top - FLOOR_T) / 2
    fb_pos, fb_size = (0, 0, FLOOR_T + fbh), (hx - WALL_T, hy - WALL_T, fbh)
    # the false bottom: always compiled, its collision switched on / off per instance (contype / conaffinity), so no
    # geom moves at runtime and the compiled bounding volumes stay valid (see lh_fence's refit_bvh note)
    g.append(f'<geom name="{name}_fb" type="box" pos="{_fmt(fb_pos)}" size="{_fmt(fb_size)}" rgba="{_fmt(DARK)}" '
             f'friction="0.8 0.005 0.0001" {_SOLID} contype="2" conaffinity="1" group="0"/>')
    th = WALL_T / 2
    vis = [((hx - th / 2, 0, WALL_H / 2), (th / 2, hy, WALL_H / 2), WOOD),
           ((hx - WALL_T + th / 2, 0, WALL_H / 2), (th / 2, hy - th, WALL_H / 2), DARK),
           ((-hx + th / 2, 0, WALL_H / 2), (th / 2, hy, WALL_H / 2), WOOD),
           ((-hx + WALL_T - th / 2, 0, WALL_H / 2), (th / 2, hy - th, WALL_H / 2), DARK),
           ((0, hy - th / 2, WALL_H / 2), (hx - th, th / 2, WALL_H / 2), WOOD),
           ((0, hy - WALL_T + th / 2, WALL_H / 2), (hx - WALL_T, th / 2, WALL_H / 2), DARK),
           ((0, -hy + th / 2, WALL_H / 2), (hx - th, th / 2, WALL_H / 2), WOOD),
           ((0, -hy + WALL_T - th / 2, WALL_H / 2), (hx - WALL_T, th / 2, WALL_H / 2), DARK),
           ((0, 0, FLOOR_T / 2), (hx, hy, FLOOR_T / 2), WOOD),
           ((0, 0, FLOOR_T + 0.0005), (hx - WALL_T, hy - WALL_T, 0.0005), DARK)]
    for p, s, c in vis:
        mat = f' material="{name}_dark"' if c == DARK else ""
        g.append(f'<geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}"{mat} contype="0" conaffinity="0" '
                 f'group="1"/>')
    g.append(f'<geom name="{name}_fbvis" type="box" pos="{_fmt(fb_pos)}" size="{_fmt(fb_size)}" rgba="{_fmt(DARK)}" '
             f'material="{name}_dark" contype="0" conaffinity="0" group="1"/>')
    m, L = LID_MASS, LX
    lid_c = (-L / 2, 0, LID_T / 2)
    lid_s = (L / 2, hy, LID_T / 2)
    stem = (-L - STEM_L / 2, 0, LID_T / 2)
    ball = (-L - STEM_L - KNOB_R * 0.7, 0, LID_T / 2)
    lid = [f'<inertial pos="{_fmt(lid_c)}" mass="{m}" diaginertia="{m * (LY ** 2 + LID_T ** 2) / 12:.6f} '
           f'{m * (L ** 2 + LID_T ** 2) / 12:.6f} {m * (LY ** 2 + L ** 2) / 12:.6f}"/>',
           f'<joint name="{name}_hinge" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="0 3.0" '
           f'damping="{HINGE_DAMP}" frictionloss="{HINGE_FRIC}" armature="0.0005"/>',
           # stiffer contacts for the lid plate, and solmix so that they win against the items' (default, soft) ones:
           # with soft contacts an arm pressing the lid down onto the 72 g block pushed the lid through it (measured)
           f'<geom name="{name}_lid" type="box" pos="{_fmt(lid_c)}" size="{_fmt(lid_s)}" rgba="{_fmt(LID_RGBA)}" mass="0" '
           f'friction="0.8 0.005 0.0001" solref="{LID_SOLREF}" solimp="{LID_SOLIMP}" solmix="1000" contype="4" '
           f'conaffinity="1" group="0"/>',
           f'<geom type="box" pos="{_fmt(lid_c)}" size="{_fmt(lid_s)}" rgba="{_fmt(LID_RGBA)}" contype="0" conaffinity="0" group="1"/>',
           f'<geom name="{name}_stem" type="cylinder" pos="{_fmt(stem)}" quat="0.7071 0 0.7071 0" size="0.0045 {STEM_L / 2:.4f}" '
           f'rgba="{_fmt(KNOB_RGBA)}" mass="0" friction="1 0.005 0.0001" contype="4" conaffinity="1" group="0"/>',
           f'<geom type="cylinder" pos="{_fmt(stem)}" quat="0.7071 0 0.7071 0" size="0.0045 {STEM_L / 2:.4f}" '
           f'rgba="{_fmt(KNOB_RGBA)}" contype="0" conaffinity="0" group="1"/>',
           f'<geom name="{name}_knob" type="sphere" pos="{_fmt(ball)}" size="{KNOB_R:.4f}" rgba="{_fmt(KNOB_RGBA)}" mass="0" '
           f'friction="1 0.005 0.0001" contype="4" conaffinity="1" group="0"/>',
           f'<geom type="sphere" pos="{_fmt(ball)}" size="{KNOB_R:.4f}" rgba="{_fmt(KNOB_RGBA)}" contype="0" conaffinity="0" group="1"/>']
    shell = "\n        ".join(g)
    lid_xml = "\n          ".join(lid)
    # matte near-black inside (no specular highlight): the floor height is not readable from above
    return f"""<mujoco model="{name}">
  <asset>
    <material name="{name}_dark" rgba="{_fmt(DARK)}" specular="0" shininess="0" reflectance="0"/>
  </asset>
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


class FBLidSkills(LidSkills):
    """LidSkills with the hinge axis in any horizontal direction (here forward-back), plus the task's per-step
    monitor. While the knob is held, the commanded displacement is projected onto the tangent of the knob's circle
    about the hinge (<= SMAX per control step), with radial and axial corrections."""

    def _hinge_frame(self, e):
        h, a = self.task.hinge_point(), self.task.hinge_axis()
        v = e - h
        al = float(v @ a)
        r = v - al * a
        return a, al, r, max(float(np.linalg.norm(r)), 1e-6)

    def _act(self, dpos, drot, grip):
        if self.latched:
            e = self.eef_pos()
            if self.grip < 0 or np.linalg.norm(e - self.task.knob_xyz()) > 0.035:
                self.latched = False  # opened, or the knob slipped out of the fingers
            else:
                a, al, r, rn = self._hinge_frame(e)
                t_open = np.cross(self.env.sim.data.xaxis[self.task.jid], r) / rn  # knob motion for an opening lid
                s = float(np.clip(np.asarray(dpos) @ t_open, -SMAX, SMAX))
                if s < 0 and self.task.lid_contacts():
                    s = 0.0  # compliant wrist: it does not force the lid down onto an item (else the lid tunnels through it)
                dpos = s * t_open + (self.rad0 - rn) * r / rn + (self.y0 - al) * a
                self.knob_steps += 1
        else:
            p = self.task.robot_lid_contact()
            if p is not None and self.task.lid_contacts():
                # the same limit for a hand pressing on the lid itself: no push towards closing while the lid rests on
                # an item (measured without it: pressing 5 cm "into" a lid lying on the block left the block poking
                # through the lid)
                _, _, r, rn = self._hinge_frame(p)
                t_open = np.cross(self.env.sim.data.xaxis[self.task.jid], r) / rn
                sc = float(np.asarray(dpos) @ t_open)
                if sc < 0:  # drop the closing component and yield a little (a force-limited arm gives way)
                    dpos = np.asarray(dpos) + (BACKOFF - sc) * t_open
        Skills._act(self, dpos, drot, grip)
        self.task.monitor_step()

    def set_gripper(self, close, steps=15):
        rep = Skills.set_gripper(self, close, steps)
        if close and self.holding_knob():
            _, al, _, rn = self._hinge_frame(self.eef_pos())
            self.latched, self.rad0, self.y0 = True, rn, al
            self.knob_grasps += 1
        elif not close:
            self.latched = False
        return rep


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


@register_task
class FalseBottomLid(LFFTask):
    name = "lh_false_bottom_lid"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Plan", "Restore")
    protocol = "cross"
    max_attempts = 3
    agent_max_steps = 120  # careful: lid 2 x ~6 + 3 items x ~7 = ~35 commands; laying the block down (tip + pick) ~8;
    #                        fixing a blocked lid in place (reopen, take out, tip, pick, put in, close) ~30; + looking
    instruction = ("Put the red block and the two blue cubes into the wooden box and close its lid. The lid opens by "
                   "lifting the black knob on its edge.")
    instruction_indirect = ("Pack the tall red piece and both small blue cubes into the wooden chest and shut it "
                            "(the lid lifts by the black ball on its edge).")
    agent_object_names = ["red block", "blue cube", "blue cube", "wooden box", "lid knob"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        box = register_generated("LffFblBox", fb_box_xml(BOX_MODEL), free=False)
        red = register_generated("LffFblRedBlock", box_xml("lff_fbl_red_block", (RED_W / 2, RED_W / 2, RED_H / 2),
                                                           (0.85, 0.10, 0.08, 1.0), density=RED_DENS,
                                                           friction=(ITEM_MU, 0.005, 0.0001)).replace(_SOLID, "").replace(
            'group="0" />', 'condim="4" group="0" />'))
        objs = [(RED, red, "red_region", (-0.35, -0.25, -0.33, -0.23))]
        for i, n in enumerate(BLUES):
            cat = register_generated(f"LffFblBlueCube{'AB'[i]}",
                                     box_xml(f"lff_fbl_blue_cube_{'ab'[i]}", (CUBE_E / 2,) * 3, (0.15, 0.30, 0.85, 1.0),
                                             density=CUBE_DENS, friction=(ITEM_MU, 0.005, 0.0001)).replace(_SOLID, "").replace(
                                         'group="0" />', 'condim="4" group="0" />'))
            objs.append((n, cat, f"blue_{'ab'[i]}_region", (-0.35, -0.09 + 0.16 * i, -0.33, -0.07 + 0.16 * i)))
        fx = [(BOX, box, "box_region", (-0.16, -0.01, -0.14, 0.01))]
        return write_bddl(self.name, "put the red block and the blue cubes into the box and close the lid", objs, fx)

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
        self.fb_gid = m.geom_name2id(pre + "_fb")
        self.fbvis_gid = m.geom_name2id(pre + "_fbvis")
        self.robot_gids = set(envs.robot_geom_ids(self.env))
        self.item_gids = {n: set(envs.obj_geom_ids(self.env, n)) for n in ITEMS}
        self.inst = None
        self.mon = None

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        bx, by = float(rng.uniform(-0.17, -0.14)), float(rng.uniform(-0.02, 0.02))
        hinge = "left" if rng.random() < 0.5 else "right"
        stop = float(rng.uniform(*STOP))
        perm = rng.permutation(len(START_Y))
        start = {n: [float(START_X + rng.uniform(-0.01, 0.01)), float(START_Y[perm[j]] + rng.uniform(-0.01, 0.01))]
                 for j, n in enumerate(ITEMS)}
        start_yaw = {n: float(rng.uniform(-0.3, 0.3)) for n in ITEMS}
        # balanced blocks like lh_lid_box: exactly 10 of every 20 consecutive seeds have the false bottom
        fb = bool((int(seed) * 7) % 20 < round(20 * P_FB))
        return dict(seed=int(seed), box_xy=[bx, by], hinge=hinge, stop_deg=stop, false_bottom=fb,
                    inner_depth_cm=round(100 * (FB_DEPTH if fb else WALL_H - FLOOR_T), 1), start=start,
                    start_yaw=start_yaw)

    def box_yaw(self, inst=None):
        inst = inst or self.inst
        return np.pi / 2 if inst["hinge"] == "left" else -np.pi / 2  # local +x (hinge edge) -> world +y / -y

    def apply_instance(self, inst):
        self.inst = inst
        m, d = self.env.sim.model, self.env.sim.data
        bx, by = inst["box_xy"]
        yaw = self.box_yaw(inst)
        m.body_pos[self.box_bid] = [bx, by, envs.TABLE_Z]
        m.body_quat[self.box_bid] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        m.jnt_range[self.jid] = [0.0, np.radians(inst["stop_deg"])]
        m.jnt_solref[self.jid] = [0.01, 1.0]
        fb = bool(inst["false_bottom"])
        m.geom_contype[self.fb_gid] = 2 if fb else 0
        m.geom_conaffinity[self.fb_gid] = 1 if fb else 0
        m.geom_rgba[self.fbvis_gid, 3] = 1.0 if fb else 0.0
        d.qpos[self.qa] = 0.0
        d.qvel[self.dof] = 0.0
        for n, xy in inst["start"].items():
            envs.place_upright(self.env, n, xy, yaw=inst["start_yaw"][n])
        for n in ITEMS:
            envs.set_friction(self.env, n, ITEM_MU)

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.mon = None
        self.sk = FBLidSkills(self.env, self, recorder=recorder)
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ geometry helpers
    def floor_z(self):
        """height of the surface items rest on inside the box (world z)"""
        return envs.TABLE_Z + ((WALL_H - FB_DEPTH) if self.inst["false_bottom"] else FLOOR_T)

    def rim_z(self):
        return envs.TABLE_Z + WALL_H

    def long_axis(self):
        """unit vector (world xy) along the box's long side (= the hinge axis), pointing forward"""
        return np.array([1.0, 0.0])

    def to_box(self, xy):
        """world xy -> box frame (u along the long side / forward, v across / left)"""
        return np.asarray(xy, float)[:2] - np.asarray(self.inst["box_xy"])

    def from_box(self, uv):
        return np.asarray(self.inst["box_xy"]) + np.asarray(uv, float)

    def hinge_sign(self):
        return 1.0 if self.inst["hinge"] == "left" else -1.0

    def hinge_point(self):
        bx, by = self.inst["box_xy"]
        return np.array([bx, by + self.hinge_sign() * LX / 2, envs.TABLE_Z + WALL_H])

    def hinge_axis(self):
        return np.array([1.0, 0.0, 0.0])

    def lid_q(self):
        return float(self.env.sim.data.qpos[self.qa])

    def lid_deg(self):
        return float(np.degrees(self.lid_q()))

    def knob_xyz(self):
        return self.env.sim.data.geom_xpos[self.knob_gid].copy()

    def held(self, name):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.objects_dict[name]))

    def item_axis(self, name):
        """body z axis in world (the red block's long axis)"""
        return self.env.sim.data.body_xmat[envs.body_id(self.env, name)].reshape(3, 3)[:, 2].copy()

    def red_pose(self):
        c = abs(float(self.item_axis(RED)[2]))
        return "upright" if c >= UPRIGHT_COS else "lying" if c <= LYING_COS else "tilted"

    def in_cavity_xy(self, name):
        """the item's whole footprint (its collision box corners, also when tilted) inside the inner walls"""
        m, d = self.env.sim.model, self.env.sim.data
        g = next(iter(self.item_gids[name]))
        R = d.geom_xmat[g].reshape(3, 3)
        half = m.geom_size[g][:3]
        c = self.to_box(d.geom_xpos[g][:2])
        ext = np.abs(R[:2, :]) @ half  # world-axis half extents of the footprint
        lim = np.array([L_IN / 2, S_IN / 2]) + IN_TOL
        return bool(np.all(np.abs(c) + ext <= lim))

    def in_box(self, name):
        return bool(self.in_cavity_xy(name) and envs.obj_max_z(self.env, name) <= self.rim_z() + RIM_TOL
                    and envs.obj_min_z(self.env, name) >= envs.TABLE_Z + FLOOR_T - 0.005 and not self.held(name))

    def over_box(self, name):
        """centre above the cavity (in it, or standing in it but sticking out)"""
        c = self.to_box(envs.obj_pos(self.env, name)[:2])
        return bool(abs(c[0]) <= L_IN / 2 and abs(c[1]) <= S_IN / 2)

    def lid_closed(self):
        return self.lid_q() <= LID_CLOSED

    def lid_contacts(self):
        """items the lid touches now"""
        d = self.env.sim.data
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            for n in ITEMS:
                g = self.item_gids[n]
                if (c.geom1 in g and c.geom2 in self.lid_gids) or (c.geom2 in g and c.geom1 in self.lid_gids):
                    out.add(n)
        return out

    def robot_lid_contact(self):
        """world position of a robot-lid contact (None if the robot does not touch the lid)"""
        d = self.env.sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in self.robot_gids and c.geom2 in self.lid_gids) or (c.geom2 in self.robot_gids
                                                                              and c.geom1 in self.lid_gids):
                return np.array(c.pos, float)
        return None

    def robot_touching(self):
        """'lid' / item names the robot is in contact with now (finishing while pressing the lid down does not count)"""
        d = self.env.sim.data
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.robot_gids:
                    if b in self.lid_gids:
                        out.add("lid")
                    for n in ITEMS:
                        if b in self.item_gids[n]:
                            out.add("red block" if n == RED else "blue cube")
        return sorted(out)

    def stage_values(self):
        v = {f"{n}_in_box": self.in_box(n) for n in ITEMS}
        v["lid_closed"] = self.lid_closed()
        return v

    # ------------------------------------------------------------ per-step monitor
    def _init_monitor(self):
        self.mon = dict(state={f"{n}_in_box": False for n in ITEMS}, ever=set(), lost=set(), events=[],
                        last_touch={n: None for n in ITEMS}, step=0, lid_opened=False, lid_open_max=0.0,
                        lid_prev=0.0, lid_still=0, blocked_now=False, n_blocked=0, blocked_max_deg=0.0,
                        red_pose=self.red_pose(), red_pose_cand=None, red_pose_n=0, red_up_in_box=False,
                        red_relaid=0, closes=0, cand={f"{n}_in_box": 0 for n in ITEMS})

    def _touches(self):
        d = self.env.sim.data
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            for n in ITEMS:
                g = self.item_gids[n]
                other = c.geom2 if c.geom1 in g else (c.geom1 if c.geom2 in g else None)
                if other is None:
                    continue
                if other in self.lid_gids:
                    out[n] = "lid"
                elif other in self.robot_gids and out.get(n) != "lid":
                    out[n] = "robot"
        return out

    def monitor_step(self):
        mon = self.mon
        if mon is None:
            return
        mon["step"] += 1
        step = self.sk.n_steps if self.sk is not None else mon["step"]
        for n, who in self._touches().items():
            mon["last_touch"][n] = (who, step)
        q = self.lid_deg()
        mon["lid_open_max"] = max(mon["lid_open_max"], q)
        if not mon["lid_opened"] and q > 90.0:
            mon["lid_opened"] = True
            mon["blocked_now"] = False
            mon["events"].append(dict(step=step, pred="lid", ev="opened_past_vertical"))
        elif mon["lid_opened"] and q <= np.degrees(LID_CLOSED):
            mon["lid_opened"] = False
            mon["closes"] += 1
            mon["events"].append(dict(step=step, pred="lid", ev="closed"))
        # lid blocked: it came down from past vertical and rests (not held) on an item at 5-80 deg for 1 s
        if abs(q - mon["lid_prev"]) < 0.05:
            mon["lid_still"] += 1
        else:
            mon["lid_still"] = 0
        mon["lid_prev"] = q
        if mon["lid_opened"] and not mon["blocked_now"] and 5.0 < q < 80.0 and mon["lid_still"] >= 20 \
                and not self.sk.latched:
            on = sorted(self.lid_contacts())
            if on:
                mon["blocked_now"] = True
                mon["n_blocked"] += 1
                mon["blocked_max_deg"] = max(mon["blocked_max_deg"], q)
                mon["events"].append(dict(step=step, pred="lid", ev="lid_blocked", lid_deg=round(q, 1), on=on,
                                          red_pose=self.red_pose()))
        if mon["blocked_now"] and q > 90.0:
            mon["blocked_now"] = False
        if mon["step"] % 2:
            return  # predicates every other control step (0.1 s)
        # red block pose (debounced 1 s): upright <-> lying
        p = self.red_pose()
        if p != "tilted" and p != mon["red_pose"]:
            if mon["red_pose_cand"] == p:
                mon["red_pose_n"] += 1
            else:
                mon["red_pose_cand"], mon["red_pose_n"] = p, 1
            if mon["red_pose_n"] >= 10 and not self.held(RED):
                where = "box" if self.over_box(RED) else "table"
                lt = mon["last_touch"][RED]
                cause = lt[0] if lt is not None and step - lt[1] <= 60 else "other"
                ev = "item_relaid" if p == "lying" else "stood_up"
                if p == "lying":
                    mon["red_relaid"] += 1
                mon["events"].append(dict(step=step, pred="red_pose", ev=ev, where=where, cause=cause))
                mon["red_pose"] = p
                mon["red_pose_cand"] = None
        else:
            mon["red_pose_cand"] = None
        if not mon["red_up_in_box"] and mon["red_pose"] == "upright" and self.over_box(RED) and not self.held(RED) \
                and envs.obj_min_z(self.env, RED) < self.rim_z():
            mon["red_up_in_box"] = True
            top = envs.obj_max_z(self.env, RED) - self.rim_z()
            mon["events"].append(dict(step=step, pred="red_pose", ev="placed_upright_in_box",
                                      top_above_rim_cm=round(100 * top, 1)))
        for n in ITEMS:  # debounced: a change counts after it held for DEBOUNCE checks (fingers rubbing a released item)
            key = f"{n}_in_box"
            now = self.in_box(n)
            if now == mon["state"][key]:
                mon["cand"][key] = 0
                continue
            mon["cand"][key] += 1
            if mon["cand"][key] >= DEBOUNCE:
                mon["cand"][key] = 0
                self._event(key, now, step, n)

    def _event(self, key, now, step, name):
        mon = self.mon
        mon["state"][key] = now
        if now:
            ev = "restored" if key in mon["lost"] else "achieved"
            mon["ever"].add(key)
            mon["lost"].discard(key)
            e = dict(step=step, pred=key, ev=ev)
            if name == RED:
                e["red_pose"] = self.red_pose()
            mon["events"].append(e)
            return
        if key not in mon["ever"]:
            return  # never achieved: nothing lost
        cause = "other"
        if self.held(name):
            cause = "picked_up"
        else:
            lt = mon["last_touch"][name]
            if lt is not None and step - lt[1] <= 60:
                cause = lt[0]
        mon["lost"].add(key)
        mon["events"].append(dict(step=step, pred=key, ev="lost", cause=cause, lid_deg=round(self.lid_deg(), 1)))

    # ------------------------------------------------------------ outcome
    def _static(self):
        d = self.env.sim.data
        for n in ITEMS:
            _, da = envs.free_joint_addr(self.env, n)
            if np.linalg.norm(d.qvel[da:da + 3]) > 0.02 or np.linalg.norm(d.qvel[da + 3:da + 6]) > 0.5:
                return False
        return abs(float(d.qvel[self.dof])) < 0.2

    def _where(self, name):
        """F2-style location of an item relative to the box, robot-centric words"""
        rel = self.to_box(envs.obj_pos(self.env, name)[:2])
        return f"{_fb(rel[0])} and {_lr(rel[1])} of the box's centre"

    def outcome(self, inst):
        envs.settle(self.env, 30)
        mon = self.mon
        vals = self.stage_values()
        static = self._static()
        held = [n for n in ITEMS if self.held(n)]
        touching = self.robot_touching()
        success = bool(all(vals.values()) and static and not held and not touching)
        regress = [e for e in mon["events"] if e["ev"] == "lost"]
        restored = [e for e in mon["events"] if e["ev"] == "restored"]
        on = sorted(self.lid_contacts())
        red_top = envs.obj_max_z(self.env, RED) - self.rim_z()
        parts = []
        failure = None
        if success:
            parts.append("The red block and both blue cubes are inside the box and the lid is closed.")
        else:
            out_items = [n for n in ITEMS if not vals[f"{n}_in_box"]]
            if not vals["lid_closed"] and on:
                failure = "lid_blocked"
            elif held:
                failure = "held"
            elif touching and all(vals.values()):
                failure = "robot_touching"
            elif out_items:
                failure = ("red_sticks_out" if out_items == [RED] and self.over_box(RED) and red_top > RIM_TOL
                           else "item_not_in_box")
            elif not vals["lid_closed"]:
                failure = "lid_open"
            else:
                failure = "not_static"
            if not vals["lid_closed"]:
                txt = f"The lid is not closed (open {self.lid_deg():.0f} degrees)"
                if on:
                    names = " and ".join("the red block" if n == RED else "a blue cube" for n in on)
                    txt += f": it rests on {names}"
                parts.append(txt + ".")
            for n in out_items:
                nm = "The red block" if n == RED else "A blue cube"
                if self.held(n):
                    parts.append(f"{nm} is still held by the gripper.")
                elif self.over_box(n) and envs.obj_max_z(self.env, n) > self.rim_z() + RIM_TOL:
                    top = envs.obj_max_z(self.env, n) - self.rim_z()
                    pose = self.red_pose() if n == RED else "resting"
                    parts.append(f"{nm} is in the box but sticks out {100 * top:.1f} cm above the rim ({pose}).")
                else:
                    parts.append(f"{nm} is not in the box: its centre is {self._where(n)}.")
            if touching and all(vals.values()):
                parts.append("The robot was still pressing on the " + " and ".join(touching) + " at the end.")
            if not static and all(vals.values()):
                parts.append("Something was still moving at the end.")
        if mon["n_blocked"]:
            parts.append(f"The lid came down on an item {mon['n_blocked']} time(s) during the attempt.")
        out = dict(success=success, failure=failure, stages=vals, static=bool(static), held=held, touching=touching,
                   events=mon["events"], n_regressions=len(regress), n_restored=len(restored),
                   false_bottom=bool(inst["false_bottom"]), n_lid_blocked=mon["n_blocked"],
                   lid_blocked_max_deg=round(mon["blocked_max_deg"], 1), red_relaid=mon["red_relaid"],
                   red_placed_upright_in_box=mon["red_up_in_box"], red_pose=self.red_pose(),
                   red_top_above_rim_cm=round(100 * red_top, 1), lid_on=on,
                   lid_open_max=round(mon["lid_open_max"], 1), lid_final_deg=round(self.lid_deg(), 1),
                   lid_closes=mon["closes"], knob_grasps=self.sk.knob_grasps, knob_steps=self.sk.knob_steps,
                   detail=" ".join(parts))
        return out

    # ------------------------------------------------------------ scripted primitives (perfect perception)
    def _settle_item(self, name, max_checks=20):
        """keep the arm still until `name` has stopped moving (a tipped block rolls on for a moment)"""
        d = self.env.sim.data
        _, da = envs.free_joint_addr(self.env, name)
        for _ in range(max_checks):
            if np.linalg.norm(d.qvel[da:da + 3]) < 0.01 and np.linalg.norm(d.qvel[da + 3:da + 6]) < 0.2:
                return
            self.sk.hold(2)

    def _grasp_yaw(self, name):
        """fingers across a 4 cm side: for the lying red block across its long axis, else yaw mod 90 deg"""
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, name)].reshape(3, 3)
        if name == RED and self.red_pose() != "upright":
            ax = R[:, 2]
            yaw = float(np.arctan2(ax[1], ax[0]))  # closing direction (-sin, cos) is then across the long axis
            return float((yaw + np.pi / 2) % np.pi - np.pi / 2)
        i = int(np.argmin(np.abs(R[2])))
        yaw = float(np.arctan2(R[1, i], R[0, i]))
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def _grasp_z(self, name):
        """grasp point height above the table for a top-down grasp of `name` where it is"""
        zmin = envs.obj_min_z(self.env, name) - envs.TABLE_Z
        zmax = envs.obj_max_z(self.env, name) - envs.TABLE_Z
        if name == RED and self.red_pose() == "upright":
            z = zmin + 0.0425  # 4.25 cm up the 7.5 cm block (0.5 cm above its centre)
        else:
            z = (zmin + zmax) / 2
        if self.over_box(name):  # the hand (2.4 cm above the grasp point) stays above the rim
            z = max(z, Z_HAND_MIN, WALL_H + 0.016 if zmax > WALL_H + 0.02 else 0.0)
            z = min(z, zmax - 0.004)
        return z

    def pick(self, name, z_carry=Z_CARRY, tries=2):
        """top-down grasp of `name` where it is, lift to z_carry (a missed grasp is retried once, as anyone would);
        returns the grasp point's height above the item's bottom, or None if the grasp failed"""
        sk = self.sk
        Z = envs.TABLE_Z
        for _ in range(tries):
            self._settle_item(name)
            p = envs.obj_pos(self.env, name)
            yaw = self._grasp_yaw(name)
            zg = self._grasp_z(name)
            zbot = envs.obj_min_z(self.env, name) - Z
            sk.set_gripper(False, steps=8)
            sk.move_to([p[0], p[1], Z + max(z_carry, zg + 0.06)], yaw=yaw)
            sk.hold(8)  # let a large wrist turn settle before going down (else the hand drifts ~1 cm near the table)
            sk.move_to([p[0], p[1], Z + zg], tol=0.003, max_steps=120)
            sk.set_gripper(True, steps=15)
            sk.move_to([p[0], p[1], Z + z_carry], speed=0.3)
            if self.held(name):
                return zg - zbot
            sk.set_gripper(False, steps=8)
        return None

    def place(self, dest_xy, z_rel, yaw, z_carry=Z_CARRY):
        """carry to above dest at z_carry, lower to z_rel (or until the item / hand rests on something), release, lift"""
        sk = self.sk
        Z = envs.TABLE_Z
        sk.move_to([dest_xy[0], dest_xy[1], Z + z_carry], yaw=yaw)
        sk.move_to([dest_xy[0], dest_xy[1], Z + z_rel], tol=0.003, max_steps=80)
        sk.set_gripper(False, steps=12)
        sk.move_to([dest_xy[0], dest_xy[1], Z + z_carry])

    def put_in_box(self, name, u, v=0.0, noise=None, z_carry=Z_CARRY):
        """pick `name` as it is (the red block upright or lying) and lower it into the box at (u, v) (box frame) as if
        the box were deep (the hand stops above the rim), release. The lying red block is turned across the box."""
        off = self.pick(name, z_carry=z_carry)
        if off is None:
            return False
        yaw = np.pi / 2 if name == RED and self.red_pose() == "lying" else self.sk.yaw
        d = self.from_box([u, v]) + (np.asarray(noise, float) if noise is not None else 0.0)
        self.place(d, max(Z_HAND_MIN, FLOOR_T + off + 0.005), yaw, z_carry=z_carry)
        return True

    def red_slot(self):
        return SLOT_U["red_lying"] if self.red_pose() == "lying" else SLOT_U["red_upright"]

    def tip_over(self, name, direction, z_push=0.055, speed=0.10):
        """push the standing item's upper part with the closed fingers along `direction` (xy) so that it falls over"""
        sk = self.sk
        Z = envs.TABLE_Z
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        p = envs.obj_pos(self.env, name)
        s = p[:2] - d * 0.06
        e = p[:2] + d * 0.035
        yaw = float(np.arctan2(d[1], d[0])) - np.pi / 2  # finger closing direction along the push
        sk.set_gripper(True, steps=8)
        sk.move_to([s[0], s[1], Z + 0.13], yaw=yaw)
        sk.move_to([s[0], s[1], Z + z_push], tol=0.004)
        sk.move_to([e[0], e[1], Z + z_push], speed=speed, tol=0.006)
        sk.move_to([e[0], e[1], Z + 0.13], speed=0.3)
        self._settle_item(name)
        return self.red_pose() if name == RED else None

    def _tip_direction(self, name):
        """outward along y (away from the other items, which stand 16 cm apart in a row)"""
        y = envs.obj_pos(self.env, name)[1]
        return np.array([0.0, 1.0 if y >= 0 else -1.0])

    def lay_down(self, name=RED, **kw):
        """tip the standing block over on the table (up to two pushes); True if it lies"""
        for _ in range(2):
            if self.red_pose() == "lying":
                return True
            self.tip_over(name, self._tip_direction(name), **kw)
        return self.red_pose() == "lying"

    def take_out_red(self):
        """pick the red block out of the box and stand it on its start spot"""
        off = self.pick(RED)
        if off is None:
            return False
        self.place(self.inst["start"][RED], off + 0.004, self.sk.yaw)
        self._settle_item(RED)
        return True

    def _grasp_knob(self, yaw=-np.pi / 2):
        """open the fingers, come down on the knob at wrist yaw -90 deg (fingers across the stem), close. Always -90
        (not its gripper-symmetric twin +90): with the lid hanging on its stop beside the box the arm reached the knob
        at -90 in every instance, but at +90 it stopped 2-3 cm short against the lid (validate seed 1004: the lying
        block's grasp yaw had landed on +90 by numerical chance and the final close missed the knob twice). A 180 deg
        wrist turn is avoided by passing through yaw 0 (see Skills.wrap_yaw)."""
        sk = self.sk
        k = self.knob_xyz()
        for attempt in range(2):
            sk.set_gripper(False, steps=8)
            if abs(sk.yaw - yaw) > np.pi / 2 + 0.1:
                sk.move_to([k[0], k[1], k[2] + 0.08], yaw=0.0, wrap_yaw=False)
            sk.move_to([k[0], k[1], k[2] + 0.08], yaw=yaw, wrap_yaw=False)
            sk.hold(5)
            sk.move_to([k[0], k[1], k[2]], tol=0.003)
            sk.set_gripper(True, steps=15)
            if sk.holding_knob() or attempt:
                break
            # missed (measured once in 40 validation attempts: after a long sequence of wrist turns the arm's posture
            # had drifted and it stopped 3-4 cm short of the knob hanging far to the side): open, go back to a neutral
            # pose over the box at yaw 0 (the OSC's null-space term straightens the arm there) and try once more
            sk.set_gripper(False, steps=8)
            e = sk.eef_pos()
            sk.move_to([e[0], e[1], e[2] + 0.10])
            c = self.from_box([0.0, 0.0])
            sk.move_to([c[0], c[1], envs.TABLE_Z + 0.28], yaw=0.0, wrap_yaw=False)
            sk.hold(10)
        return k

    def open_lid(self, over=0.04, up=0.16):
        """grasp the knob (fingers across its stem); straight up 10 cm; straight to `over` m beyond the hinge, `up` m
        above it; release; lift"""
        sk = self.sk
        k = self._grasp_knob()
        h = self.hinge_point()
        sg = self.hinge_sign()
        sk.move_to([k[0], k[1], k[2] + 0.10])
        sk.move_to([h[0], h[1] + sg * over, h[2] + up])
        sk.set_gripper(False, steps=10)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1] - sg * 0.04, e[2] + 0.05])
        sk.hold(25)

    def close_lid(self, away=0.08, up=0.10):
        """grasp the knob where it is; straight up 10 cm; straight to `away` m on the knob side of the hinge; release"""
        sk = self.sk
        k = self._grasp_knob()
        h = self.hinge_point()
        sg = self.hinge_sign()
        sk.move_to([k[0], k[1], k[2] + up])
        sk.move_to([h[0], h[1] - sg * away, max(k[2], h[2]) + up])
        sk.set_gripper(False, steps=10)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], e[2] + 0.06])
        sk.hold(30)

    def press_lid(self):
        """the natural retry when the lid did not shut: grasp the knob where it is and push it down towards closed"""
        sk = self.sk
        k = self._grasp_knob()
        h = self.hinge_point()
        sg = self.hinge_sign()
        sk.move_to([k[0], h[1] - sg * (LX + 0.01), h[2] - 0.01], speed=0.1)
        sk.set_gripper(False, steps=10)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], e[2] + 0.08])
        sk.hold(30)

    def _fallen(self):
        """items that ended up outside the box (not merely sticking out of it)"""
        return [n for n in ITEMS if not self.in_box(n) and not self.over_box(n) and not self.held(n)]

    def _replace_fallen(self, done, noise=None, z_carry=Z_CARRY):
        for n in [n for n in self._fallen() if n in done]:
            self.put_in_box(n, self.red_slot() if n == RED else SLOT_U[n], noise=(noise or {}).get(n), z_carry=z_carry)

    # ------------------------------------------------------------ scripted policies
    def execute(self, inst, params):
        noise = params.get("noise") or {}
        zc = params.get("z_carry", Z_CARRY)
        self.open_lid(**params.get("open", {}))
        done = []
        for n in ITEMS:
            if n == RED and params.get("lay_first"):
                self.lay_down(RED)
            self.put_in_box(n, self.red_slot() if n == RED else SLOT_U[n], noise=noise.get(n), z_carry=zc)
            done.append(n)
            self._replace_fallen(done, noise, zc)  # looks after every step: re-places an item that fell outside
        self.close_lid(**params.get("close", {}))
        if params.get("check") == "naive":
            self.naive_check(noise, zc)
        elif params.get("check") == "fix":
            self.check_and_fix()
        self.sk.hold(10)

    def naive_check(self, noise=None, z_carry=Z_CARRY):
        """Sonnet-like look before finishing, without the insight: a lid that did not shut is pushed down once (re-closed
        if it is still open past upright); an item outside the box is put back as it is (the red block upright)"""
        if not self.lid_closed():
            if self.lid_deg() > 90:
                self.close_lid()
            else:
                self.press_lid()
        if self._fallen():
            if self.lid_deg() < 90:
                self.open_lid()
            self._replace_fallen(list(ITEMS), noise, z_carry)
            self.close_lid()

    def check_and_fix(self, rounds=2):
        """look at every stage; if the red block is not inside below the rim (it stands up out of a shallow box and
        holds the lid open), take it out, lay it down on the table and put it in lying; re-place anything outside the
        box; close the lid"""
        for _ in range(rounds):
            v = self.stage_values()
            if all(v.values()) and not any(self.held(n) for n in ITEMS):
                return True
            bad = [n for n in ITEMS if not v[f"{n}_in_box"]]
            if self.lid_deg() < 90 and (bad or self.lid_contacts()):
                self.open_lid()
            for n in bad:
                if n == RED and self.red_pose() != "lying":
                    if self.over_box(RED):
                        self.take_out_red()
                    self.lay_down(RED)
                self.put_in_box(n, self.red_slot() if n == RED else SLOT_U[n])
            if self.lid_deg() > 90 or not self.lid_closed():
                self.close_lid()
        return all(self.stage_values().values())

    def default_params(self, inst):
        """Sonnet-like: open the lid, items in the listed order, each put in as found (the block upright), carried
        16 cm up, looks after every step (re-places an item that fell outside), closes the lid, pushes a lid that did
        not shut down once, finishes"""
        return dict(lay_first=False, check="naive")

    def oracle_params(self, inst):
        """knows the inner depth: lays the block down first in a shallow box; checks every stage at the end"""
        return dict(lay_first=bool(inst["false_bottom"]), check="fix")

    def adapt_params(self, inst, history):
        """attempt >= 2 (after a failure): lay the block down before putting it in, then check and fix"""
        return dict(lay_first=True, check="fix")

    def blind_params(self, inst, attempt, rng):
        """the natural behaviour again with small random variations, no history"""
        return dict(lay_first=False, check="naive",
                    noise={n: rng.uniform(-0.005, 0.005, 2).tolist() for n in ITEMS},
                    z_carry=float(rng.uniform(0.13, 0.19)),
                    open=dict(over=float(rng.uniform(0.02, 0.06)), up=float(rng.uniform(0.14, 0.18))),
                    close=dict(away=float(rng.uniform(0.06, 0.10))))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """oracle / naive: one attempt; adaptive: attempt 1 = the natural placement plus a real check and fix (takes the
        block out and lays it down when the lid is blocked), later attempts lay it down first; blind: natural with
        random variations; switch (extra control): natural, then lay it down first from attempt 2, no real check"""
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
                params = dict(lay_first=False, check="fix") if a == 0 else self.adapt_params(inst, history)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "switch":
                params = self.default_params(inst) if a == 0 else dict(lay_first=True, check="naive")
            elif kind == "oracle_noise":  # robustness check: oracle with +-5 mm placement noise
                params = dict(self.oracle_params(inst), noise={n: rng.uniform(-0.005, 0.005, 2).tolist() for n in ITEMS})
            else:
                raise ValueError(kind)
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive", "oracle_noise"):
                break
        return history
