"""LH tray_delivery v2 (long horizon, state restoration): load three bottles onto a tray in the loading area, then move
the loaded tray over a low threshold strip into the delivery area. Pushing the tray over the strip makes it rock, and
the top-heavy bottles fall; holding the tray by its handle while moving it does not.

Family: long-horizon tasks in which the agent's own normal later-stage action physically breaks earlier progress
(docs/11). Stages:
  S1-S3  stand the red / green / blue bottle upright on the tray while the tray is inside the yellow loading area;
  S4     move the tray (with the bottles) over the black threshold strip completely into the purple delivery area.
Success (F1): at the end the tray (board and handle) is completely inside the delivery area, all three bottles stand
upright on it, the gripper does not touch them, the scene is static, AND the event log shows that all three bottles
stood on the tray at the same time while the tray was still completely inside the loading area (load first, then
deliver). Re-standing a fallen bottle on the tray after the move is allowed (counts as a restoration).

v2 (2026-10-09, after the Sonnet 5.5 pilot lh1, see the task card): v1's 15.6 cm bottles on a 2.2 cm foot were hard to
grasp and to stand, the narrow handle was hard to push straight, "completely inside" needed 2 cm precision, and the
agents pushed at 3-5 cm/s, far below v1's 0.15 m/s topple threshold, so the hidden variable never mattered. Now:
- bottles 10.5 cm tall: 4 x 4 x 7.5 cm square body with a flat bottom + a 2.4 cm round neck (3 cm tall), 0.25 kg;
- T handle, 2.5 cm tall: a 16 cm push bar at the back (push it with the closed gripper) on a 3 cm wide, 5 cm long stem
  (grasp it across its width);
- delivery area 28 x 42 cm for the 18.5 x 34 cm tray footprint (4.75 / 4 cm slack per side; v1: 2 / 3 cm);
- a black threshold strip between the areas (a rounded sill 1 cm high, 5.3 cm wide, 46 cm long) that the tray has to
  cross. Measured: with stocky bottles speed alone cannot be the lever (on a flat table a top-heavy bottle stayed up in
  closed-gripper pushes up to 0.3 m/s; only the controller's top speed, ~0.65 m/s, toppled it). On the sill the tray
  first climbs (front up ~5 deg), then rocks over the crest under gravity; when it is pushed, nothing holds its trailing
  edge, which jumps up (max tray tilt 7.2-9.3 deg) and the top-heavy bottles fall over the leading edge,
  at every push speed tried (2-6 cm/s and the top speed). When the stem is grasped, the hand holds the trailing edge
  (max tilt 5.7-7.5 deg) and nothing falls. A hand hovering right over the handle (3 cm handle pushed with the
  grasp point ~1 cm above the table) also held it down; hence the 2.5 cm handle.
- failure codes report the first unmet requirement: not_loaded / order / bottle_fallen / bottle_off_tray /
  not_delivered / held / not_static.

Scene: wooden tray = an 11 x 34 x 1.2 cm board, long side running forward / back, with the T handle sticking out of the
middle of the long side that faces away from the delivery area; three bottles (same shape, red / green / blue) standing
in a row on the table on the far side of the loading area; two painted rectangles (no collision): yellow loading area
(tray inside it at the start) and purple delivery area 32-34 cm (centre to centre) to its left or right (random); the
threshold strip lies across the path in the gap between them. The tray travels sideways (the arm's comfortable
workspace is wide in y but short in x).

Hidden variable (same appearance): each bottle's centre-of-mass height (an invisible internal weight, 85 % of its
mass): "stable" 2.0-3.5 cm, "top-heavy" 6.6-7.6 cm above the bottom; 0 / 1 / 2 top-heavy bottles with p = 0.35 / 0.40 /
0.25. Measured on 28 instances (scripted loading, same loaded state for every move): closed-gripper pushes (2, 3, 4.5,
6 cm/s, top speed, 3 stops, slow approach; agent-style pushes through the tool layer with the grasp point 1.0 / 1.3 /
2.5 cm above the table) toppled 24/24 top-heavy bottles and 0/60 stable ones in every variant; grasp-and-drag at 2 and
4.5 cm/s toppled 0/84. Above a COM of 8 cm falling bottles sometimes land standing on their neck (cannot be stood up
again with yaw-only control without first knocking them over), so the range stops at 7.6 cm. Tray-table friction
0.45-0.60 is a nuisance variable; the sill's own friction (0.4) wins in its contacts (with the tray's 0.85 the tray
jammed at the sill and jumped over it, toppling nothing).
Restoration: a fallen bottle can be stood up again: grasp its round neck (fingers closing across the neck), lift it --
it swings down and hangs from the neck -- and set it down on the tray.

Simulation notes (as in v1): COM heights are set at reset through body_ipos / inertia and the bottle's collision BVH is
shifted with it (MuJoCo keeps it in the inertial frame); the tray uses the table's soft contact (solref 0.02); the
bottles' own contacts are stiffer (solref 0.005) so a bottle lowered onto the board by its neck is not pressed through.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

SOFT = 'solimp="0.9 0.95 0.001" solref="0.02 1"'  # the LIBERO table's contact parameters

# bottle: square body with a flat 4 x 4 cm bottom + round neck; body origin at the middle of the total height
BODY = (0.020, 0.020, 0.0375)  # half sizes of the body (4 x 4 x 7.5 cm)
NECK_R, NECK_H = 0.012, 0.015  # neck radius, half height (2.4 cm wide, 3 cm tall) -> 10.5 cm in total
BH = BODY[2] + NECK_H  # half of the total height
DAMP_BOT = 0.003  # free-joint damping of the bottles (air drag; damps the swing of a bottle hanging from its neck)
BOT_MASS = 0.25
W_FRAC = 0.85  # share of the mass in the hidden internal weight
MU_BOT = 1.0
COS_UP = 0.95  # upright: long axis within ~18 deg of vertical
BOT_SOLREF = (0.005, 1.0)  # the bottles win in all their contacts (priority 3)
BOT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)

# tray: board + T handle (stem + cross bar) sticking out of the middle of the trailing long side (tray frame -x)
PLATE = (0.055, 0.17, 0.006)  # 11 x 34 x 1.2 cm board
STEM = (0.025, 0.015, 0.0125)  # 5 long x 3 wide x 2.5 cm tall: grasp it across its width
CROSS = (0.0125, 0.08, 0.0125)  # 2.5 deep x 16 wide x 2.5 cm tall bar at the back: push it with the closed gripper
# Handle height 2.5 cm (measured on 6 instances x 6 push heights, see the card): the Panda palm reaches ~3-4 cm ahead
# of the fingers and ~2 cm above the grasp point. With a 3 cm handle a hand pushing low (grasp point ~1 cm above the
# table) hovered right over the handle and kept the trailing edge from jumping up on the sill (like a grasp: no topple
# in 3/6); with 2.5 cm every push with the grasp point 1-2.5 cm above the table toppled the top-heavy bottles (36/36).
# A hand pushing at >= 3 cm only touches the handle with its fingertips and can slide over it.
PLATE_TOP = envs.TABLE_Z + 2 * PLATE[2]
BACK_X = -PLATE[0] - 2 * STEM[0] - 2 * CROSS[0]  # tray-frame x of the handle's back face (-0.13)
FOOT_DX = (PLATE[0] + BACK_X) / 2  # x of the footprint centre (board + handle) relative to the board centre
# slots 1.5 cm towards the leading edge: a hand grasping the cross bar with its long side along the travel direction
# (palm 10.5 cm ahead of the grasp point) still clears the middle bottle by ~7 mm
SLOT_LX = 0.015
SLOT_LY = (-0.12, 0.0, 0.12)
MU_HANDLE = 0.6

# painted areas (inner half sizes: along travel, along the tray's long side)
LOAD_IN = (0.1125, 0.20)  # tray footprint 18.5 x 34 cm -> 2 / 3 cm slack per side
DELIV_IN = (0.14, 0.21)  # -> 4.75 / 4 cm slack per side
LINE_W = 0.008
AREA_DIST = (0.32, 0.34)  # centre to centre

# threshold strip: a capsule lying across the path, sunk into the table so that it sticks out RIDGE_H (a rounded sill
# 2 * sqrt(2 R h - h^2) = 5.3 cm wide)
RIDGE_H = 0.010
RIDGE_R = 0.040
RIDGE_HL = 0.23  # half length of the straight part (46 cm: longer than the tray)
MU_RIDGE = 0.4  # wins in its contacts (priority 4)

BOTTLES = ("bottle_red_1", "bottle_green_1", "bottle_blue_1")
COLOR_OF = {"bottle_red_1": "red", "bottle_green_1": "green", "bottle_blue_1": "blue"}
RGBA = {"red": (0.82, 0.12, 0.10, 1.0), "green": (0.12, 0.60, 0.20, 1.0), "blue": (0.12, 0.30, 0.85, 1.0)}
TRAY = "tray_1"
LOAD, DELIVER, RIDGE = "load_area_1", "deliver_area_1", "threshold_1"

# hidden COM heights above the bottle's bottom (m); calibration in the task card
HC_STABLE = (0.020, 0.035)
HC_TOPHEAVY = (0.066, 0.076)
N_TOP_P = (0.35, 0.40, 0.25)  # P(0 / 1 / 2 top-heavy bottles)
MU_TRAY = (0.45, 0.60)

DEBOUNCE = 5  # control steps a predicate must keep a new value before an event is recorded (0.25 s)

# scripted reference moves
V_NAIVE = 0.045  # the pilot agents pushed at 3-6 cm/s
V_CAREFUL = 0.02  # the LA interface's slowest speed
GRASP_TOP = 0.012  # scripted grasp point: on the neck, this far below the bottle's top (the palm is ~2 cm above it)
SAFE_Z = envs.TABLE_Z + 0.17  # scripted transits: fingertips ~5 cm above the tops of bottles standing on the tray
SAFE_Z_HELD = envs.TABLE_Z + 0.25  # ... and with a bottle hanging from the gripper: its bottom ~4 cm above them
PUSH_Z = envs.TABLE_Z + 0.015  # closed-gripper pushes: grasp point 1.5 cm above the table (fingertips ~0.5 cm)
DRAG_Z = envs.TABLE_Z + 0.015  # grasping the stem: finger pads over its whole height


def bottle_xml(name, rgba):
    zb = -BH + BODY[2]
    zn = -BH + 2 * BODY[2] + NECK_H
    neck_rgba = tuple(0.8 * c for c in rgba[:3]) + (1.0,)
    parts = [("box", zb, _fmt(BODY), rgba), ("cylinder", zn, f"{NECK_R:.5f} {NECK_H:.5f}", neck_rgba)]
    gs = []
    for k, (typ, z, size, c) in enumerate(parts):
        gs.append(f'        <geom name="{name}_g{k}" type="{typ}" pos="0 0 {z:.5f}" size="{size}" rgba="{_fmt(c)}" '
                  f'density="500" friction="{MU_BOT} 0.005 0.0001" {SOFT} group="0" />')
        gs.append(f'        <geom type="{typ}" pos="0 0 {z:.5f}" size="{size}" rgba="{_fmt(c)}" conaffinity="0" '
                  f'contype="0" group="1" />')
    return _wrap(name, "\n".join(gs), BH, float(np.hypot(BODY[0], BODY[1])))


def tray_xml(name, rgba=(0.62, 0.45, 0.27, 1.0)):
    hz = -PLATE[2] + STEM[2]  # handle bottom flush with the board's bottom
    sp = (-PLATE[0] - STEM[0], 0.0, hz)
    cp = (-PLATE[0] - 2 * STEM[0] - CROSS[0], 0.0, hz)
    hrgba = (0.30, 0.20, 0.12, 1.0)
    gs = []
    for k, (p, s, c, mu) in enumerate((((0, 0, 0), PLATE, rgba, 0.6), (sp, STEM, hrgba, MU_HANDLE),
                                       (cp, CROSS, hrgba, MU_HANDLE))):
        gs.append(f'        <geom name="{name}_g{k}" type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" '
                  f'density="500" friction="{mu} 0.005 0.0001" {SOFT} group="0" />')
        gs.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" conaffinity="0" '
                  f'contype="0" group="1" />')
    return _wrap(name, "\n".join(gs), PLATE[2], float(np.hypot(PLATE[0] - BACK_X, PLATE[1])))


def area_xml(name, inner, fill, line):
    """Flat, visual-only rectangle painted on the table (light fill + dark outline, no collisions)."""
    ix, iy = inner
    ox, oy = ix + LINE_W, iy + LINE_W
    parts = [((0, 0, 0.0004), (ox, oy, 0.0004), fill),
             ((0, iy + LINE_W / 2, 0.0010), (ox, LINE_W / 2, 0.0002), line),
             ((0, -iy - LINE_W / 2, 0.0010), (ox, LINE_W / 2, 0.0002), line),
             ((ix + LINE_W / 2, 0, 0.0010), (LINE_W / 2, oy, 0.0002), line),
             ((-ix - LINE_W / 2, 0, 0.0010), (LINE_W / 2, oy, 0.0002), line)]
    g = "\n".join(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(c)}" '
                  f'contype="0" conaffinity="0" group="1" />' for p, sz, c in parts)
    return _wrap(name, g, 0.0012, float(np.hypot(ox, oy)))


def sill_xml(name, rgba=(0.12, 0.12, 0.13, 1.0)):
    """the threshold strip: a capsule along the fixture's local y (quat turns z into y), origin on its axis; the
    fixture is placed with the axis RIDGE_R - RIDGE_H below the table top (static: no contacts with the table)"""
    q = "0.70711 0.70711 0 0"
    g = (f'        <geom name="{name}_g0" type="capsule" quat="{q}" size="{RIDGE_R:.5f} {RIDGE_HL:.5f}" '
         f'rgba="{_fmt(rgba)}" friction="{MU_RIDGE} 0.005 0.0001" {SOFT} group="0" />\n'
         f'        <geom type="capsule" quat="{q}" size="{RIDGE_R:.5f} {RIDGE_HL:.5f}" rgba="{_fmt(rgba)}" '
         f'conaffinity="0" contype="0" group="1" />')
    return _wrap(name, g, RIDGE_R, RIDGE_HL + RIDGE_R)


def _rotz2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


class TraySkills(Skills):
    """Skills whose every control step feeds the task's stage monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class TrayDelivery(LFFTask):
    name = "lh_tray_delivery"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Plan")
    protocol = "cross"
    max_attempts = 3
    # scripted primitives (~ agent commands): careful oracle 37; naive push + same-attempt repair of 1-2 bottles 46-63
    agent_max_steps = 110
    instruction = ("Load the red, green and blue bottles onto the wooden tray: stand each bottle upright on the tray "
                   "while the tray is still inside the yellow loading area. Only after all three bottles are on the "
                   "tray, move the tray with the bottles on it over the black threshold strip into the purple "
                   "delivery area. At the end the whole tray, including its handle, must be inside the purple area "
                   "with all three bottles standing upright on it.")
    instruction_indirect = ("Stand the three bottles up on the wooden board that lies in the yellow rectangle while "
                            "the board is still there; then bring the board with the bottles on it across the black "
                            "strip into the purple rectangle. At the end the whole board, including its dark handle, "
                            "must lie inside the purple rectangle with all three bottles standing on it.")
    agent_object_names = ["red bottle", "green bottle", "blue bottle", "wooden tray", "yellow loading area",
                          "purple delivery area", "black threshold strip"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs = []
        for i, b in enumerate(BOTTLES):
            c = COLOR_OF[b]
            cat = register_generated(f"LffTdBottle{c.capitalize()}", bottle_xml(f"lff_td_bottle_{c}", RGBA[c]))
            x0 = -0.25 + 0.12 * i
            objs.append((b, cat, f"bottle_{c}_region", (x0, 0.30, x0 + 0.01, 0.31)))
        tray = register_generated("LffTdTray", tray_xml("lff_td_tray"))
        objs.append((TRAY, tray, "tray_region", (0.0, -0.05, 0.01, -0.04)))
        load = register_generated("LffTdLoadArea", area_xml("lff_td_load_area", LOAD_IN, (0.98, 0.93, 0.55, 1.0),
                                                            (0.85, 0.62, 0.05, 1.0)), free=False)
        deliv = register_generated("LffTdDeliverArea", area_xml("lff_td_deliver_area", DELIV_IN,
                                                                (0.86, 0.78, 0.95, 1.0), (0.45, 0.20, 0.65, 1.0)),
                                   free=False)
        sill = register_generated("LffTdThreshold", sill_xml("lff_td_threshold"), free=False)
        fx = [(LOAD, load, "load_region", (-0.21, -0.41, -0.20, -0.40)),
              (DELIVER, deliv, "deliver_region", (0.15, -0.41, 0.16, -0.40)),
              (RIDGE, sill, "threshold_region", (0.15, 0.38, 0.16, 0.39))]
        return write_bddl(self.name, self.instruction, objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        env = self.env
        m = env.sim.model
        self.area_bid = {a: m.body_name2id(env.fixtures_dict[a].root_body) for a in (LOAD, DELIVER, RIDGE)}
        self.bid = {n: envs.body_id(env, n) for n in BOTTLES + (TRAY,)}
        mm = m._model
        self._bvh0 = {}
        for b in BOTTLES:  # compiled inertial frame + BVH (shifted with the hidden COM at every reset)
            bid = self.bid[b]
            adr, num = mm.body_bvhadr[bid], mm.body_bvhnum[bid]
            self._bvh0[b] = (m.body_ipos[bid].copy(), mm.bvh_aabb[adr:adr + num].copy(), adr, num)
        # geom -> object index for the contact checks (0..2 bottles, 3 tray)
        self._obj_of = np.full(m.ngeom, -1, int)
        for k, n in enumerate(BOTTLES + (TRAY,)):
            self._obj_of[envs.obj_geom_ids(env, n)] = k
        self._is_robot = np.zeros(m.ngeom, bool)
        self._is_robot[envs.robot_geom_ids(env)] = True
        sb = self.area_bid[RIDGE]
        self._is_sill = np.array([m.geom_bodyid[g] == sb or m.body_parentid[m.geom_bodyid[g]] == sb
                                  for g in range(m.ngeom)], bool)
        self._dof = {n: envs.free_joint_addr(env, n)[1] for n in BOTTLES + (TRAY,)}
        self.inst = None
        self.mon = None

    def sample_instance(self, seed):
        """Layout: the tray's long side runs forward / back (world x); it travels sideways (world y) from the loading
        area to the delivery area, 32-34 cm centre to centre, to the left or to the right, over the threshold strip in
        the middle of the gap between the areas. The bottles stand in a row on the table on the far side of the
        loading area. Tray frame: local +x = direction of travel, handle at local -x."""
        rng = np.random.default_rng(seed)
        s = 1.0 if rng.random() < 0.5 else -1.0  # +1: the tray travels to the robot's left (+y)
        xl, yl = float(rng.uniform(-0.18, -0.15)), float(-s * rng.uniform(0.11, 0.13))
        dist = float(rng.uniform(*AREA_DIST))
        deliver = [xl + float(rng.uniform(-0.01, 0.01)), yl + s * dist]
        yaw = s * np.pi / 2
        R = _rotz2(yaw)
        g0, g1 = LOAD_IN[0] + LINE_W, dist - DELIV_IN[0] - LINE_W  # the gap between the areas, along travel
        sill_xy = [(xl + deliver[0]) / 2, float((np.array([xl, yl]) + R @ np.array([(g0 + g1) / 2, 0.0]))[1])]
        jit = np.array([rng.uniform(-0.008, 0.008), rng.uniform(-0.005, 0.005)])  # tray frame
        tray_xy = (np.array([xl, yl]) + R @ (np.array([-FOOT_DX, 0.0]) + jit)).tolist()
        order = [int(i) for i in rng.permutation(3)]  # bottle -> position in the start row (along x)
        by = yl - s * (LOAD_IN[0] + LINE_W + 0.04)
        start = {}
        for k, b in enumerate(BOTTLES):
            start[b] = [xl + (-0.12, 0.0, 0.12)[order[k]] + float(rng.uniform(-0.008, 0.008)),
                        by + float(rng.uniform(-0.005, 0.005)), float(rng.uniform(-0.35, 0.35))]
        n_top = int(rng.choice(3, p=N_TOP_P))
        top = set(int(i) for i in rng.choice(3, n_top, replace=False))
        hc = {b: float(rng.uniform(*(HC_TOPHEAVY if k in top else HC_STABLE))) for k, b in enumerate(BOTTLES)}
        return dict(seed=int(seed), travel="left" if s > 0 else "right", area_yaw=float(yaw), load_xy=[xl, yl],
                    deliver_xy=deliver, sill_xy=sill_xy, tray_xy=tray_xy, tray_yaw=float(yaw), bottle_start=start,
                    com_height=hc, top_heavy=[b for k, b in enumerate(BOTTLES) if k in top],
                    mu_tray=float(rng.uniform(*MU_TRAY)))

    def _set_com(self, b, hc):
        """Hidden internal weight: bottle COM at height hc above its bottom (mass and inertia consistent)."""
        m = self.env.sim.model
        mm = m._model
        bid = self.bid[b]
        a, c = BODY[0], BODY[2]
        vb, vn = 8 * a * a * c, np.pi * NECK_R ** 2 * 2 * NECK_H
        zs = ((-BH + c) * vb + (-BH + 2 * c + NECK_H) * vn) / (vb + vn)  # shell COM (origin: middle of the height)
        ms, mw = BOT_MASS * (1 - W_FRAC), BOT_MASS * W_FRAC
        zw = float(np.clip((BOT_MASS * (hc - BH) - ms * zs) / mw, -BH + 0.01, BH - 0.01))
        zc = (ms * zs + mw * zw) / BOT_MASS
        ww = 0.025  # the weight is a 2.5 cm cube
        Ixx = ms / 3 * (a * a + c * c) + ms * (zs - zc) ** 2 + mw * (zw - zc) ** 2 + mw * ww * ww / 6
        Izz = ms * 2 * a * a / 3 + mw * ww * ww / 6
        m.body_mass[bid] = BOT_MASS
        m.body_ipos[bid] = [0.0, 0.0, zc]
        m.body_iquat[bid] = [1.0, 0.0, 0.0, 0.0]
        m.body_inertia[bid] = [Ixx, Ixx, Izz]
        ip0, A0, adr, num = self._bvh0[b]
        mm.bvh_aabb[adr:adr + num, :3] = A0[:, :3] - (m.body_ipos[bid] - ip0)  # BVH lives in the inertial frame
        return zc + BH

    def apply_instance(self, inst):
        self.inst = inst
        env = self.env
        m = env.sim.model
        ya = inst["area_yaw"]
        qa = [np.cos(ya / 2), 0.0, 0.0, np.sin(ya / 2)]
        for a, key in ((LOAD, "load_xy"), (DELIVER, "deliver_xy")):
            m.body_pos[self.area_bid[a]] = [inst[key][0], inst[key][1], envs.TABLE_Z + 0.0002]
            m.body_quat[self.area_bid[a]] = qa
        m.body_pos[self.area_bid[RIDGE]] = [inst["sill_xy"][0], inst["sill_xy"][1], envs.TABLE_Z + RIDGE_H - RIDGE_R]
        m.body_quat[self.area_bid[RIDGE]] = qa
        for g in np.flatnonzero(self._is_sill):  # the sill's friction wins (the tray's would otherwise: priority 2)
            m.geom_friction[g] = [MU_RIDGE, 0.005, 0.0001]
            m.geom_priority[g] = 4
        envs.set_obj_pose(env, TRAY, [inst["tray_xy"][0], inst["tray_xy"][1], envs.TABLE_Z + PLATE[2] + 0.0005],
                          yaw=inst["tray_yaw"])
        envs.set_friction(env, TRAY, inst["mu_tray"], priority=2)
        for b in BOTTLES:
            self._set_com(b, inst["com_height"][b])
            for g in envs.obj_geom_ids(env, b):  # bottle params win in all its contacts (tray, table, fingers)
                m.geom_friction[g] = [MU_BOT, 0.005, 0.0001]
                m.geom_condim[g] = 3
                m.geom_priority[g] = 3
                m.geom_solref[g] = BOT_SOLREF
                m.geom_solimp[g] = BOT_SOLIMP
            m.dof_damping[self._dof[b]:self._dof[b] + 6] = DAMP_BOT
            x, y, yaw = inst["bottle_start"][b]
            envs.set_obj_pose(env, b, [x, y, envs.TABLE_Z + BH + 0.0005], yaw=yaw)

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = TraySkills(self.env, self, recorder=recorder)
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ geometry / predicates
    def tray_pose(self):
        d = self.env.sim.data
        bid = self.bid[TRAY]
        return d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()

    def tray_footprint(self):
        p, R = self.tray_pose()
        loc = np.array([[PLATE[0], PLATE[1]], [PLATE[0], -PLATE[1]], [-PLATE[0], PLATE[1]], [-PLATE[0], -PLATE[1]],
                        [BACK_X, CROSS[1]], [BACK_X, -CROSS[1]]])
        return p[:2] + loc @ R[:2, :2].T

    @staticmethod
    def _word(v):
        """robot-centric word for a horizontal world direction"""
        if abs(v[0]) >= abs(v[1]):
            return "forward" if v[0] > 0 else "back"
        return "left" if v[1] > 0 else "right"

    def area_excess(self, key):
        """How far the tray footprint sticks out of the area (m, >0 = outside) per edge, keyed by robot-centric word."""
        c = np.asarray(self.inst[key])
        Ra = _rotz2(self.inst["area_yaw"])
        pts = (self.tray_footprint() - c) @ Ra  # area frame
        inner = LOAD_IN if key == "load_xy" else DELIV_IN
        out = {}
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            u = np.zeros(2)
            u[axis] = sign
            out[self._word(Ra @ u)] = float((sign * pts[:, axis]).max() - inner[axis])
        return out

    def tray_inside(self, key):
        return max(self.area_excess(key).values()) <= 0.0

    def _contacts(self):
        d = self.env.sim.data
        n = d.ncon
        if not n:
            return np.zeros(0, int), np.zeros(0, int)
        return np.asarray(d.contact.geom1)[:n], np.asarray(d.contact.geom2)[:n]

    def robot_touch(self):
        """indices (0..2 bottles, 3 tray) of the objects the robot touches now"""
        g1, g2 = self._contacts()
        hit = np.concatenate([self._obj_of[g2[self._is_robot[g1]]], self._obj_of[g1[self._is_robot[g2]]]])
        return set(int(i) for i in hit[hit >= 0])

    def tray_on_sill(self):
        g1, g2 = self._contacts()
        tray = self._obj_of == 3
        return bool(np.any((tray[g1] & self._is_sill[g2]) | (tray[g2] & self._is_sill[g1])))

    def bottle_state(self, b, touched=False):
        d = self.env.sim.data
        bid = self.bid[b]
        p = d.body_xpos[bid]
        up = float(d.body_xmat[bid].reshape(3, 3)[2, 2])
        tp, R = self.tray_pose()
        loc = R.T @ (p - tp)
        on_xy = abs(loc[0]) <= PLATE[0] and abs(loc[1]) <= PLATE[1]
        resting = abs(loc[2] - (PLATE[2] + BH)) < 0.012
        upright = up >= COS_UP
        return dict(upright=bool(upright), on_tray=bool(on_xy and resting), touched=bool(touched),
                    ok=bool(upright and on_xy and resting and not touched), loc=loc, pos=p.copy(), cos=up)

    def predicates(self):
        t = self.robot_touch()
        st = {b: self.bottle_state(b, k in t)["ok"] for k, b in enumerate(BOTTLES)}
        st["tray_in_delivery"] = self.tray_inside("deliver_xy")
        st["tray_in_loading"] = self.tray_inside("load_xy")
        return st, t

    def _tray_speed(self):
        da = self._dof[TRAY]
        return float(np.linalg.norm(self.env.sim.data.qvel[da:da + 2]))

    # ------------------------------------------------------------ monitor
    def _init_monitor(self):
        st, _ = self.predicates()
        self.mon = dict(stable=dict(st), cnt={k: 0 for k in st}, ever=set(k for k, v in st.items() if v),
                        lost=set(), events=[], order_step=None, last_touch={k: -10 ** 6 for k in range(4)},
                        last_tray_move=-10 ** 6, last_sill=-10 ** 6, tray_vmax_loaded=0.0, left_unloaded=None,
                        on_in_loading=set())

    def monitor_step(self):
        mon = self.mon
        if mon is None:
            return
        n = self.sk.n_steps
        st, touch = self.predicates()
        for k in touch:
            mon["last_touch"][k] = n
        v = self._tray_speed()
        if v > 0.01:
            mon["last_tray_move"] = n
        if self.tray_on_sill():
            mon["last_sill"] = n
        if all(mon["stable"][b] for b in BOTTLES):
            mon["tray_vmax_loaded"] = max(mon["tray_vmax_loaded"], v)
        for k, val in st.items():
            if val == mon["stable"][k]:
                mon["cnt"][k] = 0
                continue
            mon["cnt"][k] += 1
            if mon["cnt"][k] < DEBOUNCE:
                continue
            mon["stable"][k] = val
            mon["cnt"][k] = 0
            step = n - DEBOUNCE + 1
            if val:
                ev = "restored" if k in mon["lost"] else "achieved"
                mon["ever"].add(k)
                mon["events"].append(dict(step=step, pred=k, ev=ev))
            else:
                mon["lost"].add(k)
                e = dict(step=step, pred=k, ev="lost")
                if k == "tray_in_loading" and mon["order_step"] is None and mon["left_unloaded"] is None:
                    mon["left_unloaded"] = step  # the tray left the loading area before it was loaded
                if k in BOTTLES:
                    j = BOTTLES.index(k)
                    moving = n - mon["last_tray_move"] <= 30
                    # tray motion first: a bottle toppled by the move often falls against the moving hand
                    e["cause"] = ("threshold" if moving and n - mon["last_sill"] <= 40 else
                                  "tray_moving" if moving else
                                  "robot" if n - mon["last_touch"][j] <= 20 else "other")
                    # a bottle taken off (or knocked off by the robot) while the tray is still in the loading area:
                    # the tray is no longer loaded -> the order latch has to be earned again (unload / reload trick)
                    if (e["cause"] not in ("threshold", "tray_moving") and mon["stable"]["tray_in_loading"]
                            and mon["order_step"] is not None):
                        mon["order_step"] = None
                        mon["lost"].add("loaded_in_loading_area")
                        mon["events"].append(e)
                        e = dict(step=step, pred="loaded_in_loading_area", ev="lost")
                mon["events"].append(e)
        if mon["stable"]["tray_in_loading"]:
            mon["on_in_loading"] |= {b for b in BOTTLES if mon["stable"][b]}
        if mon["order_step"] is None and mon["stable"]["tray_in_loading"] and all(mon["stable"][b] for b in BOTTLES):
            mon["order_step"] = n
            mon["events"].append(dict(step=n, pred="loaded_in_loading_area",
                                      ev="restored" if "loaded_in_loading_area" in mon["lost"] else "achieved"))

    # ------------------------------------------------------------ outcome / F2
    def _static(self):
        d = self.env.sim.data
        for nm in BOTTLES + (TRAY,):
            da = self._dof[nm]
            if np.linalg.norm(d.qvel[da:da + 3]) > 0.01 or np.linalg.norm(d.qvel[da + 3:da + 6]) > 0.15:
                return False
        return True

    def _where(self, b, bs):
        """robot-centric description of where a bottle is (relative to the tray)"""
        if bs["on_tray"]:
            return "on the tray"
        tp, R = self.tray_pose()
        lx, ly = R[:2, :2].T @ (bs["pos"][:2] - tp[:2])
        if abs(lx) <= PLATE[0] and abs(ly) <= PLATE[1]:
            return "on the tray"
        if abs(lx) <= PLATE[0] + 0.04 and abs(ly) <= PLATE[1] + 0.04:
            return "partly on the tray"
        v = R[:2, 0] * np.sign(lx) if abs(lx) - PLATE[0] >= abs(ly) - PLATE[1] else R[:2, 1] * np.sign(ly)
        side = {"forward": "in front of", "back": "behind", "left": "to the left of",
                "right": "to the right of"}[self._word(v)]
        return f"on the table, {side} the tray"

    @staticmethod
    def _names(bs):
        cols = [COLOR_OF[b] for b in bs]
        if len(cols) == 1:
            return f"the {cols[0]} bottle"
        return "the " + ", the ".join(cols[:-1]) + f" and the {cols[-1]} bottles"

    def outcome(self, inst):
        sk = self.sk
        sk.hold(30)  # let everything settle (keeps the gripper as it is)
        mon = self.mon
        touch = self.robot_touch()
        bst = {b: self.bottle_state(b, k in touch) for k, b in enumerate(BOTTLES)}
        delivered = self.tray_inside("deliver_xy")
        static = self._static()
        order_ok = mon["order_step"] is not None
        all_ok = all(s["ok"] for s in bst.values())
        success = bool(order_ok and delivered and all_ok and static)
        ev = mon["events"]
        stage_preds = set(BOTTLES) | {"tray_in_delivery"}
        n_reg = sum(1 for e in ev if e["ev"] == "lost" and e["pred"] in stage_preds)
        n_res = sum(1 for e in ev if e["ev"] == "restored" and e["pred"] in stage_preds)
        fell_moving = sorted({e["pred"] for e in ev if e["ev"] == "lost" and e.get("cause") in ("threshold",
                                                                                                  "tray_moving")})
        fell_sill = sorted({e["pred"] for e in ev if e["ev"] == "lost" and e.get("cause") == "threshold"})
        fallen = [b for b in BOTTLES if not bst[b]["upright"]]
        off = [b for b in BOTTLES if bst[b]["upright"] and not bst[b]["on_tray"]]
        held = [b for b in BOTTLES if bst[b]["upright"] and bst[b]["on_tray"] and bst[b]["touched"]]
        # failure code = the first unmet requirement, in task order: loading -> order -> bottles -> delivery -> rest
        if not order_ok:
            failure = "order" if mon["left_unloaded"] is not None else "not_loaded"
        elif fallen:
            failure = "bottle_fallen"
        elif off:
            failure = "bottle_off_tray"
        elif not delivered:
            failure = "not_delivered"
        elif held:
            failure = "held"
        elif not static:
            failure = "not_static"
        else:
            failure = None
        # F2 text (robot-centric words, no coordinates); the first sentence names the failure
        parts = []
        if failure == "not_loaded":
            missing = [b for b in BOTTLES if b not in mon["on_in_loading"]]
            parts.append(f"Loading was not completed: {self._names(missing)} never stood on the tray while it was in "
                         f"the yellow loading area." if missing else
                         "Loading was not completed: the three bottles never stood on the tray at the same time while "
                         "it was in the yellow loading area.")
        elif failure == "order":
            parts.append("The tray left the yellow loading area before all three bottles stood on it (load all three "
                         "first, then move the loaded tray).")
        bottle_txt = []
        for b in BOTTLES:
            s, col = bst[b], COLOR_OF[b]
            if s["ok"]:
                bottle_txt.append(f"The {col} bottle stands upright on the tray.")
            elif not s["upright"]:
                bottle_txt.append(f"The {col} bottle has fallen over and lies {self._where(b, s)}.")
            elif not s["on_tray"]:
                bottle_txt.append(f"The {col} bottle stands upright {self._where(b, s)}, not on the tray.")
            else:
                bottle_txt.append(f"The {col} bottle stands on the tray but the gripper is still touching it.")
        if delivered:
            tray_txt = "The tray is completely inside the purple delivery area."
        else:
            ex = self.area_excess("deliver_xy")
            out = [f"{max(100 * v, 0.1):.1f} cm beyond its {k} edge" for k, v in ex.items() if v > 0]
            tray_txt = "The tray is not completely inside the purple delivery area: it sticks out " + \
                " and ".join(out) + "."
        if failure in ("not_delivered",):
            parts += [tray_txt] + bottle_txt
        else:
            parts += bottle_txt + [tray_txt]
        if fell_moving:
            how = " while the tray crossed the threshold strip" if fell_sill == fell_moving else \
                " while the tray was moving"
            txt = self._names(fell_moving)
            parts.append(txt[0].upper() + txt[1:] + f" fell over{how}.")
        if failure == "not_static":
            parts.append("Something was still moving at the end.")
        return dict(success=success, failure=failure,
                    stages={**{COLOR_OF[b]: bool(bst[b]["ok"]) for b in BOTTLES}, "tray_in_delivery": bool(delivered),
                            "loaded_in_loading_area": bool(order_ok), "static": bool(static)},
                    events=list(ev), n_regressions=int(n_reg), n_restored=int(n_res),
                    order_step=mon["order_step"], left_loading_unloaded=mon["left_unloaded"],
                    tray_vmax_loaded=round(mon["tray_vmax_loaded"], 3),
                    fell_while_moving=[COLOR_OF[b] for b in fell_moving],
                    fell_at_threshold=[COLOR_OF[b] for b in fell_sill], detail=" ".join(parts))

    # ------------------------------------------------------------ scripted primitives (move_to + gripper only)
    def _slot_world(self, j):
        tp, R = self.tray_pose()
        return tp[:2] + R[:2, :2] @ np.array([SLOT_LX, SLOT_LY[j]])

    @staticmethod
    def _yaw_window(a):
        """the gripper-symmetric equivalent of yaw `a` in [-0.5, pi - 0.5) (keeps the last wrist joint off its limit)"""
        return float((a + 0.5) % np.pi - 0.5)

    def tray_yaw(self):
        return _yaw_of(self.tray_pose()[1])

    def transit(self, xy, z, yaw=None, z_safe=SAFE_Z):
        """up to z_safe (if lower), across, then down to z"""
        sk = self.sk
        e = sk.eef_pos()
        if e[2] < z_safe - 0.01:
            sk.move_to([e[0], e[1], z_safe], speed=0.3)
        sk.move_to([xy[0], xy[1], z_safe], yaw=None if yaw is None else self._yaw_window(yaw), wrap_yaw=False)
        sk.move_to([xy[0], xy[1], z], tol=0.003, max_steps=200)

    def _travel_dir(self):
        tp, R = self.tray_pose()
        u = R[:2, 0] / np.linalg.norm(R[:2, 0])
        foot = tp[:2] + R[:2, :2] @ np.array([FOOT_DX, 0.0])
        return tp, R, u, float((np.asarray(self.inst["deliver_xy"]) - foot) @ u)

    def push_tray(self, speed, gap=0.03, segments=1, z=PUSH_Z, yaw=None, approach=None):
        """the pilot agents' move: closed gripper `gap` behind the handle's cross bar (hand's long side along the bar),
        push along the travel direction until the footprint centre is level with the delivery area's centre;
        `segments` > 1 stops in between; `approach` (m/s): first creep up to the bar at that speed, stop, then push"""
        sk = self.sk
        tp, R, u, dist = self._travel_dir()
        back = tp[:2] + R[:2, :2] @ np.array([BACK_X, 0.0])
        if yaw is None:  # yaw 0 = fingers closing along world y; this yaw puts the hand's long side across u
            yaw = float(np.arctan2(u[1], u[0]))
        s0 = back - u * (0.012 + gap)  # the closed fingers are ~2.4 cm thick along the push
        sk.set_gripper(True, steps=8)
        self.transit(s0, z, yaw=yaw)
        if approach:
            s0 = back - u * 0.010  # just touching
            sk.move_to([s0[0], s0[1], z], speed=approach, tol=0.003, max_steps=int(gap / approach / 0.05) + 40)
            sk.hold(6)
            gap = 0.0
        total = gap + dist
        for k in range(segments):
            e = s0 + u * total * (k + 1) / segments
            sk.move_to([e[0], e[1], z], speed=speed, tol=0.004,
                       max_steps=int(total / segments / speed / 0.05) + 60 if speed else 250)
            sk.hold(6)
        e = sk.eef_pos()[:2] - u * 0.03
        sk.move_to([e[0], e[1], z], speed=0.1)
        sk.move_to([e[0], e[1], z + 0.15], speed=0.25)

    def drag_tray(self, speed, z=DRAG_Z, cross=False):
        """the careful move: grasp the handle and move the tray with it (the hand keeps the trailing edge down on the
        sill). Default: the stem, fingers closing across its 3 cm width (hand's long side along the tray's long side);
        cross=True: the cross bar, fingers closing across its 2.5 cm depth (hand's long side along the travel)."""
        sk = self.sk
        tp, R, u, dist = self._travel_dir()
        lx = -PLATE[0] - 2 * STEM[0] - CROSS[0] if cross else -PLATE[0] - STEM[0]
        c = tp[:2] + R[:2, :2] @ np.array([lx, 0.0])
        yaw = float(np.arctan2(u[1], u[0])) + (np.pi / 2 if cross else 0.0)
        sk.set_gripper(False, steps=8)
        self.transit(c, z, yaw=yaw)
        sk.set_gripper(True, steps=15)
        e = c + u * dist
        sk.move_to([e[0], e[1], z], speed=speed, tol=0.004,
                   max_steps=int(dist / speed / 0.05) + 80 if speed else 250)
        sk.hold(6)
        sk.set_gripper(False, steps=12)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], e[2] + 0.15], speed=0.25)

    def pick_upright(self, b, jitter=None, yaw=0.0):
        """grasp the neck of a standing bottle from above (the palm stays above the bottle's top)"""
        sk = self.sk
        p = envs.obj_pos(self.env, b)
        top = envs.obj_max_z(self.env, b)
        xy = p[:2] + (np.zeros(2) if jitter is None else jitter)
        sk.set_gripper(False, steps=8)
        self.transit(xy, top - GRASP_TOP, yaw)
        sk.set_gripper(True, steps=15)
        sk.move_to([xy[0], xy[1], SAFE_Z_HELD], speed=0.25)
        return sk.eef_pos()[2] - envs.obj_min_z(self.env, b)  # grasp height actually held

    def place_at(self, b, xy, held_h, release=0.006):
        sk = self.sk
        self.transit(xy, PLATE_TOP + release + held_h + 0.03, z_safe=SAFE_Z_HELD)
        sk.hold(10)  # let a hanging bottle stop swinging
        sk.move_to([xy[0], xy[1], PLATE_TOP + release + held_h], speed=0.08, tol=0.003, max_steps=150)
        sk.hold(4)
        sk.set_gripper(False, steps=12)
        sk.move_to([xy[0], xy[1], SAFE_Z], speed=0.25)

    def load_all(self, rng=None):
        """every bottle goes to the tray slot on the same side as its start position (a carried bottle never passes
        over a standing one); the hand's long side across the row. With `rng`: +-5 mm on every grasp / placement."""
        tp, R = self.tray_pose()
        ay = R[:2, 1]
        yaw = self.tray_yaw() - np.pi / 2
        by_y = sorted(BOTTLES, key=lambda b: float(np.asarray(self.inst["bottle_start"][b][:2]) @ ay))
        for j, b in enumerate(by_y):
            jg, jp = (None, None) if rng is None else (rng.uniform(-0.005, 0.005, 2), rng.uniform(-0.005, 0.005, 2))
            h = self.pick_upright(b, jitter=jg, yaw=yaw)
            self.place_at(b, self._slot_world(j) + (0 if jp is None else jp), h)

    def free_slot(self, exclude):
        """a tray slot at least 7 cm from every other bottle (standing or lying); among those the one farthest from the
        robot; else the emptiest slot"""
        tp, R = self.tray_pose()
        others = [R[:2, :2].T @ (self.bottle_state(o)["pos"][:2] - tp[:2]) for o in BOTTLES if o not in exclude]
        cand = []
        for j, ly in enumerate(SLOT_LY):
            dmin = min([np.linalg.norm(np.array([SLOT_LX, ly]) - o) for o in others] + [1.0])
            cand.append((dmin >= 0.07, float(self._slot_world(j)[0]) if dmin >= 0.07 else dmin, j))
        return self._slot_world(max(cand)[2])

    def restand(self, b):
        """a fallen bottle: grasp its neck across (fingers closing across the neck), lift -- it swings down and hangs
        from the neck -- and set it down on a free slot. A bottle standing off the tray: pick it up by the neck."""
        sk = self.sk
        st = self.bottle_state(b)
        if st["upright"]:
            h = self.pick_upright(b, yaw=self.tray_yaw() - np.pi / 2)
            self.place_at(b, self.free_slot({b}), h)
            return
        for tries in range(2):
            st = self.bottle_state(b)
            R = self.env.sim.data.body_xmat[self.bid[b]].reshape(3, 3)
            ax = R[:, 2]  # bottle axis (bottom -> neck)
            u = ax[:2] / max(np.linalg.norm(ax[:2]), 1e-6)
            neck = st["pos"] + ax * (BH - GRASP_TOP)
            yaw = float(np.arctan2(u[1], u[0]))  # fingers close across the neck
            sk.set_gripper(False, steps=8)
            self.transit(neck[:2], max(neck[2], envs.TABLE_Z + 0.008), yaw)
            sk.set_gripper(True, steps=15)
            if sk.holding(b) or tries:
                break
            # a finger landed on the tray's edge next to the neck: slide the bottle 5 cm away from the tray, retry
            tp = self.tray_pose()[0]
            v = st["pos"][:2] - tp[:2]
            v = v / max(np.linalg.norm(v), 1e-6)
            sk.set_gripper(False, steps=8)
            sk.move_to(sk.eef_pos() + [0.0, 0.0, 0.05], speed=0.1)
            self.push_point(st["pos"][:2] - v * 0.015, v, 0.065, speed=0.05, z=PLATE_TOP + 0.012)
        sk.move_to([neck[0], neck[1], SAFE_Z_HELD], speed=0.15)
        sk.hold(20)
        held = sk.eef_pos()[2] - envs.obj_min_z(self.env, b)
        self.place_at(b, self.free_slot({b}), held)

    def push_point(self, p, u, dist, speed=0.03, z=None):
        """closed gripper: from 3 cm behind point `p` (direction `u`) push `dist` metres along u, back off, lift"""
        sk = self.sk
        z = PUSH_Z if z is None else z
        s0 = np.asarray(p) - u * 0.03
        sk.set_gripper(True, steps=8)
        self.transit(s0, z, yaw=float(np.arctan2(u[1], u[0])))
        e = s0 + u * (0.018 + dist)
        sk.move_to([e[0], e[1], z], speed=speed, tol=0.003, max_steps=600)
        sk.hold(4)
        e = sk.eef_pos()[:2] - u * 0.03
        sk.move_to([e[0], e[1], z], speed=0.1)
        sk.move_to([e[0], e[1], SAFE_Z], speed=0.25)

    def fix_tray(self, speed=V_CAREFUL):
        """straighten the tray by slowly pushing the trailing corner that lags behind, then grasp the handle's stem
        and move the tray, without turning it, until its footprint is centred in the delivery area"""
        sk = self.sk
        y1 = self.inst["area_yaw"]
        u1 = _rotz2(y1)[:, 0]
        for _ in range(3):
            err = (self.tray_yaw() - y1 + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < np.radians(3):
                break
            tp, R = self.tray_pose()
            corners = [tp[:2] + R[:2, :2] @ np.array([-PLATE[0], e * (PLATE[1] - 0.02)]) for e in (1.0, -1.0)]
            a = [float((c - tp[:2]) @ u1) for c in corners]
            k = int(np.argmin(a))
            self.push_point(corners[k], u1, 0.6 * (max(a) - min(a)), speed=speed)
        tp, R = self.tray_pose()
        h0 = tp[:2] + R[:2, :2] @ np.array([-PLATE[0] - STEM[0], 0.0])
        foot = tp[:2] + R[:2, :2] @ np.array([FOOT_DX, 0.0])
        h1 = h0 + (np.asarray(self.inst["deliver_xy"]) - foot)
        d = float(np.linalg.norm(h1 - h0))
        if d < 0.004:
            return
        sk.set_gripper(False, steps=8)
        self.transit(h0, DRAG_Z, yaw=self.tray_yaw())  # fingers close across the stem's width
        sk.set_gripper(True, steps=15)
        sk.move_to([h1[0], h1[1], DRAG_Z], speed=speed, tol=0.003, max_steps=int(d / speed / 0.05) + 80)
        sk.hold(5)
        sk.set_gripper(False, steps=12)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], SAFE_Z], speed=0.2)

    def check_and_restore(self, rounds=3):
        """the final check: bottles not upright on the tray -> stand them up on a free slot (farthest from the robot
        first); then, if the tray is not completely in the delivery area, move it there slowly by the handle"""
        for _ in range(rounds):
            bad = [b for b in BOTTLES if not self.bottle_state(b)["ok"]]
            if not bad and self.tray_inside("deliver_xy"):
                return
            for b in sorted(bad, key=lambda o: -float(self.bottle_state(o)["pos"][0])):
                self.restand(b)
            self.sk.hold(10)
            if not self.tray_inside("deliver_xy"):
                self.fix_tray()
                self.sk.hold(10)

    def execute(self, inst, params):
        rng = np.random.default_rng(params.get("seed")) if params.get("jitter") else None
        self.load_all(rng)
        if params.get("mode", "push") == "drag":
            self.drag_tray(params["speed"])
        else:
            self.push_tray(params["speed"], gap=params.get("gap", 0.03), segments=params.get("segments", 1))
        if params.get("check"):
            self.check_and_restore()

    def default_params(self, inst):
        """naive = the pilot agents' style: load all three, push the handle with the closed gripper at ~4.5 cm/s"""
        return dict(speed=V_NAIVE, mode="push", check=False)

    def oracle_params(self, inst):
        return dict(speed=V_CAREFUL, mode="drag", check=True)

    def adapt_params(self, inst, history):
        return dict(speed=V_CAREFUL, mode="drag", check=True)

    def blind_params(self, inst, attempt, rng):
        return dict(speed=float(rng.uniform(0.03, 0.06)), mode="push", segments=int(rng.integers(1, 3)),
                    check=False, jitter=True, seed=int(rng.integers(1 << 30)))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """cross protocol; adaptive = naive + check-and-restore in attempt 1, careful (drag by the handle) + check
        afterwards"""
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
                params = dict(self.default_params(inst), check=True) if a == 0 else self.adapt_params(inst, history)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
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
