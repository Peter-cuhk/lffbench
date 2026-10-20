"""LH lh_wobbly_table (2026-10-16): stand four bottles on the four coloured spots of a small side table -- one of its
legs is 1.5 cm short. Set the heavy bottle down on the corner over that leg while other bottles already stand on the
table and the table rocks down onto the short leg with a knock that throws the tall light bottles over.

Family: long-horizon self-disturbance ("state_restoration"): four plain pick-and-place stages, but a later stage (the heavy
bottle on the wobbly corner) physically undoes the earlier ones without the robot touching them. The agent has to
notice the fallen bottles and stand them up again (the rocked table is stable, so they stay up), or learn across
attempts to put that bottle down first (the table then rocks while nothing stands on it).

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * a small wooden side table (free body, 1.1 kg): 28 x 28 x 1.6 cm top, 8.5 cm high, four 1.6 cm square legs at
    +-12.5 cm; centre 53-57 cm in front of the robot base, +-4 cm left/right; four coloured spots (7.2 x 6.4 cm,
    painted, colours permuted) centred 8 cm from the middle on both axes (one per corner);
  * four square glass bottles (red / blue / green / yellow; 3.2 x 3.2 x 10 cm body, 2.4 cm flat foot, shoulder, 1.4 cm
    square neck, a thin stem and a 1.2 cm white ball stopper; 14.7 cm tall) standing in a row on the main table in
    front of the side table or beside it (random). Three are empty and light (75-90 g, centre of mass 7.2-8.0 cm up:
    they tip over beyond ~9 deg); the fourth looks the same but is heavy (0.34-0.38 kg, sand in the bottom, centre of
    mass 3.0-3.6 cm up).

Hidden variables: which leg is short (its collision capsule is 14-16 mm shorter; the visual leg is drawn full length,
so all legs look alike and touch the table) and which bottle is heavy. Coupled by construction: the heavy bottle's spot
is the corner over the short leg. Balanced over seeds: in the scripted naive order (spots farthest from the supply
first, then left first) the heavy bottle's spot comes 1st / 2nd / 3rd / 4th for exactly 5 of every 20 consecutive seeds
(1st = benign for that order: the table rocks while it is empty).

Physics (measured, scratch scripts t2-t24, see the task card and results/lh_wobbly_table/physics.json):
  * The legs' collision geoms are capsules (rounded feet). With flat 1.6 cm box feet the rocking axis moved 1.1 cm
    towards the loaded corner and the table's own weight stopped the rock half way (0.5 deg, balanced on two legs).
  * The empty table rests level on its three long legs: its centre of mass is 5 mm towards the corner opposite the
    short leg (1.1 kg x 5 mm). The light bottles on their own spots (the other three corners) never tip it, also let go
    4 cm above the top with a 2.6 cm error towards the short corner (4/4 seeds); the heavy bottle on the short corner
    does (0.36 kg x 11.3 cm), also against a light bottle on the opposite corner. (A light bottle put on the short
    corner's spot -- a wrong spot -- also tips the empty table: 85 g x 11.3 cm > 1.1 kg x 5 mm.) Rocked onto the short
    leg the table tilts 4.5-5.1 deg and stays there with the heavy bottle on it, also when the light ones are added;
    lifting the heavy bottle off, or pressing 1.25 N down near the raised corner, rocks it back (measured).
  * The knock topples the light bottles standing on the table: 18/18 with teleported (zero-velocity) placements at gaps
    13-18 mm (12 mm: 8-10/18, 10 mm: 0/18); they fall towards the low corner, some off the table edge. The heavy bottle
    stays up. The robot does not touch the bottles that fall (logged per rock).
  * Bottle and table-top contacts are overdamped (solref 0.01, damping ratio 3): with stiff critically damped contacts
    a light bottle let go 1-4 cm above the ROCKED table landed on its uphill edge and fell over (21/27 at 3.4 deg);
    damped, 0/54 at every gap 13-18 mm, so the cross-attempt fix (heavy bottle first) works with the agent's usual
    release 1-4 cm above the surface.
  * A 22 x 22 cm draft table (spots 15.6 cm apart diagonally, 15.7 cm draft bottles) let the bottle on the opposite corner
    fall onto the heavy one and lean on it; its stopper could not be grasped without knocking the heavy bottle over
    (after which the table rocks back and forth on every repair step). At 28 cm nothing leans on it.
  * Grasps: the Panda hand's collision hull starts 3 cm above the grasp point and the finger pads reach 1.2 cm above
    it. Everything above the square neck is narrower than the neck: fingers that overlap the neck (grasp point 0.4-0.9
    cm below its top) close on its flat faces and hold the bottle rigidly (the heavy one stayed in the fingers through
    three moves at the controller's top speed 12/12, a light one 11/12); fingers on the ball alone (grasp point at or
    above the neck's top) touch it at one point each and the bottle pivots (top-speed carries lost it 6/6). Drafts: a
    1.8 cm ball on a 1.3 cm round neck (the fingers always closed on the ball; the heavy bottle swung out in 5/6 fast
    carries and once in the first smoke test), a 1.2 x 2.2 cm cylinder stopper (line contacts: a lying bottle lifted by
    it stuck at ~26 deg and fell over when set down). The instruction states which grip is rigid and which swings (as
    lh_bystander states its cap).
  * Restoring (yaw-only top-down gripper): grasp the ball stopper of a lying bottle from above with the fingers across
    its axis (torsional friction 0.0005), lift: it swings down and hangs straight (0.1 deg); lower it onto its spot and
    let go. A bottle that fell off onto the main table right beside the small table often cannot be reached this way:
    the hand (20 cm wide along the finger axis, 3 cm above the grasp point) would hit the table top. Where the hand,
    the open fingers or the forearm would touch a standing bottle, the scripted repair first lifts the fallen bottle by
    its body and lays it down in free space.
  * Reach: the arm stalls 2-3 cm short of low targets beyond ~0.72 m from the base; every scripted grasp point stays
    within 0.70 m.

Success predicates (evaluated every other control step by a monitor, debounced over 0.5 s): each bottle stands upright
(tilt <= 10 deg from vertical; on the rocked table they stand at ~5.5 deg) on the small table's top with the centre of
its bottom inside its spot shrunk by 6 mm (3.0 cm forward-back, 2.6 cm left-right of the spot's centre), not held.
Success = all four at the end, gripper open and not touching a bottle, scene static, and the small table within 3 cm /
10 deg of where it stood and standing on its legs (tilt <= 8 deg; the rocked tilt counts as standing). The monitor logs
achieved / lost / restored per bottle and table_rocked / table_rocked_back events (the bottle released just before,
the bottles standing at that moment, which of them fell, whether the robot touched them before / while they fell) for
the server log only (F1 gives the agent only success / failure).

Protocol: cross (reset to the same instance before every attempt), k = 3.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

TZ = envs.TABLE_Z
COLOURS = ["red", "blue", "green", "yellow"]
BOTTLES = [f"bottle_{c}_1" for c in COLOURS]
TABLE = "side_table_1"
RGBA = {"red": (0.80, 0.14, 0.12, 1.0), "blue": (0.15, 0.33, 0.80, 1.0), "green": (0.14, 0.58, 0.24, 1.0),
        "yellow": (0.90, 0.74, 0.12, 1.0)}
SPOT_RGBA = {"red": (0.62, 0.12, 0.11, 1.0), "blue": (0.12, 0.22, 0.58, 1.0), "green": (0.10, 0.42, 0.17, 1.0),
             "yellow": (0.70, 0.56, 0.08, 1.0)}
CAP_RGBA = (0.93, 0.93, 0.91, 1.0)  # white round stopper
WOOD = (0.52, 0.36, 0.22, 1.0)
LEG_RGBA = (0.40, 0.27, 0.16, 1.0)

# ---------------------------------------------------------------- side table (free body)
TOP_HW = 0.140  # 28 x 28 cm top
TOP_T = 0.016
TABLE_H = 0.085  # top surface above the main table
LEG_HW = 0.008  # 1.6 x 1.6 cm legs (visual); collision: capsules of radius 8 mm (rounded feet)
LEG_O = 0.125  # leg centre offset from the table centre (x and y)
LEG_L = TABLE_H - TOP_T
TBL_MASS = 1.1
TBL_SOLREF = 'solimp="0.95 0.99 0.001" solref="0.004 1"'  # legs: stiff (a sharp knock when the short leg lands)
TOP_SOLREF = 'solimp="0.95 0.99 0.001" solref="0.01 3"'  # top: overdamped like the bottles
LEG_MU = 1.0
TOP_MU = 0.8
# corners in the table frame (= world axes; +x forward, +y left): index -> (sx, sy); k and k + 2 are diagonal
CORNERS = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
CORNER_NAME = {(1, 1): "far-left", (1, -1): "far-right", (-1, -1): "near-right", (-1, 1): "near-left"}
SPOT_O = 0.080  # spot centre offset from the table centre (x and y)
SPOT_HX, SPOT_HY = 0.036, 0.032  # spot half sizes (forward-back, left-right): 7.2 x 6.4 cm
SHORT_H = (0.014, 0.016)  # hidden: one leg this much shorter (collision only; the visual leg reaches the table)
COM_BIAS = 0.005  # table centre of mass this far towards the leg opposite the short one: the empty table rests level

# ---------------------------------------------------------------- bottles (square glass bottles with a round stopper)
RB, HB = 0.016, 0.100  # body half width, height (3.2 x 3.2 x 10 cm)
RF, HF = 0.012, 0.005  # foot: flat contact 2.4 x 2.4 cm (rounded glass base)
RS, HS = 0.011, 0.010  # shoulder
RN, HN = 0.007, 0.016  # square neck: 1.4 cm wide, 1.6 cm long (11.0-12.6 cm above the bottom)
RT, HT = 0.0035, 0.010  # thin round stem above it: 0.7 cm wide, 1.0 cm long
RK, CAP_SINK = 0.006, 0.001  # ball stopper: 1.2 cm wide, on the stem (top 14.7 cm above the bottom)
# Why this top (measured, t8_hand.py / t19_slip.py / t22_restand.py): the Panda hand's collision hull starts ~3 cm above
# the grasp point and the finger pads reach 1.2 cm above it. Everything above the square neck is narrower than the
# neck, so fingers that overlap the neck close on its flat faces (several contact points: the bottle is held rigidly),
# and fingers on the ball alone touch it at one point each (low torsional friction: a lying bottle swings down and
# hangs straight). Drafts: a 1.8 cm ball on a 1.3 cm round neck -- the fingers always closed on the ball, the heavy
# bottle pivoted and swung out in 5/6 fast carries; a 1.2 x 2.2 cm cylinder stopper -- line contacts, a lifted lying
# bottle stuck at ~26 deg and fell over when set down (in-attempt repair 6/15).
H_TOT = HB + HS + HN + HT + 2 * RK - CAP_SINK  # 14.7 cm
CAP_Z = H_TOT - RK  # stopper centre above the bottle's bottom
NECK_Z = (HB + HS, HB + HS + HN)  # square neck span above the bottom
BOT_MU = 0.7
CAP_MU_T = 0.0005  # torsional friction of the stopper (condim 4): held by it, a lying bottle swings down and hangs
BOT_DAMP = 0.003  # free-joint damping of the bottles (damps the swing of a bottle hanging from its stopper)
# overdamped contacts (damping ratio 3): with critically damped stiff contacts a light bottle set down 1-4 cm above a
# table top tilted 3.4 deg landed on its uphill edge and fell over (measured 21/27); damped: 0/54 at tilts up to 5 deg
BOT_SOLREF = 'solimp="0.95 0.99 0.001" solref="0.01 3"'
LIGHT_MASS = (0.075, 0.090)
LIGHT_COM = (0.072, 0.080)  # above the bottom (empty glass bottle, thick shoulder)
HEAVY_MASS = (0.34, 0.38)
HEAVY_COM = (0.030, 0.036)  # sand in the bottom

# ---------------------------------------------------------------- predicates / monitor
TILT_OK = np.radians(10.0)  # a bottle stands upright (world vertical; the rocked table itself tilts ~4.3 deg)
TOL_X = SPOT_HX - 0.006  # bottle centre within the spot shrunk by 6 mm (3.0 cm forward-back, 2.6 cm left-right)
TOL_Y = SPOT_HY - 0.006
ON_TOP = 0.006  # the bottle's bottom within this of the table top
HYST = 0.003
DEBOUNCE = 10  # control steps (0.5 s)
ROCK_ON, ROCK_OFF = np.radians(1.5), np.radians(0.8)  # table tilt thresholds of the rock detector
FELL_WINDOW = 40  # control steps after a rock in which lost stages are attributed to it (2 s)
TABLE_MOVE_TOL = 0.03  # success: the table centre within 3 cm of where it stood
TABLE_YAW_TOL = np.radians(10.0)
TABLE_TILT_MAX = np.radians(8.0)

# ---------------------------------------------------------------- scripted motion
Z_CARRY = 0.40  # grasp point above the main table while carrying (bottle bottom ~27 cm up: clears standing bottles)
Z_HANG = 0.42  # ... carrying a bottle that hangs from its stopper (its bottom 27 cm up)
GRASP_NECK = 0.006  # scripted transport grasp point: this far below the neck's top (pads 1.1 cm on the neck, the hand
#                    3 mm above the stopper; measured: 4-9 mm below the top hold rigidly, 4 mm above it close on the ball)
NAIVE_GAP = (0.010, 0.040)  # Sonnet releases 1-4 cm above the surface
NAIVE_BIAS = (0.008, 0.020)  # systematic forward placement error (lh_nudge_row pilot), per instance
NAIVE_RAND = 0.006
CAREFUL_GAP = 0.004


def bottle_xml(name, rgba):
    """square glass bottle: rounded foot (smaller flat contact), body, shoulder, square neck, round stopper. Collision and
    visual geoms separate; the inertial (mass, COM) is replaced per instance."""
    h = H_TOT / 2
    zf, zb = -h + HF / 2, -h + HF + (HB - HF) / 2
    zs, zn = -h + HB + HS / 2, -h + HB + HS + HN / 2
    zt = -h + HB + HS + HN + HT / 2
    zk = -h + HB + HS + HN + HT + RK - CAP_SINK
    parts = [("foot", "box", (0, 0, zf), (RF, RF, HF / 2), rgba, f'friction="{BOT_MU} 0.005 0.0001"'),
             ("body", "box", (0, 0, zb), (RB, RB, (HB - HF) / 2), rgba, f'friction="{BOT_MU} 0.005 0.0001"'),
             ("shoulder", "box", (0, 0, zs), (RS, RS, HS / 2), rgba, f'friction="{BOT_MU} 0.005 0.0001"'),
             ("neck", "box", (0, 0, zn), (RN, RN, HN / 2), rgba, f'friction="{BOT_MU} 0.005 0.0001"'),
             ("stem", "cylinder", (0, 0, zt), (RT, HT / 2), CAP_RGBA, f'friction="{BOT_MU} 0.005 0.0001"'),
             ("cap", "sphere", (0, 0, zk), (RK,), CAP_RGBA, f'friction="1 {CAP_MU_T} 0.0001" condim="4" priority="2"')]
    s = [f'        <inertial pos="0 0 {-h + 0.05:.4f}" mass="0.08" diaginertia="1.5e-4 1.5e-4 2.0e-5" />']
    for nm, typ, pos, size, c, extra in parts:
        s.append(f'        <geom name="{name}_{nm}" type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" '
                 f'{extra} {BOT_SOLREF} group="0" />')
        s.append(f'        <geom type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" contype="0" '
                 f'conaffinity="0" group="1" />')
    # a pale label (visual) so the bottles read as bottles
    s.append(f'        <geom type="box" pos="0 0 {zb + 0.008:.4f}" size="{RB + 0.0006:.4f} {RB + 0.0006:.4f} 0.016" '
             f'rgba="0.94 0.92 0.84 1" contype="0" conaffinity="0" group="1" />')
    return _wrap(name, "\n".join(s), h, RB * 1.42)


def table_xml(name):
    """side table, origin at the centre of the footprint at mid height. Leg k sits under corner CORNERS[k]."""
    h = TABLE_H / 2
    zt = h - TOP_T / 2
    zl = -h + LEG_L / 2
    s = [f'        <inertial pos="0 0 {0.02:.4f}" mass="{TBL_MASS}" diaginertia="0.004 0.004 0.007" />',
         f'        <geom name="{name}_top" type="box" pos="0 0 {zt:.5f}" size="{TOP_HW:.5f} {TOP_HW:.5f} {TOP_T / 2:.5f}" '
         f'rgba="{_fmt(WOOD)}" friction="{TOP_MU} 0.005 0.0001" {TOP_SOLREF} group="0" />',
         f'        <geom type="box" pos="0 0 {zt:.5f}" size="{TOP_HW:.5f} {TOP_HW:.5f} {TOP_T / 2:.5f}" '
         f'rgba="{_fmt(WOOD)}" contype="0" conaffinity="0" group="1" />']
    for k, (sx, sy) in enumerate(CORNERS):
        p = (sx * LEG_O, sy * LEG_O, zl)
        # collision: a capsule inscribed in the square leg -> a rounded, point-like foot (a flat 1.6 cm foot moves the
        # rocking axis 1.1 cm towards the loaded corner, and the table's own weight then stops the rock; measured)
        s.append(f'        <geom name="{name}_leg{k}" type="capsule" pos="{_fmt(p)}" '
                 f'size="{LEG_HW:.5f} {LEG_L / 2 - LEG_HW:.5f}" rgba="{_fmt(LEG_RGBA)}" '
                 f'friction="{LEG_MU} 0.005 0.0001" {TBL_SOLREF} group="0" />')
        s.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt((LEG_HW, LEG_HW, LEG_L / 2))}" '
                 f'rgba="{_fmt(LEG_RGBA)}" contype="0" conaffinity="0" group="1" />')
    for k, (sx, sy) in enumerate(CORNERS):  # coloured spots (visual); colours are set per instance
        s.append(f'        <geom name="{name}_spot{k}" type="box" pos="{sx * SPOT_O:.5f} {sy * SPOT_O:.5f} {h + 0.0004:.5f}" '
                 f'size="{SPOT_HX:.5f} {SPOT_HY:.5f} 0.0004" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0" '
                 f'group="1" />')
    return _wrap(name, "\n".join(s), h, TOP_HW * 1.42)


def _fb(d):
    return f"{100 * abs(d):.1f} cm {'forward' if d > 0 else 'back'}"


def _lr(d):
    return f"{100 * abs(d):.1f} cm to the {'left' if d > 0 else 'right'}"


def _ang(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class WobblySkills(Skills):
    """Skills that feed the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class WobblyTable(LFFTask):
    name = "lh_wobbly_table"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Restore", "Plan")
    protocol = "cross"
    max_attempts = 3
    # careful: 4 bottles x ~7 commands + standing up to three fallen bottles again x ~9 + looking / corrections
    agent_max_steps = 120
    instruction = ("Put each of the four bottles on the spot of its own colour on the small wooden table. At the end "
                   "every bottle must stand upright on its spot, the gripper must be open and clear of the bottles, and "
                   "the small table must still stand where it is now. (Each bottle has a square neck with a small round "
                   "white stopper on top. Gripped on the neck, a bottle stays rigid in the fingers; gripped by the "
                   "stopper alone, it swings freely and hangs straight down.)")
    instruction_indirect = ("Set the four coloured bottles out on the little wooden side table, each on the patch that "
                            "matches its colour. When you finish, all four have to be standing upright on their patches, "
                            "the gripper open and away from them, and the side table left where it stands. (Gripped "
                            "on its square neck a bottle stays rigid; gripped by the round white stopper alone it swings "
                            "freely and hangs straight down.)")
    agent_object_names = ["bottles (red, blue, green, yellow; each with a square neck and a small round white stopper)",
                          "small wooden side table with four coloured spots (red, blue, green, yellow)"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs = []
        for i, (c, b) in enumerate(zip(COLOURS, BOTTLES)):
            cat = register_generated(f"LffLhwBottle{c.capitalize()}", bottle_xml(f"lff_lhw_bottle_{c}", RGBA[c]))
            objs.append((b, cat, f"bottle{i}_region", (-0.30 + 0.13 * i, 0.30, -0.29 + 0.13 * i, 0.31)))
        cat = register_generated("LffLhwSideTable", table_xml("lff_lhw_table"))
        objs.append((TABLE, cat, "table_region", (-0.05, -0.30, -0.04, -0.29)))
        return write_bddl(self.name, "Stand each bottle on the spot of its colour on the small table", objs, [])

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.tbid = envs.body_id(self.env, TABLE)
        pre = f"{TABLE}_lff_lhw_table"
        self.leg_gid = [m.geom_name2id(f"{pre}_leg{k}") for k in range(4)]
        self.top_gid = m.geom_name2id(f"{pre}_top")
        self.leg_size0 = [m.geom_size[g].copy() for g in self.leg_gid]
        self.leg_pos0 = [m.geom_pos[g].copy() for g in self.leg_gid]
        self.tbl_ipos0 = m.body_ipos[self.tbid].copy()
        self.spot_gid = [m.geom_name2id(f"{pre}_spot{k}") for k in range(4)]
        self.bbid = {b: envs.body_id(self.env, b) for b in BOTTLES}
        self.cap_gid = {b: m.geom_name2id(f"{b}_lff_lhw_bottle_{c}_cap") for c, b in zip(COLOURS, BOTTLES)}
        self.dof = {b: envs.free_joint_addr(self.env, b)[1] for b in BOTTLES}
        self.tbl_dof = envs.free_joint_addr(self.env, TABLE)[1]
        self.bgeoms = {b: envs.obj_geom_ids(self.env, b) for b in BOTTLES}
        self.finger_geoms, self.hand_geoms, self.arm_geoms = [], [], []
        for g in envs.robot_geom_ids(self.env):
            n = m.geom_id2name(g) or ""
            (self.finger_geoms if "finger" in n else self.hand_geoms if n.startswith("gripper0_")
             else self.arm_geoms).append(g)
        # per-geom lookup tables for the substep contact check: bottle index (-1 none), robot part (1 finger, 2 hand,
        # 3 arm), table geom
        self.owner = np.full(m.ngeom, -1, int)
        for i, b in enumerate(BOTTLES):
            self.owner[self.bgeoms[b]] = i
        self.part = np.zeros(m.ngeom, int)
        self.part[self.finger_geoms] = 1
        self.part[self.hand_geoms] = 2
        self.part[self.arm_geoms] = 3
        self.is_table = np.zeros(m.ngeom, bool)
        self.is_table[envs.obj_geom_ids(self.env, TABLE)] = True
        self._sub_rob, self._sub_obj, self._sub_tbl = {}, {}, 0
        # contacts are collected after every MuJoCo substep (same technique as lh_nudge_row; wrappers chain)
        sim = self.env.sim
        if not getattr(sim, "_lhw_wrapped", False):
            orig = sim.step

            def step(*a, **k):
                orig(*a, **k)
                t = getattr(sim, "_lhw_task", None)
                if t is not None and t.mon is not None:
                    t._substep()
            sim.step = step
            sim._lhw_wrapped = True
        sim._lhw_task = self
        self.inst = None
        self.mon = None
        self.sk = None
        self._held_prev = None

    def _substep(self):
        d = self.env.sim.data
        n = d.ncon
        if not n:
            return
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        o1, o2 = self.owner[g1], self.owner[g2]
        for a, b, pa, pb in zip(g1, g2, o1, o2):
            if pa < 0 and pb < 0:
                if (self.is_table[a] and self.part[b]) or (self.is_table[b] and self.part[a]):
                    self._sub_tbl = 1
                continue
            if pa >= 0 and pb >= 0:
                if pa != pb:
                    self._sub_obj[int(pa)] = int(pb)
                    self._sub_obj[int(pb)] = int(pa)
                continue
            p, o = (pa, b) if pa >= 0 else (pb, a)
            part = int(self.part[o])
            if part:
                self._sub_rob[int(p)] = min(self._sub_rob.get(int(p), 9), part)

    # ------------------------------------------------------------ instances
    @staticmethod
    def _supply(side, tx, ty, j):
        """slot j (0-3) of the row of bottles on the main table: in front of the side table (robot side) or beside it;
        every grasp point stays within 0.70 m (horizontal) of the robot base (the arm stalled 2-3 cm short of low
        targets beyond ~0.72 m, measured)"""
        if side == "front":
            return [tx - 0.215, ty + (j - 1.5) * 0.085]
        sg = 1.0 if side == "left" else -1.0
        return [tx - 0.05 + (j - 1.5) * 0.075, ty + sg * 0.22]

    @staticmethod
    def _far_first(spot_of, side, tx, ty):
        """colours in the order of the scripted naive policy: the spots farthest from the supply first (a careful agent
        does not reach over bottles it has already put down), ties broken to the left, then far"""
        sup = np.mean([WobblyTable._supply(side, tx, ty, j) for j in range(4)], axis=0)
        sc = []
        for c, k in spot_of.items():
            sx, sy = CORNERS[k]
            p = np.array([tx + sx * SPOT_O, ty + sy * SPOT_O])
            sc.append((-round(float(np.linalg.norm(p - sup)), 3), -sy, -sx, c))
        return [t[-1] for t in sorted(sc)]

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        tx = float(rng.uniform(-0.13, -0.09))  # side table centre: 53-57 cm in front of the robot base
        ty = float(rng.uniform(-0.04, 0.04))
        side = str(rng.choice(["left", "right", "front"]))
        perm = [int(v) for v in rng.permutation(4)]  # perm[corner] = colour index on that corner's spot
        spot_of = {COLOURS[perm[k]]: k for k in range(4)}
        # the heavy bottle and the short leg under its spot: balanced over seeds against the scripted naive order (far
        # spots first): its spot comes 1st / 2nd / 3rd / 4th in that order for exactly 5 of every 20 consecutive seeds
        order = self._far_first(spot_of, side, tx, ty)
        pos = int((int(seed) * 7) % 20 // 5)
        heavy = order[pos]
        short = spot_of[heavy]
        slots = [int(v) for v in rng.permutation(4)]
        start = {}
        for i in range(4):
            x, y = self._supply(side, tx, ty, slots[i])
            start[COLOURS[i]] = [x + float(rng.uniform(-0.006, 0.006)), y + float(rng.uniform(-0.006, 0.006))]
        mass = {c: float(rng.uniform(*LIGHT_MASS)) for c in COLOURS}
        com = {c: float(rng.uniform(*LIGHT_COM)) for c in COLOURS}
        mass[heavy] = float(rng.uniform(*HEAVY_MASS))
        com[heavy] = float(rng.uniform(*HEAVY_COM))
        return dict(seed=int(seed), table_xy=[tx, ty], short_leg=int(short), short_corner=CORNER_NAME[CORNERS[short]],
                    heavy=heavy, heavy_rank_naive=pos + 1, spot_of=spot_of, supply_side=side, start=start, mass=mass,
                    com=com, short_h=float(rng.uniform(*SHORT_H)), com_bias=COM_BIAS,
                    yaw0={c: float(rng.uniform(-0.3, 0.3)) for c in COLOURS},
                    naive_bias_x=float(rng.uniform(*NAIVE_BIAS)))

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        k = inst["short_leg"]
        h = float(inst["short_h"])
        for j, g in enumerate(self.leg_gid):
            m.geom_size[g] = self.leg_size0[j]
            m.geom_pos[g] = self.leg_pos0[j]
        g = self.leg_gid[k]
        m.geom_size[g][1] = self.leg_size0[k][1] - h / 2  # capsule half length
        m.geom_pos[g][2] = self.leg_pos0[k][2] + h / 2
        opp = CORNERS[(k + 2) % 4]
        b = float(inst.get("com_bias", COM_BIAS)) / np.sqrt(2)
        m.body_ipos[self.tbid] = [opp[0] * b, opp[1] * b, self.tbl_ipos0[2]]
        envs.refit_bounds(m, self.tbid)
        for c in COLOURS:
            m.geom_rgba[self.spot_gid[inst["spot_of"][c]]] = SPOT_RGBA[c]
        for c, bot in zip(COLOURS, BOTTLES):
            bid = self.bbid[bot]
            M, zc = inst["mass"][c], inst["com"][c]
            m.body_mass[bid] = M
            m.body_ipos[bid] = [0.0, 0.0, -H_TOT / 2 + zc]
            # glass shell + internal weight, roughly: radius of gyration ~ 0.35 x height about horizontal axes
            ixx = M * ((2 * RB) ** 2 / 12 + (0.35 * H_TOT) ** 2)
            m.body_inertia[bid] = [ixx, ixx, M * (2 * RB) ** 2 / 6]
            envs.refit_bounds(m, bid)
            m.dof_damping[self.dof[bot]:self.dof[bot] + 6] = BOT_DAMP
        self.env.sim.forward()
        tx, ty = inst["table_xy"]
        # yaw pi: the canonical quaternion LIBERO gives generated objects is a 180 deg yaw (envs.canonical_quat); with
        # it the table frame is the world frame
        envs.place_upright(self.env, TABLE, [tx, ty], yaw=np.pi, clearance=0.0003)
        for c, bot in zip(COLOURS, BOTTLES):
            envs.place_upright(self.env, bot, inst["start"][c], yaw=inst["yaw0"][c], clearance=0.0005)
        self.env.sim.data.qvel[self.tbl_dof:self.tbl_dof + 6] = 0.0

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = WobblySkills(self.env, self, recorder=recorder)
        self._sub_rob, self._sub_obj, self._sub_tbl = {}, {}, 0
        self._held_prev = None
        self.table0 = dict(pos=self.table_pos().tolist(), yaw=self.table_yaw())
        n = len(BOTTLES)
        self.mon = dict(state=[False] * n, raw_since=[0] * n, ever=[False] * n, lost=[False] * n, events=[],
                        lost_cause=[None] * n, last_robot=[None] * n, last_obj=[None] * n, last_release=None,
                        last_table_touch=-10 ** 6, rocked=False, rock_since=0, rocks=[], n_steps=0,
                        max_table_tilt_deg=0.0)
        return self.sk

    # ------------------------------------------------------------ geometry / predicates
    def table_R(self):
        return self.env.sim.data.body_xmat[self.tbid].reshape(3, 3).copy()

    def table_pos(self):
        return self.env.sim.data.body_xpos[self.tbid].copy()

    def table_yaw(self):
        R = self.table_R()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def table_tilt(self):
        return float(np.arccos(np.clip(self.table_R()[2, 2], -1, 1)))

    def top_point(self, local_xy):
        """world position of a point on the table's top surface (table frame xy)"""
        return self.table_pos() + self.table_R() @ np.array([local_xy[0], local_xy[1], TABLE_H / 2])

    def spot_local(self, colour):
        sx, sy = CORNERS[self.inst["spot_of"][colour]]
        return np.array([sx * SPOT_O, sy * SPOT_O])

    def spot_world(self, colour):
        return self.top_point(self.spot_local(colour))

    def top_z_at(self, xy):
        """z of the table's top plane above world point xy"""
        R, p = self.table_R(), self.table_pos()
        n = R[:, 2]
        c = p + n * TABLE_H / 2
        return float(c[2] - (n[0] * (xy[0] - c[0]) + n[1] * (xy[1] - c[1])) / max(n[2], 1e-6))

    def over_table(self, xy):
        loc = self.table_R().T @ (np.array([xy[0], xy[1], self.table_pos()[2]]) - self.table_pos())
        return abs(loc[0]) < TOP_HW and abs(loc[1]) < TOP_HW

    def bR(self, b):
        return self.env.sim.data.body_xmat[self.bbid[b]].reshape(3, 3).copy()

    def bpos(self, b):
        return self.env.sim.data.body_xpos[self.bbid[b]].copy()

    def foot(self, b):
        """world position of the centre of the bottle's bottom face"""
        return self.bpos(b) - self.bR(b)[:, 2] * H_TOT / 2

    def tilt(self, b):
        return float(np.arccos(np.clip(self.bR(b)[2, 2], -1.0, 1.0)))

    def cap_xyz(self, b):
        return self.env.sim.data.geom_xpos[self.cap_gid[b]].copy()

    def held(self, b):
        return self.sk is not None and self.sk.holding(b)

    def offset(self, i):
        """bottle i's foot centre relative to its spot centre, in the table frame (x forward-back, y left-right), and
        the height of its bottom above the table top"""
        b, c = BOTTLES[i], COLOURS[i]
        loc = self.table_R().T @ (self.foot(b) - self.table_pos())
        return loc[:2] - self.spot_local(c), float(loc[2] - TABLE_H / 2)

    def on_spot(self, i, extra=0.0):
        b = BOTTLES[i]
        if self.tilt(b) > TILT_OK + (np.radians(2.0) if extra else 0.0):
            return False
        off, dz = self.offset(i)
        if abs(off[0]) > TOL_X + extra or abs(off[1]) > TOL_Y + extra or abs(dz) > ON_TOP:
            return False
        return not self.held(b)

    def stages(self):
        return [self.on_spot(i) for i in range(len(BOTTLES))]

    def table_ok(self):
        p0 = np.asarray(self.table0["pos"])
        moved = float(np.linalg.norm(self.table_pos()[:2] - p0[:2]))
        dyaw = abs(_ang(self.table_yaw() - self.table0["yaw"]))
        return bool(moved <= TABLE_MOVE_TOL and dyaw <= TABLE_YAW_TOL and self.table_tilt() <= TABLE_TILT_MAX), moved, dyaw

    def low_corner(self):
        """robot-centric name of the table corner that is lowest (after a rock: the corner over the short leg)"""
        zs = {k: self.top_point([sx * TOP_HW, sy * TOP_HW])[2] for k, (sx, sy) in enumerate(CORNERS)}
        return CORNER_NAME[CORNERS[min(zs, key=zs.get)]]

    # ------------------------------------------------------------ monitor
    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        mon["n_steps"] += 1
        t = int(self.sk.n_steps)
        rob, obj, tbl = dict(self._sub_rob), dict(self._sub_obj), self._sub_tbl
        self._sub_rob, self._sub_obj, self._sub_tbl = {}, {}, 0
        gstate = "open" if self.sk.grip < 0 else "closed"
        held_now = None
        for c, b in zip(COLOURS, BOTTLES):
            if gstate == "closed" and self.held(b):
                held_now = c
        for i, part in rob.items():
            if COLOURS[i] != held_now:
                mon["last_robot"][i] = [{1: "finger", 2: "hand", 3: "arm"}[part], t, gstate]
        for i, j in obj.items():
            mon["last_obj"][i] = [COLOURS[j], t]
        if tbl:
            mon["last_table_touch"] = t
        if held_now is None and self._held_prev is not None:  # the gripper has just let go of a bottle
            mon["last_release"] = [self._held_prev, t]
        self._held_prev = held_now
        tl = self.table_tilt()
        mon["max_table_tilt_deg"] = max(mon["max_table_tilt_deg"], round(float(np.degrees(tl)), 2))
        if mon["n_steps"] % 2:
            return
        # ---- table rock detector (debounced)
        want = tl > ROCK_ON if not mon["rocked"] else tl > ROCK_OFF
        if want != mon["rocked"]:
            mon["rock_since"] += 2
            if mon["rock_since"] >= 4:
                mon["rocked"] = want
                mon["rock_since"] = 0
                self._rock_event(want, t - 3)
        else:
            mon["rock_since"] = 0
        for r in mon["rocks"]:
            if not r.get("closed") and t - r["step"] > FELL_WINDOW:
                self._close_rock(r)
        # ---- stage predicates
        for i in range(len(BOTTLES)):
            b = BOTTLES[i]
            raw = self.on_spot(i, HYST if mon["state"][i] else 0.0)
            if raw == mon["state"][i]:
                mon["raw_since"][i] = 0
                continue
            mon["raw_since"][i] += 2
            if mon["raw_since"][i] < DEBOUNCE:
                continue
            mon["state"][i] = raw
            mon["raw_since"][i] = 0
            t0 = int(t - DEBOUNCE + 1)
            if raw:
                restored = mon["lost"][i]
                e = dict(stage=COLOURS[i], event="restored" if restored else "achieved", step=t0)
                if restored:
                    e["after"] = mon["lost_cause"][i]
                mon["events"].append(e)
                mon["ever"][i] = True
            else:
                mon["lost"][i] = True
                cause = self._cause(i, t0)
                mon["lost_cause"][i] = cause
                off, dz = self.offset(i)
                mon["events"].append(dict(stage=COLOURS[i], event="lost", step=t0, cause=cause,
                                          tilt_deg=round(float(np.degrees(self.tilt(b))), 1),
                                          offset_cm=[round(100 * float(v), 1) for v in off]))

    def _rock_event(self, down, t):
        mon = self.mon
        rel = mon["last_release"]
        e = dict(event="table_rocked" if down else "table_rocked_back", step=int(t),
                 table_tilt_deg=round(float(np.degrees(self.table_tilt())), 2), low_corner=self.low_corner(),
                 released=rel[0] if rel is not None and t - rel[1] <= 40 else None,
                 robot_touched_table=bool(t - mon["last_table_touch"] <= 20),
                 standing=[COLOURS[i] for i in range(len(BOTTLES)) if mon["state"][i]])
        # the cheap explanation would be "my hand bumped them": which of the standing bottles did the robot touch in
        # the 2 s before the rock (should be none)
        e["robot_touched_before"] = [COLOURS[i] for i in range(len(BOTTLES)) if mon["state"][i] and
                                     mon["last_robot"][i] is not None and t - mon["last_robot"][i][1] <= 40]
        mon["events"].append(e)
        mon["rocks"].append(dict(step=int(t), down=bool(down), standing=list(e["standing"]), event=e))

    def _close_rock(self, r):
        """FELL_WINDOW after a rock: which of the bottles that stood on their spots fell, and did the robot touch any
        of them after the rock began (a falling bottle hitting the open fingers)"""
        r["closed"] = True
        mon = self.mon
        fell, touched = [], []
        for c in r["standing"]:
            i = COLOURS.index(c)
            if self.tilt(BOTTLES[i]) > np.radians(30) or not mon["state"][i]:
                fell.append(c)
                lr = mon["last_robot"][i]
                if lr is not None and lr[1] >= r["step"]:
                    touched.append(c)
        r["event"]["fell"] = fell
        r["event"]["robot_touched_while_falling"] = touched
        r["fell"] = fell

    def _cause(self, i, t, window=60):
        if self.held(BOTTLES[i]):
            return "picked_up"
        mon = self.mon
        for r in reversed(mon["rocks"]):
            if 0 <= t - r["step"] <= FELL_WINDOW + 10 and COLOURS[i] in r["standing"]:
                return "table_rocked" if r["down"] else "table_rocked_back"
        lr, lo = mon["last_robot"][i], mon["last_obj"][i]
        cands = []
        if lr is not None and t - lr[1] <= window:
            cands.append((lr[1], f"robot_{lr[0]}"))
        if lo is not None and t - lo[1] <= window:
            cands.append((lo[1] - 30, "bottle:" + lo[0]))
        return max(cands)[1] if cands else "other"

    def _finalize_monitor(self, raw):
        mon = self.mon
        if mon is None:
            return
        for r in mon["rocks"]:
            if not r.get("closed"):
                self._close_rock(r)
        t = int(self.sk.n_steps)
        for i in range(len(BOTTLES)):
            if raw[i] == mon["state"][i]:
                continue
            mon["state"][i] = raw[i]
            if raw[i]:
                restored = mon["lost"][i]
                e = dict(stage=COLOURS[i], event="restored" if restored else "achieved", step=t, final=True)
                if restored:
                    e["after"] = mon["lost_cause"][i]
                mon["events"].append(e)
                mon["ever"][i] = True
            else:
                mon["lost"][i] = True
                cause = self._cause(i, t, window=10 ** 6)
                mon["lost_cause"][i] = cause
                mon["events"].append(dict(stage=COLOURS[i], event="lost", step=t, cause=cause, final=True))

    # ------------------------------------------------------------ outcome
    def _touching(self):
        d = self.env.sim.data
        for j in range(d.ncon):
            c = d.contact[j]
            if (self.owner[c.geom1] >= 0 and self.part[c.geom2]) or (self.owner[c.geom2] >= 0 and self.part[c.geom1]):
                return True
        return False

    def outcome(self, inst):
        sk = self.sk
        sk.hold(10)
        p0 = {b: self.bpos(b) for b in BOTTLES}
        t0 = self.table_pos()
        touching = False
        for _ in range(20):
            sk.hold(1)
            touching = touching or self._touching()
        moved = max([float(np.linalg.norm(self.bpos(b) - p0[b])) for b in BOTTLES] +
                    [float(np.linalg.norm(self.table_pos() - t0))])
        static = moved < 0.003
        raw = self.stages()
        self._finalize_monitor(raw)
        held = [COLOURS[i] for i, b in enumerate(BOTTLES) if self.held(b)]
        tok, tmoved, tyaw = self.table_ok()
        mon = self.mon
        ev = list(mon["events"])
        rocks = [r for r in mon["rocks"] if r["down"]]
        damaging = [r for r in mon["rocks"] if r.get("fell")]
        n_reg = sum(e.get("event") == "lost" and e.get("cause") != "picked_up" for e in ev)
        n_lost = sum(e.get("event") == "lost" for e in ev)
        n_res = sum(e.get("event") == "restored" and e.get("after") != "picked_up" for e in ev)
        success = bool(all(raw) and static and not held and not touching and tok)
        tilted = [i for i in range(len(BOTTLES)) if self.tilt(BOTTLES[i]) > TILT_OK and COLOURS[i] not in held]
        ever = mon["ever"]
        fell_by_rock = sorted({c for r in damaging for c in r["fell"]})
        if success:
            failure = "none"
        elif not tok:
            failure = "table_moved"
        elif held:
            failure = "held"
        elif tilted:
            failure = "toppled_by_rock" if any(COLOURS[i] in fell_by_rock for i in tilted) else "toppled"
        elif not all(raw):
            failure = "not_on_spot"
        elif touching:
            failure = "touching"
        else:
            failure = "not_static"
        parts = []
        if success:
            parts.append("Every bottle stands upright on the spot of its colour.")
        for i in range(len(BOTTLES)):
            if raw[i]:
                continue
            c, b = COLOURS[i], BOTTLES[i]
            if c in held:
                parts.append(f"The {c} bottle is still held by the gripper.")
                continue
            off, dz = self.offset(i)
            tl = float(np.degrees(self.tilt(b)))
            on_small = abs(dz) < 0.05 and max(abs(off + self.spot_local(c))) < TOP_HW + 0.02
            if tl > np.degrees(TILT_OK):
                txt = (f"The {c} bottle lies on its side on the {'small' if on_small else 'big'} table" if tl > 45 else
                       f"The {c} bottle leans {tl:.0f} degrees") + "."
            elif abs(dz) > ON_TOP:
                txt = f"The {c} bottle is not on the small table."
            else:
                txt = f"The {c} bottle stands {_fb(off[0])} and {_lr(off[1])} of its spot's centre, not on its spot."
            if ever[i] and c in fell_by_rock:
                txt += " It had stood on its spot and fell when the small table rocked."
            elif ever[i]:
                txt += " It had stood on its spot earlier."
            parts.append(txt)
        if not tok:
            parts.append(f"The small table was moved ({100 * tmoved:.1f} cm, {np.degrees(tyaw):.0f} degrees).")
        elif not success and all(raw):
            if touching:
                parts.append("The gripper is still touching a bottle.")
            elif not static:
                parts.append("Something was still moving.")
        if rocks:
            r = rocks[0]["event"]
            txt = (f"When the {r['released']} bottle was set down, the small table rocked down onto its "
                   f"{r['low_corner']} leg" if r.get("released") else
                   f"The small table rocked down onto its {r['low_corner']} leg")
            fl = r.get("fell", [])
            txt += (" and the " + " and ".join(fl) + f" bottle{'s' if len(fl) > 1 else ''} standing on it fell over."
                    if fl else ".")
            txt += f" That leg is {100 * float(inst['short_h']):.1f} cm shorter than the other three."
            parts.append(txt)
        trap = bool(damaging)
        return dict(success=success, stages={COLOURS[i]: bool(raw[i]) for i in range(len(BOTTLES))}, events=ev,
                    n_regressions=int(n_reg), n_lost=int(n_lost), n_restored=int(n_res), n_rocks=len(rocks),
                    rocked=bool(rocks), trap_fired=trap, recovered_in_attempt=bool(trap and success),
                    fell_by_rock=fell_by_rock, failure=failure, held=held, touching=bool(touching),
                    static=bool(static), table_ok=bool(tok), table_moved_cm=round(100 * tmoved, 2),
                    table_tilt_deg=round(float(np.degrees(self.table_tilt())), 2),
                    max_table_tilt_deg=mon["max_table_tilt_deg"],
                    offsets_cm={COLOURS[i]: [round(100 * float(v), 2) for v in self.offset(i)[0]] for i in range(4)},
                    tilts_deg={COLOURS[i]: round(float(np.degrees(self.tilt(BOTTLES[i]))), 1) for i in range(4)},
                    heavy=inst["heavy"], short_corner=inst["short_corner"], detail=" ".join(parts))

    # ------------------------------------------------------------ scripted primitives (move_to + gripper only)
    def _go(self, xyz, yaw=None, speed=None, tol=0.004, max_steps=300):
        return self.sk.move_to(xyz, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)

    def _lift_to(self, z, speed=0.2):
        cur = self.sk.eef_pos()
        if cur[2] < TZ + z - 0.005:
            self._go([cur[0], cur[1], TZ + z], speed=speed)

    def _settle(self, b, max_checks=25):
        d = self.env.sim.data
        da = self.dof[b]
        for _ in range(max_checks):
            if np.linalg.norm(d.qvel[da:da + 3]) < 0.01 and np.linalg.norm(d.qvel[da + 3:da + 6]) < 0.2:
                return
            self.sk.hold(2)

    def upright(self, b):
        return self.tilt(b) < np.radians(20)

    def pick_neck(self, b, z_carry=Z_CARRY, yaw=0.0, v=0.2):
        """Sonnet-like top-down pick of a standing bottle: open, over it at carry height, straight down to the upper
        neck, close, lift to carry height. Returns True if held."""
        sk = self.sk
        self._settle(b)
        p = self.foot(b)
        zg = p[2] + NECK_Z[1] - GRASP_NECK
        sk.set_gripper(False, steps=8)
        self._lift_to(z_carry)
        self._go([p[0], p[1], TZ + z_carry], yaw=yaw, speed=v)
        self._go([p[0], p[1], zg + 0.02], speed=0.12)
        self._go([p[0], p[1], zg], speed=0.04, tol=0.003, max_steps=150)
        sk.set_gripper(True, steps=15)
        self._go([p[0], p[1], TZ + z_carry], speed=0.12)
        sk.hold(5)
        if not self.held(b):
            sk.set_gripper(False, steps=8)
            return False
        return True

    def put_held(self, b, xy, gap, z_carry=Z_CARRY, yaw=None, v=0.2):
        """carry the held bottle so that its foot is over world xy, straight down until its bottom is `gap` above the
        surface below, let go, lift"""
        sk = self.sk
        rel = self.foot(b)[:2] - sk.eef_pos()[:2]
        q = np.asarray(xy, float) - rel
        self._go([q[0], q[1], TZ + z_carry], yaw=yaw, speed=v)
        sk.hold(10)
        rel = self.foot(b)[:2] - sk.eef_pos()[:2]  # a bottle hanging from its stopper swings while carried
        q = np.asarray(xy, float) - rel
        surf = self.top_z_at(xy) if self.over_table(xy) else TZ
        dz = sk.eef_pos()[2] - self.foot(b)[2]
        zr = surf + dz + gap
        self._go([q[0], q[1], zr + 0.02], speed=0.10)
        self._go([q[0], q[1], zr], speed=0.03, tol=0.003, max_steps=150)
        sk.hold(4)
        sk.set_gripper(False, steps=12)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], cur[2] + 0.06], speed=0.10)
        self._go([cur[0], cur[1], TZ + z_carry], speed=0.2)
        sk.hold(3)

    def surface_z(self, xy):
        return self.top_z_at(xy) if self.over_table(xy) else TZ

    def pick_stopper(self, b, v=0.15):
        """grasp a lying (or leaning) bottle by its round stopper from above, fingers across the bottle's axis, lift it
        until it hangs free. The fingertips stay >= 4 mm above the surface under the stopper: pressing the raised half
        of the rocked table rocks it back (measured), and the knock when it comes down again topples standing bottles.
        Returns True if held."""
        sk = self.sk
        self._settle(b)
        for _ in range(2):
            k, zg, yaw = self.stopper_grasp(b)  # fingers across the bottle's axis
            sk.set_gripper(False, steps=8)
            self._lift_to(Z_HANG)
            self._go([k[0], k[1], TZ + Z_HANG], yaw=yaw, speed=v)
            sk.hold(5)
            k, zg, _ = self.stopper_grasp(b)
            self._go([k[0], k[1], zg + 0.03], speed=0.10)
            self._go([k[0], k[1], zg], speed=0.03, tol=0.003, max_steps=120)
            sk.set_gripper(True, steps=15)
            if not self.held(b):
                sk.set_gripper(False, steps=8)
                self._go([k[0], k[1], zg + 0.08], speed=0.1)
                continue
            self._go([k[0], k[1], zg + 0.05], speed=0.05)
            self._go([k[0], k[1], TZ + Z_HANG], speed=0.10)
            sk.hold(20)
            return True
        return False

    # gripper footprint (measured, t8_hand): the hand's collision hull is a slab 20 cm along the finger axis and ~4 cm
    # across, from 3 cm above the grasp point up; open fingers 3.9-6.6 cm out along the finger axis, from 1 cm below
    # the grasp point to 4.5 cm above it
    HAND_U, HAND_V, HAND_Z0 = 0.100, (-0.018, 0.024), 0.03
    FING_U, FING_V, FING_Z = (0.039, 0.066), (-0.009, 0.013), (-0.010, 0.045)
    # wrist and forearm (t16_arm, measured at low grasps): lowest point above the grasp point vs the distance from the
    # grasp point towards the robot base; they span ~13 cm to either side of that line
    ARM_D = (-0.05, 0.05, 0.10, 0.15, 0.20)
    ARM_Z = (0.097, 0.100, 0.155, 0.185, 0.31)
    ARM_LAT = 0.13

    def _footprint(self, q, yaw):
        """sample points (xy) of the hand slab and of the open fingers for a grasp at q with gripper yaw `yaw`"""
        a = np.array([-np.sin(yaw), np.cos(yaw)])  # finger axis
        c = np.array([np.cos(yaw), np.sin(yaw)])
        hand = [q + u * a + v * c for u in np.linspace(-self.HAND_U, self.HAND_U, 21)
                for v in np.linspace(*self.HAND_V, 5)]
        fing = [q + sg * u * a + v * c for sg in (-1, 1) for u in np.linspace(*self.FING_U, 4)
                for v in np.linspace(*self.FING_V, 3)]
        return np.array(hand), np.array(fing)

    def grasp_clearance(self, q, zg, yaw, ignore=()):
        """smallest horizontal clearance (m) between the hand / open fingers of a grasp at (q, zg) and any standing
        bottle (other than `ignore`) or the small table; negative = collision"""
        q = np.asarray(q, float)[:2]
        hand, fing = self._footprint(q, yaw)
        best = 1.0
        for b in BOTTLES:
            if b in ignore or self.held(b) or self.tilt(b) > np.radians(30):
                continue
            p = self.foot(b)[:2]
            top = self.foot(b)[2] + H_TOT
            r = RB * 1.42
            if top > zg + self.HAND_Z0:
                best = min(best, float(np.min(np.linalg.norm(hand - p, axis=1))) - r)
            best = min(best, float(np.min(np.linalg.norm(fing - p, axis=1))) - r)
            # the wrist / forearm sloping back towards the robot base over a tall standing bottle
            ub = np.array([-0.66, 0.0]) - q
            ub /= max(float(np.linalg.norm(ub)), 1e-6)
            along, lat = float((p - q) @ ub), float((p - q) @ np.array([-ub[1], ub[0]]))
            if self.ARM_D[0] <= along <= self.ARM_D[-1] and abs(lat) < self.ARM_LAT + r:
                if top > zg + float(np.interp(along, self.ARM_D, self.ARM_Z)) - 0.01:
                    best = min(best, -0.01)
        # the small table's top plate (and legs) when the grasp is below its top surface
        R, P = self.table_R(), self.table_pos()
        ztop = P[2] + TABLE_H / 2
        for pts, z0, z1 in ((hand, zg + self.HAND_Z0, zg + 0.13), (fing, zg + self.FING_Z[0], zg + self.FING_Z[1])):
            if z0 > ztop + 0.003:
                continue
            loc = (pts - P[:2]) @ R[:2, :2]
            if z1 > ztop - TOP_T:  # the plate's height band
                d = np.maximum(np.abs(loc) - TOP_HW, 0.0)
                inside = np.all(np.abs(loc) < TOP_HW, axis=1)
                dist = np.where(inside, -0.01, np.linalg.norm(d, axis=1))
                best = min(best, float(np.min(dist)))
        return best

    def stopper_grasp(self, b):
        """(xy, z, yaw) of the stopper grasp of a lying bottle: fingers across its axis, fingertips >= 4 mm above the
        surface below"""
        k = self.cap_xyz(b)
        ax = self.bR(b)[:2, 2]
        yaw = float(np.arctan2(ax[1], ax[0])) if np.linalg.norm(ax) > 0.2 else self.sk.yaw
        zg = max(float(k[2]), self.surface_z(k[:2]) + envs.FINGERTIP_BELOW_SITE + 0.004)
        return k[:2], zg, yaw

    def body_grasp(self, b, along=None):
        """(xy, z, yaw) of a grasp across a lying bottle's body, `along` m from its bottom (default: the middle)"""
        R = self.bR(b)
        a = HF + (HB - HF) / 2 if along is None else float(along)
        c = self.foot(b) + R[:, 2] * a
        ax = R[:2, 2]
        yaw = float(np.arctan2(ax[1], ax[0])) if np.linalg.norm(ax) > 0.2 else self.sk.yaw
        zg = max(float(c[2]), self.surface_z(c[:2]) + envs.FINGERTIP_BELOW_SITE + 0.004)
        return c[:2], zg, yaw

    def staging_spot(self, b):
        """a free place on the main table to lay a fallen bottle down: far from the small table and the other bottles,
        within the arm's comfortable reach"""
        tx, ty = self.table_pos()[:2]
        best, bq = -1.0, None
        for x in np.arange(tx - 0.30, tx + 0.06, 0.03):
            for y in np.arange(ty - 0.30, ty + 0.31, 0.03):
                # within reach, and >= 14 cm inside the main table's edges (it spans x, y = +-0.5, +-0.6): a bottle laid
                # down too close to the near edge rolled off onto the floor (measured)
                if np.hypot(x + 0.66, y) > 0.66 or np.hypot(x + 0.66, y) < 0.33 or x < -0.36 or abs(y) > 0.46:
                    continue
                loc = self.table_R()[:2, :2].T @ (np.array([x, y]) - np.array([tx, ty]))
                dt = float(np.linalg.norm(np.maximum(np.abs(loc) - TOP_HW, 0.0)))
                if np.all(np.abs(loc) < TOP_HW):
                    continue
                do = min([float(np.linalg.norm(self.foot(o)[:2] - [x, y])) for o in BOTTLES if o != b] +
                         [float(np.linalg.norm(self.cap_xyz(o)[:2] - [x, y])) for o in BOTTLES if o != b])
                sc = min(dt - 0.10, do - 0.14)  # the bottle lies ~16 cm long around the point
                if sc > best and min(self.grasp_clearance([x, y], TZ + 0.016, yw, ignore=(b,))
                                     for yw in (0.0, np.pi / 2)) > 0.01:
                    best, bq = sc, np.array([x, y])
        return bq

    def stage(self, b, v=0.12):
        """the stopper of a fallen bottle cannot be reached: grasp its body at a point where the hand is clear (the
        bottle then hangs from the grasp), carry it to free space on the main table, let it down there so that it falls
        over away from everything. Returns True if it was moved."""
        sk = self.sk
        for along in (0.055, 0.035, 0.075, 0.015, 0.095):
            q, zg, yaw = self.body_grasp(b, along)
            if self.grasp_clearance(q, zg, yaw, ignore=(b,)) >= 0.004:
                break
        else:
            return False
        dst = self.staging_spot(b)
        if dst is None:
            return False
        sk.set_gripper(False, steps=8)
        self._lift_to(Z_CARRY)
        self._go([q[0], q[1], TZ + Z_CARRY], yaw=yaw, speed=v)
        self._go([q[0], q[1], zg + 0.03], speed=0.10)
        self._go([q[0], q[1], zg], speed=0.03, tol=0.003, max_steps=120)
        sk.set_gripper(True, steps=15)
        if not self.held(b):
            sk.set_gripper(False, steps=8)
            self._go([q[0], q[1], TZ + Z_CARRY], speed=0.1)
            return False
        self._go([q[0], q[1], zg + 0.04], speed=0.05)
        self._go([q[0], q[1], TZ + Z_HANG], speed=0.10)
        sk.hold(15)
        self._go([dst[0], dst[1], TZ + Z_HANG], speed=0.10)
        sk.hold(15)
        drop = envs.obj_min_z(self.env, b) - (TZ + 0.012)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], cur[2] - drop], speed=0.08)
        sk.set_gripper(False, steps=12)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], TZ + Z_HANG], speed=0.15)
        self._settle(b, 40)
        return True

    def restand(self, i, gap=CAREFUL_GAP, err=None):
        """stand bottle i up again on its spot: if the stopper cannot be grasped without the hand touching a standing
        bottle or the small table, first lay the bottle down in free space; grasp the stopper, let the bottle hang,
        carry it over the spot, lower it until its bottom is `gap` above the table top, let go"""
        b, c = BOTTLES[i], COLOURS[i]
        q, zg, yaw = self.stopper_grasp(b)
        if self.grasp_clearance(q, zg, yaw, ignore=(b,)) < 0.005:
            self.stage(b)
        if not self.pick_stopper(b):
            return False
        tgt = self.spot_world(c)[:2] + (np.asarray(err, float) if err is not None else 0.0)
        self.put_held(b, tgt, gap, z_carry=Z_HANG, v=0.10)
        self._settle(b)
        return self.on_spot(i)

    def place(self, i, gap, err=None, z_carry=Z_CARRY, yaw=0.0, v=0.2):
        """one stage: pick bottle i (by the neck if it stands, by the stopper if it lies) and put it on its spot"""
        b, c = BOTTLES[i], COLOURS[i]
        tgt = self.spot_world(c)[:2] + (np.asarray(err, float) if err is not None else 0.0)
        if self.upright(b):
            if not self.pick_neck(b, z_carry=z_carry, yaw=yaw, v=v) and not self.pick_neck(b, z_carry=z_carry,
                                                                                              yaw=yaw, v=v):
                return False
            self.put_held(b, tgt, gap, z_carry=z_carry, yaw=yaw, v=v)
        else:
            if not self.pick_stopper(b):
                return False
            self.put_held(b, tgt, gap, z_carry=Z_HANG, v=0.10)
        self._settle(b)
        return True

    def repair_order(self, idx):
        """bottles to fix: one standing upright off its spot first (it may be the heavy one: re-placing it rocks the
        table), then lying ones whose stopper is easiest to reach"""
        def key(j):
            b = BOTTLES[j]
            if self.upright(b):
                return (0, 0.0)
            q, zg, yaw = self.stopper_grasp(b)
            return (1, -self.grasp_clearance(q, zg, yaw, ignore=(b,)))
        return sorted(idx, key=key)

    def fix(self, i, gap=CAREFUL_GAP):
        if self.upright(BOTTLES[i]):
            self.place(i, gap)
        else:
            self.restand(i, gap)

    def check_and_repair(self, rounds=3, gap=CAREFUL_GAP):
        """look at every bottle; stand up / re-place every one that is not on its spot"""
        for _ in range(rounds):
            bad = [i for i in range(4) if not self.on_spot(i)]
            if not bad:
                return True
            for i in self.repair_order(bad):
                if not self.on_spot(i):
                    self.fix(i, gap)
        return all(self.stages())

    def order(self, mode, rng=None, first=None):
        o = self._far_first(self.inst["spot_of"], self.inst["supply_side"], *self.inst["table_xy"])
        if mode == "near_first":
            o = o[::-1]
        elif mode == "random":
            o = [o[j] for j in rng.permutation(4)]
        if first is not None:
            o = [first] + [c for c in o if c != first]
        return [COLOURS.index(c) for c in o]

    def execute(self, inst, params):
        rng = np.random.default_rng(params.get("noise_seed", 0))
        order = self.order(params.get("order", "far_first"), rng, params.get("first"))
        gap = params.get("gap")  # None: random in NAIVE_GAP per placement
        bias = np.array([float(params.get("bias_x", 0.0)), 0.0])
        rand = float(params.get("rand", 0.0))
        for i in order:
            g = float(rng.uniform(*NAIVE_GAP)) if gap is None else float(gap)
            e = bias + (rng.uniform(-rand, rand, 2) if rand else 0.0)
            self.place(i, g, err=e, z_carry=params.get("z_carry", Z_CARRY), v=params.get("v", 0.2))
            if params.get("repair") == "each":  # looks at all bottles after every step and stands fallen ones up
                for _ in range(2):
                    bad = [j for j in range(4) if self.mon["ever"][j] and not self.on_spot(j)]
                    for j in self.repair_order(bad):
                        if not self.on_spot(j):
                            self.fix(j)
        if params.get("check"):
            self.check_and_repair()
        self._lift_to(Z_CARRY)
        self.sk.hold(10)

    # naive = Sonnet-like habits (lh_nudge_row / lh_bystander pilots): careful, far spots first, carries high, vertical
    # descent, releases 1-4 cm above the surface, places ~1-2 cm too far forward +- 0.6 cm, no special plan for the
    # table, does not look back at the bottles it has already put down
    def default_params(self, inst):
        return dict(order="far_first", gap=None, bias_x=float(inst["naive_bias_x"]), rand=NAIVE_RAND,
                    noise_seed=int(inst["seed"]) * 13, repair=None, check=False)

    def oracle_params(self, inst):
        """knows the short leg: puts the heavy bottle on its corner first (the table rocks while nothing stands on it),
        sets the others down gently, checks at the end"""
        return dict(order="far_first", first=inst["heavy"], gap=CAREFUL_GAP, bias_x=0.0, rand=0.0, repair="each",
                    check=True)

    def adapt_params(self, inst, history):
        """attempt >= 2: the bottle whose placement rocked the table (from the event of the failed attempt) first; the
        same imprecise habits otherwise; stands fallen bottles up and checks at the end"""
        first = None
        for h in history:
            for e in h["outcome"].get("events", []):
                if e.get("event") == "table_rocked" and e.get("released"):
                    first = e["released"]
                    break
            if first:
                break
        return dict(self.default_params(inst), first=first, repair="each", check=True)

    def blind_params(self, inst, attempt, rng):
        """the naive behaviour with a random order and random variation, no history, no look back"""
        return dict(self.default_params(inst), order="random", noise_seed=int(rng.integers(1 << 30)),
                    z_carry=float(rng.uniform(0.40, 0.45)), v=float(rng.uniform(0.12, 0.25)),
                    bias_x=float(rng.uniform(*NAIVE_BIAS)))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """oracle / naive: one attempt. adaptive: attempt 1 = naive + stands fallen bottles up after every step and checks
        at the end; attempts >= 2: the rocking bottle first. blind: naive with a random order.
        Extra kinds (controls): naive_heavy_first (naive habits, 1-4 cm drops, but the heavy bottle first),
        naive_repair (= adaptive's first attempt), naive_near_first, switch (order change only, no repair),
        oracle_noise (+-5 mm)."""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        single = ("oracle", "naive", "naive_heavy_first", "naive_repair", "oracle_noise", "naive_near_first")
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":
                params = (dict(self.default_params(inst), repair="each", check=True) if a == 0 else
                          self.adapt_params(inst, history))
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "switch":
                params = (self.default_params(inst) if a == 0 else
                          dict(self.adapt_params(inst, history), repair=None, check=False))
            elif kind == "naive_heavy_first":
                params = dict(self.default_params(inst), first=inst["heavy"])
            elif kind == "naive_repair":
                params = dict(self.default_params(inst), repair="each", check=True)
            elif kind == "naive_near_first":
                params = dict(self.default_params(inst), order="near_first")
            elif kind == "oracle_noise":
                params = dict(self.oracle_params(inst), rand=0.005, noise_seed=int(seed) * 7 + a)
            else:
                raise ValueError(kind)
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in single:
                break
        return history
