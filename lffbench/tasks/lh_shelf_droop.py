"""LH lh_shelf_droop (2026-10-16; v2 the same day): put four jars on a wall shelf. The shelf board hangs on a loose,
spring-loaded fold-down bracket with a weak catch: it stays level as long as the moment of its load about the hinge at
the wall is below the bracket's preload; once the load exceeds it the catch slips and the board folds down to its stop
within about a second, and the jars on it slide off its free edge onto the table, where they lie on their sides.
Jars set down in the middle of the board's depth -- where a careful agent puts them -- push the moment over the preload
with the third jar: a median 0.4 s (90 % within 0.8 s) after that jar is down the board is past 8 deg, and the first
jar slides off a median 1.5 s (1-3 s) after it, while the arm lifts away. The same four jars standing near the wall
(centres up to ~6 cm from it) stay under the preload. Unloaded, the board springs back level.

Family: long-horizon self-damage ("state_restoration"): a later stage (the third placement) undoes the earlier ones, a
moment after it was done, once the gripper has let go. Within an attempt the agent has to notice the fallen jars, stand
them up again (by their knobs) and put them back -- near the wall, or they slide off again; across attempts it has to
load the shelf near the wall. R4: "I put them down badly / dropped them too hard / my arm knocked them" vs "the shelf
gives way under a load far from the wall".

v1 -> v2 (measured, scratchpad risk sweeps): v1 had a preloaded spring without a catch (preload 0.45-0.49 N m, 0.6 N m/rad,
damping 2.6-3.4, a slow first-order droop). Jars placed with their centres 6-7 cm from the wall overloaded it only with
the fourth jar, and the board drooped (or the jars slid off) only during the final settle, after `finish` -- 15 of 74
failing scripted attempts (20 %) showed nothing before finish; 4 jars 6 cm from the wall failed in 9 / 9 seeds. v2 raises
the preload (4 jars up to ~6 cm from the wall hold) and adds the catch + a soft spring + light damping, so every overload
folds the board down within a second, before the agent's next look.

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * a wall panel (2.5 cm thick, 34 cm + 4 cm long) along x on the robot's right or left (|y| of its face 25-27 cm, centre
    x in [-0.17, -0.12]), rising 4.5 cm above the shelf board;
  * the shelf board: 34 cm (along the wall) x 20 cm (deep) x 1.2 cm, light wood, top 11 cm above the table, hinged at
    the wall face (axis along x at mid-thickness); hard stop at 25 deg of droop. The agentview camera (behind the table,
    +x, looking at the robot) sees the board from its end, so the droop is an in-image rotation of the slab;
  * four square jars (red, green, blue, yellow; body 5 x 5 x 6 cm, white lid, a 2.2 cm dark ball knob on a 1.2 cm stem;
    10.2 cm tall, 0.24-0.26 kg, centre of mass 2.6 cm up) standing in a row on the table on the other side (|y| 13-17
    cm), 9 cm apart, in random order.

Hidden variable: the bracket. "loose" (14 of every 20 consecutive seeds): preload 0.62-0.68 N m; "stiff" (6 of 20):
preload 2.5 N m (no placement on the board reaches it). Both look identical. The preload is a spring (0.25 N m/rad
beyond the preload) plus a catch of 0.12 N m that fades out over the first 3 deg of droop; damping 0.45-0.60 N m s/rad.
As the board droops the jars' centres of mass (3.2 cm above the hinge axis) move outwards (+0.08 N m/rad per jar), so
with two or more jars on it nothing stops a board whose catch has slipped before its 25 deg stop. Jar-on-board friction
0.27-0.33 per jar (they slide at ~14.5 deg at 0.30 in MuJoCo's soft contacts).
Moments (jar weight 2.45 N x lever arm of its centre of mass): four jars against the wall (centres 3.5 cm from the
hinge) 0.34 N m; four at 6 cm 0.59 (holds); four at 7 cm 0.69 (over); in the middle of the depth (10.4 cm) one jar 0.25,
two 0.51 (holds), three 0.76 (over).

Repair: fallen jars lie on the table in front of the board's free edge. Held by the knob (ball, torsional friction
0.0005, condim 4) a lying jar turns between the fingers and hangs straight down (like lh_bystander's bottle cap); the
instruction says so. Put back near the wall, the board stays level.

Success (final-state judge after a settle of at least 15 s of simulated time with the arm holding still -- longer if
the board is still moving -- so that finishing right after the last placement does not beat the bracket): every jar
stands upright on the board (its bottom on the board's top, within its outline, axis within 10 deg of the board's
normal), not held; the board within 15 deg of level (a board that has folded down holds no standing jar anyway),
touched by nothing but the jars (closes "prop the board" loopholes), the gripper not touching a jar or the board, the
scene static.

Per-step monitor (server log only; F1 gives the agent success / failure): stage achieved / lost (cause: slid_off,
robot, jar:<colour>, picked_up, other) / restored per jar; board sag_start (> 1.5 deg), sag (> 8 deg = trap_fired) and
sag_end, each with the time since the placement that pushed the moment over the preload and the moment itself;
jar_fell (with the time since its own placement and since the sag started). outcome() also reports what the agent's
last images showed at `finish` (board angle, jars already off the board).

Protocol: cross (reset to the same instance before every attempt), k = 3.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

TZ = envs.TABLE_Z
G = 9.81
DT = 0.05  # control period

# ---------------------------------------------------------------- jars (square glass jars with a ball knob on the lid)
COLOURS = ["red", "green", "blue", "yellow"]
JARS = [f"jar_{c}_1" for c in COLOURS]
GLASS_RGBA = {"red": (0.78, 0.20, 0.16, 1.0), "green": (0.20, 0.58, 0.28, 1.0), "blue": (0.20, 0.38, 0.80, 1.0),
              "yellow": (0.90, 0.72, 0.15, 1.0)}
LID_RGBA = (0.93, 0.93, 0.91, 1.0)
KNOB_RGBA = (0.22, 0.22, 0.24, 1.0)
JB = 0.025  # half width of the square body (5 x 5 cm)
BODY_H = 0.060
LID_H, LID_B = 0.008, 0.026  # lid height, half width
NECK_R, NECK_H = 0.006, 0.012
KNOB_R = 0.011
JAR_H = BODY_H + LID_H + NECK_H + 2 * KNOB_R  # 0.102
KNOB_Z = BODY_H + LID_H + NECK_H + KNOB_R  # knob centre above the jar's bottom (0.091)
JAR_COM = 0.026  # centre of mass above the bottom (a filled jar)
JAR_MASS = (0.24, 0.26)
JAR_MU = (0.27, 0.33)  # jar body friction (per jar); the board's own is lower, so this one applies on the board
LID_MU = 0.8
KNOB_MU_T = 0.0005  # torsional friction of the knob (condim 4): a jar held by its knob swings down and hangs straight
JAR_ROT_DAMP = 0.004  # free-joint angular damping (air drag; damps the swing of a jar hanging from its knob)

# ---------------------------------------------------------------- shelf (fixture: wall panel + hinged board)
SHELF = "shelf_1"
SHELF_MODEL = "lff_lhs_shelf"
BOARD_L = 0.34  # along the wall (world x)
BOARD_D = 0.20  # depth, from the wall to the free edge
BOARD_T = 0.012
BOARD_TOP = 0.110  # top surface above the table when level
PANEL_T = 0.025
LIP = 0.045  # the wall rises this far above the board's top
BOARD_MASS = 0.30
BOARD_MU = 0.10  # (max rule: the jars' 0.27-0.33 applies)
GAP = 0.004  # board edge to the wall face (the board's lower corner swings towards the wall when it droops)
WOOD = (0.72, 0.55, 0.36, 1.0)
WALL_RGBA = (0.86, 0.86, 0.84, 1.0)
DROOP_MAX = np.radians(25.0)  # hard stop of the hinge (at 28 deg the free edge would touch the table)
K_SPRING = 0.25  # N m / rad: the bracket's spring beyond the preload (soft: once the catch slips the board folds down)
LATCH = 0.12  # N m: part of the preload held by the bracket's catch, released over the first LATCH_TH of droop
LATCH_TH = np.radians(3.0)
PRELOAD_LOOSE = (0.62, 0.68)  # N m (spring + catch at level)
PRELOAD_STIFF = 2.5
DAMPING = (0.45, 0.60)  # N m s / rad
P_LOOSE = 0.70  # 14 of every 20 consecutive seeds
WALL_Y = (0.25, 0.27)  # |y| of the wall face
SHELF_X = (-0.17, -0.12)  # centre of the board along the wall
SLOT_U = (-0.12, -0.04, 0.04, 0.12)  # slot positions along the board (scripted policies)
SUP_Y = (0.13, 0.17)  # |y| of the supply row (other side of the table)
SUP_U = (-0.135, -0.045, 0.045, 0.135)

# predicates / monitor
TILT_OK = np.radians(10.0)  # jar axis vs the board's normal
BOARD_OK = np.radians(15.0)  # final judge: the board within 15 deg of level (jars slide off at ~13-16 deg)
SAG0 = np.radians(1.5)  # sag_start
SAG_END = np.radians(0.5)
SAG_DEG = np.radians(8.0)  # trap fired
DEBOUNCE = 10  # control steps (0.5 s)
SETTLE_MIN, SETTLE_MAX = 300, 600  # final judge: 15-30 s of simulated time with the arm holding still

# scripted motion
Z_CARRY = 0.25  # grasp point above the table while carrying a jar (its bottom then clears the board by ~5 cm)
R_MID = GAP + BOARD_D / 2  # 0.104: the middle of the board's depth (from the wall face)
R_BACK = JB + 0.010  # 0.035: a jar against the wall (1 cm gap)
NAIVE_BIAS = 0.010  # per instance: Sonnet aims at the middle +- 1 cm (towards the wall or the free edge)
NAIVE_RAND = 0.006  # per jar
NAIVE_V, NAIVE_V_DOWN, NAIVE_WAITS = 0.35, 0.15, 0.5  # pace: ~8 s of simulated time from one placement to the next
#                                                    (Sonnet, lh_nudge_row pilot np2: 3.2-8.3 s, median 5.1 s)


def jar_xml(name, rgba):
    h = JAR_H / 2
    z0 = -h  # bottom
    m = 0.25
    parts = [("body", "box", (0, 0, z0 + BODY_H / 2), (JB, JB, BODY_H / 2), rgba, 'friction="0.3 0.005 0.0001"'),
             ("lid", "box", (0, 0, z0 + BODY_H + LID_H / 2), (LID_B, LID_B, LID_H / 2), LID_RGBA,
              f'friction="{LID_MU} 0.005 0.0001"'),
             ("neck", "cylinder", (0, 0, z0 + BODY_H + LID_H + NECK_H / 2), (NECK_R, NECK_H / 2), KNOB_RGBA,
              f'friction="{LID_MU} 0.005 0.0001"'),
             ("knob", "sphere", (0, 0, z0 + KNOB_Z), (KNOB_R,), KNOB_RGBA,
              f'friction="1 {KNOB_MU_T} 0.0001" condim="4" priority="2"')]
    ixx = m * ((2 * JB) ** 2 + BODY_H ** 2) / 12
    izz = m * (2 * (2 * JB) ** 2) / 12
    s = [f'        <inertial pos="0 0 {z0 + JAR_COM:.4f}" mass="{m}" diaginertia="{ixx:.3e} {ixx:.3e} {izz:.3e}" />']
    for nm, typ, pos, size, c, extra in parts:
        s.append(f'        <geom name="{name}_{nm}" type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" '
                 f'{extra} group="0" />')
        s.append(f'        <geom type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" contype="0" '
                 f'conaffinity="0" group="1" />')
    # a lighter label band on the body (visual only)
    s.append(f'        <geom type="box" pos="0 0 {z0 + BODY_H * 0.55:.4f}" size="{JB + 0.0006:.4f} {JB + 0.0006:.4f} 0.012" '
             f'rgba="0.95 0.93 0.86 1" contype="0" conaffinity="0" group="1" />')
    return _wrap(name, "\n".join(s), h, JB * 1.42)


def shelf_xml(name):
    """fixture frame: origin on the table top below the middle of the hinge line; local +y points from the wall towards
    the board's free edge, local x along the wall. Wall panel (static) + board (child body on a hinge about -x at the
    wall face, so a positive joint angle lowers the free edge). Wall and board do not collide with each other
    (contype 2 / conaffinity 1); both collide with everything else."""
    zc = BOARD_TOP - BOARD_T / 2  # hinge axis height (board mid-thickness)
    wall_h = BOARD_TOP + LIP
    hw = (BOARD_L + 0.04) / 2
    wall = (f'type="box" pos="0 {-PANEL_T / 2:.5f} {wall_h / 2:.5f}" size="{hw:.5f} {PANEL_T / 2:.5f} {wall_h / 2:.5f}" '
            f'rgba="{_fmt(WALL_RGBA)}"')
    g = [f'<geom name="{name}_wall" {wall} friction="0.6 0.005 0.0001" contype="2" conaffinity="1" group="0" />',
         f'<geom {wall} contype="0" conaffinity="0" group="1" />']
    slab = (f'type="box" pos="0 {BOARD_D / 2 + GAP:.5f} 0" size="{BOARD_L / 2:.5f} {BOARD_D / 2:.5f} {BOARD_T / 2:.5f}" '
            f'rgba="{_fmt(WOOD)}"')
    b = [f'<inertial pos="0 {BOARD_D / 2 + GAP:.5f} 0" mass="{BOARD_MASS}" '
         f'diaginertia="{BOARD_MASS * BOARD_D ** 2 / 12:.3e} {BOARD_MASS * BOARD_L ** 2 / 12:.3e} '
         f'{BOARD_MASS * (BOARD_L ** 2 + BOARD_D ** 2) / 12:.3e}" />',
         # stiffness / springref / damping are set per instance (apply_instance)
         f'<joint name="hinge" type="hinge" axis="-1 0 0" pos="0 0 0" limited="true" range="0 {DROOP_MAX:.4f}" '
         f'stiffness="{K_SPRING}" springref="-1" damping="3.0" armature="0.002" solreflimit="0.004 1" '
         f'solimplimit="0.95 0.99 0.001" />',
         f'<geom name="{name}_board" {slab} friction="{BOARD_MU} 0.005 0.0001" contype="2" conaffinity="1" '
         f'group="0" />',
         f'<geom {slab} contype="0" conaffinity="0" group="1" />',
         # darker front edge strip and hinge barrel (visual)
         f'<geom type="box" pos="0 {BOARD_D + GAP - 0.003:.5f} 0" size="{BOARD_L / 2 + 0.0005:.5f} 0.0032 '
         f'{BOARD_T / 2 + 0.0005:.5f}" rgba="0.50 0.36 0.22 1" contype="0" conaffinity="0" group="1" />',
         f'<geom type="cylinder" pos="0 0.004 {-BOARD_T / 2:.5f}" size="0.005 {BOARD_L / 2 - 0.01:.5f}" '
         f'quat="0.70711 0 0.70711 0" rgba="0.55 0.55 0.58 1" contype="0" conaffinity="0" group="1" />']
    nl = "\n          "
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
          {nl.join(g)}
        <body name="board" pos="0 0 {zc:.5f}">
          {nl.join(b)}
        </body>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {wall_h:.5f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{hw:.5f} {BOARD_D:.5f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
"""


def _wrap_h(a):
    """wrap to [-pi/2, pi/2) (the gripper is symmetric under 180 deg)"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


class DroopSkills(Skills):
    """Skills that feed the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class ShelfDroop(LFFTask):
    name = "lh_shelf_droop"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Restore", "Plan")
    protocol = "cross"
    max_attempts = 3
    # careful: ~10 commands per jar (40) + standing up and re-placing three fallen jars (~40) + looking / corrections
    agent_max_steps = 120
    instruction = ("Put the four jars (red, green, blue, yellow) on the shelf. At the end all four must be standing "
                   "upright on the shelf board, and the gripper must be open and clear of the jars and the shelf. "
                   "(Each jar has a round knob on its lid; held by the knob, a jar hangs straight down.)")
    instruction_indirect = ("Put the four little square jars with knobbed lids up on the wooden shelf by the wall. "
                            "When you are done, all of them have to stand upright on the shelf board, and the gripper "
                            "must be open and away from the jars and the shelf. (Held by its round knob, a jar hangs "
                            "straight down.)")
    agent_object_names = ["jars (red, green, blue, yellow; each with a round knob on its lid -- held by the knob it "
                          "hangs straight down)", "wooden shelf board on the wall panel"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs = []
        for i, (c, j) in enumerate(zip(COLOURS, JARS)):
            cat = register_generated(f"LffLhsJar{c.capitalize()}", jar_xml(f"lff_lhs_jar_{c}", GLASS_RGBA[c]))
            objs.append((j, cat, f"jar{i}_region", (-0.30 + 0.12 * i, 0.30, -0.29 + 0.12 * i, 0.31)))
        sh = register_generated("LffLhsShelf", shelf_xml(SHELF_MODEL), free=False)
        fx = [(SHELF, sh, "shelf_region", (0.20, -0.31, 0.21, -0.30))]  # moved in apply_instance
        return write_bddl(self.name, "Put the four jars on the shelf", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.shelf_bid = m.body_name2id(self.env.fixtures_dict[SHELF].root_body)
        self.jid = m.joint_name2id(f"{SHELF}_hinge")
        self.qa, self.da = int(m.jnt_qposadr[self.jid]), int(m.jnt_dofadr[self.jid])
        self.board_bid = int(m.jnt_bodyid[self.jid])
        pre = f"{SHELF}_{SHELF_MODEL}"
        self.board_gid = m.geom_name2id(pre + "_board")
        self.wall_gid = m.geom_name2id(pre + "_wall")
        self.jgeoms = {j: envs.obj_geom_ids(self.env, j) for j in JARS}
        self.jbid = {j: int(m.geom_bodyid[self.jgeoms[j][0]]) for j in JARS}  # body that carries the inertial
        self.body_gid = {j: m.geom_name2id(f"{j}_lff_lhs_jar_{c}_body") for c, j in zip(COLOURS, JARS)}
        self.knob_gid = {j: m.geom_name2id(f"{j}_lff_lhs_jar_{c}_knob") for c, j in zip(COLOURS, JARS)}
        self.jdof = {j: envs.free_joint_addr(self.env, j)[1] for j in JARS}
        # per-geom lookup tables for the substep contact check
        self.owner = np.full(m.ngeom, -1, int)  # jar index
        for i, j in enumerate(JARS):
            self.owner[self.jgeoms[j]] = i
        self.is_robot = np.zeros(m.ngeom, bool)
        self.is_robot[envs.robot_geom_ids(self.env)] = True
        self.is_board = np.zeros(m.ngeom, bool)
        self.is_board[self.board_gid] = True
        self._sub_rob, self._sub_obj, self._sub_board = {}, {}, set()
        sim = self.env.sim
        if not getattr(sim, "_lhs_wrapped", False):
            orig = sim.step

            def step(*a, **k):
                orig(*a, **k)
                t = getattr(sim, "_lhs_task", None)
                if t is not None:
                    t._bracket()
                    if t.mon is not None:
                        t._substep()
            sim.step = step
            sim._lhs_wrapped = True
        sim._lhs_task = self
        self.inst = None
        self.mon = None
        self.sk = None

    def _bracket(self):
        """the bracket's catch (every MuJoCo substep): a torque holding the board up that fades out over the first
        LATCH_TH of droop, so a board that starts to droop does not stop part-way but folds down to its stop"""
        d = self.env.sim.data
        th = float(d.qpos[self.qa])
        d.qfrc_applied[self.da] = -LATCH * float(np.clip(1.0 - th / LATCH_TH, 0.0, 1.0))

    def _substep(self):
        d = self.env.sim.data
        n = d.ncon
        if not n:
            return
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        for a, b in zip(g1, g2):
            pa, pb = self.owner[a], self.owner[b]
            if self.is_board[a] or self.is_board[b]:
                o = b if self.is_board[a] else a
                if self.owner[o] < 0:  # the board touched by something that is not a jar (robot, table, ...)
                    self._sub_board.add("robot" if self.is_robot[o] else (self.env.sim.model.geom_id2name(int(o)) or "?"))
                continue
            if pa < 0 and pb < 0:
                continue
            if pa >= 0 and pb >= 0:
                if pa != pb:
                    self._sub_obj[int(pa)] = int(pb)
                    self._sub_obj[int(pb)] = int(pa)
                continue
            p, o = (pa, b) if pa >= 0 else (pb, a)
            if self.is_robot[o]:
                self._sub_rob[int(p)] = True

    # ------------------------------------------------------------ instances
    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        loose = bool((int(seed) * 7) % 20 < round(20 * P_LOOSE))  # balanced: 14 of every 20 consecutive seeds
        side = "right" if rng.random() < 0.5 else "left"  # where the wall stands (robot's view)
        s = -1.0 if side == "right" else 1.0
        x0 = float(rng.uniform(*SHELF_X))
        yw = float(rng.uniform(*WALL_Y))
        preload = float(rng.uniform(*PRELOAD_LOOSE)) if loose else PRELOAD_STIFF
        damping = float(rng.uniform(*DAMPING))
        ys = -s * float(rng.uniform(*SUP_Y))
        order = rng.permutation(4)
        start = {COLOURS[i]: [float(x0 + SUP_U[order[i]] + rng.uniform(-0.01, 0.01)), float(ys + rng.uniform(-0.008, 0.008))]
                 for i in range(4)}
        yaw0 = {c: float(rng.uniform(-0.3, 0.3)) for c in COLOURS}
        mu = {c: float(rng.uniform(*JAR_MU)) for c in COLOURS}
        mass = {c: float(rng.uniform(*JAR_MASS)) for c in COLOURS}
        return dict(seed=int(seed), loose=loose, bracket="loose" if loose else "stiff", side=side, shelf_x=x0,
                    wall_y=float(s * yw), preload=preload, damping=damping, stiffness=K_SPRING, start=start, yaw0=yaw0,
                    mu=mu, mass=mass, naive_bias=float(rng.uniform(-NAIVE_BIAS, NAIVE_BIAS)))

    def apply_instance(self, inst):
        self.inst = inst
        m, d = self.env.sim.model, self.env.sim.data
        s = -1.0 if inst["side"] == "right" else 1.0
        # fixture local +y (towards the free edge) = world -s*y
        m.body_pos[self.shelf_bid] = [inst["shelf_x"], inst["wall_y"], TZ]
        m.body_quat[self.shelf_bid] = [1.0, 0.0, 0.0, 0.0] if s < 0 else [0.0, 0.0, 0.0, 1.0]
        wb = BOARD_MASS * G * (BOARD_D / 2 + GAP)  # the board's own weight about the hinge
        m.jnt_stiffness[self.jid] = inst["stiffness"]
        # spring + catch hold the board level up to a load moment of `preload` (the catch's part fades with the droop)
        m.qpos_spring[self.qa] = -(inst["preload"] - LATCH + wb) / inst["stiffness"]
        m.dof_damping[self.da] = inst["damping"]
        d.qpos[self.qa] = 0.0
        d.qvel[self.da] = 0.0
        d.qfrc_applied[self.da] = -LATCH
        for c, j in zip(COLOURS, JARS):
            b = self.jbid[j]
            M = inst["mass"][c]
            m.body_mass[b] = M
            m.body_inertia[b] = [M * ((2 * JB) ** 2 + BODY_H ** 2) / 12] * 2 + [M * (2 * JB) ** 2 / 6]
            m.geom_friction[self.body_gid[j]] = [inst["mu"][c], 0.005, 0.0001]
            m.dof_damping[self.jdof[j] + 3:self.jdof[j] + 6] = JAR_ROT_DAMP
        self.env.sim.forward()
        for c, j in zip(COLOURS, JARS):
            envs.place_upright(self.env, j, inst["start"][c], yaw=inst["yaw0"][c], clearance=0.0005)

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = DroopSkills(self.env, self, recorder=recorder)
        self._sub_rob, self._sub_obj, self._sub_board = {}, {}, set()
        n = len(JARS)
        self.mon = dict(state=[False] * n, raw_since=[0] * n, ever=[False] * n, lost=[False] * n, lost_cause=[None] * n,
                        events=[], last_robot=[None] * n, last_obj=[None] * n, placed_step=[None] * n,
                        last_place=None, raw_prev=[False] * n, raw_on=[None] * n, in_sag=False, sag_start=None, sag_fired=False, sag_max=0.0, n_sag=0,
                        fell_in_sag=[], max_deg=0.0, board_touch=[], n_steps=0, finish_step=None)
        return self.sk

    # ------------------------------------------------------------ geometry / predicates
    def angle(self):
        return float(self.env.sim.data.qpos[self.qa])

    def board_frame(self):
        d = self.env.sim.data
        return d.xpos[self.board_bid].copy(), d.xmat[self.board_bid].reshape(3, 3).copy()

    def to_board(self, p):
        o, R = self.board_frame()
        return R.T @ (np.asarray(p, float) - o)

    def board_point(self, u, r, h=BOARD_T / 2):
        """world point at (u along the wall, r from the hinge across the board, h above the hinge axis) in the board
        frame"""
        o, R = self.board_frame()
        return o + R @ np.array([u, r, h])

    def level_point(self, u, r):
        """world xy of board coordinates (u, r) with the board level (planning)"""
        s = -1.0 if self.inst["side"] == "right" else 1.0
        return np.array([self.inst["shelf_x"] - s * u, self.inst["wall_y"] - s * r])

    def board_top_z(self):
        return TZ + BOARD_TOP

    def jar_axis(self, j):
        return self.env.sim.data.xmat[self.jbid[j]].reshape(3, 3)[:, 2].copy()

    def jar_bottom(self, j):
        return envs.obj_pos(self.env, j) - self.jar_axis(j) * JAR_H / 2

    def knob_xyz(self, j):
        return self.env.sim.data.geom_xpos[self.knob_gid[j]].copy()

    def tilt(self, j):
        return float(np.arccos(np.clip(self.jar_axis(j)[2], -1.0, 1.0)))

    def held(self, j):
        return self.sk is not None and self.sk.holding(j)

    def on_board(self, i, extra=0.0):
        """jar i stands on the shelf board: its bottom (centre) on the board's top, within the board's outline, its
        axis within 10 deg of the board's normal, not held"""
        j = JARS[i]
        o, R = self.board_frame()
        ax = R.T @ self.jar_axis(j)
        if ax[2] < np.cos(TILT_OK + extra * 5):
            return False
        bc = R.T @ (self.jar_bottom(j) - o)
        if not (BOARD_T / 2 - 0.004 <= bc[2] <= BOARD_T / 2 + 0.005 + extra):
            return False
        if abs(bc[0]) > BOARD_L / 2 + extra or not (GAP - extra <= bc[1] <= GAP + BOARD_D + extra):
            return False
        return not self.held(j)

    def stages(self):
        return [self.on_board(i) for i in range(len(JARS))]

    def lever(self, j):
        """horizontal distance (m) of the jar's centre of mass from the hinge line, towards the free edge"""
        o, R = self.board_frame()
        n = R[:, 1].copy()
        n[2] = 0.0
        n /= max(np.linalg.norm(n), 1e-9)
        return float((self.env.sim.data.xipos[self.jbid[j]] - o) @ n)

    def moment(self, only_state=True):
        """load moment (N m) of the jars standing on the board about the hinge"""
        mon = self.mon
        tot = 0.0
        for i, j in enumerate(JARS):
            on = mon["state"][i] if (only_state and mon is not None) else self.on_board(i)
            if on:
                tot += self.inst["mass"][COLOURS[i]] * G * self.lever(j)
        return float(tot)

    def r_cm(self, i):
        """distance of jar i's bottom centre from the wall face (cm), in the board frame"""
        return 100 * float(self.to_board(self.jar_bottom(JARS[i]))[1])

    # ------------------------------------------------------------ monitor
    def _t(self):
        return int(self.sk.n_steps) if self.sk is not None else 0

    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        mon["n_steps"] += 1
        t = self._t()
        rob, obj, brd = dict(self._sub_rob), dict(self._sub_obj), set(self._sub_board)
        self._sub_rob, self._sub_obj, self._sub_board = {}, {}, set()
        for i in rob:
            if not self.held(JARS[i]):
                mon["last_robot"][i] = int(t)
        for i, k in obj.items():
            mon["last_obj"][i] = [COLOURS[k], int(t)]
        for b in brd:
            if b not in mon["board_touch"]:
                mon["board_touch"].append(b)
        mon["_board_touch_now"] = sorted(brd)
        th = self.angle()
        mon["max_deg"] = max(mon["max_deg"], float(np.degrees(th)))
        if th > np.radians(5.0):
            mon["droop_t"] = int(t)  # last step with the board drooped > 5 deg (it springs back once the jars are off)
        self._sag(th, t)
        if mon["n_steps"] % 2:
            return
        for i in range(len(JARS)):
            raw = self.on_board(i, 0.002 if mon["state"][i] else 0.0)
            if raw and not mon["raw_prev"][i]:
                mon["raw_on"][i] = int(t)  # set down on the board (raw; the stage event below is debounced)
            mon["raw_prev"][i] = raw
            if raw == mon["state"][i]:
                mon["raw_since"][i] = 0
                continue
            mon["raw_since"][i] += 2
            if mon["raw_since"][i] < DEBOUNCE:
                continue
            mon["state"][i] = raw
            mon["raw_since"][i] = 0
            t0 = int(t - DEBOUNCE + 1)
            c = COLOURS[i]
            if raw:
                restored = mon["lost"][i]
                e = dict(stage=c, event="restored" if restored else "achieved", step=t0,
                         r_cm=round(self.r_cm(i), 1), moment_Nm=round(self.moment(), 3))
                if restored:
                    e["after"] = mon["lost_cause"][i]
                mon["events"].append(e)
                mon["ever"][i] = True
                mon["placed_step"][i] = t0
                mon["last_place"] = (c, t0)
            else:
                mon["lost"][i] = True
                cause = self._cause(i, t)
                mon["lost_cause"][i] = cause
                e = dict(stage=c, event="lost", step=t0, cause=cause, board_deg=round(float(np.degrees(th)), 1))
                mon["events"].append(e)
                if cause == "slid_off":
                    ps = mon["placed_step"][i]
                    ss = mon["sag_start"]["step"] if mon["sag_start"] else None
                    sc = mon.get("sag_cause")
                    mon["events"].append(dict(jar=c, event="jar_fell", step=t0,
                                              dt_since_placed_s=None if ps is None else round((t0 - ps) * DT, 2),
                                              dt_since_sag_start_s=None if ss is None else round((t0 - ss) * DT, 2),
                                              after_placing=None if sc is None else sc[0],
                                              dt_since_cause_s=None if sc is None else round((t0 - sc[1]) * DT, 2),
                                              in_final_settle=mon["finish_step"] is not None,
                                              board_deg=round(float(np.degrees(th)), 1)))
                    mon["fell_in_sag"].append(c)

    def _sag(self, th, t):
        mon = self.mon
        if not mon["in_sag"]:
            if th > SAG0:
                mon["in_sag"] = True
                mon["sag_fired"] = False
                mon["sag_max"] = th
                mon["fell_in_sag"] = []
                lp = self._last_set_down(t)
                mon["sag_cause"] = lp
                e = dict(event="sag_start", step=int(t), board_deg=round(float(np.degrees(th)), 2),
                         moment_Nm=round(self.moment(only_state=False), 3),
                         preload_Nm=round(float(self.inst["preload"]), 3),
                         after_placing=None if lp is None else lp[0],
                         dt_since_placement_s=None if lp is None else round((t - lp[1]) * DT, 2),
                         n_on_board=int(sum(self.on_board(i) for i in range(len(JARS)))))
                mon["sag_start"] = e
                mon["events"].append(e)
            return
        mon["sag_max"] = max(mon["sag_max"], th)
        if not mon["sag_fired"] and th > SAG_DEG:
            mon["sag_fired"] = True
            mon["n_sag"] += 1
            ss = mon["sag_start"]
            lp = self._last_set_down(t)  # the placement that drove the board past 8 deg (sag_start may be earlier:
            mon["sag_cause"] = lp        # two jars in the middle already make it creep a few degrees)
            mon["events"].append(dict(event="sag", step=int(t), board_deg=round(float(np.degrees(th)), 2),
                                      moment_Nm=round(self.moment(only_state=False), 3),
                                      after_placing=None if lp is None else lp[0],
                                      sag_start_after=ss["after_placing"],
                                      dt_since_placement_s=None if lp is None else round((t - lp[1]) * DT, 2),
                                      dt_since_sag_start_s=round((t - ss["step"]) * DT, 2),
                                      in_final_settle=mon["finish_step"] is not None))
        mon["below"] = mon.get("below", 0) + 1 if th < SAG_END else 0
        if mon["below"] >= 2 * DEBOUNCE:  # back level for 1 s (the jars that slid off have been registered)
            mon["in_sag"] = False
            mon["below"] = 0
            mon["events"].append(dict(event="sag_end", step=int(t), max_deg=round(float(np.degrees(mon["sag_max"])), 1),
                                      fell=list(mon["fell_in_sag"]),
                                      duration_s=round((t - mon["sag_start"]["step"]) * DT, 2)))

    def _last_set_down(self, t):
        """(colour, step) of the jar most recently set down on the board (raw predicate) that still stands on it"""
        mon = self.mon
        cands = [(mon["raw_on"][i], COLOURS[i]) for i in range(len(JARS))
                 if mon["raw_on"][i] is not None and mon["raw_prev"][i] and mon["raw_on"][i] <= t]
        if not cands:
            return None
        st, c = max(cands)
        return (c, st)

    def _cause(self, i, t, window=60):
        mon = self.mon
        if self.held(JARS[i]):
            return "picked_up"
        if (mon["in_sag"] and mon["sag_max"] > np.radians(5.0)) or (mon.get("droop_t") is not None
                                                                   and t - mon["droop_t"] <= window):
            return "slid_off"
        lr, lo = mon["last_robot"][i], mon["last_obj"][i]
        if lr is not None and t - lr <= window:
            return "robot"
        if lo is not None and t - lo[1] <= window:
            return "jar:" + lo[0]
        return "other"

    def _finalize_monitor(self, raw):
        mon = self.mon
        t = self._t()
        for i in range(len(JARS)):
            if raw[i] == mon["state"][i]:
                continue
            mon["state"][i] = raw[i]
            c = COLOURS[i]
            if raw[i]:
                restored = mon["lost"][i]
                e = dict(stage=c, event="restored" if restored else "achieved", step=t, final=True)
                if restored:
                    e["after"] = mon["lost_cause"][i]
                mon["events"].append(e)
                mon["ever"][i] = True
            else:
                mon["lost"][i] = True
                cause = self._cause(i, t, window=10 ** 6)
                mon["lost_cause"][i] = cause
                mon["events"].append(dict(stage=c, event="lost", step=t, cause=cause, final=True))

    # ------------------------------------------------------------ outcome
    def _touching(self):
        d = self.env.sim.data
        jar_touch, board_touch = False, False
        for k in range(d.ncon):
            c = d.contact[k]
            a, b = c.geom1, c.geom2
            if (self.is_robot[a] and self.owner[b] >= 0) or (self.is_robot[b] and self.owner[a] >= 0):
                jar_touch = True
            if (self.is_robot[a] and self.is_board[b]) or (self.is_robot[b] and self.is_board[a]):
                board_touch = True
        return jar_touch, board_touch

    def _moving(self):
        d = self.env.sim.data
        if abs(float(d.qvel[self.da])) > np.radians(0.3):
            return True
        for j in JARS:
            da = self.jdof[j]
            if np.linalg.norm(d.qvel[da:da + 3]) > 0.01 or np.linalg.norm(d.qvel[da + 3:da + 6]) > 0.3:
                return True
        return False

    def outcome(self, inst):
        sk = self.sk
        mon = self.mon
        mon["finish_step"] = self._t()
        # what the agent's last images show (finish returns no image): board angle and the jars already off the board
        th_fin = self.angle()
        off_fin = [COLOURS[i] for i in range(4) if mon["ever"][i] and not self.on_board(i) and not self.held(JARS[i])]
        max_fin = float(mon["max_deg"])
        # settle with the arm holding still: >= 15 s of simulated time, up to 30 s while the board or a jar still moves
        # (the droop is slow: finishing right after the last placement must not beat it)
        n = 0
        while n < SETTLE_MAX:
            sk.hold(10)
            n += 10
            if n >= SETTLE_MIN and not self._moving() and not mon["in_sag"]:
                break
        settle_s = n * DT
        p0 = {j: envs.obj_pos(self.env, j) for j in JARS}
        th0 = self.angle()
        jar_touch = board_touch = False
        board_other = set()
        for _ in range(20):
            sk.hold(1)
            jt, bt = self._touching()
            jar_touch, board_touch = jar_touch or jt, board_touch or bt
            board_other |= {b for b in mon.get("_board_touch_now", []) if b != "robot"}
        moved = max(float(np.linalg.norm(envs.obj_pos(self.env, j) - p0[j])) for j in JARS)
        static = moved < 0.003 and abs(self.angle() - th0) < np.radians(0.3)
        raw = self.stages()
        self._finalize_monitor(raw)
        th = self.angle()
        held = [COLOURS[i] for i, j in enumerate(JARS) if self.held(j)]
        level = th <= BOARD_OK
        propped = sorted(board_other)
        success = bool(all(raw) and level and static and not held and not jar_touch and not board_touch and not propped)
        ev = list(mon["events"])
        n_reg = sum(e.get("event") == "lost" and e.get("cause") != "picked_up" for e in ev)
        n_lost = sum(e.get("event") == "lost" for e in ev)
        n_res = sum(e.get("event") == "restored" and e.get("after") != "picked_up" for e in ev)
        fell = [e for e in ev if e.get("event") == "jar_fell"]
        sags = [e for e in ev if e.get("event") == "sag"]
        sag_vis = [e for e in sags if not e.get("in_final_settle")]
        starts = [e for e in ev if e.get("event") == "sag_start"]
        ever = mon["ever"]
        lost_final = [i for i in range(4) if not raw[i] and ever[i]]
        never = [i for i in range(4) if not raw[i] and not ever[i]]
        if success:
            failure = "none"
        elif held:
            failure = "held"
        elif propped or board_touch:
            failure = "propped"
        elif jar_touch:
            failure = "touching"
        elif lost_final:
            causes = {mon["lost_cause"][i] for i in lost_final}
            failure = ("slid_off" if "slid_off" in causes else "knocked_off" if "robot" in causes else "regressed")
        elif never:
            failure = "not_on_shelf"
        elif not level:
            failure = "board_drooped"
        else:
            failure = "not_static"
        # ---- F2 text
        parts = []
        if success:
            parts.append("All four jars stand upright on the shelf board.")
        for i in range(4):
            if raw[i]:
                continue
            c, j = COLOURS[i], JARS[i]
            if c in held:
                parts.append(f"The {c} jar is still held by the gripper.")
                continue
            pb = self.to_board(self.jar_bottom(j))
            lying = self.tilt(j) > np.radians(45)
            if pb[2] < -0.02:  # below the board: on the table
                where = "lies on its side on the table" if lying else "stands on the table"
                dist = 100 * float(pb[1] - (GAP + BOARD_D))
                txt = f"The {c} jar {where}" + (f", {dist:.0f} cm out from the shelf's front edge." if dist > 0
                                                 else ", under the shelf.")
            elif lying:
                txt = f"The {c} jar lies on its side on the shelf board."
            else:
                txt = f"The {c} jar is not standing on the shelf board."
            if ever[i]:
                why = mon["lost_cause"][i]
                txt += {"slid_off": " It had stood on the shelf; it slid off when the shelf board drooped under the load.",
                        "robot": " It had stood on the shelf; the robot knocked it off.",
                        }.get(why, " It had stood on the shelf earlier.")
            parts.append(txt)
        if all(raw) and not success:
            if not level:
                parts.append(f"The shelf board droops by {np.degrees(th):.0f} degrees.")
            if held:
                pass
            elif jar_touch or board_touch:
                parts.append("The gripper is still touching " + ("a jar." if jar_touch else "the shelf board."))
            elif propped:
                parts.append("The shelf board rests on something below it.")
            elif not static:
                parts.append("Something was still moving.")
        if sags:
            parts.append(f"During the attempt the shelf board drooped {len(sags)} time(s) (down to {mon['max_deg']:.0f} "
                         f"degrees) and {len(fell)} time(s) a jar slid off; {n_res} time(s) a jar was put back.")
        trap = bool(sags)
        lp = lambda e: e.get("dt_since_placement_s")  # noqa: E731
        return dict(success=success, stages={COLOURS[i]: bool(raw[i]) for i in range(4)}, events=ev,
                    n_regressions=int(n_reg), n_lost=int(n_lost), n_restored=int(n_res), n_fell=len(fell),
                    n_sag=len(sags), trap_fired=trap, sag_visible=bool(sag_vis),
                    recovered_in_attempt=bool(trap and success),
                    first_sag_dt_s=lp(sags[0]) if sags else None,
                    first_sag_start_dt_s=lp(starts[0]) if starts else None,
                    first_sag_after=sags[0].get("after_placing") if sags else None,
                    first_fall_dt_s=fell[0].get("dt_since_cause_s") if fell else None,
                    fell_before_finish=bool([e for e in fell if not e.get("in_final_settle")]),
                    board_deg_at_finish=round(float(np.degrees(th_fin)), 1), off_at_finish=off_fin,
                    max_droop_before_finish_deg=round(max_fin, 1),
                    visible_before_finish=bool(sag_vis or off_fin or max_fin > np.degrees(SAG_DEG)),
                    fell_jars=sorted({e["jar"] for e in fell}), failure=failure, held=held, touching=bool(jar_touch),
                    board_touched=bool(board_touch), propped=propped, static=bool(static),
                    board_deg=round(float(np.degrees(th)), 2), max_droop_deg=round(mon["max_deg"], 1),
                    r_cm={COLOURS[i]: round(self.r_cm(i), 1) for i in range(4)},
                    moment_Nm=round(self.moment(only_state=False), 3), preload_Nm=round(float(inst["preload"]), 3),
                    loose=bool(inst["loose"]), settle_s=settle_s, finish_step=mon["finish_step"],
                    detail=" ".join(parts))

    # ------------------------------------------------------------ scripted primitives (move_to + gripper only)
    def _go(self, xyz, yaw=None, speed=None, tol=0.004, max_steps=300):
        return self.sk.move_to(xyz, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)

    def _lift_to(self, z, speed=0.15):
        cur = self.sk.eef_pos()
        if cur[2] < TZ + z - 0.005:
            self._go([cur[0], cur[1], TZ + z], speed=speed)

    def _settle_jar(self, j, max_checks=30):
        d = self.env.sim.data
        da = self.jdof[j]
        for _ in range(max_checks):
            if np.linalg.norm(d.qvel[da:da + 3]) < 0.01 and np.linalg.norm(d.qvel[da + 3:da + 6]) < 0.2:
                return
            self.sk.hold(2)

    def wait_board(self, max_steps=200):
        """wait (arm still) until the board stops moving"""
        for _ in range(max_steps // 5):
            if abs(float(self.env.sim.data.qvel[self.da])) < np.radians(0.3) and not self.mon["in_sag"]:
                return
            self.sk.hold(5)

    def lying(self, j):
        return self.tilt(j) > np.radians(45)

    def pick(self, j, z_carry=Z_CARRY, yaw=0.0, v=0.2, err=None, tries=2, v_down=0.10, waits=1.0):
        """grasp the jar by its knob and lift it to carry height. Standing jar: from above, wrist yaw `yaw`. Lying jar:
        fingers across its axis; lifted, it turns between the fingers and hangs straight down."""
        sk = self.sk
        for _ in range(tries):
            self._settle_jar(j)
            lying = self.lying(j)
            k = self.knob_xyz(j)
            if lying:
                ax = self.jar_axis(j)[:2]
                gy = float(np.arctan2(ax[1], ax[0])) if np.linalg.norm(ax) > 0.2 else sk.yaw
            else:
                gy = yaw
            q = k[:2] + (np.asarray(err, float) if err is not None else 0.0)
            sk.set_gripper(False, steps=8)
            cur = sk.eef_pos()
            self._go([cur[0], cur[1], max(cur[2], TZ + min(z_carry, 0.22))], speed=0.15)
            self._go([q[0], q[1], max(k[2] + 0.06, TZ + min(z_carry, 0.22))], yaw=gy, speed=v)
            sk.hold(max(1, int(4 * waits)))
            k = self.knob_xyz(j)
            q = k[:2] + (np.asarray(err, float) if err is not None else 0.0)
            self._go([q[0], q[1], k[2] + 0.015], speed=v_down)
            self._go([q[0], q[1], k[2]], speed=0.04, tol=0.003, max_steps=150)
            sk.set_gripper(True, steps=15)
            if lying:  # lift slowly until it hangs free, let the swing die down
                cur = sk.eef_pos()
                self._go([cur[0], cur[1], TZ + KNOB_Z + 0.04], speed=0.08)
                sk.hold(20)
            cur = sk.eef_pos()
            self._go([cur[0], cur[1], TZ + z_carry], speed=max(0.12, v_down))
            sk.hold(max(1, int(5 * waits)))
            if self.held(j):
                return True
            sk.set_gripper(False, steps=8)
            cur = sk.eef_pos()
            self._go([cur[0], cur[1], cur[2] + 0.05], speed=0.1)
        return False

    def put(self, j, u, r, z_carry=Z_CARRY, h_rel=0.010, v=0.2, yaw=0.0, v_down=0.10, waits=1.0):
        """carry the held jar over board point (u, r) (planned with the board level), straight down until its bottom is
        h_rel above the board top, let go, lift back to carry height"""
        sk = self.sk
        xy = self.level_point(u, r)
        rel = self.jar_bottom(j)[:2] - sk.eef_pos()[:2]
        q = xy - rel
        cur = sk.eef_pos()
        self._go([q[0], q[1], max(cur[2], TZ + z_carry)], yaw=yaw, speed=v)
        sk.hold(max(1, int(12 * waits)))  # let the hanging jar stop swinging
        rel = self.jar_bottom(j)[:2] - sk.eef_pos()[:2]
        q = xy - rel
        dz = sk.eef_pos()[2] - envs.obj_min_z(self.env, j)
        zr = self.board_top_z() + dz + h_rel
        self._go([q[0], q[1], zr + 0.02], speed=v_down)
        self._go([q[0], q[1], zr], speed=0.03, tol=0.003, max_steps=150)
        sk.set_gripper(False, steps=12)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], TZ + z_carry], speed=max(0.15, v_down))
        sk.hold(3)

    def place(self, i, u, r, p, rng, rand=0.0):
        """one stage: pick jar i from wherever it is and put it on the board at (u, r) +- rand"""
        j = JARS[i]
        e = rng.uniform(-rand, rand, 2) if rand else np.zeros(2)
        gerr = rng.uniform(-0.002, 0.002, 2) if rand else None
        kw = dict(z_carry=p.get("z_carry", Z_CARRY), v=p.get("v", 0.2), v_down=p.get("v_down", 0.10),
                  waits=p.get("waits", 1.0))
        if not self.pick(j, err=gerr, **kw):
            return False
        self.put(j, u + e[0], r + e[1], h_rel=p.get("h_rel", 0.010), **kw)
        self._settle_jar(j)
        return True

    def supply_order(self):
        """jar indices by their supply position: the one whose slot is farthest from the robot first (Sonnet fills a
        row from the far end)"""
        return sorted(range(4), key=lambda i: -self.inst["start"][COLOURS[i]][0])

    def slot_u(self, k):
        """board u of slot k (k = 0 farthest from the robot)"""
        s = -1.0 if self.inst["side"] == "right" else 1.0
        # u runs along local x = -s * world x  -> farthest (largest world x) first
        return -s * SLOT_U[3 - k]

    def fallen(self):
        return [i for i in range(4) if self.mon["ever"][i] and not self.on_board(i) and not self.held(JARS[i])]

    def repair(self, r_target, rng, p, rand=0.0, used=None):
        """move every jar on the board to depth r_target (same slot), then stand up the fallen ones and put them into the
        free slots at r_target"""
        sk = self.sk
        self.wait_board()
        slots = dict(used or {})
        for i in range(4):
            if self.on_board(i) and abs(self.r_cm(i) / 100 - r_target) > 0.02:
                self.place(i, slots.get(i, self.slot_u(0)), r_target, p, rng, rand)
                self.wait_board()
        for i in self.fallen():
            free = [k for k in range(4) if self.slot_u(k) not in slots.values()] or [0]
            u = slots.get(i, self.slot_u(free[0]))
            slots[i] = u
            self.place(i, u, r_target, p, rng, rand)
            self.wait_board()
        sk.hold(5)
        return slots

    def execute(self, inst, params):
        rng = np.random.default_rng(params.get("noise_seed", 0))
        r = float(params["r"])
        rand = float(params.get("rand", 0.0))
        order = params.get("order") or self.supply_order()
        used = {}
        mode_back = False
        for k, i in enumerate(order):
            u = self.slot_u(k)
            used[i] = u
            rr = R_BACK + params.get("back_off", 0.0) if mode_back else r
            self.place(i, u, rr, params, rng, rand)
            # looks at the jar it just placed (Sonnet): re-places it if it is not standing on the board
            if not self.on_board(i) and params.get("self_check", True):
                self.place(i, u, rr, params, rng, rand)
            if params.get("look_back") and (self.fallen() or self.angle() > np.radians(3.0)):
                # the shelf sags / jars fell: from now on load it near the wall, and move what is on it there too
                mode_back = True
                used = self.repair(R_BACK + params.get("back_off", 0.0), rng, params, rand, used)
        if params.get("final_check"):
            for _ in range(2):
                if all(self.stages()):
                    break
                rr = R_BACK + params.get("back_off", 0.0) if (mode_back or params.get("look_back")) else r
                used = self.repair(rr, rng, params, rand, used)
        if not params.get("fast_finish"):
            self._lift_to(Z_CARRY)
            self.sk.hold(5)

    # naive = Sonnet-like (DESIGN_INSIGHTS §3; lh_nudge_row / lh_bystander pilots): careful top-down pick and place, jars
    # set down in the visual middle of the shelf's depth, released ~1-4 cm above, wrist yaw 0, looks at the jar it just
    # placed, no look back at the others before finishing
    def default_params(self, inst):
        sd = int(inst["seed"])
        return dict(r=R_MID + float(inst["naive_bias"]), rand=NAIVE_RAND, noise_seed=sd * 13,
                    h_rel=0.010 + 0.030 * ((sd * 7) % 11) / 10, z_carry=Z_CARRY, look_back=False, final_check=False,
                    v=NAIVE_V, v_down=NAIVE_V_DOWN, waits=NAIVE_WAITS)

    def oracle_params(self, inst):
        return dict(r=R_BACK, rand=0.0, h_rel=0.005, z_carry=Z_CARRY, look_back=True, final_check=True)

    def adapt_params(self, inst, history):
        """from attempt 2: load the shelf near the wall (same imprecision), look back after every jar and at the end"""
        return dict(self.default_params(inst), r=R_BACK + 0.008, look_back=True, final_check=True)

    def blind_params(self, inst, attempt, rng):
        """the naive behaviour with random variation (depth on the board, order, heights), no history, no look back"""
        order = [int(x) for x in rng.permutation(4)]
        return dict(self.default_params(inst), noise_seed=int(rng.integers(1 << 30)), r=float(rng.uniform(0.06, 0.15)),
                    order=order, z_carry=float(rng.uniform(0.24, 0.28)), h_rel=float(rng.uniform(0.005, 0.04)),
                    v=float(rng.uniform(0.12, 0.3)))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        single = ("oracle", "naive", "oracle_noise", "naive_front", "naive_back", "naive_check_mid", "naive_fast")
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":  # attempt 1: naive + looks back after every jar; repairs near the wall
                params = (dict(self.default_params(inst), look_back=True, final_check=True, back_off=0.008) if a == 0
                          else self.adapt_params(inst, history))
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "switch":  # control: cross-attempt learning only (near the wall from attempt 2), no repair
                params = (self.default_params(inst) if a == 0 else
                          dict(self.default_params(inst), r=R_BACK + 0.008))
            elif kind == "oracle_noise":
                params = dict(self.oracle_params(inst), rand=0.005, noise_seed=int(seed) * 7 + a)
            elif kind == "naive_front":  # control: naive, but jars set down near the free edge
                params = dict(self.default_params(inst), r=GAP + BOARD_D - JB - 0.015)
            elif kind == "naive_back":  # control: naive pace and imprecision, jars against the wall -> no droop
                params = dict(self.default_params(inst), r=R_BACK + 0.008)
            elif kind == "naive_check_mid":  # control: looks back and repairs, but puts the jars back in the middle
                params = dict(self.default_params(inst), final_check=True)
            elif kind == "naive_fast":  # control: naive at the fastest pace, finishing right after the last jar
                params = dict(self.default_params(inst), v=0.6, v_down=0.3, waits=0.1, h_rel=0.03, fast_finish=True)
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
