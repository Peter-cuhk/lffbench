"""LH lh_tidy_stack v2 (2026-10-16): tidy the table -- stack the four storage boxes (red, green, blue, yellow; alike,
6 x 6 x 5 cm) into one pile on a grey mat, in any order, then stand the cup on top. In 18 of every 20 instances one box
(random colour) is an empty, soft carton: it carries one thing, but as soon as a second thing comes to rest above it,
it gives way -- standing it folds sideways, lying on a side face it is crushed flat -- and everything above it slides
off the pile: the stage that made the pile two higher destroys the stage before it.

Family: long-horizon self-disturbance ("state_restoration"): the later stages' ordinary action (putting the next box on)
undoes earlier ones; the agent has to see which box gave way, take it out of the pile and rebuild with it on top (within
an attempt) or remember its colour (across attempts). The cheapest explanation of a toppled pile ("I did not centre the
box", "I let go too high") is wrong: perfectly centred boxes let go 5 mm above the pile topple it just the same
(control naive_centred).

v1 -> v2 (Sonnet 5.5 pilot np3_lh_tidy_stack, seeds 3000 / 3002 / 3003, k=3; logs measured by the coordinator and
replayed here): in s3002 attempt 2 boxes that had fallen on their sides were restacked lying (a lying box looks the
same: they are nearly cubes); the soft box lay on its side at level 2 under two boxes and the cup, and the v1 law, which
measured the load along the box axis only, saw -0.06 N and never fired -> judged a success. Sonnet also presses what it
puts down 0.5-1.5 cm below contact (its own logs: the gripper stops 0.6-1.5 cm above the commanded release height), and
in s3003 a cup pressed onto the soft box on top -- where it belongs -- crushed it under one item (v1: 2.45 N). v2:
  (1) the load is measured along the world vertical, whichever face the carton rests on (`vertical_loads`: the downward
      push on its upper side, limited by the support under it);
  (2) a carton lying on a side face (seam vertical) cannot fold under gravity, so it is crushed instead: both halves
      flatten towards the face they rest on, the upper half by 70 % and the lower by 30 % (runtime geom sizes +
      envs.refit_bounds), the top becomes a 2-3 cm step and what is on it tips off;
  (3) only what rests on the carton counts, at most with its own weight (+0.3 N for impacts): something the robot still
      holds, the robot pushing down through an item it has just let go of, and the fingers pressing the held carton
      itself onto something do not count; the robot's own fingers / hand pushing on its top do (a hard press may crush
      it: the closed gripper 3 mm onto its top does; logged as cause "press", not counted as the trap unless it costs a
      stage);
  (4) items weigh 200 g (box and cup; v1 180 / 160 g) and f_crit = 3.45 N: one item 1.96 N, two 3.92 N;
  (5) the rigid boxes' joints are stiffer (20 Nm/rad, stiff limit; at 3 Nm/rad Sonnet's cup press on a 4-high pile bent
      every box ~4 deg and the pile leaned 10 deg in the replay); after a fold the carton straightens along a rest
      shape that moves back to straight over 1.5 s (released at once, a stiff spring flipped it onto its side);
  (6) stacking tolerance 3 cm instead of 2.5 (a correctly stacked oracle_noise pile failed at 2.51 cm).
v1 code and results: scratchpad lh_tidy_stack_v1.py, results/_old/lh_tidy_stack_v1/.

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * a grey mat (13 x 13 cm, visual only), centre x in [-0.11, -0.05], |y| <= 5 cm;
  * four storage boxes (6 x 6 x 5 cm, bright red / green / blue / yellow with a darker lid line, 200 g; friction 0.5)
    15, 18.5, 22 and 25.5 cm (+-1 cm) from the mat centre, at random angles 30-150 deg from forward on its left and right,
    yaw +-17 deg, colours shuffled;
  * a dark teal cup (5 cm wide, 7 cm tall, 200 g) on the robot's side of the mat, 23-26 cm away.

Hidden variable: which box is the soft carton (60 g; mass is not observable). Instances come in balanced blocks of 20
consecutive seeds ((seed x 7) mod 20 = k): k >= 18 benign (no soft box); otherwise the soft box is the one at rank
k mod 4 of the distance to the mat (0 = nearest = the bottom of a nearest-first pile, 3 = the top): 5 / 5 / 4 / 4.
A nearest-first stacker therefore meets it below the top in 14 of 20 instances.

Mechanism (physics; nothing is teleported). Every box is built alike: a lower half (the root body) and an upper half
joined by a ball joint at mid-height (the seam is invisible). In the three rigid boxes the joint is locked (range
0.003 rad, stiffness 3 Nm/rad). The soft carton follows a load law (`_law_step`, every MuJoCo substep): its compressive
load (above, low-passed with 0.4 s) is compared with f_crit (instance key "f_crit": the fragility knob). Below it the
joint keeps 3 Nm/rad (one box 1.3 cm off-centre tilts it < 1 deg: looks like the rigid boxes). Above it:
  * standing or upside down (seam horizontal): the joint stiffness drops to 0.03 Nm/rad -- a column with m g h ~ 0.27
    Nm/rad on it is unstable, centred or not -- and its rest shape bends 12 deg towards a per-instance side: the upper
    half folds sideways (38-54 deg measured, joint limit 54 deg) and what is on it slides off;
  * lying on a side face (seam vertical): crushed as above within ~0.1 s.
Unloaded (< 0.5 N) for 0.3 s it recovers over ~1.5 s (still visibly bent / flattened in the images of the next one or
two commands). One item on it is carried (a box let go 4 cm above adds < 0.5 N to the filtered load), two are not. So
the pile survives only with the soft box on top (carrying just the cup): soft at level 1 -> it gives way when box 3
lands (boxes 2 and 3 fall), level 2 -> when box 4 lands, level 3 -> when the cup lands (box 4 and the cup fall; standing,
the cup ends on its side, which a top-down gripper cannot undo, so that attempt cannot be repaired). Measured (v2, naive
without rebuilding, the 14 trap instances of seeds 1000-1019): standing 14/14 folded at exactly that stage, the soft
box stayed upright in the pile 14/14 (fold 37-55 deg), the cup fell 4/4 at level 3; lying (control naive_lying: the
soft box starts on a side face and is stacked lying) crushed at exactly that stage in every trap instance.

Why not the linear spring of the brief (a ball joint with one fixed stiffness k): the load that k must carry (the cup)
and the load that must topple it (one more box) differ by a factor of ~2-3, and an inverted pendulum close to its
critical load amplifies placement offsets (tilt = W e / (k - m g h)); measured (results/lh_tidy_stack/linear.json,
instance key law="linear"): no k gives both a soft box that carries the cup on top without sagging and a pile that
topples however carefully it is centred (k <= 0.25 Nm/rad: the correct pile fails 5-10/10; k = 0.5: a centred
level-3 soft box carries box + cup, 2/2). A load threshold (walls that
hold until they buckle) keeps the soft box looking and behaving rigid until the second item, independent of centring.

Success predicate (from the instruction): all four boxes in one chain -- the bottom one on the mat (lowest point <= 8 mm
above the table, centre within 3.5 cm of the mat centre in x and y: its footprint on the 13 cm mat), each next one
resting on the previous (its lowest point within 1 cm of the top of the box below, centres within 3 cm), every box
flat (some face within 10 deg of horizontal: a fallen box cannot be stood up again by a vertical gripper, so a box on
its side counts), not bent (kink <= 10 deg) and not crushed (<= 15 %) --; the cup upright (<= 10 deg) on the top box
(bottom within 1 cm, centre within 3 cm); nothing held, the gripper not touching anything, the scene static (< 3 mm
in 1 s). Tolerance: 3 cm = the box's half width, i.e. the upper box's centre of mass is over the lower box's top face
(physically resting on it; the static check rejects anything that does not stay). Sonnet's measured placement error is
0.4-1.3 cm, so two placements differ by up to 2.6 cm plus a few mm of landing drift (v2: the 2.5 cm of v1 failed one
correctly stacked oracle_noise pile, seed 1008, at 2.51 cm).

Event monitor (server log only; F1 gives the agent only success / failure): stage predicates red / green / blue /
yellow (box in the pile) and cup (cup on a complete pile), debounced over 0.5 s: achieved / lost (cause buckle / robot
/ picked_up / other) / restored; `buckle` (= trap fired: load, the robot's share of it, fold or crush, lying, level,
items on it), `collapse` (a buckle that cost a stage). Outcome: success, failure (none / held / touching / collapsed /
regressed / not_stacked / cup_fallen / cup_not_on_top / not_static), trap_fired, recovered_in_attempt, n_buckles,
n_collapses, buckle_how, max_kink_deg, max_crush, detail (F2). The law's state rides along in task.mon["law"]; a
crushed carton's geom sizes are restored at every reset.

Protocol: cross (reset to the same instance before every attempt), k = 3.
"""
import mujoco
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

TZ = envs.TABLE_Z
G = 9.81
COLOURS = ["red", "green", "blue", "yellow"]
BOXES = [f"box_{c}_1" for c in COLOURS]
COL = dict(zip(BOXES, COLOURS))
CUP, MAT = "cup_1", "mat_1"
RGBA = {"red": (0.80, 0.16, 0.13, 1.0), "green": (0.17, 0.58, 0.25, 1.0), "blue": (0.16, 0.34, 0.80, 1.0),
        "yellow": (0.90, 0.74, 0.14, 1.0)}
CUP_RGBA = (0.10, 0.45, 0.50, 1.0)  # dark teal ceramic, darker inside
CUP_INSIDE = (0.20, 0.20, 0.19, 1.0)
MAT_RGBA = (0.45, 0.45, 0.47, 1.0)

# ---------------------------------------------------------------- boxes (all four built alike: base + upper part)
HX, HY, H = 0.030, 0.030, 0.050  # 6 x 6 x 5 cm (the gripper opens 7.9 cm)
HB = 0.025  # base part height = height of the seam (ball joint) above the bottom: mid-height, so the upper
#              half can fold ~56 deg before its edge reaches the surface below
HU = H - HB
BOX_MASS = 0.20  # a filled storage box (v1: 0.18)
SOFT_MASS = 0.06  # the empty carton (mass is not observable by the agent)
BOX_MU = 0.5
CONTACT = 'solimp="0.95 0.99 0.001" solref="0.01 1"'
# soft-carton law (see SoftLaw): stiff until the compressive load through its walls exceeds F_CRIT, then soft
K_HI = 3.0  # Nm/rad: the intact soft carton (one box 1.3 cm off-centre tilts it ~0.5 deg; a cup pressed onto it
#              bends it ~4 deg)
K_RIGID = 20.0  # Nm/rad + a stiff joint limit: the rigid boxes (at 3 Nm/rad a cup pressed onto a 4-high pile bent every
#                 box 4 deg and the pile leaned 10 deg -- replay of pilot s3003 on v2; at 20 for the soft carton too
#                 its folds flipped it onto its side 4/14 times)
K_LO = 0.03  # Nm/rad: buckled (one box on it destabilises it: m g h ~ 0.12 Nm/rad)
DAMP = 0.01
F_CRIT = 3.45  # N, vertical load from above (resting things + whatever the robot pushes or holds onto it): one
#                item weighs 1.96 N and a held item pressed onto the carton adds <= ~0.9 N before it slips in the
#                fingers (measured, t10_press); two items weigh 3.92 N (v1: 2.45 N with 1.77 / 1.57 N items)
F_REL = 0.5  # N: unloaded below this ...
REL_T = 0.3  # ... for this long -> it springs back
TAU_F = 0.4  # s: low-pass of the load (impacts of a box let go 1-4 cm above do not count)
K_RISE = 1.5  # s: the folded carton straightens along a reference that moves back to straight in this time
BEND_REF = np.radians(12.0)  # reference bend of the buckled carton (the weak side; direction per instance)
RANGE_SOFT, RANGE_RIGID = 0.95, 0.003  # rad: kink limit (54 deg) of the soft carton; the rigid boxes are locked
# v2: a carton lying on a side face (its seam vertical) cannot fold under gravity; it is crushed instead: both halves
# flatten towards the surface below, the upper half much more, so the top becomes a 2-3 cm step and what is on it
# tips off (geom sizes changed at runtime + envs.refit_bounds)
CRUSH_ALPHA = (0.30, 0.70)  # fraction of the height lost by the base half / the upper half when fully crushed
CRUSH_TAU = (0.08, 0.7)  # s: crushing / recovering
FOLD_MIN_UP = 0.7  # |seam normal . vertical| above this: the fold (upright or upside down); below: the crush

# ---------------------------------------------------------------- cup
CUP_R, CUP_H, CUP_MASS = 0.025, 0.070, 0.20  # v1: 0.16

# ---------------------------------------------------------------- mat and layout
MAT_HALF = 0.065  # 13 x 13 cm
MAT_X, MAT_Y = (-0.11, -0.05), (-0.05, 0.05)
BOX_D = (0.15, 0.185, 0.22, 0.255)  # distances of the four boxes from the mat centre (+- jitter)
BOX_X = (-0.33, 0.07)  # world x range of the box positions (reach)
BOX_YMAX = 0.30
BOX_YAW = 0.30
P_BENIGN = 0.10  # 2 of every 20 consecutive seeds: no soft box

# ---------------------------------------------------------------- predicates
TILT_OK = np.radians(10.0)
KINK_OK = np.radians(10.0)
ON_MAT = 0.035  # bottom box centre within 3.5 cm of the mat centre (x and y): its footprint is on the 13 cm mat
STACK_XY = 0.030  # centre of a box within 3 cm (= the half width) of the centre of the box below: its centre of
#                   mass is over the lower box's top face (v2; 2.5 cm judged a resting box 2.51 cm off as not stacked)
STACK_Z = 0.010
CUP_XY = 0.030
DEBOUNCE = 10  # control steps (0.5 s)

# ---------------------------------------------------------------- scripted motion
Z_CARRY = 0.18
NAIVE_ERR = (0.004, 0.013)  # Sonnet's placement error 0.4-1.3 cm (random direction)
NAIVE_GAP = (0.010, 0.040)  # Sonnet lets go 1-4 cm above the surface
NAIVE_BIAS = (0.0, 0.020)  # forward bias of the first box on the mat (oblique camera)


def box_xml(name, rgba):
    """storage box: base part (root body, 1.6 cm) + upper part (child body) on a ball joint at the seam. Rigid boxes
    get a ~0 kink range at reset; the soft carton a 31.5 deg range and the load-dependent stiffness of SoftLaw.
    Inertials are replaced per instance (mass)."""
    mb, mu = BOX_MASS * HB / H, BOX_MASS * HU / H
    zb = -H / 2 + HB / 2
    dark = [0.72 * c for c in rgba[:3]] + [1.0]
    g = [f'        <inertial pos="0 0 {zb:.5f}" mass="{mb:.5f}" diaginertia="1e-5 1e-5 1e-5" />',
         f'        <geom name="{name}_base" type="box" size="{HX:.5f} {HY:.5f} {HB / 2:.5f}" pos="0 0 {zb:.5f}" '
         f'rgba="{_fmt(rgba)}" friction="{BOX_MU} 0.005 0.0001" {CONTACT} group="0" />',
         f'        <geom type="box" size="{HX:.5f} {HY:.5f} {HB / 2:.5f}" pos="0 0 {zb:.5f}" rgba="{_fmt(rgba)}" '
         f'contype="0" conaffinity="0" group="1" />',
         f'        <body name="{name}_upper" pos="0 0 {-H / 2 + HB:.5f}">',
         f'          <joint name="{name}_kink" type="ball" pos="0 0 0" range="0 {RANGE_SOFT}" stiffness="{K_HI}" '
         f'damping="{DAMP}" armature="0.0001" />',
         f'          <inertial pos="0 0 {HU / 2:.5f}" mass="{mu:.5f}" diaginertia="1e-5 1e-5 1e-5" />',
         f'          <geom name="{name}_upper" type="box" size="{HX:.5f} {HY:.5f} {HU / 2:.5f}" pos="0 0 {HU / 2:.5f}" '
         f'rgba="{_fmt(rgba)}" friction="{BOX_MU} 0.005 0.0001" {CONTACT} group="0" />',
         f'          <geom type="box" size="{HX:.5f} {HY:.5f} {HU / 2:.5f}" pos="0 0 {HU / 2:.5f}" rgba="{_fmt(rgba)}" '
         f'contype="0" conaffinity="0" group="1" />',
         # lid line (visual only): a darker band 8 mm below the top, on all four boxes
         f'          <geom type="box" size="{HX + 0.0006:.5f} {HY + 0.0006:.5f} 0.0012" pos="0 0 {HU - 0.008:.5f}" '
         f'rgba="{_fmt(dark)}" contype="0" conaffinity="0" group="1" />',
         '        </body>']
    return _wrap(name, "\n".join(g), H / 2, float(np.hypot(HX, HY)))


def cup_xml(name):
    """open cup, 5 cm wide, 7 cm tall: 12 wall boxes + an octagonal base of two boxes (box contacts; a light MuJoCo
    cylinder creeps on a flat surface, lh_nudge_row). Visual: a smooth cylinder with a dark inside."""
    r, h = CUP_R, CUP_H
    nseg, wt, bt = 12, 0.004, 0.006
    g = [f'        <inertial pos="0 0 {-h / 2 + 0.022:.4f}" mass="{CUP_MASS}" diaginertia="1.5e-4 1.5e-4 1.0e-4" />']
    s = r * np.sin(np.pi / nseg) + 0.0008
    rc = r * np.cos(np.pi / nseg) - wt / 2
    for k in range(nseg):
        a = 2 * np.pi * k / nseg
        q = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
        g.append(f'        <geom name="{name}_w{k}" type="box" size="{wt / 2:.5f} {s:.5f} {(h - 0.001) / 2:.5f}" '
                 f'pos="{rc * np.cos(a):.5f} {rc * np.sin(a):.5f} 0.0005" quat="{_fmt(q)}" rgba="{_fmt(CUP_RGBA)}" '
                 f'friction="0.8 0.005 0.0001" {CONTACT} group="0" />')
    a0 = r * 0.70
    for k, ang in enumerate((0.0, np.pi / 4)):
        q = [np.cos(ang / 2), 0, 0, np.sin(ang / 2)]
        g.append(f'        <geom name="{name}_b{k}" type="box" size="{a0:.5f} {a0:.5f} {bt / 2:.5f}" '
                 f'pos="0 0 {-h / 2 + bt / 2:.5f}" quat="{_fmt(q)}" rgba="{_fmt(CUP_RGBA)}" '
                 f'friction="0.6 0.005 0.0001" {CONTACT} group="0" />')
    g.append(f'        <geom type="cylinder" size="{r:.5f} {h / 2:.5f}" rgba="{_fmt(CUP_RGBA)}" contype="0" '
             f'conaffinity="0" group="1" />')
    g.append(f'        <geom type="cylinder" size="{r - wt:.5f} 0.0005" pos="0 0 {h / 2 - 0.0004:.5f}" '
             f'rgba="{_fmt(CUP_INSIDE)}" contype="0" conaffinity="0" group="1" />')
    return _wrap(name, "\n".join(g), h / 2, r)


def mat_xml(name):
    g = (f'        <geom type="box" size="{MAT_HALF:.5f} {MAT_HALF:.5f} 0.0005" rgba="{_fmt(MAT_RGBA)}" contype="0" '
         f'conaffinity="0" group="1" />')
    return _wrap(name, g, 0.0005, MAT_HALF * 1.42)


def free_adr(env, name):
    """(qpos, dof) address of the object's FREE joint. envs.free_joint_addr takes joints[0], which for an object with a
    jointed child body is the child's joint (robosuite lists the subtree's joints before the free joint it appends)."""
    m = env.sim.model
    for jn in env.objects_dict[name].joints:
        j = m.joint_name2id(jn)
        if int(m.jnt_type[j]) == 0:
            return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    raise KeyError(name)


def place(env, name, xy, yaw=0.0, surface_z=TZ, clearance=0.0005, tilt=None):
    """teleport upright (yaw about z; `tilt` rad about the object's own x axis first, e.g. pi/2 = lying on a side
    face -- controls only), lowest collision point `clearance` above surface_z (reset only)"""
    qa, da = free_adr(env, name)
    d = env.sim.data
    d.qpos[qa:qa + 3] = [xy[0], xy[1], surface_z + 0.3]
    q = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    if tilt:
        w1, z1 = q[0], q[3]
        c, s_ = np.cos(tilt / 2), np.sin(tilt / 2)
        q = np.array([w1 * c, w1 * s_, z1 * s_, z1 * c])  # q_yaw * q_x(tilt)
    d.qpos[qa + 3:qa + 7] = q
    d.qvel[da:da + 6] = 0.0
    env.sim.forward()
    d.qpos[qa + 2] -= envs.obj_min_z(env, name) - (surface_z + clearance)
    env.sim.forward()


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


def _lr(v):
    return f"{100 * abs(v):.1f} cm to the {'left' if v > 0 else 'right'}"


def _wrap_q(a, period):
    """wrap an angle to [-period/2, period/2)"""
    return float((a + period / 2) % period - period / 2)


class StackSkills(Skills):
    """Skills that feed the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class TidyStack(LFFTask):
    name = "lh_tidy_stack"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Restore", "Plan")
    protocol = "cross"
    max_attempts = 3
    # careful: 5 pick-and-places x ~8 commands (40) + a collapse: unstack 1-2 boxes (~16) and rebuild 3-4 (~32)
    # + looking / small corrections (~30), with some margin
    agent_max_steps = 150
    instruction = ("Tidy up the table: stack the four storage boxes (red, green, blue, yellow) into a single pile on the "
                   "grey mat, in any order you like, and then put the cup upright on top of the pile. At the end each "
                   "box must rest flat on the box below it (the bottom one on the mat), the cup must stand on the top "
                   "box, and the gripper must be open and clear of everything.")
    instruction_indirect = ("Clean up the table: pile the four coloured boxes on top of each other on the dark grey "
                            "square, in whatever order, and finish by standing the teal mug on top of the pile. In the "
                            "end every box has to sit flat on the one below it (the lowest on the grey square), the mug "
                            "has to stand on the top box, and the gripper must be open and away from everything.")
    agent_object_names = ["storage boxes (red, green, blue, yellow)", "grey mat", "cup"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs = []
        for i, (c, b) in enumerate(zip(COLOURS, BOXES)):
            cat = register_generated(f"LffLtsBox{c.capitalize()}", box_xml(f"lff_lts_box_{c}", RGBA[c]))
            objs.append((b, cat, f"box{i}_region", (-0.30 + 0.13 * i, 0.28, -0.29 + 0.13 * i, 0.29)))
        cat = register_generated("LffLtsCup", cup_xml("lff_lts_cup"))
        objs.append((CUP, cat, "cup_region", (-0.30, -0.29, -0.29, -0.28)))
        mat = register_generated("LffLtsMat", mat_xml("lff_lts_mat"), free=False)
        fx = [(MAT, mat, "mat_region", (0.0, -0.29, 0.01, -0.28))]
        return write_bddl(self.name, "Stack the four boxes on the grey mat and put the cup on top", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.mat_bid = m.body_name2id(self.env.fixtures_dict[MAT].root_body)
        self.jid, self.base_bid, self.upper_bid, self.base_gid, self.upper_gid = {}, {}, {}, {}, {}
        for b in BOXES:
            for g in envs.obj_geom_ids(self.env, b):
                n = m.geom_id2name(g)
                if n.endswith("_upper"):
                    self.upper_gid[b] = g
                elif n.endswith("_base"):
                    self.base_gid[b] = g
            self.upper_bid[b] = int(m.geom_bodyid[self.upper_gid[b]])
            self.base_bid[b] = int(m.geom_bodyid[self.base_gid[b]])
            j = next(m.joint_name2id(jn) for jn in self.env.objects_dict[b].joints if jn.endswith("_kink"))
            self.jid[b] = j
        # every geom of each half (collision and visual) with its compiled size / pos: the crush rescales them
        self.half_geoms = {}
        for b in BOXES:
            hg = []
            for bid in (self.base_bid[b], self.upper_bid[b]):
                gs = [g for g in range(int(m.body_geomadr[bid]), int(m.body_geomadr[bid] + m.body_geomnum[bid]))
                      if int(m.geom_type[g]) == 6]
                hg.append((bid, gs, m.geom_size[gs].copy(), m.geom_pos[gs].copy()))
            self.half_geoms[b] = hg
        self.cup_gids = list(envs.obj_geom_ids(self.env, CUP))
        self.obj_bodies = {b: [self.base_bid[b], self.upper_bid[b]] for b in BOXES}
        self.obj_bodies[CUP] = [envs.body_id(self.env, CUP)]
        # per-geom lookup: owner index 0-3 = boxes, 4 = cup, -1 other; robot part 1 = finger, 2 = hand / arm
        self.owner = np.full(m.ngeom, -1, int)
        for i, b in enumerate(BOXES):
            self.owner[[self.base_gid[b], self.upper_gid[b]]] = i
        self.owner[self.cup_gids] = 4
        self.part = np.zeros(m.ngeom, int)
        for g in envs.robot_geom_ids(self.env):
            n = m.geom_id2name(g) or ""
            self.part[g] = 1 if "finger" in n else 2
        self.inst = None
        self.law = None
        self.mon = None
        self.sk = None
        self._sub_rob = {}
        self._held = set()  # owner indices (0-3 boxes, 4 cup) held by the gripper, refreshed every control step
        self._rest_cap = np.inf  # upper bound of the resting load on the soft carton, refreshed every control step
        sim = self.env.sim
        if not getattr(sim, "_lts_wrapped", False):
            orig = sim.step

            def step(*a, **k):
                t = getattr(sim, "_lts_task", None)
                if t is not None and t.law is not None:
                    t._law_step()
                orig(*a, **k)
                if t is not None and t.mon is not None:
                    t._substep()
            sim.step = step
            sim._lts_wrapped = True
        sim._lts_task = self

    # ------------------------------------------------------------ instances
    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        mx, my = float(rng.uniform(*MAT_X)), float(rng.uniform(*MAT_Y))
        ca = np.pi + float(rng.uniform(-0.5, 0.5))  # the cup stands on the robot's side of the mat, 23-26 cm away
        cd = float(rng.uniform(0.23, 0.26))
        cup = [mx + cd * np.cos(ca), my + cd * np.sin(ca)]
        for _ in range(500):
            ds = np.array(BOX_D) + rng.uniform(-0.01, 0.01, 4)
            ang = rng.uniform(np.radians(30), np.radians(150), 4) * rng.choice([-1.0, 1.0], 4)
            pos = np.stack([mx + ds * np.cos(ang), my + ds * np.sin(ang)], 1)
            if not (np.all((pos[:, 0] >= BOX_X[0]) & (pos[:, 0] <= BOX_X[1])) and np.all(np.abs(pos[:, 1]) <= BOX_YMAX)):
                continue
            dd = np.linalg.norm(pos[:, None] - pos[None], axis=2) + np.eye(4)
            if dd.min() < 0.12 or np.linalg.norm(pos - np.asarray(cup), axis=1).min() < 0.14:
                continue
            break
        else:
            raise RuntimeError("layout")
        perm = rng.permutation(4)
        start = {BOXES[i]: [float(v) for v in pos[perm[i]]] for i in range(4)}
        yaw = {b: float(rng.uniform(-BOX_YAW, BOX_YAW)) for b in BOXES}
        mat = np.array([mx, my])
        near = sorted(BOXES, key=lambda b: float(np.linalg.norm(np.asarray(start[b]) - mat)))
        # balanced blocks of 20 consecutive seeds: 2 benign (no soft box); else the soft carton is the box at rank
        # r = 0..3 (0 = nearest the mat, i.e. the bottom of a nearest-first pile; 3 = the top): 5 / 5 / 4 / 4
        k = (int(seed) * 7) % 20
        benign = k >= round(20 * (1 - P_BENIGN))
        rank = None if benign else k % 4
        soft = None if benign else near[rank]
        return dict(seed=int(seed), mat=[mx, my], cup=cup, start=start, yaw=yaw, cup_yaw=float(rng.uniform(-np.pi, np.pi)),
                    soft=soft, soft_colour=COL[soft] if soft else None, benign=bool(benign),
                    soft_rank_nearest=rank, nearest_order=[COL[b] for b in near],
                    buckle_dir=float(np.pi / 2 if rng.random() < 0.5 else -np.pi / 2),
                    f_crit=F_CRIT, naive_gap=float(rng.uniform(*NAIVE_GAP)), naive_bias=float(rng.uniform(*NAIVE_BIAS)))

    def apply_instance(self, inst):
        self.inst = inst
        m, d = self.env.sim.model, self.env.sim.data
        m.body_pos[self.mat_bid] = [inst["mat"][0], inst["mat"][1], TZ + 0.0005]
        for b in BOXES:
            j = self.jid[b]
            soft = b == inst["soft"]
            M = SOFT_MASS if soft else BOX_MASS
            for bid, frac, h in ((self.base_bid[b], HB / H, HB), (self.upper_bid[b], HU / H, HU)):
                mm = M * frac
                m.body_mass[bid] = mm
                m.body_inertia[bid] = [mm * (4 * HY * HY + h * h) / 12, mm * (4 * HX * HX + h * h) / 12,
                                       mm * (4 * HX * HX + 4 * HY * HY) / 12]
            m.jnt_range[j] = [0.0, RANGE_SOFT if soft else RANGE_RIGID]
            m.jnt_solref[j] = [0.02, 1.0] if soft else [0.004, 1.0]  # rigid boxes: a stiff limit
            m.jnt_stiffness[j] = K_HI if soft else K_RIGID
            qa, da = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            m.qpos_spring[qa:qa + 4] = [1.0, 0.0, 0.0, 0.0]
            d.qpos[qa:qa + 4] = [1.0, 0.0, 0.0, 0.0]
            d.qvel[da:da + 3] = 0.0
            self._set_crush(b, 0.0, 2, -1.0)  # undo a crush left over from the previous attempt
            place(self.env, b, inst["start"][b], yaw=inst["yaw"][b], tilt=inst.get("lying", {}).get(COL[b]))
        place(self.env, CUP, inst["cup"], yaw=inst["cup_yaw"])
        self.law = None
        if inst["soft"]:
            self.law = dict(box=inst["soft"], f=0.0, fmax=0.0, buckled=False, n=0, rel_t=0.0, t=0.0, buckles=[],
                            f_crit=float(inst.get("f_crit", F_CRIT)), dir=float(inst["buckle_dir"]),
                            mode=str(inst.get("law", "threshold")), how=None, crush=0.0, crush_set=0.0,
                            crush_axis=2, crush_sign=-1.0, f_rest=0.0, f_robot=0.0)
            if self.law["mode"] == "linear":  # control only: the brief's fixed-stiffness spring (no load threshold)
                m.jnt_stiffness[self.jid[inst["soft"]]] = float(inst["k_lin"])

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.law = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = StackSkills(self.env, self, recorder=recorder)
        self._sub_rob = {}
        self._held = set()
        self._rest_cap = np.inf
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ soft-carton law
    def _part_axial(self, gid, axis):
        """sum of contact forces on geom gid (from any other geom), projected on `axis` (world)"""
        m, d = self.env.sim.model, self.env.sim.data
        n = d.ncon
        if not n:
            return 0.0
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        idx = np.nonzero((g1 == gid) | (g2 == gid))[0]
        tot = 0.0
        f6 = np.zeros(6)
        for i in idx:
            mujoco.mj_contactForce(m._model, d._data, int(i), f6)
            fw = d.contact[int(i)].frame.reshape(3, 3).T @ f6[:3]  # force on geom2, world frame
            tot += float(fw @ axis) * (-1.0 if g1[i] == gid else 1.0)
        return tot

    def vertical_loads(self, b):
        """vertical contact forces on box b (N), whichever face it rests on: (resting, robot, held, support).
        resting = downward push on its upper side (contact point above its centre) from things resting on it;
        robot = the same from the robot's own fingers / hand pushing on it (not counted while the robot holds b
        itself); held = the same from something the robot holds (pressing it down onto b); support = upward push on
        its lower side. Its two halves never touch (parent and child), so only real contacts count."""
        m, d = self.env.sim.model, self.env.sim.data
        n = d.ncon
        if not n:
            return 0.0, 0.0, 0.0, 0.0
        gb, gu = self.base_gid[b], self.upper_gid[b]
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        idx = np.nonzero((g1 == gb) | (g1 == gu) | (g2 == gb) | (g2 == gu))[0]
        if not len(idx):
            return 0.0, 0.0, 0.0, 0.0
        cz = float(d.xpos[self.base_bid[b]][2])  # the root body sits at the box's centre
        held = self._held
        b_held = BOXES.index(b) in held
        rest = robot = via = sup = 0.0
        f6 = np.zeros(6)
        for i in idx:
            c = d.contact[int(i)]
            mujoco.mj_contactForce(m._model, d._data, int(i), f6)
            fz = float(c.frame.reshape(3, 3)[:, 2] @ f6[:3])  # world z of the force on geom2
            mine1 = g1[i] in (gb, gu)
            fz = -fz if mine1 else fz  # force on box b
            other = int(g2[i] if mine1 else g1[i])
            pz = float(c.pos[2])
            if pz > cz + 0.004 and fz < 0.0:
                o = self.owner[other]
                if self.part[other] > 0:
                    if not b_held:
                        robot -= fz
                elif o >= 0 and o in held:
                    via -= fz
                else:
                    rest -= fz
            elif pz < cz - 0.004 and fz > 0.0:
                sup += fz
        # what rests on it pushes at most with its own weight: the robot pressing down through an item it has let go
        # of (Sonnet releases with the cup pushed up against the hand, pilot s3003) is not a resting load
        rest = min(rest, self._rest_cap)
        return rest, robot, via, sup

    def compression(self, b):
        """compressive force (N) on the carton, (total, resting): the downward push on its upper side from things
        resting on it plus the robot's fingers / hand pushing directly on it, limited by the support under it --
        along the world vertical, whichever face it rests on (v1 measured it along the box axis only: a carton lying
        on its side never buckled). Something the robot still holds does not count: Sonnet presses what it puts down
        0.5-1.5 cm below contact (pilot), which must not crush a carton that will carry the item once let go."""
        rest, robot, via, sup = self.vertical_loads(b)
        return float(min(rest + robot, sup)), float(min(rest, sup))

    def _law_step(self):
        """per MuJoCo substep: low-pass the compressive load; above f_crit the carton gives way -- standing or upside
        down it folds (its kink joint loses its stiffness, its rest shape bends towards its weak side and the load
        topples it), lying on a side face it is crushed (both halves flatten, the upper half more); unloaded for
        REL_T it slowly recovers its shape"""
        law = self.law
        b = law["box"]
        m = self.env.sim.model
        dt = float(m.opt.timestep)
        tot, rest = self.compression(b)
        law["f"] += (tot - law["f"]) * dt / TAU_F
        law["f_rest"] += (rest - law["f_rest"]) * dt / TAU_F
        law["fmax"] = max(law["fmax"], law["f"])
        j = self.jid[b]
        qa = int(m.jnt_qposadr[j])
        if law["mode"] == "linear":  # control: constant stiffness; a "buckle" = the kink passes 15 deg (re-arms < 5)
            kk = self.kink(b)
            if not law["buckled"] and kk > np.radians(15.0):
                law["buckled"] = True
                law["n"] += 1
                law["buckles"].append(dict(t=round(law["t"], 3), load_N=round(law["f"], 2)))
                self._on_buckle()
            elif law["buckled"] and kk < np.radians(5.0):
                law["buckled"] = False
            law["t"] += dt
            return
        if not law["buckled"]:
            if law["f"] > law["f_crit"]:
                law["buckled"] = True
                law["n"] += 1
                law["rel_t"] = 0.0
                R = self.env.sim.data.xmat[self.base_bid[b]].reshape(3, 3)
                law["unbend_q0"] = None
                if abs(R[2, 2]) >= FOLD_MIN_UP:  # seam horizontal: fold
                    law["how"] = "fold"
                    u = R.T @ np.array([np.cos(law["dir"]), np.sin(law["dir"]), 0.0])
                    ax = np.array([-u[1], u[0], 0.0])
                    ax /= max(np.linalg.norm(ax), 1e-9)
                    m.qpos_spring[qa:qa + 4] = [np.cos(BEND_REF / 2), *(np.sin(BEND_REF / 2) * ax)]
                else:  # seam vertical (lying on a side face): crush towards the face it rests on
                    law["how"] = "crush"
                    k = int(np.argmax(np.abs(R[2, :2])))
                    law["crush_axis"] = k
                    law["crush_sign"] = -float(np.sign(R[2, k]) or 1.0)  # local direction of "down" along axis k
                # cause: "load" = what rests on it is enough by itself (>= 60 % of f_crit, i.e. more than one item);
                # "press" = the robot's own push made the difference (a hard press with the gripper)
                cause = "load" if law["f_rest"] >= 0.6 * law["f_crit"] else "press"
                law["buckles"].append(dict(t=round(law["t"], 3), load_N=round(law["f"], 2), how=law["how"],
                                           resting_N=round(law["f_rest"], 2), cause=cause))
                self._on_buckle()
        elif law["f"] < F_REL:
            law["rel_t"] += dt
            if law["rel_t"] > REL_T:
                law["buckled"] = False
                # straighten smoothly: the rest shape starts at the current bend and moves back to straight over
                # K_RISE (a stiff spring released at once from a 40-50 deg bend flips the light carton over)
                q0 = np.array(self.env.sim.data.qpos[qa:qa + 4], float)
                q0 = q0 / max(np.linalg.norm(q0), 1e-9) * (1.0 if q0[0] >= 0 else -1.0)
                law["unbend_q0"], law["unbend_s"] = q0.tolist(), 0.0
                m.qpos_spring[qa:qa + 4] = q0
        else:
            law["rel_t"] = 0.0
        k = float(m.jnt_stiffness[j])
        if law["buckled"] and law["how"] == "fold":
            m.jnt_stiffness[j] = k + (K_LO - k) * min(1.0, dt / 0.02)
        else:
            m.jnt_stiffness[j] = K_HI
            if law.get("unbend_q0") is not None:
                law["unbend_s"] = min(1.0, law["unbend_s"] + dt / K_RISE)
                q = (1.0 - law["unbend_s"]) * np.asarray(law["unbend_q0"]) + law["unbend_s"] * np.array([1.0, 0, 0, 0])
                m.qpos_spring[qa:qa + 4] = q / np.linalg.norm(q)
                if law["unbend_s"] >= 1.0:
                    law["unbend_q0"] = None
        target = 1.0 if (law["buckled"] and law["how"] == "crush") else 0.0
        if target or law["crush"] > 0.0:
            tau = CRUSH_TAU[0] if target else CRUSH_TAU[1]
            law["crush"] += (target - law["crush"]) * min(1.0, dt / tau)
            if not target and law["crush"] < 0.002:
                law["crush"] = 0.0
            if abs(law["crush"] - law["crush_set"]) > 0.004 or (law["crush"] == 0.0 and law["crush_set"] > 0.0):
                self._set_crush(b, law["crush"], law["crush_axis"], law["crush_sign"])
                law["crush_set"] = law["crush"]
        law["t"] += dt

    def _set_crush(self, b, c, axis, sign):
        """flatten box b's two halves along their local `axis` by c (0..1) x CRUSH_ALPHA, keeping the faces on the
        `sign` side (the side facing down) in place; then refresh the collision bounds"""
        m = self.env.sim.model
        for (bid, gs, size0, pos0), alpha in zip(self.half_geoms[b], CRUSH_ALPHA):
            size, pos = size0.copy(), pos0.copy()
            if c > 0.0:
                s0 = size0[:, axis]
                s1 = s0 * (1.0 - alpha * c)
                size[:, axis] = s1
                pos[:, axis] = pos0[:, axis] + sign * (s0 - s1)
            m.geom_size[gs] = size
            m.geom_pos[gs] = pos
        envs.refit_bounds(m, [self.half_geoms[b][0][0], self.half_geoms[b][1][0]])

    def _on_buckle(self):
        mon = self.mon
        if mon is None:
            return
        b = self.law["box"]
        ch = self.pile()
        on = self.items_on(b)
        last = self.law["buckles"][-1] if self.law["buckles"] else {}
        e = dict(event="buckle", box=COL[b], step=int(self.sk.n_steps) if self.sk else 0,
                 load_N=round(self.law["f"], 2), level=(ch.index(b) + 1) if b in ch else None,
                 items_on=[COL.get(x, "cup") for x in on], how=last.get("how"), cause=last.get("cause"),
                 resting_N=last.get("resting_N"),
                 lying=bool(abs(self.R(self.base_bid[b])[2, 2]) < FOLD_MIN_UP))
        mon["events"].append(e)
        mon["buckle_steps"].append(e["step"])
        mon["pending_collapse"] = dict(step=e["step"], pile=list(ch), cup=self.cup_on_pile(ch))

    # ------------------------------------------------------------ geometry / predicates
    def R(self, bid):
        return self.env.sim.data.xmat[bid].reshape(3, 3)

    def centre(self, name):
        return envs.obj_pos(self.env, name)

    def flat_tilt(self, b):
        """angle (rad) between world vertical and the base part's axis closest to it (a box may lie on any face)"""
        R = self.R(self.base_bid[b])
        return float(np.arccos(np.clip(np.max(np.abs(R[2, :])), -1.0, 1.0)))

    def kink(self, b):
        """bend of the upper part against the base part (rad)"""
        zb = self.R(self.base_bid[b])[:, 2]
        zu = self.R(self.upper_bid[b])[:, 2]
        return float(np.arccos(np.clip(zb @ zu, -1.0, 1.0)))

    def cup_tilt(self):
        return float(np.arccos(np.clip(envs.obj_upright_cos(self.env, CUP), -1.0, 1.0)))

    def held(self, name):
        return self.sk is not None and self.sk.holding(name)

    def crushed(self, b):
        """how far box b is crushed (0..1; only the soft carton, lying on a side face, can be)"""
        law = self.law
        return float(law["crush"]) if (law is not None and law["box"] == b) else 0.0

    def flat(self, b):
        return (self.flat_tilt(b) <= TILT_OK and self.kink(b) <= KINK_OK and self.crushed(b) <= 0.15
                and not self.held(b))

    def rests_on(self, upper, lower):
        cu, cl = self.centre(upper), self.centre(lower)
        return (np.linalg.norm(cu[:2] - cl[:2]) <= STACK_XY
                and abs(envs.obj_min_z(self.env, upper) - envs.obj_max_z(self.env, lower)) <= STACK_Z)

    def on_mat(self, b):
        off = self.centre(b)[:2] - np.asarray(self.inst["mat"])
        return bool(np.all(np.abs(off) <= ON_MAT) and envs.obj_min_z(self.env, b) <= TZ + 0.008)

    def pile(self):
        """the longest chain of flat boxes: the bottom one on the mat, each next resting on the previous (centres within
        2 cm, faces within 1 cm)"""
        flat = [b for b in BOXES if self.flat(b)]
        best = []
        for b0 in [b for b in flat if self.on_mat(b)]:
            ch = [b0]
            while True:
                nxt = [b for b in flat if b not in ch and self.rests_on(b, ch[-1])]
                if not nxt:
                    break
                ch.append(min(nxt, key=lambda b: float(np.linalg.norm(self.centre(b)[:2] - self.centre(ch[-1])[:2]))))
            if len(ch) > len(best):
                best = ch
        return best

    def cup_on(self, b):
        c = self.centre(CUP)
        return bool(self.cup_tilt() <= TILT_OK and not self.held(CUP)
                    and np.linalg.norm(c[:2] - self.centre(b)[:2]) <= CUP_XY
                    and abs(envs.obj_min_z(self.env, CUP) - envs.obj_max_z(self.env, b)) <= STACK_Z)

    def cup_on_pile(self, ch=None):
        ch = self.pile() if ch is None else ch
        return bool(len(ch) == len(BOXES) and self.cup_on(ch[-1]))

    def items_on(self, b):
        """boxes / cup resting (directly or higher up) on box b, by height and overlap"""
        cb = self.centre(b)
        top = envs.obj_max_z(self.env, b)
        out = []
        for x in BOXES + [CUP]:
            if x == b:
                continue
            c = self.centre(x)
            if envs.obj_min_z(self.env, x) > top - 0.015 and np.linalg.norm(c[:2] - cb[:2]) < 0.06:
                out.append(x)
        return out

    def stage_values(self):
        ch = self.pile()
        v = {COL[b]: b in ch for b in BOXES}
        v["cup"] = self.cup_on_pile(ch)
        return v, ch

    # ------------------------------------------------------------ monitor
    def _init_monitor(self):
        keys = COLOURS + ["cup"]
        self.mon = dict(state={k: False for k in keys}, cand={k: 0 for k in keys}, ever={k: False for k in keys},
                        lost={k: False for k in keys}, lost_cause={k: None for k in keys}, events=[], n=0,
                        last_robot={k: -10 ** 6 for k in keys}, buckle_steps=[], pending_collapse=None,
                        max_kink={COL[b]: 0.0 for b in BOXES}, max_crush={COL[b]: 0.0 for b in BOXES},
                        max_height=0, n_collapses=0,
                        law=self.law)  # the carton's law state rides along in the robot server's regression snapshots

    def _substep(self):
        d = self.env.sim.data
        n = d.ncon
        if not n:
            return
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        o1, o2 = self.owner[g1], self.owner[g2]
        p1, p2 = self.part[g1], self.part[g2]
        sel = ((o1 >= 0) & (p2 > 0)) | ((o2 >= 0) & (p1 > 0))
        for i in np.nonzero(sel)[0]:
            o = o1[i] if o1[i] >= 0 else o2[i]
            self._sub_rob[int(o)] = True

    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        mon["n"] += 1
        t = int(self.sk.n_steps)
        self._held = {i for i, x in enumerate(BOXES + [CUP]) if self.held(x)}
        if self.law is not None:  # weight of the things (not held) on the soft carton, + 0.3 N for impacts
            m = self.env.sim.model
            self._rest_cap = 0.3 + G * sum(float(np.sum(m.body_mass[self.obj_bodies[x]]))
                                           for x in self.items_on(self.law["box"])
                                           if (BOXES + [CUP]).index(x) not in self._held)
        keys = COLOURS + ["cup"]
        for i in list(self._sub_rob):
            mon["last_robot"][keys[i]] = t
        self._sub_rob = {}
        for b in BOXES:
            mon["max_kink"][COL[b]] = max(mon["max_kink"][COL[b]], round(float(np.degrees(self.kink(b))), 1))
            mon["max_crush"][COL[b]] = max(mon["max_crush"][COL[b]], round(self.crushed(b), 2))
        if mon["n"] % 2:
            return
        vals, ch = self.stage_values()
        mon["max_height"] = max(mon["max_height"], len(ch))
        for k in keys:
            raw = vals[k]
            if raw == mon["state"][k]:
                mon["cand"][k] = 0
                continue
            mon["cand"][k] += 2
            if mon["cand"][k] < DEBOUNCE:
                continue
            mon["cand"][k] = 0
            mon["state"][k] = raw
            t0 = t - DEBOUNCE + 1
            if raw:
                restored = mon["lost"][k]
                e = dict(stage=k, event="restored" if restored else "achieved", step=t0)
                if restored:
                    e["after"] = mon["lost_cause"][k]
                    mon["lost"][k] = False
                mon["ever"][k] = True
                mon["events"].append(e)
            elif mon["ever"][k]:
                cause = self._cause(k, t)
                mon["lost"][k] = True
                mon["lost_cause"][k] = cause
                mon["events"].append(dict(stage=k, event="lost", step=t0, cause=cause))
                pc = mon["pending_collapse"]
                if cause == "buckle" and pc is not None and not pc.get("counted"):
                    pc["counted"] = True
                    mon["n_collapses"] += 1
                    mon["events"].append(dict(event="collapse", step=t0, buckled=self.inst["soft_colour"],
                                              pile_before=[COL[b] for b in pc["pile"]]))

    def _cause(self, k, t):
        name = CUP if k == "cup" else BOXES[COLOURS.index(k)]
        if self.held(name):
            return "picked_up"
        bs = self.mon["buckle_steps"]
        if bs and t - bs[-1] <= 80:  # within 4 s of a buckle
            return "buckle"
        if t - self.mon["last_robot"][k] <= 30:
            return "robot"
        return "other"

    def _finalize_monitor(self, vals):
        mon = self.mon
        t = int(self.sk.n_steps)
        for k, raw in vals.items():
            if raw == mon["state"][k]:
                continue
            mon["state"][k] = raw
            if raw:
                restored = mon["lost"][k]
                e = dict(stage=k, event="restored" if restored else "achieved", step=t, final=True)
                if restored:
                    e["after"] = mon["lost_cause"][k]
                    mon["lost"][k] = False
                mon["ever"][k] = True
                mon["events"].append(e)
            elif mon["ever"][k]:
                cause = self._cause(k, t)
                mon["lost"][k] = True
                mon["lost_cause"][k] = cause
                mon["events"].append(dict(stage=k, event="lost", step=t, cause=cause, final=True))

    # ------------------------------------------------------------ outcome
    def _touching(self):
        d = self.env.sim.data
        n = d.ncon
        if not n:
            return False
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        return bool(np.any(((self.owner[g1] >= 0) & (self.part[g2] > 0)) | ((self.owner[g2] >= 0) & (self.part[g1] > 0))))

    def _static_pos(self):
        return {x: self.centre(x).copy() for x in BOXES + [CUP]}

    def outcome(self, inst):
        sk = self.sk
        sk.hold(10)
        p0 = self._static_pos()
        touching = False
        for _ in range(20):
            sk.hold(1)
            touching = touching or self._touching()
        moved = max(float(np.linalg.norm(self.centre(x) - p0[x])) for x in p0)
        static = moved < 0.003
        vals, ch = self.stage_values()
        self._finalize_monitor(vals)
        mon = self.mon
        ev = list(mon["events"])
        held = [COL.get(x, "cup") for x in BOXES + [CUP] if self.held(x)]
        success = bool(all(vals.values()) and static and not held and not touching)
        buckles = [e for e in ev if e.get("event") == "buckle"]
        n_reg = sum(e.get("event") == "lost" and e.get("cause") != "picked_up" for e in ev)
        n_res = sum(e.get("event") == "restored" and e.get("after") != "picked_up" for e in ev)
        soft = inst["soft"]
        buckled_now = bool(self.law is not None and self.law["buckled"])
        cup_fallen = self.cup_tilt() > np.radians(45)
        if success:
            failure = "none"
        elif held:
            failure = "held"
        elif touching and all(vals.values()):
            failure = "touching"
        elif not all(vals[c] for c in COLOURS):
            if buckles and any(mon["ever"][c] for c in COLOURS):
                failure = "collapsed"
            elif any(mon["lost"][c] for c in COLOURS):
                failure = "regressed"
            else:
                failure = "not_stacked"
        elif not vals["cup"]:
            failure = "cup_fallen" if cup_fallen else "cup_not_on_top"
        else:
            failure = "not_static"
        # F2 detail (robot-centric words)
        parts = []
        if success:
            parts.append("All four boxes are stacked on the mat and the cup stands on top.")
        else:
            if ch:
                parts.append("The pile on the mat is " + " / ".join(COL[b] for b in ch) + " (bottom to top).")
            else:
                parts.append("There is no box resting flat on the mat.")
            mat = np.asarray(inst["mat"])
            for b in BOXES:
                if b in ch:
                    continue
                c = COL[b]
                if self.held(b):
                    parts.append(f"The {c} box is still held by the gripper.")
                    continue
                off = self.centre(b)[:2] - mat
                pose = ("lies tilted" if self.flat_tilt(b) > TILT_OK else
                        "is crushed flat" if self.crushed(b) > 0.15 else
                        f"is bent ({np.degrees(self.kink(b)):.0f} degrees)" if self.kink(b) > KINK_OK else "stands")
                parts.append(f"The {c} box {pose} {_fb(off[0])} and {_lr(off[1])} of the mat's centre, not in the pile.")
            if not vals["cup"]:
                if self.held(CUP):
                    parts.append("The cup is still held by the gripper.")
                elif cup_fallen:
                    parts.append("The cup has fallen over.")
                else:
                    parts.append("The cup is not standing on top of a complete pile.")
            if all(vals.values()):
                if touching:
                    parts.append("The gripper is still touching the pile.")
                elif not static:
                    parts.append("Something was still moving.")
        loads = [e for e in buckles if e.get("cause") != "press"]
        presses = [e for e in buckles if e.get("cause") == "press"]
        if loads:
            how = ("bent sideways or was crushed" if len({e.get("how") for e in loads}) > 1 else
                   "was crushed flat" if loads[0].get("how") == "crush" else "bent sideways")
            parts.append(f"During the attempt the {loads[0]['box']} box {how} under the weight on it "
                         f"{len(loads)} time(s) and everything above it slid off.")
        if presses:
            parts.append(f"The {presses[0]['box']} box gave way {len(presses)} time(s) when the gripper pushed down "
                         f"on it.")
        # trap fired = the carton gave way under what was put on it (or a giving-way cost a stage); a hard press by the
        # gripper alone on an otherwise lightly loaded carton is logged (n_press_buckles) but is not the trap
        trap = bool(loads) or int(mon["n_collapses"]) > 0
        return dict(success=success, failure=failure, stages=vals, pile=[COL[b] for b in ch], events=ev,
                    n_regressions=int(n_reg), n_restored=int(n_res), trap_fired=trap,
                    recovered_in_attempt=bool(trap and success), n_buckles=len(buckles),
                    n_collapses=int(mon["n_collapses"]), buckle_levels=[e.get("level") for e in buckles],
                    buckle_loads_N=[e.get("load_N") for e in buckles], max_kink_deg=dict(mon["max_kink"]),
                    max_crush=dict(mon["max_crush"]), buckle_how=[e.get("how") for e in buckles],
                    buckle_causes=[e.get("cause") for e in buckles], n_press_buckles=len(presses),
                    max_height=int(mon["max_height"]), soft=inst["soft_colour"], benign=bool(inst["benign"]),
                    buckled_at_end=buckled_now, cup_fallen=bool(cup_fallen), held=held, touching=bool(touching),
                    static=bool(static), detail=" ".join(parts))

    # ------------------------------------------------------------ scripted primitives (move_to + gripper only)
    def _go(self, xyz, yaw=None, speed=None, tol=0.004, max_steps=300):
        return self.sk.move_to(xyz, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)

    def _lift_to(self, z, speed=0.2):
        cur = self.sk.eef_pos()
        if cur[2] < z - 0.005:
            self._go([cur[0], cur[1], z], speed=speed)

    def _settle(self, steps=40):
        """hold until boxes / cup have stopped (max `steps` control steps)"""
        p = self._static_pos()
        for _ in range(steps // 4):
            self.sk.hold(4)
            q = self._static_pos()
            if max(float(np.linalg.norm(q[x] - p[x])) for x in p) < 0.0005:
                return
            p = q

    def settle_scene(self, max_steps=120):
        """after a collapse: wait (looking) until nothing moves and no box is visibly bent any more"""
        for _ in range(max_steps // 10):
            self._settle(10)
            if all(self.kink(b) < np.radians(3.0) for b in BOXES):
                p = self._static_pos()
                self.sk.hold(4)
                q = self._static_pos()
                if max(float(np.linalg.norm(q[x] - p[x])) for x in p) < 0.0005:
                    return

    def grasp_yaw(self, name):
        """wrist yaw that closes the fingers across a horizontal box axis: of the two, the one whose open fingers stay
        clearest of the other objects (a box that fell against another one), then the one needing the least turn"""
        if name == CUP:
            return 0.0
        R = self.R(self.base_bid[name])
        up = int(np.argmax(np.abs(R[2, :])))
        c = self.centre(name)[:2]
        others = [self.centre(x)[:2] for x in BOXES + [CUP] if x != name]
        best = None
        for a in range(3):
            if a == up:
                continue
            v = R[:2, a] / max(np.linalg.norm(R[:2, a]), 1e-9)
            yaw = _wrap_q(float(np.arctan2(v[1], v[0])) - np.pi / 2, np.pi)  # fingers close along (-sin yaw, cos yaw)
            fingers = [c + s_ * 0.045 * v for s_ in (1.0, -1.0)]  # open fingers ~4.5 cm from the centre
            clear = min([float(np.linalg.norm(f - o)) for f in fingers for o in others] + [1.0])
            key = (round(min(clear, 0.05), 3), -abs(yaw))
            if best is None or key > best[0]:
                best = (key, yaw)
        return best[1]

    def z_clear(self, extra=0.08):
        """carry height: above everything on the table"""
        top = max(envs.obj_max_z(self.env, x) for x in BOXES + [CUP])
        return max(TZ + Z_CARRY, top + extra)

    def pick(self, name, depth=0.018, z_carry=None, tries=2):
        sk = self.sk
        for _ in range(tries):
            self._settle(20)
            p = self.centre(name)
            zt = envs.obj_max_z(self.env, name)
            yaw = self.grasp_yaw(name)
            sk.set_gripper(False, steps=10)
            zc = z_carry or self.z_clear()
            self._lift_to(zc)
            self._go([p[0], p[1], zc], yaw=yaw, speed=0.25)
            self._go([p[0], p[1], zt - depth + 0.02], speed=0.12)
            self._go([p[0], p[1], max(zt - depth, TZ + 0.012)], speed=0.04, tol=0.003, max_steps=150)
            sk.set_gripper(True, steps=15)
            self._go([p[0], p[1], self.z_clear(0.10)], speed=0.15)
            if self.held(name):
                return True
            sk.set_gripper(False, steps=10)
        return False

    def put(self, name, xy, z_surface, gap, yaw=0.0):
        """carry the held object over xy (its centre), straight down until its bottom is `gap` above z_surface, let
        go, lift 8 cm"""
        sk = self.sk
        zc = max(self.z_clear(0.10), sk.eef_pos()[2])
        rel = self.centre(name)[:2] - sk.eef_pos()[:2]
        q = np.asarray(xy, float) - rel
        self._go([q[0], q[1], zc], yaw=yaw, speed=0.25)
        sk.hold(5)
        rel = self.centre(name)[:2] - sk.eef_pos()[:2]  # the object can shift in the fingers while the wrist turns
        q = np.asarray(xy, float) - rel
        dz = sk.eef_pos()[2] - envs.obj_min_z(self.env, name)
        zr = z_surface + dz + gap
        self._go([q[0], q[1], zr + 0.03], speed=0.12)
        self._go([q[0], q[1], zr], speed=0.04, tol=0.003, max_steps=150)
        sk.set_gripper(False, steps=15)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], cur[2] + 0.08], speed=0.15)
        self._settle(40)

    def free_spot(self, near_xy):
        """a free place on the table (for a box taken off the pile): >= 11 cm from everything, off the mat, reachable"""
        mat = np.asarray(self.inst["mat"])
        others = [self.centre(x)[:2] for x in BOXES + [CUP]]
        best, bd = None, -1.0
        for r in (0.17, 0.21, 0.25, 0.29):
            for a in np.radians(np.arange(0, 360, 15)):
                p = mat + r * np.array([np.cos(a), np.sin(a)])
                if not (BOX_X[0] - 0.02 <= p[0] <= BOX_X[1] and abs(p[1]) <= BOX_YMAX + 0.02):
                    continue
                d = min(float(np.linalg.norm(p - o)) for o in others)
                if d >= 0.11:
                    return p
                if d > bd:
                    best, bd = p, d
        return best

    def place_on_pile(self, name, err, gap, rng, bias=0.0):
        """put object `name` on the current pile top (or the mat) with a placement error of `err` (m, random direction)"""
        ch = [b for b in self.pile() if b != name]
        if ch:
            # aims every box at the same spot (the bottom box), as an agent that computes the pile's x, y once: errors
            # do not add up level by level
            tgt = self.centre(ch[0])[:2].copy()
            zs = envs.obj_max_z(self.env, ch[-1])
        else:
            tgt = np.asarray(self.inst["mat"], float) + np.array([bias, 0.0])
            zs = TZ
        if err:
            a = rng.uniform(0, 2 * np.pi)
            tgt = tgt + float(err) * np.array([np.cos(a), np.sin(a)])
        if not self.pick(name):
            return False
        self.put(name, tgt, zs, gap)
        return True

    def lay_aside(self, name, rng):
        if not self.pick(name):
            return False
        p = self.free_spot(self.centre(name)[:2])
        self.put(name, p, TZ, 0.01)
        return True

    # ------------------------------------------------------------ scripted policies
    def nearest_order(self):
        mat = np.asarray(self.inst["mat"])
        return sorted(BOXES, key=lambda b: float(np.linalg.norm(self.centre(b)[:2] - mat)))

    def plan(self, params, rng):
        if params.get("order") == "random":
            order = [BOXES[i] for i in rng.permutation(4)]
        else:
            order = self.nearest_order()
        soft = params.get("soft_known")
        if soft:
            order = [b for b in order if COL[b] != soft] + [b for b in order if COL[b] == soft]
        return order

    def seen_soft(self):
        """the box that visibly bent during this attempt (scripted perception of the kink in the images)"""
        mk, mc = self.mon["max_kink"], self.mon["max_crush"]
        for c in COLOURS:
            if mk[c] > 15.0 or mc[c] > 0.3:
                return c
        return None

    def build(self, order, params, rng, first=True):
        """stack `order` (boxes not yet in the pile) then the cup; returns 'ok', 'collapse' or 'fail'"""
        err = params.get("err")
        gap = params.get("gap", 0.005)
        # a box that could not be picked up is tried again after the others (one more pass)
        order = list(order) * 2
        for b in order:
            if b in self.pile():
                continue
            if params.get("soft_known") and COL[b] == params["soft_known"] and \
                    sum(x in self.pile() for x in BOXES if COL[x] != params["soft_known"]) < len(BOXES) - 1:
                continue  # the soft box only goes on once the three others are in the pile
            for _ in range(2):
                n_b = len(self.mon["buckle_steps"])
                before = self.pile()
                e = rng.uniform(*err) if err else 0.0
                if not self.place_on_pile(b, e, gap, rng, bias=params.get("bias", 0.0) if not before else 0.0):
                    break
                after = self.pile()
                if len(self.mon["buckle_steps"]) > n_b or any(x not in after for x in before):
                    return "collapse"
                if b in after:
                    break
        if len(self.pile()) < len(BOXES):
            return "fail"
        if self.cup_tilt() > np.radians(45):
            return "fail"
        for _ in range(2):
            n_b = len(self.mon["buckle_steps"])
            before = self.pile()
            e = rng.uniform(*err) if err else 0.0
            if not self.place_on_pile(CUP, e, gap, rng):
                return "fail"
            if len(self.mon["buckle_steps"]) > n_b or any(x not in self.pile() for x in before):
                return "collapse"
            if self.cup_on_pile():
                return "ok"
        return "fail"

    def rebuild_soft_top(self, soft_colour, params, rng):
        """take the soft box (and whatever is on it) off the pile, then stack the rigid boxes and the soft one last"""
        soft = BOXES[COLOURS.index(soft_colour)]
        for _ in range(6):
            ch = self.pile()
            if soft in ch:
                self.lay_aside(ch[-1], rng)  # from the top down to the soft box
            elif self.on_mat(soft) or any(soft in self.items_on(b) for b in ch):
                self.lay_aside(soft, rng)  # bent / off-centre on the mat or on the pile
            else:
                break
        order = [b for b in self.nearest_order() if b != soft] + [soft]
        return self.build(order, dict(params, soft_known=soft_colour), rng)

    def execute(self, inst, params):
        rng = np.random.default_rng(params.get("noise_seed", 0))
        order = self.plan(params, rng)
        res = self.build(order, params, rng)
        if res == "collapse":
            mode = params.get("on_collapse", "rebuild_same")
            if mode == "rebuild_same":  # the cheapest explanation: "I did not centre it" -> same order, more care
                self.settle_scene()
                res = self.build(order, dict(params, err=None, gap=0.005), rng)
            elif mode == "identify":  # sees which box bent and rebuilds with it on top
                self.settle_scene()
                soft = self.seen_soft()
                if soft and self.cup_tilt() < np.radians(45):
                    res = self.rebuild_soft_top(soft, params, rng)
        self._lift_to(self.z_clear(0.06))
        self.sk.hold(10)
        return res

    NAIVE = dict(order="nearest", soft_known=None, on_collapse="rebuild_same")

    def default_params(self, inst):
        """Sonnet-like: nearest box first, lifts above everything, straight down, lets go 1-4 cm above (per instance),
        0.4-1.3 cm placement errors, the first box up to 2 cm too far forward; after a collapse it rebuilds once in
        the same order, more carefully"""
        return dict(self.NAIVE, noise_seed=int(inst["seed"]) * 13, err=NAIVE_ERR, gap=float(inst["naive_gap"]),
                    bias=float(inst["naive_bias"]))

    def oracle_params(self, inst):
        return dict(order="nearest", soft_known=inst["soft_colour"], on_collapse="identify", err=None, gap=0.005)

    def adapt_params(self, inst, history):
        soft = None
        for h in history:
            ev = [e for e in h["outcome"]["events"] if e.get("event") == "buckle"]
            if ev:
                soft = ev[0]["box"]  # the box seen bending (the kink is visible in the images)
        return dict(self.default_params(inst), soft_known=soft, on_collapse="identify")

    def blind_params(self, inst, attempt, rng):
        return dict(self.default_params(inst), order="random", noise_seed=int(rng.integers(1 << 30)),
                    gap=float(rng.uniform(*NAIVE_GAP)), bias=float(rng.uniform(*NAIVE_BIAS)))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        single = ("oracle", "naive", "naive_centred", "oracle_noise", "naive_once", "naive_lying", "oracle_lying")
        if kind in ("naive_lying", "oracle_lying") and inst["soft"]:
            # control (v2): the soft carton starts lying on a side face, so it is stacked lying
            inst = dict(inst, lying={inst["soft_colour"]: np.pi / 2})
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":  # attempt 1: naive habits, but it looks at the collapse and rebuilds soft-on-top
                params = (dict(self.default_params(inst), on_collapse="identify") if a == 0
                          else self.adapt_params(inst, history))
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "switch":  # control: cross-attempt learning only (no within-attempt identification)
                params = self.default_params(inst) if a == 0 else dict(self.adapt_params(inst, history),
                                                                     on_collapse="rebuild_same")
            elif kind == "naive_centred":  # control (R4): perfect centring, let go 5 mm above, no rebuild
                params = dict(self.default_params(inst), err=None, gap=0.005, bias=0.0, on_collapse="none")
            elif kind == "naive_once":  # control: naive without the careful rebuild
                params = dict(self.default_params(inst), on_collapse="none")
            elif kind == "naive_lying":  # control (v2): naive_once with the soft carton lying on a side face
                params = dict(self.default_params(inst), on_collapse="none")
            elif kind == "oracle_lying":  # control (v2): soft carton lying, but on top (carries only the cup)
                params = dict(self.oracle_params(inst))
            elif kind == "oracle_noise":  # robustness: soft box on top with Sonnet-like imprecision
                params = dict(self.default_params(inst), soft_known=inst["soft_colour"], on_collapse="identify")
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
