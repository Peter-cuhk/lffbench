"""LH drawer_cabinet (2026-10-16): tidy the table -- stand two tall bottles on top of a small wooden cabinet, put two
cubes into its open drawer and close the drawer. In 14 of every 20 instances the drawer is a self-closing one: pushed
into the last 8 cm, a strong spring snaps it shut, the light cabinet lurches 2.3-4.2 cm towards the robot, and bottles
already standing on its top fall over (away from the robot, onto the gallery rail at the back of the top).

Family: long-horizon self-disturbance ("state_restoration"): a later, normal action (closing the drawer) physically
undoes an earlier stage (bottles on the cabinet). The instruction lists the bottles first but fixes no order. The agent
has to see the fallen bottles and stand them up again after the drawer is shut (within an attempt), or learn to put
the cubes in and shut the drawer before it puts the bottles on top (across attempts).

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * a small wooden cabinet (free body, 18 (x) x 22 (y) x 11 cm, centre 53-56 cm in front of the robot base, front facing
    the robot) with a 6 cm gallery rail along the back edge of its top, a 4 cm toe plinth under the drawer and a
    back-heavy carcass; its single drawer (inside 15.9 x 16.2 cm, 3 cm walls, 1.8 cm overlay front panel with a painted
    finger slot, no handle, 0.8 kg) is pulled out 11 cm towards the robot. The agentview camera looks at the cabinet's
    back: the drawer is hidden there (wrist camera only), the top and the bottles are in plain view;
  * two tall square bottles (green with a yellow ball cap, brown with a blue ball cap; 4 x 4 cm body, 19.4 cm tall,
    0.18 kg, centre of mass 9 cm up, a rounded glass base whose flat contact is 2.2 x 2.2 cm: they tip beyond ~7 deg)
    standing on the table 21-24 cm to one side of the cabinet's centre line;
  * two 3.2 cm cubes (red, blue, 23 g) 25-29 cm to the other side (picked from nearer, the forearm sweeps over the
    cabinet top and knocks bottles standing there: measured feasibility map, task card).

Hidden variable (same appearance): the drawer's closing mechanism. "self-closing" (14 of every 20 consecutive seeds):
in the last 8 cm of travel a spring pulls the drawer in with 12.5-15 N (no damper); "soft-close" (6 of 20): the same
spring with a damper, the drawer glides in at 4-6 cm/s. Nuisance variables (no effect on the class): carcass mass
0.35-0.45 kg, felt feet (friction on the table 0.15-0.22).

Mechanism (physics, no teleport; measured, see the task card): the spring is internal to the cabinet, so when it snaps
the drawer in, it also pulls the light carcass (with the bottles on its top) towards the drawer, i.e. towards the robot:
2.3-4.2 cm within ~0.07 s (drawer hitting its stop at 1.9-2.4 m/s relative to the carcass; validate's naive runs),
then the carcass stops dead. A standing bottle's tilt follows a fast displacement of its support as ~ m h dx / I_edge
(~7 rad per metre here), so the bottles tip away from the robot and come to rest leaning on the rail (45-75 deg). The
robot's hand is at the drawer front, below and 10+ cm away from the bottles, and does not touch them (0 of all slams).
Pushing slower does not help (the spring does the closing; 2-10 cm/s and push heights 3-8 cm all toppled, 32/32);
placing the bottles elsewhere on the top does not help (46/46 over six placements). A soft-close drawer moves the
carcass < 3 mm (0/38 bottles fell). Only an uncontrolled full-speed push (~0.6 m/s, not an agent habit) differs: the
hand then shoves the cabinet itself (self-closing 5/8 fell, soft-close 4/4 fell, onto the arm). Placing the bottles after the drawer is shut is safe. Prototype findings that fixed
the numbers: with a light drawer, a heavier carcass or more friction the bottles only rocked (<= 11 deg): the snap's
jerk and the friction stop cancel; a 2 kg carcass held by friction during the snap instead jerked away from the robot
and tipped the bottles towards (onto) the arm. Without the rail, fallen bottles lay with their caps right behind the
cabinet, where the palm (3 cm behind the grasp point) cannot get at them; with the heavy drawer pulled out and no
plinth, ~3 N pressing on the drawer front tipped the cabinet forward.

Restoring: the cap is a ball with low torsional friction; grasped from above with the fingers across the lying
bottle's axis, the cap turns between the fingers, the lifted bottle swings down and hangs straight, and it can be set
down on the cabinet top (lh_bystander's mechanism; the instruction says that a bottle held by its cap hangs straight).
The neck has torsional friction: a bottle pinched at the neck just under the cap is carried without swinging.

Success (final state, follows from the instruction): both bottles upright (<= 10 deg) with their whole base on the
cabinet top (inside the rail), not held; both cubes inside the drawer; the drawer closed (<= 5 mm); the cabinet
upright and not moved more than 8 cm (the snap itself moves it 2.3-4.2 cm); the gripper open, holding nothing and
touching no task object; scene static. Per-step monitor (server log only; F1 tells the agent success / failure): stage
achieved / lost / restored (with cause: drawer_slam / robot_hand / robot_arm / bottle / picked_up / other),
drawer_slam (impact speed, carcass jerk, bottles on top, which of them fell, whether the robot had touched them before
they started to fall), bottle_fell, trap_fired (a bottle on top was lost to a slam), recovered_in_attempt.

Protocol: cross (reset to the same instance before every attempt), k = 3. Once shut, the drawer cannot be opened again
(no handle; the self-closing spring would drag the light carcass along anyway).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

Z = envs.TABLE_Z

# ---------------------------------------------------------------- cabinet (body frame: front face at -x, origin at the
# centre of the footprint, z = 0 on the table)
CAB = "cabinet_1"
CAB_MODEL = "lff_lhd_cabinet"
CD, CW, CH = 0.18, 0.22, 0.11  # carcass depth (x), width (y), height
CT = 0.010  # board thickness
PF = 0.018  # drawer front panel thickness (overlay: it sits in front of the carcass when closed)
DL = 0.165  # drawer box length (x), measured from the inner face of the front panel
DWT = 0.006  # drawer wall thickness
DWH = 0.030  # drawer wall height above its floor
D_FLOOR = CT + 0.003  # drawer floor bottom above the table
FLOOR_TOP = D_FLOOR + DWT
D_IN_Y = CW / 2 - CT - 0.003 - DWT  # drawer inner half width (0.081)
D_IN_X = DL - DWT  # inner length (from the front panel to the back wall)
Q_MAX = 0.14  # slide range (opening distance)
Q_OPEN = 0.11  # the drawer starts pulled out this far (exposed inside: 11 x 16.2 cm)
WOOD = (0.66, 0.50, 0.33, 1.0)
WOOD_D = (0.55, 0.40, 0.25, 1.0)
PANEL = (0.76, 0.60, 0.40, 1.0)
INNER = (0.84, 0.72, 0.52, 1.0)
SLOT = (0.18, 0.13, 0.09, 1.0)
DRAWER_MASS = 0.80
PLINTH = 0.04  # the bottom board reaches this far in front of the carcass, under the drawer: with the heavy drawer
#                pulled out, the cabinet would otherwise tip forward under a ~3 N push down on the drawer's front
CAB_COM_X = 0.04  # carcass centre of mass towards the back (back-heavy carcass): same reason
RAIL_H, RAIL_T = 0.060, 0.008  # gallery rail along the back edge of the top: a bottle that tips over backwards comes
#                                to rest leaning on it (~50 deg) instead of falling off behind the cabinet, where its
#                                cap would lie against the cabinet's back and the palm could not get at it
SLIDE_FRIC = 1.0  # N, slide frictionloss: the open drawer stays where it is
CARCASS_MASS = (0.35, 0.45)
FEET_MU = (0.15, 0.22)

# drawer closing mechanism (hidden): spring in the last ZONE of travel, optional damper (soft-close)
ZONE = 0.08
F_SPRING = (12.5, 15.0)
SOFT_V = (0.04, 0.06)  # soft-close glide speed (damping = force / speed)
P_SLAM = 0.70  # 14 of every 20 consecutive seeds self-closing
CLOSED = 0.005  # drawer counts as closed
V_SLAM = 0.30  # m/s: an impact on the stop faster than this is a slam (soft-close: ~0.05)

# ---------------------------------------------------------------- bottles (square body, shoulder, neck, ball cap)
BOTTLES = ("green_bottle_1", "brown_bottle_1")
B_NAME = {"green_bottle_1": "green", "brown_bottle_1": "brown"}
B_RGBA = {"green_bottle_1": (0.16, 0.48, 0.26, 1.0), "brown_bottle_1": (0.50, 0.26, 0.10, 1.0)}
CAP_RGBA = {"green_bottle_1": (0.92, 0.80, 0.15, 1.0), "brown_bottle_1": (0.15, 0.32, 0.85, 1.0)}
RB, HB = 0.020, 0.115  # body half width, height (incl. the foot)
RF, HF = 0.011, 0.005  # rounded glass base: flat contact 2.2 x 2.2 cm
RS, HS = 0.015, 0.010  # shoulder
RN, HN = 0.009, 0.050  # neck: long enough to be pinched below the cap with the palm clear of the cap
RK = 0.012  # ball cap
CAP_SINK = 0.005
H_TOT = HB + HS + HN + 2 * RK - CAP_SINK  # 0.194
CAP_Z = H_TOT - RK  # cap centre above the bottle's bottom when standing
B_MASS = 0.18
B_COM = 0.090  # above the bottom
B_MU = 0.8
CAP_MU_T = 0.0005  # torsional friction of the cap (condim 4): the grasped cap turns freely
NECK_MU_T = 0.01  # torsional friction of the neck (condim 4): a bottle pinched by its neck does not swing
NECK_GRASP = H_TOT - 0.027  # scripted grasp height above the bottom: the palm (3.1 cm above the grasp point between
#                              the fingers, measured) clears the cap top by 4 mm; the pads hold the neck top and the cap

# ---------------------------------------------------------------- cubes
CUBES = ("red_cube_1", "blue_cube_1")
CUBE_COLOR = {"red_cube_1": "red", "blue_cube_1": "blue"}
CUBE_RGBA = dict(red=(0.85, 0.10, 0.08, 1.0), yellow=(0.92, 0.76, 0.10, 1.0), blue=(0.15, 0.30, 0.85, 1.0))
CUBE_E = 0.032
CUBE_DENS = 700.0  # 23 g

# ---------------------------------------------------------------- predicates / monitor
UPRIGHT = np.radians(10.0)
TILT_START = np.radians(6.0)  # a standing bottle beyond this has started to fall (it tips at ~6.8 deg)
FALLEN = np.radians(40.0)  # lying, or leaning on the gallery rail (45-75 deg)
CAB_MOVE_MAX = 0.08
CAB_TILT_MAX = np.radians(10.0)
DEBOUNCE = 10  # control steps (0.5 s) a predicate change must persist (checked every other step)
WIN_ROBOT = 40  # control steps: a robot contact this recent can explain a loss
WIN_SLAM = 60

# ---------------------------------------------------------------- scripted motion
NAIVE_ERR = 0.008  # Sonnet's placement error 0.4-1.3 cm
FINGER_AHEAD = 0.021  # closed fingertips reach this far ahead of the grasp point (yaw 0, +x) -- measured
REACH_MAX = 0.78  # scripted policies: farthest grasp point at table height (horizontal distance from the base)


def _box(name, pos, size, rgba, extra, vis=True, vis_rgba=None):
    s = [f'<geom name="{name}" type="box" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(rgba)}" {extra} group="0"/>']
    if vis:
        s.append(f'<geom type="box" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(vis_rgba or rgba)}" contype="0" '
                 f'conaffinity="0" group="1"/>')
    return s


def cabinet_xml(name):
    """free carcass (bottom, top, two sides, back) + a drawer on a slide joint along -x (opening towards the robot).
    Carcass geoms contype 2 / conaffinity 1, drawer geoms contype 4 / conaffinity 1: they never collide with each other
    but both collide with the robot, the table and the free objects (contype / conaffinity 1). The slide joint comes
    first in the object's joint list (robosuite appends the free joint), see DrawerCabinet.__init__."""
    hx, hy = CD / 2, CW / 2
    cg = 'friction="0.8 0.005 0.0001" contype="2" conaffinity="1" solimp="0.95 0.99 0.001" solref="0.004 1"'
    carc = []
    carc += _box(f"{name}_bottom", (-PLINTH / 2, 0, CT / 2), (hx + PLINTH / 2, hy, CT / 2), WOOD_D, cg)
    carc += _box(f"{name}_top", (0, 0, CH - CT / 2), (hx, hy, CT / 2), WOOD, cg)
    carc += _box(f"{name}_side_l", (0, hy - CT / 2, CH / 2), (hx, CT / 2, CH / 2 - CT), WOOD, cg)
    carc += _box(f"{name}_side_r", (0, -hy + CT / 2, CH / 2), (hx, CT / 2, CH / 2 - CT), WOOD, cg)
    carc += _box(f"{name}_back", (hx - CT / 2, 0, CH / 2), (CT / 2, hy - CT, CH / 2 - CT), WOOD, cg)
    carc += _box(f"{name}_rail", (hx - RAIL_T / 2, 0, CH + RAIL_H / 2), (RAIL_T / 2, hy, RAIL_H / 2), WOOD_D, cg)
    dg = 'friction="0.8 0.005 0.0001" contype="4" conaffinity="1" solimp="0.95 0.99 0.001" solref="0.004 1"'
    x0 = -hx  # inner face of the front panel (closed)
    zf = D_FLOOR
    dr = []
    dr += _box(f"{name}_front", (x0 - PF / 2, 0, (CT + 0.002 + CH - 0.004) / 2), (PF / 2, hy, (CH - 0.006 - CT) / 2),
               PANEL, dg)
    dr += _box(f"{name}_dfloor", (x0 + DL / 2, 0, zf + DWT / 2), (DL / 2, D_IN_Y + DWT, DWT / 2), INNER, dg)
    zw = zf + DWT + DWH / 2
    dr += _box(f"{name}_dside_l", (x0 + DL / 2, D_IN_Y + DWT / 2, zw), (DL / 2, DWT / 2, DWH / 2), INNER, dg)
    dr += _box(f"{name}_dside_r", (x0 + DL / 2, -D_IN_Y - DWT / 2, zw), (DL / 2, DWT / 2, DWH / 2), INNER, dg)
    dr += _box(f"{name}_dback", (x0 + DL - DWT / 2, 0, zw), (DWT / 2, D_IN_Y, DWH / 2), INNER, dg)
    # flush finger pull (visual only): a dark slot near the top of the front panel
    dr.append(f'<geom type="box" pos="{_fmt((x0 - PF - 0.0005, 0, CH - 0.022))}" size="0.0008 0.035 0.006" '
              f'rgba="{_fmt(SLOT)}" contype="0" conaffinity="0" group="1"/>')
    m = DRAWER_MASS
    ixx, iyy, izz = m * (CW ** 2 + 0.06 ** 2) / 12, m * (DL ** 2 + 0.06 ** 2) / 12, m * (DL ** 2 + CW ** 2) / 12
    drawer = (f'<body name="drawer" pos="0 0 0">\n'
              f'  <inertial pos="{x0 + DL * 0.4:.4f} 0 0.04" mass="{m}" diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>\n'
              f'  <joint name="{name}_slide" type="slide" axis="-1 0 0" pos="0 0 0" limited="true" range="0 {Q_MAX}" '
              f'frictionloss="{SLIDE_FRIC}" damping="0.3" armature="0.01" solreflimit="0.004 1" '
              f'solimplimit="0.95 0.99 0.001"/>\n' + "\n".join("  " + s for s in dr) + "\n</body>")
    m0 = 0.4
    inertial = (f'<inertial pos="{CAB_COM_X:.4f} 0 {CH * 0.5:.4f}" mass="{m0}" diaginertia="{m0 * (CW ** 2 + CH ** 2) / 12:.6f} '
                f'{m0 * (CD ** 2 + CH ** 2) / 12:.6f} {m0 * (CD ** 2 + CW ** 2) / 12:.6f}"/>')
    body = "\n        ".join([inertial] + carc + drawer.split("\n"))
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        {body}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {CH:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{CD / 2:.4f} {CW / 2:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>"""


def bottle_xml(name, rgba, cap_rgba):
    """square bottle: rounded base (narrow flat foot), body, shoulder, neck, ball cap; origin at mid height"""
    h = H_TOT / 2
    zf, zb, zs, zn = -h + HF / 2, -h + HF + (HB - HF) / 2, -h + HB + HS / 2, -h + HB + HS + HN / 2
    zk = -h + HB + HS + HN + RK - CAP_SINK
    m = B_MASS
    ixx = m * ((2 * RB) ** 2 + H_TOT ** 2) / 14
    izz = m * (2 * (2 * RB) ** 2) / 12
    fr = f'friction="{B_MU} 0.005 0.0001"'
    sol = 'solimp="0.95 0.99 0.001" solref="0.005 1"'
    parts = [("foot", "box", (0, 0, zf), (RF, RF, HF / 2), rgba, fr + ' priority="2"'),
             ("body", "box", (0, 0, zb), (RB, RB, (HB - HF) / 2), rgba, fr),
             ("shoulder", "box", (0, 0, zs), (RS, RS, HS / 2), rgba, fr),
             ("neck", "cylinder", (0, 0, zn), (RN, HN / 2), rgba,
              f'friction="{B_MU} {NECK_MU_T} 0.0001" condim="4" priority="2"'),
             ("cap", "sphere", (0, 0, zk), (RK,), cap_rgba, f'friction="1 {CAP_MU_T} 0.0001" condim="4" priority="2"')]
    s = [f'        <inertial pos="0 0 {-h + B_COM:.4f}" mass="{m}" diaginertia="{ixx:.6f} {ixx:.6f} {izz:.6f}" />']
    for nm, typ, pos, size, c, extra in parts:
        s.append(f'        <geom name="{name}_{nm}" type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" '
                 f'{extra} {sol} group="0" />')
        if nm != "foot":
            s.append(f'        <geom type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" contype="0" '
                     f'conaffinity="0" group="1" />')
    # rounded base (visual): a slightly narrower bottom band, and a label
    s.append(f'        <geom type="box" pos="0 0 {zf:.4f}" size="{RB - 0.004:.4f} {RB - 0.004:.4f} {HF / 2:.4f}" '
             f'rgba="{_fmt(rgba)}" contype="0" conaffinity="0" group="1" />')
    s.append(f'        <geom type="box" pos="0 0 {zb + 0.005:.4f}" size="{RB + 0.0008:.4f} {RB + 0.0008:.4f} 0.028" '
             f'rgba="0.92 0.90 0.80 1" contype="0" conaffinity="0" group="1" />')
    return _wrap(name, "\n".join(s), h, RB * 1.42)


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


def _lr(v):
    return f"{100 * abs(v):.1f} cm to the {'left' if v > 0 else 'right'}"


def _reach(p):
    return float(np.hypot(p[0] + 0.66, p[1]))


class DrawerSkills(Skills):
    """Skills that feed the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class DrawerCabinet(LFFTask):
    name = "lh_drawer_cabinet"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Restore", "Plan")
    protocol = "cross"
    max_attempts = 3
    # careful: 2 bottles x ~8 commands + 3 cubes x ~6 + closing the drawer ~5 + looking ~15; standing 2 fallen bottles
    # up again ~20; + slack
    agent_max_steps = 120
    instruction = ("Tidy up the table: stand the green bottle and the brown bottle upright on top of the small wooden "
                   "cabinet, put the red cube and the blue cube into the cabinet's open drawer (it is pulled "
                   "out towards the robot), and close the drawer. When you finish, both bottles must stand upright on the cabinet top, both cubes must be in "
                   "the closed drawer, the cabinet must still stand where it is, and the gripper must be open and not "
                   "touching anything. (Each bottle has a round cap; held by its cap, a bottle hangs straight down.)")
    instruction_indirect = ("Clean up: set the two tall bottles (the green one and the brown one) upright on the little "
                            "wooden cabinet, drop the two small cubes (red and blue) into its pulled-out drawer and "
                            "shut the drawer. In the end both bottles stand on top of the cabinet, the cubes are inside the shut drawer, "
                            "the cabinet stays in its place, and the open gripper touches nothing. (A bottle held by "
                            "its round cap hangs straight down.)")
    agent_object_names = ["green bottle (yellow round cap)", "brown bottle (blue round cap)", "red cube", "blue cube",
                          "small wooden cabinet with its drawer pulled out (the drawer front faces the robot)"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        cab = register_generated("LffLhdCabinet", cabinet_xml(CAB_MODEL))
        objs = [(CAB, cab, "cabinet_region", (0.10, -0.02, 0.12, 0.0))]
        for i, b in enumerate(BOTTLES):
            c = B_NAME[b]
            cat = register_generated(f"LffLhdBottle{c.title()}", bottle_xml(f"lff_lhd_bottle_{c}", B_RGBA[b], CAP_RGBA[b]))
            objs.append((b, cat, f"{c}_bottle_region", (-0.30, 0.20 + 0.13 * i, -0.28, 0.22 + 0.13 * i)))
        for i, n in enumerate(CUBES):
            c = CUBE_COLOR[n]
            cat = register_generated(f"LffLhdCube{c.title()}",
                                     box_xml(f"lff_lhd_cube_{c}", (CUBE_E / 2,) * 3, CUBE_RGBA[c], density=CUBE_DENS,
                                             friction=(0.8, 0.005, 0.0001)))
            objs.append((n, cat, f"{c}_cube_region", (-0.30, -0.45 + 0.13 * i, -0.28, -0.43 + 0.13 * i)))
        return write_bddl(self.name, "tidy up the table", objs, [])

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        pre = f"{CAB}_{CAB_MODEL}"
        self.jid = m.joint_name2id(pre + "_slide")
        self.qa = int(m.jnt_qposadr[self.jid])
        self.dof = int(m.jnt_dofadr[self.jid])
        self.cab_bid = envs.body_id(self.env, CAB)
        self.drawer_bid = m.body_name2id(f"{CAB}_drawer")
        # the cabinet's free joint is the LAST of its joints (robosuite appends it after the drawer slide), so
        # envs.free_joint_addr / place_upright (joints[0]) must not be used for it
        fj = m.joint_name2id(self.env.objects_dict[CAB].joints[-1])
        assert int(m.jnt_type[fj]) == 0
        self.cab_qa, self.cab_da = int(m.jnt_qposadr[fj]), int(m.jnt_dofadr[fj])
        self.feet_gid = m.geom_name2id(pre + "_bottom")
        self.cab_gids = {g for g in range(m.ngeom) if m.geom_bodyid[g] in (self.cab_bid, self.drawer_bid)
                         and m.geom_contype[g] != 0}
        self.bottle_gids = {b: set(envs.obj_geom_ids(self.env, b)) for b in BOTTLES}
        self.cube_gids = {c: set(envs.obj_geom_ids(self.env, c)) for c in CUBES}
        self.cap_gid = {b: m.geom_name2id(f"{b}_lff_lhd_bottle_{B_NAME[b]}_cap") for b in BOTTLES}
        # per-geom lookup tables for the substep contact scan
        self.part = np.zeros(m.ngeom, int)  # 1 = hand / fingers, 2 = arm links
        for g in envs.robot_geom_ids(self.env):
            n = m.geom_id2name(g) or ""
            self.part[g] = 1 if n.startswith("gripper0_") else 2
        self.bidx = np.full(m.ngeom, -1, int)
        for i, b in enumerate(BOTTLES):
            self.bidx[list(self.bottle_gids[b])] = i
        self.task_obj = np.zeros(m.ngeom, bool)
        for s in list(self.bottle_gids.values()) + list(self.cube_gids.values()) + [self.cab_gids]:
            self.task_obj[list(s)] = True
        self.catch = dict(force=0.0, zone=ZONE, damp=0.0)
        sim = self.env.sim
        if not getattr(sim, "_lhd_wrapped", False):
            orig = sim.step

            def step(*a, **k):
                orig(*a, **k)
                t = getattr(sim, "_lhd_task", None)
                if t is not None:
                    t._substep()
            sim.step = step
            sim._lhd_wrapped = True
        sim._lhd_task = self
        self.inst = None
        self.mon = None
        self.sk = None
        self._sub = {}

    # ------------------------------------------------------------ drawer mechanism (every MuJoCo substep)
    def q(self):
        return float(self.env.sim.data.qpos[self.qa])

    def catch_force(self, q, qd):
        """generalised force on the slide (positive = opening): inside the last ZONE of travel a constant pull towards
        closed (ramped in over 5 mm), plus a damper on closing motion for soft-close drawers"""
        c = self.catch
        z = c["zone"]
        if q >= z or c["force"] <= 0:
            return 0.0
        ramp = float(np.clip((z - q) / 0.005, 0.0, 1.0))
        f = -c["force"] * ramp
        if qd < 0 and c["damp"] > 0:
            f -= c["damp"] * qd * ramp
        return f

    def _substep(self):
        d = self.env.sim.data
        q, qd = float(d.qpos[self.qa]), float(d.qvel[self.dof])
        d.qfrc_applied[self.dof] = self.catch_force(q, qd)
        if self.mon is None:
            return
        sub = self._sub
        # slam detection: the closing speed just before the drawer reaches its stop
        if q > 0.002:
            sub["approach_v"] = -qd
            sub["armed"] = True
            if q < self.catch["zone"] and sub.get("zone_xy") is None:
                sub["zone_xy"] = self.cab_xy().tolist()
            elif q > self.catch["zone"] + 0.005:
                sub["zone_xy"] = None
        elif sub.get("armed"):
            sub["armed"] = False
            v = float(sub.get("approach_v", 0.0))
            sub.setdefault("impacts", []).append((v, sub.get("zone_xy")))
            sub["zone_xy"] = None
        n = d.ncon
        if not n:
            return
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        for a, b in zip(g1, g2):
            ia, ib = self.bidx[a], self.bidx[b]
            if ia < 0 and ib < 0:
                continue
            if ia >= 0 and ib >= 0:
                if ia != ib:
                    sub.setdefault("bb", set()).update((int(ia), int(ib)))
                continue
            i, o = (ia, b) if ia >= 0 else (ib, a)
            p = int(self.part[o])
            if p:
                r = sub.setdefault("rob", {})
                r[int(i)] = 2 if (p == 2 or r.get(int(i)) == 2) else 1

    # ------------------------------------------------------------ instances
    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        slam = bool((int(seed) * 7) % 20 < round(20 * P_SLAM))  # balanced: 14 of every 20 consecutive seeds
        cx, cy = float(rng.uniform(-0.13, -0.10)), float(rng.uniform(-0.04, 0.04))
        s = 1.0 if rng.random() < 0.5 else -1.0  # bottles on the left (+1) or right (-1), cubes on the other side
        bottles = {}
        order = rng.permutation(2)
        for j, i in enumerate(order):
            bx = cx + (-0.05, 0.05)[j] + float(rng.uniform(-0.01, 0.01))
            bottles[BOTTLES[i]] = [bx, cy + s * float(rng.uniform(0.21, 0.24))]
        perm = rng.permutation(2)
        # cubes 25-29 cm to the side of the cabinet's centre line: picked from nearer, the forearm sweeps over the
        # cabinet top and knocks the bottles standing there (measured feasibility map, see the task card)
        cubes = {CUBES[i]: [cx + (-0.16, -0.04)[j] + float(rng.uniform(-0.01, 0.01)),
                            cy - s * float(rng.uniform(0.25, 0.29))] for j, i in enumerate(perm)}
        force = float(rng.uniform(*F_SPRING))
        soft_v = float(rng.uniform(*SOFT_V))
        return dict(seed=int(seed), self_closing=slam, drawer="self-closing" if slam else "soft-close",
                    cab_xy=[cx, cy], bottle_side="left" if s > 0 else "right", bottle_start=bottles,
                    bottle_yaw={b: float(rng.uniform(-0.3, 0.3)) for b in BOTTLES}, cube_start=cubes,
                    cube_yaw={c: float(rng.uniform(-0.3, 0.3)) for c in CUBES},
                    spring_n=force, damp=0.0 if slam else force / soft_v,
                    carcass_kg=float(rng.uniform(*CARCASS_MASS)), feet_mu=float(rng.uniform(*FEET_MU)),
                    naive_noise_seed=int(rng.integers(1 << 30)))

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        M = float(inst["carcass_kg"])
        m.body_mass[self.cab_bid] = M
        m.body_inertia[self.cab_bid] = [M * (CW ** 2 + CH ** 2) / 12, M * (CD ** 2 + CH ** 2) / 12,
                                        M * (CD ** 2 + CW ** 2) / 12]
        m.geom_friction[self.feet_gid] = [float(inst["feet_mu"]), 0.005, 0.0001]
        m.geom_priority[self.feet_gid] = 3
        self.catch = dict(force=float(inst["spring_n"]), zone=ZONE, damp=float(inst["damp"]))
        self.set_drawer(Q_OPEN)
        self.place_cabinet(inst["cab_xy"], 0.0)
        for b in BOTTLES:
            envs.place_upright(self.env, b, inst["bottle_start"][b], yaw=inst["bottle_yaw"][b], clearance=0.0005)
        for c in CUBES:
            envs.place_upright(self.env, c, inst["cube_start"][c], yaw=inst["cube_yaw"][c], clearance=0.001)

    def set_drawer(self, q):
        d = self.env.sim.data
        d.qpos[self.qa] = q
        d.qvel[self.dof] = 0.0

    def place_cabinet(self, xy, yaw=0.0, clearance=0.0005):
        d = self.env.sim.data
        a, v = self.cab_qa, self.cab_da
        d.qpos[a:a + 3] = [xy[0], xy[1], Z + clearance]  # body origin = bottom of the carcass
        d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[v:v + 6] = 0.0
        self.env.sim.forward()

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = DrawerSkills(self.env, self, recorder=recorder)
        self._sub = {}
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ geometry helpers (world frame)
    def cab_R(self):
        return self.env.sim.data.body_xmat[self.cab_bid].reshape(3, 3).copy()

    def cab_origin(self):
        return self.env.sim.data.body_xpos[self.cab_bid].copy()

    def cab_xy(self):
        return self.cab_origin()[:2]

    def to_cab(self, p):
        """world point -> cabinet frame (x forward, origin at the footprint centre on the floor)"""
        return self.cab_R().T @ (np.asarray(p, float) - self.cab_origin())

    def from_cab(self, p):
        return self.cab_origin() + self.cab_R() @ np.asarray(p, float)

    def top_z(self):
        return float(self.from_cab([0.0, 0.0, CH])[2])

    def cab_front_x(self):
        """world x of the carcass's front face (cabinet assumed close to yaw 0)"""
        return float(self.from_cab([-CD / 2, 0.0, 0.0])[0])

    def drawer_front_x(self):
        """world x of the drawer front panel's outer face"""
        return self.cab_front_x() - self.q() - PF

    def cab_tilt(self):
        return float(np.arccos(np.clip(self.cab_R()[2, 2], -1.0, 1.0)))

    def cab_moved(self):
        return float(np.linalg.norm(self.cab_xy() - np.asarray(self.inst["cab_xy"])))

    def R(self, name):
        return self.env.sim.data.body_xmat[envs.body_id(self.env, name)].reshape(3, 3).copy()

    def tilt(self, name):
        return float(np.arccos(np.clip(self.R(name)[2, 2], -1.0, 1.0)))

    def bottle_base(self, b):
        """world position of the centre of the bottle's bottom face"""
        return envs.obj_pos(self.env, b) - self.R(b)[:, 2] * H_TOT / 2

    def cap_xyz(self, b):
        return self.env.sim.data.geom_xpos[self.cap_gid[b]].copy()

    def held(self, name):
        return self.sk is not None and self.sk.holding(name)

    # ------------------------------------------------------------ stage predicates
    def bottle_on_top(self, b, tol=0.0, tilt=UPRIGHT):
        if self.tilt(b) > tilt or self.held(b):
            return False
        c = self.to_cab(self.bottle_base(b))
        if c[0] < -CD / 2 + RF - tol or c[0] > CD / 2 - RAIL_T - RF + tol or abs(c[1]) > CW / 2 - RF + tol:
            return False
        return bool(-0.006 <= c[2] - CH <= 0.012)

    def cube_in_drawer(self, c):
        """cube centre inside the drawer box (drawer frame), its footprint within the walls (+3 mm), resting low enough
        to pass under the cabinet top; not held"""
        if self.held(c):
            return False
        d = self.env.sim.data
        Rd = d.body_xmat[self.drawer_bid].reshape(3, 3)
        p = Rd.T @ (envs.obj_pos(self.env, c) - d.body_xpos[self.drawer_bid])
        x0 = -CD / 2
        e = CUBE_E / 2 * 1.2  # footprint half size (a yawed cube reaches a little farther)
        tol = 0.003
        return bool(x0 + e - tol <= p[0] <= x0 + D_IN_X - e + tol and abs(p[1]) <= D_IN_Y - e + tol
                    and FLOOR_TOP - 0.005 <= p[2] - CUBE_E / 2 <= FLOOR_TOP + DWH + 0.01)

    def drawer_closed(self):
        return self.q() <= CLOSED

    def stage_keys(self):
        return [f"{B_NAME[b]}_bottle_on_top" for b in BOTTLES] + [f"{CUBE_COLOR[c]}_cube_in_drawer" for c in CUBES] + [
            "drawer_closed"]

    def stage_values(self, hyst=None):
        hyst = hyst or {}
        v = {}
        for b in BOTTLES:
            k = f"{B_NAME[b]}_bottle_on_top"
            v[k] = self.bottle_on_top(b, 0.003, UPRIGHT + np.radians(3.0)) if hyst.get(k) else self.bottle_on_top(b)
        for c in CUBES:
            v[f"{CUBE_COLOR[c]}_cube_in_drawer"] = self.cube_in_drawer(c)
        v["drawer_closed"] = self.q() <= (CLOSED + 0.003 if hyst.get("drawer_closed") else CLOSED)
        return v

    # ------------------------------------------------------------ per-step monitor
    def _init_monitor(self):
        keys = self.stage_keys()
        st = {k: False for k in keys}
        self.mon = dict(state=st, ever=[], lost=[], lost_cause={}, events=[], cand={k: 0 for k in keys}, step=0,
                        last_robot=[None, None], last_bb=None, slams=[], standing_since=[None, None],
                        fall_start=[None, None], touched_before_fall=[False, False], max_tilt=[0.0, 0.0],
                        n_fallen=0, fallen=[False, False])

    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        mon["step"] += 1
        t = int(self.sk.n_steps)
        sub, self._sub = self._sub, {k: v for k, v in self._sub.items() if k in ("approach_v", "armed", "zone_xy")}
        for i, part in sub.get("rob", {}).items():
            b = BOTTLES[i]
            if not self.held(b):
                mon["last_robot"][i] = ["hand" if part == 1 else "arm", t, round(float(np.degrees(self.tilt(b))), 1)]
        if sub.get("bb"):
            mon["last_bb"] = t
        for v, zxy in sub.get("impacts", []):
            if v >= V_SLAM:
                lurch = None
                if zxy is not None:
                    dxy = self.cab_xy() - np.asarray(zxy)
                    lurch = [round(100 * float(dxy[0]), 2), round(100 * float(dxy[1]), 2)]
                on = [B_NAME[b] for b in BOTTLES if mon["state"][f"{B_NAME[b]}_bottle_on_top"]]
                e = dict(event="drawer_slam", step=t, speed_mps=round(float(v), 2), cabinet_jerk_cm=lurch,
                         bottles_on_top=on, bottles_fell=[], robot_touched_before_fall=[])
                mon["events"].append(e)
                mon["slams"].append(e)
            else:
                mon["events"].append(dict(event="drawer_shut", step=t, speed_mps=round(float(v), 3)))
        # fall tracking (R4: did the robot touch a bottle before it started to fall?)
        for i, b in enumerate(BOTTLES):
            tl = self.tilt(b)
            mon["max_tilt"][i] = max(mon["max_tilt"][i], float(np.degrees(tl)))
            if self.held(b):
                mon["standing_since"][i] = None
                continue
            if tl < np.radians(3.0):
                if mon["standing_since"][i] is None:
                    mon["standing_since"][i] = t
                mon["fall_start"][i] = None
            elif tl > TILT_START and mon["standing_since"][i] is not None and mon["fall_start"][i] is None:
                mon["fall_start"][i] = t
                lr = mon["last_robot"][i]
                mon["touched_before_fall"][i] = bool(lr is not None and t - lr[1] <= 20 and lr[2] < np.degrees(TILT_START))
            if not mon["fallen"][i] and tl > FALLEN:
                mon["fallen"][i] = True
                mon["n_fallen"] += 1
                mon["events"].append(dict(event="bottle_fell", bottle=B_NAME[b], step=t, cause=self._cause(i, t),
                                          robot_touched_before_fall=bool(mon["touched_before_fall"][i]),
                                          direction=self._fall_dir(b)))
            elif mon["fallen"][i] and tl < UPRIGHT:
                mon["fallen"][i] = False
        if mon["step"] % 2:
            return
        vals = self.stage_values(hyst=mon["state"])
        for k, now in vals.items():
            self._check(k, now, t)

    def _fall_dir(self, b):
        ax = self.R(b)[:2, 2]
        if np.linalg.norm(ax) < 0.2:
            return "none"
        return ("forward" if ax[0] > 0 else "backward") if abs(ax[0]) >= abs(ax[1]) else ("left" if ax[1] > 0 else "right")

    def _cause(self, i, t):
        """why bottle i left the cabinet top"""
        b = BOTTLES[i]
        mon = self.mon
        if self.held(b):
            return "picked_up"
        lr = mon["last_robot"][i]
        robot_recent = lr is not None and t - lr[1] <= WIN_ROBOT
        if robot_recent and mon["touched_before_fall"][i]:
            return f"robot_{lr[0]}"
        if robot_recent and lr[2] < np.degrees(TILT_START):
            return f"robot_{lr[0]}"
        if mon["slams"] and t - mon["slams"][-1]["step"] <= WIN_SLAM:
            return "drawer_slam"
        if mon["last_bb"] is not None and t - mon["last_bb"] <= WIN_ROBOT:
            return "bottle"
        if robot_recent:
            return f"robot_{lr[0]}"
        return "other"

    def _check(self, key, now, t):
        mon = self.mon
        if now == mon["state"][key]:
            mon["cand"][key] = 0
            return
        mon["cand"][key] += 2
        if mon["cand"][key] < DEBOUNCE:
            return
        mon["cand"][key] = 0
        mon["state"][key] = now
        t0 = int(t - DEBOUNCE + 1)
        if now:
            restored = key in mon["lost"]
            e = dict(stage=key, event="restored" if restored else "achieved", step=t0)
            if restored:
                e["after"] = mon["lost_cause"].get(key)
                mon["lost"].remove(key)
            if key not in mon["ever"]:
                mon["ever"].append(key)
            mon["events"].append(e)
            return
        if key not in mon["ever"]:
            return
        if key.endswith("bottle_on_top"):
            i = [f"{B_NAME[b]}_bottle_on_top" for b in BOTTLES].index(key)
            cause = self._cause(i, t)
            if cause == "drawer_slam":
                s = mon["slams"][-1]
                if B_NAME[BOTTLES[i]] not in s["bottles_fell"]:
                    s["bottles_fell"].append(B_NAME[BOTTLES[i]])
                    s["robot_touched_before_fall"].append(bool(mon["touched_before_fall"][i]))
        elif key == "drawer_closed":
            cause = "opened"
        else:
            c = CUBES[[f"{CUBE_COLOR[x]}_cube_in_drawer" for x in CUBES].index(key)]
            cause = "picked_up" if self.held(c) else "other"
        if key not in mon["lost"]:
            mon["lost"].append(key)
        mon["lost_cause"][key] = cause
        mon["events"].append(dict(stage=key, event="lost", step=t0, cause=cause))

    def _finalize_monitor(self, vals):
        """close the event log at finish: a stage the debounced monitor still holds but the final judge does not (or
        vice versa) is logged now (final=True)"""
        mon = self.mon
        t = int(self.sk.n_steps)
        for k, now in vals.items():
            if now == mon["state"][k]:
                continue
            n0 = len(mon["events"])
            mon["cand"][k] = DEBOUNCE  # force the transition
            self._check(k, now, t + DEBOUNCE - 1)
            if len(mon["events"]) > n0:
                mon["events"][-1]["final"] = True

    # ------------------------------------------------------------ outcome
    def _static(self):
        d = self.env.sim.data
        for n in BOTTLES + CUBES:
            _, da = envs.free_joint_addr(self.env, n)
            if np.linalg.norm(d.qvel[da:da + 3]) > 0.02 or np.linalg.norm(d.qvel[da + 3:da + 6]) > 0.5:
                return False
        v = self.cab_da
        return bool(np.linalg.norm(d.qvel[v:v + 3]) < 0.02 and abs(float(d.qvel[self.dof])) < 0.02)

    def robot_touching(self):
        d = self.env.sim.data
        out = set()
        for j in range(d.ncon):
            c = d.contact[j]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if self.part[a] and self.task_obj[b]:
                    out.add("a bottle" if self.bidx[b] >= 0 else "the cabinet" if b in self.cab_gids else "a cube")
        return sorted(out)

    def outcome(self, inst):
        sk = self.sk
        sk.hold(10)
        touching = set()
        for _ in range(20):
            sk.hold(1)
            touching.update(self.robot_touching())
        touching = sorted(touching)
        static = self._static()
        vals = self.stage_values()
        held = [n for n in BOTTLES + CUBES if self.held(n)]
        cab_ok = bool(self.cab_tilt() <= CAB_TILT_MAX and self.cab_moved() <= CAB_MOVE_MAX)
        grip_open = bool(sk.grip < 0)
        self._finalize_monitor(vals)
        mon = self.mon
        ev = list(mon["events"])
        success = bool(all(vals.values()) and cab_ok and static and not held and not touching and grip_open)
        lost = [e for e in ev if e.get("event") == "lost"]
        regress = [e for e in lost if e.get("cause") != "picked_up"]
        restored = [e for e in ev if e.get("event") == "restored" and e.get("after") != "picked_up"]
        slams = [e for e in ev if e.get("event") == "drawer_slam"]
        slam_losses = [e for e in lost if e.get("cause") == "drawer_slam"]
        trap = bool(slam_losses)
        bkeys = [f"{B_NAME[b]}_bottle_on_top" for b in BOTTLES]
        never_bottle = [B_NAME[b] for b, k in zip(BOTTLES, bkeys) if not vals[k] and k not in mon["ever"]]
        lost_bottle = [B_NAME[b] for b, k in zip(BOTTLES, bkeys) if not vals[k] and k in mon["ever"]]
        cubes_out = [CUBE_COLOR[c] for c in CUBES if not vals[f"{CUBE_COLOR[c]}_cube_in_drawer"]]
        if success:
            failure = "none"
        elif held:
            failure = "held"
        elif not cab_ok:
            failure = "cabinet_moved"
        elif lost_bottle:
            failure = "bottle_knocked"
        elif never_bottle:
            failure = "bottle_not_on_top"
        elif cubes_out:
            failure = "cube_not_in_drawer"
        elif not vals["drawer_closed"]:
            failure = "drawer_open"
        elif touching:
            failure = "touching"
        elif not grip_open:
            failure = "gripper_closed"
        else:
            failure = "not_static"
        detail = self._detail(vals, held, touching, cab_ok, static, grip_open, success, ev)
        return dict(success=success, failure=failure, stages=vals, events=ev, n_regressions=len(regress),
                    n_lost=len(lost), n_restored=len(restored), trap_fired=trap,
                    recovered_in_attempt=bool(trap and success), n_slams=len(slams),
                    slam_speeds=[e["speed_mps"] for e in slams], slam_jerk_cm=[e["cabinet_jerk_cm"] for e in slams],
                    bottles_lost_to_slam=sorted({e["stage"].split("_")[0] for e in slam_losses}),
                    robot_touched_before_fall=[bool(x) for x in mon["touched_before_fall"]],
                    bottle_max_tilt_deg=[round(x, 1) for x in mon["max_tilt"]], n_fallen=mon["n_fallen"],
                    held=held, touching=touching, static=bool(static), gripper_open=grip_open,
                    drawer_q_cm=round(100 * self.q(), 2), cabinet_moved_cm=round(100 * self.cab_moved(), 2),
                    cabinet_tilt_deg=round(float(np.degrees(self.cab_tilt())), 1),
                    bottle_tilts_deg={B_NAME[b]: round(float(np.degrees(self.tilt(b))), 1) for b in BOTTLES},
                    self_closing=bool(inst["self_closing"]), detail=detail)

    def _detail(self, vals, held, touching, cab_ok, static, grip_open, success, ev):
        parts = []
        if success:
            parts.append("Both bottles stand on the cabinet, the two cubes are in the closed drawer.")
        top_c = self.from_cab([0.0, 0.0, CH])
        for b in BOTTLES:
            k = f"{B_NAME[b]}_bottle_on_top"
            if vals[k]:
                continue
            nm = B_NAME[b]
            if b in held:
                parts.append(f"The {nm} bottle is still held by the gripper.")
                continue
            tl = float(np.degrees(self.tilt(b)))
            base = self.bottle_base(b)
            rel = base - top_c
            on_top = abs(float(rel[2])) < 0.03
            ax = self.R(b)[:, 2]
            where = f"its base is {_fb(rel[0])} and {_lr(rel[1])} of the centre of the cabinet top"
            if tl <= np.degrees(UPRIGHT):
                txt = (f"The {nm} bottle stands but not on the cabinet top: {where}." if not on_top else
                       f"The {nm} bottle stands with part of its base over the edge of the cabinet top: {where}.")
            elif on_top and ax[0] > 0.3 and self.to_cab(self.cap_xyz(b))[0] > CD / 2 - RAIL_T:
                txt = (f"The {nm} bottle has tipped over forward on the cabinet top and leans {tl:.0f} degrees against "
                       f"the rail along the top's far edge; {where}.")
            elif on_top:
                txt = f"The {nm} bottle has tipped over on the cabinet top ({tl:.0f} degrees); {where}."
            elif base[2] < Z + 0.03:
                txt = f"The {nm} bottle has fallen over and lies on the table; {where}."
            else:
                txt = f"The {nm} bottle has fallen over and leans {tl:.0f} degrees against something; {where}."
            lost = [e for e in ev if e.get("stage") == k and e.get("event") == "lost"]
            if lost:
                why = lost[-1].get("cause", "other")
                txt += {"drawer_slam": " It had stood on the cabinet top; it fell when the drawer snapped shut and the "
                                       "cabinet jerked.",
                        "robot_hand": " It had stood on the cabinet top; the gripper knocked it.",
                        "robot_arm": " It had stood on the cabinet top; the robot's arm knocked it.",
                        "bottle": " It had stood on the cabinet top; the other bottle knocked it."}.get(
                    why, " It had stood on the cabinet top earlier.")
            parts.append(txt)
        for c in CUBES:
            if vals[f"{CUBE_COLOR[c]}_cube_in_drawer"]:
                continue
            nm = CUBE_COLOR[c]
            if c in held:
                parts.append(f"The {nm} cube is still held by the gripper.")
                continue
            p = envs.obj_pos(self.env, c)
            rel = self.to_cab(p)
            if abs(rel[0]) < CD / 2 and abs(rel[1]) < CW / 2 and rel[2] > CH - 0.01:
                parts.append(f"The {nm} cube lies on the cabinet top, not in the drawer.")
            else:
                parts.append(f"The {nm} cube is not in the drawer: it is {_fb(rel[0] + CD / 2)} and {_lr(rel[1])} of the "
                             f"middle of the cabinet's front.")
        if not vals["drawer_closed"]:
            parts.append(f"The drawer is still {100 * self.q():.1f} cm open.")
        if not cab_ok:
            if self.cab_tilt() > CAB_TILT_MAX:
                parts.append("The cabinet has been knocked over.")
            else:
                parts.append(f"The cabinet was moved {100 * self.cab_moved():.0f} cm from where it stood.")
        if not success and all(vals.values()) and cab_ok:
            if touching:
                parts.append("The gripper is still touching " + " and ".join(touching) + ".")
            elif not grip_open:
                parts.append("The gripper is closed.")
            elif not static:
                parts.append("Something was still moving.")
        slams = [e for e in ev if e.get("event") == "drawer_slam" and e.get("bottles_fell")]
        if slams and success:
            parts.append("(The drawer snapped shut and knocked bottles off the cabinet; they were stood up again.)")
        return " ".join(parts)

    # ------------------------------------------------------------ scripted primitives (move_to + gripper only)
    def _go(self, xyz, yaw=None, speed=None, tol=0.004, max_steps=300):
        return self.sk.move_to(xyz, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)

    def _lift_to(self, z, speed=0.15):
        cur = self.sk.eef_pos()
        if cur[2] < z - 0.005:
            self._go([cur[0], cur[1], z], speed=speed)

    def _settle(self, name, max_checks=25):
        d = self.env.sim.data
        _, da = envs.free_joint_addr(self.env, name)
        for _ in range(max_checks):
            if np.linalg.norm(d.qvel[da:da + 3]) < 0.01 and np.linalg.norm(d.qvel[da + 3:da + 6]) < 0.2:
                return
            self.sk.hold(2)

    def _box_yaw(self, name):
        R = self.R(name)
        i = int(np.argmin(np.abs(R[2])))
        yaw = float(np.arctan2(R[1, i], R[0, i]))
        y = float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        cands = [y + k * np.pi / 2 for k in (-1, 0, 1)]
        return min(cands, key=lambda c: abs(c - self.sk.yaw) if abs(c) <= np.pi / 2 else 9.0)

    def pick_bottle(self, b, grasp="neck", z_clear=0.05, yaw=None, v=0.15):
        """Sonnet-like pick of a standing bottle from above: over it, straight down, close, lift until its bottom
        clears the cabinet top by z_clear. grasp: neck (1 cm below its top; the body cannot be reached from above, the
        palm would land on the cap) | cap (the bottle then hangs from a pivot)"""
        sk = self.sk
        self._settle(b)
        p = envs.obj_pos(self.env, b)
        bot = self.bottle_base(b)[2]
        zg = {"neck": bot + NECK_GRASP, "cap": bot + CAP_Z}[grasp]
        yaw = self._box_yaw(b) if yaw is None else yaw
        sk.set_gripper(False, steps=8)
        top = bot + H_TOT
        self.transit(p[:2], top + 0.06, yaw=yaw, v=v)
        self._go([p[0], p[1], zg + 0.015], speed=0.1)
        self._go([p[0], p[1], zg], speed=0.04, tol=0.003, max_steps=150)
        sk.set_gripper(True, steps=15)
        lift = self.top_z() + z_clear + (zg - bot)
        self._go([p[0], p[1], lift], speed=0.10)
        sk.hold(5)
        if not self.held(b):
            sk.set_gripper(False, steps=8)
            return False
        return True

    def put_bottle(self, b, xy, gap=0.01, z_clear=0.05, v=0.12, settle=15):
        """carry the held bottle over xy (its axis, not the grasp point), straight down until its bottom is `gap` above
        the cabinet top, let go, lift"""
        sk = self.sk
        rel = envs.obj_pos(self.env, b)[:2] - sk.eef_pos()[:2]
        q = np.asarray(xy, float) - rel
        zc = self.top_z() + z_clear + (sk.eef_pos()[2] - self.bottle_base(b)[2])
        self.transit(q, zc, v=v, carried=b, margin=z_clear)
        sk.hold(settle)
        rel = envs.obj_pos(self.env, b)[:2] - sk.eef_pos()[:2]
        q = np.asarray(xy, float) - rel
        dz = sk.eef_pos()[2] - envs.obj_min_z(self.env, b)
        zr = self.top_z() + dz + gap
        self._go([q[0], q[1], zr + 0.02], speed=0.08)
        self._go([q[0], q[1], zr], speed=0.03, tol=0.003, max_steps=150)
        sk.hold(5)
        sk.set_gripper(False, steps=12)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], self.top_z() + H_TOT + 0.06], speed=0.08)
        self._settle(b)

    def obstacle_top(self, p0, p1, ignore=(), r=0.09):
        """highest top (world z) of standing bottles within r of the straight path p0 -> p1 (xy), and of the cabinet
        top if the path passes over it (expanded by 4 cm)"""
        a, b = np.asarray(p0, float)[:2], np.asarray(p1, float)[:2]
        ab = b - a
        L2 = max(float(ab @ ab), 1e-9)

        def dist(q):
            u = float(np.clip((q - a) @ ab / L2, 0.0, 1.0))
            return float(np.linalg.norm(a + u * ab - q))
        top = Z
        for o in BOTTLES:
            if o in ignore or self.held(o):
                continue
            if dist(envs.obj_pos(self.env, o)[:2]) < r:
                top = max(top, envs.obj_max_z(self.env, o))
        for u in np.linspace(0.0, 1.0, 21):
            c = self.to_cab(np.r_[a + u * ab, Z])
            if abs(c[0]) < CD / 2 + 0.04 + PF + max(self.q(), 0.0) and abs(c[1]) < CW / 2 + 0.04:
                top = max(top, self.top_z() + RAIL_H)
                break
        return top

    def transit(self, xy, z_min, yaw=None, v=0.15, carried=None, margin=0.06):
        """Sonnet-like transit: up (to clear what is in the way), straight across, the caller then goes down"""
        sk = self.sk
        cur = sk.eef_pos()
        obst = self.obstacle_top(cur, xy, ignore=(carried,) if carried else ())
        below = (sk.eef_pos()[2] - envs.obj_min_z(self.env, carried)) if carried else 0.03
        zt = max(cur[2], z_min, obst + margin + below)
        if cur[2] < zt - 0.005:
            self._go([cur[0], cur[1], zt], speed=v)
        self._go([xy[0], xy[1], zt], yaw=yaw, speed=v)

    def pick_cube(self, c, z_carry=0.15, v=0.15):
        sk = self.sk
        for _ in range(2):
            self._settle(c)
            p = envs.obj_pos(self.env, c)
            yaw = self._box_yaw(c)
            sk.set_gripper(False, steps=8)
            self.transit(p[:2], Z + z_carry, yaw=yaw, v=v)
            self._go([p[0], p[1], p[2] + 0.015], speed=0.1)
            self._go([p[0], p[1], p[2]], speed=0.04, tol=0.003, max_steps=120)
            sk.set_gripper(True, steps=15)
            self._go([p[0], p[1], max(Z + z_carry, sk.eef_pos()[2])], speed=v)
            if self.held(c):
                return True
            sk.set_gripper(False, steps=8)
        return False

    def drawer_slot(self, k, noise=None):
        """world xy of cube slot k (0..1) in the exposed part of the open drawer: a column along the drawer's middle
        (with the wrist at yaw 0 the open fingers, 13.2 cm across, stay clear of the side walls and of the cube
        already in; the palm, 3.9 cm ahead of the grasp point, stays clear of the carcass; the far slot first)"""
        q = min(self.q(), Q_OPEN)
        x = self.cab_front_x() - 0.045 - k * (q - 0.070)
        y = self.cab_xy()[1]
        out = np.array([x, y])
        return out + (np.asarray(noise, float) if noise is not None else 0.0)

    def put_cube(self, c, xy, gap=0.02, z_carry=0.15, v=0.15):
        sk = self.sk
        self.transit(xy, Z + z_carry, yaw=0.0, v=v, carried=c, margin=0.04)
        zr = Z + FLOOR_TOP + CUBE_E / 2 + gap
        self._go([xy[0], xy[1], zr + 0.015], speed=0.1)
        self._go([xy[0], xy[1], zr], speed=0.04, tol=0.003, max_steps=120)
        sk.set_gripper(False, steps=12)
        self._go([xy[0], xy[1], Z + z_carry], speed=v)
        self._settle(c)

    def close_drawer(self, v=0.04, z_push=0.05, overshoot=0.0, y_off=0.0, approach=0.03):
        """closed gripper in front of the drawer, then a straight push forward to where the front panel is flush"""
        sk = self.sk
        y = self.cab_xy()[1] + y_off
        xs = self.drawer_front_x() - FINGER_AHEAD - approach
        x_end = self.cab_front_x() - PF - FINGER_AHEAD + overshoot
        sk.set_gripper(True, steps=10)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], max(cur[2], Z + 0.15)], speed=0.15)
        self._go([xs, y, Z + 0.15], yaw=0.0, speed=0.15)
        self._go([xs, y, Z + z_push], speed=0.1, tol=0.003)
        self._go([x_end, y, Z + z_push], speed=v, tol=0.003, max_steps=int(0.25 / max(v, 0.01) / 0.05) + 60)
        self.sk.hold(10)
        cur = sk.eef_pos()
        self._go([cur[0] - 0.06, cur[1], cur[2]], speed=0.08)
        self._go([cur[0] - 0.06, cur[1], Z + 0.20], speed=0.15)
        sk.hold(10)

    def top_spots(self, mode="pair", noise=None):
        """world xy targets on the cabinet top for the two bottles (left one first): side by side across the middle"""
        c = self.from_cab([0.0, 0.0, CH])[:2]
        spots = [c + np.array([0.0, 0.05]), c + np.array([0.0, -0.05])]
        if noise is not None:
            spots = [s + np.asarray(n, float) for s, n in zip(spots, noise)]
        return spots

    def free_top_spot(self, b, start=None):
        """a spot on the cabinet top (cabinet frame candidates) at least 6.5 cm from the other bottle if that one stands
        on the top, and -- if `start` (xy the bottle comes from) is given -- whose straight path from `start` passes
        farthest from it"""
        other = [o for o in BOTTLES if o != b][0]
        cands = [np.array([dx, dy]) for dx in (0.0, -0.03, 0.03) for dy in (0.055, -0.055, 0.0)]
        ok_other = self.bottle_on_top(other, 0.01, np.radians(20))
        oc = self.to_cab(self.bottle_base(other))[:2]
        if not ok_other:
            return self.from_cab([cands[0][0], cands[0][1], CH])[:2] if start is None else min(
                (self.from_cab([c[0], c[1], CH])[:2] for c in cands[:2]),
                key=lambda w: float(np.linalg.norm(w - np.asarray(start)[:2])))
        best, score = None, -1.0
        st = None if start is None else self.to_cab(np.r_[np.asarray(start)[:2], Z])[:2]
        for c in cands:
            if np.linalg.norm(c - oc) < 0.065:
                continue
            if st is None:
                return self.from_cab([c[0], c[1], CH])[:2]
            ab = c - st
            u = float(np.clip((oc - st) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0))
            clear = float(np.linalg.norm(st + u * ab - oc))
            if clear > score:
                best, score = c, clear
        if best is None:
            best = max(cands, key=lambda c: float(np.linalg.norm(c - oc)))
        return self.from_cab([best[0], best[1], CH])[:2]

    def restand_bottle(self, b, xy=None):
        """stand a fallen bottle up again on the cabinet top: grasp the ball cap from above (fingers across the lying
        bottle's axis), lift until it hangs free with its bottom above the gallery rail, carry it over a free spot,
        lower until the bottom is ~3 mm above the top, let go. Returns True if it stands on the top."""
        sk = self.sk
        self._settle(b)
        for _ in range(2):
            if _reach(self.cap_xyz(b)) > REACH_MAX and self.tilt(b) > FALLEN:
                self.pull_in_bottle(b)
            k = self.cap_xyz(b)
            ax = self.R(b)[:2, 2]
            yaw = float(np.arctan2(ax[1], ax[0])) if np.linalg.norm(ax) > 0.2 else sk.yaw  # finger axis across it
            sk.set_gripper(False, steps=8)
            self.transit(k[:2], k[2] + 0.08, yaw=yaw, v=0.15)
            sk.hold(5)
            k = self.cap_xyz(b)
            self._go([k[0], k[1], k[2] + 0.015], speed=0.1)
            self._go([k[0], k[1], k[2]], speed=0.04, tol=0.003, max_steps=120)
            sk.set_gripper(True, steps=15)
            if not self.held(b):
                sk.set_gripper(False, steps=8)
                self._go([k[0], k[1], k[2] + 0.08], speed=0.1)
                continue
            zl = self.top_z() + RAIL_H + 0.03 + CAP_Z
            self._go([k[0], k[1], zl], speed=0.10)
            sk.hold(15)
            tx, ty = self.free_top_spot(b, start=k[:2]) if xy is None else xy
            self._go([tx, ty, zl], speed=0.10)
            sk.hold(30)
            self._go([tx, ty, self.top_z() + CAP_Z + 0.015], speed=0.08)
            self._go([tx, ty, self.top_z() + CAP_Z + 0.003], speed=0.03, tol=0.003, max_steps=120)
            sk.set_gripper(False, steps=12)
            self._go([tx, ty, self.top_z() + H_TOT + 0.04], speed=0.08)
            sk.hold(20)
            self._settle(b)
            if self.bottle_on_top(b):
                return True
        return self.bottle_on_top(b)

    def pull_in_bottle(self, b):
        """the lying bottle's cap is out of reach: grasp the body near its bottom end (fingers across it), lift it
        3 cm, pull it towards the robot until the cap is within reach, set it down"""
        sk = self.sk
        Rb = self.R(b)
        ax = Rb[:, 2]
        g = envs.obj_pos(self.env, b) - ax * (H_TOT / 2 - 0.04)
        zg = max(float(g[2] - Z), 0.02)
        yaw = float(np.arctan2(ax[1], ax[0]))
        need = _reach(self.cap_xyz(b)) - (REACH_MAX - 0.05)
        to_base = np.array([-0.66, 0.0]) - g[:2]
        to_base /= max(np.linalg.norm(to_base), 1e-6)
        dst = g[:2] + to_base * need
        sk.set_gripper(False, steps=8)
        self._go([g[0], g[1], Z + 0.15], yaw=yaw, speed=0.15)
        self._go([g[0], g[1], Z + zg + 0.015], speed=0.1)
        self._go([g[0], g[1], Z + zg], speed=0.04, tol=0.003, max_steps=120)
        sk.set_gripper(True, steps=15)
        self._go([g[0], g[1], Z + zg + 0.03], speed=0.08)
        self._go([dst[0], dst[1], Z + zg + 0.03], speed=0.08)
        self._go([dst[0], dst[1], Z + zg + 0.005], speed=0.04)
        sk.set_gripper(False, steps=12)
        self._go([dst[0], dst[1], Z + 0.15], speed=0.1)
        self._settle(b)

    # ------------------------------------------------------------ scripted policies
    def check_and_fix(self, params, rounds=2):
        """look at every stage at the end: stand fallen bottles up on the top, put stray cubes into the drawer if it is
        still open"""
        for _ in range(rounds):
            v = self.stage_values()
            if all(v.values()):
                break
            for b in BOTTLES:
                if not self.bottle_on_top(b):
                    if self.tilt(b) > UPRIGHT or self.bottle_base(b)[2] < self.top_z() - 0.03:
                        if self.tilt(b) > UPRIGHT:
                            self.restand_bottle(b)
                        elif self.pick_bottle(b, grasp="neck"):
                            self.put_bottle(b, self.free_top_spot(b), gap=0.005)
            if not self.drawer_closed():
                for k, c in enumerate(CUBES):
                    if not self.cube_in_drawer(c) and self.pick_cube(c):
                        self.put_cube(c, self.drawer_slot(k))
                if all(self.cube_in_drawer(c) for c in CUBES):
                    self.close_drawer(v=params.get("push_v", 0.04), z_push=params.get("push_z", 0.05))
        self._lift_to(self.top_z() + 0.15)
        self.sk.set_gripper(False, steps=8)
        self.sk.hold(10)

    def do_bottles(self, params, rng):
        noise = params.get("noise")
        spots = self.top_spots(noise=noise)
        side = 1.0 if self.inst["bottle_side"] == "left" else -1.0
        spots = sorted(spots, key=lambda q: side * float(q[1]))  # the spot farther from the bottles' side first
        order = list(BOTTLES) if params.get("bottle_order", "listed") == "listed" else list(BOTTLES)[::-1]
        for b, xy in zip(order, spots):
            for _ in range(2):
                if self.pick_bottle(b, grasp=params.get("grasp", "neck"), z_clear=params.get("z_clear", 0.05)):
                    self.put_bottle(b, xy, gap=params.get("gap", 0.02), z_clear=params.get("z_clear", 0.05))
                    break

    def do_cubes(self, params, rng):
        noise = params.get("cube_noise") or [None] * 3
        for k, c in enumerate(self._cube_order()):
            if self.pick_cube(c):
                self.put_cube(c, self.drawer_slot(k, noise[k]), gap=params.get("cube_gap", 0.02))

    def _cube_order(self):
        """nearest the drawer first"""
        d = np.array([self.cab_front_x() - Q_OPEN / 2, self.cab_xy()[1]])
        return sorted(CUBES, key=lambda c: float(np.linalg.norm(envs.obj_pos(self.env, c)[:2] - d)))

    def execute(self, inst, params):
        rng = np.random.default_rng(params.get("noise_seed", 0))
        order = params.get("order", "BCD")  # B = bottles on top, C = cubes into the drawer, D = close the drawer
        for step in order:
            if step == "B":
                self.do_bottles(params, rng)
            elif step == "C":
                self.do_cubes(params, rng)
            elif step == "D":
                self.close_drawer(v=params.get("push_v", 0.04), z_push=params.get("push_z", 0.05),
                                  overshoot=params.get("overshoot", 0.0))
        if params.get("check"):
            self.check_and_fix(params)
        else:
            self._lift_to(self.top_z() + 0.15)
            self.sk.set_gripper(False, steps=8)
            self.sk.hold(10)

    def _noise(self, rng, err=NAIVE_ERR):
        return dict(noise=[rng.uniform(-err, err, 2).tolist() for _ in BOTTLES],
                    cube_noise=[rng.uniform(-err, err, 2).tolist() for _ in CUBES])

    def default_params(self, inst):
        """Sonnet-like: the order of the instruction (bottles, cubes, drawer); bottles grasped by the body from above,
        lifted until they clear the cabinet top by 5 cm, lowered straight down and let go 1-4 cm above it; cubes into
        the drawer; the drawer pushed shut slowly (4 cm/s) with the closed gripper; no final look back"""
        rng = np.random.default_rng(inst["naive_noise_seed"])
        p = dict(order="BCD", check=False, grasp="neck", gap=float(rng.uniform(0.01, 0.04)),
                 push_v=float(rng.uniform(0.03, 0.05)), push_z=float(rng.uniform(0.04, 0.06)),
                 overshoot=float(rng.uniform(0.0, 0.004)), noise_seed=int(inst["naive_noise_seed"]))
        p.update(self._noise(rng))
        return p

    def oracle_params(self, inst):
        """knows the drawer: cubes in, drawer shut, then the bottles; checks every stage at the end"""
        return dict(order="CDB", check=True, grasp="neck", gap=0.008, push_v=0.04, push_z=0.05, overshoot=0.0)

    def adapt_params(self, inst, history):
        """attempt >= 2 (after the bottles fell when the drawer was shut): shut the drawer before the bottles go on
        top, and look at every stage at the end"""
        return dict(self.default_params(inst), order="CDB", check=True)

    def blind_params(self, inst, attempt, rng):
        """the natural behaviour with random variations (order of the three sub-tasks, push speed / height, release
        height, placement noise), no history, no final look"""
        order = str(rng.choice(["BCD", "CBD", "CDB"], p=[0.5, 0.3, 0.2]))
        p = dict(order=order, check=False, grasp=str(rng.choice(["neck", "cap"])), gap=float(rng.uniform(0.01, 0.04)),
                 push_v=float(rng.uniform(0.02, 0.08)), push_z=float(rng.uniform(0.03, 0.07)),
                 overshoot=float(rng.uniform(0.0, 0.006)), noise_seed=int(rng.integers(1 << 30)))
        p.update(self._noise(rng))
        return p

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """oracle / naive: one attempt; adaptive: attempt 1 = naive + a final look that stands fallen bottles up again,
        attempts >= 2 drawer first; blind: naive with random variations.
        Extra kinds (not in validate's table): switch (naive, then drawer first without a final look), oracle_noise
        (oracle with +-5 mm placement noise), naive_slow (naive pushing at the slowest agent speed, 2 cm/s),
        naive_check (naive + final look = adaptive's first attempt)"""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        single = ("oracle", "naive", "oracle_noise", "naive_slow", "naive_check", "naive_neck", "naive_cap")
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
            elif kind == "switch":
                params = self.default_params(inst) if a == 0 else dict(self.default_params(inst), order="CDB")
            elif kind == "oracle_noise":
                params = dict(self.oracle_params(inst), **self._noise(np.random.default_rng(seed * 7 + a), 0.005))
            elif kind == "naive_slow":
                params = dict(self.default_params(inst), push_v=0.02)
            elif kind == "naive_check":
                params = dict(self.default_params(inst), check=True)
            elif kind == "naive_neck":
                params = dict(self.default_params(inst), grasp="neck")
            elif kind == "naive_cap":
                params = dict(self.default_params(inst), grasp="cap")
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
