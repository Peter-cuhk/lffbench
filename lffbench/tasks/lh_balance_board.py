"""LH balance_board: put four look-alike cubes on their marks on a plank that rests loose across a narrow block (a
seesaw); loading it in the wrong order tips the plank, and every cube already on it slides down to the low end.

Family: long-horizon self-disturbance ("state_restoration"): each stage is a plain pick-and-place of a cube onto its
mark, but a later placement can tip the plank, which sends the cubes placed before it (earlier progress) sliding off
their marks. The agent has to notice it, re-level the plank and re-place them before finishing (within an attempt), and
learn a loading order that keeps the plank level (across attempts; the final-state judge does not enforce an order).

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * a grey support (fixture: two 1.5 cm blocks with a 1 cm slot between them, together 4 cm along the plank x 10 cm x
    6 cm high, flat tops) under the middle of
  * a dark wooden plank (free body, 46 cm long, pointing away from the robot, 8 cm wide, 1.5 cm thick, 0.5 kg) lying
    loose on it, level, with a 2 cm lip across each end. A thin keel under the plank's middle hangs in the slot and keeps
    the plank from wandering along the support (+-3 mm); nothing holds it down. It stays level while the combined centre
    of mass of plank + cubes is above the support's 4 cm top; otherwise it tips until its low end touches the table
    (about 17 degrees);
  * four marks (5.8 cm squares, outline + light fill) on the plank, two on each side of the block: an inner one
    (8.5-9.5 cm from the middle) and an outer one (15.5-16.5 cm); the colour of each mark is random;
  * four 4.5 cm cubes (red, green, blue, yellow) on the table, two left and two right of the plank (random places).
    They look alike except for the colour. Cube and plank surfaces are slick (friction 0.2 < tan 17 deg): on a tipped
    plank every cube slides down to the lip at the low end.
Why along x with lips (measured, see the task card): with the plank left-right and open ends, the cubes slid off the
ends onto the table 0-11 cm past them, and a default top-down grasp there (fingers left-right, the Panda hand is 21 cm
wide along them) pressed the plank's end down and tipped it again -- restoring became a loop. Along x the default grasp
is across the plank, and the lips keep the cubes on the plank within reach.

Hidden variable: the cube masses. "Hazard" instances (13 of every 20 consecutive seeds): the two cubes that belong on
the inner marks are heavy (stone, 0.19-0.24 kg), the two for the outer marks are light (48-56 g); the final layout is
balanced. A heavy cube alone on its inner mark tips the plank (1.28-1.43 x the tipping moment); a light one alone on its
outer mark does not (0.72-0.78 x); in the zig-zag order below no step exceeds 0.72-0.81 x. So the only safe loading
orders are "an outer cube, then the inner cube on the OTHER side, then the other inner cube, then the last outer cube"
(zig-zag); end-to-end, alternating outer-first (outer, outer, inner, inner) and inner-first orders all tip it.
"Benign" instances: all four cubes light (26-34 g); no order tips (worst step 0.63-0.75 x).

Why the natural order fails: nothing in the image shows which cubes are heavy. Any natural order (nearest cube first,
the marks from one end to the other, or the cautious "alternate sides, outer marks first") tips the plank in a hazard
instance; the cubes on it slide to the low end and keep it tipped. A natural plan then keeps going and finishes with
cubes off their marks. What must be learned: check the marks after each placement / before finishing, unload the low
end and re-place (within the attempt), and use the tip to infer which cubes are heavy (an inner cube tipped it; the
layout must balance in the end, so the other inner cube is heavy too) and load in the zig-zag order (across attempts).

Protocol: cross (reset to the same instance before every attempt), k = 3. Outcome: final-state judge (every cube on its
mark, plank level within 3 degrees and still on its support (10-11), nothing held, scene static) plus an event log of every stage predicate (achieved /
lost / restored, with step and cause) and every tip (which end went down, the cube released just before) for the server
log; F1 gives the agent only success / failure.
"""
import itertools

import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

TZ = envs.TABLE_Z

# ---------------------------------------------------------------- geometry / physics
# plank frame: y along the plank (+y = the end toward the robot), x across it, z up. The plank and the block are
# yawed 90 deg, so plank +y = world -x.
PL_HL, PL_HW, PL_HT = 0.23, 0.04, 0.0075  # plank half length, half width, half thickness
LIP_T, LIP_H = 0.008, 0.02  # lip across each end (inside the 46 cm), height above the plank's top
PLANK_MASS = 0.50
PLANK_ROT_DAMP = 0.01  # N m s / rad on the plank's rotation: after its low end is unloaded it swings over to ~10 deg
#                        and is calm in ~1.5 s (no damping: ~13 deg and ~2.7 s); the tip itself is not slowed (measured)
PLANK_YAW = np.pi / 2
RIDGE_HW, RIDGE_HX, RIDGE_H = 0.02, 0.05, 0.06  # support: half width along the plank (s), half depth, height
SLOT_HW = 0.005  # the support is two blocks with a 1 cm slot between them
KEEL_HW, KEEL_H = 0.002, 0.010  # keel under the plank's middle (4 mm x 8 cm x 1 cm) hanging in the slot
CUBE_E = 0.045
MU = 0.20  # cubes and plank: the cube-plank contact slides above 11 deg; ridge (1.0), table (0.6), finger pads (2.0) win
MU_RIDGE = 1.0
MARK_HALF, MARK_LINE = 0.029, 0.004
WOOD = (0.42, 0.27, 0.15, 1.0)  # dark walnut: contrasts with the light table, so the tilt reads in the images

COLORS = ("red", "green", "blue", "yellow")
RGBA = dict(red=(0.85, 0.12, 0.10, 1.0), green=(0.10, 0.62, 0.20, 1.0), blue=(0.15, 0.30, 0.88, 1.0),
            yellow=(0.93, 0.78, 0.10, 1.0))
CUBES = {c: f"{c}_cube_1" for c in COLORS}
PLANK, RIDGE = "plank_1", "ridge_1"
PLANK_MODEL = "lff_lh_bb_plank"
# mark slots in plank-frame order (+y first): the robot's end outer / inner, then the far end inner / outer
SLOTS = ("near_outer", "near_inner", "far_inner", "far_outer")
END_WORD = dict(near="the end nearer the robot", far="the far end")

# ---------------------------------------------------------------- instances / hidden variable
D_INNER = (0.085, 0.095)  # |y| of the inner marks from the plank's middle
D_OUTER = (0.155, 0.165)  # the lip stops a cube 19.95 cm out: a cube on a low-side outer mark ends >= 3.5 cm off it
P_HAZARD = 0.65  # fraction of hazard instances, balanced per 20 consecutive seeds (13 / 7)
R_HEAVY = (1.30, 1.40)  # hazard: lone heavy cube on its inner mark, moment / tipping moment
R_LIGHT = (0.72, 0.78)  # hazard: lone light cube on its outer mark
LIGHT_BENIGN = (0.026, 0.034)  # benign: every cube (kg)
H_EST, L_EST = 0.21, 0.04  # nominal masses the adaptive script uses for its hypotheses (it never sees the true ones)

# ---------------------------------------------------------------- judging
TOL = 0.020  # cube centre within 2 cm of its mark centre (per plank axis)
TOL_LOSE = 0.025  # hysteresis for the monitor
LEVEL_DEG = 3.0
TIP_DEG = 8.0  # the monitor records a tip when |tilt| exceeds this (a real tip reaches ~17 deg)
LEVEL_DEBOUNCE = 4  # monitor evaluations (0.1 s each) before plank_level changes
FLAT_COS = np.cos(np.radians(15.0))
SUPPORT_XY = 0.02  # plank on its support: its middle within 2 cm (horizontally) of the support's centre ...
SUPPORT_DZ = 0.02  # ... and not more than 2 cm below its starting height (a tip lowers it ~0.6 cm; off the block ~6 cm)

# ---------------------------------------------------------------- scripted motion
Z_CARRY = 0.17  # careful / adaptive: grasp point above the table while carrying (cube bottom 14.75 cm)
Z_CARRY_NAIVE = 0.15  # naive: just above the cubes already on the plank (their tops at 12 cm)
REL_GAP = 0.004  # release with the cube's bottom this far above the plank


def plank_xml(name):
    """Board + end lips + keel (collision) and per colour a mark (5 visual-only geoms, moved per instance).
    Long axis = body y."""
    g = []

    def box(nm, pos, size, rgba):
        g.append(f'<geom name="{nm}" type="box" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(rgba)}" '
                 f'friction="{MU} 0.005 0.0001" density="900" group="0"/>')
        g.append(f'<geom type="box" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(rgba)}" contype="0" '
                 f'conaffinity="0" group="1" mass="0"/>')

    dark = (0.25, 0.16, 0.09, 1.0)
    box(f"{name}_board", (0, 0, 0), (PL_HW, PL_HL, PL_HT), WOOD)
    for k, s in enumerate((1, -1)):
        box(f"{name}_lip{k}", (0, s * (PL_HL - LIP_T / 2), PL_HT + LIP_H / 2), (PL_HW, LIP_T / 2, LIP_H / 2), dark)
    box(f"{name}_keel", (0, 0, -PL_HT - KEEL_H / 2), (PL_HW, KEEL_HW, KEEL_H / 2), dark)
    for c in COLORS:
        rgba = RGBA[c]
        fill = tuple(0.55 + 0.45 * v for v in rgba[:3]) + (1.0,)
        o, i, w = MARK_HALF, MARK_HALF - MARK_LINE, MARK_LINE
        parts = [((0, 0, PL_HT + 0.0003), (o, o, 0.0003), fill),
                 ((0, i + w / 2, PL_HT + 0.0008), (o, w / 2, 0.0003), rgba),
                 ((0, -i - w / 2, PL_HT + 0.0008), (o, w / 2, 0.0003), rgba),
                 ((i + w / 2, 0, PL_HT + 0.0008), (w / 2, o, 0.0003), rgba),
                 ((-i - w / 2, 0, PL_HT + 0.0008), (w / 2, o, 0.0003), rgba)]
        for k, (p, sz, col) in enumerate(parts):
            g.append(f'<geom name="{name}_mk_{c}_{k}" type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" '
                     f'rgba="{_fmt(col)}" contype="0" conaffinity="0" group="1" mass="0"/>')
    return _wrap(name, "\n        ".join(g), PL_HT, float(np.hypot(PL_HL, PL_HW)))


def ridge_xml(name):
    """Split support: two 1.5 cm blocks with a 1 cm slot between them; the plank rests on both and pivots on their outer
    top edges (a 4 cm flat top). The plank's keel hangs in the slot, so the plank cannot wander along its length by
    more than +-3 mm (it stays engaged at the full tip angle)."""
    rgba = (0.35, 0.35, 0.38, 1.0)
    bh = (RIDGE_HW - SLOT_HW) / 2
    g = []
    for k, sy in enumerate((1, -1)):
        p, sz = (0, sy * (SLOT_HW + bh), RIDGE_H / 2), (RIDGE_HX, bh, RIDGE_H / 2)
        g.append(f'<geom name="{name}_g{k}" type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(rgba)}" '
                 f'friction="{MU_RIDGE} 0.005 0.0001" group="0"/>')
        g.append(f'<geom type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(rgba)}" contype="0" conaffinity="0" '
                 f'group="1"/>')
    g = "\n        ".join(g)
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        {g}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {RIDGE_H:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{RIDGE_HX:.4f} {RIDGE_HW:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>"""


def tip_ratio(items, plank_mass=PLANK_MASS, s=RIDGE_HW):
    """Static model of a loose plank on a flat-topped block: |sum m y| / ((M_plank + sum m) s); the plank tips when
    this exceeds 1 (the combined centre of mass leaves the block's top). items: [(mass, y from the middle)]."""
    if not items:
        return 0.0
    mom = sum(m * y for m, y in items)
    return abs(mom) / ((plank_mass + sum(m for m, _ in items)) * s)


def _side(y):
    return "near" if y > 0 else "far"


def _along(v):
    """plank-frame y offset in robot-centric words (+y = toward the robot)"""
    return f"{100 * abs(v):.1f} cm from its mark toward {'the robot' if v > 0 else 'the far end'}"


class BoardSkills(Skills):
    """Skills that feed the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class BalanceBoard(LFFTask):
    name = "lh_balance_board"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Plan", "Restore")
    protocol = "cross"
    max_attempts = 3
    agent_max_steps = 120  # careful zig-zag ~30-36 commands + re-placing four cubes (~30) + looking around
    instruction = ("Put each coloured cube on the square mark of the same colour on the long plank. The plank lies loose "
                   "across a narrow block under its middle. At the end all four cubes must be on their marks and the "
                   "plank must still rest level on the block.")
    instruction_indirect = ("Set the four coloured cubes onto the outlined squares of matching colour on the wooden "
                            "board that rests across the small grey block. When you are done, every cube must sit on "
                            "its square and the board must still rest level on the block.")
    agent_object_names = ["red cube", "green cube", "blue cube", "yellow cube", "plank", "support block",
                          "red mark", "green mark", "blue mark", "yellow mark"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        pl = register_generated("LffLhBbPlank", plank_xml(PLANK_MODEL))
        rg = register_generated("LffLhBbRidge", ridge_xml("lff_lh_bb_ridge"), free=False)
        objs = [(PLANK, pl, "plank_region", (0.18, -0.01, 0.20, 0.01))]
        e = CUBE_E / 2
        for i, c in enumerate(COLORS):
            cat = register_generated(f"LffLhBbCube{c.title()}",
                                     box_xml(f"lff_lh_bb_cube_{c}", (e, e, e), RGBA[c], density=500.0,
                                             friction=(MU, 0.005, 0.0001)).replace(
                                         'solimp="0.998 0.998 0.001" solref="0.001 1"', ""))
            objs.append((CUBES[c], cat, f"{c}_cube_region", (-0.32, -0.33 + 0.2 * i, -0.30, -0.31 + 0.2 * i)))
        fx = [(RIDGE, rg, "ridge_region", (-0.12, 0.30, -0.10, 0.32))]
        return write_bddl(self.name, "put each cube on its mark on the plank and keep the plank level", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.ridge_bid = m.body_name2id(self.env.fixtures_dict[RIDGE].root_body)
        self.plank_bid = envs.body_id(self.env, PLANK)
        pre = f"{PLANK}_{PLANK_MODEL}"
        self.mark_gids = {c: [m.geom_name2id(f"{pre}_mk_{c}_{k}") for k in range(5)] for c in COLORS}
        self.mark_pos0 = {c: [m.geom_pos[g].copy() for g in self.mark_gids[c]] for c in COLORS}
        self.cube_gids = {c: set(envs.obj_geom_ids(self.env, CUBES[c])) for c in COLORS}
        self.plank_gids = set(envs.obj_geom_ids(self.env, PLANK))
        self.robot_gids = set(envs.robot_geom_ids(self.env))
        self.inst = None
        self.mon = None
        self.obs = []  # adaptive script: (configuration, tipped) observations, kept across the attempts of one run

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        x0, y0 = float(rng.uniform(-0.18, -0.14)), float(rng.uniform(-0.03, 0.03))
        perm = [str(c) for c in rng.permutation(COLORS)]  # colour of the mark in SLOTS order
        slot_of = {c: s for c, s in zip(perm, SLOTS)}
        dist = dict(near_outer=float(rng.uniform(*D_OUTER)), near_inner=float(rng.uniform(*D_INNER)),
                    far_inner=-float(rng.uniform(*D_INNER)), far_outer=-float(rng.uniform(*D_OUTER)))
        marks = {c: dist[slot_of[c]] for c in COLORS}  # plank-frame y of each mark (+ = toward the robot)
        hazard = bool((int(seed) * 7) % 20 < round(20 * P_HAZARD))  # balanced like l7_key_turn / lh_lid_box
        r_h, r_l = float(rng.uniform(*R_HEAVY)), float(rng.uniform(*R_LIGHT))
        light = {c: float(rng.uniform(*LIGHT_BENIGN)) for c in COLORS}
        s, pm = RIDGE_HW, PLANK_MASS
        if hazard:
            # equal moments on both sides -> the finished layout is exactly balanced
            dn, do = dist["near_inner"], dist["near_outer"]
            y_h = r_h * pm * s / (dn - r_h * s) * dn  # heavy on the near inner mark: moment r_h x tipping moment
            x_l = r_l * pm * s / (do - r_l * s) * do
            mass = {c: (y_h if "inner" in slot_of[c] else x_l) / abs(dist[slot_of[c]]) for c in COLORS}
        else:
            mass = light
        # start: two cubes left and two right of the plank, 17-19 cm from its axis, at two depths (random places)
        spots = [(dx, sy) for sy in (1.0, -1.0) for dx in (-0.08, 0.08)]
        order = rng.permutation(4)
        start = {c: [x0 + spots[j][0] + float(rng.uniform(-0.01, 0.01)),
                     y0 + spots[j][1] * float(rng.uniform(0.17, 0.19))] for c, j in zip(COLORS, order)}
        start_yaw = {c: float(rng.uniform(-0.3, 0.3)) for c in COLORS}
        heavy = sorted(c for c in COLORS if mass[c] > 0.1)
        return dict(seed=int(seed), plank_xy=[x0, y0], marks=marks, slot_of=slot_of, hazard=hazard, mass=mass,
                    heavy=heavy, r_heavy=r_h if hazard else None, r_light=r_l if hazard else None, start=start,
                    start_yaw=start_yaw)

    def apply_instance(self, inst):
        self.inst = inst
        env = self.env
        m = env.sim.model
        x0, y0 = inst["plank_xy"]
        q = [np.cos(PLANK_YAW / 2), 0.0, 0.0, np.sin(PLANK_YAW / 2)]
        m.body_pos[self.ridge_bid] = [x0, y0, TZ]
        m.body_quat[self.ridge_bid] = q
        envs.scale_mass(env, PLANK, PLANK_MASS)
        _, da = envs.free_joint_addr(env, PLANK)
        m.dof_damping[da + 3:da + 6] = PLANK_ROT_DAMP
        for c in COLORS:
            for g, p0 in zip(self.mark_gids[c], self.mark_pos0[c]):
                m.geom_pos[g] = p0 + np.array([0.0, inst["marks"][c], 0.0])
            envs.scale_mass(env, CUBES[c], inst["mass"][c])
        envs.set_obj_pose(env, PLANK, [x0, y0, TZ + RIDGE_H + PL_HT + 0.0005], quat_wxyz=np.array(q))
        for c in COLORS:
            envs.place_upright(env, CUBES[c], inst["start"][c], yaw=inst["start_yaw"][c])

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.plank_z0 = float(self.plank_pos()[2])
        self.sk = BoardSkills(self.env, self, recorder=recorder)
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ state helpers
    def plank_R(self):
        return self.env.sim.data.body_xmat[self.plank_bid].reshape(3, 3).copy()

    def plank_pos(self):
        return self.env.sim.data.body_xpos[self.plank_bid].copy()

    def tilt_deg(self):
        """+ = the plank's end toward the robot is up"""
        return float(np.degrees(np.arcsin(np.clip(self.plank_R()[2, 1], -1.0, 1.0))))

    def is_level(self, deg=LEVEL_DEG):
        return abs(self.tilt_deg()) <= deg

    def on_support(self):
        """the plank still rests on the support block (10-11: Sonnet lh2 s3001 pushed it off onto the table, where it
        lies level, and loaded the cubes there; 'level' alone accepted that)"""
        p, r = self.plank_pos(), self.env.sim.data.body_xpos[self.ridge_bid]
        return bool(np.linalg.norm(p[:2] - r[:2]) <= SUPPORT_XY and p[2] >= self.plank_z0 - SUPPORT_DZ)

    def in_plank(self, p):
        """world point -> plank frame (x across, y along from the middle, z above the plank's mid-plane)"""
        return self.plank_R().T @ (np.asarray(p, float) - self.plank_pos())

    def mark_world(self, c):
        return self.plank_pos() + self.plank_R() @ np.array([0.0, self.inst["marks"][c], PL_HT])

    def cube_pos(self, c):
        return envs.obj_pos(self.env, CUBES[c])

    def held(self, c):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper,
                                          object_geoms=self.env.objects_dict[CUBES[c]]))

    def on_plank(self, c):
        """resting on the plank's top (plank frame), anywhere along it"""
        q = self.in_plank(self.cube_pos(c))
        return bool(abs(q[0]) <= PL_HW + 0.01 and abs(q[1]) <= PL_HL and abs(q[2] - PL_HT - CUBE_E / 2) <= 0.012)

    def mark_offset(self, c):
        q = self.in_plank(self.cube_pos(c))
        return np.array([q[0], q[1] - self.inst["marks"][c]])

    def _flat(self, c):
        Rc = self.env.sim.data.body_xmat[envs.body_id(self.env, CUBES[c])].reshape(3, 3)
        return bool(np.max(np.abs(self.plank_R()[:, 2] @ Rc)) >= FLAT_COS)

    def on_mark(self, c, tol=TOL):
        if not self.on_plank(c):
            return False
        o = self.mark_offset(c)
        return bool(np.max(np.abs(o)) <= tol and self._flat(c) and not self.held(c))

    def stage_values(self):
        v = {f"{c}_on_mark": self.on_mark(c) for c in COLORS}
        v["plank_level"] = self.is_level()
        v["plank_on_support"] = self.on_support()
        return v

    # ------------------------------------------------------------ per-step monitor (server log only)
    def _init_monitor(self):
        self.mon = dict(state={f"{c}_on_mark": False for c in COLORS} | {"plank_level": True, "plank_on_support": True}, ever=set(),
                        lost=set(), events=[], step=0, last_touch={c: None for c in COLORS},
                        held_prev={c: False for c in COLORS}, last_release=None, tipping=False, tips=[],
                        level_cnt=0, calm=0, tilt_max=0.0, win_max=0.0)

    def _robot_touches(self):
        """cubes touched by the robot or by a cube the robot is carrying, in this control step"""
        d = self.env.sim.data
        out = set()
        carried = [c for c in COLORS if self.mon["held_prev"][c]]
        hit = {c: self.robot_gids.union(*[self.cube_gids[h] for h in carried if h != c]) for c in COLORS}
        for i in range(d.ncon):
            con = d.contact[i]
            for c in COLORS:
                g = self.cube_gids[c]
                if (con.geom1 in g and con.geom2 in hit[c]) or (con.geom2 in g and con.geom1 in hit[c]):
                    out.add(c)
        return out

    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        mon["step"] += 1
        step = self.sk.n_steps
        for c in self._robot_touches():
            mon["last_touch"][c] = step
        tilt = self.tilt_deg()
        mon["tilt_max"] = max(mon["tilt_max"], abs(tilt))
        mon["win_max"] = max(mon["win_max"], abs(tilt))
        if not mon["tipping"] and abs(tilt) > TIP_DEG:
            # one tip episode = from |tilt| > 8 deg until the plank has stayed level for 1 s (the swing back and any
            # rocking belong to the same episode)
            mon["tipping"] = True
            mon["calm"] = 0
            rel = mon["last_release"]
            tip = dict(step=step, down="near" if tilt < 0 else "far",
                       after_release=(rel[0] if rel is not None and step - rel[1] <= 60 else None),
                       on_plank=[c for c in COLORS if self.on_plank(c)],
                       on_mark=[c for c in COLORS if mon["state"][f"{c}_on_mark"]])
            mon["tips"].append(tip)
            mon["events"].append(dict(step=step, pred="plank", ev="tipped", down=tip["down"],
                                      after_release=tip["after_release"], on_mark=tip["on_mark"]))
        elif mon["tipping"]:
            mon["calm"] = mon["calm"] + 1 if abs(tilt) < LEVEL_DEG else 0
            if mon["calm"] >= 20:
                mon["tipping"] = False
        if mon["step"] % 2:
            return  # predicates every other control step (0.1 s)
        for c in COLORS:
            h = self.held(c)
            if mon["held_prev"][c] and not h:
                mon["last_release"] = (c, step)
            mon["held_prev"][c] = h
            key = f"{c}_on_mark"
            was = mon["state"][key]
            now = self.on_mark(c, TOL if not was else TOL_LOSE)
            if now != was:
                self._event(key, now, step, c)
        lv = self.is_level()
        if lv != mon["state"]["plank_level"]:
            mon["level_cnt"] += 1
            if mon["level_cnt"] >= LEVEL_DEBOUNCE:
                mon["level_cnt"] = 0
                self._event("plank_level", lv, step, None)
        else:
            mon["level_cnt"] = 0
        sup = self.on_support()
        if sup != mon["state"]["plank_on_support"]:
            self._event("plank_on_support", sup, step, None)

    def _event(self, key, now, step, color):
        mon = self.mon
        mon["state"][key] = now
        if now:
            ev = "restored" if key in mon["lost"] else "achieved"
            mon["ever"].add(key)
            mon["lost"].discard(key)
            mon["events"].append(dict(step=step, pred=key, ev=ev))
            return
        cause = "other"
        recent_tip = bool(mon["tips"]) and (mon["tipping"] or step - mon["tips"][-1]["step"] <= 60)
        if color is None:
            cause = "tip" if recent_tip else "other"
        elif self.held(color):
            cause = "picked_up"
        elif recent_tip:
            cause = "tip"
        elif mon["last_touch"][color] is not None and step - mon["last_touch"][color] <= 60:
            cause = "robot"
        mon["lost"].add(key)
        mon["events"].append(dict(step=step, pred=key, ev="lost", cause=cause, tilt=round(self.tilt_deg(), 1)))

    # ------------------------------------------------------------ outcome
    def _static(self):
        d = self.env.sim.data
        for name in [CUBES[c] for c in COLORS] + [PLANK]:
            _, da = envs.free_joint_addr(self.env, name)
            if np.linalg.norm(d.qvel[da:da + 3]) > 0.02 or np.linalg.norm(d.qvel[da + 3:da + 6]) > 0.3:
                return False
        return True

    def _where(self, c):
        """robot-centric description of where a cube that is not on its mark is"""
        if self.held(c):
            return "is still held by the gripper"
        if self.on_plank(c):
            o = self.mark_offset(c)
            return f"lies on the plank, {_along(o[1])}"
        p = self.cube_pos(c)
        if np.linalg.norm(p[:2] - np.asarray(self.inst["start"][c])) < 0.02:
            return "is still where it started on the table"
        q = self.in_plank(p)
        if abs(q[1]) > PL_HL:
            return f"lies on the table beyond {END_WORD[_side(q[1])]} of the plank"
        return "lies on the table beside the plank"

    def outcome(self, inst):
        envs.settle(self.env, 30)
        mon = self.mon
        vals = self.stage_values()
        static = self._static()
        held = [c for c in COLORS if self.held(c)]
        success = bool(all(vals.values()) and static and not held)
        regress = [e for e in mon["events"] if e["ev"] == "lost"]
        restored = [e for e in mon["events"] if e["ev"] == "restored"]
        parts = []
        off = [c for c in COLORS if not vals[f"{c}_on_mark"]]
        if success:
            failure = None
            parts.append("All four cubes are on their marks and the plank is level.")
        else:
            lost_final = [c for c in off if f"{c}_on_mark" in mon["ever"]]
            failure = ("held" if held else "plank_off_support" if not vals["plank_on_support"] else
                       "regressed" if lost_final else "incomplete" if off else
                       "not_level" if not vals["plank_level"] else "not_static")
            for c in off:
                txt = f"The {c} cube is not on its mark: it {self._where(c)}"
                if f"{c}_on_mark" in mon["ever"]:
                    why = [e for e in regress if e["pred"] == f"{c}_on_mark"]
                    cause = why[-1].get("cause") if why else None
                    txt += {"tip": " (it had been on its mark; it slid off when the plank tipped).",
                            "robot": " (it had been on its mark; the robot knocked it off).",
                            "picked_up": " (it had been on its mark; it was picked up again)."}.get(
                        cause, " (it had been on its mark earlier).")
                else:
                    txt += " (it was never on its mark)."
                parts.append(txt)
            if not vals["plank_on_support"]:
                parts.append("The plank no longer rests on the support block: it was pushed or knocked off it.")
            if not vals["plank_level"]:
                t = self.tilt_deg()
                parts.append(f"The plank is not level: it is tilted {abs(t):.0f} degrees, "
                             f"{END_WORD['near' if t < 0 else 'far']} down.")
            if not static and all(vals.values()):
                parts.append("Something was still moving at the end.")
        if mon["tips"]:
            descr = []
            for tp in mon["tips"]:
                d = f"{END_WORD[tp['down']]} down"
                if tp["after_release"]:
                    d += f" right after the {tp['after_release']} cube was set down"
                descr.append(d)
            parts.append(f"During the attempt the plank tipped {len(mon['tips'])} time(s): " + "; ".join(descr) + ".")
        return dict(success=success, failure=failure, stages=vals, static=bool(static), held=held,
                    events=mon["events"], n_regressions=len(regress), n_restored=len(restored),
                    n_tips=len(mon["tips"]), tips=mon["tips"], tilt_max=round(mon["tilt_max"], 1),
                    tilt_final=round(self.tilt_deg(), 1), hazard=bool(inst["hazard"]),
                    mark_offsets_cm={c: (100 * self.mark_offset(c)).round(1).tolist() for c in COLORS},
                    on_plank={c: self.on_plank(c) for c in COLORS}, detail=" ".join(parts))

    # ------------------------------------------------------------ scripted policies (perfect perception)
    def _grasp_yaw(self, c):
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, CUBES[c])].reshape(3, 3)
        i = int(np.argmin(np.abs(R[2])))
        yaw = float(np.arctan2(R[1, i], R[0, i]))
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def _y_on_plank(self):
        """cubes resting on the plank now -> their position along it"""
        return {c: float(self.in_plank(self.cube_pos(c))[1]) for c in COLORS if self.on_plank(c) and not self.held(c)}

    def plank_calm(self):
        _, da = envs.free_joint_addr(self.env, PLANK)
        return abs(self.tilt_deg()) < 1.0 and float(np.linalg.norm(self.env.sim.data.qvel[da + 3:da + 6])) < 0.05

    def wait_calm(self, max_steps=60):
        """hold the arm still until the plank has stopped swinging (a careful agent looks before it acts)"""
        n = 0
        for _ in range(max_steps):
            if self.plank_calm():
                n += 1
                if n >= 5:
                    return True
            else:
                n = 0
            self.sk.hold(1)
        return False

    def pick(self, c, z_carry):
        sk = self.sk
        p = self.cube_pos(c)
        yaw = self._grasp_yaw(c)
        sk.set_gripper(False, steps=8)
        sk.move_to([p[0], p[1], max(TZ + z_carry, p[2] + 0.06)], yaw=yaw)
        sk.move_to([p[0], p[1], p[2]], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([p[0], p[1], TZ + z_carry], speed=0.3)
        if not self.held(c):
            sk.set_gripper(False, steps=8)
            sk.move_to([p[0], p[1], TZ + z_carry])
            return False
        return True

    def put(self, c, z_carry=Z_CARRY, noise=None, record=False, careful=False):
        """Pick cube c wherever it is and put it on its mark: carry it over the mark, lower it until its bottom is
        REL_GAP above the plank, release, lift. Returns whether the plank tipped (|tilt| > TIP_DEG) during the
        placement, or None if the cube could not be picked. record=True appends the observation (cubes on the plank
        right before the release + this one on its mark, tipped?) for the adaptive script."""
        for _ in range(2):
            if self.pick(c, z_carry):
                break
        else:
            return None
        sk = self.sk
        d = self.mark_world(c)
        if noise is not None:
            d = d + np.array([noise[0], noise[1], 0.0])
        sk.move_to([d[0], d[1], TZ + z_carry], yaw=0.0)
        if careful:
            self.wait_calm()
            d = self.mark_world(c) + (np.array([noise[0], noise[1], 0.0]) if noise is not None else 0.0)
        sk.move_to([d[0], d[1], d[2] + CUBE_E / 2 + REL_GAP], tol=0.003)
        was_level = self.plank_calm()
        conf = self._y_on_plank()
        conf[c] = float(self.inst["marks"][c])
        self.mon["win_max"] = abs(self.tilt_deg())
        sk.set_gripper(False, steps=12)
        sk.move_to([d[0], d[1], TZ + z_carry])
        sk.hold(12)
        tipped = self.mon["win_max"] > TIP_DEG
        if tipped:
            envs.settle(self.env, 30)  # let the cubes slide down
        if record and was_level:
            self.obs.append(dict(conf=conf, tipped=bool(tipped), after=c))
        return tipped

    def park(self, c):
        """take cube c off the plank and set it on a free spot of the table beside the plank"""
        x0, y0 = self.inst["plank_xy"]
        others = [self.cube_pos(o)[:2] for o in COLORS if o != c]
        cands = [np.array([x0 + dx, y0 + sy * dy]) for sy in (1, -1) for dy in (0.18, 0.25)
                 for dx in (-0.12, -0.04, 0.04, 0.12)]
        best = max(cands, key=lambda q: min(np.linalg.norm(q - o) for o in others))
        if not self.pick(c, Z_CARRY):
            return False
        sk = self.sk
        sk.move_to([best[0], best[1], TZ + Z_CARRY], yaw=0.0)
        sk.move_to([best[0], best[1], TZ + CUBE_E / 2 + 0.004], tol=0.003)
        sk.set_gripper(False, steps=12)
        sk.move_to([best[0], best[1], TZ + Z_CARRY])
        return True

    # ---- planning with the static model
    def plan(self, todo, hyps):
        """Best order to (re)place the cubes in `todo`, given the cubes now on the plank, minimising the worst tip ratio
        over the mass hypotheses `hyps` (list of {colour: kg}). A cube that is on the plank off its mark is lifted
        (its moment removed) and then set on its mark. Returns (order, worst ratio)."""
        cur0 = self._y_on_plank()
        best, best_w = list(todo), np.inf
        for order in itertools.permutations(todo):
            w = 0.0
            for mass in hyps:
                cur = dict(cur0)
                for c in order:
                    if c in cur:
                        del cur[c]
                        w = max(w, tip_ratio([(mass[o], y) for o, y in cur.items()]))
                    cur[c] = self.inst["marks"][c]
                    w = max(w, tip_ratio([(mass[o], y) for o, y in cur.items()]))
            if w < best_w - 1e-9:
                best, best_w = list(order), w
        return best, best_w

    def hypotheses(self, use_obs=True):
        """Heavy / light assignments (nominal masses) that (1) leave the finished layout balanced, (2) admit a loading
        order that never tips the plank (the task is solvable) and (3) agree with every observed placement (tipped or
        not, with a +-10 % band in which either outcome counts as consistent). Falls back to (1)+(2) if no assignment
        agrees with all observations."""
        marks = self.inst["marks"]
        prior = []
        for k in range(5):
            for heavy in itertools.combinations(COLORS, k):
                mass = {c: (H_EST if c in heavy else L_EST) for c in COLORS}
                if tip_ratio([(mass[c], marks[c]) for c in COLORS]) > 0.6:
                    continue
                for order in itertools.permutations(COLORS):
                    cur, w = [], 0.0
                    for c in order:
                        cur.append((mass[c], marks[c]))
                        w = max(w, tip_ratio(cur))
                    if w <= 0.9:
                        prior.append(mass)
                        break
        if not use_obs:
            return prior
        post = []
        for mass in prior:
            if all(not ((ob["tipped"] and r < 0.9) or (not ob["tipped"] and r > 1.1))
                   for ob in self.obs for r in [tip_ratio([(mass[o], y) for o, y in ob["conf"].items()])]):
                post.append(mass)
        return post or prior

    def unload_until_level(self, max_moves=4):
        """while the plank is tipped: take the cube farthest out on its low side off the plank"""
        for _ in range(max_moves):
            envs.settle(self.env, 20)
            self.wait_calm()
            if self.is_level():
                return True
            t = self.tilt_deg()
            yo = self._y_on_plank()
            low = [c for c, y in yo.items() if (y > 0) == (t < 0)]
            if not low:
                return self.is_level()
            self.park(min(low, key=lambda o: abs(yo[o])))  # the top of the pile first: the hand clears the others
        envs.settle(self.env, 20)
        self.wait_calm()
        return self.is_level()

    def restore(self, hyps_fn, rounds=3, record=False):
        """Check every stage; if something is wrong: unload the low end if the plank is tipped, then (re)place every
        cube that is not on its mark in the order the static model rates safest under the current mass hypotheses;
        start over after a tip. Up to `rounds` passes."""
        for _ in range(rounds):
            envs.settle(self.env, 20)
            if all(self.stage_values().values()):
                return True
            self.unload_until_level()
            todo = [c for c in COLORS if not self.on_mark(c)]
            if not todo:
                continue
            hyps = hyps_fn()
            order, worst = self.plan(todo, hyps)
            if worst > 0.9:  # cubes left on the plank make every order risky: clear it and start from empty
                for c in [c for c in COLORS if self.on_plank(c)]:
                    self.park(c)
                order, worst = self.plan(list(COLORS), hyps)
            for c in order:
                if self.on_mark(c):
                    continue
                if self.put(c, Z_CARRY, record=record, careful=True):
                    break  # tipped: look again
        envs.settle(self.env, 20)
        return all(self.stage_values().values())

    def naive_order(self, noise_sd=0.0, rng=None):
        """nearest cube first, starting from the gripper's position and then from each mark just loaded"""
        p = self.sk.eef_pos()[:2]
        left, order = list(COLORS), []
        while left:
            dist = {c: np.linalg.norm(self.cube_pos(c)[:2] - p) + (rng.normal(0, noise_sd) if rng is not None else 0.0)
                    for c in left}
            c = min(left, key=dist.get)
            order.append(c)
            left.remove(c)
            p = self.mark_world(c)[:2]
        return order

    def slot_order(self, kind):
        so = {s: c for c, s in self.inst["slot_of"].items()}
        seq = dict(end2end=SLOTS, alt_outer=("near_outer", "far_outer", "near_inner", "far_inner"),
                   alt_inner=("near_inner", "far_inner", "near_outer", "far_outer"))[kind]
        return [so[s] for s in seq]

    def execute(self, inst, params):
        rng = np.random.default_rng(params.get("seed", 0))
        mode = params["mode"]
        if mode == "plan":
            hyps = [dict(inst["mass"])] if params.get("true_masses") else self.hypotheses()
            order, _ = self.plan(list(COLORS), hyps)
        elif mode in ("end2end", "alt_outer", "alt_inner"):
            order = self.slot_order(mode)
        else:  # nearest first (naive / blind)
            order = self.naive_order(params.get("order_noise", 0.0), rng)
        noise = params.get("noise", 0.0)
        z_carry = Z_CARRY if params.get("careful") else Z_CARRY_NAIVE
        for c in order:
            nz = rng.uniform(-noise, noise, 2) if noise else None
            tipped = self.put(c, z_carry, noise=nz, record=params.get("record", False), careful=params.get("careful", False))
            if tipped and params.get("check_each"):
                break  # look at what happened, then restore below
        if params.get("check"):
            hyps_fn = (lambda: [dict(inst["mass"])]) if params.get("true_masses") else self.hypotheses
            self.restore(hyps_fn, record=params.get("record", False))
        self.sk.hold(10)
        return order

    def default_params(self, inst):
        return dict(mode="nearest", check=False)

    def oracle_params(self, inst):
        return dict(mode="plan", true_masses=True, careful=True, check=True)

    def adapt_params(self, inst, history):
        """from attempt 2: load in the order the observations of earlier attempts make safest, check after every
        placement and restore at the end"""
        return dict(mode="plan", careful=True, check=True, check_each=True, record=True)

    def blind_params(self, inst, attempt, rng):
        """nearest-first again with small random variation (order noise, 4 mm placement noise), no history, no check"""
        return dict(mode="nearest", check=False, order_noise=0.03, noise=0.004, seed=int(rng.integers(1 << 30)))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        self.obs = []
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":  # attempt 1: natural order, but look after every placement and restore
                params = (dict(mode="nearest", check=True, check_each=True, record=True) if a == 0 else
                          self.adapt_params(inst, history))
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind in ("end2end", "alt_outer", "alt_inner"):  # extra controls: other natural orders, no check
                params = dict(mode=kind, check=False)
            elif kind == "oracle_noise":  # extra control: the oracle with +-5 mm (uniform) placement noise
                params = dict(self.oracle_params(inst), noise=0.005, seed=int(rng.integers(1 << 30)))
            elif kind == "switch":  # extra control: no in-attempt repair, learn the order across attempts only
                params = (dict(mode="nearest", check=False, record=True) if a == 0 else
                          dict(mode="plan", careful=True, check=False, record=True))
            else:
                raise ValueError(kind)
            order = self.execute(inst, params)
            out = self.outcome(inst)
            out["order"] = order
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "oracle_noise", "naive", "end2end", "alt_outer", "alt_inner"):
                break
        return history
