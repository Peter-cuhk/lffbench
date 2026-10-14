"""LH bystander: put three cubes into an open box while a tall glass bottle that is not part of the task stands right
in front of the box -- lowering a cube into the box, the robot's own forearm comes down onto the bottle and knocks it
over.

Family: long-horizon self-disturbance, environment level ("state_restoration"; LIBERO-Recover's damage category L4: the
damaged thing is not part of the task). Three easy pick-and-place stages; the instruction says plainly that nothing
else may be moved and that the bottle must be left standing where it is. The agent has to notice the fallen bottle and
stand it up again on its spot before finishing (within an attempt), or learn to keep its arm off the bottle (across
attempts).

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * an open wooden box (fixture): cavity 14 x 14 cm, walls 6 cm high (rim 6.6 cm above the table), 0.8 cm thick;
    centre x in [0.00, 0.04], 6-10 cm left or right of the middle;
  * three 4 cm cubes (red / yellow / blue, 32 g) in a row beside the box on the other side (19-23 cm from the middle,
    from 10 cm behind to 2 cm in front of the box centre, shuffled, yaw +-17 deg), so that a straight carry to the box
    never passes over the bottle;
  * the bystander: a square green glass bottle with a label and a round red cap, 28.4 cm tall (body 6 x 6 x 18.5 cm,
    shoulder, neck, cap; a rounded base whose flat contact is 3.6 x 3.6 cm), 0.25 kg, centre of mass 12 cm up (an empty
    bottle: it tips beyond 8.5 deg; pressed down on its cap, beyond 3.7 deg).

Instance variable (where the bottle stands): "hazard" (15 of every 20 consecutive seeds): right in front of the box,
between the box and the robot, 1.0-2.5 cm from its front wall, 1 cm right to 4 cm left of its centre line; "benign":
beside the box on its outer side, 8-11 cm from the wall. Both look like "a bottle standing next to the box"; nothing in
the physics is hidden. What the agent cannot see before acting is its own arm: the wrist camera shows the fingers, not
the forearm, and the agent steers the grasp point.

Disturbance (physics, no teleport): lowering a cube onto the box floor (grasp point ~4 cm above the table), the
fingers and the cube stay 3-5 cm clear of the bottle (the box's front wall is between them), but the Panda's forearm
and wrist (links 5-7) slope back from the hand towards the robot base, starting ~12 cm above the grasp point; over the
space in front of the box they come down to 15-20 cm above the table, onto the upper part of a 28 cm bottle standing
there. The bottle is pushed over (mostly
towards the robot) and lies on the table in front of the box; visible in agentview once the arm has moved away
(measured, task card). Not every contact knocks it: a descent can come to rest on the cap (pinched between links 5
and 6, pressed down like a column), the arm stalls 4-6 cm short of its target and the bottle rocks back when the arm
lifts (measured: the smoke-test sequence on seed 3000; 6-7 of 8 hazard seeds still knocked with slow or near-first
descents, see the task card). The hit is yaw-independent (hit maps at yaw 0 / 45 / 90 deg) and not caused by the hand's
width. Transits at the agent's usual 11-20 cm mostly pass (the wrist end of the forearm is ~27 cm up at a 15 cm
transit). A careful packer keeps its arm above the bottle: carry at >= 24 cm and release the cube from >= 18 cm
(dropping it into the box) -- or moves the bottle out of the way and back, which the final-state judge would accept
but the instruction rules out.

Restoring (re-standable by a top-down gripper that only turns about the vertical): the body is square (a lying bottle
does not roll away from the closing fingers) and the cap is a ball with low torsional friction: grasped from above with
the fingers across the lying bottle's axis, the cap turns between the fingers, the lifted bottle swings down and hangs
straight; lowered onto the table it stands (measured: placement error ~1 mm). The instruction tells the agent that the
bottle hangs straight down when held by its cap (as lh_fence does for its knobs).

Protocol: cross (reset to the same instance before every attempt), k = 3. Outcome: final-state judge (every cube in the
box -- footprint inside the cavity, top below the rim, not held --, the bottle upright within 10 deg and its base
within 2 cm of where it stood, nothing held or touched, scene static) plus a per-step event log for the server log:
cube stages achieved / lost / restored; bottle_in_place lost / restored (with cause: arm / hand / cube, the robot geom,
the grasp point) plus the marker events bystander_knocked / bystander_restored / fallen. F1 gives the agent only
success / failure.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

Z = envs.TABLE_Z
# ---------------------------------------------------------------- geometry / physics
BOX = "box_1"
BOX_MODEL = "lff_lhb_box"
BOX_IN = 0.070  # cavity half width (14 x 14 cm)
WALL_T, FLOOR_T, WALL_H = 0.008, 0.006, 0.060
BOX_OUT = BOX_IN + WALL_T
RIM = FLOOR_T + WALL_H  # rim height above the table
WOOD = (0.62, 0.45, 0.28, 1.0)

CUBES = ("red_cube_1", "yellow_cube_1", "blue_cube_1")
CUBE_COLOR = {"red_cube_1": "red", "yellow_cube_1": "yellow", "blue_cube_1": "blue"}
CUBE_RGBA = dict(red=(0.85, 0.10, 0.08, 1.0), yellow=(0.92, 0.76, 0.10, 1.0), blue=(0.15, 0.30, 0.85, 1.0))
CUBE_E = 0.040
CUBE_DENS = 500.0  # 32 g
CUBE_MU = 0.8

BOTTLE = "bottle_1"
BOTTLE_MODEL = "lff_lhb_bottle"
BOTTLE_CLASS = "LffLhbBottle"
RB, HB = 0.030, 0.185  # square body: half width, height
RF, HF = 0.018, 0.006  # foot: a rounded glass base whose flat contact is 3.6 x 3.6 cm (0.6 cm high inset). With the
#                         4.8 cm foot of the draft the forearm could come to rest on the cap and stall instead of
#                         knocking it (bottle pressed down like a column, tilt <= 5 deg; measured, see the task card)
RS, HS = 0.022, 0.015  # shoulder
RN, HN = 0.0115, 0.060  # neck
RK = 0.015  # ball cap radius; its centre sits RK - 6 mm above the neck top
CAP_SINK = 0.006
H_TOT = HB + HS + HN + 2 * RK - CAP_SINK  # 0.284
CAP_Z = H_TOT - RK  # cap centre above the bottle's bottom when standing
BOTTLE_MASS = 0.25
BOTTLE_COM = 0.12  # above the bottom (an empty glass bottle: ~42 % of its height)
BOTTLE_MU = 0.6
CAP_MU_T = 0.0005  # torsional friction of the cap (condim 4): the grasped cap turns freely, the swing is damped
GLASS = (0.16, 0.42, 0.26, 1.0)
CAP_RGBA = (0.78, 0.12, 0.10, 1.0)

# instances
P_HAZARD = 0.75  # 15 of every 20 consecutive seeds: bottle in front of the box
HAZ_GAP = (0.010, 0.025)  # bottle body to the box's front wall
HAZ_LAT = (-0.010, 0.040)  # bottle centre left (+y) of the box's centre line
BEN_GAP = (0.080, 0.110)  # beside the box, on its outer side (away from the cubes)
BEN_DX = (-0.030, 0.030)
BOX_X = (0.0, 0.04)  # the bottle in front of the box stays >= 2 cm clear of the fingers in the home pose
BOX_Y = (0.06, 0.10)  # |y| of the box centre; the side is random
CUBE_Y = (0.19, 0.23)  # |y| of the cube row (other side)
CUBE_DX = (-0.10, -0.04, 0.02)  # cube row beside the box (x relative to the box centre; the Panda reaches ~0.8 m):
#                                  the straight carry to the box passes >= 2 cm from the bottle in front of it

# success predicate
POS_TOL = 0.020  # bottle base within 2 cm of where it stood
UPRIGHT = np.radians(10.0)
KNOCK_TILT = np.radians(10.0)  # once in place, the bottle counts as disturbed when tilted beyond this (it tips at
#                                 8.5 deg: beyond, it is falling or leaning on something) or moved
POS_HYST = 0.003  # > POS_TOL + POS_HYST for DEBOUNCE checks (a nudge that rocks back within 0.6 s is no loss)
FALLEN = np.radians(60.0)
IN_TOL = 0.002  # a cube's footprint may reach this far past the inner walls
RIM_TOL = 0.004
BELOW_RIM = 0.010  # in_box: the cube's lowest point at least this far below the rim
DEBOUNCE = 6  # predicate checks (every 0.1 s) a change must persist

# scripted motion (heights of the grasp point above the table)
Z_CARRY = 0.15  # Sonnet's transit: 11-20 cm, often lifted to ~20 before moving
Z_REL = 0.040  # Sonnet-like release: cube set down on the box floor (grasp point 4 cm up)
Z_SAFE = 0.26  # careful transit
Z_DROP = 0.19  # careful release: the cube is dropped from above the rim, the forearm stays above the bottle
NAIVE_ERR = 0.008  # Sonnet's placement error 0.4-1.3 cm
SLOT = 0.033  # slot offset from the box centre (cavity 14 cm, cubes 4 cm)


def bottle_xml(name):
    h = H_TOT / 2
    zf, zb, zs, zn = -h + HF / 2, -h + HF + (HB - HF) / 2, -h + HB + HS / 2, -h + HB + HS + HN / 2
    zk = -h + HB + HS + HN + RK - CAP_SINK
    m = BOTTLE_MASS
    ixx = m * ((2 * RB) ** 2 + H_TOT ** 2) / 16  # rough: most of the glass is in the body and the base
    izz = m * (2 * (2 * RB) ** 2) / 12
    parts = [("foot", "box", (0, 0, zf), (RF, RF, HF / 2), GLASS, f'friction="{BOTTLE_MU} 0.005 0.0001"'),
             ("body", "box", (0, 0, zb), (RB, RB, (HB - HF) / 2), GLASS, f'friction="{BOTTLE_MU} 0.005 0.0001"'),
             ("shoulder", "box", (0, 0, zs), (RS, RS, HS / 2), GLASS, f'friction="{BOTTLE_MU} 0.005 0.0001"'),
             ("neck", "cylinder", (0, 0, zn), (RN, HN / 2), GLASS, f'friction="{BOTTLE_MU} 0.005 0.0001"'),
             ("cap", "sphere", (0, 0, zk), (RK,), CAP_RGBA, f'friction="1 {CAP_MU_T} 0.0001" condim="4" priority="2"')]
    s = [f'        <inertial pos="0 0 {-h + BOTTLE_COM:.4f}" mass="{m}" diaginertia="{ixx:.6f} {ixx:.6f} {izz:.6f}" />']
    for nm, typ, pos, size, c, extra in parts:
        s.append(f'        <geom name="{name}_{nm}" type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" '
                 f'{extra} group="0" />')
        s.append(f'        <geom type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" contype="0" '
                 f'conaffinity="0" group="1" />')
    # a lighter label band (visual only) so the bottle reads as a bottle
    s.append(f'        <geom type="box" pos="0 0 {zb + 0.01:.4f}" size="{RB + 0.0008:.4f} {RB + 0.0008:.4f} 0.035" '
             f'rgba="0.92 0.90 0.80 1" contype="0" conaffinity="0" group="1" />')
    return _wrap(name, "\n".join(s), h, RB * 1.42)


def box_fixture_xml(name):
    """open wooden box, origin at the centre of its footprint, z = 0 at the table"""
    ix = BOX_IN
    parts = [((0, 0, FLOOR_T / 2), (BOX_OUT, BOX_OUT, FLOOR_T / 2)),
             ((ix + WALL_T / 2, 0, RIM / 2), (WALL_T / 2, BOX_OUT, RIM / 2)),
             ((-ix - WALL_T / 2, 0, RIM / 2), (WALL_T / 2, BOX_OUT, RIM / 2)),
             ((0, ix + WALL_T / 2, RIM / 2), (ix, WALL_T / 2, RIM / 2)),
             ((0, -ix - WALL_T / 2, RIM / 2), (ix, WALL_T / 2, RIM / 2))]
    g = []
    for k, (p, sz) in enumerate(parts):
        g.append(f'        <geom name="{name}_w{k}" type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(WOOD)}" '
                 f'friction="0.8 0.005 0.0001" {_SOLID} group="0" />')
        g.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(WOOD)}" contype="0" '
                 f'conaffinity="0" group="1" />')
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
{chr(10).join(g)}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {RIM:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{BOX_OUT:.4f} {BOX_OUT:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>"""


REACH_MAX = 0.74  # scripted policies: farthest grasp point (horizontal distance from the robot base) they rely on


def _reach(p):
    return float(np.hypot(p[0] + 0.66, p[1]))


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


def _lr(v):
    return f"{100 * abs(v):.1f} cm to the {'left' if v > 0 else 'right'}"


class BystanderSkills(Skills):
    """Skills that feed the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class Bystander(LFFTask):
    name = "lh_bystander"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Restore", "Plan")
    protocol = "cross"
    max_attempts = 3
    agent_max_steps = 100  # careful: 3 cubes x ~7 commands + looking; standing the bottle up again ~8-10; + slack
    instruction = ("Put the red, yellow and blue cubes into the open wooden box. Do not move anything else: the tall "
                   "glass bottle must be left standing upright exactly where it is now. (The bottle has a round red "
                   "cap; held by its cap, the bottle hangs straight down.)")
    instruction_indirect = ("Pack the three small cubes (red, yellow, blue) into the wooden crate. Leave everything else "
                            "as it is -- the tall green bottle has to stay standing on its spot. (Its round red cap: held "
                            "by the cap, the bottle hangs straight down.)")
    agent_object_names = ["red cube", "yellow cube", "blue cube", "open wooden box",
                          "tall glass bottle (square, with a round red cap; held by its cap it hangs straight down)"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        box = register_generated("LffLhbBox", box_fixture_xml(BOX_MODEL), free=False)
        objs = []
        for i, n in enumerate(CUBES):
            c = CUBE_COLOR[n]
            cat = register_generated(f"LffLhbCube{c.title()}",
                                     box_xml(f"lff_lhb_cube_{c}", (CUBE_E / 2,) * 3, CUBE_RGBA[c], density=CUBE_DENS,
                                             friction=(CUBE_MU, 0.005, 0.0001)).replace(_SOLID, ""))
            objs.append((n, cat, f"{c}_cube_region", (-0.35, -0.25 + 0.15 * i, -0.33, -0.23 + 0.15 * i)))
        bt = register_generated(BOTTLE_CLASS, bottle_xml(BOTTLE_MODEL))
        objs.append((BOTTLE, bt, "bottle_region", (-0.35, 0.24, -0.33, 0.26)))
        fx = [(BOX, box, "box_region", (-0.06, -0.01, -0.04, 0.01))]
        return write_bddl(self.name, "put the cubes into the box and leave the bottle where it is", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.box_bid = m.body_name2id(self.env.fixtures_dict[BOX].root_body)
        self.box_gids = {g for g in range(m.ngeom) if m.geom_bodyid[g] in
                         {self.box_bid} | {c for c in range(m.nbody) if m.body_parentid[c] == self.box_bid}}
        pre = f"{BOTTLE}_{BOTTLE_MODEL}"
        self.cap_gid = m.geom_name2id(pre + "_cap")
        self.bottle_gids = set(envs.obj_geom_ids(self.env, BOTTLE))
        self.cube_gids = {n: set(envs.obj_geom_ids(self.env, n)) for n in CUBES}
        # robot parts: 1 = hand / fingers (what the agent sees and steers), 2 = arm links (wrist, forearm, ...)
        self.part = np.zeros(m.ngeom, int)
        for g in envs.robot_geom_ids(self.env):
            n = m.geom_id2name(g) or ""
            self.part[g] = 1 if n.startswith("gripper0_") else 2
        self.other = np.zeros(m.ngeom, int)  # 3 = a cube, 4 = the box
        for n in CUBES:
            self.other[list(self.cube_gids[n])] = 3
        self.other[list(self.box_gids)] = 4
        self.is_bottle = np.zeros(m.ngeom, bool)
        self.is_bottle[list(self.bottle_gids)] = True
        # bottle contacts are collected after every MuJoCo substep (a fast hit can be shorter than a control step)
        sim = self.env.sim
        if not getattr(sim, "_lhb_wrapped", False):
            orig = sim.step

            def step(*a, **k):
                orig(*a, **k)
                t = getattr(sim, "_lhb_task", None)
                if t is not None and t.mon is not None:
                    t._substep()
            sim.step = step
            sim._lhb_wrapped = True
        sim._lhb_task = self
        self._sub = {}
        self.inst = None
        self.mon = None
        self.sk = None

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        s = 1.0 if rng.random() < 0.5 else -1.0  # box on the left (+1) or right (-1)
        bx, by = float(rng.uniform(*BOX_X)), s * float(rng.uniform(*BOX_Y))
        # balanced blocks like lh_lid_box: exactly round(20 * P_HAZARD) of every 20 consecutive seeds are hazards
        hazard = bool((int(seed) * 7) % 20 < round(20 * P_HAZARD))
        if hazard:
            gap, lat = float(rng.uniform(*HAZ_GAP)), float(rng.uniform(*HAZ_LAT))
            bottle = [bx - BOX_OUT - gap - RB, by + lat]
        else:
            gap, lat = float(rng.uniform(*BEN_GAP)), float(rng.uniform(*BEN_DX))
            bottle = [bx + lat, by + s * (BOX_OUT + gap + RB)]
        cy = -s * float(rng.uniform(*CUBE_Y))
        perm = rng.permutation(len(CUBES))
        start = {CUBES[i]: [float(bx + CUBE_DX[j] + rng.uniform(-0.01, 0.01)), float(cy + rng.uniform(-0.01, 0.01))]
                 for j, i in enumerate(perm)}
        start_yaw = {n: float(rng.uniform(-0.3, 0.3)) for n in CUBES}
        return dict(seed=int(seed), hazard=hazard, bottle_side="front" if hazard else "outer side",
                    box_side="left" if s > 0 else "right", box_xy=[bx, by], bottle_xy=[float(bottle[0]), float(bottle[1])],
                    bottle_yaw=float(rng.uniform(-0.25, 0.25)), gap=gap, lat=lat, start=start, start_yaw=start_yaw)

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        bx, by = inst["box_xy"]
        m.body_pos[self.box_bid] = [bx, by, Z]
        m.body_quat[self.box_bid] = [1.0, 0.0, 0.0, 0.0]
        for n, xy in inst["start"].items():
            envs.place_upright(self.env, n, xy, yaw=inst["start_yaw"][n], clearance=0.001)
            envs.set_friction(self.env, n, CUBE_MU)
        envs.place_upright(self.env, BOTTLE, inst["bottle_xy"], yaw=inst["bottle_yaw"], clearance=0.001)

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = BystanderSkills(self.env, self, recorder=recorder)
        self._sub = {}
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ state helpers
    def R(self, name):
        return self.env.sim.data.body_xmat[envs.body_id(self.env, name)].reshape(3, 3).copy()

    def bottle_tilt(self):
        return float(np.arccos(np.clip(self.R(BOTTLE)[2, 2], -1.0, 1.0)))

    def bottle_base(self):
        """xy of the centre of the bottle's bottom face"""
        d = self.env.sim.data
        return (envs.obj_pos(self.env, BOTTLE) - self.R(BOTTLE)[:, 2] * H_TOT / 2)[:2]

    def bottle_offset(self):
        return self.bottle_base() - np.asarray(self.inst["bottle_xy"])

    def cap_xyz(self):
        return self.env.sim.data.geom_xpos[self.cap_gid].copy()

    def held(self, name):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.objects_dict[name]))

    def bottle_in_place(self, tol=POS_TOL, tilt=UPRIGHT):
        return bool(self.bottle_tilt() <= tilt and np.linalg.norm(self.bottle_offset()) <= tol
                    and envs.obj_min_z(self.env, BOTTLE) < Z + 0.01 and not self.held(BOTTLE))

    def to_box(self, xy):
        return np.asarray(xy, float)[:2] - np.asarray(self.inst["box_xy"])

    def in_box(self, name):
        m, d = self.env.sim.model, self.env.sim.data
        g = next(iter(self.cube_gids[name]))
        Rg = d.geom_xmat[g].reshape(3, 3)
        c = self.to_box(d.geom_xpos[g][:2])
        ext = np.abs(Rg[:2, :]) @ m.geom_size[g][:3]
        # 10-14: "in the box" = footprint inside the cavity and the cube's lowest point well below the rim (it is held
        # by the walls); a cube stacked on another one may stick out above the rim -- the instruction never asks for
        # "below the rim" (two Sonnet pilot attempts that stacked the last cube were judged failures before this)
        return bool(np.all(np.abs(c) + ext <= BOX_IN + IN_TOL) and envs.obj_min_z(self.env, name) <= Z + RIM - BELOW_RIM
                    and not self.held(name))

    def over_box(self, name):
        c = self.to_box(envs.obj_pos(self.env, name)[:2])
        return bool(np.all(np.abs(c) <= BOX_OUT))

    def stage_values(self):
        v = {f"{CUBE_COLOR[n]}_in_box": self.in_box(n) for n in CUBES}
        v["bottle_in_place"] = self.bottle_in_place()
        return v

    # ------------------------------------------------------------ per-step monitor
    def _init_monitor(self):
        keys = [f"{CUBE_COLOR[n]}_in_box" for n in CUBES] + ["bottle_in_place"]
        st = {k: False for k in keys}
        st["bottle_in_place"] = self.bottle_in_place()
        self.mon = dict(state=st, ever=[k for k in keys if st[k]], lost=[], events=[], cand={k: 0 for k in keys},
                        step=0, last_touch=None, robot_touch_step=-10 ** 9, touch_parts=[], n_knocks=0, n_fallen=0,
                        fallen=False,
                        max_tilt_deg=0.0, max_offset_cm=0.0, first_knock_step=None, first_touch_step=None)

    def _substep(self):
        d = self.env.sim.data
        n = d.ncon
        if not n:
            return
        g1 = np.asarray(d.contact.geom1)[:n]
        g2 = np.asarray(d.contact.geom2)[:n]
        b1, b2 = self.is_bottle[g1], self.is_bottle[g2]
        sel = b1 ^ b2
        if not sel.any():
            return
        other = np.where(b1, g2, g1)[sel]
        for o in other:
            p = int(self.part[o])
            if p:
                who = "hand" if p == 1 else "arm"
                prev = self._sub.get("robot")
                self._sub["robot"] = "arm" if (who == "arm" or prev == "arm") else "hand"
                self._sub.setdefault("geom", self.env.sim.model.geom_id2name(int(o)))
            elif self.other[o] == 3:
                self._sub["cube"] = True

    def monitor_step(self):
        mon = self.mon
        if mon is None:
            return
        mon["step"] += 1
        step = self.sk.n_steps if self.sk is not None else mon["step"]
        sub, self._sub = self._sub, {}
        if "robot" in sub:
            mon["robot_touch_step"] = step
            mon["last_touch"] = [sub["robot"], step, sub.get("geom", "")]
            if sub["robot"] not in mon["touch_parts"]:
                mon["touch_parts"].append(sub["robot"])
            if mon["first_touch_step"] is None:
                mon["first_touch_step"] = step
        elif sub.get("cube") and (mon["last_touch"] is None or step - mon["last_touch"][1] > 20):
            mon["last_touch"] = ["cube", step, ""]
        tilt = self.bottle_tilt()
        mon["max_tilt_deg"] = round(max(mon["max_tilt_deg"], float(np.degrees(tilt))), 1)
        mon["max_offset_cm"] = round(max(mon["max_offset_cm"], 100 * float(np.linalg.norm(self.bottle_offset()))), 1)
        if mon["step"] % 2:
            return  # predicates every other control step (0.1 s)
        if not mon["fallen"] and tilt > FALLEN and not self.held(BOTTLE):
            mon["fallen"] = True
            mon["n_fallen"] += 1
            mon["events"].append(dict(step=step, pred="bottle", ev="fallen", cause=self._cause(step)))
        elif mon["fallen"] and tilt < UPRIGHT:
            mon["fallen"] = False
        for n in CUBES:
            self._check(f"{CUBE_COLOR[n]}_in_box", self.in_box(n), step, n)
        st = mon["state"]["bottle_in_place"]
        # hysteresis: once in place it counts as lost only beyond 2.3 cm / 10 deg, held for 0.6 s (a nudge that rocks
        # back is no loss)
        now = self.bottle_in_place(POS_TOL + POS_HYST, KNOCK_TILT) if st else self.bottle_in_place()
        self._check("bottle_in_place", now, step, BOTTLE)

    def _cause(self, step):
        lt = self.mon["last_touch"]
        if lt is not None and step - lt[1] <= 60:
            return lt[0]
        return "other"

    def _check(self, key, now, step, name):
        mon = self.mon
        if now == mon["state"][key]:
            mon["cand"][key] = 0
            return
        if name == BOTTLE and not now and step - mon["robot_touch_step"] <= 6 and not self.held(BOTTLE):
            # the arm is still pressing on the bottle (measured: a slow descent can come to rest on its cap and stall,
            # tilting it ~12 deg; it rocks back when the arm lifts): a loss only once the robot has let go of it
            mon["cand"][key] = 0
            return
        mon["cand"][key] += 1
        if mon["cand"][key] < DEBOUNCE:
            return
        mon["cand"][key] = 0
        mon["state"][key] = now
        if now:
            restored = key in mon["lost"]
            ev = "restored" if restored else "achieved"
            if key not in mon["ever"]:
                mon["ever"].append(key)
            if restored:
                mon["lost"].remove(key)
            mon["events"].append(dict(step=step, pred=key, ev=ev))
            if key == "bottle_in_place" and restored:
                mon["events"].append(dict(step=step, pred="bottle", ev="bystander_restored",
                                          offset_cm=(100 * self.bottle_offset()).round(1).tolist()))
            return
        if key not in mon["ever"]:
            return
        cause = "picked_up" if self.held(name) else self._cause(step)
        if key not in mon["lost"]:
            mon["lost"].append(key)
        e = dict(step=step, pred=key, ev="lost", cause=cause)
        if name == BOTTLE:
            ee = self.sk.eef_pos() if self.sk is not None else np.zeros(3)
            e["eef_cm"] = [round(100 * float(v), 1) for v in (ee[0] - self.inst["box_xy"][0], ee[1] - self.inst["box_xy"][1],
                                                               ee[2] - Z)]  # grasp point rel. to the box centre / table
            e.update(tilt_deg=round(float(np.degrees(self.bottle_tilt())), 1),
                     offset_cm=(100 * self.bottle_offset()).round(1).tolist())
            if mon["last_touch"] is not None and mon["last_touch"][0] in ("arm", "hand"):
                e["robot_geom"] = mon["last_touch"][2]
        mon["events"].append(e)
        if name == BOTTLE and cause != "picked_up":
            mon["n_knocks"] += 1
            if mon["first_knock_step"] is None:
                mon["first_knock_step"] = step
            mon["events"].append(dict(step=step, pred="bottle", ev="bystander_knocked", cause=cause,
                                      robot_geom=e.get("robot_geom", ""), tilt_deg=e["tilt_deg"],
                                      offset_cm=e["offset_cm"], eef_cm=e["eef_cm"]))

    # ------------------------------------------------------------ outcome
    def _static(self):
        d = self.env.sim.data
        for n in CUBES + (BOTTLE,):
            _, da = envs.free_joint_addr(self.env, n)
            if np.linalg.norm(d.qvel[da:da + 3]) > 0.02 or np.linalg.norm(d.qvel[da + 3:da + 6]) > 0.5:
                return False
        return True

    def robot_touching(self):
        d = self.env.sim.data
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if self.part[a] and (self.is_bottle[b] or self.other[b] == 3):
                    nm = "bottle" if self.is_bottle[b] else "a cube"
                    if nm not in out:
                        out.append(nm)
        return out

    def outcome(self, inst):
        envs.settle(self.env, 30)
        mon = self.mon
        vals = self.stage_values()
        static = self._static()
        held = [n for n in CUBES + (BOTTLE,) if self.held(n)]
        touching = self.robot_touching()
        success = bool(all(vals.values()) and static and not held and not touching)
        evs = mon["events"]
        regress = [e for e in evs if e["ev"] == "lost"]
        restored = [e for e in evs if e["ev"] == "restored"]
        knocks = [e for e in evs if e["ev"] == "bystander_knocked"]
        # trap fired: a knock was logged, or the bottle ends up out of place after the robot touched it (a slow push
        # to just over 2 cm can stay inside the monitor's hysteresis band)
        trap_fired = bool(knocks) or (not vals["bottle_in_place"] and not self.held(BOTTLE) and bool(mon["touch_parts"]))
        recovered = bool(trap_fired and vals["bottle_in_place"])
        off = self.bottle_offset()
        tilt = float(np.degrees(self.bottle_tilt()))
        parts = []
        failure = None
        if success:
            parts.append("All three cubes are in the box and the bottle stands where it stood.")
        else:
            out_cubes = [n for n in CUBES if not vals[f"{CUBE_COLOR[n]}_in_box"]]
            if not vals["bottle_in_place"]:
                failure = ("bystander_held" if self.held(BOTTLE) else
                           "bystander_knocked" if tilt > np.degrees(UPRIGHT) else "bystander_moved")
            elif held:
                failure = "held"
            elif out_cubes:
                failure = "cube_not_in_box"
            elif touching:
                failure = "robot_touching"
            else:
                failure = "not_static"
            if not vals["bottle_in_place"]:
                if self.held(BOTTLE):
                    parts.append("The bottle is still held by the gripper.")
                else:
                    pose = ("is lying on its side" if tilt > 60 else f"is tilted {tilt:.0f} degrees" if tilt > np.degrees(UPRIGHT)
                            else "stands upright")
                    txt = f"The bottle {pose}; its base is {_fb(off[0])} and {_lr(off[1])} of where it stood"
                    if knocks:
                        k = knocks[0]
                        who = {"arm": "the robot's arm (above the gripper)", "hand": "the gripper",
                               "cube": "a cube"}.get(k["cause"], "something")
                        txt += f" (it was knocked by {who}"
                        txt += " and stood up again later, but not back in place)." if restored and any(
                            e["pred"] == "bottle_in_place" for e in restored) else ")."
                    else:
                        txt += "."
                    parts.append(txt)
            for n in out_cubes:
                c = CUBE_COLOR[n]
                if self.held(n):
                    parts.append(f"The {c} cube is still held by the gripper.")
                elif self.over_box(n):
                    parts.append(f"The {c} cube rests on the box's rim instead of inside the box.")
                else:
                    rel = self.to_box(envs.obj_pos(self.env, n)[:2])
                    parts.append(f"The {c} cube is not in the box: its centre is {_fb(rel[0])} and {_lr(rel[1])} of the "
                                 f"box's centre.")
            if touching and all(vals.values()):
                parts.append("The robot was still touching " + " and ".join(touching) + " at the end.")
            if not static and all(vals.values()):
                parts.append("Something was still moving at the end.")
        if trap_fired and vals["bottle_in_place"]:
            parts.append(f"(The bottle was knocked out of place {len(knocks)} time(s) during the attempt and put back.)")
        out = dict(success=success, failure=failure, stages=vals, static=bool(static), held=held, touching=touching,
                   events=evs, n_regressions=len(regress), n_restored=len(restored), trap_fired=trap_fired,
                   recovered_in_attempt=recovered, n_knocks=len(knocks), n_fallen=mon["n_fallen"],
                   knock_causes=[k["cause"] for k in knocks], first_knock_step=mon["first_knock_step"],
                   bottle_touch_parts=list(mon["touch_parts"]), bottle_max_tilt_deg=mon["max_tilt_deg"],
                   bottle_max_offset_cm=mon["max_offset_cm"], bottle_tilt_deg=round(tilt, 1),
                   bottle_offset_cm=(100 * off).round(1).tolist(), hazard=bool(inst["hazard"]),
                   detail=" ".join(parts))
        return out

    # ------------------------------------------------------------ scripted primitives (perfect perception)
    def _go(self, xyz, yaw=None, speed=None, tol=0.004, max_steps=300):
        return self.sk.move_to(xyz, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)

    def _settle(self, name, max_checks=25):
        d = self.env.sim.data
        _, da = envs.free_joint_addr(self.env, name)
        for _ in range(max_checks):
            if np.linalg.norm(d.qvel[da:da + 3]) < 0.01 and np.linalg.norm(d.qvel[da + 3:da + 6]) < 0.2:
                return
            self.sk.hold(2)

    def _cube_yaw(self, name):
        R = self.R(name)
        i = int(np.argmin(np.abs(R[2])))
        yaw = float(np.arctan2(R[1, i], R[0, i]))
        y = float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        cands = [y + k * np.pi / 2 for k in (-1, 0, 1)]
        return min(cands, key=lambda c: abs(c - self.sk.yaw) if abs(c) <= np.pi / 2 else 9.0)

    def pick(self, name, z_carry, v=0.15, tries=2):
        """Sonnet-like top-down pick: over the cube at carry height, down at 0.1 to 1.5 cm above, last 1.5 cm at 0.04,
        close, lift to carry height"""
        sk = self.sk
        for _ in range(tries):
            self._settle(name)
            p = envs.obj_pos(self.env, name)
            zg = p[2] - Z
            yaw = self._cube_yaw(name)
            sk.set_gripper(False, steps=8)
            self._go([p[0], p[1], Z + max(z_carry, zg + 0.06)], yaw=yaw, speed=v)
            self._go([p[0], p[1], Z + zg + 0.015], speed=0.1)
            self._go([p[0], p[1], Z + zg], speed=0.04, tol=0.003, max_steps=120)
            sk.set_gripper(True, steps=15)
            self._go([p[0], p[1], Z + z_carry], speed=v)
            if self.held(name):
                return True
            sk.set_gripper(False, steps=8)
        return False

    def put(self, xy, z_rel, z_carry, v=0.15, v_down=None):
        """carry over xy at carry height, straight down (0.1 m/s, last 1.5 cm at 0.03; or one move at v_down), open,
        back up"""
        sk = self.sk
        self._go([xy[0], xy[1], Z + z_carry], speed=v)
        if z_rel < z_carry - 0.02 and v_down:
            self._go([xy[0], xy[1], Z + z_rel], speed=v_down, tol=0.003, max_steps=250)
        elif z_rel < z_carry - 0.02:
            self._go([xy[0], xy[1], Z + z_rel + 0.015], speed=0.1)
            self._go([xy[0], xy[1], Z + z_rel], speed=0.03, tol=0.003, max_steps=150)
        sk.set_gripper(False, steps=12)
        self._go([xy[0], xy[1], Z + z_carry], speed=v)
        sk.hold(5)

    def slot_xy(self, k, order):
        """slot k of the scripted packing pattern, in world xy (box frame: u forward, v left)"""
        u, v = order[k]
        return np.asarray(self.inst["box_xy"]) + np.array([u, v]) * SLOT

    def put_cube(self, name, k, order, z_rel, z_carry, noise=None, v=0.15, v_down=None):
        if not self.pick(name, z_carry, v=v):
            return False
        xy = self.slot_xy(k, order) + (np.asarray(noise, float) if noise is not None else 0.0)
        self.put(xy, z_rel, z_carry, v=v, v_down=v_down)
        self._settle(name)
        return True

    def restand_bottle(self, z_carry=None):
        """stand the bottle up again on its spot: grasp the ball cap from above (fingers across the lying bottle's
        axis), lift until it hangs free, carry it over its spot, lower until the bottom is ~2 mm above the table, let
        go. Returns True if it is back in place."""
        sk = self.sk
        self._settle(BOTTLE)
        for _ in range(2):
            if _reach(self.cap_xyz()) > REACH_MAX and self.bottle_tilt() > FALLEN:
                self.pull_in_bottle()
            k = self.cap_xyz()
            ax = self.R(BOTTLE)[:2, 2]
            yaw = float(np.arctan2(ax[1], ax[0])) if np.linalg.norm(ax) > 0.2 else sk.yaw  # finger axis across it
            sk.set_gripper(False, steps=8)
            cur = sk.eef_pos()
            self._go([cur[0], cur[1], max(cur[2], Z + 0.15)], speed=0.15)
            self._go([k[0], k[1], max(k[2] + 0.08, Z + 0.15)], yaw=yaw, speed=0.15)
            sk.hold(5)
            k = self.cap_xyz()
            self._go([k[0], k[1], k[2] + 0.015], speed=0.1)
            self._go([k[0], k[1], k[2]], speed=0.04, tol=0.003, max_steps=120)
            sk.set_gripper(True, steps=15)
            if not self.held(BOTTLE):
                sk.set_gripper(False, steps=8)
                self._go([k[0], k[1], k[2] + 0.08], speed=0.1)
                continue
            zl = Z + CAP_Z + 0.06
            self._go([k[0], k[1], zl], speed=0.10)
            sk.hold(15)
            tx, ty = self.inst["bottle_xy"]
            self._go([tx, ty, zl], speed=0.10)
            sk.hold(25)
            self._go([tx, ty, Z + CAP_Z + 0.015], speed=0.08)
            self._go([tx, ty, Z + CAP_Z + 0.002], speed=0.03, tol=0.003, max_steps=120)
            sk.set_gripper(False, steps=12)
            self._go([tx, ty, Z + CAP_Z + 0.06], speed=0.08)
            sk.hold(20)
            self._settle(BOTTLE)
            if self.bottle_in_place():
                return True
        return self.bottle_in_place()

    def pull_in_bottle(self):
        """the lying bottle's cap is out of reach (it fell forward over the box): grasp the body near its bottom end
        (fingers across it), lift it 3 cm, pull it towards the robot until the cap is within reach, set it down"""
        sk = self.sk
        Rb = self.R(BOTTLE)
        ax = Rb[:, 2]
        c = envs.obj_pos(self.env, BOTTLE)
        g = c - ax * (H_TOT / 2 - 0.05)  # 5 cm from the bottom end along the axis
        zg = max(float(g[2] - Z), 0.02)
        yaw = float(np.arctan2(ax[1], ax[0]))
        need = _reach(self.cap_xyz()) - (REACH_MAX - 0.05)
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
        self._settle(BOTTLE)

    def _fallen_cubes(self, done):
        return [n for n in done if not self.in_box(n) and not self.over_box(n) and not self.held(n)]

    # ------------------------------------------------------------ scripted policies
    def execute(self, inst, params):
        careful = params.get("careful", False)
        zc = params.get("z_carry", Z_SAFE if careful else Z_CARRY)
        zr = params.get("z_rel", Z_DROP if careful else Z_REL)
        order = params.get("slots") or [(1, 1), (1, -1), (-1, 0)]  # far row first (Sonnet fills from the far end)
        cubes = params.get("cubes") or self._cube_order()
        noise = params.get("noise") or {}
        v = params.get("speed", 0.15)
        done = []
        for k, n in enumerate(cubes):
            self.put_cube(n, k, order, zr, zc, noise=noise.get(n), v=v, v_down=params.get("v_down"))
            done.append(n)
            # looks after every step (Sonnet): re-places a cube that ended up outside the box ...
            for f in self._fallen_cubes(done):
                self.put_cube(f, done.index(f), order, zr, zc, v=v)
            # ... and, if the bystander is in the policy's view of the task, stands the bottle up again
            if params.get("restore") == "each" and not self.bottle_in_place():
                self.restand_bottle()
        if params.get("check"):
            self.check_and_fix(order, zr, zc)
        self.sk.hold(10)

    def check_and_fix(self, order, zr, zc, rounds=2):
        for _ in range(rounds):
            v = self.stage_values()
            if all(v.values()):
                return True
            for n in CUBES:
                if not self.in_box(n):
                    self.put_cube(n, CUBES.index(n), order, zr, zc)
            if not self.bottle_in_place():
                self.restand_bottle()
        return all(self.stage_values().values())

    def _cube_order(self):
        """the cube nearest the box first"""
        bxy = np.asarray(self.inst["box_xy"])
        return sorted(CUBES, key=lambda n: float(np.linalg.norm(np.asarray(self.inst["start"][n]) - bxy)))

    def default_params(self, inst):
        """Sonnet-like: cubes nearest-first, carried 15 cm up in straight lines, each lowered straight down onto the box
        floor (far row first), looks after every step and re-places a cube that fell out; no special care for the
        bottle (it is not part of the packing)"""
        return dict(careful=False, restore=None, check=False)

    def oracle_params(self, inst):
        """knows its arm: carries high (24 cm) and drops the cubes into the box from 16 cm, so the forearm stays above
        the bottle; checks every stage at the end"""
        return dict(careful=True, restore="each", check=True)

    def adapt_params(self, inst, history):
        """attempt >= 2: the careful packing (high carry, drop from above the rim), stand the bottle up if needed,
        check at the end"""
        return dict(careful=True, restore="each", check=True)

    def blind_params(self, inst, attempt, rng):
        """the natural behaviour again with small random variations, no history, no care for the bottle"""
        pats = [[(1, 1), (1, -1), (-1, 0)], [(1, -1), (1, 1), (-1, 0)], [(0, 1), (0, -1), (-1, 0)],
                [(1, 0), (-1, 1), (-1, -1)], [(1, 1), (-1, -1), (0, 0)]]
        return dict(careful=False, restore=None, check=False, slots=pats[int(rng.integers(len(pats)))],
                    z_carry=float(rng.uniform(0.12, 0.20)), z_rel=float(rng.uniform(0.032, 0.055)),
                    noise={n: rng.uniform(-NAIVE_ERR, NAIVE_ERR, 2).tolist() for n in CUBES})

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """oracle / naive: one attempt; adaptive: attempt 1 = naive + stands the knocked bottle up again after every
        step and checks at the end, attempts >= 2 careful; blind: naive with random variations.
        Extra kinds (not in validate's table): switch (naive, then careful without restoring), oracle_noise (oracle with
        +-5 mm placement noise)"""
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
                params = dict(careful=False, restore="each", check=True) if a == 0 else self.adapt_params(inst, history)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "switch":
                params = self.default_params(inst) if a == 0 else dict(careful=True, restore=None, check=False)
            elif kind == "oracle_noise":
                params = dict(self.oracle_params(inst), noise={n: rng.uniform(-0.005, 0.005, 2).tolist() for n in CUBES})
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
