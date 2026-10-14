"""LH lh_bridge (v2, 2026-10-09): build a small bridge (two pillars, a plank, then the red cube "car" on top); the car is
parked right behind the bridge site, and once the plank is down the agent's natural reach for it knocks the bridge down,
which must be rebuilt before the task can be finished.

Category: state_restoration, long-horizon self-inflicted regression. The user's example verbatim: "build a bridge;
while working you knock over a foundation you already built and must rebuild it before continuing".

v1 -> v2 (after the Sonnet 5.5 pilot lh1: only 1 of 3 instances was ever disturbed; Sonnet carried 12-20 cm high and
lifted over built pillars, so the v1 mechanism -- sweeping a pillar with a block carried low -- rarely happened; the one
disturbance was the hand landing on the plank while going for the red cube): the car now starts in the gap between the
two squares, 5.5-9 cm behind the bridge line (right behind where the plank will lie); a spare fifth block; the bridge
4-7 cm further from the robot and the plank nearer, so that knocked parts stay within reach; softer contacts for the
bridge parts (the stiff default made collapsing blocks fly 30-90 cm); the naive reference imitates Sonnet's measured
motion style instead of the v1 low-transit script (kept as naive_v1).

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y): two coloured square
marks (blue and green, 6 x 6 cm, visual only) 16 cm apart along y, 48-51 cm in front of the robot base; the red cube
(4 cm) in the gap between them, 5.5-9 cm behind the line through the squares (+-3 cm sideways, yaw +-20 deg); five
wooden cubes (edge 5.0-5.6 cm, friction 0.3-0.6, five distinct categories) in a row 9-11 cm behind the squares, two on
one side of the car and three on the other; the plank (24 x 5 x 1.2 cm) lying left-right 19-21 cm behind the squares.

Stages (order-free predicates, evaluated after every control step by a monitor; events are debounced over 6 steps):
  S1  two blocks stacked on the blue square (lower block resting on the table within 2.5 cm of the square's centre,
      upper block resting on it within 2.5 cm; any face down, cubes have no "right way up")
  S2  the same on the green square
  S3  the plank rests on the top blocks of both pillars: in contact with both, not with the table, tilt <= 10 deg
  S4  the red cube rests on the plank within 6 cm of the plank's centre along its length
Success = all four hold at the end simultaneously, the scene is static (< 4 mm motion in 1 s) and the robot neither
holds nor touches any object. The monitor's achieved / lost / restored events go to the server log only (F1).

Why the agent breaks the bridge (measured, see task_cards/lh_bridge.md): with the plank down, the car sits 3-6.5 cm
behind the plank's back edge, between the pillars. The Panda hand is 21 cm long along the finger axis and its underside
is only 2.4 cm above the grasp point; the fingertips hang 1.3 cm below it. Each of the three pilot runs went for the car
differently; scripted here on seeds 1000-1019: hovering ~13 cm above it while turning the wrist from the plank's 90 deg
to the car's yaw, then straight down (5/20 knock the bridge), keeping the 90 deg wrist and going straight down (the hand
lands across the plank: 20/20), or one diagonal move from above the bridge straight down to the grasp (the hand drives
through the bridge: 20/20). The plank and one or both pillar tops fall; in 22 of 24 collapses (seeds 3000-3029) every
part needed stays within reach (a block out of reach is covered by the spare; the plank was thrown out of reach twice).
What prevents it (learnable from one failure, and what Sonnet wrote after its pilot failure): move the car away before
laying the plank.

Hidden variables: none that matter. Block size and friction are instance variety; the car's distance (5.5-9 cm) and yaw
decide only the "hover" style's outcome.

Scripted references (perception = simulator state):
  naive     Sonnet-like (measured on the pilot): instruction order; per pillar the free block nearest its square first;
            hover ~12 cm above an object turned to its yaw, vertical descent, grasp 1.1 cm below the top; carry to a bare
            square in one diagonal move to 12.5 cm above it, onto a block lift first; lift over a built pillar when the
            carried block's straight path would hit it; release, lift to 15 cm; plank lifted to 7 cm, carried at 16 cm,
            lift to 21 cm; the car approach style is the pilot's mixture (seed % 3: hover / keep / direct); no final check
  naive_v1  the v1 naive (vertical / horizontal legs at 8 cm, carries at max(8 cm, release + 4 cm)), for comparison
  oracle    careful: first moves the car out of the zone behind the bridge; all transits at 25 cm, vertical approaches
            at 0.1 m/s, grasps 1.6 cm below a cube's top, gripper yaw chosen for hand / finger clearance; checks every
            stage at the end and repairs what is broken (same repair as adaptive)
  adaptive  attempt 1 = naive, then checks every stage before finishing and restores what was lost (tips a plank that
            stands on edge, puts the plank and car aside, re-stacks the pillars from usable blocks, then plank and car
            again); from attempt 2 the careful behaviour
  blind     naive with random variation (hover / carry / lift heights, diagonal or lift-first carries, car style,
            block allocation, 4 mm placement noise), no history

Protocol: cross (reset to the same instance before every attempt), k = 3.
"""
import os

import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

BLOCKS = ["block_a_1", "block_b_1", "block_c_1", "block_d_1", "block_e_1"]  # four are needed, one is a spare
PLANK, CAR = "plank_1", "car_1"
SQUARES = ["square_blue_1", "square_green_1"]  # S1 = blue, S2 = green
ITEMS = BLOCKS + [PLANK, CAR]
BLOCK_RGBA = [(0.80, 0.66, 0.46, 1.0), (0.70, 0.55, 0.37, 1.0), (0.84, 0.72, 0.56, 1.0), (0.64, 0.50, 0.36, 1.0),
              (0.76, 0.60, 0.42, 1.0)]  # wood
PLANK_HALF = (0.12, 0.025, 0.006)  # 24 x 5 x 1.2 cm, long axis = body x
CAR_HALF = 0.02
BLOCK_NOM = 0.026  # nominal half edge in the MJCF (resized per instance)
SQ_HALF = 0.03
SQ_RGBA = {"square_blue_1": (0.12, 0.30, 0.90, 1.0), "square_green_1": (0.10, 0.70, 0.25, 1.0)}
SQ_NAME = {"square_blue_1": "blue", "square_green_1": "green"}
SPAN = 0.16  # square centre distance (along y)

TZ = envs.TABLE_Z
BASE_X = -0.66
HAND_HALF_LONG = 0.106  # Panda hand collision AABB about the grasp site: +-10.6 cm along the finger axis
HAND_HALF_SHORT = 0.036  # +-3.6 cm across it
HAND_BOTTOM = 0.024  # its underside is 2.4 cm above the grasp site (measured)

SQ_TOL = 0.025  # lower block centre within 2.5 cm of the square centre
STACK_TOL = 0.025  # upper block centre within 2.5 cm of the lower one
FLAT_COS = np.cos(np.radians(12))
PLANK_TILT = np.radians(10)
CAR_MID = 0.06
DEBOUNCE = 6  # control steps a predicate must keep its new value before an event is recorded

# v2 layout (2026-10-09): the red cube ("car") is parked in the gap between the two squares, just behind the bridge
# line, so that once the plank is down a natural top-down grasp of it lands the hand on the plank / pillar tops; the
# four wooden blocks lie in a row behind the bridge, two on each side of the car; the plank lies further back.
LAYOUT = dict(bridge_x=(-0.18, -0.15), row_dx=(0.09, 0.11), y_half=0.30, gap=0.095, car_dx=(0.055, 0.09), car_dy=0.03,
              car_clear=0.08, car_yaw=0.35, plank_dx=(0.19, 0.21), block_edge=(0.050, 0.056))
KEEP_OUT = dict(dx=0.13, dy=0.20)  # careful: no parking / no material left within this zone behind the bridge line
# Sonnet-like naive (measured on the lh1 pilot, see task_cards/lh_bridge.md): heights of the grasp site above the table
SN = dict(hover=0.12, carry=0.125, lift=0.15, cross_margin=0.04, plank_lift=0.07, plank_carry=0.16, car_lift=0.17,
          end_lift=0.25, top_below=0.011, diag=True, car_style=None, car_hover=0.134, plank_end_lift=0.21,
          alloc="greedy")
# how the gripper went for the red cube once the plank was down -- each of the three pilot runs did it differently:
#   hover   (seed 3002) move to ~13.4 cm above it while turning to its yaw, then straight down
#   keep    (seed 3000) the same, but still turned 90 deg from laying the plank
#   direct  (seed 3001) one move from ~21 cm above the bridge straight (diagonally) down to the grasp
# the naive script uses style CAR_STYLES[seed % 3] (car_style=None), i.e. the pilot's mixture
CAR_STYLES = ("hover", "keep", "direct")
Z_NAIVE = 0.08  # naive_v1 transit height of the grasp site above the table
Z_SAFE = 0.25  # careful transit height
NAIVE_LIFT = 0.05  # naive: lift this far straight up after letting go, then head for the next waypoint
NAIVE_STYLE = "legs"  # legs: vertical / horizontal legs at the transit height; diag: straight lines between waypoints


def _wrap_q(a):
    """wrap to [-pi/4, pi/4) (cubes are symmetric under 90 deg)"""
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def _wrap_h(a):
    """wrap to [-pi/2, pi/2) (plank / gripper symmetric under 180 deg)"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


# Contact softness of the bridge parts. LFF-Bench's generated boxes use a very stiff contact (objects._SOLID); when a
# fast hand drives into a stacked bridge, the stiff contacts resolve the overlap with large impulses and blocks fly
# 30-90 cm (measured: up to 1.7 m/s while the hand moved at 0.5-0.8 m/s), often out of reach or off the table. A
# softer contact (MuJoCo-like defaults) keeps collapses local. None = use the stiff default.
CONTACT = os.environ.get("LFF_LHB_CONTACT", 'solimp="0.95 0.99 0.001" solref="0.01 1"')


def part_xml(model_name, half, rgba, density):
    """box like objects.box_xml, with the task's contact softness"""
    if not CONTACT or CONTACT == "stiff":
        return box_xml(model_name, half, rgba, density=density)
    g = (f'        <geom name="{model_name}_g0" type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" '
         f'density="{density}" friction="1.0 0.005 0.0001" {CONTACT} group="0" />\n'
         f'        <geom type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" conaffinity="0" contype="0" group="1" />')
    return _wrap(model_name, g, half[2], float(np.hypot(half[0], half[1])))


def mark_xml(name, rgba):
    """flat coloured square, visual only (does not touch the physics)"""
    g = (f'        <geom type="box" size="{SQ_HALF:.5f} {SQ_HALF:.5f} 0.00060" rgba="{_fmt(rgba)}" '
         f'contype="0" conaffinity="0" group="1" />')
    return _wrap(name, g, 0.0006, SQ_HALF * 1.42)


class BridgeSkills(Skills):
    """Skills that feed the task's per-step stage monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class Bridge(LFFTask):
    name = "lh_bridge"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Plan", "Reason", "Restore")
    protocol = "cross"
    max_attempts = 3
    agent_max_steps = 130
    instruction = ("Build a small bridge. First stack two of the wooden blocks on the blue square, then two on the green "
                   "square, to make two pillars; then lay the long plank across the tops of both pillars; finally put "
                   "the red cube on the middle of the plank. The finished bridge must stand on its own, with the "
                   "gripper open and not touching anything.")
    instruction_indirect = ("Build a small bridge. First make a tower of two of the wooden cubes on the blue mark on the "
                            "table, then another tower of two on the green mark; then lay the long flat board across the "
                            "tops of the two towers; finally set the small red cube on the middle of the board. At the "
                            "end everything must stand on its own, with the gripper open and not touching anything.")
    agent_object_names = ["wooden blocks (5)", "plank", "red cube", "blue square", "green square"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs = []
        for i, b in enumerate(BLOCKS):
            cat = register_generated(f"LffLhbBlock{'ABCDE'[i]}", part_xml(f"lff_lhb_block_{'abcde'[i]}", (BLOCK_NOM,) * 3,
                                                                         BLOCK_RGBA[i], 500.0))
            reg = (-0.30 + 0.12 * i, 0.30, -0.29 + 0.12 * i, 0.31) if i < 4 else (0.10, -0.25, 0.11, -0.24)
            objs.append((b, cat, f"blk{i}_region", reg))
        pl = register_generated("LffLhbPlank", part_xml("lff_lhb_plank", PLANK_HALF, (0.50, 0.33, 0.18, 1.0),
                                                       density=600.0))
        objs.append((PLANK, pl, "plank_region", (0.05, -0.01, 0.06, 0.01)))
        car = register_generated("LffLhbCar", part_xml("lff_lhb_car", (CAR_HALF,) * 3, (0.85, 0.10, 0.08, 1.0),
                                                      density=400.0))
        objs.append((CAR, car, "car_region", (-0.30, -0.31, -0.29, -0.30)))
        fx = []
        for i, s in enumerate(SQUARES):
            cat = register_generated(f"LffLhbSquare{'BG'[i]}", mark_xml(f"lff_lhb_square_{'bg'[i]}", SQ_RGBA[s]),
                                     free=False)
            fx.append((s, cat, f"sq{i}_region", (-0.10 + 0.10 * i, -0.31, -0.09 + 0.10 * i, -0.30)))
        return write_bddl(self.name, "Build a small bridge from the blocks the plank and the red cube", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.sq_bid = {s: m.body_name2id(self.env.fixtures_dict[s].root_body) for s in SQUARES}
        self.geoms = {o: envs.obj_geom_ids(self.env, o) for o in ITEMS}
        self.vis_geoms = {}
        for o in BLOCKS:
            root = envs.body_id(self.env, o)
            bodies = {b for b in range(m.nbody) if b == root or m.body_parentid[b] == root}
            self.vis_geoms[o] = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_contype[g] == 0
                                 and m.geom_conaffinity[g] == 0 and m.geom_type[g] == 6]
            self.block_body = getattr(self, "block_body", {})
            self.block_body[o] = int(m.geom_bodyid[self.geoms[o][0]])
        self.table_geoms = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith("main_table")
                            and m.geom_contype[g] != 0]
        self.robot_geoms = set(envs.robot_geom_ids(self.env))
        self.inst = None
        self.mon = None
        self.sk = None
        self.naive_lift = NAIVE_LIFT
        self.naive_style = NAIVE_STYLE

    # ------------------------------------------------------------ instances
    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        xb = float(rng.uniform(*LAYOUT["bridge_x"]))
        yc = float(rng.uniform(-0.03, 0.03))
        blue_left = bool(rng.random() < 0.5)
        sq = {"square_blue_1": [xb, yc + (SPAN / 2 if blue_left else -SPAN / 2)],
              "square_green_1": [xb, yc - (SPAN / 2 if blue_left else -SPAN / 2)]}
        L = LAYOUT
        # the car: in the gap between the two squares, just behind the bridge line
        pos = {CAR: [xb + float(rng.uniform(*L["car_dx"])), yc + float(rng.uniform(-L["car_dy"], L["car_dy"]))]}
        # five blocks in a row behind the bridge line, two on one side of the car and three on the other
        row_x = xb + float(rng.uniform(*L["row_dx"]))
        ys = []
        three = 1.0 if rng.random() < 0.5 else -1.0
        for side in (-1.0, 1.0):
            n = 3 if side == three else 2
            lo, hi = sorted((pos[CAR][1] + side * L["car_clear"], side * L["y_half"]))
            u = np.sort(rng.uniform(0.0, max((hi - lo) - (n - 1) * L["gap"], 0.0), n))
            ys += [lo + k * L["gap"] + u[k] for k in range(n)]
        order = rng.permutation(5)
        for k, j in enumerate(order):
            pos[BLOCKS[j]] = [row_x + float(rng.uniform(-0.01, 0.01)), float(ys[k])]
        plank_xy = [xb + float(rng.uniform(*L["plank_dx"])), yc + float(rng.uniform(-0.04, 0.04))]
        size = {b: float(rng.uniform(*L["block_edge"])) for b in BLOCKS}
        mu = {b: float(rng.uniform(0.3, 0.6)) for b in BLOCKS}
        yaw = {o: float(rng.uniform(-0.35, 0.35)) for o in BLOCKS}
        yaw[CAR] = float(rng.uniform(-L["car_yaw"], L["car_yaw"]))
        yaw[PLANK] = float(np.pi / 2 + rng.uniform(-0.25, 0.25))  # long axis roughly left-right
        pos[PLANK] = plank_xy
        return dict(seed=int(seed), squares=sq, blue_side="left" if blue_left else "right", pos=pos, yaw=yaw,
                    size=size, mu=mu, bridge_x=xb, bridge_yc=yc)

    def _resize_block(self, o, edge):
        m = self.env.sim.model
        h = edge / 2.0
        for g in self.geoms[o] + self.vis_geoms[o]:
            m.geom_size[g] = [h, h, h]
            m.geom_rbound[g] = float(np.sqrt(3) * h)
            m.geom_aabb[g] = [0.0, 0.0, 0.0, h, h, h]
        b = self.block_body[o]
        mass = 500.0 * edge ** 3
        m.body_mass[b] = mass
        m.body_inertia[b] = [mass * edge ** 2 / 6.0] * 3
        envs.refit_bounds(self.env, b)  # BVH of the resized block (envs.refit_bounds)

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        for s in SQUARES:
            x, y = inst["squares"][s]
            m.body_pos[self.sq_bid[s]] = [x, y, TZ + 0.0007]
            m.body_quat[self.sq_bid[s]] = [1.0, 0.0, 0.0, 0.0]
        for o in BLOCKS:
            self._resize_block(o, inst["size"][o])
            envs.set_friction(self.env, o, inst["mu"][o])
        envs.set_friction(self.env, PLANK, 0.5)
        envs.set_friction(self.env, CAR, 0.6)
        self.env.sim.forward()
        for o in ITEMS:
            envs.place_upright(self.env, o, inst["pos"][o], inst["yaw"][o], clearance=0.001)

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = BridgeSkills(self.env, self, recorder=recorder)
        self.mon = dict(state=[False] * 4, raw_since=[0] * 4, ever=[False] * 4, lost=[False] * 4, events=[])
        return self.sk

    # ------------------------------------------------------------ geometry / predicates
    def half(self, o):
        if o in BLOCKS:
            e = self.inst["size"][o] / 2
            return np.array([e, e, e])
        return np.array(PLANK_HALF) if o == PLANK else np.array([CAR_HALF] * 3)

    def R(self, o):
        return self.env.sim.data.body_xmat[envs.body_id(self.env, o)].reshape(3, 3).copy()

    def pos(self, o):
        return envs.obj_pos(self.env, o)

    def zmin(self, o):
        return envs.obj_min_z(self.env, o)

    def zmax(self, o):
        return envs.obj_max_z(self.env, o)

    def flat(self, o):
        return float(np.max(np.abs(self.R(o)[2]))) > FLAT_COS

    def yaw_of(self, o):
        """yaw of the object's body x axis (for the plank: its long axis)"""
        R = self.R(o)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def held(self, o):
        return self.sk is not None and self.sk.holding(o)

    def contacts(self, a_geoms, b_geoms):
        return envs.contacts_between(self.env, a_geoms, b_geoms)

    def pillar(self, s):
        """(lower, upper) blocks of a complete pillar on square s, else (lower or None, None)"""
        c = np.asarray(self.inst["squares"][s])
        lower = None
        for b in BLOCKS:
            p = self.pos(b)
            if np.linalg.norm(p[:2] - c) <= SQ_TOL and self.zmin(b) < TZ + 0.006 and self.flat(b) and not self.held(b):
                lower = b
                break
        if lower is None:
            return None, None
        pl = self.pos(lower)
        top = self.zmax(lower)
        for b in BLOCKS:
            if b == lower:
                continue
            p = self.pos(b)
            if (np.linalg.norm(p[:2] - pl[:2]) <= STACK_TOL and abs(self.zmin(b) - top) < 0.006 and self.flat(b)
                    and not self.held(b)):
                return lower, b
        return lower, None

    def plank_ok(self, tops):
        if any(t is None for t in tops) or self.held(PLANK):
            return False
        R = self.R(PLANK)
        if abs(R[2, 0]) > np.sin(PLANK_TILT) or abs(R[2, 2]) < np.cos(PLANK_TILT):  # either broad face up (v2 fix)
            return False
        if self.zmin(PLANK) < TZ + 0.03 or self.contacts(self.geoms[PLANK], self.table_geoms):
            return False
        return all(self.contacts(self.geoms[PLANK], self.geoms[t]) for t in tops)

    def car_ok(self):
        """the car rests on the plank's upper face (in the plank's own frame, so a slightly tilted plank is fine),
        within CAR_MID of the plank's centre along its length"""
        if self.held(CAR) or not self.contacts(self.geoms[CAR], self.geoms[PLANK]):
            return False
        R = self.R(PLANK)
        rel = R.T @ (self.pos(CAR) - self.pos(PLANK))
        up = rel[2] * (1.0 if R[2, 2] >= 0 else -1.0)  # along the broad face pointing up (a plank may lie upside down)
        return bool(abs(rel[0]) <= CAR_MID and abs(rel[1]) <= PLANK_HALF[1] + 0.01 and up > PLANK_HALF[2] + 0.01)

    def stages(self):
        """raw predicate values [S1, S2, S3, S4] and the pillars found"""
        pa = self.pillar(SQUARES[0])
        pb = self.pillar(SQUARES[1])
        s1, s2 = pa[1] is not None, pb[1] is not None
        s3 = bool(s1 and s2 and self.plank_ok([pa[1], pb[1]]))
        s4 = bool(self.car_ok())
        return [s1, s2, s3, s4], (pa, pb)

    # ------------------------------------------------------------ monitor
    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        raw, _ = self.stages()
        t = self.sk.n_steps
        for i in range(4):
            if raw[i] == mon["state"][i]:
                mon["raw_since"][i] = 0
                continue
            mon["raw_since"][i] += 1
            if mon["raw_since"][i] < DEBOUNCE:
                continue
            mon["state"][i] = raw[i]
            mon["raw_since"][i] = 0
            t0 = t - DEBOUNCE + 1
            if raw[i]:
                ev = "restored" if mon["lost"][i] else "achieved"
                mon["ever"][i] = True
            else:
                ev = "lost"
                mon["lost"][i] = True
            mon["events"].append(dict(stage=f"S{i + 1}", event=ev, step=int(t0)))

    # ------------------------------------------------------------ outcome
    def _dir_words(self, d):
        """robot-centric words for a horizontal offset d (world x = forward, away from the robot; y = robot's left)"""
        parts = []
        if abs(d[0]) >= 0.01:
            parts.append(f"{100 * abs(d[0]):.0f} cm {'forward' if d[0] > 0 else 'back'}")
        if abs(d[1]) >= 0.01:
            parts.append(f"{100 * abs(d[1]):.0f} cm {'left' if d[1] > 0 else 'right'}")
        return (" and ".join(parts) + " of") if parts else "right at"

    def outcome(self, inst):
        sk = self.sk
        sk.hold(10)
        p0 = {o: self.pos(o) for o in ITEMS}
        touching = False
        for _ in range(20):
            sk.hold(1)
            touching = touching or any(self.contacts(list(self.robot_geoms), self.geoms[o]) for o in ITEMS)
        moved = max(float(np.linalg.norm(self.pos(o) - p0[o])) for o in ITEMS)
        static = moved < 0.004
        raw, (pa, pb) = self.stages()
        held = [o for o in ITEMS if sk.holding(o)]
        mon = self.mon
        ev = list(mon["events"]) if mon else []
        n_reg = sum(e["event"] == "lost" for e in ev)
        n_res = sum(e["event"] == "restored" for e in ev)
        success = bool(all(raw) and static and not held and not touching)
        parts = []
        if success:
            failure = "none"
            parts.append("The bridge is complete: both pillars stand on their squares, the plank rests across them "
                         "and the red cube sits on the middle of the plank.")
        else:
            lost_final = [i for i in range(4) if not raw[i] and mon and mon["ever"][i]]
            never = [i for i in range(4) if not raw[i] and not (mon and mon["ever"][i])]
            if held:
                failure = "held"
            elif touching:
                failure = "touching"
            elif lost_final:
                failure = "regressed_" + "".join(f"S{i + 1}" for i in lost_final)
            elif never:
                failure = "incomplete_" + "".join(f"S{i + 1}" for i in never)
            else:
                failure = "not_static"
            for i, s in enumerate(SQUARES):
                if raw[i]:
                    continue
                low, _ = (pa, pb)[i]
                c = np.asarray(inst["squares"][s])
                txt = f"The pillar on the {SQ_NAME[s]} square is not standing:"
                if low is None:
                    txt += " no block stands on the square."
                else:
                    txt += " only one block stands on the square."
                strays = [b for b in BLOCKS if b not in (pa[0], pa[1], pb[0], pb[1])]
                if strays:
                    near = min(strays, key=lambda b: np.linalg.norm(self.pos(b)[:2] - c))
                    txt += f" The nearest loose block lies {self._dir_words(self.pos(near)[:2] - c)} the square."
                if mon and mon["ever"][i]:
                    txt += " It had been built and was knocked down later."
                parts.append(txt)
            if not raw[2]:
                pp = self.pos(PLANK)
                if self.contacts(self.geoms[PLANK], self.table_geoms):
                    where = "touches the table"
                elif self.held(PLANK):
                    where = "is still in the gripper"
                else:
                    where = "does not rest level on the tops of both pillars"
                bc = np.mean([inst["squares"][s] for s in SQUARES], axis=0)
                parts.append(f"The plank {where}; its centre is {self._dir_words(pp[:2] - bc)} the middle between "
                             f"the squares." + (" It had been laid and was disturbed later." if mon and mon["ever"][2]
                                                and not raw[2] else ""))
            if not raw[3]:
                if self.held(CAR):
                    parts.append("The red cube is still in the gripper.")
                elif self.contacts(self.geoms[CAR], self.geoms[PLANK]):
                    parts.append("The red cube is on the plank but not on its middle part.")
                else:
                    bc = np.mean([inst["squares"][s] for s in SQUARES], axis=0)
                    parts.append(f"The red cube is not on the plank; it lies {self._dir_words(self.pos(CAR)[:2] - bc)} "
                                 "the middle between the squares.")
            if all(raw) and held:
                parts.append("The gripper still holds an object.")
            elif touching:
                parts.append("The gripper is still touching the bridge.")
            elif all(raw) and not static:
                parts.append("The bridge was still moving.")
        if n_reg:
            parts.append(f"During the attempt earlier stages were undone {n_reg} time(s) and restored {n_res} time(s).")
        if any(w in " ".join(parts) for w in (" forward of", " back of", " back and", " forward and")):
            parts.append("(Forward = away from the robot.)")
        return dict(success=success, stages={f"S{i + 1}": bool(raw[i]) for i in range(4)}, events=ev,
                    n_regressions=int(n_reg), n_restored=int(n_res), failure=failure, held=held,
                    touching=bool(touching), static=bool(static), detail=" ".join(parts))

    # ------------------------------------------------------------ scripted policies (move_to + gripper only)
    def _grip_yaws(self, o):
        """candidate gripper yaws for a top-down grasp of o (fingers across a face)"""
        if o == PLANK:  # fingers (which open along the gripper's y at yaw 0) across the plank's width
            return [_wrap_h(self.yaw_of(PLANK))]
        a = _wrap_q(self.yaw_of(o))
        return [a, _wrap_h(a + np.pi / 2)]

    def footprint(self, o):
        """sample points on the horizontal footprint of o (corners, edge midpoints, centre)"""
        R, c, h = self.R(o), self.pos(o), self.half(o)
        pts = []
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in ((-1, 1) if o != PLANK else (0,)):
                    pts.append((c + R @ (h * [sx, sy, sz]))[:2])
        if o == PLANK:
            for t_ in np.linspace(-1, 1, 9):
                for sy in (-1, 1):
                    pts.append((c + R @ (h * [t_, sy, 0]))[:2])
        return np.array(pts)

    def _hand_clearance(self, xy, yaw, z_site, ignore=()):
        """smallest horizontal clearance (m) between the gripper at a grasp-site height z_site and everything else:
        the hand body (+-10.6 x +-3.6 cm, underside 2.4 cm above the site) against items taller than its underside,
        the open fingers (two 2 x 2.4 cm boxes 3-5 cm out along the finger axis, tips 1.3 cm below the site)
        against items taller than the fingertips. Negative = the gripper would land on something."""
        f = np.array([-np.sin(yaw), np.cos(yaw)])  # finger axis
        g = np.array([np.cos(yaw), np.sin(yaw)])
        best = 1.0
        for o in ITEMS:
            if o in ignore or self.held(o):
                continue
            top = self.zmax(o)
            if top < z_site - 0.013:
                continue
            d = self.footprint(o) - np.asarray(xy)
            u, v = np.abs(d @ f), np.abs(d @ g)
            c = float(np.min(np.maximum(np.abs(u - 0.040) - 0.010, v - 0.012)))  # fingers
            if top >= z_site + HAND_BOTTOM - 0.002:
                c = min(c, float(np.min(np.maximum(u - HAND_HALF_LONG, v - HAND_HALF_SHORT))))
            best = min(best, c)
        return best

    @staticmethod
    def _pick_yaw(yaws, clearance):
        """the gripper yaw closest to straight (0) among those with >= 1 cm clearance, else the clearest one (large
        wrist turns near the edge of the workspace make the OSC stop short)"""
        c = {a: clearance(a) for a in yaws}
        ok = [a for a in yaws if c[a] >= 0.01]
        return min(ok, key=abs) if ok else max(yaws, key=c.get)

    def _go(self, xyz, yaw=None, speed=None, tol=0.004, max_steps=250):
        return self.sk.move_to(xyz, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)

    def _grasp_z(self, o, careful):
        if o in BLOCKS:
            return self.zmax(o) - (0.016 if careful else self.inst["size"][o] / 2)
        if o == PLANK:
            return max(self.zmax(PLANK) + 0.0003 - 0.0125, TZ + 0.011)
        return self.zmax(CAR) - (0.016 if careful else CAR_HALF)

    def pick(self, o, careful, noise=None, z_t=None):
        """top-down grasp; returns the drop of the object's bottom below the grasp site, or None if not held"""
        sk = self.sk
        p = self.pos(o)
        if noise is not None:
            p = p + np.array([noise[0], noise[1], 0.0])
        zg = self._grasp_z(o, careful)
        yaws = self._grip_yaws(o)
        if careful:
            yaw = self._pick_yaw(yaws, lambda a: self._hand_clearance(p[:2], a, zg, ignore=(o,)))
            cur = sk.eef_pos()
            self._go([cur[0], cur[1], TZ + Z_SAFE])
            self._go([p[0], p[1], TZ + Z_SAFE], yaw=yaw)
            sk.set_gripper(False, steps=8)
            self._go([p[0], p[1], zg + 0.03], speed=0.25)
            self._go([p[0], p[1], zg], speed=0.1, tol=0.003, max_steps=120)
        else:
            yaw = min(yaws, key=lambda a: abs(a))
            zt = TZ + (Z_NAIVE if z_t is None else z_t)
            sk.set_gripper(False, steps=8)
            if self.naive_style == "legs":  # rise to the transit height if below it, across, straight down
                cur = sk.eef_pos()
                if cur[2] < zt - 0.003:
                    self._go([cur[0], cur[1], zt])
                self._go([p[0], p[1], max(cur[2], zt)], yaw=yaw)
            else:  # one straight line to the approach point above the object
                self._go([p[0], p[1], max(zt, self.zmax(o) + 0.03)], yaw=yaw)
            self._go([p[0], p[1], zg], tol=0.003, max_steps=120)
        sk.set_gripper(True, steps=15)
        drop = float(sk.eef_pos()[2] - self.zmin(o))
        if careful:
            self._go([p[0], p[1], TZ + Z_SAFE], speed=0.3)
        else:
            zt = TZ + (Z_NAIVE if z_t is None else z_t)
            self._go([p[0], p[1], max(zt, zg + 0.05)])
        if not sk.holding(o):
            sk.set_gripper(False, steps=8)
            return None
        return drop

    def place(self, o, xy, support_top, drop, yaw, careful, z_t=None, release_gap=0.005, lift=None):
        sk = self.sk
        zr = support_top + drop + release_gap
        if careful and o != PLANK:  # cubes are symmetric under 90 deg: release with the fingers where there is room
            yaw = self._pick_yaw([yaw, _wrap_h(yaw + np.pi / 2)], lambda a: self._hand_clearance(xy, a, zr, ignore=(o,)))
        if careful:
            cur = sk.eef_pos()
            self._go([cur[0], cur[1], TZ + Z_SAFE])
            self._go([xy[0], xy[1], TZ + Z_SAFE], yaw=yaw)
            self._go([xy[0], xy[1], zr + 0.03], speed=0.25)
            self._go([xy[0], xy[1], zr], speed=0.08, tol=0.003, max_steps=120)
            sk.set_gripper(False, steps=12)
            self._go([xy[0], xy[1], zr + 0.03], speed=0.1)
            self._go([xy[0], xy[1], TZ + Z_SAFE], speed=0.3)
        else:
            if self.naive_style == "legs":
                zt = TZ + (Z_NAIVE if z_t is None else z_t)
                cur = sk.eef_pos()
                h = max(cur[2], zt, zr + 0.04)
                if cur[2] < h - 0.003:
                    self._go([cur[0], cur[1], h])
                self._go([xy[0], xy[1], h], yaw=yaw)
            else:
                self._go([xy[0], xy[1], zr + 0.04], yaw=yaw)
            self._go([xy[0], xy[1], zr], tol=0.003, max_steps=120)
            sk.set_gripper(False, steps=12)
            self._go([xy[0], xy[1], zr + (self.naive_lift if lift is None else lift)])
        sk.hold(5)

    # placing targets ---------------------------------------------------------
    def square_xy(self, s):
        return np.asarray(self.inst["squares"][s], float)

    def bridge_mid(self):
        return np.mean([self.square_xy(s) for s in SQUARES], axis=0)

    def free_blocks(self):
        _, (pa, pb) = self.stages()
        used = set()
        for low, up in (pa, pb):
            if up is not None:
                used |= {low, up}
        return [b for b in BLOCKS if b not in used]

    def _blocker(self, b):
        """for a careful top-down grasp of block b: None if the gripper fits at some grasp yaw, else the item in the
        way (e.g. the top block of a pillar, or a block, right next to b)"""
        zg = self._grasp_z(b, True)
        p = self.pos(b)[:2]
        if max(self._hand_clearance(p, a, zg, ignore=(b,)) for a in self._grip_yaws(b)) > -0.002:
            return None
        worst = {}
        for o in ITEMS:
            if o == b or self.held(o):
                continue
            worst[o] = max(self._hand_clearance(p, a, zg, ignore=tuple(q for q in ITEMS if q != o))
                           for a in self._grip_yaws(b))
        o = min(worst, key=worst.get)
        return o if worst[o] <= -0.002 else None

    def build_pillar(self, s, careful, noise_fn, z_t=None, order="target"):
        """put blocks on square s until it carries a complete pillar (at most 5 placements)"""
        failed = {}
        for _ in range(6):
            raw, (pa, pb) = self.stages()
            low, up = (pa, pb)[SQUARES.index(s)]
            if up is not None:
                return True
            if careful and self.over_budget():
                return False
            c = self.square_xy(s)
            other = (pa, pb)[1 - SQUARES.index(s)]
            cand = [b for b in BLOCKS if b != low and b not in other and (not careful or (self.usable(b)
                                                                                       and failed.get(b, 0) < 2))]
            if not cand:
                return False
            ref = self.sk.eef_pos()[:2] if order == "gripper" else c
            cand.sort(key=lambda b: (failed.get(b, 0), np.linalg.norm(self.pos(b)[:2] - ref)))
            b = cand[0]
            if careful:
                free = [q for q in cand if self._blocker(q) is None]
                if free:
                    b = free[0]
                else:  # every usable block sits right next to something tall: take that thing's top off first
                    blk = self._blocker(cand[0])
                    if blk is not None and blk != PLANK and failed.get(blk, 0) < 2:
                        if not self.park(blk):
                            failed[blk] = failed.get(blk, 0) + 1
                        continue
            drop = self.pick(b, careful, noise=noise_fn(), z_t=z_t)
            if drop is None:
                failed[b] = failed.get(b, 0) + 1
                continue
            if low is None:
                tgt, top = c, TZ
            else:
                tgt, top = self.pos(low)[:2], self.zmax(low)
            n = noise_fn()
            self.place(b, tgt + (n if n is not None else 0.0), top, drop, 0.0, careful, z_t=z_t)
        return self.stages()[1][SQUARES.index(s)][1] is not None

    def lay_plank(self, careful, noise_fn, z_t=None):
        raw, (pa, pb) = self.stages()
        if pa[1] is None or pb[1] is None or (careful and not self.usable(PLANK)):
            return False
        drop = self.pick(PLANK, careful, noise=noise_fn(), z_t=z_t)
        if drop is None:
            return False
        raw, (pa, pb) = self.stages()
        if pa[1] is None or pb[1] is None:
            tops = [TZ + 0.11, TZ + 0.11]
            mid = self.bridge_mid()
        else:
            tops = [self.zmax(pa[1]), self.zmax(pb[1])]
            mid = (self.pos(pa[1])[:2] + self.pos(pb[1])[:2]) / 2
        n = noise_fn()
        # gripper fingers across the plank's width: plank long axis along y -> gripper yaw 90 deg
        self.place(PLANK, mid + (n if n is not None else 0.0), max(tops), drop, np.pi / 2, careful, z_t=z_t)
        return True

    def put_car(self, careful, noise_fn, z_t=None):
        drop = self.pick(CAR, careful, noise=noise_fn(), z_t=z_t)
        if drop is None:
            return False
        pp = self.pos(PLANK)
        n = noise_fn()
        self.place(CAR, pp[:2] + (n if n is not None else 0.0), self.zmax(PLANK), drop, 0.0, careful, z_t=z_t)
        return True

    def in_keep_out(self, o):
        """o lies in the zone right behind the bridge line where, once the pillars and the plank stand, a top-down
        grasp would put the hand on them (the careful policy keeps it clear)"""
        p = self.pos(o)[:2]
        dx = p[0] - self.inst["bridge_x"]
        return bool(-0.08 < dx < KEEP_OUT["dx"] and abs(p[1] - self.bridge_mid()[1]) < KEEP_OUT["dy"])

    def free_spot(self, o, avoid=()):
        """a free, comfortably reachable table spot for parking object o (the plank lies left-right), outside the
        keep-out zone right behind the bridge and away from the squares"""
        xb = self.inst["bridge_x"]
        ym = self.bridge_mid()[1]
        best, best_c = None, -1.0
        others = [q for q in ITEMS if q != o and q not in avoid]
        for x in np.arange(xb - 0.14, xb + 0.221, 0.02):
            for y in np.arange(-0.28, 0.281, 0.02):
                pts = [np.array([x, y])]
                if o == PLANK:
                    pts = [np.array([x, y + d]) for d in (-0.10, 0.0, 0.10)]
                if any(not self.reachable_xy(p) or np.hypot(p[0] - BASE_X, p[1]) > 0.64 for p in pts):
                    continue  # (a careful 25 cm transit further out straightens the elbow onto its limit, measured)
                if any(-0.08 < p[0] - xb < KEEP_OUT["dx"] and abs(p[1] - ym) < KEEP_OUT["dy"] for p in pts):
                    continue
                c = 1.0
                for p in pts:
                    for q in others:
                        c = min(c, float(np.linalg.norm(self.pos(q)[:2] - p)) - 0.04)
                    for sq in SQUARES:
                        c = min(c, float(np.linalg.norm(self.square_xy(sq) - p)) - 0.07)
                if c > best_c:
                    best, best_c = np.array([x, y]), c
        return best

    @staticmethod
    def reachable_xy(p):
        """a top-down grasp at table height works here (measured: right in front of the base, closer than 25 cm,
        the arm cannot get down to the table; at 70-74 cm it still gets within ~1 cm)"""
        rx, y = p[0] - BASE_X, p[1]
        return bool(0.25 <= rx <= 0.75 and abs(y) <= 0.34 and (rx >= 0.30 or abs(y) >= 0.15) and np.hypot(rx, y) <= 0.76)

    def usable(self, o):
        """o lies on the table (or on something) within reach"""
        return self.zmin(o) > TZ - 0.02 and self.reachable_xy(self.pos(o)[:2])

    def park(self, o, careful=True):
        if careful and not self.usable(o):
            return False
        drop = self.pick(o, careful)
        if drop is None:
            return False
        xy = self.free_spot(o)
        self.place(o, xy, TZ, drop, np.pi / 2 if o == PLANK else 0.0, careful)
        return True

    def over_budget(self):
        """the scripted repair stops where an agent's attempt would end: 3 s of simulated motion per command of the
        task's command budget (agent_max_steps = 130 -> 390 s = 7800 control steps)"""
        return self.sk.n_steps * 0.05 >= 3.0 * self.agent_max_steps

    def plank_flat(self):
        return abs(self.R(PLANK)[2, 2]) > 0.8

    def fix_plank(self):
        """a plank standing on its long edge or leaning on something cannot be picked from above: push its upper edge
        over with the closed gripper so that it falls flat (towards the side with more free table)"""
        if self.plank_flat() or self.held(PLANK):
            return False
        sk = self.sk
        R, c = self.R(PLANK), self.pos(PLANK)
        up = R[:, 1] if R[2, 1] > 0 else -R[:, 1]  # the plank's width axis, pointing up
        lean = up[:2] / max(np.linalg.norm(up[:2]), 1e-6)
        if np.linalg.norm(up[:2]) < 0.2:  # standing upright: fall to the side away from the squares
            n = R[:2, 2] / max(np.linalg.norm(R[:2, 2]), 1e-6)
            far = lambda d: min(np.linalg.norm(c[:2] + 0.05 * d - self.square_xy(q)) for q in SQUARES)  # noqa: E731
            lean = n if far(n) > far(-n) else -n
            push = lean
        else:  # leaning: push the top back over the vertical so it falls to the other side
            push = -lean
        top = c + up * 0.02
        h = max(self.zmax(PLANK) - 0.015, TZ + 0.02)
        a = top[:2] - push * 0.05
        yaw = float(np.arctan2(push[1], push[0]) - np.pi / 2)  # closed fingers across the push direction
        sk.set_gripper(True, steps=8)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], TZ + Z_SAFE])
        self._go([a[0], a[1], TZ + Z_SAFE], yaw=yaw)
        self._go([a[0], a[1], h], speed=0.25)
        b = top[:2] + push * 0.08
        self._go([b[0], b[1], h], speed=0.1, max_steps=200)
        self._go([b[0], b[1], TZ + Z_SAFE], speed=0.3)
        sk.set_gripper(False, steps=8)
        sk.hold(10)
        return True

    def clear_site(self):
        """careful: before building, move the car out of the zone right behind the bridge line"""
        if self.in_keep_out(CAR) and not self.held(CAR):
            self.park(CAR)

    def repair(self, rounds=4):
        """check every stage; rebuild what is broken (careful motions). Returns the number of fixes made."""
        nofn = lambda: None  # noqa: E731
        fixes = 0
        for _ in range(rounds):
            raw, (pa, pb) = self.stages()
            if all(raw) or self.over_budget() or not (self.usable(PLANK) or self.held(PLANK)):
                return fixes  # (a plank thrown out of reach cannot be laid again)
            fixes += 1
            if not (raw[0] and raw[1]):
                for _k in range(2):
                    if not self.plank_flat():
                        self.fix_plank()
                plank_first = self._hand_clearance(self.pos(CAR)[:2], 0.0, self._grasp_z(CAR, True),
                                                   ignore=tuple(q for q in ITEMS if q != PLANK)) < 0.0
                for o in ((PLANK, CAR) if plank_first else (CAR, PLANK)):
                    if o == CAR and (self.zmin(CAR) > TZ + 0.01 or self._near_squares(CAR) or self.in_keep_out(CAR)):
                        self.park(CAR)
                    if o == PLANK and (self.zmin(PLANK) > TZ + 0.006 or self._near_squares(PLANK)
                                       or self.in_keep_out(PLANK)):
                        self.park(PLANK)
                for i, s in enumerate(SQUARES):
                    if not self.stages()[0][i]:
                        self.build_pillar(s, True, nofn)
            raw, _ = self.stages()
            if raw[0] and raw[1] and not raw[2]:
                if not self.plank_flat():
                    self.fix_plank()
                if self.zmin(CAR) > TZ + 0.01 or self.in_keep_out(CAR):
                    self.park(CAR)
                self.lay_plank(True, nofn)
            raw, _ = self.stages()
            if raw[2] and not raw[3]:
                self.put_car(True, nofn)
        return fixes

    def _near_squares(self, o):
        p = self.pos(o)[:2]
        if o == PLANK:
            ax = self.R(PLANK)[:2, 0]
            pts = [p + s * ax * 0.11 for s in (-1, -0.5, 0, 0.5, 1)]
        else:
            pts = [p]
        return any(np.linalg.norm(q - self.square_xy(s)) < 0.06 for q in pts for s in SQUARES)

    # ------------------------------------------------------------ Sonnet-like naive (style "sonnet")
    # Imitates how Claude Sonnet 5.5 moved in the lh1 pilot (runs/claude/lh1_bridge, 4 attempts, 230 commands):
    # instruction order; for each pillar the free block nearest to its square first; a hover ~12 cm above the object
    # with the gripper turned to the object's yaw, a straight vertical descent, grasp ~1 cm below the top; the carry
    # often goes straight from the grasp to ~12.5 cm above the target (one move_eef: a diagonal climb), otherwise lift
    # first; when the carried object's straight path would run into a built pillar it lifts first and carries with the
    # object's bottom ~4 cm above that pillar; vertical descent, open, lift straight up to ~15 cm; plank: lift to 7 cm,
    # carry at 16 cm; car: lift to 17 cm, carry, place, lift to 25 cm; no final check.
    def _sn_grasp_z(self, o, sn):
        if o == PLANK:
            return self._grasp_z(PLANK, False)
        return self.zmax(o) - sn["top_below"]

    def _crossing_top(self, a, b, o, drop, hc):
        """top of the highest item that the held object o, carried straight from a to b with its bottom at hc - drop,
        would run into (any footprint point within o's half size + 1 cm of the path), else None"""
        a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
        d = b - a
        n2 = float(d @ d)
        r = float(self.half(o)[0]) + 0.01
        best = None
        for q in ITEMS:
            if q == o or self.held(q):
                continue
            top = self.zmax(q)
            if top < hc - drop:
                continue
            pts = self.footprint(q)
            t = np.clip((pts - a) @ d / n2, 0.0, 1.0) if n2 > 1e-9 else np.zeros(len(pts))
            dist = float(np.min(np.linalg.norm(pts - (a + t[:, None] * d), axis=1)))
            if dist < r and (best is None or top > best):
                best = top
        return best

    def sn_pick(self, o, sn, noise=None):
        sk = self.sk
        p = self.pos(o)
        if noise is not None:
            p = p + np.array([noise[0], noise[1], 0.0])
        zg = self._sn_grasp_z(o, sn)
        if o == PLANK:
            yaw = _wrap_h(self.yaw_of(PLANK))
        elif o == CAR and sn["car_style"] == "keep":
            yaw = None  # keeps the gripper's current yaw (the pilot's seed 3000: still turned 90 deg from the plank)
        else:
            yaw = _wrap_q(self.yaw_of(o))
        hover = TZ + (sn["car_hover"] if o == CAR else sn["hover"])
        direct = o == CAR and sn["car_style"] == "direct"  # (the pilot's seed 3001 went straight down to the car)
        if not direct:
            self._go([p[0], p[1], hover], yaw=yaw)
        self._go([p[0], p[1], zg], yaw=yaw, tol=0.003, max_steps=250 if direct else 120)
        sk.set_gripper(True, steps=15)
        if not sk.holding(o):
            sk.set_gripper(False, steps=15)
            self._go([p[0], p[1], TZ + sn["hover"]])
            return None
        return float(sk.eef_pos()[2] - self.zmin(o))

    def sn_place(self, o, xy, support_top, drop, yaw, sn, carry=None, lift=None, diag=None):
        sk = self.sk
        zr = support_top + drop + 0.004
        hc = max(TZ + (sn["carry"] if carry is None else carry), zr + 0.03)
        cur = sk.eef_pos()
        diag = sn["diag"] if diag is None else diag
        top = None if o == PLANK else self._crossing_top(cur, xy, o, drop, hc)
        if top is not None:  # a built pillar in the way: lift first, carry with the object's bottom above it
            hc = max(hc, top + drop + sn["cross_margin"])
            self._go([cur[0], cur[1], hc])
            self._go([xy[0], xy[1], hc], yaw=yaw)
        elif diag:  # one move_eef from the grasp to above the target
            self._go([xy[0], xy[1], hc], yaw=yaw)
        else:
            self._go([cur[0], cur[1], max(cur[2], hc + 0.02)])
            self._go([xy[0], xy[1], hc], yaw=yaw)
        self._go([xy[0], xy[1], zr], tol=0.003, max_steps=120)
        sk.set_gripper(False, steps=15)
        self._go([xy[0], xy[1], TZ + (sn["lift"] if lift is None else lift)])
        sk.hold(5)

    def sn_build_pillar(self, s, sn, noise_fn):
        failed = {}
        c = self.square_xy(s)
        mid = self.bridge_mid()
        for _ in range(5):
            raw, (pa, pb) = self.stages()
            low, up = (pa, pb)[SQUARES.index(s)]
            if up is not None:
                return True
            other = (pa, pb)[1 - SQUARES.index(s)]
            cand = [b for b in BLOCKS if b != low and b not in other and failed.get(b, 0) < 2]
            if not cand:
                return False
            if sn["alloc"] == "side":  # blocks on this square's side of the bridge first
                side = np.sign(c[1] - mid[1])
                same = [b for b in cand if np.sign(self.pos(b)[1] - mid[1]) == side]
                cand = same or cand
            b = min(cand, key=lambda q: (failed.get(q, 0), np.linalg.norm(self.pos(q)[:2] - c)))
            drop = self.sn_pick(b, sn, noise=noise_fn())
            if drop is None:
                failed[b] = failed.get(b, 0) + 1
                continue
            tgt, top = (c, TZ) if low is None else (self.pos(low)[:2], self.zmax(low))
            n = noise_fn()
            # onto the bare square: straight from the grasp (diagonal climb); onto a block: lift first (a diagonal
            # climb from nearby clips the lower block -- the pilot's seed 3000 did that once, then lifted first)
            self.sn_place(b, tgt + (n if n is not None else 0.0), top, drop, 0.0, sn,
                          diag=None if low is None else False)
        return self.stages()[1][SQUARES.index(s)][1] is not None

    def sn_lay_plank(self, sn, noise_fn):
        raw, (pa, pb) = self.stages()
        if pa[1] is None or pb[1] is None:
            return False
        drop = self.sn_pick(PLANK, sn, noise=noise_fn())
        if drop is None:
            return False
        cur = self.sk.eef_pos()
        self._go([cur[0], cur[1], TZ + sn["plank_lift"]])
        raw, (pa, pb) = self.stages()
        if pa[1] is None or pb[1] is None:
            tops, mid = [TZ + 0.11, TZ + 0.11], self.bridge_mid()
        else:
            tops = [self.zmax(pa[1]), self.zmax(pb[1])]
            mid = (self.pos(pa[1])[:2] + self.pos(pb[1])[:2]) / 2
        n = noise_fn()
        mid = mid + (n if n is not None else 0.0)
        self.sn_place(PLANK, mid, max(tops), drop, np.pi / 2, sn, carry=sn["plank_carry"], lift=sn["plank_end_lift"],
                      diag=True)
        return True

    def sn_put_car(self, sn, noise_fn):
        drop = self.sn_pick(CAR, sn, noise=noise_fn())
        if drop is None:  # (in the pilot, Sonnet stopped here when it saw the bridge come down)
            return False
        n = noise_fn()
        pp = self.pos(PLANK)[:2] + (n if n is not None else 0.0)
        self.sn_place(CAR, pp, self.zmax(PLANK), drop, None, sn, carry=sn["car_lift"], lift=sn["end_lift"],
                      diag=False)
        return True

    # ------------------------------------------------------------ execution
    def execute(self, inst, params):
        mode = params.get("mode", "naive")
        careful = mode == "careful"
        rng = np.random.default_rng(params.get("noise_seed", 0))
        sig = float(params.get("noise", 0.0))
        noise_fn = (lambda: rng.uniform(-sig, sig, 2)) if sig > 0 else (lambda: None)
        z_t = params.get("z_t")
        self.naive_lift = float(params.get("lift", NAIVE_LIFT))
        style = params.get("style", "sonnet")
        if not careful and style == "sonnet":
            sn = dict(SN)
            sn.update(params.get("sn", {}))
            if sn["car_style"] is None:
                sn["car_style"] = CAR_STYLES[int(inst["seed"]) % 3]
            for s in SQUARES:
                self.sn_build_pillar(s, sn, noise_fn)
            self.sn_lay_plank(sn, noise_fn)
            self.sn_put_car(sn, noise_fn)
        else:  # careful, or the v1 naive ("legs": vertical / horizontal legs at 8 cm; "diag")
            self.naive_style = style if style in ("legs", "diag") else NAIVE_STYLE
            if careful:
                self.clear_site()
            for s in SQUARES:
                self.build_pillar(s, careful, noise_fn, z_t=z_t, order=params.get("order", "target"))
            self.lay_plank(careful, noise_fn, z_t=z_t)
            self.put_car(careful, noise_fn, z_t=z_t)
        if params.get("check"):
            self.repair()
        cur = self.sk.eef_pos()
        if cur[2] < TZ + 0.20:
            self._go([cur[0], cur[1], cur[2] + 0.06], speed=0.15)

    def default_params(self, inst):
        return dict(mode="naive", style="sonnet", check=False)

    def oracle_params(self, inst):
        return dict(mode="careful", check=True)

    def adapt_params(self, inst, history):
        return dict(mode="careful", check=True)

    def blind_params(self, inst, attempt, rng):
        sn = dict(hover=float(rng.uniform(0.10, 0.14)), carry=float(rng.uniform(0.11, 0.15)),
                  lift=float(rng.uniform(0.13, 0.20)), diag=bool(rng.random() < 0.5),
                  car_style=CAR_STYLES[int(rng.integers(3))], alloc="side" if rng.random() < 0.5 else "greedy")
        return dict(mode="naive", style="sonnet", check=False, sn=sn, noise=0.004, noise_seed=int(rng.integers(1 << 30)))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """kinds: oracle, naive (Sonnet-like, car approach style by seed), adaptive, blind; for comparison naive_v1 (the
        v1 naive: vertical / horizontal legs at 8 cm) and naive_hover / naive_keep / naive_direct (Sonnet-like with
        one fixed car approach style)"""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "naive_v1":
                params = dict(mode="naive", style="legs", check=False)
            elif kind.startswith("naive_") and kind[6:] in CAR_STYLES:  # naive_hover / naive_keep / naive_direct
                params = dict(mode="naive", style="sonnet", check=False, sn=dict(car_style=kind[6:]))
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
            if out["success"] or kind == "oracle" or kind.startswith("naive"):
                break
        return history
