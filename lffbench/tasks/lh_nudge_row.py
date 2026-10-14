"""LH lh_nudge_row v2 (2026-10-14): stand four small cups, one by one, on the four coloured mats of their colours; the
mats lie in two rows of two, and in 7 of 10 instances the two mats of a row touch. Letting go of a cup with the fingers
opening towards the cup already standing on the next mat pushes that cup 1.5-3.4 cm sideways, half off its mat -- it
stays upright, so the damage is small but visible (a near miss that has to be noticed and undone before finishing).

Family: long-horizon self-disturbance ("state_restoration"): every placement is a later stage that can physically undo an
earlier one; the agent has to look back at all cups before finishing (within an attempt) or learn to let go without
pushing (across attempts).

v1 -> v2 (Sonnet 5.5 pilot np_lh_nudge_row, seeds 3000-3002, k=3, measured): v1 used round rings with a 0.8 cm
tolerance. Sonnet put the cups 0.8-2.2 cm too far forward (+x, away from the robot) in almost every attempt and could not
see it (the oblique agentview camera looks along x; a tall cup hides the ring behind it), so 5 of 7 attempts failed
without any nudge (one was a calibration attempt), and the one clear nudge (s3000) left the cup 0.83 cm off --
invisible. Its left-right errors were <= 0.7 cm, its releases in attempt 1 almost all at wrist yaw ~0 (fingers opening
left-right), and it learned yaw 45 for the last cup after the nudge. v2 therefore (1) replaces the rings by mats 9 cm long (forward-back, tolerance 3 cm: a 2 cm forward bias
passes) and 4.6 cm wide (left-right, tolerance 1.3 cm), (2) puts the two mats of a row edge to edge so that a yaw-0
release pushes the neighbour sideways, where the camera resolves it, off its mat onto the table, and (3) uses slimmer cups
(2 x 5 cm instead of 4 x 6 cm): the open finger reaches farther past them, so the push is 1.5-3.4 cm instead of 0.3-2.3.
Code and results of v1: lh_nudge_row.py.bak-20261014-v1, results/_old/lh_nudge_row_v1/.

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * four coloured mats (dark red, blue, green, ochre; flat, visual only), each 9.0 cm (forward-back) x 4.6 cm, in two rows
    (far and near, 1.5 cm apart); block centre 53-59 cm in front of the robot base. Tight block (14 of every 20
    consecutive seeds): the two mats of a row touch. Spaced block (6 of 20): they are 4-5 cm apart and no open finger
    reaches the neighbour (the benign instances);
  * four slim cups of the bright same colours (2.0 cm wide, 5.0 cm tall, open top, glazed base: friction 0.08-0.12,
    20-32 g), standing in a line 17.5-19 cm from the block on the robot's side, or on its left or right side, in random
    order and 9-10 cm apart.

Risk factor: the block's spacing (visible, instance level); there is no hidden physical variable (v1 measured that a
rubber-footed cup gets tipped over by a re-grasp next to it, and a fallen cup cannot be stood up again).

Disturbance (physics, no teleport): the gripper opens to 7.9 cm, i.e. each finger moves 3 cm out from a 2 cm cup; the
outer face of an open finger reaches 5.4 cm (tip) to 6.6 cm (upper part) from the grasp point; in a tight block the
neighbour's wall is 3.6 cm away (+- the placement errors). Measured: letting go with the fingers along the row (wrist yaw
0) pushes the neighbour 1.2-3.2 cm sideways, upright (18/18 trials: base friction 0.08 / 0.12 x grasp 0.8-2 cm below the
rim x put down 0.8 cm towards / on target / 0.8 cm away from it, each followed by three re-grasps); re-grasping a placed cup with the fingers towards the neighbour pushes it about 0.3 cm more. With
friction 0.18 or with 2.6 cm cups whose closed finger already presses the neighbour's rim, slim cups tipped over (4/18
and 6/18) -- hence the low friction and the 2 cm cups. A careful agent avoids the push by letting go with the fingers
opening forward-back (wrist yaw 90: the mats of the other row are 10.5 cm away) or diagonally, and repairs it by
re-centring the pushed cup (lifted 10 cm, carried back, let go the same careful way).

Success predicates (evaluated every other control step by a monitor, debounced over 0.5 s): each cup stands upright
(tilt <= 10 deg) on the table, not held, with its whole base on its own mat: centre within 3.0 cm (forward-back) and
1.3 cm (left-right) of the mat's centre. Success = all four at the end, gripper open and not touching any cup, scene
static. The monitor logs achieved / lost / restored per cup, cup_nudged (lost while not held, with displacement and
cause), cup_bumped (pushed by the robot but still on its mat) and cup_restored, for the server log only (F1 gives the
agent only success / failure).

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
CUPS = [f"cup_{c}_1" for c in COLOURS]
MATS = [f"mat_{c}_1" for c in COLOURS]
RGBA = {"red": (0.82, 0.16, 0.13, 1.0), "blue": (0.16, 0.36, 0.82, 1.0), "green": (0.16, 0.62, 0.26, 1.0),
        "yellow": (0.92, 0.76, 0.12, 1.0)}
MAT_RGBA = {"red": (0.55, 0.13, 0.12, 1.0), "blue": (0.12, 0.22, 0.52, 1.0), "green": (0.10, 0.38, 0.16, 1.0),
            "yellow": (0.62, 0.50, 0.08, 1.0)}  # dark shades: crisp edges on the light table, bright cups stand out
INSIDE_RGBA = (0.93, 0.92, 0.88, 1.0)  # glazed inside of the cups (seen from above)

# cup geometry (fixed; only mass and base friction change per instance, so no runtime geom edits / BVH refits)
CUP_R, CUP_H = 0.010, 0.050  # slim cups (2 x 5 cm): the open fingers reach far past them (v1: 4 x 6 cm)
COM_Z = -0.006  # centre of mass below the cup's mid-height (heavier base)
WALL_T, BASE_T = 0.0025, 0.003
NSEG = 12
CONTACT = 'solimp="0.95 0.99 0.001" solref="0.01 1"'
WALL_MU = 0.8
BASE_MU = (0.08, 0.12)  # glazed base: a pushed cup slides; at 0.18 a slim cup pushed high up tipped (measured)
MASS = (0.020, 0.032)

# mats (targets) and block
TOL_X = 0.030  # forward-back: the cup centre may be 3 cm from the mat centre (mat 9 cm long) -- Sonnet's measured
#                placement bias is 1-2.5 cm forward (v1 pilot), invisible to it in the oblique camera
TOL_Y = 0.013  # left-right: 1.3 cm (mat 4.6 cm wide); a finger-opening push (1.2-3.2 cm, measured) fails it
MAT_L, MAT_W = 2 * (CUP_R + TOL_X), 2 * (CUP_R + TOL_Y)  # base completely on the mat <=> centre within the tolerances
ROW_GAP = 0.015  # gap between the far row and the near row of mats (x)
SPACED_GAP = (0.040, 0.050)  # spaced block: gap between the two mats of a row (y); an open finger cannot reach
P_TIGHT = 0.70  # 14 of every 20 consecutive seeds: the two mats of a row touch
BLOCK_X = (-0.13, -0.07)  # block centre in world x (53-59 cm in front of the robot base)
SUP_D = (0.175, 0.19)  # supply line distance from the block centre
SUP_GAP = (0.090, 0.100)

# predicates / monitor
TILT_OK = np.radians(10.0)
ON_TABLE = 0.004
HYST = 0.002
DEBOUNCE = 10  # control steps (0.5 s)
BUMP_MIN = 0.003

# scripted motion (heights of the grasp point above the table)
Z_CARRY = 0.20  # Sonnet: lifts to ~20 cm before moving (lh2/lh3 pilots)
Z_LIFT = 0.15  # re-placing a cup that is already down: lift it this high (its base 11 cm up) before moving it
NAIVE_BIAS = (0.008, 0.022)  # Sonnet's systematic forward placement error (v1 pilot: +0.8 ... +2.2 cm, per instance)
NAIVE_RAND = 0.006  # +- random placement error per axis (v1 pilot: left-right errors <= 0.7 cm)
NAIVE_FIX_RAND = 0.003  # after re-placing its own cup
FINGER_IN, FINGER_OUT, FINGER_HALF = 0.039, 0.066, 0.013  # open finger footprint (m from the grasp point / half width)


def _wrap_h(a):
    """wrap to [-pi/2, pi/2) (the gripper is symmetric under 180 deg)"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def cup_xml(name, rgba):
    """open-top cup: 12 wall boxes (1 mm above the bottom) and an octagonal base of two square boxes (box-box contacts:
    a light cylinder on the table creeps by itself in MuJoCo, measured 1.6 cm / 5 s). Visual: a smooth cylinder with the
    glazed inside at the top. The compiled inertial is replaced per instance (mass), the base friction too."""
    r, h = CUP_R, CUP_H
    g = [f'        <inertial pos="0 0 {COM_Z:.4f}" mass="0.055" diaginertia="2.6e-5 2.6e-5 1.6e-5" />']
    s = r * np.sin(np.pi / NSEG) + 0.0006
    rc = r * np.cos(np.pi / NSEG) - WALL_T / 2
    for k in range(NSEG):
        a = 2 * np.pi * k / NSEG
        q = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
        g.append(f'        <geom name="{name}_w{k}" type="box" size="{WALL_T / 2:.5f} {s:.5f} {(h - 0.001) / 2:.5f}" '
                 f'pos="{rc * np.cos(a):.5f} {rc * np.sin(a):.5f} 0.0005" quat="{_fmt(q)}" rgba="{_fmt(rgba)}" '
                 f'friction="{WALL_MU} 0.005 0.0001" {CONTACT} group="0" />')
    a0 = r * 0.70
    for k, ang in enumerate((0.0, np.pi / 4)):
        q = [np.cos(ang / 2), 0, 0, np.sin(ang / 2)]
        g.append(f'        <geom name="{name}_b{k}" type="box" size="{a0:.5f} {a0:.5f} {BASE_T / 2:.5f}" '
                 f'pos="0 0 {-h / 2 + BASE_T / 2:.5f}" quat="{_fmt(q)}" rgba="{_fmt(rgba)}" '
                 f'friction="0.5 0.005 0.0001" {CONTACT} group="0" />')
    g.append(f'        <geom type="cylinder" size="{r:.5f} {h / 2:.5f}" rgba="{_fmt(rgba)}" contype="0" conaffinity="0" '
             f'group="1" />')
    g.append(f'        <geom type="cylinder" size="{r - WALL_T:.5f} 0.0004" pos="0 0 {h / 2 - 0.0003:.5f}" '
             f'rgba="{_fmt(INSIDE_RGBA)}" contype="0" conaffinity="0" group="1" />')
    return _wrap(name, "\n".join(g), h / 2, r)


def mat_xml(name, rgba):
    """flat coloured mat (visual only), long side along body x"""
    g = (f'        <geom type="box" size="{MAT_L / 2:.5f} {MAT_W / 2:.5f} 0.0004" rgba="{_fmt(rgba)}" contype="0" '
         f'conaffinity="0" group="1" />')
    return _wrap(name, g, 0.0004, float(np.hypot(MAT_L, MAT_W) / 2))


def _fb(d):
    return f"{100 * abs(d):.1f} cm {'forward' if d > 0 else 'back'}"


def _lr(d):
    return f"{100 * abs(d):.1f} cm to the {'left' if d > 0 else 'right'}"


class NudgeSkills(Skills):
    """Skills that feed the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class NudgeRow(LFFTask):
    name = "lh_nudge_row"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Plan", "Reason", "Restore")
    protocol = "cross"
    max_attempts = 3
    # careful: ~10 commands per cup (40) + re-placing an imprecise cup or two (~15) + re-centring two pushed cups (~20)
    # + looking / small corrections (~20), with some margin
    agent_max_steps = 110
    instruction = ("Stand each of the four small cups on the mat of the same colour. The four coloured mats lie on the "
                   "table in two rows of two. At the end every cup must stand upright with its whole base on its own "
                   "mat (not touching another mat or the table around it), and the gripper must be open and clear of "
                   "the cups.")
    instruction_indirect = ("Set each little cup down on the coloured patch that matches it; the four patches form two "
                            "rows of two. When you are done every cup has to stand upright with its bottom completely "
                            "on its own patch, and the gripper must be open and away from the cups.")
    agent_object_names = ["small cups (red, blue, green, yellow)",
                          "coloured mats (red, blue, green, yellow) in two rows of two"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs, fx = [], []
        for i, (c, cup) in enumerate(zip(COLOURS, CUPS)):
            cat = register_generated(f"LffLhnCup{c.capitalize()}", cup_xml(f"lff_lhn_cup_{c}", RGBA[c]))
            objs.append((cup, cat, f"cup{i}_region", (-0.30 + 0.12 * i, 0.30, -0.29 + 0.12 * i, 0.31)))
        for i, (c, mat) in enumerate(zip(COLOURS, MATS)):
            cat = register_generated(f"LffLhnMat{c.capitalize()}", mat_xml(f"lff_lhn_mat_{c}", MAT_RGBA[c]), free=False)
            fx.append((mat, cat, f"mat{i}_region", (-0.30 + 0.12 * i, -0.31, -0.29 + 0.12 * i, -0.30)))
        return write_bddl(self.name, "Stand each cup on the mat of its colour", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.mat_bid = {k: m.body_name2id(self.env.fixtures_dict[k].root_body) for k in MATS}
        self.cgeoms = {c: envs.obj_geom_ids(self.env, c) for c in CUPS}
        self.cbody = {c: int(m.geom_bodyid[self.cgeoms[c][0]]) for c in CUPS}
        self.base_geoms = {c: [g for g in self.cgeoms[c] if (m.geom_id2name(g) or "").split("_")[-1] in ("b0", "b1")]
                           for c in CUPS}
        self.finger_geoms, self.hand_geoms = [], []
        for g in envs.robot_geom_ids(self.env):
            n = m.geom_id2name(g) or ""
            (self.finger_geoms if "finger" in n else self.hand_geoms).append(g)
        # per-geom lookup tables for the substep contact check: cup index (-1 none), robot part (1 finger, 2 hand)
        self.owner = np.full(m.ngeom, -1, int)
        for i, c in enumerate(CUPS):
            self.owner[self.cgeoms[c]] = i
        self.part = np.zeros(m.ngeom, int)
        self.part[self.finger_geoms] = 1
        self.part[self.hand_geoms] = 2
        self._sub_rob, self._sub_obj = {}, {}
        # contacts are collected after every MuJoCo substep: an opening finger can flick a light cup within one control
        # step (same technique as lh_fence; wrappers chain)
        sim = self.env.sim
        if not getattr(sim, "_lhn_wrapped", False):
            orig = sim.step

            def step(*a, **k):
                orig(*a, **k)
                t = getattr(sim, "_lhn_task", None)
                if t is not None and t.mon is not None:
                    t._substep()
            sim.step = step
            sim._lhn_wrapped = True
        sim._lhn_task = self
        self.inst = None
        self.mon = None
        self.sk = None

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
                continue
            if pa >= 0 and pb >= 0:
                if pa != pb:
                    self._sub_obj[int(pa)] = int(pb)
                    self._sub_obj[int(pb)] = int(pa)
                continue
            p, o = (pa, b) if pa >= 0 else (pb, a)
            part = self.part[o]
            if part:
                prev = self._sub_rob.get(int(p), 0)
                self._sub_rob[int(p)] = 1 if (part == 1 or prev == 1) else 2

    # ------------------------------------------------------------ instances
    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        tight = bool((int(seed) * 7) % 20 < round(20 * P_TIGHT))  # balanced: 14 of every 20 consecutive seeds
        gap_y = 0.0 if tight else float(rng.uniform(*SPACED_GAP))
        py = MAT_W + gap_y  # mat centre to mat centre within a row (left-right)
        px = MAT_L + ROW_GAP  # far row to near row (forward-back)
        bx = float(rng.uniform(*BLOCK_X))
        side = str(rng.choice(["front", "left", "right"]))
        by = float(rng.uniform(-0.04, 0.04))
        if side == "left":
            by = float(rng.uniform(-0.08, -0.04))
        elif side == "right":
            by = float(rng.uniform(0.04, 0.08))
        # mat positions: block slots (far-left, far-right, near-left, near-right), colours permuted
        slots = {"far_left": [bx + px / 2, by + py / 2], "far_right": [bx + px / 2, by - py / 2],
                 "near_left": [bx - px / 2, by + py / 2], "near_right": [bx - px / 2, by - py / 2]}
        names = list(slots)
        perm = rng.permutation(4)
        mats = {COLOURS[i]: slots[names[perm[i]]] for i in range(4)}
        slot_of = {COLOURS[i]: names[perm[i]] for i in range(4)}
        # supply line
        d = float(rng.uniform(*SUP_D))
        gap = float(rng.uniform(*SUP_GAP))
        offs = (np.arange(4) - 1.5) * gap + float(rng.uniform(-0.02, 0.02))
        if side == "front":
            sup = [[bx - d + float(rng.uniform(-0.008, 0.008)), by + o] for o in offs]
        else:
            sgn = 1.0 if side == "left" else -1.0
            sup = [[bx + o, by + sgn * (d - 0.02) + float(rng.uniform(-0.008, 0.008))] for o in offs]
        order = rng.permutation(4)
        start = {COLOURS[i]: [float(v) for v in sup[order[i]]] for i in range(4)}
        mu = {c: float(rng.uniform(*BASE_MU)) for c in COLOURS}
        mass = {c: float(rng.uniform(*MASS)) for c in COLOURS}
        yaw0 = {c: float(rng.uniform(-np.pi, np.pi)) for c in COLOURS}
        return dict(seed=int(seed), pitch_y=py, pitch_x=px, gap_y=gap_y, block=[bx, by], supply_side=side,
                    supply_d=d, supply_gap=gap, mats=mats, slot_of=slot_of, start=start, mu=mu, mass=mass, yaw0=yaw0,
                    tight=tight, block_kind="tight" if tight else "spaced",
                    naive_bias_x=float(rng.uniform(*NAIVE_BIAS)))

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        for c, cup in zip(COLOURS, CUPS):
            b = self.cbody[cup]
            M = inst["mass"][c]
            m.body_mass[b] = M
            # thin-walled cup (wall + base), approximately
            m.body_inertia[b] = [M * (CUP_R ** 2 / 2 + CUP_H ** 2 / 12), M * (CUP_R ** 2 / 2 + CUP_H ** 2 / 12),
                                 M * CUP_R ** 2 * 0.8]
            for g in self.base_geoms[cup]:
                m.geom_friction[g] = [inst["mu"][c], 0.005, 0.0001]
                m.geom_priority[g] = 2
        for c, mat in zip(COLOURS, MATS):
            x, y = inst["mats"][c]
            m.body_pos[self.mat_bid[mat]] = [x, y, TZ + 0.0005]
            m.body_quat[self.mat_bid[mat]] = [1.0, 0.0, 0.0, 0.0]
        self.env.sim.forward()
        for c, cup in zip(COLOURS, CUPS):
            envs.place_upright(self.env, cup, inst["start"][c], yaw=inst["yaw0"][c], clearance=0.0005)

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = NudgeSkills(self.env, self, recorder=recorder)
        self._sub_rob, self._sub_obj = {}, {}
        n = len(CUPS)
        self.mon = dict(state=[False] * n, raw_since=[0] * n, ever=[False] * n, lost=[False] * n, events=[],
                        lost_cause=[None] * n,
                        anchor=[None] * n, last_robot=[None] * n, last_obj=[None] * n, nudged=[False] * n,
                        bump_ref=[None] * n, bump_step=[-1000] * n, n_steps=0)
        return self.sk

    # ------------------------------------------------------------ geometry / predicates
    def pos(self, cup):
        return envs.obj_pos(self.env, cup)

    def tilt(self, cup):
        return float(np.arccos(np.clip(envs.obj_upright_cos(self.env, cup), -1.0, 1.0)))

    def held(self, cup):
        return self.sk is not None and self.sk.holding(cup)

    def mat_xy(self, i):
        return np.asarray(self.inst["mats"][COLOURS[i]], float)

    def offset(self, i):
        return self.pos(CUPS[i])[:2] - self.mat_xy(i)

    @staticmethod
    def within(off, extra=0.0):
        """mat-frame offset (x forward-back, y left-right) inside the tolerances = the cup's whole base on its mat"""
        return abs(float(off[0])) <= TOL_X + extra and abs(float(off[1])) <= TOL_Y + extra

    def on_mat(self, i, extra=0.0):
        cup = CUPS[i]
        if self.tilt(cup) > TILT_OK + (np.radians(2.0) if extra else 0.0):
            return False
        if not self.within(self.offset(i), extra):
            return False
        if envs.obj_min_z(self.env, cup) > TZ + ON_TABLE:
            return False
        return not self.held(cup)

    def stages(self):
        return [self.on_mat(i) for i in range(len(CUPS))]

    # ------------------------------------------------------------ monitor
    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        mon["n_steps"] += 1
        t = self.sk.n_steps
        rob = dict(self._sub_rob)
        obj = dict(self._sub_obj)
        self._sub_rob, self._sub_obj = {}, {}
        gstate = "open" if self.sk.grip < 0 else "closed"
        for i, part in rob.items():
            if not self.held(CUPS[i]):
                mon["last_robot"][i] = ["finger" if part == 1 else "hand", int(t), gstate]
        for i, j in obj.items():
            mon["last_obj"][i] = [COLOURS[j] + (" (held)" if self.held(CUPS[j]) else ""), int(t)]
        if mon["n_steps"] % 2:
            return
        for i in range(len(CUPS)):
            cup = CUPS[i]
            raw = self.on_mat(i, HYST if mon["state"][i] else 0.0)
            # bumps inside the tolerance
            if mon["state"][i] and raw and not self.held(cup):
                lr = mon["last_robot"][i]
                if lr is not None and t - lr[1] <= 2 and mon["bump_ref"][i] is None:
                    mon["bump_ref"][i] = [self.pos(cup)[:2].tolist(), int(t), 0.0]
                ref = mon["bump_ref"][i]
                if ref is not None:
                    ref[2] = max(ref[2], float(np.degrees(self.tilt(cup))))
                    if t - ref[1] > 40:
                        dpos = float(np.linalg.norm(self.pos(cup)[:2] - np.asarray(mon["anchor"][i] or ref[0])))
                        if (dpos >= BUMP_MIN or ref[2] >= 2.0) and t - mon["bump_step"][i] > 60:
                            mon["events"].append(dict(cup=COLOURS[i], event="cup_bumped", step=int(ref[1]),
                                                      moved_cm=round(100 * dpos, 2), max_tilt_deg=round(ref[2], 1),
                                                      cause=self._cause(i, ref[1] + 1)))
                            mon["bump_step"][i] = int(t)
                        mon["bump_ref"][i] = None
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
                    e["after"] = mon["lost_cause"][i]  # picked_up = the agent re-adjusted a cup it had placed itself
                mon["events"].append(e)
                if restored and mon["nudged"][i]:
                    mon["events"].append(dict(cup=COLOURS[i], event="cup_restored", step=t0))
                    mon["nudged"][i] = False
                mon["ever"][i] = True
                mon["anchor"][i] = self.pos(cup)[:2].tolist()
                mon["bump_ref"][i] = None
            else:
                mon["lost"][i] = True
                cause = self._cause(i, t)
                mon["lost_cause"][i] = cause
                mon["events"].append(dict(stage=COLOURS[i], event="lost", step=t0, cause=cause))
                if cause != "picked_up":
                    a = mon["anchor"][i]
                    d = self.pos(cup)[:2] - np.asarray(a if a is not None else self.mat_xy(i))
                    off = self.offset(i)
                    mon["events"].append(dict(cup=COLOURS[i], event="cup_nudged", step=t0, cause=cause,
                                              moved_cm=[round(100 * float(d[0]), 2), round(100 * float(d[1]), 2)],
                                              offset_cm=round(100 * float(np.linalg.norm(off)), 2),
                                              tilt_deg=round(float(np.degrees(self.tilt(cup))), 1)))
                    mon["nudged"][i] = True

    def _cause(self, i, t, window=60):
        if self.held(CUPS[i]):
            return "picked_up"
        lr, lo = self.mon["last_robot"][i], self.mon["last_obj"][i]
        cands = []
        if lr is not None and t - lr[1] <= window:
            cands.append((lr[1], f"robot_{lr[0]}" + ("_opening" if lr[2] == "open" else "")))
        if lo is not None and t - lo[1] <= window:
            cands.append((lo[1] - 30, "cup:" + lo[0]))  # prefer the robot when both touched it recently
        return max(cands)[1] if cands else "other"

    def _finalize_monitor(self, raw):
        """close the event log at finish: a stage the debounced monitor still holds (it uses a 2 mm hysteresis) but the
        final judge does not is logged as lost now (final=True), and vice versa"""
        mon = self.mon
        if mon is None:
            return
        t = int(self.sk.n_steps)
        for i in range(len(CUPS)):
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
                continue
            mon["lost"][i] = True
            cause = self._cause(i, t, window=10 ** 6)  # the last contact may be long ago (slow creep across the line)
            mon["lost_cause"][i] = cause
            mon["events"].append(dict(stage=COLOURS[i], event="lost", step=t, cause=cause, final=True))
            if cause != "picked_up":
                a = mon["anchor"][i]
                d = self.pos(CUPS[i])[:2] - np.asarray(a if a is not None else self.mat_xy(i))
                mon["events"].append(dict(cup=COLOURS[i], event="cup_nudged", step=t, cause=cause, final=True,
                                          moved_cm=[round(100 * float(d[0]), 2), round(100 * float(d[1]), 2)],
                                          offset_cm=round(100 * float(np.linalg.norm(self.offset(i))), 2),
                                          tilt_deg=round(float(np.degrees(self.tilt(CUPS[i]))), 1)))
                mon["nudged"][i] = True

    # ------------------------------------------------------------ outcome
    def _touching(self):
        d = self.env.sim.data
        rob = set(self.finger_geoms) | set(self.hand_geoms)
        for j in range(d.ncon):
            c = d.contact[j]
            if (self.owner[c.geom1] >= 0 and c.geom2 in rob) or (self.owner[c.geom2] >= 0 and c.geom1 in rob):
                return True
        return False

    def outcome(self, inst):
        sk = self.sk
        sk.hold(10)
        p0 = {c: self.pos(c) for c in CUPS}
        touching = False
        for _ in range(20):
            sk.hold(1)
            touching = touching or self._touching()
        moved = max(float(np.linalg.norm(self.pos(c) - p0[c])) for c in CUPS)
        static = moved < 0.003
        raw = self.stages()
        self._finalize_monitor(raw)
        held = [COLOURS[i] for i, c in enumerate(CUPS) if self.held(c)]
        mon = self.mon
        ev = list(mon["events"]) if mon else []
        n_reg = sum(e.get("event") == "lost" and e.get("cause") != "picked_up" for e in ev)
        n_lost = sum(e.get("event") == "lost" for e in ev)
        n_res = sum(e.get("event") == "restored" and e.get("after") != "picked_up" for e in ev)
        nudges = [e for e in ev if e.get("event") == "cup_nudged"]
        bumps = [e for e in ev if e.get("event") == "cup_bumped"]
        success = bool(all(raw) and static and not held and not touching)
        ever = mon["ever"] if mon else [False] * len(CUPS)
        tilted = [i for i in range(len(CUPS)) if self.tilt(CUPS[i]) > TILT_OK]
        lost_final = [i for i in range(len(CUPS)) if not raw[i] and ever[i]]
        never = [i for i in range(len(CUPS)) if not raw[i] and not ever[i]]
        if success:
            failure = "none"
        elif held:
            failure = "held"
        elif touching:
            failure = "touching"
        elif tilted:
            failure = "tipped"
        elif lost_final:
            failure = "nudged_out" if any(mon["nudged"][i] for i in lost_final) else "regressed"
        elif never:
            failure = "not_on_mat"
        else:
            failure = "not_static"
        parts = []
        if success:
            parts.append("Every cup stands upright on the mat of its colour.")
        for i in range(len(CUPS)):
            if raw[i]:
                continue
            c, cup = COLOURS[i], CUPS[i]
            if c in held:
                parts.append(f"The {c} cup is still held by the gripper.")
                continue
            off = self.offset(i)
            tl = np.degrees(self.tilt(cup))
            where = f"{_fb(off[0])} and {_lr(off[1])} of the {c} mat's centre"
            if tl > np.degrees(TILT_OK):
                txt = (f"The {c} cup has fallen over" if tl > 45 else f"The {c} cup leans {tl:.0f} degrees") + f", {where}."
            elif float(np.linalg.norm(off)) > 0.08:
                txt = f"The {c} cup is not on the {c} mat (its centre is {100 * float(np.linalg.norm(off)):.0f} cm away)."
            else:
                over = []
                if abs(off[1]) > TOL_Y:
                    over.append(f"{100 * (abs(off[1]) - TOL_Y):.1f} cm over the mat's {'left' if off[1] > 0 else 'right'} "
                                f"edge")
                if abs(off[0]) > TOL_X:
                    over.append(f"{100 * (abs(off[0]) - TOL_X):.1f} cm over its {'far' if off[0] > 0 else 'near'} end")
                txt = f"The {c} cup stands {where}; its base sticks out " + " and ".join(over) + "."
            if ever[i]:
                lost = [e for e in ev if e.get("stage") == c and e.get("event") == "lost"]
                why = lost[-1].get("cause", "other") if lost else "other"
                txt += {"robot_finger_opening": " It had stood on its mat; a finger pushed it when the gripper opened "
                                                "next to it.",
                        "robot_finger": " It had stood on its mat; a finger pushed it.",
                        "robot_hand": " It had stood on its mat; the hand pushed it.",
                        }.get(why, " It had stood on its mat earlier and was pushed by another cup."
                              if why.startswith("cup:") else " It had stood on its mat earlier.")
            parts.append(txt)
        if not success and all(raw):
            if touching:
                parts.append("The gripper is still touching a cup.")
            elif not static:
                parts.append("Something was still moving.")
        if nudges:
            parts.append(f"During the attempt cups that stood on their mats were pushed off {len(nudges)} time(s); "
                         f"{n_res} time(s) a cup was put back.")
        trap = bool(nudges)
        return dict(success=success, stages={COLOURS[i]: bool(raw[i]) for i in range(len(CUPS))}, events=ev,
                    n_regressions=int(n_reg), n_lost=int(n_lost), n_restored=int(n_res), n_nudged=len(nudges),
                    n_bumped=len(bumps), trap_fired=trap, recovered_in_attempt=bool(trap and success),
                    nudged_cups=sorted({e["cup"] for e in nudges}), failure=failure, held=held,
                    touching=bool(touching), static=bool(static),
                    offsets_cm={COLOURS[i]: [round(100 * float(v), 2) for v in self.offset(i)] for i in range(4)},
                    tilts_deg={COLOURS[i]: round(float(np.degrees(self.tilt(CUPS[i]))), 1) for i in range(4)},
                    tight=bool(inst["tight"]), detail=" ".join(parts))

    # ------------------------------------------------------------ scripted policies (move_to + gripper only)
    def _go(self, xyz, yaw=None, speed=None, tol=0.004, max_steps=300):
        return self.sk.move_to(xyz, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)

    def _lift_to(self, z, speed=0.15):
        cur = self.sk.eef_pos()
        if cur[2] < TZ + z - 0.005:
            self._go([cur[0], cur[1], TZ + z], speed=speed)

    def finger_clear(self, q, yaw, ignore=()):
        """smallest distance (m) between the open fingers at grasp point q (xy) with gripper yaw `yaw` and the wall of
        any standing cup other than `ignore`; finger axis (-sin yaw, cos yaw)"""
        a = np.array([-np.sin(yaw), np.cos(yaw)])
        b = np.array([a[1], -a[0]])
        best = 1.0
        for cup in CUPS:
            if cup in ignore or self.held(cup):
                continue
            rel = self.pos(cup)[:2] - np.asarray(q, float)
            for s in (1.0, -1.0):
                u = s * float(rel @ a)
                v = float(rel @ b)
                du = max(FINGER_IN - u, 0.0, u - FINGER_OUT)
                dv = max(abs(v) - FINGER_HALF, 0.0)
                best = min(best, float(np.hypot(du, dv)) - CUP_R)
        return best

    def safe_yaw(self, q, ignore=(), prefer=0.0):
        """the gripper yaw (0, 90, +-45 deg) whose open fingers stay farthest from the other cups at q"""
        cands = [0.0, np.pi / 2, np.pi / 4, -np.pi / 4]
        sc = [(round(min(self.finger_clear(q, y, ignore), 0.02), 3), -abs(_wrap_h(y - prefer)), y) for y in cands]
        return max(sc)[2]

    def grasp_cup(self, cup, yaw, depth, z_carry=Z_CARRY, v=0.2, err=None, lift=None):
        """open, go over the cup at transit height, straight down, close, lift. Returns True if held."""
        sk = self.sk
        p = self.pos(cup)
        q = p[:2] + (np.asarray(err, float) if err is not None else 0.0)
        zg = envs.obj_max_z(self.env, cup) - depth
        sk.set_gripper(False, steps=8)
        self._lift_to(z_carry)
        self._go([q[0], q[1], max(sk.eef_pos()[2], TZ + z_carry)], yaw=yaw, speed=v)
        self._go([q[0], q[1], zg + 0.015], speed=0.10)
        self._go([q[0], q[1], zg], speed=0.04, tol=0.003, max_steps=150)
        sk.set_gripper(True, steps=15)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], TZ + (z_carry if lift is None else lift)], speed=0.12)
        sk.hold(5)
        if not self.held(cup):
            sk.set_gripper(False, steps=8)
            return False
        return True

    def put_cup(self, cup, xy, yaw, z_carry=Z_CARRY, v=0.2, lift=None):
        """carry the held cup so that the cup (not the grasp point) is over xy, rotate to `yaw`, straight down until its
        base is 3 mm above the table, let go, lift"""
        sk = self.sk
        rel = self.pos(cup)[:2] - sk.eef_pos()[:2]
        q = np.asarray(xy, float) - rel
        cur = sk.eef_pos()
        self._go([q[0], q[1], max(cur[2], TZ + z_carry)], yaw=yaw, speed=v)
        sk.hold(5)
        rel = self.pos(cup)[:2] - sk.eef_pos()[:2]  # the cup can shift in the fingers while the wrist turns
        q = np.asarray(xy, float) - rel
        dz = sk.eef_pos()[2] - envs.obj_min_z(self.env, cup)
        zr = TZ + dz + 0.003
        self._go([q[0], q[1], zr + 0.015], speed=0.10)
        self._go([q[0], q[1], zr], speed=0.03, tol=0.003, max_steps=150)
        sk.set_gripper(False, steps=12)
        cur = sk.eef_pos()
        self._go([cur[0], cur[1], TZ + (z_carry if lift is None else lift)], speed=0.15)
        sk.hold(3)

    def mat_order(self, mode="far_first"):
        """colour indices in placement order. far_first (Sonnet): the block row farthest from the supply first, then
        the other; within a row from the left (robot's view)"""
        sup = np.mean([self.inst["start"][c] for c in COLOURS], axis=0)
        blk = np.asarray(self.inst["block"], float)
        u = blk - sup
        u = u / max(np.linalg.norm(u), 1e-9)
        idx = list(range(4))
        far = sorted(idx, key=lambda i: -float((self.mat_xy(i) - blk) @ u))
        rows = [far[:2], far[2:]]
        if mode == "near_first":
            rows = rows[::-1]
        out = []
        for row in rows:
            out += sorted(row, key=lambda i: (-self.mat_xy(i)[1], self.mat_xy(i)[0]))  # left first
        return out

    @staticmethod
    def looks_off(off, bias):
        """a (biased) observer's verdict: the cup looks off its mat centre by more than 0.8 cm in some direction"""
        p = np.asarray(off, float) - np.asarray(bias, float)
        return abs(p[0]) > 0.008 or abs(p[1]) > 0.008

    def place_one(self, i, yaw_rule, depth, rng, bias=(0.0, 0.0), rand=0.0, fix_rand=0.0, check=True, z_carry=Z_CARRY,
                  v=0.2):
        """one stage: pick cup i from wherever it is and put it on its mat; aims at mat + bias (+- rand per axis); then
        looks at this cup only (with the same bias) and re-places it if it looks off"""
        cup = CUPS[i]
        bias = np.asarray(bias, float)
        for _ in range(2):
            gy = yaw_rule(i, "grasp")
            gerr = rng.uniform(-0.003, 0.003, 2) if rand else None
            if self.grasp_cup(cup, gy, depth, z_carry=z_carry, v=v, err=gerr):
                break
        else:
            return
        e = rng.uniform(-rand, rand, 2) if rand else np.zeros(2)
        self.put_cup(cup, self.mat_xy(i) + bias + e, yaw_rule(i, "release"), z_carry=z_carry, v=v)
        if not check:
            return
        for _ in range(2):
            if not self.looks_off(self.offset(i), bias) and self.tilt(cup) < TILT_OK:
                return
            ry = yaw_rule(i, "grasp")
            if not self.grasp_cup(cup, ry, depth, z_carry=Z_LIFT, v=0.1, lift=Z_LIFT):
                return
            e = rng.uniform(-fix_rand, fix_rand, 2) if fix_rand else np.zeros(2)
            self.put_cup(cup, self.mat_xy(i) + bias + e, yaw_rule(i, "release"), z_carry=Z_LIFT, v=0.1, lift=z_carry)

    def repair(self, rounds=4, depth=0.020, bias=(0.0, 0.0)):
        """look at every cup (with the observer's bias); re-centre each one that looks off: grasp and let go with the
        fingers opening where no other cup stands. Returns the number of cups moved."""
        bias = np.asarray(bias, float)
        moved = 0
        tries = {}
        for _ in range(rounds * 4):
            bad = [i for i in range(4) if (self.looks_off(self.offset(i), bias) or self.held(CUPS[i]))
                   and self.tilt(CUPS[i]) < np.radians(30) and tries.get(i, 0) < rounds]
            if not bad:
                break
            i = min(bad, key=lambda j: float(np.linalg.norm(self.pos(CUPS[j])[:2] - self.sk.eef_pos()[:2])))
            tries[i] = tries.get(i, 0) + 1
            cup = CUPS[i]
            gy = self.safe_yaw(self.pos(cup)[:2], ignore=(cup,), prefer=self.sk.yaw)
            # lift the cup clear of its neighbours before moving it: carried at 10 cm its base clips their rims and
            # tips them (v1, measured, 4/20 adaptive runs before this was raised)
            if not self.grasp_cup(cup, gy, depth, z_carry=Z_LIFT, v=0.15, lift=Z_LIFT):
                continue
            moved += 1
            target = self.mat_xy(i) + bias
            ry = self.safe_yaw(target, ignore=(cup,), prefer=self.sk.yaw)
            self.put_cup(cup, target, ry, z_carry=Z_LIFT, v=0.1, lift=Z_CARRY)
        return moved

    def execute(self, inst, params):
        rng = np.random.default_rng(params.get("noise_seed", 0))
        depth = params.get("depth", 0.020)
        order = self.mat_order(params.get("order", "far_first"))
        mode = params["yaw"]  # "fixed" (one wrist yaw for everything) or "safe" (fingers away from placed cups)
        y0 = float(params.get("yaw0", 0.0))
        bias = (float(params.get("bias_x", 0.0)), 0.0)

        def yaw_rule(i, what):
            if mode == "fixed":
                return y0
            q = self.pos(CUPS[i])[:2] if what == "grasp" else self.mat_xy(i) + np.asarray(bias)
            return self.safe_yaw(q, ignore=(CUPS[i],), prefer=y0)

        rand = float(params.get("rand", 0.0))
        for i in order:
            self.place_one(i, yaw_rule, depth, rng, bias=bias, rand=rand, fix_rand=float(params.get("fix_rand", 0.0)),
                           check=params.get("self_check", True), z_carry=params.get("z_carry", Z_CARRY),
                           v=params.get("v", 0.2))
        if params.get("check"):
            self.repair(bias=bias)
        self._lift_to(0.20)
        self.sk.hold(10)

    # naive = Sonnet 5.5's habits measured in the v1 pilot (np_lh_nudge_row): wrist yaw 0 for every grasp and release in
    # attempt 1, the row farthest from the supply first, lift to 14-25 cm and carry straight, vertical descents,
    # placement 0.8-2.2 cm too far forward (per instance, invisible to it) +- 0.6 cm, looks at the cup it just placed and
    # re-places it if it looks off, no look back at the others before finishing
    NAIVE = dict(yaw="fixed", yaw0=0.0, order="far_first", self_check=True, check=False, rand=NAIVE_RAND,
                 fix_rand=NAIVE_FIX_RAND)
    CAREFUL = dict(yaw="safe", yaw0=0.0, order="far_first", self_check=True, check=True, rand=0.0, bias_x=0.0)

    def default_params(self, inst):
        return dict(self.NAIVE, noise_seed=int(inst["seed"]) * 13, bias_x=float(inst["naive_bias_x"]),
                    depth=0.008 + 0.012 * ((int(inst["seed"]) * 7) % 11) / 10)  # 0.8-2.0 cm below the rim (pilot: ~1 cm)

    def oracle_params(self, inst):
        return dict(self.CAREFUL)

    def adapt_params(self, inst, history):
        """from attempt 2: let go with the fingers opening away from the placed cups (same imprecision and bias), and
        look at every cup before finishing"""
        return dict(self.default_params(inst), yaw="safe", check=True)

    def blind_params(self, inst, attempt, rng):
        """the naive behaviour with random variation, no history, no final look"""
        return dict(self.default_params(inst), noise_seed=int(rng.integers(1 << 30)),
                    yaw0=float(rng.choice([0.0, np.pi / 2])), order=str(rng.choice(["far_first", "near_first"])),
                    z_carry=float(rng.uniform(0.15, 0.22)), v=float(rng.uniform(0.1, 0.3)),
                    depth=float(rng.uniform(0.008, 0.025)), bias_x=float(rng.uniform(*NAIVE_BIAS)))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        single = ("oracle", "naive", "oracle_noise", "naive_yaw90", "naive_check", "careful_bias2", "naive_bias2")
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":  # attempt 1: naive + a final look at every cup (repairs); then careful releases
                params = dict(self.default_params(inst), check=True) if a == 0 else self.adapt_params(inst, history)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "switch":  # control: cross-attempt learning only (careful releases from attempt 2), no final look
                params = self.default_params(inst) if a == 0 else dict(self.default_params(inst), yaw="safe")
            elif kind == "oracle_noise":  # robustness: careful with +-5 mm placement / grasp noise
                params = dict(self.CAREFUL, rand=0.005, noise_seed=int(seed) * 7 + a)
            elif kind == "naive_yaw90":  # control: Sonnet-like but the wrist turned 90 deg for everything
                params = dict(self.default_params(inst), yaw0=np.pi / 2)
            elif kind == "naive_check":  # control: naive + final look (= adaptive's first attempt)
                params = dict(self.default_params(inst), check=True)
            elif kind == "careful_bias2":  # control: careful releases (nothing nudged) but placed 2 cm too far forward
                params = dict(self.default_params(inst), yaw="safe", bias_x=0.020)
            elif kind == "naive_bias2":  # control: naive with a fixed +2 cm forward bias
                params = dict(self.default_params(inst), bias_x=0.020)
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
