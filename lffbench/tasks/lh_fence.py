"""LH lh_fence (v2, 2026-10-11): build a small fence -- three square posts on square marks and two flat panels on the
strips between them, end to end in one line. Letting go of a panel with the fingers along the fence line knocks over
the post next to it (when the posts are light); the fallen post must be stood up again before finishing.

Family: long-horizon self-disturbance ("state_restoration"): every placement is a later stage that can physically
break the earlier ones; the agent has to notice and restore them (within an attempt) or learn to place without
disturbing them (across attempts).

Why v2 (measured, ../docs/11 sec. 9): in v1 the panels stood like dominoes (thickness along the row) and the hazard was
opening the fingers along the row. Sonnet 5.5 always turned the hand so that its long axis (= the finger axis) lay
along the panel's long horizontal side, which in v1 is across the row = the safe orientation, and it filled the row
from the far end: 3/3 pilot runs succeeded first time with zero disturbances. v2 turns the panels by 90 deg into a real
fence (panels end to end along the line, posts between them), so the same habit now opens the fingers ALONG the
line, towards the posts.

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * a straight fence line, left-right or forward-back: 3 dark square marks (posts) alternating with 2 dark strips
    (panels), slot pitch (post centre to post centre) w + 2 g + b;
  * 3 posts: square b = 1.55-1.65 cm, height = the panels' height, dark wood, ball knob (r 1.0 cm, 10 g) on a 0.8 cm
    stem; 2 panels: thickness t 1.4-1.6 cm (across the line), width w 4.0-4.6 cm (along it), height h 7-8 cm, light
    wood, same knob. A placed panel leaves a gap g = 1.3-1.5 cm to each post (knob to knob D = w/2 + g + b/2 =
    4.1-4.6 cm);
  * the 5 objects stand in a 2 x 3 supply grid (11 cm pitch, 13-15 cm beside the line, shifted towards one end), the
    panels with their width along the line (+-9 deg).

Hidden variable (instance level, consistent across attempts, invisible): the posts' mass distribution. "Light" posts
are plain wood (19-23 g incl. the knob; centre of mass 5.9-7.1 cm up, tips over beyond ~7 deg); "footed" posts carry a
hidden 100-140 g steel foot (centre of mass 1.5-1.8 cm up, tips over beyond 25-28 deg). They look identical. 13 of
every 20 consecutive seeds have light posts ("hazard"), 7 footed ("benign").

Disturbance (physics, no teleport): a panel is grasped by its knob; with the fingers along the panel's width (the
natural "hand along the board" grasp, and Sonnet's measured habit) the fingers lie along the fence line at the release
and, opening to 8 cm, sweep ~6 cm out from the grasp point: they hit the knob of each post that already stands next to
the panel (D 3.3-5.4 cm incl. a 0.8 cm placement error) and push it 6-18 deg. A light post falls over (away from the
panel; a middle post often ends up leaning on the panel on its other side); a footed post rocks back. Measured in the
prototype: light posts 16/16 fell for D <= 5.0 cm (2/4 at 5.5, 0/4 at 6.0), posts with a 0.12-0.25 kg foot 0/30 (max
tilt 18 deg); with the fingers across the line (along the panel's thickness) 0/9 contacts. A post set down with its
fingers along the line next to a standing panel tilts the panel <= 15 deg, which rocks back. Validation (seeds
1000-1019): the Sonnet-like naive disturbs 13/13 light-post instances and 0/7 footed ones.

So the levers are under the agent's control and the hidden weight only decides whether its habit is punished: (1)
release a panel with the fingers across the fence line (Sonnet-like naive with only this change: 19/20), or, only
partly, (2) stand both panels before the posts (15/20: a post released next to a panel with its fingers along the line
rocks the panel, which can knock the new light post). Within an attempt:
look at the fence before finishing and stand fallen posts up again (lifted by its knob a fallen post swings down and
hangs upright; a fallen panel too).

Implementation notes (found while building v2): MuJoCo keeps each body's bounding-volume hierarchy in the body's
inertial frame, computed at compile time. Changing geom sizes or body_ipos at runtime therefore needs `refit_bvh`
(else contacts are missed: a light post whose centre of mass was moved up sank 2.2 cm into the table; one moved down
never touched the fingers), and runtime body_ipos is ignored altogether when the compiled inertial frame coincides with
the body frame (v1's knob-mass shift of the centre of mass was silently ignored); the generated XML therefore declares
an inertial pos of 1 mm. With the centre of mass right, v1's knob torsional friction (0.0015) held a lifted fallen light
post at 44-61 deg instead of letting it swing down; v2 uses 0.0005 (< 3 deg).

Protocol: cross (reset to the same instance before every attempt), k = 3. Outcome: final-state judge (every mark
carries an upright post, every strip an upright panel lined up with it, nothing held or touched, scene static) plus an
event log of every stage predicate (achieved / lost / restored, with step and cause) for the server log; F1 gives the
agent only success / failure.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

TZ = envs.TABLE_Z
POSTS = [f"post_{c}_1" for c in "abc"]
PANELS = [f"panel_{c}_1" for c in "ab"]
OBJS = POSTS + PANELS
MARKS = [f"postmark_{c}_1" for c in "abc"]
STRIPS = [f"strip_{c}_1" for c in "ab"]
SLOT_KIND = ["post", "panel", "post", "panel", "post"]  # along the fence line (-u to +u)
WOOD = [(0.82, 0.66, 0.44, 1.0), (0.76, 0.60, 0.40, 1.0)]
POST_RGBA = [(0.46, 0.30, 0.18, 1.0), (0.44, 0.29, 0.17, 1.0), (0.48, 0.31, 0.19, 1.0)]
KNOB_RGBA = (0.20, 0.11, 0.06, 1.0)
STRIP_RGBA = (0.30, 0.10, 0.12, 1.0)
MARK_RGBA = (0.14, 0.14, 0.16, 1.0)
NOM_T, NOM_W, NOM_H = 0.015, 0.047, 0.075  # nominal panel (resized per instance)
NOM_B = 0.016  # nominal post width
KNOB_R, STEM_L, STEM_R = 0.010, 0.008, 0.0035
KNOB_MASS = 0.010
KNOB_MU_T = 0.0005  # torsional friction of the knob (condim 4): damps the swing of a hanging object; 0.0015 (v1) holds a lifted fallen light post at 45-60 deg (measured)
STRIP_W = 0.006

# instance ranges
P_HAZARD = 0.65  # 13 of every 20 consecutive seeds: light posts
T_RANGE = (0.014, 0.016)
W_RANGE = (0.040, 0.046)
H_RANGE = (0.070, 0.080)
B_RANGE = (0.0155, 0.0165)
GAP_RANGE = (0.013, 0.015)  # panel end to post face; 1.0-1.2 cm: a +-0.8 cm placement error let the panel end touch a light post (2/20 with the safe grasp)
DENS_RANGE = (450.0, 650.0)
MU_RANGE = (0.4, 0.8)
FOOT_RANGE = (0.10, 0.14)  # hidden steel foot of a "footed" post (kg)
FOOT_H = 0.015  # height of the foot slug (its centre 7.5 mm above the post's bottom)
SUP_PITCH = (0.105, 0.115)
SUP_D = (0.13, 0.15)
X_FAR = -0.035

# success predicate
TOL_U = 0.010  # panel centre within 1 cm of the strip centre along the line, 1.5 cm across; post 1 cm both ways
TOL_V = 0.015
TOL_POST = 0.010
TILT_OK = np.radians(5.0)
YAW_TOL = np.radians(15.0)
HYST = 0.004
DEBOUNCE = 20  # a stage counts as lost / regained only after 1 s (a post or panel that is pushed and rocks back is not a loss)
FALLEN = np.radians(45.0)

# scripted motion (z = height of the grasp point above the table)
Z_SAFE = 0.25  # careful transit (a hanging object's bottom ~15 cm up)
Z_NAIVE = 0.21  # Sonnet's usual carry height: robot-frame z 0.20 (= 21 cm above the table), measured in lh2_fence
NAIVE_ERR = 0.008  # naive placement error, uniform per axis (Sonnet: 0.4-1.3 cm from the strip centre, lh2_fence)


def _wrap_h(a):
    """wrap to [-pi/2, pi/2) (panel footprint and gripper are symmetric under 180 deg)"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def _yaw_for_axis(a):
    """gripper yaw whose finger axis (-sin yaw, cos yaw) is parallel to the horizontal direction a"""
    return _wrap_h(float(np.arctan2(-a[0], a[1])))


def _quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def refit_bvh(m, b, margin=0.001):
    """Recompute body b's bounding-volume hierarchy after runtime changes of its geoms or body_ipos. MuJoCo stores the
    boxes in the body's inertial frame, computed at compile time; stale boxes make it miss contacts (see module doc).
    Every node becomes the union of its leaf geoms' boxes (body_iquat must be identity)."""
    a = int(m.body_bvhadr[b])
    if a < 0:
        return
    ipos = np.asarray(m.body_ipos[b], float)

    def rec(i):
        g = int(m.bvh_nodeid[i])
        if g >= 0:
            R = _quat2mat(m.geom_quat[g])
            c = np.asarray(m.geom_pos[g]) + R @ np.asarray(m.geom_aabb[g, :3]) - ipos
            h = np.abs(R) @ np.asarray(m.geom_aabb[g, 3:])
            lo, hi = c - h, c + h
        else:
            l1, h1 = rec(a + int(m.bvh_child[i][0]))
            l2, h2 = rec(a + int(m.bvh_child[i][1]))
            lo, hi = np.minimum(l1, l2), np.maximum(h1, h2)
        m.bvh_aabb[i] = np.concatenate([(lo + hi) / 2, (hi - lo) / 2 + margin])
        return lo, hi
    rec(a)


def knobbed_xml(name, half, rgba):
    """box + stem + ball knob. The inertial pos is 1 mm (not 0) so that MuJoCo does not mark the inertial frame as
    identical to the body frame: then the centre of mass set at runtime (body_ipos) is used."""
    geoms = [("board", "box", (0, 0, 0), half, rgba, 'friction="0.6 0.005 0.0001"'),
             ("stem", "cylinder", (0, 0, half[2] + STEM_L / 2), (STEM_R, STEM_L / 2), KNOB_RGBA,
              'friction="1 0.005 0.0001"'),
             ("knob", "sphere", (0, 0, half[2] + STEM_L + KNOB_R), (KNOB_R,), KNOB_RGBA,
              f'friction="1 {KNOB_MU_T} 0.0001" condim="4" priority="2"')]
    s = ['        <inertial pos="0 0 0.001" mass="0.05" diaginertia="2e-5 2e-5 1e-5" />']
    for nm, typ, pos, size, c, extra in geoms:
        s.append(f'        <geom name="{name}_{nm}" type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" '
                 f'{extra} group="0" />')
        s.append(f'        <geom type="{typ}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(c)}" contype="0" '
                 f'conaffinity="0" group="1" />')
    return _wrap(name, "\n".join(s), half[2], float(np.hypot(half[0], half[1])))


def flat_xml(name, half, rgba):
    g = (f'        <geom type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" contype="0" conaffinity="0" group="1" />')
    return _wrap(name, g, half[2], float(np.hypot(half[0], half[1])))


class FenceSkills(Skills):
    """Skills that feed the task's per-step stage monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class Fence(LFFTask):
    name = "lh_fence"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Plan", "Reason", "Restore")
    protocol = "cross"
    max_attempts = 3
    agent_max_steps = 120  # careful: ~10 commands per object (50) + standing up two fallen posts (~20) + looking
    instruction = ("Build a small fence: stand one wooden post upright on each of the dark square marks on the table "
                   "and one wooden panel upright on each of the dark strips between them, the panel's long bottom edge "
                   "along its strip, so that posts and panels alternate in one straight line. Every post and panel "
                   "has a small round knob on top; an object lifted by its knob hangs straight down. At the end every "
                   "square mark must have one upright post and every strip one upright panel standing on it, with the "
                   "gripper open and clear of the fence.")
    instruction_indirect = ("Put up the little fence: the thin square sticks go upright on the dark squares, the flat "
                            "boards upright on the dark lines between them, lengthwise along their lines, so that "
                            "sticks and boards take turns in one straight row. Every piece has a round knob on top; "
                            "held by its knob a piece hangs straight down. At the end every square must carry one "
                            "upright stick and every line one upright board, and the gripper must be open and away "
                            "from the fence.")
    agent_object_names = ["wooden fence posts (square, each with a dark knob on top)",
                          "wooden fence panels (flat boards, each with a dark knob on top)",
                          "dark square marks and dark strips on the table"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs, fx = [], []
        for i, p in enumerate(POSTS):
            c = "abc"[i]
            cat = register_generated(f"LffLhfPost{c.upper()}",
                                     knobbed_xml(f"lff_lhf_post_{c}", (NOM_B / 2, NOM_B / 2, NOM_H / 2), POST_RGBA[i]))
            objs.append((p, cat, f"post{i}_region", (-0.30 + 0.12 * i, 0.30, -0.29 + 0.12 * i, 0.31)))
        for i, p in enumerate(PANELS):
            c = "ab"[i]
            cat = register_generated(f"LffLhfPanel{c.upper()}",
                                     knobbed_xml(f"lff_lhf_panel_{c}", (NOM_T / 2, NOM_W / 2, NOM_H / 2), WOOD[i]))
            objs.append((p, cat, f"panel{i}_region", (-0.30 + 0.12 * i, -0.31, -0.29 + 0.12 * i, -0.30)))
        for i, k in enumerate(MARKS):
            c = "abc"[i]
            cat = register_generated(f"LffLhfPostmark{c.upper()}",
                                     flat_xml(f"lff_lhf_postmark_{c}", (NOM_B / 2 + 0.003, NOM_B / 2 + 0.003, 0.0005),
                                              MARK_RGBA), free=False)
            fx.append((k, cat, f"mark{i}_region", (-0.30 + 0.12 * i, 0.10, -0.29 + 0.12 * i, 0.11)))
        for i, k in enumerate(STRIPS):
            c = "ab"[i]
            cat = register_generated(f"LffLhfStrip{c.upper()}",
                                     flat_xml(f"lff_lhf_strip_{c}", (STRIP_W / 2, NOM_W / 2 + 0.005, 0.0005),
                                              STRIP_RGBA), free=False)
            fx.append((k, cat, f"strip{i}_region", (-0.30 + 0.12 * i, -0.11, -0.29 + 0.12 * i, -0.10)))
        return write_bddl(self.name, "Build a small fence from the posts and panels", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.fix_bid, self.fix_vis = {}, {}
        for k in MARKS + STRIPS:
            b = m.body_name2id(self.env.fixtures_dict[k].root_body)
            self.fix_bid[k] = b
            self.fix_vis[k] = [g for g in range(m.ngeom) if m.geom_bodyid[g] in
                               {b} | {c for c in range(m.nbody) if m.body_parentid[c] == b} and m.geom_type[g] == 6]
        self.gid, self.vis, self.pbody = {}, {}, {}
        for p in OBJS:
            kind, c = p.split("_")[0], p.split("_")[1]
            self.gid[p] = {nm: m.geom_name2id(f"{p}_lff_lhf_{kind}_{c}_{nm}") for nm in ("board", "stem", "knob")}
            body = int(m.geom_bodyid[self.gid[p]["board"]])
            self.pbody[p] = body
            self.vis[p] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g] == 0
                           and m.geom_conaffinity[g] == 0]
        self.geom_owner = {}
        for p in OBJS:
            for g in self.gid[p].values():
                self.geom_owner[g] = p
        self.finger_geoms, self.hand_geoms = set(), set()
        for g in envs.robot_geom_ids(self.env):
            n = m.geom_id2name(g) or ""
            (self.finger_geoms if "finger" in n else self.hand_geoms).add(g)
        self.robot_geoms = self.finger_geoms | self.hand_geoms
        # per-geom lookup tables for the substep contact check: object index (-1 none), robot part (1 finger, 2 hand)
        self.owner = np.full(m.ngeom, -1, int)
        for i, p in enumerate(OBJS):
            self.owner[list(self.gid[p].values())] = i
        self.part = np.zeros(m.ngeom, int)
        self.part[list(self.finger_geoms)] = 1
        self.part[list(self.hand_geoms)] = 2
        self._sub_rob, self._sub_obj = {}, {}
        # Contacts are collected after every MuJoCo substep: a fast hit (a finger flicking a knob, a hanging object
        # sweeping past) can last less than one 25-substep control step (v1, measured). Wrappers chain.
        sim = self.env.sim
        if not getattr(sim, "_lhf_wrapped", False):
            orig = sim.step

            def step(*a, **k):
                orig(*a, **k)
                t = getattr(sim, "_lhf_task", None)
                if t is not None and t.mon is not None:
                    t._substep()
            sim.step = step
            sim._lhf_wrapped = True
        sim._lhf_task = self
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
        row = "lr" if rng.random() < 0.5 else "fb"
        hazard = bool((int(seed) * 7) % 20 < round(20 * P_HAZARD))  # balanced: 13 of every 20 consecutive seeds
        t, w, h = (float(rng.uniform(*T_RANGE)), float(rng.uniform(*W_RANGE)), float(rng.uniform(*H_RANGE)))
        b = float(rng.uniform(*B_RANGE))
        gap = float(rng.uniform(*GAP_RANGE))
        dens_panel, dens_post = float(rng.uniform(*DENS_RANGE)), float(rng.uniform(*DENS_RANGE))
        mu = float(rng.uniform(*MU_RANGE))
        foot = 0.0 if hazard else float(rng.uniform(*FOOT_RANGE))
        pitch = w + 2 * gap + b  # post centre to post centre
        if row == "lr":
            c = np.array([rng.uniform(-0.12, -0.06), rng.uniform(-0.05, 0.05)])
            u = np.array([0.0, 1.0])
            v = np.array([-1.0, 0.0])  # supply side: towards the robot
        else:
            side = 1.0 if rng.random() < 0.5 else -1.0
            c = np.array([rng.uniform(-0.16, -0.10), side * rng.uniform(0.09, 0.13)])
            u = np.array([1.0, 0.0])
            v = np.array([0.0, -side])  # supply side: towards the middle
        phi = float(np.arctan2(u[1], u[0]))  # direction of the fence line
        pyaw = _wrap_h(phi - np.pi / 2)  # panel yaw: body y (width) along the line, body x (thickness) across
        g = float(rng.uniform(*SUP_PITCH))
        d = float(rng.uniform(*SUP_D))
        shift = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.07))
        if row == "fb":  # keep the farthest supply slot within ~63 cm of the robot base, the far post within ~66 cm
            c[0] -= max(0.0, c[0] + shift + g - X_FAR)
            c[0] -= max(0.0, c[0] + pitch - X_FAR - 0.03)
        slots = [(c + u * (s - 2) * pitch / 2).tolist() for s in range(5)]
        sup = [c + v * (d + j * g) + u * (shift + (k - 1) * g) for j in range(2) for k in range(3)]
        pick = rng.permutation(6)[:5]
        start = {OBJS[i]: (sup[j] + rng.uniform(-0.008, 0.008, 2)).tolist() for i, j in enumerate(pick)}
        yaw0 = {p: pyaw + float(rng.uniform(-0.15, 0.15)) for p in OBJS}
        return dict(seed=int(seed), row=row, hazard=hazard, posts="light" if hazard else "footed", foot_mass=foot,
                    t=t, w=w, h=h, b=b, gap=gap, pitch=pitch, density_panel=dens_panel, density_post=dens_post, mu=mu,
                    center=c.tolist(), u=u.tolist(), v=v.tolist(), phi=phi, pyaw=pyaw, slots=slots,
                    knob_dist=w / 2 + gap + b / 2, start=start, yaw0=yaw0, supply_shift=shift, supply_pitch=g,
                    supply_d=d)

    def _set_body(self, p, half, density, foot=0.0):
        """resize p's geoms, set mass / centre of mass / inertia (board + knob + optional hidden foot), refit BVH"""
        m = self.env.sim.model
        hx, hy, hz = half
        spec = {"board": ([hx, hy, hz], [0, 0, 0], [hx, hy, hz]),
                "stem": ([STEM_R, STEM_L / 2, 0], [0, 0, hz + STEM_L / 2], [STEM_R, STEM_R, STEM_L / 2]),
                "knob": ([KNOB_R, 0, 0], [0, 0, hz + STEM_L + KNOB_R], [KNOB_R, KNOB_R, KNOB_R])}
        for nm, g in self.gid[p].items():
            sz, pos, aabb = spec[nm]
            m.geom_size[g] = sz
            m.geom_pos[g] = pos
            m.geom_aabb[g] = [0, 0, 0] + aabb
            m.geom_rbound[g] = float(np.linalg.norm(aabb))
        for nm, g in zip(("board", "stem", "knob"), self.vis[p]):
            m.geom_size[g] = spec[nm][0]
            m.geom_pos[g] = spec[nm][1]
        bd = self.pbody[p]
        X, Y, Z = 2 * hx, 2 * hy, 2 * hz
        mb = density * X * Y * Z
        zk = hz + STEM_L + KNOB_R
        zf = -hz + FOOT_H / 2
        M = mb + KNOB_MASS + foot
        zc = (KNOB_MASS * zk + foot * zf) / M
        I = np.array([mb * (Y * Y + Z * Z) / 12, mb * (X * X + Z * Z) / 12, mb * (X * X + Y * Y) / 12])
        if foot:
            I += foot * np.array([(Y * Y + FOOT_H ** 2) / 12, (X * X + FOOT_H ** 2) / 12, (X * X + Y * Y) / 12])
        par = mb * zc ** 2 + KNOB_MASS * (zk - zc) ** 2 + foot * (zf - zc) ** 2
        I[:2] += par
        m.body_mass[bd] = M
        m.body_ipos[bd] = [0, 0, zc]
        m.body_iquat[bd] = [1, 0, 0, 0]
        m.body_inertia[bd] = I
        refit_bvh(m, bd)
        return dict(mass=float(M), com_above_bottom=float(zc + hz))

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        self.body_info = {}
        for p in POSTS:
            self.body_info[p] = self._set_body(p, (inst["b"] / 2, inst["b"] / 2, inst["h"] / 2), inst["density_post"],
                                               inst["foot_mass"])
        for p in PANELS:
            self.body_info[p] = self._set_body(p, (inst["t"] / 2, inst["w"] / 2, inst["h"] / 2),
                                               inst["density_panel"])
        for p in OBJS:
            envs.set_friction(self.env, p, inst["mu"])  # board / stem / knob sliding friction ...
            m.geom_friction[self.gid[p]["knob"]] = [1.0, KNOB_MU_T, 0.0001]  # ... except the knob (grip pivot)
            m.geom_friction[self.gid[p]["stem"]] = [1.0, 0.005, 0.0001]
            m.geom_priority[self.gid[p]["knob"]] = 2
            m.geom_condim[self.gid[p]["knob"]] = 4
        self.env.sim.forward()
        q = [np.cos(inst["pyaw"] / 2), 0, 0, np.sin(inst["pyaw"] / 2)]
        for s, kind in enumerate(SLOT_KIND):
            k = MARKS[s // 2] if kind == "post" else STRIPS[s // 2]
            x, y = inst["slots"][s]
            m.body_pos[self.fix_bid[k]] = [x, y, TZ + 0.0006]
            m.body_quat[self.fix_bid[k]] = q
            for g in self.fix_vis[k]:
                m.geom_size[g] = ([inst["b"] / 2 + 0.003, inst["b"] / 2 + 0.003, 0.0005] if kind == "post" else
                                  [STRIP_W / 2, inst["w"] / 2 + 0.005, 0.0005])
        for p in OBJS:
            envs.place_upright(self.env, p, inst["start"][p], yaw=inst["yaw0"][p], clearance=0.001)

    def reset_instance(self, inst, recorder=None):
        self.mon = None
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = FenceSkills(self.env, self, recorder=recorder)
        self._sub_rob, self._sub_obj = {}, {}
        n = len(SLOT_KIND)
        self.mon = dict(state=[False] * n, raw_since=[0] * n, ever=[False] * n, lost=[False] * n, events=[],
                        last_robot={p: None for p in OBJS}, last_obj={p: None for p in OBJS},
                        fallen={p: False for p in OBJS}, occupant=[None] * n, n_steps=0)
        return self.sk

    # ------------------------------------------------------------ geometry / predicates
    @staticmethod
    def kind(p):
        return "post" if p.startswith("post") else "panel"

    def R(self, p):
        return self.env.sim.data.body_xmat[envs.body_id(self.env, p)].reshape(3, 3).copy()

    def pos(self, p):
        return envs.obj_pos(self.env, p)

    def tilt(self, p):
        return float(np.arccos(np.clip(self.R(p)[2, 2], -1.0, 1.0)))

    def knob_xyz(self, p):
        return self.env.sim.data.geom_xpos[self.gid[p]["knob"]].copy()

    def knob_stand_z(self):
        """grasp-point height (world) of a knob centre when its object stands on the table (posts = panel height)"""
        return TZ + self.inst["h"] + STEM_L + KNOB_R

    def held(self, p):
        return self.sk is not None and self.sk.holding(p)

    def on_slot(self, p, s, extra=0.0):
        """object p stands upright on fence slot s (right kind, lined up), resting on the table, not held"""
        if self.kind(p) != SLOT_KIND[s]:
            return False
        R = self.R(p)
        if R[2, 2] < np.cos(TILT_OK + (np.radians(1.0) if extra else 0.0)):
            return False
        rel = self.pos(p)[:2] - np.asarray(self.inst["slots"][s])
        du = abs(rel @ np.asarray(self.inst["u"]))
        dv = abs(rel @ np.asarray(self.inst["v"]))
        if SLOT_KIND[s] == "post":
            if max(du, dv) > TOL_POST + extra:
                return False
        else:
            if du > TOL_U + extra or dv > TOL_V + extra:
                return False
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            if abs(_wrap_h(yaw - self.inst["pyaw"])) > YAW_TOL:
                return False
        if envs.obj_min_z(self.env, p) > TZ + 0.006:
            return False
        return not self.held(p)

    def slot_occupant(self, s, extra=0.0):
        return next((p for p in OBJS if self.on_slot(p, s, extra)), None)

    def slot_filled(self, s, extra=0.0):
        return self.slot_occupant(s, extra) is not None

    def stages(self):
        return [self.slot_filled(s) for s in range(len(SLOT_KIND))]

    def stage_names(self):
        out, kc = [], {"post": 0, "panel": 0}
        for kind in SLOT_KIND:
            kc[kind] += 1
            out.append(f"{kind}{kc[kind]}")
        return out

    # ------------------------------------------------------------ monitor
    def _contacts(self):
        d = self.env.sim.data
        rob, obj = {}, {}
        for j in range(d.ncon):
            c = d.contact[j]
            a, b = c.geom1, c.geom2
            pa, pb = self.geom_owner.get(a), self.geom_owner.get(b)
            if pa is not None and pb is not None and pa != pb:
                obj[pa], obj[pb] = pb, pa
            for p, o in ((pa, b), (pb, a)):
                if p is None:
                    continue
                if o in self.finger_geoms:
                    rob[p] = "finger"
                elif o in self.hand_geoms and rob.get(p) != "finger":
                    rob[p] = "hand"
        return rob, obj

    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        mon["n_steps"] += 1
        t = self.sk.n_steps
        rob = {OBJS[i]: ("finger" if v == 1 else "hand") for i, v in self._sub_rob.items()}
        obj = {OBJS[i]: OBJS[j] for i, j in self._sub_obj.items()}
        self._sub_rob, self._sub_obj = {}, {}
        held = [p for p in OBJS if self.held(p)]
        for p, part in rob.items():
            if p not in held:
                mon["last_robot"][p] = (part, t, "open" if self.sk.grip < 0 else "closed")
        for p, q in obj.items():
            if p not in held:
                mon["last_obj"][p] = (q + (" (held)" if q in held else ""), t)
        for p in OBJS:
            f = self.tilt(p) > FALLEN
            if f != mon["fallen"][p]:
                mon["fallen"][p] = f
                if f:
                    mon["events"].append(dict(object=p, event="toppled", step=int(t), cause=self._cause(p, t)))
                else:
                    mon["events"].append(dict(object=p, event="stood_up", step=int(t)))
        if mon["n_steps"] % 2:
            return
        names = self.stage_names()
        for s in range(len(SLOT_KIND)):
            occ = self.slot_occupant(s, HYST if mon["state"][s] else 0.0)
            raw = occ is not None
            if raw:
                mon["occupant"][s] = occ
            if raw == mon["state"][s]:
                mon["raw_since"][s] = 0
                continue
            mon["raw_since"][s] += 2
            if mon["raw_since"][s] < DEBOUNCE:
                continue
            mon["state"][s] = raw
            mon["raw_since"][s] = 0
            t0 = t - DEBOUNCE + 1
            if raw:
                ev = dict(stage=names[s], event="restored" if mon["lost"][s] else "achieved", step=int(t0))
                mon["ever"][s] = True
            else:
                mon["lost"][s] = True
                ev = dict(stage=names[s], event="lost", step=int(t0), cause=self._slot_cause(s, t))
            mon["events"].append(ev)

    def _cause(self, p, t):
        if self.held(p):
            return "picked_up"
        lr, lo = self.mon["last_robot"][p], self.mon["last_obj"][p]
        cands = []
        if lr is not None and t - lr[1] <= 60:
            cands.append((lr[1], f"robot_{lr[0]}" + ("_opening" if lr[2] == "open" else "")))
        if lo is not None and t - lo[1] <= 60:
            cands.append((lo[1] - 30, "object:" + lo[0]))  # prefer the robot when both touched it recently
        return max(cands)[1] if cands else "other"

    def _slot_cause(self, s, t):
        """why slot s lost its object: the cause recorded for the object that last stood on it"""
        p = self.mon["occupant"][s]
        if p is None:
            c = np.asarray(self.inst["slots"][s])
            p = min(OBJS, key=lambda q: float(np.linalg.norm(self.pos(q)[:2] - c)))
        return self._cause(p, t)

    # ------------------------------------------------------------ outcome
    def _slot_name(self, s):
        kind = SLOT_KIND[s]
        idx = [i for i in range(len(SLOT_KIND)) if SLOT_KIND[i] == kind]
        n = len(idx)
        k = idx.index(s)  # 0-based along +u
        if self.inst["row"] == "lr":  # u = +y = robot's left: count from the robot's right
            what = "square mark" if kind == "post" else "strip"
            return f"{what} {k + 1} of {n} (counting from the robot's right)"
        what = "square mark" if kind == "post" else "strip"
        u = self.inst["u"]
        return f"{what} {k + 1 if u[0] > 0 else n - k} of {n} (counting from the one nearest the robot)"

    def _dir_words(self, d):
        parts = []
        if abs(d[0]) >= 0.01:
            parts.append(f"{100 * abs(d[0]):.0f} cm {'forward' if d[0] > 0 else 'back'}")
        if abs(d[1]) >= 0.01:
            parts.append(f"{100 * abs(d[1]):.0f} cm {'left' if d[1] > 0 else 'right'}")
        return (" and ".join(parts) + " of") if parts else "right at"

    def placed_objects(self):
        return {p for p in OBJS if any(self.on_slot(p, s) for s in range(len(SLOT_KIND)))}

    def outcome(self, inst):
        sk = self.sk
        sk.hold(10)
        p0 = {p: self.pos(p) for p in OBJS}
        touching = False
        for _ in range(20):
            sk.hold(1)
            rob, _ = self._contacts()
            touching = touching or any(p in rob for p in OBJS)
        moved = max(float(np.linalg.norm(self.pos(p) - p0[p])) for p in OBJS)
        static = moved < 0.004
        raw = self.stages()
        held = [p for p in OBJS if self.held(p)]
        mon = self.mon
        ev = list(mon["events"]) if mon else []
        n_reg = sum(e["event"] == "lost" for e in ev)
        n_res = sum(e["event"] == "restored" for e in ev)
        n_top = sum(e["event"] == "toppled" for e in ev)
        success = bool(all(raw) and static and not held and not touching)
        names = self.stage_names()
        parts = []
        ever = mon["ever"] if mon else [False] * len(raw)
        if success:
            failure = "none"
            parts.append("The fence is complete: every square mark carries an upright post and every strip an "
                         "upright panel lined up with it.")
        else:
            lost_final = [s for s in range(len(raw)) if not raw[s] and ever[s]]
            never = [s for s in range(len(raw)) if not raw[s] and not ever[s]]
            if held:
                failure = "held"
            elif touching:
                failure = "touching"
            elif lost_final:
                failure = "regressed"
            elif never:
                failure = "incomplete"
            else:
                failure = "not_static"
            placed = self.placed_objects()
            described = set()
            for s in range(len(raw)):
                if raw[s]:
                    continue
                kind = SLOT_KIND[s]
                c = np.asarray(inst["slots"][s])
                spare = [q for q in OBJS if self.kind(q) == kind and q not in placed and q not in described]
                near = min(spare, key=lambda q: float(np.linalg.norm(self.pos(q)[:2] - c))) if spare else None
                dist = float(np.linalg.norm(self.pos(near)[:2] - c)) if near else 1.0
                if near is None or (not ever[s] and dist > 0.06):
                    txt = f"{self._slot_name(s)} has no {kind} on it (none was put there)."
                else:
                    described.add(near)
                    tl = np.degrees(self.tilt(near))
                    state = ("lies fallen" if tl > 45 else f"leans {tl:.0f} degrees" if tl > np.degrees(TILT_OK)
                             else "stands")
                    txt = (f"{self._slot_name(s)} has no upright {kind} on it: the nearest {kind} {state} "
                           f"{self._dir_words(self.pos(near)[:2] - c)} it.")
                if ever[s]:
                    lost = [e for e in ev if e.get("stage") == names[s] and e["event"] == "lost"]
                    why = lost[-1]["cause"] if lost else "other"
                    txt += {"robot_finger_opening": f" The {kind} had stood there; a finger pushed it when the "
                                                    f"gripper opened.",
                            "robot_hand_opening": f" The {kind} had stood there; the hand pushed it.",
                            }.get(why, f" The {kind} had stood there and was knocked over later."
                                  if why.startswith(("robot", "object")) else f" The {kind} had stood there earlier.")
                parts.append(txt)
            if all(raw) and held:
                parts.append("The gripper still holds an object.")
            elif touching:
                parts.append("The gripper is still touching the fence.")
            elif all(raw) and not static:
                parts.append("Something was still moving.")
        if n_reg:
            parts.append(f"During the attempt objects on the fence line were knocked off {n_reg} time(s) and stood "
                         f"up again {n_res} time(s).")
        if any(w in " ".join(parts) for w in (" forward of", " back of", " back and", " forward and")):
            parts.append("(Forward = away from the robot.)")
        return dict(success=success, stages={names[s]: bool(raw[s]) for s in range(len(raw))}, events=ev,
                    n_regressions=int(n_reg), n_restored=int(n_res), n_toppled=int(n_top), failure=failure,
                    held=held, touching=bool(touching), static=bool(static),
                    tilts_deg={p: round(float(np.degrees(self.tilt(p))), 1) for p in OBJS}, detail=" ".join(parts))

    # ------------------------------------------------------------ scripted policies (move_to + gripper only)
    def _go(self, xyz, yaw=None, speed=None, tol=0.004, max_steps=300):
        return self.sk.move_to(xyz, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)

    def axis(self, p, i):
        a = self.R(p)[:2, i]
        return a / max(np.linalg.norm(a), 1e-9)

    def grasp_yaw(self, p, mode):
        """yaw for a top-down knob grasp. Lying object: fingers across its long axis (it swings up and hangs).
        Panel: 'habit' = fingers along its width (hand along the board; along the fence line once placed),
        'thin' = fingers along its thickness (across the line). Post: 'thin' = fingers across the fence line, otherwise
        the side direction needing the least rotation (a post's own orientation does not matter: square, round knob)."""
        R = self.R(p)
        if R[2, 2] < np.cos(np.radians(30)):
            ax = R[:2, 2]
            if np.linalg.norm(ax) > 1e-3:
                return _yaw_for_axis(np.array([-ax[1], ax[0]]))
        if self.kind(p) == "panel":
            return _yaw_for_axis(self.axis(p, 1) if mode == "habit" else self.axis(p, 0))
        if mode == "thin":  # careful: a post's fingers across the fence line too (they reach nothing there)
            return _yaw_for_axis(np.asarray(self.inst["v"]))
        y0 = _yaw_for_axis(self.axis(p, 0))
        return min((_wrap_h(y0), _wrap_h(y0 + np.pi / 2)), key=lambda y: abs(_wrap_h(y - self.sk.yaw)))

    def obj_yaw(self, p):
        R = self.R(p)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def pick(self, p, mode, careful, z_carry=None, noise=None, v=0.15):
        """grasp p's knob and lift it; returns (object yaw - gripper yaw) or None if not held"""
        sk = self.sk
        k = self.knob_xyz(p)
        if noise is not None:
            k = k + np.array([noise[0], noise[1], 0.0])
        yaw = self.grasp_yaw(p, mode)
        zk = max(k[2], TZ + 0.006)
        sk.set_gripper(False, steps=8)
        if careful:
            cur = sk.eef_pos()
            if cur[2] < TZ + Z_SAFE - 0.01:
                self._go([cur[0], cur[1], TZ + Z_SAFE])
            self._go([k[0], k[1], TZ + Z_SAFE], yaw=yaw, speed=0.3)
            self._go([k[0], k[1], zk + 0.04], speed=0.25)
            self._go([k[0], k[1], zk], speed=0.1, tol=0.003, max_steps=150)
        else:  # Sonnet-like: over the knob at carry height, down at 0.1 to 1.5 cm above, last 1.5 cm at 0.04
            zt = TZ + (Z_NAIVE if z_carry is None else z_carry)
            cur = sk.eef_pos()
            self._go([k[0], k[1], max(cur[2], zt, zk + 0.06)], yaw=yaw, speed=v)
            self._go([k[0], k[1], zk + 0.015], speed=0.1)
            self._go([k[0], k[1], zk], speed=0.04, tol=0.003, max_steps=150)
        sk.set_gripper(True, steps=15)
        if careful:
            self._go([k[0], k[1], TZ + Z_SAFE], speed=0.25)
        else:
            self._go([k[0], k[1], TZ + (Z_NAIVE if z_carry is None else z_carry)], speed=0.08)
        sk.hold(10)
        if not self.held(p):
            sk.set_gripper(False, steps=8)  # already lifted: the fingers open well above the object
            return None
        return float(self.obj_yaw(p) - sk.yaw)

    def release_z(self, p):
        """grasp-point height at which the held object's bottom is 2 mm above the table (a heavy object's knob can
        sit a few mm low in the fingers; pressing it onto the table would squeeze the knob out of the grip)"""
        dz = float(self.sk.eef_pos()[2] - self.knob_xyz(p)[2]) if self.held(p) else 0.0
        return self.knob_stand_z() + 0.002 + float(np.clip(dz, -0.005, 0.015))

    def place_yaw(self, p, offset):
        """gripper yaw that lines p up with the fence (panel: width along the line; post: sides along it)"""
        if self.kind(p) == "panel":
            return _wrap_h(self.inst["pyaw"] - offset)
        cands = [_wrap_h(self.inst["pyaw"] + k * np.pi / 2 - offset) for k in range(4)]
        return min(cands, key=lambda y: abs(_wrap_h(y - self.sk.yaw)))

    def place(self, p, s, offset, careful, z_carry=None, noise=None, lift=None, v=0.15):
        """carry the held object to fence slot s, lined up with the fence; set it down; let go"""
        sk = self.sk
        xy = np.asarray(self.inst["slots"][s], float) + (np.asarray(noise, float) if noise is not None else 0.0)
        yaw = self.place_yaw(p, offset)
        zr = self.release_z(p)
        if careful:
            cur = sk.eef_pos()
            if cur[2] < TZ + Z_SAFE - 0.01:
                self._go([cur[0], cur[1], TZ + Z_SAFE])
            self._go([xy[0], xy[1], TZ + Z_SAFE], yaw=yaw, speed=0.3)
            sk.hold(15)
            zr = self.release_z(p)
            self._go([xy[0], xy[1], zr + 0.03], speed=0.2)
            self._go([xy[0], xy[1], zr], speed=0.06, tol=0.003, max_steps=150)
            sk.set_gripper(False, steps=12)
            self._go([xy[0], xy[1], zr + 0.04], speed=0.1)
            self._go([xy[0], xy[1], TZ + Z_SAFE], speed=0.3)
        else:  # Sonnet-like: to above the slot at carry height, 0.1 m/s to 1.5 cm above, last 1.5 cm at 0.03, open
            zt = TZ + (Z_NAIVE if z_carry is None else z_carry)
            cur = sk.eef_pos()
            self._go([xy[0], xy[1], max(cur[2], zt)], yaw=yaw, speed=v)
            self._go([xy[0], xy[1], zr + 0.015], speed=0.1)
            self._go([xy[0], xy[1], zr], speed=0.03, tol=0.003, max_steps=150)
            sk.set_gripper(False, steps=12)
            self._go([xy[0], xy[1], TZ + (Z_NAIVE if lift is None else lift)], speed=0.15)
        sk.hold(5)

    # orders ---------------------------------------------------------------
    def supply_center(self):
        return np.mean([self.inst["start"][p] for p in OBJS], axis=0)

    def fence_order(self, far_first=True):
        """slot indices from the end of the line farthest from the supply to the near end (or reversed)"""
        sc = self.supply_center()
        first, last = np.asarray(self.inst["slots"][0]), np.asarray(self.inst["slots"][-1])
        idx = list(range(len(SLOT_KIND)))
        near_first = idx if np.linalg.norm(first - sc) <= np.linalg.norm(last - sc) else idx[::-1]
        return near_first[::-1] if far_first else near_first

    def build(self, order, careful, panel_mode, noise_fn=lambda: None, z_carry=None, lift=None, noise_pick=False,
              v=0.15):
        """put an object on every slot that has none, in the given slot order; the object for each slot: an unused
        one of the right kind, the nearest to the gripper (naive) / to the slot (careful). No looking back."""
        fails = {}
        used = set()
        for s in order:
            if self.slot_filled(s):
                continue
            for _ in range(2):
                cand = [p for p in OBJS if self.kind(p) == SLOT_KIND[s] and p not in used and fails.get(p, 0) < 2
                        and p not in self.placed_objects()]
                if not cand:
                    break
                ref = np.asarray(self.inst["slots"][s]) if careful else self.sk.eef_pos()[:2]
                p = min(cand, key=lambda q: float(np.linalg.norm(self.pos(q)[:2] - ref)))
                mode = panel_mode if self.kind(p) == "panel" else ("thin" if careful else "any")
                off = self.pick(p, mode, careful, z_carry=z_carry, noise=noise_fn() if noise_pick else None, v=v)
                if off is None:
                    fails[p] = fails.get(p, 0) + 1
                    continue
                used.add(p)
                self.place(p, s, off, careful, z_carry=z_carry, noise=noise_fn(), lift=lift, v=v)
                break

    def footprint(self, p):
        """sample points of p's collision box projected onto the table (corners, edge and face midpoints) + knob"""
        R, c = self.R(p), self.pos(p)
        if self.kind(p) == "panel":
            h = np.array([self.inst["t"], self.inst["w"], self.inst["h"]]) / 2
        else:
            h = np.array([self.inst["b"], self.inst["b"], self.inst["h"]]) / 2
        pts = [c + R @ (h * [a, b, z]) for a in (-1, 0, 1) for b in (-1, 0, 1) for z in (-1, 0, 1)]
        pts.append(self.knob_xyz(p))
        return np.array(pts)[:, :2]

    def slot_blocked(self, s, ignore=()):
        """an object that is not standing on a slot lies (partly) on slot s or right next to it"""
        c = np.asarray(self.inst["slots"][s])
        u, v = np.asarray(self.inst["u"]), np.asarray(self.inst["v"])
        hu = (self.inst["w"] if SLOT_KIND[s] == "panel" else self.inst["b"]) / 2 + 0.003
        hv = (self.inst["t"] if SLOT_KIND[s] == "panel" else self.inst["b"]) / 2 + 0.008
        placed = self.placed_objects()
        for q in OBJS:
            if q in ignore or self.held(q) or q in placed:  # objects standing on their own slots are fine
                continue
            d = self.footprint(q) - c
            if np.any((np.abs(d @ u) < hu) & (np.abs(d @ v) < hv)):
                return True
        return False

    def fingers_along_width(self, p):
        """the held panel's width axis is (nearly) parallel to the finger axis"""
        fa = np.array([-np.sin(self.sk.yaw), np.cos(self.sk.yaw)])
        return abs(float(self.axis(p, 1) @ fa)) > 0.7

    def repair(self, rounds=None):
        """look at every slot; stand fallen / misplaced objects up again by their knobs (careful motions, panels
        released with the fingers across the line). Order: an object resting on another first, then fallen / leaning
        ones from the top of the pile down, then upright ones that are off their slots. A panel lifted with its
        fingers along its width (a fallen panel can only be lifted that way) is first set down standing at a free spot
        and re-grasped. Returns the number of objects moved."""
        n = len(SLOT_KIND)
        moved = 0
        fails, moves = {}, {}
        for _ in range(rounds or 16):
            if all(self.stages()):
                break
            ok = self.placed_objects()
            loose = [p for p in OBJS if p not in ok and fails.get(p, 0) < 2 and moves.get(p, 0) < 4]
            if not loose:
                break
            on_other = [p for p in loose if envs.obj_min_z(self.env, p) > TZ + 0.006]
            down = [p for p in loose if p not in on_other and self.tilt(p) > TILT_OK]
            upright = [p for p in loose if p not in on_other and p not in down]
            if on_other:
                p = max(on_other, key=lambda q: self.knob_xyz(q)[2])
            elif down:
                p = max(down, key=lambda q: self.knob_xyz(q)[2])
            else:
                fit = {q: [s for s in range(n) if SLOT_KIND[s] == self.kind(q) and not self.slot_filled(s)
                           and not self.slot_blocked(s, ignore=(q,))] for q in upright}
                fit = {q: js for q, js in fit.items() if js}
                if not fit:
                    break
                p = min(fit, key=lambda q: min(float(np.linalg.norm(self.pos(q)[:2] - np.asarray(
                    self.inst["slots"][s]))) for s in fit[q]))
            off = self.pick(p, "thin", True)
            if off is None:
                fails[p] = fails.get(p, 0) + 1
                continue
            moved += 1
            moves[p] = moves.get(p, 0) + 1
            if self.kind(p) == "panel" and self.fingers_along_width(p):
                self.place_at(p, self.free_spot(), off)
                continue
            clear = [s for s in range(n) if SLOT_KIND[s] == self.kind(p) and not self.slot_filled(s)
                     and not self.slot_blocked(s, ignore=(p,))]
            if not clear:
                self.place_at(p, self.free_spot(), off)
                continue
            here = self.sk.eef_pos()[:2]
            s = min(clear, key=lambda j: float(np.linalg.norm(np.asarray(self.inst["slots"][j]) - here)))
            self.place(p, s, off, True)
        return moved

    def free_spot(self):
        """the reachable table spot (around the fence and the supply) farthest from every object and slot"""
        c, v = np.asarray(self.inst["center"]), np.asarray(self.inst["v"])
        u = np.asarray(self.inst["u"])
        best, bc = None, -1.0
        for a in np.linspace(-0.20, 0.20, 9):
            for b in (-0.18, -0.12, 0.10, 0.20, 0.32):
                q = c + v * b + u * a
                if not (-0.40 < q[0] < 0.08 and -0.32 < q[1] < 0.32) or np.hypot(q[0] + 0.66, q[1]) > 0.60:
                    continue
                cl = min(float(np.linalg.norm(self.pos(o)[:2] - q)) for o in OBJS)
                cl = min(cl, min(float(np.linalg.norm(np.asarray(sl) - q)) for sl in self.inst["slots"]))
                if cl > bc:
                    best, bc = q, cl
        return best

    def place_at(self, p, xy, offset):
        """careful: set the held object down standing at a free spot"""
        sk = self.sk
        zr = self.release_z(p)
        cur = sk.eef_pos()
        if cur[2] < TZ + Z_SAFE - 0.01:
            self._go([cur[0], cur[1], TZ + Z_SAFE])
        self._go([xy[0], xy[1], TZ + Z_SAFE], yaw=self.place_yaw(p, offset), speed=0.3)
        sk.hold(15)
        zr = self.release_z(p)
        self._go([xy[0], xy[1], zr + 0.03], speed=0.2)
        self._go([xy[0], xy[1], zr], speed=0.06, tol=0.003, max_steps=200)
        sk.set_gripper(False, steps=12)
        self._go([xy[0], xy[1], zr + 0.04], speed=0.1)
        self._go([xy[0], xy[1], TZ + Z_SAFE], speed=0.3)

    # ------------------------------------------------------------ policies
    def execute(self, inst, params):
        careful = params["careful"]
        noise = params.get("noise", 0.0)
        rng = np.random.default_rng(params.get("noise_seed", 0))
        nf = (lambda: rng.uniform(-noise, noise, 2)) if noise else (lambda: None)
        if params.get("order") == "panels_first":
            fo = self.fence_order(True)
            order = [s for s in fo if SLOT_KIND[s] == "panel"] + [s for s in fo if SLOT_KIND[s] == "post"]
        elif params.get("order") == "posts_first":
            fo = self.fence_order(True)
            order = [s for s in fo if SLOT_KIND[s] == "post"] + [s for s in fo if SLOT_KIND[s] == "panel"]
        else:
            order = self.fence_order(params.get("far_first", True))
        self.build(order, careful, params["panel_mode"], noise_fn=nf, z_carry=params.get("z_carry"),
                   lift=params.get("lift"), noise_pick=params.get("noise_pick", False), v=params.get("v", 0.15))
        if params.get("check"):
            self.repair()
        cur = self.sk.eef_pos()
        if cur[2] < TZ + 0.20:
            self._go([cur[0], cur[1], TZ + 0.20], speed=0.3)
        self.sk.hold(10)

    # naive = Sonnet 5.5's habits measured in the lh2_fence pilot: hand (finger axis) along the board's long side, fill
    # from the far end, carry 21 cm above the table at 0.15 m/s, slow last 1.5 cm, ~1 cm placement error, no re-check
    NAIVE = dict(careful=False, panel_mode="habit", far_first=True, check=False, noise=NAIVE_ERR)
    CAREFUL = dict(careful=True, panel_mode="thin", far_first=True, check=True)
    V1_NAIVE = dict(careful=False, panel_mode="thin", far_first=False, check=False)  # v1's scripted naive habits

    def default_params(self, inst):
        return dict(self.NAIVE, noise_seed=int(inst["seed"]) * 11)

    def oracle_params(self, inst):
        return dict(self.CAREFUL)

    def adapt_params(self, inst, history):
        """from attempt 2 (after a failure): panels released with the fingers across the line, careful, final check"""
        return dict(self.CAREFUL)

    def blind_params(self, inst, attempt, rng):
        """the naive behaviour with random variations, no history, no check"""
        return dict(self.NAIVE, noise_seed=int(rng.integers(1 << 30)), z_carry=float(rng.uniform(0.15, 0.23)),
                    lift=float(rng.uniform(0.15, 0.23)), v=float(rng.uniform(0.08, 0.25)))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        single = ("oracle", "naive", "oracle_noise", "v1_naive", "naive_thin", "naive_panels_first",
                  "naive_posts_first")
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
            elif kind == "switch":  # control: cross-attempt learning only (careful from attempt 2), no check
                params = self.default_params(inst) if a == 0 else dict(self.CAREFUL, check=False)
            elif kind == "oracle_noise":  # check: careful with +-5 mm (uniform, per axis) placement noise
                params = dict(self.CAREFUL, noise=0.005, noise_seed=int(seed) * 7 + a)
            elif kind == "v1_naive":  # control: v1's scripted naive (fingers across the line, near end first)
                params = dict(self.V1_NAIVE)
            elif kind == "naive_thin":  # control: Sonnet-like but panels released with the fingers across the line
                params = dict(self.default_params(inst), panel_mode="thin")
            elif kind == "naive_panels_first":  # control: Sonnet-like grasp, both panels before the posts
                params = dict(self.default_params(inst), order="panels_first")
            elif kind == "naive_posts_first":  # control: Sonnet-like grasp, all posts first (how fences are built)
                params = dict(self.default_params(inst), order="posts_first")
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
