"""L3 card_shelf (2026-10-12): a deepened l3_card_edge. The thin card lies on a raised SHELF (thin top on four legs)
and hides a weight in one end; sliding the heavy end over the edge far enough to grasp it tips the card off the shelf
onto the table, where it lies flat and cannot be picked up: the attempt is lost.

Why (../research_notes/P1P2_加深与常识任务设计.md sec. 1, 3.2): l3_card_edge saturated. Sonnet 5.5 (cla4) failed one
flat grasp, then in the same attempt slid the card 2-3 cm over the board's edge and grasped the overhang (both runs
succeeded on attempt 1): the failed grasp costs one command and changes nothing. Here the first common-sense
interaction (slide the nearest end out ~3 cm) has an irreversible cost whenever that end is the heavy one.

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y): a wooden shelf
(top 22 x 26 x 1 cm on four 2 x 2 cm legs, top surface 10 cm above the table) with a thin red card (9.0 x 5.6 x 0.2 cm)
lying flat on it, long side along x or along y (+-8 deg). Its near end is 2.5-4 cm from the shelf edge it points to (an
x-card's near end is always the FORWARD one: a card that tips off the backward edge lands behind the shelf, hidden from
the front camera), the far end 9-14 cm from the opposite edge, both long sides >= 5 cm inside; a grey bin (16 x 16 cm
inside, 5 cm high) stands on the table beside the shelf (left or right, 6 cm gap). Same card size, mass (10 g),
friction and bin as l3_card_edge.

Hidden variable (instance level, the same in every attempt, invisible): 8 of the card's 10 g are a weight hidden in one
end; its centre of mass sits COM_E = 2.8 cm from the middle towards that end. Both ends look identical. In 14 of every
20 consecutive seeds the heavy end is the NEAR end (the end a Sonnet-like agent slides out), else the far end.

Mechanism (physics, no teleport): an end slid out by h tips over the edge once the centre of mass passes it. Measured
(card teleported with overhang h, 2 s): heavy end stays for h <= 1.7 cm, tips for h >= 1.8 cm; light end stays up to
7.0 cm; a uniform card up to ~4.5 cm. The grasp of an overhang needs the fingers past the edge: light end 1.8-3.8 cm
out -> 11/12 grasped; the heavy end only in a narrow window (1.3-1.5 cm out, grasp 0.8-1.2 cm beyond the edge: 12/36).
A tipped card is flat on the table within 0.4 s (56/56 tilt 0), where it cannot be gripped (0/56 regrasps) and no edge
is in reach. The fix is to slide the OTHER end out (a longer slide across the shelf), learnable from the failure: "only
~2 cm out and it already tipped -> that end is heavy" (the cheaper explanation "I pushed too far" leads to pushing
less, which leaves too little overhang to grasp). Validation (seeds 1000-1019): Sonnet-like naive tips 14/14 heavy-near
instances and succeeds on 6/6 light-near ones; the same habit succeeds 20/20 on l3_card_edge and 20/20 here with a
uniform card.

Implementation notes: the centre of mass is fixed in the generated MJCF (explicit <inertial pos="COM_E 0 0">) and the
heavy end is chosen by the card's yaw (the card is symmetric), so the body's inertial frame is never changed at runtime
(no bounding-volume refit needed; see lh_fence / envs.refit_bounds for that MuJoCo pitfall). The inertia is written in
scientific notation: objects._fmt (5 decimals) rounds a 10-30 g card's inertia to ~0 (a 10 g card does not compile; a
30 g card with rounded inertia got pinched and lifted by the flat grasp 4/16). Mass 10 g as in l3_card_edge: the flat
grasp then fails 20/20 (fingers slide over the card). The shelf is a static multi-geom fixture; its legs stand inside
its top's footprint (robot-leg contacts: 0 in all scripted runs).

Protocol: cross (reset to the same instance before every attempt), k = 3. Outcome: success = the card lies in the bin,
not held; plus a per-step event log (end / side slid over an edge, tipped_off, lifted, released, in_bin, hand_hit_leg)
and R6 fields (tipped_off, trap_fired, grasp_tries) for the server log; F1 gives the agent only success / failure.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, bin_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

TZ = envs.TABLE_Z
CARD_HALF = (0.045, 0.028, 0.001)  # 9.0 x 5.6 x 0.2 cm (as l3_card_edge)
CARD_MASS = 0.010  # as l3_card_edge (flat grasp fails 20/20; a 30 g card was pinched and lifted 4/16)
SLUG_MASS = 0.008  # hidden weight (+ 2 g of card); slug centre 3.5 cm from the middle
COM_E = 0.028  # centre of mass this far from the card's middle, towards the heavy end (the card's local +x)
MU = 0.3  # card and shelf sliding friction (as l3_card_edge)
SHELF_HALF = (0.11, 0.13)  # top half extents (as l3_card_edge's board)
SHELF_H = 0.10  # top surface above the table
TOP_T = 0.01  # top thickness
LEG = 0.01  # leg half width (2 x 2 cm legs)
LEG_IN = 0.012  # leg centre to the top's edges
SHELF_TOP = TZ + SHELF_H
CARD_REST_Z = SHELF_TOP + CARD_HALF[2]
BIN_IN = 0.08  # inner half width
BIN_WALL_H = 0.05
BIN_GAP = 0.06  # shelf edge to bin outer wall
BIN_T = 0.006
BIN_HALF_OUT = BIN_IN + BIN_T
TIP_BELOW_SITE = envs.FINGERTIP_BELOW_SITE
LOW_CLOSE = 0.02  # a close counts as a grasp try when the fingertips are at most this far above the card

# instance ranges
P_HEAVY_NEAR = 0.7  # 14 of every 20 consecutive seeds
NEAR_GAP = (0.025, 0.040)  # near end to its shelf edge
SIDE_CLEAR = 0.05  # long sides to the shelf's side edges, at least
YAW_JIT = np.radians(8.0)

# scripted policies
GRASP_OUT = 0.022  # grasp point this far beyond the edge (fingers must clear the edge)
DRAG_SPEED = 0.08
NAIVE_STEPS = (0.0, 0.01, 0.02, 0.03)  # Sonnet-like: slide the end to the edge, then out in 1 cm steps, lift & look
ORACLE_STEPS = (0.02, 0.03)  # l3_card_edge oracle: to 1 cm short, look, the rest

EDGES = ("forward", "backward", "left", "right")  # shelf edges in robot-centric words (+x, -x, +y, -y)
EDGE_U = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
OPPOSITE = {"forward": "backward", "backward": "forward", "left": "right", "right": "left"}


def card_xml(name, rgba=(0.85, 0.15, 0.12, 1.0)):
    """Uniform-looking card; the inertial (10 g, centre of mass COM_E along local +x) models an 8 g weight hidden in
    the +x end. Explicit inertial pos != 0, so MuJoCo keeps a separate inertial frame (see module doc)."""
    h = CARD_HALF
    ms = CARD_MASS - SLUG_MASS
    xs = COM_E * CARD_MASS / SLUG_MASS  # slug centre
    sl = (0.0075, 0.021, h[2])  # slug half extents (1.5 x 4.2 x 0.2 cm)
    box_i = lambda mm, a: mm / 3 * np.array([a[1] ** 2 + a[2] ** 2, a[0] ** 2 + a[2] ** 2, a[0] ** 2 + a[1] ** 2])
    par = lambda mm, r: mm * np.array([r[1] ** 2 + r[2] ** 2, r[0] ** 2 + r[2] ** 2, r[0] ** 2 + r[1] ** 2])
    c = np.array([COM_E, 0.0, 0.0])
    inertia = box_i(ms, h) + par(ms, -c) + box_i(SLUG_MASS, sl) + par(SLUG_MASS, np.array([xs, 0.0, 0.0]) - c)
    g = (f'        <inertial pos="{_fmt(c)}" mass="{CARD_MASS}" diaginertia="{" ".join(f"{v:.6e}" for v in inertia)}" />\n'
         f'        <geom name="{name}_g0" type="box" size="{_fmt(h)}" rgba="{_fmt(rgba)}" '
         f'friction="{MU} 0.005 0.0001" {_SOLID} group="0" condim="4"/>\n'
         f'        <geom type="box" size="{_fmt(h)}" rgba="{_fmt(rgba)}" contype="0" conaffinity="0" group="1"/>')
    return _wrap(name, g, h[2], float(np.hypot(h[0], h[1])))


def shelf_xml(name, rgba=(0.62, 0.45, 0.27, 1.0), leg_rgba=(0.50, 0.36, 0.21, 1.0)):
    """Static shelf: top plate on four legs; body origin at mid height (top surface at +SHELF_H / 2)."""
    hx, hy = SHELF_HALF
    H = SHELF_H / 2
    parts = [((0.0, 0.0, H - TOP_T / 2), (hx, hy, TOP_T / 2), rgba)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(((sx * (hx - LEG_IN), sy * (hy - LEG_IN), -TOP_T / 2), (LEG, LEG, H - TOP_T / 2), leg_rgba))
    g = []
    for k, (p, s, c) in enumerate(parts):
        g.append(f'        <geom name="{name}_g{k}" type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" '
                 f'density="500" friction="{MU} 0.005 0.0001" {_SOLID} group="0" />')
        g.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" contype="0" '
                 f'conaffinity="0" group="1" />')
    return _wrap(name, "\n".join(g), H, float(np.hypot(hx, hy)))


def _yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class ShelfSkills(Skills):
    """Skills that feed the task's per-step monitor and log every close of the gripper."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task
        self.closes = []

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()

    def set_gripper(self, close, steps=15):
        was_open = self.grip < 0
        before = self.task.close_context() if (close and was_open) else None
        rep = super().set_gripper(close, steps)
        if before is not None:
            before.update(held=self.holding("card_1"), width=round(self.gripper_width(), 4))
            self.closes.append(before)
        return rep


@register_task
class CardShelf(LFFTask):
    name = "l3_card_shelf"
    level = "L3"
    category = "strategy_switching"
    capabilities = ("Perceive", "Reason", "Plan")
    protocol = "cross"
    max_attempts = 3
    agent_max_steps = 40
    instruction = "Pick up the thin red card lying on the wooden shelf and put it into the grey bin."
    instruction_indirect = "Put the flat red rectangle that lies on the raised wooden platform into the grey container."

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        card = register_generated("LffShelfCard", card_xml("lff_shelf_card"))
        shelf = register_generated("LffCardShelf", shelf_xml("lff_card_shelf"), free=False)
        bin_ = register_generated("LffShelfBin", bin_xml("lff_shelf_bin", (BIN_IN, BIN_IN), BIN_WALL_H, wall_t=BIN_T,
                                                         rgba=(0.55, 0.56, 0.60, 1.0)), free=False)
        objs = [("card_1", card, "card_region", (0.10, 0.20, 0.12, 0.22))]
        fx = [("shelf_1", shelf, "shelf_region", (-0.11, -0.01, -0.09, 0.01)),
              ("bin_1", bin_, "bin_region", (-0.11, -0.31, -0.09, -0.29))]
        return write_bddl(self.name, self.instruction, objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.shelf_bid = m.body_name2id(self.env.fixtures_dict["shelf_1"].root_body)
        self.bin_bid = m.body_name2id(self.env.fixtures_dict["bin_1"].root_body)
        g0 = int(m.body_geomadr[self.shelf_bid])
        self.leg_geoms = {g for g in range(g0, g0 + int(m.body_geomnum[self.shelf_bid]))
                          if (m.geom_id2name(g) or "").endswith(("_g1", "_g2", "_g3", "_g4"))}
        self.robot_geoms = set(envs.robot_geom_ids(self.env))
        self.inst = None
        self.mon = None

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        bx, by = float(rng.uniform(-0.14, -0.06)), float(rng.uniform(-0.08, 0.08))
        along = str(rng.choice(["x", "y"]))
        near_sign = 1.0 if rng.random() < 0.5 else -1.0
        if along == "x":  # the near end of an x-card is the forward one: a card that tips off the backward edge lands
            near_sign = 1.0  # behind the shelf, hidden from the front camera (measured); forward = nearest = away from the robot
        jit = float(rng.uniform(-YAW_JIT, YAW_JIT))
        gap = float(rng.uniform(*NEAR_GAP))
        heavy_near = bool((int(seed) * 7) % 20 < round(20 * P_HEAVY_NEAR))  # balanced per 20 consecutive seeds
        a_long, a_side = (0, 1) if along == "x" else (1, 0)  # world axis of the card's long / short side
        ext_long = CARD_HALF[0] * abs(np.cos(jit)) + CARD_HALF[1] * abs(np.sin(jit))  # half extents incl. the jitter
        ext_side = CARD_HALF[0] * abs(np.sin(jit)) + CARD_HALF[1] * abs(np.cos(jit))
        c = [bx, by]
        c[a_long] += near_sign * (SHELF_HALF[a_long] - gap - ext_long)
        room = SHELF_HALF[a_side] - ext_side - SIDE_CLEAR
        c[a_side] += float(rng.uniform(-room, room))
        u_near = np.zeros(2)
        u_near[a_long] = near_sign
        near_edge = {(1.0, 0.0): "forward", (-1.0, 0.0): "backward", (0.0, 1.0): "left", (0.0, -1.0): "right"}[
            (float(u_near[0]), float(u_near[1]))]
        heavy_dir = u_near if heavy_near else -u_near  # the card's local +x (heavy end) points this way (+- jitter)
        yaw = float(np.arctan2(heavy_dir[1], heavy_dir[0]) + jit)
        side = 1.0 if rng.random() < 0.5 else -1.0
        bin_xy = [bx + float(rng.uniform(-0.04, 0.04)), by + side * (SHELF_HALF[1] + BIN_GAP + BIN_HALF_OUT)]
        return dict(seed=int(seed), shelf_xy=[bx, by], card_xy=[float(c[0]), float(c[1])], card_yaw=yaw,
                    card_along=along, near_edge=near_edge, far_edge=OPPOSITE[near_edge], near_gap=gap,
                    heavy_near=heavy_near, heavy_edge=near_edge if heavy_near else OPPOSITE[near_edge],
                    com_offset=COM_E, bin_xy=bin_xy, bin_side="left" if side > 0 else "right")

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        bx, by = inst["shelf_xy"]
        m.body_pos[self.shelf_bid] = [bx, by, TZ + SHELF_H / 2]
        m.body_quat[self.shelf_bid] = [1.0, 0.0, 0.0, 0.0]
        cx, cy = inst["bin_xy"]
        m.body_pos[self.bin_bid] = [cx, cy, TZ + (BIN_WALL_H + BIN_T) / 2]
        m.body_quat[self.bin_bid] = [1.0, 0.0, 0.0, 0.0]
        x, y = inst["card_xy"]
        envs.set_obj_pose(self.env, "card_1", [x, y, CARD_REST_Z + 0.0003], yaw=inst["card_yaw"])

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = ShelfSkills(self.env, self, recorder=recorder)
        p = self.card_pose()
        self.mon = dict(start_xy=p["xy"].copy(), max_lift=0.0, max_overhang=0.0, fell=False, lifted=False,
                        out=dict(heavy=0.0, light=0.0, side=0.0), out_edge=dict(heavy=None, light=None, side=None),
                        flagged=set(), events=[], in_bin=False, tip=None, leg_hits=0, edge_used=None)
        return self.sk

    # ------------------------------------------------------------ geometry
    def card_pose(self):
        p = envs.obj_pos(self.env, "card_1")
        return dict(xy=p[:2].copy(), z=float(p[2]), yaw=_yaw_of(envs.obj_quat(self.env, "card_1")))

    def _card_frame(self):
        d = self.env.sim.data
        bid = envs.body_id(self.env, "card_1")
        return d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()

    def card_corners(self):
        p, R = self._card_frame()
        h = np.array(CARD_HALF)
        return np.array([p + R @ (h * [sx, sy, 0.0]) for sx in (-1, 1) for sy in (-1, 1)])

    def _beyond(self, xy):
        """{edge: how far the point lies beyond that edge of the shelf top (m, negative = inside)}"""
        b = np.asarray(xy, float) - np.asarray(self.inst["shelf_xy"])
        return {"forward": b[0] - SHELF_HALF[0], "backward": -b[0] - SHELF_HALF[0],
                "left": b[1] - SHELF_HALF[1], "right": -b[1] - SHELF_HALF[1]}

    def overhang(self):
        """(how far the card sticks out beyond the shelf's edges (m, max over its corners), edge word)"""
        best = (-1.0, None)
        for c in self.card_corners():
            for e, v in self._beyond(c[:2]).items():
                if v > best[0]:
                    best = (float(v), e)
        return best

    def part_overhangs(self):
        """{'heavy' | 'light' | 'side': (overhang m, edge)}: the heavy end (local +x), light end and long sides,
        each measured at its two corners."""
        p, R = self._card_frame()
        h = np.array(CARD_HALF)
        groups = {"heavy": [(1, -1), (1, 1)], "light": [(-1, -1), (-1, 1)]}
        out = {}
        for k, cs in groups.items():
            best = (-1.0, None)
            for sx, sy in cs:
                c = p + R @ (h * [sx, sy, 0.0])
                for e, v in self._beyond(c[:2]).items():
                    if v > best[0]:
                        best = (float(v), e)
            out[k] = best
        best = (-1.0, None)  # a long side: its middle point (an end pushed out also moves two corners out)
        for sy in (-1, 1):
            c = p + R @ (h * [0.0, sy, 0.0])
            for e, v in self._beyond(c[:2]).items():
                if v > best[0]:
                    best = (float(v), e)
        out["side"] = best
        return out

    def end_edge(self, which):
        """shelf edge that the card's `which` end ('heavy' | 'light' | 'near' | 'far') points to, from the live pose"""
        p = self.card_pose()
        d = np.array([np.cos(p["yaw"]), np.sin(p["yaw"])])  # heavy end direction
        if which == "near" or which == "far":
            gaps = {}
            for s in (1.0, -1.0):
                u = d * s
                e = max(EDGE_U, key=lambda k: float(np.asarray(EDGE_U[k]) @ u))
                tip = p["xy"] + u * CARD_HALF[0]
                gaps[e] = -self._beyond(tip)[e]
            near = min(gaps, key=gaps.get)
            return near if which == "near" else OPPOSITE[near]
        u = d if which == "heavy" else -d
        return max(EDGE_U, key=lambda k: float(np.asarray(EDGE_U[k]) @ u))

    def on_shelf(self):
        p = self.card_pose()
        rel = np.abs(p["xy"] - np.asarray(self.inst["shelf_xy"]))
        return bool(rel[0] <= SHELF_HALF[0] and rel[1] <= SHELF_HALF[1] and abs(p["z"] - CARD_REST_Z) < 0.012)

    def on_table(self):
        return bool(self.card_pose()["z"] < TZ + 0.03 and not self.in_bin())

    def tilt_deg(self):
        _, R = self._card_frame()
        return float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0.0, 1.0))))

    def in_bin(self):
        p = self.card_pose()
        rel = np.abs(p["xy"] - np.asarray(self.inst["bin_xy"]))
        return bool(rel[0] <= BIN_IN and rel[1] <= BIN_IN and p["z"] < TZ + BIN_T + BIN_WALL_H)

    # ------------------------------------------------------------ monitor
    def _event(self, ev, **kw):
        self.mon["events"].append(dict(event=ev, step=int(self.sk.n_steps), **kw))

    def monitor_step(self):
        mon = self.mon
        if mon is None or self.sk is None:
            return
        p = self.card_pose()
        held = self.sk.holding("card_1")
        zmin = float(self.card_corners()[:, 2].min())
        mon["max_lift"] = max(mon["max_lift"], zmin - SHELF_TOP)
        if held and zmin - SHELF_TOP > 0.02 and not mon["lifted"]:
            mon["lifted"] = True
            self._event("lifted")
        rel = np.abs(p["xy"] - np.asarray(self.inst["shelf_xy"]))
        if (not held and abs(p["z"] - CARD_REST_Z) < 0.004 and self.tilt_deg() < 5.0 and rel[0] <= SHELF_HALF[0]
                and rel[1] <= SHELF_HALF[1]):  # lying flat on the shelf
            mon["max_overhang"] = max(mon["max_overhang"], self.overhang()[0])
            for k, (v, e) in self.part_overhangs().items():
                if v > mon["out"][k]:
                    mon["out"][k], mon["out_edge"][k] = v, e
                if v >= 0.005 and k not in mon["flagged"]:
                    mon["flagged"].add(k)
                    self._event("side_over_edge" if k == "side" else "end_over_edge", part=k, edge=e)
        if p["z"] < SHELF_TOP - 0.02 and not held and not mon["fell"]:
            mon["fell"] = True
            if not mon["lifted"]:  # never lifted: it tipped (or was pushed) off the shelf
                part = max(("heavy", "light", "side"), key=lambda k: mon["out"][k])
                mon["tip"] = dict(part=part, edge=mon["out_edge"][part], overhang=round(mon["out"][part], 4))
                self._event("tipped_off", **mon["tip"])
            else:
                self._event("released")
        d = self.env.sim.data
        for i in range(d.ncon):  # the hand / arm touching a shelf leg (should not happen in normal grasps)
            c = d.contact[i]
            if (c.geom1 in self.leg_geoms and c.geom2 in self.robot_geoms) or (c.geom2 in self.leg_geoms and
                                                                                c.geom1 in self.robot_geoms):
                if mon["leg_hits"] == 0:
                    self._event("hand_hit_leg")
                mon["leg_hits"] += 1
                break
        if not mon["in_bin"] and self.in_bin() and not held:
            mon["in_bin"] = True
            self._event("in_bin")

    def close_context(self):
        """Where the gripper is relative to the card and the shelf just before it closes."""
        sk, p = self.sk, self.card_pose()
        e = sk.eef_pos()
        c, s = np.cos(p["yaw"]), np.sin(p["yaw"])
        rel = e[:2] - p["xy"]
        local = [float(rel @ [c, s]), float(rel @ [-s, c])]  # along the card's long / short side (+ = heavy end)
        beyond = self._beyond(e[:2])
        edge = max(beyond, key=beyond.get)
        ov, ov_edge = self.overhang()
        return dict(step=sk.n_steps, local=[round(v, 4) for v in local],
                    tips_above_card=round(float(e[2] - TIP_BELOW_SITE - (p["z"] + CARD_HALF[2])), 4),
                    beyond_edge=round(float(beyond[edge]), 4), beyond_edge_name=edge,
                    card_on_shelf=self.on_shelf(), overhang=round(ov, 4), overhang_edge=ov_edge,
                    width_before=round(sk.gripper_width(), 4))

    # ------------------------------------------------------------ outcome / F2
    def outcome(self, inst):
        envs.settle(self.env, 30)
        self.monitor_step()
        sk, mon = self.sk, self.mon
        p = self.card_pose()
        held = bool(sk.holding("card_1"))
        in_bin = self.in_bin() and not held
        success = bool(in_bin)
        moved = float(np.linalg.norm(p["xy"] - mon["start_xy"]))
        lifted = mon["max_lift"] > 0.02
        tries = [c for c in sk.closes if c["tips_above_card"] <= LOW_CLOSE and c["width_before"] > 0.02]
        parts, mode = [], "success"
        rel = p["xy"] - np.asarray(inst["bin_xy"])
        where = (f"{100 * abs(rel[0]):.1f} cm {'forward' if rel[0] > 0 else 'backward'} and "
                 f"{100 * abs(rel[1]):.1f} cm {'left' if rel[1] > 0 else 'right'} of the bin's centre")
        tip = mon["tip"]
        if success:
            parts.append("The card is in the bin.")
        elif held:
            mode = "held"
            parts.append(f"The card is still held by the gripper, {where}.")
        elif lifted:
            mode = "dropped"
            loc = ("on the shelf" if self.on_shelf() else "on the table" if self.on_table() else
                   "partly on the shelf" if p["z"] > SHELF_TOP - 0.02 else "off the shelf")
            parts.append(f"The card was lifted {100 * mon['max_lift']:.1f} cm above the shelf but is not in the "
                         f"bin: it lies {loc}, its centre {where}.")
        elif mon["fell"]:
            mode = "fell_off"
            what = "a long side" if tip["part"] == "side" else "one end"
            parts.append(f"The card was never lifted: after {what} of it had been slid {100 * max(tip['overhang'], 0):.1f}"
                         f" cm beyond the shelf's {tip['edge']} edge, it tipped over that edge and fell off the shelf; "
                         + ("it lies flat on the table." if self.on_table() and self.tilt_deg() < 10 else
                            f"it rests off the shelf, tilted {self.tilt_deg():.0f} degrees."))
        else:
            mode = "not_lifted"
            parts.append("The card was never lifted; it still lies flat on the shelf"
                         + (f" (it was slid {100 * moved:.1f} cm" + (f" and stuck out up to "
                            f"{100 * mon['max_overhang']:.1f} cm beyond the shelf's edge)" if mon["max_overhang"] > 0.003
                            else ")") if moved > 0.005 else "") + ".")
        if not success and not lifted and tries:
            c = tries[-1]
            n = f"{len(tries)} grasp tries; in the last one, " if len(tries) > 1 else ""
            if c["card_on_shelf"] and c["overhang"] <= 0.0 and c["beyond_edge"] < 0.0:
                txt = ("when the gripper closed on the card lying flat on the shelf, its fingertips were on either side "
                       "of the card, at shelf level")
            else:
                pos = (f"{100 * c['beyond_edge']:.1f} cm beyond the shelf's {c['beyond_edge_name']} edge"
                       if c["beyond_edge"] >= 0.0 else
                       f"over the shelf, {100 * -c['beyond_edge']:.1f} cm inside its {c['beyond_edge_name']} edge")
                txt = (f"when the gripper closed, the card stuck out {100 * max(c['overhang'], 0.0):.1f} cm beyond the "
                       f"shelf's {c['overhang_edge']} edge and the gripper's centre was {pos}")
            txt += f" ({100 * c['tips_above_card']:+.1f} cm from the fingertips to the top of the card); the fingers closed "
            txt += ("fully, sliding over the top of the card instead of gripping its edges."
                    if c["width"] < 2 * CARD_HALF[1] - 0.01 and not c["held"] else
                    ("on the card." if c["held"] else f"to {100 * c['width']:.1f} cm without holding the card."))
            parts.append(n + txt if n else txt[0].upper() + txt[1:])
        elif not success and not lifted and not mon["fell"]:
            parts.append("The gripper never closed with its fingertips down at the card.")
        failure = "none" if success else mode
        out = dict(success=success, mode=mode, failure=failure, held=held, lifted=bool(lifted),
                   max_lift=round(mon["max_lift"], 4), max_overhang=round(mon["max_overhang"], 4), moved=round(moved, 4),
                   fell=bool(mon["fell"]), tipped_off=tip is not None, trap_fired=bool(tip is not None and
                                                                                        tip["part"] == "heavy"),
                   tip=tip, end_out={k: round(v, 4) for k, v in mon["out"].items()},
                   end_out_edge=dict(mon["out_edge"]), card_final=dict(xy=[round(float(v), 4) for v in p["xy"]],
                                                                      z_above_table=round(p["z"] - TZ, 4),
                                                                      tilt_deg=round(self.tilt_deg(), 1)),
                   leg_hits=int(mon["leg_hits"]), slid_edge=mon["edge_used"],
                   events=list(mon["events"]), grasp_tries=tries, closes=sk.closes, detail=" ".join(parts))
        return out

    # ------------------------------------------------------------ scripted policies (move_to + gripper only)
    def grasp_and_bin(self, xy, yaw, z_site, lift=0.12):
        sk = self.sk
        sk.set_gripper(False, steps=8)
        sk.move_to([xy[0], xy[1], z_site + 0.08], yaw=yaw)
        sk.move_to([xy[0], xy[1], z_site], tol=0.003, max_steps=120)
        sk.set_gripper(True, steps=15)
        sk.move_to([xy[0], xy[1], z_site + lift], speed=0.2)
        if not sk.holding("card_1"):
            sk.set_gripper(False, steps=8)
            return False
        bx, by = self.inst["bin_xy"]
        hi = SHELF_TOP + 0.10
        sk.move_to([bx, by, hi])
        sk.move_to([bx, by, TZ + BIN_WALL_H + 0.05], speed=0.2)
        sk.set_gripper(False, steps=12)
        sk.move_to([bx, by, hi])
        return True

    def slide_card(self, delta, press=0.004, at=(0.0, 0.0)):
        """closed fingertips pressed on the card (at its centre + `at`), straight move by `delta` (xy), lift"""
        sk, p = self.sk, self.card_pose()
        z = p["z"] + CARD_HALF[2] + TIP_BELOW_SITE - press
        s = p["xy"] + np.asarray(at, float)
        sk.set_gripper(True, steps=8)
        sk.move_to([s[0], s[1], z + 0.06], yaw=p["yaw"])
        sk.move_to([s[0], s[1], z], tol=0.003, max_steps=100)
        e = s + np.asarray(delta, float)
        sk.move_to([e[0], e[1], z], speed=DRAG_SPEED, tol=0.004, max_steps=300)
        sk.move_to([e[0], e[1], z + 0.06], speed=0.2)
        sk.hold(10)  # look

    def push_end(self, u, dist, behind=0.015):
        """closed fingers at shelf level `behind` m behind the card's trailing end, push it `dist` along u, lift"""
        sk, p = self.sk, self.card_pose()
        d = np.array([np.cos(p["yaw"]), np.sin(p["yaw"])])
        la = d * np.sign(d @ u) if abs(d @ u) > 0.5 else d
        tail = p["xy"] - la * CARD_HALF[0]
        s = tail - u * behind
        z = SHELF_TOP + TIP_BELOW_SITE - 0.001  # fingertips on the shelf top (2 mm above: they pass over the card)
        yaw = float(np.arctan2(u[1], u[0]) - np.pi / 2)  # finger axis across the push direction
        sk.set_gripper(True, steps=8)
        sk.move_to([s[0], s[1], z + 0.05], yaw=yaw)
        sk.move_to([s[0], s[1], z], tol=0.003, max_steps=100)
        e = s + u * (behind + dist)
        sk.move_to([e[0], e[1], z], speed=0.04, tol=0.004, max_steps=300)
        sk.move_to([e[0], e[1], z + 0.05], speed=0.2)
        sk.hold(10)

    def execute(self, inst, params):
        p = self.card_pose()
        rng = np.random.default_rng(params.get("noise_seed")) if params.get("noise_seed") is not None else None
        if params.get("probe_grasp", False) or params["mode"] == "grasp":  # flat grasp at the card's middle
            dxy = np.asarray(params.get("probe_dxy", (0.0, 0.0)))
            self.grasp_and_bin(p["xy"] + dxy, p["yaw"], SHELF_TOP + params.get("probe_dz", 0.0))
            if params["mode"] == "grasp" or self.sk.holding("card_1") or self.mon["lifted"]:
                return
        # slide one end of the card along its long side out over the edge it points to, in strokes
        edge = params.get("edge") or self.end_edge(params.get("end", "near"))
        self.mon["edge_used"] = edge
        u = np.asarray(EDGE_U[edge], float)
        b = np.asarray(inst["shelf_xy"])
        half = SHELF_HALF[0] if edge in ("forward", "backward") else SHELF_HALF[1]
        edge_pos = float(b @ u) + half
        for k, target in enumerate(params.get("steps", ORACLE_STEPS)):
            if not self.on_shelf():
                return  # tipped off: nothing left to grasp (the attempt is lost)
            p = self.card_pose()
            d = np.array([np.cos(p["yaw"]), np.sin(p["yaw"])])
            la = d * np.sign(d @ u) if abs(d @ u) >= 0.5 else np.array([-d[1], d[0]]) * np.sign(np.array([-d[1], d[0]]) @ u)
            ext = max(float(c[:2] @ u) for c in self.card_corners()) - float(p["xy"] @ u)
            tgt = float(target) + (float(rng.uniform(-1, 1)) * params.get("step_noise", 0.0) if rng is not None else 0.0)
            need = edge_pos + tgt - (float(p["xy"] @ u) + ext)  # travel still needed along u
            if need > 0.003:
                if params.get("style", "press") == "endpush":
                    self.push_end(u, need)
                else:
                    at = (np.asarray(params.get("press_at", (0.0, 0.0))) if rng is None else
                          rng.normal(0.0, params.get("press_noise", 0.0), 2))
                    self.slide_card(la * need / max(float(la @ u), 1e-6), at=at)
        self.sk.hold(10)
        if not self.on_shelf():
            return
        q = self.card_pose()
        d = np.array([np.cos(q["yaw"]), np.sin(q["yaw"])])
        la = d * np.sign(d @ u) if abs(d @ u) >= 0.5 else np.array([-d[1], d[0]]) * np.sign(np.array([-d[1], d[0]]) @ u)
        cu = float(q["xy"] @ u)
        tip_u = max(float(c[:2] @ u) for c in self.card_corners())  # outermost point of the card along u
        target_u = min(tip_u - 0.008, edge_pos + params.get("grasp_out", GRASP_OUT))
        g = q["xy"] + la * (target_u - cu) / max(float(la @ u), 1e-6)
        g = g + np.asarray(params.get("grasp_dxy", (0.0, 0.0)))
        self.grasp_and_bin(g, q["yaw"], SHELF_TOP - 0.003 + params.get("grasp_dz", 0.0))

    def default_params(self, inst):
        """Sonnet-like (cla4 l3_card_edge): one flat grasp, then the nearest end out to the edge and 1 cm at a time
        to ~3 cm, lifting the finger and looking after each stroke; grasp the overhang."""
        return dict(mode="edge", probe_grasp=True, end="near", steps=list(NAIVE_STEPS))

    def oracle_params(self, inst):
        """Knows the hidden weight: slides the light end out 3 cm (two strokes, as the l3_card_edge oracle)."""
        return dict(mode="edge", end="light", steps=list(ORACLE_STEPS))

    def blind_params(self, inst, attempt, rng):
        """naive with small random variations, no history: nearest end, 2.5-3.5 cm, pressing point / grasp jitter"""
        return dict(mode="edge", probe_grasp=True, end="near", steps=[0.0, 0.01, 0.02, float(rng.uniform(0.025, 0.035))],
                    noise_seed=int(rng.integers(1 << 30)), press_noise=0.005,
                    grasp_dxy=rng.uniform(-0.005, 0.005, 2).tolist(), grasp_dz=float(rng.uniform(-0.004, 0.003)))

    def adapt_params(self, inst, history):
        """Tipped off the shelf (never lifted) -> slide the OTHER end next time (it tipped although only ~2 cm stuck
        out: that end must be heavy). Slid out but not gripped -> 1 cm more overhang. Otherwise repeat."""
        out, prm = history[-1]["outcome"], history[-1]["params"]
        used = out.get("slid_edge") or prm.get("edge")
        nxt = dict(mode="edge", probe_grasp=False, edge=used, steps=list(ORACLE_STEPS))
        if out["mode"] == "fell_off":
            e = (out.get("tip") or {}).get("edge") or used
            nxt["edge"] = OPPOSITE.get(e)
        elif out["mode"] == "not_lifted" and used is not None:
            last = float((prm.get("steps") or ORACLE_STEPS)[-1])
            nxt["steps"] = [s + 0.01 for s in ORACLE_STEPS] if last < 0.035 else list(ORACLE_STEPS)
        if nxt["edge"] is None:
            nxt.pop("edge")
            nxt["end"] = "near"
        return nxt
