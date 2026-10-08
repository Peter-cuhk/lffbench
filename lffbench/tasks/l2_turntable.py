"""L2-C turntable_restore: take the butter off a turntable that stands half under a shelf, then turn the
turntable back exactly the way it was.

Replaces the planned l2_drawer_reclosed (feasibility test 2026-10-07, see the task card): with the gripper
always pointing straight down, the handle of LIBERO's wooden_cabinet top drawer (3.8 cm below the cabinet
top) cannot be reached -- the descent stops 1.5-4.5 cm above it, closing grasps nothing (4/4 cabinet
placements) and a pull opened the drawer only 1/4 times, by accident. The turntable keeps the idea (an
articulated fixture the agent *has to* change to get the object and has to put back afterwards) and is
operable with move_to + open/close only.

Scene: a turntable (lazy susan; disc radius 17 cm on a damped hinge about z) on the robot's right; a fixed
shelf board 10 cm above the disc covers its -y half (robot's right), so nothing under the board can be
grasped from above (measured, 20 seeds x 2 grasp yaws: the hand stops on the board 10-14 cm above the grasp
height, 0/40 lifts). Four red pegs stand near the rim; grasping a peg and moving the gripper along the rim
turns the disc. The butter lies on the disc under the board; painted marks (yellow stripe, blue square,
green dot; visual only) make the disc's angle readable. The basket stands on the robot's left.

Why the first attempt fails (the L2 event is the agent's own, necessary change): to reach the butter the
agent must turn the disc by 136-179 deg (start angle random; measured on seeds 1000-1019). An agent that puts the butter into the basket
and stops leaves the turntable turned -> failure. To repair it the agent has to turn it back to its start
angle (+-15 deg); after the main turn the marks are elsewhere and nothing in the scene shows the start angle:
it is only in the first frame (or in the agent's memory of how far it turned). The F2 text says *that* the
turntable is not turned the way it was, never by how much or which way.

Protocol: within (one episode, no reset). Attempt 1 = the main task, every later attempt = one corrective
round; the outcome (success + F2 text) is computed after each round.

Scripted references (perception stand-in: simulator poses; motions: Skills.move_to / set_gripper only)
  oracle    turns, takes the butter, then immediately turns the disc back to the saved start angle.
  naive     turns, takes the butter, done.
  adaptive  saves the start angle (Save); after F2 "not turned the way it was" turns back to it (Retrieve +
            Utilize); redoes the main task if the butter is not in the basket.
  blind     no history, no feedback: every later round redoes the main task (lifts the butter out of the
            basket and drops it back, grasp jittered by 5 mm).
  nomem     (extra control, not in the acceptance table) gets F2 but has no start frame: turns the disc by a
            random angle in (-180, 180] deg each round.

Physics notes (all measured while building the task, 2026-10-07): the HOPE butter weighs 5 g -> set to 0.2 kg;
a box on a cylinder gets a single contact point and no torsional friction, so the butter span on the disc at
~4 deg/s even at rest -> the disc's collision surface is the inscribed square plate (visual disc stays round).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, register_generated
from ..task_base import LFFTask, register_task

TT = "turntable_1"
TARGET = "butter_1"
ITEMS = (TARGET,)
BASKET = "basket_1"
MASS = {"butter_1": 0.20}  # real-world order (the HOPE box weighs 5 g)

# turntable geometry (fixture frame: origin on the table top at the disc axis)
R = 0.17  # disc radius
DISC_T = 0.012  # disc thickness
BASE_H = 0.012  # pedestal height
DISC_TOP = BASE_H + DISC_T  # disc top above the table
PEG_R, PEG_RAD, PEG_H = 0.009, 0.155, 0.035  # radius, distance from the axis, height above the disc
#   (low enough that the hand clears the pegs by 1 cm when it grasps the butter between them)
PEG_ANG = tuple(np.pi / 4 + k * np.pi / 2 for k in range(4))  # disc frame
STRIPE_ANG = np.pi  # disc frame
BOARD_Z = DISC_TOP + 0.10  # underside of the shelf board above the table
BOARD_EDGE = 0.01  # the board covers fixture y < +1 cm (the -y half of the disc)
HINGE = dict(damping=1.0, armature=0.3, frictionloss=0.01)  # the disc stops within ~1 deg when released
ITEM_R = 0.07  # butter's distance from the axis (disc frame); open fingers at a held peg clear it by >= 1.2 cm

ANG_TOL = np.radians(15.0)  # success: disc within 15 deg of its start angle (4 cm at the pegs, 2.6 cm at the stripe end)
BASKET_IN = 0.060  # LIBERO basket: half size of the inner floor
BASKET_RIM = 0.150
BASKET_FLOOR = 0.017
SOLID = 'solimp="0.998 0.998 0.001" solref="0.001 1"'

NAMES = {TARGET: "butter"}
DESCR = {TARGET: "small orange box"}


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _wrap_half(a):
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def turntable_xml(model_name):
    """Static pedestal + shelf board + side wall; a hinged disc (child body) with 4 pegs and painted marks."""
    def geoms(kind, pos, size, rgba, coll=True, vis=True):
        s = f'type="{kind}" pos="{_fmt(pos)}" size="{_fmt(size)}" rgba="{_fmt(rgba)}"'
        out = [f'<geom {s} conaffinity="0" contype="0" group="1" />'] if vis else []
        if coll:
            out.insert(0, f'<geom {s} friction="1.0 0.005 0.0001" {SOLID} density="400" group="0" />')
        return out

    shelf = (0.82, 0.82, 0.80, 1.0)
    fixed = geoms("cylinder", (0, 0, BASE_H / 2), (0.06, BASE_H / 2), (0.3, 0.3, 0.3, 1.0))
    by0, by1, bx = -(R + 0.04), BOARD_EDGE, R + 0.04
    fixed += geoms("box", (0, (by0 + by1) / 2, BOARD_Z + 0.004), (bx, (by1 - by0) / 2, 0.004), shelf)
    fixed += geoms("box", (0, by0 + 0.005, BOARD_Z / 2), (bx, 0.005, BOARD_Z / 2), shelf)
    # the round disc is visual only; its collision surface is the inscribed square plate: a box on a cylinder gets a
    # single contact point and no torsional friction, and the butter span on the disc at ~4 deg/s even at rest (measured)
    disc = geoms("cylinder", (0, 0, 0), (R, DISC_T / 2), (0.62, 0.42, 0.24, 1.0), coll=False)
    disc += geoms("box", (0, 0, 0), (R / np.sqrt(2), R / np.sqrt(2), DISC_T / 2), (0.62, 0.42, 0.24, 1.0), vis=False)
    for a in PEG_ANG:
        disc += geoms("cylinder", (PEG_RAD * np.cos(a), PEG_RAD * np.sin(a), DISC_T / 2 + PEG_H / 2),
                      (PEG_R, PEG_H / 2), (0.75, 0.08, 0.08, 1.0))
    # painted marks (visual only) that make the disc's angle readable: yellow stripe, blue square, green dot
    disc += geoms("box", (0.085 * np.cos(STRIPE_ANG), 0.085 * np.sin(STRIPE_ANG), DISC_T / 2 + 0.0006),
                  (0.055, 0.007, 0.0006), (0.95, 0.85, 0.10, 1.0), coll=False)
    disc += geoms("box", (0.10 * np.cos(np.pi / 2), 0.10 * np.sin(np.pi / 2), DISC_T / 2 + 0.0006),
                  (0.016, 0.016, 0.0006), (0.15, 0.35, 0.85, 1.0), coll=False)
    disc += geoms("cylinder", (0.10 * np.cos(-np.pi / 2), 0.10 * np.sin(-np.pi / 2), DISC_T / 2 + 0.0006),
                  (0.016, 0.0006), (0.15, 0.65, 0.25, 1.0), coll=False)
    nl = "\n          "
    return f"""<mujoco model="{model_name}">
  <worldbody>
    <body>
      <body name="object">
        {nl.join(fixed)}
        <body name="disc" pos="0 0 {BASE_H + DISC_T / 2:.5f}">
          <joint name="spin" type="hinge" axis="0 0 1" pos="0 0 0" limited="false" damping="{HINGE['damping']}" armature="{HINGE['armature']}" frictionloss="{HINGE['frictionloss']}" />
          {nl.join(disc)}
        </body>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {BOARD_Z + 0.008:.5f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{R:.5f} {R:.5f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
"""


@register_task
class TurntableRestore(LFFTask):
    name = "l2_turntable_restore"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Save", "Retrieve", "Utilize")
    protocol = "within"
    max_attempts = 5
    instruction = ("Put the butter into the basket. The butter is on the turntable, under the shelf. You can turn the "
                   "turntable by its red pegs. When you are done, the turntable must be turned exactly the way it is now.")
    instruction_indirect = ("Put the small orange box into the wicker basket. It is on the round turntable, under the "
                            "shelf. You can turn the turntable by its red pegs. When you are done, the turntable must be "
                            "turned exactly the way it is now.")

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self._tt_bid = m.body_name2id(self.env.fixtures_dict[TT].root_body)
        jid = m.joint_name2id(f"{TT}_spin")
        self._qa, self._qv = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.start = {}
        self.turn_log = []  # per peg grasp: commanded sweep vs measured disc rotation (diagnostics)
        self._home = None
        self._c = np.zeros(2)

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        tt = register_generated("LffTurntableShelf", turntable_xml("lff_turntable_shelf"), free=False)
        objs = [(n, n[:-2], f"r_{i}", (0.20, -0.25 + 0.15 * i, 0.22, -0.23 + 0.15 * i)) for i, n in enumerate(ITEMS)]
        objs.append((BASKET, "basket", "r_basket", (-0.10, 0.24, -0.08, 0.26)))
        fx = [(TT, tt, "tt_region", (0.25, 0.30, 0.27, 0.32))]  # moved in apply_instance
        return write_bddl(self.name, self.instruction.replace(";", ","), objs, fixtures=fx)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        c = np.array([rng.uniform(-0.08, -0.02), rng.uniform(-0.12, -0.08)])
        th = {TARGET: float(rng.uniform(-0.14, 0.14))}  # disc-frame angle of the butter (stripe at 180 deg)
        rad = {n: float(ITEM_R + rng.uniform(-0.005, 0.005)) for n in ITEMS}
        tilt = {n: float(rng.uniform(-0.17, 0.17)) for n in ITEMS}  # box yaw relative to radial
        # start angle: the butter sits under the board (world azimuth -90 +- 40 deg about the axis)
        q0 = float(_wrap(-np.pi / 2 - th[TARGET] + rng.uniform(-np.radians(40), np.radians(40))))
        return dict(seed=int(seed), center=c.round(4).tolist(), q0=q0, theta=th, rad=rad, tilt=tilt,
                    basket_xy=[float(rng.uniform(-0.12, -0.06)), float(rng.uniform(0.25, 0.28))])

    def disc_angle(self):
        return float(self.env.sim.data.qpos[self._qa])

    def to_world(self, r, th, q=None):
        q = self.disc_angle() if q is None else q
        return self._c + r * np.array([np.cos(th + q), np.sin(th + q)])

    def to_disc(self, xy, q=None):
        q = self.disc_angle() if q is None else q
        return _rot(-q) @ (np.asarray(xy, float)[:2] - self._c)

    def place(self, name, xy, yaw, surface_z):
        """Teleport a box flat (its mesh frame is flat) with yaw about world z, lowest point 2 mm above surface_z."""
        envs.set_obj_pose(self.env, name, [xy[0], xy[1], surface_z + 0.3],
                          quat_wxyz=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]))
        self.env.sim.forward()
        dz = envs.obj_min_z(self.env, name) - (surface_z + 0.002)
        qa, _ = envs.free_joint_addr(self.env, name)
        self.env.sim.data.qpos[qa + 2] -= dz
        self.env.sim.forward()

    def apply_instance(self, inst):
        m, d = self.env.sim.model, self.env.sim.data
        self._c = np.array(inst["center"], float)
        m.body_pos[self._tt_bid] = [self._c[0], self._c[1], envs.TABLE_Z]
        m.body_quat[self._tt_bid] = [1, 0, 0, 0]
        d.qpos[self._qa] = inst["q0"]
        d.qvel[self._qv] = 0.0
        self.env.sim.forward()
        for n in ITEMS:
            envs.scale_mass(self.env, n, MASS[n])
            th = inst["theta"][n]
            # long axis radial (+ tilt): the gripper then closes tangentially and keeps clear of the board
            self.place(n, self.to_world(inst["rad"][n], th, inst["q0"]), th + inst["q0"] + inst["tilt"][n],
                       envs.TABLE_Z + DISC_TOP)
        envs.scale_mass(self.env, BASKET, 2.0)
        self.place(BASKET, inst["basket_xy"], 0.0, envs.TABLE_Z)
        self.env.sim.forward()

    def reset_instance(self, inst, recorder=None):
        sk = super().reset_instance(inst, recorder)
        self.start = dict(q=self.disc_angle())
        self._home = sk.eef_pos()
        return sk

    # ------------------------------------------------------------ measurements
    def footprint(self, name):
        """AABB centre, yaw of the long horizontal axis, z range (scripted 'perception')."""
        m, d = self.env.sim.model, self.env.sim.data
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        best = None
        for g in envs.obj_geom_ids(self.env, name):
            Rg = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + Rg @ m.geom_aabb[g, :3]
            e = np.abs(Rg) @ m.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
            vol = float(np.prod(m.geom_aabb[g, 3:]))
            if best is None or vol > best[0]:
                best = (vol, Rg, m.geom_aabb[g, 3:])
        _, Rg, half = best
        horiz = sorted([(half[k], Rg[:, k]) for k in range(3) if abs(Rg[2, k]) < 0.5], key=lambda t: -t[0])
        ax = horiz[0][1] if horiz else np.array([1.0, 0.0, 0.0])
        return dict(center=[float(v) for v in (lo + hi) / 2], yaw=float(np.arctan2(ax[1], ax[0])),
                    zmin=float(lo[2]), zmax=float(hi[2]))

    def item_loc(self, name):
        f = self.footprint(name)
        c = np.array(f["center"])
        held = self.sk is not None and self.sk.holding(name)
        if held and f["zmin"] > envs.TABLE_Z + DISC_TOP + 0.01:
            return "gripper", f
        bp = envs.obj_pos(self.env, BASKET)
        if np.all(np.abs(c[:2] - bp[:2]) <= BASKET_IN + 0.01) and f["zmin"] < envs.obj_min_z(self.env, BASKET) + BASKET_RIM \
                and f["zmin"] > envs.TABLE_Z + 0.005:
            return "basket", f
        on_disc = np.linalg.norm(c[:2] - self._c) < R and abs(f["zmin"] - (envs.TABLE_Z + DISC_TOP)) < 0.012
        if on_disc:
            return "turntable", f
        if f["zmin"] < envs.TABLE_Z + 0.01:
            return "table", f
        return "other", f

    def outcome(self, inst):
        envs.settle(self.env, 30)
        tloc, _ = self.item_loc(TARGET)
        dq = _wrap(self.disc_angle() - self.start["q"])
        turned_ok = abs(dq) <= ANG_TOL
        success = bool(tloc == "basket" and turned_ok)
        return dict(success=success, target_loc=tloc, angle_err_deg=round(float(np.degrees(dq)), 1),
                    turned_ok=bool(turned_ok),
                    detail=self._describe(tloc, turned_ok, dq, NAMES),
                    detail_indirect=self._describe(tloc, turned_ok, dq, DESCR),
                    detail_offset=self._describe(tloc, turned_ok, dq, NAMES, offset=True))

    @staticmethod
    def _describe(tloc, turned_ok, dq, names, offset=False):
        tn = names[TARGET]
        msgs = []
        if tloc != "basket":
            msgs.append({"gripper": f"The {tn} is still in the gripper, not in the basket.",
                         "turntable": f"The {tn} is still on the turntable, not in the basket.",
                         "table": f"The {tn} is on the table, not in the basket."}
                        .get(tloc, f"The {tn} is not in the basket."))
        if not turned_ok:
            m = "The turntable is not turned the way it was at the start."
            if offset:  # ablation tier: relative measurement, externalises the memory
                m += (f" It is turned {abs(np.degrees(dq)):.0f} deg "
                      f"{'counter-clockwise' if dq > 0 else 'clockwise'} (seen from above) from its starting angle.")
            msgs.append(m)
        if not msgs:
            msgs.append(f"The {tn} is in the basket and the turntable is turned the way it was at the start.")
        return " ".join(msgs)

    def feedback(self, inst, out, level="F2", indirect=False, offset=False):
        """F0: nothing, F1: success flag, F2: what is wrong (never by how much the turntable is off).
        offset=True (ablation tier 'F2+offset'): also the relative turn from the start angle."""
        if level == "F2":
            key = "detail_offset" if offset else ("detail_indirect" if indirect else "detail")
            out = dict(out, detail=out.get(key, out["detail"]))
        return super().feedback(inst, out, level)

    # ------------------------------------------------------------ scripted motions (Skills.move_to / set_gripper)
    SAFE_Z = envs.TABLE_Z + 0.22  # travel height between pegs / items
    PICK_AZ = np.pi / 2  # where the butter is brought: middle of the open side (at 135 deg the open finger lands on
    #                      the board edge -- measured); grasp yaw ~ +-90 deg, so the wrist is turned back to 0 while
    #                      carrying (arriving at the basket with yaw -103 deg left the arm stalled near a joint limit)
    WIN = (np.radians(30), np.radians(150))  # world azimuths where a peg can be held (open side, clear of the board)

    def go_home(self):
        sk = self.sk
        p = sk.eef_pos()
        sk.move_to([p[0], p[1], max(p[2], self._home[2])])
        sk.move_to(self._home, yaw=0.0)

    def _settle_yaw(self, tol=np.radians(3), max_steps=40):
        """move_to stops on position only; after a large wrist rotation hold until the yaw has converged (a descent
        that starts mid-rotation swipes the open fingers across the pegs -- measured)."""
        sk = self.sk
        for _ in range(max_steps // 5):
            R = sk.eef_mat() @ sk.R0.T
            if abs(_wrap(np.arctan2(R[1, 0], R[0, 0]) - sk.yaw)) < tol:
                return
            sk.hold(5)

    def peg_azimuths(self):
        return [_wrap(self.disc_angle() + a) for a in PEG_ANG]

    def turn(self, delta, step=np.radians(10)):
        """Turn the disc by `delta` rad (+ = counter-clockwise seen from above): grasp a peg on the open side,
        move the gripper along the rim in short straight segments, release, repeat."""
        sk = self.sk
        zg = envs.TABLE_Z + DISC_TOP + PEG_H - 0.012
        rem = float(delta)
        for _ in range(6):
            if abs(rem) < np.radians(1.0):
                break
            az = self.peg_azimuths()
            if rem > 0:
                cands = [a for a in az if self.WIN[0] - 1e-6 <= a <= self.WIN[1] - np.radians(10)]
                a0 = min(cands)
                sweep = min(rem, self.WIN[1] - a0)
            else:
                cands = [a for a in az if self.WIN[0] + np.radians(10) <= a <= self.WIN[1] + 1e-6]
                a0 = max(cands)
                sweep = max(rem, self.WIN[0] - a0)
            q_before = self.disc_angle()
            p = self._c + PEG_RAD * np.array([np.cos(a0), np.sin(a0)])
            sk.set_gripper(False, steps=6)
            # up, across, straight down: a long diagonal move at full speed sags ~5 cm below its line and the open
            # fingers sweep the disc (measured: turned it 38 deg the wrong way)
            cur = sk.eef_pos()
            if cur[2] < self.SAFE_Z - 0.02:
                sk.move_to([cur[0], cur[1], self.SAFE_Z])
            sk.move_to([p[0], p[1], self.SAFE_Z], yaw=np.pi / 2)  # fingers close along world x
            self._settle_yaw()
            sk.move_to([p[0], p[1], zg + 0.03], speed=0.4)
            sk.move_to([p[0], p[1], zg], tol=0.003, max_steps=80)
            sk.set_gripper(True, steps=12)
            dbg = dict(off=np.round((sk.eef_pos() - np.r_[self._c + PEG_RAD * np.array([np.cos(a0), np.sin(a0)]), zg]) * 100, 1).tolist(),
                       w=round(sk.gripper_width() * 100, 1), q_close=round(float(np.degrees(_wrap(self.disc_angle() - q_before))), 1),
                       yaw=round(float(np.degrees(sk.yaw)), 0))
            k = max(1, int(np.ceil(abs(sweep) / step)))
            for i in range(1, k + 1):
                w = self._c + PEG_RAD * np.array([np.cos(a0 + sweep * i / k), np.sin(a0 + sweep * i / k)])
                sk.move_to([w[0], w[1], zg], tol=0.003, max_steps=40)
            sk.set_gripper(False, steps=8)
            pe = sk.eef_pos()
            sk.move_to([pe[0], pe[1], zg + 0.10])
            got = _wrap(self.disc_angle() - q_before)
            self.turn_log.append(dict(peg_az=round(float(np.degrees(a0)), 1), sweep=round(float(np.degrees(sweep)), 1),
                                      got=round(float(np.degrees(got)), 1), **dbg))
            rem -= got

    def pick_place(self, name, dest_xy, support_z, fp=None, carry_z=1.20, travel_z=1.15, carry_yaw=0.0):
        """Top-down pick across the short side, carry, release just above support_z at dest_xy."""
        sk = self.sk
        f = fp or self.footprint(name)
        c = np.array(f["center"])
        gyaw = _wrap_half(f["yaw"])
        zg = max(f["zmin"] + 0.5 * (f["zmax"] - f["zmin"]), f["zmin"] + 0.0145)
        sk.set_gripper(False, steps=6)
        p = sk.eef_pos()
        if p[2] < travel_z - 0.02:
            sk.move_to([p[0], p[1], travel_z])
        sk.move_to([c[0], c[1], travel_z], yaw=gyaw)
        self._settle_yaw()
        sk.move_to([c[0], c[1], f["zmax"] + 0.03], speed=0.4)
        sk.move_to([c[0], c[1], zg], tol=0.003, max_steps=80)
        sk.set_gripper(True, steps=12)
        sk.move_to([c[0], c[1], carry_z], speed=0.3)
        sk.move_to([dest_xy[0], dest_xy[1], carry_z], yaw=carry_yaw)
        sk.move_to([dest_xy[0], dest_xy[1], support_z + (zg - f["zmin"]) + 0.006], tol=0.003, speed=0.2, max_steps=150)
        sk.hold(4)
        sk.set_gripper(False, steps=10)
        sk.move_to([dest_xy[0], dest_xy[1], travel_z], speed=0.4)

    def butter_to_basket(self, fp=None):
        """The main task: if the butter is on the turntable, turn it to the open side first."""
        loc, f = self.item_loc(TARGET)
        if loc == "turntable":
            az = np.arctan2(f["center"][1] - self._c[1], f["center"][0] - self._c[0])
            # bring the butter's long axis parallel to world y (then the hand lies along x, away from the board edge;
            # a hand turned 26 deg off caught the board -- measured); its centre ends within ~30 deg of PICK_AZ
            delta = _wrap_half(f["yaw"] - az)
            self.turn(_wrap(self.PICK_AZ - delta - az))
        b = envs.obj_pos(self.env, BASKET)
        # release with the hand above the rim and let the box drop in: lowering it to the floor puts the 20 cm long
        # hand below the rim of the 12 cm basket and the contact flips the wrist (joint 5 at its limit -- measured)
        self.pick_place(TARGET, b[:2], envs.obj_min_z(self.env, BASKET) + BASKET_RIM + 0.02, fp=fp)

    def do(self, actions):
        for act in actions:
            if act[0] == "butter_to_basket":
                self.butter_to_basket(fp=act[1] if len(act) > 1 else None)
            elif act[0] == "turn_to":  # ["turn_to", absolute disc angle (the agent's remembered start angle)]
                self.turn(_wrap(act[1] - self.disc_angle()))
            elif act[0] == "turn_by":
                self.turn(act[1])
            else:
                raise ValueError(act)

    # ------------------------------------------------------------ scripted reference policies
    def _plan(self, kind, inst, a, history, rng, memory):
        if kind == "naive" or (a == 0 and kind in ("adaptive", "blind", "nomem")):
            return [["butter_to_basket"]]
        if kind == "oracle":
            return [["butter_to_basket"], ["turn_to", memory["q"]]]
        last = history[-1]["outcome"]
        acts = [] if last["target_loc"] == "basket" else [["butter_to_basket"]]
        if kind == "adaptive":
            return acts + ([] if last["turned_ok"] else [["turn_to", memory["q"]]])
        if kind == "nomem":
            return acts + ([] if last["turned_ok"] else [["turn_by", float(rng.uniform(-np.pi, np.pi))]])
        if kind == "blind":
            fp = self.footprint(TARGET)
            fp["center"] = (np.array(fp["center"]) + np.r_[rng.normal(0, 0.005, 2), 0.0]).tolist()
            return [["butter_to_basket", fp]]
        raise ValueError(kind)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """Within-episode protocol: one reset, attempt 1 = main task, attempts 2..k = corrective rounds."""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        self.reset_instance(inst)
        memory = dict(q=self.disc_angle())  # Save: the start angle (an agent reads it off the first frame)
        for a in range(k):
            actions = self._plan(kind, inst, a, history, rng, memory)
            self.do(actions)
            self.go_home()
            out = self.outcome(inst)
            params = dict(actions=[[x[0]] + ([round(float(np.degrees(x[1])), 1)] if x[0] != "butter_to_basket" else [])
                                   for x in actions])
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history

    # cross-protocol hooks are not used by this task
    def default_params(self, inst):
        return dict(policy="naive")

    def oracle_params(self, inst):
        return dict(policy="oracle")
