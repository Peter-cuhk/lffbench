"""L7 offcenter_push: push a long box straight forward into a marked rectangle.

Category: failed_interaction_inference ("why did this contact not do what I expected?"). The code keeps level "L7";
the paper files the task under L5 (misjudged physics).

Hidden variable: the box's centre of mass is shifted along its long axis by e (|e| ~ U[4.5, 7.5] cm, either
side; implemented by moving the body's inertial frame, `body_ipos`). The box looks exactly the same for
every e (uniform colour, a symmetric tape stripe across its middle).

Why the prior fails: the natural push is at the middle of the box's back face. On a table the friction
resultant acts below the centre of mass, so a push whose line does not pass near it turns the box: the
heavy end lags behind. Measured (closed gripper at yaw 0, 8 cm/s, mu 0.5): pushing at the middle turns the
box 17 to 41 deg; the rectangle (box + 2 cm margin per side) only holds the box up to FIT_ANGLE = 10.9 deg
(exact geometry, box centred in the rectangle). Pushing within ~2 cm of the centre of mass gives a straight
translation (< 1 deg); 3 cm off gives 5-14 deg.

What must be learned from the failure: the turning direction tells which end is heavy (the end that
lagged), the amount tells roughly how far off the push was; push closer to the heavy end. F2 reports the box's
turn (direction and size, plus how far one end lags the other), where the box's centre ended relative to the
rectangle's centre, and where on the back face the gripper first touched it -- all relative measurements, never
absolute coordinates or e itself.

Pushes per attempt: not limited (2026-10-19, user). Until then the instruction allowed one push per attempt (one
move command that moves the box) and an attempt with more failed, so that the agent could not steer the box inside
an attempt; that moved trial and error that belongs inside an attempt to the next attempt. The attempt now succeeds
when the box ends completely inside the rectangle and upright, however many pushes it took (and every push was made
with the closed gripper, see below).

Error events (logged; they never end the attempt). A "push" is a move_to that moves the box > 1 cm or turns it
> 3 deg (unchanged definition). After every push the box is classified against the rectangle (`box_fit`):
over_turned (|yaw| > FIT_ANGLE: it fits at no position), overshot (a corner past the far edge), sideways (a corner
past a side edge), short (a corner still in front of the near edge, i.e. not pushed far enough yet), inside. A push
is `bad` when it leaves the box over_turned, overshot or sideways (or tipped over: upright cos < 0.5) -- not inside,
and not merely short of the rectangle (pushing straight on can fix "short", so pushing the box in in several steps is
not an error).
Outcome: n_pushes, n_bad_pushes, n_turned_pushes (bad pushes that left the box over_turned), n_trap_events (pushes
that took the box from a good state -- inside or short -- into a bad one, i.e. how often the trap fired anew),
trap_fired = n_bad_pushes > 0, recovered_in_attempt = trap_fired and success, first_push (record of the first push)
with first_push_yaw_deg and first_interaction_ok (the first push left the box inside, or short and not turned out).
irrecoverable: the box tipped over (upright cos < 0.5 after a move, < 0.95 after the final settle), fell off the
table, or went out of reach (its centre beyond world x = X_LOST, where the gripper can no longer be put down in front
of it at pushing height to push it back, or beyond |y| = Y_LOST); logged with the control step, the attempt goes on.

Closed gripper (2026-10-19): the instruction says "with the closed gripper", and a push made with the fingers open
(> OPEN_W apart when the move starts) now makes the attempt fail (failure "open_gripper" if the box is otherwise
inside). Reason, measured: open fingers touch the back face at two points ~8 cm apart, so ONE push at the middle of
the back face goes straight unless |e| > ~7 cm -- 22 of 25 instances (validation seeds 1000-1019, main 3000-3004)
succeeded that way without the centre of mass ever mattering. The box cannot be grasped, lifted or dragged by
pressing on its top (9 cm deep, the gripper opens 8 cm; measured), so no rule is needed for that.

Correction cost (measured 2026-10-19, task card): after the natural first push (middle, whole distance) the box is
turned 17-41 deg and lies across the rectangle. What repairs it: get in front of the leading (light) end and push it
back -- the box turns back with little translation -- then push the box in near the heavy end. The scripted
corrector (adaptive) did that in attempt 1 on 19 of 20 instances, 7-15 extra commands (median 7: 2 more pushes).
What does not: pushing the lagging end forward from behind turns the box back only slowly and carries it past the
rectangle (0 of 25 instances repaired that way; 11 ended out of reach). Near the reach limit the fingers ride up onto
the box instead of pushing it back. A cautious first push of 3 cm at the middle turns the box only 6.5-10.9 deg (below
FIT_ANGLE, so no error event) and shows the heavy end; probe + turn back + push in succeeded on 25 of 25 instances in
11 commands. So the task does not meet design rule R3 (a cost >= 10 commands for the common-sense first interaction);
its signal is clean success vs. in-attempt recovery (trap_fired / recovered_in_attempt) and first_push_yaw_deg.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

BOX = "box_1"
HALF = np.array([0.045, 0.11, 0.02])  # 9 (x) x 22 (y, long axis) x 4 cm: too wide for the 8 cm gripper
MASS = 0.6
MU = 0.5
MARGIN = 0.02  # target rectangle = box footprint + 2 cm on every side
RECT_HALF = HALF[:2] + MARGIN
E_MIN, E_MAX = 0.045, 0.075
Z_PUSH = envs.TABLE_Z + 0.02  # grasp-point height during the push (finger tips ~1 cm above the table)
Z_SAFE = envs.TABLE_Z + 0.12  # travel height of the scripted corrector (8 cm above the box top)
PUSH_SPEED = 0.08
FINGER_AHEAD = 0.009  # the closed fingers' front face is ~0.9 cm ahead of the grasp point (measured, yaw 0)
RUNUP = 0.04
MOVE_TOL = 0.01  # a command "moved the box" if its centre moved this much ...
TURN_TOL = np.radians(3.0)  # ... or it turned this much
TOUCH_TOL = 0.002  # the box "started to move" in a command once its centre moved this much
OPEN_W = 0.01  # finger opening above which a push counts as made with the gripper open (closed on nothing: 0.002 m)
TABLE_HALF = (0.5, 0.6)  # LIBERO main_table top, centred at the world origin (table_collision geom)
# Reach at pushing height (grasp point 2 cm above the table, yaw 0; measured 2026-10-19): the gripper gets to world
# x = 0.10 within 1 cm for |y| <= 0.15; asked for more it stops at x ~ 0.13-0.17, sagging to the table. Pushing a box
# back from in front of it (measured, box not turned): still works with the box's centre at x = 0.125 (far face at
# 0.17); with the centre at 0.135 the fingers ride up onto the box or push it further away. A box turned 25 deg with
# its centre at 0.113 (far corner at 0.20) was still pushed back. So a box whose centre is past X_LOST counts as out of
# reach. The rectangle's far edge lies at world x = -0.055 ... 0.055.
X_LOST = 0.13
Y_LOST = 0.40
X_DOWN = 0.14  # farthest x at which the scripted corrector puts the gripper down in front of the box


def _fit_angle(half=HALF[:2], rect=RECT_HALF):
    """Largest |yaw| at which the box's footprint still fits inside the rectangle (box centred in it): the
    rotated footprint's half extents are (a cos t + b sin t, a sin t + b cos t)."""
    t = np.radians(np.arange(0.0, 90.0, 0.01))
    ok = (half[0] * np.cos(t) + half[1] * np.sin(t) <= rect[0]) & (half[0] * np.sin(t) + half[1] * np.cos(t) <= rect[1])
    return float(t[np.argmin(ok)] - np.radians(0.01)) if not ok.all() else np.pi / 2


FIT_ANGLE = _fit_angle()  # 10.9 deg


def long_box_xml(name, half=HALF, mass=MASS, rgba=(0.72, 0.53, 0.33, 1.0), tape_rgba=(0.88, 0.80, 0.62, 1.0)):
    """Solid box with an explicit inertial whose position is offset from the body origin at compile time, so
    MuJoCo keeps the inertial frame separate (sameframe would otherwise ignore later changes of body_ipos)."""
    I = mass / 12 * np.array([4 * (half[1] ** 2 + half[2] ** 2), 4 * (half[0] ** 2 + half[2] ** 2),
                              4 * (half[0] ** 2 + half[1] ** 2)])
    tape = (half[0] + 0.0005, 0.012, 0.0006)
    g = (f'        <inertial pos="0 0.001 0" mass="{mass}" diaginertia="{_fmt(I)}" />\n'
         f'        <geom name="{name}_g0" type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" '
         f'friction="1.0 0.005 0.0001" {_SOLID} group="0" />\n'
         f'        <geom type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" conaffinity="0" contype="0" group="1" />\n'
         f'        <geom type="box" pos="0 0 {half[2]:.5f}" size="{_fmt(tape)}" rgba="{_fmt(tape_rgba)}" '
         f'conaffinity="0" contype="0" group="1" />')
    return _wrap(name, g, half[2], float(np.hypot(half[0], half[1])))


def rect_frame_xml(name, inner_half=RECT_HALF, line_w=0.006, line_rgba=(0.1, 0.55, 0.15, 1.0),
                   fill_rgba=(0.75, 0.92, 0.75, 1.0)):
    """Flat, visual-only rectangle drawn on the table (light fill + dark outline, no collisions)."""
    hx, hy = inner_half
    ox, oy = hx + line_w, hy + line_w
    parts = [((0, 0, 0.0004), (ox, oy, 0.0004), fill_rgba),
             ((0, hy + line_w / 2, 0.0010), (ox, line_w / 2, 0.0002), line_rgba),
             ((0, -hy - line_w / 2, 0.0010), (ox, line_w / 2, 0.0002), line_rgba),
             ((hx + line_w / 2, 0, 0.0010), (line_w / 2, oy, 0.0002), line_rgba),
             ((-hx - line_w / 2, 0, 0.0010), (line_w / 2, oy, 0.0002), line_rgba)]
    g = "\n".join(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(c)}" '
                  f'contype="0" conaffinity="0" group="1" />' for p, sz, c in parts)
    return _wrap(name, g, 0.0012, float(np.hypot(ox, oy)))


def _yaw(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _wrap_pi(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def box_corners(p_xy, yaw):
    R = _rot(yaw)
    return np.array([np.asarray(p_xy[:2], float) + R @ np.array([sx * HALF[0], sy * HALF[1]])
                     for sx in (-1, 1) for sy in (-1, 1)])


def box_fit(p_xy, yaw, t_xy):
    """The box's footprint against the (axis-aligned) target rectangle centred at t_xy. bad = it cannot get into the
    rectangle by being pushed straight on: turned beyond FIT_ANGLE, a corner past the far edge, or past a side edge."""
    c = box_corners(p_xy, yaw) - np.asarray(t_xy, float)[:2]
    eps = 1e-6
    inside = bool(np.all(np.abs(c[:, 0]) <= RECT_HALF[0] + eps) and np.all(np.abs(c[:, 1]) <= RECT_HALF[1] + eps))
    yw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # the footprint is symmetric under a half turn
    over = bool(abs(yw) > FIT_ANGLE)
    overshot = bool(np.any(c[:, 0] > RECT_HALF[0] + eps))
    sideways = bool(np.any(np.abs(c[:, 1]) > RECT_HALF[1] + eps))
    short = bool(np.any(c[:, 0] < -RECT_HALF[0] - eps))
    return dict(inside=inside, over_turned=over, overshot=overshot, sideways=sideways, short=short,
                bad=bool(over or overshot or sideways))


def _lr(v, unit="cm"):
    return f"{100 * abs(v):.1f} {unit} to the {'left' if v > 0 else 'right'}"


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


def _turn_words(deg):
    return f"{abs(deg):.0f} degrees {'counterclockwise' if deg > 0 else 'clockwise'}"


class _Budget(Exception):
    """the scripted policy used up the agent command budget of the attempt"""


class PushSkills(Skills):
    """Skills that log, for whoever drives them (scripted policy or agent), which move_to commands moved the
    box (pushes), where the gripper touched it, the box's state after every push, and whether the box was lost."""

    def __init__(self, env, bias=None, recorder=None, target=None):
        super().__init__(env, bias=bias, recorder=recorder)
        self.target = None if target is None else np.asarray(target, float)[:2]
        self.first_touch = None  # where the gripper's centre was (box frame) when the box first moved
        self._p0, self._yaw0 = self._box_pose()
        self.box_moves = []  # one entry per move_to that moved / turned the box (a "push")
        self.n_moves = 0  # move_to commands (an agent's move_eef with a motion = one move_to)
        self.lost = {}  # irrecoverable condition -> control step at which it was first seen
        self._mv = None  # the move_to in progress: box pose at its start, first touch inside it

    def _box_pose(self):
        p = envs.obj_pos(self.env, BOX)
        return p, _yaw(envs.obj_quat(self.env, BOX))

    def _local(self, p0, yaw0):
        """gripper centre in the box frame given by (p0, yaw0): (u along the short side, v along the long side)"""
        return (_rot(yaw0).T @ (self.eef_pos()[:2] - p0[:2])).tolist()

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        if self.first_touch is None or (self._mv is not None and self._mv["touch"] is None):
            # "touch" = the first control step at which the box has moved 2 mm. Detected from the box's motion,
            # not from contacts: a stiff gripper pushing a box makes a stick-slip contact that is often absent at the
            # control-step boundaries (measured), so contact polling misses pushes.
            p, _ = self._box_pose()
            if self.first_touch is None and np.linalg.norm(p[:2] - self._p0[:2]) > TOUCH_TOL:
                self.first_touch = dict(local=self._local(self._p0, self._yaw0), step=self.n_steps)
            if self._mv is not None and self._mv["touch"] is None and np.linalg.norm(p[:2] - self._mv["p0"][:2]) > TOUCH_TOL:
                self._mv["touch"] = self._local(self._mv["p0"], self._mv["yaw0"])

    def fit(self, p=None, yaw=None):
        if p is None:
            p, yaw = self._box_pose()
        return box_fit(p[:2], yaw, self.target) if self.target is not None else None

    def check_lost(self, final=False):
        """record irrecoverable box states (tipped, off the table, out of reach); returns the current ones"""
        p, _ = self._box_pose()
        up = envs.obj_upright_cos(self.env, BOX)
        now = []
        if up < (0.95 if final else 0.5):
            now.append("tipped")
        if p[2] < envs.TABLE_Z - 0.02 or abs(p[0]) > TABLE_HALF[0] or abs(p[1]) > TABLE_HALF[1]:
            now.append("off_table")
        elif p[0] > X_LOST or abs(p[1]) > Y_LOST:
            now.append("out_of_reach")
        for w in now:
            self.lost.setdefault(w, self.n_steps)
        return now

    def move_to(self, target, *args, **kwargs):
        p0, y0 = self._box_pose()
        fit0 = self.fit(p0, y0)
        bad0 = fit0 is not None and (fit0["bad"] or envs.obj_upright_cos(self.env, BOX) < 0.5)
        self._mv = dict(p0=p0, yaw0=y0, touch=None)
        width0 = self.gripper_width()
        self.n_moves += 1
        try:
            rep = super().move_to(target, *args, **kwargs)
        finally:
            mv, self._mv = self._mv, None
        p1, y1 = self._box_pose()
        dturn = _wrap_pi(y1 - y0)
        if np.linalg.norm(p1[:2] - p0[:2]) > MOVE_TOL or abs(dturn) > TURN_TOL:
            rec = dict(step=self.n_steps, moved_cm=round(100 * float(np.linalg.norm(p1[:2] - p0[:2])), 1),
                       turned_deg=round(float(np.degrees(dturn)), 1), yaw_deg=round(float(np.degrees(y1)), 1),
                       gripper_open=bool(width0 > OPEN_W))
            t = mv["touch"]
            if t is not None:
                u, v = t
                face = ("back" if u < -HALF[0] + 0.005 else "front" if u > HALF[0] - 0.005 else
                        "left_end" if v > 0 else "right_end")
                rec.update(touch_u=round(float(u), 4), touch_v=round(float(v), 4), face=face)
            fit1 = self.fit(p1, y1)
            if fit1 is not None:
                tipped = envs.obj_upright_cos(self.env, BOX) < 0.5
                bad = bool(fit1["bad"] or tipped)
                rec.update(inside=fit1["inside"] and not tipped, over_turned=fit1["over_turned"],
                           overshot=fit1["overshot"], sideways=fit1["sideways"], tipped=bool(tipped), bad=bad,
                           new_trap=bool(bad and not bad0))
            self.box_moves.append(rec)
        self.check_lost()
        return rep


@register_task
class OffcenterPush(LFFTask):
    name = "l7_offcenter_push"
    level = "L7"
    category = "failed_interaction_inference"
    capabilities = ("Perceive", "Reason", "Utilize")
    protocol = "cross"
    # 2026-10-19 (wording): the instruction states only the end state. "straight forward ... without turning" read as
    # a rule on the motion (no turning on the way, push only forward), but success is judged on the end state alone
    # (completely inside the rectangle -- which bounds the turn to FIT_ANGLE -- upright, every push with the closed
    # gripper), and the repair that works pushes the leading end back from in front of the box.
    instruction = ("Slowly push the long brown box with the closed gripper so that it ends up completely inside the "
                   "green rectangle ahead of it (farther from the robot).")
    instruction_indirect = ("Slowly push the long cardboard-coloured block with the closed gripper so that it ends up "
                            "completely inside the marked green area ahead of it (farther from the robot).")

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        box = register_generated("LffL7LongBox", long_box_xml("lff_l7_long_box"))
        zone = register_generated("LffL7TargetRect", rect_frame_xml("lff_l7_target_rect"), free=False)
        objs = [(BOX, box, "box_region", (-0.18, -0.01, -0.16, 0.01))]
        fixtures = [("zone_1", zone, "zone_region", (-0.06, -0.01, -0.04, 0.01))]
        return write_bddl(self.name, "push the long box into the green rectangle", objs, fixtures=fixtures)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        e = float(rng.choice([-1.0, 1.0]) * rng.uniform(E_MIN, E_MAX))
        return dict(seed=int(seed), box_xy=[float(rng.uniform(-0.20, -0.14)), float(rng.uniform(-0.08, 0.08))],
                    target_dist=float(rng.uniform(0.08, 0.13)), com_offset=e)

    def target_xy(self, inst):
        return np.array([inst["box_xy"][0] + inst["target_dist"], inst["box_xy"][1]])

    def apply_instance(self, inst):
        m = self.env.sim.model
        zone = self.env.fixtures_dict["zone_1"]
        for g in range(m.ngeom):  # the rectangle is a flat marker
            if (m.geom_id2name(g) or "").startswith("zone_1_"):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        t = self.target_xy(inst)
        zb = m.body_name2id(zone.root_body)
        m.body_pos[zb] = [t[0], t[1], envs.TABLE_Z + 0.0002]
        m.body_quat[zb] = [1.0, 0.0, 0.0, 0.0]
        bid = envs.body_id(self.env, BOX)
        m.body_mass[bid] = MASS
        m.body_ipos[bid] = [0.0, inst["com_offset"], 0.0]
        envs.refit_bounds(self.env, bid)  # the collision BVH lives in the inertial frame (envs.refit_bounds)
        envs.set_friction(self.env, BOX, MU)
        x0, y0 = inst["box_xy"]
        envs.set_obj_pose(self.env, BOX, [x0, y0, envs.TABLE_Z + HALF[2] + 0.001])

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = PushSkills(self.env, recorder=recorder, target=self.target_xy(inst))
        self.sim0 = self.sk.n_steps
        return self.sk

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 30)
        sk = self.sk
        p = envs.obj_pos(self.env, BOX)
        yaw = _yaw(envs.obj_quat(self.env, BOX))
        t = self.target_xy(inst)
        fit = box_fit(p[:2], yaw, t)
        inside = fit["inside"]
        upright = envs.obj_upright_cos(self.env, BOX) > 0.95
        sk.check_lost(final=True)
        pushes = sk.box_moves
        n_push = len(pushes)
        n_bad = sum(bool(m.get("bad")) for m in pushes)
        n_open = sum(bool(m.get("gripper_open")) for m in pushes)
        off = p[:2] - t
        lag = 2 * HALF[1] * np.sin(yaw)  # how far the +y end is behind (+) the -y end along x
        success = bool(inside and upright and n_open == 0)
        first = pushes[0] if pushes else None
        out = dict(success=success, inside=bool(inside), upright=bool(upright),
                   n_pushes=n_push, n_bad_pushes=n_bad,
                   n_turned_pushes=sum(bool(m.get("over_turned")) for m in pushes),
                   n_trap_events=sum(bool(m.get("new_trap")) for m in pushes),
                   trap_fired=n_bad > 0, recovered_in_attempt=bool(n_bad and success),
                   first_push=first, first_push_yaw_deg=None if first is None else first["yaw_deg"],
                   first_interaction_ok=None if first is None else bool(not first.get("bad")),
                   n_open_gripper_pushes=n_open,
                   irrecoverable=bool(sk.lost), irrecoverable_why=dict(sk.lost),
                   n_box_moving_commands=n_push, box_moves=pushes, n_move_commands=sk.n_moves,
                   sim_s=round((sk.n_steps - getattr(self, "sim0", 0)) * 0.05, 1),
                   yaw_deg=round(float(np.degrees(yaw)), 1),
                   centre_offset=[round(float(off[0]), 4), round(float(off[1]), 4)],
                   touch_offset=None)
        parts = []
        if sk.first_touch is None:
            out["failure"] = "not_touched"
            parts.append("The box did not move: the gripper never pushed it.")
        else:
            lt = sk.first_touch["local"]
            out["touch_offset"] = round(float(lt[1]), 4)
            if success:
                parts.append("The box is completely inside the rectangle.")
            elif not upright:
                out["failure"] = "tipped"
                parts.append("The box tipped over.")
            elif inside:
                out["failure"] = "open_gripper"
                parts.append("The box is completely inside the rectangle, but it was pushed with the gripper open; "
                             "the task asks for pushing with the closed gripper.")
            else:
                out["failure"] = "turned" if abs(np.degrees(yaw)) > 6 else "position"
                parts.append("The box is not completely inside the rectangle.")
            if n_push > 1:
                txt = f"It was moved by {n_push} separate move commands"
                if abs(first["yaw_deg"]) >= 1.0:
                    txt += f"; after the first one it had turned {_turn_words(first['yaw_deg'])} (seen from above)"
                parts.append(txt + ".")
            if n_open and out.get("failure") != "open_gripper":
                parts.append("The gripper was open during " + ("that push." if n_push == 1 else
                                                               f"{n_open} of the pushes."))
            turn = np.degrees(yaw)
            if abs(turn) >= 1.0:
                parts.append(f"It {'ended up turned' if n_push > 1 else 'turned'} {_turn_words(turn)} (seen from "
                             f"above): its {'left' if lag > 0 else 'right'} end ended {100 * abs(lag):.1f} cm "
                             f"further back than its {'right' if lag > 0 else 'left'} end.")
            else:
                parts.append("It did not turn (less than 1 degree)." if n_push <= 1 else
                             "It ended up not turned (less than 1 degree).")
            parts.append(f"Its centre ended {_fb(off[0])} and {_lr(off[1])} of the rectangle's centre.")
            parts.append(f"When the box started to move, the gripper's centre was {_lr(lt[1])} of the middle of "
                         f"the box (measured along the box's long side).")
        out["detail"] = " ".join(parts)
        return out

    # ------------------------------------------------------------ scripted primitives (agent-command counted)
    def _cmd(self, target, yaw=None, speed=None, tol=0.004, max_steps=250):
        """one agent command (move_eef; a gripper change rides on a move)"""
        self.ncmd = getattr(self, "ncmd", 0) + 1
        budget = getattr(self, "budget", None)
        if budget is not None and self.ncmd > budget:
            raise _Budget()
        self.sk.move_to(target, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)
        if budget is not None and self._sim_used() >= 3.0 * budget:  # the server's motion budget (3 s per command)
            raise _Budget()

    def _sim_used(self):
        return (self.sk.n_steps - getattr(self, "sim0", 0)) * 0.05

    def _push_at(self, face, v, dist, speed=PUSH_SPEED, runup=RUNUP):
        """One push on the box as it lies now, with the closed gripper rotated to the face: face "back" (push
        forward along the box's short side), "front" (push backwards) at long-side coordinate v (box frame), or
        "left_end" / "right_end" (push along the long side at the middle of that end face). `dist` = how far the
        fingers' front travels past the face. Commands: up (if low, 1), above the run-up point (1), down (1), push (1)."""
        sk = self.sk
        p, yaw = sk._box_pose()
        R = _rot(yaw)
        if face in ("back", "front"):
            s = -1.0 if face == "back" else 1.0
            n, q, gyaw = R @ np.array([s, 0.0]), p[:2] + R @ np.array([s * HALF[0], v]), yaw
        else:
            s = 1.0 if face == "left_end" else -1.0
            n, q, gyaw = R @ np.array([0.0, s]), p[:2] + R @ np.array([0.0, s * HALF[1]]), yaw + np.pi / 2
        start = q + n * (runup + FINGER_AHEAD)
        end = q - n * (dist - FINGER_AHEAD)
        e = sk.eef_pos()
        if e[2] < Z_PUSH + 0.05:
            self._cmd([e[0], e[1], Z_SAFE], speed=0.2)
        self._cmd([start[0], start[1], Z_SAFE], yaw=gyaw)
        self._cmd([start[0], start[1], Z_PUSH], tol=0.003)
        self._cmd([end[0], end[1], Z_PUSH], speed=speed, tol=0.004, max_steps=400)

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def execute(self, inst, params):
        """Closed gripper behind the box at lateral offset `c` (from the box's middle, along its long axis), then one
        slow straight move_to forward ending `end_dx` beyond the nominal end point (target centre minus half the box
        depth minus the finger offset), then lift. 4 commands (the gripper closes with the first one)."""
        sk = self.sk
        p = envs.obj_pos(self.env, BOX)  # perfect perception of the current scene
        t = self.target_xy(inst)
        c = float(params["c"])
        y = p[1] + c
        face = p[0] - HALF[0]
        x_end = t[0] - HALF[0] - FINGER_AHEAD + float(params.get("end_dx", 0.0))
        sk.set_gripper(True, steps=8)
        self._cmd([face - RUNUP, y, Z_PUSH + 0.08], yaw=0.0)
        self._cmd([face - RUNUP, y, Z_PUSH], tol=0.003)
        self._cmd([x_end, y, Z_PUSH], speed=PUSH_SPEED, tol=0.004, max_steps=400)
        self._cmd([x_end - 0.03, y, Z_PUSH + 0.10], speed=0.2)

    def default_params(self, inst):
        return dict(c=0.0, end_dx=0.0)  # Sonnet 5.5 (cla2 / cla4 / full5): yaw 0, middle of the back face, one push

    def oracle_params(self, inst):
        return dict(c=float(inst["com_offset"]), end_dx=0.0)

    STEP = 0.06  # first correction toward the lagging end (the expected |e| range is unknown to the learner)
    YAW_OK = 3.0  # deg: the corrector turns the box back until it is within this
    GAIN0 = 2.8  # deg of turn per cm of a push at the leading end from the front, 3 cm from the end (measured)
    SAFE_V = HALF[1] - 0.02  # contact points stay 2 cm inside the ends

    def bracket(self, pushes):
        """Contact point estimate from logged pushes on the back / front face: a push whose box turned
        counterclockwise passed to the right of the centre of mass when pushing forward (back face) and to its left
        when pushing backwards (front face); a long push that did not turn passed within ~2 cm of it."""
        lo, hi = -self.SAFE_V, self.SAFE_V
        lo_set = hi_set = False
        for m in pushes:
            if m.get("face") not in ("back", "front") or m.get("gripper_open"):
                continue
            v, turn = m["touch_v"], m["turned_deg"] * (1.0 if m["face"] == "back" else -1.0)
            if turn > 3.0:
                lo, lo_set = max(lo, v + 0.01), True
            elif turn < -3.0:
                hi, hi_set = min(hi, v - 0.01), True
            elif abs(turn) < 1.5 and m["moved_cm"] >= 3.0:
                lo, hi, lo_set, hi_set = max(lo, v - 0.02), min(hi, v + 0.02), True, True
        if lo > hi:  # inconsistent (contacts slide): trust the newest
            lo, hi = hi, lo
        if lo_set and hi_set:
            c = 0.5 * (lo + hi)
        elif lo_set:
            c = lo + self.STEP
        elif hi_set:
            c = hi - self.STEP
        else:
            c = 0.0
        return float(np.clip(c, -self.SAFE_V, self.SAFE_V))

    def adapt_params(self, inst, history):
        """Cross-attempt start point: push where the pushes of the earlier attempts put the centre of mass."""
        return dict(c=round(self.bracket([m for h in history for m in h["outcome"]["box_moves"]]), 4), end_dx=0.0)

    @staticmethod
    def _front_runup(p, yaw, v):
        """run-up for a push on the front face at v that keeps the gripper's put-down point within X_DOWN; None when
        even a 1.5 cm run-up does not fit (the box is too far forward to get in front of it)"""
        R = _rot(yaw)
        f, n_out = p[:2] + R @ np.array([HALF[0], v]), R @ np.array([1.0, 0.0])
        runup = min(RUNUP, (X_DOWN - f[0]) / max(n_out[0], 0.2) - FINGER_AHEAD)
        return float(runup) if runup >= 0.015 else None

    def adaptive_attempt(self, inst, c0, prior_pushes=()):
        """In-attempt corrector: the first push as `execute` (at c0: the middle in attempt 1, like the naive policy);
        then, after every push, look at the box (perfect perception) and push again until it is inside or the agent
        budget would run out. Turned more than YAW_OK -> push its leading end back from the front (it turns back with
        little translation), or if it is too far forward for that, the lagging end forward from behind; not turned
        but short / past the far edge / out at a side -> push it straight from behind / from the front at the
        estimated centre of mass, or along its long side at an end face. Returns the number of corrective pushes."""
        self.execute(inst, dict(c=c0, end_dx=0.0))
        sk = self.sk
        t = self.target_xy(inst)
        gain = self.GAIN0
        sim_budget = 3.0 * self.budget if self.budget is not None else np.inf
        n = 0
        while True:
            p, yaw = sk._box_pose()
            fit = sk.fit(p, yaw)
            if fit["inside"] or sk.lost:
                return n
            if (self.budget is not None and self.ncmd + 4 > self.budget) or self._sim_used() + 10.0 > sim_budget:
                return n
            n += 1
            yd = np.degrees(yaw)
            c_est = self.bracket(list(prior_pushes) + sk.box_moves)
            if abs(yd) > self.YAW_OK:
                for vm in (HALF[1] - 0.03, 0.06, 0.04, 0.02):  # the leading (light) end, or nearer the middle if
                    v_lead = -np.sign(yd) * vm  # the end is out of reach; +v = the box's left end
                    runup = self._front_runup(p, yaw, v_lead)
                    if runup is not None:
                        break
                if runup is not None:
                    d = float(np.clip(abs(yd) / gain, 1.0, 10.0)) / 100
                    self._push_at("front", v_lead, d, runup=runup)
                    dy = abs(yd) - abs(np.degrees(sk._box_pose()[1]))
                    if dy > 1.0:
                        gain = float(np.clip(dy / (100 * d), 1.0, 5.0))
                else:
                    self._push_at("back", np.sign(yd) * (HALF[1] - 0.01), 0.03)
            else:
                dx = t[0] - p[0]
                if fit["overshot"]:
                    runup = self._front_runup(p, yaw, c_est)
                    if runup is None:
                        return n
                    self._push_at("front", c_est, max(0.005, -dx), runup=runup)
                elif fit["sideways"]:
                    dyy = t[1] - p[1]
                    self._push_at("right_end" if dyy > 0 else "left_end", 0.0, max(0.005, abs(dyy) - 0.005))
                else:
                    self._push_at("back", c_est, max(0.005, dx))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """Cross protocol with the agent's command budget per attempt (agent_max_steps, default 40 commands and
        3 s of motion per command; one move_eef = one command). naive / oracle / blind: one push, then finish.
        adaptive: in-attempt corrector (`adaptive_attempt`); attempt 1 starts with the naive push, later attempts
        at the contact point estimated from all earlier pushes."""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            self.ncmd, self.budget = 0, int(getattr(self, "agent_max_steps", 40))
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "adaptive":
                params = self.default_params(inst) if a == 0 else self.adapt_params(inst, history)
            else:
                raise ValueError(kind)
            try:
                if kind == "adaptive":
                    prior = [m for h in history for m in h["outcome"]["box_moves"]]
                    params["corrections"] = self.adaptive_attempt(inst, params["c"], prior)
                else:
                    self.execute(inst, params)
            except _Budget:
                params["budget_used_up"] = True
            self.budget = None
            out = self.outcome(inst)
            out["commands"] = self.ncmd
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history

    blind_sigma = 0.008

    def blind_params(self, inst, attempt, rng):
        """Same plan (push at the middle) with a small random variation of the contact point; no history."""
        return dict(c=float(rng.normal(0.0, self.blind_sigma)), end_dx=0.0)
