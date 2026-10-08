"""L7 offcenter_push: push a long box straight forward into a marked rectangle with ONE push.

Category: failed_interaction_inference ("why did this contact not do what I expected?").

Hidden variable: the box's centre of mass is shifted along its long axis by e (|e| ~ U[4.5, 7.5] cm, either
side; implemented by moving the body's inertial frame, `body_ipos`). The box looks exactly the same for
every e (uniform colour, a symmetric tape stripe across its middle).

Why the prior fails: the natural push is at the middle of the box's back face. On a table the friction
resultant acts below the centre of mass, so a push whose line does not pass near it turns the box: the
heavy end lags behind. Measured (closed gripper at yaw 0, 8 cm/s, mu 0.5): pushing at the middle turns the
box 22 deg (e = 4 cm) to 42 deg (e = 7 cm); the rectangle (box + 2 cm margin per side) tolerates ~9 deg.
Pushing within ~2 cm of the centre of mass gives a straight translation (< 1 deg); 3 cm off gives 5-14 deg.

What must be learned from the failure: the turning direction tells which end is heavy (the end that
lagged), the amount tells roughly how far off the push was; next time move the contact point toward the
heavy end. F2 reports the box's turn (direction and size, plus how far one end lags the other), where the
box's centre ended relative to the rectangle's centre, and where on the back face the gripper first
touched it -- all relative measurements, never absolute coordinates or e itself.

Rule (stated in the instruction and enforced): the box may be moved by a single move_to command (one
straight push). Otherwise an agent could steer the box with several corrective pushes inside one attempt
and the task would no longer test what is learnt across attempts.
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
PUSH_SPEED = 0.08
FINGER_AHEAD = 0.009  # the closed fingers' front face is ~0.9 cm ahead of the grasp point (measured, yaw 0)
RUNUP = 0.04
MOVE_TOL = 0.01  # a command "moved the box" if its centre moved this much ...
TURN_TOL = np.radians(3.0)  # ... or it turned this much


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


def _lr(v, unit="cm"):
    return f"{100 * abs(v):.1f} {unit} to the {'left' if v > 0 else 'right'}"


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


class PushSkills(Skills):
    """Skills that log, for whoever drives them (scripted policy or agent), which move_to commands moved the
    box and where the gripper first touched it."""

    def __init__(self, env, bias=None, recorder=None):
        super().__init__(env, bias=bias, recorder=recorder)
        self.first_touch = None  # where the gripper's centre was (box frame) when the box first moved
        self._p0, self._yaw0 = self._box_pose()
        self.box_moves = []  # one entry per move_to that moved / turned the box

    def _box_pose(self):
        p = envs.obj_pos(self.env, BOX)
        return p, _yaw(envs.obj_quat(self.env, BOX))

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        if self.first_touch is None:
            # "first touch" = the first control step at which the box has moved 2 mm. Detected from the box's motion,
            # not from contacts: a stiff gripper pushing a box makes a stick-slip contact that is often absent at the
            # control-step boundaries (measured), so contact polling misses pushes.
            p, _ = self._box_pose()
            if np.linalg.norm(p[:2] - self._p0[:2]) > 0.002:
                R = np.array([[np.cos(self._yaw0), np.sin(self._yaw0)], [-np.sin(self._yaw0), np.cos(self._yaw0)]])
                local = R @ (self.eef_pos()[:2] - self._p0[:2])  # gripper centre in the box's starting frame
                self.first_touch = dict(local=local.tolist(), step=self.n_steps)

    def move_to(self, target, *args, **kwargs):
        p0, y0 = self._box_pose()
        rep = super().move_to(target, *args, **kwargs)
        p1, y1 = self._box_pose()
        dturn = (y1 - y0 + np.pi) % (2 * np.pi) - np.pi
        if np.linalg.norm(p1[:2] - p0[:2]) > MOVE_TOL or abs(dturn) > TURN_TOL:
            self.box_moves.append(dict(step=self.n_steps, moved_cm=round(100 * float(np.linalg.norm(p1[:2] - p0[:2])), 1),
                                       turned_deg=round(float(np.degrees(dturn)), 1)))
        return rep


@register_task
class OffcenterPush(LFFTask):
    name = "l7_offcenter_push"
    level = "L7"
    category = "failed_interaction_inference"
    capabilities = ("Perceive", "Reason", "Utilize")
    protocol = "cross"
    instruction = ("Slowly push the long brown box straight forward (away from the robot) with the closed gripper so "
                   "that it ends up completely inside the green rectangle, without turning. Use a single push: after "
                   "placing the gripper behind the box, move the box with one move command only.")
    instruction_indirect = ("Slowly push the long cardboard-coloured block straight forward (away from the robot) with "
                            "the closed gripper so that it ends up completely inside the marked green area, without "
                            "turning. Use a single push: after placing the gripper behind the block, move it with one "
                            "move command only.")

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
        envs.set_friction(self.env, BOX, MU)
        x0, y0 = inst["box_xy"]
        envs.set_obj_pose(self.env, BOX, [x0, y0, envs.TABLE_Z + HALF[2] + 0.001])

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = PushSkills(self.env, recorder=recorder)
        return self.sk

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 30)
        sk = self.sk
        p = envs.obj_pos(self.env, BOX)
        yaw = _yaw(envs.obj_quat(self.env, BOX))
        t = self.target_xy(inst)
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = [p[:2] + R @ np.array([sx * HALF[0], sy * HALF[1]]) for sx in (-1, 1) for sy in (-1, 1)]
        inside = all(abs(c[0] - t[0]) <= RECT_HALF[0] and abs(c[1] - t[1]) <= RECT_HALF[1] for c in corners)
        upright = envs.obj_upright_cos(self.env, BOX) > 0.95
        n_push = len(sk.box_moves)
        off = p[:2] - t
        lag = 2 * HALF[1] * np.sin(yaw)  # how far the +y end is behind (+) the -y end along x
        success = bool(inside and upright and n_push <= 1)
        out = dict(success=success, inside=bool(inside), upright=bool(upright), n_box_moving_commands=n_push,
                   box_moves=sk.box_moves, yaw_deg=round(float(np.degrees(yaw)), 1),
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
            elif n_push > 1:
                out["failure"] = "several_pushes"
                parts.append(f"The box was moved by {n_push} separate move commands; only one push is allowed.")
            elif not upright:
                out["failure"] = "tipped"
                parts.append("The box tipped over.")
            else:
                out["failure"] = "turned" if abs(np.degrees(yaw)) > 6 else "position"
                parts.append("The box is not completely inside the rectangle.")
            turn = np.degrees(yaw)
            if abs(turn) >= 1.0:
                parts.append(f"It turned {abs(turn):.0f} degrees {'counterclockwise' if turn > 0 else 'clockwise'} "
                             f"(seen from above): its {'left' if lag > 0 else 'right'} end ended {100 * abs(lag):.1f} cm "
                             f"further back than its {'right' if lag > 0 else 'left'} end.")
            else:
                parts.append("It did not turn (less than 1 degree).")
            parts.append(f"Its centre ended {_fb(off[0])} and {_lr(off[1])} of the rectangle's centre.")
            parts.append(f"When the box started to move, the gripper's centre was {_lr(lt[1])} of the middle of "
                         f"the box (measured along the box's long side).")
        out["detail"] = " ".join(parts)
        return out

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def execute(self, inst, params):
        """Closed gripper behind the box at lateral offset `c` (from the box's middle, along its long axis), then one
        slow straight move_to forward ending `end_dx` beyond the nominal end point (target centre minus half the box
        depth minus the finger offset)."""
        sk = self.sk
        p = envs.obj_pos(self.env, BOX)  # perfect perception of the current scene
        t = self.target_xy(inst)
        c = float(params["c"])
        y = p[1] + c
        face = p[0] - HALF[0]
        x_end = t[0] - HALF[0] - FINGER_AHEAD + float(params.get("end_dx", 0.0))
        sk.set_gripper(True, steps=8)
        sk.move_to([face - RUNUP, y, Z_PUSH + 0.08], yaw=0.0)
        sk.move_to([face - RUNUP, y, Z_PUSH], tol=0.003)
        sk.move_to([x_end, y, Z_PUSH], speed=PUSH_SPEED, tol=0.004, max_steps=400)
        sk.move_to([x_end - 0.03, y, Z_PUSH + 0.10], speed=0.2)

    def default_params(self, inst):
        return dict(c=0.0, end_dx=0.0)

    def oracle_params(self, inst):
        return dict(c=float(inst["com_offset"]), end_dx=0.0)

    STEP = 0.06  # first correction toward the lagging end (the expected |e| range is unknown to the learner)

    def adapt_params(self, inst, history):
        """Scripted F2 learner. A counterclockwise turn (left end lagged) means the push passed to the right of the
        centre of mass -> move the contact point left, and vice versa. Keeps a bracket [lo, hi] on the contact
        offset from the measured touch points: bisect once both sides are known, otherwise step by STEP. Also
        corrects the push length from the measured forward error once a push went straight."""
        lo, hi = -np.inf, np.inf
        end_dx = history[-1]["params"].get("end_dx", 0.0)
        for h in history:
            o = h["outcome"]
            c = o["touch_offset"] if o.get("touch_offset") is not None else h["params"]["c"]
            if o["yaw_deg"] > 3.0:  # left end lagged: centre of mass is left of the contact
                lo = max(lo, c)
            elif o["yaw_deg"] < -3.0:
                hi = min(hi, c)
        last = history[-1]
        if abs(last["outcome"]["yaw_deg"]) <= 3.0 and last["outcome"].get("touch_offset") is not None:
            c_new = last["params"]["c"]
            end_dx = end_dx - last["outcome"]["centre_offset"][0]
        elif np.isfinite(lo) and np.isfinite(hi):
            c_new = 0.5 * (lo + hi)
        elif np.isfinite(lo):
            c_new = lo + self.STEP
        elif np.isfinite(hi):
            c_new = hi - self.STEP
        else:
            c_new = last["params"]["c"]
        c_new = float(np.clip(c_new, -HALF[1] + 0.02, HALF[1] - 0.02))
        return dict(c=round(c_new, 4), end_dx=round(float(end_dx), 4))

    blind_sigma = 0.008

    def blind_params(self, inst, attempt, rng):
        """Same plan (push at the middle) with a small random variation of the contact point; no history."""
        return dict(c=float(rng.normal(0.0, self.blind_sigma)), end_dx=0.0)
