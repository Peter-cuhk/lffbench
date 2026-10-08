"""L7 key_turn: unlock a lock whose key turns only one way; which way is hidden.

Category: failed_interaction_inference ("why did this contact not do what I expected?").

Scene: a grey lock box (9 x 9 x 5 cm) on the table with a brass key standing in the keyhole on its top; the key's
flat head (3.6 cm wide, 1.6 cm thick, 3 cm tall) sticks up so the gripper can grasp it from above and turn it by
rotating the wrist. A red bolt sticks out of the box's front face (+x, towards the front camera) while it is locked
and slides in once the key has been turned to the open position.

Hidden variable: the turning direction that opens the lock, clockwise (CW) or counterclockwise (CCW) seen from
above (alternates with the seed parity, so every block of instances is balanced). The other way the key has 3 deg of
play and then a hard stop. Both look exactly the same.

Rule (stated in the instruction and enforced): one turn per attempt -- grasp the key, ONE move command that turns the
wrist while holding it, release. A "turn" is any move_to that starts with the key held and changes the commanded yaw
by >= 5 deg, or that rotates the key by > 3 deg. Without the rule an agent could try both ways inside one attempt and
the task would no longer test what is learnt across attempts (see the task card for the within-attempt variant).

Why the prior fails: the user's example -- many people first try counterclockwise. The scripted prior turns CCW by
90 deg, so it fails on every CW instance (half of them): the key rotates by its 3 deg of play, then the wrist stalls.
What must be learned from the failure: "the key would not turn this way at all" (not "my grip slipped") -> turn it
the other way. One failure is enough.

Mechanics notes (measured in the prototype, 2026-10-08):
  * the Panda's finger servo squeezes with kp * (finger opening) N, i.e. only ~4 N on a 7 mm head; the head then
    twisted between the fingers and wedged them open. A 16 mm thick head gives ~8 N and turns reliably.
  * without a stalling wrist the OSC keeps twisting against a blocked key, the fingers wedge open and on release
    they knock the key 15-45 deg the free way (leaking the answer). `KeySkills` therefore stops rotating the wrist
    once the held key sits on a stop and the gripper has twisted 6 deg past it (a torque-limited wrist), and limits
    the wrist to 90 deg/s while turning.
  * the stops use a stiff solref/solimp: with the defaults a hard twist pushed the key 10-15 deg past its stop.
  * commanded yaw is taken literally in this task (nearest representation to the current yaw, clipped to
    +-113 deg). The shared Skills.wrap_yaw maps yaw modulo 180 deg, which is right for grasping but would turn
    "yaw = +95" into -85 deg, i.e. turn the key the wrong way.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, register_generated
from ..skills import DT, Skills, _rot_err, _rotz
from ..task_base import LFFTask, register_task

LOCK = "lock_1"
MODEL = "lff_l7_lock"
LX, LY, LH = 0.045, 0.045, 0.05  # housing half x, half y, full height
HEAD = (0.018, 0.008, 0.015)  # key head half extents: local x = width, y = thickness, z = height
HEAD_Z = 0.004 + HEAD[2]  # head centre above the housing top
BOLT = (0.008, 0.012, 0.006)  # red bolt (visual only) half extents
BOLT_OUT = (LX + BOLT[0], 0.0, 0.022)
BOLT_IN = (LX - BOLT[0] - 0.002, 0.0, 0.022)
PLAY = np.radians(3.0)
STOP = np.radians(90.0)
UNLOCK_AT = np.radians(80.0)
KEY_FRICTION = 0.04  # N m
W_MAX = np.radians(90.0)  # wrist speed limit while turning (rad/s)
TWIST_MAX = np.radians(6.0)  # wrist stalls when it has twisted this far past a blocked key
STALL_EXIT = 20  # control steps of stalling before the move gives up
TURN_YAW = np.radians(5.0)
TURN_KEY = np.radians(3.0)
Z_GRIP = envs.TABLE_Z + LH + HEAD_Z  # grasp-point height for the key head (its centre)


def lock_xml(name, body_rgba=(0.55, 0.57, 0.60, 1.0), key_rgba=(0.80, 0.64, 0.22, 1.0), bolt_rgba=(0.80, 0.12, 0.10, 1.0)):
    g = [f'<geom name="{name}_housing" type="box" pos="0 0 {LH / 2:.4f}" size="{_fmt((LX, LY, LH / 2))}" '
         f'rgba="{_fmt(body_rgba)}" friction="1 0.005 0.0001" {_SOLID} group="0"/>',
         f'<geom type="box" pos="0 0 {LH / 2:.4f}" size="{_fmt((LX, LY, LH / 2))}" rgba="{_fmt(body_rgba)}" '
         f'contype="0" conaffinity="0" group="1"/>',
         f'<geom name="{name}_bolt" type="box" pos="{_fmt(BOLT_OUT)}" size="{_fmt(BOLT)}" rgba="{_fmt(bolt_rgba)}" '
         f'contype="0" conaffinity="0" group="1"/>',
         f'<geom type="cylinder" pos="0 0 {LH + 0.0005:.4f}" size="0.011 0.0005" rgba="0.35 0.36 0.38 1" '
         f'contype="0" conaffinity="0" group="1"/>']
    k = ['<inertial pos="0 0 0.01" mass="0.03" diaginertia="0.00001 0.00001 0.00001"/>',
         # placeholder range: the open direction is set per instance (apply_instance)
         f'<joint name="{name}_key_joint" type="hinge" axis="0 0 1" pos="0 0 0" limited="true" range="-0.05 1.6" '
         f'damping="0.01" frictionloss="{KEY_FRICTION}" armature="0.0002"/>',
         f'<geom name="{name}_head" type="box" pos="0 0 {HEAD_Z:.4f}" size="{_fmt(HEAD)}" rgba="{_fmt(key_rgba)}" '
         f'friction="1 0.005 0.0001" {_SOLID} group="0"/>',
         f'<geom type="box" pos="0 0 {HEAD_Z:.4f}" size="{_fmt(HEAD)}" rgba="{_fmt(key_rgba)}" contype="0" '
         f'conaffinity="0" group="1"/>',
         f'<geom type="cylinder" pos="0 0 0.002" size="0.006 0.002" rgba="{_fmt(key_rgba)}" contype="0" '
         f'conaffinity="0" group="1"/>']
    shell = "\n        ".join(g)
    key = "\n          ".join(k)
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        {shell}
        <body name="key" pos="0 0 {LH:.4f}">
          {key}
        </body>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {LH:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{LX:.4f} {LY:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>"""


def _wrap_pi(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _wrap_half(a):
    """the gripper is symmetric under a rotation by pi"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def _rot_words(deg):
    return f"{abs(deg):.0f} deg {'counterclockwise' if deg > 0 else 'clockwise'}"


def _lr(v):
    return f"{100 * abs(v):.1f} cm to the {'left' if v > 0 else 'right'}"


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


class KeySkills(Skills):
    """Skills for the key task: literal yaw, wrist speed limit, stalling wrist, and a log of turns / grasps / unlock."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task
        m = env.sim.model
        self._grip_geoms = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(("gripper0_", "robot0_"))]
        self.turns = []  # one entry per move_to that counts as a turn
        self.closes = []  # close_gripper events: gripper centre relative to the key head, held afterwards?
        self.stall_steps = 0
        self._grasp_off = None  # gripper yaw minus key yaw when the key was grasped
        self.touch_since_report = False

    # ------------------------------------------------------------ state
    def eef_yaw(self):
        R = self.eef_mat() @ self.R0.T
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def holding_key(self):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=[self.task.head_geom_name]))

    def _touching_key(self):
        d = self.env.sim.data
        g, kg = set(self._grip_geoms), self.task.head_gid
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in g and c.geom2 == kg) or (c.geom2 in g and c.geom1 == kg):
                return True
        return False

    def wrap_yaw(self, yaw):
        """Literal yaw: the representation of `yaw` closest to the current wrist yaw, clipped to the wrist range."""
        cur = self.eef_yaw()
        y = cur + _wrap_pi(float(yaw) - cur)
        return float(np.clip(y, -self.YAW_LIMIT, self.YAW_LIMIT))

    # ------------------------------------------------------------ low level
    def _act(self, dpos, drot, grip):
        t = self.task
        if self.grip > 0 and self._grasp_off is not None:
            q = t.key_q()
            lo, hi = t.key_range()
            twist = _wrap_half(self.eef_yaw() - t.key_world_yaw() - self._grasp_off)
            at_hi, at_lo = q >= hi - np.radians(1.5), q <= lo + np.radians(1.5)
            if (at_hi and twist > TWIST_MAX and drot[2] > 0) or (at_lo and twist < -TWIST_MAX and drot[2] < 0):
                drot = np.array(drot, float).copy()
                drot[2] = -np.sign(twist) * 0.02  # force-limited wrist: back off instead of twisting further
                self.stall_steps += 1
        super()._act(dpos, drot, grip)
        t.check_unlock()
        if self._touching_key():
            self.touch_since_report = True

    def move_to(self, target, yaw=None, speed=None, tol=0.004, max_steps=250, apply_bias=True, wrap_yaw=True):
        """Skills.move_to with the wrist reference rotating at <= W_MAX and an early stop when the wrist stalls.
        Logs the move as a turn when it starts with the key held and rotates the wrist, or when it rotates the key."""
        t = self.task
        tgt = np.asarray(target, float) + (self.bias if apply_bias else 0.0)
        if yaw is not None:
            self.yaw = self.wrap_yaw(float(yaw)) if wrap_yaw else float(yaw)
        yaw0 = self.eef_yaw()
        dyaw = self.yaw - yaw0
        R_goal = self._R()
        p0 = self.eef_pos()
        q0, held0 = t.key_q(), self.holding_key()
        dist = np.linalg.norm(tgt - p0)
        direction = (tgt - p0) / max(dist, 1e-9)
        s = 0.0
        self.stall_steps = 0
        for k in range(max_steps):
            if speed is not None:
                s = min(dist, s + speed * DT)
                ref = p0 + direction * s + direction * min(dist - s, speed * DT * 1.5)
            else:
                ref = tgt
            frac = min(1.0, (k + 1) * DT * W_MAX / max(abs(dyaw), 1e-6))
            R = _rotz(yaw0 + frac * dyaw) @ self.R0
            self._act(ref - self.eef_pos(), _rot_err(R, self.eef_mat()), self.grip)
            if self.stall_steps >= STALL_EXIT:
                break
            if (np.linalg.norm(tgt - self.eef_pos()) < tol and (speed is None or s >= dist) and frac >= 1.0
                    and np.linalg.norm(_rot_err(R_goal, self.eef_mat())) < 0.03):
                break
        stalled = self.stall_steps >= STALL_EXIT
        if stalled:
            # give up the unreached wrist angle; otherwise every later hold (e.g. while the gripper opens) keeps
            # driving the wrist towards it and the arm swings away (seen in the agent self-test, 2026-10-08)
            self.yaw = self.eef_yaw()
        rep = dict(prim="move_to", reached=bool(np.linalg.norm(tgt - self.eef_pos()) < 0.02),
                   pos=self.eef_pos().round(4).tolist(), steps=k + 1, stalled=bool(stalled))
        self.log.append(rep)
        dq = t.key_q() - q0
        if (held0 and abs(dyaw) >= TURN_YAW) or abs(dq) > TURN_KEY:
            self.turns.append(dict(step=self.n_steps, held_at_start=bool(held0), held_at_end=self.holding_key(),
                                   key_start_deg=round(float(np.degrees(q0)), 1),
                                   key_end_deg=round(float(np.degrees(t.key_q())), 1),
                                   wrist_cmd_deg=round(float(np.degrees(dyaw)), 1),
                                   wrist_turn_deg=round(float(np.degrees(self.eef_yaw() - yaw0)), 1),
                                   stalled=bool(stalled)))
        return rep

    def set_gripper(self, close, steps=15):
        was_open = self.grip < 0
        rep = super().set_gripper(close, steps)
        if close:
            if self.holding_key():
                self._grasp_off = _wrap_half(self.eef_yaw() - self.task.key_world_yaw())
            if was_open:
                off = self.eef_pos() - self.task.head_xyz()
                self.closes.append(dict(step=self.n_steps, offset=off.round(4).tolist(), held=self.holding_key(),
                                        width=round(self.gripper_width(), 4)))
        else:
            self._grasp_off = None
        return rep


@register_task
class KeyTurn(LFFTask):
    name = "l7_key_turn"
    level = "L7"
    category = "failed_interaction_inference"
    capabilities = ("Perceive", "Reason", "Plan")
    protocol = "cross"
    instruction = ("Unlock the lock on the grey box: grasp the flat head of the brass key from above and turn the key "
                   "a quarter turn (90 degrees). The red bolt on the box slides in when it is unlocked. Only one turn "
                   "is allowed per attempt: grasp the key, turn it with ONE move command while holding it, then "
                   "release it.")
    instruction_indirect = ("Open the lock of the small grey metal box: take hold of the flat gold-coloured tab that "
                            "sticks up out of its top and rotate it a quarter turn (90 degrees). The red block on the "
                            "box slides in when it is open. Only one turn is allowed per attempt: grasp the tab, rotate "
                            "it with ONE move command while holding it, then release it.")

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        lock = register_generated("LffKeyLock", lock_xml(MODEL), free=False)
        return write_bddl(self.name, "unlock the lock", [],
                          fixtures=[(LOCK, lock, "lock_region", (-0.11, -0.01, -0.09, 0.01))])

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        pre = f"{LOCK}_{MODEL}"
        self.jid = m.joint_name2id(pre + "_key_joint")
        self.qa = m.jnt_qposadr[self.jid]
        self.dof = m.jnt_dofadr[self.jid]
        self.lock_bid = m.body_name2id(self.env.fixtures_dict[LOCK].root_body)
        self.head_geom_name = pre + "_head"
        self.head_gid = m.geom_name2id(self.head_geom_name)
        self.bolt_gid = m.geom_name2id(pre + "_bolt")
        self.inst = None
        self.unlocked = False

    # agent-facing hooks -------------------------------------------------------
    def on_agent_tool(self, inst, name, args, report):
        """The lock is a fixture, so the harness's `holding_object` / `touched_object` (movable objects only) never see
        the key; add an explicit `holding_key` field and count key contacts as touches."""
        if self.sk is None or not isinstance(report, dict):
            return
        touched = self.sk.touch_since_report
        self.sk.touch_since_report = False
        if report.get("ok") and name in ("move_to", "open_gripper", "close_gripper", "push"):
            report["holding_key"] = self.sk.holding_key()
            if touched and "touched_object" in report:
                report["touched_object"] = True

    # ------------------------------------------------------------ state helpers
    def key_q(self):
        return float(self.env.sim.data.qpos[self.qa])

    def key_range(self):
        return self.env.sim.model.jnt_range[self.jid]

    def key_world_yaw(self):
        return float(self.inst["lock_yaw"]) + self.key_q()

    def head_xyz(self):
        return self.env.sim.data.geom_xpos[self.head_gid].copy()

    def open_sign(self):
        return 1.0 if self.inst["direction"] == "CCW" else -1.0

    def check_unlock(self):
        """latch: the bolt is thrown back once the key reaches the open position (and stays back)"""
        if not self.unlocked and self.open_sign() * self.key_q() >= UNLOCK_AT:
            self.unlocked = True
            self.env.sim.model.geom_pos[self.bolt_gid] = BOLT_IN

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        return dict(seed=int(seed), direction="CCW" if seed % 2 == 0 else "CW",
                    lock_xy=[float(rng.uniform(-0.15, -0.02)), float(rng.uniform(-0.15, 0.15))],
                    lock_yaw=float(rng.uniform(-np.radians(15), np.radians(15))))

    def apply_instance(self, inst):
        self.inst = inst
        m, d = self.env.sim.model, self.env.sim.data
        cx, cy = inst["lock_xy"]
        a = inst["lock_yaw"]
        m.body_pos[self.lock_bid] = [cx, cy, envs.TABLE_Z]
        m.body_quat[self.lock_bid] = [np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]
        m.jnt_range[self.jid] = [-PLAY, STOP] if inst["direction"] == "CCW" else [-STOP, PLAY]
        m.jnt_solref[self.jid] = [0.004, 1.0]
        m.jnt_solimp[self.jid] = [0.99, 0.999, 0.001, 0.5, 2.0]
        m.dof_frictionloss[self.dof] = KEY_FRICTION
        m.geom_pos[self.bolt_gid] = BOLT_OUT
        d.qpos[self.qa] = 0.0
        d.qvel[self.dof] = 0.0
        self.unlocked = False

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = KeySkills(self.env, self, recorder=recorder)
        return self.sk

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 20)
        sk = self.sk
        n_turns = len(sk.turns)
        success = bool(self.unlocked and n_turns <= 1)
        open_deg = float(np.degrees(self.open_sign() * self.key_q()))
        out = dict(success=success, unlocked=bool(self.unlocked), n_turns=n_turns, key_deg=round(np.degrees(self.key_q()), 1),
                   open_deg=round(open_deg, 1), turns=sk.turns, closes=sk.closes, first_turn=None)
        parts = []
        grasp_txt = ""
        if sk.closes:
            c = sk.closes[0]
            o = c["offset"]
            top = envs.TABLE_Z + LH + HEAD_Z + HEAD[2]
            below = top - (o[2] + self.head_xyz()[2])
            grasp_txt = (f"When the gripper first closed, its centre was {_fb(o[0])} and {_lr(o[1])} of the centre of "
                         f"the key's head and {100 * below:.1f} cm below the top of the head; it "
                         + ("was holding the key." if c["held"] else "did not get hold of the key."))
        if success:
            out["failure"] = None
            parts.append("The lock is unlocked.")
        elif n_turns == 0:
            out["failure"] = "no_turn"
            parts.append("The lock is still locked: the key was never turned.")
        else:
            out["failure"] = "several_turns" if n_turns > 1 else "still_locked"
            if n_turns > 1:
                parts.append(f"The key was turned (or pushed round) in {n_turns} separate moves; only one turn is "
                             f"allowed per attempt, so this attempt failed.")
            parts.append("The lock is unlocked." if self.unlocked else "The lock is still locked.")
            t = sk.turns[0]
            out["first_turn"] = t
            dk = t["key_end_deg"] - t["key_start_deg"]
            txt = f"During your (first) turn the wrist rotated {_rot_words(t['wrist_turn_deg'])} (seen from above)"
            if abs(dk) < 1.0:
                txt += " and the key did not rotate at all"
            else:
                txt += f" and the key rotated {_rot_words(dk)}"
            if t["stalled"]:
                txt += f", then it would not turn any further and the wrist stopped there"
            if t["held_at_start"]:
                txt += ("; the gripper lost hold of the key during the turn." if not t["held_at_end"]
                        else "; the gripper held the key throughout the turn.")
            else:
                txt += "; the gripper was not holding the key when that move started."
            parts.append(txt)
        if grasp_txt and not success:
            parts.append(grasp_txt)
        out["detail"] = " ".join(parts)
        return out

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def execute(self, inst, params):
        """Grasp the key head from above across its thickness (+ jitter), ONE wrist turn of `turn` degrees
        (+ = counterclockwise seen from above), release, lift."""
        sk = self.sk
        p = self.head_xyz() + np.r_[params.get("dxy", (0.0, 0.0)), params.get("dz", 0.0)]  # perfect perception
        yaw0 = self.key_world_yaw() + np.radians(params.get("dyaw", 0.0))
        sk.set_gripper(False, steps=8)
        sk.move_to([p[0], p[1], p[2] + 0.08], yaw=yaw0)
        sk.move_to([p[0], p[1], p[2]], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([p[0], p[1], p[2]], yaw=sk.eef_yaw() + np.radians(float(params["turn"])), max_steps=200)
        sk.set_gripper(False, steps=8)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], e[2] + 0.08])

    def default_params(self, inst):
        """the prior from the user's example: try counterclockwise first"""
        return dict(turn=90.0)

    def oracle_params(self, inst):
        return dict(turn=90.0 * (1 if inst["direction"] == "CCW" else -1))

    def adapt_params(self, inst, history):
        """Scripted F2 learner. A turn after which the key rotated less than 10 deg the commanded way (it stopped)
        -> that way is blocked -> turn the other way. A turn that rotated the key further but not far enough -> same
        way, a bit more. Anything else (no turn, several turns, lost grip) -> repeat the last direction."""
        blocked = set()
        good = 0
        for h in history:
            t = h["outcome"].get("first_turn")
            if t is None:
                continue
            sgn = int(np.sign(h["params"]["turn"]))
            dk = t["key_end_deg"] - t["key_start_deg"]
            if t["held_at_start"] and sgn * dk < 10.0:
                blocked.add(sgn)
            elif sgn * dk >= 10.0:
                good = sgn
        if good:
            return dict(turn=100.0 * good)
        last = int(np.sign(history[-1]["params"]["turn"]))
        sgn = -last if last in blocked else last
        if sgn in blocked:
            sgn = -sgn
        return dict(turn=90.0 * sgn)

    def blind_params(self, inst, attempt, rng):
        """Same plan (counterclockwise quarter turn) with small random variation; no history."""
        return dict(turn=float(np.clip(rng.normal(90.0, 8.0), 70.0, 105.0)), dxy=rng.normal(0.0, 0.002, 2).tolist(),
                    dyaw=float(rng.normal(0.0, 4.0)))
