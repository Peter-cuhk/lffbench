"""L5-B slippery_lift: pick up a heavy, slippery metal bar and hold it level.

Category dynamics_adaptation (L5): "figure out how this object moves".

Scene: one long steel-grey bar (4 x 28 x 4 cm, lying along the robot's left-right axis) on the table. It looks
perfectly uniform, but a heavy slug is hidden inside it at an unknown position, so its centre of mass sits
e = 2.5-7 cm from the bar's midpoint, towards the left or the right end. The bar is heavy (0.8-1.2 kg) and its
surface is slippery (finger friction 0.40-0.60): the two finger pads can hold its weight, but they cannot hold
much torque.

Why the prior fails. The natural grasp is across the bar at its midpoint. With the centre of mass a few cm to
one side, gravity twists the bar in the pads as soon as it is lifted: the heavy end swings down and the bar
ends up hanging tilted by ~10-50 deg (measured). Slow lifting does not help (the torque is static).

What has to be learned. Which end sank tells on which side of the grasp the centre of mass lies; the grasp
has to move towards that end until it is within ~1.5 cm of the hidden centre of mass (measured window). The
tilt magnitude saturates, so the distance has to be found by bracketing over a few attempts.

Measured physics (2026-10-07, `l5dyn/bar_win.py`): mass 0.8-1.2 kg, mu 0.4-0.6, lift 15 cm at 0.15 m/s:
|grasp - COM| <= 1.0 cm -> tilt <= 2 deg; 1.5 cm -> 0-41 deg; >= 2 cm -> mostly 30-50 deg; the tilt does not
creep during a 3 s hold. mu 0.8 gave non-monotone tilts and is not used; mu 0.3 + 1.3 kg slips out entirely.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

BAR = "bar_1"
HALF = (0.02, 0.14, 0.02)  # 4 (x) x 28 (y) x 4 (z) cm, long axis along world y
_SOLID = 'solimp="0.998 0.998 0.001" solref="0.001 1"'
W_FRAC = 0.6  # fraction of the bar's mass in the hidden slug
W_HALF = 0.015  # slug half size (for its inertia)
E_MIN, E_MAX = 0.025, 0.070  # |COM offset| from the bar's midpoint (m)
LIFT_MIN = 0.10  # the bar's lowest point must be this high above the table
TILT_MAX = 8.0  # deg
Z_GRASP = envs.TABLE_Z + HALF[2]  # gripper site at the bar's mid height (finger tips 0.7 cm above the table)
Z_LIFT = Z_GRASP + 0.15
YAW_ACROSS = np.pi / 2  # jaws close along world x, i.e. across the bar


def weighted_box_xml(name, half, rgba, mass=1.0, friction=(1.0, 0.005, 0.0001)):
    """Box with an explicit inertial. Its pos is off-centre at compile time so MuJoCo keeps
    body_sameframe = 0 / body_simple = 0 and honours the per-instance body_ipos set in apply_instance."""
    I = [mass / 3 * (half[1] ** 2 + half[2] ** 2), mass / 3 * (half[0] ** 2 + half[2] ** 2),
         mass / 3 * (half[0] ** 2 + half[1] ** 2)]
    g = (f'        <inertial pos="0 0.001 0.001" mass="{mass}" diaginertia="{_fmt(I)}" />\n'
         f'        <geom name="{name}_g0" type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" '
         f'friction="{_fmt(friction)}" {_SOLID} group="0" />\n'
         f'        <geom type="box" size="{_fmt(half)}" rgba="{_fmt(rgba)}" conaffinity="0" contype="0" group="1" />')
    return _wrap(name, g, half[2], float(np.hypot(half[0], half[1])))


def set_hidden_weight(env, obj, half, mass, w_frac, w_pos, w_half=W_HALF):
    """Uniform shell of mass (1 - w_frac) * mass plus a hidden cube slug of w_frac * mass at w_pos (body frame).
    Sets the body's mass, centre of mass and inertia; returns the centre of mass (body frame)."""
    m = env.sim.model
    bid = envs.body_id(env, obj)
    ms, mw = (1 - w_frac) * mass, w_frac * mass
    w_pos = np.asarray(w_pos, float)
    c = mw * w_pos / mass
    hx, hy, hz = half

    def par(mm, r):  # parallel-axis terms (diagonal)
        return mm * np.array([r[1] ** 2 + r[2] ** 2, r[0] ** 2 + r[2] ** 2, r[0] ** 2 + r[1] ** 2])

    I = (ms / 3 * np.array([hy ** 2 + hz ** 2, hx ** 2 + hz ** 2, hx ** 2 + hy ** 2]) + par(ms, -c)
         + mw / 6 * (2 * w_half) ** 2 * np.ones(3) + par(mw, w_pos - c))
    m.body_mass[bid] = mass
    m.body_ipos[bid] = c
    m.body_iquat[bid] = [1.0, 0.0, 0.0, 0.0]
    m.body_inertia[bid] = I
    return c


def bar_axis(env):
    """World direction of the bar's long axis (body y)."""
    return env.sim.data.body_xmat[envs.body_id(env, BAR)].reshape(3, 3)[:, 1].copy()


def bar_tilt_deg(env):
    """Signed tilt of the bar's long axis from horizontal; > 0: the bar's +y end is lower."""
    ax = bar_axis(env)
    if ax[1] < 0:  # orient the axis towards the robot's left so the sign is about world left/right
        ax = -ax
    return float(np.degrees(np.arcsin(np.clip(-ax[2], -1.0, 1.0))))


def side_words(sign):
    return "left (+y)" if sign > 0 else "right (-y)"


class BarSkills(Skills):
    """Skills that record where along the bar the jaws closed, whoever drives them (scripted or agent)."""

    def __init__(self, env, bias=None, recorder=None):
        super().__init__(env, bias=bias, recorder=recorder)
        self.closes = []
        self.zmax = float(envs.obj_pos(env, BAR)[2])
        self.z0 = self.zmax

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.zmax = max(self.zmax, float(envs.obj_pos(self.env, BAR)[2]))

    def set_gripper(self, close, steps=15):
        rep = super().set_gripper(close, steps)
        if close:
            c = envs.obj_pos(self.env, BAR)
            ax = bar_axis(self.env)
            if ax[1] < 0:
                ax = -ax
            along = float((self.eef_pos() - c) @ ax)  # + : grasp on the left (+y) side of the midpoint
            self.closes.append(dict(along=along, holding=self.holding(BAR), step=self.n_steps))
        return rep


@register_task
class SlipperyLift(LFFTask):
    name = "l5_slippery_lift"
    level = "L5"
    category = "dynamics_adaptation"
    capabilities = ("Perceive", "Reason", "Utilize")
    protocol = "cross"
    instruction = ("Pick up the metal bar and hold it level (tilted less than 8 degrees), with its lowest point at "
                   "least 10 cm above the table. Keep holding it when you finish.")
    instruction_indirect = ("Pick up the long grey rod lying on the table and hold it horizontal (tilted less than 8 "
                            "degrees), with its lowest point at least 10 cm above the table. Keep holding it when you "
                            "finish.")

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        bar = register_generated("LffSteelBar", weighted_box_xml("lff_steel_bar", HALF, (0.52, 0.55, 0.58, 1.0)))
        objs = [(BAR, bar, "bar_region", (-0.11, -0.01, -0.09, 0.01))]
        return write_bddl(self.name, "pick up the bar and hold it level", objs)  # BDDL cannot parse parentheses

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        e = float(rng.choice([-1.0, 1.0]) * rng.uniform(E_MIN, E_MAX))
        return dict(seed=int(seed),
                    bar_xy=[float(rng.uniform(-0.12, -0.03)), float(rng.uniform(-0.06, 0.06))],
                    com_offset=e,  # along the bar, + = towards the robot's left (+y) end
                    mass=float(rng.uniform(0.8, 1.2)),
                    mu=float(rng.uniform(0.40, 0.60)))

    def apply_instance(self, inst):
        x, y = inst["bar_xy"]
        envs.set_obj_pose(self.env, BAR, [x, y, envs.TABLE_Z + HALF[2] + 0.001])
        envs.set_friction(self.env, BAR, inst["mu"])
        set_hidden_weight(self.env, BAR, HALF, inst["mass"], W_FRAC, [0.0, inst["com_offset"] / W_FRAC, 0.0])

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = BarSkills(self.env, bias=self.skill_bias(inst), recorder=recorder)
        return self.sk

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        sk = self.sk
        sk.hold(30)  # 1.5 s with the gripper as the agent left it (envs.settle would open the jaws)
        held = sk.holding(BAR)
        tilt = bar_tilt_deg(self.env)
        bottom = envs.obj_min_z(self.env, BAR) - envs.TABLE_Z
        lifted = (sk.zmax - sk.z0) > 0.03
        close = next((c for c in reversed(sk.closes) if c["holding"]), sk.closes[-1] if sk.closes else None)
        success = bool(held and abs(tilt) <= TILT_MAX and bottom >= LIFT_MIN)
        out = dict(success=success, held=bool(held), tilt_deg=round(tilt, 1), bottom_cm=round(100 * bottom, 1),
                   grasp_along=None if close is None else round(close["along"], 4))
        g = ""
        if close is not None and close["holding"]:
            a = close["along"]
            g = (f" You grasped the bar {100 * abs(a):.1f} cm to the {side_words(a)} of its midpoint."
                 if abs(a) >= 0.0005 else " You grasped the bar at its midpoint.")
        tilt_txt = (f"hangs tilted by {abs(tilt):.0f} degrees, its {side_words(tilt).split()[0]} end lower than its "
                    f"{side_words(-tilt).split()[0]} end")
        if success:
            out["failure"] = None
            detail = f"The bar is held level (tilt {abs(tilt):.0f} degrees), {100 * bottom:.0f} cm above the table." + g
        elif held and abs(tilt) > TILT_MAX:
            out["failure"] = "tilted"
            detail = ("The bar did not stay level: it rotated in the gripper while it was being lifted and now "
                      f"{tilt_txt}." + g)
            if bottom < LIFT_MIN:
                detail += f" Its lowest point is {100 * bottom:.0f} cm above the table."
        elif held:
            out["failure"] = "too_low"
            detail = (f"The bar is held level (tilt {abs(tilt):.0f} degrees) but its lowest point is only "
                      f"{100 * bottom:.0f} cm above the table." + g)
        elif lifted:
            out["failure"] = "dropped"
            detail = ("The bar was lifted but slipped out of the gripper and fell back onto the table "
                      f"(it now lies tilted by {abs(tilt):.0f} degrees)." + g)
        elif close is None:
            out["failure"] = "no_grasp"
            detail = "The gripper never closed on the bar; it is still lying on the table."
        else:
            out["failure"] = "no_grasp"
            detail = "The gripper closed but did not catch the bar; it is still lying on the table."
        out["detail"] = detail
        out["detail_indirect"] = detail.replace("bar", "rod")
        return out

    def feedback(self, inst, out, level="F2", indirect=False):
        if indirect and level == "F2":
            out = dict(out, detail=out.get("detail_indirect", out["detail"]))
        return super().feedback(inst, out, level)

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def execute(self, inst, params):
        """Grasp across the bar at `along` (m from the visible midpoint, + = left / +y) and lift 15 cm."""
        sk = self.sk
        c = envs.obj_pos(self.env, BAR)  # visible bar pose (perfect perception)
        ax = bar_axis(self.env)
        if ax[1] < 0:
            ax = -ax
        g = c + ax * float(params["along"])
        sk.set_gripper(False, steps=5)
        sk.move_to([g[0], g[1], Z_GRASP + 0.10], yaw=YAW_ACROSS)
        sk.move_to([g[0], g[1], Z_GRASP], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([g[0], g[1], Z_LIFT], speed=0.15, tol=0.004, max_steps=300)

    def default_params(self, inst):
        return dict(along=0.0)

    def oracle_params(self, inst):
        return dict(along=float(inst["com_offset"]))

    MARGIN = 0.008  # a tilt towards +y at grasp g means COM > g + MARGIN (window is ~+-1.5 cm)
    STEP = 0.030  # first step towards the heavy end when only one side is bounded

    def adapt_params(self, inst, history):
        """Bracketing on the grasp point: the end that sank is the heavy side of the last grasp."""
        lo, hi = -HALF[1], HALF[1]
        lo_set = hi_set = False
        for h in history:
            o = h["outcome"]
            g = o["grasp_along"] if o.get("grasp_along") is not None else h["params"]["along"]
            if o.get("failure") == "tilted" or (o.get("failure") == "dropped" and abs(o["tilt_deg"]) > 3):
                if o["tilt_deg"] > 0:  # left end sank -> COM left of the grasp
                    lo, lo_set = max(lo, g + self.MARGIN), True
                else:
                    hi, hi_set = min(hi, g - self.MARGIN), True
        last = history[-1]["params"]["along"]
        if lo_set and hi_set:
            a = 0.5 * (lo + hi)
        elif lo_set:
            a = lo + self.STEP - self.MARGIN
        elif hi_set:
            a = hi - self.STEP + self.MARGIN
        else:  # no direction information (e.g. too low): repeat
            a = last
        return dict(along=float(np.clip(a, -0.12, 0.12)))

    def blind_params(self, inst, attempt, rng):
        return dict(along=float(rng.normal(0.0, 0.01)))
