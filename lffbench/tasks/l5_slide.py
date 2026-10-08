"""L5-A slide_to_target: one push per attempt; the cube must come to rest inside the target zone.

Hidden variable: cube-table sliding friction mu (log-uniform in [0.035, 0.10]) -- the cube looks the
same for every mu. With the default push speed the cube overshoots on slippery instances and stops
short on grippy ones; the only way to get it right is to infer from where the cube stopped how hard
to push next time (cf. MemMimic push_cube, iterative pushing).

Rule (stated in the instruction and enforced, 2026-10-08): ONE strike. Only one move command may push the cube,
and after the gripper first touches it the gripper may advance at most ADV_MAX further; the cube has to slide
the rest on its own. Otherwise a slow push carries the cube to the target and friction never matters (seen with
Claude in exp1 and cla1: 0.05 m/s pushes, one of them in two segments).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

CUBE = "cube_1"
CUBE_HALF = 0.02
ADV_MAX = 0.04  # how far the gripper may keep advancing (+x) after it first touched the cube
MOVE_EPS = 0.002  # the cube counts as touched once it moved 2 mm (contact polling alone misses stick-slip)
BEHIND_HALF_W = 0.025  # the advance counts while the gripper centre is within cube half-width + this of the cube's y
PUSH_STEP = 0.0005  # horizontal gripper motion per control step that counts as pushing (1 cm/s at 20 Hz)
PUSH_NEAR = 0.05  # a step pushes the cube if the gripper is this close (centres) and moving towards it ...
ACC_EPS = 0.0002  # ... while the cube speeds up by this much per control step. A sliding cube only slows down
#                   (friction), so speeding up means the gripper drives it. Contact polling alone misses most fast
#                   strikes (measured: 0 contacts in several 0.2-0.6 m/s strikes); "cube moving near the gripper"
#                   alone also counted the lift right after a strike (measured).


class SlideSkills(Skills):
    """Skills that watch, every control step, how the cube is pushed: which move_to commands push it (the gripper
    moves towards the cube, close to it, while the cube speeds up) and how far the gripper keeps advancing after the
    first touch."""

    def __init__(self, env, bias=None, recorder=None):
        super().__init__(env, bias=bias, recorder=recorder)
        m = env.sim.model
        self._robot = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(("gripper0_", "robot0_"))]
        self._cube = envs.obj_geom_ids(env, CUBE)
        self._c0 = envs.obj_pos(env, CUBE)
        self._prev = self.eef_pos()
        self._cprev = self._c0
        self._cspeed = 0.0  # cube displacement in the last control step
        self._pushing = False
        self.touch_x = None  # gripper x at the first touch
        self.max_advance = 0.0  # furthest the gripper went in +x beyond touch_x
        self.pushes = []  # one entry per move_to command that pushed the cube

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        e = self.eef_pos()
        c = envs.obj_pos(self.env, CUBE)
        contact = envs.contacts_between(self.env, self._robot, self._cube)
        if self.touch_x is None and (contact or np.linalg.norm(c[:2] - self._c0[:2]) > MOVE_EPS):
            self.touch_x = float(e[0])
        if self.touch_x is not None and abs(float(e[1] - c[1])) < CUBE_HALF + BEHIND_HALF_W:
            # only while the gripper is still behind the cube (a later move beside it is not a push; measured with
            # Sonnet in cla2: strike, step aside 8 cm, move forward -> was wrongly counted as carrying)
            self.max_advance = max(self.max_advance, float(e[0]) - self.touch_x)
        step = e[:2] - self._prev[:2]
        cspeed = float(np.linalg.norm(c[:2] - self._cprev[:2]))
        toward = float(np.dot(step, c[:2] - e[:2])) > 0
        if (np.linalg.norm(step) > PUSH_STEP and toward and np.linalg.norm(c[:2] - e[:2]) < PUSH_NEAR
                and cspeed - self._cspeed > ACC_EPS):
            self._pushing = True
        self._prev, self._cprev, self._cspeed = e, c, cspeed

    def move_to(self, target, *args, **kwargs):
        self._pushing = False
        c0 = envs.obj_pos(self.env, CUBE)
        rep = super().move_to(target, *args, **kwargs)
        if self._pushing:
            self.pushes.append(dict(step=self.n_steps,
                                    cube_moved_cm=round(100 * float(np.linalg.norm(envs.obj_pos(self.env, CUBE)[:2] - c0[:2])), 1)))
        return rep
INNER_HALF = 0.03  # 6 cm square target; the 4 cm cube must end completely inside it (centre within +-1 cm)
LINE_W = 0.006


def square_frame_xml(model_name, inner_half=INNER_HALF, line_w=LINE_W,
                     line_rgba=(0.1, 0.55, 0.15, 1.0), fill_rgba=(0.75, 0.92, 0.75, 1.0)):
    """Flat, visual-only square drawn on the table: a light fill plus a dark outline (no collisions)."""
    o = inner_half + line_w
    parts = [((0, 0, 0.0004), (o, o, 0.0004), fill_rgba),
             ((0, inner_half + line_w / 2, 0.0010), (o, line_w / 2, 0.0002), line_rgba),
             ((0, -inner_half - line_w / 2, 0.0010), (o, line_w / 2, 0.0002), line_rgba),
             ((inner_half + line_w / 2, 0, 0.0010), (line_w / 2, o, 0.0002), line_rgba),
             ((-inner_half - line_w / 2, 0, 0.0010), (line_w / 2, o, 0.0002), line_rgba)]
    g = "\n".join(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(c)}" '
                  f'contype="0" conaffinity="0" group="1" />' for p, sz, c in parts)
    return _wrap(model_name, g, 0.0012, o)
Z_PUSH = envs.TABLE_Z + 0.02
V_MIN, V_MAX = 0.12, 0.60


@register_task
class SlideToTarget(LFFTask):
    name = "l5_slide_to_target"
    level = "L5"
    category = "dynamics_adaptation"
    capabilities = ("Reason", "Utilize")
    instruction = ("Push the red cube once, straight away from the robot, so that it slides and comes to rest "
                   "completely inside the green square marked on the table. Strike it with a single move command: "
                   f"after the gripper first touches the cube it may move at most {100 * ADV_MAX:.0f} cm further "
                   "forward, and the cube must slide the rest of the way on its own.")
    instruction_indirect = instruction
    tol_along = INNER_HALF - CUBE_HALF  # cube completely inside the square
    runup = 0.22  # long run-up so the gripper reaches the commanded speed before contact
    tol_across = INNER_HALF - CUBE_HALF

    def make_bddl(self):
        cube = register_generated("LffRedCube", box_xml("lff_red_cube", (CUBE_HALF,) * 3, (0.85, 0.15, 0.15, 1),
                                                        density=1000))
        # static fixture (no joint): a free body without collisions would fall through the table
        zone = register_generated("LffTargetSquare", square_frame_xml("lff_target_square"), free=False)
        objs = [("cube_1", cube, "cube_region", (-0.21, -0.01, -0.19, 0.01))]
        fixtures = [("zone_1", zone, "zone_region", (-0.11, -0.01, -0.09, 0.01))]
        return write_bddl(self.name, self.instruction, objs, fixtures=fixtures)

    def apply_instance(self, inst):
        m = self.env.sim.model
        zone = self.env.fixtures_dict["zone_1"]
        # the target zone is a flat marker: no collisions
        for g in range(m.ngeom):
            if (m.geom_id2name(g) or "").startswith("zone_1_"):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        x0, y0 = inst["cube_xy"]
        envs.set_obj_pose(self.env, "cube_1", [x0, y0, envs.TABLE_Z + CUBE_HALF + 0.001])
        bid = m.body_name2id(zone.root_body)
        m.body_pos[bid] = [x0 + inst["target_dist"], y0, envs.TABLE_Z + 0.0002]
        m.body_quat[bid] = [1.0, 0.0, 0.0, 0.0]
        envs.set_friction(self.env, "cube_1", inst["mu"])

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = SlideSkills(self.env, recorder=recorder)
        return self.sk

    @staticmethod
    def max_reach(mu):
        """Approximate slide distance at the top speed, fitted to a sweep (runup 0.22 m):
        mu 0.04 -> 20.8 cm, 0.06 -> 14.8, 0.08 -> 11.8, 0.10 -> 10.1 cm."""
        return 0.0103 / mu ** 0.95

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        while True:  # reject instances the top speed cannot reach (keeps every instance solvable)
            inst = dict(seed=int(seed),
                        cube_xy=[float(rng.uniform(-0.23, -0.17)), float(rng.uniform(-0.10, 0.10))],
                        target_dist=float(rng.uniform(0.05, 0.12)),
                        mu=float(np.exp(rng.uniform(np.log(0.035), np.log(0.10)))))
            if inst["target_dist"] <= 0.85 * self.max_reach(inst["mu"]):
                return inst

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 40)
        x0, y0 = inst["cube_xy"]
        c = envs.obj_pos(self.env, "cube_1")
        along = (c[0] - x0) - inst["target_dist"]
        across = c[1] - y0
        on_table = c[2] > envs.TABLE_Z
        inside = bool(abs(along) <= self.tol_along and abs(across) <= self.tol_across and on_table)
        sk = self.sk
        n_push = len(getattr(sk, "pushes", []))
        adv = float(getattr(sk, "max_advance", 0.0))
        carried = adv > ADV_MAX
        success = bool(inside and n_push <= 1 and not carried)
        rule = []
        if n_push > 1:
            rule.append(f"The cube was pushed by {n_push} separate move commands; only one strike is allowed.")
        if carried:
            rule.append(f"After first touching the cube the gripper kept moving {100 * adv:.1f} cm further forward, "
                        f"more than the allowed {100 * ADV_MAX:.0f} cm: the cube was carried instead of sliding on its own.")
        where = ("completely inside the square" if inside else "not completely inside the square")
        if along > 0:
            detail = f"The cube stopped {100 * along:.1f} cm past the centre of the green square ({where})."
        else:
            detail = f"The cube stopped {100 * -along:.1f} cm short of the centre of the green square ({where})."
        if abs(across) > self.tol_across:
            detail += f" It also drifted {100 * abs(across):.1f} cm to the {'left' if across > 0 else 'right'} (robot's view)."
        if rule:
            detail = " ".join(rule) + " " + detail
        out = dict(success=success, along_err=float(along), across_err=float(across), detail=detail,
                   inside=inside, n_pushes=n_push, max_advance=round(adv, 4))
        if not success:
            out["failure"] = ("several_pushes" if n_push > 1 else "carried" if carried
                              else "not_touched" if sk.touch_x is None else "position")
        return out

    # ------------------------------------------------------------ execution
    def execute(self, inst, params):
        x0, y0 = inst["cube_xy"]
        sk = self.sk
        sk.set_gripper(True, steps=5)
        sk.move_to([x0 - self.runup, y0, Z_PUSH + 0.10], yaw=0.0)
        sk.move_to([x0 - self.runup, y0, Z_PUSH], tol=0.003)
        sk.move_to([x0 - CUBE_HALF + 0.015, y0, Z_PUSH], speed=float(np.clip(params["speed"], V_MIN, V_MAX)),
                   tol=0.01, max_steps=80)
        sk.move_to([x0 - CUBE_HALF + 0.015, y0, Z_PUSH + 0.08], speed=0.3)

    def default_params(self, inst):
        return dict(speed=0.30)

    def blind_params(self, inst, attempt, rng):
        return dict(speed=float(np.clip(0.30 + rng.normal(0, 0.05), V_MIN, V_MAX)))

    def adapt_params(self, inst, history):
        """Bracketing secant search on speed using the signed stopping error."""
        pts = sorted((h["params"]["speed"], h["outcome"]["along_err"]) for h in history)
        lo = [p for p in pts if p[1] < 0]
        hi = [p for p in pts if p[1] > 0]
        if lo and hi:
            (v0, e0), (v1, e1) = max(lo, key=lambda p: p[0]), min(hi, key=lambda p: p[0])
            v = v0 + (v1 - v0) * (-e0) / (e1 - e0)
        elif hi:  # always overshooting: slow down
            v = min(p[0] for p in hi) * 0.7
        else:  # always short: speed up
            v = max(p[0] for p in lo) * 1.35
        return dict(speed=float(np.clip(v, V_MIN, V_MAX)))

    def oracle_params(self, inst):
        """Privileged: bisection on speed by simulating pushes (uses the true mu)."""
        if "_oracle_speed" in inst:
            return dict(speed=inst["_oracle_speed"])
        lo, hi = V_MIN, V_MAX
        best = None
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            self.reset_instance(inst)
            self.execute(inst, dict(speed=mid))
            e = self.outcome(inst)["along_err"]
            if best is None or abs(e) < abs(best[1]):
                best = (mid, e)
            if e > 0:
                hi = mid
            else:
                lo = mid
        inst["_oracle_speed"] = best[0]
        self.reset_instance(inst)
        return dict(speed=best[0])
