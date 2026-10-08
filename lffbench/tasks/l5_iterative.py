"""[PARKED 2026-10-06 — fails acceptance: oracle 50% (n=12, k=3). The push response is not stationary:
the same commanded speed carries the cube 7.9 cm on one push and 11.4 cm on the next, because the arm reaches
different speeds at different extensions. Needs a configuration-independent striker before use.]

L5-B iterative_push: several pushes in ONE episode (MemMimic "Iterative Pushing" analogue).

The cube starts far from the target zone; each push is a straight strike away from the robot whose
only free parameter is the speed. Overshooting past the zone is unrecoverable (episode fails).
Hidden variable: cube-table friction mu, as in l5_slide_to_target. The information needed to choose
the next push (how far a push of a given speed carries the cube) is only available from the
outcomes of the previous pushes in the same episode -> within-episode learning from failure.

"Attempt" granularity for the history: one push.
"""
import numpy as np

from .. import envs
from ..task_base import register_task
from .l5_slide import CUBE_HALF, V_MAX, V_MIN, Z_PUSH, SlideToTarget


@register_task
class IterativePush(SlideToTarget):
    name = "l5_iterative_push"
    level = "L5"
    capabilities = ("Reason", "Save", "Utilize")
    protocol = "within"
    max_attempts = 3  # pushes per episode: too few to creep up on the target with small taps
    instruction = ("Push the red cube away from the robot so that it comes to rest inside the green target zone. "
                   "You have at most three separate pushes. If the cube passes the zone, the task fails.")
    instruction_indirect = instruction
    tol_along = 0.015

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        return dict(seed=int(seed),
                    cube_xy=[float(rng.uniform(-0.27, -0.23)), float(rng.uniform(-0.08, 0.08))],
                    target_dist=float(rng.uniform(0.22, 0.30)),
                    mu=float(np.exp(rng.uniform(np.log(0.035), np.log(0.08)))))

    def make_bddl(self):
        return super().make_bddl()

    # --------------------------------------------------------------- one push from the current cube pose
    def push_once(self, inst, speed):
        c = envs.obj_pos(self.env, "cube_1")
        x, y = c[0], c[1]
        sk = self.sk
        sk.set_gripper(True, steps=3)
        sk.move_to([x - self.runup, y, Z_PUSH + 0.10], yaw=0.0)
        sk.move_to([x - self.runup, y, Z_PUSH], tol=0.003)
        sk.move_to([x - CUBE_HALF + 0.015, y, Z_PUSH], speed=float(np.clip(speed, V_MIN, V_MAX)), tol=0.01,
                   max_steps=80)
        sk.move_to([x - CUBE_HALF + 0.015, y, Z_PUSH + 0.10], speed=0.3)
        envs.settle(self.env, 40)
        c1 = envs.obj_pos(self.env, "cube_1")
        return float(c1[0] - x)

    def outcome(self, inst):
        x0, y0 = inst["cube_xy"]
        c = envs.obj_pos(self.env, "cube_1")
        along = (c[0] - x0) - inst["target_dist"]
        across = c[1] - y0
        success = bool(abs(along) <= self.tol_along and abs(across) <= self.tol_across and c[2] > envs.TABLE_Z)
        overshoot = bool(along > self.tol_along)
        if success:
            detail = "The cube is inside the target zone."
        elif overshoot:
            detail = f"The cube passed the target zone: it is {100 * along:.1f} cm beyond its centre. The task has failed."
        else:
            detail = f"The cube is {100 * -along:.1f} cm short of the centre of the target zone."
        return dict(success=success, overshoot=overshoot, along_err=float(along), across_err=float(across),
                    detail=detail)

    # --------------------------------------------------------------- scripted references
    S0 = 0.02  # distance the cube travels during the contact stroke itself

    def _plan_speed(self, remaining, model, pushes_left=1):
        """Model: push distance d(v) = s0 + k v^2. With more than one push left, cover all but a short
        final leg (a short push is more predictable); with one push left, aim at the zone centre."""
        if model is None:
            return 0.30
        s0, k = model
        goal = remaining if pushes_left <= 1 else max(remaining - 0.07, 0.0)
        v = np.sqrt(max(goal - s0, 0.0) / max(k, 1e-6))
        return float(np.clip(v, V_MIN, V_MAX))

    @classmethod
    def _fit(cls, pts):
        """pts: [(v, d)]. One point -> fix s0; two or more -> least squares on (s0, k)."""
        if len(pts) == 1:
            v, d = pts[0]
            return cls.S0, max(d - cls.S0, 1e-4) / v ** 2
        A = np.array([[1.0, v ** 2] for v, _ in pts])
        b = np.array([d for _, d in pts])
        s0, k = np.linalg.lstsq(A, b, rcond=None)[0]
        if k <= 0:
            return cls.S0, max(np.mean(b) - cls.S0, 1e-4) / np.mean([v ** 2 for v, _ in pts])
        return float(np.clip(s0, 0.0, 0.05)), float(k)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        self.reset_instance(inst)
        history = []
        model = None
        if kind == "oracle":
            model = self._oracle_model(inst)
        for a in range(k):
            out0 = self.outcome(inst)
            remaining = -out0["along_err"]
            if kind == "naive" or kind == "blind":
                speed = 0.30 if kind == "naive" else float(np.clip(0.30 + rng.normal(0, 0.05), V_MIN, V_MAX))
                if remaining < 0.06:  # naive heuristic: tap gently near the end
                    speed = 0.20
            else:
                speed = self._plan_speed(remaining, model, pushes_left=k - a)
            moved = self.push_once(inst, speed)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=dict(speed=speed), outcome=dict(out, moved=moved)))
            if kind == "adaptive" and moved > 0.005:
                pts = [(h["params"]["speed"], h["outcome"]["moved"]) for h in history if h["outcome"]["moved"] > 0.005]
                model = self._fit(pts)
            if out["success"] or out["overshoot"]:
                break
        # mark only the final state as the episode outcome
        return history

    def _oracle_model(self, inst):
        """Privileged: measure the push model with the true mu in scratch rollouts."""
        pts = []
        for v in (0.2, 0.35, 0.5):
            self.reset_instance(inst)
            pts.append((v, self.push_once(inst, v)))
        self.reset_instance(inst)
        return self._fit(pts)
