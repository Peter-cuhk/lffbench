"""Motion primitives executed with robosuite's OSC_POSE controller.

All positions are world-frame metres (LIBERO table top is z = TABLE_Z = 0.90).
`bias` is a hidden calibration error added to every commanded position (used by L1 tasks);
agents never see it.
"""
import numpy as np
import robosuite.utils.transform_utils as T

from .envs import TABLE_Z

DT = 0.05  # control period (20 Hz)
ROT_SCALE = 0.5
# Faster OSC than LIBERO's default (default caps the end effector at ~0.25 m/s; this reaches ~0.6 m/s).
CONTROLLER = dict(kp=300, output_max=[0.1, 0.1, 0.1, 0.5, 0.5, 0.5], output_min=[-0.1, -0.1, -0.1, -0.5, -0.5, -0.5])


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_err(R_target, R_cur):
    """axis-angle (world frame) rotating R_cur to R_target"""
    R = R_target @ R_cur.T
    q = T.mat2quat(R)  # xyzw
    return T.quat2axisangle(q)


class Skills:
    def __init__(self, env, bias=None, recorder=None):
        self.env = env
        self.pos_scale = float(env.robots[0].controller.output_max[0])
        self.bias = np.zeros(3) if bias is None else np.asarray(bias, float)
        self.grip = -1.0  # -1 open, +1 closed
        self.R0 = self.eef_mat().copy()  # top-down orientation at reset == yaw 0
        self.yaw = 0.0
        self.recorder = recorder  # callable(env) invoked every `record_every` control steps
        self.record_every = 4
        self.n_steps = 0
        self.log = []

    # ------------------------------------------------------------------ state
    def eef_pos(self):
        return self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id].copy()

    def eef_mat(self):
        return self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id].reshape(3, 3).copy()

    def gripper_width(self):
        g = self.env.robots[0].gripper
        q = [self.env.sim.data.qpos[self.env.sim.model.joint_name2id(j)] for j in g.joints]
        return float(abs(q[0] - q[1]))

    def holding(self, obj_name):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper,
                                          object_geoms=self.env.objects_dict[obj_name]))

    # ------------------------------------------------------------------ low level
    def _act(self, dpos, drot, grip):
        a = np.zeros(7)
        a[:3] = np.clip(dpos / self.pos_scale, -1, 1)
        a[3:6] = np.clip(drot / ROT_SCALE, -1, 1)
        a[6] = grip
        self.env.step(a)
        self.n_steps += 1
        if self.recorder is not None and self.n_steps % self.record_every == 0:
            self.recorder(self.env)

    def hold(self, steps=10):
        p, R = self.eef_pos(), self._R()
        for _ in range(steps):
            self._act(p - self.eef_pos(), _rot_err(R, self.eef_mat()), self.grip)

    def _R(self, yaw=None):
        return _rotz(self.yaw if yaw is None else yaw) @ self.R0

    # ------------------------------------------------------------------ primitives
    YAW_LIMIT = np.pi / 2 + 0.4

    def wrap_yaw(self, yaw):
        """A parallel-jaw gripper is symmetric under a rotation by pi, so pick the equivalent yaw closest
        to the current one inside |yaw| <= pi/2 + 0.4. Without this, asking for ~180 deg of wrist rotation
        makes the OSC (whose axis-angle error is ambiguous at pi) wind the arm into its joint limits and
        it never recovers (found by the L3 developer: push towards -y -> 0/6 success, 2026-10-06)."""
        cands = [yaw + k * np.pi for k in range(-3, 4)]
        ok = [c for c in cands if abs(c) <= self.YAW_LIMIT] or cands
        return float(min(ok, key=lambda c: abs(c - self.yaw)))

    def move_to(self, target, yaw=None, speed=None, tol=0.004, max_steps=250, apply_bias=True, wrap_yaw=True):
        """Move the gripper site to `target` (world xyz). If `speed` (m/s) is given the reference
        moves along a straight line at that speed; otherwise go as fast as the controller allows.
        `yaw` is wrapped to the closest gripper-symmetric equivalent unless wrap_yaw=False.
        Returns a report dict."""
        tgt = np.asarray(target, float) + (self.bias if apply_bias else 0.0)
        if yaw is not None:
            self.yaw = self.wrap_yaw(float(yaw)) if wrap_yaw else float(yaw)
        R = self._R()
        p0 = self.eef_pos()
        dist = np.linalg.norm(tgt - p0)
        direction = (tgt - p0) / max(dist, 1e-9)
        s = 0.0
        for k in range(max_steps):
            if speed is not None:
                s = min(dist, s + speed * DT)
                ref = p0 + direction * s
                # lead the reference a little so the OSC lag does not throttle the speed
                ref = ref + direction * min(dist - s, speed * DT * 1.5)
            else:
                ref = tgt
            cur = self.eef_pos()
            self._act(ref - cur, _rot_err(R, self.eef_mat()), self.grip)
            err = np.linalg.norm(tgt - self.eef_pos())
            if err < tol and (speed is None or s >= dist):
                break
        rep = dict(prim="move_to", reached=bool(np.linalg.norm(tgt - self.eef_pos()) < 0.02),
                   pos=self.eef_pos().round(4).tolist(), steps=k + 1)
        self.log.append(rep)
        return rep

    def set_gripper(self, close, steps=15):
        self.grip = 1.0 if close else -1.0
        self.hold(steps)
        rep = dict(prim="close" if close else "open", width=round(self.gripper_width(), 4))
        self.log.append(rep)
        return rep

    def grasp_at(self, xy, z_grasp, yaw=0.0, approach=0.10, lift=0.15):
        """Composite: above -> descend -> close -> lift. For scripted policies."""
        self.set_gripper(False, steps=5)
        self.move_to([xy[0], xy[1], z_grasp + approach], yaw=yaw)
        self.move_to([xy[0], xy[1], z_grasp], tol=0.003)
        self.set_gripper(True, steps=15)
        self.move_to([xy[0], xy[1], z_grasp + lift], speed=0.25)

    def place_at(self, xy, z_release, approach=0.10):
        self.move_to([xy[0], xy[1], z_release + approach])
        self.move_to([xy[0], xy[1], z_release], tol=0.004)
        self.set_gripper(False, steps=12)
        self.move_to([xy[0], xy[1], z_release + approach])

    def push(self, start_xy, direction_xy, distance, speed, z=None, pre=0.06):
        """Straight-line push: go to `pre` m behind the start point at height z with the gripper closed,
        then sweep `pre + distance` along direction at `speed`, stop, and lift away."""
        d = np.asarray(direction_xy, float)
        d = d / np.linalg.norm(d)
        z = TABLE_Z + 0.02 if z is None else z
        s = np.asarray(start_xy, float) - d * pre
        self.set_gripper(True, steps=5)
        self.move_to([s[0], s[1], z + 0.12], yaw=np.arctan2(d[1], d[0]) - np.pi / 2)
        self.move_to([s[0], s[1], z], tol=0.003)
        e = s + d * (pre + distance)
        rep = self.move_to([e[0], e[1], z], speed=speed, tol=0.01, max_steps=400)
        self.hold(2)
        self.move_to([e[0], e[1], z + 0.12], speed=0.3)
        return rep
