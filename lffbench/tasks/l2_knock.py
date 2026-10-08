"""L2-A knock_restore: move the ketchup from the tray into the basket; everything else must stay on the tray.

Constructed L2 event: the first time the ketchup is lifted off the tray (grasped and raised 3 cm), one of
its two neighbours on the tray is "knocked" off the tray and lands upright on the table beside it. Which
neighbour and to which side of the tray (left / right in the camera image) it falls are hidden per-instance
variables; only the scene *after* the failure shows which item is missing and where it is. The main task itself still succeeds, so an agent that stops after
the main task fails; it has to notice the side effect and put the item back (within-episode recovery).

Protocols
  within (default, `l2_knock_restore`)  one episode; attempt 1 = the main task, every later attempt is one
      corrective round chosen by the policy; the outcome (success + F2 text) is computed after each round.
  cross (`l2_knock_restore_cross`)      reset to the same instance between attempts; the knock happens again
      on every attempt, so a learner must anticipate it. Validated strategy: restore the reported item
      right after the main task. The rule also allows prevention (the knock only hits a neighbour that is
      still within 12 cm of the ketchup's start spot), but picking a neighbour up in place is mostly blocked
      by the tray's end wall (Panda hand), so prevention is not a validated strategy (see the task card).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..skills import Skills
from ..task_base import LFFTask, register_task

TARGET = "ketchup_1"
NEIGHBOR_POOL = ("cream_cheese_1", "butter_1", "chocolate_pudding_1")
TRAY, BASKET = "wooden_tray_1", "basket_1"
PARK_XY = {"cream_cheese_1": (0.0, 1.2), "butter_1": (0.2, 1.2), "chocolate_pudding_1": (-0.2, 1.2)}

# wooden_tray collision geometry in its body frame (from the LIBERO MJCF; long axis = body x)
TRAY_IN_X = (-0.128, 0.143)
TRAY_IN_Y = (-0.081, 0.072)
TRAY_OUT_X = (-0.142, 0.156)
TRAY_OUT_Y = (-0.088, 0.080)
TRAY_CX, TRAY_CY = 0.0075, -0.0045  # centre of the inner floor
TRAY_FLOOR = 0.0072  # floor top above the tray's lowest point
BASKET_IN = 0.060  # half size of the basket's inner floor (square, centred on the body origin)
BASKET_RIM = 0.150  # rim height above the basket's lowest point
BASKET_FLOOR = 0.017  # floor top above the basket's lowest point

NAMES = {"ketchup_1": "ketchup", "cream_cheese_1": "cream cheese", "butter_1": "butter",
         "chocolate_pudding_1": "chocolate pudding"}
DESCR = {"ketchup_1": "red bottle", "cream_cheese_1": "blue box", "butter_1": "small orange box",
         "chocolate_pudding_1": "brown box"}
SIDES = ("left", "right")  # side of the tray in the agentview image: left = world -y, right = world +y


def _qz(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _wrap_half(a):
    """wrap an angle to [-pi/2, pi/2) (boxes and parallel grippers are symmetric under 180 deg)"""
    return (a + np.pi / 2) % np.pi - np.pi / 2


class KnockSkills(Skills):
    """Skills with a hook called after every control step (used for the failure injection)."""

    def __init__(self, env, bias=None, recorder=None, hook=None):
        super().__init__(env, bias=bias, recorder=recorder)
        self.hook = hook

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        if self.hook is not None:
            self.hook()


@register_task
class KnockRestore(LFFTask):
    name = "l2_knock_restore"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Plan")
    protocol = "within"
    instruction = "Move the ketchup from the tray into the basket. Everything else must stay on the tray."
    instruction_indirect = ("Move the red bottle from the wooden tray into the wicker basket. "
                            "Everything else must stay on the tray.")
    knock_lift = 0.03  # trigger: ketchup grasped and lifted this far above its resting height
    knock_radius = 0.12  # only a victim still within this distance of the ketchup's start spot is knocked

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        # upright orientations (wxyz), fixed here so the task does not depend on LIBERO's placement convention:
        # the ketchup mesh is modelled lying down; tray / basket / boxes are upright in their mesh frame
        # (identity => tray long axis along world x, as the TRAY_* constants assume).
        self.q_up = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in self.env.objects_dict}
        self.q_up[TARGET] = np.array([0.5, 0.5, 0.5, 0.5])
        self._inj = None
        self.start = {}

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        names = [TARGET, *NEIGHBOR_POOL, TRAY, BASKET]
        objs = []
        for i, n in enumerate(names):  # layout is overridden in apply_instance; regions only need to be valid
            x0 = -0.30 + 0.13 * (i % 3)
            y0 = -0.25 + 0.25 * (i // 3)
            objs.append((n, n[:-2], f"r_{i}", (x0, y0, x0 + 0.02, y0 + 0.02)))
        return write_bddl(self.name, self.instruction, objs)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        nb = [str(x) for x in rng.choice(NEIGHBOR_POOL, size=2, replace=False)]  # [robot side, far side]
        victim = int(rng.integers(2))
        side = str(rng.choice(SIDES))
        return dict(seed=int(seed),
                    tray_xy=[float(rng.uniform(-0.07, 0.01)), float(rng.uniform(-0.11, -0.08))],
                    tray_yaw=float(rng.uniform(-0.10, 0.10)),
                    basket_xy=[float(rng.uniform(-0.08, 0.02)), float(rng.uniform(0.24, 0.26))],
                    target_dx=float(rng.uniform(-0.01, 0.01)),
                    target_yaw=float(np.pi / 2 + rng.uniform(-0.2, 0.2)),
                    neighbors=nb,
                    nb_gap=[float(rng.uniform(0.066, 0.078)) for _ in range(2)],
                    nb_dy=[float(rng.uniform(-0.012, 0.012)) for _ in range(2)],
                    nb_yaw=[float(np.pi / 2 + rng.uniform(-0.15, 0.15)) for _ in range(2)],
                    # hidden variables of the injected failure
                    victim=nb[victim], side=side,
                    land_gap=float(rng.uniform(0.015, 0.035 if side == "right" else 0.04)),  # wall -> object
                    land_off=float(rng.uniform(-0.025, 0.025)),  # along the wall, from the victim's spot
                    land_dyaw=float(rng.uniform(-0.35, 0.35)))

    def _tray_frame(self):
        p = envs.obj_pos(self.env, TRAY)
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, TRAY)].reshape(3, 3)
        return p, R

    def tray_yaw(self):
        _, R = self._tray_frame()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def tray_to_world(self, lx, ly):
        p, R = self._tray_frame()
        return p[:2] + R[:2, :2] @ np.array([lx, ly])

    def world_to_tray(self, xy):
        p, R = self._tray_frame()
        return R[:2, :2].T @ (np.asarray(xy[:2]) - p[:2])

    def place(self, name, xy, yaw, surface_z=envs.TABLE_Z):
        """Teleport an object upright (yaw about world z) with its lowest point 2 mm above surface_z."""
        q = envs._quat_mul(_qz(yaw), self.q_up[name])
        envs.set_obj_pose(self.env, name, [xy[0], xy[1], surface_z + 0.3], quat_wxyz=q)
        self.env.sim.forward()
        dz = envs.obj_min_z(self.env, name) - (surface_z + 0.002)
        qa, _ = envs.free_joint_addr(self.env, name)
        self.env.sim.data.qpos[qa + 2] -= dz
        self.env.sim.forward()

    def apply_instance(self, inst):
        env = self.env
        envs.scale_mass(env, TRAY, 2.0)  # heavy enough not to be dragged around by the gripper
        envs.scale_mass(env, BASKET, 2.0)
        for n in NEIGHBOR_POOL:
            if n not in inst["neighbors"]:
                self.place(n, PARK_XY[n], 0.0, surface_z=0.0)  # unused neighbour: on the floor, out of view
        self.place(TRAY, inst["tray_xy"], inst["tray_yaw"])
        self.place(BASKET, inst["basket_xy"], 0.0)
        floor = envs.TABLE_Z + TRAY_FLOOR
        ty = inst["tray_yaw"]
        cx = TRAY_CX + inst["target_dx"]
        self.place(TARGET, self.tray_to_world(cx, TRAY_CY), ty + inst["target_yaw"], floor)
        for i, n in enumerate(inst["neighbors"]):  # in a row along the tray: [robot side, ketchup, far side]
            lx = cx + (-1 if i == 0 else 1) * inst["nb_gap"][i]
            self.place(n, self.tray_to_world(lx, TRAY_CY + inst["nb_dy"][i]), ty + inst["nb_yaw"][i], floor)
        env.sim.forward()

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.start = {n: self.footprint(n) for n in [TARGET] + inst["neighbors"]}
        self._inj = dict(inst=inst, done=False, knocked=False, step=None)
        self.sk = KnockSkills(self.env, bias=self.skill_bias(inst), recorder=recorder, hook=self._knock_hook)
        return self.sk

    # ------------------------------------------------------------ measurements
    def footprint(self, name):
        """Measured pose summary of an object: AABB centre, yaw of the long horizontal axis, half sizes, z range.
        Scripted policies use it as a stand-in for perception."""
        m, d = self.env.sim.model, self.env.sim.data
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        best = None
        for g in envs.obj_geom_ids(self.env, name):
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            e = np.abs(R) @ m.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
            vol = float(np.prod(m.geom_aabb[g, 3:]))
            if best is None or vol > best[0]:
                best = (vol, R, m.geom_aabb[g, 3:])
        _, R, half = best
        horiz = sorted([(half[k], R[:, k]) for k in range(3) if abs(R[2, k]) < 0.5], key=lambda t: -t[0])
        ax = horiz[0][1] if horiz else np.array([1.0, 0.0, 0.0])
        return dict(center=[float(v) for v in (lo + hi) / 2], yaw=float(np.arctan2(ax[1], ax[0])),
                    half_xy=[float(v) for v in ((hi - lo) / 2)[:2]], zmin=float(lo[2]), zmax=float(hi[2]))

    def _corners_tray(self, name):
        """All collision-box corners of `name` in tray coordinates."""
        m, d = self.env.sim.model, self.env.sim.data
        pts = []
        for g in envs.obj_geom_ids(self.env, name):
            Rg = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + Rg @ m.geom_aabb[g, :3]
            h = m.geom_aabb[g, 3:]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(self.world_to_tray(c + Rg @ (h * np.array([sx, sy, sz]))))
        return np.array(pts)

    # ------------------------------------------------------------ failure injection
    def _knock_hook(self):
        st = self._inj
        if st is None or st["done"] or self.sk is None:
            return
        if envs.obj_min_z(self.env, TARGET) < self.start[TARGET]["zmin"] + self.knock_lift:
            return
        if not self.sk.holding(TARGET):
            return
        st["done"] = True
        inst = st["inst"]
        vxy = np.array(self.footprint(inst["victim"])["center"][:2])
        if np.linalg.norm(vxy - np.array(self.start[TARGET]["center"][:2])) > self.knock_radius:
            return  # the victim was moved out of the way beforehand: nothing happens
        self.knock(inst)
        st["knocked"] = True
        st["step"] = self.sk.n_steps

    def knock(self, inst):
        """Teleport the victim off the tray onto the table beside the long wall on `side`: upright, long axis
        roughly perpendicular to the wall (so a gripper closing along the wall can pick it up), nearest point
        `land_gap` from the wall's outer face, `land_off` along the wall from where it stood.
        (The robot-side end of the tray was tried too: the end wall hides a flat box there from agentview.)"""
        v, side = inst["victim"], inst["side"]
        loc = self.world_to_tray(self.footprint(v)["center"])
        axis, sgn = 1, (1 if side == "right" else -1)
        wall = TRAY_OUT_Y[1] if sgn > 0 else TRAY_OUT_Y[0]
        self.place(v, self.tray_to_world(loc[0] + inst["land_off"], wall + sgn * 0.08),
                   self.tray_yaw() + np.pi / 2 + inst["land_dyaw"])
        pts = self._corners_tray(v)
        near = pts[:, axis].min() if sgn > 0 else pts[:, axis].max()
        shift = (wall + sgn * inst["land_gap"]) - near
        _, R = self._tray_frame()
        qa, _ = envs.free_joint_addr(self.env, v)
        self.env.sim.data.qpos[qa:qa + 2] += R[:2, axis] * shift
        self.env.sim.forward()

    # ------------------------------------------------------------ outcome
    def where(self, name):
        """Classify where an object is: 'tray', 'basket', 'gripper', 'table' or 'other'."""
        f = self.footprint(name)
        c = np.array(f["center"])
        lt = self.world_to_tray(c)
        if self.sk is not None and self.sk.holding(name) and f["zmin"] > envs.TABLE_Z + 0.02:
            return "gripper", f
        if (TRAY_IN_X[0] - 0.005 <= lt[0] <= TRAY_IN_X[1] + 0.005 and TRAY_IN_Y[0] - 0.005 <= lt[1] <= TRAY_IN_Y[1] + 0.005
                and f["zmin"] > envs.obj_min_z(self.env, TRAY) + 0.002):
            return "tray", f
        bp = envs.obj_pos(self.env, BASKET)
        if np.all(np.abs(c[:2] - bp[:2]) <= BASKET_IN + 0.01) and f["zmin"] < envs.obj_min_z(self.env, BASKET) + BASKET_RIM:
            return "basket", f
        if f["zmin"] < envs.TABLE_Z + 0.01:
            return "table", f
        return "other", f

    def rel_to_tray(self, c):
        """Where a point on the table is relative to the tray, in robot-centric words (never image left/right).
        Distances are from the tray's outer edge to the object's centre."""
        lt = self.world_to_tray(c)
        dy = lt[1] - TRAY_OUT_Y[1] if lt[1] > TRAY_OUT_Y[1] else (lt[1] - TRAY_OUT_Y[0] if lt[1] < TRAY_OUT_Y[0] else 0.0)
        dx = lt[0] - TRAY_OUT_X[1] if lt[0] > TRAY_OUT_X[1] else (lt[0] - TRAY_OUT_X[0] if lt[0] < TRAY_OUT_X[0] else 0.0)
        # robot-centric words (tray yaw is within +-0.1 rad, so tray axes ~ world axes): +y = robot's left,
        # +x = forward (away from the robot)
        parts = []
        if dy:
            parts.append(f"{100 * abs(dy):.0f} cm to the robot's {'left (+y)' if dy > 0 else 'right (-y)'} of the tray")
        if dx:
            parts.append(f"{100 * abs(dx):.0f} cm {'forward (+x, away from the robot)' if dx > 0 else 'backward (-x, toward the robot)'} "
                         f"of the tray")
        return " and ".join(parts) if parts else "right next to the tray"

    def _describe(self, inst, loc, names):
        t_loc = loc[TARGET][0]
        tn = names[TARGET]
        msgs = []
        if t_loc != "basket":
            msgs.append({"gripper": f"The {tn} is still in the gripper, not in the basket.",
                         "tray": f"The {tn} is still on the tray, not in the basket.",
                         "table": f"The {tn} is on the table ({self.rel_to_tray(loc[TARGET][1]['center'])}), "
                                  "not in the basket."}.get(t_loc, f"The {tn} is not in the basket."))
        for n in inst["neighbors"]:
            l, f = loc[n]
            if l == "tray":
                continue
            msgs.append({"table": f"The {names[n]} is no longer on the tray: it is on the table "
                                  f"{self.rel_to_tray(f['center'])}.",
                         "basket": f"The {names[n]} is no longer on the tray: it is in the basket.",
                         "gripper": f"The {names[n]} is no longer on the tray: it is still in the gripper."}
                        .get(l, f"The {names[n]} is no longer on the tray."))
        if not msgs:
            msgs.append(f"The {tn} is in the basket and everything else is on the tray.")
        return " ".join(msgs)

    def outcome(self, inst):
        envs.settle(self.env, 30)
        loc = {n: self.where(n) for n in [TARGET] + inst["neighbors"]}
        off = [n for n in inst["neighbors"] if loc[n][0] != "tray"]
        success = bool(loc[TARGET][0] == "basket" and not off)
        return dict(success=success,
                    target_loc=loc[TARGET][0],
                    locations={n: l for n, (l, _) in loc.items()},
                    off_tray=off,
                    footprints={n: f for n, (_, f) in loc.items()},
                    knocked=bool(self._inj and self._inj["knocked"]),
                    detail=self._describe(inst, loc, NAMES),
                    detail_indirect=self._describe(inst, loc, DESCR))

    def feedback(self, inst, out, level="F2", indirect=False):
        """F0: nothing, F1: success flag, F2: which item is where. `indirect` uses descriptive names."""
        if indirect and level == "F2":
            out = dict(out, detail=out.get("detail_indirect", out["detail"]))
        return super().feedback(inst, out, level)

    # ------------------------------------------------------------ scripted motions
    def pick_place(self, name, dest_xy, support_z, dest_yaw=None, carry_z=1.10, clearance=0.012, fp=None,
                   travel_z=1.15):
        """Top-down pick of `name` at its (measured) footprint, carry at eef height `carry_z`, release with the
        object bottom `clearance` above `support_z` at dest_xy. Moves between places at `travel_z` (above the
        basket rim) so the arm does not sweep through the scene."""
        sk = self.sk
        f = fp or self.footprint(name)
        c = np.array(f["center"])
        h = f["zmax"] - f["zmin"]
        gyaw = _wrap_half(f["yaw"])  # fingers close along eef y (= world y at yaw 0): across the short side
        if name == TARGET:
            zg = f["zmax"] - 0.035
        else:
            zg = max(f["zmin"] + 0.5 * h, f["zmin"] + 0.0145)  # fingertips reach ~1.3 cm below the grip site
        sk.set_gripper(False, steps=6)
        p = sk.eef_pos()
        if p[2] < travel_z - 0.02:
            sk.move_to([p[0], p[1], travel_z])
        sk.move_to([c[0], c[1], travel_z], yaw=gyaw)
        sk.move_to([c[0], c[1], f["zmax"] + 0.03], speed=0.4)
        sk.move_to([c[0], c[1], zg], tol=0.003, max_steps=80)
        sk.set_gripper(True, steps=12)
        sk.move_to([c[0], c[1], carry_z], speed=0.3)
        gyaw2 = gyaw if dest_yaw is None else gyaw + _wrap_half(dest_yaw - f["yaw"])
        sk.move_to([dest_xy[0], dest_xy[1], carry_z], yaw=gyaw2)
        sk.move_to([dest_xy[0], dest_xy[1], support_z + (zg - f["zmin"]) + clearance], tol=0.004, speed=0.25,
                   max_steps=120)
        sk.set_gripper(False, steps=10)
        sk.move_to([dest_xy[0], dest_xy[1], travel_z], speed=0.4)

    def main_task(self, fp=None):
        """Ketchup -> basket (the instruction, as an agent without failure knowledge would do it)."""
        b = envs.obj_pos(self.env, BASKET)
        floor = envs.obj_min_z(self.env, BASKET) + BASKET_FLOOR
        self.pick_place(TARGET, b[:2], floor, carry_z=1.27, clearance=0.006, fp=fp)

    def restore_spot(self, name, inst):
        """A free spot on the tray for `name`: the ketchup's start spot or `name`'s own start spot, whichever
        is further from everything else currently on the tray."""
        others = [n for n in [TARGET] + inst["neighbors"] if n != name and self.where(n)[0] == "tray"]
        best, best_d = None, -1.0
        for cand in (TARGET, name):
            xy = np.array(self.start[cand]["center"][:2])
            dmin = min([np.linalg.norm(xy - np.array(self.footprint(o)["center"][:2])) for o in others] + [1.0])
            if dmin > best_d + 0.01:
                best, best_d = xy, dmin
        return best

    def restore(self, name, inst):
        """Put `name` back onto the tray (long axis across the tray, fingers closing along the tray)."""
        xy = self.restore_spot(name, inst)
        self.pick_place(name, xy, envs.TABLE_Z + TRAY_FLOOR, dest_yaw=self.tray_yaw() + np.pi / 2, carry_z=1.08,
                        clearance=0.013)

    def do(self, actions, inst):
        for act in actions:
            if act[0] == "ketchup_to_basket":
                self.main_task(fp=act[1] if len(act) > 1 else None)
            elif act[0] == "restore":
                self.restore(act[1], inst)
            else:
                raise ValueError(act)

    # ------------------------------------------------------------ scripted reference policies
    def _plan(self, kind, inst, a, history, rng):
        """Actions of one attempt. JSON-friendly lists: ["ketchup_to_basket"] / ["restore", name]."""
        if kind == "naive" or (a == 0 and kind in ("adaptive", "blind")):
            return [["ketchup_to_basket"]]
        if kind == "oracle":  # knows the victim: restores it right after the main task, without feedback
            return [["ketchup_to_basket"], ["restore", inst["victim"]]]
        if kind == "adaptive":  # F2 says what is wrong; perception (measured pose) says where it is
            last = history[-1]["outcome"]
            acts = [] if last["target_loc"] == "basket" else [["ketchup_to_basket"]]
            return acts + [["restore", n] for n in last["off_tray"]]
        if kind == "blind":  # ignores the outcome: redoes the main task (ketchup wherever it is now, jittered grasp)
            fp = self.footprint(TARGET)
            fp["center"] = (np.array(fp["center"]) + np.r_[rng.normal(0, 0.005, 2), 0.0]).tolist()
            return [["ketchup_to_basket", fp]]
        raise ValueError(kind)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """Within-episode protocol: one reset, attempt 1 = main task, attempts 2..k = corrective rounds.
        After every attempt the outcome (success + F2 detail) is measured; stops at success
        (oracle / naive stop after attempt 1)."""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        self.reset_instance(inst)
        for a in range(k):
            actions = self._plan(kind, inst, a, history, rng)
            self.do(actions, inst)
            out = self.outcome(inst)
            params = dict(actions=[x[:2] if x[0] == "restore" else x[:1] for x in actions])
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history


@register_task
class KnockRestoreCross(KnockRestore):
    """Cross-attempt variant: reset to the same instance before every attempt; the knock happens again each
    time. Learned behaviour: anticipate it (restore the reported item right after the main task)."""
    name = "l2_knock_restore_cross"
    protocol = "cross"

    def _plan(self, kind, inst, a, history, rng):
        if kind == "adaptive" and a > 0:  # remembers which item(s) fell last time; finds them by looking
            fallen = sorted({n for h in history for n in h["outcome"]["off_tray"]})
            return [["ketchup_to_basket"]] + [["restore", n] for n in fallen]
        return super()._plan(kind, inst, a, history, rng)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            actions = self._plan(kind, inst, a, history, rng)
            self.do(actions, inst)
            out = self.outcome(inst)
            params = dict(actions=[x[:2] if x[0] == "restore" else x[:1] for x in actions])
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history
