"""L1-A bias_place: pick up the cream cheese and put it into the bowl, with a miscalibrated arm.

Level L1 (action fine-tuning): the method (top-down pick across the cheese's short side, carry, release
above the bowl) and the scene understanding are right; only execution precision is off.

Hidden variable: a fixed calibration offset b = r (cos th, sin th, 0), r ~ U[R_MIN, R_MAX], th ~ U[0, 2 pi),
added to every commanded gripper position (`Skills(bias=b)`). The agent never sees b; positions reported
back to it must be in the commanded frame (`TrackedSkills.reported_eef_pos`), otherwise b leaks.

Robot: the gripper is a short-stroke parallel gripper (max opening GRIPPER_MAX_OPEN = 5.0 cm, e.g. a
Robotiq Hand-E; implemented by narrowing the Panda finger actuators' ctrlrange). The cream cheese is
4.27 cm wide, so the open fingers clear it by ~3.5 mm per side.

Why the prior fails. Sent to the true cheese position, the gripper arrives b away. If the component of b
across the cheese (jaw-closing direction) exceeds ~0.7-1.0 cm (measured), a finger comes down on top of the
cheese, the hand stops ~1.5 cm too high and the jaws close above it -> missed grasp, visible in both
cameras. With the stock 8 cm Panda stroke the measured tolerance is ~2.4 cm, so most r in [2, 4] cm would
not fail; that is why the stroke is reduced. The component of b along the cheese does not matter much
(the finger pads can grip anywhere along its 8.1 cm length, and the in-hand offset cancels at the place
because the same b is added there), and the bowl is forgiving.

What has to be learned: the F2 feedback reports where the gripper closed relative to the cheese's centre
(forward/backward, left/right in the robot frame). That offset is b itself, so after one failure b can be
estimated and subtracted from every commanded position (pick and place alike).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..skills import Skills
from ..task_base import LFFTask, register_task

CHEESE, BOWL = "cream_cheese_1", "akita_black_bowl_1"
JUICE = "orange_juice_1"
DISTRACTORS = ("butter_1", "chocolate_pudding_1", JUICE)
ALL = (CHEESE, BOWL) + DISTRACTORS
NAMES = {CHEESE: "cream cheese", BOWL: "bowl", "butter_1": "butter", "chocolate_pudding_1": "chocolate pudding",
         JUICE: "orange juice carton"}
# descriptive names for the indirect variant (same wording as l2_knock_restore where objects overlap)
DESCR = {CHEESE: "blue box", BOWL: "grey bowl", "butter_1": "small orange box", "chocolate_pudding_1": "brown box",
         JUICE: "tall juice carton"}
# Cream cheese collision box: 8.12 x 4.27 x 1.79 cm, lying flat, long side along its x axis.
# Realistic masses (LIBERO's are 5-10 g): a 6 g cheese is shoved aside by a finger landing on its edge.
MASS = {CHEESE: 0.227, "butter_1": 0.250, "chocolate_pudding_1": 0.200, BOWL: 0.300}

# finger tips (0.97 cm below the gripper site) just clear the table; the pads cover 0.6-1.8 cm of the cheese
Z_GRASP = envs.TABLE_Z + 0.0105
Z_CARRY = envs.TABLE_Z + 0.20  # clears the 13 cm juice carton with the cheese hanging below the fingers
BOWL_RIM = 0.0505  # rim height of the akita bowl above its lowest point
BOWL_R_IN = 0.046  # inner radius of the bowl at the rim
RELEASE_ABOVE_RIM = 0.012  # gripper site above the rim at release (finger pads clear the rim)

R_MIN, R_MAX = 0.025, 0.040  # bias magnitude range (m)
GRIPPER_MAX_OPEN = 0.050  # short-stroke gripper: max finger gap (m); the cheese is 4.27 cm wide
CHEESE_YAW_MAX = np.deg2rad(10)


HOME_XY = np.array([-0.21, 0.0])  # gripper xy at reset


def _seg_dist(p, a, b):
    """distance from point p to segment ab (2-D)"""
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(a + u * ab - p))


def _q(axis, a):
    h = a / 2
    return {"y": np.array([np.cos(h), 0, np.sin(h), 0]), "z": np.array([np.cos(h), 0, 0, np.sin(h)])}[axis]


def _qmul(a, b):
    """wxyz quaternion product a*b"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def set_gripper_stroke(env, max_open):
    """Limit the jaw opening: each finger's position actuator gets the range [0, max_open/2] (robosuite maps
    the open/close command onto the actuator ctrlrange, so "open" now means max_open). The pad faces sit
    0.5 mm inside the joint position, hence the small correction."""
    m = env.sim.model
    half = max_open / 2 + 0.0005
    a1, a2 = [m.actuator_name2id(a) for a in env.robots[0].gripper.actuators]
    m.actuator_ctrlrange[a1] = [0.0, half]
    m.actuator_ctrlrange[a2] = [-half, 0.0]


def dir_words(dx, dy):
    """Robot-frame description: +x = forward (away from the robot), +y = the robot's left."""
    parts = []
    if abs(dy) >= 0.0005:
        parts.append(f"{100 * abs(dy):.1f} cm to the robot's {'left (+y)' if dy > 0 else 'right (-y)'}")
    if abs(dx) >= 0.0005:
        parts.append(f"{100 * abs(dx):.1f} cm {'forward (+x, away from the robot)' if dx > 0 else 'backward (-x, toward the robot)'}")
    return " and ".join(parts) if parts else "exactly at the position"


class TrackedSkills(Skills):
    """Skills that log what the outcome measurement needs, whoever drives them (scripted policy or agent):
    where the gripper was and what the fingers touched when the jaws closed, and where the cheese was when
    the jaws opened while holding it."""

    def __init__(self, env, bias=None, recorder=None):
        super().__init__(env, bias=bias, recorder=recorder)
        self.events = []
        self.cheese_z0 = float(envs.obj_pos(env, CHEESE)[2])
        self.cheese_zmax = self.cheese_z0
        m = env.sim.model
        self._grip_geoms = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith("gripper0_")]
        self._obj_geoms = {n: envs.obj_geom_ids(env, n) for n in ALL}
        # where the cheese was before the gripper last touched it (the reference for the grasp offset, so that
        # a finger shoving the cheese aside on the way down does not corrupt the measurement)
        self.cheese_ref = envs.obj_pos(env, CHEESE)

    def reported_eef_pos(self):
        """End-effector position as the miscalibrated robot believes it to be (commanded frame). Anything
        shown to the agent must use this, never eef_pos(), which is the true position and reveals b."""
        return self.eef_pos() - self.bias

    def move_to(self, target, *args, **kwargs):
        rep = super().move_to(target, *args, **kwargs)
        rep["pos"] = np.round(self.reported_eef_pos(), 4).tolist()  # do not leak b through the report
        return rep

    def touching(self):
        """Objects the gripper is in contact with right now."""
        return [n for n, gs in self._obj_geoms.items() if envs.contacts_between(self.env, self._grip_geoms, gs)]

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.cheese_zmax = max(self.cheese_zmax, float(envs.obj_pos(self.env, CHEESE)[2]))
        if self.grip < 0 and not envs.contacts_between(self.env, self._grip_geoms, self._obj_geoms[CHEESE]):
            self.cheese_ref = envs.obj_pos(self.env, CHEESE)

    def set_gripper(self, close, steps=15):
        if close and self.grip < 0:
            self.events.append(dict(kind="close", eef=self.eef_pos().tolist(),
                                    cheese=envs.obj_pos(self.env, CHEESE).tolist(),
                                    cheese_ref=np.asarray(self.cheese_ref).tolist(), touching=self.touching(),
                                    cheese_top=envs.obj_max_z(self.env, CHEESE), step=self.n_steps))
        elif not close and self.grip > 0 and self.holding(CHEESE):
            self.events.append(dict(kind="release", eef=self.eef_pos().tolist(),
                                    cheese=envs.obj_pos(self.env, CHEESE).tolist(),
                                    bowl=envs.obj_pos(self.env, BOWL).tolist(), step=self.n_steps))
        return super().set_gripper(close, steps)


@register_task
class BiasPlace(LFFTask):
    name = "l1_bias_place"
    level = "L1"
    category = "system_identification"
    capabilities = ("Perceive", "Utilize")
    protocol = "cross"
    # LIBERO calls the bowl "akita black bowl", but under our renderer it looks silver-grey with a gold rim, so
    # the colour is not used in the direct instruction (there is only one bowl).
    instruction = "Pick up the cream cheese and put it into the bowl."
    instruction_indirect = "Pick up the flat blue box with the white oval label and put it into the round grey bowl."
    pick_lift_min = 0.03  # cheese counted as picked up if it rose this much

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        # regions are only used by LIBERO's reset-time sampler (the layout is overridden in apply_instance)
        objs = [(CHEESE, "cream_cheese", "cheese_region", (-0.16, -0.22, -0.14, -0.20)),
                (BOWL, "akita_black_bowl", "bowl_region", (-0.01, 0.15, 0.01, 0.17)),
                ("butter_1", "butter", "butter_region", (-0.16, 0.10, -0.14, 0.12)),
                ("chocolate_pudding_1", "chocolate_pudding", "pudding_region", (0.04, -0.17, 0.06, -0.15)),
                (JUICE, "orange_juice", "juice_region", (-0.15, -0.03, -0.13, -0.01))]
        return write_bddl(self.name, self.instruction, objs)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        # LIBERO's own upright orientation of every object, read off the reset state
        self._base_quat = {n: envs.obj_quat(self.env, n) for n in ALL}

    def _place(self, name, xy, yaw):
        q = _qmul(_q("z", yaw), self._base_quat[name])
        envs.set_obj_pose(self.env, name, [xy[0], xy[1], envs.TABLE_Z + 0.3], quat_wxyz=q)
        self.env.sim.forward()
        qa, _ = envs.free_joint_addr(self.env, name)
        self.env.sim.data.qpos[qa + 2] -= envs.obj_min_z(self.env, name) - (envs.TABLE_Z + 0.0005)
        self.env.sim.forward()

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        while True:
            c = np.array([rng.uniform(-0.16, 0.04), rng.uniform(-0.20, 0.20)])
            bowl = np.array([rng.uniform(-0.15, 0.08), rng.uniform(-0.24, 0.24)])
            if np.linalg.norm(bowl - c) < 0.20:
                continue
            placed = [c, bowl]
            dis = {}
            for name in DISTRACTORS:
                for _ in range(300):
                    p = np.array([rng.uniform(-0.22, 0.10), rng.uniform(-0.27, 0.27)])
                    if any(np.linalg.norm(p - o) < (0.13 if k < 2 else 0.11) for k, o in enumerate(placed)):
                        continue
                    if name == JUICE:
                        # the 13 cm carton is taller than the hand when it grasps / releases: the Panda hand spans
                        # +-10.5 cm across the jaws, so keep the carton >= 17 cm from the cheese and the bowl;
                        # it must also not hide them from the front camera, and must stay >= 10 cm off the
                        # straight paths home->cheese, home->bowl and cheese->bowl
                        if np.linalg.norm(p - c) < 0.17 or np.linalg.norm(p - bowl) < 0.17:
                            continue
                        if any(p[0] > t[0] - 0.03 and abs(p[1] - t[1]) < 0.14 for t in (c, bowl)):
                            continue
                        if any(_seg_dist(p, a, b) < 0.10 for a, b in ((c, bowl), (HOME_XY, c), (HOME_XY, bowl))):
                            continue
                    break
                else:
                    break
                placed.append(p)
                dis[name] = dict(xy=p.round(5).tolist(), yaw=float(rng.uniform(-np.pi, np.pi)))
            if len(dis) == len(DISTRACTORS):
                break
        psi = float(rng.uniform(-CHEESE_YAW_MAX, CHEESE_YAW_MAX))
        n = np.array([-np.sin(psi), np.cos(psi)])  # jaw-closing direction for a grasp across the cheese
        l = np.array([np.cos(psi), np.sin(psi)])
        r = rng.uniform(R_MIN, R_MAX)
        th = rng.uniform(0, 2 * np.pi)
        b = np.array([r * np.cos(th), r * np.sin(th)])
        return dict(seed=int(seed), cheese_xy=c.round(5).tolist(), cheese_yaw=psi, bowl_xy=bowl.round(5).tolist(),
                    distractors=dis, bias=[float(b[0]), float(b[1]), 0.0], bias_r=float(r), bias_theta=float(th),
                    bias_across=float(b @ n), bias_along=float(b @ l))

    def apply_instance(self, inst):
        set_gripper_stroke(self.env, GRIPPER_MAX_OPEN)
        for name, mass in MASS.items():
            envs.scale_mass(self.env, name, mass)
        self._place(CHEESE, inst["cheese_xy"], inst["cheese_yaw"])
        self._place(BOWL, inst["bowl_xy"], 0.0)
        for name, v in inst["distractors"].items():
            self._place(name, v["xy"], v["yaw"])

    def skill_bias(self, inst):
        return np.array(inst["bias"], float)

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = TrackedSkills(self.env, bias=self.skill_bias(inst), recorder=recorder)
        self.start_pos = {n: envs.obj_pos(self.env, n) for n in ALL}
        return self.sk

    # ------------------------------------------------------------ outcome
    def in_bowl(self):
        k = envs.obj_pos(self.env, BOWL)
        c = envs.obj_pos(self.env, CHEESE)
        base = envs.obj_min_z(self.env, BOWL)
        return bool(np.linalg.norm(c[:2] - k[:2]) < BOWL_R_IN and c[2] < base + BOWL_RIM
                    and envs.obj_upright_cos(self.env, BOWL) > 0.9 and base < envs.TABLE_Z + 0.01)

    def outcome(self, inst):
        envs.settle(self.env, 40)
        sk = self.sk
        k = envs.obj_pos(self.env, BOWL)
        c = envs.obj_pos(self.env, CHEESE)
        held = sk.holding(CHEESE)
        success = bool(self.in_bowl() and not held)
        picked = sk.cheese_zmax > sk.cheese_z0 + self.pick_lift_min
        closes = [e for e in sk.events if e["kind"] == "close"]
        last_close = closes[-1] if closes else None
        releases = [e for e in sk.events if e["kind"] == "release" and last_close and e["step"] >= last_close["step"]]
        end_off = c[:2] - k[:2]
        moved = {n: float(np.linalg.norm(envs.obj_pos(self.env, n)[:2] - self.start_pos[n][:2]))
                 for n in DISTRACTORS}
        out = dict(success=success, picked=bool(picked), held_at_end=bool(held),
                   cheese_end=c.round(4).tolist(), bowl_end=k.round(4).tolist(), end_offset=end_off.round(4).tolist(),
                   grasp_offset=None, release_offset=None, touched_at_close=None,
                   distractors_moved_cm={n: round(100 * v, 1) for n, v in moved.items()})
        grasp_txt = ""
        if last_close is not None:
            go = np.array(last_close["eef"][:2]) - np.array(last_close["cheese_ref"][:2])
            out["grasp_offset"] = go.round(4).tolist()
            out["touched_at_close"] = last_close["touching"]
            out["cheese_pushed_cm"] = round(100 * float(np.linalg.norm(
                np.array(last_close["cheese"][:2]) - np.array(last_close["cheese_ref"][:2]))), 1)
            grasp_txt = (f"The gripper closed {dir_words(*go)} of the centre of the cream cheese (measured where "
                         f"the cheese was before the fingers touched it)")
        if releases:
            r = releases[-1]
            out["release_offset"] = (np.array(r["cheese"][:2]) - np.array(r["bowl"][:2])).round(4).tolist()
        if success:
            out["failure"] = None
            detail = "The cream cheese is in the bowl."
            if grasp_txt:
                detail += f" {grasp_txt}."
        elif not picked:
            out["failure"] = "missed_grasp"
            if last_close is None:
                detail = "The gripper never closed; the cream cheese was not picked up."
            else:
                on = [NAMES[n] for n in last_close["touching"] if n != CHEESE]
                why = ""
                if on:
                    why = (f" A finger landed on top of the {' and the '.join(on)} instead of going down beside the "
                           f"cheese, so the jaws closed above the cheese.")
                elif CHEESE in last_close["touching"] and not sk.holding(CHEESE):
                    why = " A finger came down on top of the cream cheese instead of beside it."
                elif not last_close["touching"] and last_close.get("cheese_top") is not None:
                    above = last_close["eef"][2] - envs.FINGERTIP_BELOW_SITE - last_close["cheese_top"]
                    if above > 0.002:  # closed in the air, entirely above the cheese
                        why = (f" The fingers closed in the air without touching anything: the fingertips were "
                               f"{100 * above:.1f} cm above the top of the cream cheese.")
                detail = f"{grasp_txt} and did not pick it up.{why} The cream cheese is still on the table."
        elif held:
            out["failure"] = "not_released"
            detail = f"{grasp_txt} and picked it up, but the cream cheese is still in the gripper."
        elif releases:
            out["failure"] = "missed_place"
            detail = (f"{grasp_txt} and picked it up. When the gripper opened, the cream cheese was "
                      f"{dir_words(*out['release_offset'])} of the bowl's centre; it did not end up in the bowl "
                      f"(it is now {dir_words(*end_off)} of the bowl's centre).")
        else:
            out["failure"] = "dropped"
            detail = (f"{grasp_txt} and picked it up, but it slipped out of the gripper before it was released; "
                      f"it is now {dir_words(*end_off)} of the bowl's centre.")
        out["detail"] = detail
        ind = detail
        for k in (JUICE, "chocolate_pudding_1", "butter_1", CHEESE, BOWL):  # longest names first
            ind = ind.replace(NAMES[k], DESCR[k])
        out["detail_indirect"] = ind
        return out

    def feedback(self, inst, out, level="F2", indirect=False):
        """F0: nothing, F1: success flag, F2: measured offsets. `indirect` uses descriptive object names."""
        if indirect and level == "F2":
            out = dict(out, detail=out.get("detail_indirect", out["detail"]))
        return super().feedback(inst, out, level)

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def execute(self, inst, params):
        """Top-down pick across the cheese's short side, carry, release above the bowl.
        params['offset'] (xy) is the policy's own correction, added to every commanded position."""
        sk = self.sk
        off = np.array(params.get("offset", [0.0, 0.0]), float)
        c = envs.obj_pos(self.env, CHEESE)  # perfect perception of the current scene
        yaw = self.grasp_yaw()
        k = envs.obj_pos(self.env, BOWL)
        z_rel = envs.obj_min_z(self.env, BOWL) + BOWL_RIM + RELEASE_ABOVE_RIM
        g, p = c[:2] + off, k[:2] + off
        sk.set_gripper(False, steps=8)
        sk.move_to([g[0], g[1], Z_CARRY], yaw=yaw)  # approach from above, clear of the 13 cm carton
        sk.move_to([g[0], g[1], Z_GRASP + 0.08], tol=0.002)
        sk.move_to([g[0], g[1], Z_GRASP], tol=0.003, max_steps=80)
        sk.set_gripper(True, steps=15)
        sk.move_to([g[0], g[1], Z_CARRY], speed=0.3)
        if not sk.holding(CHEESE):  # missed: a real controller sees the jaws closed on nothing and stops
            return
        sk.move_to([p[0], p[1], Z_CARRY], tol=0.003)
        sk.move_to([p[0], p[1], z_rel], tol=0.003, max_steps=100)
        sk.set_gripper(False, steps=12)
        sk.move_to([p[0], p[1], Z_CARRY], speed=0.3)

    def grasp_yaw(self):
        """Gripper yaw that closes the jaws across the cheese's short side (yaw 0 closes along world y)."""
        xmat = self.env.sim.data.body_xmat[envs.body_id(self.env, CHEESE)].reshape(3, 3)
        a = np.arctan2(xmat[1, 0], xmat[0, 0])  # direction of the cheese's long (x) axis
        return float((a + np.pi / 2) % np.pi - np.pi / 2)

    def default_params(self, inst):
        return dict(offset=[0.0, 0.0])

    def oracle_params(self, inst):
        return dict(offset=(-np.array(inst["bias"][:2])).tolist())

    def adapt_params(self, inst, history):
        """Scripted F2 learner: the reported grasp offset (where the gripper closed relative to the cheese's
        centre) measures the calibration offset; accumulate it into the correction subtracted from every
        commanded position. If the gripper never closed, fall back to the release offset."""
        last = history[-1]
        off = np.array(last["params"]["offset"], float)
        o = last["outcome"]
        if o.get("grasp_offset") is not None:
            off = off - np.array(o["grasp_offset"])
        elif o.get("release_offset") is not None:
            off = off - np.array(o["release_offset"])
        return dict(offset=off.round(4).tolist())

    blind_sigma = 0.005

    def blind_params(self, inst, attempt, rng):
        """Retry the same plan with a small random perturbation of the targets; no use of history."""
        return dict(offset=rng.normal(0.0, self.blind_sigma, 2).round(4).tolist())
