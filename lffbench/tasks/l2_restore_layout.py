"""L2-B restore_layout: put the ketchup into the basket; every other item must stay exactly where it was.

Scene: the ketchup (upright) and three flat boxes (cream cheese, butter, chocolate pudding) stand on the bare
table in an irregular layout (random positions and yaws, no tray, no markers, no grid), the basket on the
robot's left. Nothing in the scene marks where an item "belongs": its original pose exists only in the
first frame.

Constructed L2 event (physical, continuous): the first time the ketchup is grasped and lifted 3 cm, the
neighbouring box (the "victim", a hidden variable) is knocked: it receives a velocity impulse away from the
ketchup (+-35 deg) and a spin, slides 9-14 cm over the table and comes to rest turned by 45-80 deg. It is
real MuJoCo motion over ~0.2 s (no teleport), so a video shows it sliding away. The main task itself still
succeeds; an agent that stops after it fails. To repair the side effect the agent has to put the victim
back where it was -- within 3 cm and 25 deg (mod 180, the boxes are symmetric) -- and that pose is only in
its memory of the start of the episode (the F2 text says *which* item is out of place, never where it was).

Protocol: within (one episode, no reset). Attempt 1 = the main task, every later attempt = one corrective
round chosen by the policy; the outcome (success + F2 text) is computed after each round.

Scripted references
  oracle    knows the victim: right after the main task it puts the victim back at its saved start pose.
  naive     main task, then done.
  adaptive  saves the start layout from the first frame (Save); after a failure reads from F2 which items are
            out of place (Retrieve their saved poses) and puts them back (Utilize).
  blind     no history, no feedback: every later round redoes the main task (lifts the ketchup out of the
            basket and puts it back, grasp jittered by 5 mm), like l2_knock_restore's blind.
  nomem     (extra control, not in the acceptance table) gets the F2 text but has no start frame: it knows
            which box is out of place and guesses where to put it (random free spot within 15 cm, random yaw).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..task_base import LFFTask, register_task

TARGET = "ketchup_1"
BOXES = ("cream_cheese_1", "butter_1", "chocolate_pudding_1")
BASKET = "basket_1"
ITEMS = (TARGET,) + BOXES

# LIBERO basket (body frame = world frame at yaw 0), measured by the l2_knock_restore developer
BASKET_IN = 0.060  # half size of the inner floor
BASKET_RIM = 0.150  # rim height above the basket's lowest point
BASKET_FLOOR = 0.017  # floor top above the basket's lowest point
BASKET_OUT = 0.11  # conservative half size of the outer footprint (keep-out radius for the layout)

POS_TOL = 0.03  # success: box centre within 3 cm of its start
YAW_TOL = np.radians(25.0)  # ... and long axis within 25 deg of its start (mod 180)
STILL_TOL = 0.02  # the knock only fires if the victim is still within 2 cm of its start

REGION = (-0.26, 0.06, -0.27, 0.10)  # x0, x1, y0, y1 (world) for item centres
KNOCK_REGION = (-0.28, 0.08, -0.29, 0.12)  # where the victim may come to rest
MAX_REACH = 0.70  # max horizontal distance of an item centre from the robot base
BASE_X = -0.66
R_BOX, R_KETCHUP = 0.046, 0.034  # footprint circumradii (boxes 8 x 4.3 cm; ketchup 5.7 x 3.7 cm)
GAP = 0.03  # free space between footprints (finger room)
SLIDE_K = 0.0522  # measured: slide distance [m] = SLIDE_K * v0^2 [m/s] for these boxes on the table
SPIN_W = 40.0  # rad/s, applied each control step until the target turn is reached

NAMES = {"ketchup_1": "ketchup", "cream_cheese_1": "cream cheese", "butter_1": "butter",
         "chocolate_pudding_1": "chocolate pudding", BASKET: "basket"}
DESCR = {"ketchup_1": "red bottle", "cream_cheese_1": "blue box", "butter_1": "small orange box",
         "chocolate_pudding_1": "brown box", BASKET: "wicker basket"}


def _qz(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _wrap_half(a):
    """wrap an angle to [-pi/2, pi/2) (boxes and parallel grippers are symmetric under 180 deg)"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def _seg_dist(p, a, b):
    """distance from point p to segment a-b (2D)"""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _in_rect(xy, r):
    return r[0] <= xy[0] <= r[1] and r[2] <= xy[1] <= r[3]


def _reach_ok(xy):
    return np.hypot(xy[0] - BASE_X, xy[1]) <= MAX_REACH


@register_task
class RestoreLayout(LFFTask):
    name = "l2_restore_layout"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Save", "Retrieve", "Utilize")
    protocol = "within"
    max_attempts = 5
    instruction = ("Put the ketchup into the basket. Every other item on the table must stay exactly where it "
                   "is now, in the same position and orientation.")
    instruction_indirect = ("Put the red bottle into the wicker basket. Every other item on the table must stay "
                            "exactly where it is now, in the same position and orientation.")
    knock_lift = 0.03  # trigger: ketchup grasped and lifted this far above its resting height

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        # upright orientations (wxyz), fixed like in l2_knock_restore: the ketchup mesh is modelled lying down;
        # basket and boxes are upright in their mesh frame
        self.q_up = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in self.env.objects_dict}
        self.q_up[TARGET] = np.array([0.5, 0.5, 0.5, 0.5])
        self._inj = None
        self.start = {}
        self._home = None
        # per-control-step monitor on env.step: also sees attempts driven by the agent harness
        orig_step = self.env.step

        def step(action):
            ret = orig_step(action)
            if self._inj is not None:
                self._knock_step()
            return ret

        self.env.step = step

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        names = [*ITEMS, BASKET]
        objs = []
        for i, n in enumerate(names):  # layout is overridden in apply_instance; regions only need to be valid
            x0 = -0.30 + 0.13 * (i % 3)
            y0 = -0.25 + 0.25 * (i // 3)
            objs.append((n, n[:-2], f"r_{i}", (x0, y0, x0 + 0.02, y0 + 0.02)))
        return write_bddl(self.name, self.instruction, objs)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(5000):
            basket = np.array([rng.uniform(-0.08, 0.0), rng.uniform(0.25, 0.27)])
            k = np.array([rng.uniform(-0.17, -0.03), rng.uniform(-0.14, -0.02)])
            victim = str(rng.choice(BOXES))
            others = [b for b in BOXES if b != victim]
            a_v = rng.uniform(0, 2 * np.pi)
            r_v = rng.uniform(R_BOX + R_KETCHUP + GAP, R_BOX + R_KETCHUP + GAP + 0.02)
            pos = {TARGET: k, victim: k + r_v * np.array([np.cos(a_v), np.sin(a_v)])}
            ok = _in_rect(pos[victim], REGION) and _reach_ok(pos[victim])
            for o in others:
                if not ok:
                    break
                for _ in range(200):
                    p = np.array([rng.uniform(*REGION[:2]), rng.uniform(*REGION[2:])])
                    if not _reach_ok(p):
                        continue
                    if np.linalg.norm(p - k) < R_BOX + R_KETCHUP + GAP:
                        continue
                    if any(np.linalg.norm(p - pos[q]) < 2 * R_BOX + GAP for q in pos if q != TARGET):
                        continue
                    pos[o] = p
                    break
                else:
                    ok = False
            if not ok:
                continue
            if any(np.linalg.norm(pos[n] - basket) < BASKET_OUT + R_BOX + 0.02 for n in ITEMS):
                continue
            # the knock: away from the ketchup +-35 deg, 9-14 cm, turn 45-80 deg (random sign)
            ang = a_v + rng.uniform(-np.radians(35), np.radians(35))
            dist = rng.uniform(0.09, 0.14)
            u = np.array([np.cos(ang), np.sin(ang)])
            end = pos[victim] + dist * u
            if not (_in_rect(end, KNOCK_REGION) and _reach_ok(end)):
                continue
            if np.linalg.norm(end - basket) < BASKET_OUT + R_BOX + 0.03:
                continue
            # the path and the resting spot keep clear of the other boxes (finger room around the victim too)
            if any(_seg_dist(pos[o], pos[victim], end) < 2 * R_BOX + 0.01 for o in others):
                continue
            if any(np.linalg.norm(end - pos[o]) < 2 * R_BOX + GAP for o in others):
                continue
            if np.linalg.norm(end - k) < R_BOX + R_KETCHUP + GAP:
                continue
            turn = float(rng.choice([-1, 1]) * rng.uniform(np.radians(45), np.radians(80)))
            return dict(seed=int(seed),
                        basket_xy=basket.round(4).tolist(),
                        xy={n: pos[n].round(4).tolist() for n in ITEMS},
                        yaw={**{b: float(rng.uniform(-np.pi / 2, np.pi / 2)) for b in BOXES},
                             TARGET: float(np.pi / 2 + rng.uniform(-0.3, 0.3))},
                        # hidden variables of the injected failure
                        victim=victim, knock_dir=float(ang), knock_dist=float(dist), knock_turn=turn,
                        knock_end=end.round(4).tolist())
        raise RuntimeError("could not sample a layout")

    def place(self, name, xy, yaw, surface_z=envs.TABLE_Z):
        """Teleport an object upright (yaw about world z) with its lowest point 2 mm above surface_z (reset only)."""
        q = envs._quat_mul(_qz(yaw), self.q_up[name])
        envs.set_obj_pose(self.env, name, [xy[0], xy[1], surface_z + 0.3], quat_wxyz=q)
        self.env.sim.forward()
        dz = envs.obj_min_z(self.env, name) - (surface_z + 0.002)
        qa, _ = envs.free_joint_addr(self.env, name)
        self.env.sim.data.qpos[qa + 2] -= dz
        self.env.sim.forward()

    def apply_instance(self, inst):
        envs.scale_mass(self.env, BASKET, 2.0)  # heavy enough not to be dragged by the gripper (as in l2_knock)
        self.place(BASKET, inst["basket_xy"], 0.0)
        for n in ITEMS:
            self.place(n, inst["xy"][n], inst["yaw"][n])
        self.env.sim.forward()

    def reset_instance(self, inst, recorder=None):
        self._inj = None
        sk = super().reset_instance(inst, recorder)
        self.start = {n: self.footprint(n) for n in ITEMS}
        self._inj = dict(inst=inst, done=False, knocked=False, step=None, spin_left=0.0, spin_steps=0)
        self._home = sk.eef_pos()
        return sk

    # ------------------------------------------------------------ measurements
    def footprint(self, name):
        """Measured pose summary: AABB centre, yaw of the long horizontal axis, half sizes, z range.
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

    def body_yaw(self, name):
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, name)].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    # ------------------------------------------------------------ failure injection (physical knock)
    def _knock_step(self):
        st = self._inj
        if st["done"]:
            if st["spin_steps"] > 0:  # keep spinning until the target turn is reached (<= 6 control steps)
                self._spin(st)
            return
        if self.sk is None:
            return
        if envs.obj_min_z(self.env, TARGET) < self.start[TARGET]["zmin"] + self.knock_lift:
            return
        if not self.sk.holding(TARGET):
            return
        st["done"] = True
        inst = st["inst"]
        v = inst["victim"]
        if np.linalg.norm(np.array(self.footprint(v)["center"][:2]) - np.array(self.start[v]["center"][:2])) > STILL_TOL:
            return  # the victim was moved beforehand: nothing happens
        qa, da = envs.free_joint_addr(self.env, v)
        v0 = np.sqrt(inst["knock_dist"] / SLIDE_K)
        dvec = np.array([np.cos(inst["knock_dir"]), np.sin(inst["knock_dir"])])
        d = self.env.sim.data
        d.qvel[da:da + 3] = [v0 * dvec[0], v0 * dvec[1], 0.0]
        st.update(knocked=True, step=int(self.sk.n_steps), yaw0=self.body_yaw(v), spin_left=inst["knock_turn"],
                  spin_steps=6)
        self._spin(st)

    def _spin(self, st):
        v = st["inst"]["victim"]
        _, da = envs.free_joint_addr(self.env, v)
        turned = _wrap_half(self.body_yaw(v) - st["yaw0"]) if abs(st["spin_left"]) < np.pi / 2 else 0.0
        rest = st["spin_left"] - turned
        st["spin_steps"] -= 1
        if abs(rest) < np.radians(4) or np.sign(rest) != np.sign(st["spin_left"]):
            st["spin_steps"] = 0
            return
        # measured: one control step at w rad/s turns these boxes by ~0.35-0.8 deg per rad/s; aim a bit short
        w = float(np.clip(np.degrees(abs(rest)) / 0.6, 5.0, SPIN_W))
        self.env.sim.data.qvel[da + 5] = np.sign(rest) * w

    # ------------------------------------------------------------ outcome
    def where_target(self):
        """'basket', 'gripper', 'table' or 'other' for the ketchup."""
        f = self.footprint(TARGET)
        c = np.array(f["center"])
        if self.sk is not None and self.sk.holding(TARGET) and f["zmin"] > envs.TABLE_Z + 0.02:
            return "gripper", f
        bp = envs.obj_pos(self.env, BASKET)
        if np.all(np.abs(c[:2] - bp[:2]) <= BASKET_IN + 0.01) and f["zmin"] < envs.obj_min_z(self.env, BASKET) + BASKET_RIM:
            return "basket", f
        if f["zmin"] < envs.TABLE_Z + 0.01:
            return "table", f
        return "other", f

    def box_state(self, name):
        """Pose error of a box w.r.t. its start pose and where it is."""
        f = self.footprint(name)
        s = self.start[name]
        c, c0 = np.array(f["center"][:2]), np.array(s["center"][:2])
        dpos = float(np.linalg.norm(c - c0))
        dyaw = _wrap_half(f["yaw"] - s["yaw"])
        held = bool(self.sk is not None and self.sk.holding(name))
        upright = envs.obj_upright_cos(self.env, name) > 0.9
        on_table = f["zmin"] < envs.TABLE_Z + 0.01
        bp = envs.obj_pos(self.env, BASKET)
        loc = ("gripper" if held else
               "basket" if np.all(np.abs(c - bp[:2]) <= BASKET_IN + 0.02) and not on_table else
               "table" if on_table and upright else
               "table_tipped" if on_table else "other")
        ok = bool(loc == "table" and dpos <= POS_TOL and abs(dyaw) <= YAW_TOL)
        return dict(ok=ok, loc=loc, dpos=dpos, dyaw=float(dyaw), center=f["center"], yaw=f["yaw"],
                    offset=(c - c0).tolist())

    @staticmethod
    def _offset_words(off, dyaw):
        """robot-centric words for a displacement (current - start) and turn (for the offset tier only)"""
        parts = []
        if abs(off[0]) >= 0.005:
            parts.append(f"{100 * abs(off[0]):.0f} cm {'forward' if off[0] > 0 else 'backward'}")
        if abs(off[1]) >= 0.005:
            parts.append(f"{100 * abs(off[1]):.0f} cm to the {'left' if off[1] > 0 else 'right'}")
        s = (" and ".join(parts) + " of its starting spot") if parts else "at its starting spot"
        if abs(dyaw) >= np.radians(3):
            s += f", turned {abs(np.degrees(dyaw)):.0f} deg {'counter-clockwise' if dyaw > 0 else 'clockwise'} (seen from above)"
        return s

    def _describe(self, tloc, boxes, names, offset=False):
        tn = names[TARGET]
        msgs = []
        if tloc != "basket":
            msgs.append({"gripper": f"The {tn} is still in the gripper, not in the basket.",
                         "table": f"The {tn} is on the table, not in the basket."}
                        .get(tloc, f"The {tn} is not in the basket."))
        for n in BOXES:
            b = boxes[n]
            if b["ok"]:
                continue
            if b["loc"] == "gripper":
                msgs.append(f"The {names[n]} is in the gripper; it must be back where it was at the start.")
                continue
            if b["loc"] == "basket":
                msgs.append(f"The {names[n]} is in the basket; it must be back where it was at the start.")
                continue
            if b["loc"] == "table_tipped":
                msgs.append(f"The {names[n]} has tipped over and is not where it was at the start.")
                continue
            if b["loc"] != "table":
                msgs.append(f"The {names[n]} is not where it was at the start.")
                continue
            what = []
            if b["dpos"] > POS_TOL:
                what.append("position")
            if abs(b["dyaw"]) > YAW_TOL:
                what.append("orientation")
            m = f"The {names[n]} is not where it was at the start ({' and '.join(what)} changed)."
            if offset:
                m += f" It is {self._offset_words(b['offset'], b['dyaw'])}."
            msgs.append(m)
        if not msgs:
            msgs.append(f"The {tn} is in the basket and every other item is where it was at the start.")
        return " ".join(msgs)

    def outcome(self, inst):
        envs.settle(self.env, 30)
        tloc, tf = self.where_target()
        boxes = {n: self.box_state(n) for n in BOXES}
        moved = [n for n in BOXES if not boxes[n]["ok"]]
        success = bool(tloc == "basket" and not moved)
        return dict(success=success, target_loc=tloc, moved=moved,
                    boxes={n: dict(ok=b["ok"], loc=b["loc"], dpos=round(b["dpos"], 4),
                                   dyaw_deg=round(float(np.degrees(b["dyaw"])), 1)) for n, b in boxes.items()},
                    knocked=bool(self._inj and self._inj["knocked"]),
                    detail=self._describe(tloc, boxes, NAMES),
                    detail_indirect=self._describe(tloc, boxes, DESCR),
                    detail_offset=self._describe(tloc, boxes, NAMES, offset=True))

    def feedback(self, inst, out, level="F2", indirect=False, offset=False):
        """F0: nothing, F1: success flag, F2: which item is out of place (never where it was).
        offset=True (ablation tier 'F2+offset'): also the displacement from the start pose in robot-centric
        words -- this externalises the memory and is not the default."""
        if level == "F2":
            key = "detail_offset" if offset else ("detail_indirect" if indirect else "detail")
            out = dict(out, detail=out.get(key, out["detail"]))
        return super().feedback(inst, out, level)

    # ------------------------------------------------------------ scripted motions (Skills.move_to / set_gripper only)
    def go_home(self):
        """End of an attempt: lift the arm out of the camera's way."""
        sk = self.sk
        p = sk.eef_pos()
        sk.move_to([p[0], p[1], max(p[2], self._home[2])])
        sk.move_to(self._home)

    def pick_place(self, name, dest_xy, support_z, dest_yaw=None, carry_z=1.10, clearance=0.006, fp=None,
                   travel_z=1.15):
        """Top-down pick of `name` at its (measured) footprint, carry at eef height `carry_z`, release with the
        object bottom `clearance` above `support_z` at dest_xy; `dest_yaw` = desired yaw of the long axis."""
        sk = self.sk
        f = fp or self.footprint(name)
        c = np.array(f["center"])
        h = f["zmax"] - f["zmin"]
        gyaw = _wrap_half(f["yaw"])  # fingers close along eef y (= world y at yaw 0): across the short side
        if name == TARGET:
            zg = f["zmax"] - 0.035
        else:
            zg = max(f["zmin"] + 0.5 * h, f["zmin"] + 0.0145)  # finger tips reach ~1.3 cm below the grip site
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
        sk.move_to([dest_xy[0], dest_xy[1], support_z + (zg - f["zmin"]) + clearance], tol=0.003, speed=0.2,
                   max_steps=150)
        sk.hold(4)
        sk.set_gripper(False, steps=10)
        sk.move_to([dest_xy[0], dest_xy[1], travel_z], speed=0.4)

    def main_task(self, fp=None):
        """Ketchup -> basket (the instruction, as an agent without failure knowledge would do it)."""
        b = envs.obj_pos(self.env, BASKET)
        floor = envs.obj_min_z(self.env, BASKET) + BASKET_FLOOR
        self.pick_place(TARGET, b[:2], floor, carry_z=1.27, clearance=0.006, fp=fp)

    def restore(self, name, pose):
        """Put box `name` back at `pose` (a remembered footprint: centre xy + long-axis yaw)."""
        self.pick_place(name, pose["center"][:2], envs.TABLE_Z, dest_yaw=pose["yaw"], carry_z=1.05,
                        clearance=0.004)

    def do(self, actions, inst):
        for act in actions:
            if act[0] == "ketchup_to_basket":
                self.main_task(fp=act[1] if len(act) > 1 else None)
            elif act[0] == "restore":  # ["restore", name, pose]
                self.restore(act[1], act[2])
            else:
                raise ValueError(act)

    # ------------------------------------------------------------ scripted reference policies
    def _guess_pose(self, name, rng):
        """nomem control: a random free spot within 15 cm of where the box is now, random yaw."""
        cur = np.array(self.footprint(name)["center"][:2])
        others = [np.array(self.footprint(o)["center"][:2]) for o in BOXES if o != name]
        for _ in range(200):
            r, a = rng.uniform(0.03, 0.15), rng.uniform(0, 2 * np.pi)
            p = cur + r * np.array([np.cos(a), np.sin(a)])
            if _in_rect(p, KNOCK_REGION) and _reach_ok(p) and all(np.linalg.norm(p - o) > 2 * R_BOX + 0.01 for o in others):
                return dict(center=[float(p[0]), float(p[1]), 0.0], yaw=float(rng.uniform(-np.pi / 2, np.pi / 2)))
        return dict(center=[float(cur[0]), float(cur[1]), 0.0], yaw=float(rng.uniform(-np.pi / 2, np.pi / 2)))

    def _plan(self, kind, inst, a, history, rng, memory):
        """Actions of one round: ["ketchup_to_basket"] / ["restore", name, pose]."""
        if kind == "naive" or (a == 0 and kind in ("adaptive", "blind", "nomem")):
            return [["ketchup_to_basket"]]
        if kind == "oracle":  # knows the victim: restores it right after the main task (start pose from memory)
            return [["ketchup_to_basket"], ["restore", inst["victim"], memory[inst["victim"]]]]
        last = history[-1]["outcome"]
        if kind == "adaptive":  # F2 says which items are out of place; the saved first frame says where they were
            acts = [] if last["target_loc"] == "basket" else [["ketchup_to_basket"]]
            return acts + [["restore", n, memory[n]] for n in last["moved"]]
        if kind == "nomem":  # F2 but no start frame: guesses
            acts = [] if last["target_loc"] == "basket" else [["ketchup_to_basket"]]
            return acts + [["restore", n, self._guess_pose(n, rng)] for n in last["moved"]]
        if kind == "blind":  # ignores the outcome: redoes the main task (ketchup wherever it is now, jittered grasp)
            fp = self.footprint(TARGET)
            fp["center"] = (np.array(fp["center"]) + np.r_[rng.normal(0, 0.005, 2), 0.0]).tolist()
            return [["ketchup_to_basket", fp]]
        raise ValueError(kind)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """Within-episode protocol: one reset, attempt 1 = main task, attempts 2..k = corrective rounds."""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        self.reset_instance(inst)
        memory = {n: dict(center=self.start[n]["center"], yaw=self.start[n]["yaw"]) for n in BOXES}  # Save
        for a in range(k):
            actions = self._plan(kind, inst, a, history, rng, memory)
            self.do(actions, inst)
            self.go_home()
            out = self.outcome(inst)
            params = dict(actions=[x[:1] if x[0] == "ketchup_to_basket" else
                                   [x[0], x[1], [round(v, 3) for v in x[2]["center"][:2]], round(float(x[2]["yaw"]), 3)]
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
