"""L1-B height_bias: pick up a thin tile from the top of a wooden block and put it on the plate, with an arm whose
vertical calibration is off.

Category system_identification (level L1, action fine-tuning): the method (top-down pick across the tile's short
side with the fingertips just above the block's top, carry, release above the plate) and the scene understanding
are right; only the robot's own vertical execution is off.

Hidden variable: a fixed vertical calibration offset dz ~ U[2.0, 3.5] cm (always upward), added to every
commanded gripper position (`Skills(bias=(0, 0, dz))`): the gripper stops dz higher than commanded. The agent never
sees dz; positions reported back to it are in the commanded frame (`HeightSkills.reported_eef_pos`, and the agent
harness reports eef - bias), so the robot "believes" it reached the commanded height.

Why the prior fails: the tile is 1.0 cm thick. Sent to the right height (fingertips just above the block's top), the
fingertips actually stop 2-3.5 cm higher, i.e. 1.1-2.6 cm above the tile's top face, and the jaws close in the air
above it (measured tolerance, 6 layouts: upward errors up to 0.4 cm still grasp, >= 0.7 cm never do, i.e. the fingers
must overlap the tile by ~4 mm). dz >= 2 cm also defeats a prior that aims lower: even with the grasp point commanded
level with the block's top (fingertips ~1 cm into the block), the fingertips end up >= 1.05 cm above the block's
top, i.e. not below the tile's top face -> missed.
Visible in the agentview image (gap between the closed fingers and the tile) and in the wrist image (fingers closed
over the tile).
A downward error would not fail (the fingertips land on the block's top and the compliant OSC still closes on the
tile; measured for -1 and -2 cm), so the offset is upward only.

Why a raised block: the agent's move_to clips commanded targets to >= table + 0.5 cm. With the tile on the table,
compensating a 2-3.5 cm upward offset would need commands below the table top, which the interface refuses; on a
6-10 cm block the compensated commands stay well inside the workspace.

What has to be learned: F2 reports how far above the tile's top face the grasp point and the fingertips were when the
jaws closed. Compared with where the policy meant to put them, that is dz itself, so after one failure dz can be
estimated and subtracted from every commanded height (pick and place alike).
"""
import re

import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

TILE, STAND, PLATE = "tile_1", "stand_1", "plate_1"
DISTRACTORS = ("butter_1", "chocolate_pudding_1")
FREE_OBJECTS = (TILE, PLATE) + DISTRACTORS
NAMES = {TILE: "red tile", STAND: "wooden block", PLATE: "plate", "butter_1": "butter",
         "chocolate_pudding_1": "chocolate pudding"}
DESCR = {TILE: "thin red slab", STAND: "large wooden pedestal", PLATE: "round white dish", "butter_1": "small orange box",
         "chocolate_pudding_1": "brown box"}

TILE_HALF = (0.040, 0.0225, 0.005)  # 8.0 x 4.5 x 1.0 cm (long side along the tile's x axis)
TILE_T = 2 * TILE_HALF[2]
STAND_HALF_XY = (0.075, 0.085)  # 15 x 17 cm top
STAND_H_RANGE = (0.06, 0.10)
MASS = {TILE: 0.072, "butter_1": 0.250, "chocolate_pudding_1": 0.200, PLATE: 0.350}

FINGER_BELOW_SITE = 0.0095  # lowest point of the finger collision meshes below the grasp site (measured 0.95 cm)
SITE_ABOVE_SURFACE = 0.0105  # scripted grasp height: grasp point 1.05 cm above the block top (fingertips ~1 mm above)
CARRY_ABOVE = 0.12  # carry height above the block top
RELEASE_GAP = 0.012  # tile bottom this far above the plate's rim at release
PLATE_R_OK = 0.040  # success: tile centre within this horizontal distance of the plate centre
DZ_MIN, DZ_MAX = 0.020, 0.035
TILE_YAW_MAX = np.deg2rad(40)
HOME_XY = np.array([-0.21, 0.0])


def _seg_dist(p, a, b):
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(a + u * ab - p))


def _q(axis, a):
    h = a / 2
    return {"z": np.array([np.cos(h), 0, 0, np.sin(h)])}[axis]


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def dir_words(dx, dy):
    """Robot-frame description: +x = forward (away from the robot), +y = the robot's left."""
    parts = []
    if abs(dy) >= 0.0005:
        parts.append(f"{100 * abs(dy):.1f} cm to the robot's {'left (+y)' if dy > 0 else 'right (-y)'}")
    if abs(dx) >= 0.0005:
        parts.append(f"{100 * abs(dx):.1f} cm {'forward (+x, away from the robot)' if dx > 0 else 'backward (-x, toward the robot)'}")
    return " and ".join(parts) if parts else "exactly at the position"


def vert_words(h, what):
    if h >= 0.0005:
        return f"{100 * h:.1f} cm above {what}"
    if h <= -0.0005:
        return f"{100 * -h:.1f} cm below {what}"
    return f"level with {what}"


class HeightSkills(Skills):
    """Skills that log what the outcome measurement needs, whoever drives them (scripted policy or agent)."""

    def __init__(self, env, bias=None, recorder=None, stand_geoms=()):
        super().__init__(env, bias=bias, recorder=recorder)
        self.events = []
        m = env.sim.model
        self._finger_geoms = {
            side: [m.geom_name2id(g) for g in env.robots[0].gripper.important_geoms[side]]
            for side in ("left_finger", "right_finger")}
        self._grip_geoms = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith("gripper0_")]
        self._obj_geoms = {n: envs.obj_geom_ids(env, n) for n in FREE_OBJECTS}
        self._obj_geoms[STAND] = list(stand_geoms)
        self.tile_z0 = float(envs.obj_pos(env, TILE)[2])
        self.tile_zmax = self.tile_z0
        self.tile_ref = envs.obj_pos(env, TILE)  # tile position before the fingers last touched it
        self.tile_ref_yaw = self.tile_yaw()

    def reported_eef_pos(self):
        """End-effector position as the miscalibrated robot believes it to be (commanded frame). Anything shown to
        the agent must use this, never eef_pos(), which is the true position and reveals the bias."""
        return self.eef_pos() - self.bias

    def move_to(self, target, *args, **kwargs):
        rep = super().move_to(target, *args, **kwargs)
        rep["pos"] = np.round(self.reported_eef_pos(), 4).tolist()
        return rep

    def touching(self):
        return [n for n, gs in self._obj_geoms.items() if envs.contacts_between(self.env, self._grip_geoms, gs)]

    def holding_tile(self):
        """Both fingers in contact with the tile (robosuite's _check_grasp needs the finger *pads*, which miss a
        1 cm tile when it is gripped low on the finger meshes)."""
        tg = self._obj_geoms[TILE]
        return all(envs.contacts_between(self.env, gs, tg) for gs in self._finger_geoms.values())

    def holding(self, obj_name):
        """The agent harness reports `holding_object` through this; use the finger-contact test for the tile so a
        tile gripped low on the fingers is not reported as dropped."""
        if obj_name == TILE:
            return self.holding_tile()
        return super().holding(obj_name)

    def tile_yaw(self):
        xmat = self.env.sim.data.body_xmat[envs.body_id(self.env, TILE)].reshape(3, 3)
        return float(np.arctan2(xmat[1, 0], xmat[0, 0]))

    def finger_min_z(self):
        return float(self.eef_pos()[2] - FINGER_BELOW_SITE)

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.tile_zmax = max(self.tile_zmax, float(envs.obj_pos(self.env, TILE)[2]))
        if self.grip < 0 and not envs.contacts_between(self.env, self._grip_geoms, self._obj_geoms[TILE]):
            self.tile_ref = envs.obj_pos(self.env, TILE)
            self.tile_ref_yaw = self.tile_yaw()

    def set_gripper(self, close, steps=15):
        if close and self.grip < 0:
            self.events.append(dict(kind="close", eef=self.eef_pos().tolist(), finger_z=self.finger_min_z(),
                                    tile=envs.obj_pos(self.env, TILE).tolist(),
                                    tile_ref=np.asarray(self.tile_ref).tolist(), tile_ref_yaw=self.tile_ref_yaw,
                                    touching=self.touching(),
                                    step=self.n_steps))
        elif not close and self.grip > 0 and self.holding_tile():
            self.events.append(dict(kind="release", eef=self.eef_pos().tolist(),
                                    tile=envs.obj_pos(self.env, TILE).tolist(),
                                    plate=envs.obj_pos(self.env, PLATE).tolist(), step=self.n_steps))
        return super().set_gripper(close, steps)


@register_task
class HeightBias(LFFTask):
    name = "l1_height_bias"
    level = "L1"
    category = "system_identification"
    capabilities = ("Perceive", "Utilize")
    protocol = "cross"
    instruction = "Pick up the red tile from the top of the wooden block and put it on the plate."
    instruction_indirect = ("Pick up the thin flat red slab lying on the large wooden pedestal and put it on the "
                            "round white dish.")
    pick_lift_min = 0.03

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        tile = register_generated("LffHbTile", box_xml("lff_hb_tile", TILE_HALF, (0.80, 0.16, 0.12, 1.0),
                                                       density=2000))
        stand = register_generated("LffHbStand", box_xml("lff_hb_stand", STAND_HALF_XY + (STAND_H_RANGE[1] / 2,),
                                                         (0.62, 0.47, 0.30, 1.0), density=1000), free=False)
        objs = [(TILE, tile, "tile_region", (-0.21, -0.26, -0.19, -0.24)),
                (PLATE, "plate", "plate_region", (-0.01, 0.24, 0.01, 0.26)),
                ("butter_1", "butter", "butter_region", (-0.21, 0.24, -0.19, 0.26)),
                ("chocolate_pudding_1", "chocolate_pudding", "pudding_region", (0.15, -0.26, 0.17, -0.24))]
        fx = [(STAND, stand, "stand_region", (0.09, -0.01, 0.11, 0.01))]
        return write_bddl(self.name, self.instruction, objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        self._base_quat = {n: envs.obj_quat(self.env, n) for n in FREE_OBJECTS}
        m = self.env.sim.model
        bid = m.body_name2id(self.env.fixtures_dict[STAND].root_body)

        def under(b):
            while b > 0:
                if b == bid:
                    return True
                b = m.body_parentid[b]
            return False

        self._stand_bid = bid
        self._stand_geoms = [g for g in range(m.ngeom) if under(m.geom_bodyid[g])]

    def _place(self, name, xy, yaw, z0=envs.TABLE_Z):
        q = _qmul(_q("z", yaw), self._base_quat[name])
        envs.set_obj_pose(self.env, name, [xy[0], xy[1], z0 + 0.3], quat_wxyz=q)
        self.env.sim.forward()
        qa, _ = envs.free_joint_addr(self.env, name)
        self.env.sim.data.qpos[qa + 2] -= envs.obj_min_z(self.env, name) - (z0 + 0.0005)
        self.env.sim.forward()

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        while True:
            s = np.array([rng.uniform(-0.10, 0.02), rng.uniform(-0.17, 0.17)])
            plate = np.array([rng.uniform(-0.17, 0.08), rng.uniform(-0.25, 0.25)])
            if np.linalg.norm(plate - s) < 0.24 or abs(plate[1] - s[1]) < 0.12:
                continue  # not right behind the block as seen from the front camera
            placed = [s, plate]
            dis = {}
            for name in DISTRACTORS:
                for _ in range(300):
                    p = np.array([rng.uniform(-0.22, 0.12), rng.uniform(-0.27, 0.27)])
                    if np.linalg.norm(p - s) < 0.16 or any(np.linalg.norm(p - o) < 0.12 for o in placed[1:]):
                        continue
                    if any(_seg_dist(p, a, b) < 0.09 for a, b in ((s, plate), (HOME_XY, s))):
                        continue
                    break
                else:
                    break
                placed.append(p)
                dis[name] = dict(xy=p.round(5).tolist(), yaw=float(rng.uniform(-np.pi, np.pi)))
            if len(dis) == len(DISTRACTORS):
                break
        stand_h = float(rng.uniform(*STAND_H_RANGE))
        tile_xy = s + rng.uniform(-0.02, 0.02, 2)
        tile_yaw = float(rng.uniform(-TILE_YAW_MAX, TILE_YAW_MAX))
        dz = float(rng.uniform(DZ_MIN, DZ_MAX))
        return dict(seed=int(seed), stand_xy=s.round(5).tolist(), stand_h=stand_h, tile_xy=tile_xy.round(5).tolist(),
                    tile_yaw=tile_yaw, plate_xy=plate.round(5).tolist(), distractors=dis, bias=[0.0, 0.0, dz],
                    bias_dz=dz)

    def stand_top(self, inst):
        return envs.TABLE_Z + inst["stand_h"]

    def apply_instance(self, inst):
        m = self.env.sim.model
        h = inst["stand_h"]
        for g in self._stand_geoms:
            m.geom_size[g, 2] = h / 2
        m.body_pos[self._stand_bid] = [inst["stand_xy"][0], inst["stand_xy"][1], envs.TABLE_Z + h / 2]
        m.body_quat[self._stand_bid] = [1, 0, 0, 0]
        for name, mass in MASS.items():
            envs.scale_mass(self.env, name, mass)
        self.env.sim.forward()
        self._place(TILE, inst["tile_xy"], inst["tile_yaw"], z0=self.stand_top(inst))
        self._place(PLATE, inst["plate_xy"], 0.0)
        for name, v in inst["distractors"].items():
            self._place(name, v["xy"], v["yaw"])

    def skill_bias(self, inst):
        return np.array(inst["bias"], float)

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = HeightSkills(self.env, bias=self.skill_bias(inst), recorder=recorder, stand_geoms=self._stand_geoms)
        self.start_pos = {n: envs.obj_pos(self.env, n) for n in FREE_OBJECTS}
        return self.sk

    # ------------------------------------------------------------ outcome
    def plate_rim_z(self):
        lo, hi = self._aabb(PLATE)
        return float(hi[2])

    def _aabb(self, name):
        m, d = self.env.sim.model, self.env.sim.data
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in envs.obj_geom_ids(self.env, name):
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            hh = np.abs(R) @ m.geom_aabb[g, 3:]
            lo, hi = np.minimum(lo, c - hh), np.maximum(hi, c + hh)
        return lo, hi

    def on_plate(self):
        p = envs.obj_pos(self.env, PLATE)
        t = envs.obj_pos(self.env, TILE)
        return bool(np.linalg.norm(t[:2] - p[:2]) < PLATE_R_OK and t[2] < self.plate_rim_z() + 0.015
                    and envs.obj_upright_cos(self.env, PLATE) > 0.9 and envs.obj_min_z(self.env, PLATE) < envs.TABLE_Z + 0.01)

    def outcome(self, inst):
        envs.settle(self.env, 40)
        sk = self.sk
        top = self.stand_top(inst)
        p = envs.obj_pos(self.env, PLATE)
        t = envs.obj_pos(self.env, TILE)
        held = sk.holding_tile()
        success = bool(self.on_plate() and not held)
        picked = sk.tile_zmax > sk.tile_z0 + self.pick_lift_min
        closes = [e for e in sk.events if e["kind"] == "close"]
        last_close = closes[-1] if closes else None
        releases = [e for e in sk.events if e["kind"] == "release" and last_close and e["step"] >= last_close["step"]]
        end_off = t[:2] - p[:2]
        on_stand = bool(t[2] > top - 0.005)
        out = dict(success=success, picked=bool(picked), held_at_end=bool(held), tile_end=t.round(4).tolist(),
                   plate_end=p.round(4).tolist(), end_offset=end_off.round(4).tolist(), grasp_height=None,
                   fingertip_gap=None, grasp_xy_offset=None, touched_at_close=None, release_offset=None)
        grasp_txt = ""
        over_tile = True
        if last_close is not None:
            tile_top = last_close["tile_ref"][2] + TILE_HALF[2]
            gh = last_close["eef"][2] - tile_top
            fg = last_close["finger_z"] - tile_top
            go = np.array(last_close["eef"][:2]) - np.array(last_close["tile_ref"][:2])
            a = last_close.get("tile_ref_yaw", 0.0)
            along, across = go @ [np.cos(a), np.sin(a)], go @ [-np.sin(a), np.cos(a)]
            # grasp point over the tile's footprint (the open fingers straddle it across its short side)
            over_tile = bool(abs(along) <= TILE_HALF[0] and abs(across) <= TILE_HALF[1] + 0.01)
            out["grasp_height"] = round(gh, 4)
            out["fingertip_gap"] = round(fg, 4)
            out["grasp_xy_offset"] = go.round(4).tolist()
            out["grasp_over_tile"] = over_tile
            out["touched_at_close"] = last_close["touching"]
            face = "the top face of the red tile" if over_tile else "the level of the red tile's top face"
            grasp_txt = (f"When the jaws closed, the grasp point (midway between the fingertips) was "
                         f"{vert_words(gh, face)} and the lowest points of the fingertips "
                         f"were {vert_words(fg, 'it')} (the tile is {100 * TILE_T:.1f} cm thick)")
            if not over_tile or np.linalg.norm(go) >= 0.01:
                grasp_txt += (f"; horizontally the grasp point was {dir_words(*go)} of the tile's centre"
                              + ("" if over_tile else ", i.e. not over the tile"))
        if success:
            out["failure"] = None
            detail = "The red tile is on the plate."
            if grasp_txt:
                detail += f" {grasp_txt}."
        elif not picked:
            out["failure"] = "missed_grasp"
            if last_close is None:
                detail = "The gripper never closed; the red tile was not picked up."
            else:
                touch = last_close["touching"]
                if not over_tile and TILE not in touch:
                    why = " The gripper was not over the tile when it closed."
                elif fg > 0.0005 and TILE not in touch:
                    why = " The fingers closed in the air above the tile without touching it."
                elif STAND in touch and TILE not in touch:
                    why = " The fingertips were pressed on the top of the wooden block but did not grip the tile."
                elif TILE in touch:
                    why = " The fingers touched the tile but did not get a grip on it."
                else:
                    why = ""
                where = "still on the wooden block" if on_stand else "no longer on the wooden block"
                detail = f"{grasp_txt}, and the tile was not picked up.{why} The tile is {where}."
        elif held:
            out["failure"] = "not_released"
            detail = f"{grasp_txt}, and the tile was picked up, but it is still in the gripper."
        elif releases:
            r = releases[-1]
            out["release_offset"] = (np.array(r["tile"][:2]) - np.array(r["plate"][:2])).round(4).tolist()
            out["failure"] = "missed_place"
            detail = (f"{grasp_txt}, and the tile was picked up. When the gripper opened, the tile was "
                      f"{dir_words(*out['release_offset'])} of the plate's centre; it did not end up on the plate "
                      f"(it is now {dir_words(*end_off)} of the plate's centre).")
        else:
            out["failure"] = "dropped"
            detail = (f"{grasp_txt}, and the tile was lifted, but it slipped out of the fingers before it was "
                      f"released; it is now {dir_words(*end_off)} of the plate's centre.")
        out["detail"] = detail
        ind = detail
        for k in sorted(NAMES, key=lambda n: -len(NAMES[n])):
            ind = ind.replace(NAMES[k], DESCR[k])
        ind = re.sub(r"\btile\b", "slab", ind)  # the bare noun would give the direct name away
        out["detail_indirect"] = ind
        return out

    def feedback(self, inst, out, level="F2", indirect=False):
        if indirect and level == "F2":
            out = dict(out, detail=out.get("detail_indirect", out["detail"]))
        return super().feedback(inst, out, level)

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def grasp_yaw(self):
        """Gripper yaw that closes the jaws across the tile's short side (yaw 0 closes along world y)."""
        xmat = self.env.sim.data.body_xmat[envs.body_id(self.env, TILE)].reshape(3, 3)
        a = np.arctan2(xmat[1, 0], xmat[0, 0])
        return float((a + np.pi / 2) % np.pi - np.pi / 2)

    def execute(self, inst, params):
        """Top-down pick across the tile's short side with the fingertips just above the block top, carry, release
        above the plate. params['dz'] is the policy's own height correction, added to every commanded z."""
        sk = self.sk
        dz = float(params.get("dz", 0.0))
        t = envs.obj_pos(self.env, TILE)  # perfect perception of the current scene
        top = self.stand_top(inst)
        yaw = self.grasp_yaw()
        p = envs.obj_pos(self.env, PLATE)
        z_grasp = top + SITE_ABOVE_SURFACE + dz
        z_carry = top + CARRY_ABOVE + dz
        z_rel = self.plate_rim_z() + RELEASE_GAP + SITE_ABOVE_SURFACE + dz
        sk.set_gripper(False, steps=8)
        sk.move_to([t[0], t[1], z_carry], yaw=yaw)
        sk.move_to([t[0], t[1], z_grasp + 0.05], tol=0.002)
        sk.move_to([t[0], t[1], z_grasp], tol=0.003, max_steps=80)
        sk.set_gripper(True, steps=15)
        sk.move_to([t[0], t[1], z_carry], speed=0.3)
        if not sk.holding_tile():  # missed: a real controller sees the jaws closed on nothing and stops
            return
        sk.move_to([p[0], p[1], z_carry], tol=0.003)
        sk.move_to([p[0], p[1], z_rel], tol=0.003, max_steps=100)
        sk.set_gripper(False, steps=12)
        sk.move_to([p[0], p[1], z_carry], speed=0.3)

    def default_params(self, inst):
        return dict(dz=0.0)

    def oracle_params(self, inst):
        return dict(dz=-float(inst["bias"][2]))

    def adapt_params(self, inst, history):
        """Scripted F2 learner: the reported grasp height above the tile's top face, minus the height this policy
        meant to put the grasp point at (SITE_ABOVE_SURFACE - tile thickness), is the vertical offset; accumulate
        it into the correction subtracted from every commanded height."""
        last = history[-1]
        dz = float(last["params"]["dz"])
        o = last["outcome"]
        if o.get("grasp_height") is not None:
            dz -= o["grasp_height"] - (SITE_ABOVE_SURFACE - TILE_T)
        return dict(dz=round(dz, 4))

    blind_sigma = 0.005

    def blind_params(self, inst, attempt, rng):
        """Retry the same plan with a small random perturbation of the commanded heights; no use of history."""
        return dict(dz=round(float(rng.normal(0.0, self.blind_sigma)), 4))
