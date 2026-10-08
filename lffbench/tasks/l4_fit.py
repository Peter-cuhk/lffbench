"""L4-A fit_container: put the phone flat inside the one box it fits into.

v3 (2026-10-08, user review of cla2): in v2 the only hidden information was which box is big enough, and box and
phone sizes are visible: all 4 Sonnet runs compared the sizes in the images and chose the right box first time.
v3 makes the phone itself bigger than it looks: it wears clear bumper caps on both ends (collision only, nothing is
rendered), CAP_RANGE = 0.8-0.9 cm each, so it is 1.6-1.8 cm longer than the visible phone. The box sizes are drawn
against the VISIBLE phone so that the boxes that look roomiest are too short for the capped phone:
  fit          visible margin +2.8..3.2 cm along the phone's length, only +0.8..0.9 cm across (looks tight),
  snug         +1.15..1.25 cm along, +2.5..3.0 cm across: the most comfortable-looking box, too short with the caps,
  short_wide   +0.3..0.6 cm along, +3..4 cm across: the biggest box, too short with the caps,
  narrow_long  +3.5..4.0 cm along but 0.3..0.6 cm too narrow,
  small        0.5..1.0 cm too short.
A phone that does not fit lies on the rim of the box (or tips in at one end), visibly not inside, although in the
images it looks shorter than the opening. What has to be learned from that: the phone needs more length than it
appears to (by about the overlap seen) -> the box with the most length to spare.

The text below is the v2 description (layout, execution, feedback) and still applies, with "phone length" read as
the capped length. v2 is kept in l4_fit.py.bak-20261008-height.

v2 description:

Scene: a dark "phone" (14.0 x 6.8 x 1.2 cm) lies in the front row of a 2 x 3 grid of slots; five open
boxes of the same colour and the same height fill the other five slots (inner sizes: see INNER_RANGES):
  fit          inner = phone + 1.0-1.5 cm in both directions (phone long side along box long side),
  short_wide   0.5-1.5 cm too short but 3.5-5 cm wider than the phone (looks biggest),
  narrow_long  0.5-1.5 cm too narrow but 2.5-3.5 cm longer than the phone (the longest),
  small        0.5-1.2 cm too short and 0.3-0.6 cm too narrow (most phone-like proportions),
  short        0.8-2.0 cm too short, as wide as the fitting box.
Each box is either roughly lined up with the phone's current long axis or roughly perpendicular to it
(+-15 deg jitter), independently with p = 0.5 -- so half of the time the fitting box needs the phone to be
turned by ~90 deg.

Hidden: which box fits (only careful size comparison reveals it -- the boxes look alike), and whether the
fitting box needs the phone rotated. Box sizes are rebuilt per instance by resizing the bin geoms in
sim.model, so a single env serves every instance.

Prior (naive): take the nearest box and line the phone's long side up with the box's long side (changed
2026-10-06 at the user's request: before, the prior kept the phone's orientation, so half of the first
failures were plain misalignment instead of a size mismatch). F2 feedback after a failed attempt reports where the phone ended up and by how much it is too
long / too wide along the box's long / short side; a learner can then rule out boxes that are not bigger
in that dimension and turn the phone.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, bin_xml, register_generated
from ..skills import _rot_err
from ..task_base import LFFTask, register_task

# phone (local x = long axis); PL is the visible length, the caps add 2 * cap (inst["cap"]) to it
PL, PW, PT = 0.140, 0.068, 0.012
CAP_RANGE = (0.008, 0.009)  # clear bumper cap length per end (hidden)
# boxes (local x = long axis of the opening)
WALL_T, FLOOR_T, WALL_H = 0.006, 0.006, 0.035
BOX_HALF_H = (WALL_H + FLOOR_T) / 2  # body origin is at mid height
RIM_Z = envs.TABLE_Z + FLOOR_T + WALL_H
N_BOXES = 5
ROWS_X = (-0.30, -0.08)  # back row (near the robot), front row
COLS_Y = (-0.22, 0.0, 0.22)
DROP = 0.012  # phone bottom this far above the rim when released
Z_CARRY = envs.TABLE_Z + 0.15
PARK = (-0.45, 0.0, envs.TABLE_Z + 0.35)  # out of the agentview camera's way
PERC_SIGMA = 0.003  # perception noise (m) of the scripted adaptive policy's box-size estimates
JITTER_DEG = 15.0
# inner size of each box kind = phone size + uniform(lo, hi) offset (m), (long side, short side).
# The distractors are chosen so that no coarse size heuristic finds the fitting box: the "short_wide" box has
# the largest area and width, "narrow_long" is the longest, "small" has the most phone-like aspect ratio.
# Only comparing both box dimensions with the phone's (or the F2 feedback) singles out the fitting box.
# v3: offsets are relative to the VISIBLE phone (PL x PW); only "fit" takes the capped phone (PL + 2 cap)
BOX_KINDS = ("fit", "snug", "short_wide", "narrow_long", "small")
INNER_RANGES = {
    "fit": ((0.028, 0.008), (0.032, 0.009)),  # +2.8..3.2 cm along, +0.8..0.9 cm across (looks the tightest)
    "snug": ((0.0115, 0.025), (0.0125, 0.030)),  # +1.15..1.25 along, +2.5..3.0 across: looks the most comfortable
    "short_wide": ((0.003, 0.030), (0.006, 0.040)),  # +0.3..0.6 along, +3..4 across: the biggest box
    "narrow_long": ((0.035, -0.006), (0.040, -0.003)),  # +3.5..4.0 along, 0.3..0.6 cm too narrow
    "small": ((-0.010, 0.003), (-0.005, 0.006)),  # 0.5..1.0 cm too short
}
VISIBLE_FIT = ("fit", "snug", "short_wide")  # kinds that fit the phone as it looks (no caps)
BOX_RGBA = (0.70, 0.60, 0.46, 1.0)


def phone_xml(model_name):
    hx, hy, hz = PL / 2, PW / 2, PT / 2
    body = (0.13, 0.13, 0.15, 1.0)
    screen = (0.02, 0.025, 0.04, 1.0)
    g = (f'        <geom name="{model_name}_g0" type="box" size="{_fmt((hx, hy, hz))}" rgba="{_fmt(body)}" '
         f'density="1500" friction="1.0 0.005 0.0001" {_SOLID} group="0" />\n'
         f'        <geom type="box" size="{_fmt((hx, hy, hz))}" rgba="{_fmt(body)}" conaffinity="0" contype="0" '
         f'group="1" />\n'
         f'        <geom type="box" pos="0 0 {hz:.5f}" size="{_fmt((hx - 0.004, hy - 0.003, 0.0003))}" '
         f'rgba="{_fmt(screen)}" conaffinity="0" contype="0" group="1" />')
    # v3: clear bumper caps on both ends: collision only, fully transparent, no visual twin (nothing is rendered).
    # Resized per instance in apply_instance (inst["cap"]).
    c = CAP_RANGE[1]
    for k, sx in enumerate((1, -1)):
        g += (f'\n        <geom name="{model_name}_cap{k}" type="box" pos="{_fmt((sx * (hx + c / 2), 0, 0))}" '
              f'size="{_fmt((c / 2, hy - 0.001, hz))}" rgba="1 1 1 0" density="400" friction="1.0 0.005 0.0001" '
              f'{_SOLID} group="0" />')
    return _wrap(model_name, g, hz, float(np.hypot(hx + c, hy)))


def wrap_half(a):
    """angle modulo pi into [-pi/2, pi/2) (the phone and the boxes are 180-deg symmetric)"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def wrap_quarter(a):
    """angle modulo pi/2 into [-pi/4, pi/4)"""
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def fits_some_yaw(a, b, clearance=0.002, length=PL):
    """Does the phone's footprint (length x PW) fit inside an a x b opening at any yaw (with `clearance` to spare in
    total)?"""
    th = np.linspace(0.0, np.pi / 2, 181)
    e1 = length * np.cos(th) + PW * np.sin(th)
    e2 = length * np.sin(th) + PW * np.cos(th)
    return bool(np.any((e1 <= a - clearance) & (e2 <= b - clearance)))


def _rect_corners(xy, yaw, half):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * np.asarray(half)
    return pts @ R.T + np.asarray(xy)


def _rects_overlap(r1, r2):
    """separating-axis test for two rotated rectangles given as (xy, yaw, half)"""
    c1, c2 = _rect_corners(*r1), _rect_corners(*r2)
    for yaw in (r1[1], r1[1] + np.pi / 2, r2[1], r2[1] + np.pi / 2):
        ax = np.array([np.cos(yaw), np.sin(yaw)])
        p1, p2 = c1 @ ax, c2 @ ax
        if p1.max() < p2.min() or p2.max() < p1.min():
            return False
    return True


@register_task
class FitContainer(LFFTask):
    name = "l4_fit_container"
    level = "L4"
    category = "geometry_inference"
    capabilities = ("Perceive", "Reason")
    instruction = "Put the phone flat inside a box that it fits into completely."
    instruction_indirect = "Put the black rectangular gadget flat inside a box that it fits into completely."

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        phone = register_generated("LffFitPhone", phone_xml("lff_fit_phone"))
        # one category per box (BDDL lists one object per line; same-category lines would overwrite each
        # other). All share the same MJCF; the opening is resized per instance in apply_instance.
        cats = [register_generated(f"LffFitBox{c}", bin_xml(f"lff_fit_box_{c.lower()}", (0.075, 0.040), WALL_H,
                                                              wall_t=WALL_T, floor_t=FLOOR_T, rgba=BOX_RGBA,
                                                              density=3000))
                for c in "ABCDE"]
        objs = [("phone_1", phone, "phone_region", (-0.09, -0.01, -0.07, 0.01))]
        slots = [(x, y) for x in ROWS_X for y in COLS_Y if not (x == ROWS_X[1] and y == 0.0)]
        for i, (x, y) in enumerate(slots):
            objs.append((f"box_{i + 1}", cats[i], f"box{i + 1}_region", (x - 0.01, y - 0.01, x + 0.01, y + 0.01)))
        return write_bddl(self.name, self.instruction, objs)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        j = np.radians(JITTER_DEG)
        for _ in range(500):
            col = int(rng.integers(3))
            phone_xy = [ROWS_X[1] + rng.uniform(-0.01, 0.01), COLS_Y[col] + rng.uniform(-0.01, 0.01)]
            phone_yaw = float(rng.uniform(-j, j))
            slots = [(x, y) for x in ROWS_X for y in COLS_Y if not (x == ROWS_X[1] and y == COLS_Y[col])]
            kinds = [BOX_KINDS[k] for k in rng.permutation(N_BOXES)]
            cap = float(rng.uniform(*CAP_RANGE))
            boxes = []
            for i, (kind, (x, y)) in enumerate(zip(kinds, slots)):
                lo, hi = INNER_RANGES[kind]
                a, b = PL + rng.uniform(lo[0], hi[0]), PW + rng.uniform(lo[1], hi[1])
                assert (kind == "fit") == fits_some_yaw(a, b, length=PL + 2 * cap), (kind, a, b, cap)
                assert (kind in VISIBLE_FIT) == fits_some_yaw(a, b), (kind, a, b)
                crossed = bool(rng.random() < 0.5)
                yaw = wrap_half(phone_yaw + (np.pi / 2 if crossed else 0.0) + rng.uniform(-j, j))
                boxes.append(dict(name=f"box_{i + 1}", kind=kind, crossed=crossed,
                                  xy=[float(x + rng.uniform(-0.01, 0.01)), float(y + rng.uniform(-0.01, 0.01))],
                                  yaw=yaw, inner=[float(a), float(b)]))
            rects = [(b["xy"], b["yaw"], (b["inner"][0] / 2 + WALL_T + 0.007, b["inner"][1] / 2 + WALL_T + 0.007))
                     for b in boxes]
            rects.append((phone_xy, phone_yaw, (PL / 2 + 0.01, PW / 2 + 0.01)))
            if not any(_rects_overlap(rects[p], rects[q]) for p in range(len(rects)) for q in range(p)):
                break
        else:
            raise RuntimeError("could not sample a non-overlapping layout")
        prng = np.random.default_rng(seed + 7919)
        perc = [[float(b["inner"][0] + prng.normal(0, PERC_SIGMA)), float(b["inner"][1] + prng.normal(0, PERC_SIGMA))]
                for b in boxes]
        return dict(seed=int(seed), phone_xy=[float(v) for v in phone_xy], phone_yaw=phone_yaw, boxes=boxes,
                    fit=int(kinds.index("fit")), fit_crossed=boxes[kinds.index("fit")]["crossed"], cap=cap, _perc=perc)

    def _reshape_box(self, name, a, b):
        ix, iy, H = a / 2, b / 2, BOX_HALF_H
        parts = [((0, 0, -H + FLOOR_T / 2), (ix + WALL_T, iy + WALL_T, FLOOR_T / 2)),
                 ((ix + WALL_T / 2, 0, FLOOR_T / 2), (WALL_T / 2, iy + WALL_T, WALL_H / 2)),
                 ((-ix - WALL_T / 2, 0, FLOOR_T / 2), (WALL_T / 2, iy + WALL_T, WALL_H / 2)),
                 ((0, iy + WALL_T / 2, FLOOR_T / 2), (ix, WALL_T / 2, WALL_H / 2)),
                 ((0, -iy - WALL_T / 2, FLOOR_T / 2), (ix, WALL_T / 2, WALL_H / 2))]
        m = self.env.sim.model
        o = self.env.objects_dict[name]
        for names in (o.contact_geoms, o.visual_geoms):
            assert len(names) == len(parts), names
            for k, n in enumerate(names):
                g = m.geom_name2id(n)
                pos, size = parts[k]
                m.geom_pos[g] = pos
                m.geom_size[g] = size
                m.geom_rbound[g] = float(np.linalg.norm(size))
                m.geom_aabb[g] = [0, 0, 0, *size]

    def _set_caps(self, cap):
        """resize the two hidden bumper caps to `cap` (m) per end"""
        m = self.env.sim.model
        if not hasattr(self, "_cap_ids"):
            self._cap_ids = [g for g in envs.obj_geom_ids(self.env, "phone_1")
                             if m.geom_rgba[g][3] == 0.0 and m.geom_contype[g]]
            assert len(self._cap_ids) == 2, self._cap_ids
        for g in self._cap_ids:
            sx = 1.0 if m.geom_pos[g][0] > 0 else -1.0
            size = (cap / 2, PW / 2 - 0.001, PT / 2)
            m.geom_pos[g] = (sx * (PL / 2 + cap / 2), 0.0, 0.0)
            m.geom_size[g] = size
            m.geom_rbound[g] = float(np.linalg.norm(size))
            m.geom_aabb[g] = [0, 0, 0, *size]

    def apply_instance(self, inst):
        # walls are moved inwards at run time, out of the compile-time per-body BVH leaf boxes, so the collision
        # mid-phase can silently skip them (fingers / phone pass through; found on l4_nest_order 2026-10-07).
        # Testing all geom pairs of nearby bodies is exact; the run-time geom_rbound / geom_aabb are set below.
        import mujoco
        self.env.sim.model._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self._set_caps(inst["cap"])
        for b in inst["boxes"]:
            self._reshape_box(b["name"], *b["inner"])
            envs.set_obj_pose(self.env, b["name"], [*b["xy"], envs.TABLE_Z + BOX_HALF_H + 0.0005], yaw=b["yaw"])
        envs.set_obj_pose(self.env, "phone_1", [*inst["phone_xy"], envs.TABLE_Z + PT / 2 + 0.0005],
                          yaw=inst["phone_yaw"])

    # ------------------------------------------------------------ outcome
    def _pose2d(self, name):
        bid = envs.body_id(self.env, name)
        R = self.env.sim.data.body_xmat[bid].reshape(3, 3)
        p = self.env.sim.data.body_xpos[bid].copy()
        return p, float(np.arctan2(R[1, 0], R[0, 0])), float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def outcome(self, inst):
        envs.settle(self.env, 40)
        p, yaw, tilt = self._pose2d("phone_1")
        held = bool(self.sk is not None and self.sk.holding("phone_1"))
        # which box is the phone in / on: smallest distance of the phone centre outside a box opening
        best = None
        for j, b in enumerate(inst["boxes"]):
            c, byaw, btilt = self._pose2d(b["name"])
            d = np.array([[np.cos(byaw), np.sin(byaw)], [-np.sin(byaw), np.cos(byaw)]]) @ (p[:2] - c[:2])
            out = np.maximum(np.abs(d) - np.array(b["inner"]) / 2, 0.0)
            dist = float(np.linalg.norm(out))
            if best is None or dist < best[0]:
                best = (dist, j, c, byaw, btilt)
        dist, j, c, byaw, btilt = best
        b = inst["boxes"][j]
        a_in, b_in = b["inner"]
        floor_top = c[2] - BOX_HALF_H + FLOOR_T
        height = float(p[2] - PT / 2 - floor_top)  # phone bottom above the box floor
        in_opening = dist < 1e-9
        on_floor = height < 0.004 and tilt < 5.0
        success = bool(in_opening and on_floor and not held and btilt < 10.0)
        where = self.box_label(inst, j)  # relative description only: no absolute coordinates in feedback
        # footprint of the phone (as it lies now) against the opening of that box
        dl = yaw - byaw
        cs, sn = abs(np.cos(dl)), abs(np.sin(dl))
        L = PL + 2 * inst.get("cap", 0.0)  # the capped phone
        over_long = L * cs + PW * sn - a_in
        over_short = L * sn + PW * cs - b_in
        along_long = "length" if cs >= sn else "width"
        fb = dict(box=j, box_xy=[round(float(c[0]), 2), round(float(c[1]), 2)], along_long=along_long,
                  over_long_cm=round(100 * over_long, 1) if over_long > 0.001 else None,
                  over_short_cm=round(100 * over_short, 1) if over_short > 0.001 else None)
        res = dict(success=success, box=j, box_kind=b["kind"], height_cm=round(100 * height, 2),
                   tilt_deg=round(tilt, 1), held=held, phone_xy=[float(p[0]), float(p[1])], feedback=fb)
        if held:
            res["detail"] = "The phone is still in the gripper."
            res["failure"] = "held"
            return res
        if dist > 0.02 or (not in_opening and p[2] < envs.TABLE_Z + PT / 2 + 0.006):
            res["detail"] = (f"The phone is not in any box: it is lying on the table, about {100 * dist:.0f} cm "
                             f"outside the opening of {where}.")
            res["failure"] = "table"
            return res
        if success:
            res["detail"] = f"The phone is lying flat on the bottom of {where}."
            res["failure"] = None
            return res
        pose_txt = ("lying flat on top of the walls" if tilt < 5.0 else
                    f"tilted about {tilt:.0f} degrees with one end down inside the box")
        txt = f"The phone did not go in: it is resting on the walls of {where}, {pose_txt}."
        dim_l = "long" if along_long == "length" else "wide"
        dim_s = "wide" if along_long == "length" else "long"
        if fb["over_long_cm"]:
            txt += f" It is about {fb['over_long_cm']:.1f} cm too {dim_l} for that box along the box's long side."
        if fb["over_short_cm"]:
            txt += f" It is about {fb['over_short_cm']:.1f} cm too {dim_s} for that box along the box's short side."
        if not fb["over_long_cm"] and not fb["over_short_cm"]:
            txt += " Its footprint is smaller than the opening, but it was not lined up with the box and caught on the rim."
        res["detail"] = txt
        res["failure"] = "rim"
        return res

    @staticmethod
    def box_label(inst, j):
        """Name box j by its place among the boxes in robot-centric terms (left = +y = the robot's left)."""
        ys = [b["xy"][1] for b in inst["boxes"]]
        xs = [b["xy"][0] for b in inst["boxes"]]
        n = len(ys)
        order = sorted(range(n), key=lambda i: -ys[i])  # robot's left first
        r = order.index(j)
        close = [i for i in range(n) if i != j and abs(ys[i] - ys[j]) < 0.04]
        if close:  # boxes side by side in y: tell them apart by distance from the robot
            nearer = all(xs[j] < xs[i] for i in close)
            return f"the box {'nearest to' if nearer else 'farthest from'} the robot among those at the same left-right position"
        if r == 0:
            return "the box furthest to the robot's left (+y)"
        if r == n - 1:
            return "the box furthest to the robot's right (-y)"
        return "the middle box (between the others, left to right)"

    # ------------------------------------------------------------ execution
    def execute(self, inst, params):
        """params: box (index into inst['boxes']), place_yaw (world yaw of the phone's long side when released).
        Optional (tolerance checks only): dx / dy = release offset of the phone centre from the box centre
        along the box's long / short axis (m), drop = height of the phone bottom above the rim (m)."""
        sk, env = self.sk, self.env
        px, py = inst["phone_xy"]
        zc = envs.TABLE_Z + PT / 2
        g0 = wrap_half(inst["phone_yaw"])  # fingers close across the phone's width
        sk.set_gripper(False, steps=5)
        sk.move_to([px, py, zc + 0.10], yaw=g0)
        sk.move_to([px, py, zc + 0.001], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([px, py, Z_CARRY], speed=0.25)
        b = inst["boxes"][int(params["box"])]
        c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
        dx, dy = params.get("dx", 0.0), params.get("dy", 0.0)
        bx, by = b["xy"][0] + c * dx - s * dy, b["xy"][1] + s * dx + c * dy
        sk.move_to([bx, by, Z_CARRY], yaw=wrap_half(params["place_yaw"]))
        # put the phone's centre (not the gripper's) over the box centre: compensates in-hand slip
        off = envs.obj_pos(env, "phone_1") - sk.eef_pos()
        drop = params.get("drop", DROP)
        sk.move_to([bx - off[0], by - off[1], RIM_Z + max(0.05, drop + 0.01) - off[2]], tol=0.003)
        tgt = self._servo_phone([bx, by], RIM_Z + drop)
        sk.grip = -1.0  # open while holding the arm at a fixed target (Skills.hold re-anchors and drifts)
        self._hold_at(tgt, 12)
        sk.move_to([tgt[0], tgt[1], RIM_Z + 0.10])
        sk.move_to(list(PARK), yaw=0.0)

    def _hold_at(self, target, steps):
        sk = self.sk
        R = sk._R()
        for _ in range(steps):
            sk._act(np.asarray(target) - sk.eef_pos(), _rot_err(R, sk.eef_mat()), sk.grip)

    def _servo_phone(self, xy, z_bottom, max_steps=60, tol=0.0015):
        """Closed-loop: move the gripper until the held phone's centre is over xy and its bottom at z_bottom.
        Returns the final gripper target."""
        sk = self.sk
        R = sk._R()
        goal = np.array([xy[0], xy[1], z_bottom + PT / 2])
        tgt = e_prev = sk.eef_pos()
        for i in range(max_steps):
            ph, e = envs.obj_pos(self.env, "phone_1"), sk.eef_pos()
            if i >= 8 and np.linalg.norm(ph - goal) < tol and np.linalg.norm(e - e_prev) < 0.0005:
                break
            tgt = goal - (ph - e)
            sk._act(tgt - e, _rot_err(R, sk.eef_mat()), sk.grip)
            e_prev = e
        return tgt

    # ------------------------------------------------------------ scripted policies
    def _naive_yaw(self, inst, j):
        """keep the phone's orientation, only snap it to the nearest axis of box j"""
        p = inst["phone_yaw"]
        return float(p + wrap_quarter(inst["boxes"][j]["yaw"] - p))

    def default_params(self, inst):
        """v3 prior: careful size comparison against the phone as it looks (exact visible sizes: the prior models the
        trap; perception noise is only given to the adaptive learner) -- the box with the most room to spare in its
        tighter direction, i.e. "snug" by construction; phone's long side along the box's long side"""
        marg = [min(b["inner"][0] - PL, b["inner"][1] - PW) for b in inst["boxes"]]
        j = int(np.argmax(marg))
        return dict(box=j, place_yaw=float(inst["boxes"][j]["yaw"]))

    def nearest_params(self, inst):
        """v2 prior (kept for reference): nearest box, lined up"""
        d = [np.hypot(b["xy"][0] - inst["phone_xy"][0], b["xy"][1] - inst["phone_xy"][1]) for b in inst["boxes"]]
        j = int(np.argmin(d))
        return dict(box=j, place_yaw=float(inst["boxes"][j]["yaw"]))

    def oracle_params(self, inst):
        j = inst["fit"]
        return dict(box=j, place_yaw=float(inst["boxes"][j]["yaw"]))

    def blind_params(self, inst, attempt, rng):
        """no history: a random box (may repeat), lined up like the prior"""
        j = int(rng.integers(N_BOXES))
        return dict(box=j, place_yaw=float(inst["boxes"][j]["yaw"]))

    def naive_keep_yaw_params(self, inst):
        """old prior (kept for reference): nearest box, keep the phone's orientation"""
        d = [np.hypot(b["xy"][0] - inst["phone_xy"][0], b["xy"][1] - inst["phone_xy"][1]) for b in inst["boxes"]]
        j = int(np.argmin(d))
        return dict(box=j, place_yaw=self._naive_yaw(inst, j))

    def adapt_params(self, inst, history):
        """Scripted learner on F2 feedback.

        Sees the boxes with noisy size estimates (inst['_perc'], sigma = PERC_SIGMA) but does not know the
        phone's size. Each 'too long / too wide by X along the box's long / short side' message gives an
        estimate of the phone's length or width (perceived box dimension + X). It then picks an untried box
        that is big enough in every estimated dimension (preferring the biggest in the still-unknown one) and
        lines the phone's long side up with the box's long side.
        """
        perc = inst["_perc"]
        L_est = W_est = None
        tried_aligned = set()
        for h in history:
            o = h["outcome"]
            fb = o.get("feedback") or {}
            if o.get("failure") != "rim":
                continue
            j = fb["box"]
            a, b = perc[j]
            if fb["along_long"] == "length":
                if fb["over_long_cm"] or fb["over_short_cm"]:  # a size verdict on this box
                    tried_aligned.add(j)
                if fb["over_long_cm"]:
                    L_est = max(L_est or 0.0, a + fb["over_long_cm"] / 100)
                if fb["over_short_cm"]:
                    W_est = max(W_est or 0.0, b + fb["over_short_cm"] / 100)
            else:
                if fb["over_short_cm"]:
                    L_est = max(L_est or 0.0, b + fb["over_short_cm"] / 100)
                if fb["over_long_cm"]:
                    W_est = max(W_est or 0.0, a + fb["over_long_cm"] / 100)
        thr = 0.002
        cands = []
        for j, (a, b) in enumerate(perc):
            if j in tried_aligned:
                continue
            mL = None if L_est is None else a - L_est
            mW = None if W_est is None else b - W_est
            feasible = (mL is None or mL > thr) and (mW is None or mW > thr)
            if mL is not None and mW is not None:
                key = min(mL, mW)
            elif mL is not None:
                key = b
            elif mW is not None:
                key = a
            else:
                key = a * b
            cands.append((feasible, key, j))
        if not cands:
            j = int(np.argmax([a * b for a, b in perc]))
        else:
            j = max(cands)[2]
        return dict(box=int(j), place_yaw=float(inst["boxes"][j]["yaw"]))

    def elim_params(self, inst, history, rng):
        """sensitivity variant: uses the feedback only to learn the orientation (always lines the phone up
        after the first attempt) and to never retry a box that got a size verdict; picks among the remaining
        boxes at random, i.e. no size reasoning / perception."""
        bad = set()
        for h in history:
            fb = h["outcome"].get("feedback") or {}
            if h["outcome"].get("failure") == "rim" and fb.get("along_long") == "length" and (
                    fb.get("over_long_cm") or fb.get("over_short_cm")):
                bad.add(int(fb["box"]))
        left = [j for j in range(N_BOXES) if j not in bad] or list(range(N_BOXES))
        j = int(rng.choice(left))
        return dict(box=j, place_yaw=float(inst["boxes"][j]["yaw"]))

    EXTRA_KINDS = ("elim",)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        if kind not in self.EXTRA_KINDS:
            return super().run_scripted(kind, inst, k=k, seed=seed, on_attempt=on_attempt)
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            if a == 0:
                params = self.default_params(inst)
            else:
                params = self.elim_params(inst, history, rng)
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"]:
                break
        return history


# ---------------------------------------------------------------- physical validity checks
def physics_check(seed0=1000, n=20, out=None):
    """Every (box, orientation) pair on each instance, released centred from DROP above the rim.
    Expect success iff the box is the fitting one and the phone's long side is along the box's long side."""
    import json
    task = FitContainer()
    rows = []
    for seed in range(seed0, seed0 + n):
        inst = task.sample_instance(seed)
        for j, b in enumerate(inst["boxes"]):
            for orient in ("aligned", "crossed"):
                task.reset_instance(inst)
                task.execute(inst, dict(box=j, place_yaw=float(b["yaw"] + (0.0 if orient == "aligned" else np.pi / 2))))
                o = task.outcome(inst)
                rows.append(dict(seed=seed, box=j, kind=b["kind"], inner_cm=[round(100 * v, 2) for v in b["inner"]],
                                 orient=orient, success=o["success"], expect=bool(j == inst["fit"] and orient == "aligned"),
                                 failure=o["failure"], height_cm=o["height_cm"], tilt_deg=o["tilt_deg"], detail=o["detail"]))
                r = rows[-1]
                print(f"seed {seed} box {j} {b['kind']:6s} {orient:7s} succ={r['success']} h={r['height_cm']} "
                      f"tilt={r['tilt_deg']}" + ("" if r["success"] == r["expect"] else "  <-- MISMATCH"), flush=True)
        if out:
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
    print("total", len(rows), "mismatches", sum(r["success"] != r["expect"] for r in rows))
    return rows


def stress_check(seed=1002, out=None):
    """Release errors (offset / yaw / drop height) at the edges of the size ranges: the tightest fitting box
    (+1.0 cm in both directions) and boxes only 0.5 cm too short / too narrow."""
    import copy
    import json
    task = FitContainer()
    base = task.sample_instance(seed)
    j = base["fit"]
    rows = []
    for name, a, b in [("fit_min", PL + 0.010, PW + 0.010), ("short_0.5", PL - 0.005, PW + 0.010),
                       ("narrow_0.5", PL + 0.010, PW - 0.005)]:
        for dx, dy, dyaw, drop in [(0, 0, 0, DROP), (0.004, 0, 0, DROP), (0, 0.004, 0, DROP), (0.007, 0, 0, DROP),
                                   (0, 0, 3, DROP), (0, 0, 6, DROP), (0, 0, 0, 0.05), (0.004, 0.003, 3, 0.05)]:
            inst = copy.deepcopy(base)
            inst["boxes"][j]["inner"] = [a, b]
            task.reset_instance(inst)
            task.execute(inst, dict(box=j, place_yaw=float(inst["boxes"][j]["yaw"] + np.radians(dyaw)),
                                    dx=dx, dy=dy, drop=drop))
            o = task.outcome(inst)
            rows.append(dict(case=name, inner_cm=[round(a * 100, 2), round(b * 100, 2)], dx_mm=dx * 1000,
                             dy_mm=dy * 1000, dyaw_deg=dyaw, drop_cm=drop * 100, success=o["success"],
                             height_cm=o["height_cm"], tilt_deg=o["tilt_deg"], detail=o["detail"]))
            print(f"{name:11s} dx={dx * 1000:.0f}mm dy={dy * 1000:.0f}mm dyaw={dyaw}deg drop={drop * 100:.1f}cm -> "
                  f"succ={o['success']} h={o['height_cm']} tilt={o['tilt_deg']} | {o['detail']}", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    # python -m lffbench.tasks.l4_fit physics results/l4_fit_container/physics_check.json
    # python -m lffbench.tasks.l4_fit stress results/l4_fit_container/stress_check.json
    import sys
    if sys.argv[1] == "physics":
        physics_check(out=sys.argv[2] if len(sys.argv) > 2 else None)
    elif sys.argv[1] == "stress":
        stress_check(out=sys.argv[2] if len(sys.argv) > 2 else None)
