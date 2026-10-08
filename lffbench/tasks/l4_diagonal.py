"""L4-B diagonal_fit: put a long bar flat into an open box whose inside is shorter than the bar.

Scene: one open box (inner a x b, a slightly longer than b, 3 cm walls) and a long, thin wooden bar
(L x 1.2 x 1.4 cm) lying on the table in front of it. The bar is 1.2-2.2 cm longer than the box's inside
length, so it never fits lined up with a side of the box -- it fits only when turned towards the box's
diagonal, at an angle that depends on a, b and L (36-45 deg off the long side; the fitting window at zero
centring error is 12-25 deg wide, see task card).

Hidden: the exact dimensions (the bar is only 8-17 % longer than the box -- easy to miss at a glance), and
that a diagonal placement exists. Box and bar are resized per instance by editing geom sizes in sim.model,
so one env serves every instance (same technique as l4_fit_container).

Prior (naive): tidy placement -- line the bar up with the box's long side. It always ends lying across the
box on the rim (visible). F2 after a failure: where the bar lies, the angle between bar and box, and by how
much the bar's footprint at that angle is longer than the inside of the box along the box's long / short
side. A learner can work out the bar length from that and turn the bar towards the diagonal; a second
failure (angle slightly off) gives the over-length along the other side, which tells it which way to correct.

Only move_to / open / close are needed: grasp the bar in the middle, turn the wrist, release above the box.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import bin_xml, box_xml, register_generated
from ..skills import _rot_err
from ..task_base import LFFTask, register_task

# bar (local x = long axis); length set per instance
BAR_W, BAR_H = 0.012, 0.014
BAR_L_MAX = 0.160
BAR_RGBA = (0.72, 0.36, 0.14, 1.0)
# box (local x = long axis of the opening); inner size set per instance
WALL_T, FLOOR_T, WALL_H = 0.006, 0.006, 0.030
BOX_HALF_H = (WALL_H + FLOOR_T) / 2  # body origin at mid height
RIM_Z = envs.TABLE_Z + FLOOR_T + WALL_H
BOX_RGBA = (0.55, 0.62, 0.70, 1.0)
# sampling ranges (m)
INNER_A = (0.115, 0.130)  # inside length of the box
A_MINUS_B = (0.004, 0.020)  # the box is a little longer than wide
OVER_LEN = (0.010, 0.020)  # bar length minus inside length of the box
MIN_MARGIN = 0.011  # best-angle clearance (total) the sampler insists on
MIN_WINDOW_DEG = 8.0  # width of the angle window with >= 6 mm total clearance
# execution
Z_GRASP = envs.TABLE_Z + 0.0105  # gripper site; pads span ~0.55-2.25 cm above the table, tips just clear it
Z_CARRY = envs.TABLE_Z + 0.15
DROP = 0.010  # bar bottom this far above the rim at release
PARK = (-0.45, 0.0, envs.TABLE_Z + 0.35)
PERC_SIGMA = 0.003  # perception noise of the scripted adaptive learner (box inner size, bar width)


MIDPHASE_OFF = True


def disable_midphase(env):
    """Box and bar are resized at run time by editing geom pos / size in sim.model, but MuJoCo's per-body bounding
    volume hierarchy (bvh_aabb, used by the collision mid-phase) is computed once at compile time for the registered
    size. A box wall moved inwards out of its compiled leaf box is silently skipped by the mid-phase (found on the
    nest_order cups, 2026-10-07: the fingers passed through resized walls). Without the mid-phase MuJoCo tests all
    geom pairs of two nearby bodies (still filtered by the run-time geom_rbound set when resizing)."""
    import mujoco
    m = env.sim.model._model
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)


def wrap_half(a):
    """angle modulo pi into [-pi/2, pi/2) (bar and box are 180-deg symmetric)"""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def fold_quarter(a):
    """|angle| between two pi-symmetric axes, in [0, pi/2]"""
    a = abs(wrap_half(a))
    return float(min(a, np.pi - a))


def footprint(L, W, rel):
    """extent of an L x W rectangle turned by rel against the box axes: (along long side, along short side)"""
    c, s = abs(np.cos(rel)), abs(np.sin(rel))
    return L * c + W * s, L * s + W * c


def margins(L, W, a, b, rel):
    e1, e2 = footprint(L, W, rel)
    return a - e1, b - e2


def best_angle(L, W, a, b, n=1801):
    """angle in [0, pi/2] maximising the smaller of the two clearances; returns (angle, clearance, window)"""
    th = np.linspace(0.0, np.pi / 2, n)
    m1, m2 = margins(L, W, a, b, th)
    m = np.minimum(m1, m2)
    i = int(np.argmax(m))
    ok = th[m >= 0.006]
    win = float(np.degrees(ok.max() - ok.min())) if len(ok) else 0.0
    return float(th[i]), float(m[i]), win


def _rect_corners(xy, yaw, half):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * np.asarray(half)
    return pts @ R.T + np.asarray(xy)


def _rects_overlap(r1, r2):
    c1, c2 = _rect_corners(*r1), _rect_corners(*r2)
    for yaw in (r1[1], r1[1] + np.pi / 2, r2[1], r2[1] + np.pi / 2):
        ax = np.array([np.cos(yaw), np.sin(yaw)])
        p1, p2 = c1 @ ax, c2 @ ax
        if p1.max() < p2.min() or p2.max() < p1.min():
            return False
    return True


def _rel_dir(dx, dy):
    """robot-centric text for an offset in the world frame (forward = +x, left = +y), ending in 'of' / 'from'"""
    parts = []
    if abs(dx) >= 0.005:
        parts.append(f"{100 * abs(dx):.1f} cm {'forward' if dx > 0 else 'backward'}")
    if abs(dy) >= 0.005:
        parts.append(f"{100 * abs(dy):.1f} cm to the {'left' if dy > 0 else 'right'}")
    return " and ".join(parts) + " of" if parts else "within 0.5 cm of"


def _turn_text(signed_deg):
    """relative orientation of the bar against the box's long side (no absolute angles)"""
    if abs(signed_deg) < 2.0:
        return "lies along the box's long side (within 2 degrees)"
    side = "counterclockwise (towards the left)" if signed_deg > 0 else "clockwise (towards the right)"
    return f"is turned about {abs(signed_deg):.0f} degrees {side}, seen from above, from the box's long side"


@register_task
class DiagonalFit(LFFTask):
    name = "l4_diagonal_fit"
    level = "L4"
    category = "geometry_inference"
    capabilities = ("Perceive", "Reason")
    instruction = "Put the wooden bar into the box so that it lies flat on the bottom of the box."
    instruction_indirect = "Put the long brown stick into the grey open tray so that it lies flat on the tray's floor."

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        bar = register_generated("LffDiagBar", box_xml("lff_diag_bar", (BAR_L_MAX / 2, BAR_W / 2, BAR_H / 2),
                                                       BAR_RGBA, density=700))
        box = register_generated("LffDiagBox", bin_xml("lff_diag_box", (0.065, 0.065), WALL_H, wall_t=WALL_T,
                                                       floor_t=FLOOR_T, rgba=BOX_RGBA, density=3000))
        objs = [("bar_1", bar, "bar_region", (0.02, -0.01, 0.04, 0.01)),
                ("box_1", box, "box_region", (-0.19, -0.01, -0.17, 0.01))]
        return write_bddl(self.name, self.instruction, objs)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            a = float(rng.uniform(*INNER_A))
            b = float(a - rng.uniform(*A_MINUS_B))
            L = float(a + rng.uniform(*OVER_LEN))
            th, m, win = best_angle(L, BAR_W, a, b)
            if m < MIN_MARGIN or win < MIN_WINDOW_DEG:
                continue
            box_xy = [float(rng.uniform(-0.22, -0.13)), float(rng.uniform(-0.10, 0.10))]
            box_yaw = float(rng.uniform(-np.pi / 4, np.pi / 4))
            bar_xy = [float(rng.uniform(-0.01, 0.05)), float(rng.uniform(-0.13, 0.13))]
            bar_yaw = float(rng.uniform(-np.pi / 2, np.pi / 2))
            r_box = (box_xy, box_yaw, (a / 2 + WALL_T + 0.03, b / 2 + WALL_T + 0.03))
            r_bar = (bar_xy, bar_yaw, (L / 2 + 0.01, BAR_W / 2 + 0.03))
            if _rects_overlap(r_box, r_bar):
                continue
            break
        else:
            raise RuntimeError("could not sample an instance")
        prng = np.random.default_rng(seed + 7919)
        perc = [float(a + prng.normal(0, PERC_SIGMA)), float(b + prng.normal(0, PERC_SIGMA)),
                float(BAR_W + prng.normal(0, PERC_SIGMA / 2))]
        return dict(seed=int(seed), inner=[a, b], bar_len=L, bar_xy=bar_xy, bar_yaw=bar_yaw, box_xy=box_xy,
                    box_yaw=box_yaw, fit_angle_deg=float(np.degrees(th)), fit_margin_cm=float(100 * m),
                    fit_window_deg=win, _perc=perc)

    def _reshape_box(self, a, b):
        ix, iy, H = a / 2, b / 2, BOX_HALF_H
        parts = [((0, 0, -H + FLOOR_T / 2), (ix + WALL_T, iy + WALL_T, FLOOR_T / 2)),
                 ((ix + WALL_T / 2, 0, FLOOR_T / 2), (WALL_T / 2, iy + WALL_T, WALL_H / 2)),
                 ((-ix - WALL_T / 2, 0, FLOOR_T / 2), (WALL_T / 2, iy + WALL_T, WALL_H / 2)),
                 ((0, iy + WALL_T / 2, FLOOR_T / 2), (ix, WALL_T / 2, WALL_H / 2)),
                 ((0, -iy - WALL_T / 2, FLOOR_T / 2), (ix, WALL_T / 2, WALL_H / 2))]
        m = self.env.sim.model
        o = self.env.objects_dict["box_1"]
        for names in (o.contact_geoms, o.visual_geoms):
            assert len(names) == len(parts), names
            for k, n in enumerate(names):
                g = m.geom_name2id(n)
                pos, size = parts[k]
                m.geom_pos[g] = pos
                m.geom_size[g] = size
                m.geom_rbound[g] = float(np.linalg.norm(size))
                m.geom_aabb[g] = [0, 0, 0, *size]

    def _resize_bar(self, L):
        m = self.env.sim.model
        o = self.env.objects_dict["bar_1"]
        for n in list(o.contact_geoms) + list(o.visual_geoms):
            g = m.geom_name2id(n)
            m.geom_size[g] = [L / 2, BAR_W / 2, BAR_H / 2]
            m.geom_rbound[g] = float(np.linalg.norm(m.geom_size[g]))
            m.geom_aabb[g] = [0, 0, 0, *m.geom_size[g]]
        # keep the mass of the resized bar consistent with its volume (density 700); the geoms live in the
        # child body "object", not in the root body that carries the free joint
        bid = int(m.geom_bodyid[m.geom_name2id(o.contact_geoms[0])])
        mass = 700.0 * L * BAR_W * BAR_H
        m.body_mass[bid] = mass
        m.body_inertia[bid] = mass / 12.0 * np.array([BAR_W ** 2 + BAR_H ** 2, L ** 2 + BAR_H ** 2,
                                                      L ** 2 + BAR_W ** 2])

    def apply_instance(self, inst):
        if MIDPHASE_OFF:
            disable_midphase(self.env)
        self._reshape_box(*inst["inner"])
        self._resize_bar(inst["bar_len"])
        envs.set_obj_pose(self.env, "box_1", [*inst["box_xy"], envs.TABLE_Z + BOX_HALF_H + 0.0005],
                          yaw=inst["box_yaw"])
        envs.set_obj_pose(self.env, "bar_1", [*inst["bar_xy"], envs.TABLE_Z + BAR_H / 2 + 0.0005],
                          yaw=inst["bar_yaw"])

    # ------------------------------------------------------------ outcome
    def _pose(self, name):
        bid = envs.body_id(self.env, name)
        R = self.env.sim.data.body_xmat[bid].reshape(3, 3)
        return self.env.sim.data.body_xpos[bid].copy(), R

    def outcome(self, inst):
        envs.settle(self.env, 40)
        a, b = inst["inner"]
        L = inst["bar_len"]
        p, R = self._pose("bar_1")
        ax = R[:, 0]
        bar_yaw = float(np.arctan2(ax[1], ax[0]))
        tilt = float(np.degrees(np.arcsin(min(1.0, abs(ax[2])))))  # long axis vs horizontal
        c, Rb = self._pose("box_1")
        byaw = float(np.arctan2(Rb[1, 0], Rb[0, 0]))
        btilt = float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1))))
        held = bool(self.sk is not None and self.sk.holding("bar_1"))
        rel = fold_quarter(bar_yaw - byaw)
        # bar footprint corners in the box frame
        Rz = np.array([[np.cos(byaw), np.sin(byaw)], [-np.sin(byaw), np.cos(byaw)]])
        corners = _rect_corners(p[:2], bar_yaw, (L / 2 * np.cos(np.radians(tilt)), BAR_W / 2))
        loc = (corners - c[:2]) @ Rz.T
        out_x = float(np.max(np.abs(loc[:, 0])) - a / 2)
        out_y = float(np.max(np.abs(loc[:, 1])) - b / 2)
        centre = Rz @ (p[:2] - c[:2])
        floor_top = c[2] - BOX_HALF_H + FLOOR_T
        height = float(envs.obj_min_z(self.env, "bar_1") - floor_top)
        in_opening = out_x <= 0.002 and out_y <= 0.002
        on_floor = height < 0.005 and tilt < 6.0
        success = bool(in_opening and on_floor and not held and btilt < 10.0)
        m_long, m_short = margins(L, BAR_W, a, b, rel)
        fb = dict(rel_deg=round(float(np.degrees(rel)), 1),
                  over_long_cm=round(-100 * m_long, 1) if m_long < -0.001 else None,
                  over_short_cm=round(-100 * m_short, 1) if m_short < -0.001 else None,
                  spare_long_cm=round(100 * max(m_long, 0.0), 1) if m_long >= -0.001 else None,
                  spare_short_cm=round(100 * max(m_short, 0.0), 1) if m_short >= -0.001 else None,
                  centre_offset_cm=[round(100 * float(centre[0]), 1), round(100 * float(centre[1]), 1)])
        res = dict(success=success, height_cm=round(100 * height, 2), tilt_deg=round(tilt, 1), held=held,
                   rel_deg=fb["rel_deg"], bar_xy=[float(p[0]), float(p[1])], feedback=fb)
        signed = float(np.degrees(wrap_half(bar_yaw - byaw)))
        fb["rel_signed_deg"] = round(signed, 1)
        off_w = p[:2] - c[:2]  # bar centre relative to the box centre, world axes (only relative values are reported)
        if held:
            res["detail"] = "The bar is still in the gripper."
            res["failure"] = "held"
            return res
        over_box = abs(centre[0]) < a / 2 + WALL_T and abs(centre[1]) < b / 2 + WALL_T
        on_table = envs.obj_min_z(self.env, "bar_1") < envs.TABLE_Z + 0.008  # (floor_top is FLOOR_T above the table)
        if not over_box and on_table:
            res["detail"] = (f"The bar is not in the box: it is lying on the table, its centre "
                             f"{_rel_dir(*off_w)} the box's centre.")
            res["failure"] = "table"
            return res
        if success:
            res["detail"] = "The bar is lying flat on the bottom of the box."
            res["failure"] = None
            return res
        if not over_box:
            res["detail"] = (f"The bar is not in the box: it ended partly on the box's rim, "
                             f"{100 * (p[2] - envs.TABLE_Z):.1f} cm above the table, its centre "
                             f"{_rel_dir(*off_w)} the box's centre.")
            res["failure"] = "off"
            return res
        pose_txt = (f"lying across the top of the box on its rim, {100 * height:.1f} cm above the box floor"
                    if tilt < 6.0 else
                    f"tilted about {tilt:.0f} degrees, with one end down inside the box and the other end on the rim")
        txt = f"The bar did not go in: it is {pose_txt}. The bar {_turn_text(signed)}."
        parts = []
        if fb["over_long_cm"]:
            parts.append(f"{fb['over_long_cm']:.1f} cm longer than the inside of the box along the box's long side")
        if fb["over_short_cm"]:
            parts.append(f"{fb['over_short_cm']:.1f} cm longer than the inside of the box along the box's short side")
        if parts:
            txt += " At that angle the bar's footprint is " + " and ".join(parts) + "."
        else:
            txt += (f" At that angle the bar's footprint is smaller than the opening (spare {fb['spare_long_cm']:.1f} cm "
                    f"along the long side, {fb['spare_short_cm']:.1f} cm along the short side), but it caught on the "
                    f"rim (bar centre {_rel_dir(*off_w)} the box's centre).")
        res["detail"] = txt
        res["failure"] = "rim"
        return res

    # ------------------------------------------------------------ execution
    def execute(self, inst, params):
        """params: rel_yaw (rad) = angle of the bar's long axis against the box's long side at release.
        Optional (tolerance checks): dx / dy release offset of the bar centre from the box centre along the
        box's long / short axis (m); drop = bar bottom above the rim at release (m)."""
        sk, env = self.sk, self.env
        px, py = inst["bar_xy"]
        g0 = wrap_half(inst["bar_yaw"])  # yaw 0 closes the fingers along world y = across a bar lying along x
        sk.set_gripper(False, steps=5)
        sk.move_to([px, py, Z_GRASP + 0.10], yaw=g0)
        sk.move_to([px, py, Z_GRASP], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([px, py, Z_CARRY], speed=0.25)
        byaw = inst["box_yaw"]
        c, s = np.cos(byaw), np.sin(byaw)
        dx, dy = params.get("dx", 0.0), params.get("dy", 0.0)
        bx, by = inst["box_xy"][0] + c * dx - s * dy, inst["box_xy"][1] + s * dx + c * dy
        sk.move_to([bx, by, Z_CARRY], yaw=wrap_half(byaw + params["rel_yaw"]))
        off = envs.obj_pos(env, "bar_1") - sk.eef_pos()
        drop = params.get("drop", DROP)
        sk.move_to([bx - off[0], by - off[1], RIM_Z + drop + BAR_H / 2 + 0.04 - off[2]], tol=0.003)
        tgt = self._servo_bar([bx, by], RIM_Z + drop)
        sk.grip = -1.0
        self._hold_at(tgt, 12)
        sk.move_to([tgt[0], tgt[1], RIM_Z + 0.10])
        sk.move_to(list(PARK), yaw=0.0)

    def _hold_at(self, target, steps):
        sk = self.sk
        R = sk._R()
        for _ in range(steps):
            sk._act(np.asarray(target) - sk.eef_pos(), _rot_err(R, sk.eef_mat()), sk.grip)

    def _servo_bar(self, xy, z_bottom, max_steps=60, tol=0.0015):
        """closed loop on the held bar's centre (scripted policies only)"""
        sk = self.sk
        R = sk._R()
        goal = np.array([xy[0], xy[1], z_bottom + BAR_H / 2])
        tgt = e_prev = sk.eef_pos()
        for i in range(max_steps):
            pb, e = envs.obj_pos(self.env, "bar_1"), sk.eef_pos()
            if i >= 8 and np.linalg.norm(pb - goal) < tol and np.linalg.norm(e - e_prev) < 0.0005:
                break
            tgt = goal - (pb - e)
            sk._act(tgt - e, _rot_err(R, sk.eef_mat()), sk.grip)
            e_prev = e
        return tgt

    # ------------------------------------------------------------ scripted policies
    def _signed(self, inst, th):
        """+th or -th, whichever needs the smaller wrist turn from the bar's current orientation"""
        cur = wrap_half(inst["bar_yaw"] - inst["box_yaw"])
        return min((th, -th), key=lambda t: abs(wrap_half(t - cur)))

    def default_params(self, inst):
        """tidy prior: bar lined up with the box's long side"""
        return dict(rel_yaw=0.0)

    def oracle_params(self, inst):
        return dict(rel_yaw=self._signed(inst, np.radians(inst["fit_angle_deg"])))

    def blind_params(self, inst, attempt, rng):
        """no history: the prior again with a little jitter in angle and position"""
        return dict(rel_yaw=float(rng.normal(0.0, np.radians(4.0))), dx=float(rng.normal(0, 0.003)),
                    dy=float(rng.normal(0, 0.003)))

    def adapt_params(self, inst, history):
        """Scripted learner on F2 feedback.

        Knows the box's inner size and the bar's width only through noisy perception (inst['_perc'],
        sigma = PERC_SIGMA), not the bar's length. Every 'X cm longer than the inside along the long / short side
        at angle d' gives an equation for the bar length L (L cos d + W sin d = a + X, or L sin d + W cos d = b + X);
        'spare' values give the same with a negative X. It takes the largest length estimate (conservative) and turns
        the bar to the angle with the largest clearance for that length; if the last attempt was already at a
        feasible angle and only missed the centre, it keeps the angle."""
        a, b, W = inst["_perc"]
        Ls = []
        for h in history:
            fb = h["outcome"].get("feedback") or {}
            if h["outcome"].get("failure") != "rim":
                continue
            d = np.radians(fb["rel_deg"])
            c, s = np.cos(d), np.sin(d)
            if c > 0.3:
                x = fb["over_long_cm"] if fb["over_long_cm"] is not None else -fb["spare_long_cm"]
                Ls.append((a + x / 100 - W * s) / c)
            if s > 0.3:
                x = fb["over_short_cm"] if fb["over_short_cm"] is not None else -fb["spare_short_cm"]
                Ls.append((b + x / 100 - W * c) / s)
        L_est = max(Ls) if Ls else a + 0.015
        th, m, _ = best_angle(L_est, W, a, b)
        return dict(rel_yaw=self._signed(inst, th), L_est=float(L_est))

    def random_yaw_params(self, inst, history, rng):
        """sensitivity variant: a random orientation each retry, no history"""
        return dict(rel_yaw=float(rng.uniform(-np.pi / 2, np.pi / 2)))

    EXTRA_KINDS = ("random_yaw",)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        if kind not in self.EXTRA_KINDS:
            return super().run_scripted(kind, inst, k=k, seed=seed, on_attempt=on_attempt)
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            params = self.default_params(inst) if a == 0 else self.random_yaw_params(inst, history, rng)
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"]:
                break
        return history


# ---------------------------------------------------------------- physical validity checks
def physics_check(seed0=1000, n=10, out=None):
    """Release angles 0 / 90 deg (must fail), +-best angle (must succeed) and best +- 6 deg (report)."""
    import json
    task = DiagonalFit()
    rows = []
    for seed in range(seed0, seed0 + n):
        inst = task.sample_instance(seed)
        th = np.radians(inst["fit_angle_deg"])
        for name, rel, expect in [("aligned", 0.0, False), ("crossed", np.pi / 2, False), ("best+", th, True),
                                  ("best-", -th, True), ("best-6", th - np.radians(6), None),
                                  ("best+6", th + np.radians(6), None)]:
            task.reset_instance(inst)
            task.execute(inst, dict(rel_yaw=float(rel)))
            o = task.outcome(inst)
            rows.append(dict(seed=seed, case=name, rel_deg=float(np.degrees(rel)), success=o["success"], expect=expect,
                             failure=o["failure"], height_cm=o["height_cm"], tilt_deg=o["tilt_deg"], detail=o["detail"],
                             inner_cm=[round(100 * v, 2) for v in inst["inner"]], bar_cm=round(100 * inst["bar_len"], 2),
                             fit_margin_cm=round(inst["fit_margin_cm"], 2), fit_window_deg=round(inst["fit_window_deg"], 1)))
            flag = "" if expect is None or expect == o["success"] else "  <-- MISMATCH"
            print(f"seed {seed} {name:8s} rel={np.degrees(rel):+6.1f} succ={o['success']} h={o['height_cm']} "
                  f"tilt={o['tilt_deg']} margin={inst['fit_margin_cm']:.2f}cm win={inst['fit_window_deg']:.1f}{flag}",
                  flush=True)
        if out:
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
    bad = [r for r in rows if r["expect"] is not None and r["expect"] != r["success"]]
    print("total", len(rows), "mismatches", len(bad))
    return rows


def window_check(seed0=1000, n=6, out=None):
    """Release angle swept around the best angle (-16..+16 deg, step 2), bar centred: the physical fitting window
    against the geometric clearance."""
    import json
    task = DiagonalFit()
    rows = []
    for seed in range(seed0, seed0 + n):
        inst = task.sample_instance(seed)
        th = inst["fit_angle_deg"]
        L, (a, b) = inst["bar_len"], inst["inner"]
        for d in range(-16, 17, 2):
            rel = np.radians(th + d)
            task.reset_instance(inst)
            task.execute(inst, dict(rel_yaw=float(rel)))
            o = task.outcome(inst)
            m1, m2 = margins(L, BAR_W, a, b, rel)
            rows.append(dict(seed=seed, d_deg=d, rel_deg=round(th + d, 1), success=o["success"],
                             clearance_mm=round(1000 * float(min(m1, m2)), 1)))
        print(f"seed {seed} best={th:.1f} " + " ".join(f"{r['d_deg']:+d}:{'S' if r['success'] else '.'}"
                                                   for r in rows if r["seed"] == seed), flush=True)
        if out:
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
    return rows


def robustness_check(seed0=1000, n=6, out=None):
    """Execution errors at the best angle: release off-centre along the box's long / short axis, angle off by
    +-4 / +-8 deg, released 3 cm above the rim."""
    import json
    task = DiagonalFit()
    rows = []
    cases = [("best", {}), ("dx4", dict(dx=0.004)), ("dy4", dict(dy=0.004)), ("dx7", dict(dx=0.007)),
             ("dy7", dict(dy=0.007)), ("ang+4", dict(dang=4)), ("ang-4", dict(dang=-4)), ("ang+8", dict(dang=8)),
             ("ang-8", dict(dang=-8)), ("drop3cm", dict(drop=0.03)), ("dx4_ang4_drop3", dict(dx=0.004, dang=4, drop=0.03))]
    for seed in range(seed0, seed0 + n):
        inst = task.sample_instance(seed)
        th = inst["fit_angle_deg"]
        for name, c in cases:
            params = dict(rel_yaw=float(np.radians(th + c.get("dang", 0))))
            params.update({k: v for k, v in c.items() if k != "dang"})
            task.reset_instance(inst)
            task.execute(inst, params)
            o = task.outcome(inst)
            rows.append(dict(seed=seed, case=name, success=o["success"], detail=o["detail"],
                             fit_margin_cm=round(inst["fit_margin_cm"], 2)))
            print(f"seed {seed} {name:15s} succ={o['success']} | {o['detail'][:160]}", flush=True)
        if out:
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
    for name, _ in cases:
        r = [x["success"] for x in rows if x["case"] == name]
        print(f"{name:15s} {sum(r)}/{len(r)}")
    return rows


if __name__ == "__main__":
    # python -m lffbench.tasks.l4_diagonal physics results/l4_diagonal_fit/physics_check.json
    import sys
    # python -m lffbench.tasks.l4_diagonal window results/l4_diagonal_fit/window_check.json
    # python -m lffbench.tasks.l4_diagonal robust results/l4_diagonal_fit/robustness_check.json
    if sys.argv[1] == "physics":
        physics_check(out=sys.argv[2] if len(sys.argv) > 2 else None,
                      n=int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    elif sys.argv[1] == "window":
        window_check(out=sys.argv[2] if len(sys.argv) > 2 else None, n=int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    elif sys.argv[1] == "robust":
        robustness_check(out=sys.argv[2] if len(sys.argv) > 2 else None, n=int(sys.argv[3]) if len(sys.argv) > 3 else 6)
