"""L4-C nest_order: nest three cups one inside another; the tallest cup is not the widest.

Scene: three open round cups (red, green, blue; 1.5 mm walls leaning outwards by TAPER_DEG = 5 deg, like slightly
flared tumblers) stand apart on the table. Each cup has its own bottom width and its own height (re-drawn per
instance, 3.5-10 cm tall). Cups are resized per instance by editing their geoms in sim.model (wall segments +
floor), so one env serves every instance (same technique as l4_fit_container).

Trap (common sense): "the bigger cup goes outside", and a cup looks big mostly by its height. On (1 - P_AGREE) of
the instances the tallest cup is the middle one by width: it is at least INV_DH = 3.5 cm taller than the widest cup.
Their openings then differ by only 3-9 mm in diameter, and from the wrist camera at its start height the taller
one's opening is closer to the camera and looks about as big (about half the instances: bigger); from the front
camera the taller one looks clearly bigger. The narrowest cup is the shortest, so the very first move of the naive
order (widest cup into the tallest one) already fails, out on the table.

A variant where the innermost cup was taller than the middle one (and the widest cup the tallest) was dropped
2026-10-08: its failure (middle cup on the rim of the innermost cup, both inside the outer one) looked like a nest
from both cameras -- the same flaw as the 14-deg version.

What the failure looks like (the point of this version, user review 2026-10-08): which cup fits into which is
decided by the bottoms (same flare for all cups), and the sizes are drawn so that a cup whose bottom is too wide
does not go in at all -- it stays on the other cup's rim or wedges in the opening, its bottom at most STUCK_MAX
= 0.5 cm below the rim. So a wrong placement leaves a cup standing on top of another one, its whole body above the
rim: visible in both cameras without any text feedback. A cup that fits goes down to the floor (centring tolerance
>= TOL_MIN = 7.5 mm thanks to the flare); in the right nest the taller, narrower cups stick out of the shorter,
wider ones, but each stands inside the next.

Prior (naive): tallest cup outermost, then the next tallest. Learner: a cup that ended up on top of another cup
must go outside it; swap that pair. Feedback F1 (default) says only failed / succeeded; F2 (ablation) describes
which cup stands in / on which, in words, with no sizes or positions.

Only move_to / open / close are needed: grasp a cup across its upper wall, carry it over the other cup, release it
just above that cup's rim.
"""
import itertools

import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, register_generated
from ..skills import _rot_err
from ..task_base import LFFTask, register_task

N_SEG = 20  # wall segments per cup (regular polygon)
TAN_SEG = float(np.tan(np.pi / N_SEG))
COS_SEG = float(np.cos(np.pi / N_SEG))
TAPER_DEG = 5.0  # wall lean (from vertical)
TAPER = float(np.radians(TAPER_DEG))
TT = float(np.tan(TAPER))
WALL_T = 0.0015  # wall thickness (normal to the wall)
WALL_TH = WALL_T / float(np.cos(TAPER))  # horizontal wall thickness
FLOOR_T = 0.003
Z0 = 0.0003  # walls start this far above the cup's underside (the floor disc carries the cup)
H_RANGE = (0.035, 0.100)  # total cup height
RB_WIDE = (0.023, 0.030)  # bottom outer apothem of the cup with the widest bottom
STEP = (0.0055, 0.0075)  # bottom apothem difference between neighbours in the true order
RIM_MAX = 0.033  # outer apothem of the widest opening (gripper opens 7.9 cm)
CLEAR_MIN = 0.003  # radial gap (worst case) of a fitting cup standing on the other cup's floor
TOL_MIN = 0.0075  # a fitting cup goes in if its bottom lands within this of the other cup's axis
STUCK_MAX = 0.005  # a too-wide cup's bottom stops no deeper than this below the other cup's rim (5-deg walls
# self-lock: a cup pressed 1.2 cm into the opening wedged, and the narrow cup tipped over with it)
INV_DH = 0.035  # in the swapped pair the narrower cup is taller by at least this
KEEP_DH = 0.010  # in an unswapped pair the wider cup is taller by at least this
P_AGREE = 0.15  # share of instances whose height order is the right nesting order
SEAT_H = 0.010  # "standing on the floor": bottom less than this above the other cup's floor
SEAT_TILT = 15.0  # deg; a tall cup in a short, wider one can lean a little
WRIST_CAM_H = 0.367  # wrist camera height above the table at the start pose (measured 2026-10-08)
COLORS = {"red": (0.78, 0.22, 0.20, 1.0), "green": (0.28, 0.62, 0.32, 1.0), "blue": (0.22, 0.42, 0.80, 1.0)}
CUP_NAMES = ("cup_1", "cup_2", "cup_3")
DENSITY = 600.0
# execution
GRASP_BELOW_RIM = 0.012  # gripper site this far below the rim when grasping
Z_CARRY = envs.TABLE_Z + 0.24
DROP = 0.006  # held cup's bottom this far above the receiving rim at release
FINGER_CLEAR = 0.015  # gripper site at least this far above any rim around the target at release: fingers opened
# inside a taller outer cup press on its wall and carry it away on the way up (seed 1009, 2026-10-08)
PARK = (-0.45, 0.0, envs.TABLE_Z + 0.35)
H_NOISE = 0.03  # relative noise of the blind policy's height judgement


# ---------------------------------------------------------------- cup geometry
def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _seg_quat(phi):
    """segment frame: about z by phi, then tilted about its own y (tangential) axis by TAPER, so that its local z
    (along the wall) leans outwards and its local x is the wall's outward normal"""
    qz = np.array([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)])
    qy = np.array([np.cos(TAPER / 2), 0.0, np.sin(TAPER / 2), 0.0])
    return _quat_mul(qz, qy)


def seg_half_len(rb, H):
    """tangential half length of a wall segment: one polygon side at the rim (+0.2 mm overlap)"""
    return (rb + TT * H) * TAN_SEG + 0.0002


def rim_radius(rb, H):
    """outer apothem of the opening"""
    return rb + TT * H


def inner_radius(rb, z):
    """inner apothem at height z above the cup's underside"""
    return rb + TT * z - WALL_TH


def bottom_reach(rb, H):
    """largest horizontal radius of the cup's outside at its bottom (the segment tips stick out a little beyond
    the polygon because every segment is as wide as a rim-level polygon side)"""
    return float(np.hypot(rb + TT * Z0, seg_half_len(rb, H)))


def fit_clearance(rb_in, H_in, rb_out):
    """radial clearance (worst case over relative yaw) of cup `in` standing on the floor of cup `out`;
    negative = does not go down to the floor. The flare is the same for both cups, so the bottom is the tightest
    height."""
    return inner_radius(rb_out, FLOOR_T) - bottom_reach(rb_in, H_in)


def centring_tol(rb_in, H_in, rb_out, H_out):
    """how far (worst case) cup `in`'s bottom centre may be off cup `out`'s axis when it is lowered in at the rim
    and still get inside (the flared wall then guides it down)"""
    return inner_radius(rb_out, H_out) - bottom_reach(rb_in, H_in)


def stop_height(rb_in, rb_out, H_out):
    """height (above cup `out`'s underside) where a too-wide cup `in`, lowered centred, stops: where its bottom
    meets the sloping inner wall, or on the rim if that is higher than the cup"""
    return min((rb_in - rb_out + WALL_TH) / TT, H_out)


def _cup_parts(rb, H):
    """[(pos, half size, quat)] of the wall segments in the cup body frame (origin at mid height)."""
    zc = (Z0 + H) / 2
    hs = (H - Z0) / (2 * np.cos(TAPER))
    rc = rb + TT * zc - WALL_T / 2 * np.cos(TAPER)
    zz = zc + WALL_T / 2 * np.sin(TAPER) - H / 2
    hl = seg_half_len(rb, H)
    out = []
    for k in range(N_SEG):
        phi = 2 * np.pi * k / N_SEG
        out.append(((rc * np.cos(phi), rc * np.sin(phi), zz), (WALL_T / 2, hl, hs), _seg_quat(phi)))
    return out


def _floor_part(rb, H):
    return (0.0, 0.0, -H / 2 + FLOOR_T / 2), (rb - 0.0003, FLOOR_T / 2, 0.0), np.array([1.0, 0.0, 0.0, 0.0])


def cup_xml(model_name, rgba):
    """Open flared cup: N_SEG tilted wall boxes + a cylinder floor (sizes are overwritten per instance)."""
    geoms = []
    rb, H = RB_WIDE[1], H_RANGE[1]
    for k, (p, s, q) in enumerate(_cup_parts(rb, H)):
        geoms.append(f'        <geom name="{model_name}_g{k}" type="box" pos="{_fmt(p)}" size="{_fmt(s)}" '
                     f'quat="{_fmt(q)}" rgba="{_fmt(rgba)}" density="{DENSITY:.0f}" friction="0.8 0.005 0.0001" '
                     f'{_SOLID} group="0" />')
        geoms.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" quat="{_fmt(q)}" rgba="{_fmt(rgba)}" '
                     f'conaffinity="0" contype="0" group="1" />')
    fp, fs, _ = _floor_part(rb, H)
    geoms.append(f'        <geom name="{model_name}_g{N_SEG}" type="cylinder" pos="{_fmt(fp)}" size="{_fmt(fs[:2])}" '
                 f'rgba="{_fmt(rgba)}" density="{DENSITY:.0f}" friction="0.8 0.005 0.0001" {_SOLID} group="0" />')
    geoms.append(f'        <geom type="cylinder" pos="{_fmt(fp)}" size="{_fmt(fs[:2])}" rgba="{_fmt(rgba)}" '
                 f'conaffinity="0" contype="0" group="1" />')
    return _wrap(model_name, "\n".join(geoms), H / 2, rim_radius(rb, H) / COS_SEG)


def disable_midphase(env):
    """The cups are resized at run time by editing geom pos / size in sim.model, but MuJoCo's per-body bounding
    volume hierarchy (bvh_aabb, used by the collision mid-phase) is computed once at compile time for the registered
    maximum size. A resized wall that lies outside its compiled leaf box is silently skipped: the gripper fingers
    passed straight through the walls of the smaller cups (found 2026-10-07). Without the mid-phase MuJoCo tests all
    geom pairs of two nearby bodies (still filtered by the run-time geom_rbound set in _reshape_cup)."""
    import mujoco
    m = env.sim.model._model
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)


def kendall(a, b):
    """number of pairs ordered differently in the two orders"""
    pa, pb = {c: k for k, c in enumerate(a)}, {c: k for k, c in enumerate(b)}
    return sum((pa[x] < pa[y]) != (pb[x] < pb[y]) for x, y in itertools.combinations(a, 2))


@register_task
class NestOrder(LFFTask):
    name = "l4_nest_order"
    level = "L4"
    category = "geometry_inference"
    capabilities = ("Perceive", "Reason", "Plan")
    instruction = "Nest the red, green and blue cups one inside another."
    instruction_indirect = "Put the three cups one inside another, like a set of nesting cups."

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        cats = [register_generated(f"LffNestCup{c.capitalize()}", cup_xml(f"lff_nest_cup_{c}", COLORS[c]))
                for c in COLORS]
        regions = [(-0.25, -0.15), (-0.05, 0.0), (-0.25, 0.15)]
        objs = [(n, cat, f"{n}_region", (x - 0.01, y - 0.01, x + 0.01, y + 0.01))
                for n, cat, (x, y) in zip(CUP_NAMES, cats, regions)]
        return write_bddl(self.name, self.instruction, objs)

    @staticmethod
    def _heights_ok(case, h):
        """h[s] = height of the cup with the s-th narrowest bottom (0 = innermost in the true nest)"""
        if case == "agree":  # taller = wider: the height order is the nesting order
            return h[2] >= h[1] + KEEP_DH and h[1] >= h[0] + KEEP_DH
        if case == "swap_wide":  # the middle cup is the tallest, the narrowest one the shortest
            return h[1] >= h[2] + INV_DH and h[2] >= h[0] + KEEP_DH
        raise ValueError(case)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        u = rng.random()
        case = "agree" if u < P_AGREE else "swap_wide"
        for _ in range(200000):
            heights = [float(rng.uniform(*H_RANGE)) for _ in range(3)]  # by true size: 0 = narrowest bottom
            if not self._heights_ok(case, heights):
                continue
            rb3 = float(rng.uniform(*RB_WIDE))
            rb2 = rb3 - float(rng.uniform(*STEP))
            rb1 = rb2 - float(rng.uniform(*STEP))
            rbs = [rb1, rb2, rb3]
            rims = [rim_radius(rb, h) for rb, h in zip(rbs, heights)]
            if max(rims) > RIM_MAX:
                continue
            ok = True
            for p, q in itertools.permutations(range(3), 2):  # p into q
                if p < q:
                    ok &= (fit_clearance(rbs[p], heights[p], rbs[q]) >= CLEAR_MIN
                           and centring_tol(rbs[p], heights[p], rbs[q], heights[q]) >= TOL_MIN)
                else:
                    ok &= heights[q] - stop_height(rbs[p], rbs[q], heights[q]) <= STUCK_MAX
            if ok:
                break
        else:
            raise RuntimeError("could not sample cup sizes")
        xy = []
        while len(xy) < 3:
            xy = []
            for _k in range(3):
                for _t in range(200):
                    p = [float(rng.uniform(-0.27, 0.02)), float(rng.uniform(-0.17, 0.17))]
                    if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= 0.14 for q in xy):
                        xy.append(p)
                        break
        # the colour is fixed per object (cup_1 red, cup_2 green, cup_3 blue: one generated class each); shuffle
        # which object gets which size / height / position
        perm = [int(i) for i in rng.permutation(3)]  # perm[size index] = object slot
        cups = [None] * 3
        for s in range(3):
            cups[perm[s]] = dict(name=CUP_NAMES[perm[s]], color=list(COLORS)[perm[s]], rb=rbs[s], height=heights[s],
                                 rim=rims[s], xy=xy[s], size_rank=s)
        order = sorted(range(3), key=lambda j: -cups[j]["rb"])  # outermost first (indices into cups)
        naive = sorted(range(3), key=lambda j: -cups[j]["height"])
        assert (naive == order) == (case == "agree"), (seed, case)
        by = lambda s: perm[s]  # noqa: E731  object index of size rank s
        return dict(seed=int(seed), case=case, cups=cups, order=order, naive_order=naive,
                    bottom_cm=[round(200 * c["rb"], 2) for c in cups],
                    opening_cm=[round(200 * c["rim"], 2) for c in cups],
                    height_cm=[round(100 * c["height"], 2) for c in cups],
                    # relative size of each opening in the wrist camera at its start height (1 = largest)
                    wrist_apparent=[round(float(v), 3) for v in (lambda a: a / a.max())(np.array(
                        [c["rim"] / (WRIST_CAM_H - c["height"]) for c in cups]))],
                    clear_mm=[round(1000 * fit_clearance(rbs[s], heights[s], rbs[s + 1]), 2) for s in range(2)],
                    tol_mm=[round(1000 * centring_tol(rbs[s], heights[s], rbs[s + 1], heights[s + 1]), 2)
                            for s in range(2)],
                    # wrong neighbour pairs: how deep (cm) the wider cup's bottom gets below the narrower cup's rim
                    stuck_cm={f"{by(s + 1)}in{by(s)}": round(100 * (heights[s] - stop_height(
                        rbs[s + 1], rbs[s], heights[s])), 2) for s in range(2)})

    def _reshape_cup(self, name, rb, H):
        m = self.env.sim.model
        o = self.env.objects_dict[name]
        parts = _cup_parts(rb, H) + [_floor_part(rb, H)]
        for names in (o.contact_geoms, o.visual_geoms):
            assert len(names) == len(parts), (name, len(names))
            for k, n in enumerate(names):
                g = m.geom_name2id(n)
                pos, size, quat = parts[k]
                m.geom_pos[g] = pos
                m.geom_size[g] = size
                m.geom_quat[g] = quat
                if k < N_SEG:
                    m.geom_rbound[g] = float(np.linalg.norm(size))
                    m.geom_aabb[g] = [0, 0, 0, *size]
                else:
                    m.geom_rbound[g] = float(np.hypot(size[0], size[1]))
                    m.geom_aabb[g] = [0, 0, 0, size[0], size[0], size[1]]
        # mass / inertia of the resized cup: conical shell + disc floor
        bid = int(m.geom_bodyid[m.geom_name2id(o.contact_geoms[0])])
        slant = (H - Z0) / np.cos(TAPER)
        r_mid = rb + TT * (H + Z0) / 2
        wall_m = DENSITY * 2 * np.pi * r_mid * WALL_T * slant
        floor_m = DENSITY * np.pi * rb ** 2 * FLOOR_T
        mass = wall_m + floor_m
        # wall centroid height (from the underside): area-weighted, r grows linearly with z
        r0, r1 = rb + TT * Z0, rb + TT * H
        z_w = Z0 + (H - Z0) * (r0 + 2 * r1) / (3 * (r0 + r1))
        z_cm = (wall_m * z_w + floor_m * FLOOR_T / 2) / mass
        Izz = wall_m * r_mid ** 2 + floor_m * rb ** 2 / 2
        Ixx = (wall_m * (r_mid ** 2 / 2 + (H - Z0) ** 2 / 12 + (z_w - z_cm) ** 2)
               + floor_m * (rb ** 2 / 4 + (FLOOR_T / 2 - z_cm) ** 2))
        m.body_mass[bid] = mass
        m.body_inertia[bid] = [Ixx, Ixx, Izz]
        m.body_ipos[bid] = [0.0, 0.0, z_cm - H / 2]
        m.body_iquat[bid] = [1.0, 0.0, 0.0, 0.0]

    def apply_instance(self, inst):
        disable_midphase(self.env)
        for c in inst["cups"]:
            self._reshape_cup(c["name"], c["rb"], c["height"])
            envs.set_obj_pose(self.env, c["name"], [*c["xy"], envs.TABLE_Z + c["height"] / 2 + 0.0005])

    # ------------------------------------------------------------ state
    def _cup_state(self, inst, j):
        c = inst["cups"][j]
        bid = envs.body_id(self.env, c["name"])
        p = self.env.sim.data.body_xpos[bid].copy()
        R = self.env.sim.data.body_xmat[bid].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        bottom = p - R[:, 2] * c["height"] / 2  # centre of the floor's underside
        rim = p + R[:, 2] * c["height"] / 2
        return dict(p=p, tilt=tilt, bottom=bottom, rim=rim, floor_top=bottom[2] + FLOOR_T * R[2, 2])

    def relations(self, inst):
        """For each cup: which cup it stands in / on (if any), how high, seated or not."""
        st = [self._cup_state(inst, j) for j in range(3)]
        rel = []
        for j in range(3):
            sj = st[j]
            best = None
            for i in range(3):
                if i == j:
                    continue
                ci, si = inst["cups"][i], st[i]
                if si["tilt"] > 45:
                    continue
                d = float(np.hypot(*(sj["bottom"][:2] - si["p"][:2])))
                if d > rim_radius(ci["rb"], ci["height"]) + 0.01:
                    continue  # not above / in cup i
                hb = float(sj["bottom"][2] - si["floor_top"])  # j's bottom above i's floor
                # (the last test keeps a cup from being reported as standing in the cup it contains)
                if hb < -0.004 or hb > ci["height"] + 0.02 or sj["bottom"][2] < si["bottom"][2] + FLOOR_T / 2:
                    continue
                if best is None or hb < best[1]:
                    best = (i, hb, d)
            if best is None:
                on_table = sj["bottom"][2] < envs.TABLE_Z + 0.01 and sj["tilt"] < 30
                rel.append(dict(cup=j, on=None, table=bool(on_table), tilt=sj["tilt"], xy=sj["p"][:2].tolist(),
                                fallen=bool(sj["tilt"] > 45)))
                continue
            i, hb, d = best
            ci = inst["cups"][i]
            seated = hb < SEAT_H and sj["tilt"] < SEAT_TILT
            on_rim = hb > ci["height"] - FLOOR_T - 0.004
            rel.append(dict(cup=j, on=i, height=hb, seated=bool(seated), on_rim=bool(on_rim), tilt=sj["tilt"],
                            offset=d, xy=sj["p"][:2].tolist(), fallen=bool(sj["tilt"] > 45)))
        return rel

    def outcome(self, inst):
        envs.settle(self.env, 40)
        cups = inst["cups"]
        rel = self.relations(inst)
        held = [j for j in range(3) if self.sk is not None and self.sk.holding(cups[j]["name"])]
        col = [c["color"] for c in cups]
        seated_in = {r["cup"]: r["on"] for r in rel if r.get("on") is not None and r["seated"]}
        # success: a chain outer <- mid <- inner, all seated and upright, the outer cup standing on the table
        success = False
        for a, b, c in itertools.permutations(range(3)):
            if seated_in.get(b) == a and seated_in.get(c) == b and _on_table_upright(rel, a):
                success = True
        success = bool(success and not held)
        fb = dict(seated=[[j, i] for j, i in seated_in.items()], blocked=[], offcentre=[], held=held)
        lines = []
        res = dict(success=success, relations=rel, feedback=fb)
        if held:
            lines.append(f"The {col[held[0]]} cup is still in the gripper.")
        if success:
            a = [j for j in range(3) if j not in seated_in][0]
            b = [j for j, i in seated_in.items() if i == a][0]
            c = [j for j, i in seated_in.items() if i == b][0]
            res["detail"] = (f"All three cups are nested: the {col[c]} cup stands inside the {col[b]} cup, "
                             f"which stands inside the {col[a]} cup.")
            res["failure"] = None
            return res
        # bottom-up description
        st = {r["cup"]: self._cup_state(inst, r["cup"]) for r in rel}
        for r in sorted(rel, key=lambda r: st[r["cup"]]["bottom"][2]):
            j = r["cup"]
            if j in held:
                continue
            if r.get("on") is None:
                if r["fallen"]:
                    lines.append(f"The {col[j]} cup has fallen over.")
                elif r["table"]:
                    lines.append(f"The {col[j]} cup is standing on the table, not in another cup.")
                else:
                    lines.append(f"The {col[j]} cup is neither standing on the table nor inside another cup.")
                continue
            i = r["on"]
            if r["seated"]:
                lines.append(f"The {col[j]} cup is inside the {col[i]} cup, standing on its bottom.")
                continue
            # did not go in. The text says only what a camera shows (no sizes); the structured field keeps the
            # nominal verdict for the scripted learner
            tilt = f", tilted about {r['tilt']:.0f} degrees" if r["tilt"] >= 8 else ""
            if r["on_rim"]:
                lines.append(f"The {col[j]} cup is sitting on the rim of the {col[i]} cup, not inside it{tilt}.")
            else:
                lines.append(f"The {col[j]} cup is stuck in the opening of the {col[i]} cup, not down inside "
                             f"it{tilt}.")
            over = cups[j]["rb"] - cups[i]["rb"]
            if over > 0:
                fb["blocked"].append([j, i, round(100 * r["height"], 1)])
            else:
                fb["offcentre"].append([j, i, round(100 * r["height"], 1)])
        res["detail"] = " ".join(lines) if lines else "The cups are not nested."
        res["failure"] = "blocked" if fb["blocked"] else ("offcentre" if fb["offcentre"] else
                                                          ("held" if held else "incomplete"))
        return res

    # ------------------------------------------------------------ execution
    def execute(self, inst, params):
        """params: order = [outer, middle, inner] (indices into inst['cups']). Puts the middle cup into the outer
        one; if it went in (judged from the scene, as an agent would see it), puts the inner cup into the middle
        one. Optional (robustness checks): dxy = (dx, dy) release offset (m), drop (m), grasp_below (m),
        carry_speed (m/s or None = as fast as the controller allows)."""
        o, mid, inn = [int(v) for v in params["order"]]
        ok = self._move_cup(inst, mid, o, params)
        if ok:
            self._move_cup(inst, inn, mid, params)
        self.sk.move_to(list(PARK), yaw=0.0)

    def _move_cup(self, inst, j, i, params):
        sk = self.sk
        sj = self._cup_state(inst, j)
        x, y = sj["p"][:2]
        rim = sj["rim"][2]
        sk.set_gripper(False, steps=5)
        sk.move_to([x, y, Z_CARRY], yaw=0.0)
        sk.move_to([x, y, rim + 0.06], tol=0.004)
        sk.move_to([x, y, rim - params.get("grasp_below", GRASP_BELOW_RIM)], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([x, y, Z_CARRY], speed=params.get("carry_speed", 0.3))
        si = self._cup_state(inst, i)
        dx, dy = params.get("dxy", (0.0, 0.0))
        tx, ty = si["p"][0] + dx, si["p"][1] + dy
        drop = params.get("drop", DROP)
        sk.move_to([tx, ty, Z_CARRY], tol=0.004)
        off = self._cup_state(inst, j)["bottom"] - sk.eef_pos()
        # approach above the highest rim there (the target cup may stand inside a taller one)
        top = max(s["rim"][2] for s in (self._cup_state(inst, q) for q in range(3) if q != j)
                  if np.hypot(s["p"][0] - tx, s["p"][1] - ty) < 0.04)
        sk.move_to([tx - off[0], ty - off[1], max(top, si["rim"][2]) + drop + 0.03 - off[2]], tol=0.003)
        z_rel = si["rim"][2] + drop
        if top > si["rim"][2] + 0.002:  # the target stands inside a taller cup: let go above that rim
            z_rel = max(z_rel, top + FINGER_CLEAR - (inst["cups"][j]["height"]
                                                     - params.get("grasp_below", GRASP_BELOW_RIM)))
        tgt = self._servo_cup(inst, j, [tx, ty], z_rel)
        sk.grip = -1.0
        self._hold_at(tgt, 12)
        sk.move_to([tgt[0], tgt[1], Z_CARRY], speed=0.3)
        envs.settle(self.env, 20)
        r = [q for q in self.relations(inst) if q["cup"] == j][0]
        return bool(r.get("on") == i and r["seated"])

    def _hold_at(self, target, steps):
        sk = self.sk
        R = sk._R()
        for _ in range(steps):
            sk._act(np.asarray(target) - sk.eef_pos(), _rot_err(R, sk.eef_mat()), sk.grip)

    def _servo_cup(self, inst, j, xy, z_bottom, max_steps=60, tol=0.0015):
        """closed loop on the held cup's bottom centre (scripted policies only). Stops where the cup is held up
        (a too-wide cup on / in another cup's rim) instead of pressing on: a person lets go there too, and
        pressing on tips the cups over."""
        sk = self.sk
        R = sk._R()
        goal = np.array([xy[0], xy[1], z_bottom])
        tgt = e_prev = sk.eef_pos()
        zs = []
        for k in range(max_steps):
            b, e = self._cup_state(inst, j)["bottom"], sk.eef_pos()
            if k >= 8 and np.linalg.norm(b - goal) < tol and np.linalg.norm(e - e_prev) < 0.0005:
                break
            zs.append(b[2])
            if k >= 12 and b[2] - goal[2] > 0.003 and zs[-6] - b[2] < 0.0005:
                return e  # blocked: release here
            tgt = goal - (b - e)
            sk._act(tgt - e, _rot_err(R, sk.eef_mat()), sk.grip)
            e_prev = e
        return tgt

    # ------------------------------------------------------------ scripted policies
    def default_params(self, inst):
        """prior: the bigger-looking cup goes outside, and the cups look big by their height: tallest outermost"""
        return dict(order=list(inst["naive_order"]))

    def oracle_params(self, inst):
        return dict(order=list(inst["order"]))

    def blind_params(self, inst, attempt, rng):
        """no history: the same height judgement again, with a little noise"""
        est = [c["height"] * (1 + rng.normal(0, H_NOISE)) for c in inst["cups"]]
        return dict(order=[int(i) for i in np.argsort(est)[::-1]])

    @staticmethod
    def constraints(history):
        """pairs (outer, inner) implied by the F2 feedback of earlier attempts"""
        pairs = set()
        for h in history:
            fb = h["outcome"].get("feedback") or {}
            for j, i in fb.get("seated", []):
                pairs.add((int(i), int(j)))  # j sits in i -> i is outside j
            for j, i, _ in fb.get("blocked", []):
                pairs.add((int(j), int(i)))  # j does not fit into i -> j must be outside i
            for j, i, _ in fb.get("offcentre", []):
                pairs.add((int(i), int(j)))  # j would fit into i
        return pairs

    def adapt_params(self, inst, history):
        """Scripted learner. It does not perceive the bottoms at all; it uses only which cup went into / stayed on
        top of which (visible in the camera images; the structured part of the outcome): every such relation gives
        an (outer, inner) pair; among the untried orders consistent with all pairs it takes the one closest (fewest
        swapped pairs) to its prior, the height order; ties go to the larger height agreement."""
        pairs = self.constraints(history)
        rims = [c["height"] for c in inst["cups"]]
        prior = list(inst["naive_order"])
        tried = {tuple(h["params"]["order"]) for h in history if (h["outcome"].get("feedback") or {}).get("blocked")}
        cands = []
        for order in itertools.permutations(range(3)):
            pos = {c: k for k, c in enumerate(order)}
            consistent = all(pos[a] < pos[b] for a, b in pairs)
            agree = sum(rims[order[k]] - rims[order[k + 1]] for k in range(2))
            cands.append((consistent, tuple(order) not in tried, -kendall(order, prior), agree, order))
        best = max(cands)
        return dict(order=list(best[4]))

    def random_order_params(self, inst, history, rng):
        """sensitivity variant: a random order each retry, no history"""
        return dict(order=[int(i) for i in rng.permutation(3)])

    EXTRA_KINDS = ("random_order",)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        if kind not in self.EXTRA_KINDS:
            return super().run_scripted(kind, inst, k=k, seed=seed, on_attempt=on_attempt)
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            params = self.default_params(inst) if a == 0 else self.random_order_params(inst, history, rng)
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"]:
                break
        return history


def _on_table_upright(rel, j):
    r = [q for q in rel if q["cup"] == j][0]
    return r.get("on") is None and r["table"] and r["tilt"] < 8.0 and not r["fallen"]


# ---------------------------------------------------------------- physical validity checks
def physics_check(seed0=1000, n=8, out=None):
    """All 6 orders on each instance; expect success iff the order is the true one."""
    import json
    task = NestOrder()
    rows = []
    for seed in range(seed0, seed0 + n):
        inst = task.sample_instance(seed)
        for order in itertools.permutations(range(3)):
            task.reset_instance(inst)
            task.execute(inst, dict(order=list(order)))
            o = task.outcome(inst)
            expect = list(order) == inst["order"]
            # for a cup that did not go in: how far (cm) its bottom is below the other cup's rim
            below_rim = [round(100 * (inst["cups"][r["on"]]["height"] - FLOOR_T - r["height"]), 2)
                         for r in o["relations"] if r.get("on") is not None and not r["seated"]]
            rows.append(dict(seed=seed, case=inst["case"], order=list(order), success=o["success"], expect=expect,
                             failure=o["failure"], detail=o["detail"], below_rim_cm=below_rim,
                             clear_mm=inst["clear_mm"], tol_mm=inst["tol_mm"], bottom_cm=inst["bottom_cm"],
                             opening_cm=inst["opening_cm"], height_cm=inst["height_cm"]))
            print(f"seed {seed} order {order} succ={o['success']} expect={expect} | {o['detail']}"
                  + ("" if o["success"] == expect else "  <-- MISMATCH"), flush=True)
        if out:
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
    print("total", len(rows), "mismatches", sum(r["success"] != r["expect"] for r in rows))
    return rows


def robustness_check(seed0=1000, n=6, out=None):
    """True order with execution errors: release off-centre (both moves) by 0-9 mm in a random direction, released
    3 cm above the rim, grasped lower, carried at full controller speed."""
    import json
    task = NestOrder()
    rows = []
    cases = [("centred", {}), ("off3", dict(off=0.003)), ("off5", dict(off=0.005)), ("off7", dict(off=0.007)),
             ("off9", dict(off=0.009)), ("drop3cm", dict(drop=0.03)), ("off4_drop3cm", dict(off=0.004, drop=0.03)),
             ("grasp2cm", dict(grasp_below=0.02)), ("fast_carry", dict(carry_speed=None))]
    for seed in range(seed0, seed0 + n):
        inst = task.sample_instance(seed)
        rng = np.random.default_rng(seed)
        for name, c in cases:
            a = rng.uniform(0, 2 * np.pi)
            off = c.get("off", 0.0)
            params = dict(order=list(inst["order"]), dxy=(off * np.cos(a), off * np.sin(a)))
            params.update({k: v for k, v in c.items() if k != "off"})
            task.reset_instance(inst)
            task.execute(inst, params)
            o = task.outcome(inst)
            rows.append(dict(seed=seed, case=name, success=o["success"], detail=o["detail"], tol_mm=inst["tol_mm"],
                             height_cm=inst["height_cm"]))
            print(f"seed {seed} {name:13s} succ={o['success']} | {o['detail']}", flush=True)
        if out:
            with open(out, "w") as f:
                json.dump(rows, f, indent=1)
    for name, _ in cases:
        r = [x["success"] for x in rows if x["case"] == name]
        print(f"{name:13s} {sum(r)}/{len(r)}")
    return rows


if __name__ == "__main__":
    # python -m lffbench.tasks.l4_nest physics results/l4_nest_order/physics_check.json [n]
    # python -m lffbench.tasks.l4_nest robust results/l4_nest_order/robustness_check.json [n]
    import sys
    n_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if sys.argv[1] == "physics":
        physics_check(out=sys.argv[2] if len(sys.argv) > 2 else None, n=n_arg or 8)
    elif sys.argv[1] == "robust":
        robustness_check(out=sys.argv[2] if len(sys.argv) > 2 else None, n=n_arg or 6)
