"""L3 card_edge: a thin card lying flat on a board cannot be pinched; slide it over the board's edge, then grasp it.

Category: strategy_switching ("this way of doing it cannot work"), the user's taxonomy L3 example verbatim:
"卡片贴着桌面捏不起来 -> 先把卡片滑到桌沿，再夹起来".

Scene: a wooden board (22 x 26 x 5 cm) on the table with a thin red card (9.0 x 5.6 x 0.2 cm, 10 g) lying flat on it,
long side roughly along x or along y (+-10 deg); a grey bin (16 x 16 cm inside) stands on the table beside the board.
The board stands in for the table edge of the user's example: the real table edge is out of reach (near edge 16 cm
from the robot base, side edges at |y| = 0.6 m) and the harness does not let the grasp point go below the table top.

Why the prior fails (measured): a top-down grasp across the card's short side. The Panda fingertips land on the
board on either side of the card, but when they close on its 2 mm edge the contact flips to "slide over the top" and
both fingertips ride up over the card and close above it: 0/36 lifts for 1.5 / 2 / 3 mm cards, grasp-point height
-6..+3 mm, xy jitter 5 mm, yaw jitter 0.1 rad (prototype, 2026-10-08). Retrying with small variations fails the same
way. (A 4 mm card could be pinched -- the reason the earlier `l3_flat_card_edge` was dropped.)

What must be learned from the failure: the card cannot be picked up while it lies flat -> change the strategy: press
the closed fingertips onto the card and slide it (move_to with the fingertips pressing on it) until part of it hangs
over the board's edge, then grasp the overhanging part with the fingertips below the card: then the fingers close on
the card's full edge and it lifts (18/18 for 2.5-3.5 cm of overhang in the prototype). More than half the card's
length over the edge and it tips off the board.

Protocol: cross (reset to the same instance before every attempt), k = 5.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, bin_xml, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

CARD_HALF = (0.045, 0.028, 0.001)  # 9.0 x 5.6 x 0.2 cm
CARD_MASS = 0.010
MU = 0.3  # card and board sliding friction (the fingers' 1.0 / 2.0 win in finger contacts: MuJoCo takes the max)
BOARD_HALF = (0.11, 0.13, 0.025)
BOARD_TOP = envs.TABLE_Z + 2 * BOARD_HALF[2]
BIN_IN = 0.08  # inner half width
BIN_WALL_H = 0.05
BIN_GAP = 0.06  # board edge to bin outer wall
BIN_T = 0.006
BIN_HALF_OUT = BIN_IN + BIN_T
TIP_BELOW_SITE = envs.FINGERTIP_BELOW_SITE
LOW_CLOSE = 0.02  # a close counts as a grasp try when the fingertips are at most this far above the card
OVERHANG = 0.03  # scripted target overhang (the grasp works for ~2.5-4.0 cm; the card tips off past ~4.5 cm)
GRASP_OUT = 0.022  # scripted grasp point: this far beyond the board edge (fingers must clear the edge)
DRAG_SPEED = 0.08


def card_xml(name, rgba=(0.85, 0.15, 0.12, 1.0)):
    g = (f'        <geom name="{name}_g0" type="box" size="{_fmt(CARD_HALF)}" rgba="{_fmt(rgba)}" mass="{CARD_MASS}" '
         f'friction="{MU} 0.005 0.0001" {_SOLID} group="0" condim="4"/>\n'
         f'        <geom type="box" size="{_fmt(CARD_HALF)}" rgba="{_fmt(rgba)}" contype="0" conaffinity="0" group="1"/>')
    return _wrap(name, g, CARD_HALF[2], float(np.hypot(CARD_HALF[0], CARD_HALF[1])))


def _yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


EDGES = ("forward", "backward", "left", "right")  # board edges in robot-centric words (+x, -x, +y, -y)


class CardSkills(Skills):
    """Skills that feed the task's per-step monitor and log every close of the gripper."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task
        self.closes = []

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()

    def set_gripper(self, close, steps=15):
        was_open = self.grip < 0
        before = self.task.close_context() if (close and was_open) else None
        rep = super().set_gripper(close, steps)
        if before is not None:
            before.update(held=self.holding("card_1"), width=round(self.gripper_width(), 4))
            self.closes.append(before)
        return rep


@register_task
class CardEdge(LFFTask):
    name = "l3_card_edge"
    level = "L3"
    category = "strategy_switching"
    capabilities = ("Perceive", "Reason", "Plan")
    protocol = "cross"
    instruction = "Pick up the thin red card lying on the wooden board and put it into the grey bin."
    instruction_indirect = "Put the flat red rectangle that lies on the wooden block into the grey container."

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        card = register_generated("LffThinCard", card_xml("lff_thin_card"))
        board = register_generated("LffCardBoard", box_xml("lff_card_board", BOARD_HALF, (0.62, 0.45, 0.27, 1.0),
                                                           friction=(MU, 0.005, 0.0001)), free=False)
        bin_ = register_generated("LffCardBin", bin_xml("lff_card_bin", (BIN_IN, BIN_IN), BIN_WALL_H, wall_t=BIN_T,
                                                        rgba=(0.55, 0.56, 0.60, 1.0)), free=False)
        objs = [("card_1", card, "card_region", (0.10, 0.20, 0.12, 0.22))]
        fx = [("board_1", board, "board_region", (-0.11, -0.01, -0.09, 0.01)),
              ("bin_1", bin_, "bin_region", (-0.11, -0.31, -0.09, -0.29))]
        return write_bddl(self.name, self.instruction, objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.board_bid = m.body_name2id(self.env.fixtures_dict["board_1"].root_body)
        self.bin_bid = m.body_name2id(self.env.fixtures_dict["bin_1"].root_body)
        self.inst = None
        self.mon = None

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        bx, by = float(rng.uniform(-0.14, -0.06)), float(rng.uniform(-0.08, 0.08))
        along = str(rng.choice(["x", "y"]))
        yaw = (0.0 if along == "x" else np.pi / 2) + float(rng.uniform(-np.radians(10), np.radians(10)))
        if along == "x":  # keep the whole card >= 3 cm (long axis) / 2 cm (short axis) inside the board
            off = [rng.uniform(-0.03, 0.03), rng.uniform(-0.07, 0.07)]
        else:
            off = [rng.uniform(-0.055, 0.055), rng.uniform(-0.05, 0.05)]
        side = 1.0 if rng.random() < 0.5 else -1.0
        bin_xy = [bx + float(rng.uniform(-0.04, 0.04)), by + side * (BOARD_HALF[1] + BIN_GAP + BIN_HALF_OUT)]
        return dict(seed=int(seed), board_xy=[bx, by], card_xy=[bx + float(off[0]), by + float(off[1])],
                    card_yaw=float(yaw), card_along=along, bin_xy=bin_xy, bin_side="left" if side > 0 else "right")

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        bx, by = inst["board_xy"]
        m.body_pos[self.board_bid] = [bx, by, envs.TABLE_Z + BOARD_HALF[2]]
        m.body_quat[self.board_bid] = [1.0, 0.0, 0.0, 0.0]
        cx, cy = inst["bin_xy"]
        m.body_pos[self.bin_bid] = [cx, cy, envs.TABLE_Z + (BIN_WALL_H + BIN_T) / 2]
        m.body_quat[self.bin_bid] = [1.0, 0.0, 0.0, 0.0]
        x, y = inst["card_xy"]
        envs.set_obj_pose(self.env, "card_1", [x, y, BOARD_TOP + CARD_HALF[2] + 0.0003], yaw=inst["card_yaw"])

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 15)
        self.sk = CardSkills(self.env, self, recorder=recorder)
        p = self.card_pose()
        self.mon = dict(start_xy=p["xy"].copy(), max_lift=0.0, max_overhang=0.0, fell=False)
        return self.sk

    # ------------------------------------------------------------ geometry
    def card_pose(self):
        p = envs.obj_pos(self.env, "card_1")
        return dict(xy=p[:2].copy(), z=float(p[2]), yaw=_yaw_of(envs.obj_quat(self.env, "card_1")))

    def card_corners(self):
        d = self.env.sim.data
        bid = envs.body_id(self.env, "card_1")
        R = d.body_xmat[bid].reshape(3, 3)
        h = np.array(CARD_HALF)
        return np.array([d.body_xpos[bid] + R @ (h * [sx, sy, 0.0]) for sx in (-1, 1) for sy in (-1, 1)])

    def overhang(self):
        """(how far the card sticks out beyond the board's edges (m, max over its corners), edge word)"""
        c = self.card_corners()[:, :2] - np.asarray(self.inst["board_xy"])
        prot = {"forward": c[:, 0].max() - BOARD_HALF[0], "backward": -c[:, 0].min() - BOARD_HALF[0],
                "left": c[:, 1].max() - BOARD_HALF[1], "right": -c[:, 1].min() - BOARD_HALF[1]}
        e = max(prot, key=prot.get)
        return float(prot[e]), e

    def on_board(self):
        p = self.card_pose()
        rel = np.abs(p["xy"] - np.asarray(self.inst["board_xy"]))
        return bool(rel[0] <= BOARD_HALF[0] and rel[1] <= BOARD_HALF[1] and abs(p["z"] - BOARD_TOP) < 0.012)

    def in_bin(self):
        p = self.card_pose()
        rel = np.abs(p["xy"] - np.asarray(self.inst["bin_xy"]))
        return bool(rel[0] <= BIN_IN and rel[1] <= BIN_IN and p["z"] < envs.TABLE_Z + BIN_T + BIN_WALL_H)

    # ------------------------------------------------------------ monitor
    def monitor_step(self):
        mon = self.mon
        if mon is None:
            return
        zmin = float(self.card_corners()[:, 2].min())
        mon["max_lift"] = max(mon["max_lift"], zmin - BOARD_TOP)
        if self.on_board() or abs(self.card_pose()["z"] - BOARD_TOP) < 0.012:
            mon["max_overhang"] = max(mon["max_overhang"], self.overhang()[0])
        if self.card_pose()["z"] < BOARD_TOP - 0.02 and not self.sk.holding("card_1"):
            mon["fell"] = True

    def close_context(self):
        """Where the gripper is relative to the card and the board just before it closes."""
        sk, p = self.sk, self.card_pose()
        e = sk.eef_pos()
        c, s = np.cos(p["yaw"]), np.sin(p["yaw"])
        rel = e[:2] - p["xy"]
        local = [float(rel @ [c, s]), float(rel @ [-s, c])]  # along the card's long / short side
        b = e[:2] - np.asarray(self.inst["board_xy"])
        beyond = {"forward": b[0] - BOARD_HALF[0], "backward": -b[0] - BOARD_HALF[0],
                  "left": b[1] - BOARD_HALF[1], "right": -b[1] - BOARD_HALF[1]}
        edge = max(beyond, key=beyond.get)
        ov, ov_edge = self.overhang()
        return dict(step=sk.n_steps, local=[round(v, 4) for v in local],
                    tips_above_card=round(float(e[2] - TIP_BELOW_SITE - (p["z"] + CARD_HALF[2])), 4),
                    beyond_edge=round(float(beyond[edge]), 4), beyond_edge_name=edge,
                    card_on_board=self.on_board(), overhang=round(ov, 4), overhang_edge=ov_edge,
                    width_before=round(sk.gripper_width(), 4))

    # ------------------------------------------------------------ outcome / F2
    def outcome(self, inst):
        envs.settle(self.env, 30)
        sk, mon = self.sk, self.mon
        p = self.card_pose()
        held = bool(sk.holding("card_1"))
        in_bin = self.in_bin() and not held
        success = bool(in_bin)
        moved = float(np.linalg.norm(p["xy"] - mon["start_xy"]))
        lifted = mon["max_lift"] > 0.02
        tries = [c for c in sk.closes if c["tips_above_card"] <= LOW_CLOSE and c["width_before"] > 0.02]
        parts, mode = [], "success"
        if success:
            parts.append("The card is in the bin.")
        else:
            rel = p["xy"] - np.asarray(inst["bin_xy"])
            where = (f"{100 * abs(rel[0]):.1f} cm {'forward' if rel[0] > 0 else 'backward'} and "
                     f"{100 * abs(rel[1]):.1f} cm {'left' if rel[1] > 0 else 'right'} of the bin's centre")
            if held:
                mode = "held"
                parts.append(f"The card is still held by the gripper, {where}.")
            elif lifted:
                mode = "dropped"
                loc = "on the board" if self.on_board() else ("on the table" if p["z"] < BOARD_TOP - 0.02 else
                                                              "partly on the board")
                parts.append(f"The card was lifted {100 * mon['max_lift']:.1f} cm above the board but is not in the "
                             f"bin: it lies {loc}, its centre {where}.")
            elif mon["fell"]:
                mode = "fell_off"
                parts.append("The card was never lifted; it fell off the board and lies flat on the table.")
            else:
                mode = "not_lifted"
                parts.append("The card was never lifted; it still lies flat on the board"
                             + (f" (it was slid {100 * moved:.1f} cm" + (f" and stuck out up to "
                                f"{100 * mon['max_overhang']:.1f} cm beyond the board's edge)" if mon["max_overhang"] > 0.003
                                else ")") if moved > 0.005 else "") + ".")
            if not lifted:
                if not tries:
                    parts.append("The gripper never closed with its fingertips down at the card.")
                else:
                    c = tries[-1]
                    n = f"{len(tries)} grasp tries; in the last one, " if len(tries) > 1 else ""
                    if c["card_on_board"] and c["overhang"] <= 0.0 and c["beyond_edge"] < 0.0:
                        txt = ("when the gripper closed on the card lying flat on the board, its fingertips were on "
                               "either side of the card, at board level")
                    else:
                        if c["beyond_edge"] >= 0.0:
                            pos = f"{100 * c['beyond_edge']:.1f} cm beyond the board's {c['beyond_edge_name']} edge"
                        else:
                            pos = f"over the board, {100 * -c['beyond_edge']:.1f} cm inside its {c['beyond_edge_name']} edge"
                        txt = (f"when the gripper closed, the card stuck out {100 * max(c['overhang'], 0.0):.1f} cm "
                               f"beyond the board's {c['overhang_edge']} edge and the gripper's centre was {pos}")
                    txt += (f" ({100 * c['tips_above_card']:+.1f} cm from the fingertips to the top of the card); the "
                            "fingers closed ")
                    txt += ("fully, sliding over the top of the card instead of gripping its edges."
                            if c["width"] < 2 * CARD_HALF[1] - 0.01 and not c["held"] else
                            ("on the card." if c["held"] else f"to {100 * c['width']:.1f} cm without holding the card."))
                    parts.append(n + txt if n else txt[0].upper() + txt[1:])
        out = dict(success=success, mode=mode, held=held, lifted=bool(lifted), max_lift=round(mon["max_lift"], 4),
                   max_overhang=round(mon["max_overhang"], 4), moved=round(moved, 4), fell=bool(mon["fell"]),
                   grasp_tries=tries, closes=sk.closes, detail=" ".join(parts))
        return out

    # ------------------------------------------------------------ scripted policies (move_to + gripper only)
    def _edge_plan(self, inst):
        """the board edge the card is slid to: along its long side; the near (backward) edge for x-cards,
        the closer side edge for y-cards"""
        if inst["card_along"] == "x":
            return "backward"
        return "left" if inst["card_xy"][1] >= inst["board_xy"][1] else "right"

    def grasp_and_bin(self, xy, yaw, z_site, lift=0.12):
        sk = self.sk
        sk.set_gripper(False, steps=8)
        sk.move_to([xy[0], xy[1], z_site + 0.08], yaw=yaw)
        sk.move_to([xy[0], xy[1], z_site], tol=0.003, max_steps=120)
        sk.set_gripper(True, steps=15)
        sk.move_to([xy[0], xy[1], z_site + lift], speed=0.2)
        if not sk.holding("card_1"):
            sk.set_gripper(False, steps=8)
            return False
        bx, by = self.inst["bin_xy"]
        hi = envs.TABLE_Z + BIN_WALL_H + 0.12
        sk.move_to([bx, by, hi])
        sk.move_to([bx, by, envs.TABLE_Z + BIN_WALL_H + 0.05], speed=0.2)
        sk.set_gripper(False, steps=12)
        sk.move_to([bx, by, hi])
        return True

    def slide_card(self, delta, press=0.004):
        """closed fingertips pressed on the card's centre, straight move by `delta` (xy), lift"""
        sk, p = self.sk, self.card_pose()
        z = p["z"] + CARD_HALF[2] + TIP_BELOW_SITE - press
        sk.set_gripper(True, steps=8)
        sk.move_to([p["xy"][0], p["xy"][1], z + 0.06], yaw=p["yaw"])
        sk.move_to([p["xy"][0], p["xy"][1], z], tol=0.003, max_steps=100)
        e = p["xy"] + np.asarray(delta, float)
        sk.move_to([e[0], e[1], z], speed=DRAG_SPEED, tol=0.004, max_steps=300)
        sk.move_to([e[0], e[1], z + 0.06], speed=0.2)
        sk.hold(5)

    def execute(self, inst, params):
        p = self.card_pose()
        if params["mode"] == "grasp":
            xy = p["xy"] + np.asarray(params.get("dxy", (0.0, 0.0)))
            self.grasp_and_bin(xy, p["yaw"] + params.get("dyaw", 0.0), BOARD_TOP + params.get("dz", 0.0))
            return
        # edge: slide the card along its long side until `overhang` of it sticks out over the chosen edge
        edge = params.get("edge") or self._edge_plan(inst)
        u = {"forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1)}[edge]
        u = np.asarray(u, float)
        c, s = np.cos(p["yaw"]), np.sin(p["yaw"])
        long_ax = np.array([c, s]) if abs(np.array([c, s]) @ u) >= abs(np.array([-s, c]) @ u) else np.array([-s, c])
        long_ax = long_ax * np.sign(long_ax @ u)
        b = np.asarray(inst["board_xy"])
        half = BOARD_HALF[0] if edge in ("forward", "backward") else BOARD_HALF[1]
        edge_pos = (b @ u) + half  # coordinate of the edge along u
        ext = CARD_HALF[0] * abs(long_ax @ u) + CARD_HALF[1] * abs(np.array([-long_ax[1], long_ax[0]]) @ u)
        for short in (0.01, 0.0):  # slide to 1 cm short, look again, slide the rest (one stroke overshoots ~0.7 cm)
            p = self.card_pose()
            need = edge_pos + float(params["overhang"]) - short - (p["xy"] @ u + ext)  # travel still needed along u
            if need > 0.003:
                self.slide_card(long_ax * need / max(long_ax @ u, 1e-6))
        q = self.card_pose()
        cu = float(q["xy"] @ u)
        tip_u = cu + CARD_HALF[0] * float(long_ax @ u)  # centre of the card's far end, along u
        target_u = min(tip_u - 0.008, edge_pos + GRASP_OUT)
        g = q["xy"] + long_ax * (target_u - cu) / max(float(long_ax @ u), 1e-6)
        self.grasp_and_bin(g, q["yaw"], BOARD_TOP - 0.003)

    def default_params(self, inst):
        return dict(mode="grasp")

    def oracle_params(self, inst):
        return dict(mode="edge", overhang=OVERHANG)

    def blind_params(self, inst, attempt, rng):
        return dict(mode="grasp", dxy=rng.normal(0.0, 0.006, 2).tolist(), dyaw=float(rng.normal(0.0, 0.12)),
                    dz=float(rng.uniform(-0.006, 0.003)))

    def adapt_params(self, inst, history):
        """not lifted while flat on the board (fingers slid over it) -> slide it over the edge first, 3 cm.
        Edge tried: card fell off -> 1 cm less; fingers closed without holding and < 4 cm overhang -> 1 cm more."""
        out, prm = history[-1]["outcome"], history[-1]["params"]
        if prm["mode"] == "grasp":
            return dict(mode="edge", overhang=OVERHANG) if out["mode"] in ("not_lifted", "fell_off") else dict(prm)
        h = float(prm["overhang"])
        if out["mode"] == "fell_off":
            h -= 0.01
        elif out["mode"] == "not_lifted":
            h += 0.01 if out["max_overhang"] < 0.04 else -0.005
        return dict(mode="edge", overhang=float(np.clip(h, 0.025, 0.045)))
