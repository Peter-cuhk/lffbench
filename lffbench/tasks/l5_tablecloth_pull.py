"""L5-C tablecloth_pull: yank a board out from under a cube so that the cube lands inside a band.

Category dynamics_adaptation (L5): "figure out how this object moves". Replaces l5_tip_threshold, which could not
be built robustly with this robot (see task_cards/l5_tip_threshold.md).

Scene: a 30 x 8 x 1 cm wooden board lies on the table, pointing away from the robot, with a black handle block
at its near end. A red 4 cm cube stands on the far part of the board. A green band (6 cm deep, across the
table) is painted on the table under the board, a few cm closer to the robot than the cube. The agent may touch
only the board / handle, never the cube.

Physics. If the board is pulled slowly the cube rides along on it (static friction). If it is yanked, the board
slides out from under the cube; while it slides, friction drags the cube a little towards the robot, so the
cube drops onto the table having moved d towards the robot. d DEcreases with the pull speed (counter-intuitive:
a harder yank moves the cube less) and increases with the cube-board friction mu.

Hidden variable: the cube's friction mu (0.08-0.15; the cube looks the same for every mu), which shifts the whole
speed -> displacement curve and the speed below which the cube rides along.

Why the prior fails. The classic tablecloth prior is to yank as fast as possible (0.6 m/s, the arm's top speed);
the band is always placed >= 1.3 cm beyond the displacement such a yank produces, so the cube lands short of it. F2 reports where the cube ended relative to the band (robot-frame words, cm) or that it
rode along; the agent has to infer which way to change the pull speed (faster = less displacement) and by how
much.

Rule (stated in the instruction and enforced, 2026-10-08): ONE pull. Only one move command may drive the board
(>= PULL_EPS), the cube must still be where it started (<= PRE_TOL, grasp nudges) when that pull begins, and all
other move commands together may drive the board at most STRAY_MAX. Gripper open / close commands never count.
Without it every cla2 success (Sonnet, 4/4) dragged the board slowly first so that the cube rode to the band and the
last pull only added a little (one dragged it 4 cm, then yanked it sideways; one moved it with 11 commands, most
of them open-gripper strikes): the friction never mattered.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

BOARD, CUBE, BAND = "board_1", "cube_1", "band_1"
B_HALF = (0.15, 0.04, 0.005)  # 30 x 8 x 1 cm
H_HALF = (0.0125, 0.02, 0.015)  # handle 2.5 x 4 x 3 cm on top of the near end
C_HALF = 0.02  # 4 cm cube
LF = 0.05  # cube centre to the board's far edge
BAND_HALF_X = 0.03  # band 6 cm deep: the 4 cm cube must be fully inside -> centre within +-1 cm
BAND_HALF_Y = 0.12
LINE_W = 0.005
TOL = BAND_HALF_X - C_HALF
MU_BOARD = 0.3  # board-table friction (fixed)
STROKE = 0.25  # scripted pull length
V_MIN, V_MAX = 0.15, 0.80
# one-pull rule (2026-10-08). Board motion is the larger of its two ends' 3-D paths and is only counted while the robot
# drives the board (touches it, or it speeds up; a board sliding on its own only slows down), so coasting after a
# strike or a release is not charged to the next command. Measured on the cla2 agent runs and the scripted policy:
# approach moves 0 mm; closing on the handle 0.6-2 mm (agents) / 4.6 mm (script); release after the pull 3-10 mm
# (gripper commands never count as pulls); the loophole drags 13-41 mm per command, open-gripper strikes 11-54 mm.
PULL_EPS = 0.003  # a move command that drives the board this far is a pull
PRE_TOL = 0.01  # the cube may have moved at most this much (grasp nudges) when the pull starts
STRAY_MAX = 0.005  # board motion driven by all the other move commands together (each below PULL_EPS)
ACC_EPS = 0.0002  # board speed-up per control step (m/step) that counts as the robot driving it
_SOLID = 'solimp="0.998 0.998 0.001" solref="0.001 1"'


def board_xml(name):
    hp = (-B_HALF[0] + H_HALF[0], 0.0, B_HALF[2] + H_HALF[2])
    wood, black = (0.62, 0.45, 0.28, 1.0), (0.12, 0.12, 0.12, 1.0)
    g = (f'        <inertial pos="0.001 0 0.001" mass="0.3" diaginertia="0.0016 0.0023 0.0039" />\n'
         f'        <geom name="{name}_g0" type="box" size="{_fmt(B_HALF)}" rgba="{_fmt(wood)}" {_SOLID} group="0" />\n'
         f'        <geom name="{name}_g1" type="box" pos="{_fmt(hp)}" size="{_fmt(H_HALF)}" rgba="{_fmt(black)}" '
         f'{_SOLID} group="0" />\n'
         f'        <geom type="box" size="{_fmt(B_HALF)}" rgba="{_fmt(wood)}" conaffinity="0" contype="0" group="1" />\n'
         f'        <geom type="box" pos="{_fmt(hp)}" size="{_fmt(H_HALF)}" rgba="{_fmt(black)}" conaffinity="0" '
         f'contype="0" group="1" />')
    # small horizontal radius: only LIBERO's reset-time sampler reads it (the layout is overridden anyway)
    return _wrap(name, g, B_HALF[2], 0.03)


def band_xml(name, line_rgba=(0.1, 0.55, 0.15, 1.0), fill_rgba=(0.75, 0.92, 0.75, 1.0)):
    """Flat visual band across the table (no collisions): light fill + dark lines on its two long edges."""
    hx, hy = BAND_HALF_X, BAND_HALF_Y
    parts = [((0, 0, 0.0004), (hx, hy, 0.0004), fill_rgba),
             ((hx + LINE_W / 2, 0, 0.0010), (LINE_W / 2, hy, 0.0002), line_rgba),
             ((-hx - LINE_W / 2, 0, 0.0010), (LINE_W / 2, hy, 0.0002), line_rgba)]
    g = "\n".join(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(c)}" '
                  f'contype="0" conaffinity="0" group="1" />' for p, sz, c in parts)
    return _wrap(name, g, 0.0012, 0.03)


def d_range(mu):
    """Band distances sampled for a given mu (m). Calibrated 2026-10-07 (LF 5 cm, board at x 0.05): the cube
    moves d(0.6 m/s) ~ 0.33 * mu - 0.85 cm when yanked at the controller's top speed (mu 0.08 -> 1.8 cm,
    0.10 -> 2.3, 0.12 -> 3.1, 0.15 -> 4.3 cm) and >= 0.5 * mu at a pull ~0.05 m/s above the ride-along speed (0.08 -> 4.1 cm at
    0.35 m/s, 0.15 -> 7.6 cm at 0.45 m/s). The band centre is put ~1.5 cm beyond the top-speed displacement, so
    the 'yank as fast as possible' prior always lands the cube short of the band."""
    return 0.33 * mu + 0.0065, 0.50 * mu


class ClothSkills(Skills):
    """Skills that watch, every control step, whether the robot touches the cube (not allowed) and how each
    move_to / gripper command moves the board and the cube (one pull only)."""

    def __init__(self, env, bias=None, recorder=None):
        super().__init__(env, bias=bias, recorder=recorder)
        m = env.sim.model
        self._robot = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(("gripper0_", "robot0_"))]
        self._cube = envs.obj_geom_ids(env, CUBE)
        self._board = envs.obj_geom_ids(env, BOARD)
        self.touched_cube = False
        self.cube0 = envs.obj_pos(env, CUBE)
        self._ends = self._board_ends()
        self._bstep = 0.0  # board motion in the previous control step
        self._seg = None
        self.segments = []  # one entry per move_to / set_gripper command

    def _board_ends(self):
        """Both ends of the board's long axis (world xyz): their motion also catches turning and tilting."""
        b = envs.obj_pos(self.env, BOARD)
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, BOARD)].reshape(3, 3)
        return np.stack([b - R[:, 0] * B_HALF[0], b + R[:, 0] * B_HALF[0]])

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        if not self.touched_cube and envs.contacts_between(self.env, self._robot, self._cube):
            self.touched_cube = True
        ends = self._board_ends()
        step = float(np.max(np.linalg.norm(ends - self._ends, axis=1)))
        touch = envs.contacts_between(self.env, self._robot, self._board)
        speed_up = step - self._bstep > ACC_EPS
        self._ends, self._bstep = ends, step
        s = self._seg
        if s is not None:
            s["path"] += step
            s["touch"] += int(touch)
            if touch or speed_up:
                s["driven"] += step

    def _segment(self, kind, fn, *args, **kwargs):
        c0, b0 = envs.obj_pos(self.env, CUBE), envs.obj_pos(self.env, BOARD)
        self._seg = dict(kind=kind, step0=self.n_steps, path=0.0, driven=0.0, touch=0)
        try:
            rep = fn(*args, **kwargs)
        finally:
            s, self._seg = self._seg, None
        c1, b1 = envs.obj_pos(self.env, CUBE), envs.obj_pos(self.env, BOARD)
        s.update(steps=self.n_steps - s["step0"], board_dxy=(b1 - b0)[:2].round(4).tolist(),
                 cube_dxy=(c1 - c0)[:2].round(4).tolist(), cube_from_start=(c0 - self.cube0)[:2].round(4).tolist())
        self.segments.append(s)
        return rep

    def move_to(self, target, *args, **kwargs):
        return self._segment("move", super().move_to, target, *args, **kwargs)

    def set_gripper(self, close, steps=15):
        return self._segment("close" if close else "open", super().set_gripper, close, steps)

    def pull_check(self):
        """Pulls (move commands that drove the board >= PULL_EPS), where the cube was when the (first) pull started,
        and the board motion driven by the other move commands."""
        moves = [s for s in self.segments if s["kind"] == "move"]
        pulls = [s for s in moves if s["driven"] >= PULL_EPS]
        stray = sum(s["driven"] for s in moves if s["driven"] < PULL_EPS)
        pre = (float(np.linalg.norm(pulls[0]["cube_from_start"])) if pulls
               else float(np.linalg.norm((envs.obj_pos(self.env, CUBE) - self.cube0)[:2])))
        return dict(n_pulls=len(pulls), pulls_cm=[round(100 * s["driven"], 1) for s in pulls],
                    pre_pull_cube_cm=round(100 * pre, 1), stray_cm=round(100 * stray, 2))


@register_task
class TableclothPull(LFFTask):
    name = "l5_tablecloth_pull"
    level = "L5"
    category = "dynamics_adaptation"
    capabilities = ("Perceive", "Reason", "Utilize")
    protocol = "cross"
    instruction = ("Pull the wooden board out from under the red cube so that the cube ends up standing on the table "
                   "completely inside the green band. Touch only the board and its black handle, never the cube. "
                   "Only one pull is allowed per attempt: grasp the black handle, pull the board out with ONE move "
                   "command, then release it. Any other move that drags, pushes or nudges the board, before or after "
                   "the pull, counts as another pull.")
    instruction_indirect = ("Pull the long brown plank out from under the small red block so that the block ends up "
                            "standing on the table completely inside the green stripe. Touch only the plank and its "
                            "black handle, never the block. Only one pull is allowed per attempt: grasp the black "
                            "handle, pull the plank out with ONE move command, then release it. Any other move that "
                            "drags, pushes or nudges the plank, before or after the pull, counts as another pull.")

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        board = register_generated("LffClothBoard", board_xml("lff_cloth_board"))
        cube = register_generated("LffClothCube", box_xml("lff_cloth_cube", (C_HALF,) * 3, (0.85, 0.15, 0.15, 1.0),
                                                          density=2300))
        band = register_generated("LffClothBand", band_xml("lff_cloth_band"), free=False)
        objs = [(BOARD, board, "board_region", (0.09, -0.26, 0.11, -0.24)),
                (CUBE, cube, "cube_region", (-0.21, 0.24, -0.19, 0.26))]
        fixtures = [(BAND, band, "band_region", (-0.31, -0.01, -0.29, 0.01))]
        return write_bddl(self.name, "pull the board out from under the cube", objs, fixtures=fixtures)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        mu = float(np.exp(rng.uniform(np.log(0.08), np.log(0.15))))
        lo, hi = d_range(mu)
        d = float(rng.uniform(lo, max(lo, hi)))
        return dict(seed=int(seed), board_xy=[float(rng.uniform(0.03, 0.08)), float(rng.uniform(-0.08, 0.08))],
                    mu=mu, band_d=d)

    def cube_start(self, inst):
        bx, by = inst["board_xy"]
        return np.array([bx + B_HALF[0] - LF, by])

    def band_center(self, inst):
        c = self.cube_start(inst)
        return np.array([c[0] - inst["band_d"], c[1]])

    def apply_instance(self, inst):
        m = self.env.sim.model
        for g in range(m.ngeom):  # the band is a flat marker: no collisions
            if (m.geom_id2name(g) or "").startswith(BAND + "_"):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        bx, by = inst["board_xy"]
        envs.set_obj_pose(self.env, BOARD, [bx, by, envs.TABLE_Z + B_HALF[2] + 0.0005])
        c = self.cube_start(inst)
        envs.set_obj_pose(self.env, CUBE, [c[0], c[1], envs.TABLE_Z + 2 * B_HALF[2] + C_HALF + 0.0005])
        envs.set_friction(self.env, BOARD, MU_BOARD, priority=1)
        envs.set_friction(self.env, CUBE, inst["mu"], priority=2)  # cube-board and cube-table contacts use mu
        bid = m.body_name2id(self.env.fixtures_dict[BAND].root_body)
        bc = self.band_center(inst)
        m.body_pos[bid] = [bc[0], bc[1], envs.TABLE_Z + 0.0002]
        m.body_quat[bid] = [1.0, 0.0, 0.0, 0.0]

    def reset_instance(self, inst, recorder=None):
        from robosuite.utils.errors import RandomizationError
        for _ in range(20):  # LIBERO's reset-time placement sampler occasionally fails; the layout is overridden
            try:
                self.env.reset()
                break
            except RandomizationError:
                continue
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = ClothSkills(self.env, bias=self.skill_bias(inst), recorder=recorder)
        return self.sk

    # ------------------------------------------------------------ outcome
    def cube_on_board(self):
        c = envs.obj_pos(self.env, CUBE)
        b = envs.obj_pos(self.env, BOARD)
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, BOARD)].reshape(3, 3)
        local = R.T @ (c - b)
        return bool(abs(local[0]) < B_HALF[0] + 0.005 and abs(local[1]) < B_HALF[1] + 0.005
                    and c[2] > envs.TABLE_Z + 2 * B_HALF[2] + C_HALF - 0.004)

    def outcome(self, inst):
        sk = self.sk
        envs.settle(self.env, 60)
        c = envs.obj_pos(self.env, CUBE)
        c0 = self.cube_start(inst)
        bc = self.band_center(inst)
        moved = float(c0[0] - c[0])  # towards the robot
        off = c[:2] - bc  # + x: cube forward (away from the robot) of the band's centre line
        upright = envs.obj_upright_cos(self.env, CUBE) > 0.95
        on_board = self.cube_on_board()
        on_table = c[2] < envs.TABLE_Z + C_HALF + 0.004
        in_band = bool(abs(off[0]) <= TOL and abs(off[1]) <= BAND_HALF_Y - C_HALF)
        pc = sk.pull_check()
        rule = []
        if pc["n_pulls"] > 1:
            rule.append(f"The board was moved by {pc['n_pulls']} separate move commands "
                        f"({', '.join(f'{v:.1f}' for v in pc['pulls_cm'])} cm); only one pull is allowed.")
        if pc["n_pulls"] and pc["pre_pull_cube_cm"] > 100 * PRE_TOL:
            rule.append(f"When the pull started the cube had already been moved {pc['pre_pull_cube_cm']:.1f} cm by "
                        f"earlier small moves of the board; only one pull is allowed.")
        if pc["stray_cm"] > 100 * STRAY_MAX:
            rule.append(f"Besides the pull, other move commands moved the board {pc['stray_cm']:.1f} cm in total; "
                        f"only one pull is allowed.")
        success = bool(in_band and on_table and upright and not on_board and not sk.touched_cube and not rule)
        b_moved = float(np.linalg.norm(envs.obj_pos(self.env, BOARD)[:2] - np.asarray(inst["board_xy"])))
        out = dict(success=success, touched_cube=bool(sk.touched_cube), on_board=on_board, upright=bool(upright),
                   moved_toward_robot=round(moved, 4), band_offset_x=round(float(off[0]), 4),
                   err=round(float(moved - inst["band_d"]), 4), board_moved=round(b_moved, 4),
                   one_pull=not rule, **pc)
        side = ("forward (+x, away from the robot)" if off[0] > 0 else "backward (-x, toward the robot)")
        where = (f"its centre is {100 * abs(off[0]):.1f} cm {side} of the band's centre line"
                 if abs(off[0]) >= 0.0005 else "its centre is on the band's centre line")
        if sk.touched_cube:
            out["failure"] = "touched_cube"
            detail = "The robot touched the cube, which is not allowed."
        elif on_board and b_moved < 0.02:
            out["failure"] = "board_not_moved"
            detail = "The board was not pulled; the cube is still standing on it."
        elif on_board:
            out["failure"] = "rode_along"
            detail = (f"The board was pulled {100 * b_moved:.0f} cm, but the cube rode along on it: it is still on the "
                      f"board and moved {100 * moved:.1f} cm toward the robot.")
        elif not upright:
            out["failure"] = "toppled"
            detail = f"The board came out from under the cube, but the cube fell over; {where}."
        elif in_band and on_table and upright:
            out["failure"] = None
            detail = (f"The board came out from under the cube and the cube is standing inside the green band; {where}. "
                      f"It moved {100 * moved:.1f} cm toward the robot.")
        else:
            out["failure"] = "missed_band"
            detail = (f"The board came out from under the cube and the cube is standing on the table, but not "
                      f"completely inside the green band: {where}. It moved {100 * moved:.1f} cm toward the robot.")
        if rule and not sk.touched_cube:
            out["failure"] = "several_pulls"
            detail = " ".join(rule) + " " + detail
        out["detail"] = detail
        out["detail_indirect"] = (detail.replace("board", "plank").replace("cube", "block")
                                  .replace("green band", "green stripe").replace("band's", "stripe's"))
        return out

    def feedback(self, inst, out, level="F2", indirect=False):
        if indirect and level == "F2":
            out = dict(out, detail=out.get("detail_indirect", out["detail"]))
        return super().feedback(inst, out, level)

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def execute(self, inst, params):
        """Grasp the handle (jaws across its 4 cm width) and pull the board STROKE m toward the robot at `speed`."""
        sk = self.sk
        b = envs.obj_pos(self.env, BOARD)
        hx, hy = b[0] - B_HALF[0] + H_HALF[0], b[1]
        zg = envs.TABLE_Z + 2 * B_HALF[2] + H_HALF[2]
        sk.set_gripper(False, steps=5)
        sk.move_to([hx, hy, zg + 0.08], yaw=0.0)
        sk.move_to([hx, hy, zg], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([hx - STROKE, hy, zg], speed=float(np.clip(params["speed"], V_MIN, V_MAX)), tol=0.01,
                   max_steps=300)
        sk.set_gripper(False, steps=8)
        sk.move_to([hx - STROKE, hy, zg + 0.10], speed=0.3)

    def default_params(self, inst):
        """Prior: the classic tablecloth trick -- yank as fast as the arm can."""
        return dict(speed=0.60)

    def blind_params(self, inst, attempt, rng):
        return dict(speed=float(np.clip(0.60 + rng.normal(0, 0.06), V_MIN, V_MAX)))

    def adapt_params(self, inst, history):
        """Bracketing search on the pull speed with the signed displacement error (riding along = far too much
        displacement). Faster -> less displacement."""
        pts = []
        for h in history:
            o = h["outcome"]
            if o.get("failure") in ("rode_along", "board_not_moved"):
                e = 0.10
            elif o.get("failure") in ("touched_cube", "several_pulls"):
                continue
            else:
                e = o["err"]
            pts.append((h["params"]["speed"], e))
        hi_err = [p for p in pts if p[1] > 0]  # moved too much -> needs a faster pull
        lo_err = [p for p in pts if p[1] < 0]  # moved too little -> needs a slower pull
        if hi_err and lo_err:
            (v0, e0) = max(hi_err, key=lambda p: p[0])  # fastest pull that still moved it too much
            (v1, e1) = min(lo_err, key=lambda p: p[0])  # slowest pull that moved it too little
            if e0 >= 0.05:  # rode along: no usable slope -> bisect
                v = 0.5 * (v0 + v1)
            else:
                v = v0 + (v1 - v0) * e0 / (e0 - e1)
        elif hi_err:
            v = max(p[0] for p in hi_err) * 1.25
        else:
            v = min(p[0] for p in lo_err) * 0.8
        return dict(speed=float(np.clip(v, V_MIN, V_MAX)))

    def oracle_params(self, inst):
        """Privileged: bisection on the speed by simulating pulls with the true mu."""
        if "_oracle_speed" in inst:
            return dict(speed=inst["_oracle_speed"])
        lo, hi = V_MIN, V_MAX
        best = None
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            self.reset_instance(inst)
            self.execute(inst, dict(speed=mid))
            o = self.outcome(inst)
            e = 0.10 if o["failure"] in ("rode_along", "board_not_moved") else o["err"]
            if o["success"]:
                best = (mid, e)
                break
            if best is None or abs(e) < abs(best[1]):
                best = (mid, e)
            if e > 0:
                lo = mid
            else:
                hi = mid
        inst["_oracle_speed"] = best[0]
        self.reset_instance(inst)
        return dict(speed=best[0])
