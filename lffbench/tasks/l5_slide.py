"""L5-A slide_to_target: strike the cube so that it slides on its own and comes to rest inside the target zone.

Hidden variable: cube-table sliding friction mu (log-uniform in [0.035, 0.10]) -- the cube looks the
same for every mu. With the default push speed the cube overshoots on slippery instances and stops
short on grippy ones; the only way to get it right is to infer from where the cube stopped how hard
to push next time (cf. MemMimic push_cube, iterative pushing).

Strikes per attempt: not limited (2026-10-19, user). From 2026-10-08 the instruction allowed ONE strike per attempt
(one move command pushing the cube, the gripper advancing at most 4 cm after the first touch), to stop slow pushes that
carry the cube to the target so that friction never matters (exp1 / cla1 Claude, la2 GPT: 0.05 m/s pushes). That rule
also moved within-attempt correction (overshot -> fetch the cube back -> strike again) to the next attempt. The
loophole is now closed by the success rule instead:

Success: when the attempt ends the cube is at rest, on the table, in the square (see "Target" below), AND it got there
by a free slide of at least D_MIN: from the last moment the robot touched it (any robot or gripper geom, contacts
checked at every 2 ms physics substep; holding it counts as touching) to its final position the cube slid >= D_MIN with
nothing touching it. A touch that leaves a resting cube where it was (< 1 mm) is ignored. This fails a quasi-static push
into the square (free slide 0-1.2 cm measured), placing the cube there with the gripper (0 cm) and putting it just in
front of the square and tapping it (< D_MIN); see the task card for the adversarial checks and why D_MIN = 3 cm.

Error events: every strike (a robot contact that moved the cube followed by a free slide of >= STRIKE_MIN) is logged
when the cube comes to rest; it is "missed" if the cube then lies outside the square (past / short / beside).
`n_missed_strikes`, `trap_fired` (>= 1 missed strike) and `recovered_in_attempt` (trap fired and the attempt then
succeeded) are in the outcome, as in l7_key_turn; `irrecoverable` = the cube fell off the table or lies outside the
arm's reach (the attempt is not ended for it).

Observation fix (2026-10-19): a strike used to return while the cube was still sliding, and `observe` does not advance
the simulation, so agents saw the cube half-way (cla2: "2.2 cm short", it then slid to 7.0 cm past the centre).
SlideSkills.move_to / set_gripper now hold the arm still after the motion until the cube is at rest (or WAIT_MAX),
so every reply and image shows the cube where it stopped. The waiting time counts as simulated motion time.

Target (2026-10-19, user; before: completely inside the square on all four sides, centre within +-1 cm): the agentview
camera looks at the square from the far side (the cube is pushed towards it), and a cube that sticks out past the far
edge by up to ~1 cm covers the far green line but shows < 2 px of red beyond it in the 512 x 512 image, which looks like
"inside / on the line" (self-test: 0.8 cm past the inner edge looked inside). So on the far side the cube may now lie on
the far green line: the instruction says it must not stick out beyond that line; judged, its centre may be up to
TOL_PAST = 2.0 cm past the square's centre (front face <= 4 mm beyond the line's outer edge). From 2.0 cm on, every one
of 25 measured instances shows >= 2 px of red beyond the line (task card). The near side and the sides stay "completely
inside" (+-1 cm): the near-side band cannot be made visible in the agentview by any tolerance (the cube's own body hides
the near line), and relaxing it would let the 0.30 m/s strike pass on short-stopping instances (task card).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, box_xml, register_generated
from ..skills import DT, Skills, _rot_err
from ..task_base import LFFTask, register_task

CUBE = "cube_1"
CUBE_HALF = 0.02
D_MIN = 0.03  # the cube's last free slide (after its last contact with the robot) must be at least this long
STRIKE_MIN = 0.01  # a contact that moved the cube, followed by a free slide of >= 1 cm, counts as a strike
TOUCH_MOVE = 0.001  # a contact episode moved the cube if the cube ends >= 1 mm (3-D) from where it was at its start
EP_GAP = 25  # physics substeps (one control step, 50 ms) without robot contact end a contact episode
V_REST, W_REST = 0.003, 0.05  # cube at rest: linear speed below 3 mm/s and angular speed below 0.05 rad/s ...
REST_SUB = 25  # ... for 25 consecutive substeps (50 ms)
WAIT_MAX = 100  # control steps (5 s) a command may hold the arm waiting for the cube to come to rest
ON_TABLE = 0.005  # cube resting on the table: centre within 5 mm of TABLE_Z + CUBE_HALF
OFF_TABLE_Z = envs.TABLE_Z - 0.05  # cube centre below this: it fell off the table
REACH_MARGIN = 0.03  # cube centre this close to the commanded-target limits (agent/tools.py WS_LO / WS_HI): unreachable


class CubeWatch:
    """Follows the cube at every physics substep (installed on env.sim by reset_instance; outcome() reads it).

    contact episode: substeps with a robot-cube contact, merged across gaps shorter than EP_GAP substeps. Closed
      EP_GAP substeps after its last contact; it "moved" the cube if the cube is then >= TOUCH_MOVE from where it was
      when the episode began, or still moving.
    release: the cube's xy at the last contact of the last episode that moved it, i.e. where its current free slide
      began (None while the robot has never moved the cube).
    strike: an episode that moved the cube followed by a free slide of >= STRIKE_MIN, logged when the cube comes to
      rest (or when the robot touches it again before that: result "interrupted")."""

    def __init__(self, env, target_xy, tol_short, tol_past, tol_across):
        m = env.sim.model
        self.env = env
        self.is_robot = np.zeros(m.ngeom, bool)
        self.is_robot[envs.robot_geom_ids(env)] = True
        self.is_cube = np.zeros(m.ngeom, bool)
        self.is_cube[envs.obj_geom_ids(env, CUBE)] = True
        self.qa, self.da = envs.free_joint_addr(env, CUBE)
        self.target = np.asarray(target_xy, float)
        self.tol = (float(tol_short), float(tol_past), float(tol_across))  # centre offsets: short of / past / beside
        self.n_sub = 0
        self.ep = None  # open contact episode
        self.gap = 0
        self.quiet = 0  # consecutive substeps at rest
        self.release = None
        self.slide = None  # free slide in progress: where it started (cube at the episode start) and was released
        self.moves = []  # closed contact episodes that moved the cube
        self.strikes = []

    # ------------------------------------------------------------ state
    def pos(self):
        return self.env.sim.data.qpos[self.qa:self.qa + 3].copy()

    def at_rest(self):
        qv = self.env.sim.data.qvel[self.da:self.da + 6]
        return bool(np.linalg.norm(qv[:3]) < V_REST and np.linalg.norm(qv[3:]) < W_REST)

    def judge(self, p):
        """Where a cube at p lies relative to the square: (along, across, in / past / short / beside / off).
        in = centre at most tol_short short of / tol_past past the square's centre and within tol_across beside it."""
        along, across = float(p[0] - self.target[0]), float(p[1] - self.target[1])
        if abs(p[2] - (envs.TABLE_Z + CUBE_HALF)) > ON_TABLE:
            return along, across, "off"
        short, past, beside = self.tol
        if -short <= along <= past and abs(across) <= beside:
            return along, across, "in"
        if along > past:
            return along, across, "past"
        if along < -short:
            return along, across, "short"
        return along, across, "beside"

    # ------------------------------------------------------------ per substep
    def substep(self):
        d = self.env.sim.data
        n = d.ncon
        touch = False
        if n:
            g1 = np.asarray(d.contact.geom1)[:n]
            g2 = np.asarray(d.contact.geom2)[:n]
            touch = bool(np.any((self.is_robot[g1] & self.is_cube[g2]) | (self.is_robot[g2] & self.is_cube[g1])))
        self.n_sub += 1
        p = self.pos()
        rest = self.at_rest()
        self.quiet = self.quiet + 1 if rest else 0
        if touch:
            if self.ep is None:
                if self.slide is not None:
                    self._end_slide(p, interrupted=not rest)
                self.ep = dict(start=p, sub0=self.n_sub)
            self.ep["last"] = p
            self.ep["sub1"] = self.n_sub
            self.gap = 0
        elif self.ep is not None:
            self.gap += 1
            if self.gap >= EP_GAP:
                self._close(p, rest)
        if self.ep is None and self.slide is not None and self.quiet >= REST_SUB:
            self._end_slide(p, interrupted=False)

    def _close(self, p, rest):
        ep, self.ep = self.ep, None
        if np.linalg.norm(p - ep["start"]) < TOUCH_MOVE and rest:
            return  # touched a resting cube and left it where it was
        self.release = ep["last"][:2].copy()
        self.moves.append(dict(sub0=ep["sub0"], sub1=ep["sub1"], start=ep["start"].round(4).tolist(),
                               release=ep["last"].round(4).tolist()))
        self.slide = dict(start=ep["start"], release=ep["last"], sub=ep["sub1"])

    def _end_slide(self, p, interrupted):
        s, self.slide = self.slide, None
        free = float(np.linalg.norm(p[:2] - s["release"][:2]))
        if free < STRIKE_MIN:
            return  # a push or a put-down, not a strike
        along, across, res = self.judge(p)
        self.strikes.append(dict(sub=int(s["sub"]), start=s["start"].round(4).tolist(),
                                 release=s["release"].round(4).tolist(), rest=p.round(4).tolist(),
                                 pushed_cm=round(100 * float(np.linalg.norm(s["release"][:2] - s["start"][:2])), 2),
                                 free_slide_cm=round(100 * free, 2), along_err_cm=round(100 * along, 2),
                                 across_err_cm=round(100 * across, 2),
                                 result="interrupted" if interrupted else res,
                                 missed=bool(not interrupted and res != "in")))

    def flush(self):
        """End of attempt: settle the bookkeeping. Returns True if the robot is touching the cube now and that contact
        has moved it (then the cube's current free slide is 0)."""
        p = self.pos()
        if self.ep is not None:
            if np.linalg.norm(p - self.ep["start"]) >= TOUCH_MOVE or not self.at_rest():
                self.release = p[:2].copy()
                self.slide = None
                return True
            return False
        if self.slide is not None and self.at_rest():
            self._end_slide(p, interrupted=False)
        return False


def _install_watch(sim, watch):
    """Call watch.substep() after every MuJoCo step of this sim (one wrapper per sim, like agent/tools.ContactTracker;
    it only reads the state). A fast strike touches the cube for 3-16 substeps (6-32 ms, measured), less than one
    control step, so contact checks at control steps alone would miss it."""
    if not getattr(sim, "_lff_slide_wrapped", False):
        orig = sim.step

        def step(*a, **k):
            orig(*a, **k)
            w = getattr(sim, "_lff_slide_watch", None)
            if w is not None:
                w.substep()
        sim.step = step
        sim._lff_slide_wrapped = True
    sim._lff_slide_watch = watch


class SlideSkills(Skills):
    """Skills whose motion commands return only once the cube has come to rest (see the module docstring)."""

    def __init__(self, env, bias=None, recorder=None):
        super().__init__(env, bias=bias, recorder=recorder)
        self._qa, self._da = envs.free_joint_addr(env, CUBE)
        self.waits = []  # (control step, steps waited) for every command that had to wait

    def cube_moving(self):
        qv = self.env.sim.data.qvel[self._da:self._da + 6]
        return bool(np.linalg.norm(qv[:3]) >= V_REST or np.linalg.norm(qv[3:]) >= W_REST)

    def wait_cube_rest(self):
        """Hold the arm where it is until the cube is at rest (two control steps in a row) or WAIT_MAX steps passed.
        No wait if the cube is already at rest or is held (it then moves with the arm)."""
        if not self.cube_moving() or self.holding(CUBE):
            return 0
        p, R = self.eef_pos(), self._R()
        quiet = 0
        for k in range(WAIT_MAX):
            self._act(p - self.eef_pos(), _rot_err(R, self.eef_mat()), self.grip)
            quiet = 0 if self.cube_moving() else quiet + 1
            if quiet >= 2:
                break
        self.waits.append((self.n_steps, k + 1))
        return k + 1

    def move_to(self, target, *args, **kwargs):
        rep = super().move_to(target, *args, **kwargs)
        n = self.wait_cube_rest()
        if n:
            rep["waited_s"] = round(n * DT, 2)
        return rep

    def set_gripper(self, close, steps=15):
        rep = super().set_gripper(close, steps)
        n = self.wait_cube_rest()
        if n:
            rep["waited_s"] = round(n * DT, 2)
        return rep


INNER_HALF = 0.03  # 6 cm square target (inner edge of the outline): near side and sides, the cube's centre within 1 cm
LINE_W = 0.006
# Far side (away from the robot, facing the agentview camera): the cube may lie on the far line. Instruction: no part of
# it beyond the line's outer edge (centre <= INNER_HALF + LINE_W - CUBE_HALF = 1.6 cm past the centre); judged: centre
# <= 2.0 cm past it, i.e. the front face up to 4 mm beyond the line, which in the 512 x 512 agentview is < 2 px of red
# beyond the line (from 2.0 cm on >= 2 px on all 25 measured instances, task card 2026-10-19).
TOL_PAST = 0.020


def square_frame_xml(model_name, inner_half=INNER_HALF, line_w=LINE_W,
                     line_rgba=(0.1, 0.55, 0.15, 1.0), fill_rgba=(0.75, 0.92, 0.75, 1.0)):
    """Flat, visual-only square drawn on the table: a light fill plus a dark outline (no collisions)."""
    o = inner_half + line_w
    parts = [((0, 0, 0.0004), (o, o, 0.0004), fill_rgba),
             ((0, inner_half + line_w / 2, 0.0010), (o, line_w / 2, 0.0002), line_rgba),
             ((0, -inner_half - line_w / 2, 0.0010), (o, line_w / 2, 0.0002), line_rgba),
             ((inner_half + line_w / 2, 0, 0.0010), (line_w / 2, o, 0.0002), line_rgba),
             ((-inner_half - line_w / 2, 0, 0.0010), (line_w / 2, o, 0.0002), line_rgba)]
    g = "\n".join(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(sz)}" rgba="{_fmt(c)}" '
                  f'contype="0" conaffinity="0" group="1" />' for p, sz, c in parts)
    return _wrap(model_name, g, 0.0012, o)


Z_PUSH = envs.TABLE_Z + 0.02
V_MIN, V_MAX = 0.12, 0.60
Z_GRASP = envs.TABLE_Z + CUBE_HALF  # grasp point at the cube's centre height (fingertips ~1 cm above the table)


@register_task
class SlideToTarget(LFFTask):
    name = "l5_slide_to_target"
    level = "L5"
    category = "dynamics_adaptation"
    capabilities = ("Reason", "Utilize")
    # no parentheses or semicolons: the instruction is also the BDDL :language string (';' starts a comment there)
    instruction = ("Push the red cube straight away from the robot so that it slides and comes to rest in the green "
                   "square marked on the table. On the side of the square nearest the robot and on its left and "
                   "right, no part of the cube may lie on or beyond the green border line. On the far side, away from "
                   "the robot, the cube may lie on the border line, but no part of it may stick out beyond that line. "
                   "The cube must slide the last part of the way on its own: after it leaves the gripper it has to "
                   f"slide at least {100 * D_MIN:.0f} cm, with nothing touching it, before it comes to rest in the "
                   "square.")
    instruction_indirect = instruction
    tol_along = INNER_HALF - CUBE_HALF  # short of the centre (near side): completely inside the square (1.0 cm)
    tol_past = TOL_PAST  # past the centre (far side): may lie on the far line (2.0 cm, see TOL_PAST)
    runup = 0.22  # long run-up so the gripper reaches the commanded speed before contact
    tol_across = INNER_HALF - CUBE_HALF

    def make_bddl(self):
        cube = register_generated("LffRedCube", box_xml("lff_red_cube", (CUBE_HALF,) * 3, (0.85, 0.15, 0.15, 1),
                                                        density=1000))
        # static fixture (no joint): a free body without collisions would fall through the table
        zone = register_generated("LffTargetSquare", square_frame_xml("lff_target_square"), free=False)
        objs = [("cube_1", cube, "cube_region", (-0.21, -0.01, -0.19, 0.01))]
        fixtures = [("zone_1", zone, "zone_region", (-0.11, -0.01, -0.09, 0.01))]
        return write_bddl(self.name, self.instruction, objs, fixtures=fixtures)

    def apply_instance(self, inst):
        m = self.env.sim.model
        zone = self.env.fixtures_dict["zone_1"]
        # the target zone is a flat marker: no collisions
        for g in range(m.ngeom):
            if (m.geom_id2name(g) or "").startswith("zone_1_"):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        x0, y0 = inst["cube_xy"]
        envs.set_obj_pose(self.env, "cube_1", [x0, y0, envs.TABLE_Z + CUBE_HALF + 0.001])
        bid = m.body_name2id(zone.root_body)
        m.body_pos[bid] = [x0 + inst["target_dist"], y0, envs.TABLE_Z + 0.0002]
        m.body_quat[bid] = [1.0, 0.0, 0.0, 0.0]
        envs.set_friction(self.env, "cube_1", inst["mu"])

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        x0, y0 = inst["cube_xy"]
        self.watch = CubeWatch(self.env, (x0 + inst["target_dist"], y0), self.tol_along, self.tol_past,
                               self.tol_across)
        _install_watch(self.env.sim, self.watch)
        self.sk = SlideSkills(self.env, recorder=recorder)
        return self.sk

    @staticmethod
    def max_reach(mu):
        """Approximate slide distance at the top speed, fitted to a sweep (runup 0.22 m):
        mu 0.04 -> 20.8 cm, 0.06 -> 14.8, 0.08 -> 11.8, 0.10 -> 10.1 cm."""
        return 0.0103 / mu ** 0.95

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        while True:  # reject instances the top speed cannot reach (keeps every instance solvable)
            inst = dict(seed=int(seed),
                        cube_xy=[float(rng.uniform(-0.23, -0.17)), float(rng.uniform(-0.10, 0.10))],
                        target_dist=float(rng.uniform(0.05, 0.12)),
                        mu=float(np.exp(rng.uniform(np.log(0.035), np.log(0.10)))))
            if inst["target_dist"] <= 0.85 * self.max_reach(inst["mu"]):
                return inst

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 40)
        w = self.watch
        for _ in range(12):  # still moving (e.g. a budget ended the attempt mid-slide): let it come to rest
            if w.at_rest():
                break
            envs.settle(self.env, 5)
        touching = w.flush()
        x0, y0 = inst["cube_xy"]
        c = envs.obj_pos(self.env, CUBE)
        along = (c[0] - x0) - inst["target_dist"]
        across = c[1] - y0
        _, _, where = w.judge(w.pos())
        inside = where == "in"
        at_rest = w.at_rest()
        free = float(np.linalg.norm(c[:2] - w.release)) if w.release is not None else 0.0
        slid = bool(w.release is not None and not touching and free >= D_MIN)
        success = bool(inside and at_rest and slid)
        strikes = w.strikes
        n_missed = sum(s["missed"] for s in strikes)
        from ..agent.tools import WS_HI, WS_LO
        reach = bool(np.all(c[:2] >= WS_LO[:2] + REACH_MARGIN) and np.all(c[:2] <= WS_HI[:2] - REACH_MARGIN))
        irrecoverable = bool(c[2] < OFF_TABLE_Z or not reach)

        parts = []
        if w.release is None and np.linalg.norm(c[:2] - np.asarray(inst["cube_xy"])) < STRIKE_MIN:
            parts.append("The cube was not pushed.")
        if c[2] < OFF_TABLE_Z:
            parts.append("The cube fell off the table.")
        else:
            state = {"in": "in the square", "past": "it sticks out beyond the far green line",
                     "short": "it sticks out past the edge of the square nearest the robot"}.get(where,
                                                                                            "not in the square")
            if along > 0:
                parts.append(f"The cube stopped {100 * along:.1f} cm past the centre of the green square ({state}).")
            else:
                parts.append(f"The cube stopped {100 * -along:.1f} cm short of the centre of the green square "
                             f"({state}).")
            if abs(across) > self.tol_across:
                parts.append(f"It also drifted {100 * abs(across):.1f} cm to the {'left' if across > 0 else 'right'} "
                             f"(robot's view).")
            if where == "off" and c[2] > OFF_TABLE_Z:
                parts.append("It is not resting on the table.")
        if inside and not at_rest:
            parts.append("It was still moving when the attempt ended.")
        if touching:
            parts.append("The gripper was still touching (or holding) the cube when the attempt ended.")
        elif inside and w.release is not None and free < D_MIN:
            parts.append(f"After it last left the gripper it slid only {100 * free:.1f} cm on its own; it has to slide "
                         f"at least {100 * D_MIN:.0f} cm.")
        if success:
            failure = None
        elif w.release is None and not inside:
            failure = "not_touched"
        elif not inside:
            failure = "position"
        elif not at_rest:
            failure = "not_at_rest"
        else:
            failure = "slide_too_short"
        out = dict(success=success, along_err=float(along), across_err=float(across), detail=" ".join(parts),
                   inside=inside, at_rest=at_rest, free_slide_cm=round(100 * free, 2),
                   release_xy=None if w.release is None else w.release.round(4).tolist(),
                   touching_at_end=bool(touching), n_moves=len(w.moves), n_strikes=len(strikes),
                   n_missed_strikes=int(n_missed), trap_fired=n_missed > 0,
                   recovered_in_attempt=bool(n_missed and success), irrecoverable=irrecoverable,
                   strikes=strikes, waits=len(getattr(self.sk, "waits", [])), failure=failure)
        return out

    # ------------------------------------------------------------ execution
    def execute(self, inst, params):
        """One straight strike from the cube's initial position: closed gripper, 22 cm run-up, stop 1.5 cm past the
        point of contact, lift (the move returns once the cube has come to rest)."""
        x0, y0 = inst["cube_xy"]
        sk = self.sk
        sk.set_gripper(True, steps=5)
        sk.move_to([x0 - self.runup, y0, Z_PUSH + 0.10], yaw=0.0)
        sk.move_to([x0 - self.runup, y0, Z_PUSH], tol=0.003)
        sk.move_to([x0 - CUBE_HALF + 0.015, y0, Z_PUSH], speed=float(np.clip(params["speed"], V_MIN, V_MAX)),
                   tol=0.01, max_steps=80)
        sk.move_to([x0 - CUBE_HALF + 0.015, y0, Z_PUSH + 0.08], speed=0.3)

    def default_params(self, inst):
        return dict(speed=0.30)

    def blind_params(self, inst, attempt, rng):
        return dict(speed=float(np.clip(0.30 + rng.normal(0, 0.05), V_MIN, V_MAX)))

    def adapt_params(self, inst, history):
        """Cross-attempt learner (one strike per attempt; used by the scripted agent backend): bracketing secant search
        on speed using the signed stopping error. validate.py's adaptive policy is run_scripted (within the attempt)."""
        pts = sorted((h["params"]["speed"], h["outcome"]["along_err"]) for h in history)
        lo = [p for p in pts if p[1] < 0]
        hi = [p for p in pts if p[1] > 0]
        if lo and hi:
            (v0, e0), (v1, e1) = max(lo, key=lambda p: p[0]), min(hi, key=lambda p: p[0])
            v = v0 + (v1 - v0) * (-e0) / (e1 - e0)
        elif hi:  # always overshooting: slow down
            v = min(p[0] for p in hi) * 0.7
        else:  # always short: speed up
            v = max(p[0] for p in lo) * 1.35
        return dict(speed=float(np.clip(v, V_MIN, V_MAX)))

    def oracle_params(self, inst):
        """Privileged: bisection on speed by simulating pushes (uses the true mu)."""
        if "_oracle_speed" in inst:
            return dict(speed=inst["_oracle_speed"])
        lo, hi = V_MIN, V_MAX
        best = None
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            self.reset_instance(inst)
            self.execute(inst, dict(speed=mid))
            e = self.outcome(inst)["along_err"]
            if best is None or abs(e) < abs(best[1]):
                best = (mid, e)
            if e > 0:
                hi = mid
            else:
                lo = mid
        inst["_oracle_speed"] = best[0]
        self.reset_instance(inst)
        return dict(speed=best[0])

    # ------------------------------------------------------------ scripted within-attempt correction
    AGENT_CMDS = 40  # robot_server default per-attempt budget (no agent_max_steps here): 40 commands ...
    AGENT_SIM_S = 3.0 * AGENT_CMDS  # ... and 3 s of simulated motion per command
    RESTRIKE_MARGIN = 0.02  # strike again from where the cube stopped only if it is >= D_MIN + this short

    def _strike_here(self, speed):
        """Strike the cube where it lies (perfect perception), straight towards +x. 4 agent commands: above the run-up
        point + close, down, strike (returns at rest), lift."""
        c = envs.obj_pos(self.env, CUBE)
        sk = self.sk
        sk.move_to([c[0] - self.runup, c[1], Z_PUSH + 0.10], yaw=0.0)
        sk.set_gripper(True, steps=5)
        sk.move_to([c[0] - self.runup, c[1], Z_PUSH], tol=0.003)
        sk.move_to([c[0] - CUBE_HALF + 0.015, c[1], Z_PUSH], speed=float(np.clip(speed, V_MIN, V_MAX)), tol=0.01,
                   max_steps=80)
        sk.move_to([c[0] - CUBE_HALF + 0.015, c[1], Z_PUSH + 0.08], speed=0.3)
        return 4

    def _carry_back(self, inst):
        """Grasp the cube from above, carry it back to its initial position and put it down. 6 agent commands:
        above it + open, down + close, lift, carry, down + open, lift."""
        c = envs.obj_pos(self.env, CUBE)
        x0, y0 = inst["cube_xy"]
        sk = self.sk
        sk.move_to([c[0], c[1], Z_GRASP + 0.10], yaw=0.0)
        sk.set_gripper(False, steps=10)
        sk.move_to([c[0], c[1], Z_GRASP], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([c[0], c[1], Z_GRASP + 0.06])
        sk.move_to([x0, y0, Z_GRASP + 0.06])
        sk.move_to([x0, y0, Z_GRASP + 0.003], tol=0.003)
        sk.set_gripper(False, steps=12)
        sk.move_to([x0, y0, Z_GRASP + 0.10])
        return 6

    @staticmethod
    def _plan_speed(want, pts):
        """Speed for a slide of `want` m from strikes seen so far, pts = [(speed, slide)]: none -> the 0.30 m/s
        prior; one -> slide ~ speed^2 through it; several -> log-log interpolation between the bracketing strikes
        (or extrapolation from the two nearest, exponent clipped to [0.5, 3])."""
        pts = sorted((v, d) for v, d in pts if d > 0.002)
        if not pts:
            return 0.30
        if len(pts) == 1:
            v, d = pts[0]
            return float(np.clip(v * (want / d) ** 0.5, V_MIN, V_MAX))
        lo = [p for p in pts if p[1] <= want]
        hi = [p for p in pts if p[1] > want]
        if lo and hi:
            (v0, d0), (v1, d1) = max(lo, key=lambda p: p[1]), min(hi, key=lambda p: p[1])
        else:
            (v0, d0), (v1, d1) = sorted(pts, key=lambda p: abs(np.log(p[1] / want)))[:2]
        if abs(np.log(d1 / d0)) < 1e-6 or abs(np.log(v1 / v0)) < 1e-6:
            return float(np.clip(v0 * (want / d0) ** 0.5, V_MIN, V_MAX))
        k = float(np.clip(np.log(d1 / d0) / np.log(v1 / v0), 0.5, 3.0))
        return float(np.clip(v0 * (want / d0) ** (1.0 / k), V_MIN, V_MAX))

    def _adaptive_attempt(self, inst, pts):
        """One attempt of the scripted within-attempt learner: strike; look where the cube stopped (perfect
        perception, as on the images); if short by >= D_MIN + 2 cm strike again from there, otherwise (overshot,
        a little short, or drifted sideways) carry it back to its initial position and strike again, with the speed
        planned from every strike seen so far (`pts`, also from earlier attempts). Stops when the cube is in the square
        or the next correction would not fit in the agent's command / motion budget."""
        x0, y0 = inst["cube_xy"]
        tx = x0 + inst["target_dist"]
        sk = self.sk
        n0 = sk.n_steps
        n_cmd, speeds, path = 0, [], []
        while True:
            c = envs.obj_pos(self.env, CUBE)
            want = tx - c[0]
            near_start = np.linalg.norm(c[:2] - np.asarray(inst["cube_xy"])) < 0.01
            local = [(v, d) for v, d, s in pts if s == near_start] or [(v, d) for v, d, _ in pts]
            speed = self._plan_speed(want, local)
            n_cmd += self._strike_here(speed)
            c1 = envs.obj_pos(self.env, CUBE)
            pts.append((speed, float(c1[0] - c[0]), near_start))
            speeds.append(round(speed, 4))
            err, across = float(c1[0] - tx), float(c1[1] - y0)
            path.append("strike")
            if -self.tol_along <= err <= self.tol_past and abs(across) <= self.tol_across:
                break
            used_s = (sk.n_steps - n0) * DT
            if err < 0 and -err >= D_MIN + self.RESTRIKE_MARGIN and abs(across) < 0.005:
                if n_cmd + 4 > self.AGENT_CMDS or used_s + 12.0 > self.AGENT_SIM_S:
                    break
                continue
            if n_cmd + 10 > self.AGENT_CMDS or used_s + 30.0 > self.AGENT_SIM_S:
                break
            n_cmd += self._carry_back(inst)
            path.append("carry_back")
        return dict(speed=speeds[0], speeds=speeds, path=path, n_commands=n_cmd,
                    sim_s=round((sk.n_steps - n0) * DT, 1))

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """oracle / naive / blind: one strike per attempt (LFFTask.run_scripted, cross protocol). adaptive: corrects
        within the attempt (_adaptive_attempt) and carries what it saw into the next attempt."""
        if kind != "adaptive":
            return super().run_scripted(kind, inst, k=k, seed=seed, on_attempt=on_attempt)
        k = k or self.max_attempts
        history, pts = [], []
        for a in range(k):
            self.reset_instance(inst)
            params = self._adaptive_attempt(inst, pts)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"]:
                break
        return history
