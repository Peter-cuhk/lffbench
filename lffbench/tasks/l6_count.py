"""L6-B count_into_box: put exactly N identical cubes into an opaque drop box; one carry is interrupted by a slip.

Scene: T identical orange 4 cm cubes scattered on the table and a dark, closed drop box (static fixture) at
one side of the table. The box lid has a square opening on the robot side; a cube released above it falls in,
lands on a low-friction ramp and slides under the closed part of the lid, where no camera can see it (once past
the opening it is moved to a storage slot at the far end, so that cubes never queue back under the opening).
Cubes cannot be taken back out.

Instruction: put exactly N cubes into the box (N in {4, 5}, given in the instruction).

Interruption: during the (p+1)-th carry (p cubes are already in the box, p in {1..N-1}, hidden) the carried
cube "slips": the gripper opens and the cube lands back on the table next to the spot it was picked from. The
number of cubes left on the table is R = N + e with e in {0, 1} drawn independently of p (T = R + p), so
neither the box nor the table tells how many cubes are inside: progress ("how many have I put in already")
is only in the agent's own history. Over-filling is irreversible (terminal failure); under-filling can be
completed in a later segment.

Protocol: within (one episode, no reset), like l6_swap_interrupt. An attempt (segment) ends when the agent
declares done; the outcome is measured; if the box holds fewer than N cubes the agent continues from the
current state. The slip is injected once per episode, inside attempt 1, by a per-control-step monitor (so it
also fires for agents driven by the harness). Events reported immediately (`drain_events`): "A cube dropped
into the box." after every cube that falls in, and the slip.

Scripted references (perception = simulator state, as in l6_swap):
  oracle    privileged box count: carries cubes until the box holds N.
  adaptive  counts its own successful drops (Save), notices the slip (the cube is back on the table, the
            drop count is unchanged), carries cubes until its count reaches N (Retrieve + Utilize). Later
            segments: if F2 says the box holds fewer than N, adds one cube per segment.
  naive     memoryless continuation (1 attempt): loses its count at the slip; re-plans from the current frame,
            which shows no progress, so it guesses how many cubes are already inside (uniform over 0..N-1)
            and adds the rest.
  blind     memoryless restart (k attempts): at the slip it does not know where it was and starts over
            ("the box must get N cubes"), every later segment again puts N cubes in (repeated steps).
Extra controls (not in the acceptance table): `skip` (counts carries instead of drops, i.e. counts the
slipped carry; with memory + F2 it adds the missing cube in segment 2) and `inc1` (stateless, adds one cube
per segment; shows that the within protocol's stop-at-first-success leaks progress to such a policy).
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, box_xml, register_generated
from ..task_base import LFFTask, register_task

CUBE_HALF = 0.02
CUBE_RGBA = (0.95, 0.50, 0.10, 1.0)
N_CUBES = 11  # objects declared in the BDDL; unused ones are parked on the floor out of view
LETTERS = "ABCDEFGHIJK"
CUBES = [f"cube_{i + 1}" for i in range(N_CUBES)]

# drop box (local frame: origin at the centre of the footprint on the table top, +x away from the robot)
WALL_T = 0.008
IN_HX, IN_HY = 0.100, 0.110  # inner half sizes
LID_Z = 0.095  # underside of the lid above the table
LID_T = 0.008
HOLE_X = (-0.090, -0.020)  # opening in the lid (local x range), on the robot side
HOLE_HY = 0.035
RAMP_Z = (0.045, 0.006)  # top surface of the ramp at the near (-x) and far (+x) inner walls
RAMP_T = 0.006
RAMP_MU = 0.03
BOX_RGBA = (0.30, 0.20, 0.13, 1.0)
INNER_RGBA = (0.08, 0.07, 0.06, 1.0)
HOLE_C = np.array([(HOLE_X[0] + HOLE_X[1]) / 2, 0.0])
OUT_HX, OUT_HY = IN_HX + WALL_T, IN_HY + WALL_T
RAMP_SLOPE = (RAMP_Z[0] - RAMP_Z[1]) / (2 * IN_HX)
RAMP_TH = float(np.arctan(RAMP_SLOPE))
# Storage slots under the closed part of the lid. A cube that has slid past the opening is moved to the next free
# slot (out of every camera's view) so that the cubes do not queue up along the slide line back under the opening.
# Order: the lane straight behind the opening (y = 0) is filled last, so that a newly dropped cube can always slide
# past the opening before it is moved; then a second layer on the far row. (x, y, layer)
_ROW = [IN_HX - CUBE_HALF - 0.002 - 0.044 * r for r in range(2)]
SLOTS = ([(_ROW[r], y, 0) for r in range(2) for y in (-0.088, 0.088, -0.044, 0.044)]
         + [(_ROW[0], 0.0, 0), (_ROW[1], 0.0, 0)] + [(_ROW[0], y, 1) for y in (-0.088, 0.088, -0.044, 0.044, 0.0)])

BASE_X = -0.66
REACH = 0.70  # max horizontal distance of a cube from the robot base
CUBE_REGION = (-0.30, 0.07, -0.28, 0.28)
CUBE_MIN_DIST = 0.085
BOX_CLEAR = 0.12  # cube centre at least this far from the box footprint (the hand must not hit the lid)
Z_GRASP = envs.TABLE_Z + CUBE_HALF
Z_CARRY = envs.TABLE_Z + 0.20
Z_RELEASE = envs.TABLE_Z + LID_Z + LID_T + 0.012 + CUBE_HALF  # cube bottom ~1.2 cm above the lid
PARK = [(-0.5 + 0.1 * i, 1.25) for i in range(N_CUBES)]
LIFTED = 0.05  # slip trigger: cube at least this far above its resting height
HAND_HALF = 0.11  # the Panda hand reaches about this far along the closing axis (it hits 4 cm cubes there)
HAND_CLEAR = 0.06  # neighbour centre at least this far from that segment for a collision-free grasp


def hand_box_clear(p, a, box, half=HAND_HALF):
    """Smallest distance from the hand segment (closing axis at yaw a) to the drop box footprint."""
    ax = np.array([-np.sin(a), np.cos(a)])
    return min(float(np.linalg.norm(np.maximum(np.abs(np.asarray(p) + t * ax - box) - [OUT_HX, OUT_HY], 0.0)))
               for t in np.linspace(-half, half, 9))


def axis_clear(p, a, others, half=HAND_HALF):
    """Smallest distance from a neighbour centre to the hand segment p +- half * (closing axis at yaw a)."""
    ax = np.array([-np.sin(a), np.cos(a)])
    best = 1.0
    for o in others:
        v = np.asarray(o) - p
        t = float(np.clip(v @ ax, -half, half))
        best = min(best, float(np.linalg.norm(v - t * ax)))
    return best


def drop_box_xml(model_name):
    """Closed box with a square opening in the lid and an internal ramp (static fixture, collisions on)."""
    parts = []  # (pos, half size, rgba, quat or None, friction or None)
    H = LID_Z / 2
    # walls
    parts.append(((IN_HX + WALL_T / 2, 0, H), (WALL_T / 2, OUT_HY, H), BOX_RGBA))
    parts.append(((-IN_HX - WALL_T / 2, 0, H), (WALL_T / 2, OUT_HY, H), BOX_RGBA))
    parts.append(((0, IN_HY + WALL_T / 2, H), (IN_HX, WALL_T / 2, H), BOX_RGBA))
    parts.append(((0, -IN_HY - WALL_T / 2, H), (IN_HX, WALL_T / 2, H), BOX_RGBA))
    # lid around the opening
    zc = LID_Z + LID_T / 2
    x0, x1 = HOLE_X
    parts.append((((-OUT_HX + x0) / 2, 0, zc), ((x0 + OUT_HX) / 2, OUT_HY, LID_T / 2), BOX_RGBA))  # near strip
    parts.append((((x1 + OUT_HX) / 2, 0, zc), ((OUT_HX - x1) / 2, OUT_HY, LID_T / 2), BOX_RGBA))  # far part
    for s in (-1, 1):
        parts.append((((x0 + x1) / 2, s * (HOLE_HY + OUT_HY) / 2, zc), ((x1 - x0) / 2, (OUT_HY - HOLE_HY) / 2,
                                                                        LID_T / 2), BOX_RGBA))
    # floor (dark) and ramp
    parts.append(((0, 0, 0.002), (IN_HX, IN_HY, 0.002), INNER_RGBA))
    dz = RAMP_Z[0] - RAMP_Z[1]
    th = float(np.arctan2(dz, 2 * IN_HX))
    L = float(np.hypot(dz, 2 * IN_HX)) / 2
    zmid = (RAMP_Z[0] + RAMP_Z[1]) / 2 - RAMP_T / 2 / np.cos(th)
    ramp = ((0, 0, zmid), (L, IN_HY - 0.001, RAMP_T / 2), INNER_RGBA, (np.cos(th / 2), 0, np.sin(th / 2), 0), RAMP_MU)
    geoms = []
    for k, p in enumerate(parts + [ramp]):
        pos, half, rgba = p[:3]
        quat = p[3] if len(p) > 3 else None
        mu = p[4] if len(p) > 4 else None
        q = f' quat="{_fmt(quat)}"' if quat is not None else ""
        fr = f' friction="{mu} 0.005 0.0001" priority="2"' if mu is not None else ' friction="1.0 0.005 0.0001"'
        geoms.append(f'        <geom name="{model_name}_g{k}" type="box" pos="{_fmt(pos)}" size="{_fmt(half)}"{q} '
                     f'rgba="{_fmt(rgba)}" density="1000"{fr} group="0" />')
        geoms.append(f'        <geom type="box" pos="{_fmt(pos)}" size="{_fmt(half)}"{q} rgba="{_fmt(rgba)}" '
                     f'conaffinity="0" contype="0" group="1" />')
    return _wrap(model_name, "\n".join(geoms), (LID_Z + LID_T) / 2, float(np.hypot(OUT_HX, OUT_HY)))


def _wrap_q(a):
    """cubes are symmetric under 90 deg rotations: wrap to [-pi/4, pi/4)"""
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


@register_task
class CountIntoBox(LFFTask):
    name = "l6_count_into_box"
    level = "L6"
    category = "procedural_memory"
    capabilities = ("Perceive", "Save", "Retrieve", "Utilize")
    protocol = "within"
    agent_max_steps = 80  # robot commands per attempt for agents (a VLM needs ~40-55)
    max_attempts = 5
    _TEMPLATE = ("Put exactly {n} of the orange cubes into the dark box, through the opening in its lid. "
                 "Cubes cannot be taken back out of the box.")
    _TEMPLATE_INDIRECT = ("Drop exactly {n} of the small orange blocks into the closed brown container through the "
                          "square hole in its top. Blocks cannot be taken back out of the container.")
    instruction = _TEMPLATE.format(n="N")  # replaced per instance (sample_instance / agent_instruction)
    instruction_indirect = _TEMPLATE_INDIRECT.format(n="N")
    inject = True
    announce_drops = True  # False: no "A cube dropped into the box." event (harder variant, counting from images)
    agent_object_names = ["orange cube (several identical)", "drop box"]

    WORDS = {3: "three", 4: "four", 5: "five", 6: "six"}

    def agent_instruction(self, inst, variant="direct"):
        t = self._TEMPLATE_INDIRECT if variant == "indirect" else self._TEMPLATE
        return t.format(n=self.WORDS[inst["n_target"]])

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        self._mon = False
        orig_step = self.env.step

        def step(action):  # per-control-step monitor: also sees attempts driven by the agent harness
            ret = orig_step(action)
            if self._mon:
                self._monitor()
            return ret

        self.env.step = step

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs = []
        for i, L in enumerate(LETTERS[:N_CUBES]):
            cat = register_generated(f"LffCountCube{L}", box_xml(f"lff_count_cube_{L.lower()}", (CUBE_HALF,) * 3,
                                                                 CUBE_RGBA, density=600))
            gx, gy = -0.30 + 0.15 * (i % 4), -0.40 + 0.2 * (i // 4)
            objs.append((CUBES[i], cat, f"c{i + 1}_region", (gx - 0.01, gy - 0.01, gx + 0.01, gy + 0.01)))
        box = register_generated("LffDropBox", drop_box_xml("lff_drop_box"), free=False)
        fx = [("drop_box_1", box, "box_region", (0.19, 0.39, 0.21, 0.41))]
        return write_bddl(self.name, self._TEMPLATE.format(n="N"), objs, fx)

    def _box_body(self):
        return self.env.sim.model.body_name2id(self.env.fixtures_dict["drop_box_1"].root_body)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.choice([4, 5]))
        p = int(rng.integers(1, n))  # cubes already inside when the slip happens
        e = int(rng.integers(0, 2))
        T = n + e + p
        for _ in range(500):
            side = int(rng.choice([-1, 1]))
            box = np.array([rng.uniform(-0.08, -0.02), side * rng.uniform(0.22, 0.25)])
            P = np.column_stack([rng.uniform(*CUBE_REGION[:2], 800), rng.uniform(*CUBE_REGION[2:], 800)])
            P = P[np.hypot(P[:, 0] - BASE_X, P[:, 1]) <= REACH]
            P = P[np.linalg.norm(np.maximum(np.abs(P - box) - np.array([OUT_HX, OUT_HY]), 0.0), axis=1) >= BOX_CLEAR]
            cubes = []
            for c in P:  # random sequential addition
                if all(np.linalg.norm(c - q) >= CUBE_MIN_DIST for q in cubes):
                    cubes.append(c)
                    if len(cubes) == T:
                        break
            if len(cubes) < T:
                continue
            yaws = []  # each cube gets a yaw whose (small-rotation) grasp axis is free for the hand
            for i in range(T):
                oth = [cubes[j] for j in range(T) if j != i]
                for _ in range(30):
                    y = float(rng.uniform(-np.pi / 4, np.pi / 4))
                    if axis_clear(cubes[i], y, oth) >= HAND_CLEAR and hand_box_clear(cubes[i], y, box) >= 0.035:
                        # |y| <= 45 deg: the grasp needs no large wrist rotation
                        yaws.append(y)
                        break
                else:
                    break
            if len(yaws) == T:
                break
        else:
            raise RuntimeError("could not sample a layout")
        inst = dict(seed=int(seed), n_target=n, p=p, e=e, n_cubes=T, box_xy=box.round(4).tolist(),
                    cubes=np.array(cubes).round(4).tolist(),
                    yaws=yaws,
                    trigger_dist=float(rng.uniform(0.05, 0.09)),
                    slip_r=float(rng.uniform(0.005, 0.02)), slip_ang=float(rng.uniform(0, 2 * np.pi)),
                    slip_yaw=float(rng.uniform(-np.pi / 4, np.pi / 4)))
        # make_video prints task.instruction: keep it in sync with the last sampled instance
        self.instruction = self.agent_instruction(inst, "direct")
        self.instruction_indirect = self.agent_instruction(inst, "indirect")
        return inst

    def apply_instance(self, inst):
        m = self.env.sim.model
        bid = self._box_body()
        m.body_pos[bid] = [inst["box_xy"][0], inst["box_xy"][1], envs.TABLE_Z]
        m.body_quat[bid] = [1, 0, 0, 0]
        T = inst["n_cubes"]
        for i, c in enumerate(CUBES):
            if i < T:
                envs.set_obj_pose(self.env, c, [*inst["cubes"][i], envs.TABLE_Z + CUBE_HALF + 0.001],
                                  yaw=inst["yaws"][i])
            else:
                envs.set_obj_pose(self.env, c, [*PARK[i], CUBE_HALF + 0.001])
        self.env.sim.forward()
        self._inst = inst
        self.active = CUBES[:T]
        self.slip = None
        self.events = []
        self._event_cursor = 0
        self._outcome_cursor = 0
        self.in_box = []  # cubes that fell into the box, in order
        self.stored = []  # cubes moved to a storage slot
        self._bad = set()  # scripted policies: cubes whose grasp failed
        self.rest_xy = {c: np.array(inst["cubes"][i]) for i, c in enumerate(self.active)}
        self._mon = False

    def reset_instance(self, inst, recorder=None):
        self._mon = False
        sk = super().reset_instance(inst, recorder=recorder)
        self._home = sk.eef_pos()
        self._mon = True
        return sk

    def go_home(self):
        p = self.sk.eef_pos()
        self.sk.move_to([p[0], p[1], max(p[2], self._home[2])])
        self.sk.move_to(self._home)

    # ------------------------------------------------------------ geometry helpers
    def box_local(self, xyz):
        b = self._inst["box_xy"]
        return np.array([xyz[0] - b[0], xyz[1] - b[1], xyz[2] - envs.TABLE_Z])

    def hole_world(self):
        b = self._inst["box_xy"]
        return np.array([b[0] + HOLE_C[0], b[1] + HOLE_C[1]])

    def inside_box(self, c):
        q = self.box_local(envs.obj_pos(self.env, c))
        return bool(abs(q[0]) < IN_HX and abs(q[1]) < IN_HY and q[2] < LID_Z)

    def on_lid(self, c):
        q = self.box_local(envs.obj_pos(self.env, c))
        return bool(abs(q[0]) < OUT_HX + 0.01 and abs(q[1]) < OUT_HY + 0.01 and q[2] >= LID_Z)

    def on_table(self, c):
        p = envs.obj_pos(self.env, c)
        return bool(p[2] < envs.TABLE_Z + CUBE_HALF + 0.01 and not self.inside_box(c))

    # ------------------------------------------------------------ monitor (every control step)
    def _monitor(self):
        env = self.env
        for c in self.active:
            if c in self.in_box:
                if c not in self.stored and self.box_local(envs.obj_pos(env, c))[0] > HOLE_X[1] + 0.02:
                    self._store(c)
                continue
            p = envs.obj_pos(env, c)
            if self.inside_box(c):
                self.in_box.append(c)
                if self.announce_drops:
                    self.events.append(dict(step=int(self.sk.n_steps), kind="drop", obj=c,
                                            text="A cube dropped into the box."))
                continue
            if p[2] < envs.TABLE_Z + CUBE_HALF + 0.005:  # resting on the table: remember where
                self.rest_xy[c] = p[:2].copy()
                continue
            if (self.inject and self.slip is None and p[2] > envs.TABLE_Z + CUBE_HALF + LIFTED
                    and len(self.in_box) == self._inst["p"]
                    and np.linalg.norm(p[:2] - self.rest_xy[c]) >= self._inst["trigger_dist"]
                    and self.sk.holding(c)):
                self._do_slip(c)

    def _store(self, c):
        sx, sy, layer = SLOTS[min(len(self.stored), len(SLOTS) - 1)]
        b = self._inst["box_xy"]
        z = RAMP_Z[0] - (sx + IN_HX) * RAMP_SLOPE + CUBE_HALF / np.cos(RAMP_TH) + 0.002 + layer * (2 * CUBE_HALF + 0.003)
        envs.set_obj_pose(self.env, c, [b[0] + sx, b[1] + sy, envs.TABLE_Z + z],
                          quat_wxyz=[np.cos(RAMP_TH / 2), 0.0, np.sin(RAMP_TH / 2), 0.0])
        self.stored.append(c)

    def _do_slip(self, c):
        inst = self._inst
        src = self.rest_xy[c]
        others = [envs.obj_pos(self.env, o)[:2] for o in self.active if o != c and o not in self.in_box]
        xy, yaw = src, inst["slip_yaw"]
        found = False
        for r in (inst["slip_r"], inst["slip_r"] / 2, 0.0):  # next to the pick-up spot, still graspable
            cand = src + r * np.array([np.cos(inst["slip_ang"]), np.sin(inst["slip_ang"])])
            if not all(np.linalg.norm(cand - o) >= 0.07 for o in others):
                continue
            for dy in (0.0, 0.3, -0.3, 0.6, -0.6):
                y = _wrap_q(inst["slip_yaw"] + dy)
                if axis_clear(cand, y, others) >= HAND_CLEAR and \
                        hand_box_clear(cand, y, np.asarray(inst["box_xy"])) >= 0.035:
                    xy, yaw, found = cand, y, True
                    break
            if found:
                break
        envs.set_obj_pose(self.env, c, [xy[0], xy[1], envs.TABLE_Z + CUBE_HALF + 0.002], yaw=yaw)
        self.sk.grip = -1.0  # the gripper opens
        self.env.sim.forward()
        text = "A cube slipped out of the gripper and fell back onto the table."
        self.slip = dict(obj=c, xy=[float(v) for v in xy], from_xy=[float(v) for v in src], step=int(self.sk.n_steps),
                         in_box=len(self.in_box))
        self.events.append(dict(step=int(self.sk.n_steps), kind="slip", obj=c, text=text))

    def drain_events(self):
        ev = self.events[self._event_cursor:]
        self._event_cursor = len(self.events)
        return [e["text"] for e in ev]

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 20)
        n_in = len(self.in_box)
        N = inst["n_target"]
        held = [c for c in self.active if self.sk.holding(c)]
        n_table = sum(self.on_table(c) for c in self.active if c not in self.in_box)
        n_lid = sum(self.on_lid(c) for c in self.active if c not in self.in_box)
        success = bool(n_in == N and not held)
        terminal = bool(n_in > N)
        if success:
            head = f"The box holds exactly {N} cubes."
        elif terminal:
            head = (f"The box holds more than {N} cubes. Cubes cannot be taken out of the box, so the task can no "
                    f"longer be completed.")
        elif n_in == N:
            head = f"The gripper is still holding a cube."
        else:
            head = f"The box holds fewer than {N} cubes."
        state = f"Now {n_table} cube{'s are' if n_table != 1 else ' is'} on the table"
        if n_lid:
            state += f" and {n_lid} on the box lid"
        state += "."
        # F2 = verdict + currently observable state + the slip of this segment (never the number inside)
        new = [e for e in self.events[self._outcome_cursor:] if e["kind"] == "slip"]
        self._outcome_cursor = len(self.events)
        detail = head + " " + state
        if new:
            detail = "During this attempt: " + " ".join(e["text"] for e in new) + " " + detail
        return dict(success=success, terminal=terminal, n_in_box=n_in, n_target=N, n_table=n_table, n_lid=n_lid,
                    held=len(held), slip=self.slip, detail=detail)

    # ------------------------------------------------------------ motion (scripted policies)
    def _grasp_yaw(self, c):
        """The cube's grasp axis with the smaller wrist rotation (|yaw| <= 45 deg). Rotations near +-90 deg drive
        the Panda wrist into its joint-6 limit at some positions (measured), so the other axis is used only if
        the hand would hit a neighbour or the box on this one."""
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, c)].reshape(3, 3)
        yaw = _wrap_q(float(np.arctan2(R[1, 0], R[0, 0])))
        p = envs.obj_pos(self.env, c)[:2]
        others = [envs.obj_pos(self.env, o)[:2] for o in self.active if o != c and o not in self.in_box]
        box = np.asarray(self._inst["box_xy"])

        def score(a):
            return min(axis_clear(p, a, others), hand_box_clear(p, a, box) + 0.03)
        if score(yaw) >= HAND_CLEAR:
            return yaw
        alt = yaw - np.pi / 2 if yaw >= 0 else yaw + np.pi / 2
        return alt if score(alt) > score(yaw) + 0.02 else yaw

    def _pick(self, c):
        sk = self.sk
        for _ in range(2):
            p = envs.obj_pos(self.env, c)
            sk.grasp_at(p[:2], Z_GRASP, yaw=self._grasp_yaw(c), lift=0.12)
            if sk.holding(c):
                return True
            sk.set_gripper(False, steps=8)
            sk.move_to([p[0], p[1], Z_GRASP + 0.12])
        return False

    def _carry(self, c):
        """Pick cube c and release it above the opening. Returns 'in', 'slipped', 'missed' or 'no_grasp'."""
        n_ev = len(self.events)
        n_in = len(self.in_box)
        if not self._pick(c):
            self._bad.add(c)
            self.go_home()  # un-wind the arm before the next try
            return "no_grasp"
        h = self.hole_world()
        sk = self.sk
        sk.move_to([h[0], h[1], Z_CARRY], yaw=0.0)  # the slip may fire here; any yaw fits the opening
        if any(e["kind"] == "slip" for e in self.events[n_ev:]):
            return "slipped"
        sk.move_to([h[0], h[1], Z_RELEASE], tol=0.004)
        sk.set_gripper(False, steps=12)
        sk.move_to([h[0], h[1], Z_CARRY])
        sk.hold(10)
        return "in" if len(self.in_box) > n_in else "missed"

    def _next_cube(self):
        """nearest table cube to the opening (what any policy sees in the current frame)"""
        h = self.hole_world()
        cands = [c for c in self.active if c not in self.in_box and (self.on_table(c) or self.on_lid(c))]
        cands = [c for c in cands if c not in self._bad] or cands  # try another cube after a failed grasp
        if not cands:
            return None
        return min(cands, key=lambda c: np.linalg.norm(envs.obj_pos(self.env, c)[:2] - h))

    def _add(self, k, log, stop_on_slip=False):
        """Carry until k cubes went in (by the policy's own observation of each drop). Returns (added, slipped)."""
        added, slipped = 0, False
        guard = 0
        while added < k and guard < 3 * k + 3:
            guard += 1
            c = self._next_cube()
            if c is None:
                break
            r = self._carry(c)
            log.append(r)
            if r == "in":
                added += 1
            elif r == "slipped":
                slipped = True
                if stop_on_slip:
                    break
        return added, slipped

    # ------------------------------------------------------------ scripted references (within protocol)
    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed + 7919)
        self.reset_instance(inst)
        N = inst["n_target"]
        history = []
        for a in range(k):
            log = []
            plan = dict(kind=kind, carries=log)
            if a == 0:
                if kind == "oracle":
                    while len(self.in_box) < N and len(log) < 3 * N + 3:  # privileged count
                        c = self._next_cube()
                        if c is None:
                            break
                        log.append(self._carry(c))
                    plan["recovery"] = "oracle: privileged box count"
                elif kind in ("adaptive", "skip"):
                    added, slipped = self._add(N, log) if kind == "adaptive" else (0, False)
                    if kind == "skip":  # counts carries, not drops
                        n_car = 0
                        while n_car < N and len(log) < 3 * N:
                            c = self._next_cube()
                            if c is None:
                                break
                            r = self._carry(c)
                            log.append(r)
                            if r in ("in", "slipped"):
                                n_car += 1
                            if r == "in":
                                added += 1
                        plan["recovery"] = "counts carries (the slipped one too)"
                    else:
                        plan["recovery"] = "own drop count; the slipped carry is not counted"
                else:  # memoryless after the slip
                    added, slipped = self._add(N, log, stop_on_slip=True)
                    if slipped:
                        if kind == "naive":
                            guess = int(rng.integers(0, N))
                            plan["recovery"] = f"memoryless: guesses {guess} already inside, adds {N - guess}"
                            self._add(N - guess, log)
                        elif kind == "blind":
                            plan["recovery"] = f"memoryless restart: puts {N} cubes in"
                            self._add(N, log)
                        elif kind == "inc1":
                            plan["recovery"] = "stateless: adds one cube per segment"
                            self._add(1, log)
                        else:
                            raise ValueError(kind)
                    else:
                        plan["recovery"] = "none (no slip)"
            else:
                if kind in ("adaptive", "skip"):
                    plan["recovery"] = "memory + F2 'fewer than N': add one cube"
                    self._add(1, log)
                elif kind == "blind":
                    plan["recovery"] = f"memoryless restart: puts {N} cubes in"
                    self._add(N, log)
                elif kind == "inc1":
                    plan["recovery"] = "stateless: adds one cube"
                    self._add(1, log)
                elif kind == "oracle":
                    while len(self.in_box) < N and len(log) < 3 * N + 3:
                        c = self._next_cube()
                        if c is None:
                            break
                        log.append(self._carry(c))
                else:
                    raise ValueError(kind)
            self.go_home()
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=plan, outcome=out))
            if on_attempt:
                on_attempt(a, plan, out)
            if out["success"] or out["terminal"] or kind in ("oracle", "naive"):
                break
        return history

    # cross-protocol hooks are not used by this task
    def default_params(self, inst):
        return dict(policy="naive")

    def oracle_params(self, inst):
        return dict(policy="oracle")
