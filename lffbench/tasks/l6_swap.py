"""L6-A swap_interrupt: swap two items between identical pads; the second carry is interrupted by a slip.

Scene: four identical dark round pads (flat markers, no collision). The cream cheese (X) starts on pad A,
the butter (Y) on pad B, pads C and D are empty. Pad positions, which pad is which and the drop spot are
random per seed. Goal: X on B and Y on A.

Standard procedure: X: A -> C (temporary), Y: B -> A, X: C -> B. During the second carry (whichever item is
carried while the other one has already left its start pad), the carried item "slips": the gripper opens
and the item lands on the table at a hidden spot that is on no pad. Afterwards X sits on C, Y lies on the
table and A, B, D are empty -- the current image no longer shows which empty pad was whose. The information
needed to finish ("what has been done, where did each item start") is only in the agent's own history.

The drop spot is placed next to a *lure* pad: the spare pad ("spare", 50 %), the carried item's own start pad
("source", 25 %) or its true destination ("dest", 25 %). So "put it on the nearest pad" is usually wrong and
proximity is close to uninformative; a stronger bias towards the start pad would open a memoryless shortcut
(tidy both items onto their nearest pads = undo, then swap again), see the task card.

Protocol: within (one episode, no reset). Attempt granularity: an attempt is everything the agent does until
it declares the task done; after each attempt the outcome is measured (F1/F2 feedback) and, if it failed, the
agent continues from the current state. The slip is injected once per episode, inside attempt 1 (it is an
in-attempt failure event, reported immediately: see `drain_events`). Scripted references:

  oracle    knows the true start layout; after the slip puts Y on A and X on B.
  adaptive  saves the start layout from its first frame (Save), recalls it after the slip (Retrieve) and
            finishes the swap from the right step (Utilize); later attempts fix residual misplacements from
            memory + F2 feedback.
  naive     memoryless continuation (one attempt): after the slip it re-plans from the current frame only --
            the stray item goes to the nearest empty pad, the item that is still on a pad goes to the nearest
            remaining empty pad.
  blind     memoryless restart (k attempts): does not know which step it was at; tidies the stray item onto
            the nearest empty pad and then performs the whole swap again from the current frame; every later
            attempt again swaps the two items (repeated swap, cf. RMBench).
  blind_continue  (extra control, not in the acceptance table) attempt 1 = naive, later attempts = swap again.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _wrap, register_generated
from ..task_base import LFFTask, register_task

N_PADS = 4
PAD_R = 0.055  # pad radius (11 cm discs); items are 8 x 4.3 x 1.8 cm boxes lying flat
PAD_HALF_T = 0.0008
PAD_RGBA = (0.22, 0.24, 0.28, 1.0)
SUCC_R = 0.04  # item centre within 4 cm of the pad centre counts as "on that pad" for success
PAD_MIN_DIST = 0.17
PAD_REGION = (-0.26, 0.04, -0.25, 0.25)  # x0, x1, y0, y1 (world)
PAD_REACH = 0.71  # max horizontal distance of a pad centre from the robot base
DROP_REGION = (-0.31, 0.07, -0.30, 0.30)
# beyond ~0.75 m from the base the arm ends up fully stretched (elbow at its joint limit) and the OSC gets stuck
MAX_REACH = 0.73
DROP_CLEAR = PAD_R + 0.05  # drop centre this far from every pad centre -> item fully off the pads
LURE_MARGIN = 0.03  # the lure pad is nearer to the drop spot than any other pad by at least this much
GRASP_DZ = 0.012  # gripper-site height above the table for a top-down grasp of the flat boxes
LIFT = 0.15
BASE_X = -0.66  # robot base x (world); base-frame coordinates = (x - BASE_X, y)
LURE_P = (("source", 0.25), ("spare", 0.5), ("dest", 0.25))

X, Y = "cream_cheese_1", "butter_1"
NAME = {X: "cream cheese", Y: "butter"}
PADS = [f"pad_{i + 1}" for i in range(N_PADS)]


def _pad_xml(model_name):
    c = " ".join(f"{v:.3f}" for v in PAD_RGBA)
    g = (f'        <geom name="{model_name}_g0" type="cylinder" size="{PAD_R:.4f} {PAD_HALF_T:.4f}" rgba="{c}" '
         f'density="10" contype="0" conaffinity="0" group="0" />\n'
         f'        <geom type="cylinder" size="{PAD_R:.4f} {PAD_HALF_T:.4f}" rgba="{c}" conaffinity="0" '
         f'contype="0" group="1" />')
    # flat marker without collisions: gravity compensation keeps it from sinking through the table
    return _wrap(model_name, g, PAD_HALF_T, PAD_R).replace('<body name="object">',
                                                           '<body name="object" gravcomp="1">')


def _yaw_of(env, obj):
    R = env.sim.data.body_xmat[envs.body_id(env, obj)].reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))  # long axis of the boxes = body x
    return (yaw + np.pi / 2) % np.pi - np.pi / 2  # boxes are symmetric under pi


def drop_spot(pads, src, dest, other_xy, lure, phase, dist):
    """Geometry of the injected drop. pads: (N,2); src / dest: start pads of the carried / other item;
    other_xy: current xy of the other item. Returns (xy, lure_pad) or (None, None)."""
    pads = np.asarray(pads, float)
    rest = [i for i in range(len(pads)) if i not in (src, dest)]
    # spare = an empty pad that is neither start pad and not the one the other item is parked on
    occ = [i for i in rest if np.linalg.norm(pads[i] - other_xy) < PAD_R]
    spare_c = [i for i in rest if i not in occ] or rest
    spare = max(spare_c, key=lambda i: np.linalg.norm(pads[i] - other_xy))
    lp = {"source": src, "dest": dest, "spare": spare}[lure]
    for d in (dist, dist + 0.01, dist - 0.01, dist + 0.02):
        for ang in phase + np.linspace(0, 2 * np.pi, 48, endpoint=False):
            xy = pads[lp] + d * np.array([np.cos(ang), np.sin(ang)])
            if not (DROP_REGION[0] <= xy[0] <= DROP_REGION[1] and DROP_REGION[2] <= xy[1] <= DROP_REGION[3]):
                continue
            if np.hypot(xy[0] - BASE_X, xy[1]) > MAX_REACH:
                continue
            dd = np.linalg.norm(pads - xy, axis=1)
            if dd.min() < DROP_CLEAR or np.linalg.norm(xy - other_xy) < 0.12:
                continue
            if np.any(np.delete(dd, lp) < dd[lp] + LURE_MARGIN):
                continue
            return xy, int(lp)
    return None, None


@register_task
class SwapInterrupt(LFFTask):
    name = "l6_swap_interrupt"
    level = "L6"
    category = "procedural_memory"
    capabilities = ("Perceive", "Save", "Retrieve", "Utilize")
    protocol = "within"
    max_attempts = 5
    instruction = ("Swap the positions of the cream cheese and the butter: each must end up on the round pad "
                   "where the other one started. You may use the empty pads as temporary spots.")
    instruction_indirect = ("Swap the positions of the pale blue box with the oval logo and the red-and-yellow "
                            "box: each must end up on the round pad where the other one started. You may use the "
                            "empty pads as temporary spots.")
    inject = True  # set False to run the same scene without the slip (sanity check)

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        cats = [register_generated(f"LffSwapPad{L}", _pad_xml(f"lff_swap_pad_{L.lower()}")) for L in "ABCD"]
        objs = [(X, "cream_cheese", "cc_region", (-0.21, -0.21, -0.19, -0.19)),
                (Y, "butter", "bu_region", (-0.21, 0.19, -0.19, 0.21)),
                (PADS[0], cats[0], "p1_region", (0.05, -0.21, 0.07, -0.19)),
                (PADS[1], cats[1], "p2_region", (0.05, 0.19, 0.07, 0.21)),
                (PADS[2], cats[2], "p3_region", (-0.06, -0.01, -0.04, 0.01)),
                (PADS[3], cats[3], "p4_region", (-0.31, -0.01, -0.29, 0.01))]
        return write_bddl(self.name, self.instruction, objs)

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            pads = []
            while len(pads) < N_PADS:
                p = np.array([rng.uniform(*PAD_REGION[:2]), rng.uniform(*PAD_REGION[2:])])
                if np.hypot(p[0] - BASE_X, p[1]) > PAD_REACH:
                    continue
                if all(np.linalg.norm(p - q) >= PAD_MIN_DIST for q in pads):
                    pads.append(p)
            pads = np.array(pads)
            x_pad, y_pad = 0, 1  # pad order is already random
            u, acc = rng.uniform(), 0.0
            for lure, pr in LURE_P:
                acc += pr
                if u < acc:
                    break
            phase, dist = float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0.115, 0.14))
            # canonical execution: X parked on the empty pad nearest to its start pad, then Y is carried
            empt = [i for i in range(N_PADS) if i not in (x_pad, y_pad)]
            buf = min(empt, key=lambda i: np.linalg.norm(pads[i] - pads[x_pad]))
            xy, lp = drop_spot(pads, y_pad, x_pad, pads[buf], lure, phase, dist)
            if xy is None:
                continue
            return dict(seed=int(seed), pads=pads.round(4).tolist(), x_pad=x_pad, y_pad=y_pad,
                        yaw_x=float(rng.uniform(-0.3, 0.3)), yaw_y=float(rng.uniform(-0.3, 0.3)),
                        lure=lure, drop_phase=phase, drop_dist=dist, drop_yaw=float(rng.uniform(-1.4, 1.4)),
                        trigger_dist=float(rng.uniform(0.05, 0.10)),
                        canonical_buffer=int(buf), canonical_drop_xy=xy.round(4).tolist())
        raise RuntimeError("could not sample a layout")

    def apply_instance(self, inst):
        for i, p in enumerate(inst["pads"]):
            envs.set_obj_pose(self.env, PADS[i], [p[0], p[1], envs.TABLE_Z + PAD_HALF_T + 0.0001])
        envs.place_upright(self.env, X, inst["pads"][inst["x_pad"]], inst["yaw_x"])
        envs.place_upright(self.env, Y, inst["pads"][inst["y_pad"]], inst["yaw_y"])
        self.env.sim.forward()
        self.slip = None
        self.events = []
        self._event_cursor = 0  # for drain_events (harness, per tool call)
        self._outcome_cursor = 0  # for outcome (per attempt)
        self._inst = inst

    def reset_instance(self, inst, recorder=None):
        sk = super().reset_instance(inst)
        self._user_recorder = recorder
        sk.recorder = self._on_step  # slip trigger is checked on every control step
        sk.record_every = 1
        self._home = sk.eef_pos()
        return sk

    def go_home(self):
        """Lift the arm out of the camera's way (end of an attempt)."""
        p = self.sk.eef_pos()
        self.sk.move_to([p[0], p[1], max(p[2], self._home[2])])
        self.sk.move_to(self._home)

    # ------------------------------------------------------------ slip injection (state triggered)
    def _on_step(self, env):
        if self.inject and self.slip is None:
            self._maybe_slip()
        if self._user_recorder is not None and self.sk.n_steps % 4 == 0:
            self._user_recorder(env)

    def _maybe_slip(self):
        """Fires during the first carry of an item while the other item has already left its start pad,
        i.e. the second transport of a swap, whatever order the agent chose."""
        inst, env = self._inst, self.env
        pads = np.asarray(inst["pads"])
        start = {X: inst["x_pad"], Y: inst["y_pad"]}
        for o, other in ((X, Y), (Y, X)):
            p = envs.obj_pos(env, o)
            if p[2] < envs.TABLE_Z + 0.05:
                continue
            if np.linalg.norm(p[:2] - pads[start[o]]) < inst["trigger_dist"]:
                continue
            q = envs.obj_pos(env, other)
            if np.linalg.norm(q[:2] - pads[start[other]]) < PAD_R:
                continue
            if not self.sk.holding(o):
                continue
            self._do_slip(o, other, q[:2])
            return

    def _do_slip(self, o, other, other_xy):
        inst = self._inst
        start = {X: inst["x_pad"], Y: inst["y_pad"]}
        args = (inst["pads"], start[o], start[other], other_xy)
        xy, lp = drop_spot(*args, inst["lure"], inst["drop_phase"], inst["drop_dist"])
        lure = inst["lure"]
        for alt in ("source", "spare", "dest"):  # fallback (only if the agent did something unusual)
            if xy is not None:
                break
            xy, lp = drop_spot(*args, alt, inst["drop_phase"], inst["drop_dist"])
            lure = alt
        if xy is None:
            xy, lp, lure = np.asarray(inst["canonical_drop_xy"]), -1, "fallback"
        envs.place_upright(self.env, o, xy, inst["drop_yaw"])
        self.sk.grip = -1.0  # the gripper opens
        self.env.sim.forward()
        text = f"The {NAME[o]} slipped out of the gripper and fell onto the table."
        self.slip = dict(obj=o, xy=[float(v) for v in xy], lure=lure, lure_pad=lp, step=int(self.sk.n_steps))
        self.events.append(dict(step=int(self.sk.n_steps), kind="slip", obj=o, text=text))

    def drain_events(self):
        """Event texts since the last call (for the harness: append to the current tool-call report)."""
        ev = self.events[self._event_cursor:]
        self._event_cursor = len(self.events)
        return [e["text"] for e in ev]

    # ------------------------------------------------------------ perception helpers (what an agent can see)
    def pad_xy(self):
        return np.array([envs.obj_pos(self.env, p)[:2] for p in PADS])

    def on_pad(self, o, r=PAD_R):
        d = np.linalg.norm(self.pad_xy() - envs.obj_pos(self.env, o)[:2], axis=1)
        return int(np.argmin(d)) if d.min() < r else None

    def empty_pads(self):
        occ = {self.on_pad(X), self.on_pad(Y)}
        return [i for i in range(N_PADS) if i not in occ]

    def _nearest(self, cands, xy):
        pads = self.pad_xy()
        return min(cands, key=lambda i: np.linalg.norm(pads[i] - np.asarray(xy)[:2]))

    @staticmethod
    def _bf(xy):
        return f"({xy[0] - BASE_X:.2f}, {xy[1]:+.2f})"

    def pad_label(self, i, cur=None):
        """Robot-centric name of pad i relative to the other pads (no coordinates in agent feedback)."""
        cur = self.pad_xy() if cur is None else cur
        c = cur.mean(axis=0)
        fb = "far" if cur[i][0] > c[0] else "near"
        lr = "left" if cur[i][1] > c[1] else "right"
        same = [j for j in range(len(cur)) if ("far" if cur[j][0] > c[0] else "near") == fb
                and ("left" if cur[j][1] > c[1] else "right") == lr]
        lab = f"the {fb}-{lr} pad"
        if len(same) > 1:  # two pads in the same quadrant: tell them apart left to right
            order = sorted(same, key=lambda j: -cur[j][1])
            side = "more to the robot's left" if order.index(i) == 0 else "more to the robot's right"
            lab += f" ({side} of the two)"
        return lab

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 20)
        pads = np.asarray(inst["pads"])
        target = {X: inst["y_pad"], Y: inst["x_pad"]}
        res, ok = {}, True
        for o in (X, Y):
            p = envs.obj_pos(self.env, o)
            dt = float(np.linalg.norm(p[:2] - pads[target[o]]))
            resting = bool(envs.TABLE_Z - 0.01 < p[2] < envs.TABLE_Z + 0.05 and not self.sk.holding(o))
            res[o] = dict(xy=[float(p[0]), float(p[1])], on_pad=self.on_pad(o), dist_to_target=dt, resting=resting)
            ok = ok and resting and dt <= SUCC_R
        # F2 detail: only what is observable now + events of this attempt; nothing about the start layout
        parts = []
        cur = self.pad_xy()
        for o in (X, Y):
            r = res[o]
            if r["on_pad"] is not None:
                parts.append(f"the {NAME[o]} is on {self.pad_label(r['on_pad'], cur)}")
            else:
                dd = np.linalg.norm(cur - np.asarray(r["xy"]), axis=1)
                d = dd.min() - PAD_R
                parts.append(f"the {NAME[o]} is on the table, on no pad "
                             f"({100 * d:.0f} cm from the edge of {self.pad_label(int(dd.argmin()), cur)})")
        emp = [self.pad_label(i, cur) for i in self.empty_pads()]
        detail = ("Now " + "; ".join(parts) + "; empty: " + (", ".join(emp) if emp else "none")
                  + " (left/right = the robot's left/right, near/far = toward/away from the robot).")
        new_ev = [e["text"] for e in self.events[self._outcome_cursor:]]
        self._outcome_cursor = len(self.events)
        if new_ev:
            detail = "During this attempt: " + " ".join(new_ev) + " " + detail
        return dict(success=bool(ok), items={NAME[o]: res[o] for o in (X, Y)},
                    slip=self.slip, detail=detail)

    # ------------------------------------------------------------ motion
    def _pick(self, o):
        sk = self.sk
        for _ in range(2):
            p = envs.obj_pos(self.env, o)
            sk.grasp_at(p[:2], envs.TABLE_Z + GRASP_DZ, yaw=_yaw_of(self.env, o), lift=LIFT)
            if sk.holding(o):
                return True
            sk.set_gripper(False, steps=8)
            sk.move_to([p[0], p[1], envs.TABLE_Z + LIFT])
        return False

    def _move(self, o, pad, log):
        """Pick item o and put it on pad `pad`. Returns False if the item slipped on the way."""
        n_ev = len(self.events)
        frm = self.on_pad(o)
        ok = self._pick(o)
        rec = dict(obj=NAME[o], from_pad=frm, to_pad=int(pad), grasped=ok)
        log.append(rec)
        if not ok:
            return True
        xy = self.pad_xy()[pad]
        sk = self.sk
        sk.move_to([xy[0], xy[1], envs.TABLE_Z + GRASP_DZ + LIFT])  # carry (the slip may fire here)
        if len(self.events) > n_ev:
            rec["slipped"] = True
            return False
        sk.move_to([xy[0], xy[1], envs.TABLE_Z + GRASP_DZ + 0.004], tol=0.004)
        sk.set_gripper(False, steps=12)
        sk.move_to([xy[0], xy[1], envs.TABLE_Z + GRASP_DZ + 0.10])
        return True

    def _run(self, moves, log):
        """moves: list of (item, pad) computed up front (an open-loop plan). Stops at a slip."""
        for o, pad in moves:
            if not self._move(o, pad, log):
                return False
        return True

    def _swap_now(self, log, rng=None):
        """The standard swap of the two items' *current* pads via an empty pad (no memory)."""
        a, b = self.on_pad(X), self.on_pad(Y)
        empt = self.empty_pads()
        if a is None or b is None or not empt:
            return
        buf = int(rng.choice(empt)) if rng is not None else self._nearest(empt, self.pad_xy()[a])
        self._run([(X, buf), (Y, a), (X, b)], log)

    def _tidy_stray(self, log):
        for o in (X, Y):
            if self.on_pad(o) is None and self.empty_pads():
                self._move(o, self._nearest(self.empty_pads(), envs.obj_pos(self.env, o)), log)

    def _fixup(self, target, log):
        """Move items until each is on its target pad (target: item -> pad index)."""
        pads = self.pad_xy()
        for _ in range(6):
            todo = [o for o in (Y, X) if np.linalg.norm(envs.obj_pos(self.env, o)[:2] - pads[target[o]]) > SUCC_R]
            if not todo:
                return
            free = [o for o in todo if self.on_pad({X: Y, Y: X}[o], r=PAD_R) != target[o]]
            if free:
                self._move(free[0], target[free[0]], log)
            else:  # the two items block each other: park one on an empty pad that is no one's target
                o = todo[0]
                spare = [i for i in self.empty_pads() if i not in target.values()] or self.empty_pads()
                self._move(o, self._nearest(spare, envs.obj_pos(self.env, o)), log)

    # ------------------------------------------------------------ scripted references (within protocol)
    def _memory(self):
        """Save: what the agent sees in its first frame -- which pad (by position) each item starts on."""
        return {X: self.pad_xy()[self.on_pad(X)].tolist(), Y: self.pad_xy()[self.on_pad(Y)].tolist()}

    def saved_memory(self, first):
        """What the adaptive policy recalls after the slip (identity; patched by the mismatched-memory control)."""
        return first

    def _attempt1(self, kind, inst, mem, rng, first):
        log = []
        plan = dict(kind=kind, moves=log)
        a = self._nearest(range(N_PADS), first[X])  # the open-loop plan is made from the first frame
        b = self._nearest(range(N_PADS), first[Y])
        empt = [i for i in range(N_PADS) if i not in (a, b)]
        buf = self._nearest(empt, self.pad_xy()[a])
        plan["plan"] = f"X->{buf}, Y->{a}, X->{b}"
        if self._run([(X, buf), (Y, a), (X, b)], log):
            plan["recovery"] = "none (no slip)"
            return plan
        # ---- the carry was interrupted by the slip: recover
        if kind == "oracle":  # privileged: true start layout
            plan["recovery"] = "oracle: true layout"
            self._fixup({X: inst["y_pad"], Y: inst["x_pad"]}, log)
        elif kind == "adaptive":  # Retrieve the saved first frame and the step log; finish from the right step
            ta = self._nearest(range(N_PADS), mem[X])
            tb = self._nearest(range(N_PADS), mem[Y])
            plan["recovery"] = f"memory: butter -> pad {ta} (cream cheese's start), cream cheese -> pad {tb}"
            self._fixup({X: tb, Y: ta}, log)
        elif kind in ("naive", "blind_continue"):  # memoryless continuation from the current frame
            stray = [o for o in (X, Y) if self.on_pad(o) is None]
            parked = [o for o in (X, Y) if self.on_pad(o) is not None]
            plan["recovery"] = "current frame: stray item -> nearest empty pad; parked item -> nearest empty pad"
            for o in stray:
                self._move(o, self._nearest(self.empty_pads(), envs.obj_pos(self.env, o)), log)
            for o in parked:
                if self.empty_pads():
                    self._move(o, self._nearest(self.empty_pads(), envs.obj_pos(self.env, o)), log)
        elif kind == "blind":  # memoryless restart: tidy up, then the whole swap again
            plan["recovery"] = "restart: stray item -> nearest empty pad, then swap again"
            self._tidy_stray(log)
            self._swap_now(log, rng)
        else:
            raise ValueError(kind)
        return plan

    def _retry(self, kind, inst, mem, rng):
        log = []
        plan = dict(kind=kind, moves=log)
        if kind == "adaptive":
            ta = self._nearest(range(N_PADS), mem[X])
            tb = self._nearest(range(N_PADS), mem[Y])
            plan["recovery"] = "memory + F2: fix items not on their targets"
            self._fixup({X: tb, Y: ta}, log)
        elif kind in ("blind", "blind_continue"):
            plan["recovery"] = "no history: tidy, then swap the two items again"
            self._tidy_stray(log)
            self._swap_now(log, rng)
        else:
            raise ValueError(kind)
        return plan

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        self.reset_instance(inst)
        first = self._memory()
        mem = self.saved_memory(first)
        history = []
        for a in range(k):
            plan = self._attempt1(kind, inst, mem, rng, first) if a == 0 else self._retry(kind, inst, mem, rng)
            self.go_home()  # "done": the agent declares the attempt finished
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=plan, outcome=out))
            if on_attempt:
                on_attempt(a, plan, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history

    # cross-protocol hooks are not used by this task
    def default_params(self, inst):
        return dict(policy="naive")

    def oracle_params(self, inst):
        return dict(policy="oracle")
