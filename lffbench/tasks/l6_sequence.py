"""L6-C sequence_resume: weigh four items one by one in a given order; an emergency stop resets part of the work.

Scene: four grocery items (cream cheese, butter, chocolate pudding, popcorn -- small flat boxes), each on its
own round pad (four identical dark pads, flat markers), and a scale (grey square platform, static fixture).
Pad / scale positions and the order are random per instance.

Instruction: weigh the items one at a time in the given order. Weighing = put the item on the scale and let go;
the scale records a reading after the item has rested on it for 0.5 s and beeps. Then the item goes back on its
own pad before the next item is picked up. Every item must be weighed exactly once; the scale log cannot be
edited, so a second reading of an item or a reading out of turn is a terminal failure.

Interruption ("emergency stop", once per episode, inside attempt 1): while the robot carries an item, the gripper
opens, a safety routine puts that item back on its own pad and the arm returns to its home pose. Hidden: how
many items had been weighed when it happened, w in {1..4} (uniform), and the phase: the carried item was on its
way to the scale (not yet weighed, phase A) or on its way back (already weighed, phase B); w = 4 is always
phase B. After the stop every item sits on its pad and the arm is at home: the picture looks like the start of
the task. Which item comes next (order[w], or "nothing, call done" for w = 4) is only in the agent's own history
(what it carried, whether the scale had beeped).

Protocol: within (like l6_swap_interrupt). Events reported immediately (`drain_events`): the scale's beep after
every recorded reading, an error beep for a reading out of turn, and the emergency stop. F2 after `done` gives the
verdict, whether the log is complete, and which items are not on their pads (distance only), never how many
items have been weighed.

Scripted references (perception = simulator state):
  oracle    privileged scale log: always weighs order[len(log)] next.
  adaptive  keeps its own record of readings (an item counts as weighed when the scale beeped while it lay on
            it); after the stop it resumes from order[len(record)] (redoes the interrupted item in phase A, goes on
            with the next one in phase B, calls done when all four are recorded).
  naive     memoryless continuation (1 attempt): the picture after the stop shows no progress; it guesses how far
            it got (uniform over 0..4 items weighed) and continues from there.
  blind     memoryless restart (k attempts): starts the sequence over from the first item -> a second reading of
            an already weighed item (repeated step).
Extra control (not in the acceptance table): `skip` counts items it *started* to weigh (pick-ups) instead of
readings: correct after a phase-B stop, skips the interrupted item after a phase-A stop.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _fmt, _wrap, register_generated
from ..skills import ROT_SCALE, _rot_err
from ..task_base import LFFTask, register_task
from .l6_swap import _pad_xml as _swap_pad_xml

ITEMS = ["cream_cheese_1", "butter_1", "chocolate_pudding_1", "popcorn_1"]
CATS = {"cream_cheese_1": "cream_cheese", "butter_1": "butter", "chocolate_pudding_1": "chocolate_pudding",
        "popcorn_1": "popcorn"}
NAME = {"cream_cheese_1": "cream cheese", "butter_1": "butter", "chocolate_pudding_1": "chocolate pudding",
        "popcorn_1": "popcorn"}
DESCR = {"cream_cheese_1": "the pale blue box with the oval logo", "butter_1": "the red box with white lettering",
         "chocolate_pudding_1": "the dark brown box", "popcorn_1": "the blue-and-yellow box"}
N_ITEMS = len(ITEMS)
PADS = [f"pad_{i + 1}" for i in range(N_ITEMS)]
PAD_R = 0.055
PAD_HALF_T = 0.0008

SCALE_HALF = 0.06  # platform half size (12 x 12 cm)
SCALE_BASE_HALF = 0.067
SCALE_TOP = 0.020  # platform top above the table
SCALE_BASE_H = 0.012
SCALE_RGBA = (0.78, 0.79, 0.80, 1.0)
SCALE_BASE_RGBA = (0.20, 0.20, 0.22, 1.0)

BASE_X = -0.66
REGION = (-0.27, 0.05, -0.27, 0.27)
REACH = 0.70
PAD_MIN = 0.16
SCALE_MIN = 0.175
SUCC_R = 0.04  # item centre within 4 cm of its pad centre
GRASP_DZ = 0.012  # gripper-site height above the support surface for a top-down grasp of the flat boxes
Z_CARRY = envs.TABLE_Z + 0.15
DWELL = 10  # control steps (0.5 s) at rest on the scale before a reading is recorded
LIFTED = 0.04


def scale_xml(model_name):
    parts = [((0, 0, SCALE_BASE_H / 2), (SCALE_BASE_HALF, SCALE_BASE_HALF, SCALE_BASE_H / 2), SCALE_BASE_RGBA),
             ((0, 0, (SCALE_BASE_H + SCALE_TOP) / 2), (SCALE_HALF, SCALE_HALF, (SCALE_TOP - SCALE_BASE_H) / 2),
              SCALE_RGBA)]
    g = []
    for k, (pos, half, rgba) in enumerate(parts):
        g.append(f'        <geom name="{model_name}_g{k}" type="box" pos="{_fmt(pos)}" size="{_fmt(half)}" '
                 f'rgba="{_fmt(rgba)}" density="1000" friction="1.0 0.005 0.0001" group="0" />')
        g.append(f'        <geom type="box" pos="{_fmt(pos)}" size="{_fmt(half)}" rgba="{_fmt(rgba)}" '
                 f'conaffinity="0" contype="0" group="1" />')
    return _wrap(model_name, "\n".join(g), SCALE_TOP / 2, float(np.hypot(SCALE_BASE_HALF, SCALE_BASE_HALF)))


def _yaw_of(env, obj):
    R = env.sim.data.body_xmat[envs.body_id(env, obj)].reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))  # long axis of the boxes = body x
    return (yaw + np.pi / 2) % np.pi - np.pi / 2


@register_task
class SequenceResume(LFFTask):
    name = "l6_sequence_resume"
    level = "L6"
    category = "procedural_memory"
    capabilities = ("Perceive", "Save", "Retrieve", "Utilize")
    protocol = "within"
    agent_max_steps = 100  # robot commands per attempt for agents (a VLM needs ~60-75)
    max_attempts = 5
    _TEMPLATE = ("Weigh the four grocery items one at a time, in this order: {order}. To weigh an item, put it on "
                 "the grey scale and let go of it; the scale beeps when it has recorded the reading. Then put the "
                 "item back on its own round pad before you pick up the next item. Weigh every item exactly once: "
                 "the scale logs every reading, and the log cannot be edited.")
    instruction = _TEMPLATE.format(order="(given per instance)")
    instruction_indirect = instruction
    inject = True
    agent_object_names = [NAME[i] for i in ITEMS] + ["scale", "round pads"]

    def agent_instruction(self, inst, variant="direct"):
        names = DESCR if variant == "indirect" else NAME
        order = [names[o] for o in inst["order"]]
        return self._TEMPLATE.format(order=", ".join(order[:-1]) + ", then " + order[-1])

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        self._mon = False
        self._latch = None
        orig_step = self.env.step

        def step(action):  # per-control-step monitor (also for attempts driven by the agent harness)
            if self._latch is not None:
                action = self._homing_action()
            ret = orig_step(action)
            if self._mon:
                self._monitor()
            return ret

        self.env.step = step

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        pads = [register_generated(f"LffSeqPad{L}", _swap_pad_xml(f"lff_seq_pad_{L.lower()}")) for L in "ABCD"]
        scale = register_generated("LffSeqScale", scale_xml("lff_seq_scale"), free=False)
        objs = [(o, CATS[o], f"r{i}", (-0.26 + 0.15 * i - 0.01, -0.31, -0.26 + 0.15 * i + 0.01, -0.29))
                for i, o in enumerate(ITEMS)]
        objs += [(PADS[i], pads[i], f"p{i}", (-0.26 + 0.15 * i - 0.01, 0.29, -0.26 + 0.15 * i + 0.01, 0.31))
                 for i in range(N_ITEMS)]
        fx = [("scale_1", scale, "scale_region", (0.19, -0.01, 0.21, 0.01))]
        return write_bddl(self.name, "Weigh the four items one at a time in the given order", objs, fx)  # no ; in BDDL

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(5000):
            pts = []
            while len(pts) < N_ITEMS + 1:
                p = np.array([rng.uniform(*REGION[:2]), rng.uniform(*REGION[2:])])
                if np.hypot(p[0] - BASE_X, p[1]) > REACH:
                    continue
                pts.append(p)
            scale, pads = pts[0], np.array(pts[1:])
            dp = [np.linalg.norm(pads[i] - pads[j]) for i in range(N_ITEMS) for j in range(i)]
            if min(dp) >= PAD_MIN and min(np.linalg.norm(pads - scale, axis=1)) >= SCALE_MIN:
                break
        else:
            raise RuntimeError("could not sample a layout")
        order = [ITEMS[i] for i in rng.permutation(N_ITEMS)]
        w = int(rng.integers(1, N_ITEMS + 1))  # items weighed when the stop happens
        phase = "B" if w == N_ITEMS else str(rng.choice(["A", "B"]))
        inst = dict(seed=int(seed), order=order, pads=pads.round(4).tolist(), scale_xy=scale.round(4).tolist(),
                    item_off=[[float(v) for v in rng.uniform(-0.008, 0.008, 2)] for _ in range(N_ITEMS)],
                    item_yaw=[float(v) for v in rng.uniform(-0.3, 0.3, N_ITEMS)],
                    stop_w=w, stop_phase=phase, trigger_dist=float(rng.uniform(0.05, 0.09)),
                    stop_off=[float(v) for v in rng.uniform(-0.008, 0.008, 2)],
                    stop_yaw=float(rng.uniform(-0.3, 0.3)))
        inst["stop_item"] = order[w] if phase == "A" else order[w - 1]
        self.instruction = self.agent_instruction(inst, "direct")  # make_video prints task.instruction
        self.instruction_indirect = self.agent_instruction(inst, "indirect")
        return inst

    def apply_instance(self, inst):
        m = self.env.sim.model
        for i, p in enumerate(inst["pads"]):
            envs.set_obj_pose(self.env, PADS[i], [p[0], p[1], envs.TABLE_Z + PAD_HALF_T + 0.0001])
        bid = m.body_name2id(self.env.fixtures_dict["scale_1"].root_body)
        m.body_pos[bid] = [inst["scale_xy"][0], inst["scale_xy"][1], envs.TABLE_Z]
        m.body_quat[bid] = [1, 0, 0, 0]
        for i, o in enumerate(ITEMS):
            p = np.asarray(inst["pads"][i]) + inst["item_off"][i]
            envs.place_upright(self.env, o, p, inst["item_yaw"][i])
        self.env.sim.forward()
        self._inst = inst
        self.pad_of = {o: i for i, o in enumerate(ITEMS)}  # item i starts on pad i (pad positions are random)
        self.log = []  # scale readings (item names)
        self.error = None
        self.stop = None
        self.events = []
        self._event_cursor = 0
        self._outcome_cursor = 0
        self._dwell = {o: 0 for o in ITEMS}
        self._visit = {o: False for o in ITEMS}
        self._prev = {}
        self._latch = None
        self._mon = False

    def reset_instance(self, inst, recorder=None):
        self._mon = False
        self._latch = None
        sk = super().reset_instance(inst, recorder=recorder)
        self._home = sk.eef_pos()
        self.rest_dz = {o: float(envs.obj_pos(self.env, o)[2] - envs.TABLE_Z) for o in ITEMS}
        self._prev = {o: envs.obj_pos(self.env, o) for o in ITEMS}
        self._mon = True
        return sk

    def go_home(self):
        p = self.sk.eef_pos()
        self.sk.move_to([p[0], p[1], max(p[2], self._home[2])])
        self.sk.move_to(self._home)

    # ------------------------------------------------------------ helpers
    def scale_xy(self):
        return np.asarray(self._inst["scale_xy"], float)

    def pad_xy(self, o):
        return np.asarray(self._inst["pads"][self.pad_of[o]], float)

    def on_scale(self, o, p=None):
        p = envs.obj_pos(self.env, o) if p is None else p
        d = np.abs(p[:2] - self.scale_xy())
        return bool(d.max() <= SCALE_HALF and abs(p[2] - (envs.TABLE_Z + SCALE_TOP + self.rest_dz[o])) < 0.008)

    def on_own_pad(self, o):
        p = envs.obj_pos(self.env, o)
        return bool(np.linalg.norm(p[:2] - self.pad_xy(o)) <= SUCC_R
                    and abs(p[2] - (envs.TABLE_Z + self.rest_dz[o])) < 0.01 and not self.sk.holding(o))

    # ------------------------------------------------------------ monitor: scale readings + emergency stop
    def _monitor(self):
        env = self.env
        if self._latch is not None:
            if len(self.sk.log) > self._latch and np.linalg.norm(self.sk.eef_pos() - self._home) < 0.01:
                self._latch = None
        for o in ITEMS:
            p = envs.obj_pos(env, o)
            still = np.linalg.norm(p - self._prev[o]) < 0.0008
            self._prev[o] = p
            if self.on_scale(o, p):
                if still and not self.sk.holding(o):
                    self._dwell[o] += 1
                    if self._dwell[o] >= DWELL and not self._visit[o]:
                        self._visit[o] = True
                        self._record(o)
                else:
                    self._dwell[o] = 0
            else:
                self._dwell[o] = 0
                if np.abs(p[:2] - self.scale_xy()).max() > SCALE_HALF or \
                        p[2] > envs.TABLE_Z + SCALE_TOP + self.rest_dz[o] + 0.02:
                    self._visit[o] = False
        if self.inject and self.stop is None:
            self._maybe_stop()

    def _record(self, o):
        self.log.append(o)
        k = len(self.log)
        ok = self.log == self._inst["order"][:k]
        if ok:
            self.events.append(dict(step=int(self.sk.n_steps), kind="beep", obj=o,
                                    text="The scale beeped: reading recorded."))
        else:
            if self.error is None:
                self.error = dict(item=o, position=k, step=int(self.sk.n_steps),
                                  kind="repeat" if self.log.count(o) > 1 else "out_of_order")
            self.events.append(dict(step=int(self.sk.n_steps), kind="error", obj=o,
                                    text="The scale beeped three times: error, this reading does not follow the "
                                         "requested order. The log cannot be edited."))

    def _maybe_stop(self):
        inst = self._inst
        o, w = inst["stop_item"], inst["stop_w"]
        if self.log != inst["order"][:w]:
            return
        p = envs.obj_pos(self.env, o)
        if inst["stop_phase"] == "A":
            if p[2] < envs.TABLE_Z + self.rest_dz[o] + LIFTED:
                return
            if np.linalg.norm(p[:2] - self.pad_xy(o)) < inst["trigger_dist"]:
                return
        else:
            if p[2] < envs.TABLE_Z + SCALE_TOP + self.rest_dz[o] + LIFTED:
                return
            if np.linalg.norm(p[:2] - self.scale_xy()) < inst["trigger_dist"]:
                return
        if not self.sk.holding(o):
            return
        self._do_stop(o)

    def _do_stop(self, o):
        inst = self._inst
        xy = self.pad_xy(o) + np.asarray(inst["stop_off"])
        envs.place_upright(self.env, o, xy, inst["stop_yaw"])
        self.sk.grip = -1.0
        self.sk.yaw = 0.0
        self._latch = len(self.sk.log)  # drive the arm home until the current primitive has returned
        self._prev[o] = envs.obj_pos(self.env, o)
        self.env.sim.forward()
        text = ("Emergency stop! The gripper opened, the safety routine put the item the robot was holding back on "
                "its pad and moved the arm to its home pose. You may continue the task.")
        self.stop = dict(item=o, phase=inst["stop_phase"], weighed=len(self.log), step=int(self.sk.n_steps))
        self.events.append(dict(step=int(self.sk.n_steps), kind="stop", obj=o, text=text))

    def _homing_action(self):
        sk = self.sk
        a = np.zeros(self.env.action_dim)
        a[:3] = np.clip((self._home - sk.eef_pos()) / sk.pos_scale, -1, 1)
        a[3:6] = np.clip(_rot_err(sk.R0, sk.eef_mat()) / ROT_SCALE, -1, 1)
        a[6] = -1.0
        return a

    def drain_events(self):
        ev = self.events[self._event_cursor:]
        self._event_cursor = len(self.events)
        return [e["text"] for e in ev]

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 20)
        order = inst["order"]
        complete = self.log == order
        terminal = self.error is not None
        off = [o for o in ITEMS if not self.on_own_pad(o)]
        success = bool(complete and not off and not terminal)
        if success:
            head = "All four items were weighed once, in the requested order, and are back on their pads."
        elif terminal:
            head = ("The scale log does not follow the requested order (an item was weighed a second time or out "
                    "of turn). The log cannot be edited, so the task can no longer be completed.")
        elif complete:
            head = "Every item has been weighed, but not every item is back on its own pad."
        else:
            head = "Not every item has been weighed yet."
        parts = []
        if not success and not terminal:
            for o in off:
                p = envs.obj_pos(self.env, o)
                if self.sk.holding(o):
                    parts.append(f"the {NAME[o]} is still in the gripper")
                elif self.on_scale(o, p):
                    parts.append(f"the {NAME[o]} is on the scale")
                else:
                    d = np.linalg.norm(p[:2] - self.pad_xy(o))
                    parts.append(f"the {NAME[o]} is {100 * d:.0f} cm from the centre of its pad")
        detail = head + (" Also: " + "; ".join(parts) + "." if parts else "")
        new = [e for e in self.events[self._outcome_cursor:] if e["kind"] == "stop"]
        self._outcome_cursor = len(self.events)
        if new:
            detail = "During this attempt: " + " ".join(e["text"] for e in new) + " " + detail
        return dict(success=success, terminal=terminal, log=[NAME[o] for o in self.log], n_weighed=len(self.log),
                    complete=complete, error=self.error, stop=self.stop,
                    off_pad=[NAME[o] for o in off], detail=detail)

    # ------------------------------------------------------------ motion (scripted policies)
    def _support_z(self, o):
        p = envs.obj_pos(self.env, o)
        return envs.TABLE_Z + (SCALE_TOP if self.on_scale(o, p) or
                               (np.abs(p[:2] - self.scale_xy()).max() < SCALE_HALF) else 0.0)

    def _n_stops(self):
        return sum(e["kind"] == "stop" for e in self.events)

    def _pick(self, o):
        sk = self.sk
        for _ in range(2):
            p = envs.obj_pos(self.env, o)
            sk.grasp_at(p[:2], self._support_z(o) + GRASP_DZ, yaw=_yaw_of(self.env, o), lift=0.12)
            if sk.holding(o):
                return True
            sk.set_gripper(False, steps=8)
            sk.move_to([p[0], p[1], self._support_z(o) + GRASP_DZ + 0.12])
        return False

    def _carry_to(self, o, xy, z_support):
        """Carry the held item to xy and put it down. Returns False if an emergency stop happened."""
        sk = self.sk
        n0 = self._n_stops()
        sk.move_to([xy[0], xy[1], Z_CARRY])
        if self._n_stops() > n0:
            return False
        sk.move_to([xy[0], xy[1], z_support + GRASP_DZ + 0.004], tol=0.004)
        sk.set_gripper(False, steps=12)
        sk.move_to([xy[0], xy[1], z_support + GRASP_DZ + 0.10])
        return self._n_stops() == n0

    def _weigh(self, o):
        """Weigh one item (pad -> scale -> pad). Returns (status, beeped): status 'ok' / 'stop_A' / 'stop_B' /
        'no_grasp'; beeped = the scale beeped while the item lay on it (what the agent hears)."""
        n_beep = sum(e["kind"] in ("beep", "error") for e in self.events)
        if not self._pick(o):
            return "no_grasp", False
        if not self._carry_to(o, self.scale_xy(), envs.TABLE_Z + SCALE_TOP):
            return "stop_A", False
        self.sk.hold(6)
        beeped = sum(e["kind"] in ("beep", "error") for e in self.events) > n_beep
        if not self._pick(o):
            return "no_grasp", beeped
        if not self._carry_to(o, self.pad_xy(o), envs.TABLE_Z):
            return "stop_B", beeped
        return "ok", beeped

    def _fix_pads(self, log):
        for o in ITEMS:
            if not self.on_own_pad(o) and not self.on_scale(o):
                if self._pick(o):
                    self._carry_to(o, self.pad_xy(o), envs.TABLE_Z)
                    log.append((NAME[o], "back to pad"))

    def _run_from(self, k, record, log, stop_on_stop=False):
        """Weigh order[k:] in turn, keeping `record` (the policy's own list of readings it heard)."""
        order = self._inst["order"]
        i = k
        guard = 0
        while i < len(order) and guard < 10 and self.error is None:
            guard += 1
            o = order[i]
            st, beeped = self._weigh(o)
            log.append((NAME[o], st, beeped))
            if beeped:
                record.append(o)
            if st == "no_grasp":
                continue
            if st.startswith("stop"):
                if stop_on_stop:
                    return "stopped"
                i = len(record)  # resume from the own record
                continue
            i += 1
        return "ok"

    # ------------------------------------------------------------ scripted references (within protocol)
    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed + 104729)
        self.reset_instance(inst)
        order = inst["order"]
        history = []
        record = []  # adaptive's Save: readings it heard
        for a in range(k):
            log = []
            plan = dict(kind=kind, steps=log)
            if a == 0:
                if kind == "oracle":
                    while len(self.log) < len(order) and self.error is None and len(log) < 12:
                        o = order[len(self.log)]  # privileged scale log
                        st, _ = self._weigh(o)
                        log.append((NAME[o], st))
                    plan["recovery"] = "oracle: privileged scale log"
                elif kind == "adaptive":
                    self._run_from(0, record, log)
                    plan["recovery"] = "own record of readings: resume from order[len(record)]"
                elif kind == "skip":
                    started = []
                    i = 0
                    while i < len(order) and self.error is None and len(log) < 12:
                        o = order[i]
                        st, beeped = self._weigh(o)
                        log.append((NAME[o], st, beeped))
                        if st == "no_grasp":
                            continue
                        started.append(o)
                        i = len(started)  # counts started items, not readings
                    plan["recovery"] = "counts items it started to weigh (the interrupted one too)"
                else:  # memoryless after the stop
                    r = self._run_from(0, record, log, stop_on_stop=True)
                    if r == "stopped":
                        if kind == "naive":
                            g = int(rng.integers(0, len(order) + 1))
                            plan["recovery"] = f"memoryless: guesses {g} items already weighed"
                            self._run_from(g, [], log)
                        elif kind == "blind":
                            plan["recovery"] = "memoryless restart from the first item"
                            self._run_from(0, [], log)
                        else:
                            raise ValueError(kind)
                    else:
                        plan["recovery"] = "none (no stop)"
            else:
                if kind in ("adaptive", "oracle", "skip"):
                    plan["recovery"] = "memory + F2: put items back on their pads / finish from own record"
                    if kind == "adaptive" and len(record) < len(order):
                        self._run_from(len(record), record, log)
                    if kind == "skip" and not history[-1]["outcome"]["complete"]:
                        # F2 says the log is incomplete: with memory, the item carried at the stop was not weighed
                        stopped = [s[0] for h in history for s in h["params"]["steps"] if str(s[1]).startswith("stop")]
                        if stopped:
                            o = next(i for i in ITEMS if NAME[i] == stopped[-1])
                            st, _ = self._weigh(o)
                            log.append((NAME[o], st))
                            plan["recovery"] = "memory + F2 'not every item weighed': weigh the item carried at the stop"
                    self._fix_pads(log)
                elif kind == "blind":
                    plan["recovery"] = "memoryless restart from the first item"
                    self._run_from(0, [], log)
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

    def default_params(self, inst):
        return dict(policy="naive")

    def oracle_params(self, inst):
        return dict(policy="oracle")
