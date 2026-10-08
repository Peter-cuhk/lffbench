"""Scripted backend: no model. Runs a task's scripted reference policy (adaptive by default) through the
agent tool interface, to test the harness end to end.

How: the task's own `execute(inst, params)` runs in a worker thread with `task.sk` replaced by a proxy.
Every Skills primitive it calls (move_to / set_gripper / push / grasp_at / place_at) becomes an agent
tool call that is handed to the harness, executed by the ToolExecutor on the main thread, and its
report is returned to the worker. Only one thread runs at a time (queues hand control back and forth),
so MuJoCo is never touched concurrently. Only schema-valid tool calls are emitted, so this also shows
that the tool interface is expressive enough for the task.

The scripted policy is privileged: it reads the instance (layout) like the task's reference policies
do. It sees earlier attempts only through the memory mode chosen by the harness and only the outcome
fields allowed by the feedback level (F2: structured measurement, F1: success flag, F0: nothing); if
the adaptive rule cannot run on what it sees, it falls back to the task's blind retry rule.
Supported: cross-protocol tasks that implement execute/default_params/adapt_params (+ oracle/blind).
"""
import math
import queue
import threading
import traceback

import numpy as np

from ..context import ToolCall
from .base import Backend, BackendResponse

TIMEOUT_S = 600


class _Abort(Exception):
    pass


class _Channel:
    def __init__(self):
        self.req = queue.Queue()
        self.res = queue.Queue()

    def call(self, name, args):  # worker side
        self.req.put(("call", name, args))
        kind, payload = self.res.get(timeout=TIMEOUT_S)
        if kind == "abort":
            raise _Abort()
        return payload


class SkillsProxy:
    """Stands in for task.sk inside task.execute(); primitives become agent tool calls."""

    def __init__(self, real, channel):
        self._real = real
        self._ch = channel
        self.untooled_hold_steps = 0
        self.ignored_wrap_yaw_false = 0

    def move_to(self, target, yaw=None, speed=None, tol=None, max_steps=None, apply_bias=True, wrap_yaw=True):
        if not wrap_yaw:  # the move_to tool always wraps yaw mod pi (Skills default); counted, not fatal
            self.ignored_wrap_yaw_false += 1
        if not apply_bias:
            raise NotImplementedError("apply_bias=False cannot be expressed through the agent tools")
        yaw_deg = math.degrees(self._real.yaw if yaw is None else float(yaw))
        rep = self._ch.call("move_to", dict(x=float(target[0]), y=float(target[1]), z=float(target[2]),
                                            yaw=round(yaw_deg, 3), speed=None if speed is None else float(speed)))
        return dict(prim="move_to", reached=rep.get("reached"), pos=rep.get("eef_pos"))

    def set_gripper(self, close, steps=15):
        rep = self._ch.call("close_gripper" if close else "open_gripper", {})
        return dict(prim="close" if close else "open", width=rep.get("gripper_width_m"))

    def push(self, start_xy, direction_xy, distance, speed, z=None, pre=0.06):
        rep = self._ch.call("push", dict(x=float(start_xy[0]), y=float(start_xy[1]), dir_x=float(direction_xy[0]),
                                         dir_y=float(direction_xy[1]), distance=float(distance), speed=float(speed),
                                         z=None if z is None else float(z), runup=float(pre)))
        return dict(prim="push", reached=rep.get("sweep_reached"), pos=rep.get("eef_pos"))

    def grasp_at(self, xy, z_grasp, yaw=0.0, approach=0.10, lift=0.15):
        self.set_gripper(False, steps=5)
        self.move_to([xy[0], xy[1], z_grasp + approach], yaw=yaw)
        self.move_to([xy[0], xy[1], z_grasp])
        self.set_gripper(True)
        self.move_to([xy[0], xy[1], z_grasp + lift], speed=0.25)

    def place_at(self, xy, z_release, approach=0.10):
        self.move_to([xy[0], xy[1], z_release + approach])
        self.move_to([xy[0], xy[1], z_release])
        self.set_gripper(False)
        self.move_to([xy[0], xy[1], z_release + approach])

    def hold(self, steps=10):
        # waiting in place has no tool; executed directly (only physics, no new command) and counted
        self.untooled_hold_steps += int(steps)
        self._real.hold(steps)

    def __getattr__(self, k):  # eef_pos, eef_mat, gripper_width, holding, yaw, ...
        return getattr(self._real, k)


class ScriptedBackend(Backend):
    name = "scripted"

    def __init__(self, policy="adaptive", seed=0):
        assert policy in ("adaptive", "naive", "oracle", "blind"), policy
        self.policy = policy
        self.seed = seed
        self.info = None
        self.rng = None

    def config(self):
        return dict(name=self.name, policy=self.policy, seed=self.seed)

    def begin_run(self, info):
        if info.protocol != "cross":
            raise NotImplementedError("ScriptedBackend supports cross-protocol tasks only (execute/adapt_params)")
        self.info = info
        self.rng = np.random.default_rng(self.seed + info.seed)
        self._oracle = None
        if self.policy == "oracle":  # privileged scratch rollouts happen before the harness resets the scene
            self._oracle = info.task.oracle_params(info.inst)

    # ------------------------------------------------------------------ params from visible history
    def _visible(self, rec):
        out = rec.outcome
        if self.info.feedback == "F2":
            vis = dict(out)
        elif self.info.feedback == "F1":
            vis = dict(success=out.get("success"))
        else:
            vis = {}
        return dict(attempt=rec.attempt, params=rec.backend_meta.get("params"), outcome=vis)

    def _params(self, attempt, history):
        task, inst = self.info.task, self.info.inst
        if self.policy == "oracle":
            return dict(self._oracle), "oracle"
        if self.policy == "naive" or not history and self.policy == "adaptive":
            return task.default_params(inst), "default"
        if self.policy == "blind":
            if attempt == 1:
                return task.default_params(inst), "default"
            return task.blind_params(inst, attempt - 1, self.rng), "blind"
        hist = [self._visible(r) for r in history if r.backend_meta.get("params") is not None]
        if not hist:
            return task.blind_params(inst, attempt - 1, self.rng), "blind(no usable history)"
        try:
            return task.adapt_params(inst, hist), "adapt"
        except (KeyError, TypeError, ValueError):
            return task.blind_params(inst, attempt - 1, self.rng), "blind(feedback too coarse)"

    # ------------------------------------------------------------------ attempt
    def begin_attempt(self, ainfo):
        self.params, self.rule = self._params(ainfo.attempt, ainfo.history)
        self.ch = _Channel()
        self.thread = None
        self.pending = False
        self.last_report = None
        self.n = 0
        self.proxy = None

    def _worker(self):
        try:
            self.info.task.execute(self.info.inst, self.params)
            self.ch.req.put(("finished", None))
        except _Abort:
            pass
        except Exception:
            self.ch.req.put(("error", traceback.format_exc()))

    def act(self, conv, tools):
        if self.thread is None:
            task = self.info.task
            self.real_sk = task.sk
            self.proxy = SkillsProxy(task.sk, self.ch)
            task.sk = self.proxy
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        elif self.pending:
            self.ch.res.put(("result", self.last_report or {}))
            self.pending = False
        kind, *rest = self.ch.req.get(timeout=TIMEOUT_S)
        self.n += 1
        if kind == "call":
            name, args = rest[0], rest[1]
            self.pending = True
            tc = ToolCall(id=f"scripted_{self.n}", name=name, arguments=args)
            text = None
            if self.n == 1:
                text = f"[scripted {self.policy}: rule={self.rule}, params={self.params}]"
            return BackendResponse(tool_calls=[tc], text=text)
        if kind == "finished":
            return BackendResponse(tool_calls=[ToolCall(id=f"scripted_{self.n}", name="done",
                                                        arguments=dict(reason=f"scripted {self.policy} policy finished "
                                                                              f"with params {self.params}"))])
        raise RuntimeError("scripted policy crashed:\n" + str(rest[0]))

    def on_tool_result(self, call, result):
        self.last_report = result.report

    def end_attempt(self):
        if self.thread is not None:
            if self.thread.is_alive():
                if self.pending:
                    self.ch.res.put(("abort", None))
                self.thread.join(timeout=30)
            self.info.task.sk = self.real_sk
        meta = dict(policy=self.policy, rule=self.rule, params=self.params)
        if self.proxy is not None and self.proxy.untooled_hold_steps:
            meta["untooled_hold_steps"] = self.proxy.untooled_hold_steps
        if self.proxy is not None and self.proxy.ignored_wrap_yaw_false:
            meta["ignored_wrap_yaw_false"] = self.proxy.ignored_wrap_yaw_false
        return meta
