"""Task interface shared by all LFF-Bench tasks.

A task = a LIBERO scene + an instance sampler (layout and hidden variables) + a success / outcome
measurement + scripted reference policies used to validate that the task really requires learning
from failure:

  oracle    knows the hidden variables; should succeed on attempt 1
  naive     acts on the prior (what an agent would do without the failure information)
  adaptive  scripted learner that updates its parameters from structured (F2) feedback
  blind     repeats the naive behaviour with small random variation, no history

Tasks whose policies can be expressed as "execute(params)" get the cross-attempt protocol for free.
Tasks that need within-episode recovery override `run_scripted`.
"""
import numpy as np

from . import envs
from .skills import Skills

TASKS = {}


def register_task(cls):
    TASKS[cls.name] = cls
    return cls


class LFFTask:
    name = "base"
    level = "L?"
    # the user's 7 categories (2026-10-06): hidden information -> required capability -> adapted action
    category = ""  # system_identification | dynamics_adaptation | failed_interaction_inference |
    #                state_restoration | procedural_memory | strategy_switching | geometry_inference
    capabilities = ()
    max_attempts = 5
    protocol = "cross"  # "cross": reset between attempts; "within": one episode, recover in place
    instruction = ""  # direct reference
    instruction_indirect = ""  # descriptive reference (perception / semantic reasoning variant)
    cam_res = 256

    def __init__(self, cam_res=None):
        if cam_res:
            self.cam_res = cam_res
        self.bddl_path = self.make_bddl()
        self.env = envs.make_env(self.bddl_path, cam_res=self.cam_res)
        self.env.reset()
        self.sk = None

    # ------------------------------------------------------------ to implement
    def make_bddl(self):
        raise NotImplementedError

    def sample_instance(self, seed):
        """Return a JSON-serialisable dict with layout + hidden variables."""
        raise NotImplementedError

    def apply_instance(self, inst):
        """Place objects / set hidden physics after env.reset()."""
        raise NotImplementedError

    def outcome(self, inst):
        """dict with at least {'success': bool}; may contain structured measurements."""
        raise NotImplementedError

    def feedback(self, inst, out, level="F2"):
        """Text the agent receives after an attempt. F0: nothing, F1: success flag, F2: measurement."""
        if level == "F0":
            return ""
        s = "Attempt succeeded." if out["success"] else "Attempt failed."
        if level == "F1":
            return s
        return s + (" " + out["detail"] if out.get("detail") else "")

    def skill_bias(self, inst):
        return None

    # scripted references (cross protocol) ----------------------------------
    def default_params(self, inst):
        raise NotImplementedError

    def oracle_params(self, inst):
        raise NotImplementedError

    def adapt_params(self, inst, history):
        raise NotImplementedError

    def blind_params(self, inst, attempt, rng):
        return self.default_params(inst)

    def execute(self, inst, params):
        raise NotImplementedError

    # ------------------------------------------------------------ shared
    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = Skills(self.env, bias=self.skill_bias(inst), recorder=recorder)
        return self.sk

    def snapshot(self, cam="agentview", res=None):
        return envs.render(self.env, cam, res or self.cam_res)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """Cross-attempt protocol with a scripted policy. Returns the per-attempt history."""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":
                params = self.default_params(inst) if a == 0 else self.adapt_params(inst, history)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            else:
                raise ValueError(kind)
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history
