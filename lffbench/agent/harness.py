"""Protocol controller: runs a backend on task instances, attempt by attempt.

  cross protocol : reset to the same instance (same layout + hidden variables) before every attempt
  within protocol: one continuous episode; each `done` call closes a segment ("attempt") that is checked;
                   the agent continues from the current state if it failed and budget remains
Both protocols pack finished attempts the same way (keyframes <= 4, action summary, feedback F0/F1/F2) and
show them according to the memory mode (none / full / last / mismatched).
"""
import time
from dataclasses import asdict, dataclass, field

from .context import (FEEDBACK_LEVELS, MEMORY_MODES, AttemptRecord, Conversation, PromptBuilder, StepRecord,
                      pick_keyframes, select_history, summarize_actions)
from .cost import call_cost, estimate_request_tokens
from .metrics import run_summary
from .backends.base import AttemptInfo, RunInfo


@dataclass
class HarnessConfig:
    memory: str = "full"
    feedback: str = "F2"
    k: int = None  # attempts (cross) / done calls (within); default task.max_attempts
    image_res: int = 512
    keyframe_res: int = None  # default = image_res
    n_keyframes: int = 4
    cams: tuple = ("agentview", "wrist")
    auto_obs: bool = True  # send new images after every action tool
    max_steps: int = 30  # tool calls per attempt
    max_no_tool: int = 3  # consecutive replies without a tool call before the attempt is ended
    privileged_locate: bool = False
    grasp_report: str = "bool"  # "bool" | "name" | "none"
    instruction: str = "direct"  # "direct" | "indirect"
    early_stop: str = "auto"  # "auto" (stop at first success unless F0), "yes", "no"
    settle_within: int = 20  # control steps held still before checking a within-protocol segment
    jpeg_quality: int = 90
    save_images: bool = True
    model_for_cost: str = "gpt-6-astra"
    extra: dict = field(default_factory=dict)

    def validate(self):
        assert self.memory in MEMORY_MODES, self.memory
        assert self.feedback in FEEDBACK_LEVELS, self.feedback
        assert self.grasp_report in ("bool", "name", "none")
        assert self.instruction in ("direct", "indirect")
        assert self.early_stop in ("auto", "yes", "no")
        assert 1 <= self.n_keyframes <= 4 or self.n_keyframes == 0


def clean_inst(inst):
    return {k: v for k, v in inst.items() if not str(k).startswith("_")}


class AgentHarness:
    def __init__(self, task, backend, cfg: HarnessConfig, logger, payload_probe=None, executor_factory=None,
                 cam_axes=None):
        cfg.validate()
        self.task, self.backend, self.cfg, self.log = task, backend, cfg, logger
        self.protocol = getattr(task, "protocol", "cross")
        self.k = cfg.k or task.max_attempts
        self.payload_probe = payload_probe  # optional OpenAIResponsesBackend used only to size requests (dry run)
        self.executor_factory = executor_factory  # tests: build a fake executor instead of ToolExecutor
        self._cam_axes = cam_axes

    # ------------------------------------------------------------------ helpers
    def instruction(self, inst):
        hook = getattr(self.task, "agent_instruction", None)
        if hook is not None:
            return hook(inst, self.cfg.instruction)
        if self.cfg.instruction == "indirect" and getattr(self.task, "instruction_indirect", ""):
            return self.task.instruction_indirect
        return self.task.instruction

    def feedback(self, inst, out, level):
        import inspect
        fn = self.task.feedback
        if "indirect" in inspect.signature(fn).parameters:  # tasks that phrase feedback per instruction variant
            return fn(inst, out, level, indirect=self.cfg.instruction == "indirect")
        return fn(inst, out, level)

    def early_stop(self):
        if self.cfg.early_stop == "auto":
            return self.cfg.feedback != "F0"
        return self.cfg.early_stop == "yes"

    def terminal(self, out):
        hook = getattr(self.task, "is_terminal", None)
        if hook is not None:
            return bool(hook(out))
        return bool(out.get("terminal") or out.get("overshoot"))

    def _executor(self, inst):
        if self.executor_factory is not None:
            return self.executor_factory(self, inst)
        from .tools import ToolExecutor
        return ToolExecutor(self.task, inst, image_res=self.cfg.image_res, cams=self.cfg.cams,
                            privileged_locate=self.cfg.privileged_locate, grasp_report=self.cfg.grasp_report,
                            jpeg_quality=self.cfg.jpeg_quality, image_sink=self._sink)

    def _sink(self, ref):
        self._img_counter += 1
        rel = f"seed{self._seed}/a{self._attempt}_s{self._step:02d}_{self._img_counter:03d}_{ref.camera}.jpg"
        return self.log.save_image(ref, rel)

    def _keyframe(self, ref):
        """Separate (possibly smaller) copy of an agentview frame for the history."""
        from .images import ImageRef
        kres = self.cfg.keyframe_res or self.cfg.image_res
        kf = ImageRef(camera=ref.camera, array=ref.array, label=ref.label)
        if kres == self.cfg.image_res:
            kf.jpeg, kf.path = ref.jpeg, ref.path
            return kf
        self._img_counter += 1
        rel = f"seed{self._seed}/a{self._attempt}_key_{self._img_counter:03d}_{ref.camera}.jpg"
        return self.log.save_image(kf, rel, res=kres)

    def prompts(self, executor):
        if self._cam_axes is None:
            from . import camera as C
            self._cam_axes = {c: C.describe_axes(self.task.env, c, self.cfg.image_res) for c in C.CAMERAS}
        names = getattr(self.task, "agent_object_names", None) or list(self.task.env.objects_dict)
        return PromptBuilder(self.task, self.protocol, self.k, self.cfg.memory, self.cfg.feedback, self.cfg.image_res,
                             self._cam_axes, privileged_locate=self.cfg.privileged_locate, object_names=names,
                             grasp_report=self.cfg.grasp_report, cams=self.cfg.cams)

    # ------------------------------------------------------------------ run
    def run_instance(self, seed, donor=None):
        """donor: AttemptRecords of another instance (memory=mismatched). Returns (summary, records)."""
        task, cfg = self.task, self.cfg
        inst = task.sample_instance(seed)
        self._seed, self._attempt, self._step, self._img_counter = seed, 0, 0, 0
        self._llm_calls = []
        t_run = time.time()
        sim_wall = 0.0
        info = RunInfo(task=task, inst=inst, seed=seed, k=self.k, protocol=self.protocol, feedback=cfg.feedback,
                       memory=cfg.memory)
        self.backend.begin_run(info)
        self.log.event("run_start", seed=seed, inst=clean_inst(inst), task=task.name, protocol=self.protocol, k=self.k,
                       memory=cfg.memory, feedback=cfg.feedback, donor_seed=donor[0].seed if donor else None)
        records, executor, status = [], None, "ok"
        for a in range(1, self.k + 1):
            self._attempt, self._step = a, 0
            if self.protocol == "cross" or executor is None:
                t0 = time.time()
                task.reset_instance(inst)
                executor = self._executor(inst)
                sim_wall += time.time() - t0
            visible = select_history(cfg.memory, records, donor)
            rec, attempt_sim = self._attempt_loop(inst, seed, a, visible, executor,
                                                  continuing=self.protocol == "within" and a > 1)
            sim_wall += attempt_sim
            records.append(rec)
            if rec.ended_by == "backend_error":
                status = "error"
                break
            if rec.success and self.early_stop():
                break
            if self.protocol == "within" and (rec.success or self.terminal(rec.outcome)):
                break
        self.backend.end_run()
        summ = run_summary(seed, records, self.k, status=status, llm_calls=self._llm_calls,
                           extra=dict(sim_wall_s=round(sim_wall, 2), wall_s=round(time.time() - t_run, 2),
                                      donor_seed=donor[0].seed if donor else None))
        self.log.event("run_end", **summ)
        summ["_llm_calls"] = self._llm_calls
        return summ, records

    def _attempt_loop(self, inst, seed, a, visible, ex, continuing):
        cfg, task, backend = self.cfg, self.task, self.backend
        P = self.prompts(ex)
        tools = self.tool_specs()
        conv = Conversation(P.system_prompt())
        backend.begin_attempt(AttemptInfo(attempt=a, history=visible, continuing=continuing))
        sim_t = 0.0
        t0 = time.time()
        obs0 = ex.observe(label="start")
        sim_t += time.time() - t0
        conv.add_user(P.history_parts(self.instruction(inst), visible), live=False)
        conv.add_user(P.current_parts(a, obs0, ex.proprio(), continuing), live=True)
        steps, diagnosis, done_reason, ended_by = [], None, None, None
        n_calls, no_tool = 0, 0
        while ended_by is None:
            if len(steps) >= cfg.max_steps:
                ended_by = "step_budget"
                break
            try:
                resp = backend.act(conv, tools)
            except Exception as e:  # API failure after retries, crashed scripted policy, ...
                self.log.event("llm_error", seed=seed, attempt=a, call=n_calls + 1, error=repr(e)[:4000])
                ended_by = "backend_error"
                break
            n_calls += 1
            call_rec = self._log_llm_call(seed, a, n_calls, resp, conv, tools)
            self._llm_calls.append(call_rec)
            if n_calls == 1 and resp.text:
                diagnosis = resp.text
            conv.add_assistant(resp.text, resp.tool_calls, raw=resp.raw, backend=backend.name)
            if not resp.tool_calls:
                no_tool += 1
                if no_tool >= cfg.max_no_tool:
                    ended_by = "no_tool_call"
                    break
                conv.add_user(P.nudge(), live=True)
                continue
            no_tool = 0
            acted = None
            for tc in resp.tool_calls:
                self._step = len(steps) + 1
                res = ex.call(tc.name, tc.raw_arguments or tc.arguments)
                sim_t += res.wall_s
                conv.add_tool_output(tc.id, tc.name, res.text())
                backend.on_tool_result(tc, res)
                st = StepRecord(step=self._step, name=tc.name, args=res.args, report=res.report, ok=res.ok,
                                wall_s=round(res.wall_s, 3), sim_steps=res.sim_steps)
                steps.append(st)
                if res.images:
                    conv.add_user(P.observation_parts(f"Images returned by get_observation (step {self._step})",
                                                      res.images, ex.proprio()), live=True)
                if tc.name in ("move_to", "open_gripper", "close_gripper", "push") and res.ok:
                    acted = st
                self.log.event("tool_call", seed=seed, attempt=a, step=self._step, call_id=tc.id, name=tc.name,
                               args=res.args, ok=res.ok, report=res.report, wall_s=round(res.wall_s, 3),
                               sim_steps=res.sim_steps, images=[i.path for i in res.images])
                if res.ended:
                    done_reason = res.args.get("reason")
                    ended_by = "done"
                    break
                if len(steps) >= cfg.max_steps:
                    break
            if ended_by is None and acted is not None:
                t1 = time.time()
                if cfg.auto_obs:
                    obs = ex.observe(label=f"step{acted.step}")
                    conv.add_user(P.step_observation(acted.step, acted.name, obs, ex.proprio()), live=True)
                    acted.frame = next((o for o in obs if o.camera == "agentview"), obs[0])
                else:  # still keep an agentview frame for the history keyframes / logs
                    from .images import ImageRef
                    acted.frame = self._sink(ImageRef(camera="agentview", array=ex.render("agentview"),
                                                      label=f"step{acted.step}"))
                sim_t += time.time() - t1
                self.log.event("observation", seed=seed, attempt=a, step=acted.step,
                               images=[acted.frame.path] if not cfg.auto_obs else [o.path for o in obs],
                               proprio=ex.proprio())
        meta = backend.end_attempt() or {}
        # ------------------------------------------------------------ outcome + packing
        t1 = time.time()
        home = getattr(task, "go_home", None)  # task-defined end-of-attempt motion (e.g. L6: arm out of view)
        if home is not None and ended_by != "backend_error":
            home()
        if self.protocol == "within" and cfg.settle_within:
            ex.sk.hold(cfg.settle_within)
        out = task.outcome(inst)
        from .images import ImageRef
        self._step = 99
        final = self._sink(ImageRef(camera="agentview", array=ex.render("agentview", cfg.keyframe_res or cfg.image_res),
                                    label="end"))
        sim_t += time.time() - t1
        fb = self.feedback(inst, out, cfg.feedback)
        fb_full = self.feedback(inst, out, "F2")
        init_frame = next((o for o in obs0 if o.camera == "agentview"), None)
        for s in steps:
            if s.frame is not None and (cfg.keyframe_res or cfg.image_res) != cfg.image_res:
                s.frame = self._keyframe(s.frame)
        if init_frame is not None and (cfg.keyframe_res or cfg.image_res) != cfg.image_res:
            init_frame = self._keyframe(init_frame)
        kfs = pick_keyframes(steps, init_frame, final, cfg.n_keyframes) if cfg.n_keyframes else []
        rec = AttemptRecord(seed=seed, attempt=a, steps=steps, keyframes=kfs, outcome=dict(out),
                            success=bool(out.get("success")), feedback=fb, done_reason=done_reason, diagnosis=diagnosis,
                            ended_by=ended_by, backend_meta=meta)
        calls = [c for c in self._llm_calls if c["attempt"] == a]
        self.log.event("attempt_end", seed=seed, attempt=a, success=rec.success, outcome=out, feedback_shown=fb,
                       feedback_f2=fb_full, ended_by=ended_by, done_reason=done_reason, diagnosis=diagnosis,
                       n_steps=len(steps), n_llm_calls=len(calls), backend_meta=meta,
                       history_shown=[(h.seed, h.attempt) for h in visible],
                       keyframes=[(c, r.path) for c, r in kfs], action_summary=summarize_actions(steps),
                       llm_latency_s=round(sum(c["latency_s"] for c in calls), 3), sim_wall_s=round(sim_t, 3))
        return rec, sim_t

    def tool_specs(self):
        from .tools import tool_specs
        names = getattr(self.task, "agent_object_names", None) or list(self.task.env.objects_dict)
        return tool_specs(self.cfg.privileged_locate, names, self.cfg.image_res)

    def _log_llm_call(self, seed, a, n, resp, conv, tools):
        rec = dict(seed=seed, attempt=a, call=n, latency_s=round(resp.latency_s, 3), usage=resp.usage,
                   cost_usd=call_cost(resp.usage, self.cfg.model_for_cost) if resp.usage else None,
                   status=resp.status, error=resp.error, text=resp.text, reasoning_summary=resp.reasoning_summary,
                   tool_calls=[dict(id=t.id, name=t.name, arguments=t.arguments) for t in resp.tool_calls],
                   request=resp.request_stats, conv_images=conv.n_images())
        if self.payload_probe is not None:  # dry run: size of the request GPT-6 would have received
            stats = self.payload_probe.request_stats(self.payload_probe.build_request(conv, tools))
            stats["est_input_tokens"] = estimate_request_tokens(stats, self.cfg.image_res)
            rec["dry_run_request"] = stats
        self.log.event("llm_call", **rec)
        return rec


def config_dict(cfg):
    return asdict(cfg)
