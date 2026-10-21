"""Replay logged LA-interface agent runs (Codex, Claude Code, the API driver) on the same task code, verify the replay
against the logged robot reports, and (record mode) write a dense 20 Hz trajectory with annotations, the input of the
LeRobot export (scripts/lerobot_export.py).

  python scripts/lerobot_replay.py check  <run_dir> [<run_dir> ...] [--json out.json]   # replay only, print fidelity
  python scripts/lerobot_replay.py record <run_dir> --out <raw_root> [--res 256]

Why an exact replay is possible: the robot server owns the simulator, and the agent changes it only through robot
commands, which events.jsonl logs with their arguments in order (`call` / `done`). MuJoCo is deterministic, robosuite
is built with initialization_noise=None, and the instance comes from the seed. The replay re-issues the commands exactly
like robot_server._handle_v2 / _done (same defaults, same rejection rules, same resets between attempts, the same
outcome() calls, which also step the simulator) and compares every logged report (eef position, yaw, gripper width,
ok, holding / touched flags) and every logged outcome dict with the replayed one. Only runs whose contract.json code
hashes equal the current files are expected to match (check_code()).

Record mode wraps env.step: before each control step it stores the observation of that step (agentview + wrist RGB,
proprioception, MuJoCo state) and the 7-D OSC action passed to env.step, i.e. frame t = (o_t, a_t). Episodes:
protocol "cross" -> one episode per attempt (the scene is reset between attempts); protocol "within" -> one episode
per run (the scene persists; attempts are segments of the episode). Rendering does not change the physics, and the
fidelity check runs in record mode too. Per episode the recorder writes <raw_root>/<batch>/<run>/epNN/:
  agentview.mp4, wrist.mp4   lossless RGB H.264 (intermediate; the export re-encodes)
  frames.npz                 per-frame numeric arrays (see Recorder.state) + labels attempt / command / phase
  sim_state.npy              MuJoCo state per frame (sim.get_state().flatten(): time, qpos, qvel)
  final_*.png, final.npz     observation after the last control step (the state the outcome was judged in)
  annotation.json            episode annotation (build_annotation)
and <raw_root>/<batch>/<run>/replay.json (fidelity) when the whole run is done.
"""
import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import pkgutil
import re
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

LFF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LFF_ROOT)

from lffbench import envs  # noqa: E402
from lffbench import tasks as task_pkg  # noqa: E402
from lffbench.agent import camera as C  # noqa: E402
from lffbench.agent.tools import WS_HI, WS_LO, ToolExecutor, _json_default  # noqa: E402
from lffbench.task_base import TASKS  # noqa: E402

for _m in pkgutil.iter_modules(task_pkg.__path__):
    importlib.import_module(f"lffbench.tasks.{_m.name}")

# per-frame phase label: move / gripper = inside an agent's move_eef command; judge = control steps run by the
# task's outcome() check after `finish` (the arm holds still; not an agent action)
PHASES = {"move": 0, "gripper": 1, "judge": 2}
FPS = 20
SCHEMA = "lff-replay-1"

# error events in the outcome dict (scripts/outcomes.py): an attempt-1 success with one of these is "IR"
ERROR_FLAGS = ("trap_fired", "recovered_in_attempt")
ERROR_COUNTS = ("n_bad_pushes", "n_turned_pushes", "n_wrong_way", "n_regressions", "n_load_falls", "n_trap_events",
                "n_missed_strikes", "n_open_gripper_pushes")


def load_jsonl(p):
    out = []
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def norm_json(o):
    return json.loads(json.dumps(o, default=_json_default))


def sha256(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_code(run_dir):
    """[(file, logged hash, current hash)] for the server code files whose hash differs from the run's contract."""
    p = os.path.join(run_dir, "contract.json")
    if not os.path.exists(p):
        return [("contract.json", None, None)]
    files = json.load(open(p))["contract"].get("server_code_files", {})
    bad = []
    for f, h in files.items():
        q = os.path.join(LFF_ROOT, f)
        cur = sha256(q) if os.path.exists(q) else None
        if cur != h:
            bad.append((f, h, cur))
    return bad


def diff_json(a, b, path="", tol=0.0, out=None, limit=20):
    """Differences between two JSON values (numbers compared with an absolute tolerance)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                diff_json(a[k], b[k], f"{path}.{k}", tol, out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff_json(x, y, f"{path}[{i}]", tol, out, limit)
    elif isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if a != b:
            out.append((path, a, b))
    elif not (abs(float(a) - float(b)) <= tol or (math.isnan(float(a)) and math.isnan(float(b)))):
        out.append((path, a, b))
    return out


def error_event(o):
    """True / False if the outcome dict says whether a mistake happened in the attempt, None if the task logs none."""
    if not isinstance(o, dict):
        return None
    keys = [k for k in ERROR_FLAGS + ERROR_COUNTS if k in o]
    if not keys:
        return None
    return any(bool(o[k]) for k in ERROR_FLAGS if k in o) or any((o.get(k) or 0) > 0 for k in ERROR_COUNTS if k in o)


# ====================================================================================================== logs -> notes

class RunLog:
    """Everything about a run that comes from its logs alone (no simulator): commands with timing, the agent's
    narration around every robot command (trajectory.jsonl), attempt outcomes, harness / model, batch condition."""

    def __init__(self, run_dir, manifest=None):
        self.run_dir = os.path.abspath(run_dir.rstrip("/"))
        self.run = os.path.basename(self.run_dir)
        self.batch = os.path.basename(os.path.dirname(self.run_dir))
        self.ev = load_jsonl(os.path.join(self.run_dir, "events.jsonl"))
        self.start = next(e for e in self.ev if e["ev"] == "session_start")
        self.calls, self.dones, self.harness_status = {}, [], []
        last = None
        for e in self.ev:
            if e["ev"] == "call":
                self.calls[(int(e["attempt"]), int(e["step"]))] = e
                last = e
            elif e["ev"] == "done":
                self.dones.append(e)
                last = e
            elif e["ev"] == "http" and last is not None and "think_s" not in last:
                ok = (last["ev"] == "call" and e.get("cmd") == last.get("cmd")) or \
                     (last["ev"] == "done" and e.get("cmd") in ("finish", "done", "move_eef", "observe"))
                if ok:
                    last["think_s"], last["server_s"] = e.get("think_s"), e.get("server_s")
        self.manifest = manifest or {}
        self.entry = next((r for r in self.manifest.get("runs", []) if r.get("run") == self.run), {}) or {}
        p = os.path.join(self.run_dir, "harness.json")
        self.harness = json.load(open(p)) if os.path.exists(p) else {}
        self.traj = load_jsonl(os.path.join(self.run_dir, "trajectory.jsonl"))
        self._narration()

    # ---------------------------------------------------------------- agent narration
    def _narration(self):
        """notes[(attempt, step)] = what the agent wrote / did since the previous robot command, up to this one;
        finish_notes[attempt] likewise for the agent's finish call; final_report = its last message."""
        self.notes, self.finish_notes, self.final_report, self.headers = {}, {}, "", []
        buf = dict(messages=[], reasoning=[], shell=[], viewed=[])
        attempt = 1

        def take():
            nonlocal buf
            out = {k: v for k, v in buf.items() if v}
            buf = dict(messages=[], reasoning=[], shell=[], viewed=[])
            return out

        last_msg = ""
        for t in self.traj:
            if t.get("kind") == "header":
                self.headers.append(t.get("agent") or {})
                continue
            if t.get("kind") != "step":
                continue
            if (t.get("message") or "").strip():
                buf["messages"].append(t["message"].strip())
                last_msg = t["message"].strip()
            if (t.get("reasoning_content") or "").strip():
                buf["reasoning"].append(t["reasoning_content"].strip())
            res = {r.get("source_call_id"): r for r in (t.get("observation") or {}).get("results", [])}
            for c in t.get("tool_calls") or []:
                fn, args = c.get("function_name"), c.get("arguments") or {}
                if fn == "shell":
                    buf["shell"].append(str(args.get("command", ""))[:2000])
                elif fn == "view_images":
                    buf["viewed"].extend(args.get("paths") or [])
                elif fn == "robot":
                    cmd = str(args.get("command", "")).strip()
                    r = res.get(c.get("tool_call_id")) or {}
                    try:
                        rep = json.loads(r.get("content") or "{}")
                    except ValueError:
                        rep = {}
                    word = cmd.split()[0] if cmd else ""
                    m = re.search(r"a(\d+)_s(\d+)_state\.json", str(rep.get("state_file", "")))
                    if word in ("move_eef", "observe") and m:
                        a, s = int(m.group(1)), int(m.group(2))
                        n = take()
                        n["command"] = cmd
                        self.notes[(a, s)] = n
                        attempt = a
                    elif word in ("finish", "done"):
                        n = take()
                        n["command"] = cmd
                        n["reply"] = {k: rep.get(k) for k in ("feedback", "message") if rep.get(k)}
                        self.finish_notes.setdefault(attempt, n)
                        if rep.get("ended") and not rep.get("message", "").startswith("The episode is over"):
                            attempt += 1
                    # status / help / errors: their narration stays in the buffer for the next command
        self.final_report = last_msg

    # ---------------------------------------------------------------- run-level labels
    def attempts(self):
        out = []
        for e in self.dones:
            o = e.get("outcome") or {}
            if isinstance(o, str):
                try:
                    o = ast.literal_eval(o)
                except (ValueError, SyntaxError):
                    o = {}
            out.append(dict(attempt=int(e["attempt"]), success=e.get("success") in (True, "True"), outcome=o,
                            reason=e.get("reason") or ""))
        return sorted(out, key=lambda x: x["attempt"])

    def condition(self):
        s, en = self.start, self.entry
        mem = s.get("memory", en.get("mem"))
        chain = en.get("chain")
        if mem == "full":
            cond = "with_history"
        elif chain:
            cond = "memory_free_retry"
        else:
            cond = "independent_single_attempt"
        return cond, chain

    def outcome_class(self):
        """CS / IR / S? / XR / F for history runs (scripts/outcomes.py), else None."""
        att = self.attempts()
        if not att or self.start.get("memory") != "full":
            return None, "".join("S" if a["success"] else "F" for a in att)
        hist = "".join("S" if a["success"] else "F" for a in att)
        if att[0]["success"]:
            ev = error_event(att[0]["outcome"])
            cls = "S?" if ev is None else ("IR" if ev else "CS")
        else:
            cls = "XR" if any(a["success"] for a in att) else "F"
        return cls, hist

    def agent_info(self):
        h, en = self.harness, self.entry
        hdr = self.headers[0] if self.headers else {}
        return dict(harness=h.get("harness") or en.get("agent_harness") or hdr.get("name"),
                    harness_version=h.get("version") or hdr.get("version"),
                    harness_binary_sha256=(h.get("binary") or {}).get("sha256"),
                    model=h.get("model") or en.get("model") or hdr.get("model_name"),
                    effort=h.get("effort") or en.get("effort") or hdr.get("effort"),
                    provider=h.get("provider") or en.get("provider"),
                    codex_catalog=en.get("codex_catalog"), auth=en.get("codex_auth") or en.get("claude_auth"),
                    jail=en.get("jail"))


# ====================================================================================================== recording

class EpisodeSink:
    """Streams one episode to disk: two lossless RGB videos while recording, numeric arrays when closed."""

    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.w = None
        self.num = defaultdict(list)
        self.n = 0

    def add(self, st, imgs):
        import imageio
        if self.w is None:
            self.w = [imageio.get_writer(os.path.join(self.path, f"{k}.mp4"), fps=FPS, codec="libx264rgb",
                                         pixelformat="rgb24", macro_block_size=1,
                                         ffmpeg_params=["-crf", "0", "-preset", "veryfast", "-threads", "1"], ffmpeg_log_level="error")
                      for k in ("agentview", "wrist")]
        for w, im in zip(self.w, imgs):
            w.append_data(im)
        for k, v in st.items():
            self.num[k].append(v)
        self.n += 1

    def close(self, final_state, final_imgs):
        from PIL import Image
        if self.w is not None:
            for w in self.w:
                w.close()
        arr = {k: np.asarray(v) for k, v in self.num.items() if k != "sim_state"}
        np.savez(os.path.join(self.path, "frames.npz"), **arr)
        if self.num.get("sim_state"):
            np.save(os.path.join(self.path, "sim_state.npy"), np.asarray(self.num["sim_state"], np.float64))
        np.savez(os.path.join(self.path, "final.npz"), **{k: np.asarray(v) for k, v in final_state.items()})
        for k, im in zip(("agentview", "wrist"), final_imgs):
            Image.fromarray(im).save(os.path.join(self.path, f"final_{k}.png"))


class Recorder:
    """Captures (o_t, a_t) before every env.step while active and streams it into the current EpisodeSink."""

    def __init__(self, rp, res):
        self.rp, self.res = rp, int(res)
        self.active = False
        self.sink = None
        self.cur = dict(attempt=1, command=-1, phase=PHASES["move"])
        env = rp.task.env
        self.env, self.rob = env, env.robots[0]
        self.cams = (C.mj_cam("agentview"), C.mj_cam("wrist"))
        orig = env.step

        def step(a):
            if self.active and self.sink is not None:
                self.capture(np.asarray(a, np.float64))
            out = orig(a)
            if self.active and self.sink is not None:
                self.rp.poll_events(self.sink.n - 1)
            return out

        env.step = step

    @property
    def n(self):
        return self.sink.n if self.sink is not None else 0

    def state(self):
        import robosuite.utils.transform_utils as T
        env, rob = self.env, self.rob
        d = env.sim.data
        eef = np.array(d.site_xpos[rob.eef_site_id], np.float64)
        q_xyzw = T.convert_quat(np.array(d.get_body_xquat(rob.robot_model.eef_name), np.float64), to="xyzw")
        gq = np.array([d.qpos[x] for x in rob._ref_gripper_joint_pos_indexes], np.float64)
        try:
            flat = np.asarray(env.sim.get_state().flatten(), np.float64)
        except Exception:
            flat = np.concatenate([[d.time], d.qpos, d.qvel]).astype(np.float64)
        sk = self.rp.task.sk
        return dict(eef_pos=eef, eef_quat_xyzw=q_xyzw, eef_axis_angle=np.asarray(T.quat2axisangle(q_xyzw), np.float64),
                    gripper_qpos=gq, joint_pos=np.array(d.qpos[rob._ref_joint_pos_indexes], np.float64),
                    joint_vel=np.array(d.qvel[rob._ref_joint_vel_indexes], np.float64),
                    gripper_width=np.float64(abs(gq[0] - gq[1])), eef_site_mat=np.array(d.site_xmat[rob.eef_site_id], np.float64),
                    eef_yaw_cmd=np.float64(getattr(sk, "yaw", 0.0)), gripper_cmd=np.float64(getattr(sk, "grip", -1.0)),
                    sim_time=np.float64(d.time), sim_state=flat)

    def images(self):
        return [envs.render(self.env, c, self.res) for c in self.cams]

    def capture(self, a):
        st = self.state()
        st.update(action=a, attempt=np.int64(self.cur["attempt"]), command=np.int64(self.cur["command"]),
                  phase=np.int64(self.cur["phase"]))
        self.sink.add(st, self.images())


class RunReplay:
    def __init__(self, run_dir, record=False, res=256, out_root=None, manifest=None):
        self.run_dir = os.path.abspath(run_dir.rstrip("/"))
        self.run_name = os.path.basename(self.run_dir)
        self.log = RunLog(self.run_dir, manifest)
        self.ev = self.log.ev
        self.start = self.log.start
        cp = os.path.join(self.run_dir, "contract.json")
        self.contract = json.load(open(cp))["contract"] if os.path.exists(cp) else {}
        spec = self.contract.get("image_spec") or {}
        self.image_res = int(spec.get("width", 512))
        self.code_diff = check_code(self.run_dir)
        s = self.start
        self.task = TASKS[s["task"]](cam_res=self.image_res)
        inst = self.task.sample_instance(s["seed"])
        logged = s.get("inst") or {}
        sampled = norm_json({k: v for k, v in inst.items() if not k.startswith("_")})
        self.inst_diff = diff_json(sampled, logged, "inst")
        if s.get("arm", "none") not in (None, "none") or logged.get("nominal"):
            inst.update(logged)  # controlled arms / nominal: the server overrode these instance fields
            self.inst_diff = []
        self.inst = inst
        self.protocol = getattr(self.task, "protocol", "cross")
        self.k = int(s.get("k") or self.task.max_attempts)
        self.attempt = 1
        self.mismatch = []  # (where, what, logged, replayed)
        self.n_checked = 0
        self.max_dev_m = 0.0
        self.unfinished = None
        self.rec = Recorder(self, res) if record else None
        self.res = int(res)
        self.out_root = out_root
        self.n_mon = 0
        self.events, self.spans, self.ep_attempts, self.ep_index = [], [], [], 0
        self.episodes = []  # summaries of the written episodes
        self.task.reset_instance(self.inst)
        self.ex = ToolExecutor(self.task, self.inst, image_res=self.image_res, cams=("agentview", "wrist"))
        sim = self.task.env.sim
        bid = sim.model.body_name2id("robot0_base")
        self.base_off = np.array(sim.data.body_xpos[bid], float).copy()
        if self.rec:
            self._open_episode()

    # ------------------------------------------------------------------ episodes
    def _ep_dir(self):
        return os.path.join(self.out_root, self.log.batch, self.run_name, f"ep{self.ep_index:02d}")

    def _open_episode(self):
        self.rec.sink = EpisodeSink(self._ep_dir())
        self.rec.cur = dict(attempt=self.attempt, command=-1, phase=PHASES["move"])
        self.rec.active = True
        self.events, self.spans, self.ep_attempts = [], [], []
        self.ep_initial = self.rec.state()  # state right after the reset (before frame 0)

    def close_episode(self):
        if not self.rec:
            self.ep_attempts = []
            return
        sink = self.rec.sink
        st = self.rec.state()
        imgs = self.rec.images()
        st.pop("sim_state", None)
        sink.close(dict(st, sim_state=self.rec.state()["sim_state"]), imgs)
        ann = build_annotation(self, sink)
        with open(os.path.join(sink.path, "annotation.json"), "w") as f:
            json.dump(ann, f, indent=1, default=_json_default)
        self.episodes.append(dict(dir=os.path.relpath(sink.path, self.out_root), n_frames=sink.n,
                                  attempts=[a["attempt"] for a in self.ep_attempts]))
        self.rec.active = False
        self.rec.sink = None
        self.ep_index += 1

    # ------------------------------------------------------------------ monitor events (task-side error events)
    def poll_events(self, frame):
        mon = getattr(self.task, "mon", None)
        evs = mon.get("events") if isinstance(mon, dict) else None
        if not isinstance(evs, list):
            return
        if len(evs) < self.n_mon:  # monitor was rebuilt
            self.n_mon = 0
        for e in evs[self.n_mon:]:
            self.events.append(dict(frame=int(frame), attempt=self.attempt,
                                    event=norm_json(e) if isinstance(e, dict) else str(e)))
        self.n_mon = len(evs)

    # ------------------------------------------------------------------ state as the server reports it
    def proprio_la(self):
        p = self.ex.proprio()
        pos = [round(float(v), 4) for v in np.asarray(p["eef_pos"], float) - self.base_off]
        return dict(eef_pos=pos, eef_yaw_deg=p["eef_yaw_deg"], gripper_width_m=p["gripper_width_m"])

    def compare(self, where, rep_logged, rep):
        self.n_checked += 1
        for k in ("eef_pos", "eef_yaw_deg", "gripper_width_m"):
            if k not in rep_logged or k not in rep:
                continue
            a, b = rep_logged[k], rep[k]
            if k == "eef_pos":
                dev = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
                self.max_dev_m = max(self.max_dev_m, dev)
                if dev > 1.5e-4:
                    self.mismatch.append((where, k, a, b))
            elif abs(float(a) - float(b)) > (0.11 if k == "eef_yaw_deg" else 1.5e-4):
                self.mismatch.append((where, k, a, b))
        if bool(rep_logged.get("ok", True)) != bool(rep.get("ok", True)):
            self.mismatch.append((where, "ok", rep_logged.get("ok"), rep.get("ok")))

    def _span(self, e, f0, kind):
        if self.rec:
            self.spans.append(dict(attempt=self.attempt, step=int(e.get("step", -1)), kind=kind, frame_start=f0,
                                   frame_end=self.rec.n))

    # ------------------------------------------------------------------ commands (robot_server._handle_v2)
    def move_eef(self, e):
        args = e.get("args") or {}
        where = f"a{e['attempt']}s{e['step']}"
        logged = e.get("report") or {}
        f0 = self.rec.n if self.rec else 0
        p = self.ex.proprio()
        if set(args) - {"x", "y", "z", "yaw", "gripper", "speed"}:
            self._span(e, f0, "error")
            return self.compare(where, logged, dict(ok=False))
        off = self.base_off
        try:
            cur = np.asarray(p["eef_pos"], float) - off
            tgt = [float(args.get(k, cur[i])) for i, k in enumerate("xyz")]
            yaw = float(args.get("yaw", p["eef_yaw_deg"]))
            speed = None if args.get("speed") in (None, "") else float(args["speed"])
        except (TypeError, ValueError):
            self._span(e, f0, "error")
            return self.compare(where, logged, dict(ok=False))
        grip = args.get("gripper")
        if grip not in (None, "open", "close"):
            self._span(e, f0, "error")
            return self.compare(where, logged, dict(ok=False))
        moving = any(k in args for k in ("x", "y", "z", "yaw"))
        tgt_w = np.asarray(tgt) + off
        if moving and not (np.all(tgt_w >= WS_LO) and np.all(tgt_w <= WS_HI)):
            self._span(e, f0, "rejected")
            return self.compare(where, logged, dict(ok=False, **self.proprio_la()))
        raw = {}
        if self.rec:
            self.rec.cur = dict(attempt=self.attempt, command=int(e["step"]), phase=PHASES["move"])
        if moving:
            mv = dict(x=float(tgt_w[0]), y=float(tgt_w[1]), z=float(tgt_w[2]), yaw=yaw)
            if speed is not None:
                mv["speed"] = speed
            raw.update(self.ex.call("move_to", mv).report)
        if grip is not None:
            if self.rec:
                self.rec.cur = dict(attempt=self.attempt, command=int(e["step"]), phase=PHASES["gripper"])
            raw.update(self.ex.call("open_gripper" if grip == "open" else "close_gripper", {}).report)
        rep = dict(ok=bool(raw.get("ok", True)), **self.proprio_la())
        self.compare(where, logged, rep)
        lr = e.get("raw") or {}
        for k in ("holding_object", "touched_object"):
            if k in lr and k in raw and bool(lr[k]) != bool(raw[k]):
                self.mismatch.append((where, "raw." + k, lr[k], raw[k]))
        self._span(e, f0, "move_eef")

    def observe(self, e):
        f0 = self.rec.n if self.rec else 0
        self._span(e, f0, "observe")

    def done(self, e):
        f0 = self.rec.n if self.rec else 0
        if self.rec:
            self.rec.cur = dict(attempt=self.attempt, command=-1, phase=PHASES["judge"])
        out = self.task.outcome(self.inst)
        succ = bool(out["success"])
        where = f"a{e['attempt']}done"
        if succ != bool(e.get("success") in (True, "True")):
            self.mismatch.append((where, "success", e.get("success"), succ))
        lo = e.get("outcome") or {}
        ro = norm_json({k: v for k, v in out.items() if k != "feedback"})
        if isinstance(lo, dict):
            for d in diff_json(lo, ro, "outcome", tol=1e-6, limit=8):
                self.mismatch.append((where,) + d)
        info = dict(attempt=self.attempt, success=succ, outcome=ro, finish_reason=e.get("reason", ""),
                    feedback_shown=e.get("feedback", ""), auto_ended="(automatic)" in str(e.get("reason", "")),
                    think_s=e.get("think_s"))
        try:
            info["feedback_f2"] = self.task.feedback(self.inst, out, "F2")
        except Exception as ex_:
            info["feedback_f2"] = f"<unavailable: {type(ex_).__name__}>"
        if self.rec:
            info["judge_frames"] = [f0, self.rec.n]
        self.ep_attempts.append(info)
        last = self.attempt >= self.k or succ or bool(out.get("terminal"))
        if self.protocol == "cross" or last:
            self.close_episode()
        if last:
            return True
        self.attempt += 1
        if self.protocol == "cross":
            self.task.reset_instance(self.inst)
            self.n_mon = 0
            if self.rec:
                self._open_episode()
        elif self.rec:
            self.rec.cur = dict(attempt=self.attempt, command=-1, phase=PHASES["move"])
        self.ex = ToolExecutor(self.task, self.inst, image_res=self.image_res, cams=("agentview", "wrist"))
        return False

    def run(self):
        t0 = time.time()
        finished = False
        for e in self.ev:
            if e["ev"] == "call":
                if e.get("cmd") == "move_eef":
                    self.move_eef(e)
                elif e.get("cmd") == "observe":
                    self.observe(e)
                else:
                    self.mismatch.append((f"a{e.get('attempt')}s{e.get('step')}", "unsupported command", e.get("cmd"), None))
            elif e["ev"] == "done":
                finished = self.done(e)
                if finished:
                    break
        self.unfinished = not finished
        if not finished and self.rec and self.rec.sink is not None:
            # the run ended without a final finish (agent stopped, harness died): keep it, outcome not judged
            if self.rec.n or self.ep_attempts:
                self.ep_attempts.append(dict(attempt=self.attempt, success=None, outcome=None, unjudged=True,
                                             finish_reason="", feedback_shown="", auto_ended=False))
                self.close_episode()
            else:
                shutil.rmtree(self.rec.sink.path, ignore_errors=True)
                self.rec.sink = None
        self.wall_s = time.time() - t0
        return self

    def summary(self):
        return dict(run=self.run_name, batch=self.log.batch, task=self.start["task"], seed=self.start["seed"],
                    protocol=self.protocol, code_diff=[f for f, _, _ in self.code_diff], inst_diff=len(self.inst_diff),
                    n_checked=self.n_checked, n_mismatch=len(self.mismatch), max_eef_dev_m=round(self.max_dev_m, 6),
                    unfinished=self.unfinished, wall_s=round(getattr(self, "wall_s", 0.0), 1),
                    first_mismatches=[list(map(str, m)) for m in self.mismatch[:5]])


# ====================================================================================================== annotation

def task_info(task, inst):
    cls = type(task)
    return dict(name=cls.name, level=cls.level, family="long_horizon" if cls.name.startswith("lh_") else "hidden_variable",
                category=cls.category, capabilities=list(cls.capabilities), protocol=getattr(cls, "protocol", "cross"),
                max_attempts=cls.max_attempts, agent_max_steps=int(getattr(cls, "agent_max_steps", 40)),
                instruction=cls.instruction)


def build_annotation(rp, sink):
    """Episode annotation: provenance, agent, task + hidden variables, per-attempt outcome (incl. the privileged F2
    failure description), run-level outcome class, every robot command with its frame span and the agent's narration,
    task-side error events with frame indices."""
    log = rp.log
    s = rp.start
    cond, chain = log.condition()
    cls, hist = log.outcome_class()
    att_all = log.attempts()
    attempts = []
    for a in rp.ep_attempts:
        a = dict(a)
        n = a["attempt"]
        later = [x for x in att_all if x["attempt"] > n]
        a["corrected_later_in_run"] = (None if a.get("success") is not False else any(x["success"] for x in later))
        a["is_last_attempt_of_run"] = not later
        fn = log.finish_notes.get(n)
        if fn:
            a["agent_finish"] = fn
        a["error_event_in_attempt"] = error_event(a.get("outcome"))
        attempts.append(a)
    # post-failure reflection: what the agent wrote right after the failure feedback (before its next robot command)
    for a in attempts:
        nxt = sorted((k for k in log.notes if k[0] == a["attempt"] + 1), key=lambda k: k[1])
        if a.get("success") is False and nxt:
            note = log.notes[nxt[0]]
            a["agent_reflection_after_feedback"] = {k: note[k] for k in ("messages", "reasoning") if k in note}
    commands = []
    for sp in rp.spans:
        e = log.calls.get((sp["attempt"], sp["step"])) or {}
        rep = e.get("report") or {}
        c = dict(attempt=sp["attempt"], step=sp["step"], kind=sp["kind"], frame_start=sp["frame_start"],
                 frame_end=sp["frame_end"], command=" ".join([e.get("cmd", "")] + [f"{k}={v}" for k, v in (e.get("args") or {}).items()]),
                 args=e.get("args") or {}, ok=rep.get("ok"), error=rep.get("error"),
                 stopped_from_target=rep.get("stopped_from_target"), eef_pos_after=rep.get("eef_pos"),
                 gripper_width_after=rep.get("gripper_width_m"), holding_after=(e.get("raw") or {}).get("holding_object"),
                 touched_during=(e.get("raw") or {}).get("touched_object"), think_s=e.get("think_s"),
                 agent_obs_images=rep.get("images"))
        note = log.notes.get((sp["attempt"], sp["step"]))
        if note:
            c["agent_note"] = {k: v for k, v in note.items() if k != "command"}
        commands.append(c)
    t_all = [x for x in (log.calls.get(k, {}).get("think_s") for k in log.calls) if x]
    return dict(
        schema=SCHEMA,
        source=dict(batch=log.batch, run=log.run, attempts_in_episode=[a["attempt"] for a in attempts],
                    episode_in_run=rp.ep_index, condition=cond, memory_free_chain_of=chain,
                    memory_free_rep=(log.entry.get("rep") if cond == "memory_free_retry" else None),
                    feedback_level=s.get("feedback"), hide_k=s.get("k_shown", s.get("k")) != s.get("k"),
                    k=s.get("k"), perception=s.get("perception"), instruction_variant=s.get("instruction"),
                    contract_sha256=(json.load(open(os.path.join(rp.run_dir, "contract.json"))).get("contract_sha256")
                                     if os.path.exists(os.path.join(rp.run_dir, "contract.json")) else None)),
        agent=log.agent_info(),
        task=dict(task_info(rp.task, rp.inst), seed=s.get("seed"),
                  instance={k: v for k, v in norm_json(rp.inst).items() if not str(k).startswith("_")}),
        run_outcome=dict(outcome_class=cls, attempt_history=hist, n_attempts=len(att_all),
                         run_success=any(x["success"] for x in att_all) if att_all else None,
                         agent_final_report=log.final_report[:6000] or None,
                         mean_think_s=(round(float(np.mean(t_all)), 2) if t_all else None)),
        attempts=attempts,
        commands=commands,
        task_events=rp.events,
        frames=dict(n=sink.n, fps=FPS, image_res=rp.res,
                    phase_codes=PHASES, initial_state=dict(eef_pos=rp.ep_initial["eef_pos"],
                                                           joint_pos=rp.ep_initial["joint_pos"],
                                                           gripper_qpos=rp.ep_initial["gripper_qpos"])),
    )


# ====================================================================================================== CLI

def load_manifest(run_dir):
    p = os.path.join(os.path.dirname(os.path.abspath(run_dir.rstrip("/"))), "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "record"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="record again even if replay.json exists")
    a = ap.parse_args()
    rows = []
    for r in a.runs:
        r = r.rstrip("/")
        try:
            if a.mode == "check":
                s = RunReplay(r, manifest=load_manifest(r)).run().summary()
            else:
                batch, name = os.path.basename(os.path.dirname(os.path.abspath(r))), os.path.basename(r)
                rd = os.path.join(a.out, batch, name)
                if os.path.exists(os.path.join(rd, "replay.json")) and not a.force:
                    print(json.dumps(dict(run=name, skipped="exists")), flush=True)
                    continue
                shutil.rmtree(rd, ignore_errors=True)
                rp = RunReplay(r, record=True, res=a.res, out_root=a.out, manifest=load_manifest(r)).run()
                s = rp.summary()
                s["episodes"] = rp.episodes
                with open(os.path.join(rd, "replay.json"), "w") as f:
                    json.dump(s, f, indent=1)
        except Exception:
            import traceback
            s = dict(run=os.path.basename(r), error=traceback.format_exc()[-2000:])
        rows.append(s)
        print(json.dumps(s), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
