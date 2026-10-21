"""Pack the recorded agent replays (scripts/lerobot_replay.py record) into one LeRobot v3.0 dataset plus LFF sidecars.

Runs in a Python with lerobot 0.4.x (it does not import lffbench / MuJoCo). Three steps, so that the video encoding
can run in parallel shards (contiguous episode ranges, merged in order, so episode indices follow the plan):

  python scripts/lerobot_export.py plan  --raw <raw_root>[,<raw_root2>] --runs-root <lffbench>/runs/api \
      --paper <paper/latex> --task-set <lffbench>/sets/main_v2_1inst.txt --work <work_dir> [--batches ...] [--limit N]
  python scripts/lerobot_export.py shard --work <work_dir> --shard i --nshards N
  python scripts/lerobot_export.py merge --work <work_dir> --nshards N --out <dataset_root>

Only runs whose replay matched the log exactly are exported (replay.json: no code / instance difference, 0
mismatching reports or outcomes). Episodes without a judged attempt (the agent was stopped mid-attempt) are dropped.

LeRobot features (fps 20, one frame per robosuite control step; frame t = observation before action t, action t):
  observation.images.image        agentview RGB 256x256 (upright; LIBERO naming)
  observation.images.wrist_image  wrist (eye-in-hand) RGB 256x256
  observation.state               8: eef xyz (world, m), eef axis-angle (3), gripper finger qpos (2)  (LIBERO layout)
  observation.joint_pos / joint_vel   7 arm joints (rad, rad/s)
  observation.eef_quat            4: xyzw;  observation.gripper_width 1 (m)
  action                          7: robosuite OSC_POSE input in [-1, 1]: dpos (x 0.1 m), drot axis-angle (x 0.5 rad),
                                  gripper (-1 open, +1 close)
  annotation.attempt_index        attempt (1-based) the frame belongs to
  annotation.command_index        robot-command step within the attempt (-1: outcome check after `finish`)
  annotation.phase                0 arm motion, 1 gripper open/close, 2 outcome check (not an agent action)
  annotation.attempt_success      1 / 0 / -1 (succeeded / failed / not judged) for the frame's attempt
  annotation.task_event           number of task-monitor events (error / progress events) registered at this step
Sidecars under lff/: annotations/episode_*.json (full annotation), episodes.jsonl (flat per-episode table),
runs.jsonl (attempt chains), tasks.json (task metadata + paper description), sim_state/episode_*.npy (MuJoCo state
per frame), final/episode_* (observation after the last step = the state the outcome was judged in), code/.
"""
import argparse
import glob
import json
import os
import re
import shutil
import time

import numpy as np

FPS = 20
RES = 256
BATCH_ORDER = ["cx_main1", "cc_main1", "cc_opus1", "cx_native1", "cx_pilot6c", "cx_pilot6b", "cx_pilot6", "cx_smoke",
               "cc_smoke", "cc_opus_smoke"]
BATCH_NOTES = {
    "cx_main1": "GPT main table: gpt-6-astra (high) in Codex 0.163, shared (generic) model catalog",
    "cc_main1": "Claude main table: claude-sonnet-5-5 (high) in Claude Code 2.1.267",
    "cc_opus1": "Claude main table: claude-opus-5-5 (high) in Claude Code 2.1.299",
    "cx_native1": "control: gpt-6-astra in Codex with its native model catalog (code mode), 8 instances",
    "cx_pilot6c": "pilot: gpt-6-astra in Codex, 3 history runs + memory-free retries (early-exit resume fixed)",
    "cx_pilot6b": "pilot: gpt-6-astra in Codex; Codex sometimes exited early (before the resume fix), so the "
                  "with-history continuation of failed runs is incomplete",
    "cx_pilot6": "pilot: gpt-6-astra in Codex, stopped early",
    "cx_smoke": "smoke test: single attempt", "cc_smoke": "smoke test: single attempt",
    "cc_opus_smoke": "smoke test: single attempt",
}
PHASES = {"move": 0, "gripper": 1, "judge": 2}
CAUSES = {"L1": "execution offset (the arm does not go exactly where commanded)",
          "L2": "disturbed state to restore (incl. damage the agent causes)",
          "L3": "infeasible manipulation strategy (grasp / contact / tool must change)",
          "L4": "misjudged geometry (size, shape, fit, hidden structure)",
          "L5": "misjudged physics (mass distribution, friction, dynamics)"}


def features():
    vid = lambda: {"dtype": "video", "shape": (RES, RES, 3), "names": ["height", "width", "channels"]}
    f32 = lambda n, names: {"dtype": "float32", "shape": (n,), "names": names}
    i64 = lambda: {"dtype": "int64", "shape": (1,), "names": None}
    return {
        "observation.images.image": vid(),
        "observation.images.wrist_image": vid(),
        "observation.state": f32(8, ["eef_x", "eef_y", "eef_z", "eef_rot_aa_x", "eef_rot_aa_y", "eef_rot_aa_z",
                                     "finger_qpos_left", "finger_qpos_right"]),
        "observation.joint_pos": f32(7, [f"joint{i}" for i in range(1, 8)]),
        "observation.joint_vel": f32(7, [f"joint{i}" for i in range(1, 8)]),
        "observation.eef_quat": f32(4, ["qx", "qy", "qz", "qw"]),
        "observation.gripper_width": f32(1, ["width_m"]),
        "action": f32(7, ["eef_delta_x", "eef_delta_y", "eef_delta_z", "eef_delta_rot_aa_x", "eef_delta_rot_aa_y",
                          "eef_delta_rot_aa_z", "gripper_close"]),
        "annotation.attempt_index": i64(),
        "annotation.command_index": i64(),
        "annotation.phase": i64(),
        "annotation.attempt_success": i64(),
        "annotation.task_event": i64(),
    }


# ------------------------------------------------------------------------------------------------ paper metadata

def _clean_tex(s):
    s = re.sub(r"%.*", "", s)
    for a, b in [(r"\{\\times\}", "×"), (r"\\times", "×"), (r"\\,", " "), (r"\\mu", "μ"), (r"\^\\circ", "°"),
                 (r"\\pm", "±"), (r"\\leftrightarrow", "↔"), (r"\\geq", "≥"), (r"\\leq", "≤"), (r"\\%", "%"),
                 (r"\\_", "_"), (r"\\eg", "e.g."), (r"\\ie", "i.e."), (r"\\ding\{51\}", "yes"), (r"\\ding\{55\}", "no"),
                 (r"\\approx", "≈"), (r"\\to", "→"), (r"\\Delta", "Δ"), (r"\\theta", "θ"), (r"\\alpha", "α"),
                 (r"\\cdot", "·"), (r"\\ldots", "…"), (r"\\dots", "…"), (r"\\&", "&"), (r"\\textasciitilde", "~")]:
        s = re.sub(a, b, s)
    s = re.sub(r"\\(?:cref|ref|label|cite)\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:TODO|todo)\{[^}]*\}", "(to be written)", s)
    for _ in range(3):
        s = re.sub(r"\\(?:emph|texttt|textbf|textit|mathrm|text|mathit|mathbf)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.replace("``", '"').replace("''", '"').replace("---", "—").replace("--", "–").replace("~", " ")
    s = re.sub(r"\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n", "\n\n", s)).strip()


def paper_tasks(paper_dir):
    """{task: {name, cause, hidden_variable, repair_within_attempt, oracle_pct, naive_pct, description}} from the
    paper's task table and task appendix (empty if the paper sources are not there)."""
    out = {}
    if not paper_dir:
        return out
    app = os.path.join(paper_dir, "sections", "appendix_tasks.tex")
    tab = os.path.join(paper_dir, "sections", "tab_tasks.tex")
    if os.path.exists(app):
        txt = open(app).read()
        parts = re.split(r"(\\paragraph\{|\\subsection\{|\\section\{)", txt)
        for i in range(1, len(parts) - 1, 2):
            if parts[i] != "\\paragraph{":
                continue
            body = parts[i + 1]
            m = re.match(r"(.*?)\s*\(\\texttt\{([a-z0-9\\_]+)\};\s*([^)]*)\)\.\}", body, re.S)
            if not m:
                continue
            name, code, cause = m.group(1).strip(), m.group(2).replace("\\_", "_"), m.group(3).strip()
            desc = _clean_tex(body[m.end():])
            out[code] = dict(name=name, cause=_clean_tex(cause), description=desc)
    if os.path.exists(tab):
        names = {v["name"].lower(): k for k, v in out.items()}
        for line in open(tab):
            cells = [c.strip() for c in line.split("&")]
            if len(cells) != 6 or not cells[-1].endswith("\\\\"):
                continue
            nm = re.sub(r"\\contscene", "", cells[0]).strip().lower()
            code = names.get(nm)
            if not code:
                continue
            out[code].update(table_cause=_clean_tex(cells[1]), hidden_variable=_clean_tex(cells[2]),
                             repair_within_attempt=_clean_tex(cells[3]) == "yes",
                             oracle_first_attempt_pct=_clean_tex(cells[4]),
                             naive_first_attempt_pct=_clean_tex(cells[5].rstrip("\\").strip()))
    return out


# ------------------------------------------------------------------------------------------------ collection

def manifest_status(runs_root, batch, run, cache={}):
    if batch not in cache:
        p = os.path.join(runs_root, batch, "manifest.json")
        cache[batch] = json.load(open(p)) if os.path.exists(p) else {}
    m = cache[batch]
    for r in m.get("runs", []):
        if r.get("run") == run:
            return r.get("status", "?"), r
    for r in m.get("superseded", []):
        if r.get("run") == run:
            return "superseded_" + str(r.get("status", "?")), r
    return "not_in_manifest", {}


def collect(raw_roots, runs_root, batches, task_order):
    """Verified episodes in dataset order. A run recorded under several raw roots is taken from the first root."""
    eps, rejected, seen = [], [], set()
    for root in raw_roots:
        for rj in sorted(glob.glob(os.path.join(root, "*", "*", "replay.json"))):
            rep = json.load(open(rj))
            batch, run = rep.get("batch") or rj.split(os.sep)[-3], rep["run"]
            if (batches and batch not in batches) or (batch, run) in seen:
                continue
            seen.add((batch, run))
            ok = not rep.get("error") and rep.get("n_mismatch") == 0 and not rep.get("code_diff") and not rep.get("inst_diff")
            status, entry = manifest_status(runs_root, batch, run)
            if not ok:
                never = rep.get("code_diff") == ["contract.json"] and not rep.get("n_checked")
                rejected.append(dict(run=run, batch=batch, run_status=status,
                                     reason="run never started (no contract, no robot command)" if never else
                                     "replay does not reproduce the log (the log has gaps or the replay diverged)",
                                     detail=rep.get("first_mismatches")))
                continue
            for ep in rep.get("episodes", []):
                d = os.path.join(root, ep["dir"])
                ann = json.load(open(os.path.join(d, "annotation.json")))
                judged = [a for a in ann["attempts"] if a.get("success") is not None]
                if not judged or ep["n_frames"] == 0:
                    rejected.append(dict(run=run, batch=batch, episode=ep["dir"],
                                         reason="no judged attempt" if not judged else "no frames"))
                    continue
                ann["source"]["run_status"] = status
                ann["source"]["manifest_id"] = entry.get("id")
                ann["source"]["replay"] = dict(n_reports_checked=rep["n_checked"], n_mismatch=rep["n_mismatch"],
                                               max_eef_dev_m=rep["max_eef_dev_m"])
                key = (BATCH_ORDER.index(batch) if batch in BATCH_ORDER else 99, task_order.get(ann["task"]["name"], 99),
                       {"with_history": 0, "memory_free_retry": 1}.get(ann["source"]["condition"], 2),
                       ann["source"].get("memory_free_rep") or 0, run, ann["source"]["episode_in_run"])
                eps.append(dict(key=key, dir=d, n=int(ep["n_frames"]), ann=ann))
    eps.sort(key=lambda x: x["key"])
    return eps, rejected


# ------------------------------------------------------------------------------------------------ writing

def frames_of(path):
    import av
    with av.open(path) as c:
        for fr in c.decode(video=0):
            yield fr.to_ndarray(format="rgb24")


def flat_row(i, ann, n, idx0):
    s, ag, t, ro = ann["source"], ann["agent"], ann["task"], ann["run_outcome"]
    att = ann["attempts"]
    a0 = att[0]
    return dict(episode_index=i, dataset_from_index=idx0, dataset_to_index=idx0 + n, length=n,
                batch=s["batch"], run=s["run"], run_status=s.get("run_status"), condition=s["condition"],
                memory_free_rep=s.get("memory_free_rep"), memory_free_chain_of=s.get("memory_free_chain_of"),
                episode_in_run=s["episode_in_run"], attempts=[a["attempt"] for a in att],
                harness=ag.get("harness"), harness_version=ag.get("harness_version"), model=ag.get("model"),
                effort=ag.get("effort"), codex_catalog=ag.get("codex_catalog"),
                task=t["name"], level_code=t["level"], paper_cause=(t.get("paper") or {}).get("cause"),
                category=t["category"], protocol=t["protocol"], seed=t["seed"], instruction=t["instruction"],
                success=[a.get("success") for a in att], failure=[(a.get("outcome") or {}).get("failure") for a in att],
                feedback_f2=[a.get("feedback_f2") for a in att], finish_reason=[a.get("finish_reason") for a in att],
                auto_ended=[a.get("auto_ended") for a in att],
                corrected_later_in_run=[a.get("corrected_later_in_run") for a in att],
                error_event_in_attempt=[a.get("error_event_in_attempt") for a in att],
                run_outcome_class=ro.get("outcome_class"), run_attempt_history=ro.get("attempt_history"),
                n_commands=len(ann["commands"]), n_task_events=len(ann["task_events"]),
                first_attempt_success=a0.get("success") if a0["attempt"] == 1 else None)


def cmd_plan(a):
    task_order = {}
    if a.task_set and os.path.exists(a.task_set):
        for i, line in enumerate(l for l in open(a.task_set) if l.strip() and not l.startswith("#")):
            task_order[line.split(":")[0].strip()] = i
    batches = [b for b in a.batches.split(",") if b] if a.batches else None
    eps, rejected = collect([r for r in a.raw.split(",") if r], a.runs_root, batches, task_order)
    if a.limit:
        eps = eps[:a.limit]
    paper = paper_tasks(a.paper)
    idx0 = 0
    for i, e in enumerate(eps):
        ann = e["ann"]
        tname = ann["task"]["name"]
        if tname in paper:
            ann["task"]["paper"] = {k: v for k, v in paper[tname].items() if k != "description"}
        ann["episode_index"] = i
        ann["dataset_index_range"] = [idx0, idx0 + e["n"]]
        e["index"], e["from"] = i, idx0
        idx0 += e["n"]
        del e["key"]
    os.makedirs(a.work, exist_ok=True)
    with open(os.path.join(a.work, "plan.json"), "w") as f:
        json.dump(dict(episodes=eps, rejected=rejected, paper=paper, total_frames=idx0, repo_id=a.repo_id), f,
                  ensure_ascii=False)
    print(f"plan: {len(eps)} episodes, {idx0} frames, {len(rejected)} rejected", flush=True)


def shard_range(n_eps, i, k):
    return (n_eps * i) // k, (n_eps * (i + 1)) // k


def cmd_shard(a):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    t0 = time.time()
    plan = json.load(open(os.path.join(a.work, "plan.json")))
    eps = plan["episodes"]
    lo, hi = shard_range(len(eps), a.shard, a.nshards)
    out = os.path.join(a.work, "shards", f"shard_{a.shard:03d}")
    if os.path.exists(os.path.join(out, "DONE")):
        print(f"shard {a.shard}: done already", flush=True)
        return
    shutil.rmtree(out, ignore_errors=True)
    ds = LeRobotDataset.create(repo_id=f"{plan['repo_id']}-shard{a.shard:03d}", fps=FPS, features=features(), root=out,
                               robot_type="franka_panda", use_videos=True, image_writer_threads=a.threads,
                               vcodec=a.vcodec)
    nfr = 0
    for e in eps[lo:hi]:
        d, ann, n = e["dir"], e["ann"], e["n"]
        fr = np.load(os.path.join(d, "frames.npz"))
        assert len(fr["action"]) == n, d
        succ = {x["attempt"]: (-1 if x.get("success") is None else int(bool(x["success"]))) for x in ann["attempts"]}
        evc = np.zeros(n, np.int64)
        for ev in ann["task_events"]:
            if 0 <= ev["frame"] < n:
                evc[ev["frame"]] += 1
        state = np.concatenate([fr["eef_pos"], fr["eef_axis_angle"], fr["gripper_qpos"]], 1).astype(np.float32)
        instr = ann["task"]["instruction"]
        k = 0
        for img, wr in zip(frames_of(os.path.join(d, "agentview.mp4")), frames_of(os.path.join(d, "wrist.mp4"))):
            if k >= n:
                break
            ds.add_frame({
                "observation.images.image": img, "observation.images.wrist_image": wr,
                "observation.state": state[k],
                "observation.joint_pos": fr["joint_pos"][k].astype(np.float32),
                "observation.joint_vel": fr["joint_vel"][k].astype(np.float32),
                "observation.eef_quat": fr["eef_quat_xyzw"][k].astype(np.float32),
                "observation.gripper_width": np.array([fr["gripper_width"][k]], np.float32),
                "action": fr["action"][k].astype(np.float32),
                "annotation.attempt_index": np.array([fr["attempt"][k]], np.int64),
                "annotation.command_index": np.array([fr["command"][k]], np.int64),
                "annotation.phase": np.array([fr["phase"][k]], np.int64),
                "annotation.attempt_success": np.array([succ.get(int(fr["attempt"][k]), -1)], np.int64),
                "annotation.task_event": np.array([evc[k]], np.int64),
                "task": instr,
            })
            k += 1
        if k != n:
            raise RuntimeError(f"{d}: {k} video frames for {n} numeric frames")
        ds.save_episode()
        nfr += n
    ds.finalize()
    shutil.rmtree(os.path.join(out, "images"), ignore_errors=True)
    open(os.path.join(out, "DONE"), "w").write(f"{lo} {hi} {nfr}\n")
    print(f"shard {a.shard}: episodes {lo}-{hi}, {nfr} frames, {time.time() - t0:.0f}s", flush=True)


def cmd_merge(a):
    from pathlib import Path
    from lerobot.datasets.aggregate import aggregate_datasets
    t0 = time.time()
    plan = json.load(open(os.path.join(a.work, "plan.json")))
    eps = plan["episodes"]
    roots, ids = [], []
    for i in range(a.nshards):
        lo, hi = shard_range(len(eps), i, a.nshards)
        if hi <= lo:
            continue
        r = os.path.join(a.work, "shards", f"shard_{i:03d}")
        if not os.path.exists(os.path.join(r, "DONE")):
            raise SystemExit(f"shard {i} is not done")
        roots.append(Path(r))
        ids.append(f"{plan['repo_id']}-shard{i:03d}")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    aggregate_datasets(repo_ids=ids, aggr_repo_id=plan["repo_id"], roots=roots, aggr_root=Path(a.out))
    print(f"aggregated {len(roots)} shards in {time.time() - t0:.0f}s", flush=True)
    write_sidecars(a, plan)
    print(f"merge done in {time.time() - t0:.0f}s", flush=True)


def write_sidecars(a, plan):
    eps, paper = plan["episodes"], plan["paper"]
    lff = os.path.join(a.out, "lff")
    for sub in ("annotations", "sim_state", "final", "code"):
        os.makedirs(os.path.join(lff, sub), exist_ok=True)
    rows, runs, tasks = [], {}, {}
    for e in eps:
        i, d, ann, n = e["index"], e["dir"], e["ann"], e["n"]
        name = f"episode_{i:06d}"
        if os.path.exists(os.path.join(d, "sim_state.npy")):
            shutil.copy(os.path.join(d, "sim_state.npy"), os.path.join(lff, "sim_state", name + ".npy"))
        shutil.copy(os.path.join(d, "final.npz"), os.path.join(lff, "final", name + ".npz"))
        for cam, key in (("agentview", "image"), ("wrist", "wrist_image")):
            shutil.copy(os.path.join(d, f"final_{cam}.png"), os.path.join(lff, "final", f"{name}_{key}.png"))
        with open(os.path.join(lff, "annotations", name + ".json"), "w") as f:
            json.dump(ann, f, indent=1, ensure_ascii=False)
        rows.append(flat_row(i, ann, n, e["from"]))
        tname = ann["task"]["name"]
        rk = (ann["source"]["batch"], ann["source"]["run"])
        r = runs.setdefault(rk, dict(batch=rk[0], run=rk[1], manifest_id=ann["source"].get("manifest_id"),
                                     run_status=ann["source"].get("run_status"), condition=ann["source"]["condition"],
                                     memory_free_chain_of=ann["source"].get("memory_free_chain_of"),
                                     memory_free_rep=ann["source"].get("memory_free_rep"), task=tname,
                                     seed=ann["task"]["seed"], model=ann["agent"].get("model"),
                                     harness=ann["agent"].get("harness"),
                                     harness_version=ann["agent"].get("harness_version"),
                                     codex_catalog=ann["agent"].get("codex_catalog"),
                                     outcome_class=ann["run_outcome"].get("outcome_class"),
                                     attempt_history=ann["run_outcome"].get("attempt_history"),
                                     agent_final_report=ann["run_outcome"].get("agent_final_report"),
                                     episodes=[]))
        for x in ann["attempts"]:
            r["episodes"].append(dict(attempt=x["attempt"], episode_index=i, success=x.get("success")))
        if tname not in tasks:
            tasks[tname] = {k: v for k, v in ann["task"].items() if k not in ("seed", "instance", "paper")}
            if tname in paper:
                tasks[tname]["paper"] = paper[tname]
            tasks[tname]["instances"] = {}
        tasks[tname]["instances"][str(ann["task"]["seed"])] = ann["task"]["instance"]
    # memory-free retries <-> the history run whose first attempt they share (manifest "chain" = history run id)
    by_id = {(r["batch"], r["manifest_id"]): rn for (b, rn), r in runs.items() if r.get("manifest_id")}
    for r in runs.values():
        if r.get("memory_free_chain_of"):
            r["memory_free_chain_of_run"] = by_id.get((r["batch"], r["memory_free_chain_of"]))
    for (b, rn), r in runs.items():
        r["memory_free_retries"] = sorted((x["run"] for x in runs.values()
                                           if x["batch"] == b and x.get("memory_free_chain_of_run") == rn),
                                          key=lambda q: runs[(b, q)].get("memory_free_rep") or 0)
    with open(os.path.join(lff, "episodes.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(lff, "runs.jsonl"), "w") as f:
        for r in runs.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(lff, "tasks.json"), "w") as f:
        json.dump(dict(causes=CAUSES, phase_codes=PHASES, tasks=tasks), f, indent=1, ensure_ascii=False)
    with open(os.path.join(lff, "batches.json"), "w") as f:
        json.dump(BATCH_NOTES, f, indent=1)
    with open(os.path.join(lff, "rejected.json"), "w") as f:
        json.dump(plan["rejected"], f, indent=1)
    here = os.path.dirname(os.path.abspath(__file__))
    for s in ("lerobot_replay.py", "lerobot_export.py"):
        shutil.copy(os.path.join(here, s), os.path.join(lff, "code", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", "shard", "merge"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--raw", default="")
    ap.add_argument("--runs-root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--paper", default="")
    ap.add_argument("--task-set", default="")
    ap.add_argument("--repo-id", default="lffbench/agent-replays")
    ap.add_argument("--batches", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--vcodec", default="libsvtav1")
    a = ap.parse_args()
    {"plan": cmd_plan, "shard": cmd_shard, "merge": cmd_merge}[a.step](a)


if __name__ == "__main__":
    main()
