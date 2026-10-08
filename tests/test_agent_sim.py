"""Simulator tests (LIBERO/MuJoCo, CPU): camera geometry, tool reports, hidden bias, and the end-to-end
ScriptedBackend run on l5_slide_to_target (n=3). Takes a few minutes on the shared 8-core machine."""
import json
import os

import numpy as np
import pytest

from lffbench import envs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def l5():
    from lffbench.tasks.l5_slide import SlideToTarget
    task = SlideToTarget()
    inst = task.sample_instance(1001)
    task.reset_instance(inst)
    return task, inst


def make_ex(task, inst, **kw):
    from lffbench.agent.tools import ToolExecutor
    task.reset_instance(inst)
    return ToolExecutor(task, inst, image_res=kw.pop("image_res", 256), **kw)


def test_pixel_to_world_roundtrip(l5):
    from lffbench.agent import camera as C
    task, inst = l5
    task.reset_instance(inst)
    top = envs.obj_pos(task.env, "cube_1") + np.array([0, 0, 0.02])  # centre of the cube's top face
    for cam, tol in (("agentview", 0.006), ("wrist", 0.003)):
        u, v, _ = C.world_to_pixel(task.env, cam, top, 512)
        assert 0 <= u < 512 and 0 <= v < 512
        rgb = envs.render(task.env, C.mj_cam(cam), 512)
        r, g, b = rgb[int(round(v)), int(round(u))].astype(int)
        assert r > 120 and g < 80 and b < 80, "projected pixel should be on the red cube (image orientation)"
        p = C.pixel_to_world(task.env, cam, u, v, 512)["xyz"]
        assert np.linalg.norm(p - top) < tol
    ax = C.describe_axes(task.env, "agentview", 512)
    assert ax == dict(right="+y", down="+x")


def test_tool_reports_and_hidden_bias(l5):
    task, inst = l5
    ex = make_ex(task, inst)
    bias = np.array([0.03, -0.02, 0.0])
    ex.sk.bias = bias.copy()
    tgt = np.array([-0.30, 0.05, 1.05])
    r = ex.call("move_to", dict(x=tgt[0], y=tgt[1], z=tgt[2], yaw=0.0, speed=None))
    assert r.ok and r.report["reached"]
    # proprioception is in the robot's (miscalibrated) frame: it reports the commanded point ...
    assert np.linalg.norm(np.array(r.report["eef_pos"]) - tgt) < 0.01
    # ... while the gripper truly sits at target + bias
    assert np.linalg.norm(ex.sk.eef_pos() - (tgt + bias)) < 0.01
    assert set(r.report) >= {"reached", "eef_pos", "eef_yaw_deg", "gripper", "gripper_width_m", "holding_object",
                             "touched_object", "sim_time_s"}
    # camera-based localisation stays in the true world frame
    from lffbench.agent import camera as C
    top = envs.obj_pos(task.env, "cube_1") + np.array([0, 0, 0.02])
    u, v, _ = C.world_to_pixel(task.env, "agentview", top, 256)
    pw = ex.call("pixel_to_world", dict(camera="agentview", u=u, v=v))
    assert pw.ok and np.linalg.norm(np.array(pw.report["world_xyz"]) - top) < 0.01


def test_tool_errors_clipping_and_privileged_locate(l5):
    task, inst = l5
    ex = make_ex(task, inst)
    assert not ex.call("fly", {}).ok
    assert "not valid JSON" in ex.call("move_to", "{x: 1").report["error"]
    assert not ex.call("move_to", dict(x=0.0, y=0.0)).ok  # missing args
    assert not ex.call("locate", dict(object_name="cube_1")).ok  # not enabled
    r = ex.call("move_to", dict(x=-0.2, y=0.0, z=0.5, yaw=0, speed=None))  # below the table -> clipped
    assert r.ok and r.report["target_clipped_to"][2] > envs.TABLE_Z
    r = ex.call("close_gripper", {})
    assert r.report["gripper"] == "closed"
    ex2 = make_ex(task, inst, privileged_locate=True, grasp_report="name")
    loc = ex2.call("locate", dict(object_name="cube_1"))
    assert loc.ok and np.linalg.norm(np.array(loc.report["position"]) - envs.obj_pos(task.env, "cube_1")) < 1e-3
    obs = ex2.call("get_observation", {})
    assert obs.ok and [i.camera for i in obs.images] == ["agentview", "wrist"] and obs.images[0].array.shape == (256, 256, 3)
    push = ex2.call("push", dict(x=inst["cube_xy"][0] - 0.02, y=inst["cube_xy"][1], dir_x=1, dir_y=0, distance=0.01,
                                 speed=0.2, z=None, runup=None))
    assert push.ok and "cube_1" in push.report["touched_objects"]


def _validate_is_fresh():
    vj = os.path.join(ROOT, "results", "l5_slide_to_target", "validate.json")
    src = os.path.join(ROOT, "lffbench", "tasks", "l5_slide.py")
    return os.path.exists(vj) and os.path.getmtime(vj) > os.path.getmtime(src), vj


def test_scripted_end_to_end_l5_n3(tmp_path):
    """ScriptedBackend (adaptive policy, F2, full memory) through the whole pipeline on 3 instances."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_agent
    run_dir, agg, runs = run_agent.main(["--task", "l5_slide_to_target", "--backend", "scripted", "--memory", "full",
                                         "--feedback", "F2", "--n", "3", "--k", "5", "--seed0", "1000",
                                         "--out", str(tmp_path), "--dry-run-payload", "--dump-requests", "2"])
    assert agg["n_runs"] == 3 and agg["n_error_runs"] == 0
    assert agg["succ_at"][-1] == 1.0, agg  # validate.json: adaptive succ@5 = 0.95 over 20 seeds; seeds 1000-1002 succeed
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    assert len(summ["runs"]) == 3 and all(r["status"] == "ok" for r in summ["runs"])
    ev = [json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl"))]
    calls = [e for e in ev if e["type"] == "llm_call"]
    tools = [e for e in ev if e["type"] == "tool_call"]
    ends = [e for e in ev if e["type"] == "attempt_end"]
    assert calls and all("latency_s" in c and "dry_run_request" in c for c in calls)
    assert all(c["dry_run_request"]["n_images"] >= 2 for c in calls)
    assert {t["name"] for t in tools} <= {"move_to", "close_gripper", "open_gripper", "push", "done"}
    for e in ends:
        assert 1 <= len(e["keyframes"]) <= 4
        for _, p in e["keyframes"]:
            assert os.path.exists(os.path.join(run_dir, p))
        assert e["ended_by"] == "done"
    assert len(os.listdir(os.path.join(run_dir, "requests"))) == 2
    # the scripted policy driven through the tool interface reproduces the task's own reference rollouts
    fresh, vj = _validate_is_fresh()
    if fresh:
        ref = {r["seed"]: r for r in json.load(open(vj))["runs"] if r["kind"] == "adaptive"}
        for e in ends:
            h = ref[e["seed"]]["history"][e["attempt"] - 1]
            assert abs(h["outcome"]["along_err"] - e["outcome"]["along_err"]) < 0.002, (e["seed"], e["attempt"])
            assert abs(h["params"]["speed"] - e["backend_meta"]["params"]["speed"]) < 1e-3


def test_mismatched_with_auto_donor(tmp_path):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_agent
    run_dir, agg, runs = run_agent.main(["--task", "l5_slide_to_target", "--backend", "scripted", "--memory",
                                         "mismatched", "--n", "1", "--k", "3", "--seed0", "1002", "--out", str(tmp_path),
                                         "--image-res", "256", "--keyframes", "2"])
    ev = [json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl"))]
    start = next(e for e in ev if e["type"] == "run_start")
    assert start["donor_seed"] == 11002
    for e in ev:
        if e["type"] == "attempt_end" and e["attempt"] > 1:
            assert e["history_shown"] and all(s == 11002 for s, _ in e["history_shown"])
    assert os.path.exists(os.path.join(run_dir, "donors", "events.jsonl"))
