"""python scripts/l6_controls.py <l6_count_into_box|l6_sequence_resume> [n]

Extra controls for the two L6 tasks: thumbnails right after the interruption, extra scripted policies,
no-injection sanity check, analytic memoryless baselines. Writes results/<task>/controls.json."""
import sys, json, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, imageio
from lffbench import envs
import lffbench.tasks.l6_count, lffbench.tasks.l6_sequence  # noqa
from lffbench.task_base import TASKS
name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
out_dir = os.path.join(ROOT, "results", name)
os.makedirs(out_dir, exist_ok=True)
task = TASKS[name](cam_res=384)
extra = ["skip", "inc1"] if name == "l6_count_into_box" else ["skip"]
res = dict(task=name, n=n, seeds=list(range(1000, 1000 + n)), extra={}, no_inject={}, analytic={})
t0 = time.time()

# 1) frames right after the interruption (what the agent sees next), seeds 1000-1002, adaptive policy
def interrupted():
    return (task.slip if name == "l6_count_into_box" else task.stop) is not None


for seed in range(1000, 1003):
    inst = task.sample_instance(seed)
    shot = {}
    orig_step = task.env.step

    def step(a, _o=orig_step):
        r = _o(a)
        if "f" not in shot and interrupted():
            shot["cnt"] = shot.get("cnt", 0) + 1
            if shot["cnt"] >= (4 if name == "l6_count_into_box" else 60):  # count: 0.2 s; sequence: arm back home
                shot["f"] = np.concatenate([envs.render(task.env, "agentview", 384),
                                            envs.render(task.env, "robot0_eye_in_hand", 384)], 1)
        return r
    task.env.step = step
    task.run_scripted("adaptive", inst, k=5, seed=seed)
    task.env.step = orig_step
    if "f" in shot:
        imageio.imwrite(os.path.join(out_dir, f"interrupt_{seed}.png"), shot["f"])
print("thumbs", f"{time.time()-t0:.0f}s", flush=True)

# 2) extra scripted policies
for kind in extra:
    fs = []
    for i in range(n):
        seed = 1000 + i
        inst = task.sample_instance(seed)
        h = task.run_scripted(kind, inst, k=5, seed=seed)
        s = next((x["attempt"] for x in h if x["outcome"]["success"]), None)
        fs.append(s)
    res["extra"][kind] = dict(first_success=fs, succ_at_k=[float(np.mean([s is not None and s <= j for s in fs])) for j in range(1, 6)])
    print(kind, res["extra"][kind]["succ_at_k"], f"{time.time()-t0:.0f}s", flush=True)

# 3) no-injection sanity (oracle + naive, first 10 seeds)
task.inject = False
for kind in ("adaptive", "naive"):
    fs = []
    for i in range(10):
        seed = 1000 + i
        inst = task.sample_instance(seed)
        h = task.run_scripted(kind, inst, k=1, seed=seed)
        fs.append(bool(h[0]["outcome"]["success"]))
    res["no_inject"][kind] = float(np.mean(fs))
    print("no inject", kind, res["no_inject"][kind], flush=True)
task.inject = True

# 4) analytic memoryless baselines over the same seeds (perfect execution assumed)
if name == "l6_count_into_box":
    insts = [task.sample_instance(1000 + i) for i in range(n)]
    res["analytic"]["uniform_guess_0..N-1@1"] = float(np.mean([1 / x["n_target"] for x in insts]))
    best = max(range(0, 6), key=lambda g: np.mean([x["n_target"] - x["p"] == g for x in insts]))
    res["analytic"]["best_fixed_number_to_add@1"] = dict(add=best, rate=float(np.mean([x["n_target"] - x["p"] == best for x in insts])))
    res["analytic"]["p_hist"] = {str(v): int(sum(x["p"] == v for x in insts)) for v in range(1, 6)}
    res["analytic"]["N_hist"] = {str(v): int(sum(x["n_target"] == v for x in insts)) for v in (4, 5)}
    big = [task.sample_instance(5000 + i) for i in range(2000)]
    res["analytic"]["uniform_guess@1_2000inst"] = float(np.mean([1 / x["n_target"] for x in big]))
    res["analytic"]["best_fixed_add@1_2000inst"] = float(max(np.mean([x["n_target"] - x["p"] == g for x in big]) for g in range(1, 5)))
else:
    insts = [task.sample_instance(1000 + i) for i in range(n)]
    res["analytic"]["w_hist"] = {str(v): int(sum(x["stop_w"] == v for x in insts)) for v in range(1, 5)}
    res["analytic"]["phase_hist"] = {p: int(sum(x["stop_phase"] == p for x in insts)) for p in "AB"}
    res["analytic"]["uniform_guess_0..4@1"] = 0.2
    big = [task.sample_instance(5000 + i) for i in range(2000)]
    res["analytic"]["best_fixed_guess@1_2000inst"] = float(max(np.mean([x["stop_w"] == g for x in big]) for g in range(0, 5)))
    res["analytic"]["skip_expected"] = float(np.mean([x["stop_phase"] == "B" or x["stop_w"] == 3 for x in big]))
res["seconds"] = time.time() - t0
json.dump(res, open(os.path.join(out_dir, "controls.json"), "w"), indent=1)
print(json.dumps(res["analytic"]))
