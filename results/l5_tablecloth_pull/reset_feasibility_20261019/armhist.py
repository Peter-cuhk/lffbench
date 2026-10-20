"""same object state, same pull, different arm history beforehand (no object touched): landing x difference"""
import sys, json, numpy as np
from lffbench.tasks.l5_tablecloth_pull import TableclothPull, CUBE
from lffbench import envs
seeds = [int(x) for x in sys.argv[1].split(",")]; speeds = [float(x) for x in sys.argv[2].split(",")]; out = sys.argv[3]
task = TableclothPull()
paths = {"none": [], "left_right": [[-0.15, 0.25, 1.05], [-0.15, -0.25, 1.05]], "high": [[-0.2, 0.0, 1.3]],
         "far_front": [[0.10, 0.15, 1.0], [-0.1, 0.0, 1.05]]}
rows = []
for seed in seeds:
    inst = task.sample_instance(seed)
    for v in speeds:
        res = {}
        for name, pts in paths.items():
            task.reset_instance(inst)
            for p in pts:
                task.sk.move_to(p, yaw=0.0)
            task.execute(inst, dict(speed=v)); envs.settle(task.env, 60)
            res[name] = float(envs.obj_pos(task.env, CUBE)[0])
        d = {k: round(100 * (x - res["none"]), 2) for k, x in res.items() if k != "none"}
        rows.append(dict(seed=seed, v=v, dx_cm=d)); print(seed, v, d, flush=True)
        json.dump(rows, open(out, "w"), indent=1)
