"""landing sensitivity to reset error: teleport cube (on a board at its start) or board+cube by dx, pull at v"""
import sys, json, numpy as np
from lffbench.tasks.l5_tablecloth_pull import TableclothPull, BOARD, CUBE, B_HALF, C_HALF
from lffbench import envs
seeds = [int(x) for x in sys.argv[1].split(",")]; speeds = [float(x) for x in sys.argv[2].split(",")]; out = sys.argv[3]
task = TableclothPull(); TZ = envs.TABLE_Z
rows = []
def run(inst, v, cdx, bdx):
    task.reset_instance(inst)
    bx, by = inst["board_xy"]; c = task.cube_start(inst)
    envs.set_obj_pose(task.env, BOARD, [bx + bdx, by, TZ + B_HALF[2] + 0.0005])
    envs.set_obj_pose(task.env, CUBE, [c[0] + bdx + cdx, c[1], TZ + 2 * B_HALF[2] + C_HALF + 0.0005])
    task.env.sim.forward(); envs.settle(task.env, 10)
    task.execute(inst, dict(speed=v)); envs.settle(task.env, 60)
    p = envs.obj_pos(task.env, CUBE)
    return float(p[0]), bool(task.cube_on_board())
for seed in seeds:
    inst = task.sample_instance(seed)
    for v in speeds:
        x0, _ = run(inst, v, 0, 0)
        res = {}
        for cdx in (-0.005, -0.003, 0.003, 0.005):
            x, ob = run(inst, v, cdx, 0); res[f"cube{1000*cdx:+.0f}mm"] = round(100 * (x - x0), 2) if not ob else "R"
        for bdx in (-0.005, 0.005):
            x, ob = run(inst, v, 0, bdx); res[f"board{1000*bdx:+.0f}mm"] = round(100 * (x - x0), 2) if not ob else "R"
        rows.append(dict(seed=seed, mu=inst["mu"], v=v, shifts_cm=res))
        print(seed, round(inst["mu"], 3), v, res, flush=True)
        json.dump(rows, open(out, "w"), indent=1)
