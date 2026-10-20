"""best-effort scripted reset (reset3: low approach to the far cube, cube onto its spot on the pulled-back board,
board pushed back at 0.03 m/s with the cube riding on it) on many seeds; per-command stall log; control pull from a
teleported exact start with the same arm state."""
import sys, json, time, numpy as np
sys.path.insert(0, sys.argv[0].rsplit('/',1)[0])
from lffbench.tasks.l5_tablecloth_pull import TableclothPull, CUBE
from lffbench import envs
import reset_lib
from reset_lib import Runner, teleport_start
seeds = [int(x) for x in sys.argv[1].split(",")]
speeds = [float(x) for x in sys.argv[2].split(",")]
out = sys.argv[3]
task = TableclothPull()
env = task.env; sim = env.sim; rb = env.robots[0]
lim4 = sim.model.jnt_range[rb._ref_joint_indexes][3]
stalls = []
orig = Runner.cmd
def cmd(self, xyz=None, yaw=None, grip=None, speed=None, tol=0.004, max_steps=250):
    orig(self, xyz, yaw, grip, speed, tol, max_steps)
    if xyz is not None:
        err = float(np.linalg.norm(self.sk.eef_pos() - np.asarray(xyz)))
        q4 = float(sim.data.qpos[rb._ref_joint_pos_indexes][3])
        if err > 0.01:
            stalls.append(dict(cmd=self.ncmd, tgt=np.round(xyz, 3).tolist(), err_cm=round(100 * err, 1),
                               elbow_at_limit=bool(lim4[1] - q4 < 0.01)))
Runner.cmd = cmd
rows = []
for seed in seeds:
    inst = task.sample_instance(seed)
    for v in speeds:
        task.reset_instance(inst)
        r = Runner(task, inst)
        stalls.clear()
        t0 = time.time()
        r.pull(v); L0 = r.landing()
        n_before = r.ncmd
        R = r.reset3()
        st = sim.get_state(); n_steps = task.sk.n_steps; nc = r.ncmd
        teleport_start(task, inst); r.pull(v); C = r.landing()
        sim.set_state(st); sim.forward(); task.sk.n_steps = n_steps; r.ncmd = nc
        task.sk._ends = task.sk._board_ends(); task.sk._bstep = 0.0
        r.pull(v); L1 = r.landing()
        ok = R["board_cm"] <= 0.5 and abs(R["board_yaw_deg"]) <= 2 and R["cube_cm"] <= 0.5 and R["cube_on_board"]
        row = dict(seed=seed, mu=inst["mu"], v=v, L0=L0, R=R, C=C, L1=L1, ok=ok, stalls=list(stalls),
                   cube_land_x=L0["cube_xy"][0], wall=round(time.time() - t0, 1))
        rows.append(row)
        print(seed, v, "land_x", L0["cube_xy"][0], "ok", ok, "reset", (R["board_cm"], R["board_yaw_deg"], R["cube_dx_cm"], R["cube_dy_cm"], R["reset_cmds"], R["reset_sim_s"]),
              "dx_actual", round(100 * (L1["cube_xy"][0] - L0["cube_xy"][0]), 2), "dx_teleport", round(100 * (C["cube_xy"][0] - L0["cube_xy"][0]), 2),
              "stalls", [(s["cmd"], s["err_cm"], s["elbow_at_limit"]) for s in stalls], flush=True)
        json.dump(rows, open(out, "w"), indent=1)
