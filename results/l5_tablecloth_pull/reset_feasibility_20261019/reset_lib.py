"""Feasibility of a robot-side reset in l5_tablecloth_pull (2026-10-19; experiment record, not used by the task).

v1 Runner.reset: cube aside -> board back -> cube onto its start spot (unreachable: x 0.13-0.18 m).
v2 Runner.reset_ride / reset2: cube onto its spot on the pulled-back board, push the board back with the cube riding.
v3 Runner.reset3 (used by feas3.py): as v2, but the far cube is approached and left at a low height; push at 0.03 m/s.
Run from the repo root after `source run_env.sh`, e.g.
  python results/l5_tablecloth_pull/reset_feasibility_20261019/feas3.py 1000,1003 0.6,0.45 /tmp/out.json
  python results/l5_tablecloth_pull/reset_feasibility_20261019/sens.py 1000,1001 0.6,0.45 /tmp/sens.json
  python results/l5_tablecloth_pull/reset_feasibility_20261019/armhist.py 1000,1001 0.6,0.45 /tmp/arm.json

Commands are counted like the la interface: one move_eef = a motion optionally followed by a gripper action.
Perception is privileged (true object poses) plus optional Gaussian noise `pn` (m) on every perceived position.
"""
import numpy as np

from lffbench import envs
from lffbench.tasks.l5_tablecloth_pull import (B_HALF, BOARD, C_HALF, CUBE, H_HALF, STROKE, V_MAX, V_MIN)

TZ = envs.TABLE_Z
Z_HANDLE = TZ + 2 * B_HALF[2] + H_HALF[2]
Z_CUBE_TABLE = TZ + C_HALF
Z_CUBE_BOARD = TZ + 2 * B_HALF[2] + C_HALF


def board_pose(task):
    env = task.env
    b = envs.obj_pos(env, BOARD)
    R = env.sim.data.body_xmat[envs.body_id(env, BOARD)].reshape(3, 3)
    return b, float(np.arctan2(R[1, 0], R[0, 0]))


def cube_yaw(task):
    """yaw of a horizontal face normal of the cube (whichever face it rests on), folded to (-45, 45] deg"""
    R = task.env.sim.data.body_xmat[envs.body_id(task.env, CUBE)].reshape(3, 3)
    k = int(np.argmin(np.abs(R[2, :])))  # the body axis that is most horizontal
    a = np.arctan2(R[1, k], R[0, k])
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


class Runner:
    def __init__(self, task, inst, pn=0.0, rng=None):
        self.task, self.inst, self.sk = task, inst, task.sk
        self.pn = pn
        self.rng = rng or np.random.default_rng(0)
        self.ncmd = 0
        bx, by = inst["board_xy"]
        self.board0 = np.array([bx, by])
        self.cube0 = task.cube_start(inst)

    def see(self, xy):
        return np.asarray(xy, float)[:2] + (self.rng.normal(0, self.pn, 2) if self.pn else 0.0)

    def cmd(self, xyz=None, yaw=None, grip=None, speed=None, tol=0.004, max_steps=250):
        """one la move_eef"""
        self.ncmd += 1
        if xyz is not None:
            kw = dict(yaw=yaw) if yaw is not None else {}
            self.sk.move_to(xyz, speed=speed, tol=tol, max_steps=max_steps, **kw)
        if grip is not None:
            self.sk.set_gripper(grip == "close")

    def eef(self):
        return self.sk.eef_pos()

    # ------------------------------------------------------------ pull (same motion as task.execute)
    def pull(self, speed):
        b, _ = board_pose(self.task)
        bs = self.see(b)
        hx, hy = bs[0] - B_HALF[0] + H_HALF[0], bs[1]
        self.cmd([hx, hy, Z_HANDLE + 0.08], yaw=0.0)
        self.cmd([hx, hy, Z_HANDLE], tol=0.003, grip="close")
        self.cmd([hx - STROKE, hy, Z_HANDLE], speed=float(np.clip(speed, V_MIN, V_MAX)), tol=0.01, max_steps=300,
                 grip="open")
        self.cmd([hx - STROKE, hy, Z_HANDLE + 0.10], speed=0.3)

    # ------------------------------------------------------------ reset pieces
    def pick_cube(self):
        c = envs.obj_pos(self.task.env, CUBE)
        cs = self.see(c)
        zc = c[2]  # height: the agent can tell table vs board
        yaw = cube_yaw(self.task)
        self.cmd([cs[0], cs[1], zc + 0.10], yaw=yaw)
        self.cmd([cs[0], cs[1], zc + 0.003], tol=0.003, grip="close")
        self.cmd([cs[0], cs[1], zc + 0.08], speed=0.25)

    def place_cube(self, xy, z_centre, yaw=0.0):
        self.cmd([xy[0], xy[1], z_centre + 0.08], yaw=yaw)
        self.cmd([xy[0], xy[1], z_centre + 0.005], tol=0.003, grip="open")
        self.cmd([xy[0], xy[1], z_centre + 0.08])

    def move_board(self, target_xy, speed=0.15):
        """grasp the handle, slide the board so its centre goes to target_xy (perceived), release, lift."""
        b, _ = board_pose(self.task)
        bs = self.see(b)
        hx, hy = bs[0] - B_HALF[0] + H_HALF[0], bs[1]
        self.cmd([hx, hy, Z_HANDLE + 0.08], yaw=0.0)
        self.cmd([hx, hy, Z_HANDLE], tol=0.003, grip="close")
        # where the board is now relative to the gripper (perceived again after closing)
        b1, _ = board_pose(self.task)
        d = np.asarray(target_xy) - self.see(b1)
        e = self.eef()
        self.cmd([e[0] + d[0], e[1] + d[1], Z_HANDLE], speed=speed, tol=0.002, max_steps=400, grip="open")
        self.cmd([e[0] + d[0], e[1] + d[1], Z_HANDLE + 0.08])

    def errors(self):
        b, yaw = board_pose(self.task)
        c = envs.obj_pos(self.task.env, CUBE)
        return dict(board_cm=round(100 * float(np.linalg.norm(b[:2] - self.board0)), 2),
                    board_dx_cm=round(100 * float(b[0] - self.board0[0]), 2),
                    board_yaw_deg=round(float(np.degrees(yaw)), 2),
                    cube_cm=round(100 * float(np.linalg.norm(c[:2] - self.cube0)), 2),
                    cube_dx_cm=round(100 * float(c[0] - self.cube0[0]), 2),
                    cube_dy_cm=round(100 * float(c[1] - self.cube0[1]), 2),
                    cube_on_board=self.task.cube_on_board(),
                    cube_up=round(envs.obj_upright_cos(self.task.env, CUBE), 3))

    def reset(self, board_tol=0.003, max_fix=2, aside=None):
        """cube aside -> board back to its start (with up to max_fix corrections) -> cube onto its start spot."""
        n0, s0 = self.ncmd, self.sk.n_steps
        c = envs.obj_pos(self.task.env, CUBE)
        by = self.board0[1]
        side = -1.0 if by > 0 else 1.0
        aside = np.array([c[0], by + side * 0.14]) if aside is None else np.asarray(aside)
        self.pick_cube()
        self.place_cube(aside, Z_CUBE_TABLE)
        self.move_board(self.board0)
        nfix = 0
        while nfix < max_fix:
            b, yaw = board_pose(self.task)
            if np.linalg.norm(self.see(b) - self.board0) <= board_tol:
                break
            self.move_board(self.board0, speed=0.05)
            nfix += 1
        self.pick_cube()
        self.place_cube(self.see(self.cube0), Z_CUBE_BOARD)
        envs.settle(self.task.env, 20)
        r = self.errors()
        r.update(reset_cmds=self.ncmd - n0, reset_sim_s=round((self.sk.n_steps - s0) * 0.05, 1), board_fixes=nfix)
        return r

    def landing(self):
        envs.settle(self.task.env, 60)
        c = envs.obj_pos(self.task.env, CUBE)
        return dict(cube_xy=c[:2].round(4).tolist(), moved_cm=round(100 * float(self.cube0[0] - c[0]), 2),
                    on_board=self.task.cube_on_board(), up=round(envs.obj_upright_cos(self.task.env, CUBE), 3))


def board_marker_xy(task, inst):
    """world xy of the cube's start spot on the board, wherever the board is now"""
    b, yaw = board_pose(task)
    off = task.cube_start(inst) - np.asarray(inst["board_xy"])  # (B_HALF[0] - LF, 0) in the board frame
    c, s = np.cos(yaw), np.sin(yaw)
    return b[:2] + np.array([c * off[0] - s * off[1], s * off[0] + c * off[1]])


def reset_ride(self, push_speed=0.08, board_tol=0.003, max_fix=2):
    """cube onto its spot on the pulled-back board -> push the board (cube riding on it) back to its start."""
    n0, s0 = self.ncmd, self.sk.n_steps
    self.pick_cube()
    _, byaw = board_pose(self.task)
    tgt = board_marker_xy(self.task, self.inst)
    self.place_cube(self.see(tgt), Z_CUBE_BOARD, yaw=byaw)
    self.move_board(self.board0, speed=push_speed)
    nfix = 0
    while nfix < max_fix:
        b, yaw = board_pose(self.task)
        if np.linalg.norm(self.see(b) - self.board0) <= board_tol:
            break
        self.move_board(self.board0, speed=0.04)
        nfix += 1
    envs.settle(self.task.env, 20)
    r = self.errors()
    r.update(reset_cmds=self.ncmd - n0, reset_sim_s=round((self.sk.n_steps - s0) * 0.05, 1), board_fixes=nfix)
    return r


Runner.reset_ride = reset_ride


# ------------------------------------------------------------ v2: keep the arm bent (no lifting at the far end)
def pick_cube2(self):
    c = envs.obj_pos(self.task.env, CUBE)
    cs = self.see(c)
    zc = c[2]
    yaw = cube_yaw(self.task)
    self.cmd([cs[0], cs[1], zc + 0.05], yaw=yaw)
    self.cmd([cs[0], cs[1], zc + 0.003], tol=0.003, grip="close")
    self.cmd([cs[0] - 0.06, cs[1], zc + 0.03], speed=0.15)  # back toward the robot first, low


def place_cube2(self, xy, z_centre, yaw=0.0):
    self.cmd([xy[0], xy[1], z_centre + 0.03], yaw=yaw)
    self.cmd([xy[0], xy[1], z_centre + 0.003], tol=0.003, grip="open")
    self.cmd([xy[0], xy[1], z_centre + 0.06])


def move_board2(self, target_xy, speed=0.03):
    b, _ = board_pose(self.task)
    bs = self.see(b)
    hx, hy = bs[0] - B_HALF[0] + H_HALF[0], bs[1]
    self.cmd([hx, hy, Z_HANDLE + 0.06], yaw=0.0)
    self.cmd([hx, hy, Z_HANDLE], tol=0.003, grip="close")
    b1, _ = board_pose(self.task)
    d = np.asarray(target_xy) - self.see(b1)
    e = self.eef()
    self.cmd([e[0] + d[0], e[1] + d[1], Z_HANDLE], speed=speed, tol=0.002, max_steps=600, grip="open")
    self.cmd([e[0] + d[0], e[1] + d[1], Z_HANDLE + 0.06])


def reset2(self, push_speed=0.03, board_tol=0.002, max_fix=2):
    n0, s0 = self.ncmd, self.sk.n_steps
    self.pick_cube2()
    _, byaw = board_pose(self.task)
    tgt = board_marker_xy(self.task, self.inst)
    self.place_cube2(self.see(tgt), Z_CUBE_BOARD, yaw=byaw)
    self.move_board2(self.board0, speed=push_speed)
    nfix = 0
    while nfix < max_fix:
        b, yaw = board_pose(self.task)
        if np.linalg.norm(self.see(b) - self.board0) <= board_tol:
            break
        self.move_board2(self.board0, speed=0.02)
        nfix += 1
    envs.settle(self.task.env, 20)
    r = self.errors()
    r.update(reset_cmds=self.ncmd - n0, reset_sim_s=round((self.sk.n_steps - s0) * 0.05, 1), board_fixes=nfix)
    return r


Runner.pick_cube2, Runner.place_cube2, Runner.move_board2, Runner.reset2 = pick_cube2, place_cube2, move_board2, reset2


def teleport_start(task, inst):
    """put board and cube exactly at their start poses (control: arm state as it is)"""
    bx, by = inst["board_xy"]
    envs.set_obj_pose(task.env, BOARD, [bx, by, TZ + B_HALF[2] + 0.0005])
    c = task.cube_start(inst)
    envs.set_obj_pose(task.env, CUBE, [c[0], c[1], TZ + 2 * B_HALF[2] + C_HALF + 0.0005])
    task.env.sim.forward()
    envs.settle(task.env, 10)


def pick_cube3(self, back=0.08, zt=0.95, spd=0.1):
    """approach a far cube from the robot side at a low height, grasp, come back the same way"""
    c = envs.obj_pos(self.task.env, CUBE)
    cs = self.see(c)
    zc = c[2]
    yaw = cube_yaw(self.task)
    self.cmd([cs[0] - back, cs[1], zt], yaw=yaw)
    self.cmd([cs[0], cs[1], zt], speed=spd)
    self.cmd([cs[0], cs[1], zc + 0.003], tol=0.003, grip="close")
    self.cmd([cs[0] - back, cs[1], zt + 0.01], speed=spd)


def reset3(self, push_speed=0.03, board_tol=0.002, max_fix=2):
    n0, s0 = self.ncmd, self.sk.n_steps
    self.pick_cube3()
    _, byaw = board_pose(self.task)
    tgt = board_marker_xy(self.task, self.inst)
    self.place_cube2(self.see(tgt), Z_CUBE_BOARD, yaw=byaw)
    self.move_board2(self.board0, speed=push_speed)
    nfix = 0
    while nfix < max_fix:
        b, yaw = board_pose(self.task)
        if np.linalg.norm(self.see(b) - self.board0) <= board_tol:
            break
        self.move_board2(self.board0, speed=0.02)
        nfix += 1
    envs.settle(self.task.env, 20)
    r = self.errors()
    r.update(reset_cmds=self.ncmd - n0, reset_sim_s=round((self.sk.n_steps - s0) * 0.05, 1), board_fixes=nfix)
    return r


Runner.pick_cube3, Runner.reset3 = pick_cube3, reset3
