"""L3-C under_table_tool: a cube sits under a low glass side table; put it into the bin (tool use).

v2 (2026-10-08, user review of cla2): in v1 the hidden information was "the hand does not fit under the glass top".
Every Sonnet run (4/4) saw the T-shaped stick lying next to the low table and used it straight away, i.e. the task
was solved by reasoning before acting and nothing had to be learned from a failure. v2 keeps the scene and moves the
hidden information to something no image shows:

Hidden: the side of the low table AWAY from the stick is closed by a clear acrylic panel (between the two legs,
tabletop to glass top; collision only, nothing is rendered). The bin stands on that side, beyond the panel. The other
three sides are open.

Why the prior fails: the stick lies on one side with its head pointing at the low table, and the bin is on the
opposite side, so the obvious plan is "slide the stick in and push the cube straight across, out towards the bin".
The cube then stops against the invisible panel, still completely under the glass top (the hand cannot reach it
there: the glass is 7 cm above the tabletop). Pushing harder or again from the same side fails the same way.

What must be learned from the failure (F1: from the images -- the cube stopped at the edge and did not come out):
something blocks that side -> change the push: put the stick down behind the low table (the robot's side) pointing
forward, push the cube out of the FRONT side (away from the robot), then grasp it and carry it to the bin.

Layout (randomised per instance): position of the low table, cube position under it (+-1 cm around the centre),
the stick lying beside the low table (left or right) with its head pointing at it (+-10 deg), the panel and the bin
on the other side.

Protocol: cross (reset to the same instance before every attempt), k = 5. v1 is kept in l3_tool.py.bak-20261008-pane.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, bin_xml, box_xml, register_generated
from ..skills import DT, _rot_err
from ..task_base import LFFTask, register_task

CUBE_HALF = 0.02
TOP_HALF = (0.06, 0.06, 0.005)  # glass top 12 x 12 cm, 1 cm thick
TOP_UNDER = 0.07  # underside of the glass top above the tabletop
LEG_HALF = 0.008
HANDLE_HALF = (0.12, 0.008, 0.01)  # stick handle 24 x 1.6 x 2 cm, along local x
HEAD_HALF = (0.008, 0.025, 0.01)  # crossbar 1.6 x 5 x 2 cm at the +x end
BIN_IN = (0.06, 0.06)
BIN_WALL_H = 0.05
GRIP_END = 0.035  # scripted grasp point: this far from the free end of the handle
GRIP_FRONT = 0.07  # v2 front push: grasp this far from the free end (keeps the hand clear of the glass edge)
Z_STICK = envs.TABLE_Z + 0.0125  # site height when grasping / sliding the stick (pads 0.75-2.4 cm above the table)
TIP_BELOW_SITE = 0.0095
# clear acrylic panel between the two legs of one y side, tabletop to glass underside (collision only, invisible)
PANE_HALF = (TOP_HALF[0] - 2 * LEG_HALF, 0.002, TOP_UNDER / 2)
PANE_Y = TOP_HALF[1] - LEG_HALF  # in the plane of the leg centres
STICK_COM = 0.0221  # the stick's centre of mass is this far from the handle centre, towards the head
BIN_GAP = (0.04, 0.06)  # tabletop gap between the low table's far side and the bin's outer wall


def low_table_xml(model_name):
    glass, leg = (0.70, 0.85, 0.95, 0.30), (0.35, 0.25, 0.18, 1.0)
    z_top = TOP_UNDER + TOP_HALF[2]
    parts = [((0, 0, z_top), TOP_HALF, glass)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(((sx * (TOP_HALF[0] - LEG_HALF), sy * (TOP_HALF[1] - LEG_HALF), TOP_UNDER / 2),
                          (LEG_HALF, LEG_HALF, TOP_UNDER / 2), leg))
    g = []
    for k, (p, s, c) in enumerate(parts):
        g.append(f'        <geom name="{model_name}_g{k}" type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" '
                 f'density="800" friction="1.0 0.005 0.0001" {_SOLID} group="0" />')
        g.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" conaffinity="0" '
                 f'contype="0" group="1" />')
    # acrylic panels on both y sides; apply_instance turns the collision of the one next to the stick off.
    # Fully transparent (rgba alpha 0) and no visual twin: nothing is rendered.
    for k, sy in enumerate((1, -1)):
        g.append(f'        <geom name="{model_name}_pane{k}" type="box" pos="{_fmt((0, sy * PANE_Y, PANE_HALF[2]))}" '
                 f'size="{_fmt(PANE_HALF)}" rgba="1 1 1 0" density="800" friction="0.6 0.005 0.0001" {_SOLID} '
                 f'group="0" />')
    # origin at the tabletop (z = 0 is the bottom of the legs)
    return _wrap(model_name, "\n".join(g), 0.0, float(np.hypot(TOP_HALF[0], TOP_HALF[1])))


def stick_xml(model_name):
    c = (0.80, 0.62, 0.30, 1.0)
    parts = [((0, 0, 0), HANDLE_HALF), ((HANDLE_HALF[0] + HEAD_HALF[0], 0, 0), HEAD_HALF)]
    g = []
    for k, (p, s) in enumerate(parts):
        g.append(f'        <geom name="{model_name}_g{k}" type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" '
                 f'density="600" friction="1.0 0.005 0.0001" {_SOLID} group="0" />')
        g.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" conaffinity="0" '
                 f'contype="0" group="1" />')
    return _wrap(model_name, "\n".join(g), HANDLE_HALF[2], HANDLE_HALF[0] + 2 * HEAD_HALF[0])


def _yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@register_task
class UnderTableTool(LFFTask):
    name = "l3_under_table_tool"
    level = "L3"
    category = "strategy_switching"
    capabilities = ("Perceive", "Reason", "Plan")
    protocol = "cross"
    instruction = "Put the red cube into the gray bin."
    instruction_indirect = "Put the small red block that is under the low glass table into the gray container."

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        self._mon = None
        self.inst = None
        orig_step = self.env.step

        def step(action):
            ret = orig_step(action)
            if self._mon is not None:
                self._monitor_step()
            return ret

        self.env.step = step

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        cube = register_generated("LffBenchCube", box_xml("lff_bench_cube", (CUBE_HALF,) * 3, (0.85, 0.12, 0.12, 1.0),
                                                          density=800))
        stick = register_generated("LffBenchStick", stick_xml("lff_bench_stick"))
        table = register_generated("LffLowGlassTable", low_table_xml("lff_low_glass_table"), free=False)
        tray = register_generated("LffBenchBin", bin_xml("lff_bench_bin", BIN_IN, BIN_WALL_H), free=False)
        objs = [("cube_1", cube, "cube_region", (-0.21, -0.26, -0.19, -0.24)),
                ("stick_1", stick, "stick_region", (-0.01, 0.24, 0.01, 0.26))]
        fx = [("lowtable_1", table, "lowtable_region", (0.19, -0.01, 0.21, 0.01)),
              ("bin_1", tray, "bin_region", (0.19, 0.29, 0.21, 0.31))]
        return write_bddl(self.name, self.instruction, objs, fx)

    def _fixture(self, name):
        if not hasattr(self, "_fx"):
            self._fx = {}
        if name not in self._fx:
            m = self.env.sim.model
            bid = m.body_name2id(self.env.fixtures_dict[name].root_body)

            def under(b):
                while b > 0:
                    if b == bid:
                        return True
                    b = m.body_parentid[b]
                return False

            self._fx[name] = (bid, [g for g in range(m.ngeom) if under(m.geom_bodyid[g])])
        return self._fx[name]

    def _panes(self):
        """[(geom id, +1 / -1 = which y side of the low table)] of the two acrylic panel geoms"""
        if not hasattr(self, "_pane_ids"):
            m = self.env.sim.model
            self._pane_ids = [(g, int(np.sign(m.geom_pos[g][1]))) for g in self._fixture("lowtable_1")[1]
                              if np.allclose(m.geom_size[g], PANE_HALF, atol=1e-6)]
            assert len(self._pane_ids) == 2, self._pane_ids
        return self._pane_ids

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        side = int(rng.choice([-1, 1]))  # the stick lies on the left (+1) or right (-1) of the low table
        # v2: 4 cm further from the robot than v1 (-0.16..-0.10), so that a stick pushed forward from behind the
        # low table is within reach (gripper about 25 cm in front of the robot base at the start of that push)
        t_xy = np.array([rng.uniform(-0.12, -0.06), -side * rng.uniform(0.04, 0.07)])
        cube_xy = t_xy + rng.uniform(-0.01, 0.01, 2)
        # the stick lies beside the low table, its head pointing roughly at it (+-10 deg), 2.5-4.5 cm from its edge
        yaw = -side * np.pi / 2 + float(rng.uniform(-np.radians(10), np.radians(10)))
        d = np.array([np.cos(yaw), np.sin(yaw)])
        head_front = np.array([cube_xy[0] + rng.uniform(-0.01, 0.01), t_xy[1] + side * (TOP_HALF[1] + rng.uniform(0.025, 0.045))])
        stick_xy = head_front - d * (HANDLE_HALF[0] + 2 * HEAD_HALF[0])
        # v2: the bin stands on the far side (the panel's side), so pushing the cube straight across looks like the
        # way to the bin
        gap = rng.uniform(*BIN_GAP)
        bin_xy = np.array([t_xy[0] + rng.uniform(-0.02, 0.02),
                           t_xy[1] - side * (TOP_HALF[1] + gap + BIN_IN[1] + 0.006)])
        return dict(seed=int(seed), table_xy=t_xy.tolist(), cube_xy=cube_xy.tolist(), side=side, pane_side=-side,
                    stick_xy=stick_xy.tolist(), stick_yaw=float(yaw), bin_xy=bin_xy.tolist())

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        bid, _ = self._fixture("lowtable_1")
        m.body_pos[bid] = [inst["table_xy"][0], inst["table_xy"][1], envs.TABLE_Z]
        m.body_quat[bid] = [1, 0, 0, 0]
        for g, sy in self._panes():  # only the panel on the far side (away from the stick) collides
            on = sy == inst.get("pane_side", -inst["side"])
            m.geom_contype[g] = 1 if on else 0
            m.geom_conaffinity[g] = 1 if on else 0
        bbid, _ = self._fixture("bin_1")
        m.body_pos[bbid] = [inst["bin_xy"][0], inst["bin_xy"][1], envs.TABLE_Z + (BIN_WALL_H + 0.006) / 2]
        m.body_quat[bbid] = [1, 0, 0, 0]
        cx, cy = inst["cube_xy"]
        envs.set_obj_pose(self.env, "cube_1", [cx, cy, envs.TABLE_Z + CUBE_HALF + 0.0005])
        sx, sy = inst["stick_xy"]
        envs.set_obj_pose(self.env, "stick_1", [sx, sy, envs.TABLE_Z + HANDLE_HALF[2] + 0.0005], yaw=inst["stick_yaw"])

    def reset_instance(self, inst, recorder=None):
        self._mon = None
        sk = super().reset_instance(inst, recorder=recorder)
        self._start_monitor()
        return sk

    # ------------------------------------------------------------ state
    def stick_pose(self):
        d = self.env.sim.data
        bid = envs.body_id(self.env, "stick_1")
        return d.body_xpos[bid].copy(), d.body_xmat[bid].reshape(3, 3).copy()

    def observe(self):
        p, R = self.stick_pose()
        ax = R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-9)
        return dict(cube=envs.obj_pos(self.env, "cube_1"), table_xy=np.array(self.inst["table_xy"]),
                    bin_xy=np.array(self.inst["bin_xy"]), stick_xy=p[:2], stick_dir=ax,
                    free_end=p[:2] - ax * HANDLE_HALF[0], head=p[:2] + ax * (HANDLE_HALF[0] + HEAD_HALF[0]))

    def under_table(self, xy=None, margin=0.0):
        """is the cube's footprint (any part) under the glass top?"""
        xy = envs.obj_pos(self.env, "cube_1")[:2] if xy is None else np.asarray(xy)
        rel = np.abs(xy - np.array(self.inst["table_xy"]))
        return bool((rel < np.array(TOP_HALF[:2]) + CUBE_HALF + margin).all())

    def outside_dist(self):
        """how far the cube's footprint is outside the glass top (m, < 0: still partly under it)"""
        rel = np.abs(envs.obj_pos(self.env, "cube_1")[:2] - np.array(self.inst["table_xy"]))
        return float(max(rel - np.array(TOP_HALF[:2]) - CUBE_HALF))

    def in_bin(self):
        p = envs.obj_pos(self.env, "cube_1")
        rel = np.abs(p[:2] - np.array(self.inst["bin_xy"]))
        return bool(p[2] < envs.TABLE_Z + BIN_WALL_H and (rel < np.array(BIN_IN)).all())

    # ------------------------------------------------------------ contacts / monitor
    def _groups(self):
        if not hasattr(self, "_gg"):
            m = self.env.sim.model
            robot = set(envs.robot_geom_ids(self.env))
            tbl = self._fixture("lowtable_1")[1]
            top = {g for g in tbl if m.geom_contype[g] and abs(m.geom_size[g, 0] - TOP_HALF[0]) < 1e-6}
            self._gg = (robot, set(tbl), top, set(envs.obj_geom_ids(self.env, "stick_1")),
                        set(envs.obj_geom_ids(self.env, "cube_1")))
        return self._gg

    def _robot_contacts(self):
        """labels of what the robot touches: 'glass_top', 'pane', 'table_leg', 'stick', 'cube'"""
        robot, tbl, top, stick, cube = self._groups()
        panes = {g for g, _ in self._panes()}
        d = self.env.sim.data
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in robot:
                o = c.geom2
            elif c.geom2 in robot:
                o = c.geom1
            else:
                continue
            if o in top:
                out.add("glass_top")
            elif o in panes:
                out.add("pane")
            elif o in tbl:
                out.add("table_leg")
            elif o in stick:
                out.add("stick")
            elif o in cube:
                out.add("cube")
        return out

    def _start_monitor(self):
        c = envs.obj_pos(self.env, "cube_1")
        self._mon = dict(start=c.copy(), top_hit=None, stick_held=False, stick_touched_cube=False, max_cube_dz=0.0,
                         cube_out=not self.under_table(), cube_hit_pane=False, stick_hit_pane=False,
                         robot_hit_pane=False)

    def _monitor_step(self):
        mon, sk = self._mon, self.sk
        if sk is None:
            return
        hits = self._robot_contacts()
        if "glass_top" in hits:
            e = sk.eef_pos()
            rel = np.abs(e[:2] - np.array(self.inst["table_xy"]))
            mon["top_hit"] = dict(above=bool((rel < np.array(TOP_HALF[:2])).all()),
                                  tip=float(e[2] - TIP_BELOW_SITE - envs.TABLE_Z),
                                  outside=float(max(rel - np.array(TOP_HALF[:2]))))
        if sk.holding("stick_1"):
            mon["stick_held"] = True
        if "pane" in hits:
            mon["robot_hit_pane"] = True
        robot, tbl, top, stick, cube = self._groups()
        panes = {g for g, _ in self._panes()}
        d = self.env.sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            pair = (c.geom1, c.geom2)
            if (pair[0] in stick and pair[1] in cube) or (pair[1] in stick and pair[0] in cube):
                mon["stick_touched_cube"] = True
            if (pair[0] in panes and pair[1] in cube) or (pair[1] in panes and pair[0] in cube):
                mon["cube_hit_pane"] = True
            if (pair[0] in panes and pair[1] in stick) or (pair[1] in panes and pair[0] in stick):
                mon["stick_hit_pane"] = True
        p = envs.obj_pos(self.env, "cube_1")
        mon["max_cube_dz"] = max(mon["max_cube_dz"], float(p[2] - mon["start"][2]))
        if not self.under_table():
            mon["cube_out"] = True

    # ------------------------------------------------------------ outcome / F2
    def outcome(self, inst):
        envs.settle(self.env, 30)
        mon = self._mon or {}
        p = envs.obj_pos(self.env, "cube_1")
        held = bool(self.sk.holding("cube_1")) if self.sk is not None else False
        success = bool(self.in_bin() and not held)
        moved = float(np.linalg.norm(p[:2] - mon.get("start", p)[:2]))
        under = self.under_table()
        brel = p[:2] - np.array(inst["bin_xy"])
        parts, mode = [], "success"
        top = mon.get("top_hit")
        if success:
            parts.append("The cube is in the bin.")
        elif p[2] < envs.TABLE_Z - 0.05:
            mode = "fell_off"
            parts.append("The cube fell off the table.")
        elif held:
            mode = "held"
            parts.append("The cube is still held by the gripper; it was not released into the bin.")
        elif under:
            mode = "under"
            if top is not None and not mon.get("stick_touched_cube"):
                mode = "blocked"
                if top["above"]:
                    parts.append(f"The gripper came down on the glass top of the low table: its fingertips stopped "
                                 f"{100 * max(top['tip'], 0.0):.1f} cm above the tabletop, while the cube is underneath "
                                 f"the glass (the glass underside is {100 * TOP_UNDER:.1f} cm above the tabletop), so the "
                                 f"gripper never got to the cube.")
                else:
                    parts.append(f"The hand ran into the edge of the glass top of the low table (fingertips "
                                 f"{100 * max(top['tip'], 0.0):.1f} cm above the tabletop, gripper "
                                 f"{100 * max(top['outside'], 0.0):.1f} cm outside the edge) and could not move in "
                                 f"under the glass.")
            if mon.get("cube_hit_pane"):
                mode = "pane"
                parts.append(f"The cube was pushed against a clear acrylic panel that closes the "
                             f"{'left' if inst.get('pane_side', -inst['side']) > 0 else 'right'} side of the low "
                             f"table (between its two legs, from the tabletop up to the glass); it cannot leave the "
                             f"low table on that side.")
            if moved > 0.01:
                parts.append(f"The cube was moved {100 * moved:.1f} cm but is still under the low table "
                             f"({100 * -self.outside_dist():.1f} cm from getting completely out from under it).")
            else:
                parts.append("The cube did not move; it is still under the low table.")
        else:
            mode = "out_not_in_bin"
            lifted = mon.get("max_cube_dz", 0.0) > 0.03
            parts.append(("The cube was lifted but ended outside the bin" if lifted else
                          "The cube is out from under the low table but not in the bin") +
                         f": it is {100 * abs(brel[0]):.1f} cm {'forward' if brel[0] > 0 else 'backward'} and "
                         f"{100 * abs(brel[1]):.1f} cm {'left' if brel[1] > 0 else 'right'} of the bin's centre.")
        out = dict(success=success, mode=mode, moved=moved, under=under, cube_rel_bin=[float(brel[0]), float(brel[1])],
                   stick_used=bool(mon.get("stick_held")), terminal=bool(p[2] < envs.TABLE_Z - 0.05),
                   cube_hit_pane=bool(mon.get("cube_hit_pane")), stick_hit_pane=bool(mon.get("stick_hit_pane")),
                   robot_hit_pane=bool(mon.get("robot_hit_pane")), detail=" ".join(parts))
        self._start_monitor()
        return out

    # ------------------------------------------------------------ primitives for the scripted policies
    def _yaw_near(self, yaw):
        cands = [yaw + k * np.pi for k in (-2, -1, 0, 1, 2)]
        cands = [c for c in cands if abs(c) <= np.pi / 2 + 0.4] or cands
        return float(min(cands, key=lambda c: abs(c - self.sk.yaw)))

    def _align(self, tol=0.03, max_steps=60):
        sk = self.sk
        for _ in range(max_steps):
            R = sk._R() @ sk.eef_mat().T
            if np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) < tol:
                break
            sk.hold(1)

    def _guarded_descent(self, xyz, speed=0.12, max_steps=80):
        sk = self.sk
        p0 = sk.eef_pos()
        tgt = np.asarray(xyz, float)
        dist = max(np.linalg.norm(tgt - p0), 1e-6)
        R = sk._R()
        for k in range(max_steps):
            ref = p0 + (tgt - p0) * min(1.0, (k + 1) * speed * DT / dist)
            sk._act(ref - sk.eef_pos(), _rot_err(R, sk.eef_mat()), sk.grip)
            if self._robot_contacts() - {"cube"} and sk.eef_pos()[2] > tgt[2] + 0.005:
                break
            if np.linalg.norm(tgt - sk.eef_pos()) < 0.003:
                break
        sk.hold(4)

    def grasp_cube_and_bin(self, dxy=(0.0, 0.0), dyaw=0.0):
        """Top-down grasp of the cube where it is (+ offsets); if held, drop it into the bin."""
        sk, o = self.sk, self.observe()
        c = o["cube"]
        gx, gy = c[0] + dxy[0], c[1] + dxy[1]
        # the hand is 20 cm long along the closing direction: keep that direction parallel to the low table's edge
        v = c[:2] - o["table_xy"]
        yaw = (np.pi / 2 if abs(v[1]) > abs(v[0]) else 0.0) + dyaw
        sk.set_gripper(False, steps=8)
        sk.move_to([gx, gy, envs.TABLE_Z + 0.15], yaw=self._yaw_near(yaw))
        self._align()
        self._guarded_descent([gx, gy, envs.TABLE_Z + 0.017])
        sk.set_gripper(True, steps=15)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], envs.TABLE_Z + 0.16], speed=0.25)
        if sk.holding("cube_1"):
            bx, by = o["bin_xy"]
            sk.move_to([bx, by, envs.TABLE_Z + 0.16])
            sk.move_to([bx, by, envs.TABLE_Z + BIN_WALL_H + 0.05])
            sk.set_gripper(False, steps=12)
            sk.move_to([bx, by, envs.TABLE_Z + 0.16])
        else:
            sk.set_gripper(False, steps=6)

    def tool_push_out(self, extra=0.0):
        """Pick the stick up by the end of its handle (it already points at the low table), lift it a few mm and slide
        it along its own axis under the glass top so that its head pushes the cube out on the far side; put it down."""
        sk, o = self.sk, self.observe()
        ax = o["stick_dir"]
        g = o["free_end"] + ax * GRIP_END
        sk.set_gripper(False, steps=8)
        sk.move_to([g[0], g[1], Z_STICK + 0.08], yaw=self._yaw_near(float(np.arctan2(ax[1], ax[0]))))
        self._align()
        sk.move_to([g[0], g[1], Z_STICK], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([g[0], g[1], Z_STICK + 0.006], speed=0.05)
        o2 = self.observe()
        c, t_xy = o2["cube"][:2], o2["table_xy"]
        n = np.array([0.0, -float(self.inst["side"])])  # push straight across the low table (normal of its far edge)
        perp = np.array([-n[1], n[0]])
        head = o2["head"]
        # line the head up with the cube by sliding the stick sideways (no turning: a turned stick slips in the grasp)
        lat = float((c - head) @ perp)
        e = sk.eef_pos()
        sk.move_to([e[0] + perp[0] * lat, e[1] + perp[1] * lat, e[2]], speed=0.05)
        o3 = self.observe()
        head_front = float((o3["head"] + o3["stick_dir"] * HEAD_HALF[0]) @ n)
        gap = max(float(c @ n) - CUBE_HALF - head_front, 0.0)
        need = (TOP_HALF[1] + CUBE_HALF + 0.05 + extra) - float((c - t_xy) @ n)  # cube travel until it is clear
        e = sk.eef_pos()
        tgt = e[:2] + n * (gap + need)
        sk.move_to([tgt[0], tgt[1], e[2]], speed=0.08, tol=0.004, max_steps=500)
        sk.hold(3)
        for _ in range(3):  # still (partly) under the glass top: push on a little
            if not self.under_table() or not sk.holding("stick_1"):
                break
            e = sk.eef_pos()
            tgt = e[:2] + n * 0.025
            sk.move_to([tgt[0], tgt[1], e[2]], speed=0.06, tol=0.004, max_steps=100)
            sk.hold(3)
        ax = n
        back = sk.eef_pos()[:2] - ax * 0.05
        sk.move_to([back[0], back[1], e[2]], speed=0.08)
        sk.set_gripper(False, steps=10)
        sk.move_to([back[0], back[1], envs.TABLE_Z + 0.12], speed=0.2)

    def tool_push_front(self, extra=0.0):
        """v2 reference solution. Pick the stick up at its centre of mass, turn it to point forward (+x, away from the
        robot) in the air, put it down behind the low table in line with the cube, re-grasp it GRIP_FRONT from its
        free end and slide it forward under the glass top so that its head pushes the cube out of the FRONT side;
        pull the stick back a little and put it down.

        Reach: the re-grasp point is about 25 cm in front of the robot base at the start of the push; it stays
        >= 3 cm behind the glass top's back edge at the end (the hand is about 4-13 cm above the fingertips and
        would hit the glass edge)."""
        sk, o = self.sk, self.observe()
        ax = o["stick_dir"]
        com = o["stick_xy"] + ax * STICK_COM
        phi = float(np.arctan2(ax[1], ax[0]))  # the stick points this way (handle -> head); about +-90 deg
        sk.set_gripper(False, steps=8)
        # grasp with gripper yaw = phi (not phi +- pi), so that turning the stick to point forward is a wrist turn to
        # yaw 0, inside the wrist's range
        sk.move_to([com[0], com[1], Z_STICK + 0.08], yaw=phi, wrap_yaw=False)
        self._align()
        sk.move_to([com[0], com[1], Z_STICK], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([com[0], com[1], Z_STICK + 0.10], speed=0.15)
        c, t_xy = o["cube"][:2], o["table_xy"]
        head_front_x = t_xy[0] - TOP_HALF[0] - 0.006  # just outside the glass top's back edge
        put_x = head_front_x - (HANDLE_HALF[0] + 2 * HEAD_HALF[0]) + STICK_COM
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], e[2]], yaw=0.0, wrap_yaw=False)
        self._align()
        sk.move_to([put_x, c[1], Z_STICK + 0.10], speed=0.2)
        sk.move_to([put_x, c[1], Z_STICK + 0.004], speed=0.08, tol=0.003)
        sk.set_gripper(False, steps=10)
        sk.move_to([put_x, c[1], Z_STICK + 0.06], speed=0.15)
        # re-grasp near the free end
        o2 = self.observe()
        ax2 = o2["stick_dir"]
        g = o2["free_end"] + ax2 * GRIP_FRONT
        sk.move_to([g[0], g[1], Z_STICK + 0.06], yaw=self._yaw_near(float(np.arctan2(ax2[1], ax2[0]))))
        self._align()
        sk.move_to([g[0], g[1], Z_STICK], tol=0.003)
        sk.set_gripper(True, steps=15)
        sk.move_to([g[0], g[1], Z_STICK + 0.006], speed=0.05)
        # line the head up with the cube (sideways slide), then push forward
        o3 = self.observe()
        lat = float(c[1] - o3["head"][1])
        e = sk.eef_pos()
        sk.move_to([e[0], e[1] + lat, e[2]], speed=0.05)
        o4 = self.observe()
        head_front = float(o4["head"][0] + HEAD_HALF[0])
        cube_tgt = t_xy[0] + TOP_HALF[0] + CUBE_HALF + 0.03 + extra
        travel = (cube_tgt - CUBE_HALF) - head_front
        e = sk.eef_pos()
        sk.move_to([e[0] + travel, e[1], e[2]], speed=0.08, tol=0.004, max_steps=500)
        sk.hold(3)
        e = sk.eef_pos()
        sk.move_to([e[0] - 0.05, e[1], e[2]], speed=0.08)
        sk.set_gripper(False, steps=10)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], envs.TABLE_Z + 0.12], speed=0.2)

    def execute(self, inst, params):
        if params["mode"] == "tool":
            self.tool_push_out(params.get("extra", 0.0))
        elif params["mode"] == "front":
            self.tool_push_front(params.get("extra", 0.0))
        self.grasp_cube_and_bin(params.get("dxy", (0.0, 0.0)), params.get("dyaw", 0.0))

    # ------------------------------------------------------------ scripted references
    def default_params(self, inst):
        """v2 prior: what every cla2 Sonnet run did -- take the stick and push the cube straight across (towards
        the bin, which is now behind the hidden panel)"""
        return dict(mode="tool")

    def oracle_params(self, inst):
        return dict(mode="front")

    def blind_params(self, inst, attempt, rng):
        """no history: the prior with a random extra push length"""
        return dict(mode="tool", extra=float(abs(rng.normal(0, 0.03))))

    def adapt_params(self, inst, history):
        """v2: the cube stayed under the low table after a push across -> something blocks that side: push it out of
        the front instead. Hand grasp blocked by the glass -> use the stick (the v1 lesson). Front push that did not
        get it out -> push further."""
        out, prm = history[-1]["outcome"], history[-1]["params"]
        if prm["mode"] == "grasp":
            return dict(mode="tool") if out["mode"] in ("blocked", "under") else dict(prm)
        if prm["mode"] == "tool":
            return dict(mode="front") if out["mode"] in ("pane", "under", "blocked") else dict(prm)
        if out["mode"] in ("under", "blocked"):
            return dict(prm, extra=prm.get("extra", 0.0) + 0.02)
        return dict(prm)
