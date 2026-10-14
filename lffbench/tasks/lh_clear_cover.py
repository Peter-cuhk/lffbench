"""LH clear_cover: sort four cubes into two open boxes by colour -- but in half of the instances one box is covered by a
loose clear acrylic lid. A cube carried over that box and released stops on the lid at rim height; it is not in the box.

Family: long-horizon self-disturbance / common sense ("state_restoration"; design P2-2 in
research_notes/P1P2_加深与常识任务设计.md): four pick-and-place stages that are each easy for a frontier model; the
trap is a commonsense one -- "a box can have a transparent lid" -- and the failure is only visible from the right
viewpoint: from the side (agentview) the cube visibly sits on top of the box, at rim height, with the box floor visible
below it; from straight above (the wrist camera) it lies inside the box's outline and looks almost the same as a cube
that is inside (design rule R5: a top-down view hides height differences).

Scene (world frame, robot base at x = -0.66; "forward" = +x, away from the robot; "left" = +y):
  * two open boxes (fixtures; inside 12 x 12 cm, walls 5 cm high and 8 mm thick, rim 5.6 cm above the table, light
    floor) with red and with blue walls, centres at x in [-0.14, -0.08], y = +-(0.12-0.14) (24-28 cm apart); which
    colour is on the left is random. The instruction names the boxes by colour, not "left / right" (the design draft's
    wording), so that no failure comes from whose left is meant;
  * four 4 cm cubes (38 g), two red and two blue, in a row in front of the boxes (x = -0.33, near the robot; 14 cm
    apart, shuffled, yaw +-23 deg);
  * the clear lid (covered instances only): a loose 13.4 x 13.4 cm, 8 mm clear plate lying on one box's walls,
    1 mm inside their outer faces (no overhang). v2 (2026-10-13, after the Sonnet pilot np_lh_clear_cover): rendered
    nearly invisible -- alpha 0.03, matte (no specular glint), no edge highlight (v1: alpha 0.10 + bright edges, and the
    pilot agent named "the transparent lid" from the first images); heavy (1 kg; mass is not visible) with friction 1.0
    and priority for all its contacts and medium-stiff contacts, so incidental contact by the hand does not knock it off
    (v1, 0.12 kg / mu 0.3 / 6 mm overhang / very stiff contacts: a probe descent between the boxes knocked it off in the
    pilot's very first motions). Transparent geoms do not write MuJoCo's depth buffer: in the depth maps
    (--perception rgbd) the lid is not there at all. In uncovered instances the lid is parked on the floor far outside
    the workspace and the cameras' view.

Hidden variable: whether a box is covered and which one. Exactly 10 of every 20 consecutive seeds are covered (5 red /
5 blue box): (seed * 7) mod 20 < 10, like lh_lid_box's balanced blocks.

Why the natural plan fails: carry each cube over its box and release it a few cm above the rim. Over the covered box the
cube lands on the lid and stays there (v2, measured: drops of 1-8 cm, 48/48 stay on the lid, the lid does not move;
pressing a held cube down towards the box floor stops the arm 4.2 cm above the target, the lid moves <= 0.8 mm, 12/12
still covering). After the release the cube visibly sits on top of the box in agentview (its bottom 0.8 cm above the
rim, the box floor visible below it); checking only from above it looks as if it were in the box (~1.3x bigger).
What must be learned: look from the side / compare heights before finishing, notice the cubes resting at rim height,
remove the lid (push its edge horizontally with the closed gripper, fingertips between the rim and the lid's top face --
what the pilot agent did when it saw the v1 lid) and put the cubes in (within the attempt), or check for a lid and
remove it before placing anything (across attempts). Pressing the fingertips onto the lid and dragging no longer moves
it (v2: hand friction = rim friction, by design, because a hand landing on the lid while moving sideways is exactly
what knocked the v1 lid off).
Measured, 2 covered instances x 86 Sonnet-like incidental motions (probe descents to 2-5 cm next to / between the
boxes, transits at 11-20 cm over the covered box, low lateral moves next to the boxes): the box got uncovered 1/172
(v1: 57/172), the lid moved > 2 cm 20/172 (v1: 97/172); a crash sweep through the boxes at 3 cm still knocks it (1/2).
Deliberate edge push, agent-style commands: fingertip target 2-10 mm below the lid's top face 10/10 (+ 8/8 in the
parameter sweep), at the top face 1/2, above it 0/2.

Protocol: cross (reset to the same instance before every attempt), k = 3. Outcome: final-state judge (every cube in the
cavity of the box of its colour -- centre inside the cavity, lowest point below the rim --, nothing held, the lid not on
a box, scene static) plus a per-step event log (cube stages achieved / lost / restored, rested_on_cover, left_cover,
cover_removed, cover_on_box, in_wrong_box) for the server log; F1 gives the agent only success / failure.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, _wrap, box_xml, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

Z = envs.TABLE_Z
TIP = envs.FINGERTIP_BELOW_SITE
# ---------------------------------------------------------------- geometry / physics
CUBE_E = 0.04
CUBE_DENS = 600.0  # -> 38 g
BOX_IN = 0.06  # inner half width (12 x 12 cm cavity)
WALL_T, FLOOR_T, WALL_H = 0.008, 0.006, 0.05
BOX_OUT = BOX_IN + WALL_T
RIM = FLOOR_T + WALL_H  # rim height above the table
WALL_MU = 0.3  # walls (cube contacts; the lid's own friction has priority in its contacts)
LID_T = 0.008  # thick enough that a horizontal push at 'lid height' finds its edge (v1 draft: 4 mm, see the card)
LID_HALF = BOX_OUT - 0.001  # v2: no overhang (1 mm inside the walls' outer faces): a hand landing on its edge cannot tip it
LID_MASS = 1.0  # v2: heavy (a thick glass-like plate; mass is not visible): incidental bumps move it much less
LID_MU = 1.0  # v2: with priority, used for ALL lid contacts (rim, hand, fingers, cubes): pressing on the lid and moving
#               sideways cannot drag it (hand friction == rim friction, and the rim also carries the lid's weight)
LID_SOLREF = (0.005, 1.0)  # v2: medium-stiff contacts (v1 used _SOLID: a finger that caught the lid's edge while the
LID_SOLIMP = (0.95, 0.99, 0.001)  # arm pressed on could squeeze it against the wall and launch it 16-37 cm)
LID_TOP = RIM + LID_T  # lid top surface above the table when it lies on a box
LID_RGBA = (0.85, 0.90, 0.95, 0.03)  # v2: near-invisible, no edge highlight, no specular (material below)
PARK = (0.0, 1.4)  # lid parked on the floor (world z = 0), out of reach and out of view (uncovered instances)

COLORS = ("red", "blue")
CUBE_RGBA = dict(red=(0.88, 0.10, 0.08, 1.0), blue=(0.10, 0.25, 0.88, 1.0))
WALL_RGBA = dict(red=(0.72, 0.26, 0.22, 1.0), blue=(0.22, 0.34, 0.72, 1.0))
FLOOR_RGBA = (0.86, 0.85, 0.82, 1.0)
CUBES = [f"{c}_cube_{i}" for c in COLORS for i in (1, 2)]
BOXES = {c: f"{c}_box_1" for c in COLORS}
LID = "clear_lid_1"

IN_MARGIN = 0.010  # a cube counts as in a box if its centre is within BOX_IN - IN_MARGIN of the box centre (each axis)
BELOW_RIM = 0.005  # ... and its lowest point is at least this far below the rim
Z_CARRY = 0.15  # scripted transit height of the grasp point above the table (Sonnet: 11-20 cm)
DROP = 0.04  # scripted release: cube bottom this far above the rim (Sonnet-like 3-5 cm)
SLOT = 0.028  # two cubes per box: +-2.8 cm from the centre along y
PUSH_DEPTH = 0.004  # scripted edge push: fingertips this far below the lid's top face (lid 8 mm thick)
PUSH_SPEED = 0.06


def _color(name):
    return name.split("_")[0]


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


def _lr(v):
    return f"{100 * abs(v):.1f} cm to the {'left' if v > 0 else 'right'}"


def box_xml_cc(name, wall_rgba):
    """open-top box, body origin at the centre of its footprint on the table (z = 0 = table top)"""
    o = BOX_OUT
    parts = [((0, 0, FLOOR_T / 2), (o, o, FLOOR_T / 2), FLOOR_RGBA, 1.0),
             ((BOX_IN + WALL_T / 2, 0, RIM / 2), (WALL_T / 2, o, RIM / 2), wall_rgba, WALL_MU),
             ((-BOX_IN - WALL_T / 2, 0, RIM / 2), (WALL_T / 2, o, RIM / 2), wall_rgba, WALL_MU),
             ((0, BOX_IN + WALL_T / 2, RIM / 2), (BOX_IN, WALL_T / 2, RIM / 2), wall_rgba, WALL_MU),
             ((0, -BOX_IN - WALL_T / 2, RIM / 2), (BOX_IN, WALL_T / 2, RIM / 2), wall_rgba, WALL_MU)]
    g = []
    for k, (p, s, c, mu) in enumerate(parts):
        g.append(f'        <geom name="{name}_g{k}" type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" '
                 f'friction="{mu} 0.005 0.0001" {_SOLID} group="0" />')
        g.append(f'        <geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(c)}" conaffinity="0" '
                 f'contype="0" group="1" />')
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
{chr(10).join(g)}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {RIM:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{o:.4f} {o:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>"""


def lid_xml(name):
    """clear plate: one collision box + a near-invisible visual twin with a matte material (no specular glint)"""
    h, t = LID_HALF, LID_T / 2
    g = [f'        <geom name="{name}_g0" type="box" size="{_fmt((h, h, t))}" rgba="1 1 1 0" mass="{LID_MASS}" '
         f'friction="{LID_MU} 0.005 0.0001" solref="{_fmt(LID_SOLREF)}" solimp="{_fmt(LID_SOLIMP)}" group="0" />',
         f'        <geom type="box" size="{_fmt((h, h, t))}" rgba="{_fmt(LID_RGBA)}" material="{name}_mat" '
         f'conaffinity="0" contype="0" group="1" />']
    xml = _wrap(name, "\n".join(g), t, float(np.hypot(h, h)))
    mat = (f'  <asset>\n    <material name="{name}_mat" rgba="{_fmt(LID_RGBA)}" specular="0" shininess="0" '
           f'reflectance="0" />\n  </asset>\n')
    return xml.replace("  <worldbody>", mat + "  <worldbody>", 1)


class CoverSkills(Skills):
    """Skills that feed the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class ClearCover(LFFTask):
    name = "lh_clear_cover"
    level = "L2"
    category = "state_restoration"
    capabilities = ("Perceive", "Reason", "Plan", "Utilize")
    protocol = "cross"
    max_attempts = 3
    # careful oracle: 4 pick-and-places (~7 commands each = 28) + pushing the lid off (~5) + looking around (~15);
    # one recovery: pushing the lid off + re-placing the cubes that were on it (~5 + 2 x 7) -> ~70; Sonnet looks after
    # every step and calibrates the wrist camera (cla4: 10+ observe / probe commands)
    agent_max_steps = 110
    instruction = "Put the two red cubes into the red box and the two blue cubes into the blue box."
    instruction_indirect = "Sort the four small cubes by colour: each cube goes into the box whose walls have its colour."
    agent_object_names = ["red cube", "red cube", "blue cube", "blue cube", "red box", "blue box"]

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        objs, fx = [], []
        for j, c in enumerate(COLORS):
            bx = register_generated(f"LffCc{c.title()}Box", box_xml_cc(f"lff_cc_{c}_box", WALL_RGBA[c]), free=False)
            fx.append((BOXES[c], bx, f"{c}_box_region", (-0.12, 0.12 - 0.26 * j, -0.10, 0.14 - 0.26 * j)))
        k = 0
        for c in COLORS:
            for i in (1, 2):
                ab = "AB"[i - 1]
                cat = register_generated(f"LffCc{c.title()}Cube{ab}",
                                         box_xml(f"lff_cc_{c}_cube_{ab.lower()}", (CUBE_E / 2,) * 3, CUBE_RGBA[c],
                                                 density=CUBE_DENS))
                objs.append((f"{c}_cube_{i}", cat, f"{c}_cube_{i}_region",
                             (-0.34, -0.22 + 0.14 * k, -0.32, -0.20 + 0.14 * k)))
                k += 1
        lid = register_generated("LffCc2ClearLid", lid_xml("lff_cc2_clear_lid"))
        objs.append((LID, lid, "lid_region", (0.20, -0.01, 0.22, 0.01)))
        return write_bddl(self.name, "put the cubes into the boxes of their colour", objs, fx)

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        self.box_bid = {c: m.body_name2id(self.env.fixtures_dict[BOXES[c]].root_body) for c in COLORS}
        self.lid_gids = set(envs.obj_geom_ids(self.env, LID))
        self.cube_gids = {n: set(envs.obj_geom_ids(self.env, n)) for n in CUBES}
        self.robot_gids = set(envs.robot_geom_ids(self.env))
        self.inst = None
        self.mon = None

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        bx = float(rng.uniform(-0.13, -0.09))
        left = str(rng.choice(COLORS))
        boxes = {}
        for c in COLORS:
            s = 1.0 if c == left else -1.0
            boxes[c] = [bx + float(rng.uniform(-0.01, 0.01)), s * float(rng.uniform(0.12, 0.14))]
        slots = [-0.21, -0.07, 0.07, 0.21]
        perm = rng.permutation(4)
        start = {CUBES[i]: [float(-0.33 + rng.uniform(-0.01, 0.01)), float(slots[j] + rng.uniform(-0.01, 0.01))]
                 for j, i in enumerate(perm)}
        # balanced like lh_lid_box: exactly 10 of every 20 consecutive seeds covered (5 red box, 5 blue box)
        r = (int(seed) * 7) % 20
        cover = "red" if r < 5 else ("blue" if r < 10 else None)
        return dict(seed=int(seed), boxes=boxes, left_box=left, start=start,
                    start_yaw={k: float(rng.uniform(-0.4, 0.4)) for k in CUBES}, cover=cover,
                    cover_side=(None if cover is None else ("left" if cover == left else "right")),
                    lid_off=[float(v) for v in rng.uniform(-0.004, 0.004, 2)], lid_yaw=float(rng.uniform(-0.05, 0.05)))

    def apply_instance(self, inst):
        self.inst = inst
        m = self.env.sim.model
        for c in COLORS:
            m.body_pos[self.box_bid[c]] = [inst["boxes"][c][0], inst["boxes"][c][1], Z]
            m.body_quat[self.box_bid[c]] = [1.0, 0.0, 0.0, 0.0]
        for name, xy in inst["start"].items():
            envs.place_upright(self.env, name, xy, yaw=inst["start_yaw"][name])
        if inst["cover"]:
            b = inst["boxes"][inst["cover"]]
            envs.set_obj_pose(self.env, LID, [b[0] + inst["lid_off"][0], b[1] + inst["lid_off"][1],
                                              Z + RIM + LID_T / 2 + 0.0005], yaw=inst["lid_yaw"])
        else:
            envs.set_obj_pose(self.env, LID, [PARK[0], PARK[1], LID_T / 2 + 0.002])
        envs.set_friction(self.env, LID, LID_MU, priority=2)

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.mon = None
        self.sk = CoverSkills(self.env, self, recorder=recorder)
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ state helpers
    def held(self, name):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.objects_dict[name]))

    def cube_rel(self, name, color):
        return envs.obj_pos(self.env, name)[:2] - np.asarray(self.inst["boxes"][color])

    def in_box(self, name, color, margin=IN_MARGIN):
        """centre inside the cavity (per axis), lowest point below the rim, not held"""
        rel = self.cube_rel(name, color)
        return bool(np.max(np.abs(rel)) <= BOX_IN - margin and envs.obj_min_z(self.env, name) < Z + RIM - BELOW_RIM
                    and not self.held(name))

    def where_box(self, name):
        for c in COLORS:
            if self.in_box(name, c):
                return c
        return None

    def lid_pos(self):
        return envs.obj_pos(self.env, LID)

    def covering(self):
        """colour of the box the lid lies on (centre within the box's outer footprint + 1 cm, above the floor), or None"""
        p = self.lid_pos()
        for c in COLORS:
            rel = np.abs(p[:2] - np.asarray(self.inst["boxes"][c]))
            if rel[0] <= BOX_OUT + 0.01 and rel[1] <= BOX_OUT + 0.01 and p[2] > Z + 0.02:
                return c
        return None

    def _touching(self, gids_a, gids_b):
        return envs.contacts_between(self.env, gids_a, gids_b)

    def on_cover(self, name, moving_ok=False):
        """the cube rests on the lid while the lid lies on a box (contact, not held, slow -- unless `moving_ok`: the
        monitor keeps a cube that came to rest on the lid "on" while it rides along with the lid being pushed)"""
        if self.covering() is None or self.held(name):
            return False
        if not self._touching(self.cube_gids[name], self.lid_gids):
            return False
        if moving_ok:
            return True
        _, da = envs.free_joint_addr(self.env, name)
        return bool(np.linalg.norm(self.env.sim.data.qvel[da:da + 3]) < 0.05)

    def stage_values(self):
        v = {f"{n}_in_box": self.in_box(n, _color(n)) for n in CUBES}
        v["cover_off_boxes"] = self.covering() is None
        return v

    # ------------------------------------------------------------ per-step monitor
    def _init_monitor(self):
        self.mon = dict(step=0, state={f"{n}_in_box": False for n in CUBES}, ever=set(), lost=set(), events=[],
                        on_cover={n: False for n in CUBES}, cover=self.covering(), wrong={n: None for n in CUBES},
                        last_touch={n: None for n in CUBES}, rested=[], cover_removed_step=None)

    def _touches(self):
        d = self.env.sim.data
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            for n, g in self.cube_gids.items():
                other = c.geom2 if c.geom1 in g else (c.geom1 if c.geom2 in g else None)
                if other is None:
                    continue
                if other in self.robot_gids:
                    out[n] = "robot"
                elif other in self.lid_gids and out.get(n) != "robot":
                    out[n] = "lid"
        return out

    def monitor_step(self):
        mon = self.mon
        if mon is None:
            return
        mon["step"] += 1
        step = self.sk.n_steps if self.sk is not None else mon["step"]
        for n, who in self._touches().items():
            if who == "robot":
                mon["last_touch"][n] = step
        if mon["step"] % 2:
            return  # predicates every other control step (0.1 s)
        cov = self.covering()
        if cov != mon["cover"]:
            if mon["cover"] is not None and cov is None:
                mon["events"].append(dict(step=step, pred="cover", ev="cover_removed", box=mon["cover"]))
                mon["cover_removed_step"] = step
            elif cov is not None:
                mon["events"].append(dict(step=step, pred="cover", ev="cover_on_box", box=cov))
            mon["cover"] = cov
        for n in CUBES:
            oc = self.on_cover(n, moving_ok=mon["on_cover"][n])
            if oc != mon["on_cover"][n]:
                mon["on_cover"][n] = oc
                if oc:
                    mon["rested"].append(dict(step=step, cube=n))
                mon["events"].append(dict(step=step, pred=f"{n}_on_cover", ev="rested_on_cover" if oc else "left_cover",
                                          cube=n, box=cov))
            key = f"{n}_in_box"
            was = mon["state"][key]
            now = self.in_box(n, _color(n), IN_MARGIN if not was else IN_MARGIN - 0.005)
            if now != was:
                self._event(key, now, step, n)
            other = [c for c in COLORS if c != _color(n)][0]
            wr = other if self.in_box(n, other) else None
            if wr != mon["wrong"][n]:
                mon["wrong"][n] = wr
                if wr:
                    mon["events"].append(dict(step=step, pred=key, ev="in_wrong_box", cube=n, box=wr))

    def _event(self, key, now, step, n):
        mon = self.mon
        mon["state"][key] = now
        if now:
            ev = "restored" if key in mon["lost"] else "achieved"
            mon["ever"].add(key)
            mon["lost"].discard(key)
            mon["events"].append(dict(step=step, pred=key, ev=ev))
            return
        if self.held(n):
            cause = "picked_up"
        else:
            lt = mon["last_touch"][n]
            cause = "robot" if lt is not None and step - lt <= 60 else "other"
        mon["lost"].add(key)
        mon["events"].append(dict(step=step, pred=key, ev="lost", cause=cause))

    # ------------------------------------------------------------ outcome
    def _static(self):
        d = self.env.sim.data
        for name in CUBES + [LID]:
            _, da = envs.free_joint_addr(self.env, name)
            if np.linalg.norm(d.qvel[da:da + 3]) > 0.02 or np.linalg.norm(d.qvel[da + 3:da + 6]) > 0.5:
                return False
        return True

    def outcome(self, inst):
        envs.settle(self.env, 30)
        mon = self.mon
        vals = self.stage_values()
        static = self._static()
        held = [n for n in CUBES if self.held(n)]
        on_cover = [n for n in CUBES if self.on_cover(n)]
        wrong = [n for n in CUBES if not vals[f"{n}_in_box"] and self.where_box(n) is not None]
        missing = [n for n in CUBES if not vals[f"{n}_in_box"]]
        cov = self.covering()
        success = bool(all(vals.values()) and static and not held)
        failure = None
        if not success:
            failure = ("held" if held else "cube_on_cover" if on_cover else "wrong_box" if wrong else
                       "cube_not_in_box" if missing else "cover_on_box" if cov is not None else "not_static")
        regress = [e for e in mon["events"] if e["ev"] == "lost"]
        restored = [e for e in mon["events"] if e["ev"] == "restored"]
        rested_cubes = sorted({r["cube"] for r in mon["rested"]})
        parts = []
        if success:
            parts.append("All four cubes are in the boxes of their colour.")
        for n in missing:
            c = _color(n)
            txt = f"A {c} cube is not in the {c} box: "
            if n in held:
                parts.append(txt + "it is still held by the gripper.")
                continue
            p = envs.obj_pos(self.env, n)
            rel = self.cube_rel(n, c)
            if n in on_cover:
                txt += (f"it rests on a clear acrylic lid that lies on the {cov} box, "
                        f"{100 * (envs.obj_min_z(self.env, n) - Z - FLOOR_T):.1f} cm above the box floor")
            elif self.where_box(n) is not None:
                txt += f"it is in the {self.where_box(n)} box"
            else:
                h = envs.obj_min_z(self.env, n) - Z
                txt += ("it lies on the table" if h < 0.005 else f"it lies {100 * h:.1f} cm above the table")
                txt += f", its centre {_fb(rel[0])} and {_lr(rel[1])} of the {c} box's centre"
            if n in rested_cubes and n not in on_cover:
                txt += " (it had come to rest on the clear lid earlier)"
            parts.append(txt + ".")
        if cov is not None:
            parts.append(f"A clear acrylic lid lies on the {cov} box.")
        if not static and all(vals.values()) and not held:
            parts.append("Something was still moving at the end.")
        if mon["cover_removed_step"] is not None and cov is None:
            parts.append("The clear lid was pushed off its box.")
        out = dict(success=success, failure=failure, stages=vals, static=bool(static), held=held,
                   events=mon["events"], n_regressions=len(regress), n_restored=len(restored),
                   covered=inst["cover"], trap_fired=bool(mon["rested"]), n_rested_on_cover=len(mon["rested"]),
                   rested_cubes=rested_cubes, cover_removed=mon["cover_removed_step"] is not None,
                   cover_removed_step=mon["cover_removed_step"], cover_final=cov, on_cover_final=on_cover,
                   recovered_in_attempt=bool(mon["rested"]) and success,
                   cube_offsets_cm={n: (100 * self.cube_rel(n, _color(n))).round(1).tolist() for n in CUBES},
                   detail=" ".join(parts))
        return out

    # ------------------------------------------------------------ scripted policies (perfect perception)
    def _grasp_yaw(self, name):
        R = self.env.sim.data.body_xmat[envs.body_id(self.env, name)].reshape(3, 3)
        i = int(np.argmin(np.abs(R[2])))  # the most horizontal body axis (works whichever face the cube rests on)
        yaw = float(np.arctan2(R[1, i], R[0, i]))
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def _finger_axis(self, yaw):
        """world xy direction along which the fingers close at gripper yaw `yaw` (Skills: yaw 0 = fingers along y)"""
        return np.array([-np.sin(yaw), np.cos(yaw)])

    def _clear_yaw(self, name, room=0.04):
        """grasp yaw of the cube: the smallest wrist turn (within +-45 deg) unless one of the open fingertips (4.5 cm
        either side) would come down within `room` of another cube's centre (cubes put side by side or pushed off on
        the lid stand 1.5-2.5 cm apart); then the 90-deg turned grasp if it has more room"""
        p = envs.obj_pos(self.env, name)[:2]
        others = [envs.obj_pos(self.env, n)[:2] for n in CUBES if n != name]

        def dmin(yaw):
            a = self._finger_axis(yaw)
            return min([np.linalg.norm(t - o) for t in (p + a * 0.045, p - a * 0.045) for o in others] or [1.0])

        y0 = self._grasp_yaw(name)
        if dmin(y0) >= room:
            return y0
        y1 = float((y0 + np.pi / 2 + np.pi / 2) % np.pi - np.pi / 2)
        return y1 if dmin(y1) > dmin(y0) else y0

    def pick(self, name, z_carry=Z_CARRY):
        sk = self.sk
        p = envs.obj_pos(self.env, name)
        sk.set_gripper(False, steps=8)
        sk.move_to([p[0], p[1], Z + max(z_carry, p[2] - Z + 0.06)], yaw=self._clear_yaw(name))
        sk.move_to([p[0], p[1], p[2]], tol=0.003, max_steps=120)
        sk.set_gripper(True, steps=15)
        sk.move_to([p[0], p[1], Z + z_carry], speed=0.3)
        if not self.held(name):
            sk.set_gripper(False, steps=8)
            return False
        return True

    def place_in(self, name, slot, drop=DROP, z_carry=Z_CARRY, noise=None):
        """carry the held cube over its box (slot -1 / +1: 2.8 cm right / left of the centre) and release it with its
        bottom `drop` above the rim"""
        sk = self.sk
        b = np.asarray(self.inst["boxes"][_color(name)], float)
        d = b + [0.0, SLOT * slot] + (np.asarray(noise, float) if noise is not None else 0.0)
        z_rel = Z + RIM + drop + CUBE_E / 2
        sk.move_to([d[0], d[1], Z + z_carry], yaw=0.0)
        sk.move_to([d[0], d[1], z_rel], tol=0.003, max_steps=120)
        sk.set_gripper(False, steps=12)
        sk.move_to([d[0], d[1], Z + z_carry])

    def pick_place(self, name, slot, drop=DROP, noise=None):
        if self.pick(name):
            self.place_in(name, slot, drop=drop, noise=noise)
            return True
        return False

    def remove_cover(self, direction=None, depth=PUSH_DEPTH, mode="edge"):
        """push the lid off its box: closed gripper, fingertips `depth` below the lid's top face, start 3 cm outside
        its near edge (mode "edge") or 3 cm inside it, pressing on the lid (mode "press": press and drag -- v2: does not
        move the lid any more, see the module docstring), sweep
        straight on until the lid has cleared the box (default: sideways, away from the other box), lift"""
        cov = self.covering()
        if cov is None:
            return False
        sk = self.sk
        lp = self.lid_pos()
        b = np.asarray(self.inst["boxes"][cov], float)
        if direction is None:
            o = np.asarray(self.inst["boxes"][[c for c in COLORS if c != cov][0]], float)
            direction = (0.0, float(np.sign(b[1] - o[1])))
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        back = LID_HALF + 0.03 if mode == "edge" else LID_HALF - 0.03
        s = lp[:2] - d * back
        top = lp[2] + LID_T / 2
        z = top - depth + TIP
        sk.set_gripper(True, steps=8)
        sk.move_to([s[0], s[1], z + 0.08], yaw=float(np.arctan2(d[1], d[0]) - np.pi / 2))
        sk.move_to([s[0], s[1], z], tol=0.003, max_steps=80)
        # sweep until the lid's near edge is 3 cm beyond the box's far wall
        need = float((b - lp[:2]) @ d) + BOX_OUT + back + 0.03
        e = s + d * need
        sk.move_to([e[0], e[1], z], speed=PUSH_SPEED, tol=0.006, max_steps=int(need / PUSH_SPEED / 0.05) + 80)
        sk.move_to([e[0], e[1], z + 0.10], speed=0.2)
        sk.set_gripper(False, steps=8)
        sk.hold(10)
        return self.covering() is None

    def slots(self, order):
        """slot of each cube: the first cube of a colour goes right of the centre, the second left"""
        seen, out = {c: 0 for c in COLORS}, {}
        for n in order:
            out[n] = (-1, 1)[seen[_color(n)] % 2]
            seen[_color(n)] += 1
        return out

    def wrist_fix(self, rounds=1):
        """what a top-down check can see: cubes whose centre is not over the right box's outline (on the table, in
        the other box, still held) get placed again; heights are not judged"""
        for _ in range(rounds):
            bad = []
            for n in CUBES:
                rel = self.cube_rel(n, _color(n))
                if self.held(n) or np.max(np.abs(rel)) > BOX_OUT:
                    bad.append(n)
            if not bad:
                return
            for n in bad:
                if self.held(n):
                    self.sk.set_gripper(False, steps=12)
                    continue
                self.pick_place(n, -1 if self.cube_rel(n, _color(n))[1] < 0 else 1)

    def set_aside(self, name):
        """pick the cube (e.g. off the lid) and put it down on the table where it started (that spot is free)"""
        if not self.pick(name):
            return False
        sk = self.sk
        x, y = self.inst["start"][name]
        sk.move_to([x, y, Z + Z_CARRY], yaw=0.0)
        sk.move_to([x, y, Z + CUBE_E / 2 + 0.004], tol=0.003, max_steps=120)
        sk.set_gripper(False, steps=12)
        sk.move_to([x, y, Z + Z_CARRY])
        return True

    def side_fix(self, rounds=2):
        """the careful check (agentview / heights): every cube must be inside its box, below the rim; if cubes rest on
        a lid (or a lid lies on a box): take the cubes off the lid (v2: the heavy lid cannot be pushed off with cubes
        standing on it -- the hand runs into them), push the lid off; then place every cube that is not in its box"""
        for _ in range(rounds):
            v = self.stage_values()
            if all(v.values()):
                return True
            if not v["cover_off_boxes"]:
                for n in CUBES:
                    if self.on_cover(n) or (self.covering() is not None and self._touching(self.cube_gids[n], self.lid_gids)):
                        self.set_aside(n)
                self.remove_cover()
            for n in CUBES:
                if not self.in_box(n, _color(n)):
                    if self.held(n):
                        self.sk.set_gripper(False, steps=12)
                        continue
                    others = [m for m in CUBES if m != n and _color(m) == _color(n)]
                    rel_o = self.cube_rel(others[0], _color(n)) if others else np.zeros(2)
                    self.pick_place(n, 1 if rel_o[1] < 0 else -1)
        return all(self.stage_values().values())

    def execute(self, inst, params):
        order = params.get("order") or list(CUBES)
        noise = params.get("noise") or {}
        drops = params.get("drops") or {}
        if params["plan"] == "lid_first":
            self.remove_cover(params.get("push_dir"), params.get("push_depth", PUSH_DEPTH), params.get("push_mode", "edge"))
        sl = self.slots(order)
        for n in order:
            self.pick_place(n, sl[n], drop=drops.get(n, DROP), noise=noise.get(n))
        if params.get("check") == "side":
            self.side_fix()
        elif params.get("check") == "wrist":
            self.wrist_fix()
        self.sk.hold(10)

    def default_params(self, inst):
        return dict(plan="naive", check="wrist")

    def oracle_params(self, inst):
        return dict(plan="lid_first", check="side")

    def adapt_params(self, inst, history):
        """from attempt 2 on: look for a lid and push it off first, then place, then check from the side"""
        return dict(plan="lid_first", check="side")

    def blind_params(self, inst, attempt, rng):
        """the natural plan again with small random variations (order, 5 mm placement noise, 3-5 cm drop), no history"""
        order = [CUBES[i] for i in rng.permutation(len(CUBES))]
        return dict(plan="naive", check="wrist", order=order,
                    noise={n: rng.uniform(-0.005, 0.005, 2).tolist() for n in CUBES},
                    drops={n: float(rng.uniform(0.03, 0.05)) for n in CUBES})

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history = []
        for a in range(k):
            self.reset_instance(inst)
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "adaptive":
                params = dict(plan="naive", check="side") if a == 0 else self.adapt_params(inst, history)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "switch":  # extra control (not in validate's table): cross-attempt learning only, no side check
                params = self.default_params(inst) if a == 0 else dict(plan="lid_first", check="wrist")
            else:
                raise ValueError(kind)
            self.execute(inst, params)
            out = self.outcome(inst)
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history
