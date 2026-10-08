"""L7 hidden_mechanism: open a small cabinet door whose opening mechanism is hidden.

Category: failed_interaction_inference ("why did this contact not do what I expected?").

Scene: a small wooden cabinet (14 x 24 x 10 cm) whose door faces away from the robot (towards the front
camera), with a black knob post in the middle of the door. The robot reaches over the cabinet, grasps the
knob from above and pulls.

Hidden variable: the door's mechanism, one of four (uniform): swings on a hinge at its left edge (HL) or at
its right edge (HR), or slides to the left (SL) or to the right (SR). The door, the knob and the cabinet look
exactly the same for all four (hinge pins and slide rails are not modelled visually).

Rule (stated in the instruction and enforced): one pull per attempt -- grasp the knob, ONE straight move_to
while holding it, release. A "pull" is any move_to that starts with the knob held, moves the door, or pushes
against the door / knob sideways. Without this rule an agent could probe the door with small wiggles inside
one attempt and the task would no longer test what is learnt across attempts.

Why the prior fails (measured): the natural pull is straight out from the door (forward, +x), 12 cm.
  hinged:  the knob has to follow a circle around the hinge; the door swings open 31-36 deg and the knob is
           pulled out of the fingers. The knob ends ~6 cm forward and ~4.5 cm towards the hinge side.
  sliding: nothing moves; the fingers slide off the knob.
Success needs >= 60 deg (hinged) or >= 12 cm (sliding).

What must be learned: from how the knob / door moved (F2 gives the knob's displacement relative to where it
started, plus the images) -> which mechanism -> the pull direction that opens it:
  hinged on the left:  pull forward-left (~55 deg from forward), ~22 cm  -> 70-88 deg (measured)
  hinged on the right: pull forward-right                                  -> 70-88 deg
  sliding:             pull sideways ~18 cm (18 cm); after "nothing moved" the side is not observable,
                       so a learner needs one more attempt in the worst case (left fails -> right).
Straight pulls at 30 / 45 deg towards the hinge give 44 / ~62 deg; diagonal pulls on a slider give 2-4 cm.
"""
import mujoco
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import _SOLID, _fmt, register_generated
from ..skills import Skills
from ..task_base import LFFTask, register_task

CAB = "cab_1"
MODEL = "lff_l7_mech_cabinet"
# cabinet frame: origin at the centre of the footprint, z = 0 on the table; the open side faces +x
D, W, H, T = 0.14, 0.24, 0.10, 0.01  # depth (x), width (y), height, wall thickness
DW, DH, DT = 0.232, 0.09, 0.012  # door width, height, thickness
GAP = 0.002
KZ = 0.045  # knob stem height above the table
STEM_L, STEM_R = 0.032, 0.005
POST_R, POST_H = 0.009, 0.038
POST_TOP = KZ - 0.006 + POST_H  # 7.7 cm above the table
Z_GRIP = envs.TABLE_Z + 0.075  # grasp-point height for grasping the knob (verified across cabinet positions)
MECHS = ("HL", "HR", "SL", "SR")
HINGE_OK = np.radians(60.0)
SLIDE_OK = 0.12
SLIDE_RANGE = 0.22
PULL_SPEED = 0.10


def cabinet_xml(name, rgba=(0.45, 0.32, 0.22, 1.0), door_rgba=(0.80, 0.70, 0.55, 1.0), knob_rgba=(0.12, 0.12, 0.13, 1.0)):
    hx, hy = D / 2, W / 2
    shell = [((-hx + T / 2, 0, H / 2), (T / 2, hy, H / 2)), ((0, hy - T / 2, H / 2), (hx, T / 2, H / 2)),
             ((0, -hy + T / 2, H / 2), (hx, T / 2, H / 2)), ((0, 0, H - T / 2), (hx, hy, T / 2)),
             ((0, 0, T / 2), (hx, hy, T / 2))]
    g = []
    for k, (p, s) in enumerate(shell):
        g.append(f'<geom name="{name}_s{k}" type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(rgba)}" '
                 f'friction="0.6 0.005 0.0001" {_SOLID} group="0"/>')
        g.append(f'<geom type="box" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(rgba)}" contype="0" conaffinity="0" group="1"/>')
    dx = hx + GAP + DT / 2
    dz = 0.005 + DH / 2
    door = [f'<inertial pos="0 0 0" mass="0.35" diaginertia="0.0004 0.0016 0.0016"/>',
            # placeholder joint: type, axis, anchor and range are set per instance (apply_instance)
            f'<joint name="{name}_door_joint" type="hinge" axis="0 0 1" pos="0 0 0" limited="true" range="-0.01 0.01" '
            f'damping="0.2" frictionloss="0.02"/>',
            f'<geom name="{name}_panel" type="box" size="{_fmt((DT / 2, DW / 2, DH / 2))}" rgba="{_fmt(door_rgba)}" '
            f'friction="0.6 0.005 0.0001" {_SOLID} group="0"/>',
            f'<geom type="box" size="{_fmt((DT / 2, DW / 2, DH / 2))}" rgba="{_fmt(door_rgba)}" contype="0" conaffinity="0" group="1"/>']
    sz = KZ - dz
    stem_p = (DT / 2 + STEM_L / 2, 0, sz)
    post_p = (DT / 2 + STEM_L, 0, sz - 0.006 + POST_H / 2)
    for tag, typ, p, s in (("stem", "box", stem_p, (STEM_L / 2 + 0.002, STEM_R, STEM_R)),
                           ("post", "cylinder", post_p, (POST_R, POST_H / 2))):
        door.append(f'<geom name="{name}_{tag}" type="{typ}" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(knob_rgba)}" '
                    f'friction="1.0 0.005 0.0001" {_SOLID} group="0"/>')
        door.append(f'<geom type="{typ}" pos="{_fmt(p)}" size="{_fmt(s)}" rgba="{_fmt(knob_rgba)}" contype="0" conaffinity="0" group="1"/>')
    body = ("\n        ".join(g) + f'\n        <body name="door" pos="{_fmt((dx, 0, dz))}">\n          '
            + "\n          ".join(door) + "\n        </body>")
    return f"""<mujoco model="{name}">
  <worldbody>
    <body>
      <body name="object">
        {body}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {H:.4f}" name="top_site" />
      <site rgba="0 0 0 0" size="0.005" pos="{D / 2:.4f} {W / 2:.4f} 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>"""


def _lr(v):
    return f"{100 * abs(v):.1f} cm to the {'left' if v > 0 else 'right'}"


def _fb(v):
    return f"{100 * abs(v):.1f} cm {'forward' if v > 0 else 'backward'}"


class MechSkills(Skills):
    """Skills that log, for whoever drives them (scripted policy or agent): door pulls, the first grasp of the
    knob, and how the knob moved during the first pull."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task
        m = env.sim.model
        self._grip_geoms = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(("gripper0_", "robot0_"))]
        self._door_geoms = task.door_geom_ids
        self.knob0 = task.knob_xyz()
        self.pulls = []  # one entry per move_to that counts as a pull
        self.closes = []  # close_gripper events: gripper centre relative to the knob, held afterwards?
        self._touch = False
        self.touch_since_report = False

    def holding_knob(self):
        return bool(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.task.knob_geom_names))

    def _touching_door(self):
        d = self.env.sim.data
        g = set(self._grip_geoms)
        dg = set(self._door_geoms)
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in g and c.geom2 in dg) or (c.geom2 in g and c.geom1 in dg):
                return True
        return False

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        if self._touching_door():
            self._touch = True  # per move_to (pull detection)
            self.touch_since_report = True  # per agent command (report field)

    def move_to(self, target, *args, **kwargs):
        q0 = self.task.door_q()
        hold0 = self.holding_knob()
        p0 = self.eef_pos()
        k0 = self.task.knob_xyz()
        self._touch = False
        rep = super().move_to(target, *args, **kwargs)
        dq = abs(self.task.door_q() - q0)
        moved = dq > (np.radians(1.0) if self.task.hinged else 0.003)
        horiz = float(np.linalg.norm(np.asarray(target, float)[:2] - p0[:2]))
        if hold0 or moved or (self._touch and horiz >= 0.015):
            k1 = self.task.knob_xyz()
            self.pulls.append(dict(step=self.n_steps, held_at_start=bool(hold0), held_at_end=self.holding_knob(),
                                   knob_start=k0.tolist(), knob_end=k1.tolist(), door_moved=bool(moved),
                                   gripper_move=(np.asarray(target, float)[:2] - p0[:2]).round(4).tolist()))
        return rep

    def set_gripper(self, close, steps=15):
        was_open = self.grip < 0
        rep = super().set_gripper(close, steps)
        if close and was_open:
            off = self.eef_pos() - self.task.knob_xyz()
            self.closes.append(dict(step=self.n_steps, offset=off.round(4).tolist(), held=self.holding_knob(),
                                    width=round(self.gripper_width(), 4)))
        return rep


@register_task
class HiddenMechanism(LFFTask):
    name = "l7_hidden_mechanism"
    level = "L7"
    category = "failed_interaction_inference"
    capabilities = ("Perceive", "Reason", "Plan")
    protocol = "cross"
    instruction = ("Open the door of the small wooden cabinet wide (a door that swings must end at least 60 degrees "
                   "open, a door that slides at least 12 cm). The door faces away from you; grasp its black knob from "
                   "above. Only one pull is allowed per attempt: grasp the knob, move the gripper with ONE straight "
                   "move command while holding it, then release it. Pushing the door or the knob counts as a pull "
                   "too.")
    instruction_indirect = ("Open the front panel of the small brown box wide (a panel that swings must end at least 60 "
                            "degrees open, a panel that slides at least 12 cm). The panel faces away from you; grasp the "
                            "small black handle on it from above. Only one pull is allowed per attempt: grasp the handle, "
                            "move the gripper with ONE straight move command while holding it, then release it. "
                            "Pushing the panel or the handle counts as a pull too.")

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        cab = register_generated("LffL7MechCabinet", cabinet_xml(MODEL), free=False)
        return write_bddl(self.name, "open the cabinet door", [],
                          fixtures=[(CAB, cab, "cab_region", (-0.14, -0.01, -0.12, 0.01))])

    def __init__(self, cam_res=None):
        super().__init__(cam_res)
        m = self.env.sim.model
        pre = f"{CAB}_{MODEL}"
        self.jid = m.joint_name2id(pre + "_door_joint")
        self.dof = m.jnt_dofadr[self.jid]
        self.qa = m.jnt_qposadr[self.jid]
        self.cab_bid = m.body_name2id(self.env.fixtures_dict[CAB].root_body)
        self.post_gid = m.geom_name2id(pre + "_post")
        self.knob_geom_names = [pre + "_post", pre + "_stem"]
        self.door_geom_ids = [m.geom_name2id(pre + s) for s in ("_post", "_stem", "_panel")]
        self.hinged = True

    # agent-facing hooks -------------------------------------------------------
    def on_agent_tool(self, inst, name, args, report):
        """The cabinet is a fixture, so the harness's `holding_object` / `touched_object` (movable objects only) never
        see the knob or the door; add an explicit `holding_knob` field and count door contacts as touches."""
        if self.sk is None or not isinstance(report, dict):
            return
        touched = self.sk.touch_since_report
        self.sk.touch_since_report = False
        if report.get("ok") and name in ("move_to", "open_gripper", "close_gripper", "push"):
            report["holding_knob"] = self.sk.holding_knob()
            if touched and "touched_object" in report:
                report["touched_object"] = True

    # ------------------------------------------------------------ state helpers
    def door_q(self):
        return float(self.env.sim.data.qpos[self.qa])

    def knob_xyz(self):
        return self.env.sim.data.geom_xpos[self.post_gid].copy()

    def sample_instance(self, seed):
        rng = np.random.default_rng(seed)
        return dict(seed=int(seed), mech=MECHS[int(rng.integers(4))],
                    cab_xy=[float(rng.uniform(-0.15, -0.10)), float(rng.uniform(-0.07, 0.07))])

    def apply_instance(self, inst):
        m, d = self.env.sim.model, self.env.sim.data
        cx, cy = inst["cab_xy"]
        m.body_pos[self.cab_bid] = [cx, cy, envs.TABLE_Z]
        m.body_quat[self.cab_bid] = [1.0, 0.0, 0.0, 0.0]
        mech = inst["mech"]
        jid, dof = self.jid, self.dof
        self.hinged = mech in ("HL", "HR")
        if self.hinged:
            s = 1 if mech == "HL" else -1
            m.jnt_type[jid] = mujoco.mjtJoint.mjJNT_HINGE
            m.jnt_axis[jid] = [0, 0, 1]
            m.jnt_pos[jid] = [-DT / 2, s * DW / 2, 0]  # back face of the door, at its left / right edge
            m.jnt_range[jid] = [0.0, 1.9] if s > 0 else [-1.9, 0.0]
            m.dof_damping[dof] = 0.05
            m.dof_frictionloss[dof] = 0.03
        else:
            s = 1 if mech == "SL" else -1
            m.jnt_type[jid] = mujoco.mjtJoint.mjJNT_SLIDE
            m.jnt_axis[jid] = [0, s, 0]
            m.jnt_pos[jid] = [0, 0, 0]
            m.jnt_range[jid] = [0.0, SLIDE_RANGE]
            m.dof_damping[dof] = 2.0
            m.dof_frictionloss[dof] = 0.5
        # MuJoCo scales constraint softness (here: the joint limits) with dof_invweight0, computed once at compile
        # time for the placeholder hinge through the door's centre. Changing the joint type without updating it
        # made the slide limit ~200x too soft: a slider pulled the wrong way moved 17 cm past its stop (measured).
        door_mass, door_izz = 0.35, 0.0016
        m.dof_invweight0[dof] = (1.0 / (door_izz + door_mass * (DW / 2) ** 2)) if self.hinged else 1.0 / door_mass
        m.jnt_solref[jid] = [0.004, 1.0]  # stiff stops (default 0.02 let a hard pull push a slider 8 mm past its stop)
        d.qpos[self.qa] = 0.0
        d.qvel[dof] = 0.0

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.sk = MechSkills(self.env, self, recorder=recorder)
        return self.sk

    def opening(self):
        """Hinge: degrees open (>= 0); slide: cm open (>= 0)."""
        q = abs(self.door_q())
        return float(np.degrees(q)) if self.hinged else 100.0 * q

    # ------------------------------------------------------------ outcome
    def outcome(self, inst):
        envs.settle(self.env, 20)
        sk = self.sk
        q = abs(self.door_q())
        is_open = q >= (HINGE_OK if self.hinged else SLIDE_OK)
        n_pulls = len(sk.pulls)
        success = bool(is_open and n_pulls <= 1)
        kd = self.knob_xyz() - sk.knob0
        out = dict(success=success, door_open=bool(is_open), opening=round(self.opening(), 1), n_pulls=n_pulls,
                   knob_disp=kd.round(4).tolist(), pulls=sk.pulls, closes=sk.closes, first_pull_disp=None,
                   lost_hold=None)
        parts = []
        grasp_txt = ""
        if sk.closes:
            c = sk.closes[0]
            o = c["offset"]
            below = (envs.TABLE_Z + POST_TOP) - (o[2] + sk.knob0[2])  # knob top minus gripper centre height
            grasp_txt = (f"When the gripper first closed, its centre was {_fb(o[0])} and {_lr(o[1])} of the knob's "
                         f"centre and {100 * below:.1f} cm below the top of the knob; it "
                         + ("was holding the knob." if c["held"] else "did not get hold of the knob."))
        if success:
            out["failure"] = None
            parts.append("The door is open wide.")
        elif n_pulls == 0:
            out["failure"] = "no_pull"
            parts.append("The door did not open: the knob was never pulled.")
        else:
            out["failure"] = "several_pulls" if n_pulls > 1 else "not_open"
            if n_pulls > 1:
                parts.append(f"The door or knob was pulled or pushed in {n_pulls} separate moves; only one pull is "
                             f"allowed per attempt, so this attempt failed.")
            elif is_open:
                parts.append("The door is open wide.")
            else:
                parts.append("The door did not open wide enough.")
            p = sk.pulls[0]
            # net effect of the first pull once things came to rest: up to the start of the next pull, or now
            k_after = np.array(sk.pulls[1]["knob_start"]) if n_pulls > 1 else self.knob_xyz()
            disp = k_after - np.array(p["knob_start"])
            out["first_pull_disp"] = disp.round(4).tolist()
            out["lost_hold"] = bool(p["held_at_start"] and not p["held_at_end"])
            if np.linalg.norm(disp[:2]) < 0.003:
                txt = "During your (first) pull the knob did not move at all (less than 0.3 cm)"
            else:
                txt = f"During your (first) pull the knob moved {_fb(disp[0])} and {_lr(disp[1])}"
            if p["held_at_start"]:
                txt += ("; the gripper lost hold of the knob during the pull." if out["lost_hold"]
                        else "; the gripper still held the knob at the end of the pull.")
            else:
                txt += "; the gripper was not holding the knob when that move started."
            parts.append(txt)
        if grasp_txt and not success:
            parts.append(grasp_txt)
        out["detail"] = " ".join(parts)
        return out

    # ------------------------------------------------------------ scripted execution (cross protocol)
    def execute(self, inst, params):
        """Grasp the knob from above (yaw 0), one straight pull in direction `alpha` (deg from forward, + = left)
        over `dist` metres, release, lift."""
        sk = self.sk
        p = self.knob_xyz()  # perfect perception of the current scene
        sk.set_gripper(False, steps=8)
        sk.move_to([p[0], p[1], Z_GRIP + 0.08], yaw=0.0)
        sk.move_to([p[0], p[1], Z_GRIP], tol=0.003)
        sk.set_gripper(True, steps=15)
        a = np.radians(float(params["alpha"]))
        dist = float(params["dist"])
        sk.move_to([p[0] + dist * np.cos(a), p[1] + dist * np.sin(a), Z_GRIP], speed=PULL_SPEED, tol=0.004,
                   max_steps=300)
        sk.set_gripper(False, steps=8)
        e = sk.eef_pos()
        sk.move_to([e[0], e[1], Z_GRIP + 0.10])

    def default_params(self, inst):
        return dict(alpha=0.0, dist=0.12)

    ORACLE = dict(HL=(55.0, 0.22), HR=(-55.0, 0.22), SL=(90.0, 0.18), SR=(-90.0, 0.18))

    def oracle_params(self, inst):
        a, dist = self.ORACLE[inst["mech"]]
        return dict(alpha=a, dist=dist)

    def adapt_params(self, inst, history):
        """Scripted F2 learner using the knob's displacement during the (first) pull of each earlier attempt:
          moved forward and sideways  -> it swings on a hinge on that side -> pull ~55 deg towards that side
                                         (more if a hinge pull already fell short)
          moved only sideways          -> it slides that way -> pull sideways
          did not move                 -> not a hinge; it must slide: try left, then right."""
        hinge_side, slide_side = 0, 0
        tried_still = set()  # pull directions (sign of alpha, only for sideways pulls) after which nothing moved
        hinge_tries = 0
        for h in history:
            o = h["outcome"]
            d = o.get("first_pull_disp")
            a = h["params"]["alpha"]
            if d is None:
                continue
            dx, dy = d[0], d[1]
            if dx >= 0.01 and abs(dy) >= 0.01:
                hinge_side = int(np.sign(dy))
                if abs(a) >= 50:
                    hinge_tries += 1
            elif abs(dy) >= 0.01:
                slide_side = int(np.sign(dy))
            elif abs(a) >= 80:
                tried_still.add(int(np.sign(a)))
        if hinge_side:
            a = 55.0 + 5.0 * min(hinge_tries, 2)
            return dict(alpha=hinge_side * a, dist=0.22 + 0.02 * min(hinge_tries, 2))
        if slide_side:
            return dict(alpha=90.0 * slide_side, dist=0.18)
        side = 1 if 1 not in tried_still else -1
        return dict(alpha=90.0 * side, dist=0.18)

    def blind_params(self, inst, attempt, rng):
        """Same plan (pull straight out) with small random variation; no history."""
        return dict(alpha=float(rng.normal(0.0, 8.0)), dist=float(np.clip(rng.normal(0.12, 0.02), 0.06, 0.2)))
