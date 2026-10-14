"""L5 loaded_beam (2026-10-12): a deepened l5_slippery_lift -- lift a loaded bar level without spilling its load.

Why (../research_notes/P1P2_加深与常识任务设计.md sec. 1, 2, 3.1): l5_slippery_lift saturated. Sonnet 5.5 (cla4)
grasped the bar's middle, lifted, saw the 45-60 deg tilt, put the bar down and regrasped near the centre of mass,
all inside attempt 1 (both runs succeeded on attempt 1): the tilting lift cost nothing. Rule R3: the first
common-sense interaction must cost something. Here the tilting lift spills the load.

Scene (world frame, robot base at x = -0.66; "forward" = +x, "left" = +y): the l5_slippery_lift bar (steel grey,
4 x 28 x 4 cm, long axis along the robot's left-right axis, same layout and hidden weight per seed) with two small red
blocks (2.5 cm cubes, 30 g, smooth: friction 0.2, they slide on the steel beyond ~11 deg of tilt) resting on its top
near the ends, 12.5 cm left and right of its middle.

Hidden variable (same in every attempt, invisible): a slug inside the bar puts its centre of mass e = 2.5-7 cm from
the middle, towards the left or right end (bar 0.8-1.2 kg, finger friction 0.40-0.60; l5_slippery_lift). The blocks
are symmetric, so the centre of mass of bar + blocks sits at M e / (M + 2 m) (2.3-6.7 cm).

Mechanism (physics, nothing teleported): grasped at its middle and lifted, the bar rotates in the slippery finger
pads and hangs tilted; the block on the low (heavy) end slides off, drops onto the table and, once the bar is set down
again, lies 2-5 cm beyond the bar's end; the other block slides along the bar to the fingers. Measured (20 seeds,
middle grasp): lift 15 cm at 0.1 m/s -> tilt 29-54 deg, spill 20/20; 15 cm at 0.03 m/s -> spill 20/20 (block friction =
the bar's, task card); a 5.5 cm "half lift" -> 20-22 deg, spill 20/20; 3 cm -> 10-11 deg, spill 2/20; 1 / 1.5 cm test
lifts -> 0.6-2.5 / 2.5-4.5 deg, spill 0/20 each (the low end stays on the table). Putting it right is a real repair:
set the bar down, release, pick the fallen block from the table, put it back on the end, move the other block back to
its end, regrasp nearer the heavy end, lift again: 20 commands for the scripted policy (5 for the whole oracle; the
same repair costs 7 commands in l5_slippery_lift, where nothing spills). What helps: a ~1 cm test lift shows the tilt
direction without spilling, or, after the first failure, grasping on the side of the end that sank (the heavy end),
ideally at the combined centre of mass (+-5 mm: 80/80 level lifts, nothing spilled).

Success: the bar is held (after a 1.5 s hold), tilted <= 8 deg, its lowest point >= 10 cm above the table (the
l5_slippery_lift condition), and both blocks rest on the bar, one within 3 cm of each end position (+-12.5 cm from the
middle; either block may sit on either end). Protocol: cross (reset before every attempt), k = 3.

Outcome (server log only; the agent gets F1): success / failure code, the bar's tilt and height, final block states,
per-step events (bar_lifted, bar_tilted, load_fell = the trap fired, load_displaced, load_removed, load_restored with
by_robot, bar_set_down), the lifts with their grasp points and maximum tilt, trap_fired, recovered_in_attempt,
first_interaction_ok and an F2-style `detail` in robot-centric words.
"""
import numpy as np

from .. import envs
from ..bddl import write_bddl
from ..objects import box_xml, register_generated
from ..task_base import register_task
from .l5_slippery_lift import (BAR, HALF, LIFT_MIN, TILT_MAX, YAW_ACROSS, Z_GRASP, BarSkills, SlipperyLift,
                               bar_tilt_deg, weighted_box_xml)

TZ = envs.TABLE_Z
LOADS = ("block_1", "block_2")  # block_1 starts on the left (+y) end, block_2 on the right (-y) end
LOAD_SIGN = {"block_1": 1.0, "block_2": -1.0}
LOAD_H = 0.0125  # 2.5 cm cubes
LOAD_MASS = 0.030
LOAD_RGBA = (0.85, 0.20, 0.12, 1.0)
LOAD_MU = 0.20  # block sliding friction (geom priority 3: used against the bar, the table and the fingers): the blocks
#                 slide on the steel bar beyond ~11 deg of tilt. With the bar's own mu (0.4-0.6, slide beyond 22-31 deg)
#                 a 5.5 cm "half lift" (20 deg) spilled nothing (0/10) and was a free probe; with 0.2 it spills.
SLOT = 0.125  # block centre this far from the bar's middle (along the bar)
SLOT_TOL = 0.03  # success: within 3 cm of an end position
BAR_TOP = TZ + 2 * HALF[2]
LIFTED = 0.02  # the bar's centre this much higher than at rest (while held) -> "lifted" (a ~1 cm test lift is not)
DOWN = 0.008  # ... and back below this -> "set down"
DEBOUNCE = 5  # control steps a new block state must persist before it counts
SAFE_ALONG = 0.09  # scripted grasps stay within +-9 cm of the middle (the hand must not touch the blocks)


class _Budget(Exception):
    """the scripted policy used up the agent command budget of the attempt"""


def bar_frame(env):
    """(centre, along, side, up): `along` = the bar's long axis oriented towards the robot's left (+y); `up` = the
    bar's cross-section axis closest to world up (the face the blocks rest on); `side` completes the frame."""
    bid = envs.body_id(env, BAR)
    d = env.sim.data
    c = d.body_xpos[bid].copy()
    R = d.body_xmat[bid].reshape(3, 3)
    al = R[:, 1].copy()
    if al[1] < 0:
        al = -al
    up = max((R[:, 0].copy(), R[:, 2].copy()), key=lambda v: abs(v[2]))
    if up[2] < 0:
        up = -up
    side = np.cross(up, al)
    return c, al, side, up


def _end_word(s):
    return "left" if s > 0 else "right"


class BeamSkills(BarSkills):
    """BarSkills (records where along the bar the jaws closed) + the task's per-step monitor."""

    def __init__(self, env, task, recorder=None):
        super().__init__(env, recorder=recorder)
        self.task = task

    def _act(self, dpos, drot, grip):
        super()._act(dpos, drot, grip)
        self.task.monitor_step()


@register_task
class LoadedBeam(SlipperyLift):
    name = "l5_loaded_beam"
    level = "L5"
    category = "dynamics_adaptation"
    capabilities = ("Perceive", "Reason", "Plan", "Utilize")
    protocol = "cross"
    max_attempts = 3
    agent_max_steps = 40
    instruction = ("Lift the metal bar together with the two small red blocks resting near its ends, and hold it level "
                   "(tilted less than 8 degrees) with its lowest point at least 10 cm above the table. Both blocks must "
                   "stay on the bar near its ends. Keep holding the bar when you finish.")
    instruction_indirect = ("Raise the long grey rod with the two little red cubes sitting near its ends and hold it "
                            "horizontal (tilted less than 8 degrees), its lowest point at least 10 cm above the table. "
                            "Both cubes must stay on the rod near its ends. Keep holding the rod when you finish.")

    # ------------------------------------------------------------ scene
    def make_bddl(self):
        bar = register_generated("LffBeamBar", weighted_box_xml("lff_beam_bar", HALF, (0.52, 0.55, 0.58, 1.0)))
        dens = LOAD_MASS / (2 * LOAD_H) ** 3
        la = register_generated("LffBeamBlockA", box_xml("lff_beam_block_a", (LOAD_H,) * 3, LOAD_RGBA, density=dens))
        lb = register_generated("LffBeamBlockB", box_xml("lff_beam_block_b", (LOAD_H,) * 3, LOAD_RGBA, density=dens))
        # LIBERO's sampler puts the bar anywhere within +-13 cm of its region's centre (it "keeps the bar's 14 cm
        # radius inside" a 2 cm region) and rejects block placements closer than 14 + 1.8 cm to it: the block regions
        # are >= 19 cm from every possible bar position (the first layout, 2 cm beside the bar's range, failed with
        # "Cannot place all objects" in 13 of 400 resets, this one 0 of 400). apply_instance moves everything anyway.
        objs = [(BAR, bar, "bar_region", (-0.11, -0.01, -0.09, 0.01)),
                ("block_1", la, "block1_region", (-0.11, 0.32, -0.09, 0.34)),
                ("block_2", lb, "block2_region", (-0.11, -0.34, -0.09, -0.32))]
        return write_bddl(self.name, "lift the loaded bar and hold it level", objs)

    def __init__(self, cam_res=None):
        self.mon = None
        self.budget = None  # scripted policies only: command budget of the current attempt
        self.ncmd = 0
        super().__init__(cam_res)

    def sample_instance(self, seed):
        inst = super().sample_instance(seed)  # same bar layout and hidden weight as l5_slippery_lift for this seed
        m = inst["mass"]
        inst.update(load_mass=LOAD_MASS, slot=SLOT,
                    com_total=float(m * inst["com_offset"] / (m + 2 * LOAD_MASS)))  # bar + both blocks on their ends
        return inst

    def apply_instance(self, inst):
        super().apply_instance(inst)  # bar pose, friction, hidden weight (+ envs.refit_bounds)
        x, y = inst["bar_xy"]
        for nm in LOADS:
            envs.set_obj_pose(self.env, nm, [x, y + LOAD_SIGN[nm] * SLOT, BAR_TOP + LOAD_H + 0.0005])
            if LOAD_MU is not None:
                envs.set_friction(self.env, nm, LOAD_MU, priority=3)

    def reset_instance(self, inst, recorder=None):
        self.env.reset()
        self.apply_instance(inst)
        self.env.sim.forward()
        envs.settle(self.env, 10)
        self.mon = None
        self.sk = BeamSkills(self.env, self, recorder=recorder)
        self._init_monitor()
        return self.sk

    # ------------------------------------------------------------ state
    def bar_bottom(self):
        return envs.obj_min_z(self.env, BAR) - TZ

    def load_rel(self, nm):
        """block centre in the bar frame: (along (+ = towards the bar's left end), side, height above the bar's centre)"""
        c, al, side, up = bar_frame(self.env)
        r = envs.obj_pos(self.env, nm) - c
        return float(r @ al), float(r @ side), float(r @ up)

    def _block_held(self, nm):
        sk = self.sk
        if sk is None or sk.grip < 0:
            return False
        if np.linalg.norm(envs.obj_pos(self.env, nm) - sk.eef_pos()) > 0.06:
            return False
        return sk.holding(nm)

    def load_state(self, nm, tol=SLOT_TOL):
        """'slot+' / 'slot-' (on the bar within `tol` of its left / right end position), 'bar' (on the bar
        elsewhere), 'held', 'table', 'air' (anything else: falling, resting on the gripper ...)"""
        if self._block_held(nm):
            return "held"
        a, s, u = self.load_rel(nm)
        if abs(u - (HALF[2] + LOAD_H)) <= 0.008 and abs(s) <= HALF[0] + 0.003 and abs(a) <= HALF[1] + 0.003:
            if abs(a - SLOT) <= tol:
                return "slot+"
            if abs(a + SLOT) <= tol:
                return "slot-"
            return "bar"
        if envs.obj_pos(self.env, nm)[2] < TZ + 0.027:
            return "table"
        return "air"

    # ------------------------------------------------------------ monitor
    def _init_monitor(self):
        st = {nm: self.load_state(nm) for nm in LOADS}
        self.mon = dict(stable=dict(st), cand={nm: None for nm in LOADS}, cnt={nm: 0 for nm in LOADS},
                        last_on_bar={nm: self.load_rel(nm)[0] for nm in LOADS}, lost=set(), carried=set(), events=[],
                        lifts=[],
                        lifted=False, tilted=False, max_tilt=0.0)

    def _event(self, ev, step=None, **kw):
        self.mon["events"].append(dict(step=int(self.sk.n_steps if step is None else step), ev=ev, **kw))

    def _last_grasp(self):
        return next((c for c in reversed(self.sk.closes) if c["holding"]), None)

    def monitor_step(self):
        mon, sk = self.mon, self.sk
        if mon is None or sk is None:
            return
        n = sk.n_steps
        # --- the bar: lifts and tilt
        rise = float(envs.obj_pos(self.env, BAR)[2]) - (TZ + HALF[2])
        held = sk.grip > 0 and sk.holding(BAR)
        if not mon["lifted"] and held and rise > LIFTED:
            g = self._last_grasp()
            mon["lifted"], mon["tilted"], mon["max_tilt"] = True, False, 0.0
            mon["lifts"].append(dict(step_up=n, step_down=None, grasp_along=None if g is None else round(g["along"], 4),
                                     max_tilt_deg=0.0, falls=[]))
            self._event("bar_lifted", grasp_along=mon["lifts"][-1]["grasp_along"])
        if mon["lifted"]:
            t = bar_tilt_deg(self.env)
            lift = mon["lifts"][-1]
            if abs(t) > abs(lift["max_tilt_deg"]):
                lift["max_tilt_deg"] = round(t, 1)
            if not mon["tilted"] and abs(t) > TILT_MAX and held:
                mon["tilted"] = True
                self._event("bar_tilted", tilt_deg=round(t, 1), low_end=_end_word(t))
            if rise < DOWN:
                mon["lifted"] = False
                lift["step_down"] = n
                self._event("bar_set_down", max_tilt_deg=lift["max_tilt_deg"])
        # --- the blocks (debounced state machine)
        for nm in LOADS:
            # hysteresis: a block that has left its end position must come back within 2 cm of an end to count as
            # restored (a block sliding along the tilted bar hovered around the 3 cm limit and flickered)
            s = self.load_state(nm, SLOT_TOL - 0.01 if nm in mon["lost"] else SLOT_TOL)
            if s in ("slot+", "slot-", "bar"):
                mon["last_on_bar"][nm] = self.load_rel(nm)[0]
            old = mon["stable"][nm]
            if s == old:
                mon["cand"][nm], mon["cnt"][nm] = None, 0
                continue
            if s != mon["cand"][nm]:
                mon["cand"][nm], mon["cnt"][nm] = s, 0
            mon["cnt"][nm] += 1
            if mon["cnt"][nm] < DEBOUNCE:
                continue
            step = n - DEBOUNCE + 1
            mon["stable"][nm], mon["cand"][nm], mon["cnt"][nm] = s, None, 0
            start = _end_word(LOAD_SIGN[nm])
            if s == "held":
                mon["carried"].add(nm)
            if s in ("slot+", "slot-"):
                if nm in mon["lost"]:
                    # by_robot: the gripper carried it back (else physics did, e.g. the high-end block sliding back out
                    # while the bar swings after the other block fell: 3-4 of 59 blind attempts)
                    mon["lost"].discard(nm)
                    self._event("load_restored", step, load=nm, start_end=start, end=_end_word(1 if s == "slot+" else -1),
                                by_robot=nm in mon["carried"])
                    mon["carried"].discard(nm)
            elif old in ("slot+", "slot-") and s == "bar":
                mon["lost"].add(nm)
                self._event("load_displaced", step, load=nm, start_end=start, along_cm=round(100 * self.load_rel(nm)[0], 1))
            elif old in ("slot+", "slot-", "bar", "air") and s == "table":
                mon["lost"].add(nm)
                end = _end_word(mon["last_on_bar"][nm])
                self._event("load_fell", step, load=nm, start_end=start, off_end=end)
                if mon["lifted"]:
                    mon["lifts"][-1]["falls"].append(nm)
            elif old == "held" and s == "table":
                mon["lost"].add(nm)
                self._event("load_removed", step, load=nm, start_end=start)
            elif old in ("slot+", "slot-") and s == "held":
                mon["lost"].add(nm)  # picked up by the agent: counts as lost until it is back on an end

    # ------------------------------------------------------------ outcome / F2
    def _where(self, nm):
        """robot-centric description of a block that is not on an end position"""
        st = self.load_state(nm)
        a, s, u = self.load_rel(nm)
        if st == "held":
            return "is held in the gripper"
        if st == "bar":
            e = 1.0 if a >= 0 else -1.0
            return (f"is still on the bar but {100 * (SLOT - abs(a)):.0f} cm in from its {_end_word(e)} end position "
                    f"({100 * abs(a):.0f} cm {_end_word(e)} of the bar's middle)")
        c, al, side, up = bar_frame(self.env)
        e = 1.0 if a >= 0 else -1.0
        u = al[:2] / max(np.linalg.norm(al[:2]), 1e-6)  # the bar's direction seen from above
        end = c[:2] + al[:2] * e * HALF[1]  # the end's position seen from above (also while the bar hangs tilted)
        r = envs.obj_pos(self.env, nm)[:2] - end
        beyond = float(r @ u) * e
        lateral = float(r @ np.array([-u[1], u[0]]))
        where = "on the table" if st == "table" else "off the bar"
        if beyond > 0 and abs(lateral) < 0.04:
            return f"lies {where}, {100 * beyond:.0f} cm beyond the bar's {_end_word(e)} end"
        fx = f"{100 * abs(r[0]):.0f} cm {'forward' if r[0] > 0 else 'back'}"
        fy = f"{100 * abs(r[1]):.0f} cm {'left' if r[1] > 0 else 'right'}"
        return f"lies {where}, {fx} and {fy} of the bar's {_end_word(e)} end"

    def outcome(self, inst):
        sk, mon = self.sk, self.mon
        sk.hold(30)  # 1.5 s with the gripper as the agent left it (envs.settle would open the jaws)
        held = bool(sk.holding(BAR))
        tilt = bar_tilt_deg(self.env)
        bottom = self.bar_bottom()
        states = {nm: self.load_state(nm) for nm in LOADS}
        on_plus = [nm for nm in LOADS if states[nm] == "slot+"]
        on_minus = [nm for nm in LOADS if states[nm] == "slot-"]
        loads_ok = len(on_plus) == 1 and len(on_minus) == 1
        level_ok = held and abs(tilt) <= TILT_MAX and bottom >= LIFT_MIN
        success = bool(level_ok and loads_ok)
        lifted_ever = bool(mon["lifts"]) or (sk.zmax - sk.z0) > 0.03
        g = self._last_grasp()
        evs = mon["events"]
        falls = [e for e in evs if e["ev"] == "load_fell"]
        restored = [e for e in evs if e["ev"] == "load_restored"]
        lifts = mon["lifts"]
        first_ok = None if not lifts else bool(not lifts[0]["falls"])  # the first lift spilled nothing
        # --- text
        gtxt = ""
        if g is not None:
            a = g["along"]
            gtxt = (f" You last grasped the bar {100 * abs(a):.1f} cm to the {_end_word(a)} of its middle."
                    if abs(a) >= 0.0005 else " You last grasped the bar at its middle.")
        tilt_txt = (f"hangs tilted by {abs(tilt):.0f} degrees, its {_end_word(tilt)} end lower than its "
                    f"{_end_word(-tilt)} end")
        if success:
            failure = None
            bar_txt = (f"The bar is held level (tilt {abs(tilt):.0f} degrees), {100 * bottom:.0f} cm above the table, "
                       "with a block on each end.")
        elif held and abs(tilt) > TILT_MAX:
            failure = "tilted"
            bar_txt = f"The bar did not stay level: it rotated in the gripper and now {tilt_txt}."
            if bottom < LIFT_MIN:
                bar_txt += f" Its lowest point is {100 * bottom:.0f} cm above the table."
        elif held and bottom < LIFT_MIN:
            failure = "too_low"
            bar_txt = (f"The bar is held level (tilt {abs(tilt):.0f} degrees) but its lowest point is only "
                       f"{100 * bottom:.0f} cm above the table.")
        elif held:
            failure = "load_off"
            bar_txt = (f"The bar is held level (tilt {abs(tilt):.0f} degrees), {100 * bottom:.0f} cm above the table, "
                       "but the blocks are not both on its ends.")
        elif lifted_ever:
            failure = "dropped"
            bar_txt = "The bar was lifted but is no longer in the gripper; it lies on the table."
        else:
            failure = "no_grasp"
            bar_txt = ("The gripper never closed on the bar; it is still lying on the table." if not sk.closes else
                       "The gripper did not catch the bar; it is still lying on the table.")
        parts = [bar_txt]
        if falls:
            f0 = falls[0]
            lift = next((l for l in lifts if f0["load"] in l["falls"]), None)
            how = (f" while the bar hung tilted by up to {abs(lift['max_tilt_deg']):.0f} degrees with its "
                   f"{_end_word(lift['max_tilt_deg'])} end lower") if lift and abs(lift["max_tilt_deg"]) > 1 else ""
            parts.append(f"The {f0['start_end']} block slid off the bar's {f0['off_end']} end{how}."
                         + (f" Blocks fell off {len(falls)} times in this attempt." if len(falls) > 1 else ""))
        if not loads_ok:
            for nm in LOADS:
                if states[nm] in ("slot+", "slot-") and (len(on_plus) == 1 or states[nm] != "slot+") and \
                        (len(on_minus) == 1 or states[nm] != "slot-"):
                    continue
                parts.append(f"The {_end_word(LOAD_SIGN[nm])} block {self._where(nm)}.")
        detail = " ".join(parts) + gtxt
        loads = {}
        for nm in LOADS:
            a, s, u = self.load_rel(nm)
            loads[nm] = dict(state=states[nm], along_cm=round(100 * a, 1), side_cm=round(100 * s, 1),
                             above_bar_centre_cm=round(100 * u, 1))
        out = dict(success=success, failure=failure, held=held, tilt_deg=round(tilt, 1), bottom_cm=round(100 * bottom, 1),
                   grasp_along=None if g is None else round(g["along"], 4),
                   stages=dict(bar_held_level_high=bool(level_ok), loads_on_ends=bool(loads_ok)),
                   loads=loads, loads_on_ends=len(on_plus) + len(on_minus) if loads_ok else
                   int(len(on_plus) >= 1) + int(len(on_minus) >= 1),
                   trap_fired=bool(falls), n_load_falls=len(falls), n_restored=len(restored),
                   n_restored_by_robot=len([e for e in restored if e.get("by_robot")]),
                   n_regressions=len([e for e in evs if e["ev"] in ("load_fell", "load_displaced")]),
                   recovered_in_attempt=bool(falls and success), first_interaction_ok=first_ok,
                   n_lifts=len(lifts), lifts=lifts, events=list(evs), closes=sk.closes, detail=detail)
        out["detail_indirect"] = detail.replace("bar", "rod").replace("block", "cube")
        return out

    # ------------------------------------------------------------ scripted primitives (agent-command counted)
    def _cmd(self, target=None, yaw=None, speed=None, grip=None, tol=0.004, max_steps=250):
        """one agent command (move_eef with an optional gripper action, or observe when nothing is given)"""
        self.ncmd += 1
        if self.budget is not None and self.ncmd > self.budget:
            raise _Budget()
        sk = self.sk
        if target is not None:
            t = np.asarray(target, float).copy()
            t[2] = max(t[2], TZ + 0.006)  # the agent interface does not go below 0.5 cm above the table
            sk.move_to(t, yaw=yaw, speed=speed, tol=tol, max_steps=max_steps)
        if grip is not None:
            sk.set_gripper(bool(grip))

    def grasp_lift(self, along, lift=0.15, speed=0.10, dx=0.0):
        """open, above the grasp point (jaws across the bar), descend, close, lift, look: 5 commands"""
        c, al, side, up = bar_frame(self.env)
        g = c + al * float(along) + np.array([dx, 0.0, 0.0])
        z = max(float(c[2]), Z_GRASP)
        self._cmd([g[0], g[1], z + 0.10], yaw=YAW_ACROSS, grip=False)
        self._cmd([g[0], g[1], z], tol=0.003)
        self._cmd(grip=True)
        self._cmd([g[0], g[1], z + lift], speed=speed, max_steps=600)
        self._cmd()

    def put_down(self):
        """lower the held bar onto the table, release, back up: 2 commands"""
        e = self.sk.eef_pos()
        self._cmd([e[0], e[1], Z_GRASP + 0.003], speed=0.05, grip=False, max_steps=600)
        self._cmd([e[0], e[1], Z_GRASP + 0.12])

    def _block_yaw(self, nm):
        """gripper yaw whose jaws close on two faces of the block, preferring jaws that do not reach towards the bar"""
        q = envs.obj_quat(self.env, nm)
        w, x, y, z = q
        psi = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        c, al, side, up = bar_frame(self.env)
        p = envs.obj_pos(self.env, nm)
        a = float(np.clip((p - c) @ al, -HALF[1], HALF[1]))
        v = (p - (c + al * a))[:2]
        v = v / max(np.linalg.norm(v), 1e-6)
        best = None
        for k in range(2):
            ang = psi + k * np.pi / 2  # closing direction candidate (a face normal)
            n = np.array([np.cos(ang), np.sin(ang)])
            score = abs(n @ v)
            if best is None or score < best[0]:
                best = (score, ang)
        yaw = best[1] + np.pi / 2  # yaw = closing direction + 90 deg (YAW_ACROSS closes along world x)
        return float((yaw + np.pi / 2) % np.pi - np.pi / 2)

    def move_block_to_slot(self, nm, sign, tries=2):
        """pick a block (from the table or the bar) and set it on the end position `sign`: 6 commands per try"""
        sk = self.sk
        for _ in range(tries):
            p = envs.obj_pos(self.env, nm)
            yaw = self._block_yaw(nm)
            zg = max(float(p[2]), TZ + 0.008)
            self._cmd([p[0], p[1], zg + 0.08], yaw=yaw, grip=False)
            self._cmd([p[0], p[1], zg], tol=0.003, grip=True)
            self._cmd([p[0], p[1], BAR_TOP + LOAD_H + 0.07])
            if not sk.holding(nm):
                continue
            c, al, side, up = bar_frame(self.env)
            t = c + al * sign * SLOT
            ztop = envs.obj_max_z(self.env, BAR)
            self._cmd([t[0], t[1], ztop + LOAD_H + 0.06])
            self._cmd([t[0], t[1], ztop + LOAD_H + 0.004], speed=0.05, grip=False)
            self._cmd([t[0], t[1], ztop + LOAD_H + 0.08])
            if self.load_state(nm) == ("slot+" if sign > 0 else "slot-"):
                return True
        return False

    def restore_loads(self):
        """put every block that is not on an end position back onto a free end; True if both ends are taken"""
        for _ in range(2):
            st = {nm: self.load_state(nm) for nm in LOADS}
            free = [s for s in (1.0, -1.0) if not any(v == ("slot+" if s > 0 else "slot-") for v in st.values())]
            if not free:
                return True
            loose = [nm for nm in LOADS if st[nm] not in ("slot+", "slot-")]
            for sgn in free:
                if not loose:
                    break
                c, al, side, up = bar_frame(self.env)
                slot = c + al * sgn * SLOT
                nm = min(loose, key=lambda b: np.linalg.norm(envs.obj_pos(self.env, b)[:2] - slot[:2]))
                loose.remove(nm)
                self.move_block_to_slot(nm, sgn)
            self._cmd()  # look
        st = {nm: self.load_state(nm) for nm in LOADS}
        return sorted(st.values()) == ["slot+", "slot-"]

    def judge(self):
        """what an agent sees after a lift (perfect perception): (all good, tilt, blocks on their ends)"""
        tilt = bar_tilt_deg(self.env)
        st = sorted(self.load_state(nm) for nm in LOADS)
        ok_loads = st == ["slot+", "slot-"]
        good = self.sk.holding(BAR) and abs(tilt) <= TILT_MAX - 2 and self.bar_bottom() >= LIFT_MIN + 0.01 and ok_loads
        return bool(good), float(tilt), ok_loads

    # ------------------------------------------------------------ scripted policies
    def execute(self, inst, params):
        self.grasp_lift(params["along"], params.get("lift", 0.15), params.get("speed", 0.10), params.get("dx", 0.0))

    def default_params(self, inst):
        return dict(along=0.0, lift=0.15, speed=0.10)  # Sonnet 5.5 (cla4): middle, up to ~15-18 cm at 0.1 m/s

    def oracle_params(self, inst):
        return dict(along=float(inst["com_total"]), lift=0.15, speed=0.10)

    def blind_params(self, inst, attempt, rng):
        return dict(along=float(rng.normal(0.0, 0.01)), lift=float(rng.uniform(0.12, 0.18)),
                    speed=float(rng.uniform(0.05, 0.20)))

    MARGIN = 0.008  # a lift that sank towards +y at grasp g says COM > g + MARGIN
    STEP = 0.030  # first move towards the end that sank when only one side is bounded

    def bracket(self, obs):
        """grasp point from (grasp_along, tilt) observations: the end that sank is the heavy side of that grasp"""
        lo, hi = -SAFE_ALONG, SAFE_ALONG
        lo_set = hi_set = False
        for g, t in obs:
            if t > 0:
                lo, lo_set = max(lo, g + self.MARGIN), True
            else:
                hi, hi_set = min(hi, g - self.MARGIN), True
        if lo_set and hi_set:
            a = 0.5 * (lo + hi)
        elif lo_set:
            a = lo + self.STEP - self.MARGIN
        elif hi_set:
            a = hi - self.STEP + self.MARGIN
        else:
            a = 0.0
        return float(np.clip(a, -SAFE_ALONG, SAFE_ALONG))

    def adapt_params(self, inst, history):
        obs = [o for h in history for o in h["params"].get("obs", [])]
        return dict(along=self.bracket(obs))

    def adaptive_attempt(self, obs):
        """lift at the current estimate; if the bar tilts or a block falls: set the bar down, put the blocks back,
        regrasp on the side of the end that sank (bracketing), lift again -- while the command budget allows"""
        tried = []
        along = self.bracket(obs)
        while True:
            tried.append(round(along, 4))
            self.grasp_lift(along)
            good, tilt, ok_loads = self.judge()
            if good:
                return tried
            g = self._last_grasp()
            if abs(tilt) > 3.0 or not ok_loads:
                lift = self.mon["lifts"][-1] if self.mon["lifts"] else None
                t = tilt if abs(tilt) > 3.0 else (lift["max_tilt_deg"] if lift else 0.0)
                if abs(t) > 1.0:
                    obs.append((g["along"] if g is not None else along, t))
            # a repair = set down (2) + both blocks back (2 x 6: the fallen one and the one that slid to the fingers)
            # + look (1) + regrasp, lift, look (5) = 20 commands; without enough budget keep holding the bar as it is
            if self.budget is not None and self.ncmd + 20 > self.budget:
                return tried
            self.put_down()
            if not self.restore_loads():
                return tried
            along = self.bracket(obs)

    def run_scripted(self, kind, inst, k=None, seed=0, on_attempt=None):
        """cross protocol with an agent-like command budget per attempt (agent_max_steps; one move_eef with an optional
        gripper action, or one observe, = one command). naive / oracle / blind: one lift, no checking.
        adaptive: naive in attempt 1 but checks and repairs inside the attempt (`adaptive_attempt`); later attempts
        start from the grasp point learned so far."""
        k = k or self.max_attempts
        rng = np.random.default_rng(seed)
        history, obs = [], []
        for a in range(k):
            self.reset_instance(inst)
            self.ncmd, self.budget = 0, self.agent_max_steps
            if kind == "oracle":
                params = self.oracle_params(inst)
            elif kind == "naive":
                params = self.default_params(inst)
            elif kind == "blind":
                params = self.default_params(inst) if a == 0 else self.blind_params(inst, a, rng)
            elif kind == "adaptive":
                params = dict(obs=[])
            else:
                raise ValueError(kind)
            n0 = len(obs)
            try:
                if kind == "adaptive":
                    params["tried"] = self.adaptive_attempt(obs)
                else:
                    self.execute(inst, params)
            except _Budget:
                params["budget_used_up"] = True
            if kind == "adaptive":
                params["obs"] = [[round(g, 4), round(t, 1)] for g, t in obs[n0:]]
            self.budget = None
            out = self.outcome(inst)
            out["commands"] = self.ncmd
            history.append(dict(attempt=a + 1, params=params, outcome=out))
            if on_attempt:
                on_attempt(a, params, out)
            if out["success"] or kind in ("oracle", "naive"):
                break
        return history

