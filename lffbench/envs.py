"""Environment construction and low-level sim helpers on top of LIBERO / robosuite."""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
# LIBERO reads its paths from <LIBERO_CONFIG_PATH>/config.yaml (written by setup/install.sh into <repo>/.libero)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".libero"))

import numpy as np

TABLE_Z = 0.90  # LIBERO main_table top surface (table_offset z)


def make_env(bddl_path, cam_res=256, cams=("agentview", "robot0_eye_in_hand"), depth=False, controller=None):
    """Build a LIBERO env from a BDDL file without camera rendering on every step.

    Images are rendered on demand with `render(env, cam)`.
    """
    import robosuite as suite
    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero.envs import TASK_MAPPING

    cfg = suite.load_controller_config(default_controller="OSC_POSE")
    if controller is None:
        from .skills import CONTROLLER
        controller = CONTROLLER
    if controller:
        cfg.update(controller)
    info = BDDLUtils.get_problem_info(bddl_path)
    env = TASK_MAPPING[info["problem_name"]](
        bddl_path,
        robots=["Panda"],
        controller_configs=cfg,
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=10 ** 7,
        ignore_done=True,
        hard_reset=False,
        camera_names=list(cams),
        camera_heights=cam_res,
        camera_widths=cam_res,
        camera_depths=depth,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    )
    env.lff_language = info["language_instruction"]
    return env


def render(env, cam="agentview", res=256, depth=False):
    """Render an RGB image (H, W, 3, uint8), upright."""
    out = env.sim.render(camera_name=cam, width=res, height=res, depth=depth)
    if depth:
        rgb, d = out
        return rgb[::-1].copy(), d[::-1].copy()
    return out[::-1].copy()


# ---------------------------------------------------------------- object state helpers

def body_id(env, obj_name):
    return env.sim.model.body_name2id(env.objects_dict[obj_name].root_body)


def obj_pos(env, obj_name):
    return env.sim.data.body_xpos[body_id(env, obj_name)].copy()


def obj_quat(env, obj_name):
    """wxyz"""
    return env.sim.data.body_xquat[body_id(env, obj_name)].copy()


def obj_upright_cos(env, obj_name):
    """cos of the angle between the object's z axis and world z."""
    xmat = env.sim.data.body_xmat[body_id(env, obj_name)].reshape(3, 3)
    return float(xmat[2, 2])


def free_joint_addr(env, obj_name):
    jname = env.objects_dict[obj_name].joints[0]
    jid = env.sim.model.joint_name2id(jname)
    return env.sim.model.jnt_qposadr[jid], env.sim.model.jnt_dofadr[jid]


def set_obj_pose(env, obj_name, pos, yaw=0.0, quat_wxyz=None):
    """Teleport a free-joint object (used only at reset / by the protocol, never by agents)."""
    qa, da = free_joint_addr(env, obj_name)
    if quat_wxyz is None:
        quat_wxyz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    env.sim.data.qpos[qa:qa + 3] = pos
    env.sim.data.qpos[qa + 3:qa + 7] = quat_wxyz
    env.sim.data.qvel[da:da + 6] = 0.0


def obj_geom_ids(env, obj_name):
    return [env.sim.model.geom_name2id(g) for g in env.objects_dict[obj_name].contact_geoms]


def set_friction(env, obj_name, mu, priority=2):
    """Set sliding friction of an object's collision geoms and give them priority so their
    friction is used in contacts (MuJoCo otherwise takes the max of the two geoms)."""
    for g in obj_geom_ids(env, obj_name):
        env.sim.model.geom_friction[g, 0] = mu
        env.sim.model.geom_priority[g] = priority


def scale_mass(env, obj_name, mass):
    bid = body_id(env, obj_name)
    m0 = env.sim.model.body_mass[bid]
    env.sim.model.body_mass[bid] = mass
    env.sim.model.body_inertia[bid] *= mass / max(m0, 1e-9)


def settle(env, steps=20):
    """Let physics settle while holding the arm still."""
    for _ in range(steps):
        env.step(np.zeros(env.action_dim))


def contacts_between(env, geoms_a, geoms_b):
    a, b = set(geoms_a), set(geoms_b)
    d = env.sim.data
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a):
            return True
    return False


def robot_geom_ids(env):
    m = env.sim.model
    ids = []
    for g in range(m.ngeom):
        n = m.geom_id2name(g) or ""
        if n.startswith("robot0_") or n.startswith("gripper0_"):
            ids.append(g)
    return ids


# ---------------------------------------------------------------- upright placement (added 2026-10-06)

def _quat_mul(a, b):
    """wxyz quaternion product a*b"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def canonical_quat(env, obj_name):
    """The orientation LIBERO's own placement gives an object, with any random yaw set to 0 (wxyz).

    Reproduces LIBERO exactly, including its convention bug: TableRegionSampler._sample_quat builds an
    xyzw quaternion, multiplies it with obj.init_quat (xyzw), and bddl_base_domain writes that array
    straight into the wxyz qpos slot. E.g. scanned objects (plate, bowls, basket; rotation (pi/2, pi/2)
    about x) therefore end up yawed by 90 deg, not rolled; HOPE groceries (dict rotation) come out
    as (0.5, 0.5, 0.5, 0.5), which is the same under both readings. (Fixed 2026-10-06 after the L3
    developer found plates being stood on edge.)"""
    from robosuite.utils.transform_utils import quat_multiply

    o = env.objects_dict[obj_name]
    rot = getattr(o, "rotation", None)
    ax = getattr(o, "rotation_axis", None) or "z"

    def xyzw(axis, a):
        h = a / 2
        return {"x": np.array([np.sin(h), 0.0, 0.0, np.cos(h)]),
                "y": np.array([0.0, np.sin(h), 0.0, np.cos(h)]),
                "z": np.array([0.0, 0.0, np.sin(h), np.cos(h)])}[axis]

    def fixed(r):  # a random range means a free angle -> use 0 (only yaw ranges occur in practice)
        return float(r[0]) if float(min(r)) == float(max(r)) else 0.0

    if isinstance(rot, dict):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        for axis, r in rot.items():
            q = quat_multiply(xyzw(axis, fixed(r)), q)
    else:
        if rot is None:
            a = 0.0
        elif isinstance(rot, (tuple, list)):
            a = fixed(rot)
        else:
            a = float(rot)
        q = xyzw(ax, a)
    init = np.array(getattr(o, "init_quat", [0.0, 0.0, 0.0, 1.0]), dtype=float)
    q = quat_multiply(q, init)
    return np.array(q, dtype=float)  # used directly as wxyz, like LIBERO does


def obj_min_z(env, obj_name):
    """Lowest world z of the object's collision geoms (exact for boxes, AABB bound for meshes)."""
    m, d = env.sim.model, env.sim.data
    zmin = np.inf
    for g in obj_geom_ids(env, obj_name):
        R = d.geom_xmat[g].reshape(3, 3)
        c_local = m.geom_aabb[g, :3]
        half = m.geom_aabb[g, 3:]
        cz = d.geom_xpos[g][2] + R[2] @ c_local
        zmin = min(zmin, cz - np.abs(R[2]) @ half)
    return float(zmin)


def obj_max_z(env, obj_name):
    """Highest world z of the object's collision geoms (exact for boxes, AABB bound for meshes)."""
    m, d = env.sim.model, env.sim.data
    zmax = -np.inf
    for g in obj_geom_ids(env, obj_name):
        R = d.geom_xmat[g].reshape(3, 3)
        cz = d.geom_xpos[g][2] + R[2] @ m.geom_aabb[g, :3]
        zmax = max(zmax, cz + np.abs(R[2]) @ m.geom_aabb[g, 3:])
    return float(zmax)


FINGERTIP_BELOW_SITE = 0.0095  # lowest point of the Panda finger collision meshes below the grasp site (measured)


def place_upright(env, obj_name, xy, yaw=0.0, surface_z=TABLE_Z, clearance=0.002):
    """Teleport an object upright (LIBERO canonical orientation, then `yaw` about world z) so that its
    lowest collision point sits `clearance` above `surface_z`. Call env.sim.forward() / settle after."""
    q = _quat_mul(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), canonical_quat(env, obj_name))
    set_obj_pose(env, obj_name, [xy[0], xy[1], surface_z + 0.3], quat_wxyz=q)
    env.sim.forward()
    dz = obj_min_z(env, obj_name) - (surface_z + clearance)
    qa, _ = free_joint_addr(env, obj_name)
    env.sim.data.qpos[qa + 2] -= dz
    env.sim.forward()
