"""Environment construction and low-level sim helpers on top of LIBERO / robosuite."""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
# LIBERO reads its paths from <LIBERO_CONFIG_PATH>/config.yaml (written by setup/install.sh into <repo>/.libero)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".libero"))

import numpy as np

TABLE_Z = 0.90  # LIBERO main_table top surface (table_offset z)
# Upper limit of the Panda elbow (robot0_joint4; stock -0.0698). The arm is fully stretched at about -0.47 rad; a fast
# move out of a far pose could carry the elbow past that into (-0.47, -0.07], where the OSC's only way to shorten the
# arm runs into the stock limit and every later move stops 4-17 cm short for the rest of the attempt (measured
# 2026-10-19: 50 of 160 targets of the workspace box; 44 of 714 logged LA attempts). Capping the joint keeps the elbow
# on the normal side: reach is unchanged, and no scripted oracle / naive run of the main set goes above -0.73.
ELBOW_Q4_MAX = -0.55


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
    m = env.sim.model  # hard_reset=False: the model (and this limit) survives env.reset()
    m.jnt_range[m.joint_name2id("robot0_joint4"), 1] = ELBOW_Q4_MAX
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


# ---------------------------------------------------------------- collision bounds after runtime edits (2026-10-12)

def _quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _prim_bounds(m, g):
    """(half extents in the geom frame, bounding-sphere radius) of a primitive geom from geom_size; None otherwise"""
    t, s = int(m.geom_type[g]), m.geom_size[g]
    if t == 6:  # box
        return np.array(s[:3], float), float(np.linalg.norm(s[:3]))
    if t == 4:  # ellipsoid
        return np.array(s[:3], float), float(max(s[:3]))
    if t == 2:  # sphere
        return np.full(3, float(s[0])), float(s[0])
    if t == 5:  # cylinder
        return np.array([s[0], s[0], s[1]], float), float(np.hypot(s[0], s[1]))
    if t == 3:  # capsule
        return np.array([s[0], s[0], s[0] + s[1]], float), float(s[0] + s[1])
    return None


def refit_bounds(model, bodies, margin=0.0, geoms=True):
    """Make the collision bounds of `bodies` (body id or ids) consistent again after runtime edits of their geoms
    (geom_size / geom_pos / geom_quat) or of their inertial frame (body_ipos / body_iquat). `model`: an env, a
    robosuite MjModel wrapper or a raw mujoco.MjModel.

    MuJoCo (3.2.3) computes these bounds at compile time and never updates them: geom_aabb / geom_rbound (geom
    frame; broad phase and geom-pair filters) and the per-body bounding-volume hierarchy bvh_aabb, stored in the
    body's INERTIAL frame and used by the mid-phase whenever one of the two bodies of a pair has >= 2 collision
    geoms. Stale boxes make MuJoCo silently skip contacts (lh_fence v1: a post whose centre of mass was raised sank
    2.2 cm into the table; measured 2026-10-12: a body with a single collision geom whose COM moved 6 cm missed every
    contact with a 2-geom body beyond the shifted box). With geoms=True the primitive geoms' geom_aabb / geom_rbound
    are recomputed from geom_size (meshes keep theirs); then every BVH node becomes the union of its leaf geoms'
    boxes (+ margin) in the runtime inertial frame, following body_sameframe / geom_sameframe like mj_kinematics
    (on a pristine model this reproduces the compiled bvh_aabb exactly).
    Also note: a runtime body_ipos is ignored by MuJoCo if the compiled inertial pos is the body origin
    (body_sameframe = 1); bodies whose centre of mass is moved at runtime need a non-zero compiled inertial pos."""
    if hasattr(model, "sim"):
        model = model.sim.model
    m = getattr(model, "_model", model)
    for b in np.atleast_1d(bodies):
        b = int(b)
        if geoms:
            for g in range(int(m.body_geomadr[b]), int(m.body_geomadr[b] + m.body_geomnum[b])):
                pb = _prim_bounds(m, g)
                if pb is not None:
                    m.geom_aabb[g] = [0.0, 0.0, 0.0, *pb[0]]
                    m.geom_rbound[g] = pb[1]
        a, n = int(m.body_bvhadr[b]), int(m.body_bvhnum[b])
        if a < 0 or n < 1:
            continue
        sf = int(m.body_sameframe[b])  # 1: inertial frame = body frame, 3: same orientation as the body
        ip = np.zeros(3) if sf == 1 else np.array(m.body_ipos[b], float)
        iR = np.eye(3) if sf in (1, 3) else _quat2mat(m.body_iquat[b])

        def rec(i):
            g = int(m.bvh_nodeid[i])
            if g >= 0:
                gsf = int(m.geom_sameframe[g])  # 1: body frame, 2: inertial frame, 3/4: their orientation
                gp = np.zeros(3) if gsf == 1 else ip if gsf == 2 else np.array(m.geom_pos[g], float)
                gR = np.eye(3) if gsf in (1, 3) else iR if gsf in (2, 4) else _quat2mat(m.geom_quat[g])
                R = iR.T @ gR
                c = iR.T @ (gp - ip) + R @ np.asarray(m.geom_aabb[g, :3])
                h = np.abs(R) @ np.asarray(m.geom_aabb[g, 3:])
                lo, hi = c - h, c + h
            else:
                l1, h1 = rec(a + int(m.bvh_child[i][0]))
                l2, h2 = rec(a + int(m.bvh_child[i][1]))
                lo, hi = np.minimum(l1, l2), np.maximum(h1, h2)
            m.bvh_aabb[i] = np.concatenate([(lo + hi) / 2, (hi - lo) / 2 + margin])
            return lo, hi
        rec(a)
