"""Camera geometry for the agent tools: pixel -> world back-projection with the rendered depth map.

Conventions (match the images the agent receives, i.e. `envs.render`, which flips MuJoCo's
bottom-up buffer to an upright image):
  u = column index (0 = left edge), v = row index (0 = top edge), both in pixels of a res x res image.
Intrinsics / extrinsics / depth linearisation come from robosuite.utils.camera_utils.
"""
import numpy as np
from robosuite.utils import camera_utils as CU

from .. import envs

# agent-facing camera name -> MuJoCo camera name
CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}


def mj_cam(name):
    if name in CAMERAS:
        return CAMERAS[name]
    if name in CAMERAS.values():
        return name
    raise KeyError(f"unknown camera {name!r}; use one of {sorted(CAMERAS)}")


def intrinsics(env, cam, res):
    return CU.get_camera_intrinsic_matrix(env.sim, mj_cam(cam), res, res)


def cam_to_world(env, cam):
    """4x4 pose of the camera in the world, OpenCV axes (x right, y down, z forward)."""
    return CU.get_camera_extrinsic_matrix(env.sim, mj_cam(cam))


def render_depth(env, cam, res):
    """Metric depth (distance along the optical axis, metres), upright (res, res)."""
    _, d = envs.render(env, mj_cam(cam), res, depth=True)
    return CU.get_real_depth_map(env.sim, np.clip(d, 0.0, 1.0))


def world_to_pixel(env, cam, xyz, res):
    """Project a world point; returns (u, v) floats (u = column, v = row) and the depth."""
    K = intrinsics(env, cam, res)
    T = np.linalg.inv(cam_to_world(env, cam))
    p = T @ np.append(np.asarray(xyz, float), 1.0)
    uvw = K @ p[:3]
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), float(p[2])


def pixel_ray(env, cam, u, v, res):
    """Unit ray direction (world) through pixel (u, v) and the camera origin."""
    K = intrinsics(env, cam, res)
    E = cam_to_world(env, cam)
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    d = E[:3, :3] @ d_cam
    return E[:3, 3].copy(), d / np.linalg.norm(d)


def pixel_to_world(env, cam, u, v, res, depth=None):
    """Back-project pixel (u, v) of a res x res image using the depth map. Returns a dict."""
    if depth is None:
        depth = render_depth(env, cam, res)
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iu < res and 0 <= iv < res):
        raise ValueError(f"pixel ({u}, {v}) outside the {res}x{res} image")
    z = float(depth[iv, iu])
    K = intrinsics(env, cam, res)
    E = cam_to_world(env, cam)
    p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, 1.0])
    p = E @ p_cam
    far = env.sim.model.vis.map.zfar * env.sim.model.stat.extent
    return dict(xyz=p[:3], depth=z, valid=bool(z < 0.98 * far))


def image_axes(env, cam, res, plane_z=envs.TABLE_Z):
    """World directions of image-right and image-down at the image centre, projected onto the table
    plane (for describing the camera to the agent). Returns two unit 2-vectors (x, y)."""
    o, d = pixel_ray(env, cam, res / 2, res / 2, res)
    if abs(d[2]) < 1e-6:
        return None
    t = (plane_z - o[2]) / d[2]
    c = o + t * d
    out = []
    for du, dv in ((8, 0), (0, 8)):
        o2, d2 = pixel_ray(env, cam, res / 2 + du, res / 2 + dv, res)
        t2 = (plane_z - o2[2]) / d2[2]
        w = (o2 + t2 * d2 - c)[:2]
        out.append(w / max(np.linalg.norm(w), 1e-9))
    return out


def describe_axes(env, cam, res):
    """Human-readable mapping 'image right ~ +y, image down ~ +x' (dominant axis, sign)."""
    ax = image_axes(env, cam, res)
    if ax is None:
        return None
    names = []
    for w in ax:
        i = int(np.argmax(np.abs(w)))
        names.append(("+" if w[i] > 0 else "-") + "xy"[i])
    return dict(right=names[0], down=names[1])
