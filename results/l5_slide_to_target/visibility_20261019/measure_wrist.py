"""Wrist camera: is the far / near protrusion visible? Arm parked at (a) the scripted post-strike pose (above the cube's
start, 8 cm above the push height; the self-test agent ended its strikes in the same place) and (b) 25 cm above the
square's centre. Overrun of the cube silhouette past the outer edge of the far / near line, in px (512x512)."""
import io, json, os, sys
import numpy as np
from PIL import Image
from lffbench.tasks.l5_slide import SlideToTarget, CUBE_HALF, Z_PUSH
from lffbench import envs

OUT = sys.argv[1]
SEEDS = [int(s) for s in sys.argv[2].split(",")]
os.makedirs(OUT, exist_ok=True)
RES = 512
CAM = "robot0_eye_in_hand"
t = SlideToTarget(cam_res=RES)
m = t.env.sim.model
gid = {m.geom_id2name(g): g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(("zone_1", "cube_1"))}
CUBE_G = gid["cube_1_g1"]
L = {"far": gid["zone_1_g3"], "near": gid["zone_1_g4"]}
FILL = gid["zone_1_g0"]


def seg():
    s = t.env.sim.render(RES, RES, camera_name=CAM, segmentation=True)[::-1]
    return np.where(s[..., 0] == 5, s[..., 1], -1)


def place(xy, z=None):
    envs.set_obj_pose(t.env, "cube_1", [xy[0], xy[1], envs.TABLE_Z + CUBE_HALF if z is None else z])
    qa, da = envs.free_joint_addr(t.env, "cube_1")
    t.env.sim.data.qvel[da:da + 6] = 0
    t.env.sim.forward()


def line_metrics(g, gref, name):
    cube = g == CUBE_G
    ref = gref == L[name]
    fill = gref == FILL
    if not ref.any() or not fill.any() or not cube.any():
        return None
    # outward image direction of this line: from the fill centroid to the line centroid (vertical component)
    out_down = np.nonzero(ref)[0].mean() > np.nonzero(fill)[0].mean()
    cols = [c for c in range(RES) if cube[:, c].any() and ref[:, c].any()][2:-2]
    if not cols:
        return dict(cols=0)
    over, vis = [], []
    for c in cols:
        lr = np.nonzero(ref[:, c])[0]; cr = np.nonzero(cube[:, c])[0]
        over.append(int(cr.max() - lr.max()) if out_down else int(lr.min() - cr.min()))
        vis.append(int((g[:, c] == L[name]).sum()))
    return dict(cols=len(cols), over=float(np.median(over)), vis=float(np.median(vis)),
                ref=float(np.median([ref[:, c].sum() for c in cols])))


rows = []
for seed in SEEDS:
    inst = t.sample_instance(seed)
    x0, y0 = inst["cube_xy"]; tx = x0 + inst["target_dist"]
    for pose_name in ("post_strike", "above_target"):
        t.reset_instance(inst)
        sk = t.sk
        sk.set_gripper(True, steps=5)
        if pose_name == "post_strike":
            sk.move_to([x0 - CUBE_HALF + 0.015, y0, Z_PUSH + 0.08], yaw=0.0)
        else:
            sk.move_to([tx, y0, envs.TABLE_Z + 0.25], yaw=0.0)
        place([x0, y0 + 0.6], z=envs.TABLE_Z + 0.5)
        gref = seg()
        for a in np.round(np.arange(-0.030, 0.0301, 0.002), 4):
            place([tx + a, y0])
            g = seg()
            rec = dict(seed=seed, pose=pose_name, along_cm=round(100 * a, 1))
            for name in ("far", "near"):
                rec[name] = line_metrics(g, gref, name)
            rows.append(rec)
            if seed == 3000 and round(100 * a, 1) in (-2.0, -1.6, -1.0, 0.0, 1.0, 1.6, 2.0):
                Image.fromarray(envs.render(t.env, CAM, RES)).save(
                    os.path.join(OUT, f"wrist_{pose_name}_s{seed}_{100 * a:+.1f}.jpg"), quality=90)
    print(seed, flush=True)
json.dump(rows, open(os.path.join(OUT, "wrist.json"), "w"), indent=0)
