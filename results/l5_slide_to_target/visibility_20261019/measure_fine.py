"""1-mm sweep of the far / near / lateral boundary on many instances: segmentation-based overrun (px) of the cube
silhouette past the outer edge of the line, plus the same measured on colour in the JPEG q90 image."""
import io, json, os, sys
import numpy as np
from PIL import Image
from lffbench.tasks.l5_slide import SlideToTarget, CUBE_HALF
from lffbench import envs

OUT = sys.argv[1]
SEEDS = [int(s) for s in sys.argv[2].split(",")]
RES = 512
t = SlideToTarget(cam_res=RES)
m = t.env.sim.model
gid = {m.geom_id2name(g): g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(("zone_1", "cube_1"))}
CUBE_G = gid["cube_1_g1"]
L = {"left": gid["zone_1_g1"], "right": gid["zone_1_g2"], "far": gid["zone_1_g3"], "near": gid["zone_1_g4"]}


def jpeg(a, q=90):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", quality=q); b.seek(0)
    return np.asarray(Image.open(b).convert("RGB")).astype(int)


def seg(cam="agentview"):
    s = t.env.sim.render(RES, RES, camera_name=cam, segmentation=True)[::-1]
    return np.where(s[..., 0] == 5, s[..., 1], -1)


def place(xy, z=None):
    envs.set_obj_pose(t.env, "cube_1", [xy[0], xy[1], envs.TABLE_Z + CUBE_HALF if z is None else z])
    qa, da = envs.free_joint_addr(t.env, "cube_1")
    t.env.sim.data.qvel[da:da + 6] = 0
    t.env.sim.forward()


def red(rgb):
    return (rgb[..., 0] > rgb[..., 1] + 50) & (rgb[..., 0] > rgb[..., 2] + 50)


def far_metrics(g, gref, rgbj):
    cube = g == CUBE_G
    ref = gref == L["far"]
    cols = [c for c in range(RES) if cube[:, c].any() and ref[:, c].any()]
    cols = cols[2:-2]
    seg_over, col_over, vis = [], [], []
    rj = red(rgbj)
    for c in cols:
        outer = np.nonzero(ref[:, c])[0].max()
        seg_over.append(int(np.nonzero(cube[:, c])[0].max()) - int(outer))
        k = 0
        while outer + 1 + k < RES and rj[outer + 1 + k, c]:
            k += 1
        col_over.append(k)
        vis.append(int((g[:, c] == L["far"]).sum()))
    return dict(seg_over=float(np.median(seg_over)), jpeg_red_rows=float(np.median(col_over)),
                line_vis=float(np.median(vis)), line_ref=float(np.median([ref[:, c].sum() for c in cols])))


def side_metrics(g, gref, rgbj, name):
    cube = g == CUBE_G
    ref = gref == L[name]
    rows = [r for r in range(RES) if cube[r, :].any() and ref[r, :].any()][2:-2]
    rj = red(rgbj)
    so, co, vis = [], [], []
    for r in rows:
        if name == "left":  # image right
            outer = np.nonzero(ref[r, :])[0].max(); so.append(int(np.nonzero(cube[r, :])[0].max()) - int(outer))
            k = 0
            while outer + 1 + k < RES and rj[r, outer + 1 + k]:
                k += 1
        else:
            outer = np.nonzero(ref[r, :])[0].min(); so.append(int(outer) - int(np.nonzero(cube[r, :])[0].min()))
            k = 0
            while outer - 1 - k >= 0 and rj[r, outer - 1 - k]:
                k += 1
        co.append(k); vis.append(int((g[r, :] == L[name]).sum()))
    if not rows:
        return {}
    return dict(seg_over=float(np.max(so)), seg_over_med=float(np.median(so)), jpeg_red_cols=float(np.median(co)),
                line_vis=float(np.median(vis)), line_vis_min=float(np.min(vis)))


rows = []
for seed in SEEDS:
    inst = t.sample_instance(seed)
    t.reset_instance(inst)
    x0, y0 = inst["cube_xy"]; tx = x0 + inst["target_dist"]
    place([x0, y0 + 0.6], z=envs.TABLE_Z + 0.5)
    gref = seg()
    for a in np.round(np.arange(0.014, 0.0261, 0.001), 4):
        place([tx + a, y0])
        g = seg(); rgbj = jpeg(envs.render(t.env, "agentview", RES))
        rows.append(dict(seed=seed, y0=y0, kind="far", off_cm=round(100 * a, 1), **far_metrics(g, gref, rgbj)))
    for side, sgn in (("left", 1), ("right", -1)):
        for b in np.round(np.arange(0.010, 0.0221, 0.001), 4):
            place([tx, y0 + sgn * b])
            g = seg(); rgbj = jpeg(envs.render(t.env, "agentview", RES))
            rows.append(dict(seed=seed, y0=y0, kind=side, off_cm=round(100 * b, 1), **side_metrics(g, gref, rgbj, side)))
    print(seed, flush=True)
json.dump(rows, open(OUT, "w"), indent=0)
