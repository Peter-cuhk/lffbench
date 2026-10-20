"""Teleport the cube around the target square on main-table instances and quantify, from the 512x512 agentview the
server sends (same camera, same envs.render, JPEG q90), whether 'not completely inside' is visible.
Writes sweep.json (+ crops/) to the out dir."""
import io, json, os, sys
import numpy as np
from PIL import Image
from lffbench.tasks.l5_slide import SlideToTarget, CUBE_HALF, INNER_HALF, LINE_W, Z_PUSH
from lffbench import envs
from lffbench.agent import camera as C

OUT = sys.argv[1]
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [3000, 3001, 3003, 3004, 3007]
os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
RES = 512
t = SlideToTarget(cam_res=RES)
m = t.env.sim.model
gid = {m.geom_id2name(g): g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(("zone_1", "cube_1"))}
CUBE_G = gid["cube_1_g1"]
LINES = {"left": gid["zone_1_g1"], "right": gid["zone_1_g2"], "far": gid["zone_1_g3"], "near": gid["zone_1_g4"]}
FILL = gid["zone_1_g0"]


def jpeg(a, q=90):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", quality=q); b.seek(0)
    return np.asarray(Image.open(b).convert("RGB"))


def render(cam):
    rgb = envs.render(t.env, cam, RES)
    seg = t.env.sim.render(RES, RES, camera_name=cam, segmentation=True)[::-1]
    g = np.where(seg[..., 0] == 5, seg[..., 1], -1)
    return rgb, g


def place(xy, z=None):
    envs.set_obj_pose(t.env, "cube_1", [xy[0], xy[1], envs.TABLE_Z + CUBE_HALF if z is None else z])
    qa, da = envs.free_joint_addr(t.env, "cube_1")
    t.env.sim.data.qvel[da:da + 6] = 0
    t.env.sim.forward()


def green_dark(rgb):
    """line colour (0.1,0.55,0.15) -> ~ (26,140,38) lit; classify on the JPEG image"""
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    return (g > r + 40) & (g > b + 40) & (g < 175)


def metrics(g, gref, rgb, rgbref):
    """g / gref: geom id images with / without the cube.  For each line: how many of its pixels inside the cube's
    image span are hidden, how far (px) the cube silhouette extends past the line's outer edge."""
    cube = g == CUBE_G
    out = dict(cube_px=int(cube.sum()))
    if not cube.any():
        return out
    rows, cols = np.nonzero(cube)
    c0, c1 = cols.min(), cols.max()
    r0, r1 = rows.min(), rows.max()
    icols = [c for c in range(c0 + 2, c1 - 1) if cube[:, c].any()]
    irows = [r for r in range(r0 + 2, r1 - 1) if cube[r, :].any()]
    gd, gdref = green_dark(rgb), green_dark(rgbref)
    for name, lg in LINES.items():
        ref = gref == lg
        vis = g == lg
        d = {}
        if name in ("far", "near"):
            span = [c for c in icols if ref[:, c].any()]
            thick = [int(vis[:, c].sum()) for c in span]  # visible line rows in that column
            ref_thick = [int(ref[:, c].sum()) for c in span]
            d["cols"] = len(span)
            d["ref_thick_med"] = float(np.median(ref_thick)) if span else None
            d["vis_thick_med"] = float(np.median(thick)) if span else None
            d["cols_line_gone"] = int(sum(1 for x in thick if x == 0))
            # dark-green pixels in the JPEG image around the line's rows (line rows +-2), per cube column
            d["jpeg_green_med"] = float(np.median([
                gd[max(0, np.nonzero(ref[:, c])[0].min() - 2):np.nonzero(ref[:, c])[0].max() + 3, c].sum()
                for c in span])) if span else None
            if name == "far":  # far line is nearer the camera: lower in the image (larger row)
                over = [int(np.nonzero(cube[:, c])[0].max()) - int(np.nonzero(ref[:, c])[0].max()) for c in span]
                inner = [int(np.nonzero(cube[:, c])[0].max()) - int(np.nonzero(ref[:, c])[0].min()) + 1 for c in span]
            else:  # near line: the cube's front-bottom edge vs the near line (only meaningful if the front is behind it)
                over = [int(np.nonzero(ref[:, c])[0].min()) - int(np.nonzero(cube[:, c])[0].max()) for c in span]
                inner = [int(np.nonzero(ref[:, c])[0].max()) - int(np.nonzero(cube[:, c])[0].max()) + 1 for c in span]
            d["overrun_outer_med"] = float(np.median(over)) if span else None
            d["overrun_inner_med"] = float(np.median(inner)) if span else None
        else:  # side lines: along image columns
            span = [r for r in irows if ref[r, :].any()]
            thick = [int(vis[r, :].sum()) for r in span]
            d["rows"] = len(span)
            d["ref_thick_med"] = float(np.median([int(ref[r, :].sum()) for r in span])) if span else None
            d["vis_thick_med"] = float(np.median(thick)) if span else None
            d["rows_line_gone"] = int(sum(1 for x in thick if x == 0))
            if span:
                if name == "left":  # +y -> image right (larger column)
                    over = [int(np.nonzero(cube[r, :])[0].max()) - int(np.nonzero(ref[r, :])[0].max()) for r in span if cube[r, :].any()]
                    inner = [int(np.nonzero(cube[r, :])[0].max()) - int(np.nonzero(ref[r, :])[0].min()) + 1 for r in span if cube[r, :].any()]
                else:
                    over = [int(np.nonzero(ref[r, :])[0].min()) - int(np.nonzero(cube[r, :])[0].min()) for r in span if cube[r, :].any()]
                    inner = [int(np.nonzero(ref[r, :])[0].max()) - int(np.nonzero(cube[r, :])[0].min()) + 1 for r in span if cube[r, :].any()]
                d["overrun_outer_max"] = int(max(over)) if over else None
                d["overrun_inner_max"] = int(max(inner)) if inner else None
        out[name] = d
    # green fill visible in front of the cube (between its front-bottom edge and the far line), median over columns
    farref = gref == LINES["far"]
    gaps = []
    for c in icols:
        if farref[:, c].any():
            gaps.append(int(np.nonzero(farref[:, c])[0].min()) - int(np.nonzero(cube[:, c])[0].max()) - 1)
    out["front_gap_px_med"] = float(np.median(gaps)) if gaps else None
    # square depth in image rows at the cube's centre column (inner edges of far and near lines)
    cc = int(np.median(cols))
    nr, fr = gref[:, cc] == LINES["near"], gref[:, cc] == LINES["far"]
    if nr.any() and fr.any():
        out["square_rows_at_centre"] = int(np.nonzero(fr)[0].min() - np.nonzero(nr)[0].max() - 1)
    return out


def crop(a, center_uv, half=40, scale=6):
    u, v = int(round(center_uv[0])), int(round(center_uv[1]))
    c = a[max(0, v - half):v + half, max(0, u - half):u + half]
    return np.asarray(Image.fromarray(c).resize((c.shape[1] * scale, c.shape[0] * scale), Image.NEAREST))


ALONG = [round(x, 4) for x in np.arange(-0.040, 0.0401, 0.002)]
ACROSS = [round(x, 4) for x in np.arange(-0.030, 0.0301, 0.002)]
rows = []
for seed in SEEDS:
    inst = t.sample_instance(seed)
    t.reset_instance(inst)
    x0, y0 = inst["cube_xy"]
    tx = x0 + inst["target_dist"]
    # no-cube reference: put the cube far away, outside the view
    place([x0, y0 + 0.6], z=envs.TABLE_Z + 0.5)
    ref = {}
    for cam in ("agentview", "robot0_eye_in_hand"):
        ref[cam] = render(cam)
    cuv = C.world_to_pixel(t.env, "agentview", [tx, y0, envs.TABLE_Z], RES)[:2]
    positions = [("along", a, 0.0) for a in ALONG] + [("across", 0.0, b) for b in ACROSS if b != 0.0]
    for kind, a, b in positions:
        place([tx + a, y0 + b])
        rgb, g = render("agentview")
        rj = jpeg(rgb)
        rec = dict(seed=seed, kind=kind, along_cm=round(100 * a, 2), across_cm=round(100 * b, 2),
                   inside=bool(abs(a) <= INNER_HALF - CUBE_HALF + 1e-9 and abs(b) <= INNER_HALF - CUBE_HALF + 1e-9))
        rec["agentview"] = metrics(g, ref["agentview"][1], rj, jpeg(ref["agentview"][0]))
        rows.append(rec)
        tag = f"s{seed}_{kind}_{'p' if (a if kind == 'along' else b) >= 0 else 'm'}{abs(100 * (a if kind == 'along' else b)):.1f}"
        Image.fromarray(rj).save(os.path.join(OUT, "crops", tag + "_full.jpg"), quality=95)
        Image.fromarray(crop(rj, cuv)).save(os.path.join(OUT, "crops", tag + "_crop.png"))
    print(seed, "done", flush=True)
json.dump(rows, open(os.path.join(OUT, "sweep.json"), "w"), indent=1)
