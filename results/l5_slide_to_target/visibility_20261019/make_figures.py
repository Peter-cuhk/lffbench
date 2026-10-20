"""Figures for results/l5_slide_to_target/visibility_20261019/: the cube teleported to given centre offsets from the
square's centre (along = away from the robot, towards the agentview camera), rendered as the server does (512 x 512,
envs.render, JPEG q90). Crops are 80 x 60 px around the square, scaled x6 (nearest)."""
import io, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lffbench.tasks.l5_slide import SlideToTarget, CUBE_HALF, Z_PUSH
from lffbench import envs
from lffbench.agent import camera as C

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
RES = 512
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
t = SlideToTarget(cam_res=RES)
m = t.env.sim.model
gid = {m.geom_id2name(g): g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(("zone_1", "cube_1"))}
CUBE_G, FAR = gid["cube_1_g1"], gid["zone_1_g3"]


def jpeg(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="JPEG", quality=90); b.seek(0)
    return Image.open(b).convert("RGB")


def place(xy, z=None):
    envs.set_obj_pose(t.env, "cube_1", [xy[0], xy[1], envs.TABLE_Z + CUBE_HALF if z is None else z])
    qa, da = envs.free_joint_addr(t.env, "cube_1")
    t.env.sim.data.qvel[da:da + 6] = 0
    t.env.sim.forward()


def seg():
    s = t.env.sim.render(RES, RES, camera_name="agentview", segmentation=True)[::-1]
    return np.where(s[..., 0] == 5, s[..., 1], -1)


def far_over(g, gref):
    cube, ref = g == CUBE_G, gref == FAR
    cols = [c for c in range(RES) if cube[:, c].any() and ref[:, c].any()][2:-2]
    return int(np.median([np.nonzero(cube[:, c])[0].max() - np.nonzero(ref[:, c])[0].max() for c in cols]))


def verdict(a, past):
    return "in" if -0.0100001 <= a <= past + 1e-7 else ("past" if a > 0 else "short")


def grid(tiles, ncol, path, title):
    w, h = tiles[0][0].size
    nrow = (len(tiles) + ncol - 1) // ncol
    lab = 78
    M = Image.new("RGB", (ncol * (w + 8), 44 + nrow * (h + lab)), "white")
    d = ImageDraw.Draw(M)
    d.text((6, 8), title, fill="black", font=F)
    for i, (im, lines) in enumerate(tiles):
        x, y = (i % ncol) * (w + 8), 44 + (i // ncol) * (h + lab)
        M.paste(im, (x, y))
        for k, (txt, col) in enumerate(lines):
            d.text((x + 4, y + h + 2 + 24 * k), txt, fill=col, font=FS)
    M.save(path)


def agent_crops(seed, offsets, path, title):
    inst = t.sample_instance(seed); t.reset_instance(inst)
    x0, y0 = inst["cube_xy"]; tx = x0 + inst["target_dist"]
    place([x0, y0 + 0.6], z=envs.TABLE_Z + 0.5); gref = seg()
    u, v = [int(round(q)) for q in C.world_to_pixel(t.env, "agentview", [tx, y0, envs.TABLE_Z], RES)[:2]]
    tiles = []
    for a in offsets:
        place([tx + a, y0])
        im = jpeg(envs.render(t.env, "agentview", RES)).crop((u - 40, v - 36, u + 40, v + 24)).resize((480, 360), Image.NEAREST)
        old, new = verdict(a, 0.01), verdict(a, 0.02)
        lines = [(f"{100 * a:+.1f} cm   overrun {far_over(seg(), gref):+d} px" if a > 0 else f"{100 * a:+.1f} cm", "black"),
                 (f"old: {old}   new: {new}", "black" if old == new else "#b00020")]
        tiles.append((im, lines))
    grid(tiles, 4, path, title)


# 1. far side, seed 3000 (main table), agentview
agent_crops(3000, [0.0, 0.010, 0.014, 0.016, 0.018, 0.020, 0.022, 0.026],
            os.path.join(OUT, "agentview_far_s3000.png"),
            "seed 3000 agentview (512x512 JPEG q90, crop x6): cube centre past the square's centre; "
            "overrun = red rows beyond the far line's outer edge")
# 2. near side, seed 3004 (naive stops 1.66 cm short)
agent_crops(3004, [-0.010, -0.0125, -0.0166, -0.020, -0.025, -0.030, -0.040, 0.0],
            os.path.join(OUT, "agentview_near_s3004.png"),
            "seed 3004 agentview: cube centre short of the square's centre (naive 0.30 m/s strike stops at -1.66 cm); "
            "the near line is hidden by the cube in every frame")
# 3. full frames as the agent gets them
inst = t.sample_instance(3000); t.reset_instance(inst)
x0, y0 = inst["cube_xy"]; tx = x0 + inst["target_dist"]
fr = []
for a in (0.010, 0.020, 0.026):
    place([tx + a, y0]); fr.append(jpeg(envs.render(t.env, "agentview", RES)))
M = Image.new("RGB", (3 * 512 + 16, 512 + 40), "white"); d = ImageDraw.Draw(M)
for i, (im, s) in enumerate(zip(fr, ("+1.0 cm (old limit)", "+2.0 cm (new limit)", "+2.6 cm (fails, 4 px)"))):
    M.paste(im, (i * 520, 40)); d.text((i * 520 + 6, 8), "seed 3000 agentview 1:1, " + s, fill="black", font=FS)
M.save(os.path.join(OUT, "agentview_full_s3000.png"))
# 4. wrist camera from the scripted post-strike pose (above the cube's start, 8 cm above the push height)
for seed in (3000,):
    inst = t.sample_instance(seed); t.reset_instance(inst)
    x0, y0 = inst["cube_xy"]; tx = x0 + inst["target_dist"]
    t.sk.set_gripper(True, steps=5); t.sk.move_to([x0 - CUBE_HALF + 0.015, y0, Z_PUSH + 0.08], yaw=0.0)
    tiles = []
    for a in (-0.010, -0.014, -0.0166, -0.020, 0.0, 0.010, 0.016, 0.020):
        place([tx + a, y0])
        im = jpeg(envs.render(t.env, "robot0_eye_in_hand", RES)).crop((96, 0, 416, 300)).resize((320, 300))
        tiles.append((im, [(f"{100 * a:+.1f} cm", "black"),
                           (f"old: {verdict(a, 0.01)}   new: {verdict(a, 0.02)}",
                            "black" if verdict(a, 0.01) == verdict(a, 0.02) else "#b00020")]))
    grid(tiles, 4, os.path.join(OUT, f"wrist_post_strike_s{seed}.png"),
         f"seed {seed} wrist camera after a strike (gripper above the cube's start): near side visible, far side not")
print("ok")
