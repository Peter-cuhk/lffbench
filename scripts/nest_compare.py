"""Render what an agent would see for l4_nest_order: start, the naive (tallest-outermost) attempt's end, the true
order's end. Agentview + a wrist view taken from above the stack (the agent can always move there and look).

python scripts/nest_compare.py --seeds 1000,1001,1002 --out results/l4_nest_order/compare
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lffbench import envs  # noqa: E402
from lffbench.tasks.l4_nest import NestOrder  # noqa: E402

RES = 512


def views(task, inst, look_at=None):
    """agentview, and the wrist camera ~30 cm above the table over `look_at` (cup index) or at the start pose"""
    a = envs.render(task.env, "agentview", RES)
    if look_at is not None:
        x, y = task._cup_state(inst, look_at)["p"][:2]
        task.sk.move_to([x, y, envs.TABLE_Z + 0.30], yaw=0.0, tol=0.003)
        envs.settle(task.env, 10)
    w = envs.render(task.env, "robot0_eye_in_hand", RES)
    return a, w


def row(label, a, w):
    im = Image.new("RGB", (2 * RES, RES + 22), "white")
    im.paste(Image.fromarray(a), (0, 22))
    im.paste(Image.fromarray(w), (RES, 22))
    ImageDraw.Draw(im).text((6, 5), label, fill="black")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000,1001,1002")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "l4_nest_order", "compare"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    task = NestOrder(cam_res=RES)
    for seed in [int(s) for s in args.seeds.split(",")]:
        inst = task.sample_instance(seed)
        col = [c["color"] for c in inst["cups"]]
        name = lambda order: " > ".join(col[j] for j in order)  # noqa: E731
        rows = []
        task.reset_instance(inst)
        rows.append(row(f"seed {seed} ({inst['case']}) start   heights cm {dict(zip(col, inst['height_cm']))}  "
                        f"openings cm {dict(zip(col, inst['opening_cm']))}", *views(task, inst)))
        for kind, order in (("naive (tallest outermost)", inst["naive_order"]), ("true order", inst["order"])):
            task.reset_instance(inst)
            task.execute(inst, dict(order=list(order)))
            o = task.outcome(inst)
            rows.append(row(f"{kind}: {name(order)} -> {'SUCCESS' if o['success'] else 'FAIL'} | {o['detail']}"[:150],
                            *views(task, inst, look_at=order[0])))
        sheet = Image.new("RGB", (2 * RES, len(rows) * (RES + 22)), "white")
        for k, r in enumerate(rows):
            sheet.paste(r, (0, k * (RES + 22)))
        p = os.path.join(args.out, f"compare_{seed}.jpg")
        sheet.save(p, quality=88)
        print(p, flush=True)


if __name__ == "__main__":
    main()
