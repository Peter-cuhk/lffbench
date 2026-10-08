"""Record a scripted rollout of a task as an mp4 (agentview | wrist), with title cards between attempts.

python scripts/make_video.py --task l1_bias_place --kind adaptive --seed 1000
"""
import argparse
import importlib
import os
import pkgutil
import sys
import textwrap

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import lffbench.tasks as task_pkg  # noqa: E402
from lffbench import envs  # noqa: E402
from lffbench.task_base import TASKS  # noqa: E402

for m in pkgutil.iter_modules(task_pkg.__path__):
    importlib.import_module(f"lffbench.tasks.{m.name}")

RES = 320
FPS = 15


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def header(img, lines):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18 * len(lines) + 6], fill=(0, 0, 0))
    for i, s in enumerate(lines):
        d.text((6, 3 + 18 * i), s, fill=(255, 255, 255), font=font(15))
    return np.asarray(im)


def card(lines, w, h, color=(20, 20, 20)):
    im = Image.new("RGB", (w, h), color)
    d = ImageDraw.Draw(im)
    y = 20
    for s in lines:
        for part in textwrap.wrap(s, 70) or [""]:
            d.text((20, y), part, fill=(255, 255, 255), font=font(18))
            y += 24
        y += 6
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--kind", default="adaptive")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--every", type=int, default=3, help="record one frame every N control steps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = TASKS[args.task]()
    env = task.env
    inst = task.sample_instance(args.seed)
    frames = []
    state = dict(n=0, attempt=1)
    W = 2 * RES

    def grab():
        a = envs.render(env, "agentview", RES)
        w = envs.render(env, "robot0_eye_in_hand", RES)
        img = np.concatenate([a, w], axis=1)
        return header(img, [f"{args.task}  [{task.level}]  policy={args.kind}  seed={args.seed}",
                            f"attempt {state['attempt']}   t={state['n'] * 0.05:5.1f}s (sim)   left: agentview  right: wrist"])

    orig_step = env.step

    def step(a):
        out = orig_step(a)
        state["n"] += 1
        if state["n"] % args.every == 0:
            frames.append(grab())
        return out

    env.step = step

    def on_attempt(a, params, out):
        txt = [f"Attempt {a + 1}: {'SUCCESS' if out['success'] else 'FAILED'}"]
        if out.get("detail"):
            txt.append("Feedback (F2): " + out["detail"])
        p = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (params or {}).items()
             if not isinstance(v, (list, dict)) or len(str(v)) < 120}
        if p:
            txt.append(f"params: {p}")
        if not out["success"]:
            txt.append(f"-> attempt {a + 2}")
        frames.extend([card(txt, W, RES, (40, 20, 20) if not out["success"] else (20, 50, 20))] * (FPS * 3))
        state["attempt"] = a + 2

    intro = card([f"Task: {args.task}  (level {task.level}; capabilities: {', '.join(task.capabilities)})",
                  f"Instruction: {task.instruction}",
                  f"Policy: {args.kind}  (scripted reference)",
                  "Hidden variables are not visible to the policy."], W, RES)
    frames.extend([intro] * (FPS * 3))
    task.reset_instance(inst)
    frames.append(grab())
    n_cards_before = len(frames)
    hist = task.run_scripted(args.kind, inst, k=args.k, seed=args.seed, on_attempt=on_attempt)
    # tasks whose run_scripted does not call on_attempt: summarise at the end
    if not any(f is not None and f.shape[0] == RES and (f[5, 5] == np.array([40, 20, 20])).all() for f in frames[n_cards_before:]) \
            and not any((f[5, 5] == np.array([20, 50, 20])).all() for f in frames[n_cards_before:]):
        lines = []
        for h in hist:
            o = h["outcome"]
            lines.append(f"Attempt {h['attempt']}: {'SUCCESS' if o['success'] else 'FAILED'} — {o.get('detail', '')}")
        frames.extend([card(lines[-6:], W, RES)] * (FPS * 4))
    out = args.out or os.path.join(ROOT, "videos", f"{args.task}_{args.kind}_s{args.seed}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimwrite(out, frames, fps=FPS, quality=7, macro_block_size=8)
    first = next((h["attempt"] for h in hist if h["outcome"]["success"]), None)
    print(f"wrote {out}  frames={len(frames)}  attempts={len(hist)}  first_success={first}")


if __name__ == "__main__":
    main()
