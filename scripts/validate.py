"""Headroom validation of LFF-Bench tasks with scripted reference policies.

python scripts/validate.py --task l5_slide_to_target --n 20 --k 5
Writes results/<task>/validate.json and thumbnails.
"""
import argparse
import importlib
import json
import os
import pkgutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import imageio  # noqa: E402

import lffbench.tasks as task_pkg  # noqa: E402
from lffbench.task_base import TASKS  # noqa: E402

for m in pkgutil.iter_modules(task_pkg.__path__):
    importlib.import_module(f"lffbench.tasks.{m.name}")


def succ_curve(results, k):
    """run-level succ@j for j = 1..k (a run = one instance, attempts share the run's state)."""
    out = []
    for j in range(1, k + 1):
        out.append(float(np.mean([r is not None and r <= j for r in results])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--kinds", default="oracle,naive,adaptive,blind")
    ap.add_argument("--thumbs", type=int, default=3)
    args = ap.parse_args()

    task = TASKS[args.task]()
    outdir = os.path.join(ROOT, "results", args.task)
    os.makedirs(outdir, exist_ok=True)
    kinds = args.kinds.split(",")
    first_success = {k: [] for k in kinds}
    runs = []
    t0 = time.time()
    for i in range(args.n):
        seed = args.seed0 + i
        inst = task.sample_instance(seed)
        if i < args.thumbs:
            task.reset_instance(inst)
            imageio.imwrite(os.path.join(outdir, f"init_{seed}.png"), task.snapshot(res=384))
        for kind in kinds:
            hist = task.run_scripted(kind, inst, k=args.k, seed=seed)
            s = next((h["attempt"] for h in hist if h["outcome"]["success"]), None)
            first_success[kind].append(s)
            runs.append(dict(seed=seed, kind=kind, inst={k: v for k, v in inst.items() if not k.startswith("_")},
                             history=hist))
            if i < args.thumbs and kind == "naive":
                imageio.imwrite(os.path.join(outdir, f"naive_end_{seed}.png"), task.snapshot(res=384))
        print(f"[{i + 1}/{args.n}] seed={seed} " + " ".join(f"{k}={first_success[k][-1]}" for k in kinds)
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    summary = {k: succ_curve(v, args.k) for k, v in first_success.items()}
    res = dict(task=args.task, level=task.level, capabilities=list(task.capabilities), n=args.n, k=args.k,
               succ_at_k=summary, runs=runs, seconds=time.time() - t0)
    with open(os.path.join(outdir, "validate.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: [round(x, 2) for x in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
