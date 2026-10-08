"""Seed-3000 instance check: hidden params + scripted naive/oracle first attempt. Writes nothing into the repo."""
import importlib, json, pkgutil, sys, os
ROOT = "/mnt/cpfs/workspace/agentic-robotics/lffbench"
sys.path.insert(0, ROOT)
import lffbench.tasks as task_pkg
from lffbench.task_base import TASKS
for m in pkgutil.iter_modules(task_pkg.__path__):
    importlib.import_module(f"lffbench.tasks.{m.name}")
name, seed = sys.argv[1], int(sys.argv[2])
task = TASKS[name]()
inst = task.sample_instance(seed)
def short(v):
    try:
        s = json.dumps(v, default=lambda o: getattr(o, "tolist", lambda: str(o))())
    except Exception:
        s = str(v)
    return s[:300]
out = {"task": name, "seed": seed, "inst": {k: short(v) for k, v in (inst.items() if isinstance(inst, dict) else vars(inst).items())}}
for kind in ("naive", "oracle"):
    try:
        hist = task.run_scripted(kind, inst, k=1, seed=seed)
        out[kind] = short(hist)
    except Exception as e:
        out[kind] = f"ERR {type(e).__name__}: {e}"
print(json.dumps(out, ensure_ascii=False))
