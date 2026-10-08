"""Check that this machine can run LFF-Bench. Run after `source run_env.sh`:

  python scripts/check_setup.py            # Python packages, LIBERO config, simulation + rendering, the agent's shell jail
  python scripts/check_setup.py --api      # ... plus one tiny text-only request to the model endpoint (OPENAI_API_KEY /
                                           #     OPENAI_BASE_URL from the environment or from .env); costs a few tokens
  python scripts/check_setup.py --api --model gpt-5.5 --skip-sim --skip-jail

Exit code 0 when every check that ran passed.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
JAIL = "/opt/lffjail/jail_inner.sh"  # same path as lffbench/agent/robot_server.py and scripts/gpt_player.py

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    return ok


def check_packages():
    report("python 3.10", sys.version_info[:2] == (3, 10), sys.version.split()[0])
    from lffbench.agent.backends.base import import_openai
    for mod in ("numpy", "mujoco", "robosuite", "libero.libero", "openai", "imageio", "cv2"):
        try:
            m = import_openai() if mod == "openai" else __import__(mod, fromlist=["_"])
            report(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            report(f"import {mod}", False, f"{type(e).__name__}: {e}")
    cfg = os.path.join(os.environ.get("LIBERO_CONFIG_PATH", ""), "config.yaml")
    if report("LIBERO config", os.path.isfile(cfg), cfg + ("" if os.path.isfile(cfg) else " missing: run setup/install.sh")):
        import yaml
        assets = yaml.safe_load(open(cfg)).get("assets", "")
        report("LIBERO assets", os.path.isdir(assets), assets)
    report("rendering backend", bool(os.environ.get("MUJOCO_GL")), f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")


def check_sim(task_name, seed):
    import importlib
    import pkgutil

    try:
        import lffbench.tasks as task_pkg
        from lffbench.task_base import TASKS
    except Exception as e:  # noqa: BLE001
        report("import lffbench", False, f"{type(e).__name__}: {e}")
        return
    try:
        for m in pkgutil.iter_modules(task_pkg.__path__):
            importlib.import_module(f"lffbench.tasks.{m.name}")
    except Exception as e:  # noqa: BLE001
        report("import tasks", False, f"{type(e).__name__}: {e}")
        return
    report("tasks registered", len(TASKS) >= 20, f"{len(TASKS)} tasks")
    t0 = time.time()
    try:
        task = TASKS[task_name]()
        inst = task.sample_instance(seed)
        task.reset_instance(inst)
        img = task.snapshot()
        out = task.outcome(inst)
    except Exception as e:  # noqa: BLE001
        report(f"simulate + render {task_name}", False, f"{type(e).__name__}: {e}")
        return
    ok = img.ndim == 3 and img.shape[2] == 3 and float(img.std()) > 1.0
    report(f"simulate + render {task_name}", ok,
           f"image {img.shape}, pixel std {img.std():.1f}, untouched scene success={out['success']}, {time.time() - t0:.1f} s")


def check_jail():
    if not report("jail installed", os.access(JAIL, os.X_OK) and os.path.isdir("/opt/lffjail/py"),
                  JAIL if os.path.exists(JAIL) else "missing: sudo bash jail/setup_jail.sh"):
        return
    probe = r'''
echo "pwd=$(pwd)"
python3 -c "import numpy, scipy, PIL, cv2; print('py=ok')" 2>&1 | tail -1
echo "mnt_entries=$(ls -A /mnt 2>/dev/null | wc -l) home_entries=$(ls -A /home 2>/dev/null | wc -l)"
python3 -c "
import socket
s = socket.socket(); s.settimeout(3)
try:
    s.connect(('1.1.1.1', 443)); print('net=open')
except OSError as e:
    print('net=blocked')
"
touch made_in_jail && echo "write=ok"
'''
    with tempfile.TemporaryDirectory(prefix="lffjail_check_") as ws:
        argv = ["unshare", "--user", "--map-root-user", "--mount", "--net", "--", JAIL, ws,
                "timeout", "-k", "5", "60", "bash", "-c", probe]
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        kv = dict(l.split("=", 1) for l in r.stdout.split() if "=" in l)
        if r.returncode != 0 and not kv:
            hint = ""
            if "Operation not permitted" in r.stdout or "uid_map" in r.stdout:
                hint = (" | unprivileged user namespaces are blocked; on Ubuntu >= 23.10: "
                        "sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0")
            report("jail runs", False, r.stdout.strip()[-400:] + hint)
            return
        report("jail runs", True, f"workdir {kv.get('pwd')}")
        report("jail python (numpy/scipy/Pillow/OpenCV)", kv.get("py") == "ok", kv.get("py", r.stdout[-300:]))
        report("jail hides host data", kv.get("mnt_entries") == "0" and kv.get("home_entries") == "0",
               f"/mnt entries {kv.get('mnt_entries')}, /home entries {kv.get('home_entries')}")
        report("jail has no network", kv.get("net") == "blocked", kv.get("net", "?"))
        report("jail writes the workspace", kv.get("write") == "ok" and os.path.exists(os.path.join(ws, "made_in_jail")))


def load_env_file():
    path = os.environ.get("LFF_ENV_FILE") or os.path.join(ROOT, ".env")
    if not os.path.isfile(path):
        return None
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.removeprefix("export ").strip(), v.strip().strip("'\""))
    return path


def check_api(model, effort):
    src = load_env_file()
    if not os.environ.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
        os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"  # same as scripts/gpt_player.py
    if not report("OPENAI_API_KEY set", bool(os.environ.get("OPENAI_API_KEY")),
                  f"from {src}" if src else "from the environment (no .env)"):
        return
    base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1 (default)"
    from lffbench.agent.backends.openai_responses import OpenAIResponsesBackend
    t0 = time.time()
    try:
        client = OpenAIResponsesBackend(model=model, reasoning_effort=effort, timeout=120.0, max_retries=1).client()
        resp = client.responses.create(model=model, instructions="Reply with exactly the word OK.", input="ping",
                                       reasoning=dict(effort=effort), store=False, max_output_tokens=256)
        text = (resp.output_text or "").strip()
        u = resp.usage
        report(f"API {model} @ {base}", bool(text), f"reply {text[:40]!r}, {time.time() - t0:.1f} s, "
               f"tokens in {u.input_tokens} / out {u.output_tokens}")
    except Exception as e:  # noqa: BLE001
        report(f"API {model} @ {base}", False, f"{type(e).__name__}: {str(e)[:300]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", action="store_true", help="also send one tiny request to the model endpoint")
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--task", default="l5_slide_to_target")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-jail", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    check_packages()
    if not a.skip_sim:
        check_sim(a.task, a.seed)
    if not a.skip_jail:
        check_jail()
    if a.api:
        check_api(a.model, a.effort)
    n_bad = results.count(False)
    print(f"\n{len(results) - n_bad}/{len(results)} checks passed")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
