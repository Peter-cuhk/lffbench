"""Build the record of an earlier failed attempt for the full5 history arms (docs/14 §6).

  python scripts/make_dossier.py <run log dir> <out dir> [--attempt 1] [--allow-success]

The record contains only what the agent of that attempt saw: the starting images, every robot command with the
robot's reply (as logged by robot_server), the images those commands produced, the agent's closing remark and the
F1 verdict ("failed"). No outcome codes, measurements or hidden variables. The same builder is used for the own-history
arm (record of a failed independent run on the same instance) and the mismatched arm (record of a failed run on the
twin instance: same layout, different hidden variable), so the two dossiers differ only in content.
"""
import argparse
import json
import os
import shutil

DROP = ("state_file", "attempt_ended", "camera_file")


def shell_args(args):
    out = []
    for k, v in args.items():
        v = json.dumps(v) if not isinstance(v, str) else v
        out.append(f'{k}="{v}"' if (" " in v or not v) else f"{k}={v}")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--allow-success", action="store_true")
    a = ap.parse_args()
    evs = [json.loads(line) for line in open(os.path.join(a.log, "events.jsonl")) if line.strip()]
    start = next(e for e in evs if e["ev"] == "session_start")
    if start.get("style", "la") not in ("la", None) and "style" in start:
        raise SystemExit("only la-style runs are supported")
    calls = [e for e in evs if e["ev"] == "call" and e.get("attempt") == a.attempt]
    done = [e for e in evs if e["ev"] == "done" and e.get("attempt") == a.attempt]
    if not done:
        raise SystemExit(f"attempt {a.attempt} has no done event (run unfinished?)")
    done = done[0]
    if done["success"] and not a.allow_success:
        raise SystemExit(f"attempt {a.attempt} succeeded; nothing to learn from (use --allow-success to force)")
    imgs = os.path.join(a.log, "images")
    os.makedirs(os.path.join(a.out, "obs"), exist_ok=False)

    def copy(rel):
        base = os.path.basename(rel)
        src = os.path.join(imgs, f"a{a.attempt}_{base}")
        if not os.path.exists(src):
            return None
        shutil.copy(src, os.path.join(a.out, "obs", base))
        return f"obs/{base}"

    # starting images: the first images the agent saw (status at step 0)
    first = sorted(f for f in os.listdir(imgs) if f.startswith(f"a{a.attempt}_") and "s000_" in f)
    start_imgs = [copy(f.split("_", 1)[1]) for f in first]
    lines = ["# Record of the earlier attempt", "",
             "Starting scene: " + ", ".join(p for p in start_imgs if p), "",
             "## Robot commands and replies", ""]
    rec = []
    for i, e in enumerate(calls, 1):
        rep = {k: v for k, v in (e.get("report") or {}).items() if k not in DROP}
        if "images" in rep:
            rep["images"] = [p2 for p2 in (copy(p) for p in rep["images"]) if p2]
        cmd = f"./robot {e['cmd']} {shell_args(e.get('args') or {})}".strip()
        lines += [f"{i}. `{cmd}`", "   reply: `" + json.dumps(rep) + "`", ""]
        rec.append(dict(n=i, command=cmd, reply=rep))
    lines += [f"{len(calls) + 1}. `./robot finish reason=\"{done.get('reason', '')}\"`", "",
              "## Closing remark of that attempt", "", str(done.get("reason", "")), "",
              "## Result", "", "The environment judged that attempt: " + ("succeeded." if done["success"] else "failed."), ""]
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(a.out, "record.jsonl"), "w") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), os.path.basename(a.out) + ".source.json"), "w") as f:
        json.dump(dict(log=os.path.abspath(a.log), attempt=a.attempt, task=start.get("task"), seed=start.get("seed"),
                       arm=start.get("arm", "none"), n_commands=len(calls), success=done["success"]), f)
    print(json.dumps(dict(out=a.out, n_commands=len(calls), n_images=len(os.listdir(os.path.join(a.out, "obs"))))))


if __name__ == "__main__":
    main()
