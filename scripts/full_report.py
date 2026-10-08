"""Summarise a full run (all tasks x harnesses) from the robot-server logs only, so every harness is measured
the same way.

python scripts/full_report.py --batch full1 [--out ../docs/full1_table.md]

Per run: outcome of every attempt, attempts used, robot commands, close_gripper calls that ended holding an
object, "empty done" attempts (the agent ended an attempt without a single move / gripper command in it), and
wall time from the first to the last server request.
"""
import argparse
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVEL = {"l1": "L1", "l2": "L2", "l3": "L3", "l4": "L4", "l5": "L5", "l6": "L6", "l7": "L7"}
HARNESS_NAME = {"gpt_player": "ours", "rpent": "RPent", "gptpolicy": "GPT-Policy", "claude": "ours (Claude CLI)"}


def run_info(r):
    ev = [json.loads(l) for l in open(os.path.join(r["log"], "events.jsonl"))]
    calls = [e for e in ev if e["ev"] == "call"]
    dones = [e for e in ev if e["ev"] == "done"]
    http = [e for e in ev if e["ev"] == "http"]
    # v1: move_to / open_gripper / close_gripper; v2 and la: move_eef [gripper=open|close] (holding is only in `raw`,
    # the agent never sees it)
    acts_per_attempt = collections.Counter(e["attempt"] for e in calls if e["cmd"] in ("move_to", "open_gripper", "close_gripper")
                                           or (e["cmd"] == "move_eef" and (e.get("report") or {}).get("ok")))
    empty = sum(1 for d in dones if acts_per_attempt.get(d["attempt"], 0) == 0)
    closes = [e for e in calls if e["cmd"] == "close_gripper"
              or (e["cmd"] == "move_eef" and (e.get("args") or {}).get("gripper") == "close" and (e.get("report") or {}).get("ok"))]
    held = sum(1 for e in closes if (e.get("report") or {}).get("holding_object") or (e.get("raw") or {}).get("holding_object"))
    first = next((d["attempt"] for d in dones if d["success"]), None)
    wall = (http[-1]["t_send"] - http[0]["t_recv"]) / 60 if http else None
    return dict(outcomes="".join("S" if d["success"] else "F" for d in dones), attempts=len(dones), first=first,
                cmds=len(calls), closes=len(closes), held=held, empty=empty, wall=wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="full1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", default="gpt", help="runs/<runs>/<batch>/manifest.json (gpt or claude)")
    a = ap.parse_args()
    m = json.load(open(os.path.join(ROOT, "runs", a.runs, a.batch, "manifest.json")))
    rows = {}
    for r in m["runs"]:
        try:
            h = r.get("harness") or ("claude" if r.get("cond") in ("harness", "noharness") else "gpt_player")
            rows[(r["task"], h)] = dict(run_info(r), k=r["k"])
        except FileNotFoundError:
            pass
    tasks = sorted({t for t, _ in rows}, key=lambda t: (t[:2], t))
    hs = [h for h in ("gpt_player", "rpent", "gptpolicy", "claude") if any(k[1] == h for k in rows)]
    L = [f"# {a.batch}: one run per task x harness (see manifest for model / perception / style)", "",
         "Cell = outcome of each attempt (S success, F failure); `·` = run missing. Measured from robot-server logs.", "",
         "| task | level | " + " | ".join(HARNESS_NAME[h] for h in hs) + " |", "|---|---|" + "---|" * len(hs)]
    for t in tasks:
        cells = []
        for h in hs:
            x = rows.get((t, h))
            cells.append("·" if x is None else (x["outcomes"] or "—"))
        L.append(f"| {t} | {LEVEL.get(t[:2], '?')} | " + " | ".join(cells) + " |")
    L += ["", "| harness | runs | success @1 | success @3 | grasps held / close_gripper | empty-done attempts | "
          "commands / attempt | wall min / run |", "|---|---|---|---|---|---|---|---|"]
    for h in hs:
        xs = [x for (t, hh), x in rows.items() if hh == h and x["attempts"] > 0]
        n = len(xs)
        att = sum(x["attempts"] for x in xs)
        L.append(f"| {HARNESS_NAME[h]} | {n} | {sum(1 for x in xs if x['first'] == 1)} | "
                 f"{sum(1 for x in xs if x['first'])} | {sum(x['held'] for x in xs)} / {sum(x['closes'] for x in xs)} | "
                 f"{sum(x['empty'] for x in xs)} / {att} | {sum(x['cmds'] for x in xs) / max(att, 1):.1f} | "
                 f"{sum(x['wall'] or 0 for x in xs) / max(n, 1):.0f} |")
    by_level = collections.defaultdict(lambda: [0, 0])
    for (t, h), x in rows.items():
        if x["attempts"]:
            by_level[LEVEL.get(t[:2], "?")][0] += bool(x["first"])
            by_level[LEVEL.get(t[:2], "?")][1] += 1
    L += ["", "Success @3 by level (all harnesses pooled): " +
          ", ".join(f"{lv} {s}/{n}" for lv, (s, n) in sorted(by_level.items()))]
    txt = "\n".join(L)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
