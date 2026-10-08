"""Per-run table for LA-interface (LIBERO-Agent-style) GPT runs: outcomes from the robot-server log, tool use and
token cost from the player transcript.

python scripts/la_report.py --batch la1_rgb la1_rgbd [--out ../docs/la1_table.md]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from full_report import run_info  # noqa: E402


def transcript_stats(log):
    p = os.path.join(log, "gpt_transcript.jsonl")
    st = dict(turns=0, shell=0, view=0, robot=0, auto_finish=0, api_err=0, inp=0, cached=0, out=0, reason=0, ended=None)
    if not os.path.exists(p):
        au = os.path.join(log, "audit.json")  # Claude subagent runs: tool counts from the audit of its transcript
        if os.path.exists(au):
            d = json.load(open(au))
            bt = d.get("by_tool", {})
            st.update(turns=d.get("n_calls", 0), view=bt.get("Read", 0), shell=bt.get("Bash", 0),
                      ended=f"audited ({d.get('n_violations')} violations, memory_injected={d.get('memory_injected')})")
        return st
    for l in open(p):
        d = json.loads(l)
        ev = d["ev"]
        if ev == "assistant":
            st["turns"] += 1
            u = d.get("usage") or {}
            st["inp"] += u.get("input_tokens", 0)
            st["cached"] += u.get("cached_tokens", 0)
            st["out"] += u.get("output_tokens", 0)
            st["reason"] += u.get("reasoning_tokens", 0)
        elif ev == "tool":
            c = d.get("command", "")
            if c.startswith("shell:"):
                st["shell"] += 1
            elif c.startswith("view_images"):
                st["view"] += 1
            else:
                st["robot"] += 1
        elif ev == "user" and d.get("auto_finish"):
            st["auto_finish"] += 1
        elif ev == "api_error":
            st["api_err"] += 1
        elif ev in ("end", "abort"):
            st["ended"] = ev if ev == "end" else "abort: " + d.get("reason", "")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", default="gpt", help="runs/<runs>/<batch> (gpt or claude); for claude, 'shell' = all Bash "
                                                  "calls (robot + shell) and 'view' = Read calls, from audit.json")
    a = ap.parse_args()
    L = ["| batch | task | perception | outcomes | robot cmds | grasps held/closed | shell | view | turns | "
         "auto-finish | input Mtok (cached) | output ktok (reasoning) | wall min | state |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = dict(inp=0, cached=0, out=0, reason=0)
    for b in a.batch:
        m = json.load(open(os.path.join(ROOT, "runs", a.runs, b, "manifest.json")))
        for r in m["runs"]:
            try:
                x = run_info(r)
            except FileNotFoundError:
                continue
            t = transcript_stats(r["log"])
            for k in tot:
                tot[k] += t[k]
            task = r["task"] + (" (nominal)" if r.get("nominal") else "")
            L.append(f"| {b} | {task} | {r.get('perception')} | {x['outcomes'] or '—'} | {x['cmds']} | "
                     f"{x['held']}/{x['closes']} | {t['shell']} | {t['view']} | {t['turns']} | {t['auto_finish']} | "
                     f"{t['inp'] / 1e6:.2f} ({t['cached'] / 1e6:.2f}) | {t['out'] / 1e3:.0f} ({t['reason'] / 1e3:.0f}) | "
                     f"{(x['wall'] or 0):.0f} | {t['ended'] or 'running'} |")
    L.append("")
    L.append(f"Total tokens: input {tot['inp'] / 1e6:.2f} M (cached {tot['cached'] / 1e6:.2f} M), "
             f"output {tot['out'] / 1e3:.0f} k (reasoning {tot['reason'] / 1e3:.0f} k).")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
