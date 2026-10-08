"""Render one player-subagent run as a self-contained HTML trace: everything the agent received,
every command it ran, every result and image it saw, its own text, plus the server-side ground truth.

python scripts/trace_html.py --batch exp1 --sandbox <hex> --out traces/<name>.html
"""
import argparse
import base64
import html
import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPTS = os.environ.get("CLAUDE_TASKS_DIR", "")  # the Claude Code session's tasks/ dir (<agentId>.output files)

CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#57534e;--card:#fff;--bd:#e7e5e4;--acc:#2563eb;--tool:#f5f5f4;--ok:#15803d;--bad:#b91c1c;--th:#7c3aed}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1c1917;--fg:#e7e5e4;--mut:#a8a29e;--card:#292524;--bd:#44403c;--acc:#60a5fa;--tool:#231f1d;--ok:#4ade80;--bad:#f87171;--th:#c4b5fd}}
body{background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;padding:16px;max-width:1100px;margin:auto}
h1{font-size:20px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid var(--bd);padding-bottom:4px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:10px 12px;margin:8px 0}
pre{white-space:pre-wrap;word-break:break-word;background:var(--tool);border:1px solid var(--bd);border-radius:6px;padding:8px;font:12px/1.45 ui-monospace,Menlo,monospace;margin:4px 0;max-height:420px;overflow:auto}
.lbl{font-size:12px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.03em}
.cmd{color:var(--acc)}.think{color:var(--th)}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
img{max-width:256px;border-radius:4px;border:1px solid var(--bd);margin:2px}
details>summary{cursor:pointer;color:var(--mut)}
table{border-collapse:collapse}td,th{border:1px solid var(--bd);padding:3px 8px;text-align:left;font-size:13px}
.step{display:flex;gap:8px}.n{color:var(--mut);min-width:34px;font-variant-numeric:tabular-nums}
"""


def esc(s):
    return html.escape(str(s))


def ts(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="exp1")
    ap.add_argument("--sandbox", required=True, help="sandbox hex id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    m = json.load(open(os.path.join(ROOT, "runs", "claude", a.batch, "manifest.json")))
    run = next(r for r in m["runs"] if r["sandbox"].endswith(a.sandbox))
    tr = [json.loads(l) for l in open(os.path.join(TRANSCRIPTS, f"{run['agent']}.output"))]
    ev = [json.loads(l) for l in open(os.path.join(run["log"], "events.jsonl"))]
    start = next(e for e in ev if e["ev"] == "session_start")
    dones = [e for e in ev if e["ev"] == "done"]
    http = [e for e in ev if e["ev"] == "http"]
    think = [h["think_s"] for h in http if h.get("think_s")]
    srv = [h["server_s"] for h in http]

    out = [f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(a.title or 'Agent trace')}</title><style>{CSS}</style></head><body>"]
    out.append(f"<h1>{esc(a.title or run['run'])}</h1>")
    out.append("<div class='card'><table>")
    rows = [("任务", run["task"]), ("种子", run["seed"]), ("模型", run["model"]), ("条件", run["cond"] + f"（k={run['k']}）"),
            ("沙箱", run["sandbox"]), ("服务端 run 名", run["run"]),
            ("结果", "；".join(f"第{d['attempt']}次 {'成功' if d['success'] else '失败'}" for d in dones)),
            ("命令数 / 模型思考总时长 / 仿真执行总时长", f"{sum(1 for e in ev if e['ev']=='call')} / {sum(think):.0f} s / {sum(srv):.0f} s"),
            ("子代理墙钟", f"{(ts(tr[-1]['timestamp']) - ts(tr[0]['timestamp'])) / 60:.1f} min")]
    for k, v in rows:
        out.append(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>")
    out.append("</table></div>")
    inst = {k: v for k, v in start["inst"].items()}
    out.append("<div class='card'><div class='lbl'>服务端真值（agent 从未看到）</div><pre>" + esc(json.dumps(inst, indent=1, ensure_ascii=False)[:3000]) + "</pre></div>")

    # ---------------------------------------------------------------- what the agent received
    out.append("<h2>1. agent 收到的上下文</h2>")
    out.append("<div class='card'>Claude Code 内置的系统提示和工具定义不在对话记录里（由 Claude Code 注入，这里看不到原文）。下面是记录里能看到的全部注入内容和我的提示。</div>")
    for d in tr:
        if d["type"] != "attachment":
            continue
        at = d["attachment"]
        t = at.get("type")
        if t == "instructions":
            for f in at.get("files", []):
                out.append(f"<details class='card'><summary>注入附件：{esc(f.get('type'))} — {esc(f.get('path'))}（{len(f.get('content',''))} 字符）</summary><pre>{esc(f.get('content',''))}</pre></details>")
        elif t in ("environment", "model", "date", "session_context", "skill_listing", "deferred_tools_delta", "credential_org"):
            body = json.dumps({k: v for k, v in at.items() if k != "type"}, ensure_ascii=False, indent=1)
            out.append(f"<details class='card'><summary>注入附件：{esc(t)}（{len(body)} 字符）</summary><pre>{esc(body[:6000])}</pre></details>")
    first_user = next(d for d in tr if d["type"] == "user")
    out.append("<div class='card'><div class='lbl'>我给 agent 的提示（原文）</div><pre>" + esc(first_user["message"]["content"]) + "</pre></div>")

    # ---------------------------------------------------------------- timeline
    out.append("<h2>2. 逐步过程（agent 的文字、执行的命令、收到的结果和图像）</h2>")
    n = 0
    results = {}
    for d in tr:
        if d["type"] == "user" and isinstance(d["message"].get("content"), list):
            for c in d["message"]["content"]:
                if c.get("type") == "tool_result":
                    results[c["tool_use_id"]] = c
    t0 = ts(tr[0]["timestamp"])
    for d in tr:
        if d["type"] != "assistant":
            continue
        for c in d["message"].get("content", []):
            ct = c.get("type")
            tt = f"+{ts(d['timestamp']) - t0:6.0f}s"
            if ct == "thinking" and c.get("thinking"):
                out.append(f"<div class='card'><span class='n'>{tt}</span> <span class='lbl think'>思考</span><pre class='think'>{esc(c['thinking'])}</pre></div>")
            elif ct == "text" and c.get("text", "").strip():
                out.append(f"<div class='card'><span class='n'>{tt}</span> <span class='lbl'>agent 文字</span><pre>{esc(c['text'])}</pre></div>")
            elif ct == "tool_use":
                n += 1
                inp = c.get("input", {})
                if c["name"] == "Bash":
                    head = f"Bash: <span class='cmd'>{esc(inp.get('command',''))}</span>"
                elif c["name"] == "Read":
                    head = f"Read: <span class='cmd'>{esc(inp.get('file_path',''))}</span>"
                elif c["name"] == "SubagentHandback":
                    head = "最终报告（SubagentHandback）"
                else:
                    head = f"{esc(c['name'])}: {esc(json.dumps(inp, ensure_ascii=False)[:300])}"
                block = [f"<div class='card'><span class='n'>{tt} #{n}</span> <span class='lbl'>工具调用</span> {head}"]
                if c["name"] == "SubagentHandback":
                    block.append(f"<pre>{esc(inp.get('message',''))}</pre>")
                r = results.get(c["id"])
                if r is not None:
                    rc = r.get("content")
                    if isinstance(rc, str):
                        rc = [{"type": "text", "text": rc}]
                    for part in rc or []:
                        if part.get("type") == "text":
                            txt = part["text"]
                            block.append(f"<details {'open' if len(txt) < 1500 else ''}><summary>返回（{len(txt)} 字符）</summary><pre>{esc(txt)}</pre></details>")
                        elif part.get("type") == "image":
                            src = part.get("source", {})
                            if src.get("type") == "base64":
                                block.append(f"<img src='data:{src.get('media_type','image/jpeg')};base64,{src['data']}'>")
                block.append("</div>")
                out.extend(block)
    out.append("<h2>3. 服务端每次 done 的判定</h2>")
    for e in dones:
        out.append(f"<div class='card'>第 {e['attempt']} 次：<span class='{'ok' if e['success'] else 'bad'}'>{'成功' if e['success'] else '失败'}</span><pre>{esc(e.get('feedback',''))}</pre></div>")
    out.append("</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(out))
    print(a.out, os.path.getsize(a.out), "bytes; tool calls", n)


if __name__ == "__main__":
    main()
