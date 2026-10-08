"""Render one GPT-player run as a self-contained HTML trace: the instructions and tool the model received,
every request's new content (text + images), the model's reasoning summaries and commands, every server
report, the server-side ground truth and the done verdicts. Optionally embeds the rollout video.

python scripts/trace_gpt_html.py --log runs/gpt/exp2/<run> --out traces/<name>.html [--title ...] [--video]
"""
import argparse
import base64
import html
import json
import os

from trace_html import CSS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def esc(s):
    return html.escape(str(s))


def img_tag(path):
    if not os.path.exists(path):
        return f"<span class='lbl'>[missing {esc(os.path.basename(path))}]</span>"
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f"<img src='data:image/jpeg;base64,{b}' title='{esc(os.path.basename(path))}'>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--video", action="store_true", help="embed rollout.mp4 (base64)")
    a = ap.parse_args()
    sess = json.load(open(os.path.join(a.log, "session.json")))
    sb = sess["sandbox"]
    tr = [json.loads(l) for l in open(os.path.join(a.log, "gpt_transcript.jsonl"))]
    ev = [json.loads(l) for l in open(os.path.join(a.log, "events.jsonl"))]
    start = next(e for e in ev if e["ev"] == "session_start")
    dones = [e for e in ev if e["ev"] == "done"]
    http = [e for e in ev if e["ev"] == "http"]
    asst = [d for d in tr if d["ev"] == "assistant"]
    st = next(d for d in tr if d["ev"] == "start")
    end = next((d for d in tr if d["ev"] in ("end", "abort")), {})
    tot = end.get("usage_total") or {}

    out = [f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' "
           f"content='width=device-width,initial-scale=1'><title>{esc(a.title or 'GPT trace')}</title>"
           f"<style>{CSS}</style></head><body><h1>{esc(a.title or os.path.basename(a.log))}</h1><div class='card'><table>"]
    rows = [("任务 / 种子", f"{start['task']} / {start['seed']}"), ("模型", f"{st['model']}（reasoning effort {st['effort']}）"),
            ("感知", start.get("perception", "depth") + "（rgb = 只有彩色图，没有深度工具）"),
            ("结果", "；".join(f"第{d['attempt']}次 {'成功' if d['success'] else '失败'}" for d in dones) or "—"),
            ("机器人命令数", sum(1 for e in ev if e["ev"] == "call")),
            ("模型调用次数 / 平均每次延迟", f"{len(asst)} / {sum(d['latency_s'] for d in asst) / max(len(asst), 1):.1f} s"),
            ("仿真执行总时长", f"{sum(h['server_s'] for h in http):.0f} s"),
            ("token（输入 / 其中缓存 / 输出 / 其中推理）",
             f"{tot.get('input_tokens', 0)} / {tot.get('cached_tokens', 0)} / {tot.get('output_tokens', 0)} / {tot.get('reasoning_tokens', 0)}")]
    out += [f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows] + ["</table></div>"]
    inst = {k: v for k, v in start["inst"].items()}
    out.append("<div class='card'><div class='lbl'>服务端真值（模型从未看到）</div><pre>"
               + esc(json.dumps(inst, indent=1, ensure_ascii=False)[:3000]) + "</pre></div>")
    if a.video and os.path.exists(os.path.join(a.log, "rollout.mp4")):
        b = base64.b64encode(open(os.path.join(a.log, "rollout.mp4"), "rb").read()).decode()
        out.append("<h2>录像（左：第三视角，右：腕部；3 倍速）</h2><video controls style='max-width:100%' "
                   f"src='data:video/mp4;base64,{b}'></video>")

    out.append("<h2>1. 模型收到的系统指令和工具（原文，每次请求都一样）</h2>")
    out.append("<div class='card'>这就是发给 GPT 的全部固定内容：沙箱 README 原文加一句开头说明，以及唯一的 robot 工具。"
               "没有文件、记忆、深度、物体坐标。</div>")
    out.append(f"<details class='card' open><summary>instructions（{len(st['instructions'])} 字符）</summary><pre>{esc(st['instructions'])}</pre></details>")
    out.append(f"<details class='card'><summary>tool</summary><pre>{esc(json.dumps(st['tool'], indent=1, ensure_ascii=False))}</pre></details>")

    out.append("<h2>2. 逐步过程</h2>")
    t0 = tr[0]["t"]
    n = 0
    for d in tr:
        tt = f"+{d['t'] - t0:6.0f}s"
        if d["ev"] == "user":
            imgs = "".join(img_tag(os.path.join(sb, p)) for p in d.get("images", []))
            out.append(f"<div class='card'><span class='n'>{tt}</span> <span class='lbl'>发给模型</span><pre>{esc(d['text'])}</pre>{imgs}</div>")
        elif d["ev"] == "assistant":
            blk = [f"<div class='card'><span class='n'>{tt}</span> <span class='lbl'>模型回复</span> "
                   f"<span class='lbl'>（{d['latency_s']:.1f} s，输出 {d['usage'].get('output_tokens', 0)} token，"
                   f"其中推理 {d['usage'].get('reasoning_tokens', 0)}）</span>"]
            if d.get("reasoning_summary"):
                blk.append(f"<div class='lbl think'>推理摘要（模型自己的思考摘要）</div><pre class='think'>{esc(d['reasoning_summary'])}</pre>")
            if d.get("text"):
                blk.append(f"<div class='lbl'>文字</div><pre>{esc(d['text'])}</pre>")
            for c in d["tool_calls"]:
                n += 1
                blk.append(f"<div>#{n} <span class='cmd'>./robot {esc(c['arguments'].get('command', ''))}</span></div>")
            out.append("".join(blk) + "</div>")
        elif d["ev"] == "tool":
            imgs = "".join(img_tag(os.path.join(sb, p)) for p in d.get("images", []))
            txt = d["output"]
            out.append(f"<div class='card'><span class='n'>{tt}</span> <span class='lbl'>服务器返回（作为工具结果发给模型）</span>"
                       f"<details {'open' if len(txt) < 1500 else ''}><summary>{len(txt)} 字符</summary><pre>{esc(txt)}</pre></details>"
                       + (f"<div class='lbl'>随后附上的图像</div>{imgs}" if imgs else "") + "</div>")
        elif d["ev"] in ("api_error", "abort"):
            out.append(f"<div class='card bad'>{esc(json.dumps(d, ensure_ascii=False))}</div>")
    out.append("<h2>3. 服务端每次 done 的判定</h2>")
    for e in dones:
        out.append(f"<div class='card'>第 {e['attempt']} 次：<span class='{'ok' if e['success'] else 'bad'}'>"
                   f"{'成功' if e['success'] else '失败'}</span>（agent 的理由：{esc(e.get('reason', ''))}）<pre>{esc(e.get('feedback', ''))}</pre></div>")
    rep = os.path.join(a.log, "final_report.md")
    if os.path.exists(rep):
        out.append(f"<h2>4. 模型最后的报告</h2><div class='card'><pre>{esc(open(rep).read())}</pre></div>")
    out.append("</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(out))
    print(a.out, os.path.getsize(a.out), "bytes; commands", n)


if __name__ == "__main__":
    main()
