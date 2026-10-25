#!/usr/bin/env python3
"""Read-only terminal viewer for active LFF-Bench runs (ANSI images or ASCII)."""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Strip terminal controls from untrusted logged text; only this program emits ANSI.
def clean(value):
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', str(value))


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


class Tail:
    """Only consume complete JSONL records; tolerate concurrent append and rotation."""
    def __init__(self, path):
        self.path, self.offset, self.pending, self.identity = Path(path), 0, b'', None
        self.errors = 0
        self.generation = 0

    def read(self):
        try:
            with self.path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
                    self.offset, self.pending = 0, b''
                    self.generation += 1
                self.identity = identity
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except OSError:
            return []
        lines = (self.pending + chunk).split(b'\n')
        self.pending = lines.pop()
        out = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    out.append(row)
                else:
                    self.errors += 1
            except (ValueError, UnicodeDecodeError):
                self.errors += 1
        return out


class Run:
    def __init__(self, path):
        self.path = Path(path)
        self.task = self.path.name
        self.calls = 0
        self.attempt = 1
        self.outcomes = []
        self.command = '等待工具调用'
        self.command_at = None
        self.view_paths = []
        self.view_at = None
        self.view_sequence = 0
        self.embedded_sequence = 0
        self.embedded_paths = []
        self.embedded = []
        self.summary = ''
        self.status = '等待日志'
        self.notice = ''
        self.last_activity = None
        names = ['events.jsonl', 'gateway_transcript.jsonl', 'mcp_transcript.jsonl',
                 'transcript.jsonl', 'gpt_transcript.jsonl', 'codex_events.jsonl']
        self.tails = {n: Tail(self.path / n) for n in names}
        self.image_cache = {}

    def update(self):
        gateway = 'gateway_transcript.jsonl' if (self.path/'gateway_transcript.jsonl').exists() else 'mcp_transcript.jsonl'
        api = 'transcript.jsonl' if (self.path/'transcript.jsonl').exists() else 'gpt_transcript.jsonl'
        for name in ['events.jsonl', gateway, api, 'codex_events.jsonl']:
            tail = self.tails[name]
            generation = tail.generation
            rows = tail.read()
            if tail.generation != generation:
                # Avoid mixing observations and counts across replaced/truncated runs.
                self.__init__(self.path)
                self.notice = '日志已轮转，重新读取'
                return self.update()
            for row in rows:
                if row.get('t') is not None:
                    self.last_activity = max(self.last_activity or 0, row['t'])
                if name == 'events.jsonl':
                    if row.get('ev') == 'session_start':
                        self.task = row.get('task', self.task)
                        self.status = '运行中'
                    elif row.get('ev') == 'call':
                        self.calls += 1
                        self.attempt = row.get('attempt', self.attempt)
                        self.command = row.get('cmd', '') + ' ' + json.dumps(row.get('args', {}), ensure_ascii=False)
                        self.command_at = row.get('t')
                    elif row.get('ev') == 'done':
                        self.outcomes.append((row.get('attempt'), row.get('success')))
                elif name == gateway:
                    if row.get('ev') != 'tool':
                        continue
                    if row.get('finished'):
                        self.status = '任务已结束'
                    if row.get('name') == 'view_images' and not row.get('is_error') and row.get('images', 0) > 0:
                        paths = []
                        for output in row.get('output', []):
                            try:
                                obj = json.loads(output)
                                paths.extend(obj.get('attached', []))
                            except (ValueError, TypeError, AttributeError):
                                pass
                        # Only mark confirmed attachments as observed, not requested files.
                        if paths:
                            self.view_paths, self.view_at = paths, row.get('t')
                            self.view_sequence += 1
                    if row.get('name') == 'robot':
                        self.command = row.get('arguments', {}).get('command', self.command)
                        self.command_at = row.get('t')
                elif name == api:
                    if row.get('ev') in ('user', 'tool') and row.get('images'):
                        self.view_paths, self.view_at = row['images'], row.get('t')
                        self.view_sequence += 1
                    if row.get('ev') == 'assistant':
                        self.summary = row.get('reasoning_summary') or row.get('text') or ''
                    if row.get('ev') in ('end', 'abort'):
                        self.status = '已结束' if row['ev'] == 'end' else '已中断'
                elif name == 'codex_events.jsonl' and row.get('type') == 'item.completed':
                    item = row.get('item', {})
                    if item.get('type') in ('reasoning', 'agent_message'):
                        self.summary = item.get('text', '')
                    if item.get('type') == 'mcp_tool_call' and item.get('tool') == 'view_images' and not item.get('error'):
                        content = (item.get('result') or {}).get('content', [])
                        paths, raw = [], []
                        for c in content:
                            if c.get('type') == 'text':
                                try:
                                    paths.extend(json.loads(c.get('text', '')).get('attached', []))
                                except (ValueError, AttributeError):
                                    pass
                            elif c.get('type') == 'image':
                                try:
                                    raw.append(base64.b64decode(c['data'], validate=True))
                                except (ValueError, KeyError):
                                    pass
                        if raw and len(raw) == len(paths):
                            self.embedded_paths, self.embedded = paths, raw
                            self.embedded_sequence += 1
                            if not (self.path/gateway).exists():
                                self.view_paths = paths
                                self.view_sequence = self.embedded_sequence
        self.notice = f'跳过 {sum(t.errors for t in self.tails.values())} 条损坏日志' if any(t.errors for t in self.tails.values()) else self.notice

    def frames(self):
        session = read_json(self.path/'session.json') or {}
        roots = [self.path]
        if session.get('sandbox'):
            roots.append(Path(session['sandbox']))
        result = []
        for index, ref in enumerate(self.view_paths):
            if not isinstance(ref, str):
                continue
            if self.embedded_sequence == self.view_sequence and self.embedded_paths == self.view_paths and index < len(self.embedded):
                result.append((ref, self.embedded[index], '工具返回的原始图片字节'))
                continue
            for root in roots:
                target = (root/ref).resolve()
                if not target.is_relative_to(root.resolve()) or not target.is_file():
                    continue
                stat = target.stat()
                if self.view_at is not None and stat.st_mtime > self.view_at + 0.1:
                    result.append((ref, None, '文件在观察后已修改，不能确认原图'))
                    break
                key = (str(target), stat.st_mtime_ns, stat.st_size)
                if key not in self.image_cache:
                    self.image_cache = {key: target.read_bytes()}
                result.append((ref, self.image_cache[key], '日志引用的图片文件'))
                break
            else:
                result.append((ref, None, '图片文件不可用'))
        return result


def choose_run(batch):
    data = read_json(Path(batch)/'manifest.json')
    if not data:
        return None, '批次清单暂不可读'
    rows = data.get('runs', [])
    counts = {key: sum(r.get('status') == key for r in rows) for key in ['done', 'running', 'pending']}
    active = [r for r in rows if r.get('status') == 'running' and r.get('log')]
    completed = [r for r in rows if r.get('log')]
    row = active[0] if active else completed[-1] if completed else None
    summary = ' / '.join(f'{key} {count}' for key, count in counts.items())
    return row, summary


def image_lines(raw, columns, rows, color):
    from PIL import Image, ImageOps
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    width, height = columns, rows * 2
    image = ImageOps.contain(image, (width, height))
    pixels = image.load()
    lines = []
    ramp = ' .:-=+*#%@'
    for y in range(0, image.height, 2):
        line = ''
        for x in range(image.width):
            upper, lower = pixels[x,y], pixels[x,min(y+1,image.height-1)]
            if color:
                line += '\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm▀' % (*upper, *lower)
            else:
                value = (sum(upper) + sum(lower)) / 6
                line += ramp[min(9, int(value/256*10))]
        lines.append(line + ('\033[0m' if color else ''))
    return lines


# Native image transports keep source pixels rather than quantizing to text cells.
KITTY_IMAGE_ID = 1987000000 + os.getpid() % 100000

def choose_renderer(requested, environ=None, tty=True):
    env = os.environ if environ is None else environ
    if not tty:
        return 'ansi'
    # This embedded terminal was observed to swallow OSC 1337 images, leaving
    # the reserved rectangle blank. Keep a visible preview even for old commands.
    if env.get('TERM_PROGRAM') == 'claude-desktop' and requested == 'iterm':
        return 'ansi'
    if requested != 'auto':
        return requested
    # Multiplexers need terminal-specific passthrough setup; keep auto conservative.
    if env.get('TMUX') or env.get('STY'):
        return 'ansi'
    if env.get('KITTY_WINDOW_ID') or env.get('TERM') == 'xterm-kitty':
        return 'kitty'
    if env.get('TERM_PROGRAM') in ('iTerm.app', 'WezTerm'):
        return 'iterm'
    return 'ansi'


def native_image(raw_images, columns, rows, renderer):
    from PIL import Image
    decoded = [Image.open(io.BytesIO(raw)).convert('RGB') for raw in raw_images]
    # Place full-resolution views side by side without resampling source pixels.
    canvas = Image.new('RGB', (sum(im.width for im in decoded), max(im.height for im in decoded)))
    x = 0
    for im in decoded:
        canvas.paste(im, (x, 0))
        x += im.width
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode()
    if renderer == 'iterm':
        sequence = f'\033]1337;File=inline=1;width={columns};height={rows};preserveAspectRatio=1:{data}\a'
    else:
        # Preserve aspect in the usual 1:2 text cells; don't stretch square inputs.
        w = max(1, min(columns, int(rows * 2 * canvas.width / canvas.height)))
        h = max(1, min(rows, round(w * canvas.height / canvas.width / 2)))
        chunks = [data[i:i+4096] for i in range(0, len(data), 4096)]
        sequence = ''
        for n, chunk in enumerate(chunks):
            meta = f'a=T,f=100,t=d,i={KITTY_IMAGE_ID},q=2,C=1,c={w},r={h},' if n == 0 else ''
            sequence += f'\033_G{meta}m={int(n < len(chunks)-1)};{chunk}\033\\'
    # Both protocols are given a reserved rectangle; restore before advancing.
    return '\0337' + sequence + '\0338' + '\n' * (rows - 1)


def clear_native(renderer):
    return f'\033_Ga=d,d=I,i={KITTY_IMAGE_ID},q=2\033\\' if renderer == 'kitty' else ''


def age(timestamp):
    return f'{max(0,int(time.time()-timestamp))} 秒前' if timestamp else '时间未记录'


def render(run, label, width, rows, color, images, renderer="ansi", camera="all"):
    lines = [clean(label)[:width], f'{clean(run.task)} | {clean(run.status)} | 尝试 {run.attempt} | 机器人动作 {run.calls}',
             f'最近动作 ({age(run.command_at)}): {clean(run.command)[:max(15,width-22)]}',
             f'最近收到的画面 ({age(run.view_at)})：按观察事件更新，不是连续视频']
    frames = run.frames()
    if camera != "all":
        keyword = "agentview" if camera == "main" else "wrist"
        frames = [f for f in frames if keyword in f[0]][-1:]
    if not frames:
        lines.append('尚未发现已返回给 Agent 的图片。')
    if images and frames:
        shown = frames[-2:]
        cell = max(12,(width-3*(len(shown)-1))//len(shown))
        columns = []
        for ref, raw, source in shown:
            lines.append(f'  {clean(ref)} · {source}')
            try:
                columns.append(image_lines(raw, cell, rows, color) if raw else ['[原图不可用]'])
            except Exception as exc:
                columns.append([f'[无法解码图片: {type(exc).__name__}]'])
        if renderer in ('iterm', 'kitty') and any(raw for _, raw, _ in shown):
            try:
                lines.append(native_image([raw for _, raw, _ in shown if raw], width, rows, renderer))
                columns = []
            except Exception as exc:
                lines.append(f'原图显示编码失败，使用字符预览: {type(exc).__name__}')
        for y in range(max(map(len, columns), default=0)):
            parts=[]
            for c in columns:
                part=c[y] if y<len(c) else ''
                visible=len(re.sub(r'\033\[[0-9;]*m','',part))
                parts.append(part+' '*max(0,cell-visible))
            lines.append('   '.join(parts))
    elif frames:
        lines.extend('图片: '+clean(f[0]) for f in frames)
    if run.summary:
        lines.append('最近记录的摘要/说明: '+clean(run.summary)[:max(20,width-20)])
    if run.outcomes:
        lines.append('已提交结果: '+' · '.join(f'尝试 {a}: '+('成功' if ok else '失败') for a,ok in run.outcomes))
    if run.notice:
        lines.append(run.notice)
    lines.append('最近日志活动: '+age(run.last_activity))
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--batch',type=Path,help='Batch directory; follows the running task then the next one')
    group.add_argument('--run',type=Path,help='One run directory')
    parser.add_argument('--interval',type=float,default=6,help="Refresh interval in seconds (default: 6)")
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--duration',type=float,help='Stop after N seconds (useful for verification)')
    parser.add_argument('--ascii',action='store_true',help='Portable monochrome preview instead of truecolor blocks')
    parser.add_argument('--no-images',action='store_true')
    parser.add_argument('--renderer',choices=['auto','ansi','iterm','kitty'],default='auto',help='Native pixels: iterm (also VS Code with enableImages) or kitty; auto detects known terminals')
    parser.add_argument('--camera',choices=['all','main','wrist'],default='all',help='Select one camera for a larger view')
    parser.add_argument('--width',type=int,default=None)
    parser.add_argument('--image-rows',type=int,default=None)
    a=parser.parse_args(argv)
    if a.interval<0.5:parser.error('--interval must be >= 0.5 seconds')
    if a.width is not None and a.width<24:parser.error('--width must be >= 24')
    if a.image_rows is not None and a.image_rows<2:parser.error('--image-rows must be >= 2')
    if a.duration is not None and a.duration<=0:parser.error('--duration must be positive')
    if not a.no_images:
        try:import PIL
        except ImportError:parser.error('Image preview needs Pillow; use the evaluation Python or --no-images')
    renderer=choose_renderer('ansi' if a.ascii else a.renderer,tty=sys.stdout.isatty())
    if a.no_images:renderer='ansi'
    compatibility_notice = (
        'Claude 内置终端的 iTerm 图片模式未能显示；已切换为字符预览（非原图清晰度）。'
        if os.environ.get('TERM_PROGRAM') == 'claude-desktop'
        and a.renderer == 'iterm' and renderer == 'ansi'
        and not a.ascii and not a.no_images and sys.stdout.isatty() else ''
    )
    interactive=sys.stdout.isatty() and not a.once
    current=None
    started=time.monotonic()
    previous=None
    try:
        if interactive:print('\033[?1049h\033[?25l',end='',flush=True)
        while True:
            label='Agent Viewer · LIVE · Ctrl+C 退出'
            path=a.run
            if a.batch:
                selected,summary=choose_run(a.batch)
                path=Path(selected['log']) if selected else None
                label=f'{a.batch.name} | {summary}'
            if path is not None:
                if current is None or current.path!=path:current=Run(path)
                current.update()
                if a.batch and selected:current.status=selected.get('status',current.status)
                size=shutil.get_terminal_size((100,36))
                width=a.width or max(24,size.columns-1)
                rows=a.image_rows or max(3,size.lines-12-bool(compatibility_notice))
                label += f' | 刷新 {a.interval:g}s · {renderer}'
                output=render(current,label,width,rows,sys.stdout.isatty() and not a.ascii,not a.no_images,renderer,a.camera)
                if compatibility_notice:
                    header, _, body = output.partition('\n')
                    output = header + '\n' + compatibility_notice + '\n' + body
            else:output=label+'\n等待批次启动下一任务…'
            if output!=previous:
                if interactive:print(clear_native(renderer)+'\033[H\033[J'+output,end='',flush=True)
                else:print(output,flush=True)
                previous=output
            if a.once or a.duration and time.monotonic()-started>=a.duration:break
            delay=a.interval if a.duration is None else min(a.interval,max(0,a.duration-(time.monotonic()-started)))
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:print(clear_native(renderer)+'\033[0m\033[?25h\033[?1049l',end='',flush=True)


if __name__=='__main__':main()
