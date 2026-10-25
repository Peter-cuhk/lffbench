"""Export saved API-player runs as a portable, offline Agent Viewer.

No simulator/model calls are made. Original image bytes are embedded once.
Supports transcript.jsonl and gpt_transcript.jsonl (API-player event schema).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path


def read_jsonl(path):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: malformed JSON; export a completed log") from exc
    return rows


def parse_output(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def build_run(folder, assets, sandbox_override=None):
    folder = Path(folder)
    transcript = next((folder / name for name in ('transcript.jsonl', 'gpt_transcript.jsonl')
                       if (folder / name).exists()), None)
    if not transcript:
        raise ValueError(f"No supported API-player transcript in {folder}")
    rows = read_jsonl(transcript)
    if not any(row.get('ev') == 'assistant' for row in rows):
        raise ValueError(f"No API-player assistant events in {transcript}")
    session = json.loads((folder / 'session.json').read_text())
    events = read_jsonl(folder / 'events.jsonl') if (folder / 'events.jsonl').exists() else []
    start = next((r for r in rows if r.get('ev') == 'start'), {})
    server = next((r for r in events if r.get('ev') == 'session_start'), {})
    sandbox = Path(sandbox_override or session.get('sandbox', folder))
    warnings, cache = [], {}

    def images(refs, source, timestamp):
        result = []
        for ref in refs:
            if not isinstance(ref, str):
                raise ValueError('Expected image path strings in API-player transcript')
            if ref not in cache:
                relative = Path(ref)
                roots = [folder, sandbox]
                candidates = []
                # Only export images within the explicitly selected run/sandbox.
                for root in roots:
                    candidate = (root / relative).resolve()
                    if candidate.is_relative_to(root.resolve()):
                        candidates.append(candidate)
                path = next((p for p in candidates if p.is_file()), None)
                key = None
                if path:
                    raw = path.read_bytes()
                    if raw.startswith(b'\xff\xd8\xff'):
                        mime = 'image/jpeg'
                    elif raw.startswith(b'\x89PNG\r\n\x1a\n'):
                        mime = 'image/png'
                    elif raw[:6] in (b'GIF87a', b'GIF89a'):
                        mime = 'image/gif'
                    elif raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
                        mime = 'image/webp'
                    else:
                        raise ValueError(f'Unsupported image bytes: {ref}')
                    key = hashlib.sha256(raw).hexdigest()[:24]
                    assets[key] = f'data:{mime};base64,' + base64.b64encode(raw).decode()
                else:
                    warnings.append(f'图片缺失：{ref}')
                name = relative.stem
                camera = 'agentview' if 'agentview' in name else 'wrist' if 'wrist' in name else name
                cache[ref] = {'asset': key, 'path': ref, 'camera': camera}
            result.append({**cache[ref], 'source': source, 'at': timestamp})
        return result

    outcomes = [{'attempt': e.get('attempt'), 'success': e.get('success'),
                 'feedback': e.get('feedback', ''), 'at': e.get('t')}
                for e in events if e.get('ev') == 'done']
    steps, pending, latest, latest_text = [], [], [], ''
    attempt, origin, active, task_text = 1, rows[0].get('t', 0), None, ''
    for line, row in enumerate(rows, 1):
        kind = row.get('ev')
        if kind == 'user':
            latest_text = row.get('text', '')
            if not task_text:
                match = re.search(r'\{.*\}', latest_text, re.S)
                task_text = parse_output(match.group() if match else '').get('task', latest_text)
            if row.get('images'):
                latest = images(row['images'], line, row.get('t'))
        elif kind in ('api_error', 'abort'):
            pending.append({'kind': kind, 'text': row.get('error', ''), 'at': row.get('t')})
        elif kind == 'assistant':
            calls = row.get('tool_calls', [])
            active = {
                'index': len(steps), 'turn': row.get('turn'), 'attempt': attempt,
                'at': row.get('t'), 'elapsed': max(0, row.get('t', origin) - origin),
                'latency': row.get('latency_s'), 'before': list(latest),
                'inputText': latest_text, 'text': row.get('text') or '',
                'decisionSummary': row.get('reasoning_summary') or None,
                'decisionSummaryStatus': row.get('reasoning_summary_status', 'not_recorded'),
                'decisionSourceLine': line,
                'actions': [{'id': c.get('id'), 'name': c.get('name', 'tool'),
                             'arguments': c.get('arguments', {}), 'result': None,
                             'after': []} for c in calls],
                'notices': pending, 'verdict': None,
                'requestImages': row.get('request_stats', {}).get('n_images'),
            }
            pending = []
            steps.append(active)
        elif kind == 'tool':
            if active is None:
                warnings.append(f'第 {line} 行工具结果缺少前置调用')
                continue
            if row.get('turn') is not None and row.get('turn') != active['turn']:
                raise ValueError(f'Tool turn mismatch at transcript line {line}')
            unresolved = [a for a in active['actions'] if a['result'] is None]
            call_id = row.get('call_id') or row.get('tool_call_id')
            action = next((a for a in unresolved if call_id and a['id'] == call_id), None)
            if call_id and action is None:
                raise ValueError(f'Unknown tool call id at transcript line {line}')
            if action is None:
                action = next((a for a in unresolved if isinstance(a['arguments'], dict)
                               and a['arguments'].get('command') == row.get('command')), None)
            if action is None and len(unresolved) == 1:
                action = unresolved[0]
            if action is None:
                raise ValueError(f'Ambiguous tool result at transcript line {line}')
            output = row.get('output', '')
            action['result'] = output
            action['at'] = row.get('t')
            action['after'] = images(row.get('images', []), line, row.get('t'))
            parsed = parse_output(output)
            action['ok'] = parsed.get('ok')
            latest_text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            if action['after']:
                latest = list(action['after'])
            command = row.get('command') or (action['arguments'].get('command', '')
                                            if isinstance(action['arguments'], dict) else '')
            if command.split(' ', 1)[0] == 'done':
                verdict = next((o for o in outcomes if o['attempt'] == attempt), None)
                if verdict:
                    active['verdict'] = verdict
                # A failed done may immediately return RESET images for the next attempt.
                message = parsed.get('message', '')
                next_attempt = re.search(r'Attempt\s+(\d+)\s+of', message)
                if next_attempt:
                    attempt = int(next_attempt.group(1))
                    action['resetToAttempt'] = attempt
    if pending:
        steps[-1]['noticesAfter'] = pending
    for step in steps:
        step['incomplete'] = any(a['result'] is None for a in step['actions'])
    status = 'success' if any(o['success'] is True for o in outcomes) else (
        'failed' if outcomes and any(r.get('ev') == 'end' and r.get('finished') is True for r in rows) else 'incomplete')
    video = None
    video_path = folder / 'rollout.mp4'
    if video_path.is_file():
        raw = video_path.read_bytes()
        if len(raw) < 12 or raw[4:8] != b'ftyp':
            raise ValueError('rollout.mp4 is not an MP4 container')
        key = 'video-' + hashlib.sha256(raw).hexdigest()[:24]
        assets[key] = 'data:video/mp4;base64,' + base64.b64encode(raw).decode()
        video = {'asset': key, 'name': 'rollout.mp4'}
    return {'id': folder.name, 'video': video, 'task': server.get('task', folder.name),
            'taskText': task_text, 'seed': server.get('seed'), 'model': start.get('model', '未记录'),
            'perception': server.get('perception', '未记录'), 'feedback': server.get('feedback', '未记录'),
            'status': status, 'steps': steps, 'outcomes': outcomes,
            'warnings': sorted(set(warnings)), 'source': transcript.name,
            'actionCount': sum(len(s['actions']) for s in steps)}


def render(folders, output, sandbox_override=None):
    assets = {}
    runs = [build_run(folder, assets, sandbox_override) for folder in folders]
    payload = json.dumps({'version': 1, 'runs': runs, 'assets': assets}, ensure_ascii=False)
    # Logs are untrusted text, including strings containing HTML/script terminators.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = Path(__file__).with_name('viewer.html').read_text()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('/*__PAYLOAD__*/null', payload))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True, help='Repeat to include multiple runs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sandbox', type=Path, help='Relocated image directory for a single run')
    args = parser.parse_args()
    if args.sandbox and len(args.run) != 1:
        parser.error('--sandbox requires exactly one --run')
    try:
        runs = render(args.run, args.out, args.sandbox)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Export failed: {exc}\n')
    print(json.dumps({'output': str(args.out.resolve()), 'runs': [
        {'task': r['task'], 'steps': len(r['steps']), 'actions': r['actionCount'],
         'outcomes': [o['success'] for o in r['outcomes']], 'warnings': r['warnings']} for r in runs
    ]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
