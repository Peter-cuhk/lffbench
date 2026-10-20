"""Local fake model APIs for offline harness checks (no real model, no cost).

Both servers listen on 127.0.0.1, record every request body (auth headers dropped) and answer from a script, so a
native harness can be driven turn by turn without a model:
  FakeAnthropic   Anthropic Messages API (streaming SSE), used by Claude Code and by api_player --provider anthropic
  FakeResponses   OpenAI Responses API (streaming SSE), used by Codex through a custom model provider

A script is a list of turns; turn i is answered to the request that already holds i assistant turns, so retries get
the same answer. Past the end of the script the server answers with a final text. A turn is a list of steps:
  ("tool", name, args)     Anthropic: tool_use block (name as given, e.g. "mcp__lff__robot" or "Read")
  ("text", s)              a text block / assistant message
  ("exec", js)             Responses: one `exec` custom tool call (Codex code mode) with this JavaScript
  ("fcall", name, args)    Responses: a plain function call
  ("nscall", ns, name, args)  Responses: a function call into a tool namespace (Codex without code mode calls our MCP
                           tools this way: name "robot", namespace "mcp__lff"; measured with 0.163)

  srv = FakeAnthropic(script, record_dir).start(); ... srv.base_url ...; srv.stop()
"""
import itertools
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DROP_HEADERS = ("x-api-key", "authorization", "cookie", "chatgpt-account-id")


class _Base:
    final_text = "done"

    def __init__(self, script=None, record_dir=None, port=0):
        self.script = list(script or [])
        self.record_dir = record_dir
        self.port = port
        self.lock = threading.Lock()
        self.counter = itertools.count(1)
        self.requests = []  # (n, path, body)
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)

    # ------------------------------------------------------------------ plumbing
    def start(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _record(self, body):
                with outer.lock:
                    n = next(outer.counter)
                    outer.requests.append((n, self.path, body))
                if outer.record_dir:
                    with open(os.path.join(outer.record_dir, f"req_{n:03d}.json"), "w") as f:
                        json.dump(dict(method=self.command, path=self.path, body=body,
                                       headers={k: v for k, v in self.headers.items() if k.lower() not in DROP_HEADERS}),
                                  f, indent=1, ensure_ascii=False)
                return n

            def do_GET(self):
                self._record(None)
                self._send(404, b"{}", "application/json")

            def do_HEAD(self):
                self._send(404, b"", "application/json")

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length", 0) or 0))
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = raw[:4000].decode("utf-8", "replace")
                n = self._record(body)
                outer.handle(self, n, body)

            def _send(self, code, data, ctype):
                self.send_response(code)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def turn(self, i):
        return self.script[i] if i < len(self.script) else [("text", self.final_text)]

    @staticmethod
    def send_sse(h, events):
        data = b"".join(f"event: {ev}\ndata: {json.dumps(d)}\n\n".encode() for ev, d in events)
        h._send(200, data, "text/event-stream")


class FakeAnthropic(_Base):
    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def handle(self, h, n, body):
        if not h.path.startswith("/v1/messages") or "count_tokens" in h.path or not isinstance(body, dict):
            h._send(200, json.dumps({"input_tokens": 10}).encode(), "application/json")
            return
        msgs = body.get("messages", [])
        i = sum(1 for m in msgs if m.get("role") == "assistant")
        steps = self.turn(i)
        blocks = []
        for j, st in enumerate(steps):
            if st[0] == "tool":
                blocks.append(dict(type="tool_use", id=f"toolu_fake_{i:03d}_{j}", name=st[1], input=st[2]))
            else:
                blocks.append(dict(type="text", text=st[1]))
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        model = body.get("model", "fake")
        evs = [("message_start", dict(type="message_start", message=dict(
            id=f"msg_fake_{n}", type="message", role="assistant", model=model, content=[], stop_reason=None,
            stop_sequence=None, usage=dict(input_tokens=100, output_tokens=1))))]
        for k, b in enumerate(blocks):
            if b["type"] == "tool_use":
                evs.append(("content_block_start", dict(type="content_block_start", index=k, content_block=dict(
                    type="tool_use", id=b["id"], name=b["name"], input={}))))
                evs.append(("content_block_delta", dict(type="content_block_delta", index=k, delta=dict(
                    type="input_json_delta", partial_json=json.dumps(b["input"])))))
            else:
                evs.append(("content_block_start", dict(type="content_block_start", index=k,
                                                        content_block=dict(type="text", text=""))))
                evs.append(("content_block_delta", dict(type="content_block_delta", index=k,
                                                        delta=dict(type="text_delta", text=b["text"]))))
            evs.append(("content_block_stop", dict(type="content_block_stop", index=k)))
        evs.append(("message_delta", dict(type="message_delta", delta=dict(stop_reason=stop, stop_sequence=None),
                                          usage=dict(output_tokens=5))))
        evs.append(("message_stop", dict(type="message_stop")))
        self.send_sse(h, evs)

    def message_requests(self):
        return [(n, b) for n, p, b in self.requests if isinstance(b, dict) and p.startswith("/v1/messages")
                and "count_tokens" not in p]


class FakeResponses(_Base):
    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def handle(self, h, n, body):
        if not isinstance(body, dict) or "input" not in body:
            h._send(404, b"{}", "application/json")
            return
        items = body.get("input") or []
        i = sum(1 for it in items if isinstance(it, dict) and it.get("type") in
                ("custom_tool_call", "function_call") or (isinstance(it, dict) and it.get("type") == "message"
                                                          and it.get("role") == "assistant"))
        steps = self.turn(i)
        out = []
        for j, st in enumerate(steps):
            if st[0] == "exec":
                out.append(dict(type="custom_tool_call", id=f"ctc_fake_{n}_{j}", call_id=f"call_fake_{i:03d}_{j}",
                                name="exec", status="completed", input=st[1]))
            elif st[0] == "fcall":
                out.append(dict(type="function_call", id=f"fc_fake_{n}_{j}", call_id=f"call_fake_{i:03d}_{j}",
                                name=st[1], arguments=json.dumps(st[2]), status="completed"))
            elif st[0] == "nscall":
                out.append(dict(type="function_call", id=f"fc_fake_{n}_{j}", call_id=f"call_fake_{i:03d}_{j}",
                                namespace=st[1], name=st[2], arguments=json.dumps(st[3]), status="completed"))
            else:
                out.append(dict(type="message", id=f"msg_fake_{n}_{j}", role="assistant", status="completed",
                                content=[dict(type="output_text", text=st[1], annotations=[])]))
        rid = f"resp_fake_{n}"
        evs = [("response.created", dict(type="response.created", response=dict(id=rid, status="in_progress")))]
        for k, it in enumerate(out):
            evs.append(("response.output_item.done", dict(type="response.output_item.done", output_index=k, item=it)))
        evs.append(("response.completed", dict(type="response.completed", response=dict(
            id=rid, status="completed", output=out, usage=dict(
                input_tokens=100, input_tokens_details=dict(cached_tokens=0), output_tokens=3,
                output_tokens_details=dict(reasoning_tokens=0), total_tokens=103)))))
        self.send_sse(h, evs)

    def model_requests(self):
        return [(n, b) for n, p, b in self.requests if isinstance(b, dict) and "input" in b]
