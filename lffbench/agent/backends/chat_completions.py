"""OpenAI-compatible Chat Completions backend (function calling), e.g. for a local open-source VLM served
by scripts/qwen_vl_server.py or any vLLM / SGLang endpoint. Same conversation, different wire format."""
import json
import os
import time

from ..context import ToolCall, prune_live_images
from .base import Backend, BackendResponse, import_openai


def chat_tools(specs):
    return [dict(type="function", function=dict(name=s["name"], description=s["description"],
                                                 parameters=s["parameters"])) for s in specs]


class ChatCompletionsBackend(Backend):
    name = "chat"

    def __init__(self, model="qwen3-vl-8b", base_url=None, api_key=None, temperature=0.0, max_tokens=1024,
                 max_live_images=4, jpeg_quality=90, timeout=600.0, client=None, extra_body=None):
        self.model = model
        self.base_url = base_url or os.environ.get("LFF_CHAT_BASE_URL", "http://127.0.0.1:8765/v1")
        self.api_key = api_key or os.environ.get("LFF_CHAT_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_live_images = max_live_images
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.extra_body = extra_body
        self._client = client

    def config(self):
        return dict(name=self.name, model=self.model, base_url=self.base_url, temperature=self.temperature,
                    max_tokens=self.max_tokens, max_live_images=self.max_live_images)

    def client(self):
        if self._client is None:
            openai = import_openai()
            kw = dict(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=1)
            if any(h in self.base_url for h in ("://127.0.0.1", "://localhost")):
                # local server: ignore the machine's proxy variables (ALL_PROXY=socks5h://... is set here)
                kw["http_client"] = openai.DefaultHttpxClient(trust_env=False, timeout=self.timeout)
            self._client = openai.OpenAI(**kw)
        return self._client

    def build_request(self, conv, tools):
        msgs = [dict(role="system", content=conv.system)]
        for it in prune_live_images(conv.items, self.max_live_images):
            if it["kind"] == "user":
                content = []
                for p in it["parts"]:
                    if p["type"] == "text":
                        content.append(dict(type="text", text=p["text"]))
                    else:
                        content.append(dict(type="image_url", image_url=dict(url=p["image"].data_url(self.jpeg_quality))))
                msgs.append(dict(role="user", content=content))
            elif it["kind"] == "assistant":
                m = dict(role="assistant", content=it.get("text") or None)
                if it["tool_calls"]:
                    m["tool_calls"] = [dict(id=tc.id, type="function",
                                            function=dict(name=tc.name,
                                                          arguments=tc.raw_arguments or json.dumps(tc.arguments)))
                                       for tc in it["tool_calls"]]
                msgs.append(m)
            elif it["kind"] == "tool_output":
                msgs.append(dict(role="tool", tool_call_id=it["call_id"], content=it["text"]))
        req = dict(model=self.model, messages=msgs, tools=chat_tools(tools), tool_choice="auto",
                   temperature=self.temperature, max_tokens=self.max_tokens)
        if self.extra_body:
            req["extra_body"] = dict(self.extra_body)
        return req

    def act(self, conv, tools):
        req = self.build_request(conv, tools)
        n_img = sum(1 for m in req["messages"] if isinstance(m.get("content"), list)
                    for c in m["content"] if c.get("type") == "image_url")
        t0 = time.time()
        resp = self.client().chat.completions.create(**req)
        lat = time.time() - t0
        msg = resp.choices[0].message
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            raw = tc.function.arguments or ""
            try:
                args = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id or f"call_{i}", name=tc.function.name, arguments=args, raw_arguments=raw))
        u = resp.usage
        usage = dict(input_tokens=getattr(u, "prompt_tokens", 0) or 0, cached_tokens=0,
                     output_tokens=getattr(u, "completion_tokens", 0) or 0, reasoning_tokens=0) if u else {}
        return BackendResponse(tool_calls=calls, text=msg.content or None, usage=usage, latency_s=lat,
                               request_stats=dict(n_images=n_img, n_messages=len(req["messages"])))
