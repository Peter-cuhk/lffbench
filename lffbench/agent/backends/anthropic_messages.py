"""Anthropic Messages API backend (Claude Opus 5.5 / Sonnet 5.5 / Fable 5.1 ...), official SDK `anthropic`.

Same conversation as the OpenAI backends, converted to Messages API blocks:
- system prompt = conv.system; tools = the harness tool specs (strict, one call per turn like the GPT player);
- every assistant turn is stored with its full `content` (thinking blocks with signatures, text, tool_use) and replayed
  unchanged, so the model keeps its reasoning across tool calls; the request history is append-only (current Claude
  models reject edited history that carries thinking blocks), so no live-image pruning by default;
- tool outputs of one turn become `tool_result` blocks at the start of the next user message; images the harness
  attaches after them (observations, view_images) follow in the same user message;
- adaptive thinking with an explicit effort (the per-model default differs: Opus 5.5 medium, Sonnet 5.5 high);
- automatic prompt caching (top-level cache_control): every request caches its whole prefix for the next call;
- streaming, so long thinking turns do not hit the HTTP timeout.
No refusal fallback on purpose: a fallback would silently swap the model under test. A refusal is reported as
status "refusal" and the player ends the run.

Key from ANTHROPIC_API_KEY (or any credential the SDK resolves). The client is created lazily on the first real
call; tests inject a mock client.
"""
import time

from ..context import ToolCall, prune_live_images
from .base import Backend, BackendResponse, import_sdk

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(block):
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "to_dict"):
        return block.to_dict()
    return block.model_dump(mode="json", exclude_none=True)


def anthropic_tools(specs, strict=True):
    out = []
    for s in specs:
        t = dict(name=s["name"], description=s["description"], input_schema=s["parameters"])
        if strict:
            t["strict"] = True
        out.append(t)
    return out


def image_block(image, quality=None):
    url = image.data_url(quality)
    head, data = url.split(",", 1)  # data:<mime>;base64,<data>
    mime = head[len("data:"):].split(";", 1)[0]
    return dict(type="image", source=dict(type="base64", media_type=mime, data=data))


class AnthropicMessagesBackend(Backend):
    name = "anthropic"

    def __init__(self, model="claude-opus-5-5", effort="high", thinking_display="summarized", max_tokens=64000,
                 max_live_images=None, jpeg_quality=90, timeout=900.0, max_retries=2, strict_tools=True,
                 parallel_tool_calls=False, cache=True, client=None):
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
        self.model = model
        self.effort = effort
        self.thinking_display = thinking_display  # "summarized" (logged as the reasoning summary) | "omitted"
        self.max_tokens = max_tokens
        self.max_live_images = max_live_images
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.max_retries = max_retries
        self.strict_tools = strict_tools
        self.parallel_tool_calls = parallel_tool_calls
        self.cache = cache
        self._client = client

    def config(self):
        return dict(name=self.name, model=self.model, effort=self.effort, thinking_display=self.thinking_display,
                    max_tokens=self.max_tokens, max_live_images=self.max_live_images, jpeg_quality=self.jpeg_quality,
                    timeout=self.timeout, max_retries=self.max_retries, strict_tools=self.strict_tools,
                    parallel_tool_calls=self.parallel_tool_calls, cache=self.cache)

    def client(self):
        if self._client is None:
            anthropic = import_sdk("anthropic")
            self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    # ------------------------------------------------------------------ request
    def convert_items(self, items):
        msgs = []

        def push(role, blocks):
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"].extend(blocks)
            else:
                msgs.append(dict(role=role, content=list(blocks)))

        for it in items:
            kind = it["kind"]
            if kind == "user":
                blocks = []
                for p in it["parts"]:
                    if p["type"] == "text":
                        blocks.append(dict(type="text", text=p["text"]))
                    else:
                        blocks.append(image_block(p["image"], self.jpeg_quality))
                push("user", blocks)
            elif kind == "assistant":
                if it.get("raw") is not None and it.get("backend") == self.name:
                    blocks = [dict(b) for b in it["raw"]]
                else:  # a turn produced by another backend: plain text + tool calls, no thinking
                    blocks = ([dict(type="text", text=it["text"])] if it.get("text") else [])
                    blocks += [dict(type="tool_use", id=tc.id, name=tc.name, input=tc.arguments)
                               for tc in it["tool_calls"]]
                if not blocks:
                    blocks = [dict(type="text", text="(no output)")]
                push("assistant", blocks)
            elif kind == "tool_output":
                push("user", [dict(type="tool_result", tool_use_id=it["call_id"], content=it["text"])])
            else:
                raise ValueError(kind)
        return msgs

    def build_request(self, conv, tools):
        items = prune_live_images(conv.items, self.max_live_images)
        req = dict(model=self.model, max_tokens=int(self.max_tokens), system=conv.system,
                   messages=self.convert_items(items), tools=anthropic_tools(tools, self.strict_tools),
                   tool_choice=dict(type="auto", disable_parallel_tool_use=not self.parallel_tool_calls),
                   thinking=dict(type="adaptive", display=self.thinking_display),
                   output_config=dict(effort=self.effort))
        if self.cache:
            req["cache_control"] = dict(type="ephemeral")
        return req

    @staticmethod
    def request_stats(req):
        n_img, img_bytes, txt = 0, 0, len(req.get("system") or "")
        for m in req["messages"]:
            for b in m["content"]:
                t = b.get("type")
                if t == "image":
                    n_img += 1
                    img_bytes += len(b["source"]["data"])
                elif t == "text":
                    txt += len(b["text"])
                elif t == "tool_result":
                    txt += len(b["content"]) if isinstance(b["content"], str) else 0
        return dict(n_images=n_img, image_payload_bytes=img_bytes, text_chars=txt, n_messages=len(req["messages"]))

    # ------------------------------------------------------------------ call
    def act(self, conv, tools):
        req = self.build_request(conv, tools)
        stats = self.request_stats(req)
        t0 = time.time()
        with self.client().messages.stream(**req) as stream:
            msg = stream.get_final_message()
        out = self.parse_response(msg)
        out.latency_s = time.time() - t0
        out.request_stats = stats
        return out

    def parse_response(self, msg):
        calls, texts, summ = [], [], []
        content = _get(msg, "content") or []
        for b in content:
            t = _get(b, "type")
            if t == "tool_use":
                args = _get(b, "input") or {}
                calls.append(ToolCall(id=_get(b, "id"), name=_get(b, "name"), arguments=dict(args)))
            elif t == "text":
                texts.append(_get(b, "text") or "")
            elif t == "thinking":
                if _get(b, "thinking"):
                    summ.append(_get(b, "thinking"))
        u = _get(msg, "usage")
        usage = {}
        if u is not None:
            fresh = _get(u, "input_tokens", 0) or 0
            read = _get(u, "cache_read_input_tokens", 0) or 0
            write = _get(u, "cache_creation_input_tokens", 0) or 0
            # same meaning as the OpenAI backend: input_tokens = whole prompt, cached_tokens = the part read from cache
            usage = dict(input_tokens=fresh + read + write, cached_tokens=read, cache_write_tokens=write,
                         output_tokens=_get(u, "output_tokens", 0) or 0, reasoning_tokens=0)
        stop = _get(msg, "stop_reason")
        status, err = "completed", None
        if stop == "refusal":
            d = _get(msg, "stop_details")
            status, err = "refusal", f"refusal: category={_get(d, 'category')} {_get(d, 'explanation') or ''}".strip()
        elif stop == "max_tokens":
            status, err = "incomplete", "incomplete: max_tokens"
        return BackendResponse(tool_calls=calls, text="\n".join(t for t in texts if t) or None,
                               reasoning_summary="\n".join(summ) or None, usage=usage,
                               raw=[_to_dict(b) for b in content], status=status, error=err)
