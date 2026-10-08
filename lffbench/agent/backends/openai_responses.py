"""OpenAI Responses API backend (default model gpt-6-astra).

Stateless requests: every call re-sends instructions + tools + the full (pruned) conversation with
store=False; reasoning items are replayed with their encrypted content so the model keeps its chain
of thought across tool calls within an attempt. The request prefix (instructions, tools, task text,
packed history of earlier attempts) is byte-identical across the calls of an attempt, which is what
OpenAI's automatic prompt caching needs. Images go in as base64 JPEG data URLs.

Key and endpoint come from OPENAI_API_KEY / OPENAI_BASE_URL. Nothing here is executed at import time;
the client is created lazily on the first real call (tests inject a mock client).
"""
import json
import os
import time

from ..context import ToolCall, prune_live_images
from .base import Backend, BackendResponse, import_openai

# GPT-6 Astra: reasoning.effort supports low/medium/high/xhigh/max; `none` is NOT supported.
GPT6_ASTRA_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(item):
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "to_dict"):
        return item.to_dict(mode="json")
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    return dict(vars(item))


def responses_tools(specs, strict=True):
    return [dict(type="function", name=s["name"], description=s["description"], parameters=s["parameters"],
                 strict=strict) for s in specs]


class OpenAIResponsesBackend(Backend):
    name = "openai"

    def __init__(self, model="gpt-6-astra", reasoning_effort="medium", reasoning_summary=None, image_detail="auto",
                 max_output_tokens=None, service_tier=None, prompt_cache_key=None, prompt_cache_retention=None,
                 store=False, replay_reasoning=True, max_live_images=16, jpeg_quality=90, timeout=900.0,
                 max_retries=4, extra_body=None, strict_tools=True, cache_breakpoints="history+last",
                 prompt_cache_options=None, client=None, http_client=None):
        if model.startswith("gpt-6-astra") and reasoning_effort not in GPT6_ASTRA_EFFORTS:
            raise ValueError(f"gpt-6-astra supports reasoning effort {GPT6_ASTRA_EFFORTS} (no 'none'), "
                             f"got {reasoning_effort!r}")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.image_detail = image_detail
        self.max_output_tokens = max_output_tokens
        self.service_tier = service_tier
        self.prompt_cache_key = prompt_cache_key
        self.prompt_cache_retention = prompt_cache_retention
        self.store = store
        self.replay_reasoning = replay_reasoning
        self.max_live_images = max_live_images
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body
        self.strict_tools = strict_tools
        # explicit prompt-cache breakpoints (Responses API, gpt-5.6+): "none" | "history" | "history+last"
        assert cache_breakpoints in ("none", "history", "history+last"), cache_breakpoints
        self.cache_breakpoints = cache_breakpoints
        self.prompt_cache_options = prompt_cache_options
        self._client = client
        self._http_client = http_client  # optional custom transport (tests, proxies)
        self._cache_key = prompt_cache_key

    def config(self):
        return dict(name=self.name, model=self.model, reasoning_effort=self.reasoning_effort,
                    reasoning_summary=self.reasoning_summary, image_detail=self.image_detail,
                    max_output_tokens=self.max_output_tokens, service_tier=self.service_tier,
                    prompt_cache_key=self.prompt_cache_key, prompt_cache_retention=self.prompt_cache_retention,
                    store=self.store, replay_reasoning=self.replay_reasoning, max_live_images=self.max_live_images,
                    jpeg_quality=self.jpeg_quality, timeout=self.timeout, max_retries=self.max_retries,
                    extra_body=self.extra_body, cache_breakpoints=self.cache_breakpoints,
                    prompt_cache_options=self.prompt_cache_options, base_url=os.environ.get("OPENAI_BASE_URL"))

    # ------------------------------------------------------------------ client
    def client(self):
        if self._client is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set; the OpenAI backend cannot run without it")
            openai = import_openai()
            kw = dict(api_key=key, base_url=os.environ.get("OPENAI_BASE_URL") or None, timeout=self.timeout,
                      max_retries=self.max_retries)
            if self._http_client is not None:
                kw["http_client"] = self._http_client
            self._client = openai.OpenAI(**kw)
        return self._client

    def begin_run(self, info):
        if self.prompt_cache_key is None:
            # route all calls of one run to the same cache shard
            self._cache_key = f"lffbench:{getattr(info.task, 'name', 'task')}:{info.seed}"

    # ------------------------------------------------------------------ request
    def convert_items(self, items):
        out = []
        last_user = max((i for i, it in enumerate(items) if it["kind"] == "user"), default=-1)
        for i, it in enumerate(items):
            kind = it["kind"]
            if kind == "user":
                content = []
                for p in it["parts"]:
                    if p["type"] == "text":
                        content.append(dict(type="input_text", text=p["text"]))
                    else:
                        content.append(dict(type="input_image", image_url=p["image"].data_url(self.jpeg_quality),
                                            detail=self.image_detail))
                # explicit cache breakpoints: end of the stable history item (byte-identical for the whole
                # attempt) and end of the newest input (reused by the next call of the attempt)
                if content and ((self.cache_breakpoints != "none" and not it.get("live"))
                                or (self.cache_breakpoints == "history+last" and i == last_user)):
                    content[-1]["prompt_cache_breakpoint"] = dict(mode="explicit")
                out.append(dict(role="user", content=content))
            elif kind == "assistant":
                if it.get("raw") and it.get("backend") == self.name:
                    for r in it["raw"]:
                        if r.get("type") == "reasoning" and not self.replay_reasoning:
                            continue
                        out.append(r)
                else:
                    if it.get("text"):
                        out.append(dict(role="assistant", content=it["text"]))
                    for tc in it["tool_calls"]:
                        out.append(dict(type="function_call", call_id=tc.id, name=tc.name,
                                        arguments=tc.raw_arguments or json.dumps(tc.arguments)))
            elif kind == "tool_output":
                out.append(dict(type="function_call_output", call_id=it["call_id"], output=it["text"]))
            else:
                raise ValueError(kind)
        return out

    def build_request(self, conv, tools):
        items = prune_live_images(conv.items, self.max_live_images)
        req = dict(model=self.model, instructions=conv.system, input=self.convert_items(items),
                   tools=responses_tools(tools, self.strict_tools), tool_choice="auto", parallel_tool_calls=False,
                   store=self.store)
        reasoning = dict(effort=self.reasoning_effort)
        if self.reasoning_summary:
            reasoning["summary"] = self.reasoning_summary
        req["reasoning"] = reasoning
        if not self.store and self.replay_reasoning:
            req["include"] = ["reasoning.encrypted_content"]
        if self.max_output_tokens:
            req["max_output_tokens"] = int(self.max_output_tokens)
        if self.service_tier:
            req["service_tier"] = self.service_tier
        if self._cache_key:
            req["prompt_cache_key"] = self._cache_key
        if self.prompt_cache_retention:
            req["prompt_cache_retention"] = self.prompt_cache_retention
        if self.prompt_cache_options:
            req["prompt_cache_options"] = dict(self.prompt_cache_options)
        if self.extra_body:
            req["extra_body"] = dict(self.extra_body)
        return req

    @staticmethod
    def request_stats(req):
        n_img, img_bytes, txt = 0, 0, len(req.get("instructions") or "")
        for it in req["input"]:
            for c in (it.get("content") if isinstance(it.get("content"), list) else []):
                if c.get("type") == "input_image":
                    n_img += 1
                    img_bytes += len(c["image_url"])
                elif "text" in c:
                    txt += len(c["text"])
            if isinstance(it.get("content"), str):
                txt += len(it["content"])
            if it.get("type") == "function_call_output":
                txt += len(it["output"]) if isinstance(it["output"], str) else 0
            if it.get("type") == "function_call":
                txt += len(it.get("arguments", ""))
        tools_chars = len(json.dumps(req.get("tools", [])))
        return dict(n_images=n_img, image_payload_bytes=img_bytes, text_chars=txt + tools_chars,
                    n_input_items=len(req["input"]))

    # ------------------------------------------------------------------ call
    def act(self, conv, tools):
        req = self.build_request(conv, tools)
        stats = self.request_stats(req)
        t0 = time.time()
        resp = self.client().responses.create(**req)
        out = self.parse_response(resp)
        out.latency_s = time.time() - t0
        out.request_stats = stats
        return out

    def parse_response(self, resp):
        calls, texts, summ = [], [], []
        output = _get(resp, "output") or []
        for item in output:
            t = _get(item, "type")
            if t == "function_call":
                raw = _get(item, "arguments") or ""
                try:
                    args = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(id=_get(item, "call_id"), name=_get(item, "name"), arguments=args,
                                      raw_arguments=raw))
            elif t == "message":
                for c in _get(item, "content") or []:
                    if _get(c, "type") == "output_text":
                        texts.append(_get(c, "text") or "")
            elif t == "reasoning":
                for s in _get(item, "summary") or []:
                    summ.append(_get(s, "text") or "")
        u = _get(resp, "usage")
        usage = {}
        if u is not None:
            itd = _get(u, "input_tokens_details")
            otd = _get(u, "output_tokens_details")
            usage = dict(input_tokens=_get(u, "input_tokens", 0) or 0,
                         cached_tokens=(_get(itd, "cached_tokens", 0) or 0) if itd is not None else 0,
                         cache_write_tokens=(_get(itd, "cache_write_tokens", 0) or 0) if itd is not None else 0,
                         output_tokens=_get(u, "output_tokens", 0) or 0,
                         reasoning_tokens=(_get(otd, "reasoning_tokens", 0) or 0) if otd is not None else 0)
        status = _get(resp, "status") or "completed"
        err = None
        if status == "incomplete":
            err = "incomplete: " + str(_get(_get(resp, "incomplete_details"), "reason"))
        return BackendResponse(tool_calls=calls, text="\n".join(t for t in texts if t) or None,
                               reasoning_summary="\n".join(s for s in summ if s) or None, usage=usage,
                               raw=[_to_dict(i) for i in output], status=status, error=err)
