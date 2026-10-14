"""Google Gemini backend (Gemini API, official SDK `google-genai`).

Same conversation as the other backends, converted to Gemini `contents`:
- system prompt = system_instruction; tools = function declarations with the harness's JSON schemas;
- every model turn is stored as the full returned `Content` (python-mode dump, so `thought_signature` bytes survive)
  and replayed unchanged: Gemini 3 models need the signatures back to continue their reasoning after a tool call;
- tool outputs of one turn become `function_response` parts at the start of the next user content; images the
  harness attaches after them follow in the same user content;
- thinking: `thinking_level` (Gemini 3) or `thinking_budget` (Gemini 2.5), with thought summaries logged;
- implicit prompt caching is automatic on the Gemini API (cached tokens are reported in usage).

Key from GEMINI_API_KEY / GOOGLE_API_KEY (the SDK reads them). The client is created lazily; tests inject a mock.
"""
import base64
import time

from ..context import ToolCall, prune_live_images
from .base import Backend, BackendResponse, import_sdk


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _strip_additional(schema):
    """The OpenAPI subset of `parameters` has no additionalProperties; used only if the SDK lacks parameters_json_schema."""
    if isinstance(schema, dict):
        return {k: _strip_additional(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_additional(v) for v in schema]
    return schema


def _image_part(image, quality=None):
    head, data = image.data_url(quality).split(",", 1)
    mime = head[len("data:"):].split(";", 1)[0]
    return dict(inline_data=dict(mime_type=mime, data=base64.b64decode(data)))


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(self, model="gemini-3.5-pro", thinking_level="high", thinking_budget=None, include_thoughts=True,
                 max_output_tokens=65536, media_resolution=None, max_live_images=None, jpeg_quality=90,
                 timeout=900.0, client=None):
        self.model = model
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.include_thoughts = include_thoughts
        self.max_output_tokens = max_output_tokens
        self.media_resolution = media_resolution
        self.max_live_images = max_live_images
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self._client = client
        self._json_schema_field = None

    def config(self):
        return dict(name=self.name, model=self.model, thinking_level=self.thinking_level,
                    thinking_budget=self.thinking_budget, include_thoughts=self.include_thoughts,
                    max_output_tokens=self.max_output_tokens, media_resolution=self.media_resolution,
                    max_live_images=self.max_live_images, jpeg_quality=self.jpeg_quality, timeout=self.timeout)

    def client(self):
        if self._client is None:
            genai = import_sdk("google.genai")
            self._client = genai.Client(http_options=dict(timeout=int(self.timeout * 1000)))
        return self._client

    def _declarations(self, specs):
        if self._json_schema_field is None:
            try:
                types = import_sdk("google.genai.types")
                self._json_schema_field = "parameters_json_schema" in types.FunctionDeclaration.model_fields
            except ImportError:  # tests without the SDK
                self._json_schema_field = True
        out = []
        for s in specs:
            d = dict(name=s["name"], description=s["description"])
            if self._json_schema_field:
                d["parameters_json_schema"] = s["parameters"]
            else:
                d["parameters"] = _strip_additional(s["parameters"])
            out.append(d)
        return [dict(function_declarations=out)]

    # ------------------------------------------------------------------ request
    def convert_items(self, items):
        contents = []

        def push(role, parts):
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append(dict(role=role, parts=list(parts)))

        for it in items:
            kind = it["kind"]
            if kind == "user":
                parts = []
                for p in it["parts"]:
                    if p["type"] == "text":
                        parts.append(dict(text=p["text"]))
                    else:
                        parts.append(_image_part(p["image"], self.jpeg_quality))
                push("user", parts)
            elif kind == "assistant":
                if it.get("raw") is not None and it.get("backend") == self.name:
                    content = dict(it["raw"][0])
                    contents.append(dict(role="model", parts=list(content.get("parts") or [])))
                    continue
                parts = ([dict(text=it["text"])] if it.get("text") else [])
                parts += [dict(function_call=dict(name=tc.name, args=tc.arguments)) for tc in it["tool_calls"]]
                push("model", parts or [dict(text="(no output)")])
            elif kind == "tool_output":
                fr = dict(name=it["name"], response=dict(output=it["text"]))
                if it["call_id"] and not str(it["call_id"]).startswith("gemini_call_"):
                    fr["id"] = it["call_id"]
                push("user", [dict(function_response=fr)])
            else:
                raise ValueError(kind)
        return contents

    def build_request(self, conv, tools):
        items = prune_live_images(conv.items, self.max_live_images)
        thinking = dict(include_thoughts=self.include_thoughts)
        if self.thinking_budget is not None:
            thinking["thinking_budget"] = int(self.thinking_budget)
        elif self.thinking_level:
            thinking["thinking_level"] = self.thinking_level
        config = dict(system_instruction=conv.system, tools=self._declarations(tools),
                      tool_config=dict(function_calling_config=dict(mode="AUTO")),
                      automatic_function_calling=dict(disable=True), thinking_config=thinking,
                      max_output_tokens=int(self.max_output_tokens))
        if self.media_resolution:
            config["media_resolution"] = self.media_resolution
        return dict(model=self.model, contents=self.convert_items(items), config=config)

    @staticmethod
    def request_stats(req):
        n_img, img_bytes, txt = 0, 0, len(req["config"].get("system_instruction") or "")
        for c in req["contents"]:
            for p in c["parts"]:
                p = p if isinstance(p, dict) else {}
                if p.get("inline_data"):
                    n_img += 1
                    img_bytes += len(p["inline_data"]["data"])
                elif p.get("text"):
                    txt += len(p["text"])
                elif p.get("function_response"):
                    txt += len(str(p["function_response"].get("response", "")))
        return dict(n_images=n_img, image_payload_bytes=img_bytes, text_chars=txt, n_contents=len(req["contents"]))

    # ------------------------------------------------------------------ call
    def act(self, conv, tools):
        req = self.build_request(conv, tools)
        stats = self.request_stats(req)
        types = import_sdk("google.genai.types")  # typed objects: plain nested dicts are sent with snake_case keys
        t0 = time.time()
        resp = self.client().models.generate_content(
            model=req["model"], contents=[types.Content.model_validate(c) for c in req["contents"]],
            config=types.GenerateContentConfig.model_validate(req["config"]))
        out = self.parse_response(resp)
        out.latency_s = time.time() - t0
        out.request_stats = stats
        return out

    def parse_response(self, resp):
        cands = _get(resp, "candidates") or []
        cand = cands[0] if cands else None
        content = _get(cand, "content") if cand is not None else None
        parts = (_get(content, "parts") or []) if content is not None else []
        calls, texts, summ = [], [], []
        for i, p in enumerate(parts):
            fc = _get(p, "function_call")
            if fc is not None:
                cid = _get(fc, "id") or f"gemini_call_{i}"
                calls.append(ToolCall(id=cid, name=_get(fc, "name"), arguments=dict(_get(fc, "args") or {})))
            elif _get(p, "text"):
                (summ if _get(p, "thought") else texts).append(_get(p, "text"))
        u = _get(resp, "usage_metadata")
        usage = {}
        if u is not None:
            usage = dict(input_tokens=_get(u, "prompt_token_count", 0) or 0,
                         cached_tokens=_get(u, "cached_content_token_count", 0) or 0,
                         output_tokens=(_get(u, "candidates_token_count", 0) or 0) + (_get(u, "thoughts_token_count", 0) or 0),
                         reasoning_tokens=_get(u, "thoughts_token_count", 0) or 0)
        finish = str(_get(cand, "finish_reason") or "") if cand is not None else "NO_CANDIDATE"
        status, err = "completed", None
        if not parts:
            status, err = "incomplete", f"incomplete: empty response (finish_reason={finish})"
        elif "MAX_TOKENS" in finish or "MALFORMED" in finish:
            status, err = "incomplete", f"incomplete: finish_reason={finish}"
        elif "SAFETY" in finish or "PROHIBITED" in finish or "BLOCKLIST" in finish:
            status, err = "refusal", f"refusal: finish_reason={finish}"
        if content is not None and hasattr(content, "model_dump"):
            raw = [content.model_dump(exclude_none=True)]  # python mode: thought_signature stays bytes
        elif content is not None:
            raw = [dict(content)]
        else:
            raw = None
        return BackendResponse(tool_calls=calls, text="\n".join(texts) or None,
                               reasoning_summary="\n".join(summ) or None, usage=usage, raw=raw,
                               status=status, error=err)
