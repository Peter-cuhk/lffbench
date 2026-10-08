#!/usr/bin/env python3
"""Minimal OpenAI-compatible Chat Completions server (with tool calling) for Qwen3-VL.

Runs a local HF-format Qwen3-VL checkpoint with transformers (>= 4.57) on one
CUDA-compatible device (here: Alibaba PPU) and exposes:

  POST /v1/chat/completions   OpenAI chat.completion (non-streaming; stream=true
                              is emulated: the full answer is generated first
                              and then sent as SSE chunks)
  GET  /v1/models             lists the single served model
  GET  /health                liveness + device memory

Request format accepted (subset of OpenAI):
  messages: system / developer / user / assistant (with tool_calls) / tool
            user/tool content may be a string or a list of
            {"type":"text","text":...} and
            {"type":"image_url","image_url":{"url": "data:image/...;base64,..." | http(s) URL | file path}}
  tools: [{"type":"function","function":{"name","description","parameters"}}]
  tool_choice: "auto" (default) | "none" (tools hidden from the prompt) |
               "required" (generation is prefixed with "<tool_call>\n") |
               {"type":"function","function":{"name":X}} (prefixed with the call header for X)
  parallel_tool_calls: false -> only the first parsed tool call is returned
  max_tokens / max_completion_tokens (default --default-max-tokens, 1024)
  temperature (absent or 0 -> greedy), top_p, top_k, seed, stop (str or list)
Ignored: n (always 1), response_format, logprobs, presence/frequency penalty, user.

The model emits Hermes-style <tool_call>{"name":..., "arguments":{...}}</tool_call>
blocks; they are parsed into choices[0].message.tool_calls with arguments as a JSON
string, and the remaining text (if any) becomes message.content (else null).

Generation is serialised with a lock (one request on the device at a time); the
HTTP layer is a stdlib ThreadingHTTPServer so /health stays responsive.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = logging.getLogger("qwen_vl_server")

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
END_MARKERS = ("<|im_end|>", "<|endoftext|>")


# ----------------------------------------------------------------------------
# Request conversion
# ----------------------------------------------------------------------------
class BadRequest(Exception):
    pass


def load_image(url: str):
    from PIL import Image

    if url.startswith("data:"):
        try:
            _, b64 = url.split(",", 1)
        except ValueError as e:
            raise BadRequest("malformed data URL") from e
        raw = base64.b64decode(b64)
    elif url.startswith("http://") or url.startswith("https://"):
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    else:
        path = url[len("file://"):] if url.startswith("file://") else url
        with open(path, "rb") as f:
            raw = f.read()
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img.convert("RGB")


def convert_content(content, images: list):
    """OpenAI content (str | list[part] | None) -> Qwen template content.

    Images are replaced by {"type": "image"} placeholders and the decoded PIL images
    are appended to `images` in prompt order (the chat template emits one
    <|vision_start|><|image_pad|><|vision_end|> per placeholder, in the same order).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise BadRequest(f"unsupported content type: {type(content).__name__}")
    out = []
    for part in content:
        if isinstance(part, str):
            out.append({"type": "text", "text": part})
            continue
        ptype = part.get("type")
        if ptype in ("text", "input_text"):
            out.append({"type": "text", "text": part.get("text", "")})
        elif ptype in ("image_url", "input_image", "image"):
            iu = part.get("image_url", part.get("image"))
            url = iu.get("url") if isinstance(iu, dict) else iu
            if not isinstance(url, str):
                raise BadRequest("image part without url")
            try:
                images.append(load_image(url))
            except BadRequest:
                raise
            except Exception as e:  # noqa: BLE001
                raise BadRequest(f"could not load image ({url[:60]}...): {type(e).__name__}: {e}") from e
            out.append({"type": "image"})
        else:
            raise BadRequest(f"unsupported content part type: {ptype}")
    return out


def text_only(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def convert_messages(messages: list):
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty list")
    images: list = []
    system_texts = []
    conv = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            # The Qwen template only renders a leading system message; merge all of them.
            system_texts.append(text_only(m.get("content")))
        elif role == "user":
            conv.append({"role": "user", "content": convert_content(m.get("content"), images)})
        elif role == "assistant":
            c = m.get("content")
            msg = {"role": "assistant", "content": text_only(c) if not isinstance(c, str) else c}
            tcs = m.get("tool_calls") or []
            if tcs:
                norm = []
                for tc in tcs:
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    norm.append({"type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
                msg["tool_calls"] = norm
            conv.append(msg)
        elif role in ("tool", "function"):
            conv.append({"role": "tool", "content": convert_content(m.get("content"), images)})
        else:
            raise BadRequest(f"unsupported role: {role}")
    if system_texts:
        conv.insert(0, {"role": "system", "content": "\n\n".join(system_texts)})
    return conv, images


# ----------------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------------
def _loads_lenient(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # common slips: trailing garbage after the object, missing final brace
    start = s.find("{")
    if start >= 0:
        dec = json.JSONDecoder()
        try:
            obj, _ = dec.raw_decode(s[start:])
            return obj
        except json.JSONDecodeError:
            pass
        for extra in ("}", "}}", "\"}}"):
            try:
                return json.loads(s[start:] + extra)
            except json.JSONDecodeError:
                continue
    return None


def parse_tool_calls(text: str):
    """Return (content_or_None, tool_calls_list)."""
    calls = []
    spans = []
    for mt in TOOL_CALL_RE.finditer(text):
        spans.append(mt.span())
        calls.append(mt.group(1))
    # an unterminated trailing <tool_call> (e.g. hit max_tokens / model forgot the tag)
    last_end = spans[-1][1] if spans else 0
    tail_idx = text.find("<tool_call>", last_end)
    if tail_idx >= 0:
        spans.append((tail_idx, len(text)))
        calls.append(text[tail_idx + len("<tool_call>"):])

    tool_calls = []
    keep_raw = []  # unparsable blocks stay in content
    for (a, b), body in zip(spans, calls):
        obj = _loads_lenient(body)
        if not isinstance(obj, dict) or "name" not in obj:
            keep_raw.append((a, b))
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            parsed = _loads_lenient(args)
            args_str = json.dumps(parsed, ensure_ascii=False) if parsed is not None else args
        else:
            args_str = json.dumps(args, ensure_ascii=False)
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": str(obj["name"]), "arguments": args_str},
        })

    # content = text with successfully parsed blocks removed
    parsed_spans = [s for s in spans if s not in keep_raw]
    pieces, pos = [], 0
    for a, b in sorted(parsed_spans):
        pieces.append(text[pos:a])
        pos = b
    pieces.append(text[pos:])
    content = "".join(pieces).strip()
    return (content if content else None), tool_calls


# ----------------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.args = args
        self.device = args.device
        t0 = time.time()
        proc_kwargs = {}
        if args.min_pixels:
            proc_kwargs["min_pixels"] = args.min_pixels
        if args.max_pixels:
            proc_kwargs["max_pixels"] = args.max_pixels
        if args.attn == "sdpa" and not args.sdpa_native_gqa:
            # PPU quirk (measured): torch SDPA with enable_gqa=True fails for decode steps
            # (q_len=1, no mask) with "q must have shape (batch_size, seqlen_q, num_heads,
            # head_size_og)". Make transformers repeat K/V heads itself instead.
            import transformers.integrations.sdpa_attention as _sdpa

            _sdpa.use_gqa_in_sdpa = lambda attention_mask, key: False
        self.processor = AutoProcessor.from_pretrained(args.model, **proc_kwargs)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation=args.attn,
            device_map={"": self.device},
        ).eval()
        self.load_time = time.time() - t0
        self.eos_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in END_MARKERS]
        self.eos_ids = [i for i in self.eos_ids if isinstance(i, int) and i >= 0]
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_ids[-1]
        self.lock = threading.Lock()
        LOG.info("model loaded in %.1fs on %s (attn=%s); mem allocated %.2f GiB",
                 self.load_time, self.device, args.attn, torch.cuda.memory_allocated(self.device) / 2**30)

    def mem(self):
        t = self.torch
        return {
            "allocated_gib": round(t.cuda.memory_allocated(self.device) / 2**30, 2),
            "reserved_gib": round(t.cuda.memory_reserved(self.device) / 2**30, 2),
            "max_allocated_gib": round(t.cuda.max_memory_allocated(self.device) / 2**30, 2),
        }

    def chat(self, req: dict):
        torch = self.torch
        a = self.args
        messages, images = convert_messages(req.get("messages"))
        tools = req.get("tools") or None
        tool_choice = req.get("tool_choice", "auto")
        prefix = ""
        if tools:
            if tool_choice == "none":
                tools = None
            elif tool_choice == "required":
                prefix = "<tool_call>\n"
            elif isinstance(tool_choice, dict):
                fname = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
                if fname:
                    prefix = '<tool_call>\n{"name": "' + fname + '", "arguments": '
        tmpl_tools = None
        if tools:
            tmpl_tools = []
            for t in tools:
                if isinstance(t, dict) and "function" not in t and "name" in t:
                    t = {"type": "function", "function": t}
                tmpl_tools.append(t)

        prompt = self.processor.apply_chat_template(
            messages, tools=tmpl_tools, tokenize=False, add_generation_prompt=True
        ) + prefix
        inputs = self.processor(
            text=[prompt], images=images if images else None, return_tensors="pt"
        ).to(self.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        if n_prompt > a.max_prompt_tokens:
            raise BadRequest(f"prompt has {n_prompt} tokens > --max-prompt-tokens {a.max_prompt_tokens}")

        max_new = req.get("max_completion_tokens") or req.get("max_tokens") or a.default_max_tokens
        max_new = int(min(max_new, a.max_tokens_cap))
        temperature = req.get("temperature")
        if temperature is None:
            temperature = a.default_temperature
        gen_kwargs = dict(
            max_new_tokens=max_new,
            eos_token_id=self.eos_ids,
            pad_token_id=self.pad_id,
            use_cache=True,
        )
        if temperature and float(temperature) > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature),
                              top_p=float(req.get("top_p") or 0.8), top_k=int(req.get("top_k") or 20))
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if req.get("seed") is not None:
            torch.manual_seed(int(req["seed"]))

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0, n_prompt:].tolist()
        # count tokens up to and including the first EOS
        n_completion = len(new_ids)
        hit_eos = False
        for i, tid in enumerate(new_ids):
            if tid in self.eos_ids:
                n_completion = i + 1
                hit_eos = True
                break
        text = self.tokenizer.decode(new_ids[:n_completion], skip_special_tokens=False)
        for mk in END_MARKERS:
            text = text.replace(mk, "")
        text = prefix + text

        stop = req.get("stop")
        stopped_by_str = False
        if stop:
            for s in ([stop] if isinstance(stop, str) else stop):
                k = text.find(s, len(prefix))
                if k >= 0:
                    text = text[:k]
                    stopped_by_str = True

        if tools:
            content, tool_calls = parse_tool_calls(text)
            if tool_calls and req.get("parallel_tool_calls") is False:
                tool_calls = tool_calls[:1]
        else:
            content, tool_calls = (text.strip() or None), []

        if tool_calls:
            finish = "tool_calls"
        elif hit_eos or stopped_by_str:
            finish = "stop"
        else:
            finish = "length"
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                 "total_tokens": n_prompt + n_completion}
        return message, finish, usage, len(images), text


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
ENGINE: Engine | None = None
ARGS = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # route http.server logs through logging
        LOG.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str, etype: str):
        self._send_json(code, {"error": {"message": msg, "type": etype, "code": code}})

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok", "model": ARGS.served_name, "memory": ENGINE.mem(),
                                  "load_time_s": round(ENGINE.load_time, 1)})
        elif self.path.rstrip("/") == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": ARGS.served_name, "object": "model", "created": int(time.time()), "owned_by": "local"}]})
        else:
            self._error(404, f"no route {self.path}", "not_found")

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"no route {self.path}", "not_found")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            self._error(400, f"invalid JSON body: {e}", "invalid_request_error")
            return
        t0 = time.time()
        try:
            with ENGINE.lock:
                t_lock = time.time()
                message, finish, usage, n_img, raw = ENGINE.chat(req)
        except BadRequest as e:
            self._error(400, str(e), "invalid_request_error")
            return
        except Exception as e:  # noqa: BLE001
            LOG.error("generation failed: %s\n%s", e, traceback.format_exc())
            if "out of memory" in str(e).lower():
                try:
                    ENGINE.torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            self._error(500, f"{type(e).__name__}: {e}", "server_error")
            return
        dt = time.time() - t0
        LOG.info("req images=%d prompt=%d completion=%d finish=%s tool_calls=%d queue=%.2fs gen=%.2fs mem=%s",
                 n_img, usage["prompt_tokens"], usage["completion_tokens"], finish,
                 len(message.get("tool_calls", [])), t_lock - t0, time.time() - t_lock, ENGINE.mem())
        if ARGS.log_outputs:
            LOG.info("raw output: %r", raw)
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model_name = req.get("model") or ARGS.served_name
        if req.get("stream"):
            self._send_stream(cid, created, model_name, message, finish, usage, req)
            return
        self._send_json(200, {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
            "usage": usage,
            "server_timing": {"latency_s": round(dt, 3)},
        })

    def _send_stream(self, cid, created, model_name, message, finish, usage, req):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def chunk(delta, fr=None, with_usage=False):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": fr, "logprobs": None}]}
            if with_usage:
                obj["usage"] = usage
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

        chunk({"role": "assistant", "content": message.get("content") or ""})
        for i, tc in enumerate(message.get("tool_calls", [])):
            chunk({"tool_calls": [dict(tc, index=i)]})
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        chunk({}, finish, with_usage=include_usage)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    global ENGINE, ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    p.add_argument("--served-name", default="Qwen3-VL-8B-Instruct")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--sdpa-native-gqa", action="store_true",
                   help="let torch SDPA do GQA (enable_gqa=True); broken on PPU for decode, off by default")
    p.add_argument("--default-max-tokens", type=int, default=1024)
    p.add_argument("--max-tokens-cap", type=int, default=8192)
    p.add_argument("--max-prompt-tokens", type=int, default=65536)
    p.add_argument("--default-temperature", type=float, default=0.0,
                   help="used when the request has no temperature; 0 = greedy")
    p.add_argument("--min-pixels", type=int, default=0, help="override image processor min_pixels")
    p.add_argument("--max-pixels", type=int, default=0, help="override image processor max_pixels")
    p.add_argument("--log-outputs", action="store_true", help="log raw model output text per request")
    p.add_argument("--warmup", action="store_true", help="run one tiny generation before serving")
    ARGS = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    import transformers

    LOG.info("transformers %s from %s", transformers.__version__, os.path.dirname(transformers.__file__))
    ENGINE = Engine(ARGS)
    if ARGS.warmup:
        t = time.time()
        with ENGINE.lock:
            ENGINE.chat({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4})
        LOG.info("warmup done in %.1fs", time.time() - t)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    srv.daemon_threads = True
    LOG.info("serving %s on http://%s:%d/v1/chat/completions", ARGS.served_name, ARGS.host, ARGS.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
