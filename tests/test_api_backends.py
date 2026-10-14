"""Request construction and response parsing of the Anthropic and Gemini backends. No simulator, no network: the real
SDKs are exercised through in-process mock HTTP transports. Checks the points that break agent loops silently:
reasoning state (thinking blocks + signatures / thought signatures) replayed unchanged on the next call, tool results
placed first in the following user turn, images sent as base64, usage mapped to the shared meaning."""
import base64
import json

import numpy as np
import pytest

from lffbench.agent.backends.base import import_sdk
from lffbench.agent.context import Conversation
from lffbench.agent.images import ImageRef

TOOLS = [dict(name="robot", description="run a robot command",
              parameters={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"],
                          "additionalProperties": False})]


def img(cam, val=100, res=64):
    return ImageRef(camera=cam, array=np.full((res, res, 3), val, np.uint8))


def first_turn():
    conv = Conversation("SYSTEM PROMPT")
    conv.add_user([dict(type="text", text="Begin. status: {}"), dict(type="image", image=img("agentview", 10))])
    return conv


# ---------------------------------------------------------------------------------------------------- Anthropic
def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def anthropic_stream(blocks, stop_reason="tool_use", stop_details=None):
    """SSE body of a streamed Messages API reply with the given content blocks."""
    ev = [dict(type="message_start", message=dict(id="msg_1", type="message", role="assistant", model="claude-opus-5-5",
                                                   content=[], stop_reason=None, stop_sequence=None,
                                                   usage=dict(input_tokens=10, cache_read_input_tokens=100,
                                                              cache_creation_input_tokens=5, output_tokens=1)))]
    for i, b in enumerate(blocks):
        if b["type"] == "thinking":
            ev.append(dict(type="content_block_start", index=i, content_block=dict(type="thinking", thinking="", signature="")))
            ev.append(dict(type="content_block_delta", index=i, delta=dict(type="thinking_delta", thinking=b["thinking"])))
            ev.append(dict(type="content_block_delta", index=i, delta=dict(type="signature_delta", signature=b["signature"])))
        elif b["type"] == "text":
            ev.append(dict(type="content_block_start", index=i, content_block=dict(type="text", text="")))
            ev.append(dict(type="content_block_delta", index=i, delta=dict(type="text_delta", text=b["text"])))
        elif b["type"] == "tool_use":
            ev.append(dict(type="content_block_start", index=i,
                           content_block=dict(type="tool_use", id=b["id"], name=b["name"], input={})))
            ev.append(dict(type="content_block_delta", index=i,
                           delta=dict(type="input_json_delta", partial_json=json.dumps(b["input"]))))
        ev.append(dict(type="content_block_stop", index=i))
    md = dict(type="message_delta", delta=dict(stop_reason=stop_reason, stop_sequence=None), usage=dict(output_tokens=42))
    if stop_details:
        md["delta"]["stop_details"] = stop_details
    ev += [md, dict(type="message_stop")]
    return sse(ev)


def anthropic_backend(replies, seen):
    try:
        anthropic = import_sdk("anthropic")
        httpx2 = import_sdk("httpx2")
    except ImportError:
        pytest.skip("anthropic SDK not installed")
    from lffbench.agent.backends.anthropic_messages import AnthropicMessagesBackend

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=replies.pop(0))

    client = anthropic.Anthropic(api_key="test", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    return AnthropicMessagesBackend(model="claude-opus-5-5", effort="high", client=client)


def test_anthropic_request_and_replay():
    seen = []
    r1 = anthropic_stream([dict(type="thinking", thinking="Look at the table first.", signature="SIG123"),
                           dict(type="tool_use", id="toolu_1", name="robot", input={"command": "status"})])
    r2 = anthropic_stream([dict(type="text", text="Done.")], stop_reason="end_turn")
    b = anthropic_backend([r1, r2], seen)
    conv = first_turn()
    out = b.act(conv, TOOLS)
    req = seen[0]
    assert req["model"] == "claude-opus-5-5" and req["stream"] is True and req["system"] == "SYSTEM PROMPT"
    assert req["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert req["output_config"] == {"effort": "high"}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert req["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert req["tools"][0]["strict"] is True and req["tools"][0]["input_schema"]["required"] == ["command"]
    u0 = req["messages"][0]
    assert u0["role"] == "user" and [c["type"] for c in u0["content"]] == ["text", "image"]
    src = u0["content"][1]["source"]
    assert src["type"] == "base64" and src["media_type"] == "image/jpeg" and base64.b64decode(src["data"])[:2] == b"\xff\xd8"
    assert [(c.name, c.arguments) for c in out.tool_calls] == [("robot", {"command": "status"})]
    assert out.reasoning_summary == "Look at the table first." and out.status == "completed"
    assert out.usage == dict(input_tokens=115, cached_tokens=100, cache_write_tokens=5, output_tokens=42, reasoning_tokens=0)

    conv.add_assistant(out.text, out.tool_calls, raw=out.raw, backend=b.name)
    conv.add_tool_output("toolu_1", "robot", '{"ok": true}')
    conv.add_user([dict(type="text", text="Requested images:"), dict(type="image", image=img("wrist", 30))])
    out2 = b.act(conv, TOOLS)
    msgs = seen[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    asst = msgs[1]["content"]
    assert asst[0]["type"] == "thinking" and asst[0]["signature"] == "SIG123" and asst[0]["thinking"] == "Look at the table first."
    assert asst[1] == {"type": "tool_use", "id": "toolu_1", "name": "robot", "input": {"command": "status"}} or \
        {k: asst[1][k] for k in ("type", "id", "name", "input")} == {"type": "tool_use", "id": "toolu_1", "name": "robot",
                                                                     "input": {"command": "status"}}
    # tool_result first, then the attached text + image, all in one user message
    assert [c["type"] for c in msgs[2]["content"]] == ["tool_result", "text", "image"]
    assert msgs[2]["content"][0] == {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"ok": true}'}
    # the first request's messages are an exact prefix of the second one (append-only history)
    assert seen[1]["messages"][0] == seen[0]["messages"][0]
    assert out2.text == "Done." and not out2.tool_calls


def test_anthropic_refusal_is_reported():
    seen = []
    r = anthropic_stream([], stop_reason="refusal", stop_details=dict(type="refusal", category="cyber", explanation="x"))
    b = anthropic_backend([r], seen)
    out = b.act(first_turn(), TOOLS)
    assert out.status == "refusal" and "cyber" in out.error


# ---------------------------------------------------------------------------------------------------- Gemini
def gemini_backend(replies, seen):
    try:
        genai = import_sdk("google.genai")
        types = import_sdk("google.genai.types")
        httpx = import_sdk("httpx")
    except ImportError:
        pytest.skip("google-genai SDK not installed")
    from lffbench.agent.backends.gemini import GeminiBackend

    def handler(request):
        seen.append(dict(url=str(request.url), body=json.loads(request.content)))
        return httpx.Response(200, json=replies.pop(0))

    client = genai.Client(api_key="test", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
    return GeminiBackend(model="gemini-test", thinking_level="high", client=client)


def test_gemini_request_and_thought_signature_replay():
    seen = []
    sig = base64.b64encode(b"sig-bytes").decode()
    r1 = dict(candidates=[dict(content=dict(role="model", parts=[
        dict(text="Plan: check status.", thought=True),
        dict(functionCall=dict(id="fc1", name="robot", args=dict(command="status")), thoughtSignature=sig)]),
        finishReason="STOP")],
        usageMetadata=dict(promptTokenCount=120, cachedContentTokenCount=100, candidatesTokenCount=7,
                           thoughtsTokenCount=30, totalTokenCount=157))
    r2 = dict(candidates=[dict(content=dict(role="model", parts=[dict(text="Done.")]), finishReason="STOP")],
              usageMetadata=dict(promptTokenCount=200, candidatesTokenCount=3, totalTokenCount=203))
    b = gemini_backend([r1, r2], seen)
    conv = first_turn()
    out = b.act(conv, TOOLS)
    body = seen[0]["body"]
    assert "models/gemini-test:generateContent" in seen[0]["url"]
    assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM PROMPT"
    fd = body["tools"][0]["functionDeclarations"][0]
    # google-genai sends some nested fields in snake_case (the API's JSON parser accepts both spellings)
    schema = fd.get("parametersJsonSchema", fd.get("parameters_json_schema"))
    assert fd["name"] == "robot" and schema["additionalProperties"] is False
    tc = body["generationConfig"]["thinkingConfig"]
    assert tc.get("includeThoughts", tc.get("include_thoughts")) is True
    assert tc.get("thinkingLevel", tc.get("thinking_level")) == "HIGH"
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    p = body["contents"][0]["parts"]
    blob = p[1]["inlineData"]
    assert body["contents"][0]["role"] == "user" and blob.get("mimeType", blob.get("mime_type")) == "image/jpeg"
    assert base64.urlsafe_b64decode(blob["data"])[:2] == b"\xff\xd8"
    assert [(c.id, c.name, c.arguments) for c in out.tool_calls] == [("fc1", "robot", {"command": "status"})]
    assert out.reasoning_summary == "Plan: check status." and out.text is None
    assert out.usage == dict(input_tokens=120, cached_tokens=100, output_tokens=37, reasoning_tokens=30)

    conv.add_assistant(out.text, out.tool_calls, raw=out.raw, backend=b.name)
    conv.add_tool_output("fc1", "robot", '{"ok": true}')
    conv.add_user([dict(type="text", text="Requested images:"), dict(type="image", image=img("wrist", 30))])
    out2 = b.act(conv, TOOLS)
    c = seen[1]["body"]["contents"]
    assert [x["role"] for x in c] == ["user", "model", "user"]
    model_parts = c[1]["parts"]
    assert model_parts[0] == {"text": "Plan: check status.", "thought": True}
    sent = model_parts[1].get("thoughtSignature", model_parts[1].get("thought_signature"))
    assert base64.urlsafe_b64decode(sent) == b"sig-bytes" and model_parts[1]["functionCall"]["id"] == "fc1"
    fr = c[2]["parts"][0]["functionResponse"]
    assert fr == {"id": "fc1", "name": "robot", "response": {"output": '{"ok": true}'}}
    assert [list(x)[0] for x in c[2]["parts"][1:]] == ["text", "inlineData"]
    assert out2.text == "Done."


def test_gemini_empty_reply_is_incomplete():
    seen = []
    b = gemini_backend([dict(candidates=[dict(finishReason="MALFORMED_FUNCTION_CALL")])], seen)
    out = b.act(first_turn(), TOOLS)
    assert out.status == "incomplete" and not out.tool_calls
