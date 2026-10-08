"""Request construction for the OpenAI Responses backend (and the chat backend). No simulator, no network:
the real openai SDK is exercised through an in-process mock HTTP transport."""
import base64
import io
import json

import numpy as np
import pytest

from lffbench.agent.backends.base import import_openai
from lffbench.agent.backends.openai_responses import OpenAIResponsesBackend
from lffbench.agent.backends.chat_completions import ChatCompletionsBackend
from lffbench.agent.context import Conversation, ToolCall
from lffbench.agent.images import ImageRef
from lffbench.agent.cost import call_cost

TOOLS = [
    dict(name="move_to", description="move", parameters={"type": "object", "properties": {
        "x": {"type": "number"}, "speed": {"type": ["number", "null"]}}, "required": ["x", "speed"],
        "additionalProperties": False}),
    dict(name="done", description="end", parameters={"type": "object", "properties": {"reason": {"type": "string"}},
                                                      "required": ["reason"], "additionalProperties": False}),
]


def img(cam, val=100, res=512):
    return ImageRef(camera=cam, array=np.full((res, res, 3), val, np.uint8))


def make_conv():
    conv = Conversation("SYSTEM PROMPT")
    conv.add_user([dict(type="text", text="TASK: push"), dict(type="text", text="[end of attempt 1]"),
                   dict(type="image", image=img("agentview", 10))], live=False)
    conv.add_user([dict(type="text", text="attempt 2"), dict(type="image", image=img("agentview", 20)),
                   dict(type="image", image=img("wrist", 30)), dict(type="text", text="Proprioception: {}")], live=True)
    conv.add_assistant("I will push slower.", [ToolCall(id="call_1", name="move_to", arguments={"x": 0.1, "speed": None},
                                                        raw_arguments='{"x": 0.1, "speed": null}')], backend="other")
    conv.add_tool_output("call_1", "move_to", '{"ok": true}')
    conv.add_user([dict(type="text", text="obs"), dict(type="image", image=img("agentview", 40))], live=True)
    return conv


def test_build_request_structure():
    b = OpenAIResponsesBackend(reasoning_effort="high", client=object())
    req = b.build_request(make_conv(), TOOLS)
    assert req["model"] == "gpt-6-astra"
    assert req["instructions"] == "SYSTEM PROMPT"
    assert req["reasoning"] == {"effort": "high"}
    assert req["store"] is False and req["include"] == ["reasoning.encrypted_content"]
    assert req["parallel_tool_calls"] is False and req["tool_choice"] == "auto"
    assert [t["name"] for t in req["tools"]] == ["move_to", "done"]
    assert all(t["type"] == "function" and t["strict"] is True for t in req["tools"])
    kinds = [it.get("type") or it.get("role") for it in req["input"]]
    assert kinds == ["user", "user", "assistant", "function_call", "function_call_output", "user"]
    fc, fco = req["input"][3], req["input"][4]
    assert fc["call_id"] == fco["call_id"] == "call_1" and json.loads(fc["arguments"]) == {"x": 0.1, "speed": None}
    images = [c for it in req["input"] if isinstance(it.get("content"), list) for c in it["content"]
              if c["type"] == "input_image"]
    assert len(images) == 4
    for c in images:
        assert c["image_url"].startswith("data:image/jpeg;base64,") and c["detail"] == "auto"
        from PIL import Image
        im = Image.open(io.BytesIO(base64.b64decode(c["image_url"].split(",", 1)[1])))
        assert im.format == "JPEG" and im.size == (512, 512)
    # explicit cache breakpoints: end of the (stable) history item and end of the newest input
    bps = [(i, j) for i, it in enumerate(req["input"]) if isinstance(it.get("content"), list)
           for j, c in enumerate(it["content"]) if "prompt_cache_breakpoint" in c]
    assert bps == [(0, 2), (5, 1)]
    assert OpenAIResponsesBackend.request_stats(req)["n_images"] == 4


def test_effort_validation():
    with pytest.raises(ValueError):
        OpenAIResponsesBackend(reasoning_effort="none")
    for e in ("low", "medium", "high", "xhigh", "max"):
        OpenAIResponsesBackend(reasoning_effort=e)


def test_live_image_pruning_keeps_history():
    conv = Conversation("S")
    conv.add_user([dict(type="image", image=img("agentview"))], live=False)
    for i in range(10):
        conv.add_user([dict(type="image", image=img("agentview", i)), dict(type="image", image=img("wrist", i))])
    b = OpenAIResponsesBackend(max_live_images=8, client=object(), cache_breakpoints="none")
    req = b.build_request(conv, TOOLS)
    n = [sum(c["type"] == "input_image" for c in it["content"]) for it in req["input"]]
    assert n[0] == 1  # history image never pruned
    assert sum(n[1:]) == 4  # hysteresis: pruned to half the budget
    assert "omitted" in req["input"][1]["content"][0]["text"]


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAIResponsesBackend().client()


def _canned(call_id="call_A"):
    return {
        "id": "resp_1", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-6-astra",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}],
             "encrypted_content": "gAAAAenc"},
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "Last time the cube overshot.", "annotations": []}]},
            {"type": "function_call", "id": "fc_1", "call_id": call_id, "name": "move_to",
             "arguments": '{"x": 0.25, "speed": 0.2}', "status": "completed"},
        ],
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [], "temperature": 1.0, "top_p": 1.0,
        "usage": {"input_tokens": 5000, "input_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 500},
                  "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 200}, "total_tokens": 5300},
    }


def test_sdk_roundtrip_with_mock_transport(monkeypatch):
    """Real openai SDK + mock HTTP transport: checks the wire request and response parsing, no network."""
    import_openai()
    import httpx2
    seen = []

    def handler(request):
        seen.append(request)
        return httpx2.Response(200, json=_canned(f"call_{len(seen)}"))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://mock.invalid/v1")
    b = OpenAIResponsesBackend(reasoning_effort="medium", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    conv = make_conv()
    r = b.act(conv, TOOLS)
    assert len(seen) == 1
    req = seen[0]
    assert req.method == "POST" and str(req.url) == "http://mock.invalid/v1/responses"
    assert req.headers["authorization"] == "Bearer sk-test-not-real"
    body = json.loads(req.content)
    assert body["model"] == "gpt-6-astra" and body["reasoning"]["effort"] == "medium"
    assert body["input"][1]["content"][1]["type"] == "input_image"
    # parsed response
    assert [(t.name, t.arguments, t.id) for t in r.tool_calls] == [("move_to", {"x": 0.25, "speed": 0.2}, "call_1")]
    assert r.text == "Last time the cube overshot." and r.reasoning_summary == "think"
    assert r.usage == dict(input_tokens=5000, cached_tokens=4000, cache_write_tokens=500, output_tokens=300,
                           reasoning_tokens=200)
    assert abs(call_cost(r.usage) - (500 * 10 + 500 * 12.5 + 4000 * 1 + 300 * 50) / 1e6) < 1e-12
    # second call replays the reasoning item (encrypted) + function call, followed by its output
    conv.add_assistant(r.text, r.tool_calls, raw=r.raw, backend=b.name)
    conv.add_tool_output(r.tool_calls[0].id, "move_to", '{"ok": true}')
    b.act(conv, TOOLS)
    body2 = json.loads(seen[1].content)
    types = [it.get("type") for it in body2["input"]]
    i = types.index("reasoning")
    assert body2["input"][i]["encrypted_content"] == "gAAAAenc"
    assert types[i + 1:i + 4] == ["message", "function_call", "function_call_output"]
    assert body2["input"][i + 2]["call_id"] == body2["input"][i + 3]["call_id"] == "call_1"


def test_chat_backend_payload():
    b = ChatCompletionsBackend(client=object(), max_live_images=2)
    req = b.build_request(make_conv(), TOOLS)
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "user", "assistant", "tool", "user"]
    assert req["messages"][3]["tool_calls"][0]["function"]["name"] == "move_to"
    assert req["messages"][4]["tool_call_id"] == "call_1"
    assert req["tools"][0]["type"] == "function" and req["tools"][0]["function"]["name"] == "move_to"
    n_img = sum(c["type"] == "image_url" for m in req["messages"] if isinstance(m["content"], list) for c in m["content"])
    assert n_img == 2  # 1 history + newest live (budget 2 -> hysteresis keeps 1 live)
