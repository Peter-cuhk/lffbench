"""Fast offline tests of the harness adapter layer (no harness binaries, no robot server, no model).
The end-to-end checks (real Claude Code / Codex against fake APIs and a real robot_server) are in
tests/harness_selfcheck/selfcheck.py."""
import json
import os
import urllib.request

import pytest

from lffbench.harnesses import base as HB
from lffbench.harnesses import contract as C
from lffbench.harnesses import get_adapter
from lffbench.harnesses.fakeapi import FakeAnthropic, FakeResponses


@pytest.mark.parametrize("name", ["ours", "claude_code", "claude_code_full", "codex"])
def test_capabilities_complete(name):
    ad = get_adapter(name)
    HB.check_capabilities(ad.name, ad.capabilities)
    assert set(ad.capabilities) == set(HB.REQUIRED) | set(HB.FORBIDDEN)


def test_incomplete_declaration_rejected():
    with pytest.raises(HB.CapabilityError):
        class Bad(HB.HarnessAdapter):  # noqa: F841
            name = "bad"
            capabilities = dict(only_benchmark_tools=HB.Cap("yes", "launch"))


def _fake_log(tmp_path, readme="# Robot task\nat most 40 robot commands\n"):
    sb = tmp_path / "sb"
    log = tmp_path / "log"
    sb.mkdir()
    log.mkdir()
    (sb / "README.md").write_text(readme)
    (log / "session.json").write_text(json.dumps(dict(run="x", sandbox=str(sb), port=1)))
    (log / "agent_iface.json").write_text(json.dumps(dict(token="t", port=1, workspace=str(sb), agent_iface="tools",
                                                          max_tool_calls=100, perception="rgb")))
    (log / "events.jsonl").write_text(json.dumps(dict(ev="session_start", task="l1_height_bias", seed=3000, k=3,
                                                      k_shown=1, memory="full", feedback="F1", instruction="direct",
                                                      perception="rgb", protocol="cross", arm="none", dossier="")) + "\n")
    return str(log)


def test_contract_hash_deterministic_and_sensitive(tmp_path):
    log = _fake_log(tmp_path)
    c1, c2 = C.build_contract(log), C.build_contract(log)
    assert C.contract_sha256(c1) == C.contract_sha256(c2)
    assert c1["hide_k"] is True and c1["budget"]["robot_commands"] == 40
    open(os.path.join(json.load(open(os.path.join(log, "session.json")))["sandbox"], "README.md"), "a").write("x")
    assert C.contract_sha256(C.build_contract(log)) != C.contract_sha256(c1)


def test_fake_anthropic_scripted_turns():
    srv = FakeAnthropic([[("tool", "mcp__lff__robot", {"command": "status"})]]).start()
    try:
        def ask(msgs):
            req = urllib.request.Request(srv.base_url + "/v1/messages", data=json.dumps(
                dict(model="m", messages=msgs)).encode(), headers={"content-type": "application/json"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(req, timeout=10).read().decode()
        r1 = ask([dict(role="user", content="hi")])
        assert "mcp__lff__robot" in r1 and '"stop_reason": "tool_use"' in r1
        r2 = ask([dict(role="user", content="hi"), dict(role="assistant", content="x"), dict(role="user", content="y")])
        assert "end_turn" in r2
        assert len(srv.message_requests()) == 2
    finally:
        srv.stop()


def test_fake_responses_exec_turn():
    srv = FakeResponses([[("exec", "text(1)")]]).start()
    try:
        req = urllib.request.Request(srv.base_url + "/responses", data=json.dumps(dict(model="m", input=[])).encode(),
                                     headers={"content-type": "application/json"})
        out = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=10).read().decode()
        assert "custom_tool_call" in out and "text(1)" in out
    finally:
        srv.stop()


def test_claude_prompt_normalisation():
    from lffbench.harnesses.claude_code import _norm
    spec = HB.LaunchSpec(argv=[], env={}, cwd="/tmp/lffrun/0123456789ab",
                         state_dirs=dict(state="/tmp/lffstate/0123456789ab"))
    s = ("x-anthropic-billing-header: cc_version=2.1.267.5d4; CWD: /tmp/lffrun/0123456789ab Date: 2026-10-13 "
         "/tmp/lffstate/aaaaaaaaaaaa/config/projects/-tmp-lffrun-0123456789ab/memory/")
    n = _norm(s, spec)
    assert "5d4" not in n and "2026-10-13" not in n and "0123456789ab" not in n and "aaaaaaaaaaaa" not in n
