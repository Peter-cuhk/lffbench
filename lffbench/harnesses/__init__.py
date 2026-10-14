"""Harness adapters (docs/15): one way to plug any agent harness into a robot_server session.

  ours              reference driver scripts/api_player.py (main table)
  claude_code       Claude Code CLI, --bare, only our MCP tools (arm B)
  claude_code_full  Claude Code CLI without --bare: full system prompt, still one-time dirs and only MCP (arm B')
  codex             Codex CLI exec, ephemeral, empty CODEX_HOME, only our MCP tools reachable (arm F)

Use scripts/harness_run.py (one session) or scripts/api_batch.py start --harness ... (batches).
"""
import json
import os
import sys

from .base import EpisodeSpec, HarnessAdapter, HarnessRefused, first_message_text  # noqa: F401

NAMES = ("ours", "claude_code", "claude_code_full", "codex")
PROVIDER = dict(claude_code="anthropic", claude_code_full="anthropic", codex="openai")


def get_adapter(name, **kw):
    if name == "ours":
        from .reference import Reference
        return Reference(**kw)
    if name == "claude_code":
        from .claude_code import ClaudeCode
        return ClaudeCode(**kw)
    if name == "claude_code_full":
        from .claude_code import ClaudeCodeFull
        return ClaudeCodeFull(**kw)
    if name == "codex":
        from .codex import Codex
        return Codex(**kw)
    raise ValueError(f"unknown harness {name!r}; one of {NAMES}")


def render_first_message(ep):
    """Call `status` once over HTTP (exactly as the reference driver does before its first turn) and build the first
    user message for a native harness. Returns (text, status_json)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "scripts"))
    from api_player import call_server  # noqa: E402
    st = json.loads(call_server(ep.port, ep.token, "status", {}))
    readme = open(os.path.join(ep.sandbox, "README.md")).read()
    status_json = json.dumps(st)
    return first_message_text(readme, status_json), status_json
