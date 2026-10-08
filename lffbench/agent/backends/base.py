"""Backend interface: a backend maps a conversation (+ tool specs) to the next tool call(s)."""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# Normally `openai` is installed in the venv (setup/install.sh). Optionally it can live in a separate
# `pip install --target` directory, given by LFF_OPENAI_SDK_DIR, that is APPENDED to sys.path so every
# package already in the venv keeps precedence.
OPENAI_SDK_DIR = os.environ.get("LFF_OPENAI_SDK_DIR", "")


def import_openai():
    try:
        import openai  # noqa: F401
    except ImportError:
        if OPENAI_SDK_DIR and os.path.isdir(OPENAI_SDK_DIR) and OPENAI_SDK_DIR not in sys.path:
            sys.path.append(OPENAI_SDK_DIR)
        import openai  # noqa: F401
    return sys.modules["openai"]


@dataclass
class BackendResponse:
    tool_calls: list = field(default_factory=list)  # [ToolCall]
    text: Optional[str] = None  # visible assistant text (e.g. the diagnosis)
    reasoning_summary: Optional[str] = None
    usage: dict = field(default_factory=dict)  # input_tokens, cached_tokens, output_tokens, reasoning_tokens
    latency_s: float = 0.0
    raw: Optional[list] = None  # backend-specific items to replay in later requests (e.g. reasoning items)
    request_stats: dict = field(default_factory=dict)  # n_images, n_items, bytes, ...
    status: str = "ok"
    error: Optional[str] = None


@dataclass
class RunInfo:
    task: object  # privileged: only the scripted backend may look at task / inst
    inst: dict
    seed: int
    k: int
    protocol: str
    feedback: str
    memory: str


@dataclass
class AttemptInfo:
    attempt: int
    history: list  # AttemptRecords visible to the agent under the memory mode (already selected)
    continuing: bool = False  # within protocol: continuing after a failed done


class Backend:
    name = "base"

    def begin_run(self, info: RunInfo):
        pass

    def begin_attempt(self, info: AttemptInfo):
        pass

    def act(self, conv, tools) -> BackendResponse:
        raise NotImplementedError

    def on_tool_result(self, call, result):
        pass

    def end_attempt(self) -> dict:
        return {}

    def end_run(self):
        pass

    def config(self) -> dict:
        return dict(name=self.name)
