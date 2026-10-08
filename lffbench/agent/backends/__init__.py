"""Model backends. Import lazily: the OpenAI / chat backends need the `openai` package (see base.import_openai)."""
from .base import AttemptInfo, Backend, BackendResponse, RunInfo  # noqa: F401


def make_backend(kind, **kw):
    if kind == "scripted":
        from .scripted import ScriptedBackend
        return ScriptedBackend(**kw)
    if kind == "openai":
        from .openai_responses import OpenAIResponsesBackend
        return OpenAIResponsesBackend(**kw)
    if kind == "chat":
        from .chat_completions import ChatCompletionsBackend
        return ChatCompletionsBackend(**kw)
    raise ValueError(kind)
