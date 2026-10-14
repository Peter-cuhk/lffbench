"""Model backends. Import lazily: each API backend needs its provider's SDK (see base.import_sdk)."""
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
    if kind == "anthropic":
        from .anthropic_messages import AnthropicMessagesBackend
        return AnthropicMessagesBackend(**kw)
    if kind == "gemini":
        from .gemini import GeminiBackend
        return GeminiBackend(**kw)
    raise ValueError(kind)
