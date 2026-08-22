"""LLM provider factory."""

from __future__ import annotations

from ..config import settings
from .base import Call, LLMError, LLMRefusal, LLMTruncated, Provider, Usage

_CACHE: dict[tuple[str, str, str], Provider] = {}


def get_provider(
    provider: str | None = None, model: str | None = None, effort: str | None = None
) -> Provider:
    name = (provider or settings.provider).lower()
    model_id = model or settings.model
    eff = effort or settings.effort
    key = (name, model_id, eff)
    if key in _CACHE:
        return _CACHE[key]

    if name == "local":
        from .local_provider import LocalProvider

        instance: Provider = LocalProvider(model_id, eff)
    elif name == "ollama":
        from .ollama_provider import OllamaProvider

        instance = OllamaProvider(model_id, eff)
    elif name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        instance = AnthropicProvider(model_id, eff, settings.anthropic_api_key)
    elif name == "openai":
        from .openai_provider import OpenAIProvider

        instance = OpenAIProvider(model_id, eff, settings.openai_api_key)
    elif name == "gemini":
        from .gemini_provider import GeminiProvider

        instance = GeminiProvider(model_id, eff, settings.google_api_key)
    elif name == "mock":
        from .mock_provider import MockProvider

        instance = MockProvider(model_id or "mock-1", eff)
    else:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{name}'. Use one of: "
            "local, ollama, anthropic, openai, gemini, mock."
        )

    _CACHE[key] = instance
    return instance


__all__ = [
    "Call",
    "LLMError",
    "LLMRefusal",
    "LLMTruncated",
    "Provider",
    "Usage",
    "get_provider",
]
