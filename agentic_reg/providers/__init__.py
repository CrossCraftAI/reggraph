"""Provider factory: turn settings into a concrete LLM backend."""

from ..config import Settings
from .base import LLMProvider, Provider


def get_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower()

    if provider == "github":
        from .github_provider import GitHubModelsProvider

        return GitHubModelsProvider(
            settings.github_model, settings.github_token, settings.github_base_url
        )

    if provider == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(settings.ollama_model, settings.ollama_host)

    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings.anthropic_model, settings.anthropic_api_key)

    raise ValueError(
        f"Unknown LLM provider {settings.llm_provider!r}. Use 'github', 'ollama', or 'anthropic'."
    )


__all__ = ["LLMProvider", "Provider", "get_provider"]
