from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentic_reg.config import Settings
from agentic_reg.providers import LLMProvider, get_provider
from agentic_reg.providers.github_provider import GitHubModelsProvider
from agentic_reg.providers.ollama_provider import OllamaProvider


class _DummyProvider(LLMProvider):
    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        return f"system={system}; prompt={prompt}; temp={temperature}"


def test_get_provider_returns_github_provider():
    settings = Settings(
        _env_file=None,
        llm_provider="github",
        github_token="fake-token",
    )
    provider = get_provider(settings)
    assert isinstance(provider, GitHubModelsProvider)


def test_get_provider_returns_ollama_provider():
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        ollama_model="llama3.1:8b",
    )
    provider = get_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_get_provider_raises_on_unknown_provider():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="unknown")


def test_get_provider_raises_when_github_token_missing():
    settings = Settings(
        _env_file=None,
        llm_provider="github",
        github_token=None,
    )
    with patch("agentic_reg.providers.github_provider.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError
        with pytest.raises(ValueError, match="No GitHub token"):
            get_provider(settings)


def test_chat_bridge_flattens_old_message_shape():
    response = _DummyProvider().chat(
        [
            {"role": "system", "content": "Be careful."},
            {"role": "user", "content": "Question?"},
        ],
        tools=[{"type": "function", "function": {"name": "search"}}],
    )

    assert "system=Be careful." in response
    assert "user: Question?" in response
    assert "Available tools: search" in response
