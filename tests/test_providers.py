from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentic_reg.config import Settings
from agentic_reg.providers import OpenAICompatProvider, get_provider


def test_get_provider_returns_github_provider():
    settings = Settings(
        llm_provider="github",
        github_token="fake-token",
    )
    provider = get_provider(settings)
    assert isinstance(provider, OpenAICompatProvider)


def test_get_provider_returns_ollama_provider():
    settings = Settings(
        llm_provider="ollama",
        ollama_model="llama3.1:8b",
    )
    provider = get_provider(settings)
    assert isinstance(provider, OpenAICompatProvider)


def test_get_provider_raises_on_unknown_provider():
    with pytest.raises(ValidationError):
        Settings(llm_provider="unknown")


def test_get_provider_raises_when_github_token_missing():
    settings = Settings(
        llm_provider="github",
        github_token=None,
    )
    try:
        with patch("agentic_reg.providers.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            get_provider(settings)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        assert "No GitHub token" in str(exc)


def test_openai_compat_provider_builds_correct_payload():
    provider = OpenAICompatProvider(
        base_url="https://models.github.ai/inference",
        model="openai/gpt-4o",
        api_key="test-key",
    )

    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    payload = provider._build_payload(messages, tools, stream=False)
    assert payload["model"] == "openai/gpt-4o"
    assert payload["messages"] == messages
    assert payload["tools"] == tools
    assert payload["stream"] is False


def test_openai_compat_provider_extracts_content():
    response = {
        "choices": [{"message": {"content": "Hello, world!"}}],
    }
    content = OpenAICompatProvider._extract_content(response)
    assert content == "Hello, world!"


def test_openai_compat_provider_extracts_empty_content_when_missing():
    response = {"choices": [{"message": {}}]}
    content = OpenAICompatProvider._extract_content(response)
    assert content == ""
