"""LLM provider abstraction.

A single OpenAI-compatible provider covers GitHub Models, Ollama, and any other
endpoint that speaks the OpenAI chat/completions protocol. The factory
``get_provider`` reads ``Settings`` and returns the correct client.
"""

import json
import subprocess
from collections.abc import Iterator
from typing import Any, Protocol

import httpx

from agentic_reg.config import Settings

# ── message / tool shapes (OpenAI-compat) ──────────────────────────────────

Message = dict[str, Any]  # {"role": "...", "content": "..."}
ToolDef = dict[str, Any]  # OpenAI function-calling tool schema


class Provider(Protocol):
    """LLM backend interface.

    Every backend speaks this protocol. The agent loop calls ``chat``
    (or ``chat_stream`` for streaming) and passes the same message/tool
    shapes regardless of which provider is active.
    """

    def chat(self, messages: list[Message], tools: list[ToolDef] | None = None) -> str:
        """Send messages and return the model's text response."""

    def chat_stream(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Iterator[str]:
        """Streaming variant of ``chat`` — yields text deltas."""


# ── OpenAI-compatible provider ─────────────────────────────────────────────


class OpenAICompatProvider:
    """Provider for any OpenAI-compatible chat/completions endpoint.

    Covers GitHub Models (``https://models.github.ai/inference``) and
    Ollama (``http://localhost:11434/v1``) out of the box.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    # -- public interface ---------------------------------------------------

    def chat(self, messages: list[Message], tools: list[ToolDef] | None = None) -> str:
        payload = self._build_payload(messages, tools, stream=False)
        response = self._post(payload)
        return self._extract_content(response)

    def chat_stream(
        self, messages: list[Message], tools: list[ToolDef] | None = None
    ) -> Iterator[str]:
        payload = self._build_payload(messages, tools, stream=True)
        with self._client.stream("POST", self._endpoint, json=payload) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    # -- internal -----------------------------------------------------------

    @property
    def _endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = self._client.post(self._endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content: str = msg.get("content", "") or ""
        return content


# ── auth helpers ────────────────────────────────────────────────────────────


def _resolve_github_token(settings: Settings) -> str | None:
    """Return a GitHub token from settings or the ``gh`` CLI."""
    if settings.github_token:
        return settings.github_token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ── factory ─────────────────────────────────────────────────────────────────


def get_provider(settings: Settings) -> Provider:
    """Return the configured LLM provider."""
    provider_name = settings.llm_provider

    if provider_name == "github":
        token = _resolve_github_token(settings)
        if not token:
            raise RuntimeError(
                "No GitHub token found. Set GITHUB_TOKEN env var or run 'gh auth login'."
            )
        return OpenAICompatProvider(
            base_url=settings.github_base_url,
            model=settings.github_model,
            api_key=token,
        )

    if provider_name == "ollama":
        # Ollama's OpenAI-compat endpoint: http://host:port/v1
        base = settings.ollama_host.rstrip("/")
        return OpenAICompatProvider(
            base_url=f"{base}/v1",
            model=settings.ollama_model,
            api_key=None,
        )

    if provider_name == "anthropic":
        raise NotImplementedError("Anthropic provider is deferred. Use github or ollama.")

    raise ValueError(f"Unknown llm_provider: {provider_name!r}")
