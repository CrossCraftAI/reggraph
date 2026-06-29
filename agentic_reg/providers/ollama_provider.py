"""Local LLM provider backed by Ollama (https://ollama.com).

Free, runs on your machine, no API key. On Apple Silicon Ollama uses the GPU
(Metal) automatically. This is the default backend for development.
"""

from typing import Any

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, host: str) -> None:
        # Imported lazily so the package imports cleanly even if a user only
        # ever uses the Anthropic backend.
        import ollama

        self._model = model
        self._client = ollama.Client(host=host)

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "options": {"temperature": temperature},
        }
        if system is not None:
            kwargs["system"] = system
        response = self._client.generate(**kwargs)
        return response["response"].strip()
