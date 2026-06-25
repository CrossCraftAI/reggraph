"""Local LLM provider backed by Ollama (https://ollama.com).

Free, runs on your machine, no API key. On Apple Silicon Ollama uses the GPU
(Metal) automatically. This is the default backend for development.
"""

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
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
            system=system,
            options={"temperature": temperature},
        )
        return response["response"].strip()
