"""Claude (Anthropic API) provider.

Not used by default. Enable it later with::

    uv sync --extra anthropic
    # in .env: AGENTIC_REG_LLM_PROVIDER=anthropic  and  ANTHROPIC_API_KEY=sk-ant-...

Local models are great for proving the pipeline; Claude is a large quality jump
for the hard steps (extraction and multi-step legal reasoning).
"""

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None) -> None:
        import anthropic  # part of the optional `anthropic` extra

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env to use the Claude provider."
            )
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        # Adaptive thinking lets Claude decide how much to reason; it manages its
        # own sampling, so `temperature` is intentionally not forwarded here.
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if block.type == "text").strip()
