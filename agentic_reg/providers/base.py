"""The LLM provider interface.

Every part of the system that needs a language model talks to *this* interface,
never to a concrete backend. That is what makes the backend swappable: adding
a new provider means adding one class, not touching call sites.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class LLMProvider(ABC):
    """Minimal text-completion interface used across agentic-reg."""

    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        """Return the model's text response to ``prompt``.

        Args:
            prompt: The user-facing instruction / question.
            system: Optional system prompt that sets the model's role.
            temperature: Sampling temperature (0 = most deterministic). Providers
                may ignore this when the model controls its own sampling.
        """
        raise NotImplementedError

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> str:
        """Compatibility bridge for the old ReAct prototype.

        New code should use ``complete``. Older code on this branch passed
        OpenAI-style messages plus optional tools; we flatten those messages
        into a single prompt and ignore tool schemas unless a concrete provider
        overrides this method.
        """
        system = (
            "\n\n".join(
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "system"
            )
            or None
        )
        prompt_parts = [
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
            if message.get("role") != "system"
        ]
        if tools:
            tool_names = [
                str(tool.get("function", {}).get("name", "tool"))
                for tool in tools
                if isinstance(tool, dict)
            ]
            prompt_parts.append("Available tools: " + ", ".join(tool_names))
        return self.complete("\n\n".join(prompt_parts), system=system)

    def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Iterator[str]:
        """Compatibility streaming bridge for older callers."""
        yield self.chat(messages, tools=tools)


Provider = LLMProvider
