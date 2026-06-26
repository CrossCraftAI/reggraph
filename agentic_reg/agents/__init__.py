"""Agent orchestrator selection."""

from typing import Protocol

from ..agent import RegulatoryAgent
from ..config import Settings
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.vectors import VectorIndex
from ..providers.base import LLMProvider
from ..trace import ReasoningTrace
from .team import RegulatoryTeam


class Orchestrator(Protocol):
    graph: KnowledgeGraph

    def answer(self, question: str) -> ReasoningTrace:
        """Return an answer trace for ``question``."""
        ...


def get_orchestrator(
    settings: Settings,
    provider: LLMProvider,
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
) -> Orchestrator:
    """Return the configured orchestrator."""
    if settings.agent_mode.lower() == "single":
        return RegulatoryAgent(provider, vector_index, graph, settings)
    return RegulatoryTeam(provider, vector_index, graph, settings)


__all__ = ["Orchestrator", "RegulatoryAgent", "RegulatoryTeam", "get_orchestrator"]
