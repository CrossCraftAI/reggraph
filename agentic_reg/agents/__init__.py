"""Agent orchestration: choose single-agent or basic team mode by config."""

from ..agent import RegulatoryAgent
from ..config import Settings
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.vectors import VectorIndex
from ..providers.base import LLMProvider
from .team import RegulatoryTeam


def get_orchestrator(
    settings: Settings,
    provider: LLMProvider,
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
):
    if settings.agent_mode.lower() == "single":
        return RegulatoryAgent(provider, vector_index, graph, settings)
    return RegulatoryTeam(provider, vector_index, graph, settings)


__all__ = ["RegulatoryAgent", "RegulatoryTeam", "get_orchestrator"]
