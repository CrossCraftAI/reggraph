"""Basic specialist roles for the Phase 0 team orchestrator."""

from ..config import Settings
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.vectors import VectorIndex
from ..providers.base import LLMProvider
from ..retrieval import hybrid_retrieve
from .state import Finding, extract_citations

SPECIALISTS = {
    "clause_analyst": "Focus on direct clause requirements and permissions.",
    "cross_reference": "Focus on related clauses and graph reasoning chains.",
}

DEFAULT_ROLE = "clause_analyst"


def display_name(role: str) -> str:
    return role.replace("_", " ").title()


def run_specialist(
    role: str,
    question: str,
    provider: LLMProvider,
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
    settings: Settings,
) -> Finding:
    role = role if role in SPECIALISTS else DEFAULT_ROLE
    context = hybrid_retrieve(question, vector_index, graph, settings)
    prompt = f"""Question: {question}

Specialist role: {display_name(role)}
Instruction: {SPECIALISTS[role]}

{context.to_prompt_context()}

Write a concise finding grounded in the cited clauses."""
    text = provider.complete(prompt, system="You are a careful regulatory specialist.")
    return Finding(
        role=role,
        sub_question=question,
        text=text,
        citations=extract_citations(text),
        retrieved_ids=[hit.id for hit in context.vector_hits],
        graph_node_ids=[node["id"] for node in context.graph_nodes],
        graph_edges=context.graph_edges,
        multi_hop_paths=context.multi_hop_paths,
    )
