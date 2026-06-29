"""Specialist roles used by the team orchestrator."""

from collections.abc import Callable
from typing import TypedDict

from .._internal import extract_citations
from ..config import Settings
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.vectors import VectorIndex
from ..providers.base import LLMProvider
from ..retrieval import RetrievedContext, hybrid_retrieve
from .state import Finding

_CLAUSE_SYSTEM = (
    "You are a regulatory Clause Analyst. Interpret the provided clauses. "
    "Use only the clause text and cite clause ids in square brackets."
)
_CLAUSE_PROMPT = """Sub-question: {q}

{context}

In 2-4 sentences, explain what the relevant clauses require or permit. Cite
clause ids in [brackets]."""

_XREF_SYSTEM = (
    "You are a regulatory Cross-Reference Analyst. Use graph evidence to trace "
    "dependencies, conditions, and exceptions. If no relationship evidence was "
    "retrieved, say so. Cite clause ids in square brackets."
)
_XREF_PROMPT = """Sub-question: {q}

{context}

In 2-4 sentences, explain how the clauses depend on, condition, or limit one
another. Cite clause ids in [brackets]."""

_CURATOR_SYSTEM = (
    "You are a regulatory Graph Curator. Identify missing graph entities or "
    "relations only when the evidence supports them. Cite clause ids in square brackets."
)
_CURATOR_PROMPT = """Sub-question: {q}

{context}

In 2-4 sentences, name any graph relationship or concept that appears missing.
If the graph already captures the relationship, say so. Cite clause ids in
[brackets]."""


class SpecialistConfig(TypedDict):
    name: str
    system: str
    template: str
    build_context: Callable[[RetrievedContext], str]


def _clause_context(ctx: RetrievedContext) -> str:
    lines = ["RELEVANT CLAUSES:"]
    for hit in ctx.vector_hits:
        lines.append(f"- [{hit.id}] {hit.title}\n  {hit.text}")
    return "\n".join(lines)


def _node_label(node_id: str, labels: dict[str, str]) -> str:
    label = labels.get(node_id, node_id)
    return f"[{node_id}] {label}"


def _graph_context(ctx: RetrievedContext) -> str:
    lines: list[str] = []
    labels = {str(node["id"]): str(node.get("label", node["id"])) for node in ctx.graph_nodes}

    if ctx.multi_hop_paths:
        lines.append("REASONING CHAINS:")
        for path in ctx.multi_hop_paths:
            lines.append(f"- {path['text']}")

    if ctx.graph_edges:
        if lines:
            lines.append("")
        lines.append("RELATED GRAPH EDGES:")
        for edge in ctx.graph_edges:
            source = str(edge["source"])
            target = str(edge["target"])
            relation = edge.get("relation", "related")
            lines.append(
                f"- {_node_label(source, labels)} --{relation}--> {_node_label(target, labels)}"
            )

    if not lines:
        lines.append("NO GRAPH RELATIONSHIPS RETRIEVED.")

    lines.append("")
    lines.append("SOURCE CLAUSES:")
    for hit in ctx.vector_hits:
        lines.append(f"- [{hit.id}] {hit.title}\n  {hit.text}")

    return "\n".join(lines)


SPECIALISTS: dict[str, SpecialistConfig] = {
    "clause_analyst": {
        "name": "Clause Analyst",
        "system": _CLAUSE_SYSTEM,
        "template": _CLAUSE_PROMPT,
        "build_context": _clause_context,
    },
    "cross_reference": {
        "name": "Cross-Reference Analyst",
        "system": _XREF_SYSTEM,
        "template": _XREF_PROMPT,
        "build_context": _graph_context,
    },
    "graph_curator": {
        "name": "Graph Curator",
        "system": _CURATOR_SYSTEM,
        "template": _CURATOR_PROMPT,
        "build_context": _graph_context,
    },
}

DEFAULT_ROLE = "clause_analyst"


def display_name(role: str) -> str:
    if role == "supervisor":
        return "Supervisor"
    specialist = SPECIALISTS.get(role)
    return specialist["name"] if specialist else role


def run_specialist(
    role: str,
    question: str,
    provider: LLMProvider,
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
    settings: Settings,
) -> Finding:
    cfg = SPECIALISTS.get(role)
    if cfg is None:
        role = DEFAULT_ROLE
        cfg = SPECIALISTS[role]

    context = hybrid_retrieve(question, vector_index, graph, settings)
    text = provider.complete(
        cfg["template"].format(q=question, context=cfg["build_context"](context)),
        system=cfg["system"],
    )
    return Finding(
        role=role,
        sub_question=question,
        text=text,
        citations=extract_citations(text),
        retrieved_ids=[hit.id for hit in context.vector_hits],
        graph_node_ids=[str(node["id"]) for node in context.graph_nodes],
        graph_edges=context.graph_edges,
        multi_hop_paths=context.multi_hop_paths,
    )
