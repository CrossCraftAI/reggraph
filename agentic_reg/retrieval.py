"""Hybrid retrieval — vector search + graph expansion + multi-hop path-finding.

The key idea: vector search finds clauses near the question, graph expansion
pulls in structurally-related nodes, and multi-hop paths surface the reasoning
chains between them (e.g. breach -> security -> integrity principle). That last
bit is what makes this more than flat RAG.

TODO: consider adding a re-rank step after graph expansion, particularly when
graph_hops > 2 pulls in a lot of noise.
"""

from dataclasses import dataclass, field

from .config import Settings
from .knowledge.graph import KnowledgeGraph
from .knowledge.vectors import VectorHit, VectorIndex


@dataclass
class RetrievedContext:
    vector_hits: list[VectorHit]
    graph_nodes: list[dict]
    graph_edges: list[dict]
    multi_hop_paths: list[dict] = field(default_factory=list)  # {"nodes": [...], "text": "..."}

    def to_prompt_context(self) -> str:
        lines = ["RELEVANT CLAUSES (semantic match):"]
        for hit in self.vector_hits:
            lines.append(f"- [{hit.id}] {hit.title}\n  {hit.text}")

        if self.multi_hop_paths:
            lines.append("\nREASONING CHAINS (multi-hop links between clauses):")
            for path in self.multi_hop_paths:
                lines.append(f"- {path['text']}")

        if self.graph_edges:
            label = {n["id"]: n.get("label", n["id"]) for n in self.graph_nodes}
            lines.append("\nRELATED CLAUSES & CONCEPTS (knowledge graph):")
            for edge in self.graph_edges:
                src = label.get(edge["source"], edge["source"])
                dst = label.get(edge["target"], edge["target"])
                lines.append(f"- {src} --{edge['relation']}--> {dst}")
        return "\n".join(lines)


def _describe_path(graph: KnowledgeGraph, path: list[str]) -> str:
    parts = [graph.label(path[0])]
    for a, b in zip(path, path[1:], strict=False):
        parts.append(f"--{graph.edge_relation(a, b)}-->")
        parts.append(graph.label(b))
    return " ".join(parts)


def hybrid_retrieve(
    question: str,
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
    settings: Settings,
) -> RetrievedContext:
    hits = vector_index.search(question, settings.vector_top_k)

    # Ablation baseline: vector-only retrieval, no graph contribution.
    if not settings.use_graph:
        return RetrievedContext(vector_hits=hits, graph_nodes=[], graph_edges=[])

    seed_ids = [hit.id for hit in hits]
    nodes, edges = graph.expand(seed_ids, hops=settings.graph_hops)

    paths = graph.clause_paths(seed_ids, cutoff=settings.graph_hops + 1)
    multi_hop_paths = [
        {"nodes": path, "text": _describe_path(graph, path)} for path in paths if len(path) >= 3
    ]

    return RetrievedContext(
        vector_hits=hits,
        graph_nodes=nodes,
        graph_edges=edges,
        multi_hop_paths=multi_hop_paths,
    )
