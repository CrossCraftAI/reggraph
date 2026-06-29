"""Regulatory knowledge graph, backed by NetworkX (in-memory + JSON-on-disk).

Nodes represent clauses and concepts; edges are cross-references and typed
relations. Currently NetworkX — swapping for Neo4j behind this same interface
would be the natural next step if the graph grows beyond prototype scale.

TODO: the undirected walk in expand() works fine but is O(nodes_in_frontier *
avg_degree) — fine for ~100 nodes, but worth profiling before scaling.
"""

import json
from pathlib import Path

import networkx as nx

from agentic_reg._internal import extract_citations
from agentic_reg.domains import Domain
from agentic_reg.ingest import _make_id, load_chunks


class KnowledgeGraph:
    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self.g: nx.DiGraph = graph if graph is not None else nx.DiGraph()
        self._graph = self.g  # compatibility for the older eval prototype

    # --- construction ---
    def add_node(self, node_id: str, *, label: str, kind: str, **attrs: object) -> None:
        self.g.add_node(node_id, label=label, kind=kind, **attrs)

    def add_edge(self, source: str, target: str, relation: str) -> None:
        self.g.add_edge(source, target, relation=relation)

    def has_node(self, node_id: str) -> bool:
        return self.g.has_node(node_id.lower())

    def has_edge(self, source: str, target: str, relation: str | None = None) -> bool:
        source_id = source.lower()
        target_id = target.lower()
        if not self.g.has_edge(source_id, target_id):
            return False
        if relation is None:
            return True
        return self.g[source_id][target_id].get("relation") == relation.lower()

    @classmethod
    def build(cls, domain: Domain) -> "KnowledgeGraph":
        """Compatibility builder for older callers.

        The primary build path now lives in ``agentic_reg.build``. This helper
        still builds a deterministic clause graph from the domain document.
        """
        graph = cls()
        chunks = load_chunks(domain.source_path)
        known_ids = {chunk.id for chunk in chunks}
        for chunk in chunks:
            graph.add_node(chunk.id, label=chunk.title, kind="clause", text=chunk.text)
        for chunk in chunks:
            for target in extract_citations(chunk.text):
                if target in known_ids and target != chunk.id:
                    graph.add_edge(chunk.id, target, "references")
        return graph

    # --- retrieval ---
    def expand(self, seed_ids: list[str], hops: int = 1) -> tuple[list[dict], list[dict]]:
        """Return (nodes, edges) within ``hops`` of any seed (undirected walk).

        The walk is undirected so we surface relationships in either direction
        (e.g. both "Article 6 references Article 9" and the reverse view).
        """
        seeds = [n for n in seed_ids if self.g.has_node(n)]
        undirected = self.g.to_undirected(as_view=True)

        selected: set[str] = set(seeds)
        frontier: set[str] = set(seeds)
        for _ in range(max(hops, 0)):
            nxt: set[str] = set()
            for node in frontier:
                nxt.update(undirected.neighbors(node))
            nxt -= selected
            selected.update(nxt)
            frontier = nxt
            if not frontier:
                break

        nodes = [{"id": n, **self.g.nodes[n]} for n in selected]
        edges = [
            {"source": u, "target": v, "relation": data.get("relation", "")}
            for u, v, data in self.g.edges(data=True)
            if u in selected and v in selected
        ]
        return nodes, edges

    def label(self, node_id: str) -> str:
        if self.g.has_node(node_id):
            return self.g.nodes[node_id].get("label", node_id)
        return node_id

    def edge_relation(self, a: str, b: str) -> str:
        """Relation label between two adjacent nodes, in either direction."""
        if self.g.has_edge(a, b):
            return self.g[a][b].get("relation", "related")
        if self.g.has_edge(b, a):
            return self.g[b][a].get("relation", "related")
        return "related"

    def clause_paths(self, seed_ids: list[str], cutoff: int = 3) -> list[list[str]]:
        """Shortest paths connecting pairs of seed clauses (multi-hop chains).

        Restricted to clause nodes (kind == "clause") so the chains read as clean
        clause-to-clause reasoning paths (e.g. section-9 -> section-8 -> section-3)
        rather than wandering through concept nodes.
        """
        import itertools

        clause_nodes = [n for n, d in self.g.nodes(data=True) if d.get("kind") == "clause"]
        sub = self.g.subgraph(clause_nodes).to_undirected()

        seeds = sorted({n for n in seed_ids if n in sub})
        paths: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for a, b in itertools.combinations(seeds, 2):
            try:
                path = nx.shortest_path(sub, a, b)
            except nx.NetworkXNoPath:
                continue
            key = tuple(path)
            if 2 <= len(path) <= cutoff + 1 and key not in seen:
                seen.add(key)
                paths.append(path)
        return paths

    # --- persistence ---
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.g, edges="edges")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data.get("nodes"), dict):
            graph = cls()
            for node_id, attrs in data.get("nodes", {}).items():
                graph.add_node(
                    node_id,
                    label=attrs.get("label", node_id),
                    kind=attrs.get("kind", "clause"),
                    text=attrs.get("text", ""),
                    source_article=attrs.get("source_article", ""),
                )
            for edge in data.get("edges", []):
                graph.add_edge(edge["source"], edge["target"], edge.get("relation", "references"))
            return graph
        graph = nx.node_link_graph(data, directed=True, multigraph=False, edges="edges")
        return cls(graph)

    def view(self, kinds: set[str] | None = None) -> tuple[list[dict], list[dict]]:
        """Return (nodes, edges) for visualization, optionally filtered by kind."""
        nodes = [
            {"id": n, **data}
            for n, data in self.g.nodes(data=True)
            if kinds is None or data.get("kind") in kinds
        ]
        ids = {node["id"] for node in nodes}
        edges = [
            {"source": u, "target": v, "relation": data.get("relation", "")}
            for u, v, data in self.g.edges(data=True)
            if u in ids and v in ids
        ]
        return nodes, edges

    # --- stats ---
    @property
    def num_nodes(self) -> int:
        return self.g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.g.number_of_edges()

    @property
    def node_count(self) -> int:
        return self.num_nodes

    @property
    def edge_count(self) -> int:
        return self.num_edges

    def nodes(self) -> list[str]:
        return sorted(self.g.nodes())

    def edges(self) -> list[tuple[str, str, str]]:
        return sorted((u, v, data.get("relation", "")) for u, v, data in self.g.edges(data=True))


def _article_id(heading: str) -> str:
    """Backward-compatible article-id helper used by older tests."""
    return _make_id(heading)
