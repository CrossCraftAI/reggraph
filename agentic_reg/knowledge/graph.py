"""Typed knowledge graph for a regulatory domain.

Built from a domain's source markdown and backed by NetworkX.  Implements the
``ProposalGraph`` and ``GraphLike`` Protocols so the proposal and symbolic-check
modules can operate on it without importing NetworkX themselves.
"""

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from agentic_reg.domains import Domain

# ── node / edge schemas ────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[([a-z][a-z-]*-\d+)\]", flags=re.IGNORECASE)
_SECTION_RE = re.compile(
    r"^##\s+Article\s+(\d+)\s*[—–-]\s*(.+)", flags=re.MULTILINE | re.IGNORECASE
)


@dataclass
class ClauseNode:
    node_id: str
    label: str
    kind: str = "clause"
    text: str = ""
    source_article: str = ""


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str = "references"
    evidence: str = ""

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.relation)


# ── KnowledgeGraph ──────────────────────────────────────────────────────────


class KnowledgeGraph:
    """Typed regulatory knowledge graph backed by NetworkX.

    Implements ``ProposalGraph`` (for proposals.py) and ``GraphLike``
    (for symbolic.py) so those modules never import NetworkX directly.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    # -- Protocol: GraphLike -------------------------------------------------

    def has_node(self, node_id: str) -> bool:
        """Return whether *node_id* exists in the graph."""
        return node_id.lower() in self._graph

    # -- Protocol: ProposalGraph ---------------------------------------------

    def add_node(  # type: ignore[override]
        self, node_id: str, *, label: str, kind: str
    ) -> None:
        """Add a node to the graph."""
        self._graph.add_node(node_id.lower(), label=label, kind=kind)

    def add_edge(  # type: ignore[override]
        self, source_id: str, target_id: str, relation: str
    ) -> None:
        """Add a typed edge to the graph."""
        self._graph.add_edge(source_id.lower(), target_id.lower(), key=relation.lower())

    def has_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str | None = None,
    ) -> bool:
        """Return whether a matching edge exists."""
        source = source_id.lower()
        target = target_id.lower()
        if relation is None:
            return self._graph.has_edge(source, target)
        return self._graph.has_edge(source, target, relation.lower())

    # -- build / persistence -------------------------------------------------

    @classmethod
    def build(cls, domain: Domain) -> "KnowledgeGraph":
        """Build a graph from the domain's source markdown."""
        graph = cls()
        text = domain.source_path.read_text(encoding="utf-8")
        articles = _parse_articles(text)

        for article in articles:
            node_id = _article_id(article["heading"])
            graph.add_node(
                node_id,
                label=article["heading"],
                kind="clause",
            )
            graph._graph.nodes[node_id]["text"] = article["body"]
            graph._graph.nodes[node_id]["source_article"] = node_id

        # Add cross-reference edges extracted from article bodies.
        for article in articles:
            source_id = _article_id(article["heading"])
            cited = _extract_citations(article["body"])
            for target_id in cited:
                if target_id in graph._graph:
                    graph.add_edge(source_id, target_id, "references")

        return graph

    def save(self, path: str | Path) -> None:
        """Persist the graph as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nodes = {}
        for node_id, attrs in self._graph.nodes(data=True):
            nodes[node_id] = {
                "label": attrs.get("label", ""),
                "kind": attrs.get("kind", "clause"),
                "text": attrs.get("text", ""),
                "source_article": attrs.get("source_article", ""),
            }
        edges = []
        for source, target, key in self._graph.edges(keys=True):
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": key,
                }
            )
        data = {"nodes": nodes, "edges": edges}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        """Load a graph from a JSON file previously written by ``save``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls()
        for node_id, attrs in data.get("nodes", {}).items():
            graph.add_node(
                node_id,
                label=attrs.get("label", ""),
                kind=attrs.get("kind", "clause"),
            )
            graph._graph.nodes[node_id]["text"] = attrs.get("text", "")
            graph._graph.nodes[node_id]["source_article"] = attrs.get("source_article", "")
        for edge in data.get("edges", []):
            graph.add_edge(edge["source"], edge["target"], edge["relation"])
        return graph

    # -- traversal -----------------------------------------------------------

    def expand(self, node_ids: set[str] | list[str], hops: int = 2) -> set[str]:
        """Return nodes reachable within *hops* steps via typed edges (BFS)."""
        start = {nid.lower() for nid in node_ids}
        if not start or hops < 0:
            return start
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((nid, 0) for nid in start)
        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if depth < hops:
                for neighbor in self._graph.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
                for pred in self._graph.predecessors(current):
                    if pred not in visited:
                        queue.append((pred, depth + 1))
        return visited

    # -- query ---------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def nodes(self) -> list[str]:
        return sorted(self._graph.nodes())

    def edges(self) -> list[tuple[str, str, str]]:
        return sorted(self._graph.edges(keys=True))  # type: ignore[no-any-return]


# ── markdown parsing helpers ────────────────────────────────────────────────


def _parse_articles(text: str) -> list[dict[str, str]]:
    """Split a regulation markdown file into article heading + body pairs."""
    articles: list[dict[str, str]] = []
    matches = list(_SECTION_RE.finditer(text))
    for i, match in enumerate(matches):
        heading = f"Article {match.group(1)} — {match.group(2)}"
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        articles.append({"heading": heading, "body": body})
    return articles


def _article_id(heading: str) -> str:
    """Derive a stable node id from an article heading.

    >>> _article_id("Article 5 — Principles")
    'article-5'
    """
    m = re.match(r"Article\s+(\d+)", heading, flags=re.IGNORECASE)
    if not m:
        return heading.lower().replace(" ", "-")
    return f"article-{int(m.group(1))}"


def _extract_citations(text: str) -> list[str]:
    """Return deduplicated, lowercased citations from *text*."""
    seen: dict[str, None] = {}
    for match in _CITATION_RE.findall(text):
        seen.setdefault(match.lower(), None)
    return list(seen)
