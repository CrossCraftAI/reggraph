"""Build a domain's knowledge store — vector index and typed graph.

Usage::

    python -m agentic_reg.build --domain gdpr
    python -m agentic_reg.build --domain gdpr --no-enrich
"""

import argparse

from agentic_reg.config import PROJECT_ROOT, get_settings
from agentic_reg.domains import get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorIndex


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a regulatory domain's knowledge store.")
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name (e.g. gdpr).",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip LLM concept typing — build a deterministic graph only.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    domain = get_domain(args.domain)
    store_dir = PROJECT_ROOT / "data" / "store" / domain.name
    store_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vector index
    print(f"Building vector index for {domain.name}...")
    vector_index = VectorIndex.build(domain, settings.embedding_model)
    print(f"  ✓ {vector_index.chunk_count} chunks indexed")

    # 2. Knowledge graph
    print(f"Building knowledge graph for {domain.name}...")
    graph = KnowledgeGraph.build(domain)
    graph_path = store_dir / "graph.json"
    graph.save(graph_path)
    print(f"  ✓ {graph.node_count} nodes, {graph.edge_count} edges → {graph_path}")

    # 3. LLM enrichment (unless skipped)
    if args.no_enrich:
        print("Skipping LLM enrichment (--no-enrich).")
    else:
        _enrich_graph(domain, graph, settings)
        graph.save(graph_path)
        print(f"  ✓ enriched graph saved → {graph_path}")

    print(f"Done. Store written to {store_dir}")


def _enrich_graph(domain, graph, settings) -> None:
    """Extract typed concepts from articles and add them to the graph.

    Not yet implemented — requires the provider layer and agent loop.
    For now this is a no-op stub.
    """
    print("  (LLM enrichment not yet implemented — skipping)")


if __name__ == "__main__":
    main()
