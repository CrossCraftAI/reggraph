"""Build the persisted knowledge store (vectors + graph) for a domain.

    uv run python -m agentic_reg.build               # default domain (settings)
    uv run python -m agentic_reg.build --domain uk_dpa
    uv run python -m agentic_reg.build --domain gdpr --no-enrich

Safe to run without an LLM running: the vector index and the cross-reference
graph are built deterministically; LLM concept enrichment is best-effort.
"""

import argparse
import shutil

from .config import get_settings
from .domains import get_domain
from .ingest import load_chunks
from .knowledge.extract import (
    CONCEPT_EDGE,
    extract_cross_references,
    extract_graph_elements,
)
from .knowledge.graph import KnowledgeGraph
from .knowledge.vectors import VectorIndex


def build(domain_name: str | None = None, *, enrich: bool = True) -> None:
    settings = get_settings()
    domain = get_domain(domain_name or settings.domain)

    print(f"Domain: {domain.name} — {domain.title}")
    print(f"Loading document: {domain.source_path}")
    chunks = load_chunks(domain.source_path)
    print(f"  {len(chunks)} chunk(s): {', '.join(c.id for c in chunks)}")

    print("Building vector index (downloads the embedding model on first run)...")
    if domain.chroma_dir.exists():
        shutil.rmtree(domain.chroma_dir)
    vector_index = VectorIndex(domain.chroma_dir, settings.embedding_model)
    vector_index.add(chunks)
    print(f"  vector store now holds {vector_index.count()} chunk(s).")

    print("Building knowledge graph...")
    graph = KnowledgeGraph(symbolic_rules=domain.symbolic_rules)
    for chunk in chunks:
        graph.add_node(chunk.id, label=chunk.title, kind="clause", text=chunk.text)
    known_ids = {c.id for c in chunks}
    cross_refs = extract_cross_references(chunks)
    for source, target in cross_refs:
        graph.add_edge(source, target, "references")
    print(f"  {len(cross_refs)} cross-reference edge(s) from clause text.")

    if enrich:
        # Best-effort typed LLM enrichment; never blocks the build.
        try:
            from .providers import get_provider

            provider = get_provider(settings)
            print(f"Enriching via '{provider.name}' (unit='{domain.unit_label}')...")
            concepts = relations = 0
            for chunk in chunks:
                extraction = extract_graph_elements(
                    chunk, provider, known_ids, unit_label=domain.unit_label
                )
                for concept in extraction.concepts:
                    graph.add_node(concept.id, label=concept.label, kind=concept.kind)
                    graph.add_edge(
                        chunk.id, concept.id, CONCEPT_EDGE.get(concept.kind, "introduces")
                    )
                    concepts += 1
                for relation in extraction.relations:
                    graph.add_edge(relation.source_id, relation.target_id, relation.relation)
                    relations += 1
            print(f"  added {concepts} concept node(s) and {relations} typed relation(s).")
        except Exception as exc:
            print(f"  (skipped LLM enrichment: {type(exc).__name__}: {exc})")
    else:
        print("Skipping LLM enrichment (--no-enrich); deterministic graph only.")

    graph.save(domain.graph_path)
    print(f"  graph saved: {graph.num_nodes} node(s), {graph.num_edges} edge(s).")
    print(f"\nDone. Knowledge store written to {domain.store_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agentic_reg.build")
    parser.add_argument("--domain", default=None, help="domain name (default from settings)")
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="skip best-effort LLM concept/relation enrichment",
    )
    args = parser.parse_args(argv)
    build(args.domain, enrich=not args.no_enrich)


if __name__ == "__main__":
    main()
