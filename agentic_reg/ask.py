"""Ask questions through the configured Phase 0 orchestrator.

Usage::

    python -m agentic_reg.ask "What lawful bases allow processing?"
    python -m agentic_reg.ask --domain uk_dpa "When must a breach be notified?"
"""

import argparse
import sys
from pathlib import Path
from typing import cast

from agentic_reg.agents import get_orchestrator
from agentic_reg.config import AgentMode, get_settings
from agentic_reg.domains import get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorIndex
from agentic_reg.providers import get_provider
from agentic_reg.trace import ReasoningTrace


def answer(
    question: str,
    *,
    domain_name: str = "gdpr",
    mode: AgentMode | None = None,
) -> ReasoningTrace:
    """Run the configured orchestrator and return its reasoning trace."""
    settings = get_settings()
    domain = get_domain(domain_name)
    settings.domain = domain.name
    if mode is not None:
        settings.agent_mode = mode

    graph = KnowledgeGraph.load(domain.graph_path)
    vector_index = VectorIndex.load(domain.chroma_dir, settings.embedding_model)
    provider = get_provider(settings)
    orchestrator = get_orchestrator(settings, provider, vector_index, graph)
    return orchestrator.answer(question)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask a question over a regulatory store.")
    parser.add_argument("question", nargs="?", help="Question to ask.")
    parser.add_argument("--domain", default="gdpr", help="Domain name (default: gdpr).")
    parser.add_argument(
        "--mode",
        choices=["single", "team"],
        default=None,
        help="Override AGENTIC_REG_AGENT_MODE for this question.",
    )
    parser.add_argument("--output", "-o", help="Write output to this file instead of stdout.")
    parser.add_argument("--trace-only", action="store_true", help="Output only trace JSON.")
    args = parser.parse_args(argv)

    question = args.question or input("Question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(1)

    trace = answer(question, domain_name=args.domain, mode=cast(AgentMode | None, args.mode))
    output = (
        trace.to_json()
        if args.trace_only
        else f"{trace.answer}\n\n--- trace ---\n{trace.to_json()}"
    )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        print(f"Trace written to {path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
