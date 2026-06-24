"""Thin ReAct query tool for asking questions over a regulatory knowledge store.

Usage::

    python -m agentic_reg.ask "What lawful bases allow processing of special category data?"
    python -m agentic_reg.ask --domain gdpr "When must a breach be notified?"

Outputs the answer and a structured trace (JSON) to stdout, or writes it to a file
with ``--output <path>``.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_reg.config import PROJECT_ROOT, get_settings
from agentic_reg.domains import get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.symbolic import run_symbolic_checks
from agentic_reg.knowledge.vectors import VectorIndex
from agentic_reg.providers import get_provider

# ── tool definitions (OpenAI function-calling format) ──────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search regulation text. Returns chunks with article references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": "Expand from article nodes in the graph. Returns connected article IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Article node IDs to expand from, e.g. ['article-6'].",
                    },
                    "hops": {
                        "type": "integer",
                        "description": "How many hops to traverse (default 1).",
                    },
                },
                "required": ["node_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify",
            "description": (
                "Run deterministic checks on a draft answer. Returns pass/fail findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The original user question.",
                    },
                    "draft_answer": {
                        "type": "string",
                        "description": "The draft answer to verify.",
                    },
                },
                "required": ["question", "draft_answer"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a regulatory analyst answering questions about the GDPR.

You have access to tools:
- **search(query)**: find relevant regulation text by semantic search. Always use this first.
- **expand(node_ids, hops)**: traverse the regulation graph to find connected articles.
- **verify(question, draft_answer)**: run deterministic checks on your draft answer.

Workflow:
1. Search for relevant clauses.
2. Expand from matching articles to find connected requirements.
3. Draft an answer citing specific articles in [article-N] format.
4. Verify your draft answer with the verify tool.
5. If verify finds issues, fix them and verify again.
6. Return the final answer.

Always cite articles using [article-N] format. Every claim must be traceable to an article.
"""

MAX_TURNS = 6


# ── data types ──────────────────────────────────────────────────────────────


@dataclass
class Step:
    role: str  # "thought" | "tool_call" | "tool_result" | "answer"
    content: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: Any = None


@dataclass
class Trace:
    question: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# ── tool implementations ────────────────────────────────────────────────────


def _tool_search(query: str, vector_index: VectorIndex) -> str:
    chunks = vector_index.search(query, top_k=4)
    if not chunks:
        return "No relevant clauses found."
    lines = []
    for chunk in chunks:
        lines.append(f"[{chunk.article_ref}] {chunk.text[:300]}")
    return "\n\n".join(lines)


def _tool_expand(node_ids: list[str], graph: KnowledgeGraph, hops: int = 1) -> str:
    reached = graph.expand(set(node_ids), hops=hops)
    if not reached:
        return "No connected articles found."
    return "Connected articles: " + ", ".join(sorted(reached))


def _tool_verify(
    question: str,
    draft_answer: str,
    graph: KnowledgeGraph,
) -> str:
    findings = run_symbolic_checks(question, draft_answer, graph)
    if not findings:
        return "No symbolic rules triggered."
    lines = []
    for finding in findings:
        status = "PASS" if finding.passed else "FAIL"
        lines.append(f"[{status}] {finding.rule_id}: {finding.message}")
    return "\n".join(lines)


# ── ReAct loop ──────────────────────────────────────────────────────────────


def answer(
    question: str,
    *,
    domain_name: str = "gdpr",
) -> Trace:
    """Run the ReAct agent loop and return a structured trace."""
    settings = get_settings()
    domain = get_domain(domain_name)
    provider = get_provider(settings)

    store_dir = PROJECT_ROOT / "data" / "store" / domain.name
    graph_path = store_dir / "graph.json"

    graph = KnowledgeGraph.load(graph_path)
    vector_index = VectorIndex.load(str(store_dir / "chroma"), settings.embedding_model)

    trace = Trace(question=question)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _turn in range(MAX_TURNS):
        response_text = provider.chat(messages, tools=TOOLS)

        # Try to parse as a tool call. The provider returns plain text;
        # tool calls may arrive as JSON fragments or as a natural-language
        # answer.  We heuristically detect both.
        tool_call = _try_parse_tool_call(response_text)

        if tool_call is None:
            # Natural-language answer — treat as final.
            trace.steps.append(Step(role="answer", content=response_text))
            trace.answer = response_text
            break

        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        trace.steps.append(Step(role="tool_call", tool_name=name, tool_args=args))

        # Execute the tool.
        result = _execute_tool(name, args, graph, vector_index)
        trace.steps.append(Step(role="tool_result", tool_name=name, tool_result=result))

        # Feed the tool result back to the LLM.
        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
            }
        )
    else:
        # Ran out of turns without a final answer — ask for one explicitly.
        messages.append(
            {
                "role": "user",
                "content": (
                    "Please provide your final answer now, "
                    "citing relevant articles in [article-N] format."
                ),
            }
        )
        final = provider.chat(messages)
        trace.steps.append(Step(role="answer", content=final))
        trace.answer = final

    # Run symbolic checks on the final answer and attach to trace.
    if graph is not None and trace.answer:
        findings = run_symbolic_checks(question, trace.answer, graph)
        trace.findings = [f.to_dict() for f in findings]

    return trace


def _execute_tool(
    name: str,
    args: dict[str, Any],
    graph: KnowledgeGraph,
    vector_index: VectorIndex,
) -> str:
    if name == "search":
        return _tool_search(args.get("query", ""), vector_index)
    if name == "expand":
        node_ids = args.get("node_ids", [])
        hops = args.get("hops", 1)
        return _tool_expand(node_ids, graph, hops=hops)
    if name == "verify":
        return _tool_verify(
            args.get("question", ""),
            args.get("draft_answer", ""),
            graph,
        )
    return f"Unknown tool: {name}"


def _try_parse_tool_call(text: str) -> dict[str, Any] | None:
    """Heuristically detect a tool call in the LLM response.

    The OpenAI-compat API may return tool calls in the message payload
    (via ``tool_calls``), but when using plain ``chat()`` we get the text
    directly.  We look for JSON tool-call blocks or function-call markers.
    """
    text = text.strip()

    # If the response looks like JSON, try parsing it as a tool call.
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if "name" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # If the response contains a JSON block, extract and parse it.
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            parsed = json.loads(text[start:end].strip())
            if "name" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question over a regulatory knowledge store."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask (omit for interactive prompt).",
    )
    parser.add_argument(
        "--domain",
        default="gdpr",
        help="Domain name (default: gdpr).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write trace JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Output only the trace JSON (no human-readable answer).",
    )
    args = parser.parse_args(argv)

    question = args.question
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(1)

    trace = answer(question, domain_name=args.domain)

    if args.trace_only:
        output = trace.to_json()
    else:
        output = f"{trace.answer}\n\n--- trace ---\n{trace.to_json()}"

    if args.output:
        import os

        os.makedirs(args.output.rsplit("/", 1)[0] if "/" in args.output else ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
        print(f"Trace written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
