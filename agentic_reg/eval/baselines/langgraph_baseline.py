"""Legacy vector-only LangGraph baseline for direct experiments.

The Phase 1 eval CLI now uses the canonical ``single``, ``team``, and
``team-no-graph`` configs. This class remains available for older direct
experiments that want a hand-rolled vector-only LangGraph control:

* No typed knowledge graph (no concept nodes, no typed relations)
* No multi-hop clause-path traversal
* No deterministic citation verification (zero-LLM-cost hallucination detection)
* No symbolic regulatory checks (special-category, erasure, breach deadlines)
* No pluggable domains (hardcoded to whatever domain the index was built for)

Both this baseline and RegGraph use the same:
* LLM provider (same model, same API)
* Embedding model (same SentenceTransformer)
* Vector index (same ChromaDB collection)

The difference is purely the regulatory-specific architecture RegGraph adds on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import StateGraph

from agentic_reg.knowledge.vectors import VectorIndex
from agentic_reg.providers import Provider

# ── state ─────────────────────────────────────────────────────────────────────


@dataclass
class BaselineState:
    """State graph for a single-turn regulatory QA with vanilla LangGraph."""

    question: str = ""
    # retrieval
    retrieved_chunks: list[dict[str, str]] = field(default_factory=list)
    # reasoning
    reasoning: str = ""
    # final output
    answer: str = ""
    # metadata
    token_count: int = 0
    elapsed_ms: float = 0.0


# ── agent ─────────────────────────────────────────────────────────────────────

_RETRIEVAL_PROMPT = """\
You are a regulatory compliance assistant. Answer the question using ONLY the
provided regulatory text excerpts. Cite every claim with article references
in [article-N] brackets.

If the provided excerpts do not contain enough information to answer the
question fully, say so explicitly rather than guessing.

Regulatory excerpts:
{context}

Question: {question}

Answer (with [article-N] citations for every claim):"""


class LangGraphBaseline:
    """Vanilla LangGraph agent for regulatory Q&A — vector-only RAG.

    This is the **baseline** for comparative evaluation. It does everything a
    competent LangGraph + ChromaDB setup can do, but it lacks RegGraph's
    regulatory-specific architecture (typed graph, symbolic checks, deterministic
    citation verification, pluggable domains).

    The Phase 1 CLI does not instantiate this class; use ``team-no-graph`` for
    the supported graph-ablation config.
    """

    def __init__(
        self,
        provider: Provider,
        vector_index: VectorIndex,
        *,
        top_k: int = 4,
    ) -> None:
        self._provider = provider
        self._vector_index = vector_index
        self._top_k = top_k
        self._graph = self._build_graph()

    # -- public API ------------------------------------------------------------

    def answer(self, question: str) -> BaselineState:
        """Answer a regulatory question and return the full state trace."""
        t0 = time.perf_counter()
        state = BaselineState(question=question)

        # Run the LangGraph agent.
        final = self._graph.invoke(state)
        final["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        return BaselineState(**final)

    # -- graph construction ----------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Build a simple retrieve → reason → answer LangGraph pipeline."""
        builder = StateGraph(BaselineState)

        builder.add_node("retrieve", self._retrieve)
        builder.add_node("reason", self._reason)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "reason")
        builder.set_finish_point("reason")

        return builder.compile()

    # -- nodes -----------------------------------------------------------------

    def _retrieve(self, state: BaselineState) -> dict[str, Any]:
        """Vector search — same ChromaDB index RegGraph uses, but no graph expansion."""
        chunks = self._vector_index.search(state.question, top_k=self._top_k)
        return {
            "retrieved_chunks": [{"text": c.text, "article_ref": c.article_ref} for c in chunks],
        }

    def _reason(self, state: BaselineState) -> dict[str, Any]:
        """LLM reasoning over retrieved chunks — no symbolic checks, no graph context."""
        context = "\n\n---\n\n".join(
            f"[{c['article_ref']}] {c['text']}" for c in state.retrieved_chunks
        )
        prompt = _RETRIEVAL_PROMPT.format(context=context, question=state.question)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        answer = self._provider.chat(messages)

        # Rough token estimate: ~4 chars per token for English text.
        prompt_tokens = len(prompt) // 4
        answer_tokens = len(answer) // 4
        return {
            "reasoning": f"Retrieved {len(state.retrieved_chunks)} chunks from vector index.",
            "answer": answer,
            "token_count": prompt_tokens + answer_tokens,
        }
