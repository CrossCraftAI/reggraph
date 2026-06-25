"""The single reasoning agent, built on LangGraph.

A deliberately small, linear graph — ``retrieve -> reason -> answer`` — so the
structure is easy to learn. Every node appends to the reasoning trace, which is
the auditable record of how the answer was produced.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .knowledge.graph import KnowledgeGraph
from .knowledge.vectors import VectorIndex
from .providers.base import LLMProvider
from .retrieval import RetrievedContext, hybrid_retrieve
from .trace import ReasoningTrace

_SYSTEM = (
    "You are a careful regulatory analyst. Answer ONLY from the provided clauses "
    "and cite the clause ids in square brackets, e.g. [article-6] or [section-3]. If the clauses "
    "do not contain the answer, say so plainly rather than guessing."
)

_REASON_PROMPT = """Question: {question}

{context}

Think step by step about which clauses apply and how they relate to one another.
Write a brief analysis (3-6 sentences). Do not write the final answer yet."""

_ANSWER_PROMPT = """Question: {question}

{context}

Analysis:
{reasoning}

Now write the final answer for the user. Ground every claim in the clauses above
and cite clause ids in square brackets, e.g. [article-6] or [section-3]. Be concise."""


class AgentState(TypedDict):
    question: str
    context: RetrievedContext | None
    reasoning: str
    answer: str
    trace: ReasoningTrace


class RegulatoryAgent:
    def __init__(
        self,
        provider: LLMProvider,
        vector_index: VectorIndex,
        graph: KnowledgeGraph,
        settings: Settings,
    ) -> None:
        self.provider = provider
        self.vector_index = vector_index
        self.graph = graph
        self.settings = settings
        self._app = self._build()

    def _build(self):
        builder = StateGraph(AgentState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("reason", self._reason)
        builder.add_node("answer", self._answer)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "reason")
        builder.add_edge("reason", "answer")
        builder.add_edge("answer", END)
        return builder.compile()

    # --- nodes ---
    def _retrieve(self, state: AgentState) -> dict:
        ctx = hybrid_retrieve(state["question"], self.vector_index, self.graph, self.settings)
        state["trace"].add_step(
            "retrieve",
            f"Vector search found {len(ctx.vector_hits)} clause(s); graph expansion added "
            f"{len(ctx.graph_nodes)} related node(s); "
            f"{len(ctx.multi_hop_paths)} multi-hop chain(s).",
            vector_hits=[
                {"id": h.id, "title": h.title, "score": round(h.score, 4)} for h in ctx.vector_hits
            ],
            multi_hop_paths=ctx.multi_hop_paths,
            graph_edges=ctx.graph_edges,
        )
        return {"context": ctx}

    def _reason(self, state: AgentState) -> dict:
        ctx = state["context"]
        assert ctx is not None
        prompt = _REASON_PROMPT.format(question=state["question"], context=ctx.to_prompt_context())
        reasoning = self.provider.complete(prompt, system=_SYSTEM)
        state["trace"].add_step(
            "reason", "Analysed which clauses apply and how they connect.", reasoning=reasoning
        )
        return {"reasoning": reasoning}

    def _answer(self, state: AgentState) -> dict:
        ctx = state["context"]
        assert ctx is not None
        prompt = _ANSWER_PROMPT.format(
            question=state["question"],
            context=ctx.to_prompt_context(),
            reasoning=state["reasoning"],
        )
        answer = self.provider.complete(prompt, system=_SYSTEM)
        state["trace"].add_step(
            "answer", "Wrote the grounded final answer with citations.", answer=answer
        )
        state["trace"].answer = answer
        return {"answer": answer}

    # --- public API ---
    def answer(self, question: str) -> ReasoningTrace:
        """Run the full graph and return the populated reasoning trace."""
        trace = ReasoningTrace(question=question)
        self._app.invoke(
            {
                "question": question,
                "context": None,
                "reasoning": "",
                "answer": "",
                "trace": trace,
            }
        )
        return trace
