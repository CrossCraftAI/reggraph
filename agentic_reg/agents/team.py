"""Basic Phase 0 team orchestrator.

This intentionally stays small. Later PRD commits will make the team
hierarchical, add graph-curator proposals, and wire verification/revision.
"""

from ..config import Settings
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.vectors import VectorIndex
from ..providers.base import LLMProvider
from ..trace import ReasoningTrace
from .specialists import display_name, run_specialist

_SYNTH_SYSTEM = (
    "You are a careful regulatory analyst. Synthesize specialist findings into "
    "one concise answer grounded in cited clauses."
)

_SYNTH_PROMPT = """Question: {question}

Specialist findings:
{findings}

Write the final answer. Cite clause ids in square brackets."""


class RegulatoryTeam:
    """Run a small fixed set of specialists and synthesize their findings."""

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

    def answer(self, question: str) -> ReasoningTrace:
        trace = ReasoningTrace(question=question)
        roles = ["clause_analyst", "cross_reference"][: self.settings.max_subquestions]
        trace.add_step(
            "plan",
            f"Selected {len(roles)} specialist role(s).",
            roles=[display_name(role) for role in roles],
        )

        findings = [
            run_specialist(
                role,
                question,
                self.provider,
                self.vector_index,
                self.graph,
                self.settings,
            )
            for role in roles
        ]
        for finding in findings:
            trace.add_step(
                f"specialist:{finding.role}",
                f"{display_name(finding.role)} produced a finding.",
                finding=finding.text,
                citations=finding.citations,
                retrieved_ids=finding.retrieved_ids,
                graph_node_ids=finding.graph_node_ids,
                graph_edges=finding.graph_edges,
                multi_hop_paths=finding.multi_hop_paths,
            )

        findings_block = "\n".join(
            f"- ({display_name(finding.role)}) {finding.text}" for finding in findings
        )
        answer = self.provider.complete(
            _SYNTH_PROMPT.format(question=question, findings=findings_block),
            system=_SYNTH_SYSTEM,
        )
        trace.add_step("synthesize", "Synthesized specialist findings.", answer=answer)
        trace.answer = answer
        return trace
