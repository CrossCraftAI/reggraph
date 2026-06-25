from agentic_reg.agent import RegulatoryAgent
from agentic_reg.config import Settings
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorHit
from agentic_reg.providers.base import LLMProvider


class _FakeProvider(LLMProvider):
    name = "fake"

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        if "final answer" in prompt.lower():
            return "Processing needs a lawful basis [article-6]."
        return "Article 6 is the relevant lawful-basis clause."


class _FakeIndex:
    def search(self, query: str, top_k: int):
        return [
            VectorHit(
                id="article-6",
                title="Article 6",
                text="Processing is lawful with a valid basis.",
                score=0.1,
            )
        ]


def test_single_agent_returns_reasoning_trace():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="Article 6", kind="clause")
    settings = Settings(_env_file=None, agent_mode="single", use_graph=True)
    agent = RegulatoryAgent(_FakeProvider(), _FakeIndex(), graph, settings)

    trace = agent.answer("What makes processing lawful?")

    assert trace.answer == "Processing needs a lawful basis [article-6]."
    assert [step.name for step in trace.steps] == ["retrieve", "reason", "answer"]
    assert "article-6" in trace.to_json()
