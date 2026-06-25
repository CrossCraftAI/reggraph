from agentic_reg.agents import RegulatoryTeam, get_orchestrator
from agentic_reg.agent import RegulatoryAgent
from agentic_reg.config import Settings
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorHit
from agentic_reg.providers.base import LLMProvider


class _FakeProvider(LLMProvider):
    name = "fake"

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        if "Specialist role" in prompt:
            return "Article 6 supplies lawful bases [article-6]."
        return "Processing needs a lawful basis [article-6]."


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


def _graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="Article 6", kind="clause")
    return graph


def test_get_orchestrator_selects_mode():
    single_settings = Settings(_env_file=None, agent_mode="single")
    team_settings = Settings(_env_file=None, agent_mode="team")

    assert isinstance(
        get_orchestrator(single_settings, _FakeProvider(), _FakeIndex(), _graph()),
        RegulatoryAgent,
    )
    assert isinstance(
        get_orchestrator(team_settings, _FakeProvider(), _FakeIndex(), _graph()),
        RegulatoryTeam,
    )


def test_basic_team_returns_trace():
    settings = Settings(_env_file=None, agent_mode="team", max_subquestions=2)
    team = RegulatoryTeam(_FakeProvider(), _FakeIndex(), _graph(), settings)

    trace = team.answer("What makes processing lawful?")

    assert trace.answer == "Processing needs a lawful basis [article-6]."
    assert [step.name for step in trace.steps] == [
        "plan",
        "specialist:clause_analyst",
        "specialist:cross_reference",
        "synthesize",
    ]
    assert "article-6" in trace.to_json()
