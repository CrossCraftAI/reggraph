from agentic_reg.agents.team import RegulatoryTeam
from agentic_reg.config import Settings
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.providers.base import LLMProvider


class _Provider(LLMProvider):
    name = "p"

    def complete(self, *args, **kwargs):
        return ""


def _team() -> RegulatoryTeam:
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="Article 6", kind="clause")
    settings = Settings(_env_file=None, max_subquestions=3)
    return RegulatoryTeam(_Provider(), object(), graph, settings)


def test_parse_plan_is_bounded_and_validates_roles():
    payload = (
        '{"sub_questions": ['
        '{"question": "a", "specialist": "cross_reference"}, '
        '{"question": "b", "specialist": "supervisor"}, '
        '{"question": "c", "specialist": "clause_analyst"}, '
        '{"question": "d", "specialist": "graph_curator"}]}'
    )
    plan = _team()._parse_plan(payload, "original question")
    assert len(plan) == 3
    assert plan[0].role == "cross_reference"
    assert plan[1].role == "supervisor"
    assert plan[2].role == "clause_analyst"


def test_parse_plan_downgrades_unknown_roles():
    payload = '{"sub_questions": [{"question": "a", "specialist": "bogus"}]}'
    plan = _team()._parse_plan(payload, "original question")
    assert plan[0].role == "clause_analyst"


def test_parse_plan_accepts_graph_curator_role():
    payload = '{"sub_questions": [{"question": "a", "specialist": "graph_curator"}]}'
    plan = _team()._parse_plan(payload, "original question")
    assert plan[0].role == "graph_curator"


def test_parse_plan_falls_back_on_bad_json():
    plan = _team()._parse_plan("not json at all", "original question")
    assert len(plan) == 1
    assert plan[0].text == "original question"
    assert plan[0].role == "clause_analyst"
