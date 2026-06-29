import json

from agentic_reg.agent import RegulatoryAgent
from agentic_reg.agents import Orchestrator, get_orchestrator
from agentic_reg.agents.team import RegulatoryTeam
from agentic_reg.config import Settings
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorHit
from agentic_reg.providers.base import LLMProvider


class _FakeIndex:
    def search(self, query, top_k):
        return [VectorHit(id="article-6", title="Article 6", text="Lawful bases.", score=0.1)]


def _graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="Article 6", kind="clause")
    graph.add_node("article-7", label="Article 7", kind="clause")
    return graph


def _settings() -> Settings:
    return Settings(_env_file=None, max_revisions=1, max_subquestions=3)


class _ScriptedProvider(LLMProvider):
    name = "scripted"

    def complete(self, prompt, *, system=None, temperature=0.0):
        if "A verifier found these problems" in prompt:
            return "Revised: processing on consent is lawful [article-6]."
        if "Check this draft answer" in prompt:
            return '{"unsupported_claims": [], "contradictions": []}'
        if "Propose up to 3 graph updates" in prompt:
            return '{"proposals": []}'
        if "Break this into" in prompt:
            return (
                '{"sub_questions": [{"question": "q1", "specialist": "clause_analyst"}, '
                '{"question": "q2", "specialist": "cross_reference"}]}'
            )
        if "Sub-question:" in prompt:
            return "Per [article-6], consent is a lawful basis."
        return "Lawful per [article-6] and [article-99]."


def test_get_orchestrator_selects_single_and_team_modes():
    single_graph = _graph()
    single: Orchestrator = get_orchestrator(
        Settings(_env_file=None, agent_mode="single"),
        _ScriptedProvider(),
        _FakeIndex(),
        single_graph,
    )
    assert isinstance(single, RegulatoryAgent)
    assert single.graph is single_graph

    team_graph = _graph()
    team: Orchestrator = get_orchestrator(
        Settings(_env_file=None, agent_mode="team"),
        _ScriptedProvider(),
        _FakeIndex(),
        team_graph,
    )
    assert isinstance(team, RegulatoryTeam)
    assert team.graph is team_graph


def test_team_runs_full_pipeline_with_one_self_correction():
    team = RegulatoryTeam(_ScriptedProvider(), _FakeIndex(), _graph(), _settings())
    trace = team.answer("Can consent make processing lawful?")

    assert [step.name for step in trace.steps] == [
        "plan",
        "specialist:clause_analyst",
        "specialist:cross_reference",
        "task_tree",
        "synthesize",
        "graph_updates",
        "verify",
        "revise",
        "verify",
    ]
    assert trace.answer == "Revised: processing on consent is lawful [article-6]."
    json.loads(trace.to_json())


def test_team_finalizes_without_revision_when_clean():
    class _CleanProvider(LLMProvider):
        name = "clean"

        def complete(self, prompt, *, system=None, temperature=0.0):
            if "Check this draft answer" in prompt:
                return '{"unsupported_claims": [], "contradictions": []}'
            if "Propose up to 3 graph updates" in prompt:
                return '{"proposals": []}'
            if "Break this into" in prompt:
                return '{"sub_questions": [{"question": "q", "specialist": "clause_analyst"}]}'
            if "Sub-question:" in prompt:
                return "Per [article-6], yes."
            return "Processing on consent is lawful [article-6]."

    team = RegulatoryTeam(_CleanProvider(), _FakeIndex(), _graph(), _settings())
    trace = team.answer("Can consent make processing lawful?")

    assert "revise" not in [step.name for step in trace.steps]
    assert trace.answer == "Processing on consent is lawful [article-6]."


def test_team_can_spawn_bounded_recursive_tasks():
    class _RecursiveProvider(LLMProvider):
        name = "recursive"

        def complete(self, prompt, *, system=None, temperature=0.0):
            if "Question: complex branch" in prompt and "Break this into" in prompt:
                return (
                    '{"sub_questions": ['
                    '{"question": "leaf a", "specialist": "clause_analyst"}, '
                    '{"question": "leaf b", "specialist": "cross_reference"}]}'
                )
            if "Break this into" in prompt:
                return (
                    '{"sub_questions": [{"question": "complex branch", '
                    '"specialist": "supervisor"}]}'
                )
            if "Sub-question:" in prompt:
                return "Supported by [article-6]."
            return "Final answer [article-6]."

    settings = Settings(
        _env_file=None,
        max_subquestions=3,
        max_agent_depth=2,
        max_agent_tasks=4,
    )
    team = RegulatoryTeam(_RecursiveProvider(), _FakeIndex(), _graph(), settings)
    trace = team.answer("Needs hierarchy")

    task_tree = next(step for step in trace.steps if step.name == "task_tree")
    tasks = task_tree.data["tasks"]
    supervisor = next(task for task in tasks if task["text"] == "complex branch")

    assert supervisor["role"] == "supervisor"
    assert supervisor["children"] == ["task-2", "task-3"]
    assert [task["text"] for task in tasks if task["status"] == "answered"] == [
        "leaf a",
        "leaf b",
    ]


def test_task_cap_forces_supervisor_to_leaf_work():
    class _CapProvider(LLMProvider):
        name = "cap"

        def complete(self, prompt, *, system=None, temperature=0.0):
            if "Break this into" in prompt:
                return '{"sub_questions": [{"question": "too broad", "specialist": "supervisor"}]}'
            if "Sub-question:" in prompt:
                return "Handled as a leaf [article-6]."
            return "Final answer [article-6]."

    settings = Settings(
        _env_file=None,
        max_subquestions=3,
        max_agent_depth=3,
        max_agent_tasks=1,
    )
    team = RegulatoryTeam(_CapProvider(), _FakeIndex(), _graph(), settings)
    trace = team.answer("Cap this")
    tasks = next(step for step in trace.steps if step.name == "task_tree").data["tasks"]
    non_root = [task for task in tasks if task["parent_id"] is not None]

    assert len(non_root) == 1
    assert non_root[0]["role"] == "clause_analyst"
    assert non_root[0]["status"] == "answered"


def test_graph_update_mode_off_skips_proposals(tmp_path):
    class _NoProposalProvider(LLMProvider):
        name = "no-proposal"

        def complete(self, prompt, *, system=None, temperature=0.0):
            assert "Propose up to 3 graph updates" not in prompt
            if "Check this draft answer" in prompt:
                return '{"unsupported_claims": [], "contradictions": []}'
            if "Break this into" in prompt:
                return '{"sub_questions": [{"question": "q", "specialist": "clause_analyst"}]}'
            if "Sub-question:" in prompt:
                return "Supported [article-6]."
            return "Final answer [article-6]."

    settings = Settings(
        _env_file=None,
        graph_update_mode="off",
        graph_proposals_path=str(tmp_path / "proposals.jsonl"),
    )
    team = RegulatoryTeam(_NoProposalProvider(), _FakeIndex(), _graph(), settings)
    trace = team.answer("Skip proposals")
    graph_updates = next(step for step in trace.steps if step.name == "graph_updates")

    assert graph_updates.data["proposals"] == []
    assert not (tmp_path / "proposals.jsonl").exists()


def test_graph_update_mode_propose_reviews_without_mutating_graph(tmp_path):
    class _ProposalProvider(LLMProvider):
        name = "proposal"

        def complete(self, prompt, *, system=None, temperature=0.0):
            if "Check this draft answer" in prompt:
                return '{"unsupported_claims": [], "contradictions": []}'
            if "Propose up to 3 graph updates" in prompt:
                return (
                    '{"proposals": [{"action": "edge", "source_id": "article-6", '
                    '"target_id": "article-7", "relation": "depends_on", '
                    '"evidence": "Article 6 depends on Article 7.", '
                    '"citations": ["article-6", "article-7"]}]}'
                )
            if "Break this into" in prompt:
                return '{"sub_questions": [{"question": "q", "specialist": "clause_analyst"}]}'
            if "Sub-question:" in prompt:
                return "Supported [article-6] [article-7]."
            return "Final answer [article-6] [article-7]."

    graph = _graph()
    settings = Settings(
        _env_file=None,
        graph_update_mode="propose",
        graph_proposals_path=str(tmp_path / "proposals.jsonl"),
    )
    team = RegulatoryTeam(_ProposalProvider(), _FakeIndex(), graph, settings)
    trace = team.answer("Review proposal")
    graph_updates = next(step for step in trace.steps if step.name == "graph_updates")

    assert not graph.has_edge("article-6", "article-7", "depends_on")
    assert graph_updates.data["proposals"][0]["status"] == "accepted"
    assert (tmp_path / "proposals.jsonl").exists()
