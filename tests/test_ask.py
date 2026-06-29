from types import SimpleNamespace

from agentic_reg import ask as ask_module
from agentic_reg.domains import Domain
from agentic_reg.trace import ReasoningTrace


class _FakeOrchestrator:
    def __init__(self, response: str) -> None:
        self.response = response

    def answer(self, question: str) -> ReasoningTrace:
        trace = ReasoningTrace(question=question)
        trace.answer = self.response
        trace.add_step("answer", "answered", answer=self.response)
        return trace


def test_answer_delegates_to_configured_orchestrator(monkeypatch, tmp_path):
    domain = Domain(
        name="uk_dpa",
        title="UK DPA",
        description="x",
        source_path=tmp_path / "source.md",
        unit_label="section",
    )
    settings = SimpleNamespace(domain="gdpr", agent_mode="team", embedding_model="fake")
    calls = {}

    monkeypatch.setattr(ask_module, "get_settings", lambda: settings)
    monkeypatch.setattr(ask_module, "get_domain", lambda name: domain)
    monkeypatch.setattr(ask_module.KnowledgeGraph, "load", lambda path: "graph")
    monkeypatch.setattr(ask_module.VectorIndex, "load", lambda path, model: "vector")
    monkeypatch.setattr(ask_module, "get_provider", lambda loaded_settings: "provider")

    def fake_get_orchestrator(loaded_settings, provider, vector_index, graph):
        calls["settings"] = loaded_settings
        calls["provider"] = provider
        calls["vector_index"] = vector_index
        calls["graph"] = graph
        return _FakeOrchestrator("Notify within 72 hours [section-67].")

    monkeypatch.setattr(ask_module, "get_orchestrator", fake_get_orchestrator)

    trace = ask_module.answer("When notify?", domain_name="uk_dpa", mode="single")

    assert trace.answer == "Notify within 72 hours [section-67]."
    assert settings.domain == "uk_dpa"
    assert settings.agent_mode == "single"
    assert calls == {
        "settings": settings,
        "provider": "provider",
        "vector_index": "vector",
        "graph": "graph",
    }


def test_main_prints_answer_and_trace(monkeypatch, capsys):
    trace = ReasoningTrace(question="Q?")
    trace.answer = "Answer [section-67]."
    trace.add_step("answer", "answered", answer=trace.answer)
    monkeypatch.setattr(ask_module, "answer", lambda *args, **kwargs: trace)

    ask_module.main(["--domain", "uk_dpa", "Q?"])

    output = capsys.readouterr().out
    assert "Answer [section-67]." in output
    assert "--- trace ---" in output
    assert '"question": "Q?"' in output


def test_main_trace_only(monkeypatch, capsys):
    trace = ReasoningTrace(question="Q?")
    trace.answer = "Answer [article-6]."
    monkeypatch.setattr(ask_module, "answer", lambda *args, **kwargs: trace)

    ask_module.main(["--trace-only", "Q?"])

    output = capsys.readouterr().out
    assert output.lstrip().startswith("{")
    assert "--- trace ---" not in output
