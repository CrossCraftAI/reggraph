import json

from agentic_reg.trace import ReasoningTrace


def test_trace_records_steps_and_serialises():
    trace = ReasoningTrace(question="What makes processing lawful?")
    trace.add_step("retrieve", "found 2 clauses", vector_hits=[{"id": "article-6"}])
    trace.add_step("answer", "wrote answer")
    trace.answer = "Processing is lawful under [article-6]."

    data = trace.to_dict()
    assert data["question"] == "What makes processing lawful?"
    assert data["answer"] == "Processing is lawful under [article-6]."
    assert [step["name"] for step in data["steps"]] == ["retrieve", "answer"]
    assert data["steps"][0]["data"]["vector_hits"][0]["id"] == "article-6"

    # round-trips through JSON without error
    assert json.loads(trace.to_json())["answer"].startswith("Processing")
