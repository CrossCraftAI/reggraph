"""The reasoning trace — a first-class, exportable record of how an answer was
reached.

Auditability is the whole point of agentic-reg, so every step the agent takes
appends a structured entry here. The trace is JSON-serialisable so it can be
shown in the UI, saved, or diffed.
"""

import json
from dataclasses import asdict, dataclass, field


@dataclass
class TraceStep:
    name: str  # short machine name, e.g. "retrieve"
    summary: str  # one-line human explanation of what happened
    data: dict  # structured details (retrieved clauses, reasoning text, ...)


@dataclass
class ReasoningTrace:
    question: str
    answer: str = ""
    steps: list[TraceStep] = field(default_factory=list)

    def add_step(self, name: str, summary: str, **data: object) -> None:
        self.steps.append(TraceStep(name=name, summary=summary, data=dict(data)))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
