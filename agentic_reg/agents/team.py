"""Hierarchical multi-agent team."""

from dataclasses import asdict

from langgraph.graph import END, START, StateGraph

from .._internal import parse_json_object
from ..config import Settings
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.vectors import VectorIndex
from ..providers.base import LLMProvider
from ..trace import ReasoningTrace
from .specialists import DEFAULT_ROLE, SPECIALISTS, display_name, run_specialist
from .state import AgentTask, Finding, SubQuestion, TeamState

SUPERVISOR_ROLE = "supervisor"
ALLOWED_ROLES = {SUPERVISOR_ROLE, *SPECIALISTS.keys()}

_DECOMPOSE_SYSTEM = "You are a supervisor that plans regulatory analysis. Respond with JSON only."

_DECOMPOSE_PROMPT = """Question: {question}

Break this into 1-{max_sub} focused sub-questions. Use "supervisor" only when a
sub-question still needs decomposition; otherwise choose the best specialist:
- "clause_analyst": interpreting what specific clauses require or permit.
- "cross_reference": tracing dependencies, conditions, and exceptions across clauses.
- "graph_curator": identifying missing graph entities or relationships.

Return ONLY JSON:
{{"sub_questions": [{{"question": "...",
 "specialist": "supervisor|clause_analyst|cross_reference|graph_curator"}}]}}"""

_SYNTH_SYSTEM = (
    "You are a careful regulatory analyst. Write one answer grounded only in the "
    "specialist findings. Cite clause ids in square brackets."
)

_SYNTH_PROMPT = """Question: {question}

Specialist findings:
{findings}

Write the answer to the question. Ground every claim in the findings and cite
clause ids in [brackets]. Be concise."""


def _findings_block(findings: list[Finding]) -> str:
    return "\n".join(f"- ({display_name(f.role)}) {f.text}" for f in findings) or "(none)"


class RegulatoryTeam:
    """Plan recursively, dispatch specialists, and synthesize findings."""

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
        builder = StateGraph(TeamState)
        builder.add_node("plan", self._plan)
        builder.add_node("dispatch", self._dispatch)
        builder.add_node("synthesize", self._synthesize)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "dispatch")
        builder.add_edge("dispatch", "synthesize")
        builder.add_edge("synthesize", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    def _plan(self, state: TeamState) -> dict:
        raw = self.provider.complete(
            _DECOMPOSE_PROMPT.format(
                question=state["question"], max_sub=self.settings.max_subquestions
            ),
            system=_DECOMPOSE_SYSTEM,
        )
        plan = self._parse_plan(raw, state["question"])
        state["trace"].add_step(
            "plan",
            f"Supervisor decomposed the question into {len(plan)} top-level task(s).",
            sub_questions=[
                {"question": sq.text, "specialist": display_name(sq.role)} for sq in plan
            ],
        )
        return {"plan": plan}

    def _parse_plan(self, raw: str, question: str, *, fallback: bool = True) -> list[SubQuestion]:
        data = parse_json_object(raw)
        plan: list[SubQuestion] = []
        for item in data.get("sub_questions", []) or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("question", "")).strip()
            if not text:
                continue
            role = str(item.get("specialist") or item.get("role") or "").strip().lower()
            if role not in ALLOWED_ROLES:
                role = DEFAULT_ROLE
            plan.append(SubQuestion(text=text, role=role))
            if len(plan) >= self.settings.max_subquestions:
                break
        if not plan and fallback:
            plan = [SubQuestion(text=question, role=DEFAULT_ROLE)]
        return plan

    def _dispatch(self, state: TeamState) -> dict:
        tasks = [
            AgentTask(
                id="task-0",
                text=state["question"],
                role=SUPERVISOR_ROLE,
                parent_id=None,
                depth=0,
                status="spawned",
            )
        ]
        findings: list[Finding] = []

        for sub in state["plan"]:
            child_id = self._run_task(
                sub.text,
                sub.role,
                parent_id="task-0",
                depth=1,
                tasks=tasks,
                findings=findings,
                trace=state["trace"],
            )
            if child_id:
                tasks[0].children.append(child_id)

        if not findings:
            self._run_task(
                state["question"],
                DEFAULT_ROLE,
                parent_id="task-0",
                depth=1,
                tasks=tasks,
                findings=findings,
                trace=state["trace"],
                ignore_cap=True,
            )

        state["trace"].add_step(
            "task_tree",
            f"Executed {len(tasks) - 1} bounded agent task(s).",
            tasks=[asdict(task) for task in tasks],
        )
        return {"tasks": tasks, "findings": findings}

    def _run_task(
        self,
        text: str,
        role: str,
        *,
        parent_id: str,
        depth: int,
        tasks: list[AgentTask],
        findings: list[Finding],
        trace: ReasoningTrace,
        ignore_cap: bool = False,
    ) -> str | None:
        if not ignore_cap and self._task_limit_reached(tasks):
            return None

        task = AgentTask(
            id=f"task-{len(tasks)}",
            text=text,
            role=role if role in ALLOWED_ROLES else DEFAULT_ROLE,
            parent_id=parent_id,
            depth=depth,
        )
        tasks.append(task)

        if task.role == SUPERVISOR_ROLE and depth < self.settings.max_agent_depth:
            raw = self.provider.complete(
                _DECOMPOSE_PROMPT.format(question=text, max_sub=self.settings.max_subquestions),
                system=_DECOMPOSE_SYSTEM,
            )
            children = self._parse_plan(raw, text, fallback=False)
            if children:
                task.status = "spawned"
                spawned_child = False
                for child in children:
                    child_id = self._run_task(
                        child.text,
                        child.role,
                        parent_id=task.id,
                        depth=depth + 1,
                        tasks=tasks,
                        findings=findings,
                        trace=trace,
                    )
                    if child_id:
                        task.children.append(child_id)
                        spawned_child = True
                if spawned_child:
                    return task.id

        if task.role == SUPERVISOR_ROLE:
            task.role = DEFAULT_ROLE

        finding = run_specialist(
            task.role,
            text,
            self.provider,
            self.vector_index,
            self.graph,
            self.settings,
        )
        findings.append(finding)
        task.status = "answered"
        task.finding = finding.text
        task.citations = finding.citations
        task.retrieved_ids = finding.retrieved_ids
        task.graph_node_ids = finding.graph_node_ids
        trace.add_step(
            f"specialist:{task.role}",
            f"{display_name(task.role)} analysed task {task.id}: {text}",
            task_id=task.id,
            parent_id=task.parent_id,
            depth=task.depth,
            sub_question=text,
            finding=finding.text,
            citations=finding.citations,
            retrieved_ids=finding.retrieved_ids,
            graph_node_ids=finding.graph_node_ids,
            graph_edges=finding.graph_edges,
            multi_hop_paths=finding.multi_hop_paths,
        )
        return task.id

    def _task_limit_reached(self, tasks: list[AgentTask]) -> bool:
        spawned = sum(1 for task in tasks if task.parent_id is not None)
        return spawned >= max(self.settings.max_agent_tasks, 1)

    def _synthesize(self, state: TeamState) -> dict:
        draft = self.provider.complete(
            _SYNTH_PROMPT.format(
                question=state["question"], findings=_findings_block(state["findings"])
            ),
            system=_SYNTH_SYSTEM,
        )
        state["trace"].add_step(
            "synthesize", "Supervisor synthesized the findings into an answer.", draft=draft
        )
        return {"draft": draft}

    def _finalize(self, state: TeamState) -> dict:
        state["trace"].answer = state["draft"]
        return {}

    def answer(self, question: str) -> ReasoningTrace:
        trace = ReasoningTrace(question=question)
        self._app.invoke(
            {
                "question": question,
                "plan": [],
                "tasks": [],
                "findings": [],
                "draft": "",
                "trace": trace,
            }
        )
        return trace
