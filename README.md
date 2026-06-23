<div align="center">
  <h1>⚖️ RegGraph</h1>
  <p><strong>Auditable multi-agent reasoning over regulatory knowledge graphs</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
  [![CI](https://img.shields.io/badge/CI-passing-brightgreen)](.github/workflows/ci.yml)
</div>

---

## Why RegGraph?

Most regulatory RAG is **flat**: vector search → dump context → ask LLM → hope.  
It loses the cross-references, conditions, and exceptions that *are* the regulation.

RegGraph keeps that structure. It builds a **typed knowledge graph** from the regulation
itself — articles, obligations, rights, conditions, prohibitions — and reasons over the
**paths between clauses**, not just the clauses in isolation.

Every citation is checked against the real regulation graph. Hallucinated article
references are caught **deterministically, with zero LLM cost**.
High-confidence symbolic checks cover recurring regulatory patterns such as
special-category data, withdrawal and erasure, and breach notification deadlines.

| | Plain RAG | GraphRAG | RegGraph |
|---|---|---|---|
| **Semantic search** | ✓ | ✓ | ✓ |
| **Graph expansion** | ✗ | ✓ | ✓ |
| **Typed regulatory concepts** | ✗ | ✗ | ✓ |
| **Multi-hop clause paths** | ✗ | partial | ✓ |
| **Deterministic citation check** | ✗ | ✗ | ✓ |
| **Multi-agent verification** | ✗ | ✗ | ✓ |
| **Pluggable domains** | ✗ | ✗ | ✓ |

## Architecture

```
document → chunks → knowledge store → hybrid retrieval → agent team → answer + trace

                      ├─ vectors (Chroma)        ├─ vector search
                      └─ typed graph (NetworkX)   ├─ graph expansion
                                                    └─ multi-hop clause paths

                          ┌─ supervisor (decomposes)
agent team ───────────────┼─ specialists (clause & cross-reference analysts)
  (LangGraph)             ├─ synthesis
                          ├─ verification (deterministic + LLM)
                          └─ self-correction (bounded, one pass)
```

## Quick Start

```bash
git clone https://github.com/CrossCraftAI/reggraph.git && cd reggraph
uv sync
uv run python -m agentic_reg.build --domain gdpr --no-enrich
uv run streamlit run app.py
```

No API keys. No config. `--no-enrich` builds a deterministic graph without an LLM.

For LLM-backed reasoning, log into GitHub (free tier):

```bash
gh auth login
uv run python -m agentic_reg.build --domain gdpr   # adds typed concept enrichment
```

## Features

- **Typed regulatory knowledge graph** — clauses + concepts (obligation, right, condition, prohibition, …) with typed relations (requires, overrides, depends_on, exception_to, …)
- **Multi-hop reasoning chains** — clause-to-clause paths across the regulation, not isolated matches
- **Deterministic hallucination detection** — every `[article-N]` citation verified against the real graph, zero LLM cost
- **Symbolic verification** — lightweight rules catch missing lawful-basis, erasure, and breach-deadline support
- **Multi-agent team** — supervisor decomposes questions → specialists research → synthesis → verifier checks → self-corrects if needed
- **Full audit trail** — every step exported as structured JSON, every claim traceable to a real clause
- **Pluggable domains** — GDPR and UK DPA 2018 ship built-in; add a new regulation with one markdown file, no code changes
- **Evaluation harness** — citation F1, hallucination rate, multi-hop recall, LLM-judge quality scores
- **Vector-only ablation** — flip `AGENTIC_REG_USE_GRAPH=false` to isolate the graph's contribution

## How It Works

<table>
<tr><td width="50%">

**Build a domain's knowledge store**  
Ingest a regulation → chunk into clauses → extract cross-references + typed concepts → build graph + vector index.

```bash
uv run python -m agentic_reg.build --domain gdpr
uv run python -m agentic_reg.build --domain uk_dpa
```

</td><td width="50%">

**Ask a question**  
Choose single-agent or multi-agent team mode, type a question, and get a grounded answer with citations.

```bash
uv run streamlit run app.py
# → open http://localhost:8501
```

</td></tr>
<tr><td>

**See the reasoning trace**  
Every step — retrieval, specialist analysis, synthesis, verification verdict, self-correction — visible in the UI and exportable as JSON.

</td><td>

**Run evaluation**  
Compare architectures with the eval harness — same model, same cases, varying agent mode and graph usage.

```bash
uv run python -m agentic_reg.eval --configs single,team --limit 4
```

</td></tr>
</table>

## Usage

### Python API

```python
from agentic_reg.agents import get_orchestrator
from agentic_reg.config import get_settings
from agentic_reg.domains import get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorIndex
from agentic_reg.providers import get_provider

settings = get_settings()
domain = get_domain("gdpr")
vector_index = VectorIndex(domain.chroma_dir, settings.embedding_model)
graph = KnowledgeGraph.load(domain.graph_path)
provider = get_provider(settings)

orchestrator = get_orchestrator(settings, provider, vector_index, graph)
trace = orchestrator.answer("What lawful bases allow processing of special category data?")

print(trace.answer)
print(trace.to_json())
```

### LLM Backends

| Backend | Config | Auth |
|---------|--------|------|
| **GitHub Models** (default) | `AGENTIC_REG_LLM_PROVIDER=github` | `gh auth login` or `GITHUB_TOKEN` |
| **Ollama** (local) | `AGENTIC_REG_LLM_PROVIDER=ollama` | None |
| **Claude API** (paid) | `AGENTIC_REG_LLM_PROVIDER=anthropic` | `ANTHROPIC_API_KEY` |

## Plug in Your Own Domain

Add a regulation with one markdown file (sections as `## <Unit> N — Title`):

```python
from agentic_reg.domains import Domain, register
from agentic_reg.config import PROJECT_ROOT

register(Domain(
    name="my_regulation",
    title="My Regulation",
    description="A custom regulatory document.",
    source_path=PROJECT_ROOT / "data" / "my_regulation.md",
    unit_label="section",
))
```

Then `uv run python -m agentic_reg.build --domain my_regulation`.

External packages can ship domains via entry points — they auto-register on install.

## Evaluation

```bash
# Compare single agent vs multi-agent team (model held constant)
uv run python -m agentic_reg.eval --configs single,team

# Quick ablation: measure the graph's contribution
uv run python -m agentic_reg.eval --configs team,team-no-graph --limit 4
```

**Deterministic metrics** (citation F1, hallucination rate, multi-hop recall) — no LLM needed, fully reproducible.

**LLM-judge** — holistic answer quality grading, excluded from the mean on failure.

## Comparison

| Framework | Multi-agent | Graph-based | Regulatory focus | Deterministic verification | Pluggable domains |
|-----------|:-----------:|:-----------:|:----------------:|:--------------------------:|:-----------------:|
| **RegGraph** | ✓ | ✓ | ✓ | ✓ | ✓ |
| CrewAI | ✓ | ✗ | ✗ | ✗ | ✗ |
| LangGraph | ✓ | ✓ | ✗ | ✗ | ✗ |
| Microsoft GraphRAG | ✗ | ✓ | ✗ | ✗ | ✗ |
| Haystack | partial | ✗ | ✗ | ✗ | ✗ |

## Contributing

Issues and PRs welcome.

```bash
uv sync
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # tests (model-free, fast)
```

## License

[MIT](LICENSE) © 2026 CrossCraftAI
