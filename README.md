<div align="center">
  <h1>⚖️ RegGraph</h1>
  <p><strong>Auditable multi-agent reasoning over regulatory knowledge graphs</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
  [![CI](https://github.com/CrossCraftAI/reggraph/actions/workflows/ci.yml/badge.svg)](https://github.com/CrossCraftAI/reggraph/actions/workflows/ci.yml)
</div>

---

## Why RegGraph?

Legal expertise is bounded by jurisdiction — most practitioners know *their* regulation
cold but can't reason across borders. Meanwhile, startups scale at exponential pace,
their data sloshing across GDPR, CCPA, UK DPA, PIPEDA with no systematic mapping
between them. The result: either over-compliance (wasted effort) or under-compliance
(regulatory risk).

RegGraph models regulations as **comparable typed graphs** and traces reasoning chains
across them. Give it a question about one regulation; it follows cross-references into
related jurisdictions, showing equivalences, gaps, and conflicts. This is what a legal
expert does when they say "under GDPR this requires article-9 consent… but CCPA has no
special-category concept, so you need to close that gap contractually."

Under the hood, three primitives work together:

1. **Typed graph construction** — regulation markdown → NetworkX graph with
   obligations, rights, conditions, prohibitions, and cross-references. No LLM needed
   (`--no-enrich`).
2. **Deterministic citation verification** — every `[article-N]` reference checked
   against the real graph. Hallucinated citations caught with zero LLM cost.
3. **Symbolic compliance rules** — lightweight checks for recurring regulatory patterns
   (special-category → must-cite-lawful-basis, breach → must-cite-72h-deadline).

These aren't the product — they're the enablers. The product is **cross-jurisdictional
reasoning chains** that show you what applies, what's missing, and why.

| | Plain RAG | GraphRAG | RegGraph |
|---|---|---|---|
| **Semantic search** | ✓ | ✓ | ✓ |
| **Graph expansion** | ✗ | ✓ | ✓ |
| **Typed regulatory concepts** | ✗ | ✗ | ✓ |
| **Multi-hop clause paths** | ✗ | partial | ✓ |
| **Deterministic citation check** | ✗ | ✗ | ✓ |
| **Multi-agent verification** | ✗ | ✗ | ✓ |
| **Cross-jurisdictional mapping** | ✗ | ✗ | ✓ |
| **Pluggable domains** | ✗ | ✗ | ✓ |

## Architecture

```
regulation A ──→ typed graph ──┐
regulation B ──→ typed graph ──┼──→ cross-jurisdictional reasoning chains
regulation C ──→ typed graph ──┘         │
                                         ├─ equivalence mapping (A.9 ≈ B.10)
                                         ├─ gap analysis (A requires X, C doesn't)
                                         └─ delta chains (A → add Y → C compliant)

                  ┌─ supervisor (jurisdiction-aware decomposition)
agent team ───────┼─ specialists (GDPR analyst, CCPA analyst, cross-ref analyst)
  (LangGraph)     ├─ synthesis (jurisdictional delta)
                  ├─ verification (deterministic citation check + symbolic rules)
                  └─ self-correction (bounded, one pass)
```

## Quick Start

> **⚠️ Work in Progress** — the core modules (providers, domains, graph, vectors, build
> CLI, agent loop, eval harness) are under active development. The Quick Start commands
> below will work once these land. Track progress in [TODO.md](TODO.md) and
> [issue #1](https://github.com/CrossCraftAI/reggraph/issues/1).

```bash
git clone https://github.com/CrossCraftAI/reggraph.git && cd reggraph
uv sync
uv run python -m agentic_reg.build --domain gdpr --no-enrich
uv run python -m agentic_reg.ask "What lawful bases allow processing of special category data?"
```

No API keys. No config. `--no-enrich` builds a deterministic graph without an LLM.

For LLM-backed reasoning, log into GitHub (free tier):

```bash
gh auth login
uv run python -m agentic_reg.build --domain gdpr   # adds typed concept enrichment
```

## Features

- **Cross-jurisdictional reasoning chains** — map obligations, rights, and prohibitions across regulations; trace equivalence, gap, and delta paths between GDPR, CCPA, UK DPA, and custom domains
- **Typed regulatory knowledge graph** — clauses + concepts (obligation, right, condition, prohibition, …) with typed relations (requires, overrides, depends_on, exception_to, equivalent_to, conflicts_with, extends)
- **Multi-hop reasoning chains** — clause-to-clause paths across the regulation, not isolated matches
- **Deterministic citation verification** — every `[article-N]` citation verified against the real graph, zero LLM cost. No external API, no database lookup, no NLI model
- **Symbolic verification** — lightweight rules catch missing lawful-basis, erasure, and breach-deadline support. Internally enables cross-jurisdictional compliance checks
- **Multi-agent team** — supervisor decomposes questions → jurisdiction-aware specialists → synthesis with delta analysis → verifier checks → self-corrects if needed
- **Full audit trail** — every step exported as structured JSON, every claim traceable to a real clause in a real regulation
- **Pluggable domains** — GDPR and UK DPA 2018 ship built-in; add a new regulation with one markdown file, no code changes
- **Evaluation harness** — citation F1, hallucination rate, multi-hop recall, cross-jurisdictional reasoning chain recall, LLM-judge quality scores
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
Compare RegGraph against a vanilla LangGraph baseline — same model, same index, same cases.

```bash
uv run python -m agentic_reg.eval --configs reggraph,langgraph --limit 4
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
gdpr = get_domain("gdpr")
ccpa = get_domain("ccpa")  # drop in your regulation markdown

# Build or load graphs — each regulation is a typed graph
gdpr_graph = KnowledgeGraph.load(gdpr.graph_path)
ccpa_graph = KnowledgeGraph.load(ccpa.graph_path)

provider = get_provider(settings)

# Cross-jurisdictional reasoning: GDPR → CCPA mapping
orchestrator = get_orchestrator(settings, provider, gdpr_graph, ccpa_graph)
trace = orchestrator.answer(
    "Does my GDPR article-9 health data consent cover CCPA's requirements?"
)

print(trace.answer)       # "No. CCPA has no special-category concept.
                          #  Your GDPR consent satisfies article-9 but
                          #  you need addt'l contractual safeguards for
                          #  California residents because…"
print(trace.reasoning_chain)  # GDPR.article-9 → equivalent_to? none
                              # → gap: CCPA has no special-category
                              # → delta: add contractual safeguards
print(trace.to_json())    # full audit trail across both jurisdictions
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

The eval harness compares RegGraph against a **vanilla LangGraph + ChromaDB
baseline** — the same LLM, same vector index, same test cases. The only variable
is the regulatory-specific architecture RegGraph adds on top.

```bash
# RegGraph vs vanilla LangGraph baseline (the fair comparison)
uv run python -m agentic_reg.eval --configs reggraph,langgraph

# Quick ablation: measure the graph's contribution
uv run python -m agentic_reg.eval --configs reggraph,no-graph --limit 4

# All three: full pipeline, LangGraph baseline, and vector-only
uv run python -m agentic_reg.eval --configs reggraph,langgraph,no-graph

# With LLM-judge for holistic quality scoring
uv run python -m agentic_reg.eval --configs reggraph,langgraph --judge

# Live mode: real LLM calls (incurs API cost)
uv run python -m agentic_reg.eval --configs reggraph,langgraph --live --limit 4
```

**Deterministic metrics** (citation F1, hallucination rate, multi-hop recall,
symbolic check pass rate) — no LLM needed, fully reproducible. Run without
`--live` for instant, cost-free metric validation.

**Runtime metrics** (wall-clock time, token usage) — captured in `--live` mode.

**LLM-judge** — holistic answer quality grading across correctness, completeness,
citation quality, and clarity.

## Benchmark & Competitive Landscape

There is no widely-adopted open-source framework that does cross-jurisdictional
regulatory reasoning. The closest tools fall into three buckets, none of which
solve the same problem:

| Category | Examples | Stars | What they do | What they don't do |
|----------|----------|------:|--------------|-------------------|
| **General GraphRAG** | Microsoft GraphRAG, LightRAG | 9k–30k | Community-summary graphs over documents | No regulatory concepts, no cross-jurisdictional mapping, no agents |
| **Agent frameworks** | LangGraph, CrewAI | 10k–25k | Multi-agent orchestration | No typed regulatory graph, no citation verification, no domain model |
| **Fairness/privacy libs** | AIF360, Fairlearn, Google DP | 2k–3k | Bias metrics, fairness constraints, DP statistics | Point-solution libraries — complementary, not comparable |
| **Legal citation tools** | eyecite, CiteTracer | 0–250 | Extract citations from text, verify against Crossref | US case law only, no regulation graph, external API dependency |
| **Compliance engines** | CDISC CORE, ARKA, OSCAL | 50–200 | Conformance checking against specific standards | Vertical-specific, no cross-jurisdictional reasoning, no LLM agent layer |

### What makes this defensible

Three things compound into a moat:

1. **The graph is the source of truth, not an external API.** Citation verification
   is a set-membership check — `graph.has_node("article-9")` — with zero latency,
   zero cost, and zero external dependency. Every other citation verifier calls
   Crossref, Semantic Scholar, or a database. Ours is deterministic.

2. **Regulations are modeled, not searched.** The typed graph captures obligations,
   rights, conditions, prohibitions, and cross-references. This makes
   cross-jurisdictional reasoning possible — you can ask "does CCPA have an
   equivalent to GDPR article-9?" and get a reasoning chain, not a vector-similarity
   guess.

3. **Domains are pluggable packages.** `pip install reggraph-hipaa` adds a new
   regulation's typed graph. Inter-domain edges (equivalent_to, conflicts_with,
   extends) form automatically from the concept layer. Each new domain enriches
   every other domain's reasoning chains.

### The LangGraph baseline

The eval harness compares RegGraph against a vanilla LangGraph + ChromaDB
baseline — same LLM, same vector index, same test cases. LangGraph is what
every competent team builds on; the difference is purely architectural:

| Capability | LangGraph + ChromaDB | RegGraph |
|-----------|:--------------------:|:--------:|
| Multi-agent orchestration | ✓ | ✓ |
| Semantic vector search | ✓ | ✓ |
| Typed regulatory concepts | ✗ | ✓ |
| Multi-hop clause-path traversal | ✗ | ✓ |
| Deterministic citation verification | ✗ | ✓ |
| Symbolic regulatory checks | ✗ | ✓ |
| Cross-jurisdictional reasoning chains | ✗ | ✓ |
| Pluggable regulatory domains | ✗ | ✓ |

### Benchmark metrics

| Metric | What it captures | Why it matters |
|--------|-----------------|----------------|
| **Citation F1** | Precision + recall of article references | Are citations correct and complete? |
| **Hallucination rate** | Cited articles that don't exist in the regulation | Does the answer fabricate legal references? |
| **Multi-hop recall** | Connected articles found beyond direct matches | Does reasoning follow cross-references? |
| **Cross-jurisdictional recall** | Equivalent articles found across regulation graphs | Does reasoning map between jurisdictions? |
| **Symbolic pass rate** | High-confidence rule checks (special-category, erasure, breach deadlines) | Are hard regulatory requirements met? |
| **LLM-judge score** | Holistic quality: correctness, completeness, citation quality, clarity | Is the answer practically usable? |

### Running the benchmark

```bash
# Single-jurisdiction comparison
uv run python -m agentic_reg.eval --configs reggraph,langgraph --limit 5

# Cross-jurisdictional: GDPR ↔ UK DPA mapping
uv run python -m agentic_reg.eval --configs reggraph,langgraph \
    --scenario cross-jurisdictional --limit 4

# Live mode with LLM judge
uv run python -m agentic_reg.eval --configs reggraph --live --judge \
    --scenario cross-jurisdictional
```

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
