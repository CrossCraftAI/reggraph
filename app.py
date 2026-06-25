"""Streamlit demo UI for RegGraph.

Run with:

    uv run streamlit run app.py
"""

import streamlit as st

from agentic_reg.agents import get_orchestrator
from agentic_reg.config import get_settings
from agentic_reg.domains import get_domain, list_domains
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorIndex
from agentic_reg.providers import get_provider

KIND_COLORS = {
    "clause": "#cfe8ff",
    "obligation": "#ffe9c7",
    "definition": "#d7f5d7",
    "right": "#f3d7f7",
    "condition": "#fff3c4",
    "principle": "#cdeff0",
    "prohibition": "#ffd0d0",
    "concept": "#ececec",
}

MODE_LABELS = {"team": "Multi-agent team", "single": "Single agent"}


st.set_page_config(page_title="RegGraph", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge store and model...")
def load_orchestrator(domain_name: str, mode: str):
    settings = get_settings()
    settings.domain = domain_name
    settings.agent_mode = mode
    domain = get_domain(domain_name)
    vector_index = VectorIndex(domain.chroma_dir, settings.embedding_model)
    graph = KnowledgeGraph.load(domain.graph_path)
    provider = get_provider(settings)
    return get_orchestrator(settings, provider, vector_index, graph), settings


def graph_dot(nodes: list[dict], edges: list[dict]) -> str:
    parts = ['digraph G { rankdir=LR; node [shape=box style="rounded,filled" fontsize=10];']
    for node in nodes:
        color = KIND_COLORS.get(node.get("kind", "concept"), "#ececec")
        label = node.get("label", node["id"]).replace('"', "'")
        parts.append(f'"{node["id"]}" [label="{label}" fillcolor="{color}"];')
    for edge in edges:
        parts.append(
            f'"{edge["source"]}" -> "{edge["target"]}" '
            f'[label="{edge["relation"]}" fontsize=8];'
        )
    parts.append("}")
    return "\n".join(parts)


st.title("RegGraph")
st.caption("Auditable regulatory reasoning over a graph + vector knowledge store")

settings = get_settings()
domains = list_domains()
domain_names = [domain.name for domain in domains]

with st.sidebar:
    st.subheader("Domain")
    default_domain = settings.domain if settings.domain in domain_names else domain_names[0]
    domain_name = st.selectbox(
        "Regulatory domain",
        domain_names,
        index=domain_names.index(default_domain),
        format_func=lambda name: get_domain(name).title,
    )
    st.caption(get_domain(domain_name).description)

    st.subheader("Mode")
    default_mode = settings.agent_mode if settings.agent_mode in MODE_LABELS else "team"
    mode = st.radio(
        "Agent mode",
        ["team", "single"],
        index=["team", "single"].index(default_mode),
        format_func=lambda value: MODE_LABELS[value],
    )

domain = get_domain(domain_name)
if not domain.graph_path.exists():
    st.warning(
        f"Knowledge store for '{domain.name}' was not found. Build it first:\n\n"
        f"uv run python -m agentic_reg.build --domain {domain.name} --no-enrich"
    )
    st.stop()

try:
    orchestrator, loaded_settings = load_orchestrator(domain_name, mode)
except Exception as exc:
    st.error(f"Could not load the selected provider or store: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("Configuration")
    model = {
        "github": loaded_settings.github_model,
        "ollama": loaded_settings.ollama_model,
        "anthropic": loaded_settings.anthropic_model,
    }.get(loaded_settings.llm_provider, "?")
    st.write(f"LLM provider: `{loaded_settings.llm_provider}`")
    st.write(f"Model: `{model}`")
    st.write(f"Graph: {orchestrator.graph.num_nodes} nodes / {orchestrator.graph.num_edges} edges")

question = st.text_area("Question", height=110)
run = st.button("Answer", type="primary", disabled=not question.strip())

if run:
    with st.spinner("Reasoning..."):
        trace = orchestrator.answer(question.strip())

    st.subheader("Answer")
    st.write(trace.answer)

    st.subheader("Trace")
    for step in trace.steps:
        with st.expander(step.summary, expanded=False):
            st.json(step.data)
    st.download_button(
        "Download trace JSON",
        trace.to_json(),
        file_name=f"{domain.name}-trace.json",
        mime="application/json",
    )

with st.expander("Knowledge graph", expanded=False):
    nodes, edges = orchestrator.graph.view(kinds={"clause"})
    st.caption(f"Showing {len(nodes)} clause nodes and {len(edges)} edges")
    st.graphviz_chart(graph_dot(nodes, edges))
