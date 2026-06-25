"""Knowledge-graph construction — deterministic backbone + best-effort typed LLM enrichment.

extract_cross_references: scans clause text for "<Unit> N" mentions, no LLM needed.
Always produces a correct reference graph.

extract_graph_elements: one LLM call per clause for typed concept nodes
(obligation/definition/right/condition/principle/prohibition) and typed relations
(requires/overrides/depends_on/exception_to/applies_to/implies).
Never raises — malformed output yields an empty Extraction so the build always succeeds.
"""

import re
from dataclasses import dataclass, field

from .._internal import parse_json_object
from ..ingest import Chunk
from ..providers.base import LLMProvider

# Unit-agnostic reference detector: "Article 6", "Section 3", "Rule 4" ...
_REFERENCE_RE = re.compile(r"\b([A-Za-z][A-Za-z-]*)\s+(\d+)\b")
_BRACKET_REFERENCE_RE = re.compile(r"\[([a-z][a-z-]*-\d+)\]", flags=re.IGNORECASE)

CONCEPT_KINDS = {
    "obligation",
    "definition",
    "right",
    "condition",
    "principle",
    "prohibition",
}
RELATION_TYPES = {
    "requires",
    "overrides",
    "depends_on",
    "exception_to",
    "applies_to",
    "implies",
}

# How a clause connects to a concept it introduces, by concept kind.
CONCEPT_EDGE = {
    "obligation": "imposes",
    "definition": "defines",
    "right": "grants",
    "condition": "sets_condition",
    "principle": "establishes",
    "prohibition": "prohibits",
    "concept": "introduces",
}


def extract_cross_references(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """Return (source_id, target_id) edges from "<Unit> N" mentions in text.

    Over-matches (e.g. "paragraph 2") are harmless: an edge is kept only when the
    derived id is a known node, so unrelated "<word> <number>" pairs are dropped.
    """
    known_ids = {c.id for c in chunks}
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        targets = {match.lower() for match in _BRACKET_REFERENCE_RE.findall(chunk.text)}
        targets.update(
            f"{word.lower()}-{number}" for word, number in _REFERENCE_RE.findall(chunk.text)
        )
        for target in targets:
            edge = (chunk.id, target)
            if target in known_ids and target != chunk.id and edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return edges


@dataclass
class Concept:
    id: str
    label: str
    kind: str
    clause_id: str


@dataclass
class ClauseRelation:
    source_id: str
    target_id: str
    relation: str


@dataclass
class Extraction:
    concepts: list[Concept] = field(default_factory=list)
    relations: list[ClauseRelation] = field(default_factory=list)


EXTRACT_SYSTEM = "You extract a typed knowledge graph from regulatory text. Respond with JSON only."

EXTRACT_PROMPT = """\
Extract a typed knowledge graph from this clause of "{title}".

Return ONLY a JSON object with two keys:
{{
  "concepts": [
    {{"label": "<short noun phrase>",
      "kind": "obligation|definition|right|condition|principle|prohibition"}}
  ],
  "relations": [
    {{"target_unit": <number of another {unit_label} this clause relates to>,
     "type": "requires|overrides|depends_on|exception_to|applies_to|implies"}}
  ]
}}

Rules:
- "concepts": 2-5 key concepts this clause introduces, each with the best-fitting kind.
- "relations": only where the clause expresses a real dependency on another {unit_label}
  (e.g. consent processing "requires" the consent {unit_label}). Use the {unit_label} NUMBER.
- Output the JSON object only, no prose.

CLAUSE:
{text}
"""


def extract_graph_elements(
    chunk: Chunk, provider: LLMProvider, known_ids: set[str], unit_label: str = "article"
) -> Extraction:
    """Best-effort typed extraction for one chunk. Never raises."""
    try:
        raw = provider.complete(
            EXTRACT_PROMPT.format(title=chunk.title, text=chunk.text, unit_label=unit_label),
            system=EXTRACT_SYSTEM,
        )
    except Exception:
        return Extraction()

    data = parse_json_object(raw)
    result = Extraction()

    for i, item in enumerate(data.get("concepts", []) or []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        kind = str(item.get("kind", "concept")).strip().lower()
        if kind not in CONCEPT_KINDS:
            kind = "concept"
        result.concepts.append(
            Concept(id=f"{chunk.id}::concept::{i}", label=label, kind=kind, clause_id=chunk.id)
        )

    for item in data.get("relations", []) or []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_unit", "")).strip()
        match = re.search(r"\d+", target)
        if not match:
            continue
        target_id = f"{unit_label}-{match.group(0)}"
        relation = str(item.get("type", "")).strip().lower()
        if target_id in known_ids and target_id != chunk.id and relation in RELATION_TYPES:
            result.relations.append(
                ClauseRelation(source_id=chunk.id, target_id=target_id, relation=relation)
            )

    return result
