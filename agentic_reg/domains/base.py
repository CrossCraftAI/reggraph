"""The Domain plugin abstraction.

A ``Domain`` packages everything domain-specific about a regulation: its source
document, the label for its units (article / section / rule), and where its
knowledge store and benchmark live. Adding a new domain is "define a Domain +
drop in a markdown document" — no core code changes.
"""

from dataclasses import dataclass
from pathlib import Path

from ..config import PROJECT_ROOT


@dataclass(frozen=True)
class Domain:
    name: str  # short id, e.g. "gdpr"
    title: str  # human title, e.g. "EU GDPR (excerpt)"
    description: str
    source_path: Path  # the regulatory document (markdown)
    unit_label: str = "article"  # "article" | "section" | "rule" ... (UX + citation example)
    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Domain name must not be empty")

    @property
    def store_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "store" / self.name

    @property
    def chroma_dir(self) -> Path:
        return self.store_dir / "chroma"

    @property
    def graph_path(self) -> Path:
        return self.store_dir / "graph.json"

    @property
    def benchmark_path(self) -> Path:
        return PROJECT_ROOT / "data" / "benchmark" / f"{self.name}.jsonl"
