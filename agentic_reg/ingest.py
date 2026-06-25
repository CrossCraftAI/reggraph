"""Ingestion: load a regulatory document and split it into clause-level chunks.

Supports markdown (``## <Unit> N`` sections), plain text, and PDF. PDFs and
plain text are split on recognized ``<Unit> N`` heading lines. Each chunk
becomes both a vector entry and a graph node sharing a stable id (e.g.
``article-6`` or ``section-3``), which is what lets a vector hit seed graph
expansion.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    title: str
    text: str

    @property
    def article_ref(self) -> str:
        """Backward-compatible name for older article-only callers."""
        return self.id.split("::", 1)[0]


_UNIT_HEADING_LINE = re.compile(
    r"^\s*(?:Article|Section|Rule|Regulation|Clause|Paragraph)\s+\d+\b", flags=re.IGNORECASE
)


_UNIT_HEADING = re.compile(r"\b([A-Za-z][A-Za-z-]*)\s+(\d+)\b")


def _make_id(title: str) -> str:
    """Derive a stable id from a unit heading, unit-agnostic.

    "Article 6 — Lawfulness of processing" -> "article-6"; "Section 3 — Terms"
    -> "section-3". Falls back to a slug for headings without a "<Unit> N" form.
    """
    match = _UNIT_HEADING.search(title)
    if match:
        return f"{match.group(1).lower()}-{match.group(2)}"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _chunk(title: str, lines: list[str]) -> Chunk | None:
    body = "\n".join(lines).strip()
    if not body:
        return None
    return Chunk(id=_make_id(title), title=title, text=body)


def _chunks_from_markdown(text: str) -> list[Chunk]:
    """Split on level-2 (``## ``) headings; ignore the title and any preamble."""
    chunks: list[Chunk] = []
    title: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if title is not None and (chunk := _chunk(title, lines)):
                chunks.append(chunk)
            title, lines = line[3:].strip(), []
        elif title is not None:
            lines.append(line)
    if title is not None and (chunk := _chunk(title, lines)):
        chunks.append(chunk)
    return chunks


def _chunks_from_legal_text(text: str) -> list[Chunk]:
    """Split plain text / extracted PDF text on recognized unit heading lines."""
    chunks: list[Chunk] = []
    title: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if _UNIT_HEADING_LINE.match(line):
            if title is not None and (chunk := _chunk(title, lines)):
                chunks.append(chunk)
            title, lines = line.strip(), []
        elif title is not None:
            lines.append(line)
    if title is not None and (chunk := _chunk(title, lines)):
        chunks.append(chunk)
    return chunks


def _text_from_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_chunks(path: str | Path) -> list[Chunk]:
    """Parse the document at ``path`` into chunks, dispatching on file type."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _chunks_from_legal_text(_text_from_pdf(path))

    text = path.read_text(encoding="utf-8")
    if suffix in {".md", ".markdown"}:
        return _chunks_from_markdown(text)
    return _chunks_from_legal_text(text)
