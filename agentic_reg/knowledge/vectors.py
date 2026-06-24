"""Vector index for semantic chunk retrieval backed by ChromaDB."""

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from agentic_reg.domains import Domain


@dataclass
class Chunk:
    """A retrieved chunk with its source article."""

    id: str
    text: str
    article_ref: str


class _EmbeddingFunc(EmbeddingFunction):
    """Bridge from SentenceTransformer to ChromaDB's ``EmbeddingFunction`` protocol."""

    def __init__(self, model: SentenceTransformer) -> None:
        self._model = model

    def name(self) -> str:
        return f"SentenceTransformer({self._model})"  # type: ignore[no-any-return]

    def __call__(self, docs: list[str]) -> list[list[float]]:  # type: ignore[override]
        embeddings = self._model.encode(docs, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]  # type: ignore[no-any-return]


class VectorIndex:
    """ChromaDB-backed semantic search over regulation chunks."""

    COLLECTION_NAME = "regulatory_chunks"

    def __init__(self, chroma_dir: str | Path, embedding_model_name: str) -> None:
        self._chroma_dir = str(chroma_dir)
        self._model = SentenceTransformer(embedding_model_name)
        self._client = chromadb.PersistentClient(path=self._chroma_dir)
        self._ef = _EmbeddingFunc(self._model)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
        )

    # -- build ---------------------------------------------------------------

    @classmethod
    def build(cls, domain: Domain, embedding_model: str) -> "VectorIndex":
        """Build a vector index from the domain's source markdown."""
        chroma_dir = str(Path("data") / "store" / domain.name / "chroma")
        instance = cls(chroma_dir, embedding_model)

        text = domain.source_path.read_text(encoding="utf-8")
        chunks = _chunk_articles(text, domain.chunk_size, domain.chunk_overlap)

        if chunks:
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [{"article_ref": chunk.article_ref} for chunk in chunks]
            instance._collection.add(ids=ids, documents=documents, metadatas=metadatas)

        return instance

    # -- search --------------------------------------------------------------

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        """Return the top-*k* chunks for *query*."""
        results = self._collection.query(query_texts=[query], n_results=top_k)
        chunks: list[Chunk] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, chunk_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    article_ref=meta.get("article_ref", ""),
                )
            )
        return chunks

    # -- load ----------------------------------------------------------------

    @classmethod
    def load(cls, chroma_dir: str | Path, embedding_model: str) -> "VectorIndex":
        """Load an existing index from *chroma_dir*."""
        return cls(chroma_dir, embedding_model)

    # -- query ---------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return self._collection.count()  # type: ignore[no-any-return]


# ── chunking helpers ────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^##\s+Article\s+(\d+)\s*[—–-]", flags=re.MULTILINE | re.IGNORECASE)


def _chunk_articles(text: str, max_chars: int = 512, overlap: int = 64) -> list[Chunk]:
    """Split a regulation markdown into chunks keyed by article.

    Each chunk is at most *max_chars* characters with *overlap* characters
    carried from the previous chunk.  Article boundaries are always respected:
    a new article starts a fresh chunk.
    """
    chunks: list[Chunk] = []
    articles = _split_by_article(text)
    for article_ref, body in articles:
        if not body.strip():
            continue
        # Split long articles into overlapping sub-chunks.
        start = 0
        part = 0
        while start < len(body):
            end = min(start + max_chars, len(body))
            chunk_text = body[start:end]
            chunk_id = f"{article_ref}::{part}"
            chunks.append(Chunk(id=chunk_id, text=chunk_text, article_ref=article_ref))
            part += 1
            if end >= len(body):
                break
            start = end - overlap
    return chunks


def _split_by_article(text: str) -> list[tuple[str, str]]:
    """Return list of (article_id, body) pairs."""
    matches = list(_SECTION_RE.finditer(text))
    articles: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        article_num = match.group(1)
        article_id = f"article-{article_num}"
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        articles.append((article_id, body))
    return articles
