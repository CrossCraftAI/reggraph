"""The vector half of the knowledge store: Chroma + local sentence-transformers.

Embeddings are computed locally (free, no API) and stored in an embedded Chroma
database that persists to disk. Importing this module pulls in the embedding
stack (torch), so keep it out of light-weight import paths.
"""

from dataclasses import dataclass
from pathlib import Path

from ..domains import Domain
from ..ingest import Chunk, load_chunks


@dataclass
class VectorHit:
    id: str
    title: str
    text: str
    score: float  # Chroma distance — lower is more similar

    @property
    def article_ref(self) -> str:
        """Backward-compatible name used by the old ask/eval prototype."""
        return self.id


class VectorIndex:
    COLLECTION = "regulatory_chunks"

    def __init__(self, chroma_dir: str | Path, embedding_model: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from sentence_transformers import SentenceTransformer

        self._chroma_dir = str(chroma_dir)
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self._embedder = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(self.COLLECTION)

    @classmethod
    def build(cls, domain: Domain, embedding_model: str) -> "VectorIndex":
        index = cls(domain.chroma_dir, embedding_model)
        index.add(load_chunks(domain.source_path))
        return index

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts, normalize_embeddings=True).tolist()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self._embed([c.text for c in chunks]),
            metadatas=[{"title": c.title} for c in chunks],
        )

    def count(self) -> int:
        return self._collection.count()

    @property
    def chunk_count(self) -> int:
        return self.count()

    def search(self, query: str, top_k: int) -> list[VectorHit]:
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=min(top_k, total),
        )
        hits: list[VectorHit] = []
        for i in range(len(result["ids"][0])):
            hits.append(
                VectorHit(
                    id=result["ids"][0][i],
                    title=result["metadatas"][0][i].get("title", ""),
                    text=result["documents"][0][i],
                    score=float(result["distances"][0][i]),
                )
            )
        return hits

    @classmethod
    def load(cls, chroma_dir: str | Path, embedding_model: str) -> "VectorIndex":
        return cls(chroma_dir, embedding_model)


def _split_by_article(text: str) -> list[tuple[str, str]]:
    """Backward-compatible helper for older tests.

    New code should call ``load_chunks`` with a file path.
    """
    from ..ingest import _chunks_from_markdown

    return [(chunk.id, chunk.text) for chunk in _chunks_from_markdown(text)]


def _chunk_articles(text: str, max_chars: int = 512, overlap: int = 64) -> list[Chunk]:
    """Backward-compatible article chunking helper."""
    chunks: list[Chunk] = []
    for article_ref, body in _split_by_article(text):
        start = 0
        part = 0
        while start < len(body):
            end = min(start + max_chars, len(body))
            title = article_ref.replace("-", " ").title()
            chunk_id = f"{article_ref}::{part}" if len(body) > max_chars else article_ref
            chunks.append(Chunk(id=chunk_id, title=title, text=body[start:end]))
            part += 1
            if end >= len(body):
                break
            start = end - overlap
    return chunks
