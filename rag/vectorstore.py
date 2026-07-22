"""Chroma persistence layer.

Design choice: the collection stores **explicit vectors** — embedding happens in
rag/llm.py and vectors are passed via ``embeddings=`` / ``query_embeddings=``.
Chroma never embeds anything itself, so the provider switch (Gemini/OpenAI/local)
stays entirely inside the LLM layer and no embedding-function config needs to be
serialized with the collection.

The collection uses **cosine** distance (Chroma's default is L2) and is namespaced
by provider+dimension (see config.Settings.collection_name) because vector spaces
of different dimensions/providers must never be mixed.
"""
from __future__ import annotations

from dataclasses import dataclass

# Some hosts (e.g. Streamlit Community Cloud) ship a system sqlite3 older than the
# >= 3.35 chromadb requires. If pysqlite3-binary is installed (see the commented
# line in requirements.txt), transparently swap it in BEFORE chromadb imports sqlite3.
try:  # pragma: no cover - deployment-environment shim
    __import__("pysqlite3")
    import sys as _sys

    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb

from .config import Settings, get_settings
from .ingest import Chunk
from .llm import Embeddings


@dataclass(frozen=True)
class Retrieved:
    """One retrieved chunk with provenance and a [~0..1] relevance score."""

    text: str
    document_name: str
    page_number: int
    relevance: float  # 1 - cosine_distance; higher = more similar


class VectorStore:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self._client = chromadb.PersistentClient(path=self.s.chroma_dir)
        self._col = self._get_or_create()

    def _get_or_create(self):
        return self._client.get_or_create_collection(
            name=self.s.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )

    # ── introspection ────────────────────────────────────────────────────────
    def count(self) -> int:
        return self._col.count()

    def has_document(self, document_name: str) -> bool:
        got = self._col.get(where={"document_name": document_name}, limit=1)
        return bool(got["ids"])

    def list_documents(self) -> list[str]:
        got = self._col.get(include=["metadatas"])
        return sorted({m["document_name"] for m in (got["metadatas"] or [])})

    # ── writes ───────────────────────────────────────────────────────────────
    def add_chunks(self, chunks: list[Chunk], embeddings: Embeddings) -> int:
        """Embed + store chunks. A document already in the store is skipped whole
        (idempotent re-ingest). Returns the number of chunks actually added."""
        by_doc: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.document_name, []).append(c)

        to_add: list[Chunk] = []
        for doc, doc_chunks in by_doc.items():
            if not self.has_document(doc):
                to_add.extend(doc_chunks)

        if not to_add:
            return 0

        vectors = embeddings.embed_documents([c.text for c in to_add])
        self._col.add(
            ids=[c.id for c in to_add],
            documents=[c.text for c in to_add],
            embeddings=vectors,
            metadatas=[
                {
                    "document_name": c.document_name,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                }
                for c in to_add
            ],
        )
        return len(to_add)

    def reset(self) -> None:
        """Drop and recreate the collection (used by the UI's 'clear documents')."""
        self._client.delete_collection(self.s.collection_name)
        self._col = self._get_or_create()

    # ── reads ────────────────────────────────────────────────────────────────
    def query(self, query_embedding: list[float], k: int | None = None) -> list[Retrieved]:
        """Top-k nearest chunks for a pre-embedded query, with relevance scores."""
        total = self.count()
        if total == 0:
            return []
        k = min(k or self.s.top_k, total)
        res = self._col.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[Retrieved] = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                Retrieved(
                    text=text,
                    document_name=str(meta["document_name"]),
                    page_number=int(meta["page_number"]),
                    relevance=1.0 - float(dist),
                )
            )
        return out
