"""PDF ingestion: extract page text (1-based) and chunk it, preserving metadata.

Chunking happens **per page** — the key trick (borrowed from the Chroma cookbook
examples) that keeps an exact page number on every chunk, which is what makes
page-level citations possible downstream.

The splitter is a small, dependency-free recursive character splitter in the same
spirit as LangChain's ``RecursiveCharacterTextSplitter``: coarse separators first
(paragraphs, lines, sentences, words), greedy packing up to ``chunk_size`` with an
``overlap`` tail carried between consecutive chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Sequence

from pypdf import PdfReader

from .config import Settings, get_settings

# Coarse -> fine. The final "" means "hard cut" for pathological unbroken runs.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

PdfSource = str | Path | IO[bytes]


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit. Metadata travels with the text into the vector store."""

    text: str
    document_name: str
    page_number: int  # 1-based, as printed in a PDF viewer
    chunk_index: int  # 0-based, document-global ordering

    @property
    def id(self) -> str:
        return f"{self.document_name}::p{self.page_number}::c{self.chunk_index}"


# ────────────────────────────── extraction ──────────────────────────────────
def extract_pages(source: PdfSource) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with 1-based page numbers.

    Accepts a path or a binary file-like object (e.g. a Streamlit upload).
    Pages with no extractable text are returned as empty strings (text PDFs only —
    scanned/image PDFs are out of scope, no OCR).
    """
    reader = PdfReader(source)
    return [(i, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages, start=1)]


# ─────────────────────────────── splitting ──────────────────────────────────
def _atomize(text: str, chunk_size: int, seps: Sequence[str]) -> list[str]:
    """Break text into pieces each <= chunk_size using the coarsest separator that
    fits, recursing to finer separators; falls back to a hard cut."""
    if len(text) <= chunk_size:
        return [text]
    if not seps:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _atomize(text, chunk_size, rest)
    parts = text.split(sep)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep  # keep separators so no characters are lost
        if part:
            out.extend(_atomize(part, chunk_size, rest))
    return out


def _pack(atoms: Sequence[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge atoms into chunks <= chunk_size, carrying an overlap tail."""
    chunks: list[str] = []
    cur = ""
    for atom in atoms:
        if cur and len(cur) + len(atom) > chunk_size:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap > 0 else ""
        cur += atom
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split one page's text into chunks. Returns [] for empty/whitespace text."""
    if overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({overlap}) must be < chunk_size ({chunk_size})")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    return _pack(_atomize(text, chunk_size, _SEPARATORS), chunk_size, overlap)


# ─────────────────────────────── chunking ───────────────────────────────────
def chunk_pages(
    pages: Sequence[tuple[int, str]],
    document_name: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Chunk pre-extracted pages; every chunk keeps its document + page provenance."""
    chunks: list[Chunk] = []
    idx = 0
    for page_number, text in pages:
        for piece in split_text(text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    text=piece,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def chunk_pdf(
    source: PdfSource,
    document_name: str | None = None,
    settings: Settings | None = None,
) -> list[Chunk]:
    """Convenience: PDF (path or upload stream) -> chunks with metadata."""
    s = settings or get_settings()
    if document_name is None:
        if isinstance(source, (str, Path)):
            document_name = Path(source).name
        else:
            document_name = getattr(source, "name", None) or "uploaded.pdf"
    return chunk_pages(extract_pages(source), document_name, s.chunk_size, s.chunk_overlap)
