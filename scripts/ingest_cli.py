"""Ingest a folder of text PDFs into the Chroma store.

Usage:
    python scripts/ingest_cli.py                # ingests ./sample_pdfs
    python scripts/ingest_cli.py path/to/pdfs   # ingests a custom folder

Idempotent: documents already in the store (by file name) are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `rag` importable

from rag.config import get_settings
from rag.ingest import chunk_pdf
from rag.llm import get_embeddings
from rag.vectorstore import VectorStore


def main(folder: str | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    pdf_dir = Path(folder) if folder else root / "sample_pdfs"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return 1

    s = get_settings()
    store = VectorStore(s)
    emb = get_embeddings(s)

    for pdf in pdfs:
        chunks = chunk_pdf(pdf, settings=s)
        added = store.add_chunks(chunks, emb)
        note = "" if added else "  (already ingested - skipped)"
        print(f"  {pdf.name}: {len(chunks)} chunk(s), {added} added{note}")

    print(f"\ncollection={s.collection_name}  total_chunks={store.count()}")
    print(f"documents: {', '.join(store.list_documents())}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
