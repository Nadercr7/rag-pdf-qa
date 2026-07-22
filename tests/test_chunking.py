"""Chunking preserves document + page metadata and loses no content."""
from pathlib import Path

import pytest

from rag.ingest import chunk_pages, chunk_pdf, split_text

SAMPLES = Path(__file__).resolve().parent.parent / "sample_pdfs"


def test_split_text_respects_chunk_size_and_loses_nothing():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    pieces = split_text(text, chunk_size=120, overlap=20)
    assert len(pieces) > 1
    assert all(len(p) <= 120 for p in pieces)
    joined = " ".join(pieces)
    for i in range(60):  # every sentence survives into at least one chunk
        assert f"Sentence number {i} is here." in joined


def test_split_text_overlap_carries_tail_between_chunks():
    text = " ".join(f"word{i}" for i in range(200))
    pieces = split_text(text, chunk_size=100, overlap=30)
    assert len(pieces) >= 2
    for a, b in zip(pieces, pieces[1:]):
        assert a.split()[-1] in b  # last word of a chunk reappears in the next


def test_split_text_never_exceeds_chunk_size_even_with_large_atoms():
    # 90-char unbroken atoms: naive overlap-carry would emit tail+atom > chunk_size
    text = ". ".join("x" * 90 for _ in range(10))
    pieces = split_text(text, chunk_size=100, overlap=30)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces), [len(p) for p in pieces]
    assert any("x" * 90 in p for p in pieces)  # content still intact


def test_split_text_edge_cases():
    assert split_text("   \n  ", 100, 10) == []
    assert split_text("short", 100, 10) == ["short"]
    with pytest.raises(ValueError):
        split_text("hello world", chunk_size=10, overlap=10)


def test_chunk_pages_attaches_document_and_page_metadata():
    pages = [
        (1, "Alpha policy about vacation. " * 5),
        (2, "Beta policy about security. " * 5),
    ]
    chunks = chunk_pages(pages, "doc.pdf", chunk_size=80, overlap=10)
    assert chunks
    assert all(c.document_name == "doc.pdf" for c in chunks)
    assert {c.page_number for c in chunks} == {1, 2}
    for c in chunks:  # page provenance is exact, never smeared across pages
        if "Alpha" in c.text:
            assert c.page_number == 1
        if "Beta" in c.text:
            assert c.page_number == 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)  # ids unique


def test_chunk_pdf_on_sample_corpus_preserves_metadata(make_settings):
    pdf = SAMPLES / "meridian_benefits_guide.pdf"
    assert pdf.exists(), "sample corpus missing — run scripts/make_sample_pdfs.py"
    chunks = chunk_pdf(pdf, settings=make_settings())
    assert chunks
    assert all(c.document_name == "meridian_benefits_guide.pdf" for c in chunks)
    assert {c.page_number for c in chunks} == {1, 2}
    # a fact known to live on page 2 only ever appears in page-2 chunks
    hits = [c for c in chunks if "16 weeks" in c.text]
    assert hits and all(c.page_number == 2 for c in hits)
