"""VectorStore add/query mechanics with deterministic fake embeddings (offline)."""
from rag.ingest import Chunk
from rag.vectorstore import VectorStore
from tests.fakes import FakeEmbeddings

AXES = {"vacation": 0, "password": 1, "expense": 2}


def _chunks() -> list[Chunk]:
    return [
        Chunk("Employees get 20 days of vacation.", "handbook.pdf", 1, 0),
        Chunk("Passwords must be 14 characters.", "security.pdf", 1, 0),
        Chunk("Expense reports are due in 30 days.", "expense.pdf", 2, 0),
    ]


def test_query_returns_relevant_hit_with_metadata(make_settings):
    s = make_settings()
    store, emb = VectorStore(s), FakeEmbeddings(AXES)
    assert store.add_chunks(_chunks(), emb) == 3

    hits = store.query(emb.embed_query("How much vacation do I get?"), k=2)
    assert hits, "expected retrieval hits"
    top = hits[0]
    assert top.document_name == "handbook.pdf"
    assert top.page_number == 1
    assert top.relevance > 0.99  # same axis -> cosine ~1.0
    assert hits == sorted(hits, key=lambda h: h.relevance, reverse=True)


def test_add_is_idempotent_per_document(make_settings):
    s = make_settings()
    store, emb = VectorStore(s), FakeEmbeddings(AXES)
    assert store.add_chunks(_chunks(), emb) == 3
    assert store.add_chunks(_chunks(), emb) == 0  # same documents -> skipped whole
    assert store.count() == 3
    assert store.list_documents() == ["expense.pdf", "handbook.pdf", "security.pdf"]
    assert store.document_stats() == {"expense.pdf": 1, "handbook.pdf": 1, "security.pdf": 1}


def test_query_on_empty_store_returns_no_hits(make_settings):
    s = make_settings()
    store, emb = VectorStore(s), FakeEmbeddings(AXES)
    assert store.query(emb.embed_query("anything at all"), k=3) == []


def test_reset_clears_the_collection(make_settings):
    s = make_settings()
    store, emb = VectorStore(s), FakeEmbeddings(AXES)
    store.add_chunks(_chunks(), emb)
    assert store.count() == 3
    store.reset()
    assert store.count() == 0
