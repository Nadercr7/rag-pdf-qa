"""The not-found contract and citation flow, end to end (offline).

The crucial test: a low-relevance query must be answered by gate 1 with the EXACT
contractual string, WITHOUT the LLM ever being invoked (RaisingChat proves it).
"""
from rag.config import NOT_FOUND_MESSAGE
from rag.ingest import Chunk
from rag.qa import answer, parse_citations
from rag.vectorstore import VectorStore
from tests.fakes import FakeEmbeddings, RaisingChat, StaticChat

AXES = {"vacation": 0, "password": 1}


def _seeded(s):
    store, emb = VectorStore(s), FakeEmbeddings(AXES)
    store.add_chunks(
        [
            Chunk("Employees get 20 days of vacation.", "handbook.pdf", 1, 0),
            Chunk("Passwords must be 14 characters.", "security.pdf", 2, 0),
        ],
        emb,
    )
    return store, emb


def test_low_relevance_query_routes_to_not_found_without_llm_call(make_settings):
    s = make_settings(relevance_threshold=0.55)
    store, emb = _seeded(s)
    # no keyword overlap with any stored chunk -> cosine ~0 -> below threshold
    a = answer("Tell me about quantum plumbing", store, emb, RaisingChat(), s)
    assert a.not_found is True
    assert a.text == NOT_FOUND_MESSAGE  # exact contract string, verbatim
    assert a.refusal_stage == "retrieval"
    assert a.sources == []


def test_model_refusal_is_normalized_to_exact_contract_string(make_settings):
    s = make_settings()
    store, emb = _seeded(s)
    drifted = StaticChat("I couldn't find this in the documents")  # missing final period
    a = answer("How many vacation days do we get?", store, emb, drifted, s)
    assert a.not_found is True
    assert a.text == NOT_FOUND_MESSAGE
    assert a.refusal_stage == "model"


def test_grounded_answer_maps_citations_to_document_and_page(make_settings):
    s = make_settings()
    store, emb = _seeded(s)
    chat = StaticChat("You get 20 days of vacation [Source 1].")
    a = answer("How many vacation days do employees get?", store, emb, chat, s)
    assert a.not_found is False
    assert [(x.document_name, x.page_number) for x in a.sources] == [("handbook.pdf", 1)]
    # the model was shown the numbered source block for what it cited
    assert chat.calls and "[Source 1] (document: handbook.pdf, page: 1)" in chat.calls[0][1]


def test_answer_without_markers_falls_back_to_top_source(make_settings):
    s = make_settings()
    store, emb = _seeded(s)
    chat = StaticChat("You get 20 days of vacation.")  # no [Source i] markers
    a = answer("How many vacation days do employees get?", store, emb, chat, s)
    assert a.not_found is False
    assert [(x.document_name, x.page_number) for x in a.sources] == [("handbook.pdf", 1)]


def test_out_of_range_markers_are_stripped_from_display(make_settings):
    s = make_settings()
    store, emb = _seeded(s)
    chat = StaticChat("You get 20 days of vacation [Source 7].")  # source 7 doesn't exist
    a = answer("How many vacation days do employees get?", store, emb, chat, s)
    assert a.not_found is False
    assert "[Source 7]" not in a.text  # unresolvable marker removed from display
    assert a.citations == [("handbook.pdf", 1)]  # provenance falls back to top source


def test_paraphrased_refusal_without_citations_is_normalized(make_settings):
    s = make_settings()
    store, emb = _seeded(s)
    chat = StaticChat("The provided sources do not contain information about that topic.")
    a = answer("How many vacation days do employees get?", store, emb, chat, s)
    assert a.not_found is True
    assert a.text == NOT_FOUND_MESSAGE
    assert a.refusal_stage == "model"


def test_legitimate_answer_is_never_reclassified_as_refusal(make_settings):
    s = make_settings()
    store, emb = _seeded(s)
    # cites a source AND contains refusal-ish words deep in the sentence
    chat = StaticChat(
        "Restricted data must never be copied to devices; the policy does not provide "
        "exceptions [Source 1]."
    )
    a = answer("What are the vacation rules?", store, emb, chat, s)
    assert a.not_found is False


def test_citations_dedupe_same_document_and_page(make_settings):
    s = make_settings()
    store, emb = VectorStore(s), FakeEmbeddings(AXES)
    store.add_chunks(
        [
            Chunk("Vacation part one.", "handbook.pdf", 1, 0),
            Chunk("Vacation part two.", "handbook.pdf", 1, 1),
        ],
        emb,
    )
    chat = StaticChat("Both parts matter [Source 1][Source 2].")
    a = answer("vacation details?", store, emb, chat, s)
    assert len(a.sources) == 2               # two distinct chunks cited...
    assert a.citations == [("handbook.pdf", 1)]  # ...but one display citation


def test_parse_citations_variants():
    assert parse_citations("A [Source 1]. B [Source 2].", 3) == [1, 2]
    assert parse_citations("A [Sources 1, 3].", 3) == [1, 3]
    assert parse_citations("A [source 2 and 3].", 3) == [2, 3]
    assert parse_citations("Nothing cited here.", 3) == []
    assert parse_citations("Bogus [Source 9].", 3) == []  # out of range ignored
