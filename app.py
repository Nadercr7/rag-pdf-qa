"""Streamlit chat UI for the grounded RAG PDF Q&A demo.

Run locally:  streamlit run app.py
Keys come from .env locally, or host secrets when deployed (see rag/config.py).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag.config import get_settings
from rag.ingest import chunk_pdf
from rag.llm import get_chat_model, get_embeddings
from rag.qa import answer
from rag.vectorstore import VectorStore

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Grounded PDF Q&A", page_icon="📄")


def _md(text: str) -> str:
    """Escape $ so Streamlit's markdown doesn't turn '$250 ... $350' into KaTeX."""
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Loading vector store & model clients...")
def get_backend():
    s = get_settings()
    return s, VectorStore(s), get_embeddings(s), get_chat_model(s)


s, store, emb, chat = get_backend()

# First run on a fresh store: seed the sample corpus so the demo is instantly usable.
if store.count() == 0:
    with st.spinner("First run — ingesting the sample corpus..."):
        for pdf in sorted((ROOT / "sample_pdfs").glob("*.pdf")):
            store.add_chunks(chunk_pdf(pdf, settings=s), emb)

# ────────────────────────────────── sidebar ──────────────────────────────────
with st.sidebar:
    st.subheader("📚 Documents")
    docs = store.list_documents()
    st.caption(f"{store.count()} chunks from {len(docs)} document(s)")
    for d in docs:
        st.markdown(f"- {d}")

    uploads = st.file_uploader(
        "Add text PDFs (no OCR)", type="pdf", accept_multiple_files=True
    )
    if uploads and st.button("Ingest uploaded PDFs", use_container_width=True):
        with st.spinner("Chunking + embedding..."):
            added = 0
            for f in uploads:
                added += store.add_chunks(
                    chunk_pdf(f, document_name=f.name, settings=s), emb
                )
        st.success(f"Added {added} chunk(s). Already-ingested files are skipped.")
        st.rerun()

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        f"LLM: **{s.llm_provider}** · embeddings: **{s.embedding_provider}** "
        f"({s.embed_dim}d) · top-k {s.top_k} · threshold {s.relevance_threshold}"
    )
    st.caption(
        "Grounding: answers come only from retrieved sources; weak retrieval or "
        "unsupported questions return the exact not-found reply."
    )

# ─────────────────────────────────── chat ────────────────────────────────────
st.title("📄 Grounded PDF Q&A")
st.caption(
    "Ask about the ingested documents. Every answer is grounded with "
    "document + page citations — otherwise it replies: "
    "*\"I couldn't find this in the documents.\"*"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:  # replay history
    with st.chat_message(m["role"]):
        st.markdown(_md(m["content"]))
        if m.get("citations"):
            st.caption("📎 Sources: " + " · ".join(f"**{d}** — p.{p}" for d, p in m["citations"]))

if prompt := st.chat_input("e.g. How many vacation days do employees get?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(_md(prompt))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving & answering..."):
                a = answer(prompt, store, emb, chat, s)
        except Exception as exc:  # provider/network hiccup — fail visibly, keep chat alive
            st.error(f"Provider error — please retry. Details: {exc}")
            st.stop()

        st.markdown(_md(a.text))
        if a.citations:
            st.caption("📎 Sources: " + " · ".join(f"**{d}** — p.{p}" for d, p in a.citations))
        if a.retrieved and not a.not_found:
            with st.expander("🔍 Retrieved context (with relevance)"):
                for src in a.retrieved:
                    st.markdown(
                        f"**[Source {src.index}]** {src.document_name} — p.{src.page_number} "
                        f"· relevance `{src.relevance:.2f}`"
                    )
                    snippet = src.text[:300] + ("..." if len(src.text) > 300 else "")
                    st.caption(_md(snippet))

    st.session_state.messages.append(
        {"role": "assistant", "content": a.text, "citations": a.citations}
    )
