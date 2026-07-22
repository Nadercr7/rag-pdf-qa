"""Streamlit chat UI for the grounded RAG PDF Q&A demo.

Run locally:  streamlit run app.py
Keys come from .env locally, or host secrets when deployed (see rag/config.py).
Theme lives in .streamlit/config.toml; the CSS below only adds refinements on top.
"""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from rag.config import get_settings
from rag.ingest import chunk_pdf
from rag.llm import get_chat_model, get_embeddings
from rag.qa import answer
from rag.vectorstore import VectorStore

ROOT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Nadercr7/rag-pdf-qa"

AVATARS = {"user": ":material/person:", "assistant": ":material/find_in_page:"}

EXAMPLE_QUESTIONS = [
    "How many vacation days do full-time employees get?",
    "What is the minimum password length?",
    "What is the hotel budget for international travel?",
    "Does the company offer pet insurance?",  # deliberately absent -> shows the guardrail
]

st.set_page_config(
    page_title="Grounded PDF Q&A",
    page_icon=":material/find_in_page:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* chrome */
#MainMenu, footer {visibility: hidden;}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.4rem; padding-bottom: 6rem;}
[data-testid="stSidebar"] {border-right: 1px solid #E9ECF2;}

/* header */
.app-title {font-size: 1.72rem; font-weight: 700; letter-spacing: -0.02em; color: #171C26;}
.app-sub {font-size: .93rem; color: #5B6472; margin-top: .4rem; line-height: 1.5; max-width: 40rem;}
.app-rule {height: 1px; background: #ECEFF4; margin: 1.15rem 0 .35rem;}

/* citation chips */
.chips-row {display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; margin: .2rem 0 .1rem;}
.chips-label {font-size: .68rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #98A1B3;}
.source-chip {display: inline-flex; align-items: center; gap: .38rem; border: 1px solid #E4E8EF;
  background: #F7F8FA; border-radius: 999px; padding: .18rem .62rem; font-size: .78rem;
  color: #2A3242; font-weight: 500; white-space: nowrap;}
.source-chip .chip-page {color: #3B5BDB; font-weight: 600;}
.rel-badge {display: inline-block; background: #EEF1FE; color: #3B5BDB; border-radius: 999px;
  padding: .14rem .5rem; font-size: .72rem; font-weight: 600;}
.gate-note {font-size: .78rem; color: #98A1B3; margin-top: .2rem;}

/* retrieved-context rows */
.ctx-head {display: flex; flex-wrap: wrap; align-items: center; gap: .45rem; margin: .4rem 0 .05rem;}
.ctx-source {font-size: .72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; color: #5B6472;}

/* sidebar */
.side-label {font-size: .7rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
  color: #98A1B3; margin: .15rem 0 .45rem;}
.doc-row {display: flex; justify-content: space-between; align-items: baseline; gap: .6rem;
  padding: .24rem 0; border-bottom: 1px dashed #E9ECF2; font-size: .82rem; color: #2A3242;}
.doc-row:last-child {border-bottom: none;}
.doc-row span:first-child {overflow-wrap: anywhere;}
.doc-count {color: #98A1B3; font-size: .74rem; font-variant-numeric: tabular-nums; white-space: nowrap;}
.cfg-row {display: flex; justify-content: space-between; gap: .6rem; padding: .16rem 0; font-size: .8rem; color: #2A3242;}
.cfg-row .k {color: #98A1B3;}

[data-testid="stChatMessage"] {padding: .35rem 0;}
</style>
""")


def _md(text: str) -> str:
    """Escape $ so Streamlit's markdown doesn't turn '$250 ... $350' into KaTeX."""
    return text.replace("$", "\\$")


def _chips(citations: list[tuple[str, int]]) -> str:
    """Render (document, page) citations as chips. Names are user-supplied -> escaped."""
    spans = "".join(
        f'<span class="source-chip">{html.escape(doc)}'
        f'<span class="chip-page">p.{page}</span></span>'
        for doc, page in citations
    )
    return f'<div class="chips-row"><span class="chips-label">Sources</span>{spans}</div>'


def _gate_note(stage: str) -> str:
    reason = "retrieval threshold" if stage == "retrieval" else "model grounding"
    return f'<div class="gate-note">Refused by the {reason} gate - no supporting source in the documents.</div>'


@st.cache_resource(show_spinner="Loading vector store and model clients...")
def get_backend():
    s = get_settings()
    return s, VectorStore(s), get_embeddings(s), get_chat_model(s)


s, store, emb, chat = get_backend()

# First run on a fresh store: seed the sample corpus so the demo is instantly usable.
if store.count() == 0:
    with st.spinner("First run - ingesting the sample corpus..."):
        for pdf in sorted((ROOT / "sample_pdfs").glob("*.pdf")):
            store.add_chunks(chunk_pdf(pdf, settings=s), emb)

# ────────────────────────────────── sidebar ──────────────────────────────────
with st.sidebar:
    st.html('<div class="side-label">Knowledge base</div>')
    stats = store.document_stats()
    if stats:
        st.html(
            "".join(
                f'<div class="doc-row"><span>{html.escape(name)}</span>'
                f'<span class="doc-count">{count} chunks</span></div>'
                for name, count in sorted(stats.items())
            )
        )
        st.caption(f"{store.count()} chunks across {len(stats)} documents")
    else:
        st.caption("No documents yet - add PDFs below.")

    st.html('<div class="side-label" style="margin-top:1.1rem">Add documents</div>')
    uploads = st.file_uploader("Text PDFs only (no OCR)", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Add to knowledge base", icon=":material/upload_file:", use_container_width=True):
        results: list[str] = []
        with st.spinner("Chunking and embedding..."):
            for f in uploads:  # one bad file (encrypted/corrupt) must not sink the rest
                try:
                    chunks = chunk_pdf(f, document_name=f.name, settings=s)
                    if not chunks:
                        results.append(f"No text in {f.name} - scanned PDF?")
                        continue
                    added = store.add_chunks(chunks, emb)
                    if added:
                        results.append(f"Added {f.name} - {added} chunks")
                    else:
                        results.append(f"Skipped {f.name} - a document with this name is already ingested")
                except Exception as exc:  # noqa: BLE001 - surface per file, keep going
                    results.append(f"Failed {f.name} - {exc}")
        st.session_state["ingest_results"] = results
        st.rerun()  # refresh the document list; results shown after rerun

    if msgs := st.session_state.pop("ingest_results", None):
        for line in msgs:
            st.caption(line)

    st.divider()
    if st.button("Reset conversation", icon=":material/refresh:", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.html('<div class="side-label">Configuration</div>')
    chat_model = s.gemini_chat_model if s.llm_provider == "gemini" else s.openai_chat_model
    st.html(
        "".join(
            f'<div class="cfg-row"><span class="k">{k}</span><span>{html.escape(v)}</span></div>'
            for k, v in [
                ("LLM", f"{s.llm_provider} / {chat_model}"),
                ("Embeddings", f"{s.embed_model} / {s.embed_dim}d"),
                ("Top-k", str(s.top_k)),
                ("Relevance gate", f"min {s.relevance_threshold}"),
            ]
        )
    )
    st.markdown(f"[Source on GitHub]({REPO_URL})")

# ─────────────────────────────────── chat ────────────────────────────────────
st.html(
    '<div class="app-title">Grounded PDF Q&A</div>'
    '<div class="app-sub">Answers come only from the ingested documents, each with '
    'document and page citations. When the documents do not contain the answer, the '
    'assistant replies: <em>"I couldn\'t find this in the documents."</em></div>'
    '<div class="app-rule"></div>'
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# chat_input renders pinned at the bottom regardless of call position; reading it early
# lets the example box hide in the same run a question is submitted.
prompt = st.chat_input("Ask about the documents") or st.session_state.pop("pending_q", None)

if not st.session_state.messages and not prompt:
    with st.container(border=True):
        st.markdown("**Try an example**")
        st.caption("The last one is deliberately not in the documents - it demonstrates the refusal guardrail.")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"example-{i}", use_container_width=True):
                st.session_state.pending_q = q
                st.rerun()

for m in st.session_state.messages:  # replay history
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        st.markdown(_md(m["content"]))
        if m.get("citations"):
            st.html(_chips(m["citations"]))
        elif m.get("stage"):
            st.html(_gate_note(m["stage"]))

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(_md(prompt))

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Searching the documents..."):
                a = answer(prompt, store, emb, chat, s)
        except Exception as exc:  # provider/network hiccup - fail visibly, keep chat alive
            st.error(f"Provider error - please retry. Details: {exc}")
            st.stop()

        st.markdown(_md(a.text))
        if a.citations:
            st.html(_chips(a.citations))
        if a.not_found and a.refusal_stage:
            st.html(_gate_note(a.refusal_stage))
        if a.retrieved and not a.not_found:
            with st.expander("Retrieved context"):
                for src in a.retrieved:
                    st.html(
                        f'<div class="ctx-head"><span class="ctx-source">Source {src.index}</span>'
                        f'<span class="source-chip">{html.escape(src.document_name)}'
                        f'<span class="chip-page">p.{src.page_number}</span></span>'
                        f'<span class="rel-badge">relevance {src.relevance:.2f}</span></div>'
                    )
                    snippet = src.text[:300] + ("..." if len(src.text) > 300 else "")
                    st.caption(_md(snippet))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": a.text,
            "citations": a.citations,
            "stage": a.refusal_stage,
        }
    )
