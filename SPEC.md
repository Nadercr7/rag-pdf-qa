# SPEC — Grounded RAG PDF Q&A

A small, production-quality Retrieval-Augmented Generation (RAG) chatbot that answers
questions **only** from a set of text-based PDFs, with source citations (document + page)
and a strict "not found" guardrail. Built on a **100% free** stack (Google Gemini free
tier by default), provider-agnostic so the same code runs on OpenAI.

This is an **MVP prototype**, engineered cleanly — not a platform. See *Out of scope*.

---

## 1. What it does

1. **Ingest** 10–20 text PDFs (no OCR) into a persistent Chroma vector store, chunked while
   preserving `document_name` + `page_number` on every chunk.
2. **Answer** questions by retrieving the most relevant chunks and generating an answer
   grounded **only** in those chunks.
3. **Cite** sources: every grounded answer shows which document and page it came from.
4. **Refuse** when the answer is not in the documents — replying with the *exact* string:
   > `I couldn't find this in the documents.`
5. **Chat UI** (Streamlit): ask questions, see answers + citations, upload new PDFs, and
   reset the conversation.

---

## 2. Architecture

```
                 ┌──────────────────────────── app.py (Streamlit) ───────────────────────────┐
                 │  chat input · message history · upload PDFs · reset · render citations     │
                 └───────────────┬───────────────────────────────────────────┬───────────────┘
                                 │ ask(question)                              │ ingest(files)
                                 ▼                                            ▼
   rag/qa.py  ── answer() ──────────────────────────────────────    rag/ingest.py
   1. embed query (RETRIEVAL_QUERY)                                  extract_pages(pdf) -> [(page, text)]
   2. vectorstore.query(top_k) -> chunks + relevance                chunk_pages(...)    -> [Chunk{doc,page,idx,text}]
   3. GATE 1: max relevance < THRESHOLD  ─────► "not found"                 │
   4. build [Source i] context (doc + page)                                 ▼
   5. chat.generate(grounded system prompt)                        rag/vectorstore.py  (Chroma, cosine)
   6. GATE 2: model says not-answerable   ─────► "not found"       add_chunks(embeddings) · query() · reset()
   7. parse [Source i] citations -> [{document, page}]                       │
        │                                                                    │
        ▼                                                                    ▼
   rag/llm.py  ── provider-agnostic ──  get_chat_model() · get_embeddings()  (driven by rag/config.py)
        ├── gemini : gemini-2.5-flash        + gemini-embedding-001 (768-d, L2-normalized)   [FREE, default]
        ├── openai : gpt-4o-mini (configurable) + text-embedding-3-small (1536-d)            [client's target]
        └── local  : (embeddings only) sentence-transformers all-MiniLM-L6-v2 (384-d)        [zero-API-cost fallback]
```

**Data flow (query):** question → embed(query) → Chroma cosine top-k → relevance gate →
grounded prompt with numbered sources → LLM → grounding gate → citation parse → answer + sources.

**Persistence:** Chroma `PersistentClient` at `./chroma_db`. Collection name is namespaced by
provider+dimension (e.g. `pdfs_gemini_768`) because embedding spaces of different dimensions
must never be mixed in one collection. Ingest is **idempotent** — a document already present
(matched by `document_name`) is skipped.

---

## 3. Provider abstraction (the switch)

One interface, two concrete provider families, selected by env:

| Env var | Values | Effect |
|---|---|---|
| `LLM_PROVIDER` | `gemini` (default) \| `openai` | Selects **chat + embeddings** provider together (the single required switch). |
| `EMBEDDING_PROVIDER` | `gemini` \| `openai` \| `local` | *Optional* override for embeddings only. Defaults to match `LLM_PROVIDER`. Set to `local` for zero-API-cost hosting. |

- `rag/llm.py` exposes two tiny interfaces: `Embeddings` (`embed_documents`, `embed_query`)
  and `ChatModel` (`generate(system, user)`), plus factories `get_embeddings()`,
  `get_chat_model()` that read `rag/config.py`. Nothing else in the codebase knows which
  provider is active.
- **Gemini** is the running default (free tier). **OpenAI** is first-class and correct
  (the real client's target) but not live-tested here (no key on the demo).
- **Key rotation:** Gemini free tier is ~10 RPM. `GEMINI_API_KEYS` may hold a comma-separated
  list; the Gemini client round-robins and retries the next key on `429/RESOURCE_EXHAUSTED`.

---

## 4. Grounding, citations & the not-found contract

This is the core of the product. Three mechanisms, defense-in-depth:

**(A) Retrieval relevance gate (deterministic, no LLM call).**
Chroma returns cosine `distance`; we compute `relevance = 1 - distance` (∈ ~[0,1]). If the
best retrieved chunk's relevance `< RELEVANCE_THRESHOLD`, we return the exact not-found string
immediately — no model call, fully deterministic, unit-testable. Threshold is tuned
empirically against the golden set (Phase 4) and is per-provider (env `RELEVANCE_THRESHOLD`).

**(B) Grounded generation prompt.**
Retrieved chunks are rendered as numbered sources:
```
[Source 1] (document: meridian_it_security_policy.pdf, page: 1)
<chunk text>

[Source 2] (document: ...)
<chunk text>
```
The system prompt instructs the model to answer **only** from these sources, to cite the
sources it uses inline as `[Source i]`, and — if the sources do not contain the answer — to
reply with **exactly** `I couldn't find this in the documents.` and nothing else. This catches
**hard negatives**: cases where a chunk is retrieved *above* threshold but is not actually an
answer (e.g. asking about a "vehicle allowance" when the docs only mention per-mile mileage
reimbursement).

**(C) Exact-string normalization.**
The refusal must be returned verbatim. If the model's stripped response matches the refusal
(case-insensitively, ignoring trailing punctuation), code normalizes it to the canonical
constant `NOT_FOUND_MESSAGE`. Guarantees the contract exactly regardless of minor model drift.

**Citations.** After a grounded answer, code parses the `[Source i]` markers the model emitted,
maps each to its `{document, page}`, de-duplicates, and returns them for display. If the model
answered without citing (rare), we fall back to showing the top retrieved source as provenance.
Citations can only ever reference sources that were actually retrieved — the model never invents
a document or page.

`NOT_FOUND_MESSAGE = "I couldn't find this in the documents."` is defined once and reused by the
pipeline, the tests, and the eval.

---

## 5. File / module layout

```
RAG/
├── app.py                       # Streamlit chat UI (ask · citations · upload · reset)
├── rag/
│   ├── __init__.py
│   ├── config.py                # env + provider switch + constants (Settings dataclass)
│   ├── llm.py                   # provider-agnostic Embeddings + ChatModel + factories + key rotation
│   ├── ingest.py                # PDF -> pages -> chunks with {document_name, page_number, chunk_index}
│   ├── vectorstore.py           # Chroma persistence: add_chunks / query / reset / stats
│   └── qa.py                    # retrieve -> gate -> grounded prompt -> generate -> gate -> cite
├── scripts/
│   ├── make_sample_pdfs.py      # generate the sample corpus PDFs (fpdf2)
│   └── ingest_cli.py            # CLI: ingest a folder of PDFs into Chroma
├── sample_pdfs/                 # generated sample corpus (Meridian Labs docs)
├── eval/
│   ├── golden.yaml              # golden Q&A incl. not-found cases (ground truth)
│   └── eval.py                  # runs golden set, prints pass/fail table
├── tests/
│   ├── test_chunking.py         # chunks preserve document_name + page_number
│   ├── test_retrieval.py        # retrieval returns hits for in-doc queries
│   └── test_not_found.py        # low-relevance query routes to not-found (gate 1, no LLM)
├── screenshots/                 # grounded-answer + not-found proof
├── requirements.txt             # pinned
├── .env.example                 # documents every env var (no secrets)
├── .gitignore                   # .env, .venv, chroma_db/, __pycache__, .streamlit/secrets.toml
├── README.md  · DEPLOY.md · SPEC.md · PLAN.md · progress.md
```

---

## 6. Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` \| `openai` |
| `EMBEDDING_PROVIDER` | (matches `LLM_PROVIDER`) | `gemini` \| `openai` \| `local` |
| `GEMINI_API_KEY` / `GEMINI_API_KEYS` | — | single key, or comma-separated list for rotation |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` | |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | |
| `OPENAI_API_KEY` | — | for the OpenAI path |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | |
| `EMBED_DIM` | `768` (gemini) | truncation dim for Gemini; local=384, openai=1536 |
| `TOP_K` | `4` | retrieved chunks |
| `RELEVANCE_THRESHOLD` | `0.55` (tuned empirically) | gate-1 cutoff on `1 - cosine_distance`; grounded ≥ ~0.67, off-topic ~0.49 on the sample corpus |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `400` / `60` (tuned) | small chunks = topically coherent units → better relevance separation |
| `CHROMA_DIR` | `./chroma_db` | persistence path |

Config reads from `st.secrets` (deployed) falling back to environment / `.env` (local, via
`python-dotenv`). No secrets are ever committed.

---

## 7. Out of scope (explicitly NOT built)

Authentication/users · databases beyond Chroma · background queues/workers · a second service
or API server · OCR / scanned-image PDFs · multi-tenant isolation · streaming token UI (nice-to-have,
not required) · reranking models · conversation-memory retrieval (each question retrieves fresh) ·
observability stacks. If a feature isn't needed to satisfy the Definition of Done, it isn't here.

---

## 8. Stack (pinned in requirements.txt)

`google-genai` · `chromadb` · `pypdf` · `sentence-transformers` (local fallback) · `openai` ·
`streamlit` · `fpdf2` (sample-PDF generation) · `pyyaml` · `python-dotenv` · `numpy` · `pytest`.
Exact versions are frozen from a real install (Phase 3), not guessed.
