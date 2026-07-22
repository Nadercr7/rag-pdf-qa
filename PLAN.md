# PLAN — ordered, independently verifiable milestones

Each milestone ends with a **Verify** step whose output is captured as evidence, and a git
commit. Progress is appended to `progress.md` so work survives context resets.

The "iterate until green" loop lives in M7–M8.

---

### M0 — Repo scaffold + dependencies  *(Phase 3)*
- Create package layout (`rag/`, `scripts/`, `tests/`, `eval/`, `sample_pdfs/`, `screenshots/`).
- Write `requirements.txt`, install into the 3.11 venv with `uv`, then **freeze exact versions**.
- `.gitignore`, `.env.example`, `.env` (local, real keys, git-ignored).
- **Verify:** `python -c "import chromadb, google.genai, streamlit, pypdf, numpy"` prints OK;
  `pip freeze` shows pinned versions.

### M1 — `rag/config.py` (env + provider switch)
- `Settings` dataclass; provider/model/threshold/paths; `st.secrets`→env→default resolution;
  Gemini key-list parsing; `NOT_FOUND_MESSAGE` constant.
- **Verify:** `python -c "from rag.config import get_settings; print(get_settings())"` shows the
  Gemini defaults and loaded key count (keys masked).

### M2 — `rag/llm.py` (provider-agnostic LLM + embeddings)
- `Embeddings` / `ChatModel` interfaces; Gemini + OpenAI + local implementations; factories;
  Gemini key rotation on 429; Gemini 768-d truncation + **L2 normalization**; correct
  `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types.
- **Verify (live, Gemini):** embed a doc + a query → 768-d, norm≈1.0; `generate()` returns text.
  Print the vector dim, norm, and a one-line generation.

### M3 — Sample corpus PDFs
- `scripts/make_sample_pdfs.py` renders the 4 "Meridian Labs" docs (2 pages each) from the
  designed content, with real page breaks so page citations are exact.
- **Verify:** 4 PDFs in `sample_pdfs/`; `pypdf` reports 2 pages each; a known fact appears on
  the expected page.

### M4 — `rag/ingest.py` (PDF → chunks with page metadata)
- `extract_pages()` (1-based) + `chunk_pages()` (RecursiveCharacterTextSplitter, per-page so
  page numbers are preserved); each chunk carries `{document_name, page_number, chunk_index}`.
- **Verify:** chunk the sample PDFs; assert every chunk has a non-empty `document_name` and a
  `page_number ≥ 1`; print counts per document.

### M5 — `rag/vectorstore.py` (Chroma)
- `PersistentClient`, cosine collection namespaced by provider+dim, `add_chunks()` (explicit
  vectors), idempotent skip by `document_name`, `query()` returning docs+metadata+relevance,
  `reset()`, `stats()`.
- **Verify:** ingest samples; `stats()` shows chunk count; a manual in-doc query returns hits
  with correct `{document, page}` and a plausible relevance.

### M6 — `rag/qa.py` (retrieval + grounded prompt + citations + not-found)
- Orchestrate gates A/B/C from SPEC §4; numbered-source prompt; citation parsing; exact-string
  normalization; return `Answer{text, sources[], not_found}`.
- **Verify:** one grounded question → correct answer + citation (doc+page); one clearly-absent
  question → exact `NOT_FOUND_MESSAGE`.

### M7 — Tests (pytest)
- `test_chunking.py` (metadata preserved), `test_retrieval.py` (in-doc hits),
  `test_not_found.py` (a low-relevance/stub query hits gate 1 → not-found **without** an LLM call).
- **Verify:** `pytest -q` all green; show output.

### M8 — Eval (golden set)
- `eval/golden.yaml`: grounded Qs (with expected doc+page) + ≥1 not-found Q asserting the exact
  refusal (incl. a hard negative). `eval/eval.py` runs them and prints a pass/fail table.
- **Tune** `RELEVANCE_THRESHOLD` here so grounded pass and not-found refuse. Iterate until green.
- **Verify:** `python eval/eval.py` → table, all pass, including the not-found row.

### M9 — `app.py` (Streamlit UI)
- Chat (history via `session_state`), citations panel, multi-file PDF upload → ingest, reset
  button, `@st.cache_resource` backend. Reads secrets/env for keys.
- **Verify:** `streamlit run app.py`; capture screenshots of (1) grounded answer + citation,
  (2) not-found. Save to `screenshots/`.

### M10 — Adversarial review  *(Phase 5)*
- Fresh subagent sees only the diff + SPEC + PLAN; checks every requirement + edge cases
  (not-found exactness, citations, provider switch). Fix correctness gaps; re-run M7–M8.

### M11 — Docs + deploy  *(Phase 6)*
- `README.md` (setup, architecture, grounding/citation/not-found explanation, provider switch,
  eval numbers), `DEPLOY.md`, finalize `.env.example`.
- Public GitHub repo via `gh`; push (no secrets).
- Deploy free live: **HF Spaces** (if an HF token is provided — I can push + verify the URL
  myself) else **Streamlit Community Cloud** with click-by-click steps + one manual step.
- **Verify:** live URL loads and answers a sample question; record the URL.

---

**Commit cadence:** one commit per milestone (`M#: …`). **Evidence:** each Verify's real output
is shown, not asserted. **Definition of Done:** the checklist in the task brief — all boxes true.
