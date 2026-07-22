# Progress log

Append-only. What changed + what was *verified*, per milestone. Survives context resets.

## Phase 0 — key de-risk (verified)
- `gemini-2.5-flash` generation → HTTP 200 with real keys (format `AQ.…`, a valid Gemini key format).
- Embeddings: `text-embedding-004` is **404** on these keys; `gemini-embedding-001` works → used instead.
- Env: Python 3.11.15 venv, `uv` 0.11.8, git, `gh` authed as `Nadercr7` (repo scope). No HF token yet.

## M0 — scaffold + dependencies (verified)
- SPEC.md, PLAN.md written (architecture, provider switch, 3-layer grounding/citation/not-found, 12 milestones).
- `rag/` package; `.gitignore` (excludes `.env`, `.venv`, `chroma_db/`); `.env.example` documents every var; `.env` (git-ignored) holds 10 Gemini keys as a rotation list.
- `requirements.txt` pinned & installed into the 3.11 venv via `uv`; `requirements-local.txt` isolates the optional torch/sentence-transformers path (kept out of the deployed core).
- Verified: `import chromadb, google.genai, streamlit, pypdf, openai, yaml, dotenv, fpdf` all OK.
  Pins: google-genai 2.13.0 · chromadb 1.5.9 · pypdf 6.14.2 · openai 2.47.0 · streamlit 1.60.0 · fpdf2 2.8.7 · numpy 2.4.6 · pytest 9.1.1.
- `git init` (main); confirmed `.env` NOT staged.
- ⚠ Deploy note: chromadb needs sqlite ≥ 3.35 → Streamlit Cloud may need the `pysqlite3-binary` swap; HF Spaces (Docker) is fine.

## M1+M2 — config + provider-agnostic LLM layer (verified)
- `rag/config.py`: frozen `Settings`, env→st.secrets→defaults, provider validation, secret-masking repr,
  `NOT_FOUND_MESSAGE` constant, collection namespaced `pdfs_<provider>_<dim>`; local provider forces 384-d.
- `rag/llm.py`: `Embeddings`/`ChatModel` ABCs; Gemini w/ key-rotation pool (round-robin + retry-next-key on
  429/transient, backoff after full cycle), batch embed w/ per-text fallback, task types
  RETRIEVAL_DOCUMENT/RETRIEVAL_QUERY, 768-d truncation + L2 normalization; OpenAI chat+embeddings
  (`dimensions=` truncation); local MiniLM lazy-import. Factories `get_embeddings`/`get_chat_model`.
- Verified live (Gemini): batch doc embed → 2×768-d unit vectors; query 768-d unit; cosine sanity
  0.7786 (relevant) vs 0.5340 (unrelated) → ordering correct; chat generation OK via pool.
- Data point for M8 threshold tuning: relevant ≈ 0.78, same-domain-unrelated ≈ 0.53 (default 0.6 plausible).

## M3 — sample corpus PDFs (verified)
- `scripts/make_sample_pdfs.py`: 4 fictional "Meridian Labs" docs (handbook, IT security, expense/travel,
  benefits), 2 pages each, explicit per-page rendering (auto page break off) so fact→page is deterministic.
- fpdf2 gotcha fixed: full-width `multi_cell` defaults `new_x=RIGHT` → next cell gets zero width; all cells
  now return to LMARGIN.
- Verified with pypdf: every doc = 2 pages; all 8 fact→page probes OK (e.g. "20 days of paid vacation" on
  handbook p.1, "16 weeks" on benefits p.2). These probes mirror eval/golden.yaml citation ground truth.

## M4+M5 — ingest + Chroma vectorstore (verified) + chunking tuned
- `rag/ingest.py`: per-page extraction (1-based) + dependency-free recursive splitter (coarse→fine
  separators, greedy packing, overlap tail); `Chunk{text, document_name, page_number, chunk_index}`,
  id `doc::pN::cM`. Accepts paths or upload streams.
- `rag/vectorstore.py`: PersistentClient, cosine (`configuration={"hnsw":{"space":"cosine"}}`), explicit
  vectors only (provider stays in llm.py), idempotent add by document_name, relevance = 1 − distance.
- `scripts/ingest_cli.py`; `CHROMA_DIR` now anchored to repo root (same store from any CWD).
- Verified live: 4 PDFs ingested; re-run → 0 added (idempotent); retrieval lands correct doc+page.
- **Chunking experiment** (full-page 1000/150 vs 400/60): grounded best 0.69/0.67 vs absent-topic best
  0.65/0.61 at page-size — margin too thin. At 400/60 (16 chunks): grounded 0.677–0.726, off-topic
  (cake recipe) 0.494, near-domain absent 0.649–0.672. Conclusion adopted: CHUNK 400/60,
  RELEVANCE_THRESHOLD 0.55 catches off-topic deterministically; near-domain absent questions are
  intrinsically inseparable by score and are handled by the LLM grounding gate (layer 2) — eval will prove it.

## M6 — QA pipeline (verified)
- `rag/qa.py`: gate 1 (threshold, deterministic, zero LLM calls) → numbered-source grounded prompt
  (rule 4: "mentioning a topic is NOT answering") → gate 2 model refusal → gate 3 exact-string
  normalization; `[Source i]` citation parsing tolerant of `[Sources 1, 2]`/`and`; cited sources
  mapped back to {document, page}; `refusal_stage` recorded for observability.
- Verified live: grounded → "20 days ... [Source 2]" cited handbook p.1; hard-negative (vehicle
  allowance, retrieval 0.67 ABOVE threshold) → exact refusal via stage=model; off-topic (cake) →
  exact refusal via stage=retrieval. Exact-string equality confirmed for both refusals.

## M7 — pytest suite (verified: 14 passed in 2.11s, fully offline)
- `tests/fakes.py`: FakeEmbeddings (keyword→axis cosine geometry), StaticChat (records prompts),
  RaisingChat (fails the test if the LLM is invoked at all).
- test_chunking (5): size respected, nothing lost, overlap tail carried, doc+page metadata exact per
  page, sample-corpus "16 weeks" only in p.2 chunks, unique ids, bad-overlap raises.
- test_retrieval (4): hits with correct doc+page+relevance ordering, idempotent add, empty store, reset.
- test_not_found (5): low-relevance → EXACT refusal with NO LLM call (RaisingChat proof,
  stage=retrieval); drifted model refusal normalized to exact string; citations mapped to doc+page;
  no-marker fallback to top source; parse_citations variants incl. out-of-range.

## M8 — golden eval (verified: 17/17 PASS, first run)
- `eval/golden.yaml`: 12 grounded (each with expected substrings + exact doc+page citation) + 5 not-found:
  vehicle-allowance, pet-insurance, stock-options (vs 401(k) "vesting" text!), contractor-vacation
  (full-time-only policy), off-topic-cooking.
- `eval/eval.py`: self-contained (idempotent ingest), prints config + pass/fail table + refusal stages,
  exit code gates CI-style usage.
- RESULT: 17/17. All 12 grounded cited the correct document+page. 4 hard negatives refused with the
  EXACT string via stage=model (rule-4 grounding works); off-topic refused via stage=retrieval (no LLM).
- Added `Answer.citations` (unique doc+page, order-preserving) after spotting a duplicate "p.1, p.1"
  display on 401k-match; covered by an offline test. pytest now 15 passed.

## M9 — Streamlit UI + screenshots (verified)
- `app.py`: chat with history replay, per-message doc+page citation captions, "Retrieved context"
  expander with relevance scores, multi-PDF upload → idempotent ingest, Reset conversation,
  auto-seed of sample corpus on an empty store, `$`-escaping (Streamlit KaTeX gotcha),
  `@st.cache_resource` backend, visible provider/threshold footer.
- Verified via Playwright driving the real app (dev-only tool, not in requirements):
  screenshots/01_grounded_answer_with_citation.png — "20 days ... [Source 2]" + "Sources:
  meridian_employee_handbook.pdf — p.1"; screenshots/02_not_found_exact_reply.png — vehicle-allowance
  question answered with the exact refusal and no citations. DOM assert: last message == exact string.
