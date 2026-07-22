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
