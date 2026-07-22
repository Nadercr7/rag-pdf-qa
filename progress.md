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
