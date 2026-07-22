"""Golden-set evaluation over the LIVE pipeline (retrieval + LLM).

Runs every question in eval/golden.yaml through rag.qa.answer() and prints a
pass/fail table with citations and refusal stages as evidence.

  python eval/eval.py          # needs GEMINI key(s) in .env (or the active provider's key)

Self-contained: ingests sample_pdfs/ into the store first (idempotent skip if
already present). Exit code 0 iff every row passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `rag` importable

import yaml

from rag.config import NOT_FOUND_MESSAGE, get_settings
from rag.ingest import chunk_pdf
from rag.llm import get_chat_model, get_embeddings
from rag.qa import Answer, answer
from rag.vectorstore import VectorStore

ROOT = Path(__file__).resolve().parent.parent


def _judge(q: dict, a: Answer) -> tuple[bool, str]:
    """Return (passed, human-readable detail) for one golden question."""
    if q["type"] == "not_found":
        if a.not_found and a.text == NOT_FOUND_MESSAGE:  # verbatim contract
            return True, f"exact refusal (stage={a.refusal_stage})"
        return False, f"expected exact refusal, got: {a.text[:80]!r}"

    # grounded
    if a.not_found:
        return False, f"unexpected refusal (stage={a.refusal_stage})"
    missing = [t for t in q.get("expect_answer_contains", []) if t.lower() not in a.text.lower()]
    if missing:
        return False, f"answer missing {missing}: {a.text[:70]!r}"
    cited = a.citations  # unique (document, page), first-mention order
    exp = q.get("expect_citation")
    if exp and (exp["document"], exp["page"]) not in cited:
        return False, f"expected cite {exp['document']} p.{exp['page']}, got {cited}"
    cites = ", ".join(f"{d} p.{p}" for d, p in cited)
    return True, f"{a.text[:58]!r} -> {cites}"


def main() -> int:
    try:  # Windows consoles may default to a legacy codepage
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    golden = yaml.safe_load((ROOT / "eval" / "golden.yaml").read_text(encoding="utf-8"))
    s = get_settings()
    store, emb, chat = VectorStore(s), get_embeddings(s), get_chat_model(s)

    for pdf in sorted((ROOT / "sample_pdfs").glob("*.pdf")):  # idempotent ingest
        store.add_chunks(chunk_pdf(pdf, settings=s), emb)

    print(
        f"eval config: llm={s.llm_provider} embeddings={s.embedding_provider}/{s.embed_dim}d "
        f"threshold={s.relevance_threshold} chunks={s.chunk_size}/{s.chunk_overlap} "
        f"top_k={s.top_k} | store: {store.count()} chunks\n"
    )

    rows: list[tuple[str, str, bool, str]] = []
    for q in golden["questions"]:
        a = answer(q["question"], store, emb, chat, s)
        ok, detail = _judge(q, a)
        rows.append((q["id"], q["type"], ok, detail))

    id_w = max(len(r[0]) for r in rows)
    line = "-" * (id_w + 88)
    print(f"{'id':<{id_w}}  {'type':<9}  result  detail")
    print(line)
    for rid, rtype, ok, detail in rows:
        print(f"{rid:<{id_w}}  {rtype:<9}  {'PASS' if ok else 'FAIL':<6}  {detail}")
    print(line)

    n_pass = sum(ok for _, _, ok, _ in rows)
    n_nf = sum(1 for _, t, ok, _ in rows if t == "not_found" and ok)
    print(f"{n_pass}/{len(rows)} passed  ({n_nf}/{sum(1 for _, t, _, _ in rows if t == 'not_found')} not-found cases exact)")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
