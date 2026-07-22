"""Question answering: retrieve -> gate -> grounded generation -> gate -> cite.

Defense-in-depth against hallucination (see SPEC §4):

  Gate 1 (retrieval, deterministic): chunks scoring below RELEVANCE_THRESHOLD are
     dropped; if none survive, the exact NOT_FOUND_MESSAGE is returned **without
     calling the LLM at all**.
  Gate 2 (grounding prompt): the model may answer only from the numbered sources
     and must emit the exact refusal sentence when they don't contain the answer —
     including when a source merely *touches* the topic (the hard-negative case).
  Gate 3 (normalization): any refusal-shaped reply is normalized to the exact
     contractual string, so the "not found" contract holds verbatim.

Citations: sources are numbered [Source i] in the prompt; the model cites inline;
we parse the markers back and surface {document, page} — the model can only ever
cite sources it was actually given.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import NOT_FOUND_MESSAGE, Settings, get_settings
from .llm import ChatModel, Embeddings
from .vectorstore import Retrieved, VectorStore

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions using ONLY the numbered sources provided in the user message.

Rules:
1. Use ONLY information stated in the sources. Never use outside knowledge, and never guess.
2. Cite every fact with its source marker, e.g. [Source 1], placed right after the statement it supports. Cite only source numbers that exist.
3. If the sources do not contain the information needed to answer the question, reply with EXACTLY this sentence and nothing else:
{NOT_FOUND_MESSAGE}
4. A source merely mentioning the topic is NOT the same as it answering the question. If the sources touch the topic but do not state the specific answer, use the exact sentence from rule 3.
5. Be concise and factual. Quote numbers, limits, and durations exactly as written in the sources."""


@dataclass(frozen=True)
class Source:
    """A retrieved chunk as presented to the model (and to the user as a citation)."""

    index: int  # 1-based [Source i] number used in the prompt
    document_name: str
    page_number: int
    relevance: float
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]        # sources the answer actually cites (top-1 fallback)
    retrieved: list[Source]      # every above-threshold source shown to the model
    not_found: bool
    refusal_stage: str | None    # None | "retrieval" (gate 1) | "model" (gates 2+3)


# ────────────────────────────── prompt building ─────────────────────────────
def build_context(sources: list[Source]) -> str:
    blocks = [
        f"[Source {s.index}] (document: {s.document_name}, page: {s.page_number})\n{s.text}"
        for s in sources
    ]
    return "\n\n".join(blocks)


def build_user_prompt(question: str, sources: list[Source]) -> str:
    return f"Sources:\n\n{build_context(sources)}\n\nQuestion: {question}"


# ────────────────────────────── citation parse ──────────────────────────────
_CITE_RE = re.compile(r"\[\s*sources?\s+([^\]]+)\]", re.IGNORECASE)


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Extract cited source indices, in first-mention order, bounded to real sources.
    Tolerates '[Source 1]', '[source 2]', '[Sources 1, 3]', '[Source 1 and 2]'."""
    order: list[int] = []
    for match in _CITE_RE.finditer(text):
        for num in re.findall(r"\d+", match.group(1)):
            i = int(num)
            if 1 <= i <= n_sources and i not in order:
                order.append(i)
    return order


# ────────────────────────────── refusal handling ────────────────────────────
def _as_refusal(text: str) -> bool:
    """True iff the reply is the contractual refusal (tolerating trivial drift)."""
    t = text.strip().strip('"').strip()
    target = NOT_FOUND_MESSAGE.rstrip(".")
    return t.rstrip(".!").lower() == target.lower() or t.lower().startswith(target.lower())


# ─────────────────────────────── the pipeline ───────────────────────────────
def answer(
    question: str,
    store: VectorStore,
    embeddings: Embeddings,
    chat: ChatModel,
    settings: Settings | None = None,
) -> Answer:
    s = settings or get_settings()

    # 1. retrieve
    hits: list[Retrieved] = store.query(embeddings.embed_query(question), k=s.top_k)

    # 2. gate 1 — deterministic relevance threshold (no LLM call on failure)
    kept = [h for h in hits if h.relevance >= s.relevance_threshold]
    if not kept:
        return Answer(
            text=NOT_FOUND_MESSAGE, sources=[], retrieved=[],
            not_found=True, refusal_stage="retrieval",
        )

    sources = [
        Source(
            index=i,
            document_name=h.document_name,
            page_number=h.page_number,
            relevance=h.relevance,
            text=h.text,
        )
        for i, h in enumerate(kept, start=1)
    ]

    # 3. grounded generation
    reply = chat.generate(SYSTEM_PROMPT, build_user_prompt(question, sources))

    # 4. gates 2+3 — model-level refusal, normalized to the exact contract string
    if not reply or _as_refusal(reply):
        return Answer(
            text=NOT_FOUND_MESSAGE, sources=[], retrieved=sources,
            not_found=True, refusal_stage="model",
        )

    # 5. citations — map the model's [Source i] markers back to documents/pages
    cited = parse_citations(reply, n_sources=len(sources))
    if cited:
        used = [sources[i - 1] for i in cited]
    else:
        used = [sources[0]]  # answered without markers: surface best source as provenance

    return Answer(text=reply, sources=used, retrieved=sources, not_found=False, refusal_stage=None)
