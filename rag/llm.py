"""Provider-agnostic LLM + embeddings, behind two tiny interfaces.

Nothing else in the codebase knows which provider is active — it asks
``get_embeddings()`` / ``get_chat_model()`` and gets an object implementing
``Embeddings`` / ``ChatModel``. Switching provider is a single env var (see config.py).

Providers:
  * gemini  — gemini-2.5-flash + gemini-embedding-001 (768-d, L2-normalized). FREE default.
              Rotates across a pool of API keys and retries the next key on rate limits.
  * openai  — chat.completions + text-embedding-3-small (first-class; the client's target).
  * local   — sentence-transformers all-MiniLM-L6-v2 (384-d). Embeddings only, lazy import
              so the core app never pulls in torch.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from .config import Settings, get_settings


# ─────────────────────────────── interfaces ────────────────────────────────
class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus chunks (asymmetric providers use a document task type)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (asymmetric providers use a query task type)."""

    @property
    @abstractmethod
    def dim(self) -> int: ...


class ChatModel(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Return the model's text answer for a (system, user) prompt pair."""


# ─────────────────────────────── helpers ───────────────────────────────────
def _l2_normalize(values: Sequence[float]) -> list[float]:
    """L2-normalize a vector. REQUIRED for Gemini embeddings truncated below 3072 dims."""
    v = np.asarray(values, dtype="float32")
    n = float(np.linalg.norm(v))
    return v.tolist() if n == 0.0 else (v / n).tolist()


def _is_transient(err: Exception) -> bool:
    """True for rate-limit / overloaded / transient server errors worth retrying on another key."""
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if code in (429, 500, 503):
        return True
    msg = str(err).upper()
    needles = ("RESOURCE_EXHAUSTED", "429", "RATE LIMIT", "QUOTA", "UNAVAILABLE",
               "OVERLOADED", "DEADLINE_EXCEEDED", "503")
    return any(n in msg for n in needles)


def _looks_like_batch_limit(err: Exception) -> bool:
    msg = str(err).upper()
    return "INVALID_ARGUMENT" in msg or "BATCH" in msg or "ONLY SUPPORTS" in msg


# ───────────────────────── Gemini key-rotation pool ────────────────────────
class _GeminiPool:
    """Holds one client per API key; round-robins to spread free-tier RPM, and
    retries the next key on transient/rate-limit errors."""

    def __init__(self, keys: tuple[str, ...]):
        if not keys:
            raise RuntimeError(
                "No Gemini API key configured. Set GEMINI_API_KEY or GEMINI_API_KEYS."
            )
        from google import genai

        self._clients = [genai.Client(api_key=k) for k in keys]
        self._i = 0

    def _rotate(self) -> None:
        self._i = (self._i + 1) % len(self._clients)

    def call(self, fn: Callable):
        n = len(self._clients)
        attempts = max(n, 3)
        last_err: Exception | None = None
        for a in range(attempts):
            client = self._clients[self._i]
            try:
                result = fn(client)
                self._rotate()  # spread the next call onto a different key
                return result
            except Exception as err:  # noqa: BLE001 - decide by classification below
                last_err = err
                if _is_transient(err):
                    self._rotate()
                    if a + 1 >= n:            # cycled every key once — brief backoff
                        time.sleep(1.0)
                    continue
                raise
        assert last_err is not None
        raise last_err


@lru_cache(maxsize=4)
def _gemini_pool_for(keys: tuple[str, ...]) -> _GeminiPool:
    return _GeminiPool(keys)


# ─────────────────────────────── Gemini ────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        self.s = settings
        self.pool = _gemini_pool_for(settings.gemini_api_keys)

    def _call_embed(self, contents: Sequence[str], task_type: str):
        from google.genai import types

        cfg = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.s.embed_dim
        )
        return self.pool.call(
            lambda c: c.models.embed_content(
                model=self.s.gemini_embed_model, contents=list(contents), config=cfg
            )
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        out: list[list[float]] = []
        batch = 100
        for i in range(0, len(texts), batch):
            group = texts[i : i + batch]
            try:
                resp = self._call_embed(group, "RETRIEVAL_DOCUMENT")
                out.extend(_l2_normalize(e.values) for e in resp.embeddings)
            except Exception as err:  # noqa: BLE001
                if len(group) == 1 or not _looks_like_batch_limit(err):
                    raise
                for t in group:  # model rejected the batch — fall back to one-at-a-time
                    resp = self._call_embed([t], "RETRIEVAL_DOCUMENT")
                    out.append(_l2_normalize(resp.embeddings[0].values))
        return out

    def embed_query(self, text: str) -> list[float]:
        resp = self._call_embed([text], "RETRIEVAL_QUERY")
        return _l2_normalize(resp.embeddings[0].values)

    @property
    def dim(self) -> int:
        return self.s.embed_dim


class GeminiChat(ChatModel):
    def __init__(self, settings: Settings):
        self.s = settings
        self.pool = _gemini_pool_for(settings.gemini_api_keys)

    def generate(self, system: str, user: str) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.1)
        resp = self.pool.call(
            lambda c: c.models.generate_content(
                model=self.s.gemini_chat_model, contents=user, config=cfg
            )
        )
        return (resp.text or "").strip()


# ─────────────────────────────── OpenAI ────────────────────────────────────
class OpenAIEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.s = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=self.s.openai_embed_model, input=list(texts), dimensions=self.s.embed_dim
        )
        # text-embedding-3-* returns unit-normalized vectors (also when truncated via `dimensions`).
        return [d.embedding for d in resp.data]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    @property
    def dim(self) -> int:
        return self.s.embed_dim


class OpenAIChat(ChatModel):
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.s = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.s.openai_chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()


# ─────────────────────────── local (embeddings) ────────────────────────────
class LocalEmbeddings(Embeddings):
    """sentence-transformers all-MiniLM-L6-v2. Lazy import keeps torch out of the core."""

    _model = None  # process-wide singleton

    def __init__(self, settings: Settings):
        self.s = settings

    def _get_model(self):
        if LocalEmbeddings._model is None:
            from sentence_transformers import SentenceTransformer

            LocalEmbeddings._model = SentenceTransformer("all-MiniLM-L6-v2")
        return LocalEmbeddings._model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        emb = self._get_model().encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in np.asarray(emb, dtype="float32")]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    @property
    def dim(self) -> int:
        return 384


# ─────────────────────────────── factories ─────────────────────────────────
def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.embedding_provider == "gemini":
        return GeminiEmbeddings(s)
    if s.embedding_provider == "openai":
        return OpenAIEmbeddings(s)
    if s.embedding_provider == "local":
        return LocalEmbeddings(s)
    raise ValueError(f"Unknown embedding_provider: {s.embedding_provider!r}")


def get_chat_model(settings: Settings | None = None) -> ChatModel:
    s = settings or get_settings()
    if s.llm_provider == "gemini":
        return GeminiChat(s)
    if s.llm_provider == "openai":
        return OpenAIChat(s)
    raise ValueError(f"Unknown llm_provider: {s.llm_provider!r}")
