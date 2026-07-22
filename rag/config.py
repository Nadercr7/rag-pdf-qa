"""Configuration + the provider switch.

Every tunable lives here, resolved in this order:
  1. process environment / local ``.env``  (local dev, and Hugging Face Spaces secrets)
  2. ``st.secrets``                          (Streamlit Community Cloud)
  3. documented defaults

Secrets are never printed: ``Settings.__repr__`` masks all keys.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load a local .env from the repo root regardless of the current working directory.
# On a deployed host there is usually no .env — this is then a harmless no-op.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv is a pinned dep; stay defensive anyway
    pass

# The exact, contractual refusal. Defined ONCE and reused by qa.py, the tests and the eval.
NOT_FOUND_MESSAGE = "I couldn't find this in the documents."

_VALID_LLM = {"gemini", "openai"}
_VALID_EMB = {"gemini", "openai", "local"}
_DEFAULT_EMBED_DIM = {"gemini": 768, "openai": 1536, "local": 384}


def _get(key: str, default: str | None = None) -> str | None:
    """Read a config value: env var first, then Streamlit secrets, then default."""
    val = os.environ.get(key)
    if val not in (None, ""):
        return val
    try:  # st.secrets only exists under a Streamlit runtime; ignore otherwise
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


def _mask(secret: str) -> str:
    if not secret:
        return "<empty>"
    return f"{secret[:4]}..{secret[-4:]}" if len(secret) > 10 else "***"


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-resolved configuration."""

    llm_provider: str            # "gemini" | "openai"
    embedding_provider: str      # "gemini" | "openai" | "local"
    gemini_api_keys: tuple[str, ...]
    gemini_chat_model: str
    gemini_embed_model: str
    openai_api_key: str | None
    openai_chat_model: str
    openai_embed_model: str
    embed_dim: int
    top_k: int
    relevance_threshold: float
    chunk_size: int
    chunk_overlap: int
    chroma_dir: str

    @property
    def embed_model(self) -> str:
        """The embedding model actually in use for the active embedding provider."""
        if self.embedding_provider == "gemini":
            return self.gemini_embed_model
        if self.embedding_provider == "openai":
            return self.openai_embed_model
        return "all-MiniLM-L6-v2"  # local

    @property
    def collection_name(self) -> str:
        # Namespaced by provider + embed model + dim: vectors from different models or
        # dimensions must never share a collection (their spaces are incompatible).
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", self.embed_model).strip("-")
        return f"pdfs_{self.embedding_provider}_{slug}_{self.embed_dim}"

    def __repr__(self) -> str:  # never leak secrets into logs/among evidence
        shown = ", ".join(_mask(k) for k in self.gemini_api_keys[:2])
        more = ", ..." if len(self.gemini_api_keys) > 2 else ""
        return (
            f"Settings(llm_provider={self.llm_provider!r}, "
            f"embedding_provider={self.embedding_provider!r}, "
            f"gemini_keys=[{len(self.gemini_api_keys)}: {shown}{more}], "
            f"gemini_chat_model={self.gemini_chat_model!r}, "
            f"gemini_embed_model={self.gemini_embed_model!r}, "
            f"openai_api_key={_mask(self.openai_api_key or '')}, "
            f"openai_chat_model={self.openai_chat_model!r}, "
            f"embed_dim={self.embed_dim}, top_k={self.top_k}, "
            f"relevance_threshold={self.relevance_threshold}, "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
            f"collection={self.collection_name!r}, chroma_dir={self.chroma_dir!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Resolve and cache the active settings. Call ``get_settings.cache_clear()`` in tests."""
    llm_provider = (_get("LLM_PROVIDER", "gemini") or "gemini").strip().lower()
    embedding_provider = (_get("EMBEDDING_PROVIDER", llm_provider) or llm_provider).strip().lower()

    if llm_provider not in _VALID_LLM:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(_VALID_LLM)}, got {llm_provider!r}")
    if embedding_provider not in _VALID_EMB:
        raise ValueError(
            f"EMBEDDING_PROVIDER must be one of {sorted(_VALID_EMB)}, got {embedding_provider!r}"
        )

    keys_raw = _get("GEMINI_API_KEYS") or _get("GEMINI_API_KEY") or ""
    gemini_api_keys = tuple(k.strip() for k in keys_raw.split(",") if k.strip())

    # all-MiniLM-L6-v2 has a fixed 384 dims; other providers support Matryoshka truncation.
    if embedding_provider == "local":
        embed_dim = 384
    else:
        embed_dim = int(_get("EMBED_DIM", str(_DEFAULT_EMBED_DIM[embedding_provider])))

    return Settings(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        gemini_api_keys=gemini_api_keys,
        gemini_chat_model=_get("GEMINI_CHAT_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
        gemini_embed_model=_get("GEMINI_EMBED_MODEL", "gemini-embedding-001") or "gemini-embedding-001",
        openai_api_key=_get("OPENAI_API_KEY"),
        openai_chat_model=_get("OPENAI_CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        openai_embed_model=_get("OPENAI_EMBED_MODEL", "text-embedding-3-small") or "text-embedding-3-small",
        embed_dim=embed_dim,
        top_k=int(_get("TOP_K", "4")),
        # Tuned on the sample corpus: grounded >= ~0.67, off-topic ~0.49 (see progress.md).
        relevance_threshold=float(_get("RELEVANCE_THRESHOLD", "0.55")),
        # Small chunks = topically coherent units -> markedly better relevance separation.
        chunk_size=int(_get("CHUNK_SIZE", "400")),
        chunk_overlap=int(_get("CHUNK_OVERLAP", "60")),
        chroma_dir=_resolve_dir(_get("CHROMA_DIR", "./chroma_db") or "./chroma_db"),
    )


def _resolve_dir(raw: str) -> str:
    """Anchor a relative CHROMA_DIR to the repo root so every entrypoint (CLI, tests,
    Streamlit, eval) uses the same store regardless of the current working directory."""
    p = Path(raw)
    return str(p if p.is_absolute() else (_REPO_ROOT / p).resolve())
