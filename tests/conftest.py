"""Shared fixtures. The whole suite is offline: no API keys, no network."""
from __future__ import annotations

import pytest

from rag.config import Settings
from tests.fakes import FakeEmbeddings


@pytest.fixture
def make_settings(tmp_path):
    """Factory for isolated Settings pointing at a per-test Chroma directory."""

    def _make(**over) -> Settings:
        base = dict(
            llm_provider="gemini",
            embedding_provider="local",
            gemini_api_keys=(),
            gemini_chat_model="test-chat",
            gemini_embed_model="test-embed",
            openai_api_key=None,
            openai_chat_model="test-chat",
            openai_embed_model="test-embed",
            embed_dim=FakeEmbeddings.DIM,
            top_k=4,
            relevance_threshold=0.55,
            chunk_size=400,
            chunk_overlap=60,
            chroma_dir=str(tmp_path / "chroma"),
        )
        base.update(over)
        return Settings(**base)

    return _make
