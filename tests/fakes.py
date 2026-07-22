"""Deterministic offline fakes for the provider layer.

FakeEmbeddings gives controllable cosine geometry: texts sharing a keyword map to
the same axis (relevance ~1.0); texts with no shared keyword are orthogonal
(relevance ~0.0). This lets tests steer retrieval above/below the threshold
without any network access.
"""
from __future__ import annotations

import numpy as np

from rag.llm import ChatModel, Embeddings


class FakeEmbeddings(Embeddings):
    DIM = 8

    def __init__(self, keyword_axes: dict[str, int]):
        self.keyword_axes = keyword_axes

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.DIM, dtype="float32")
        t = text.lower()
        for kw, axis in self.keyword_axes.items():
            if kw in t:
                v[axis] += 1.0
        if not v.any():
            v[self.DIM - 1] = 1.0  # unknown text lands on its own lonely axis
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @property
    def dim(self) -> int:
        return self.DIM


class StaticChat(ChatModel):
    """Returns a canned reply and records every (system, user) prompt it saw."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


class RaisingChat(ChatModel):
    """Fails the test if the pipeline calls the LLM at all (gate-1 proof)."""

    def generate(self, system: str, user: str) -> str:
        raise AssertionError("LLM was called — the retrieval gate should have short-circuited")
