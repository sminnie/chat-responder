"""Bi-encoder responder scorer.

Encodes the conversation once, scores it against cached role-profile embeddings,
and returns a softmax distribution over candidate responders (always sums to 1).
This is the zero-training MVP baseline and the low-latency serving path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Role:
    name: str
    profile: str


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Default encoder: small, fast, L2-normalized embeddings for cosine scoring."""

    # CPU beats MPS/CUDA for this model at batch size 1: the transfer and dispatch
    # overhead dominates six tiny transformer layers.
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        max_seq_length: int = 128,
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = max_seq_length
        self.encode(["warmup"])

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = x / temperature
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class ResponderScorer:
    def __init__(
        self,
        roles: list[Role],
        encoder: Encoder | None = None,
        temperature: float = 0.1,
    ):
        if not roles:
            raise ValueError("roles must be non-empty")
        self.roles = list(roles)
        self.encoder = encoder or SentenceTransformerEncoder()
        self.temperature = temperature
        self._profile_emb = self.encoder.encode([r.profile for r in self.roles])
        self._index = {r.name: i for i, r in enumerate(self.roles)}

    def score(self, query: str, candidates: list[str] | None = None) -> dict[str, float]:
        """Return {role: probability} over candidates; probabilities sum to 1."""
        if candidates is None:
            candidates = [r.name for r in self.roles]
        if not candidates:
            return {}
        unknown = [c for c in candidates if c not in self._index]
        if unknown:
            raise KeyError(f"unknown roles: {unknown}")
        query_emb = self.encoder.encode([query])[0]
        idxs = [self._index[c] for c in candidates]
        sims = self._profile_emb[idxs] @ query_emb
        probs = _softmax(sims, self.temperature)
        return {c: float(p) for c, p in zip(candidates, probs)}
