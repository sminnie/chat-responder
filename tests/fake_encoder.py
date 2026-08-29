"""Deterministic fake encoder so tests run without downloading models or network."""

from __future__ import annotations

import hashlib

import numpy as np


class FakeEncoder:
    """Maps text to a stable L2-normalized vector via hashing. Keyword-sensitive."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        acc = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode()).digest()
            raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            acc += np.resize(raw, self.dim)
        norm = np.linalg.norm(acc)
        return acc / norm if norm else acc

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])
