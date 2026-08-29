"""Real-time engine: maintains a rolling conversation window and, on each new
message, returns the distribution over who should respond next (excluding the sender).
"""

from __future__ import annotations

from collections import deque

from .parser import Message
from .scorer import ResponderScorer


class RealtimeEngine:
    def __init__(self, scorer: ResponderScorer, window: int = 6):
        self.scorer = scorer
        self.history: deque[Message] = deque(maxlen=window)

    def add_message(self, sender: str, text: str) -> dict[str, float]:
        """Ingest a message; return {role: score} for candidate responders (sums to 1)."""
        self.history.append(Message(None, sender, text))
        candidates = [r.name for r in self.scorer.roles if r.name != sender]
        return self.scorer.score(self._query(), candidates)

    def _query(self) -> str:
        return "\n".join(f"{m.sender}: {m.text}" for m in self.history)
