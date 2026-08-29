"""Replay a WhatsApp export through the real-time engine to simulate live scoring."""

from __future__ import annotations

from .engine import RealtimeEngine
from .parser import Message


def replay(engine: RealtimeEngine, messages: list[Message], top_k: int = 3):
    """Yield (message, suggestions) as if messages arrived live."""
    for msg in messages:
        dist = engine.add_message(msg.sender, msg.text)
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        yield msg, ranked


def format_line(msg: Message, ranked: list[tuple[str, float]]) -> str:
    suggestions = ", ".join(f"{name} {score:.2f}" for name, score in ranked)
    return f"{msg.sender}: {msg.text}\n   -> {suggestions}"
