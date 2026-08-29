"""Local Slack/WhatsApp-style chat UI for driving the real-time engine.

Stdlib-only HTTP server: serves a single page and a small JSON API so you can
type messages as any participant and watch the responder distribution update.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .engine import RealtimeEngine

_STATIC = Path(__file__).parent / "static"
_MAX_BODY = 64 * 1024
BOT_NAME = "Responder"


def _bot_text(ranked: list[tuple[str, float]]) -> str:
    if not ranked:
        return "No one else is in this channel yet."
    top, score = ranked[0]
    line = f"{top}, this one looks like yours ({score:.0%} confident)."
    rest = ranked[1:]
    if rest:
        others = ", ".join(f"{n} {s:.0%}" for n, s in rest)
        line += f" Otherwise: {others}."
    return line


class ChatSession:
    """Engine + transcript, guarded by a lock so concurrent requests stay consistent."""

    def __init__(self, engine: RealtimeEngine, top_k: int = 3):
        self._engine = engine
        self._top_k = top_k
        self._lock = threading.Lock()
        self.transcript: list[dict] = []
        self._latencies: list[float] = []

    @property
    def roles(self):
        return self._engine.scorer.roles

    def send(self, sender: str, text: str) -> dict:
        """Post a message and return the bot's recommendation entry."""
        known = {r.name for r in self.roles}
        if sender not in known:
            raise KeyError(sender)
        with self._lock:
            started = time.perf_counter()
            dist = self._engine.add_message(sender, text)
            latency_ms = (time.perf_counter() - started) * 1000
            ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            ranked = ranked[: self._top_k]
            self._latencies.append(latency_ms)
            bot = {
                "sender": BOT_NAME,
                "kind": "bot",
                "text": _bot_text(ranked),
                "suggestions": [{"name": n, "score": s} for n, s in ranked],
                "latency_ms": round(latency_ms, 1),
                "avg_latency_ms": round(
                    sum(self._latencies) / len(self._latencies), 1
                ),
            }
            self.transcript.append(
                {"sender": sender, "kind": "user", "text": text, "suggestions": []}
            )
            self.transcript.append(bot)
        return bot

    def reset(self) -> None:
        with self._lock:
            self._engine.history.clear()
            self.transcript.clear()
            self._latencies.clear()


def _handler_class(session: ChatSession):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the console clean for the transcript
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > _MAX_BODY:
                raise ValueError("invalid body size")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _drain(self) -> None:
            """Consume an ignored body so the keep-alive connection stays in sync."""
            length = min(int(self.headers.get("Content-Length") or 0), _MAX_BODY)
            if length > 0:
                self.rfile.read(length)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(
                    200,
                    (_STATIC / "index.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif self.path == "/api/state":
                self._json(
                    200,
                    {
                        "bot": BOT_NAME,
                        "roles": [
                            {"name": r.name, "profile": r.profile} for r in session.roles
                        ],
                        "messages": session.transcript,
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                if self.path == "/api/message":
                    payload = self._read_json()
                    sender = str(payload.get("sender", "")).strip()
                    text = str(payload.get("text", "")).strip()
                    if not sender or not text:
                        self._json(400, {"error": "sender and text are required"})
                        return
                    self._json(200, session.send(sender, text))
                elif self.path == "/api/reset":
                    self._drain()
                    session.reset()
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "not found"})
            except KeyError as exc:
                self._json(400, {"error": f"unknown sender: {exc.args[0]}"})
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "malformed request"})

    return Handler


def create_server(
    session: ChatSession, host: str = "127.0.0.1", port: int = 8000
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler_class(session))
