import json
import threading
import urllib.error
import urllib.request

import pytest

from chatresponder.engine import RealtimeEngine
from chatresponder.scorer import ResponderScorer, Role
from chatresponder.ui import ChatSession, create_server

from .fake_encoder import FakeEncoder

ROLES = [
    Role("Alice", "Team lead who answers technical questions and approves plans."),
    Role("Bob", "Backend engineer who debugs APIs and fixes code."),
    Role("Carol", "Supportive teammate who celebrates wins and says thanks."),
]


@pytest.fixture
def base_url():
    scorer = ResponderScorer(ROLES, encoder=FakeEncoder())
    session = ChatSession(RealtimeEngine(scorer, window=4))
    server = create_server(session, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def test_index_serves_html(base_url):
    with urllib.request.urlopen(base_url + "/") as response:
        assert b"<html" in response.read()


def test_state_lists_roles_and_transcript(base_url):
    _post(base_url + "/api/message", {"sender": "Alice", "text": "deploy is broken"})
    with urllib.request.urlopen(base_url + "/api/state") as response:
        state = json.load(response)
    assert [r["name"] for r in state["roles"]] == ["Alice", "Bob", "Carol"]
    assert state["bot"] == "Responder"
    kinds = [m["kind"] for m in state["messages"]]
    assert kinds == ["user", "bot"]
    assert state["messages"][0]["text"] == "deploy is broken"


def test_bot_replies_with_ranked_suggestions_excluding_sender(base_url):
    bot = _post(base_url + "/api/message", {"sender": "Alice", "text": "api bug"})
    names = [s["name"] for s in bot["suggestions"]]
    scores = [s["score"] for s in bot["suggestions"]]
    assert bot["kind"] == "bot" and bot["sender"] == "Responder"
    assert names[0] in bot["text"]
    assert "Alice" not in names
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) == pytest.approx(1.0)
    assert bot["latency_ms"] >= 0
    assert bot["avg_latency_ms"] == pytest.approx(bot["latency_ms"])


def test_unknown_sender_is_rejected(base_url):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base_url + "/api/message", {"sender": "Mallory", "text": "hi"})
    assert exc.value.code == 400


def test_reset_clears_transcript(base_url):
    _post(base_url + "/api/message", {"sender": "Bob", "text": "shipped it"})
    _post(base_url + "/api/reset", {})
    with urllib.request.urlopen(base_url + "/api/state") as response:
        assert json.load(response)["messages"] == []
