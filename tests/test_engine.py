import math

from chatresponder.engine import RealtimeEngine
from chatresponder.scorer import Role, ResponderScorer

from .fake_encoder import FakeEncoder

ROLES = [
    Role("Alice", "team lead answers questions approves plans"),
    Role("Bob", "backend engineer code bugs api fixes"),
    Role("Carol", "supportive teammate encourages celebrates thanks"),
]


def _engine():
    scorer = ResponderScorer(ROLES, encoder=FakeEncoder(), temperature=0.1)
    return RealtimeEngine(scorer, window=6)


def test_sender_excluded_from_candidates():
    dist = _engine().add_message("Alice", "can we ship today?")
    assert "Alice" not in dist
    assert set(dist) == {"Bob", "Carol"}
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-6)


def test_window_is_bounded():
    engine = _engine()
    for i in range(10):
        engine.add_message("Bob", f"msg {i}")
    assert len(engine.history) == 6
