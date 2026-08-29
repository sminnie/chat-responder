import math

import pytest
from chatresponder.scorer import Role, ResponderScorer

from .fake_encoder import FakeEncoder

ROLES = [
    Role("Alice", "team lead answers questions approves plans"),
    Role("Bob", "backend engineer code bugs api fixes"),
    Role("Carol", "supportive teammate encourages celebrates thanks"),
]


def _scorer():
    return ResponderScorer(ROLES, encoder=FakeEncoder(), temperature=0.1)


def test_distribution_sums_to_one():
    dist = _scorer().score("Alice: can we ship the auth fix today")
    assert set(dist) == {"Alice", "Bob", "Carol"}
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-6)


def test_two_roles_sum_to_one():
    dist = _scorer().score("some message", candidates=["Alice", "Bob"])
    assert set(dist) == {"Alice", "Bob"}
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-6)


def test_empty_candidates_returns_empty():
    assert _scorer().score("hi", candidates=[]) == {}


def test_unknown_role_raises():
    with pytest.raises(KeyError):
        _scorer().score("hi", candidates=["Dave"])


def test_empty_roles_raises():
    with pytest.raises(ValueError):
        ResponderScorer([], encoder=FakeEncoder())
