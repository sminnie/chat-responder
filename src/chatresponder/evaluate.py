"""Evaluate a scorer against held-out examples using next-responder ground truth."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import Example
from .scorer import ResponderScorer


@dataclass
class EvalResult:
    n: int
    top1_accuracy: float
    recall_at_2: float

    def as_table(self) -> str:
        return (
            f"examples      : {self.n}\n"
            f"top-1 accuracy: {self.top1_accuracy:.3f}\n"
            f"recall@2      : {self.recall_at_2:.3f}"
        )


def evaluate(scorer: ResponderScorer, examples: list[Example]) -> EvalResult:
    role_names = {r.name for r in scorer.roles}
    top1 = 0
    top2 = 0
    scored = 0
    for ex in examples:
        if ex.responder not in role_names:
            continue
        candidates = [r.name for r in scorer.roles]
        dist = scorer.score(ex.query_text(), candidates)
        ranked = sorted(dist, key=dist.get, reverse=True)
        scored += 1
        if ranked[:1] == [ex.responder]:
            top1 += 1
        if ex.responder in ranked[:2]:
            top2 += 1
    if scored == 0:
        return EvalResult(0, 0.0, 0.0)
    return EvalResult(scored, top1 / scored, top2 / scored)
