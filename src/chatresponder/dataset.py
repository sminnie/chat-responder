"""Build training/eval examples from parsed messages.

Each example is a context window plus the label = who actually responded next.
This gives free ground truth from real chat history (no manual labeling needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .parser import Message


@dataclass
class Example:
    context: list[Message] = field(default_factory=list)
    responder: str = ""

    def query_text(self) -> str:
        return "\n".join(f"{m.sender}: {m.text}" for m in self.context)


def build_examples(
    messages: list[Message],
    window: int = 6,
    skip_consecutive: bool = True,
) -> list[Example]:
    """For each message, emit (previous `window` messages -> this message's sender)."""
    examples: list[Example] = []
    for i in range(1, len(messages)):
        responder = messages[i].sender
        if skip_consecutive and responder == messages[i - 1].sender:
            continue
        context = messages[max(0, i - window) : i]
        examples.append(Example(context=list(context), responder=responder))
    return examples


def chronological_split(
    examples: list[Example], eval_fraction: float = 0.2
) -> tuple[list[Example], list[Example]]:
    """Split chronologically so eval always follows train (no leakage)."""
    cut = int(len(examples) * (1 - eval_fraction))
    return examples[:cut], examples[cut:]
