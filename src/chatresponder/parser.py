"""Parse WhatsApp chat exports (iOS and Android formats) into structured messages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

# WhatsApp exports embed invisible LTR/RTL marks around timestamps and names.
_INVISIBLE = re.compile(r"[\u200e\u200f\u202a-\u202e]")

# iOS: [2/29/26, 10:15:32 AM] Alice: Hello
_IOS = re.compile(r"^\[(?P<ts>[^\]]+)\]\s?(?P<sender>[^:]+):\s?(?P<text>.*)$")
# Android: 2/29/26, 10:15 - Alice: Hello
_ANDROID = re.compile(
    r"^(?P<ts>\d{1,2}/\d{1,2}/\d{2,4},\s[^-]+?)\s-\s(?P<sender>[^:]+):\s(?P<text>.*)$"
)

_TS_FORMATS = (
    "%m/%d/%y, %I:%M:%S %p",
    "%m/%d/%Y, %I:%M:%S %p",
    "%m/%d/%y, %I:%M %p",
    "%m/%d/%Y, %I:%M %p",
    "%d/%m/%Y, %H:%M",
    "%d/%m/%y, %H:%M",
    "%m/%d/%y, %H:%M",
)


@dataclass
class Message:
    timestamp: datetime | None
    sender: str
    text: str


def _parse_ts(raw: str) -> datetime | None:
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def parse_whatsapp_export(text: str) -> list[Message]:
    """Parse a raw WhatsApp export. Continuation lines are appended to the prior message."""
    messages: list[Message] = []
    for raw_line in text.splitlines():
        line = _INVISIBLE.sub("", raw_line).rstrip("\n")
        match = _IOS.match(line) or _ANDROID.match(line)
        if match:
            messages.append(
                Message(
                    timestamp=_parse_ts(match.group("ts").strip()),
                    sender=match.group("sender").strip(),
                    text=match.group("text"),
                )
            )
        elif messages:
            prev = messages[-1]
            messages[-1] = Message(prev.timestamp, prev.sender, prev.text + "\n" + line)
    return messages
