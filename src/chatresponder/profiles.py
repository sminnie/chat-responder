"""Load role definitions from a JSON file: [{"name": ..., "profile": ...}, ...]."""

from __future__ import annotations

import json
from pathlib import Path

from .scorer import Role


def load_roles(path: str | Path) -> list[Role]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Role(name=item["name"], profile=item["profile"]) for item in data]
