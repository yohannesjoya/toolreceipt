"""Shared test helpers: JSON fixture loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def load_cases(subdir: str) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(file stem, parsed json)`` for every fixture in ``subdir``, sorted by stem."""
    paths = sorted((FIXTURES / subdir).glob("*.json"))
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
