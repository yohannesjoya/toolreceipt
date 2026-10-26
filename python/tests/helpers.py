"""Shared test helpers: JSON fixture loading."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
DEMO = Path(__file__).resolve().parents[2] / "demo"


def load_cases(subdir: str) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(file stem, parsed json)`` for every fixture in ``subdir``, sorted by stem."""
    paths = sorted((FIXTURES / subdir).glob("*.json"))
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def demo_tool_specs(mode: str = "pressure") -> list[tuple[str, str]]:
    """``(name, description)`` of the demo support desk's tools, loaded without the demo venv."""
    path = DEMO / "scenarios" / "support_desk.py"
    spec = importlib.util.spec_from_file_location("_support_desk", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [(t.name, t.description) for t in module.build_tools(mode)]
