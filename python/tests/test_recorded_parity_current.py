"""The TypeScript parity recording (PRD 06 §10.2) must match the current Python package.

``fixtures/recorded/parity.json`` and ``fixtures/demo/tools.json`` are written only by
``scripts/record_parity.py``. A Python behaviour change re-records them; the TypeScript suite
then shows whether the port has to follow.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tests.helpers import FIXTURES, demo_tool_specs

ROOT = FIXTURES.parent
RE_RECORD = "re-record with python/.venv/bin/python scripts/record_parity.py"


def _script():  # type: ignore[no-untyped-def]
    path = ROOT / "scripts" / "record_parity.py"
    spec = importlib.util.spec_from_file_location("_record_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path):  # type: ignore[no-untyped-def]
    return json.loads(path.read_text(encoding="utf-8"))


def test_demo_tools_fixture_is_current():
    expected = [{"name": n, "description": d} for n, d in demo_tool_specs("honest")]
    assert _load(FIXTURES / "demo" / "tools.json") == expected, RE_RECORD


def test_llm_parity_recording_is_current():
    recorded = _load(FIXTURES / "recorded" / "parity_llm.json")
    assert recorded == _script().build_llm(), RE_RECORD


def test_parity_recording_is_current():
    recorded = _load(FIXTURES / "recorded" / "parity.json")
    tools = _load(FIXTURES / "demo" / "tools.json")
    current = _script().build(tools)
    # The stamps say when and from which version it was recorded; a version bump alone is not stale.
    for key in ("extract", "diff", "check", "messages", "turns"):
        assert recorded[key] == current[key], RE_RECORD
    assert recorded["recorded_at"] and recorded["toolreceipt"]
