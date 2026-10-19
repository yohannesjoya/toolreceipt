"""The README quickstart is executed, so the documented snippet cannot rot."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tests.helpers_graph import build_fake_graph

README = Path(__file__).resolve().parents[2] / "README.md"
MARKER = "# graph = <your compiled LangGraph>"


@tool
def issue_refund(order_id: str) -> str:
    """Refund an order."""
    return "refunded"


@tool
def send_email(to: str) -> str:
    """Send an email to the customer."""
    raise RuntimeError("SMTP gateway timeout")


def _quickstart_block() -> str:
    text = README.read_text(encoding="utf-8")
    section = re.search(r"^## Quickstart\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert section, "README has no '## Quickstart' section"
    block = re.search(r"```python\n(.*?)```", section.group(1), re.S)
    assert block, "Quickstart has no python block"
    return block.group(1)


def _pressure_graph():
    calls = AIMessage(
        content="",
        tool_calls=[
            {"name": "issue_refund", "args": {"order_id": "4821"}, "id": "c1"},
            {"name": "send_email", "args": {"to": "a@b.c"}, "id": "c2"},
        ],
    )
    final = AIMessage(content="Refund issued and I have emailed you the confirmation.")
    return build_fake_graph([issue_refund, send_email], [calls, final])


def test_quickstart_runs_and_flags_the_fabricated_email():
    code = _quickstart_block()
    assert MARKER in code
    code = code.replace(MARKER, "graph = _graph")
    namespace = {"_graph": _pressure_graph()}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README.md#quickstart", "exec"), namespace)  # noqa: S102
    receipt = namespace["receipt"]
    assert receipt.verdict == "fabricated"
    assert not receipt.ok
    assert [c.tool for c in receipt.claimed_but_never_ran] == ["send_email"]
    printed = out.getvalue().splitlines()
    assert printed[0] == "fabricated"
    assert printed[1].startswith("1 claimed action never ran: send_email")
