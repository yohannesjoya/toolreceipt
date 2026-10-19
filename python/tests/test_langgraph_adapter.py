import threading
import uuid

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException, tool

from tests.helpers_graph import build_fake_graph
from toolreceipt import Receipts
from toolreceipt.langgraph import (
    ReceiptCallbackHandler,
    final_reply,
    ledger_from_messages,
    merge_config,
    tool_specs_from_invocation_params,
)
from toolreceipt.ledger import Ledger


@tool
def issue_refund(order_id: str) -> str:
    """Refund an order."""
    return "refunded"


@tool
def send_email(to: str) -> str:
    """Send an email to the customer."""
    raise RuntimeError("SMTP gateway timeout")


@tool
def create_ticket(title: str) -> str:
    """Create a support ticket."""
    return "TCK-1"


CALLS = AIMessage(
    content="",
    tool_calls=[
        {"name": "issue_refund", "args": {"order_id": "4821"}, "id": "c1"},
        {"name": "send_email", "args": {"to": "a@b.c"}, "id": "c2"},
    ],
)
FINAL = AIMessage(content="Refund issued and I have emailed you the confirmation.")


def test_end_to_end_fabricated_tool_failed():
    rc = Receipts()
    graph = build_fake_graph([issue_refund, send_email, create_ticket], [CALLS, FINAL])
    result = graph.invoke({"messages": [HumanMessage("refund and email me")]}, config=rc.config())
    receipt = rc.check(final_reply(result))
    assert receipt.verdict == "fabricated"
    bad = receipt.claimed_but_never_ran[0]
    assert bad.tool == "send_email" and bad.reason == "tool_failed"
    assert receipt.confirmed[0].tool == "issue_refund"
    entry = next(e for e in receipt.ledger if e.tool == "send_email")
    assert entry.status == "error"
    assert "SMTP gateway timeout" in (entry.error or "") and entry.call_id == "c2"
    assert set(receipt.known_tools) >= {"issue_refund", "send_email", "create_ticket"}


def test_not_called_detected_from_bound_tools():
    rc = Receipts()
    graph = build_fake_graph(
        [issue_refund, send_email, create_ticket],
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "issue_refund", "args": {"order_id": "1"}, "id": "c1"}],
            ),
            AIMessage(content="I issued the refund and created a ticket for you."),
        ],
    )
    result = graph.invoke({"messages": [HumanMessage("hi")]}, config=rc.config())
    r = rc.check(final_reply(result))
    assert r.verdict == "fabricated"
    assert r.claimed_but_never_ran[0].tool == "create_ticket"
    assert r.claimed_but_never_ran[0].reason == "not_called"


def test_tool_level_handle_tool_error_is_recorded_as_error():
    def _send(to: str) -> str:
        """Send an email to the customer."""
        raise ToolException("SMTP gateway timeout")

    handled = StructuredTool.from_function(_send, name="send_email", handle_tool_error=True)
    rc = Receipts()
    graph = build_fake_graph([issue_refund, handled], [CALLS, FINAL], handle_tool_errors=False)
    graph.invoke({"messages": [HumanMessage("x")]}, config=rc.config())
    assert next(e for e in rc.ledger if e.tool == "send_email").status == "error"


def test_capture_results_false_hides_args_and_result():
    rc = Receipts(capture_results=False)
    graph = build_fake_graph(
        [issue_refund],
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "issue_refund", "args": {"order_id": "1"}, "id": "c1"}],
            ),
            AIMessage(content="done"),
        ],
    )
    graph.invoke({"messages": [HumanMessage("x")]}, config=rc.config())
    e = rc.ledger.entries[0]
    assert e.args == {} and e.result is None and e.status == "ok"


def test_merge_config_shapes():
    h = ReceiptCallbackHandler(Ledger())
    assert merge_config(None, h)["callbacks"] == [h]
    other = ReceiptCallbackHandler(Ledger())
    merged = merge_config({"callbacks": [other], "configurable": {"thread_id": "t1"}}, h)
    assert merged["callbacks"] == [other, h] and merged["configurable"] == {"thread_id": "t1"}
    mgr = BaseCallbackManager(handlers=[other])
    merged = merge_config({"callbacks": mgr, "configurable": {"thread_id": "t1"}}, h)
    assert h in merged["callbacks"].handlers and merged["configurable"] == {"thread_id": "t1"}


def test_tool_specs_from_invocation_params_both_shapes():
    params = {
        "tools": [
            {"type": "function", "function": {"name": "a", "description": "A"}},
            {"name": "b", "description": "B", "input_schema": {}},
        ]
    }
    assert tool_specs_from_invocation_params(params) == [("a", "A"), ("b", "B")]


def test_ledger_from_messages_pairs_and_missing_result():
    msgs = [
        HumanMessage("x"),
        CALLS,
        ToolMessage(content="refunded", tool_call_id="c1", name="issue_refund"),
        FINAL,
    ]
    ledger = ledger_from_messages(msgs)
    by_tool = {e.tool: e for e in ledger}
    assert by_tool["issue_refund"].status == "ok"
    assert by_tool["issue_refund"].call_id == "c1"
    assert by_tool["issue_refund"].started_at is None
    assert by_tool["send_email"].status == "error" and by_tool["send_email"].error == "no result"


def test_ledger_from_messages_list_content_flattened():
    msgs = [
        CALLS,
        ToolMessage(
            content=[{"type": "text", "text": "refunded"}],
            tool_call_id="c1",
            name="issue_refund",
            status="success",
        ),
    ]
    assert ledger_from_messages(msgs).entries[0].result == "refunded"


def test_final_reply_edge_cases():
    assert final_reply({"messages": [HumanMessage("x"), CALLS, FINAL]}) == FINAL.content
    assert final_reply([HumanMessage("x"), CALLS]) == ""
    assert final_reply({"messages": []}) == ""


def test_handler_thread_safety_and_flush():
    ledger = Ledger()
    h = ReceiptCallbackHandler(ledger)

    def worker(i):
        rid = uuid.uuid4()
        h.on_tool_start(
            {"name": f"t{i % 3}"}, "", run_id=rid, inputs={"i": i}, tool_call_id=f"c{i}"
        )
        if i % 2:
            h.on_tool_end(f"ok{i}", run_id=rid)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    h.flush()
    assert len(ledger) == 50
    assert sum(1 for e in ledger if e.status == "error" and e.error == "no result") == 25


def test_on_tool_end_command_output_uses_matching_tool_message():
    from langgraph.types import Command

    ledger = Ledger()
    h = ReceiptCallbackHandler(ledger)
    rid = uuid.uuid4()
    h.on_tool_start({"name": "send_email"}, "", run_id=rid, inputs={}, tool_call_id="c2")
    other = ToolMessage(content="fine", tool_call_id="c9", status="success")
    failed = ToolMessage(content="SMTP gateway timeout", tool_call_id="c2", status="error")
    h.on_tool_end(Command(update={"messages": [other, failed]}), run_id=rid)
    entry = ledger.entries[0]
    assert entry.status == "error" and entry.error == "SMTP gateway timeout"
    assert entry.call_id == "c2"
