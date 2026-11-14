"""ReceiptGuard against the real ``langchain`` agent, driven by a scripted fake chat model."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver

from toolreceipt import check, check_messages
from toolreceipt.langchain import ReceiptGuard
from toolreceipt.langgraph import final_reply, ledger_from_messages

REFUNDS: list[int] = []


@tool
def issue_refund(amount: int) -> str:
    """Issue a refund to the customer."""
    REFUNDS.append(amount)
    return f"refunded {amount} (secret-result-text)"


@tool
def send_email(subject: str) -> str:
    """Send an email to the customer."""
    raise ToolException("SMTP gateway timeout (secret-error-text)")


send_email.handle_tool_error = True


@tool
def create_ticket(title: str) -> str:
    """Create a support ticket."""
    return "TCK-1"


TOOLS = [issue_refund, send_email, create_ticket]


class Scripted(BaseChatModel):
    """Plays one step per call: a string is a reply, a list is tool calls."""

    steps: list[Any]
    seen: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> Scripted:
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        step = self.steps.pop(0)
        message = (
            AIMessage(content="", tool_calls=step)
            if isinstance(step, list)
            else AIMessage(content=step)
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def call(name: str, call_id: str, **args: Any) -> list[dict[str, Any]]:
    return [{"name": name, "args": args, "id": call_id, "type": "tool_call"}]


def run(
    steps: list[Any], *, tools: list[Any] = TOOLS, checkpointer: Any = None, **guard: Any
) -> tuple[dict[str, Any], Scripted]:
    REFUNDS.clear()
    model = Scripted(steps=list(steps), seen=[])
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[ReceiptGuard(tools, **guard)],
        checkpointer=checkpointer,
    )
    return agent.invoke({"messages": [HumanMessage("Refund $32 and email me.")]}), model


def outcome(result: dict[str, Any]) -> dict[str, Any]:
    return result["messages"][-1].response_metadata["toolreceipt"]


def notes(result: dict[str, Any]) -> list[BaseMessage]:
    return [
        m for m in result["messages"] if isinstance(m, HumanMessage) and m.name == "toolreceipt"
    ]


def test_honest_reply_passes_untouched_with_no_extra_call():
    result, model = run([call("issue_refund", "c1", amount=32), "I've refunded $32."])
    assert len(model.seen) == 2
    assert result["messages"][-1].content == "I've refunded $32."
    assert notes(result) == []
    out = outcome(result)
    assert (out["outcome"], out["retries"], out["receipt"]["verdict"]) == ("passed", 0, "clean")


def test_lie_fixed_on_retry():
    result, model = run(
        [
            call("issue_refund", "c1", amount=32),
            "I've refunded $32 and created a ticket for you.",
            "I've refunded $32. I have not created a ticket.",
        ]
    )
    assert len(model.seen) == 3
    [note] = notes(result)
    assert "create_ticket" in note.content and "was not called" in note.content
    assert "Do not call tools" in note.content
    out = outcome(result)
    assert (out["outcome"], out["retries"], out["receipt"]["verdict"]) == ("fixed", 1, "clean")


def test_lie_twice_stops_after_one_note():
    result, model = run(
        [
            call("issue_refund", "c1", amount=32),
            "I've refunded $32 and created a ticket for you.",
            "I've refunded $32 and created a ticket for you.",
        ]
    )
    assert len(model.seen) == 3
    assert len(notes(result)) == 1
    out = outcome(result)
    assert (out["outcome"], out["retries"], out["receipt"]["verdict"]) == (
        "unresolved",
        1,
        "fabricated",
    )


def test_every_false_claim_is_in_the_note():
    result, _ = run(
        [
            call("issue_refund", "c1", amount=32),
            "I've refunded $32, created a ticket and emailed you the confirmation.",
            "I've refunded $32.",
        ]
    )
    [note] = notes(result)
    assert "create_ticket" in note.content and "send_email" in note.content


def test_failed_tool_is_named_as_failed_without_its_error_text():
    result, _ = run(
        [
            call("issue_refund", "c1", amount=32),
            call("send_email", "c2", subject="Refund"),
            "I've refunded $32 and emailed you the confirmation.",
            "I've refunded $32, but the email could not be sent.",
        ]
    )
    [note] = notes(result)
    assert "send_email" in note.content and "failed" in note.content
    assert "secret-error-text" not in note.content and "secret-result-text" not in note.content
    assert outcome(result)["outcome"] == "fixed"


def test_wrong_amount_names_the_observed_value_and_asks_for_no_second_refund():
    result, model = run(
        [
            call("issue_refund", "c1", amount=32),
            "I've refunded $64.",
            "I've refunded $32.",
        ]
    )
    [note] = notes(result)
    assert (
        "issue_refund" in note.content
        and "32" in note.content
        and "was not called" not in note.content
    )
    assert "secret-result-text" not in note.content
    assert REFUNDS == [32]
    assert outcome(result)["outcome"] == "fixed"


def test_retries_reset_for_each_user_turn():
    REFUNDS.clear()
    model = Scripted(
        steps=[
            call("issue_refund", "c1", amount=32),
            "I've refunded $32 and created a ticket.",
            "I've refunded $32.",
            "I've created a ticket for you.",
            "I have not created a ticket.",
        ],
        seen=[],
    )
    agent = create_agent(
        model=model, tools=TOOLS, middleware=[ReceiptGuard(TOOLS)], checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "t"}}
    first = agent.invoke({"messages": [HumanMessage("Refund $32.")]}, config)
    second = agent.invoke({"messages": [HumanMessage("Open a ticket too.")]}, config)
    assert outcome(first)["outcome"] == "fixed"
    assert (outcome(second)["outcome"], outcome(second)["retries"]) == ("fixed", 1)


def test_max_retries_zero_only_records():
    result, model = run(
        [call("issue_refund", "c1", amount=32), "I've refunded $32 and created a ticket."],
        max_retries=0,
    )
    assert len(model.seen) == 2 and notes(result) == []
    assert outcome(result)["outcome"] == "unresolved"


def test_plain_function_tools_are_read():
    def lookup_order(order_id: str) -> str:
        """Look up an order by id."""
        return "delayed"

    def open_ticket(title: str) -> str:
        """Create a support ticket."""
        return "TCK-1"

    result, _ = run(
        [
            call("lookup_order", "c1", order_id="4821"),
            "Your order is delayed, and I've created a ticket for you.",
            "Your order is delayed.",
        ],
        tools=[lookup_order, open_ticket],
    )
    assert outcome(result)["outcome"] == "fixed"
    assert "open_ticket" in notes(result)[0].content


def test_unmatched_claim_does_not_draw_a_note():
    result, model = run(
        [call("issue_refund", "c1", amount=32), "I've refunded $32 and noted your preference."]
    )
    assert len(model.seen) == 2 and notes(result) == []


def test_async_run_matches_sync():
    REFUNDS.clear()
    model = Scripted(
        steps=[
            call("issue_refund", "c1", amount=32),
            "I've refunded $32 and created a ticket.",
            "I've refunded $32.",
        ],
        seen=[],
    )
    agent = create_agent(model=model, tools=TOOLS, middleware=[ReceiptGuard(TOOLS)])
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage("x")]}))
    assert (outcome(result)["outcome"], outcome(result)["retries"]) == ("fixed", 1)


def test_guarded_history_reads_clean_after_the_fact():
    result, _ = run(
        [
            call("issue_refund", "c1", amount=32),
            "I've refunded $32 and created a ticket for you.",
            "I've refunded $32. I have not created a ticket.",
        ]
    )
    messages = result["messages"]
    assert (
        check(final_reply(messages), ledger_from_messages(messages), known_tools=TOOLS).verdict
        == "clean"
    )
    assert (
        check_messages(convert_to_openai_messages(messages), known_tools=TOOLS).verdict == "clean"
    )


def test_guard_keeps_its_tool_list_private():
    # ``AgentMiddleware.tools`` would register the list with the agent a second time.
    assert not getattr(ReceiptGuard(TOOLS), "tools", None)


def test_rejects_negative_retries():
    with pytest.raises(ValueError):
        ReceiptGuard(TOOLS, max_retries=-1)


def test_stored_outcome_carries_no_tool_results():
    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order by id."""
        return "x" * 50_000 + " private-result-text"

    result, _ = run([call("lookup_order", "c1", order_id="4821"), "Your order is on its way."],
                    tools=[lookup_order])
    out = outcome(result)
    assert "ledger" not in out["receipt"]
    assert "private-result-text" not in str(out)
    assert len(str(out)) < 2_000


def test_rejects_non_integer_retries():
    for bad in (1.5, True, "1"):
        with pytest.raises(ValueError):
            ReceiptGuard(TOOLS, max_retries=bad)


def test_missing_extra_names_the_extra():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['langchain'] = None\n"
        "try:\n"
        "    import toolreceipt.langchain\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "toolreceipt[langgraph]" in out.stdout
