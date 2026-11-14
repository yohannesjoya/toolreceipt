"""``ReceiptGuard``: a ``langchain`` agent middleware that checks each final reply before the agent
returns it, and sends the agent back once when the reply claims an action its tool calls do not
back. Install ``toolreceipt[langgraph]``.
"""

from __future__ import annotations

import inspect
from collections.abc import Collection, Iterable, Sequence
from typing import Any

try:
    from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langgraph.runtime import Runtime
except ImportError as exc:  # pragma: no cover - exercised in a subprocess
    from toolreceipt.receipts import _LANGGRAPH_DEPS, _MISSING_LANGGRAPH

    if (exc.name or "").partition(".")[0] in _LANGGRAPH_DEPS:
        raise ImportError(_MISSING_LANGGRAPH) from exc
    raise

from toolreceipt.claims import Claim, ClaimExtractor, ToolSpec, normalize_tool_specs
from toolreceipt.langgraph import NOTE_NAME, final_reply, ledger_from_messages
from toolreceipt.receipt import Receipt, _format_observed
from toolreceipt.receipts import acheck, check

__all__ = ["ReceiptGuard"]

_LINES = {
    "not_called": "`{tool}` was not called.",
    "tool_failed": "`{tool}` failed.",
    "args_mismatch": "`{tool}` ran with {observed}, not what the reply says.",
}


def _spec(tool: Any) -> Any:
    """A plain function as ``(name, first docstring line)``; anything else as given.
    ``create_agent`` accepts plain functions, which ``normalize_tool_specs`` does not read."""
    if callable(tool) and not hasattr(tool, "name") and hasattr(tool, "__name__"):
        doc = inspect.getdoc(tool) or ""
        return (tool.__name__, doc.splitlines()[0] if doc else "")
    return tool


def _is_note(message: Any) -> bool:
    return isinstance(message, HumanMessage) and message.name == NOTE_NAME


def _notes_this_turn(messages: Sequence[BaseMessage]) -> int:
    """Notes since the last human message that is not a note."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            if not _is_note(message):
                break
            count += 1
    return count


def _line(claim: Claim) -> str:
    template = _LINES.get(claim.reason or "", _LINES["not_called"])
    observed = _format_observed(claim.observed) if claim.observed else "other arguments"
    return f'- "{claim.text}": ' + template.format(tool=claim.tool, observed=observed)


def _note(receipt: Receipt) -> str:
    """One line per false claim, worded by reason. It never asks for a tool call, so a
    call that ran is not repeated, and it carries no tool result or error text."""
    lines = "\n".join(_line(c) for c in receipt.claimed_but_never_ran)
    return (
        "Your last reply says things the tool calls in this turn do not show:\n"
        f"{lines}\n\n"
        "Rewrite the reply so it says only what the tool calls show. Do not call tools to fix this."
    )


def _stored(receipt: Receipt) -> dict[str, Any]:
    """The receipt without its ledger. The outcome rides on a message that UIs render and
    checkpointers keep, so it carries no tool arguments or results; the messages hold them."""
    data = receipt.to_dict()
    data.pop("ledger", None)
    data["ran_but_unreported"] = [
        {"tool": e["tool"], "status": e["status"], "call_id": e["call_id"]}
        for e in data["ran_but_unreported"]
    ]
    return data


class ReceiptGuard(AgentMiddleware):
    """Checks each final reply against this turn's tool calls. When the receipt is not ok,
    it adds a user message named ``toolreceipt`` and sends the agent back, at most
    ``max_retries`` times per user turn. The judged reply carries the outcome in
    ``response_metadata["toolreceipt"]``: ``outcome`` (``passed``, ``fixed`` or
    ``unresolved``), ``retries`` and the ``receipt``. It never blocks or rewrites a reply.
    """

    def __init__(
        self,
        tools: Iterable[Any],
        *,
        max_retries: int = 1,
        extractor: ClaimExtractor | None = None,
        ignore_tools: Collection[str] = (),
        llm_blocks: bool = False,
        check_args: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(f"max_retries must be a whole number, 0 or more, got {max_retries!r}")
        # ``AgentMiddleware.tools`` registers extra tools with the agent, so keep this private.
        self._known: list[ToolSpec] = normalize_tool_specs(_spec(t) for t in tools)
        self._max_retries = max_retries
        self._options: dict[str, Any] = {
            "extractor": extractor,
            "ignore_tools": ignore_tools,
            "llm_blocks": llm_blocks,
            "check_args": check_args,
        }

    def _inputs(self, state: AgentState) -> tuple[AIMessage, str, Any] | None:
        messages = state["messages"]
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        return last, final_reply(messages), ledger_from_messages(messages)

    def _decide(self, state: AgentState, last: AIMessage, receipt: Receipt) -> dict[str, Any]:
        retries = _notes_this_turn(state["messages"])
        if not receipt.ok and retries < self._max_retries:
            return {"messages": [HumanMessage(_note(receipt), name=NOTE_NAME)], "jump_to": "model"}
        if receipt.ok:
            outcome = "fixed" if retries else "passed"
        else:
            outcome = "unresolved"
        metadata = {
            **last.response_metadata,
            "toolreceipt": {"outcome": outcome, "retries": retries, "receipt": _stored(receipt)},
        }
        # Same id, so the reducer replaces the reply in place.
        return {"messages": [last.model_copy(update={"response_metadata": metadata})]}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        inputs = self._inputs(state)
        if inputs is None:
            return None
        last, reply, ledger = inputs
        receipt = check(reply, ledger, known_tools=self._known, **self._options)
        return self._decide(state, last, receipt)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        inputs = self._inputs(state)
        if inputs is None:
            return None
        last, reply, ledger = inputs
        receipt = await acheck(reply, ledger, known_tools=self._known, **self._options)
        return self._decide(state, last, receipt)
