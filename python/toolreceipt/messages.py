"""Framework-free message input (PRD 04 §7): a ledger and a final reply from a message list.

Reads OpenAI Chat Completions messages (``role``, ``content``, ``tool_calls``,
``tool_call_id``) as plain dicts or SDK objects, by key and attribute, with no SDK imported.
Only the last turn is read: the messages after the last ``user`` message.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Collection, Iterable, Sequence
from typing import Any

from toolreceipt.claims import ClaimExtractor, normalize_tool_specs
from toolreceipt.ledger import NO_RESULT, Ledger, LedgerEntry, coerce_args
from toolreceipt.receipt import Receipt
from toolreceipt.receipts import ToolLike, _acheck, _check

__all__ = [
    "acheck_messages",
    "check_messages",
    "final_reply",
    "ledger_from_messages",
    "run_messages",
]

SUPPORTED = "OpenAI Chat Completions messages"
_TEXT_TYPES = frozenset({"text", "output_text", "input_text"})
_LANGCHAIN_TYPES = frozenset({"ai", "human", "tool", "system"})


def _get(obj: Any, key: str) -> Any:
    """A field by key (mappings) or attribute (SDK objects); ``None`` when missing."""
    if isinstance(obj, dict):
        return obj.get(key)
    get = getattr(obj, "get", None)
    if callable(get) and not hasattr(obj, "__dataclass_fields__"):
        try:
            return get(key)
        except TypeError:
            pass
    return getattr(obj, key, None)


def _blocks(content: Any) -> list[Any] | None:
    if isinstance(content, Sequence) and not isinstance(content, str):
        return list(content)
    return None


def _text(content: Any) -> str:
    """A string as is; a block list as its text blocks joined by newlines; else ``""``."""
    if isinstance(content, str):
        return content
    blocks = _blocks(content)
    if blocks is None:
        return ""
    parts: list[str] = []
    for block in blocks:
        text = _get(block, "text")
        if _get(block, "type") in _TEXT_TYPES and isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _is_user(message: Any) -> bool:
    """A user message that starts a turn (not one made only of ``tool_result`` blocks)."""
    if _get(message, "role") != "user":
        return False
    blocks = _blocks(_get(message, "content"))
    return not blocks or any(_get(b, "type") != "tool_result" for b in blocks)


def _reject_unsupported(message: Any) -> None:
    role = _get(message, "role")
    if role == "function" or _get(message, "function_call") is not None:
        raise ValueError(
            f"legacy OpenAI function_call messages are not supported; pass {SUPPORTED}"
        )
    if role is None and _get(message, "type") in _LANGCHAIN_TYPES:
        raise ValueError(
            "LangChain messages belong to toolreceipt.langgraph.ledger_from_messages; "
            f"toolreceipt.messages reads {SUPPORTED}"
        )


def _parse_args(arguments: Any) -> dict[str, Any]:
    """OpenAI ``arguments`` is a JSON string; unparsable text is kept as ``{"input": ...}``."""
    if isinstance(arguments, str):
        try:
            return coerce_args(json.loads(arguments))
        except ValueError:
            return coerce_args(arguments)
    return coerce_args(arguments)


def _calls(message: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """``(call id, tool name, args)`` for every tool call the message carries."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for call in _blocks(_get(message, "tool_calls")) or []:
        function = _get(call, "function")
        name = _get(function, "name") if function is not None else None
        if isinstance(name, str) and name:
            call_id = str(_get(call, "id") or "")
            out.append((call_id, name, _parse_args(_get(function, "arguments"))))
    return out


def _results(message: Any) -> list[tuple[str, str, bool]]:
    """``(call id, text, is_error)`` for every tool result the message carries."""
    if _get(message, "role") == "tool":
        return [(str(_get(message, "tool_call_id") or ""), _text(_get(message, "content")), False)]
    return []


def run_messages(messages: Sequence[Any]) -> list[Any]:
    """The last turn: the messages after the last user message (§7.4); all when there is none."""
    items = list(messages)
    for i in range(len(items) - 1, -1, -1):
        if _is_user(items[i]):
            return items[i + 1 :]
    return items


def ledger_from_messages(messages: Sequence[Any]) -> Ledger:
    """One entry per tool call of the last turn, paired with its result by id.

    A call with no result is ``status="error", error="no result"``; a result with no call is
    ignored; a call whose id is empty pairs with the next result whose id is empty; a
    repeated id pairs first-in, first-out. Status is ``ok`` for every result (nothing is
    inferred from its text). Timestamps are ``None``. Raises ``ValueError`` on a legacy
    OpenAI function-call message or a LangChain message object.
    """
    entries: list[LedgerEntry] = []
    open_calls: dict[str, deque[LedgerEntry]] = {}
    for message in run_messages(messages):
        _reject_unsupported(message)
        for call_id, name, args in _calls(message):
            entry = LedgerEntry(
                tool=name, status="error", args=args, error=NO_RESULT, call_id=call_id
            )
            entries.append(entry)
            open_calls.setdefault(call_id, deque()).append(entry)
        for call_id, text, is_error in _results(message):
            queue = open_calls.get(call_id)
            if not queue:
                continue
            entry = queue.popleft()
            if is_error:
                entry.error = text
            else:
                entry.status, entry.error, entry.result = "ok", None, text
    return Ledger(entries)


def final_reply(messages: Sequence[Any]) -> str:
    """Text of the last turn's last assistant message when it carries no tool call, else ``""``."""
    for message in reversed(run_messages(messages)):
        if _get(message, "role") == "assistant":
            return "" if _calls(message) else _text(_get(message, "content"))
    return ""


def check_messages(
    messages: Sequence[Any],
    *,
    known_tools: Iterable[ToolLike] = (),
    ignore_tools: Collection[str] = (),
    extractor: ClaimExtractor | None = None,
    llm_blocks: bool = False,
    check_args: bool = True,
) -> Receipt:
    """``check(final_reply(messages), ledger_from_messages(messages), ...)`` with no framework.

    ``known_tools`` is the ``tools`` list sent to the API; without it a tool that was never
    called cannot be reported as ``not_called``.
    """
    return _check(
        final_reply(messages), ledger_from_messages(messages), normalize_tool_specs(known_tools),
        ignore_tools, extractor, llm_blocks, check_args,
    )


async def acheck_messages(
    messages: Sequence[Any],
    *,
    known_tools: Iterable[ToolLike] = (),
    ignore_tools: Collection[str] = (),
    extractor: ClaimExtractor | None = None,
    llm_blocks: bool = False,
    check_args: bool = True,
) -> Receipt:
    """``check_messages`` for async code; see ``acheck``."""
    return await _acheck(
        final_reply(messages), ledger_from_messages(messages), normalize_tool_specs(known_tools),
        ignore_tools, extractor, llm_blocks, check_args,
    )
