"""Framework-free message input (PRD 04 §7): a ledger and a final reply from a message list.

Reads OpenAI Chat Completions messages (``role``, ``content``, ``tool_calls``,
``tool_call_id``), OpenAI Responses items (``type`` of ``message``, ``function_call``,
``function_call_output``; ``call_id``) and Anthropic Messages (``tool_use`` and
``tool_result`` content blocks) as plain dicts or SDK objects, by key and attribute, with no
SDK imported. Only the last turn is read: the messages after the last ``user`` message.
A plain transcript shape is read too: role/content lines with settled ledger entries
(``{"tool": ..., "status": "ok" | "error", ...}``) at the position where the tool settled
(PRD 05 §7.5); ``toolreceipt.turns`` builds per-turn receipts on the same pieces.
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
]

SUPPORTED = "OpenAI Chat Completions messages, OpenAI Responses items or Anthropic messages"
_TEXT_TYPES = frozenset({"text", "output_text", "input_text"})
_LANGCHAIN_TYPES = frozenset({"ai", "human", "tool", "system"})
# Tool-call shapes this reader knows about but does not read yet (0.3.2): dropping them
# would empty the ledger and make every honest claim ``not_called``.
_UNREAD_ITEM_TYPES = frozenset({
    "custom_tool_call", "custom_tool_call_output", "mcp_call", "computer_call",
    "local_shell_call", "shell_call",
})


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
    """A user message that starts a turn: one with text and no ``tool_result`` block.

    Anthropic returns tool results inside user messages, sometimes with a text block beside
    them, and a tool's screenshot comes back as an image-only user message; both continue
    the run, otherwise the turn's calls would drop out of the ledger and an honest reply
    would read as ``not_called``.
    """
    if _get(message, "role") != "user":
        return False
    content = _get(message, "content")
    if isinstance(content, str):
        return bool(content.strip())
    blocks = _blocks(content)
    if blocks is None:
        return True
    if any(_get(b, "type") == "tool_result" for b in blocks):
        return False
    return bool(_text(content).strip())


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
    item_type = _get(message, "type")
    if item_type in _UNREAD_ITEM_TYPES:
        raise ValueError(
            f"{item_type} items are not supported yet; only function tool calls are read"
        )
    for call in _blocks(_get(message, "tool_calls")) or []:
        call_type = _get(call, "type")
        if call_type is not None and call_type != "function":
            raise ValueError(
                f"tool calls of type {call_type!r} are not supported yet; only function tool "
                "calls are read"
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
    """``(call id, tool name, args)`` for every tool call the message carries: Chat
    Completions ``tool_calls``, a Responses ``function_call`` item, Anthropic ``tool_use``
    blocks. A Responses item's ``id`` is the item id, never the call id."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for call in _blocks(_get(message, "tool_calls")) or []:
        function = _get(call, "function")
        name = _get(function, "name") if function is not None else None
        if isinstance(name, str) and name:
            call_id = str(_get(call, "id") or "")
            out.append((call_id, name, _parse_args(_get(function, "arguments"))))
    if _get(message, "type") == "function_call":
        name = _get(message, "name")
        if isinstance(name, str) and name:
            call_id = str(_get(message, "call_id") or "")
            out.append((call_id, name, _parse_args(_get(message, "arguments"))))
    for block in _blocks(_get(message, "content")) or []:
        name = _get(block, "name")
        if _get(block, "type") == "tool_use" and isinstance(name, str) and name:
            out.append((str(_get(block, "id") or ""), name, coerce_args(_get(block, "input"))))
    return out


def _results(message: Any) -> list[tuple[str, str, bool]]:
    """``(call id, text, is_error)`` for every tool result the message carries: a Chat
    Completions ``tool`` message, a Responses ``function_call_output`` item, Anthropic
    ``tool_result`` blocks (the only shape with an explicit error flag)."""
    out: list[tuple[str, str, bool]] = []
    if _get(message, "role") == "tool":
        call_id = str(_get(message, "tool_call_id") or "")
        out.append((call_id, _text(_get(message, "content")), False))
    if _get(message, "type") == "function_call_output":
        call_id = str(_get(message, "call_id") or "")
        out.append((call_id, _text(_get(message, "output")), False))
    for block in _blocks(_get(message, "content")) or []:
        if _get(block, "type") == "tool_result":
            call_id = str(_get(block, "tool_use_id") or "")
            out.append((call_id, _text(_get(block, "content")), bool(_get(block, "is_error"))))
    return out


def _is_turn(message: Any) -> bool:
    """An assistant message with non-empty text: a Chat Completions ``assistant`` message, a
    Responses ``message`` item with ``role == "assistant"``, an Anthropic assistant message
    with a ``text`` block. A tool-call-only assistant message is not a turn."""
    return _get(message, "role") == "assistant" and bool(_text(_get(message, "content")).strip())


def _shape(message: Any) -> str:
    """``dict(keys: ...)`` or the type name: enough to find the item, no values."""
    if isinstance(message, dict):
        return f"dict(keys: {', '.join(sorted(str(k) for k in message))})"
    return type(message).__name__


def _settled(message: Any) -> LedgerEntry | None:
    """The ``LedgerEntry`` a settled transcript entry records, else ``None``.

    A settled entry is an item whose ``tool`` is a non-empty string and whose ``role`` and
    ``type`` are absent or ``None`` (``_get`` cannot tell the two apart). It is read with the
    ``LedgerEntry.to_dict()`` keys. Two deliberate differences from ``LedgerEntry.from_dict``:
    a ``status`` other than ``ok`` or ``error`` raises, and an ``error`` entry with no
    ``error`` text gets ``NO_RESULT``. A ``tool`` that is present but empty or not a string
    raises rather than being skipped silently.
    """
    tool = _get(message, "tool")
    if tool is None or _get(message, "role") is not None or _get(message, "type") is not None:
        return None
    # Error messages name the shape, never the payload: args, result and error text are
    # customer data (README "Privacy").
    if not isinstance(tool, str) or not tool:
        raise ValueError(
            "a settled transcript entry needs a non-empty string tool, got "
            f"{type(tool).__name__} in {_shape(message)}"
        )
    status = _get(message, "status")
    if status is None:
        status = "ok"
    if status not in ("ok", "error"):
        shown = repr(status) if isinstance(status, str) else type(status).__name__
        raise ValueError(
            f"a settled transcript entry's status must be 'ok' or 'error', got {shown} in "
            f"{_shape(message)}"
        )
    error = _get(message, "error")
    if status == "error" and not error:
        error = NO_RESULT
    call_id = _get(message, "call_id")
    return LedgerEntry(
        tool=tool,
        status=status,
        args=coerce_args(_get(message, "args")),
        result=_get(message, "result"),
        error=error if status == "error" else _get(message, "error"),
        call_id=str(call_id) if call_id is not None else None,
        parent_run_id=_get(message, "parent_run_id"),
        started_at=_get(message, "started_at"),
        ended_at=_get(message, "ended_at"),
    )


def _pair(messages: Iterable[Any]) -> list[LedgerEntry]:
    """One entry per tool call of ``messages``, in call order, paired with its result by id.

    A call with no result is ``status="error", error="no result"``; a result with no call is
    ignored; a call whose id is empty pairs with the next result whose id is empty; a
    repeated id pairs first-in, first-out. Status is ``ok`` for every result except an
    Anthropic ``tool_result`` with ``is_error: true`` (nothing is inferred from result text).
    A settled transcript entry is appended at its position and never pairs. Raises
    ``ValueError`` on an unsupported shape (``_reject_unsupported``).
    """
    entries: list[LedgerEntry] = []
    open_calls: dict[str, deque[LedgerEntry]] = {}
    for message in messages:
        _reject_unsupported(message)
        settled = _settled(message)
        if settled is not None:
            entries.append(settled)
            continue
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
    return entries


def _run_messages(messages: Sequence[Any]) -> list[Any]:
    """The last turn: the messages after the last user message that carries no tool result
    (§7.4); all when there is none."""
    items = list(messages)
    for i in range(len(items) - 1, -1, -1):
        if _is_user(items[i]):
            return items[i + 1 :]
    return items


def ledger_from_messages(messages: Sequence[Any]) -> Ledger:
    """One entry per tool call of the last turn, paired with its result by id.

    A call with no result is ``status="error", error="no result"``; a result with no call is
    ignored; a call whose id is empty pairs with the next result whose id is empty; a
    repeated id pairs first-in, first-out. Status is ``ok`` for every result except an
    Anthropic ``tool_result`` with ``is_error: true`` (nothing is inferred from result text).
    Timestamps are ``None`` unless a settled transcript entry carries them. Raises
    ``ValueError`` on a legacy OpenAI function-call message or a LangChain message object.
    """
    return Ledger(_pair(_run_messages(messages)))


def final_reply(messages: Sequence[Any]) -> str:
    """Text of the last turn's last assistant message when no tool call or settled entry
    follows it or sits in it, else ``""`` (a run that ends on a tool call has no final
    reply)."""
    for message in reversed(_run_messages(messages)):
        if _calls(message) or _settled(message) is not None:
            return ""
        if _get(message, "role") == "assistant":
            return _text(_get(message, "content"))
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
