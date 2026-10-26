"""LangGraph / LangChain adapter: callback ledger capture, post-hoc ledger, final reply.

This is the only module that imports ``langchain_core``; install ``toolreceipt[langgraph]``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from toolreceipt.claims import ToolSpec
from toolreceipt.ledger import NO_RESULT, Ledger, LedgerEntry, coerce_args, truncate

__all__ = [
    "ReceiptCallbackHandler",
    "final_reply",
    "ledger_from_messages",
    "merge_config",
    "tool_specs_from_invocation_params",
]


def tool_specs_from_invocation_params(params: Mapping[str, Any]) -> list[ToolSpec]:
    """Read bound tools from chat-model ``invocation_params``.

    Accepts the OpenAI shape ``{"type": "function", "function": {"name", "description"}}``
    and the Anthropic shape ``{"name", "description", "input_schema"}``. Entries without a
    string name are skipped.
    """
    tools = params.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, str):
        return []
    specs: list[ToolSpec] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        inner = tool.get("function")
        src: Mapping[str, Any] = inner if isinstance(inner, Mapping) else tool
        name = src.get("name")
        if isinstance(name, str) and name:
            description = src.get("description")
            specs.append(ToolSpec(name, description if isinstance(description, str) else ""))
    return specs


def _tool_message_from(output: Any, call_id: str | None) -> Any:
    """Unwrap a tool output to its ``ToolMessage`` when it carries one.

    A list (content_and_artifact tools) yields its first ``ToolMessage``. A LangGraph
    ``Command`` (duck-typed: an ``update`` mapping with ``messages``) yields the
    ``ToolMessage`` whose ``tool_call_id`` matches ``call_id``, else its first one.
    Anything else is returned unchanged.
    """
    candidates: Any = None
    if isinstance(output, list):
        candidates = output
    else:
        update = getattr(output, "update", None)
        if isinstance(update, Mapping):
            candidates = update.get("messages")
    if not isinstance(candidates, Sequence) or isinstance(candidates, str):
        return output
    messages = [m for m in candidates if isinstance(m, ToolMessage)]
    if not messages:
        return output
    return next((m for m in messages if m.tool_call_id == call_id), messages[0])


def _safe_name(serialized: Any) -> str:
    try:
        return str(serialized.get("name") or "unknown")
    except Exception:
        return "unknown"


class ReceiptCallbackHandler(BaseCallbackHandler):
    """Records every tool call of a run into ``ledger``; thread-safe.

    Entries are appended to the ledger when they close (tool end, tool error or ``flush``).
    """

    # Async callers would otherwise pay one default-executor hop per event; the handler's
    # work is microseconds under its own lock, and sync ``invoke`` is unaffected.
    run_inline = True

    def __init__(
        self, ledger: Ledger, *, capture_results: bool = True, max_result_chars: int = 2000
    ) -> None:
        super().__init__()
        self.ledger = ledger
        self.known_tools: dict[str, str] = {}
        self._capture_results = capture_results
        self._max_result_chars = max_result_chars
        self._in_flight: dict[UUID, LedgerEntry] = {}
        self._lock = threading.Lock()

    def _add_known(self, name: str, description: str) -> None:
        # Caller holds the lock. A non-empty description is never overridden.
        if not self.known_tools.get(name):
            self.known_tools[name] = description

    def _close(self, run_id: UUID) -> LedgerEntry | None:
        # Caller holds the lock.
        entry = self._in_flight.pop(run_id, None)
        if entry is not None:
            entry.ended_at = time.time()
            self.ledger.append(entry)
        return entry

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Callbacks never raise: LangChain would log and swallow the exception, losing data.
        try:
            params = kwargs.get("invocation_params")
            if not isinstance(params, Mapping):
                return
            specs = tool_specs_from_invocation_params(params)
        except Exception:
            return
        with self._lock:
            for name, description in specs:
                self._add_known(name, description)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        description: Any = None
        try:
            serialized = serialized or {}
            name = str(serialized.get("name") or "unknown")
            description = serialized.get("description")
            tool_call_id = kwargs.get("tool_call_id")
            entry = LedgerEntry(
                tool=name,
                args=coerce_args(inputs if inputs is not None else input_str)
                if self._capture_results
                else {},
                call_id=str(tool_call_id) if tool_call_id else str(run_id),
                parent_run_id=str(parent_run_id) if parent_run_id is not None else None,
                started_at=time.time(),
            )
        except Exception:
            # Still open an entry, so the call is never missing from the ledger.
            entry = LedgerEntry(
                tool=_safe_name(serialized), call_id=str(run_id), started_at=time.time()
            )
        with self._lock:
            self._add_known(entry.tool, description if isinstance(description, str) else "")
            self._in_flight[run_id] = entry

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            pending = self._in_flight.get(run_id)
            expected_id = pending.call_id if pending is not None else None
        failed = False
        text: str | None = None
        call_id: str | None = None
        try:
            output = _tool_message_from(output, expected_id)
            if isinstance(output, ToolMessage):
                failed = output.status == "error"
                text = str(output.text)
                call_id = output.tool_call_id or None
            else:
                text = str(output)
        except Exception:
            # The tool returned but its output cannot be read: record it as ran, no result.
            failed, text, call_id = False, None, None
        with self._lock:
            entry = self._close(run_id)
            if entry is None:
                return
            if call_id:
                entry.call_id = call_id
            if failed:
                entry.status = "error"
                entry.error = truncate(text, self._max_result_chars)
            elif self._capture_results:
                entry.result = truncate(text, self._max_result_chars)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            message = repr(error)
        except Exception:
            message = type(error).__name__
        with self._lock:
            entry = self._close(run_id)
            if entry is not None:
                entry.status = "error"
                entry.error = truncate(message, self._max_result_chars)

    def flush(self) -> None:
        """Close every in-flight entry as ``status="error", error="no result"``."""
        with self._lock:
            for run_id in list(self._in_flight):
                entry = self._close(run_id)
                if entry is not None:
                    entry.status = "error"
                    entry.error = NO_RESULT


def merge_config(
    base_config: Mapping[str, Any] | None, handler: BaseCallbackHandler
) -> dict[str, Any]:
    """Shallow copy of ``base_config`` with ``handler`` added to its callbacks.

    A callbacks list gets ``handler`` appended; a ``BaseCallbackManager`` is copied (the
    caller's manager is not mutated) and ``handler`` added as inheritable. Other keys are kept.
    """
    merged = dict(base_config or {})
    callbacks = merged.get("callbacks")
    if callbacks is None:
        merged["callbacks"] = [handler]
    elif isinstance(callbacks, BaseCallbackManager):
        manager = callbacks.copy()
        manager.add_handler(handler, inherit=True)
        merged["callbacks"] = manager
    elif isinstance(callbacks, Sequence) and not isinstance(callbacks, str):
        merged["callbacks"] = [*callbacks, handler]
    else:
        raise TypeError(
            "config['callbacks'] must be a list of handlers or a BaseCallbackManager, "
            f"got {type(callbacks).__name__}"
        )
    return merged


def ledger_from_messages(messages: Sequence[Any]) -> Ledger:
    """Build a ledger post hoc by pairing ``AIMessage.tool_calls`` with ``ToolMessage`` results.

    A tool call with no matching ``ToolMessage`` is recorded as ``status="error",
    error="no result"``. Timestamps are ``None``.
    """
    entries: list[LedgerEntry] = []
    open_calls: dict[str, LedgerEntry] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                call_id = call.get("id")
                entry = LedgerEntry(
                    tool=call["name"],
                    status="error",
                    args=coerce_args(call.get("args")),
                    error=NO_RESULT,
                    call_id=call_id,
                )
                entries.append(entry)
                if call_id:
                    open_calls[call_id] = entry
        elif isinstance(message, ToolMessage):
            matched = open_calls.pop(message.tool_call_id, None)
            if matched is None:
                continue
            text = str(message.text)
            if message.status == "error":
                matched.error = text
            else:
                matched.status = "ok"
                matched.error = None
                matched.result = text
    return Ledger(entries)


def final_reply(result: Mapping[str, Any] | Sequence[Any]) -> str:
    """Text of the last ``AIMessage`` with non-empty text, else ``""``.

    Accepts a graph result dict with ``messages`` or a messages list.
    """
    messages: Any = result.get("messages", []) if isinstance(result, Mapping) else result
    if not isinstance(messages, Sequence):
        return ""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = str(message.text)
            if text.strip():
                return text
    return ""
