"""Ledger of tool calls actually performed during an agent run."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["ok", "error"]


@dataclass
class LedgerEntry:
    tool: str
    status: Status = "ok"
    args: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    call_id: str | None = None
    parent_run_id: str | None = None
    started_at: float | None = None
    ended_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "call_id": self.call_id,
            "parent_run_id": self.parent_run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LedgerEntry:
        return cls(
            tool=data["tool"],
            status=data.get("status", "ok"),
            args=dict(data.get("args") or {}),
            result=data.get("result"),
            error=data.get("error"),
            call_id=data.get("call_id"),
            parent_run_id=data.get("parent_run_id"),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
        )


class Ledger:
    def __init__(self, entries: Iterable[LedgerEntry] = ()) -> None:
        self._entries: list[LedgerEntry] = list(entries)

    def append(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> list[LedgerEntry]:
        """Entries sorted by start time; untimed entries follow in insertion order."""
        indexed = list(enumerate(self._entries))
        indexed.sort(key=lambda p: (p[1].started_at is None, p[1].started_at or 0.0, p[0]))
        return [e for _, e in indexed]

    def ok_tools(self) -> set[str]:
        return {e.tool for e in self._entries if e.status == "ok"}

    def error_tools(self) -> set[str]:
        return {e.tool for e in self._entries if e.status == "error"}

    def tools(self) -> set[str]:
        return {e.tool for e in self._entries}

    def to_dict(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_dict(cls, data: Iterable[Mapping[str, Any]]) -> Ledger:
        return cls(LedgerEntry.from_dict(d) for d in data)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LedgerEntry]:
        return iter(self.entries)


def coerce_args(args: Any) -> dict[str, Any]:
    """Return a JSON-safe dict; non-dict input is wrapped as ``{"input": ...}``.

    Never raises: input JSON cannot hold (non-string keys, cycles, very deep nesting) is
    kept as ``{"input": str(args)}``.
    """
    if args is None:
        return {}
    try:
        coerced = json.loads(json.dumps(args, default=str))
    except (TypeError, ValueError, RecursionError):
        try:
            return {"input": str(args)}
        except Exception:
            return {"input": f"<unserialisable {type(args).__name__}>"}
    if isinstance(coerced, dict):
        return coerced
    return {"input": coerced}


def truncate(text: str | None, limit: int) -> str | None:
    """Cut ``text`` so the output (including the ellipsis) is at most ``limit`` long."""
    if text is None or len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    return text[: limit - 1] + "…"
