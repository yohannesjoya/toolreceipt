"""``check()`` and the ``Receipts`` facade."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Collection, Iterable, Mapping
from dataclasses import replace
from typing import Any, Protocol

from toolreceipt.claims import (
    Claim,
    ClaimExtractor,
    NamedTool,
    RuleExtractor,
    ToolSpec,
    normalize_tool_specs,
)
from toolreceipt.diff import diff
from toolreceipt.ledger import Ledger
from toolreceipt.receipt import Receipt

ToolLike = str | ToolSpec | tuple[str, str] | Mapping[str, Any] | NamedTool

_LANGGRAPH_DEPS = frozenset({"langchain_core", "langgraph"})
_MISSING_LANGGRAPH = 'toolreceipt[langgraph] is required: pip install "toolreceipt[langgraph]"'


class _Handler(Protocol):
    """What ``Receipts`` needs from the LangGraph callback handler."""

    ledger: Ledger
    known_tools: dict[str, str]

    def flush(self) -> None: ...


def _merge_tools(*groups: Iterable[ToolSpec]) -> list[ToolSpec]:
    """Union by name in first-seen order; a non-empty description is never overridden."""
    merged: dict[str, str] = {}
    for group in groups:
        for name, description in group:
            if not merged.get(name):
                merged[name] = description
    return [ToolSpec(name, description) for name, description in merged.items()]


def _specs(known: list[ToolSpec], ledger: Ledger) -> list[ToolSpec]:
    """Known tools plus every tool the ledger saw, by name."""
    return _merge_tools(known, (ToolSpec(name, "") for name in sorted(ledger.tools())))


def _diff(
    claims: list[Claim],
    ledger: Ledger,
    specs: list[ToolSpec],
    ignore_tools: Collection[str],
    llm_blocks: bool,
    check_args: bool,
) -> Receipt:
    return diff(
        claims, ledger, known_tools=[s.name for s in specs], ignore_tools=ignore_tools,
        llm_blocks=llm_blocks, check_args=check_args,
    )


def _check(
    reply: str,
    ledger: Ledger,
    known: list[ToolSpec],
    ignore_tools: Collection[str],
    extractor: ClaimExtractor | None,
    llm_blocks: bool = False,
    check_args: bool = True,
) -> Receipt:
    specs = _specs(known, ledger)
    claims = (extractor or RuleExtractor()).extract(reply, specs)
    return _diff(claims, ledger, specs, ignore_tools, llm_blocks, check_args)


async def _acheck(
    reply: str,
    ledger: Ledger,
    known: list[ToolSpec],
    ignore_tools: Collection[str],
    extractor: ClaimExtractor | None,
    llm_blocks: bool = False,
    check_args: bool = True,
) -> Receipt:
    """``_check`` that awaits an extractor's ``aextract`` coroutine when it has one."""
    specs = _specs(known, ledger)
    aextract = getattr(extractor, "aextract", None)
    if inspect.iscoroutinefunction(aextract):
        claims = await aextract(reply, specs)
    else:
        claims = (extractor or RuleExtractor()).extract(reply, specs)
    return _diff(claims, ledger, specs, ignore_tools, llm_blocks, check_args)


def check(
    reply: str,
    ledger: Ledger,
    *,
    known_tools: Iterable[ToolLike] = (),
    ignore_tools: Collection[str] = (),
    extractor: ClaimExtractor | None = None,
    llm_blocks: bool = False,
    check_args: bool = True,
) -> Receipt:
    """Framework-free check of ``reply`` against ``ledger``.

    ``llm_blocks``: let claims found only by an LLM extractor make the verdict ``fabricated``;
    by default they are reported in ``Receipt.suspected``. ``check_args``: compare the one
    amount or cued reference number in a confirmed claim's clause with the call's arguments
    (``args_mismatch``); ``False`` skips that check.
    """
    return _check(
        reply, ledger, normalize_tool_specs(known_tools), ignore_tools, extractor, llm_blocks,
        check_args,
    )


async def acheck(
    reply: str,
    ledger: Ledger,
    *,
    known_tools: Iterable[ToolLike] = (),
    ignore_tools: Collection[str] = (),
    extractor: ClaimExtractor | None = None,
    llm_blocks: bool = False,
    check_args: bool = True,
) -> Receipt:
    """``check()`` for async code: the same receipt, with an extractor's ``aextract``
    awaited when it defines one (``LLMExtractor`` runs its model call on a thread). With
    the rules alone there is no I/O and the result equals ``check()``."""
    return await _acheck(
        reply, ledger, normalize_tool_specs(known_tools), ignore_tools, extractor, llm_blocks,
        check_args,
    )


class Receipts:
    """One per agent run: injects callbacks via ``config()`` and checks the final reply."""

    def __init__(
        self,
        *,
        extractor: ClaimExtractor | None = None,
        known_tools: Iterable[ToolLike] = (),
        ignore_tools: Collection[str] = (),
        capture_results: bool = True,
        max_result_chars: int = 2000,
        llm_blocks: bool = False,
        check_args: bool = True,
    ) -> None:
        self._extractor = extractor
        self._llm_blocks = llm_blocks
        self._check_args = check_args
        self._known_tools = normalize_tool_specs(known_tools)
        self._ignore_tools = set(ignore_tools)
        self._capture_results = capture_results
        self._max_result_chars = max_result_chars
        self._handler: _Handler | None = None

    def config(self, base_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return ``base_config`` with a fresh receipt callback handler added."""
        try:
            adapter = importlib.import_module("toolreceipt.langgraph")
        except ImportError as exc:
            if (exc.name or "").partition(".")[0] in _LANGGRAPH_DEPS:
                raise ImportError(_MISSING_LANGGRAPH) from exc
            raise
        handler: _Handler = adapter.ReceiptCallbackHandler(
            Ledger(),
            capture_results=self._capture_results,
            max_result_chars=self._max_result_chars,
        )
        self._handler = handler
        merged: dict[str, Any] = adapter.merge_config(base_config, handler)
        return merged

    @property
    def ledger(self) -> Ledger | None:
        return self._handler.ledger if self._handler is not None else None

    def _resolve(self, ledger: Ledger | None) -> tuple[Ledger, list[ToolSpec]]:
        known = list(self._known_tools)
        if ledger is None:
            if self._handler is None:
                raise ValueError("no ledger: call config() before the run or pass ledger=")
            self._handler.flush()
            ledger = self._handler.ledger
            known = _merge_tools(
                known, (ToolSpec(n, d) for n, d in self._handler.known_tools.items())
            )
        if not self._capture_results:
            ledger = Ledger(replace(e, args={}, result=None) for e in ledger.entries)
        return ledger, known

    def check(self, reply: str, ledger: Ledger | None = None) -> Receipt:
        """Check ``reply``; with no ``ledger``, use the most recent ``config()`` run's."""
        ledger, known = self._resolve(ledger)
        return _check(
            reply, ledger, known, self._ignore_tools, self._extractor, self._llm_blocks,
            self._check_args,
        )

    async def acheck(self, reply: str, ledger: Ledger | None = None) -> Receipt:
        """``check()`` for async code; see the module-level ``acheck``. One ``Receipts`` per
        run: ``config()`` binds the ledger to this instance, so concurrent runs need their
        own instances."""
        ledger, known = self._resolve(ledger)
        return await _acheck(
            reply, ledger, known, self._ignore_tools, self._extractor, self._llm_blocks,
            self._check_args,
        )
