"""Support desk demo: run a LangGraph agent on Groq and show its reply, ledger and receipt.

Routes: ``GET /`` (the page), ``GET /healthz``, ``POST /run``. A live run that fails for any
reason (missing key, Groq rate limit, network) falls back to a recorded replay; with no
replay available the API answers 503 with a JSON error.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from toolreceipt import Receipts
from toolreceipt.langgraph import final_reply

from scenarios import support_desk

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "static" / "index.html"
DEFAULT_REPLAY_DIR = HERE / "replays"

log = logging.getLogger("toolreceipt.demo")

# Local runs read the repo-root .env; existing environment variables win.
load_dotenv(HERE.parent / ".env")


def _env_number(name: str, default: float) -> float:
    """Non-negative number from the environment; a bad value logs a warning and uses default."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        log.warning("ignoring invalid %s; using the default %g", name, default)
        return default
    return value


GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TIMEOUT_SECONDS = 15.0
RUN_DEADLINE_SECONDS = 45.0  # whole live run; past it the request falls back to a replay
RATE_LIMIT_SECONDS = _env_number("RATE_LIMIT_SECONDS", 5.0)
RATE_LIMIT_MAX_KEYS = 1024
# Global cap on live Groq runs across all clients; above it a replay is served.
LIVE_RUNS_PER_MINUTE = int(_env_number("LIVE_RUNS_PER_MINUTE", 10))
# The agent must answer after this many consecutive rounds in which every tool call failed
# (stops a model retrying a failing tool), or after MAX_TOOL_ROUNDS rounds in total.
MAX_FAILED_ROUNDS = 2
MAX_TOOL_ROUNDS = 6
RECURSION_LIMIT = 2 * MAX_TOOL_ROUNDS + 4
BUCKETS = ("confirmed", "claimed_but_never_ran", "unmatched")

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live-run")


FINAL_NOTE = (
    "Tools are no longer available for this conversation. Reply to the customer now, in plain text."
)


def _is_tool_use_failed(exc: Exception) -> bool:
    """Groq's 400 ``tool_use_failed``: the provider rejected a malformed or disallowed call."""
    return getattr(exc, "status_code", None) == 400 and "tool_use_failed" in str(exc)


def _flatten_tool_log(messages: list[Any]) -> str:
    """Plain-text log of every tool call and its outcome, e.g. ``f(a='x') -> ERROR: ...``."""
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    lines = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
            result = results.get(call.get("id") or "")
            if result is None:
                outcome = "ERROR: no result"
            else:
                text = " ".join(str(result.text).split())[:300]
                outcome = f"ERROR: {text}" if result.status == "error" else f"ok: {text}"
            lines.append(f"- {call['name']}({args}) -> {outcome}")
    return "\n".join(lines) or "- (no tools were called)"


def _tool_rounds(messages: list[Any]) -> tuple[int, int]:
    """(total tool rounds, trailing consecutive rounds in which every call failed).

    A round is an ``AIMessage`` with tool calls plus the ``ToolMessage``s that follow it; a
    call without a successful ``ToolMessage`` in its round counts as failed.
    """
    rounds: list[tuple[set[str], set[str]]] = []  # (call ids, ids that succeeded)
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            rounds.append(({c.get("id") or "" for c in message.tool_calls}, set()))
        elif isinstance(message, ToolMessage) and rounds and message.status != "error":
            rounds[-1][1].add(message.tool_call_id)
    failed_streak = 0
    for ids, succeeded in rounds:
        failed_streak = failed_streak + 1 if not (ids & succeeded) else 0
    return len(rounds), failed_streak


def build_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    *,
    deadline: float | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """``START -> agent``, ``tools_condition``, ``tools -> agent``; system prompt prepended.

    ``deadline`` (a ``time.monotonic()`` value) stops the run before any later model call.
    After ``MAX_FAILED_ROUNDS`` consecutive all-failed tool rounds or ``MAX_TOOL_ROUNDS``
    rounds in total, or when a round fails with Groq's
    ``tool_use_failed``, the turn goes to the unbound model with a flattened history (system
    prompt, customer message, plain-text tool log, reply-now note) and no tool-call
    messages; any tool call it still returns is dropped, so the run always ends with a reply.
    """
    bound = model.bind_tools(tools)
    system = SystemMessage(system_prompt)

    def answer_now(history: list[Any]) -> AIMessage:
        customer = next((m for m in history if isinstance(m, HumanMessage)), None)
        flat = [
            system,
            *([HumanMessage(customer.content)] if customer is not None else []),
            SystemMessage(
                "Actions taken so far (from the tool log):\n"
                + _flatten_tool_log(history)
                + "\n\n"
                + FINAL_NOTE
            ),
        ]
        reply = model.invoke(flat)
        if reply.tool_calls:
            reply = AIMessage(content=reply.content, id=reply.id)
        return reply

    def agent(state: MessagesState) -> dict[str, Any]:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("live run deadline exceeded")
        messages = [system, *state["messages"]]
        total, failed_streak = _tool_rounds(state["messages"])
        if total >= MAX_TOOL_ROUNDS or failed_streak >= MAX_FAILED_ROUNDS:
            return {"messages": [answer_now(state["messages"])]}
        try:
            return {"messages": [bound.invoke(messages)]}
        except Exception as exc:
            if not _is_tool_use_failed(exc):
                raise
            log.warning("tool_use_failed from the provider; retrying the turn without tools")
            return {"messages": [answer_now(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def make_model() -> BaseChatModel:
    """The live Groq model. Raises ``RuntimeError`` when ``GROQ_API_KEY`` is not set."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.7,
        max_tokens=400,
        max_retries=1,
        timeout=GROQ_TIMEOUT_SECONDS,
    )


def _run_live(mode: str, model: BaseChatModel, deadline: float) -> dict[str, Any]:
    tools = support_desk.build_tools(mode)
    graph = build_graph(model, tools, support_desk.SYSTEM_PROMPTS[mode], deadline=deadline)
    receipts = Receipts()
    result = graph.invoke(
        {"messages": [HumanMessage(support_desk.CUSTOMER_MESSAGE)]},
        config=receipts.config({"recursion_limit": RECURSION_LIMIT}),
    )
    reply = final_reply(result)
    if not reply.strip():
        raise RuntimeError("the agent produced no reply")
    receipt = receipts.check(reply).to_dict()
    claims = [{**claim, "bucket": bucket} for bucket in BUCKETS for claim in receipt[bucket]]
    return {
        "mode": mode,
        "source": "live",
        "customer_message": support_desk.CUSTOMER_MESSAGE,
        "reply": reply,
        "ledger": receipt["ledger"],
        "receipt": receipt,
        "claims": claims,
    }


def run_scenario(mode: str, *, model: BaseChatModel | None = None) -> dict[str, Any]:
    """Run the support desk agent live; on any failure serve a replay instead.

    Without an injected ``model`` the run uses Groq: a missing ``GROQ_API_KEY`` serves a
    replay without using a live slot, and once ``LIVE_RUNS_PER_MINUTE`` is spent a replay is
    served without calling Groq. The whole run is bounded by ``RUN_DEADLINE_SECONDS``.
    """
    if mode not in support_desk.SYSTEM_PROMPTS:
        raise ValueError(f"unknown mode {mode!r}")
    if model is None:
        if not os.environ.get("GROQ_API_KEY"):
            log.warning("GROQ_API_KEY is not set; serving a replay")
            return load_replay(mode)
        if not _take_live_slot():
            log.warning("live run cap reached; serving a replay")
            return load_replay(mode)
    deadline = time.monotonic() + RUN_DEADLINE_SECONDS
    try:
        live_model = model if model is not None else make_model()
        future = _executor.submit(_run_live, mode, live_model, deadline)
        try:
            return future.result(timeout=RUN_DEADLINE_SECONDS)
        except FutureTimeout:
            # Drop the run if it has not started; a running worker stops at its next
            # model call once the deadline has passed.
            future.cancel()
            raise
    except Exception as exc:  # any live failure -> replay; the error is never shown
        log.warning("live run failed (%s); serving a replay", type(exc).__name__)
        return load_replay(mode)


def _replay_dir() -> Path:
    return Path(os.environ.get("TOOLRECEIPT_REPLAY_DIR") or DEFAULT_REPLAY_DIR)


def load_replay(mode: str) -> dict[str, Any]:
    """A random recorded run from ``{replay_dir}/{mode}-*.json``, marked ``source="replay"``."""
    files = sorted(_replay_dir().glob(f"{mode}-*.json"))
    if not files:
        raise FileNotFoundError(f"no replay recorded for mode {mode!r}")
    data: dict[str, Any] = json.loads(random.choice(files).read_text(encoding="utf-8"))
    data["source"] = "replay"
    return data


# --- rate limit: one POST /run per client IP per RATE_LIMIT_SECONDS (in memory) ---

_last_run: OrderedDict[str, float] = OrderedDict()
_rate_lock = threading.Lock()
_live_runs: deque[float] = deque()


def reset_rate_limit() -> None:
    with _rate_lock:
        _last_run.clear()
        _live_runs.clear()


def _client_ip(request: Request) -> str:
    # Behind Render's proxy the peer is the proxy, which appends the real client address as
    # the right-most X-Forwarded-For entry. Entries to its left are client-supplied (spoofable).
    forwarded = request.headers.get("x-forwarded-for", "")
    last = forwarded.split(",")[-1].strip()
    if last:
        return last
    return request.client.host if request.client else "unknown"


def _allow(ip: str) -> bool:
    now = time.monotonic()
    with _rate_lock:
        last = _last_run.get(ip)
        if last is not None and now - last < RATE_LIMIT_SECONDS:
            return False
        _last_run[ip] = now
        _last_run.move_to_end(ip)
        while len(_last_run) > RATE_LIMIT_MAX_KEYS:  # fixed-size LRU: evict the oldest
            _last_run.popitem(last=False)
        return True


def _take_live_slot() -> bool:
    """Reserve one of ``LIVE_RUNS_PER_MINUTE`` live model runs shared by all clients."""
    now = time.monotonic()
    with _rate_lock:
        while _live_runs and now - _live_runs[0] >= 60.0:
            _live_runs.popleft()
        if len(_live_runs) >= LIVE_RUNS_PER_MINUTE:
            return False
        _live_runs.append(now)
        return True


# --- HTTP ---

app = FastAPI(title="toolreceipt demo", docs_url=None, redoc_url=None)


class RunRequest(BaseModel):
    mode: Literal["honest", "pressure"]
    replay: bool = False


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/run", response_model=None)
def run(body: RunRequest, request: Request) -> dict[str, Any] | JSONResponse:
    if not _allow(_client_ip(request)):
        return JSONResponse(
            {"error": f"One run per {RATE_LIMIT_SECONDS:g} seconds, please wait a moment."},
            status_code=429,
        )
    try:
        return load_replay(body.mode) if body.replay else run_scenario(body.mode)
    except Exception as exc:  # no replay to fall back on, or an unreadable one
        log.warning("run unavailable (%s)", type(exc).__name__)
        return JSONResponse(
            {"error": "The live agent is unavailable and no recorded run exists yet."},
            status_code=503,
        )
