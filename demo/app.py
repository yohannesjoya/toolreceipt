"""Support desk demo: run a LangGraph agent on Groq and show its reply, ledger and receipt.

Routes: ``GET /`` (the page), ``GET /healthz``, ``POST /run``. A live run that fails for any
reason (missing key, Groq rate limit, network) falls back to a recorded replay; with no
replay available the API answers 503 with a JSON error.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

GROQ_MODEL = "openai/gpt-oss-120b"
RATE_LIMIT_SECONDS = 5.0
# Tool rounds before the agent must answer; stops a model retrying a failing tool forever.
MAX_TOOL_ROUNDS = 5
RECURSION_LIMIT = 2 * MAX_TOOL_ROUNDS + 4
BUCKETS = ("confirmed", "claimed_but_never_ran", "unmatched")

log = logging.getLogger("toolreceipt.demo")

# Local runs read the repo-root .env; existing environment variables win.
load_dotenv(HERE.parent / ".env")


def build_graph(
    model: BaseChatModel, tools: list[BaseTool], system_prompt: str
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """``START -> agent``, ``tools_condition``, ``tools -> agent``; system prompt prepended.

    After ``MAX_TOOL_ROUNDS`` tool rounds the agent is invoked with ``tool_choice="none"``
    and any tool call it still makes is dropped, so the run always ends with a reply.
    """
    bound = model.bind_tools(tools)
    answer_only = model.bind_tools(tools, tool_choice="none")
    system = SystemMessage(system_prompt)

    def agent(state: MessagesState) -> dict[str, Any]:
        messages = [system, *state["messages"]]
        rounds = sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)
        if rounds < MAX_TOOL_ROUNDS:
            return {"messages": [bound.invoke(messages)]}
        reply = answer_only.invoke(messages)
        if isinstance(reply, AIMessage) and reply.tool_calls:
            reply = AIMessage(content=reply.content, id=reply.id)
        return {"messages": [reply]}

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

    return ChatGroq(model=GROQ_MODEL, temperature=0.7, max_tokens=400, max_retries=1, timeout=30)


def _run_live(mode: str, model: BaseChatModel) -> dict[str, Any]:
    tools = support_desk.build_tools(mode)
    graph = build_graph(model, tools, support_desk.SYSTEM_PROMPTS[mode])
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
    """Run the support desk agent live; on any failure serve a replay instead."""
    if mode not in support_desk.SYSTEM_PROMPTS:
        raise ValueError(f"unknown mode {mode!r}")
    try:
        return _run_live(mode, model if model is not None else make_model())
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

_last_run: dict[str, float] = {}
_rate_lock = threading.Lock()


def reset_rate_limit() -> None:
    with _rate_lock:
        _last_run.clear()


def _client_ip(request: Request) -> str:
    # Behind Render's proxy the peer is the proxy; the client is the first forwarded hop.
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


def _allow(ip: str) -> bool:
    now = time.monotonic()
    window = RATE_LIMIT_SECONDS
    with _rate_lock:
        last = _last_run.get(ip)
        if last is not None and now - last < window:
            return False
        if len(_last_run) > 10_000:  # drop stale entries so memory stays bounded
            for key in [k for k, t in _last_run.items() if now - t >= window]:
                del _last_run[key]
        _last_run[ip] = now
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
