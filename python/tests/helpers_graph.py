"""Minimal LangGraph agent graph driven by a fake chat model, for adapter tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def _openai_spec(t: BaseTool) -> dict[str, Any]:
    return {"type": "function", "function": {"name": t.name, "description": t.description}}


def build_fake_graph(
    tools: Sequence[BaseTool],
    ai_messages: Sequence[AIMessage],
    *,
    handle_tool_errors: bool = True,
) -> Any:
    """Graph ``START -> agent``, ``tools_condition``, ``tools -> agent``.

    The agent replays ``ai_messages`` in order.
    """
    model = GenericFakeChatModel(messages=iter(ai_messages)).bind(
        tools=[_openai_spec(t) for t in tools]
    )

    def agent(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        # Explicit config: on Python 3.10 LangGraph does not propagate it into async nodes.
        return {"messages": [model.invoke(state["messages"], config)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(list(tools), handle_tool_errors=handle_tool_errors))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()
