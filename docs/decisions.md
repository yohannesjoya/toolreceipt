# Decisions

## Callbacks over tool wrapping

Ledger capture uses a LangChain callback handler rather than wrapping tools. It needs no
change to the user's tools or graph, works for tools the user does not own, and the same
handler works for plain LangChain agents.

The handler implements `on_chat_model_start`, `on_tool_start`, `on_tool_end` and
`on_tool_error`. A tool built with `handle_tool_error=True` (or `handle_validation_error=True`)
reports its failure through `on_tool_end` with a `ToolMessage(status="error")` and never calls
`on_tool_error`, so `on_tool_end` reads `ToolMessage.status` rather than assuming success.
`ToolNode` runs one turn's tool calls in a thread pool, so the in-flight map, known tools and
ledger appends are guarded by one lock.

## Graphs need `handle_tool_errors=True` for failed tools to be recorded

Verified on langgraph-prebuilt 1.1.1: `ToolNode`'s default `handle_tool_errors` only catches
invalid-argument errors and re-raises tool execution errors, which aborts the run. Then there
is no final reply to check, and the agent never learns the tool failed. With
`ToolNode(tools, handle_tool_errors=True)` (or tools built with `handle_tool_error=True`), the
failure reaches the ledger as a `status="error"` entry and the agent as an error
`ToolMessage`, and the run continues. The README and demo use `handle_tool_errors=True`; the
test graph helper defaults to it.

## Known tools come from `invocation_params`

`on_chat_model_start` receives `invocation_params["tools"]` with every tool bound to the chat
model (OpenAI shape `{"type": "function", "function": {"name", "description"}}` or Anthropic
shape `{"name", "description", "input_schema"}`). The handler adds those names and
descriptions to its known tools, plus any name seen in `on_tool_start`. This is what lets a
claim about a bound tool that was never called be reported as `not_called` with no user
configuration. The post-hoc `ledger_from_messages` path cannot see bound tools, so there
`known_tools` must be passed to `Receipts` for `not_called` detection.
