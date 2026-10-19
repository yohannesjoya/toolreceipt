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

## A family-only match never accuses

A claim is scored against each bound tool: +2 when the tool name carries the claim's verb
family, +2 when it also carries an object word of the claim ("ticket" in `create_ticket` for
"I've created a ticket"), +1 when the description holds an object word the name lacks.

Honest agents often describe work done with one tool using a verb whose family also fits a
sibling tool: "I've raised this with our team" after `escalate_ticket` fits the `create` family
of `create_ticket`, "I've forwarded your complaint" after `create_ticket` fits the `send`
family of `send_email`. Matched on the verb family alone, such a claim says nothing about which
tool did the work. So a `not_called` outcome is reported only when the candidate also matched
on an object word of its name. A family-only candidate that never ran sends the claim to
`unmatched` (informational) instead of `claimed_but_never_ran`. A family-only candidate that
ran confirms the claim, and one that was attempted and failed is still `tool_failed`, since the
agent tried that exact tool.

The extractor records this evidence on each claim as a private attribute, outside the fixed
`Claim` field set and never emitted by `to_dict()`. A claim built by hand or by
`Claim.from_dict` carries no evidence, and the diff treats its candidates as fully matched.

The claim's verb stem also counts as object evidence when it is the last token of a
multi-token tool name after another token of the same family: "I refunded your order" names
the `refund` of `issue_refund`, so it is still `not_called` when `issue_refund` never ran.
"I've processed your return" does not (`process` is not a token of `issue_refund`).

Precision comes before recall here: "I've sent it" (a `send_email`/`send_sms` tie) and "I've
updated your address" against `update_customer` are no longer flagged when the tool never ran. Related lexicon choices:
"summarised" and "closed" belong to no verb family (closing a ticket is usually an update), and
"issue" is not a ticket synonym ("I've logged the issue" means the problem, not a ticket).
