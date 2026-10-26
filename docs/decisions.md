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

## Another tool of the same action makes a claim ambiguous

A claim that would be `not_called` blocks only when no successful ledger entry exists for any
tool whose name carries a verb of the claim's family. If `remove_item` ran, "I've removed the
duplicate mug from your order" may describe it even though the object matches `cancel_order`, so
the claim goes to `unmatched`. `tool_failed` is unchanged. The verb-stem evidence ("refunded" naming
the `refund` of `issue_refund`) marks a claim as object-matched but never adds to the score, so a
sibling `refund_payment` cannot outrank the tool that ran. "case" is not a ticket synonym either
("opened a case with our specialist team" after `escalate_ticket`).

## LLM-only claims are suspected, not fabricated

`LLMExtractor` (0.2) adds claims the rules cannot see: outcome phrasing, synonyms, passive
voice, status lines, other languages. A claim only the model found whose tool never ran lands
in `Receipt.suspected`, not `claimed_but_never_ran`, so `ok` and whether the verdict is
`fabricated` still come from the deterministic rule claims alone. Two runs of the same reply
block or pass the same way even though the model's output varies (temperature 0 on Groq was
not reproducible: 3 of 11 replies differed between two runs). `clean` versus `unreported` may
move, because an LLM claim that confirms a tool removes it from `ran_but_unreported`; leaving
a confirmed tool listed as unreported would make the receipt contradict itself. Callers who
accept probabilistic blocking set `llm_blocks=True`, which also turns on a sibling demotion
by tool-name verb token (`send_email` ran, the claim names `send_sms`: unmatched, not
blocking), since precision matters most when the model can block.

## Rules win on overlap; family-alone claims are kept beside the LLM claim

The extractor runs the rules first and merges the model's claims in. Where a model claim
overlaps a rule claim that already has candidates matched on an object word, the rule claim
stands. Where the rule claim is unmatched, the model's tool resolves it (an unmatched claim
never blocks, so replacing it is safe). Where the rule claim matched on the verb family
alone, both claims are kept: the merge never sees the ledger, so it cannot know whether the
rule candidate failed, and replacing it would turn a `tool_failed` block into `suspected`.
Nothing in the merge writes `_object_matched` into a rule claim. The rules' `_dedupe` does not
run over the merged list for the same reason; instead a model claim whose candidate set equals
a surviving rule claim's is dropped. The guards reuse the rule lexicon's negation, other-subject
and state checks but not its future or advice lists, because outcome claims ("you'll find the
confirmation in your inbox") are the largest measured gap and those lists would delete them.

## Description lead verb makes a verbless tool eligible; sibling demotion stays name-only

A tool whose name carries no verb token (`jira`, `stripe`, `email_customer`) is matched on
the family named by the first content word of its description, after an optional `Name:`
prefix and one filler phrase. Only the first word counts: a later verb ("Manage
subscriptions (cancel, pause)") is not evidence, and a read verb anywhere in the first
clause makes the tool ineligible, as a read verb in a name does. Such a tool joins the
scoring beside verb-named tools only when the reply names it, so no shipped fixture changed.
It can tie a name match but never outscore it (`desc_generic_noun_ties_not_displaces`), and
a noun lead word ("Email", "File") counts as the verb only when a determiner or pronoun
follows it, so "Email address validation" names no family (`desc_noun_modifier_not_lead`).
Object evidence from a description is its direct object only: the words after the lead verb
up to the first preposition or the clause end. "Create a support ticket for the customer"
gives `support ticket`, not `customer`, because any word anywhere in a description turned
"I've added a note to your customer record" into a fabrication against `zendesk` when
`hubspot` had run (`desc_crm_note_not_ticket`, `desc_generic_word_other_tool_honest`).
`family_ran` (the sibling demotion in `diff()`) keeps reading names only: widening it to
descriptions would silence a real "I've emailed you" fabrication whenever a Slack-style send
tool ran, and `diff()` receives names, not descriptions. The cost is one rare false alarm,
pinned by `fixtures/check/desc_slack_ran_keeps_email_flag.json`.

## `acheck` shares one code path with `check`; one `Receipts` per run

`acheck` resolves the ledger and known tools exactly as `check` and differs only in awaiting
an extractor's `aextract` when it defines one. `LLMExtractor.aextract` runs the user's sync
`complete` on a thread, or awaits `acomplete` when the caller has an async-only client
(0.3.1). `Receipts.config()` binds the newest ledger to the instance, so two concurrent runs
on one instance that call `check()` with no arguments read each other's ledger; the quickstart
says "one per run", and since 0.3.1 `check(reply, config=cfg)` reads the run whose config is
passed back, so one instance can serve concurrent runs when the caller keeps each config.

## Argument literals are compared one per kind against keyed top-level arguments; names are never compared

The argument check (0.3.1) reads two kinds of literal from a confirmed claim's clause: a money
amount with a currency mark, and a reference number that follows a cue word (`order 4821`,
`ticket #5521`). A clause with two of one kind is not judged for that kind ("refunded $32 of
your $64 order"), and an amount the surrounding words qualify is not judged either: one
after a preposition ("half of your $64 order"), one before the noun it prices ("your $120
plan"), a unit rate ("$16 each"), a magnitude suffix ("$1.5k") or an approximation ("around
$64"). "pounds" is not a currency word, since it is a weight as often ("3 pounds of apples"). A literal is compared only with top-level arguments whose key names the
same kind (`amount`, `refund_amount`, `amount_cents` for amounts; a key carrying the cue
word, `order_id`, for references) and whose value parses as that kind; nested values, free
text and keys such as `charge="ch_…"` are never evidence. A value agrees when it equals the
literal, equals it times 100 (Stripe-style cents), when the keyed amounts of all the tool's
calls sum to it (a refund split across calls), or when the digits appear in the tool's
result (an id the agent learned from the result; a result cut by `max_result_chars` can lose
that excuse). The claim is `args_mismatch` only when every `ok` call of its candidates is
keyed for every literal and disagrees on at least one, and no same-family tool's call
agrees; a family-only match is never judged, since it says nothing about which tool did the
work. Names, email addresses, dates and times stay out: nicknames, roles, display names and
formats make them unsafe for rules. The honest corpus with real arguments is the 19 demo
replays, the 6 demo examples and the held-out replies; `args_honest_*` fixtures pin the
shapes that nearly flagged in review (cents, line items, two amounts, `update_ticket(id=…)`
beside an order number, a date after "order").

## Message lists are read by key and attribute over the last turn; failure is never inferred from result text

`toolreceipt.messages` reads OpenAI Chat Completions messages, OpenAI Responses items and
Anthropic messages through one accessor (mapping key, else attribute), so dicts and SDK
objects behave the same and no SDK is imported. Only the last turn is read, the messages
after the last user message that carries no `tool_result` block (Anthropic puts tool results,
sometimes with a text block beside them, in user messages that continue the run), because a whole
conversation would mark every earlier turn's tools `ran_but_unreported` and compare the last
reply's amounts with earlier calls. The final reply is `""` when the run ends on a tool call,
so an abandoned run checks as "no claims, tools unreported" rather than against an earlier
preamble; the LangChain `final_reply` keeps returning the last text until 0.4 aligns it. A
tool result is `ok` unless Anthropic's explicit `is_error` says otherwise: reading "Error:"
out of a result would turn a successful call into `tool_failed`. Recognised but unsupported
shapes (the legacy `function_call` field, `role: "function"`, LangChain message objects)
raise rather than yield an empty ledger that would make every claim `not_called` or every run
`clean` silently.
