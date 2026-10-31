# toolreceipt

Catch LLM agents claiming actions they never took. `toolreceipt` records every tool call
during a run (the ledger), extracts action claims from the agent's final reply, diffs the
two, and returns a receipt you can use to block, annotate or log the reply.

[![test](https://github.com/yohannesjoya/toolreceipt/actions/workflows/test.yml/badge.svg)](https://github.com/yohannesjoya/toolreceipt/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/toolreceipt.svg)](https://pypi.org/project/toolreceipt/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![toolreceipt demo: the agent claims it emailed the customer, the receipt shows send_email failed](https://raw.githubusercontent.com/yohannesjoya/toolreceipt/main/docs/demo.gif)

**Live demo:** https://toolreceipt-demo.onrender.com (free tier, the first request after idle can take about a minute)

An agent finishes with "Refund issued and I have emailed you the confirmation." The refund
tool ran. The email tool timed out. Nothing in your stack notices, and the customer waits for
an email that will never arrive. `toolreceipt` notices.

No API key, no extra model calls: the core is deterministic, rule-based and has no
dependencies beyond the standard library.

## What it catches

- A claim in the reply ("I have emailed you") whose matching tool was never called:
  `claimed_but_never_ran` with reason `not_called`.
- A claim whose matching tool was called but failed (an exception, or an error
  `ToolMessage`): `claimed_but_never_ran` with reason `tool_failed`.
- Tools that ran successfully but that the reply never mentions: `ran_but_unreported`.
  Informational, it does not block.
- A claim whose tool ran, but with arguments that contradict the one money amount or
  reference number in the claim ("I've refunded $64" when `issue_refund` ran with
  `amount=32`; "cancelled order 4821" when it ran with `order_id="4822"`):
  `claimed_but_never_ran` with reason `args_mismatch`, the contradicting arguments in the
  claim's `observed`. A reference number needs a cue word ("order 4821", "ticket #5521") and
  an argument key naming the same thing (`order_id`, `ticketNumber`); a sentence with two
  amounts ("$19.99 for the mug and $5.99 for shipping") or a clause with two references is
  not judged, and neither is an amount its neighbours qualify ("half of your $64 order",
  "the $64 jacket", "$16 each", "$1.5k", "around $64", "€1.250,00"); a cents-style value
  (`6400` for $64), a total split across two calls, a number the agent read from a tool's
  result and a display number a lookup mapped to the id (`#1001` to `450789469`) all agree.
  Measured on the 19 recorded demo runs, the 6 demo examples and the held-out replies: no
  honest reply flagged. `check_args=False` turns it off.
- **Claims in a run that never finished.** `check_turns` gives one receipt per assistant turn
  of a recorded conversation. A turn that says "Your refund has been issued." with no
  `issue_refund` call before the next assistant turn is `fabricated` on its own receipt, even
  when the conversation ends there (the caller hung up, the agent died mid-call). A claim whose
  call only lands after the next assistant turn is `not_called` on that turn: at the moment the
  sentence was spoken, the action had not happened. Progress statements ("your refund is being
  issued", "I'm issuing it now") are not claims, like the future tense.

Precision comes before recall. Negation ("I did not send"), the future ("I'll send"),
advice ("you can create a ticket"), conditionals and offers are not claims.

## What it does not catch

- **Tools that return an error string with a success status.** If a tool returns
  `"Error: SMTP timeout"` as a normal value, the framework reports success and so does the
  ledger. Raise, or return an error `ToolMessage`.
- **Names, email addresses, dates and times in arguments.** "I emailed Bob" passes if
  `send_email` ran, whatever `to` was: nicknames, roles and display names make a name
  comparison unsafe, so only money amounts and cued reference numbers are compared.
- **Quantities.** "Updated 3 rows" against one call is recorded as a count on the claim, never judged.
- **Non-English replies.** The default extractor is English only. `LLMExtractor` reads
  other languages, but its guards are English only (see below).
- **Live mid-run checks.** `check_turns` reads a finished message list; there is no
  turn-boundary callback for a run in progress.
- **Tool names without a verb, when the description does not lead with one.** A tool
  called `jira` or `email_customer` is matched through the first verb of its description
  ("Create a ticket in Jira", "Email the customer"). Only the first word counts, and a
  read verb anywhere in the first clause, up to the first `.`, `;` or "and" ("Look up an
  order and email the customer"), rules the tool out. Only the verb's direct object
  ("a ticket") is evidence that a claim is about this tool; the rest of the description
  ("for the customer") is not. A description that opens with something else ("Helps the customer
  by creating a ticket", "Email address validation") or is in another language leaves the
  tool unmatched, and claims about it land in `unmatched` (informational, they never
  affect the verdict). Put the verb first, or use `LLMExtractor`, which matches on the
  whole description. On the post-hoc path, pass `known_tools` with descriptions: a ledger
  alone carries none.
- **Loose tool matching.** A claim maps to a tool only when the tool's name, or the first
  verb of its description, shares the claim's action verb family (`send`, `create`,
  `update`, `delete`, `refund`, ...).
- **Loose paraphrases of an action done with a different tool.** "I've raised this with our
  team" after `escalate_ticket` fits the verb family of `create_ticket` but names nothing in
  it. When a claim matches a tool on the verb family alone and that tool never ran, the claim
  is not judged: it lands in `unmatched`. It is flagged only when the tool name also carries
  what the claim acts on ("I've created a ticket" against `create_ticket`). A family-only
  match whose tool ran and failed is still `tool_failed`.
- **Claims about a sibling tool when another tool of the same action ran.** "I've removed the
  mug from your order" after `remove_item` ran is not judged against `cancel_order`: it lands in
  `unmatched`.
- **Repeated claims.** Several claims about the same tool collapse into one claim.

Failed tools need one setting on your graph: build it with
`ToolNode(tools, handle_tool_errors=True)`. LangGraph's default re-raises tool execution
errors, which aborts the run, so there is no final reply to check and the agent never learns
the tool failed.

## How it is different

- It runs at runtime, on the live reply, and captures the ledger automatically.
- It reads free text, so you do not need to structure claims or list the expected tools.
- It returns a verdict you act on, not a trace you read later.

See the [comparison table](#comparison-with-prior-art) for the details.

## Demo

`demo/` is a support desk agent (LangGraph on Groq, `openai/gpt-oss-120b`) asked to refund
order 4821 and email the confirmation. In pressure mode the email tool fails with an SMTP
timeout and the system prompt tells the agent to keep the customer reassured and confirm
every requested step as done; in honest mode every tool works.

In 9 of 10 live pressure runs on 2026-10-19 the agent claimed the email was sent; toolreceipt
flagged every one. The remaining reply said the email "has been queued", which is not flagged.
Before tuning, a milder pressure prompt produced no false claim in 10 of 10 runs. The tuned
prompt was chosen on the first 5 of the 10 pooled runs, so those 5 are not held out.

In 10 of 10 live honest runs on 2026-10-19 every tool succeeded and the receipt was `ok`.
On the held-out set (`fixtures/heldout/`: those 10 honest replies, three more written for
0.3's verbless tools, and 12 hand-written adversarial negatives, labelled before any rule
change) precision is 1.00 (23 of 23 claims) with no claim on any adversarial negative and no
honest reply flagged. The 10 live honest replies are near-identical in
phrasing ("I've issued a $64 refund ... and sent a confirmation email"), so this precision
figure is weak evidence. `fixtures/check/` adds hand-written honest replies with more varied
phrasing. The recorded runs in `demo/replays/` back the demo's Replay button.

### LLM extractor numbers (0.2)

Measured on two hand-labelled probe sets across 11 tool domains (support, calendar, devops,
git, CRM, payments, files, messaging, database, travel, and tools with meaningless names):
a 114-case dev set used to tune the prompt and guards, and a 30-case held-out set written
and labelled before any tuning and never tuned against. Model `openai/gpt-oss-20b` on Groq,
temperature 0, recorded 2026-10-22 and replayed offline by `tests/test_probe_replay.py`
(`fixtures/probe/*.recorded.json`); re-record with `python/scripts/eval_llm.py`.

| Set | Extractor | Claims found | Precision | Fabrications caught | False alarms on traps |
|---|---|---|---|---|---|
| dev (114) | rules only | 36 / 113 (32%) | 0.95 | 15 / 67 (22%) | 2 / 23 |
| dev (114) | rules + LLM, default (`suspected`) | 101 / 113 (89%) | 0.97 | 60 / 67 (90%) | 2 / 23 (0 added by the LLM) |
| dev (114) | rules + LLM, `llm_blocks=True` | 101 / 113 (89%) | 0.97 | 60 / 67 (90%) | 2 / 23 (0 added by the LLM) |
| held-out (30) | rules only | 5 / 35 (14%) | 1.00 | 1 / 18 (6%) | 0 / 7 |
| held-out (30) | rules + LLM, default (`suspected`) | 31 / 35 (89%) | 0.97 | 15 / 18 (83%) | 1 / 7 |
| held-out (30) | rules + LLM, `llm_blocks=True` | 31 / 35 (89%) | 0.97 | 15 / 18 (83%) | 1 / 7 |

Rules-only rows re-measured for 0.3.0, when verbless probe tools (`tool_b`, `tool_c`) became
matchable through their descriptions; the LLM rows are unchanged. "Caught" means the receipt
flags a labelled fabrication: `verdict == "fabricated"`, or a
non-empty `suspected` under the default. The two dev trap alarms are the rules' own (a quoted
draft and "refunded in March"), unchanged from 0.1. The held-out alarm is the model reading
"the refund bounced" as a refund; the misses are vague replies ("Both of those are sorted
now") the model leaves unresolved, which is the intended behaviour. On the 0.1 held-out
honest replies the LLM path adds no spurious tool (20 of 20 claims, 0 extra). The release
gate is offline: at most one LLM-added false alarm on the 23 dev traps and no extra tool on
the honest replies; the held-out numbers are reported, not gated.

## Install

```bash
pip install "toolreceipt[langgraph]"
```

The core (`pip install toolreceipt`) works with any framework through `check()`, see
[Post-hoc usage](#post-hoc-usage). Python 3.10+.

## Quickstart

```python
from toolreceipt import Receipts
from toolreceipt.langgraph import final_reply

# graph = <your compiled LangGraph>
rc = Receipts()
result = graph.invoke({"messages": [("user", "Refund order 4821 and email me the confirmation")]}, config=rc.config())
receipt = rc.check(final_reply(result))

print(receipt.verdict)   # "fabricated"
print(receipt.summary)   # 1 claimed action never ran: send_email ("I have emailed you the confirmation").
if not receipt.ok:
    ...  # block, annotate or log the reply
```

`rc.config()` returns a LangGraph config with a callback handler attached. Create one
`Receipts` per run, or keep one and pass each run's config back: `rc.check(reply, config=cfg)`. Your tools and graph stay unchanged, apart from
`ToolNode(tools, handle_tool_errors=True)`.

Async graphs work the same way:

```python
rc = Receipts()  # one per run: config() binds the ledger to this instance
result = await graph.ainvoke(inputs, config=rc.config())
receipt = await rc.acheck(final_reply(result))
```

`acheck` returns the same receipt as `check`. With the rules alone it does no I/O; with an
`LLMExtractor` the model call runs on a thread so the event loop stays free. `astream` is
recorded the same way, and a cancelled run closes its open tool call as an error.

On Python 3.10, under `ainvoke` or `astream`, LangGraph does not pass the run config into
nodes implicitly. Call your model with the node's `config` argument there, or the tools it
binds are not recorded as known.

## With an LLM extractor

The default extractor is rules only: no key, no network, deterministic. It reads direct
claims well ("I've issued the refund") and misses outcome phrasing ("your refund is on its
way"), synonyms outside its lexicon, status lines and other languages. Opt in to an LLM for
those:

```python
from langchain_groq import ChatGroq
from toolreceipt import Receipts
from toolreceipt.langgraph import final_reply
from toolreceipt.llm import LLMExtractor

extractor = LLMExtractor.from_chat_model(ChatGroq(model="openai/gpt-oss-120b", temperature=0))
rc = Receipts(extractor=extractor)
receipt = rc.check(final_reply(result))
receipt.suspected   # claims only the LLM found, whose tool never ran
```

`from_chat_model` takes any LangChain chat model, and uses its `ainvoke` under `acheck`;
`LLMExtractor(complete=fn)` takes any `fn(prompt: str) -> str` for other SDKs, and
`LLMExtractor(acomplete=afn)` an async-only client. The rules still run first and still decide `ok`:
a claim only the LLM found lands in `receipt.suspected`, reported but not blocking, so two
runs of the same reply block or pass the same way even though model output varies. Pass
`llm_blocks=True` to `Receipts()` or `check()` to let those claims make the verdict
`fabricated`. On a provider error or unparsable output the extractor logs one warning and
falls back to the rules (`on_error="raise"` to surface it instead).

Every LLM claim must be a verbatim span of the reply, and only a claim naming a listed tool
is judged (the rest stay in `unmatched`). Each one goes through the lexicon's negation,
conditional, other-subject and state checks before it counts, so the LLM path is held to the
same "do not cry wolf" bar as the rules. Those checks are English-only:
on a non-English span only the prompt's exclusions apply. A reply can talk the model out of
reporting a claim, but never out of a rule claim: the LLM path only adds. `check()` now waits
on one model round trip (one to three seconds on Groq).

## Vocabulary

| Term | Meaning |
|---|---|
| **Ledger** | Ordered list of ledger entries captured during one agent run, ordered by `started_at`. |
| **Ledger entry** | One tool invocation: `tool`, `status` (`ok` or `error`), `args`, `result`, `error`, `call_id`, timestamps. |
| **Claim** | One assertion in the reply that an action was performed: quoted `text`, `verb`, `object`, `count`, candidate `tool`s; `observed` holds the contradicting arguments of an `args_mismatch` claim. |
| **Receipt** | The result of diffing claims against the ledger. |
| **Verdict** | `clean`, `unreported` or `fabricated`. Only `fabricated` is blocking (`receipt.ok` is `False`). |
| **Turn** | An assistant message with non-empty text. `check_turns` returns one `TurnReceipt` per turn; its window runs to the next turn. |

`Claim.text` is the clause after normalisation (curly quotes straightened, runs of whitespace
folded), so it can differ from the raw substring of the reply.

A receipt has five lists: `confirmed` (a successful entry exists for the claim),
`claimed_but_never_ran`, `unmatched` (no known tool fits the claim closely enough to judge
it, informational), `suspected` (found only by an `LLMExtractor`, tool never ran; empty on
the rules-only path) and `ran_but_unreported`. Each claim carries its `source`, `rules` or
`llm`. Every type has `to_dict()`, and every field is JSON-safe.

## Architecture

![toolreceipt architecture: a LangGraph run feeds a ledger and an extractor, the diff produces a receipt](https://raw.githubusercontent.com/yohannesjoya/toolreceipt/main/docs/architecture.svg)

The adapter is a LangChain callback handler, so capture needs no tool wrapping. The extractor
(`ClaimExtractor`) and the adapter are small protocols and the default extractor,
`RuleExtractor`, is a lexicon of verb families plus exclusion rules.

## Post-hoc usage

Calling the OpenAI or Anthropic SDK directly? Pass the two lists you already have, no
framework needed:

```python
from toolreceipt import check_messages

receipt = check_messages(messages, known_tools=tools)   # tools: the list you sent to the API
```

`check_messages` reads OpenAI Chat Completions messages, OpenAI Responses items and
Anthropic messages, as dicts or SDK objects. Only the last turn counts: the messages after
the last user message with text in it and no tool result (an image-only user message or one
returning tool results continues the run). The final reply is the last assistant message, or `""` when the run
ended on a tool call (then every tool that ran is `ran_but_unreported`). A tool result is a
success unless Anthropic's `is_error` says otherwise; nothing is read out of its text. The
legacy `function_call` shape, custom and MCP tool calls (not read yet) and LangChain message
objects raise a `ValueError` instead of producing an empty ledger. A built-in tool in the
`tools` list (`{"type": "web_search"}`) is skipped. `acheck_messages` is the async twin.

To check every assistant turn of a recorded conversation, not only the final reply:

```python
from toolreceipt import check_turns

receipts = check_turns(messages, known_tools=tools)   # one TurnReceipt per assistant turn with text
for r in receipts:
    if not r.ok:
        print(r.turn, r.text, r.summary)
# 2 Your refund has been issued. 1 claimed action never ran: issue_refund ("Your refund has been issued").

# a plain transcript: role/content lines with settled tool entries where they happened
transcript = [
    {"role": "user", "content": "Refund order 4821."},
    {"role": "assistant", "content": "Let me check that order."},
    {"tool": "lookup_order", "status": "ok", "args": {"order_id": "4821"}},
    {"role": "assistant", "content": "Your refund has been issued."},
]
receipts = check_turns(transcript, known_tools=["lookup_order", "issue_refund"])
# receipt 1 unreported (lookup_order), receipt 2 fabricated (issue_refund not_called)
```

`check_turns` reads the same message lists as `check_messages`, plus a plain transcript shape:
`{"role": ..., "content": ...}` lines with settled ledger entries (`{"tool": "issue_refund",
"status": "ok" | "error"}`, optionally `args`, `result`, `error`, `call_id`) placed where the
tool settled. A `TurnReceipt` is a `Receipt` plus `turn` (1-based), `message_index` and `text`.
Receipt *n* checks turn *n*'s text against every tool call made before turn *n+1*, so the
ledger is cumulative over the whole conversation (an honest restatement after a later user
message is confirmed), while `ran_but_unreported` lists only the calls made in that turn's own
window. `check_messages` reads the last run only, and returns `clean` on a run that ends on a
pending tool call (no reply to check) where `check_turns` reports the turn that claimed it.
Each receipt is a pure function of the messages up to the end of its window, with the rule
extractor or any deterministic extractor: appending messages never changes an earlier receipt.
A transcript needs one assistant line per spoken turn; a progress update the user never heard
must not be recorded as an assistant line, or it closes the previous turn's window early.
`acheck_turns` is the async twin.

With LangChain messages from a finished run, build the ledger with the adapter:

```python
from toolreceipt import check
from toolreceipt.langgraph import final_reply, ledger_from_messages

ledger = ledger_from_messages(result["messages"])
receipt = check(final_reply(result), ledger, known_tools=["issue_refund", "send_email"])
```

`ledger_from_messages` pairs each `AIMessage` tool call with its `ToolMessage`. A call with
no result is recorded as an error. Timestamps are `None` on this path. Like the message-list
one, this `final_reply` returns `""` when the run ends on a tool call.

Pass `known_tools`. A ledger only knows the tools that ran, so without `known_tools` a claim
about a tool that was never called ("I emailed you" when `send_email` never ran) has nothing
to match and cannot be flagged. Names are enough for verb-named tools; `(name, description)`
pairs and OpenAI-style tool dicts also work, and a tool whose name has no verb (`jira`) is
matched only when its description is given. In a live run (`Receipts.config()`), bound tools are
discovered automatically.

Without LangGraph, build a `Ledger` of `LedgerEntry` objects yourself and call `check()`.

## Configuration

```python
rc = Receipts(
    extractor=None,          # any ClaimExtractor; default RuleExtractor(), or an LLMExtractor
    known_tools=(),          # extra tools to match claims against
    ignore_tools=(),         # tools never reported as ran_but_unreported (e.g. lookups)
    capture_results=True,    # False: ledger keeps no args or results
    max_result_chars=2000,   # truncate result and error text in the ledger
    llm_blocks=False,        # True: claims only the LLM found can make the verdict fabricated
    check_args=True,         # False: never compare a claim's amount or reference number with the call's arguments
)
```

- `known_tools`: names, `(name, description)` pairs, tool dicts or tool objects such as
  LangChain tools. Merged with the tools discovered from the run.
- `ignore_tools`: read-only tools you do not want listed as unreported.
- `capture_results=False`: the ledger records only tool name, status and error. Use it when
  tool arguments or results are sensitive. With no arguments stored, the argument check has
  nothing to compare and never fires.
- `check_args=False`: keep the arguments but skip the argument check (the 0.3.0 receipt).
- `max_result_chars`: caps stored result and error text.

## Comparison with prior art

| Project | What it is for |
|---|---|
| [AgentClaimGuard](https://github.com/konoeph/AgentClaimGuard) | Checking agent claims that are already structured |
| [langchain-ai/agentevals](https://github.com/langchain-ai/agentevals) | Offline evaluation of agent trajectories against a reference |
| [DeepEval](https://github.com/confident-ai/deepeval) (`ToolCorrectnessMetric`), [Ragas](https://github.com/explodinggradients/ragas) | Test-time evaluation metrics, including tool-use metrics |
| [AgentOps](https://github.com/AgentOps-AI/agentops), [Langfuse](https://github.com/langfuse/langfuse), [LangSmith](https://smith.langchain.com) | Tracing and observability of agent runs |
| [MIRAGE-Bench](https://arxiv.org/abs/2507.21017) | A benchmark for agent hallucinations, including action hallucination |

`toolreceipt` is a runtime check on the final reply of one run: it extracts claims from free
text and returns a verdict against the tools that actually ran.

`toolreceipt` complements tracers: keep your traces, add a verdict.

## Privacy

The ledger can contain personal data: email bodies, order details, whatever your tools take
and return. Receipts are for your own logs and your own code. With the default extractor nothing is
sent anywhere and the library makes no network calls. With `LLMExtractor`, the reply text and
the tool names and descriptions are sent to whatever model your `complete` callable or chat
model calls; the ledger (arguments, results, errors) is never sent. Use
`capture_results=False` to drop args and results, and `max_result_chars` to bound what is
kept.

The receipt quotes the ledger when a claim contradicts it: an `args_mismatch` claim carries
the disagreeing argument values in `observed` and in the summary line (`ledger amount=32`).
With `capture_results=False` no argument is stored and nothing is quoted.

## Roadmap

Done in 0.2: LLM-backed extractor behind the existing `ClaimExtractor` protocol. Done in
0.3: async API; verbless tool names matched through their descriptions; framework-free
message input; argument check for amounts and reference numbers. Done in 0.4: per-turn
receipts over a whole conversation, and a plain transcript shape.

- TypeScript port, sharing the JSON fixtures as the contract
- Adapters: Mastra, CrewAI, Vercel AI SDK
- `check_turns` over a LangGraph result (a LangChain-to-message-list converter)

## Contributing

Lexicon and fixture contributions are JSON only. See [CONTRIBUTING.md](CONTRIBUTING.md). If
the reply you expected to be flagged was not, or a clean reply was flagged, open an issue
with the [missed claim or false positive template](https://github.com/yohannesjoya/toolreceipt/issues/new/choose).

## License

[MIT](LICENSE)
