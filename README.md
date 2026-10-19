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

Precision comes before recall. Negation ("I did not send"), the future ("I'll send"),
advice ("you can create a ticket"), conditionals and offers are not claims.

## What it does not catch

- **Tools that return an error string with a success status.** If a tool returns
  `"Error: SMTP timeout"` as a normal value, the framework reports success and so does the
  ledger. Raise, or return an error `ToolMessage`.
- **Argument mismatches.** "I emailed Bob" passes if `send_email` ran, whatever `to` was.
- **Quantities.** "Updated 3 rows" against one call is recorded as a count on the claim, never judged.
- **Non-English replies.** English only.
- **Streaming and mid-run checks.** Only the final reply is checked.
- **Tool names without a verb.** A tool called `email_customer` never matches, so claims
  about it land in `unmatched` (informational, they never affect the verdict).
- **Loose tool matching.** A claim maps to a tool only when the tool name shares the
  claim's action verb family (`send`, `create`, `update`, `delete`, `refund`, ...).
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
On the held-out set (`fixtures/heldout/`: those 10 honest replies plus 12 hand-written
adversarial negatives, labelled before any rule change) precision is 1.00 (20 of 20 claims)
with no claim on any adversarial negative. The 10 honest replies are near-identical in
phrasing ("I've issued a $64 refund ... and sent a confirmation email"), so this precision
figure is weak evidence. `fixtures/check/` adds hand-written honest replies with more varied
phrasing. The recorded runs in `demo/replays/` back the demo's Replay button.

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
`Receipts` per run. Your tools and graph stay unchanged, apart from
`ToolNode(tools, handle_tool_errors=True)`.

## Vocabulary

| Term | Meaning |
|---|---|
| **Ledger** | Ordered list of ledger entries captured during one agent run, ordered by `started_at`. |
| **Ledger entry** | One tool invocation: `tool`, `status` (`ok` or `error`), `args`, `result`, `error`, `call_id`, timestamps. |
| **Claim** | One assertion in the reply that an action was performed: quoted `text`, `verb`, `object`, `count`, candidate `tool`s. |
| **Receipt** | The result of diffing claims against the ledger. |
| **Verdict** | `clean`, `unreported` or `fabricated`. Only `fabricated` is blocking (`receipt.ok` is `False`). |

`Claim.text` is the clause after normalisation (curly quotes straightened, runs of whitespace
folded), so it can differ from the raw substring of the reply.

A receipt has four lists: `confirmed` (a successful entry exists for the claim),
`claimed_but_never_ran`, `unmatched` (no known tool fits the claim closely enough to judge
it, informational) and
`ran_but_unreported`. Every type has `to_dict()`, and every field is JSON-safe.

## Architecture

![toolreceipt architecture: a LangGraph run feeds a ledger and an extractor, the diff produces a receipt](https://raw.githubusercontent.com/yohannesjoya/toolreceipt/main/docs/architecture.svg)

The adapter is a LangChain callback handler, so capture needs no tool wrapping. The extractor
(`ClaimExtractor`) and the adapter are small protocols and the default extractor,
`RuleExtractor`, is a lexicon of verb families plus exclusion rules.

## Post-hoc usage

Already have the messages from a finished run, or a different framework? Build a ledger and
call `check()`:

```python
from toolreceipt import check
from toolreceipt.langgraph import final_reply, ledger_from_messages

ledger = ledger_from_messages(result["messages"])
receipt = check(final_reply(result), ledger, known_tools=["issue_refund", "send_email"])
```

`ledger_from_messages` pairs each `AIMessage` tool call with its `ToolMessage`. A call with
no result is recorded as an error. Timestamps are `None` on this path.

Pass `known_tools`. A ledger only knows the tools that ran, so without `known_tools` a claim
about a tool that was never called ("I emailed you" when `send_email` never ran) has nothing
to match and cannot be flagged. Names are enough; `(name, description)` pairs and
OpenAI-style tool dicts also work. In a live run (`Receipts.config()`), bound tools are
discovered automatically.

Without LangGraph, build a `Ledger` of `LedgerEntry` objects yourself and call `check()`.

## Configuration

```python
rc = Receipts(
    extractor=None,          # any ClaimExtractor; default RuleExtractor()
    known_tools=(),          # extra tools to match claims against
    ignore_tools=(),         # tools never reported as ran_but_unreported (e.g. lookups)
    capture_results=True,    # False: ledger keeps no args or results
    max_result_chars=2000,   # truncate result and error text in the ledger
)
```

- `known_tools`: names, `(name, description)` pairs or tool dicts. Merged with the tools
  discovered from the run.
- `ignore_tools`: read-only tools you do not want listed as unreported.
- `capture_results=False`: the ledger records only tool name, status and error. Use it when
  tool arguments or results are sensitive.
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
and return. Receipts are for your own logs and your own code. Nothing is sent anywhere, and
the library makes no network calls. Use `capture_results=False` to drop args and results,
and `max_result_chars` to bound what is kept.

## Roadmap

- LLM-backed extractor behind the existing `ClaimExtractor` protocol
- TypeScript port, sharing the JSON fixtures as the contract
- Adapters: Mastra, CrewAI, Vercel AI SDK
- Argument check ("emailed Bob" against `to="alice"`)
- Async API

## Contributing

Lexicon and fixture contributions are JSON only. See [CONTRIBUTING.md](CONTRIBUTING.md). If
the reply you expected to be flagged was not, or a clean reply was flagged, open an issue
with the [missed claim or false positive template](https://github.com/yohannesjoya/toolreceipt/issues/new/choose).

## License

[MIT](LICENSE)
