# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-10-30

### Added

- `check_messages()` and `acheck_messages()` (`toolreceipt.messages`): a receipt from a plain
  message list with no framework installed. OpenAI Chat Completions messages, OpenAI
  Responses items and Anthropic messages, as dicts or SDK objects; only the last turn (after
  the last user message with text and no tool result) is read; the final reply is `""` when
  the run ends on a tool call; the legacy `function_call` shape, custom and MCP tool calls
  and LangChain message objects raise `ValueError`; a built-in tool spec in `known_tools` is
  skipped.
  `toolreceipt.messages.ledger_from_messages` and `final_reply` are the framework-free twins
  of the LangChain adapter's helpers.
- Argument check: a claim whose tool ran with arguments that contradict the one money amount
  or cued reference number in its clause is `claimed_but_never_ran` with the new reason
  `args_mismatch`. Only top-level arguments whose key names the same kind are compared; a
  cents-style value, a total split across calls and a number read from the tool's result
  agree, as does a display number a lookup mapped to the id; names, email addresses, dates
  and quantities are never compared. Amounts are counted over the sentence, and an amount is
  not judged when the words around it qualify it ("half of your $64 order", "the $64 jacket",
  "$16 each", "$1.5k", "around $64", "€1.250,00"). `check_args=False` on `check`, `acheck`, `check_messages`,
  `acheck_messages`, `diff` and `Receipts` turns the check off. Lexicon keys `amount_keys` and `id_cues`; both default to empty when absent.
- `Claim.observed`: the contradicting arguments of an `args_mismatch` claim (`{"amount": 32}`),
  `None` elsewhere; emitted last by `to_dict()`.
- `Receipts.check(..., config=cfg)` and `acheck(..., config=cfg)` read the run whose config is
  given, so one `Receipts` can serve concurrent runs.
- `LLMExtractor(acomplete=...)` for async-only clients, awaited natively by `aextract`;
  `from_chat_model` wires the model's `ainvoke` too. `extract()` on an async-only extractor
  raises `TypeError`.

### Changed

- Verdicts can change for ledgers that carry arguments: a `clean` 0.3.0 receipt becomes
  `fabricated` when a confirmed claim's amount or reference number contradicts the call.
  `check_args=False` restores the 0.3.0 receipt. Every existing fixture and probe recording is
  unchanged; the 19 recorded demo runs and 6 demo examples produce no `args_mismatch`.
- The summary line names the contradicting argument for an `args_mismatch` claim:
  `1 claimed action never ran: issue_refund ("I've refunded $64", ledger amount=32).`
- `NO_RESULT` lives in `toolreceipt.ledger` (still importable from `toolreceipt.langgraph`).

## [0.3.0] - 2026-10-30

### Added

- `acheck()` and `Receipts.acheck()`: the same receipt as `check()` for async code. An
  extractor's `aextract` coroutine is awaited when it defines one; the rules alone do no I/O.
- `LLMExtractor.aextract()`: the model call runs on a thread (`asyncio.to_thread`), so the
  event loop stays free. `extract()` and `aextract()` share one code path.
- Description-based tool matching: a tool whose name carries no verb (`jira`, `stripe`,
  `email_customer`) is matched through the first verb of its description ("Create a ticket
  in Jira"). Only the lead verb's direct object ("a ticket") is object evidence; the rest of
  the description is not, so a generic word such as "customer" or "order" never turns an
  honest reply about a sibling tool into a fabrication. Verb-named tools behave as before
  unless the reply names a verbless tool of the same action, which then ties with them ("a
  ticket in Jira"). A description that leads with a read verb, or names one in its first
  clause, keeps the tool unmatched. No existing fixture changes. Sibling demotion
  (`family_ran`) stays name-only.
- Lexicon keys `description_lead_skip` (filler phrases stripped before the lead verb) and
  `description_lead_nouns` (noun tokens accepted as a lead verb when an object follows them:
  "Email the customer", not "Email address validation"); both default to empty when absent,
  so a 0.2 lexicon file loads. `fixtures/descriptions/sample.json` pins the lead verb of 55
  real descriptions.
- Public helper `toolreceipt.claims.description_family(description, lexicon)`.

### Changed

- `ReceiptCallbackHandler.run_inline = True`: async callers no longer pay one executor hop
  per callback event. Sync `invoke` is unaffected.
- `email_customer`-style names (noun tokens only) now match through their description. The
  README example for verbless names changed accordingly.
- README probe table, rules-only rows re-measured: dev 36 / 113 claims (was 34), 15 / 67
  fabrications caught (was 14); held-out 5 / 35 (was 4), 1 / 18 (was 1). The verbless probe
  tools `tool_b` and `tool_c` are now matchable; the numbers match the estimate made before
  implementation. LLM rows unchanged.

### Documented

- 0.2.0 shipped these public names without listing them: `toolreceipt.llm.build_prompt`,
  `parse_output`, `message_text` and `MAX_SPAN_CHARS`.

## [0.2.0] - 2026-10-22

### Added

- `LLMExtractor` (`toolreceipt.llm`): an opt-in extractor that runs the rules first and asks
  any chat model for the claims they cannot see (outcome phrasing, synonyms, passive voice,
  status lines, other languages). Takes a plain `complete(prompt) -> str` callable, or any
  LangChain chat model through `LLMExtractor.from_chat_model(model)`. Six precision guards
  reuse the rule lexicon, including its sentence-level drops (conditional sentences, a
  subject inherited from the previous clause); on any provider error it falls back to the
  rules (`on_error`).
  No new dependency, no new extra; nothing changes without constructing one.
- `Receipt.suspected`: claims found only by the LLM whose tool never ran. Reported, never
  blocking by default. `summary` gains a suffix when it is non-empty.
- `Claim.source` (`"rules"` or `"llm"`); `from_dict()` defaults it, so existing fixtures load.
- `llm_blocks=` on `check()`, `Receipts()` and `diff()`: let LLM-only claims make the verdict
  `fabricated`. Under it, a never-called candidate whose sibling tool ran (shared verb token)
  is `unmatched` rather than blocking.
- Probe sets under `fixtures/probe/` (114-case dev set, 30-case held-out set written before
  tuning) with committed model recordings replayed offline by `tests/test_probe_replay.py`,
  and `scripts/eval_llm.py` to re-record them; `fixtures/llm/` pins the guards and the merge.

### Changed

- `Claim.to_dict()` emits `source` and `Receipt.to_dict()` emits `suspected` (additive).
  Rules-only behaviour is otherwise byte-identical to 0.1.1.

## [0.1.1] - 2026-10-20

### Fixed

- `known_tools` accepts tool objects with a `name` and a `description`, such as LangChain's
  `BaseTool`. Passing one used to raise `TypeError: expected string or bytes-like object`.
  An entry of any other shape now raises a `TypeError` that names `known_tools`.

## [0.1.0] - 2026-10-20

### Added

- `Ledger` and `LedgerEntry`: JSON-safe record of tool calls, with truncation of results and errors.
- `Claim` and `ClaimExtractor` protocol, with `RuleExtractor`, a deterministic, key-free,
  stdlib-only extractor driven by a packaged lexicon of verb families and exclusion rules
  (negation, including "nothing" and "none", future, advice, conditionals, other subjects).
- Diff and `Receipt`: `confirmed`, `claimed_but_never_ran` (reasons `not_called` and
  `tool_failed`), `unmatched` and `ran_but_unreported`, a verdict (`clean`, `unreported`,
  `fabricated`) and a one-line summary. A claim that matches a tool on the verb family alone
  is never reported as `not_called`; it lands in `unmatched` unless that tool ran or failed.
- `check()` for framework-free use and the `Receipts` facade (`config()`, `check()`, `ledger`)
  with `known_tools`, `ignore_tools`, `capture_results` and `max_result_chars`.
- LangGraph adapter (`toolreceipt[langgraph]`): `ReceiptCallbackHandler`, `merge_config`,
  `final_reply`, `ledger_from_messages` and bound-tool discovery. Callbacks never raise:
  a call whose arguments or output cannot be serialised is still recorded.
- JSON fixtures for extractor, diff and end-to-end `check()` behaviour, README quickstart
  executed as a test, `CONTRIBUTING.md`, issue templates and an architecture sketch.
- Held-out set in `fixtures/heldout/` (live honest-mode replies and adversarial negatives)
  with `tests/test_heldout.py`, which prints precision; live demo replays in `demo/replays/`
  and `demo/scripts/collect_runs.py` to record them.
