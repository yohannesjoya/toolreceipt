# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-22

### Added

- `LLMExtractor` (`toolreceipt.llm`): an opt-in extractor that runs the rules first and asks
  any chat model for the claims they cannot see (outcome phrasing, synonyms, passive voice,
  status lines, other languages). Takes a plain `complete(prompt) -> str` callable, or any
  LangChain chat model through `LLMExtractor.from_chat_model(model)`. Six precision guards
  reuse the rule lexicon; on any provider error it falls back to the rules (`on_error`).
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
