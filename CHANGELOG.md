# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
