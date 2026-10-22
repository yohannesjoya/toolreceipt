# Contributing to toolreceipt

Thanks for helping. The most useful contributions are real replies the library got wrong.
Those are JSON fixtures and need no Python knowledge.

## Run the tests

```bash
cd python
uv sync --all-extras --group dev
uv run pytest -q
uv run ruff check .
uv run mypy
```

CI runs the same three commands on Python 3.10 to 3.13. The core package (everything but
`toolreceipt/langgraph.py`) imports only the standard library. Keep it that way.

## Principle: precision over recall

A guard that cries wolf gets removed. A reply that is not a clear claim of a completed action
must not produce a claim. A new negative fixture that produces a claim is a bug. When in
doubt, add the negative fixture first.

## Add an extractor fixture

Fixtures live in `fixtures/extractor/*.json` and are picked up automatically by
`python/tests/test_extractor_fixtures.py`. Name positive cases for the behaviour
(`send_email_past_tense.json`) and negative cases with the prefix `neg_`
(`neg_future.json`). A `neg_` fixture must have `"expected": []`.

```json
{
  "text": "I assigned the ticket to the billing team.",
  "known_tools": [
    {"name": "update_ticket", "description": "Change a ticket's fields"},
    {"name": "create_ticket", "description": "Open a support ticket"}
  ],
  "expected": [
    {
      "verb": "update",
      "object": "ticket to the billing team",
      "count": null,
      "text": "I assigned the ticket to the billing team",
      "candidates": ["update_ticket"],
      "confidence": "high"
    }
  ]
}
```

`known_tools` entries are `{name, description}`. Each expected claim may list any of `verb`,
`object`, `count`, `text`, `candidates`, `confidence`; only the keys you list are compared.
The number of claims must match exactly. `candidates` is `[]` for an unmatched claim.

## Add a diff fixture

Fixtures live in `fixtures/diff/*.json` and run through `python/tests/test_diff_fixtures.py`.
They take pre-built claims and a ledger, so they test the diff alone.

```json
{
  "claims": [
    {"text": "I have emailed you", "verb": "send", "candidates": ["send_email"], "confidence": "high"}
  ],
  "ledger": [{"tool": "lookup_order", "status": "ok"}],
  "known_tools": ["send_email", "lookup_order"],
  "ignore_tools": [],
  "expected": {
    "verdict": "fabricated",
    "ok": false,
    "confirmed": [],
    "claimed_but_never_ran": [{"tool": "send_email", "reason": "not_called"}],
    "unmatched": [],
    "ran_but_unreported": ["lookup_order"],
    "summary": "1 claimed action never ran: send_email (\"I have emailed you\")."
  }
}
```

Claims and ledger entries use the field sets in the README's Vocabulary section; missing
optional fields take their defaults. In `expected`, each claim entry lists the fields to
compare, and `ran_but_unreported` is a list of tool names.

## Add an end-to-end fixture

Fixtures in `fixtures/check/*.json` run a reply through `check()` (extractor and diff
together) via `python/tests/test_check_fixtures.py`. Use them for honest replies that must
never be `fabricated`, and for fabrications that must stay caught.

```json
{
  "text": "I've raised this with our team.",
  "known_tools": [
    {"name": "create_ticket", "description": "Create a support ticket for follow-up by the team."},
    {"name": "escalate_ticket", "description": "Escalate a ticket to a specialist team."}
  ],
  "ledger": [{"tool": "escalate_ticket", "status": "ok"}],
  "expected": {"verdict": "unreported"}
}
```

## Add an LLM fixture

Fixtures in `fixtures/llm/*.json` replay recorded model output through `LLMExtractor` with no
network (`python/tests/test_llm_fixtures.py`). They pin the precision guards and the merge
with the rule claims. Record one from a real run:

```python
extractor = LLMExtractor(complete=my_call)
extractor.extract(reply, known_tools)
print(extractor.last_raw)   # paste into "llm_output"
```

`last_raw` is written on every call, so read it from an extractor that only this thread uses.

```json
{
  "reply": "Your refund is on its way and you'll find the confirmation in your inbox.",
  "known_tools": [
    {"name": "issue_refund", "description": "Issue a refund to the customer for an order."},
    {"name": "send_email", "description": "Send an email to the customer."}
  ],
  "llm_output": "{\"claims\": [{\"text\": \"Your refund is on its way\", \"tool\": \"issue_refund\"}]}",
  "expected": [
    {"text": "Your refund is on its way", "verb": "other", "candidates": ["issue_refund"], "source": "llm"}
  ],
  "model": "openai/gpt-oss-120b",
  "recorded_at": "2026-10-21"
}
```

`expected` lists every claim the extractor returns, rule claims first (`"source": "rules"`),
then LLM claims; compare any of the extractor-fixture keys plus `source`. A model output the
guards must drop has `"expected": []` (or only the rule claims). An optional `ledger` makes
the test check that `ok` agrees with the rules-only receipt under that ledger. Hand-written
`llm_output` is fine for guard cases; say so in a `note`.

## Add or extend a lexicon family

`python/toolreceipt/lexicon.json` holds the verb families (`send`, `create`, `update`,
`delete`, `refund`, `charge`, `schedule`, `save`, `submit`, `trigger`) and the exclusion
rules (negation, future, advice, conditionals, and so on). A family has:

- `forms`: past-tense verb forms found in replies ("emailed", "notified").
- `tokens`: word stems found in tool names ("email", "notify").

To extend one, add the words and add at least one positive and one negative fixture that
exercise them. A claim maps to a tool only when the tool name shares the claim's verb family,
so `tokens` is what lets `notify_customer` match "I notified the customer". Run the whole
suite: a lexicon change can create claims in existing negative fixtures.

## Add an extractor

An extractor implements the `ClaimExtractor` protocol from `toolreceipt`:

```python
from collections.abc import Sequence
from toolreceipt import Claim, ToolSpec

class MyExtractor:
    def extract(self, text: str, known_tools: Sequence[ToolSpec]) -> list[Claim]: ...
```

Return claims with `candidates` filled in (known tool names tied at the best match) and
leave `tool` and `reason` for the diff. Pass it with `Receipts(extractor=MyExtractor())` or
`check(..., extractor=MyExtractor())`. Run your extractor over the existing fixtures to see
where it differs from `RuleExtractor`.

## Add an adapter

An adapter turns a framework run into the two inputs of `check()`: a `Ledger` of
`LedgerEntry` objects (one per tool call, `status` `"ok"` or `"error"`) and the final reply
text, plus the tool names the agent could use. `python/toolreceipt/langgraph.py` is the
reference: a callback handler for live capture, `ledger_from_messages` for post-hoc use and
`final_reply`. Put framework imports in the adapter module only, never in the core, and add
it as an optional extra in `python/pyproject.toml`. The adapter needs a test graph that
runs without network access; `python/tests/helpers_graph.py` shows one built on a fake model.

## Pull requests

- Conventional commit prefixes (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Tests, `ruff` and `mypy` pass.
- User-visible changes get a line in `CHANGELOG.md` under `[Unreleased]`.

## Reporting a problem

Use the issue templates: **Missed claim** when a fabricated action was not flagged, **False
positive** when a legitimate reply was flagged. Include the reply text and the tool list or
ledger. Remember the ledger may hold personal data: redact it before posting.
