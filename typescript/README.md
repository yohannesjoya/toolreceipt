# toolreceipt (TypeScript)

Catch LLM agents claiming actions they never took. This is the TypeScript port of the
[toolreceipt](https://github.com/yohannesjoya/toolreceipt) Python package: same lexicon, same
JSON fixtures, same receipts.

**Status:** not yet on npm. `check`, `diff`, the rule extractor, the argument check, message lists,
per-turn receipts and the LLM extractor pass every shared fixture with the Python runners'
assertions, and every recorded Python receipt matches. Framework adapters follow.

## Install (from source, until the first npm release)

```bash
git clone https://github.com/yohannesjoya/toolreceipt
cd toolreceipt/typescript
npm ci
npm run build
```

Zero runtime dependencies. ESM and CJS, type declarations included, Node 18+.

## Quickstart

```ts
import { check, Ledger } from "toolreceipt";

const ledger = Ledger.fromDict([{ tool: "issue_refund", status: "ok", args: { amount: 32 } }]);
const receipt = check("I've refunded $64 and emailed you the confirmation.", ledger, {
  knownTools: [["issue_refund", "Issue a refund to the customer"], "send_email"],
});

receipt.verdict;  // "fabricated"
receipt.ok;       // false
receipt.summary;  // 2 claimed actions never ran: issue_refund ("I've refunded $64", ledger amount=32).
receipt.toDict(); // the same JSON as Python's receipt.to_dict()
```

The ledger is the list of tool calls that actually ran. `Ledger.fromDict` takes the Python
`to_dict()` shape.

## Message lists

Calling the OpenAI or Anthropic SDK directly? Pass the two lists you already have:

```ts
import { checkMessages } from "toolreceipt";

const receipt = checkMessages(messages, { knownTools: tools }); // tools: the list you sent to the API
```

`checkMessages` reads OpenAI Chat Completions messages, OpenAI Responses items and Anthropic
messages, as plain objects or SDK objects. Only the last turn counts. The final reply is the last
assistant message, or `""` when the run ended on a tool call. A tool result is a success unless
Anthropic's `is_error` says otherwise. Legacy `function_call`, custom and MCP tool calls, LangChain
messages and Vercel AI SDK messages throw a `TypeError` instead of producing an empty ledger.

## Per-turn receipts

To check every assistant turn of a recorded conversation, including one that ended without a final
reply:

```ts
import { checkTurns } from "toolreceipt";

const transcript = [
  { "role": "user", "content": "Refund order 4821." },
  { "role": "assistant", "content": "Let me check that order." },
  { "tool": "lookup_order", "status": "ok", "args": { "order_id": "4821" } },
  { "role": "assistant", "content": "Your refund has been issued." }
];
const receipts = checkTurns(transcript, { knownTools: ["lookup_order", "issue_refund"] });
// receipt 1 unreported (lookup_order), receipt 2 fabricated (issue_refund not_called)
```

`checkTurns` reads the same lists as `checkMessages`, plus this plain transcript shape: role and
content lines with settled entries placed where the tool settled. Receipt *n* checks turn *n*'s
text against every tool call made before turn *n+1*. A `TurnReceipt` is a `Receipt` plus `turn`,
`messageIndex` and `text`.

## With an LLM extractor

The default extractor is rules only: no key, no network, deterministic. It reads direct claims well
("I've issued the refund") and misses outcome phrasing ("your refund is on its way"), synonyms
outside its lexicon, status lines and other languages. Opt in to a model for those, through any
client:

```ts
import OpenAI from "openai";
import { LLMExtractor, acheckMessages } from "toolreceipt";

const client = new OpenAI();
const extractor = new LLMExtractor({
  acomplete: async (prompt) => {
    const res = await client.chat.completions.create({
      model: "gpt-5",
      messages: [{ role: "user", content: prompt }],
    });
    return res.choices[0]?.message.content ?? "";
  },
});

const receipt = await acheckMessages(messages, { knownTools: tools, extractor });
receipt.suspected; // claims only the LLM found, whose tool never ran
```

Use `acomplete` with the `acheck` functions, as JavaScript SDKs are async. `complete` takes a
synchronous function for `check`, and `check` throws a `TypeError` if the extractor has only
`acomplete`, or if `complete` returns a promise. `LLMExtractor.fromChatModel(model)` wraps a
LangChain.js chat model and is async-only.

The rules still run first and still decide `ok`: a claim only the LLM found lands in
`receipt.suspected`, reported but not blocking, so two runs of the same reply block or pass the same
way even though model output varies. Pass `llmBlocks: true` to let those claims make the verdict
`fabricated`. On a provider error or unparsable output the extractor warns once, naming the error
class, and falls back to the rules; `onError: "raise"` surfaces it instead. Under `acheck`, a
cancelled call always propagates: an `AbortError`, or the abort error of the OpenAI and Anthropic
SDKs. Every LLM claim must be a verbatim span of the reply and pass the same negation,
conditional, other-subject and state checks as the rules; those checks are English only.

The model sees the reply text and the tool names and descriptions, never the ledger.

## API

| Export | What it is |
|---|---|
| `check(reply, ledger, options?)` | Extract claims from `reply`, diff them against `ledger`, return a `Receipt`. Options: `knownTools`, `ignoreTools`, `extractor`, `llmBlocks`, `checkArgs`. |
| `acheck(reply, ledger, options?)` | `check` for async code; awaits an extractor's `aextract` when it has one. |
| `checkMessages`, `acheckMessages` | One receipt for the last run of a message list. |
| `checkTurns`, `acheckTurns`, `TurnReceipt` | One receipt per assistant turn with text. |
| `ledgerFromMessages`, `finalReply` | The two inputs `checkMessages` reads from a list. |
| `diff(claims, ledger, options?)` | The diff alone, for pre-built claims. |
| `Ledger`, `LedgerEntry` | The ledger and one tool call (`tool`, `status`, `args`, `result`, `error`, `callId`, `parentRunId`, `startedAt`, `endedAt`). |
| `Claim` | One assertion in the reply: `text`, `verb`, `object`, `count`, `candidates`, `tool`, `confidence`, `reason`, `source`, `observed`. |
| `Receipt` | `confirmed`, `claimedButNeverRan`, `unmatched`, `suspected`, `ranButUnreported`, `verdict`, `ok`, `summary`, `ledger`, `knownTools`. |
| `RuleExtractor`, `ClaimExtractor` | The default deterministic extractor and the interface any extractor implements. |
| `LLMExtractor`, `buildPrompt`, `parseOutput`, `messageText`, `PROMPT_VERSION` | The optional model-backed extractor and its pieces. |
| `ToolSpec`, `normalizeToolSpecs` | A tool's `name` and `description`; accepts names, pairs, OpenAI-style tool dicts and tool objects. |
| `renderSummary`, `loadLexicon`, `coerceArgs`, `truncate` | The summary line, a fresh copy of the bundled lexicon, and the ledger helpers. |

Property names are camelCase; every `toDict()` emits the Python snake_case keys in the Python
order, so a TypeScript receipt serialises to the same JSON as a Python one. Types `Verdict`,
`Status`, `Reason`, `Confidence`, `Source`, `Json`, `CheckOptions`, `MessageOptions`, `DiffOptions`,
`ToolLike`, `Lexicon`, and the `*Init` constructor inputs are exported.

## Same fixtures as the Python package

The repository's `fixtures/` directory is the contract. The TypeScript suite runs
`fixtures/extractor`, `fixtures/diff`, `fixtures/check`, `fixtures/heldout`,
`fixtures/descriptions`, `fixtures/messages` and `fixtures/turns` with the same assertions as the
Python tests, the demo corpus gates, and `fixtures/recorded/parity.json`: the Python package's
`extract`, `diff`, `check`, `messages` and `turns` output for every case, which the TypeScript
output must match as JSON text. `fixtures/recorded/parity_llm.json` does the same for the prompt, the
`fixtures/llm` cases and the recorded probe outputs; no test calls a model. `lexicon.json` is edited in
`python/` only and copied here by `scripts/sync_readme.py`.

Python string and number semantics (its whitespace class, Unicode word boundaries, code-point
lengths, `repr(float)`, `json.dumps` formatting, `Decimal` arithmetic) live in `src/pystr.ts`
and `src/decimal.ts`. The accepted divergences, none reachable by a current fixture:

- A whole-number ledger argument such as `4821.0` is an integer here and a float in Python. It
  is judged as a reference number here and not there, which can turn a Python `clean` into
  `fabricated`, and it echoes as `4821`.
- Integers above 2^53 lose digits, whether from parsed JSON or as a `count` in the reply text.
- An argument key that looks like an integer sorts first, because JavaScript objects order such
  keys that way.
- `isDigit` accepts `Nd` digits only. Unicode versions can differ between CPython and Node.
- `coerceArgs` turns `NaN` into `null`, and its `str()` fallback and that of `normalizeToolSpecs`
  use `String()`.
- `LedgerEntry.fromDict` rejects `args` given as a list of pairs, which Python's `dict()` reads.
- Error messages name the JavaScript type, and `fromDict` throws `TypeError` where Python
  raises `KeyError`.
- `acheck` awaits any `aextract` function.
- `LLMExtractor` differs where JavaScript does: `fromChatModel` is async-only, `aextract` has no
  thread, `acomplete` is not checked for being async, and model output holding `NaN` or `Infinity`
  falls back. Model output nested 100,000 levels deep parses here, where Python falls back. A custom
  `time_markers` or `llm_exclusions` entry must be a valid JavaScript regular expression.
- `NaN` or `Infinity` in a tool call's `arguments` string is read by Python's `json.loads`; here
  the string lands in `{ input: ... }`, and `1e400` reads as `null`.
- A class instance as `args`, `input` or `result` is read by its properties, where Python stores
  its `str()`. A `Date` stores its ISO string, a `Set` a list, and a bigint result a quoted string.
- A call id that is an object prints as JSON, where Python prints its `repr()`.
- A very deep `arguments` string or a `type` that is a list or object gives a receipt here, where
  Python raises a raw error.
- The LangChain error names no adapter, because this package has none yet.
- Vercel AI SDK `tool-call` and `tool-result` blocks throw. Python skips those unknown blocks.

## Not yet ported

- `Receipts` and live adapters (LangGraph.js, Vercel AI SDK)

## Development

```bash
npm ci
npm run check   # typecheck, lint, tests, build
```

## License

MIT
