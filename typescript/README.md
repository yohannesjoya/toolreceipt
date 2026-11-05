# toolreceipt (TypeScript)

Catch LLM agents claiming actions they never took. This is the TypeScript port of the
[toolreceipt](https://github.com/yohannesjoya/toolreceipt) Python package: same lexicon, same
JSON fixtures, same receipts.

**Status:** core ported, not yet on npm. `check`, `diff`, the rule extractor and the argument
check pass every shared fixture with the Python runners' assertions, and every recorded Python
receipt deep-equals. Message lists, per-turn receipts, the LLM extractor and framework adapters
follow.

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

The ledger is the list of tool calls that actually ran; you build it from your own framework's
trace (`Ledger.fromDict` takes the Python `to_dict()` shape). Adapters that build it for you
are planned.

## API

| Export | What it is |
|---|---|
| `check(reply, ledger, options?)` | Extract claims from `reply`, diff them against `ledger`, return a `Receipt`. Options: `knownTools`, `ignoreTools`, `extractor`, `llmBlocks`, `checkArgs`. |
| `acheck(reply, ledger, options?)` | `check` for async code; awaits an extractor's `aextract` when it has one. |
| `diff(claims, ledger, options?)` | The diff alone, for pre-built claims. |
| `Ledger`, `LedgerEntry` | The ledger and one tool call (`tool`, `status`, `args`, `result`, `error`, `callId`, `parentRunId`, `startedAt`, `endedAt`). |
| `Claim` | One assertion in the reply: `text`, `verb`, `object`, `count`, `candidates`, `tool`, `confidence`, `reason`, `source`, `observed`. |
| `Receipt` | `confirmed`, `claimedButNeverRan`, `unmatched`, `suspected`, `ranButUnreported`, `verdict`, `ok`, `summary`, `ledger`, `knownTools`. |
| `RuleExtractor`, `ClaimExtractor` | The default deterministic extractor and the interface any extractor implements. |
| `ToolSpec`, `normalizeToolSpecs` | A tool's `name` and `description`; accepts names, pairs, OpenAI-style tool dicts and tool objects. |
| `renderSummary`, `loadLexicon`, `coerceArgs`, `truncate` | The summary line, a fresh copy of the bundled lexicon, and the ledger helpers. |

Property names are camelCase; every `toDict()` emits the Python snake_case keys in the Python
order, so a TypeScript receipt serialises to the same JSON as a Python one. Types `Verdict`,
`Status`, `Reason`, `Confidence`, `Source`, `Json` are exported.

## Same fixtures as the Python package

The repository's `fixtures/` directory is the contract. The TypeScript suite runs
`fixtures/extractor`, `fixtures/diff`, `fixtures/check`, `fixtures/heldout` and
`fixtures/descriptions` with the same assertions as the Python tests, the demo corpus gates,
and `fixtures/recorded/parity.json`: the Python package's `extract`, `diff` and `check` output
for every case, which the TypeScript output must deep-equal. `lexicon.json` is edited in
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

## Not yet ported

- `checkMessages` / `checkTurns` over OpenAI, Anthropic and transcript message lists
- `LLMExtractor`
- `Receipts` and live adapters (LangGraph.js, Vercel AI SDK)

## Development

```bash
npm ci
npm run check   # typecheck, lint, tests, build
```

## License

MIT
