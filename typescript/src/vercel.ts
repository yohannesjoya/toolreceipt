/**
 * A finished Vercel AI SDK run (5, 6 or 7) as a ledger and a reply, by key, with no `ai` import.
 * Reads `steps` and each step's `content`, never `result.toolCalls`: AI SDK 7 changed that from the
 * last step to all steps. An approved tool runs before the request's first step, so its result is
 * only in `responseMessages` (AI SDK 7) or `response.messages` (5 and 6), and is read from there.
 */

import { ToolSpec, typeName } from "./claims";
import { isThenable, splitTools } from "./capture";
import { Ledger, LedgerEntry, NO_RESULT, asText, coerceArgs, errorText } from "./ledger";
import { type MessageOptions, get, pair } from "./messages";
import { pyStr, pyTruthy } from "./pystr";
import type { Receipt } from "./receipt";
import { acheck, check } from "./receipts";

function stepsOf(result: unknown): unknown[] {
  const steps = get(result, "steps");
  if (isThenable(steps)) {
    throw new TypeError("steps is a promise, as streamText returns it; pass { steps: await result.steps }");
  }
  if (!Array.isArray(steps)) {
    throw new TypeError(`a Vercel AI SDK result needs a steps array, got ${typeName(result)}`);
  }
  return steps as unknown[];
}

function contentOf(step: unknown): unknown[] {
  const content = get(step, "content");
  if (!Array.isArray(content)) throw new TypeError("a step needs a content array, which AI SDK 5 and newer have");
  return content as unknown[];
}

/** Entries for results in the response list whose calls the steps do not hold: tools that ran before
 * the first step, on a request that continues an earlier one. */
function earlierEntries(result: unknown, seen: ReadonlySet<string>): LedgerEntry[] {
  const list = get(result, "responseMessages") ?? get(get(result, "response"), "messages");
  if (!Array.isArray(list)) return [];
  const toolMessages = (list as unknown[]).filter((m) => get(m, "role") === "tool");
  return pair(toolMessages).filter((e) => !seen.has(e.callId ?? ""));
}

/** One entry per tool call of the run, paired with its result by id. A call with no result is
 * `status: "error", error: "no result"`. A preliminary result of a streaming tool does not count.
 * Under `streamText`, pass `{ steps: await result.steps, responseMessages: await result.responseMessages }`. */
export function ledgerFromResult(result: unknown): Ledger {
  const entries: LedgerEntry[] = [];
  const open = new Map<string, LedgerEntry[]>();
  for (const step of stepsOf(result)) {
    for (const part of contentOf(step)) {
      const type = get(part, "type");
      const id = pyStr(get(part, "toolCallId") ?? "");
      if (type === "tool-call") {
        const entry = new LedgerEntry({
          tool: pyStr(get(part, "toolName")), status: "error", args: coerceArgs(get(part, "input")), error: NO_RESULT, callId: id,
        });
        entries.push(entry);
        const queue = open.get(id);
        if (queue) queue.push(entry);
        else open.set(id, [entry]);
      } else if (type === "tool-result" && !pyTruthy(get(part, "preliminary"))) {
        const entry = open.get(id)?.shift();
        if (entry === undefined) continue;
        entry.status = "ok";
        entry.error = null;
        entry.result = asText(get(part, "output"));
      } else if (type === "tool-error") {
        const entry = open.get(id)?.shift();
        if (entry !== undefined) entry.error = errorText(get(part, "error"));
      }
    }
  }
  return new Ledger([...earlierEntries(result, new Set(open.keys())), ...entries]);
}

/** The last step's text, joined as `result.text` joins it, or `""` when that step holds a tool call
 * the provider did not run: a run that ends on a tool call has no final reply. */
export function finalReplyFromResult(result: unknown): string {
  const steps = stepsOf(result);
  if (steps.length === 0) return "";
  const parts = contentOf(steps[steps.length - 1]);
  if (parts.some((p) => get(p, "type") === "tool-call" && !pyTruthy(get(p, "providerExecuted")))) return "";
  return parts.filter((p) => get(p, "type") === "text").map((p) => pyStr(get(p, "text") ?? "")).join("");
}

/** `check(finalReplyFromResult(result), ledgerFromResult(result), options)`. Pass `toolOptions(tools)`:
 * without it a tool that was never called cannot be `not_called`. */
export function checkResult(result: unknown, options: MessageOptions = {}): Receipt {
  return check(finalReplyFromResult(result), ledgerFromResult(result), options);
}

/** `checkResult` for async code; see `acheck`. */
export async function acheckResult(result: unknown, options: MessageOptions = {}): Promise<Receipt> {
  return acheck(finalReplyFromResult(result), ledgerFromResult(result), options);
}

/** `knownTools` and `unseenTools` from a tools record. A tool with no `execute` is unseen: a browser
 * tool's call can sit in an earlier request, so never seeing it here proves nothing. */
export function toolOptions(tools: unknown): { knownTools: ToolSpec[]; unseenTools: ToolSpec[] } {
  const { runs, unseen } = splitTools(tools, "toolOptions");
  const spec = ([name, description]: [string, string]): ToolSpec => new ToolSpec(name, description);
  return { knownTools: runs.map(spec), unseenTools: unseen.map(spec) };
}
