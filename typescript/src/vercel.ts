/**
 * A finished Vercel AI SDK run (5, 6 or 7) as a ledger and a reply, by key, with no `ai` import.
 * Reads `steps` and each step's `content`, never `result.toolCalls`: AI SDK 7 changed that from the
 * last step to all steps.
 */

import { ToolSpec, typeName } from "./claims";
import { toolDescriptions } from "./capture";
import { Ledger, LedgerEntry, NO_RESULT, asText, coerceArgs, errorText } from "./ledger";
import { type MessageOptions, get } from "./messages";
import { pyStr, pyTruthy } from "./pystr";
import type { Receipt } from "./receipt";
import { acheck, check } from "./receipts";

function stepsOf(result: unknown): unknown[] {
  const steps = get(result, "steps");
  if (steps !== null && typeof (steps as { then?: unknown }).then === "function") {
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

/** One entry per tool call of the run, paired with its result by id. A call with no result is
 * `status: "error", error: "no result"`. A preliminary result of a streaming tool does not count. */
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
  return new Ledger(entries);
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

/** `check(finalReplyFromResult(result), ledgerFromResult(result), options)`. Pass `knownTools`, for
 * example `knownToolsFrom(tools)`: without it a tool that was never called cannot be `not_called`. */
export function checkResult(result: unknown, options: MessageOptions = {}): Receipt {
  return check(finalReplyFromResult(result), ledgerFromResult(result), options);
}

/** `checkResult` for async code; see `acheck`. */
export async function acheckResult(result: unknown, options: MessageOptions = {}): Promise<Receipt> {
  return acheck(finalReplyFromResult(result), ledgerFromResult(result), options);
}

/** The tools of a record, with their descriptions, for `knownTools`. */
export function knownToolsFrom(tools: unknown): ToolSpec[] {
  return toolDescriptions(tools, "knownToolsFrom").map(([name, description]) => new ToolSpec(name, description));
}
