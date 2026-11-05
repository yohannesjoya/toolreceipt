/**
 * Per-turn receipts (port of turns.py): one receipt per assistant turn with text. Receipt n is
 * computed from turn n's text and the tool calls made from the start of the list through the
 * message just before turn n+1, so it is a pure function of that prefix. The ledger is cumulative
 * over the whole list, while `ranButUnreported` lists only the entries whose call sits in the
 * turn's own window. The last turn closes as-is: a call with no result is `tool_failed`.
 */

import { normalizeToolSpecs } from "./claims";
import { Ledger, type LedgerEntry } from "./ledger";
import { type MessageOptions, get, isTurn, pair, text } from "./messages";
import { Receipt, type ReceiptInit, type Verdict, renderSummary } from "./receipt";
import { type CheckOptions, acheck, check } from "./receipts";
import type { Json } from "./types";

export interface TurnReceiptInit extends ReceiptInit {
  turn: number;
  messageIndex: number;
  text: string;
}

/** A `Receipt` for one assistant turn: `turn` is the 1-based ordinal across the list,
 * `messageIndex` the position of its message in the input, `text` the text that was checked. */
export class TurnReceipt extends Receipt {
  turn: number;
  messageIndex: number;
  text: string;

  constructor(init: TurnReceiptInit) {
    super(init);
    this.turn = init.turn;
    this.messageIndex = init.messageIndex;
    this.text = init.text;
  }

  override toDict(): Record<string, Json> {
    return { turn: this.turn, message_index: this.messageIndex, text: this.text, ...super.toDict() };
  }
}

/** `[turnIndex, endExclusive]` per turn. Window n runs from turn n's message (from index 0 for the
 * first turn) through just before turn n+1's message; the windows partition the list. */
export function windows(messages: readonly unknown[]): Array<[number, number]> {
  const turns: number[] = [];
  messages.forEach((m, i) => {
    if (isTurn(m)) turns.push(i);
  });
  return turns.map((index, i) => [index, turns[i + 1] ?? messages.length]);
}

/** `receipt` with `ranButUnreported` cut to the window's entries and the verdict and summary
 * recomputed with `diff()`'s rules. */
function reduce(receipt: Receipt, window: readonly LedgerEntry[], turn: number, messageIndex: number, body: string): TurnReceipt {
  const inWindow = new Set(window);
  const unreported = receipt.ranButUnreported.filter((e) => inWindow.has(e));
  let verdict: Verdict;
  if (receipt.claimedButNeverRan.length > 0) verdict = "fabricated";
  else if (unreported.length > 0) verdict = "unreported";
  else verdict = "clean";
  return new TurnReceipt({
    confirmed: receipt.confirmed,
    claimedButNeverRan: receipt.claimedButNeverRan,
    unmatched: receipt.unmatched,
    ranButUnreported: unreported,
    verdict,
    summary: renderSummary(verdict, receipt.claimedButNeverRan, unreported, receipt.suspected, receipt.confirmed.length),
    ledger: receipt.ledger,
    knownTools: receipt.knownTools,
    suspected: receipt.suspected,
    turn,
    messageIndex,
    text: body,
  });
}

/** `[turn, messageIndex, text, prefix entries, window entries]` per turn. Every message is
 * validated up front; the prefix is re-paired per turn, so each receipt is a function of its prefix
 * by construction. */
function* turnInputs(messages: readonly unknown[]): Generator<[number, number, string, LedgerEntry[], LedgerEntry[]]> {
  const items = [...messages];
  pair(items);
  let seen = 0;
  let n = 0;
  for (const [index, end] of windows(items)) {
    const entries = pair(items.slice(0, end));
    const window = entries.slice(seen);
    seen = entries.length;
    n += 1;
    yield [n, index, text(get(items[index], "content")), entries, window];
  }
}

function withSpecs(options: MessageOptions): CheckOptions {
  // Both may be one-shot iterables, and every turn reads them.
  return {
    ...options,
    knownTools: normalizeToolSpecs(options.knownTools ?? []),
    ignoreTools: [...(options.ignoreTools ?? [])],
  };
}

/**
 * One `TurnReceipt` per assistant turn with text, in message order. Receipt n checks turn n's text
 * against every tool call made before turn n+1; a claim whose call only lands after the next turn
 * is `not_called` on its own receipt. Throws `TypeError` on an unsupported shape anywhere.
 */
export function checkTurns(messages: readonly unknown[], options: MessageOptions = {}): TurnReceipt[] {
  const opts = withSpecs(options);
  const receipts: TurnReceipt[] = [];
  for (const [n, index, body, entries, window] of turnInputs(messages)) {
    receipts.push(reduce(check(body, new Ledger(entries), opts), window, n, index, body));
  }
  return receipts;
}

/** `checkTurns` for async code: awaits the extractor once per turn, in order. */
export async function acheckTurns(messages: readonly unknown[], options: MessageOptions = {}): Promise<TurnReceipt[]> {
  const opts = withSpecs(options);
  const receipts: TurnReceipt[] = [];
  for (const [n, index, body, entries, window] of turnInputs(messages)) {
    receipts.push(reduce(await acheck(body, new Ledger(entries), opts), window, n, index, body));
  }
  return receipts;
}
