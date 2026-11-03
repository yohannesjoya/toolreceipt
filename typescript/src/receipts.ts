/** `check()` and `acheck()` (port of receipts.py). STUB until Task 9. */

import type { ClaimExtractor, ToolLike } from "./claims";
import type { Ledger } from "./ledger";
import type { Receipt } from "./receipt";

export interface CheckOptions {
  knownTools?: Iterable<ToolLike> | undefined;
  ignoreTools?: Iterable<string> | undefined;
  extractor?: ClaimExtractor | null | undefined;
  llmBlocks?: boolean | undefined;
  checkArgs?: boolean | undefined;
}

export function check(_reply: string, _ledger: Ledger, _options: CheckOptions = {}): Receipt {
  throw new Error("not ported yet");
}

export function acheck(_reply: string, _ledger: Ledger, _options: CheckOptions = {}): Promise<Receipt> {
  return Promise.reject(new Error("not ported yet"));
}
