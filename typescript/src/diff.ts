/** Set-semantics diff between claims and the ledger (port of diff.py). STUB until Task 8. */

import type { Claim } from "./claims";
import type { Ledger } from "./ledger";
import type { Receipt } from "./receipt";

export interface DiffOptions {
  knownTools?: readonly string[] | undefined;
  ignoreTools?: Iterable<string> | undefined;
  llmBlocks?: boolean | undefined;
  checkArgs?: boolean | undefined;
}

export function diff(_claims: readonly Claim[], _ledger: Ledger, _options: DiffOptions = {}): Receipt {
  throw new Error("not ported yet");
}
