/** The `Receipt` result model and its one-line summary (port of receipt.py). */

import type { Claim } from "./claims";
import { type Ledger, type LedgerEntry, truncate } from "./ledger";
import { pyDumps } from "./pystr";
import type { Json } from "./types";

export type Verdict = "clean" | "unreported" | "fabricated";

export interface ReceiptInit {
  confirmed: Claim[];
  claimedButNeverRan: Claim[];
  unmatched: Claim[];
  ranButUnreported: LedgerEntry[];
  verdict: Verdict;
  summary: string;
  ledger: Ledger;
  knownTools: string[];
  suspected?: Claim[] | undefined;
}

export class Receipt {
  confirmed: Claim[];
  claimedButNeverRan: Claim[];
  unmatched: Claim[];
  ranButUnreported: LedgerEntry[];
  verdict: Verdict;
  summary: string;
  ledger: Ledger;
  knownTools: string[];
  /** Claims found only by an LLM extractor whose tool never ran. Reported, never blocking by
   * default; empty on the rules-only path. */
  suspected: Claim[];

  constructor(init: ReceiptInit) {
    this.confirmed = init.confirmed;
    this.claimedButNeverRan = init.claimedButNeverRan;
    this.unmatched = init.unmatched;
    this.ranButUnreported = init.ranButUnreported;
    this.verdict = init.verdict;
    this.summary = init.summary;
    this.ledger = init.ledger;
    this.knownTools = init.knownTools;
    this.suspected = init.suspected ?? [];
  }

  get ok(): boolean {
    return this.verdict !== "fabricated";
  }

  toDict(): Record<string, Json> {
    return {
      confirmed: this.confirmed.map((c) => c.toDict()),
      claimed_but_never_ran: this.claimedButNeverRan.map((c) => c.toDict()),
      unmatched: this.unmatched.map((c) => c.toDict()),
      suspected: this.suspected.map((c) => c.toDict()),
      ran_but_unreported: this.ranButUnreported.map((e) => e.toDict()),
      verdict: this.verdict,
      ok: this.ok,
      summary: this.summary,
      ledger: [...this.ledger].map((e) => e.toDict()),
      known_tools: [...this.knownTools],
    };
  }
}

function plural(n: number, word: string): string {
  return n === 1 ? `${n} ${word}` : `${n} ${word}s`;
}

/** `key=json` pairs of the contradicting ledger arguments, cut to 60 characters. */
function formatObserved(observed: Record<string, Json>): string {
  const text = Object.entries(observed)
    .map(([k, v]) => `${k}=${pyDumps(v)}`)
    .join(", ");
  return truncate(text, 60) ?? "";
}

/** `tool ("text")`, plus `, ledger k=v` when the claim carries `observed`. */
function quoted(claim: Claim): string {
  const appendix = claim.observed && Object.keys(claim.observed).length > 0 ? `, ledger ${formatObserved(claim.observed)}` : "";
  return `${claim.tool} ("${claim.text}"${appendix})`;
}

/**
 * One-line, specific summary. Without `suspected` the lines are the 0.1 lines. With it, the
 * suffix replaces the final period and the `clean` line counts the confirmed claims.
 */
export function renderSummary(
  verdict: Verdict,
  claimedButNeverRan: readonly Claim[],
  ranButUnreported: readonly LedgerEntry[],
  suspected: readonly Claim[] = [],
  confirmedCount = 0,
): string {
  if (verdict === "fabricated") {
    const first = claimedButNeverRan[0]!;
    const n = claimedButNeverRan.length;
    let line = `${n} claimed action${n === 1 ? "" : "s"} never ran: ${quoted(first)}`;
    if (suspected.length > 0) line += `; ${suspected.length} more suspected`;
    return line + ".";
  }
  const parts: string[] = [];
  if (suspected.length > 0 && confirmedCount > 0) parts.push(plural(confirmedCount, "claim") + " confirmed");
  else if (suspected.length > 0) parts.push("No claims");
  else parts.push("All claims confirmed");
  if (verdict === "unreported") {
    const names = [...new Set(ranButUnreported.map((e) => e.tool))];
    parts.push(`${plural(names.length, "tool")} ran unreported: ${names.join(", ")}`);
  }
  if (suspected.length > 0) parts.push(`${suspected.length} suspected (LLM only): ${quoted(suspected[0]!)}`);
  return parts.join("; ") + ".";
}
