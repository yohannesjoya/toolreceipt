/** Ledger of tool calls actually performed during an agent run (port of ledger.py). */

import { cpLength, cpSlice } from "./pystr";
import type { Json } from "./types";

export type Status = "ok" | "error";

/** `error` text of a tool call that never returned a result. */
export const NO_RESULT = "no result";

export interface LedgerEntryInit {
  tool: string;
  status?: Status | undefined;
  args?: Record<string, Json> | undefined;
  result?: string | null | undefined;
  error?: string | null | undefined;
  callId?: string | null | undefined;
  parentRunId?: string | null | undefined;
  startedAt?: number | null | undefined;
  endedAt?: number | null | undefined;
}

export class LedgerEntry {
  tool: string;
  status: Status;
  args: Record<string, Json>;
  result: string | null;
  error: string | null;
  callId: string | null;
  parentRunId: string | null;
  startedAt: number | null;
  endedAt: number | null;

  constructor(init: LedgerEntryInit) {
    this.tool = init.tool;
    this.status = init.status ?? "ok";
    this.args = init.args ?? {};
    this.result = init.result ?? null;
    this.error = init.error ?? null;
    this.callId = init.callId ?? null;
    this.parentRunId = init.parentRunId ?? null;
    this.startedAt = init.startedAt ?? null;
    this.endedAt = init.endedAt ?? null;
  }

  toDict(): Record<string, Json> {
    return {
      tool: this.tool,
      status: this.status,
      args: this.args,
      result: this.result,
      error: this.error,
      call_id: this.callId,
      parent_run_id: this.parentRunId,
      started_at: this.startedAt,
      ended_at: this.endedAt,
    };
  }

  static fromDict(data: Record<string, unknown>): LedgerEntry {
    const args = data["args"];
    const tool = data["tool"];
    if (typeof tool !== "string") throw new TypeError("ledger entry needs a string 'tool'"); // Python: KeyError('tool')
    return new LedgerEntry({
      tool,
      status: (data["status"] as Status | undefined) ?? "ok",
      args: args && typeof args === "object" ? { ...(args as Record<string, Json>) } : {},
      result: (data["result"] as string | null | undefined) ?? null,
      error: (data["error"] as string | null | undefined) ?? null,
      callId: (data["call_id"] as string | null | undefined) ?? null,
      parentRunId: (data["parent_run_id"] as string | null | undefined) ?? null,
      startedAt: (data["started_at"] as number | null | undefined) ?? null,
      endedAt: (data["ended_at"] as number | null | undefined) ?? null,
    });
  }
}

export class Ledger implements Iterable<LedgerEntry> {
  private readonly _entries: LedgerEntry[];

  constructor(entries: Iterable<LedgerEntry> = []) {
    this._entries = [...entries];
  }

  append(entry: LedgerEntry): void {
    this._entries.push(entry);
  }

  /** Entries sorted by start time; untimed entries follow in insertion order. */
  get entries(): LedgerEntry[] {
    return this._entries
      .map((entry, index) => ({ entry, index }))
      .sort((a, b) => {
        const au = a.entry.startedAt === null ? 1 : 0;
        const bu = b.entry.startedAt === null ? 1 : 0;
        if (au !== bu) return au - bu;
        const at = a.entry.startedAt ?? 0;
        const bt = b.entry.startedAt ?? 0;
        if (at !== bt) return at < bt ? -1 : 1;
        return a.index - b.index;
      })
      .map((p) => p.entry);
  }

  okTools(): Set<string> {
    return new Set(this._entries.filter((e) => e.status === "ok").map((e) => e.tool));
  }

  errorTools(): Set<string> {
    return new Set(this._entries.filter((e) => e.status === "error").map((e) => e.tool));
  }

  tools(): Set<string> {
    return new Set(this._entries.map((e) => e.tool));
  }

  toDict(): Record<string, Json>[] {
    return this.entries.map((e) => e.toDict());
  }

  static fromDict(data: Iterable<Record<string, unknown>>): Ledger {
    return new Ledger([...data].map((d) => LedgerEntry.fromDict(d)));
  }

  get length(): number {
    return this._entries.length;
  }

  [Symbol.iterator](): Iterator<LedgerEntry> {
    return this.entries[Symbol.iterator]();
  }
}

/**
 * Return a JSON-safe object; non-object input is wrapped as `{ input: ... }`.
 *
 * Never throws: input JSON cannot hold (a cycle, a getter that throws) is kept as
 * `{ input: String(args) }`. A bigint is rendered through `String()` (Python's `default=str`).
 */
export function coerceArgs(args: unknown): Record<string, Json> {
  if (args === null || args === undefined) return {};
  let coerced: unknown;
  try {
    const text = JSON.stringify(args, (_key, value: unknown) => (typeof value === "bigint" ? String(value) : value));
    if (text === undefined) return {};
    coerced = JSON.parse(text);
  } catch {
    try {
      // Python: {"input": str(args)}; String() is the accepted counterpart (PRD 06 §7.3 item 11).
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      return { input: String(args) };
    } catch {
      return { input: "<unserialisable>" };
    }
  }
  if (coerced !== null && typeof coerced === "object" && !Array.isArray(coerced)) {
    return coerced as Record<string, Json>;
  }
  return { input: coerced as Json };
}

/** Cut `text` so the output (including the ellipsis) is at most `limit` code points long. */
export function truncate(text: string | null, limit: number): string | null {
  if (text === null || cpLength(text) <= limit) return text;
  if (limit <= 0) return "";
  return cpSlice(text, 0, limit - 1) + "…";
}
