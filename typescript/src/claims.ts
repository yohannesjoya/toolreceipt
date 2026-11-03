/** Claim data model, extractor interface, tool specs and the rule extractor (port of claims.py). */

import type { Json } from "./types";

export type Confidence = "high" | "medium";
export type Reason = "tool_failed" | "not_called" | "args_mismatch";
export type Source = "rules" | "llm";

export class ToolSpec {
  constructor(
    readonly name: string,
    readonly description: string = "",
  ) {}
}

/** Any tool object with a `name` and a `description`, such as a LangChain or AI SDK tool. */
export interface NamedTool {
  name: string;
  description?: string | null | undefined;
}

export type ToolLike =
  | string
  | ToolSpec
  | readonly [string]
  | readonly [string, string]
  | Record<string, unknown>
  | Map<string, unknown>
  | NamedTool;

export interface ClaimInit {
  text: string;
  verb: string;
  object?: string | null | undefined;
  count?: number | null | undefined;
  candidates?: readonly string[] | undefined;
  tool?: string | null | undefined;
  confidence?: Confidence | null | undefined;
  reason?: Reason | null | undefined;
  source?: Source | undefined;
  observed?: Record<string, Json> | null | undefined;
}

export interface ClaimEvidence {
  /** True when the candidates matched on an object word of their tool name, false when on the
   * verb family alone, null when unknown (a claim built by hand or by `fromDict`). */
  objectMatched: boolean | null;
  /** The reply clause the argument check reads literals from, when the extractor knows it. */
  clause: string | null;
  /** The reply sentence the clause came from (amounts are counted over it). */
  sentence: string | null;
}

export class Claim {
  text: string;
  verb: string;
  object: string | null;
  count: number | null;
  candidates: string[];
  tool: string | null;
  confidence: Confidence | null;
  reason: Reason | null;
  source: Source;
  /** The ledger arguments that contradict the claim when `reason === "args_mismatch"`. */
  observed: Record<string, Json> | null;

  // Extractor evidence: deliberately not enumerable, so toDict(), JSON.stringify and object
  // spread never emit it (the field set is fixed by the Python package).
  private declare _objectMatched: boolean | null;
  private declare _clause: string | null;
  private declare _sentence: string | null;

  constructor(init: ClaimInit) {
    this.text = init.text;
    this.verb = init.verb;
    this.object = init.object ?? null;
    this.count = init.count ?? null;
    this.candidates = [...(init.candidates ?? [])];
    this.tool = init.tool ?? null;
    this.confidence = init.confidence ?? null;
    this.reason = init.reason ?? null;
    this.source = init.source ?? "rules";
    this.observed = init.observed ?? null;
    for (const key of ["_objectMatched", "_clause", "_sentence"]) {
      Object.defineProperty(this, key, { value: null, writable: true, enumerable: false, configurable: true });
    }
  }

  toDict(): Record<string, Json> {
    return {
      text: this.text,
      verb: this.verb,
      object: this.object,
      count: this.count,
      candidates: [...this.candidates],
      tool: this.tool,
      confidence: this.confidence,
      reason: this.reason,
      source: this.source,
      observed: this.observed,
    };
  }

  static fromDict(data: Record<string, unknown>): Claim {
    return new Claim({
      text: data["text"] as string,
      verb: data["verb"] as string,
      object: (data["object"] as string | null | undefined) ?? null,
      count: (data["count"] as number | null | undefined) ?? null,
      candidates: (data["candidates"] as string[] | null | undefined) ?? [],
      tool: (data["tool"] as string | null | undefined) ?? null,
      confidence: (data["confidence"] as Confidence | null | undefined) ?? null,
      reason: (data["reason"] as Reason | null | undefined) ?? null,
      source: (data["source"] as Source | undefined) ?? "rules",
      observed: (data["observed"] as Record<string, Json> | null | undefined) ?? null,
    });
  }

  evidence(): ClaimEvidence {
    return { objectMatched: this._objectMatched, clause: this._clause, sentence: this._sentence };
  }

  setEvidence(e: Partial<ClaimEvidence>): void {
    if (e.objectMatched !== undefined) this._objectMatched = e.objectMatched;
    if (e.clause !== undefined) this._clause = e.clause;
    if (e.sentence !== undefined) this._sentence = e.sentence;
  }
}

/** `dataclasses.replace` that keeps the extractor's private match evidence (diff.py `_copy`). */
export function cloneClaim(claim: Claim, changes: Partial<ClaimInit> = {}): Claim {
  const out = new Claim({
    text: claim.text,
    verb: claim.verb,
    object: claim.object,
    count: claim.count,
    candidates: claim.candidates,
    tool: claim.tool,
    confidence: claim.confidence,
    reason: claim.reason,
    source: claim.source,
    observed: claim.observed,
    ...changes,
  });
  out.setEvidence(claim.evidence());
  return out;
}

export interface ClaimExtractor {
  extract(text: string, knownTools: readonly ToolSpec[]): Claim[];
  aextract?(text: string, knownTools: readonly ToolSpec[]): Promise<Claim[]>;
}

const TYPE_ERROR =
  "known_tools entries must be a name, a (name, description) pair, a ToolSpec, a dict or an object with a name attribute; got ";

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  const proto = Object.getPrototypeOf(value) as unknown;
  return proto === Object.prototype || proto === null;
}

function typeName(value: unknown): string {
  if (value === null) return "NoneType";
  if (typeof value === "object") {
    const ctor = (value as { constructor?: { name?: string } }).constructor;
    return ctor?.name ?? "object";
  }
  return typeof value;
}

/** Normalize names, pairs, ToolSpecs, (OpenAI-style) dicts and tool objects to `ToolSpec`. */
export function normalizeToolSpecs(tools: Iterable<ToolLike>): ToolSpec[] {
  const specs: ToolSpec[] = [];
  for (const tool of tools) {
    if (typeof tool === "string") {
      specs.push(new ToolSpec(tool, ""));
    } else if (tool instanceof ToolSpec) {
      specs.push(tool);
    } else if (Array.isArray(tool)) {
      const [name, ...rest] = tool as readonly [string, ...string[]];
      specs.push(new ToolSpec(name, rest[0] ?? ""));
    } else if (tool instanceof Map || isPlainObject(tool)) {
      const get = (obj: unknown, key: string): unknown =>
        obj instanceof Map ? obj.get(key) : isPlainObject(obj) ? obj[key] : undefined;
      const inner = get(tool, "function") ?? get(tool, "custom");
      const src = inner instanceof Map || isPlainObject(inner) ? inner : tool;
      const name = get(src, "name");
      if (typeof name !== "string" || name === "") continue; // a built-in tool spec such as {"type": "web_search"}
      const description = get(src, "description");
      // Python: str(src.get("description") or ""); String() is the accepted counterpart (PRD 06 §7.3 item 11).
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      specs.push(new ToolSpec(name, description ? String(description) : ""));
    } else {
      const toolName = (tool as { name?: unknown }).name;
      if (typeof toolName !== "string") throw new TypeError(TYPE_ERROR + typeName(tool));
      const description = (tool as { description?: unknown }).description;
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      specs.push(new ToolSpec(toolName, description ? String(description) : ""));
    }
  }
  return specs;
}

// ---------------------------------------------------------------------------
// STUBS (Task 5): replaced by the rule extractor port in Tasks 6 and 7.
// ---------------------------------------------------------------------------

function notPorted(): never {
  throw new Error("not ported yet");
}

export class RuleExtractor implements ClaimExtractor {
  constructor(_lexicon?: unknown) {}
  extract(_text: string, _knownTools: readonly ToolSpec[]): Claim[] {
    return notPorted();
  }
}
export function descriptionFamily(_description: string, _lexicon?: unknown): string | null {
  return notPorted();
}
export function descriptionObject(_description: string, _lexicon?: unknown): string[] {
  return notPorted();
}
export function stemVerb(_word: string): string {
  return notPorted();
}
export function toolVerbTokens(_spec: ToolSpec, _lexicon?: unknown): Set<string> {
  return notPorted();
}
