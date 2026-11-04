/** toolreceipt: catch LLM agents claiming tool actions they never performed. */

export { Claim, RuleExtractor, ToolSpec, normalizeToolSpecs } from "./claims";
export type { ClaimEvidence, ClaimExtractor, ClaimInit, Confidence, NamedTool, Reason, Source, ToolLike } from "./claims";
export { diff } from "./diff";
export type { DiffOptions } from "./diff";
export { Ledger, LedgerEntry, NO_RESULT, coerceArgs, truncate } from "./ledger";
export type { LedgerEntryInit, Status } from "./ledger";
export { loadLexicon } from "./lexicon";
export type { Family, Lexicon } from "./lexicon";
export { Receipt, renderSummary } from "./receipt";
export type { ReceiptInit, Verdict } from "./receipt";
export { acheck, check } from "./receipts";
export type { CheckOptions } from "./receipts";
export type { Json } from "./types";
