/** toolreceipt: catch LLM agents claiming tool actions they never performed. */

export { Claim, ToolSpec, cloneClaim, normalizeToolSpecs } from "./claims";
export type { ClaimEvidence, ClaimExtractor, ClaimInit, Confidence, NamedTool, Reason, Source, ToolLike } from "./claims";
export { Ledger, LedgerEntry, NO_RESULT, coerceArgs, truncate } from "./ledger";
export type { LedgerEntryInit, Status } from "./ledger";
export { defaultLexicon, loadLexicon } from "./lexicon";
export type { Family, Lexicon } from "./lexicon";
export { Receipt, renderSummary } from "./receipt";
export type { ReceiptInit, Verdict } from "./receipt";
export type { Json } from "./types";
