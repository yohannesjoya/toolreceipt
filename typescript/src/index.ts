/** toolreceipt: catch LLM agents claiming tool actions they never performed. */

export { Claim, RuleExtractor, ToolSpec, normalizeToolSpecs } from "./claims";
export type { ClaimEvidence, ClaimExtractor, ClaimInit, Confidence, NamedTool, Reason, Source, ToolLike } from "./claims";
export { diff } from "./diff";
export type { DiffOptions } from "./diff";
export { Ledger, LedgerEntry, NO_RESULT, coerceArgs, truncate } from "./ledger";
export type { LedgerEntryInit, Status } from "./ledger";
export { loadLexicon } from "./lexicon";
export type { Family, Lexicon } from "./lexicon";
export { LLMExtractor, PROMPT_VERSION, buildPrompt, messageText, parseOutput } from "./llm";
export type { LLMExtractorOptions, OnError } from "./llm";
export { acheckMessages, checkMessages, finalReply, ledgerFromMessages } from "./messages";
export type { MessageOptions } from "./messages";
export { Receipt, renderSummary } from "./receipt";
export type { ReceiptInit, Verdict } from "./receipt";
export { acheck, check } from "./receipts";
export type { CheckOptions } from "./receipts";
export { TurnReceipt, acheckTurns, checkTurns } from "./turns";
export type { TurnReceiptInit } from "./turns";
export type { Json } from "./types";
