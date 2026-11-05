/**
 * Compile-time check: tool lists typed by interfaces, as the SDKs type them, are accepted uncast.
 * `npm run check` typechecks this file. The shapes copy openai 7.51 and @anthropic-ai/sdk 1.5,
 * which are not dependencies of this package.
 */
import { describe, expect, it } from "vitest";

import { checkMessages, checkTurns } from "../src/index";

interface FunctionDefinition { name: string; description?: string; parameters?: Record<string, unknown> }
interface ChatCompletionFunctionTool { function: FunctionDefinition; type: "function" }
interface ChatCompletionCustomTool { custom: { name: string; description?: string }; type: "custom" }
type ChatCompletionTool = ChatCompletionFunctionTool | ChatCompletionCustomTool;

interface ResponsesFunctionTool { name: string; parameters: Record<string, unknown> | null; strict: boolean | null; type: "function"; description?: string | null }
interface ResponsesWebSearchTool { type: "web_search"; search_context_size?: "low" | "medium" | "high" }
type ResponsesTool = ResponsesFunctionTool | ResponsesWebSearchTool;

interface AnthropicTool { input_schema: { type: "object" }; name: string; description?: string; type?: "custom" | null }
interface AnthropicBashTool { name: "bash"; type: "bash_20250124" }
type AnthropicToolUnion = AnthropicTool | AnthropicBashTool;

describe("SDK-typed tool lists", () => {
  it("compile uncast and normalise", () => {
    const chat: ChatCompletionTool[] = [
      { type: "function", function: { name: "issue_refund", description: "Issue a refund." } },
      { type: "custom", custom: { name: "send_email" } },
    ];
    const responses: ResponsesTool[] = [
      { type: "function", name: "issue_refund", parameters: null, strict: null },
      { type: "web_search" },
    ];
    const anthropic: AnthropicToolUnion[] = [
      { name: "issue_refund", input_schema: { type: "object" } },
      { name: "bash", type: "bash_20250124" },
    ];
    const msgs = [{ role: "user", content: "x" }, { role: "assistant", content: "Hi." }];
    expect(checkMessages(msgs, { knownTools: chat }).knownTools).toEqual(["issue_refund", "send_email"]);
    expect(checkTurns(msgs, { knownTools: responses })[0]!.knownTools).toEqual(["issue_refund"]);
    expect(checkMessages(msgs, { knownTools: anthropic }).knownTools).toEqual(["issue_refund", "bash"]);
  });
});
