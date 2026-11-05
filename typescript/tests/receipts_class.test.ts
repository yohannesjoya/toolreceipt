/** `Receipts` and `wrapTools`, without an AI SDK. Python's test_receipts_facade.py where it applies. */
import { describe, expect, it } from "vitest";

import { Claim, type ClaimExtractor, Ledger, LedgerEntry, Receipts, ToolSpec, check } from "../src/index";

const OPTIONS = { toolCallId: "c1" };
/** The SDK's call: every tool gets `(input, options)`, whatever its own signature says. */
const run = (tool: { execute: unknown }, ...args: unknown[]): Promise<unknown> => (tool.execute as (...a: unknown[]) => Promise<unknown>)(...args);
const ok = (value: unknown = "done") => ({ description: "Send an email", execute: () => Promise.resolve(value) });

describe("Receipts with a given ledger", () => {
  it("checks like check()", () => {
    const ledger = Ledger.fromDict([{ tool: "issue_refund", status: "ok", args: { amount: 32 } }]);
    const options = { knownTools: [["issue_refund", "Issue a refund"], "send_email"] as Array<[string, string] | string> };
    const reply = "I've refunded $64 and emailed you.";
    expect(new Receipts(options).check(reply, ledger).toDict()).toEqual(check(reply, ledger, options).toDict());
  });

  it("throws with no source and no wrapTools call", () => {
    expect(() => new Receipts().check("hi")).toThrow(TypeError);
    expect(() => new Receipts().check("hi")).toThrow(/wrapTools/);
  });

  it("has no ledger before wrapTools", () => {
    expect(new Receipts().ledger).toBeNull();
  });

  it("redacts a given ledger when captureResults is off, leaving the caller's alone", () => {
    const ledger = Ledger.fromDict([{ tool: "t", status: "ok", args: { a: 1 }, result: "secret" }]);
    const receipt = new Receipts({ captureResults: false }).check("", ledger);
    expect(receipt.ledger.entries[0]!.args).toEqual({});
    expect(receipt.ledger.entries[0]!.result).toBeNull();
    expect(ledger.entries[0]!.result).toBe("secret");
  });

  it("disables the argument check when captureResults is off", () => {
    const ledger = Ledger.fromDict([{ tool: "issue_refund", status: "ok", args: { amount: 32 } }]);
    const reply = "I've refunded $64.";
    expect(new Receipts().check(reply, ledger).verdict).toBe("fabricated");
    expect(new Receipts({ captureResults: false }).check(reply, ledger).verdict).toBe("clean");
    expect(new Receipts({ checkArgs: false }).check(reply, ledger).verdict).toBe("clean");
  });

  it("passes llmBlocks and the extractor through, sync and async", async () => {
    const llm: ClaimExtractor = { extract: (text) => [new Claim({ text, verb: "other", candidates: ["send_email"], source: "llm" })] };
    const options = { extractor: llm, knownTools: ["send_email"] };
    const soft = new Receipts(options).check("check your inbox", new Ledger());
    expect([soft.ok, soft.suspected.map((c) => c.tool)]).toEqual([true, ["send_email"]]);
    const hard = new Receipts({ ...options, llmBlocks: true });
    expect(hard.check("check your inbox", new Ledger()).ok).toBe(false);
    expect((await hard.acheck("check your inbox", new Ledger())).ok).toBe(false);
  });

  it("reads a one-shot ignoreTools on every check", () => {
    const receipts = new Receipts({ ignoreTools: new Set(["lookup"]).values() });
    const ledger = Ledger.fromDict([{ tool: "lookup", status: "ok" }]);
    expect(receipts.check("", ledger).verdict).toBe("clean");
    expect(receipts.check("", ledger).verdict).toBe("clean");
  });
});

describe("wrapTools", () => {
  it("does not touch the input and keeps every other property", () => {
    const original = ok();
    const extra = Symbol("extra");
    const tool = { ...original, inputSchema: { a: 1 }, [extra]: true };
    const tools = { send_email: tool };
    const wrapped = new Receipts().wrapTools(tools);
    expect(wrapped).not.toBe(tools);
    expect(tools.send_email).toBe(tool);
    expect(tool.execute).toBe(original.execute);
    expect(wrapped.send_email.execute).not.toBe(tool.execute);
    expect(wrapped.send_email.inputSchema).toBe(tool.inputSchema);
    expect(wrapped.send_email[extra]).toBe(true);
  });

  it("passes input, options and this to the tool, and its value back", async () => {
    const seen: unknown[] = [];
    const value = { a: 1 };
    const tool = {
      description: "",
      execute(this: unknown, ...args: unknown[]) {
        seen.push(this, ...args);
        return Promise.resolve(value);
      },
    };
    const input = { x: 1 };
    await expect(run(new Receipts().wrapTools({ t: tool }).t, input, OPTIONS)).resolves.toBe(value);
    expect(seen).toEqual([tool, input, OPTIONS]);
  });

  it("does not make a sync tool async", () => {
    const wrapped = new Receipts().wrapTools({ t: { execute: () => 5 } });
    expect(run(wrapped.t, {}, OPTIONS)).toBe(5);
  });

  it("keeps the prototype, own getters and a frozen tool's properties", async () => {
    class Tool {
      own = 1;
      get description(): string { return "From the class"; }
      needsApproval(): boolean { return true; }
      execute(this: Tool): Promise<number> { return Promise.resolve(this.own); }
    }
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ t: new Tool(), frozen: Object.freeze({ schema: 1, execute: () => 2 }) });
    expect(wrapped.t).toBeInstanceOf(Tool);
    expect(wrapped.t.needsApproval()).toBe(true);
    expect(wrapped.t.description).toBe("From the class");
    expect(wrapped.frozen.schema).toBe(1);
    await expect(run(wrapped.t)).resolves.toBe(1);
    expect(run(wrapped.frozen)).toBe(2);
    expect(receipts.ledger!.tools()).toEqual(new Set(["t", "frozen"]));
  });

  it("does not let a throwing Proxy output reach the run", async () => {
    const value = new Proxy({}, { has() { throw new Error("no"); }, get() { throw new Error("no"); } });
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ t: { execute: () => value } });
    expect(run(wrapped.t, {}, OPTIONS)).toBe(value);
    await Promise.resolve();
    expect(receipts.ledger!.entries[0]!.status).toBe("ok");
  });

  it("stores a thrown plain object as JSON", async () => {
    const receipts = new Receipts();
    // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
    const wrapped = receipts.wrapTools({ t: { execute: () => Promise.reject({ code: 503 }) } });
    await run(wrapped.t, {}, OPTIONS).catch(() => null);
    expect(receipts.ledger!.entries[0]!.error).toBe('{"code": 503}');
  });

  it("does not serialise a thrown non-Error with captureResults off", async () => {
    const receipts = new Receipts({ captureResults: false });
    const thrown: unknown = { token: "secret" };
    // eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
    const wrapped = receipts.wrapTools({ t: { execute: () => Promise.reject(thrown) } });
    await run(wrapped.t, {}, OPTIONS).catch(() => null);
    const [entry] = receipts.ledger!.entries;
    expect(entry!.status).toBe("error");
    expect(entry!.error).toBe("non-Error thrown");
  });

  it("records one entry per call, closed in the ledger with times in seconds", async () => {
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ send_email: ok("sent") });
    const pending = run(wrapped.send_email, { to: "a" }, OPTIONS);
    expect(receipts.ledger!.length).toBe(0); // joins when it closes, as Python
    await pending;
    const [entry] = receipts.ledger!.entries;
    expect([entry!.tool, entry!.status, entry!.result, entry!.callId]).toEqual(["send_email", "ok", "sent", "c1"]);
    expect(entry!.args).toEqual({ to: "a" });
    expect(entry!.startedAt! / Date.now()).toBeLessThan(1 / 100);
    expect(entry!.endedAt!).toBeGreaterThanOrEqual(entry!.startedAt!);
  });

  it("numbers a call with no toolCallId, so entries stay apart", async () => {
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ t: ok() });
    await run(wrapped.t, {});
    await run(wrapped.t, {});
    const ids = receipts.ledger!.entries.map((e) => e.callId);
    expect(new Set(ids).size).toBe(2);
    expect(ids.every((id) => typeof id === "string" && id !== "")).toBe(true);
  });

  it("records a rejection and rethrows the same error", async () => {
    const error = new Error("boom");
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ t: { execute: () => Promise.reject(error) } });
    await expect(run(wrapped.t, {}, OPTIONS)).rejects.toBe(error);
    const [entry] = receipts.ledger!.entries;
    expect([entry!.status, entry!.error]).toEqual(["error", "Error: boom"]);
  });

  it("records a sync throw and rethrows the same error", () => {
    const error = new Error("boom");
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ t: { execute: () => { throw error; } } });
    expect(() => run(wrapped.t, {}, OPTIONS)).toThrow(error);
    expect(receipts.ledger!.errorTools()).toEqual(new Set(["t"]));
  });

  it("cuts a long result and a long error to maxResultChars", async () => {
    const receipts = new Receipts({ maxResultChars: 5 });
    const wrapped = receipts.wrapTools({
      a: { execute: () => Promise.resolve("1234567") },
      b: { execute: () => Promise.reject(new Error("1234567")) },
    });
    await run(wrapped.a, {}, OPTIONS);
    await run(wrapped.b, {}, OPTIONS).catch(() => null);
    expect(receipts.ledger!.entries.map((e) => e.result ?? e.error)).toEqual(["1234…", "Erro…"]);
  });

  it("keeps args and result out with captureResults off, and keeps the error text", async () => {
    const receipts = new Receipts({ captureResults: false });
    const wrapped = receipts.wrapTools({
      a: { execute: () => Promise.resolve("secret") },
      b: { execute: () => Promise.reject(new Error("boom")) },
    });
    await run(wrapped.a, { card: "4242" }, OPTIONS);
    await run(wrapped.b, { card: "4242" }, OPTIONS).catch(() => null);
    const [a, b] = receipts.ledger!.entries;
    expect([a!.args, a!.result]).toEqual([{}, null]);
    expect([b!.args, b!.error]).toEqual([{}, "Error: boom"]);
  });

  it.each([
    ["a null-prototype value", () => Object.create(null) as unknown],
    ["a cycle", () => { const o: Record<string, unknown> = {}; o["self"] = o; return o; }],
    ["a throwing getter", () => ({ get x(): number { throw new Error("no"); } })],
  ])("survives %s in the output and the input", async (_name, make) => {
    const value = make();
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ t: { execute: () => Promise.resolve(value) } });
    await expect(run(wrapped.t, make(), OPTIONS)).resolves.toBe(value);
    expect(receipts.ledger!.entries[0]!.status).toBe("ok");
  });

  it("survives an error that cannot be printed, and rethrows it", async () => {
    const error: unknown = Object.create(null);
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ t: { execute: () => Promise.reject(error as Error) } });
    await expect(run(wrapped.t, {}, OPTIONS)).rejects.toBe(error);
    expect(receipts.ledger!.entries[0]!.status).toBe("error");
    expect(receipts.ledger!.entries[0]!.error).not.toBe("");
  });

  it("survives options that throw when read", async () => {
    const receipts = new Receipts();
    const wrapped = receipts.wrapTools({ t: ok() });
    const options = { get toolCallId(): string { throw new Error("no"); } };
    await expect(run(wrapped.t, {}, options)).resolves.toBe("done");
    expect(receipts.ledger!.length).toBe(1);
  });

  it("rejects what is not a record", () => {
    expect(() => new Receipts().wrapTools([] as never)).toThrow(TypeError);
    expect(() => new Receipts().wrapTools(null as never)).toThrow(TypeError);
    expect(() => new Receipts().wrapTools(new Map() as never)).toThrow(TypeError);
  });

  describe("a tool with no execute", () => {
    it("passes through and is not known, so an honest reply is not accused", () => {
      const tool = { description: "Send an email" };
      const receipts = new Receipts();
      const wrapped = receipts.wrapTools({ send_email: tool });
      expect(wrapped.send_email).toBe(tool);
      const receipt = receipts.check("I emailed you the confirmation.");
      expect(receipt.ok).toBe(true);
      expect(receipt.knownTools).toEqual([]);
    });

    it("stops a wrapped sibling taking the blame for it", () => {
      const receipts = new Receipts();
      receipts.wrapTools({ issue_refund: { description: "Issue a refund", execute: () => 1 }, refund_order: { description: "Refund an order" } });
      const receipt = receipts.check("I've refunded your order.");
      expect(receipt.verdict).toBe("clean");
      expect(receipt.unmatched.length).toBe(1);
    });

    it("stays judged when the caller names it", () => {
      const receipts = new Receipts({ knownTools: ["send_email"] });
      receipts.wrapTools({ send_email: {} });
      expect(receipts.check("I emailed you the confirmation.").verdict).toBe("fabricated");
    });
  });

  describe("known tools", () => {
    it("makes a tool the run never called judgeable, with its description", () => {
      const seen: ToolSpec[][] = [];
      const extractor: ClaimExtractor = { extract: (_text, known) => { seen.push([...known]); return []; } };
      const receipts = new Receipts({ extractor, knownTools: ["z_first", ["b", ""]] });
      receipts.wrapTools({ a: ok(), b: { description: "Do b", execute: () => 1 } });
      receipts.check("hi");
      expect(seen[0]!.map((s) => [s.name, s.description])).toEqual([["z_first", ""], ["b", "Do b"], ["a", "Send an email"]]);
    });

    it("gives fabricated for a wrapped tool that never ran", () => {
      const receipts = new Receipts();
      receipts.wrapTools({ send_email: ok() });
      expect(receipts.check("I emailed you the confirmation.").verdict).toBe("fabricated");
    });
  });

  describe("streaming tools", () => {
    async function drain(iterable: AsyncIterable<unknown>): Promise<unknown[]> {
      const out: unknown[] = [];
      for await (const v of iterable) out.push(v);
      return out;
    }

    function streaming(gen: () => AsyncGenerator<unknown>) {
      const receipts = new Receipts();
      const wrapped = receipts.wrapTools({ t: { execute: gen } });
      return { receipts, out: wrapped.t.execute() as unknown as AsyncIterable<unknown> };
    }

    it("re-yields every value and records the last", async () => {
      const { receipts, out } = streaming(async function* () { await Promise.resolve(); yield "a"; yield "b"; });
      expect(typeof (out as unknown as Promise<unknown>).then).toBe("undefined"); // the SDK tests the raw return
      expect(await drain(out)).toEqual(["a", "b"]);
      expect(receipts.ledger!.entries[0]!.result).toBe("b");
    });

    it("records a throw from the iterator and rethrows it", async () => {
      const error = new Error("boom");
      const { receipts, out } = streaming(async function* () { await Promise.resolve(); yield "a"; throw error; });
      await expect(drain(out)).rejects.toBe(error);
      expect(receipts.ledger!.entries[0]!.status).toBe("error");
    });

    it("leaves the entry open when stopped early, and closes the source", async () => {
      let closed = false;
      const { receipts, out } = streaming(async function* () {
        try { await Promise.resolve(); yield "a"; yield "b"; } finally { closed = true; }
      });
      for await (const _ of out) break;
      expect(closed).toBe(true);
      expect(receipts.ledger!.length).toBe(0);
      const receipt = receipts.check("");
      expect([receipt.ledger.entries[0]!.status, receipt.ledger.entries[0]!.error]).toEqual(["error", "no result"]);
    });
  });

  describe("runs", () => {
    it("closes an open entry at check, and a later finish is ignored", async () => {
      let release: (v: string) => void = () => undefined;
      const receipts = new Receipts();
      const wrapped = receipts.wrapTools({ send_email: { description: "Send an email", execute: () => new Promise<string>((r) => { release = r; }) } });
      const pending = run(wrapped.send_email, {}, OPTIONS);
      const receipt = receipts.check("I emailed you the confirmation.");
      expect(receipt.claimedButNeverRan.map((c) => c.reason)).toEqual(["tool_failed"]);
      release("sent");
      await expect(pending).resolves.toBe("sent");
      expect(receipts.ledger!.entries.map((e) => [e.status, e.error])).toEqual([["error", "no result"]]);
    });

    it("reads the one run by default, and a named run when asked", async () => {
      const receipts = new Receipts();
      const first = receipts.wrapTools({ send_email: ok() });
      await run(first.send_email, {}, OPTIONS);
      expect(receipts.check("").ledger.length).toBe(1);
      const second = receipts.wrapTools({ issue_refund: { description: "Issue a refund", execute: () => 1 } });
      expect(receipts.check("", first).ledger.length).toBe(1);
      expect(receipts.check("", second).ledger.length).toBe(0);
      expect((await receipts.acheck("", first)).ledger.length).toBe(1);
    });

    it("refuses to guess once there are two runs, so one request never reads another's data", async () => {
      const receipts = new Receipts();
      const a = receipts.wrapTools({ t: { execute: () => "for A" } });
      const b = receipts.wrapTools({ t: { execute: () => "for B" } });
      await run(a.t, {}, OPTIONS);
      await run(b.t, {}, OPTIONS);
      expect(() => receipts.check("")).toThrow(TypeError);
      expect(() => receipts.check("")).toThrow(/wrapTools/);
      await expect(receipts.acheck("")).rejects.toThrow(TypeError);
      expect(receipts.check("", new Ledger()).ledger.length).toBe(0);
      expect(receipts.check("", a).ledger.entries[0]!.result).toBe("for A");
    });

    it("rejects a record from another instance or never wrapped", () => {
      const receipts = new Receipts();
      receipts.wrapTools({});
      expect(() => receipts.check("", new Receipts().wrapTools({ t: ok() }))).toThrow(TypeError);
      expect(() => receipts.check("", { t: ok() })).toThrow(/wrapTools/);
    });

    it("uses only the constructor's tools for a given ledger", () => {
      const receipts = new Receipts();
      receipts.wrapTools({ send_email: ok() });
      expect(receipts.check("I emailed you.", new Ledger()).knownTools).toEqual([]);
    });
  });

  it("gives a ledger entry a LedgerEntry", async () => {
    const receipts = new Receipts();
    await run(receipts.wrapTools({ t: ok() }).t, {}, OPTIONS);
    expect(receipts.ledger!.entries[0]).toBeInstanceOf(LedgerEntry);
  });
});
