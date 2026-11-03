/**
 * Table tests for pystr.ts. Every expected value was produced by the Python expression in
 * the row's comment (CPython 3.12.3), never by running the TS code.
 */
import { describe, expect, it } from "vitest";

import {
  PY_SPACE_CHARS, cpCompare, cpFirst, cpLast, cpLength, cpSlice, fullmatch, isAlnum, isAlpha,
  isDigit, matchAt0, pyDumps, pyInt, pyLstrip, pyRegex, pyRepr, pyRstrip, pySplit, pySplitlines,
  pyStrip, reEscape,
} from "../src/pystr";

describe("PY_SPACE", () => {
  it("is exactly Python's str.isspace() set", () => {
    // [hex(i) for i in range(0x110000) if chr(i).isspace()]
    const expected = [0x9, 0xa, 0xb, 0xc, 0xd, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
      0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
      0x2028, 0x2029, 0x202f, 0x205f, 0x3000];
    expect([...PY_SPACE_CHARS].map((c) => c.codePointAt(0))).toEqual(expected);
    expect(pyRegex("\\s").test("\ufeff")).toBe(false); // JS \s includes U+FEFF, Python does not
    expect(pyRegex("\\s").test("\x85")).toBe(true);    // Python includes NEL, JS does not
    expect(pyRegex("\\S").test("\x85")).toBe(false);
  });
});

describe("pyRegex", () => {
  it("rewrites \\b to a Unicode word boundary", () => {
    expect(pyRegex("\\bago\\b").test("long ago")).toBe(true);
    expect(pyRegex("\\bago\\b").test("ago\u00e9")).toBe(false); // re.search(r"\bago\b", "agoé") is None
  });
  it("rewrites \\d inside a character class and leaves [\\b] a backspace", () => {
    expect(pyRegex("[\\d-]+").exec("\u0663-\u0664")![0]).toBe("\u0663-\u0664");
    expect(pyRegex("[\\b]").test("\b")).toBe(true);
    expect(pyRegex("[\\b]").test("a")).toBe(false);
  });
  it("keeps flags and adds u", () => {
    const re = pyRegex("\\d+", "i");
    expect(re.flags).toContain("u");
    expect(re.flags).toContain("i");
  });
  it("rewrites \\s in a lookbehind split", () => {
    expect(pyRegex("(?<=[.!?])\\s+").exec("a.\x85b")![0]).toBe("\x85");
  });
  it("rewrites \\w", () => {
    expect(pyRegex("^\\w+$").test("caf\u00e9_1")).toBe(true);
    expect(pyRegex("^\\w+$").test("a-b")).toBe(false);
  });
});

describe("reEscape", () => {
  it("escapes only u-mode syntax characters", () => {
    expect(reEscape("e-mailed")).toBe("e-mailed");
    expect(reEscape("set up")).toBe("set up");
    expect(reEscape("n't")).toBe("n't");
    expect(reEscape("a.b*c")).toBe("a\\.b\\*c");
    expect(reEscape("$5 (ok)")).toBe("\\$5 \\(ok\\)");
    expect(() => new RegExp(reEscape("e-mailed"), "u")).not.toThrow();
    expect(() => new RegExp(reEscape("a.b*c[x]{2}|y/z^$\\"), "u")).not.toThrow();
  });
});

describe("fullmatch and matchAt0", () => {
  it("anchor like re.fullmatch and re.match", () => {
    expect(fullmatch(pyRegex("\\d+"), "12")).toBeTruthy();
    expect(fullmatch(pyRegex("\\d+"), "12a")).toBeNull();
    expect(matchAt0(pyRegex("if\\b"), "if so")).toBeTruthy();
    expect(matchAt0(pyRegex("if\\b"), "so if")).toBeNull();
  });
});

describe("pySplitlines", () => {
  it("splits on Python's line boundaries", () => {
    // "a\nb\rc\r\nd\ve\ff\x1cg\x1dh\x1ei\x85j\u2028k\u2029l\n".splitlines()
    expect(pySplitlines("a\nb\rc\r\nd\ve\ff\x1cg\x1dh\x1ei\x85j\u2028k\u2029l\n"))
      .toEqual(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]);
    expect(pySplitlines("")).toEqual([]);
    expect(pySplitlines("a\n\nb")).toEqual(["a", "", "b"]);
  });
});

describe("pySplit", () => {
  it("splits on runs of Python whitespace with no separator", () => {
    // "  a\u00a0b\x1cc\u2003d \x85 e\ufeff f  ".split()
    expect(pySplit("  a\u00a0b\x1cc\u2003d \x85 e\ufeff f  ")).toEqual(["a", "b", "c", "d", "e\ufeff", "f"]);
    expect(pySplit("   ")).toEqual([]);
  });
  it("keeps empties with a separator", () => {
    expect(pySplit("a,,b", ",")).toEqual(["a", "", "b"]);
  });
});

describe("pyStrip", () => {
  it("strips Python whitespace or a character set", () => {
    expect(pyStrip("\x1c\u00a0 x \u3000\x85")).toBe("x");          // "\x1c\u00a0 x \u3000\x85".strip()
    expect(pyStrip("..a.b..", ".")).toBe("a.b");                    // "..a.b..".strip(".")
    expect(pyLstrip("xxhixx", "x")).toBe("hixx");                   // "xxhixx".lstrip("x")
    expect(pyRstrip("a.!?", ".!?,;:")).toBe("a");                   // "a.!?".rstrip(".!?,;:")
    expect(pyStrip("")).toBe("");
  });
});

describe("character classes", () => {
  it("isDigit is \\p{Nd} only (documented: Python also accepts superscripts)", () => {
    expect(isDigit("\u0663")).toBe(true);
    expect(isDigit("12")).toBe(true);
    expect(isDigit("\u00b2")).toBe(false); // "²".isdigit() is True in Python: accepted divergence
    expect(isDigit("\u00bd")).toBe(false);
    expect(isDigit("")).toBe(false);
    expect(isDigit("1a")).toBe(false);
  });
  it("isAlnum and isAlpha", () => {
    expect(isAlnum("\u00e91")).toBe(true);
    expect(isAlnum("e-")).toBe(false);
    expect(isAlnum("")).toBe(false);
    expect(isAlpha("\u00df")).toBe(true);
    expect(isAlpha("1")).toBe(false);
  });
});

describe("code points", () => {
  it("length, slice, first, last", () => {
    expect(cpLength("\u{1F600}\u{1F600}")).toBe(2);
    expect(cpSlice("a\u{1F600}b", 1, 2)).toBe("\u{1F600}");
    expect(cpSlice("abc", 1)).toBe("bc");
    expect(cpFirst("\u{1F600}x")).toBe("\u{1F600}");
    expect(cpLast("x\u{1F600}")).toBe("\u{1F600}");
    expect(cpFirst("")).toBe("");
    expect(cpLast("")).toBe("");
  });
  it("cpCompare orders by code point", () => {
    // sorted(["\U0001F600", "\uFFFD", "a"]) == ["a", "\uFFFD", "\U0001F600"]
    expect(["\u{1F600}", "\uFFFD", "a"].sort(cpCompare)).toEqual(["a", "\uFFFD", "\u{1F600}"]);
    expect(["\u{1F600}", "\uFFFD"].sort()).toEqual(["\u{1F600}", "\uFFFD"]); // JS default differs
  });
});

describe("pyInt", () => {
  it("reads any Nd digits", () => {
    expect(pyInt("\u0663\u0664")).toBe(34); // int("٣٤")
    expect(pyInt("12")).toBe(12);
    expect(pyInt("\u{1D7D8}\u{1D7D9}")).toBe(1); // int("𝟘𝟙"), mathematical double-struck digits
  });
});

describe("pyRepr", () => {
  it("matches Python repr(float)/str(int) under the int-iff-isInteger rule", () => {
    const rows: Array<[number, string]> = [
      [1e16, "10000000000000000"],            // integer in JS (item 8); Python float repr would be 1e+16
      [9999999999999998, "9999999999999998"],
      [999999999999999.9, "999999999999999.9"],
      [1234567.5, "1234567.5"],
      [1e-4, "0.0001"],
      [1e-5, "1e-05"],
      [1e-7, "1e-07"],
      [1.5e-7, "1.5e-07"],
      [123456789012345680, "123456789012345680"],
      [2.5, "2.5"],
      [32, "32"],
      [0.1 + 0.2, "0.30000000000000004"],
      // Every double >= 2**53 is integer-valued, so under the int rule (§7.3 item 8) it prints its digits;
      // the exponent branch is reachable only below 1e-4. 1e300 as a Python int would print the same digits.
      [1e300, BigInt(1e300).toString()],
      [-0.0001, "-0.0001"],
      [-2.5, "-2.5"],
      [0, "0"],
      [1.5e-10, "1.5e-10"],
      [12345.678e-9, "1.2345678e-05"],
    ];
    for (const [n, s] of rows) expect(pyRepr(n), String(n)).toBe(s);
  });
});

describe("pyDumps", () => {
  it("matches json.dumps defaults", () => {
    // json.dumps({"amount": 32, "note": "café ✓ 😀", "ok": True, "x": None, "f": 2.5, "l": [1, "a"]})
    expect(pyDumps({ amount: 32, note: "caf\u00e9 \u2713 \u{1F600}", ok: true, x: null, f: 2.5, l: [1, "a"] }))
      .toBe('{"amount": 32, "note": "caf\\u00e9 \\u2713 \\ud83d\\ude00", "ok": true, "x": null, "f": 2.5, "l": [1, "a"]}');
    expect(pyDumps('a"b\\c\n\t\x01')).toBe('"a\\"b\\\\c\\n\\t\\u0001"');
    expect(pyDumps(1e-5)).toBe("1e-05");
    expect(pyDumps([])).toBe("[]");
    expect(pyDumps({})).toBe("{}");
    expect(pyDumps("/")).toBe('"/"');
  });
});
