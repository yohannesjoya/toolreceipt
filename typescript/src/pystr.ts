/**
 * Python string, regex and JSON semantics (PRD 06 §7.3).
 *
 * Every place where the port must behave like CPython rather than like idiomatic JS goes
 * through this module, so the deliberate divergences from the JS built-ins live in one file.
 * The accepted divergences from Python are listed in PRD 06 §7.3 and docs/decisions.md.
 */

// ---------------------------------------------------------------------------
// Whitespace: Python's str.isspace() set, which JS \s is not (Python has U+001C..U+001F and
// U+0085 and lacks U+FEFF).
// ---------------------------------------------------------------------------

export const PY_SPACE_CHARS =
  "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000";
const PY_SPACE_SET = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
export const PY_SPACE_CLASS = `[${PY_SPACE_SET}]`;
const PY_NOT_SPACE_CLASS = `[^${PY_SPACE_SET}]`;

const WORD_SET = "\\p{L}\\p{N}_";
const WORD_CLASS = `[${WORD_SET}]`;
const WORD_BOUNDARY = `(?:(?<=${WORD_CLASS})(?!${WORD_CLASS})|(?<!${WORD_CLASS})(?=${WORD_CLASS}))`;

// ---------------------------------------------------------------------------
// Regular expressions
// ---------------------------------------------------------------------------

/**
 * Compile a Python `re` pattern as a JS RegExp with Python semantics: the `u` flag is added
 * and `\b` (outside classes), `\w`, `\d`, `\s`, `\S` are rewritten to their Unicode-aware
 * Python meanings. Escaped characters are otherwise left alone.
 */
export function pyRegex(source: string, flags = ""): RegExp {
  let out = "";
  let inClass = false;
  for (let i = 0; i < source.length; i++) {
    const ch = source[i]!;
    if (ch === "\\") {
      const next = source[i + 1] ?? "";
      i++;
      if (next === "b" && !inClass) out += WORD_BOUNDARY;
      else if (next === "w") out += inClass ? WORD_SET : WORD_CLASS;
      else if (next === "d") out += "\\p{Nd}";
      else if (next === "s") out += inClass ? PY_SPACE_SET : PY_SPACE_CLASS;
      else if (next === "S" && !inClass) out += PY_NOT_SPACE_CLASS;
      else out += "\\" + next;
      continue;
    }
    if (ch === "[" && !inClass) inClass = true;
    else if (ch === "]" && inClass) inClass = false;
    out += ch;
  }
  const all = flags.includes("u") ? flags : flags + "u";
  return new RegExp(out, all);
}

/** Escape a literal for a `u`-flag RegExp: only the syntax characters, never `-`, space or `#`. */
export function reEscape(text: string): string {
  return text.replace(/[\^$\\.*+?()[\]{}|/]/g, "\\$&");
}

const FULL_CACHE = new WeakMap<RegExp, RegExp>();
const AT0_CACHE = new WeakMap<RegExp, RegExp>();

function anchored(cache: WeakMap<RegExp, RegExp>, re: RegExp, suffix: string): RegExp {
  let out = cache.get(re);
  if (!out) {
    out = new RegExp(`^(?:${re.source})${suffix}`, re.flags.replace("g", ""));
    cache.set(re, out);
  }
  return out;
}

/** `re.fullmatch(re, s)`. The anchored variant is compiled once per pattern. */
export function fullmatch(re: RegExp, s: string): RegExpExecArray | null {
  return anchored(FULL_CACHE, re, "$").exec(s);
}

/** `re.match(re, s)`: anchored at position 0 only. */
export function matchAt0(re: RegExp, s: string): RegExpExecArray | null {
  return anchored(AT0_CACHE, re, "").exec(s);
}

// ---------------------------------------------------------------------------
// str methods
// ---------------------------------------------------------------------------

export function pyLower(s: string): string {
  return s.toLowerCase();
}

const SPACE_RUN = new RegExp(`${PY_SPACE_CLASS}+`, "gu");
const SPLITLINES = new RegExp("\\r\\n|[\\n\\r\\v\\f\\x1c\\x1d\\x1e\\x85\\u2028\\u2029]", "u");

/** `str.split()` (no separator: Python whitespace runs, empties dropped) or `str.split(sep)`. */
export function pySplit(s: string, sep?: string): string[] {
  if (sep === undefined) return s.split(SPACE_RUN).filter((p) => p.length > 0);
  return s.split(sep);
}

/** `str.splitlines()` without keepends. */
export function pySplitlines(s: string): string[] {
  if (s === "") return [];
  const parts = s.split(SPLITLINES);
  if (parts.length > 0 && parts[parts.length - 1] === "") parts.pop();
  return parts;
}

function stripSet(chars: string | undefined): Set<string> {
  return new Set([...(chars ?? PY_SPACE_CHARS)]);
}

export function pyLstrip(s: string, chars?: string): string {
  const set = stripSet(chars);
  const cps = [...s];
  let i = 0;
  while (i < cps.length && set.has(cps[i]!)) i++;
  return cps.slice(i).join("");
}

export function pyRstrip(s: string, chars?: string): string {
  const set = stripSet(chars);
  const cps = [...s];
  let j = cps.length;
  while (j > 0 && set.has(cps[j - 1]!)) j--;
  return cps.slice(0, j).join("");
}

/** `str.strip(chars)`: a character set, not a substring; Python whitespace by default. */
export function pyStrip(s: string, chars?: string): string {
  return pyRstrip(pyLstrip(s, chars), chars);
}

const ALNUM = /^[\p{L}\p{N}]+$/u;
const ALPHA = /^\p{L}+$/u;
const DIGIT = /^\p{Nd}+$/u;

/** A mapping in the Python sense: an object literal or a `Map`, not a class instance or array. */
export function isMapping(value: unknown): value is Record<string, unknown> | Map<unknown, unknown> {
  if (value instanceof Map) return true;
  if (value === null || typeof value !== "object") return false;
  const proto = Object.getPrototypeOf(value) as unknown;
  return proto === Object.prototype || proto === null;
}

/** Python truthiness: empty lists and mappings are false as well. */
export function pyTruthy(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (value instanceof Map) return value.size > 0;
  if (isMapping(value)) return Object.keys(value).length > 0;
  return Boolean(value);
}

export function isAlnum(s: string): boolean {
  return ALNUM.test(s);
}

export function isAlpha(s: string): boolean {
  return ALPHA.test(s);
}

/** `str.isdigit()` restricted to Nd (Python also accepts Numeric_Type=Digit; accepted divergence). */
export function isDigit(s: string): boolean {
  return DIGIT.test(s);
}

// ---------------------------------------------------------------------------
// Code points (Python indexes characters, JS indexes UTF-16 units)
// ---------------------------------------------------------------------------

export function cpLength(s: string): number {
  let n = 0;
  const it = s[Symbol.iterator]();
  while (!it.next().done) n++;
  return n;
}

export function cpSlice(s: string, start: number, end?: number): string {
  return [...s].slice(start, end).join("");
}

export function cpFirst(s: string): string {
  for (const c of s) return c;
  return "";
}

export function cpLast(s: string): string {
  let last = "";
  for (const c of s) last = c;
  return last;
}

/** Python string ordering: by code point, not by UTF-16 unit. */
export function cpCompare(a: string, b: string): number {
  const ia = a[Symbol.iterator]();
  const ib = b[Symbol.iterator]();
  for (;;) {
    const na = ia.next();
    const nb = ib.next();
    if (na.done && nb.done) return 0;
    if (na.done) return -1;
    if (nb.done) return 1;
    const ca = na.value.codePointAt(0)!;
    const cb = nb.value.codePointAt(0)!;
    if (ca !== cb) return ca < cb ? -1 : 1;
  }
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

const ND = /\p{Nd}/u;

/** The value 0 to 9 of one Nd digit: each Nd block is ten contiguous code points in order. */
function digitValue(ch: string): number {
  const cp = ch.codePointAt(0)!;
  let start = cp;
  while (start > 0 && ND.test(String.fromCodePoint(start - 1))) start--;
  return (cp - start) % 10;
}

/** Every Nd digit replaced by its ASCII digit; other characters unchanged. */
export function pyAsciiDigits(s: string): string {
  let out = "";
  for (const ch of s) out += ND.test(ch) ? String(digitValue(ch)) : ch;
  return out;
}

/** `int(digits)` for any run of Nd digits. */
export function pyInt(digits: string): number {
  let value = 0;
  for (const ch of digits) value = value * 10 + digitValue(ch);
  return value;
}

/**
 * Python `repr(float)` / `str(int)`. A JS number is a Python int iff `Number.isInteger`
 * (PRD 06 §7.3 item 8); a float uses the shortest round-trip digits in Python's notation:
 * fixed when the decimal exponent is in [-4, 16), else `d.ddde±XX`.
 */
export function pyRepr(n: number): string {
  if (!Number.isFinite(n)) return n > 0 ? "inf" : n < 0 ? "-inf" : "nan";
  if (Number.isInteger(n)) {
    return Math.abs(n) < 1e21 ? String(n) : BigInt(n).toString();
  }
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);
  const exp = abs.toExponential(); // shortest round-trip digits: "d.ddde±X"
  const [mant, e] = exp.split("e") as [string, string];
  const exponent = Number(e);
  const digits = mant.replace(".", "");
  if (exponent >= -4 && exponent < 16) {
    let fixed: string;
    if (exponent >= 0) {
      const intPart = digits.slice(0, exponent + 1).padEnd(exponent + 1, "0");
      const frac = digits.slice(exponent + 1);
      fixed = frac ? `${intPart}.${frac}` : `${intPart}.0`;
    } else {
      fixed = `0.${"0".repeat(-exponent - 1)}${digits}`;
    }
    return sign + fixed;
  }
  const expSign = exponent < 0 ? "-" : "+";
  const expDigits = String(Math.abs(exponent)).padStart(2, "0");
  return `${sign}${mant}e${expSign}${expDigits}`;
}

// ---------------------------------------------------------------------------
// json.dumps with defaults: separators ", " and ": ", ensure_ascii=True
// ---------------------------------------------------------------------------

function dumpString(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]!;
    const code = s.charCodeAt(i);
    if (ch === '"') out += '\\"';
    else if (ch === "\\") out += "\\\\";
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (ch === "\b") out += "\\b";
    else if (ch === "\f") out += "\\f";
    else if (code < 0x20 || code > 0x7e) out += "\\u" + code.toString(16).padStart(4, "0");
    else out += ch;
  }
  return out + '"';
}

/** Python `json.dumps(value)` with default arguments. Non-JSON values go through `String()`. */
export function pyDumps(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return pyRepr(value);
  if (typeof value === "string") return dumpString(value);
  if (Array.isArray(value)) return `[${value.map((v) => pyDumps(v)).join(", ")}]`;
  if (typeof value === "object") {
    const parts = Object.entries(value as Record<string, unknown>).map(
      ([k, v]) => `${dumpString(k)}: ${pyDumps(v)}`,
    );
    return `{${parts.join(", ")}}`;
  }
  if (typeof value === "bigint" || typeof value === "symbol") return dumpString(value.toString());
  return dumpString("[function]");
}
