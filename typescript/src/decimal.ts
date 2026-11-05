/**
 * Exact decimal numbers for the argument check (PRD 06 §7.6), standing in for Python's
 * `decimal.Decimal`. Value = digits / 10^scale. Only what `_args.py` needs.
 */

import { pyAsciiDigits } from "./pystr";

const DECIMAL = /^(-?)(\d*)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/u;

export class Dec {
  private constructor(
    readonly digits: bigint,
    readonly scale: number,
  ) {}

  /** `Decimal(text)` for `-?\d+(\.\d+)?([eE][+-]?\d+)?` and `-?\.\d+`; `null` for anything else. */
  static parse(text: string): Dec | null {
    const m = DECIMAL.exec(pyAsciiDigits(text)); // Decimal() reads any Nd digits
    if (!m) return null;
    const [, sign, intPart = "", frac = "", exp] = m;
    if (intPart === "" && frac === "") return null;
    let digits = BigInt((intPart || "0") + frac);
    let scale = frac.length;
    if (exp !== undefined) {
      const e = Number(exp);
      if (e < 0) scale -= e;
      else if (e >= scale) {
        digits *= 10n ** BigInt(e - scale);
        scale = 0;
      } else scale -= e;
    }
    return new Dec(sign === "-" ? -digits : digits, scale);
  }

  /** `Decimal(str(value))`; `null` when the number is not finite. */
  static fromNumber(n: number): Dec | null {
    if (!Number.isFinite(n)) return null;
    return Dec.parse(String(n));
  }

  abs(): Dec {
    return this.digits < 0n ? new Dec(-this.digits, this.scale) : this;
  }

  times100(): Dec {
    return new Dec(this.digits * 100n, this.scale);
  }

  private static aligned(a: Dec, b: Dec): [bigint, bigint, number] {
    const scale = Math.max(a.scale, b.scale);
    return [a.digits * 10n ** BigInt(scale - a.scale), b.digits * 10n ** BigInt(scale - b.scale), scale];
  }

  add(other: Dec): Dec {
    const [a, b, scale] = Dec.aligned(this, other);
    return new Dec(a + b, scale);
  }

  equals(other: Dec): boolean {
    const [a, b] = Dec.aligned(this, other);
    return a === b;
  }
}
