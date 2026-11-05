/**
 * Install the packed tarball the way a user would, and run it. Catches what the test suite cannot:
 * a missing file, a broken `exports` map, declarations that do not resolve, a stray dependency.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dir = mkdtempSync(join(tmpdir(), "toolreceipt-smoke-"));
const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] });

const SUMMARY = '2 claimed actions never ran: issue_refund ("I\'ve refunded $64", ledger amount=32).';
const BODY = `
const ledger = Ledger.fromDict([{ tool: "issue_refund", status: "ok", args: { amount: 32 } }]);
const receipt = check("I've refunded $64 and emailed you the confirmation.", ledger, {
  knownTools: [["issue_refund", "Issue a refund to the customer"], "send_email"],
});
console.log(receipt.summary);
const wrapped = new Receipts().wrapTools({ send_email: { description: "Send an email", execute: () => "sent" } });
console.log(typeof wrapped.send_email.execute);
`;
const EXPECTED = `${SUMMARY}\nfunction\n`;

try {
  run("npm", ["pack", "--pack-destination", dir], root);
  const [filename] = readdirSync(dir).filter((name) => name.endsWith(".tgz"));
  writeFileSync(join(dir, "package.json"), JSON.stringify({ name: "consumer", private: true }));
  run("npm", ["install", "--no-audit", "--no-fund", "--offline", join(dir, filename)], dir);

  const installed = JSON.parse(readFileSync(join(dir, "node_modules/toolreceipt/package.json"), "utf8"));
  if (Object.keys(installed.dependencies ?? {}).length > 0) throw new Error("the package has runtime dependencies");

  writeFileSync(join(dir, "esm.mjs"), `import { Ledger, Receipts, check } from "toolreceipt";${BODY}`);
  writeFileSync(join(dir, "cjs.cjs"), `const { Ledger, Receipts, check } = require("toolreceipt");${BODY}`);
  for (const file of ["esm.mjs", "cjs.cjs"]) {
    const out = run(process.execPath, [file], dir);
    if (out !== EXPECTED) throw new Error(`${file} printed:\n${out}`);
  }

  // Both module systems must find their declarations, under the strictest resolution a user can pick.
  writeFileSync(join(dir, "types.mts"), `import { Ledger, Receipts, check, type Receipt } from "toolreceipt";
const receipt: Receipt = check("hi", new Ledger());
const verdict: string = receipt.verdict;
const receipts: Receipts = new Receipts();
export { verdict, receipts };
`);
  writeFileSync(join(dir, "types.cts"), `import toolreceipt = require("toolreceipt");
const receipt: toolreceipt.Receipt = toolreceipt.check("hi", new toolreceipt.Ledger());
export = receipt.verdict;
`);
  writeFileSync(join(dir, "tsconfig.json"), JSON.stringify({
    compilerOptions: { noEmit: true, strict: true, skipLibCheck: false, types: [], target: "es2022", module: "nodenext", moduleResolution: "nodenext" },
    files: ["types.mts", "types.cts"],
  }));
  run(process.execPath, [join(root, "node_modules/typescript/bin/tsc"), "-p", "."], dir);
  console.log("smoke ok:", filename);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
