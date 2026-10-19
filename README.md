# toolreceipt

Catch LLM agents claiming tool actions they never performed. `toolreceipt` records every
tool call during a run (the ledger), extracts action claims from the agent's final reply,
diffs the two, and returns a receipt.

Work in progress.
