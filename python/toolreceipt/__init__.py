"""toolreceipt: catch LLM agents claiming tool actions they never performed."""

from toolreceipt.claims import Claim, ClaimExtractor, RuleExtractor, ToolSpec
from toolreceipt.ledger import Ledger, LedgerEntry
from toolreceipt.llm import LLMExtractor
from toolreceipt.receipt import Receipt, Verdict
from toolreceipt.receipts import Receipts, acheck, check

__version__ = "0.2.0"

__all__ = [
    "Claim",
    "ClaimExtractor",
    "LLMExtractor",
    "Ledger",
    "LedgerEntry",
    "Receipt",
    "Receipts",
    "RuleExtractor",
    "ToolSpec",
    "Verdict",
    "__version__",
    "acheck",
    "check",
]
