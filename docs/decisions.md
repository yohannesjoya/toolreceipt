# Decisions

## Callbacks over tool wrapping

Ledger capture uses a LangChain callback handler rather than wrapping tools. It needs no
change to the user's tools or graph, works for tools the user does not own, and the same
handler works for plain LangChain agents.
