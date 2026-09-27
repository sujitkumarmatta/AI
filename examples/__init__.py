"""Worked examples: real agent clients pointed at misfeed by one line.

These exist so the compatibility claim is demonstrated rather than asserted. Both are
driven by tests, so neither can quietly rot:

- `openai_sdk_agent` uses the official `openai` SDK, streaming and non-streaming.
- `langgraph_agent` uses LangGraph via `langchain-openai` (needs the `compat` group).

Writing them found four real bugs that no unit test would have: cassettes were
client-specific, streamed framing leaked into the next request, changing the
canonicalisation scheme failed illegibly, and the world fixture was not thread-safe.
"""
