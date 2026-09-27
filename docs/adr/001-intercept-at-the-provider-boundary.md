# 1. Intercept at the provider HTTP boundary

## Context

Faults have to reach the agent without modifying it. The candidates were: wrap the
agent's tool functions, patch the provider SDK, or sit between the agent and the
model as an OpenAI-compatible endpoint.

## Decision

Sit between the agent and the model, and apply faults by rewriting the tool-result
message in the outgoing request.

## Alternatives

**Wrap the tool functions.** Most faithful — it corrupts what the agent actually
receives, so the agent's own validation and retry code is exercised. But it requires
touching the agent's code, is specific to one language, and is different work for
every framework.

**Patch the provider SDK.** No agent changes, but monkey-patching is version-fragile
and locks the harness to the SDKs it knows.

## Consequences

Works against any framework in any language with no code changes. The agent under
test needs one environment variable pointed at the proxy.

The cost is real and is stated in the README rather than buried: this exercises the
model's reasoning about a degraded tool result, not the agent's error handling. The
agent's retry wrapper sees the genuine payload.

That half is now covered by `misfeed.toolfault`, as a second mechanism rather than a
change to this one, which is the right shape: corrupting what a function returns to its
caller cannot be done from outside the call without patching, so it is opt-in -- a
single call inserted at the agent's tool dispatch. A harness that monkey-patched a
user's tools in order to measure their reliability would be a poor advertisement for
itself.

Keeping them separate turned out to buy something neither could alone. A fault applied
in flight lives in the conversation history and is permanent by construction; a
tool-side fault can be transient, so the two can distinguish "the retry helps" from
"the failure does not clear". `tests/test_toolfault_integration.py` runs the same agent
against the same fault three ways and gets three different answers.

A second consequence: because the corruption lives in the request rather than in the
conversation, it has to be re-applied on every later request. Forgetting that was a
real bug.
