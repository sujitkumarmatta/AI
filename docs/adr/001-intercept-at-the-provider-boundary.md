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
agent's retry wrapper sees the genuine payload. Measuring that half needs an
SDK-level injector, which would be a second mechanism rather than a change to this
one.

A second consequence: because the corruption lives in the request rather than in the
conversation, it has to be re-applied on every later request. Forgetting that was a
real bug.
