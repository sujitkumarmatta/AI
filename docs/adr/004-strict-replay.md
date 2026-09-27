# 4. A replay miss is a hard error

## Context

During replay the agent may issue a request with no recorded response — because it
diverged, because a prompt changed, or because normalisation did not cover some
volatile content.

## Decision

Raise. `CassetteMiss` carries the canonical request, the rules applied, and how many
responses were recorded for that key. Over HTTP it is a 409. Nothing falls through
to an upstream.

## Alternatives

**Fall through to a live call.** The convenient choice, and what a caching proxy
would do. Rejected: it converts a divergence into a run that is partly replayed and
partly live, produces numbers that look reproducible and are not, and spends money
without asking. A harness whose entire subject is failures that hide inside apparent
success cannot itself hide a failure inside apparent success.

**Serve the nearest recorded response.** Plausible-looking and silently wrong, which
is precisely the failure class under study.

**Reuse the last response for a repeated key.** Convenient for agents that retry
identically, but it masks drift. Repeats instead consume a per-key queue in recorded
order, and running out is a miss.

## Consequences

A drifted agent fails loudly and immediately, with enough diagnostics to see why.

The cost is brittleness: an agent that puts unmasked volatile content in its prompts
cannot be replayed until a normalisation rule covers it. That is the intended
trade — a loud failure the user can fix, rather than a quiet one they cannot see.

`Engine.cassette_remaining` exposes the other direction: recorded responses never
served, meaning the run stopped earlier than when it was recorded.
