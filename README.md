# misfeed

Feed an agent a broken tool result and find out whether it notices.

**Status: early development.** The harness is being built in vertical slices;
this README describes only what is implemented and tested. Anything not listed
under "What works today" does not exist yet.

## The problem

Agents call tools, and tools misbehave. Reported production rates put individual
tool-call failure at 3–15%, and Datadog has reported roughly 1 in 20 model
requests failing while the system keeps returning plausible output.

The damaging case is not the exception. It is the success-shaped failure: HTTP
200 with an empty payload, a silently truncated list, a renamed field, a stale
value. The model reads it as fact, reasons around it, and returns a confident
wrong answer. Nothing throws. Nothing alerts. You find out when a user
complains.

You currently cannot test for this, for a structural reason: agent runs are not
reproducible. The same input takes a different path each time, so you cannot
capture a failure, change one thing, and demonstrate that the change helped.

## The approach

Record an agent's run once through an OpenAI-compatible proxy. Replay it
deterministically offline. Then corrupt a chosen tool result and record only the
divergent continuation as a branch. After that, every experiment replays from
committed cassettes — no API key, no network, same numbers on anyone's machine.

## What works today

Nothing yet. This section gets filled in as slices land, with the test that
proves each one.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
