# Architecture

Everything described here exists in code. Where something is planned but absent, it
says so.

## The shape of the problem

To test how an agent behaves when a tool misbehaves, three things have to be true
at once:

1. The same run has to be repeatable, or a fix cannot be shown to have helped.
2. The corruption has to be injectable without modifying the agent, or the result
   describes a patched agent rather than a real one.
3. The result has to be reproducible by someone else, or it is an anecdote.

(1) and (3) are in tension with (2): an agent driven by a real model does not repeat,
and once a fault changes the conversation, nothing recorded from the clean run
applies any more. The cassette tree is the resolution.

## Components

| module | responsibility |
| --- | --- |
| `misfeed.canon` | reduce a chat request to the fields that determine the completion; normalise volatile content; hash |
| `misfeed.store` | content-addressed blobs; trunk/branch trace tree; strict replay |
| `misfeed.faults` | the nine fault classes and how one is applied to a request |
| `misfeed.proxy` | the OpenAI-compatible endpoint; record, replay and inject modes |
| `misfeed.verdict` | eight outcomes from three deterministic facts |
| `misfeed.report` | aggregation, exclusion rules, markdown rendering |
| `evals.world` | the subject: a five-row SQLite world and its tools |
| `evals.tasks` | questions, SQL-computed reference answers, prompt variants |
| `evals.agent` | a plain tool-calling loop |
| `evals.stub_model` | a scripted stand-in model, so the harness runs offline |
| `evals.study` | the driver: clean run, then a branch per (fault, step) |
| `evals.demo` | narrated walkthrough from committed cassettes |

The split between `misfeed` and `evals` is load-bearing. `misfeed` is the harness
and knows nothing about orders or customers; `evals` is one agent's problem domain.
A different subject swaps `evals` and leaves the harness alone.

## The interception point

The proxy implements `POST /v1/chat/completions` and sits between the agent and the
model. A fault is applied by rewriting the **tool-result message in the outgoing
request**: the agent's tool really returned its normal payload, and only the model
sees the corrupted version.

```
agent ──┬─ calls its own tools locally (real results) ───────────┐
        │                                                        │
        └─ POST /v1/chat/completions ──> misfeed ──> model       │
                                            │                    │
                              rewrites the tool-result <──────────┘
                              message in this request
```

Because the interception point is the HTTP API rather than any SDK, there is nothing
framework-specific in the mechanism: any client speaking OpenAI chat-completions, in
any language, needs only its base URL redirected.

This is covered by tests rather than assumed. `tests/test_compat_openai_sdk.py` drives
an unmodified `openai.AsyncOpenAI` over a real socket against a uvicorn-served proxy,
records and replays, injects a fault, and checks cassette portability in both
directions. It is also the only test that exercises the real server path — uvicorn,
sockets, and the lifespan that flushes a recording on shutdown; every other test drives
the ASGI app in-process and skips all three.

The interception point is also the boundary of what it can see: the agent's own
retry wrapper and validation code run against the *real* tool result and are never
exercised. An SDK-level injector would cover that half. It does not exist.

## The cassette tree

A **trunk** is a clean recorded run. A **branch** is what happened after a fault was
injected at some step, and it stores only its divergent continuation, inheriting the
trunk's entries before its fork.

```
trunk:   [step0] [step1] [step2]
                    │
branch (fork at 1): ├─ inherits step0
                    └─ own: [step1'] [step2']
```

Resolving a branch walks the ancestry, truncating each parent at the child's fork.
Branches of branches work; a cycle raises rather than hanging.

Two consequences:

- N fault experiments against an M-step run cost far less than N full runs. In the
  committed study, branch traces store fewer entries than they resolve to, and the
  reproducibility gate confirms each resolves to a complete run.
- Every experiment replays from committed files, so a published number can be
  reproduced with no endpoint.

Blobs are addressed by the SHA-256 of their canonical JSON, so identical content is
stored once. Trace ids are caller-supplied and readable (`trunk--stub-naive--baseline--T1_shipped_total`)
rather than random, because these files are committed and reviewed by people.

## Request identity

Replay requires deciding whether two requests are the same. Two problems, both
handled explicitly rather than silently:

**Volatile prompt content.** Agents inject timestamps, UUIDs and epoch millis, so a
byte-exact hash would miss on every run. Three named rules mask exactly those
formats. Anything broader risks masking a real prompt change and making replay
quietly wrong. `canon.explain()` reports what was hashed and under which rules, and
a miss prints it.

**Wire format is not meaning.** `stream` and `stream_options` are excluded from the
key, so a streaming and a non-streaming call for identical content are one logical
request. The cassette stores the logical response. Synthesising SSE on replay is
therefore a later addition needing no cassette migration — but until it lands,
`stream=true` is refused with a 400 rather than answered with a non-streaming body.

**Streaming framing in the next request.** A client that accumulates a streamed reply
keeps the delta's `index` on each tool call and sends it back in the assistant
message; a non-streaming client has no such field. Position in the `tool_calls` array
already carries that information and the API does not read `index` there, so it is
stripped -- narrowly, inside a message's `tool_calls` only. Without it a streaming and
a non-streaming client could not share a cassette.

**Explicit nulls.** Null-valued fields are dropped at every depth. A loop that appends
a response message verbatim sends `"content": null` on an assistant tool-call message;
the official SDK's `model_dump(exclude_none=True)` omits the field. Both mean "no
content", and treating them as different requests made every cassette client-specific
with nothing saying so. A compatibility test against a second, independently written
client found it; no unit test would have.

Also excluded: `user`, `metadata`, `store` — caller bookkeeping with no effect on the
completion. `bool` is checked before the numeric branch, because `bool` subclasses
`int` in Python and `parallel_tool_calls=True` must not collide with `=1`.

### Scheme versioning

Every trace is stamped with `fingerprint()` — a short hash of `KEY_FIELDS`, the
normalisation rules and the structural rules. Loading a trace whose stamp differs
raises `CanonMismatch` naming both values.

This exists because it was needed. Normalising explicit nulls changed every stored
key, and the symptom was "the cassettes are incomplete" plus a silent re-record of the
whole study. Against a paid endpoint that is real money spent for no reason. The
fingerprint is derived rather than hand-maintained, because a version integer someone
has to remember to bump does not get bumped. `misfeed traces` marks stale traces
rather than refusing to list them, since listing is how you discover one is stale.

## Modes

| mode | cassette | upstream | on a miss |
| --- | --- | --- | --- |
| `REPLAY` | read only | never constructed | hard 409 with diagnostics |
| `RECORD` | read, then append | on miss only | call upstream, append |
| `INJECT` | read, then append to branch | on miss only | call upstream, append to branch |

`RECORD` and `INJECT` sharing "try the cassette, else go live and append" is what
makes recording resumable: a rate limit stops the run without wasting what was
already captured, and a completed experiment is free to re-run. The upstream client
is built by a factory on first live call, so a run served entirely from cassettes
opens no connection at all.

### Divergence

Once a fault actually changes a request, the trunk describes a different
conversation. From that point everything is recorded rather than matched.

This is gated on divergence, not on step number. An earlier version truncated the
trunk at the fork, which was wrong twice over: a fault that could not apply then
burned a live call re-recording an identical request, and a post-fork request that
coincidentally matched a trunk key was served but not recorded, leaving the branch
replaying short. Both have regression tests.

A fault that cannot apply — a JSON fault against a plain-text payload, or any fault
at a step with no tool result — is not an experiment. Nothing diverges, nothing is
recorded, and the study reports it as skipped.

### Fault persistence

A corrupted tool result is re-applied to every later request, keyed by
`tool_call_id`. Without this, the agent's next request carries the original payload
it actually received and the corruption evaporates after one step — which is not how
a broken tool behaves. This bug was caught by a study run reporting half its
experiments as "the corrupted result never reached the answer".

## Failure modes, and what happens

| failure | behaviour |
| --- | --- |
| replay has no recorded response | `CassetteMiss` → 409 with the canonical request and which rules were applied |
| live-request cap reached | `BudgetExceeded` → 429, nothing sent upstream |
| upstream non-2xx | 502 carrying the upstream body, truncated |
| malformed request body | 400 |
| trace id does not exist | raises at app construction, not at first request |
| trace ancestry contains a cycle | raises |
| agent fails the task with no fault | report withholds the whole cell and names it |
| fault never reached the answer | excluded from the denominator, counted separately |

The first row is the important one. A replay miss could have been made to fall
through to a live call, which would be convenient and would turn a divergence into
an invisible, billed, non-reproducible run — the exact class of failure this project
exists to expose.

## What is not here

- No SDK-level injection, so the agent's own error handling is untested.
- No MCP transport interception; faults reach tools only through the model-facing
  boundary.
- No database or server. Cassettes are files; the study is a script. Nothing is
  deployed, so there is no operations document.
- No concurrency. The proxy serves one run at a time by construction: the engine
  holds per-run step counters and divergence state. Parallelising a study means
  running separate engines, not sharing one.
