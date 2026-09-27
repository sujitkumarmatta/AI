# 2. Record a branch instead of continuing live after injection

## Context

Once a fault changes a request, the model's reply is not the recorded one, and
nothing after that point exists in the cassette. Something has to produce the rest
of the run.

## Decision

Record the divergent continuation once, as a branch that inherits the trunk's
entries before its fork. Afterwards the whole experiment replays from files.

## Alternatives

**Continue live every time.** Simple, and always current. But every reproduction of
a published number then costs money and needs credentials, which means in practice
nobody reproduces it. It also makes the numbers unstable run to run, so a
regression gate is impossible.

**Synthesise the continuation.** Cheap and fully offline, but the results would
describe the synthesiser rather than a model.

**Re-record the full run per experiment.** Correct but wasteful: N faults against an
M-step run costs N×M model calls instead of N×(M−fork).

## Consequences

`make eval` reproduces every published number on a clean clone with no key, no
endpoint and no network, and CI enforces it. This is the property the project is
built around.

Storage grows with the number of experiments, though branches share their trunk
prefix and blobs are deduplicated by content. The committed nine-fault study is
under a megabyte.

Branch explosion is bounded by the study, not by the harness: the driver picks which
(fault, step) pairs to record. There is no automatic search over injection points.

A subtlety this forced: during a fresh recording the whole trunk is available, but
once a fault applies the trunk must stop being matched, or the branch would be
served entries it never recorded and would replay short later.
