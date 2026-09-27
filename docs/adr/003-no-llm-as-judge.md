# 3. Deterministic task checks, not LLM-as-judge

## Context

Every run needs a verdict: was the answer right, and did the agent flag trouble.
The conventional choice in 2026 is an LLM judge, and most eval platforms ship one.

## Decision

Choose tasks whose correct answer is computed by a reference function, and have the
agent end with a machine-readable `ANSWER:` line. Correctness is then an exact
predicate. Use no model anywhere in grading.

## Alternatives

**LLM-as-judge.** General, handles free-form answers, and is what the surrounding
ecosystem does. Rejected for two reasons. It is circular: the thing being measured
is whether a model notices that its input is degraded, and the judge is a model with
the same weakness. And its error rate would be unknown, so every published number
would carry an unquantified error term — unacceptable when the headline metric is a
rate.

**Human labels.** Accurate and unscalable, and would make re-running the study
expensive, which defeats the reproducibility goal.

**Fuzzy text matching against an expected value.** Deterministic, but extraction
noise gets mixed into the correctness signal: a number appearing anywhere in the
message is not the same as the agent asserting it.

## Consequences

`answered` and `correct` are exact, so the primary metric has no grader error in it,
and the reproducibility gate can demand byte-identical results.

The cost is a restricted task space: only questions with a computable answer. The
fixture is therefore synthetic, which trades realism for honesty. The fault classes,
not the tasks, carry the external validity.

Whether the agent *surfaced* a problem cannot be made exact this way, so it is a
published keyword lexicon, deliberately kept out of the correctness path and
reported with the phrases that matched. Its error against hand labels is not yet
measured, which is stated wherever it is used. An LLM judge could be added later as
a *secondary* signal with its agreement against the deterministic check published —
that ordering is the point.
