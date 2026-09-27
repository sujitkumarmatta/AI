# Evaluation

## What is being measured

For an agent, a fault class and an injection point: what does the agent do when a
tool result is corrupted in that way at that point?

The primary metric is the **silent corruption rate** — the share of scored runs in
which the agent asserted a wrong answer and flagged nothing.

```
silent corruption rate = SILENT_CORRUPTION / scored runs
```

An agent that gets the wrong answer and says it hit a problem is doing its job badly.
One that returns a confident wrong figure is doing something worse, because nothing
downstream can tell.

## Ground truth

Tasks are answerable by SQL over the same five-row fixture the tools read
(`evals/world.py`). The reference answer is a query, not a constant, so changing the
fixture cannot leave a stale expected value behind.

| task | question | expected |
| --- | --- | --- |
| `T1_shipped_total` | total cents of shipped orders for 'Alcott Foods' | 6725 |
| `T2_shipped_count` | how many shipped orders 'Alcott Foods' has | 3 |

Both require two hops — name to id, id to orders — so a fault can be injected at
either. Both need aggregation over a multi-row result, which is what makes a silently
shortened list bite.

The agent is instructed to end with `ANSWER: <value>`. That makes *answered* and
*correct* exact predicates rather than text-matching guesses, and is why no model or
person grades anything. See [ADR 003](adr/003-no-llm-as-judge.md).

## Outcome classification

Three deterministic facts — answered, correct, surfaced — plus crash and loop flags,
map onto eight outcomes (`misfeed/verdict.py`). The mapping is exhaustively tested,
including that every outcome is reachable, because the primary metric is one cell of
it and a mislabelled cell would silently change every published number.

`correct` is exact. `surfaced` is a published keyword lexicon and is the weakest link
in the chain — see **Threats to validity**.

## What is excluded, and why

Three exclusion rules, each reported separately rather than merged into one number.

**Clean run as precondition.** Every cell runs the task with no fault first. If the
agent cannot answer it unfaulted, its behaviour under a fault is uninterpretable, so
the whole cell is withheld and named in `precondition_failed`. Averaging such runs in
would attribute the agent's baseline incompetence to the fault.

**Fault never applied.** A JSON-aware fault against a plain-text payload, or any
fault at a step with no tool result, produces no experiment. Nothing diverges and
nothing is recorded. Counted as `skipped_inapplicable`.

**Fault never reached the answer.** If the corrupted result was not in the context of
the request that produced the final answer — because the agent pruned it — the
experiment tested nothing. Counted as `excluded_not_reached`. Scoring it as a pass
would flatter the agent; as a failure, slander it.

In the committed study: 64 of 72 fault runs scored, 8 skipped as inapplicable, 0
excluded as unreached, 0 withheld on a failed precondition.

## Reproducibility

`make eval` re-runs the study from committed cassettes and fails unless **both**:

1. the `results` block matches the committed report byte for byte, and
2. the replay made zero live calls.

The second condition carries as much weight as the first. A run that quietly reached
an upstream to fill a gap would reproduce the numbers while proving nothing about
whether anyone else could.

The gate rebuilds the study from the report's own metadata rather than from CLI
defaults, so it reproduces the study that was actually run. It is verified to have
teeth by a test that tampers with one recorded response and asserts the gate fails.

`results` and `provenance` are separate blocks because a first run makes live calls
and a replay makes none. Keeping them together would force the guarantee to be stated
as "identical except for some fields", which erodes.

CI runs this gate with no credentials configured, deliberately: the claim is that the
numbers reproduce with no endpoint reachable, and a job with credentials could not
prove it.

## Results

From [`../evals/report.json`](../evals/report.json). **The upstream was a scripted
stand-in, not a model.** These figures describe the harness. See **Threats to
validity**.

| agent | scored | silent corruption | rate |
| --- | --- | --- | --- |
| `stub/naive` | 32 | 28 | 87.5% |
| `stub/careful` | 32 | 3 | 9.4% |

Grouped by what each fault does to the payload:

| effect on the data | faults | careful policy |
| --- | --- | --- |
| absent or malformed | `empty_success`, `error_text`, `missing_fields`, `truncated`, `injected_instruction` | abstains every time, 0 silent corruptions |
| plausible but wrong | `partial_list`, `unit_shift` | still silently corrupts |
| does not affect the answer | `stale` | correctly unaffected |

The finding worth stating: **validation defeats absence, not plausibility.** Checking
that a tool result is present and correctly shaped catches an empty payload, an error
string, a missing field, a truncated structure. It cannot catch a list silently
shortened from three rows to one, or an amount arriving scaled by 100, because those
are well-formed and look right. On exactly those faults the careful and naive policies
are indistinguishable.

`stale` scoring 0% is a result, not a gap. It shifts every date backwards and changes
nothing, because neither task's answer depends on a date. A harness that only injected
faults it already knew would bite could not tell you which faults are harmless.

## Threats to validity

Listed in rough order of how much they should reduce confidence.

1. **The subject is a scripted stand-in, not a model.** The gap between `naive` and
   `careful` is true by construction. Nothing here supports any claim about how a
   real model behaves. Every such report is stamped `synthetic: true`.
2. **The `surfaced` lexicon has unmeasured error.** It separates
   `silent_corruption` from `loud_failure` and `abstained` from `abandoned`, so its
   error propagates directly into the primary metric. Measuring it against hand
   labels on a sample of real-model runs is the single most valuable next step.
3. **Two tasks, one domain, five rows.** The task set is small enough that a single
   quirk of the fixture could drive a rate. The fault classes, not the tasks, are what
   carries external validity.
4. **The model's reasoning is measured, not the agent's error handling.** Faults are
   applied in flight, so an agent's retry and validation code sees the real payload.
   A production agent might catch faults this harness reports as uncaught.
5. **Injection points are chosen, not searched.** The study injects at steps 1–4. The
   smallest corruption that breaks a task is not searched for, so rates are
   conditional on the chosen points.
6. **Fault parameters are defaults.** `partial_list` keeps one row, `unit_shift`
   scales by 100. Different parameters would give different rates; no sensitivity
   analysis has been done.
7. **Per-fault cells are small.** Four to eight scored runs each. A rate like 37.5%
   is three runs out of eight, and no confidence interval is reported because with
   these counts it would be wider than the differences being discussed.

## Not yet done

- Measure the `surfaced` lexicon's error against hand labels.
- Record against a real endpoint and report the `baseline` vs `distrust` comparison.
- Sensitivity analysis over fault parameters.
- Search over injection points rather than fixed steps.
- Enough runs per cell to justify an interval.
