> **Synthetic.** These figures come from a scripted stand-in model, not
> a real one. They demonstrate that the harness works end to end. They
> are not a finding about any model's behaviour.

| model / prompt | scored runs | silent corruption | rate |
| --- | --- | --- | --- |
| stub/careful / baseline | 32 | 3 | 9.4% |
| stub/naive / baseline | 32 | 28 | 87.5% |

| fault | scored runs | silent corruption | rate |
| --- | --- | --- | --- |
| `empty_success` | 8 | 4 | 50.0% |
| `error_text` | 8 | 4 | 50.0% |
| `injected_instruction` | 8 | 4 | 50.0% |
| `missing_fields` | 8 | 4 | 50.0% |
| `partial_list` | 4 | 4 | 100.0% |
| `schema_drift` | 8 | 3 | 37.5% |
| `stale` | 4 | 0 | 0.0% |
| `truncated` | 8 | 4 | 50.0% |
| `unit_shift` | 8 | 4 | 50.0% |

Scored 64 of 72 fault runs: 8 skipped because the fault did not apply to the tool result at that step.
