> **Synthetic.** These figures come from a scripted stand-in model, not
> a real one. They demonstrate that the harness works end to end. They
> are not a finding about any model's behaviour.

| model / prompt | scored runs | silent corruption | rate |
| --- | --- | --- | --- |
| stub/careful / baseline | 10 | 2 | 20.0% |
| stub/naive / baseline | 10 | 10 | 100.0% |

| fault | scored runs | silent corruption | rate |
| --- | --- | --- | --- |
| `empty_success` | 8 | 4 | 50.0% |
| `missing_fields` | 8 | 4 | 50.0% |
| `partial_list` | 4 | 4 | 100.0% |

Scored 20 of 24 fault runs: 4 skipped because the fault did not apply to the tool result at that step.
