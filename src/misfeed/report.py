"""Aggregating runs into a report, and rendering it.

Three things here exist to stop a number being read as more than it is.

A clean run is a precondition, not a data point. If an agent cannot solve a task
with no fault injected, its behaviour under a fault says nothing, so the fault
results for that cell are withheld rather than averaged in. `precondition_failed`
names those cells.

Runs where the fault never applied, or never reached the context that produced the
answer, are excluded from the denominator and counted separately. Counting them as
passes would flatter the system; counting them as failures would slander it.

Every report carries `synthetic`. A report produced against a scripted stand-in
model is stamped true, because a rate measured from a stand-in is a statement
about the harness, not about any model.

The structure separates `results` from `provenance` for a reason worth stating.
`results` is the finding, and it must be byte-identical every time the study is
re-run from the same cassettes -- that is the reproducibility guarantee, and CI
checks it. `provenance` is how this particular run obtained them, and it
legitimately differs: a first run makes live calls, a replay makes none. Keeping
them in one block would either break the guarantee or force it to be stated as
"identical except for some fields", which erodes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from misfeed.verdict import Outcome, RunFacts, is_scored

__all__ = ["RunResult", "markdown_report", "summarise"]


@dataclass(frozen=True, slots=True)
class RunResult:
    """One agent run: either the clean baseline, or one fault experiment."""

    task: str
    model: str
    variant: str
    trace_id: str
    steps: int
    fault: str | None = None
    fault_step: int | None = None
    outcome: Outcome | None = None
    facts: RunFacts | None = None
    skipped: str | None = None
    answer: str | None = None
    expected: Any = None
    surfaced_phrases: list[str] = field(default_factory=list)
    total_tokens: int = 0
    live_calls: int = 0

    @property
    def is_clean(self) -> bool:
        return self.fault is None

    @property
    def cell(self) -> tuple[str, str, str]:
        return (self.model, self.variant, self.task)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "task": self.task,
            "model": self.model,
            "variant": self.variant,
            "trace_id": self.trace_id,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "answer": self.answer,
            "expected": self.expected,
        }
        if self.fault is not None:
            out["fault"] = self.fault
            out["fault_step"] = self.fault_step
        if self.outcome is not None:
            out["outcome"] = str(self.outcome)
        if self.skipped is not None:
            out["skipped"] = self.skipped
        if self.facts is not None:
            out["facts"] = self.facts.to_json()
        if self.surfaced_phrases:
            out["surfaced_phrases"] = self.surfaced_phrases
        return out


def summarise(
    results: list[RunResult], *, synthetic: bool, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Aggregate runs into the committed report structure."""
    clean = {r.cell: r for r in results if r.is_clean}
    faulted = [r for r in results if not r.is_clean]

    precondition_failed = sorted(
        f"{model}/{variant}/{task}"
        for (model, variant, task), run in clean.items()
        if not (run.facts is not None and run.facts.correct)
    )
    blocked = {
        cell for cell, run in clean.items() if not (run.facts is not None and run.facts.correct)
    }

    withheld = [r for r in faulted if r.cell in blocked]
    usable = [r for r in faulted if r.cell not in blocked]

    scored = [r for r in usable if r.facts is not None and is_scored(r.facts)]
    # Two different reasons a run is not scored, kept apart because they mean
    # different things. A fault that could not apply never became an experiment;
    # one that applied but never reached the answer was an experiment that tested
    # nothing. Reporting them as one number would hide which.
    skipped = [r for r in usable if r.facts is None]
    not_reached = [r for r in usable if r.facts is not None and not is_scored(r.facts)]

    by_variant: dict[str, dict[str, Any]] = {}
    for run in scored:
        key = f"{run.model} / {run.variant}"
        bucket = by_variant.setdefault(
            key, {"model": run.model, "variant": run.variant, "outcomes": Counter(), "scored": 0}
        )
        bucket["outcomes"][str(run.outcome)] += 1
        bucket["scored"] += 1
    for bucket in by_variant.values():
        count = bucket["scored"]
        silent = bucket["outcomes"].get(str(Outcome.SILENT_CORRUPTION), 0)
        bucket["outcomes"] = dict(sorted(bucket["outcomes"].items()))
        bucket["silent_corruption"] = silent
        bucket["silent_corruption_rate"] = round(silent / count, 4) if count else None

    by_fault: dict[str, dict[str, Any]] = {}
    for run in scored:
        assert run.fault is not None
        bucket = by_fault.setdefault(run.fault, {"scored": 0, "silent_corruption": 0})
        bucket["scored"] += 1
        if run.outcome is Outcome.SILENT_CORRUPTION:
            bucket["silent_corruption"] += 1
    for bucket in by_fault.values():
        bucket["silent_corruption_rate"] = (
            round(bucket["silent_corruption"] / bucket["scored"], 4) if bucket["scored"] else None
        )

    return {
        "synthetic": synthetic,
        "metadata": metadata or {},
        "results": {
            "totals": {
                "runs": len(results),
                "clean_runs": len(clean),
                "fault_runs": len(faulted),
                "scored": len(scored),
                "skipped_inapplicable": len(skipped),
                "excluded_not_reached": len(not_reached),
                "withheld_precondition": len(withheld),
                "total_tokens": sum(r.total_tokens for r in results),
            },
            "precondition_failed": precondition_failed,
            "by_variant": dict(sorted(by_variant.items())),
            "by_fault": dict(sorted(by_fault.items())),
            "runs": [r.to_json() for r in results],
        },
        "provenance": {
            "live_calls": sum(r.live_calls for r in results),
            "live_calls_by_trace": {r.trace_id: r.live_calls for r in results if r.live_calls},
        },
    }


def _percent(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def markdown_report(summary: dict[str, Any]) -> str:
    """A compact table for the README, generated -- never hand-edited."""
    results = summary["results"]
    lines: list[str] = []
    if summary["synthetic"]:
        lines += [
            "> **Synthetic.** These figures come from a scripted stand-in model, not",
            "> a real one. They demonstrate that the harness works end to end. They",
            "> are not a finding about any model's behaviour.",
            "",
        ]

    totals = results["totals"]
    lines += [
        "| model / prompt | scored runs | silent corruption | rate |",
        "| --- | --- | --- | --- |",
    ]
    for key, bucket in results["by_variant"].items():
        lines.append(
            f"| {key} | {bucket['scored']} | {bucket['silent_corruption']} | "
            f"{_percent(bucket['silent_corruption_rate'])} |"
        )
    if not results["by_variant"]:
        lines.append("| _no scored runs_ | 0 | 0 | n/a |")

    if results["by_fault"]:
        lines += [
            "",
            "| fault | scored runs | silent corruption | rate |",
            "| --- | --- | --- | --- |",
        ]
        for fault, bucket in results["by_fault"].items():
            lines.append(
                f"| `{fault}` | {bucket['scored']} | {bucket['silent_corruption']} | "
                f"{_percent(bucket['silent_corruption_rate'])} |"
            )

    unscored: list[str] = []
    if totals["skipped_inapplicable"]:
        unscored.append(
            f"{totals['skipped_inapplicable']} skipped because the fault did not apply "
            f"to the tool result at that step"
        )
    if totals["excluded_not_reached"]:
        unscored.append(
            f"{totals['excluded_not_reached']} excluded because the corrupted result "
            f"never reached the answer"
        )
    if totals["withheld_precondition"]:
        unscored.append(
            f"{totals['withheld_precondition']} withheld because the agent failed the "
            f"task with no fault injected"
        )
    tail = f"Scored {totals['scored']} of {totals['fault_runs']} fault runs"
    tail += f": {'; '.join(unscored)}." if unscored else "."
    lines += ["", tail]
    if results["precondition_failed"]:
        lines += [
            "",
            "Cells withheld on a failed clean run: "
            + ", ".join(f"`{cell}`" for cell in results["precondition_failed"])
            + ".",
        ]
    return "\n".join(lines) + "\n"
