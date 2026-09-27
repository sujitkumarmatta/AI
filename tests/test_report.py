"""Tests for report aggregation.

The aggregation decides what a published number means, so the cases that matter
are the ones that could quietly change a rate: a failed clean run, a fault that
never applied, and a fault that applied but never reached the answer.
"""

from __future__ import annotations

from typing import Any

from misfeed.report import RunResult, markdown_report, summarise
from misfeed.verdict import Outcome, RunFacts


def clean(
    *, correct: bool = True, task: str = "T1", model: str = "m", variant: str = "base"
) -> RunResult:
    return RunResult(
        task=task,
        model=model,
        variant=variant,
        trace_id=f"trunk--{model}--{variant}--{task}",
        steps=2,
        facts=RunFacts(
            fault_applied=False,
            fault_in_final_context=False,
            answered=True,
            correct=correct,
            surfaced=False,
        ),
    )


def faulted(
    *,
    outcome: Outcome = Outcome.SILENT_CORRUPTION,
    reached: bool = True,
    fault: str = "empty_success",
    task: str = "T1",
    model: str = "m",
    variant: str = "base",
    tokens: int = 100,
) -> RunResult:
    correct = outcome in (Outcome.RECOVERED, Outcome.UNAFFECTED)
    surfaced = outcome in (Outcome.RECOVERED, Outcome.LOUD_FAILURE, Outcome.ABSTAINED)
    return RunResult(
        task=task,
        model=model,
        variant=variant,
        trace_id=f"branch--{model}--{variant}--{task}--{fault}",
        steps=2,
        fault=fault,
        fault_step=1,
        outcome=outcome,
        facts=RunFacts(
            fault_applied=True,
            fault_in_final_context=reached,
            answered=True,
            correct=correct,
            surfaced=surfaced,
        ),
        total_tokens=tokens,
    )


def skipped(*, reason: str = "content is not JSON", fault: str = "partial_list") -> RunResult:
    return RunResult(
        task="T1",
        model="m",
        variant="base",
        trace_id="branch--skip",
        steps=2,
        fault=fault,
        fault_step=1,
        skipped=reason,
    )


def summary_of(results: list[RunResult], **kw: Any) -> dict[str, Any]:
    kw.setdefault("synthetic", False)
    return summarise(results, **kw)


class TestRates:
    def test_rate_is_silent_corruption_over_scored(self) -> None:
        out = summary_of([clean(), faulted(), faulted(), faulted(outcome=Outcome.ABSTAINED)])
        bucket = out["results"]["by_variant"]["m / base"]
        assert bucket["scored"] == 3
        assert bucket["silent_corruption"] == 2
        assert bucket["silent_corruption_rate"] == round(2 / 3, 4)

    def test_rate_is_none_with_nothing_scored(self) -> None:
        out = summary_of([clean()])
        assert out["results"]["by_variant"] == {}
        assert out["results"]["totals"]["scored"] == 0

    def test_per_fault_breakdown(self) -> None:
        out = summary_of(
            [
                clean(),
                faulted(fault="empty_success"),
                faulted(fault="partial_list", outcome=Outcome.ABSTAINED),
                faulted(fault="partial_list"),
            ]
        )
        assert out["results"]["by_fault"]["empty_success"]["silent_corruption_rate"] == 1.0
        assert out["results"]["by_fault"]["partial_list"]["silent_corruption_rate"] == 0.5

    def test_variants_are_reported_separately(self) -> None:
        out = summary_of(
            [
                clean(variant="base"),
                clean(variant="distrust"),
                faulted(variant="base"),
                faulted(variant="distrust", outcome=Outcome.ABSTAINED),
            ]
        )
        assert out["results"]["by_variant"]["m / base"]["silent_corruption_rate"] == 1.0
        assert out["results"]["by_variant"]["m / distrust"]["silent_corruption_rate"] == 0.0


class TestExclusions:
    def test_failed_clean_run_withholds_the_whole_cell(self) -> None:
        # If the agent cannot do the task unfaulted, its faulted behaviour is
        # uninterpretable, so those runs must not reach any rate.
        out = summary_of([clean(correct=False), faulted(), faulted()])
        assert out["results"]["totals"]["withheld_precondition"] == 2
        assert out["results"]["totals"]["scored"] == 0
        assert out["results"]["by_variant"] == {}
        assert out["results"]["precondition_failed"] == ["m/base/T1"]

    def test_withholding_is_scoped_to_the_failing_cell(self) -> None:
        out = summary_of(
            [
                clean(task="T1", correct=False),
                clean(task="T2", correct=True),
                faulted(task="T1"),
                faulted(task="T2"),
            ]
        )
        assert out["results"]["totals"]["withheld_precondition"] == 1
        assert out["results"]["totals"]["scored"] == 1
        assert out["results"]["precondition_failed"] == ["m/base/T1"]

    def test_inapplicable_fault_is_skipped_not_scored(self) -> None:
        out = summary_of([clean(), skipped()])
        assert out["results"]["totals"]["skipped_inapplicable"] == 1
        assert out["results"]["totals"]["excluded_not_reached"] == 0
        assert out["results"]["totals"]["scored"] == 0

    def test_unreached_fault_is_excluded_not_scored(self) -> None:
        out = summary_of([clean(), faulted(reached=False)])
        assert out["results"]["totals"]["excluded_not_reached"] == 1
        assert out["results"]["totals"]["skipped_inapplicable"] == 0
        assert out["results"]["totals"]["scored"] == 0

    def test_the_two_exclusion_reasons_are_never_merged(self) -> None:
        out = summary_of([clean(), skipped(), faulted(reached=False)])
        assert out["results"]["totals"]["skipped_inapplicable"] == 1
        assert out["results"]["totals"]["excluded_not_reached"] == 1


class TestTotals:
    def test_tokens_and_runs_are_summed(self) -> None:
        out = summary_of([clean(), faulted(tokens=10), faulted(tokens=32)])
        assert out["results"]["totals"]["runs"] == 3
        assert out["results"]["totals"]["fault_runs"] == 2
        assert out["results"]["totals"]["clean_runs"] == 1
        assert out["results"]["totals"]["total_tokens"] == 42

    def test_every_run_is_retained_for_audit(self) -> None:
        # A reader must be able to recompute the rate from the runs.
        out = summary_of([clean(), faulted(), skipped()])
        assert len(out["results"]["runs"]) == 3
        assert any(r.get("skipped") for r in out["results"]["runs"])


class TestMarkdown:
    def test_synthetic_reports_carry_a_warning(self) -> None:
        text = markdown_report(summarise([clean(), faulted()], synthetic=True))
        assert "Synthetic" in text
        assert "not a finding" in text

    def test_real_reports_carry_no_warning(self) -> None:
        text = markdown_report(summary_of([clean(), faulted()]))
        assert "Synthetic" not in text

    def test_table_shows_the_rate(self) -> None:
        text = markdown_report(summary_of([clean(), faulted()]))
        assert "| m / base | 1 | 1 | 100.0% |" in text

    def test_empty_table_is_still_valid(self) -> None:
        text = markdown_report(summary_of([clean()]))
        assert "_no scored runs_" in text

    def test_withheld_cells_are_named(self) -> None:
        text = markdown_report(summary_of([clean(correct=False), faulted()]))
        assert "`m/base/T1`" in text
        assert "failed the task with no fault injected" in text

    def test_each_exclusion_reason_is_spelled_out(self) -> None:
        text = markdown_report(summary_of([clean(), skipped(), faulted(reached=False)]))
        assert "did not apply" in text
        assert "never reached the answer" in text
