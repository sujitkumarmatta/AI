"""Tests for tool-side fault injection.

The property that justifies this module's existence is transience: a fault applied in
flight lives in the conversation history and is permanent by construction, so only a
tool-side fault can answer "does the agent's retry actually help?".
"""

from __future__ import annotations

import json

import pytest

from misfeed.faults import FaultSpec
from misfeed.toolfault import ToolFaultPlan, ToolInjector

ROWS = '{"rows": [{"id": 1, "amount": 10}, {"id": 2, "amount": 20}], "total": 30}'


def plan(fault: str = "empty_success", **kw: object) -> ToolFaultPlan:
    return ToolFaultPlan(fault=FaultSpec(fault=fault), **kw)  # type: ignore[arg-type]


class TestNoPlan:
    def test_passes_everything_through(self) -> None:
        injector = ToolInjector()
        assert injector.apply(tool_name="t", result=ROWS) == ROWS
        assert injector.applied is False
        assert injector.outcome is None

    def test_still_counts_calls(self) -> None:
        injector = ToolInjector()
        for _ in range(3):
            injector.apply(tool_name="t", result=ROWS)
        assert injector.calls == 3


class TestTargeting:
    def test_corrupts_the_first_call_by_default(self) -> None:
        injector = ToolInjector(plan())
        assert injector.apply(tool_name="t", result=ROWS) == ""
        assert injector.applied is True

    def test_occurrence_selects_a_later_call(self) -> None:
        injector = ToolInjector(plan(occurrence=1))
        assert injector.apply(tool_name="t", result=ROWS) == ROWS
        assert injector.apply(tool_name="t", result=ROWS) == ""
        assert [c.call_index for c in injector.corruptions] == [1]

    def test_tool_name_restricts_matching(self) -> None:
        injector = ToolInjector(plan(tool_name="orders"))
        assert injector.apply(tool_name="customers", result=ROWS) == ROWS
        assert injector.apply(tool_name="orders", result=ROWS) == ""
        assert injector.corruptions[0].tool_name == "orders"

    def test_occurrence_counts_matches_not_all_calls(self) -> None:
        # The second *orders* call, not the second call overall.
        injector = ToolInjector(plan(tool_name="orders", occurrence=1))
        assert injector.apply(tool_name="customers", result=ROWS) == ROWS
        assert injector.apply(tool_name="orders", result=ROWS) == ROWS
        assert injector.apply(tool_name="orders", result=ROWS) == ""
        assert injector.matches == 2

    def test_a_never_matching_plan_changes_nothing(self) -> None:
        injector = ToolInjector(plan(tool_name="absent"))
        assert injector.apply(tool_name="t", result=ROWS) == ROWS
        assert injector.applied is False
        outcome = injector.outcome
        assert outcome is not None
        assert outcome["matching_calls"] == 0

    def test_negative_occurrence_rejected(self) -> None:
        with pytest.raises(ValueError, match="occurrence must be >= 0"):
            plan(occurrence=-1)


class TestTransience:
    """The capability the in-flight injector cannot express."""

    def test_transient_by_default_so_a_retry_gets_real_data(self) -> None:
        injector = ToolInjector(plan())
        assert injector.apply(tool_name="t", result=ROWS) == ""
        # The retry succeeds, which is what makes "did the retry help?" answerable.
        assert injector.apply(tool_name="t", result=ROWS) == ROWS
        assert len(injector.corruptions) == 1

    def test_persist_breaks_every_later_call(self) -> None:
        injector = ToolInjector(plan(persist=True))
        assert injector.apply(tool_name="t", result=ROWS) == ""
        assert injector.apply(tool_name="t", result=ROWS) == ""
        assert injector.apply(tool_name="t", result=ROWS) == ""
        assert len(injector.corruptions) == 3

    def test_persist_still_respects_the_occurrence(self) -> None:
        injector = ToolInjector(plan(occurrence=1, persist=True))
        assert injector.apply(tool_name="t", result=ROWS) == ROWS
        assert injector.apply(tool_name="t", result=ROWS) == ""
        assert injector.apply(tool_name="t", result=ROWS) == ""
        assert [c.call_index for c in injector.corruptions] == [1, 2]

    def test_persist_still_respects_the_tool_name(self) -> None:
        injector = ToolInjector(plan(tool_name="orders", persist=True))
        injector.apply(tool_name="orders", result=ROWS)
        assert injector.apply(tool_name="other", result=ROWS) == ROWS
        assert injector.apply(tool_name="orders", result=ROWS) == ""
        assert len(injector.corruptions) == 2


class TestInapplicableFaults:
    def test_reports_instead_of_raising(self) -> None:
        # A JSON fault against plain text must show up in the report, not break the run.
        injector = ToolInjector(plan("partial_list"))
        assert injector.apply(tool_name="t", result="plain text") == "plain text"
        assert injector.applied is False
        assert injector.skipped[0]["skipped"] == "content is not JSON"
        assert injector.skipped[0]["fault"] == "partial_list"

    def test_an_inapplicable_call_does_not_consume_the_occurrence(self) -> None:
        # It matched and was attempted, so it counts as the occurrence; the next call
        # is not silently corrupted instead. Otherwise a skip would move the target.
        injector = ToolInjector(plan("partial_list"))
        injector.apply(tool_name="t", result="plain text")
        assert injector.apply(tool_name="t", result=ROWS) == ROWS
        assert injector.applied is False


class TestFaultsBehaveAsElsewhere:
    def test_partial_list_shortens_the_list(self) -> None:
        injector = ToolInjector(plan("partial_list"))
        out = json.loads(injector.apply(tool_name="t", result=ROWS))
        assert len(out["rows"]) == 1
        assert out["total"] == 30  # left inconsistent, as in the wire-level injector

    def test_parameters_are_passed_through(self) -> None:
        injector = ToolInjector(
            ToolFaultPlan(fault=FaultSpec(fault="unit_shift", params={"factor": 100}))
        )
        out = json.loads(injector.apply(tool_name="t", result=ROWS))
        assert out["total"] == 3000

    def test_is_deterministic(self) -> None:
        first = ToolInjector(plan("truncated")).apply(tool_name="t", result=ROWS)
        second = ToolInjector(plan("truncated")).apply(tool_name="t", result=ROWS)
        assert first == second


class TestOutcomeReporting:
    def test_outcome_describes_the_plan_and_what_happened(self) -> None:
        injector = ToolInjector(plan(tool_name="orders", occurrence=1))
        injector.apply(tool_name="orders", result=ROWS)
        injector.apply(tool_name="orders", result=ROWS)
        outcome = injector.outcome
        assert outcome is not None
        assert outcome["fault"] == "empty_success"
        assert outcome["tool_name"] == "orders"
        assert outcome["occurrence"] == 1
        assert outcome["persist"] is False
        assert outcome["applied"] is True
        assert outcome["tool_calls_seen"] == 2
        assert outcome["matching_calls"] == 2
        assert len(outcome["corruptions"]) == 1

    def test_outcome_marks_where_the_fault_was_applied(self) -> None:
        # So a report can tell a tool-side experiment from a wire-level one.
        injector = ToolInjector(plan())
        injector.apply(tool_name="t", result=ROWS)
        outcome = injector.outcome
        assert outcome is not None
        assert outcome["injected_at"] == "tool"
