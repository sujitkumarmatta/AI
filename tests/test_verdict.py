"""Tests for outcome classification.

The classification table is exhaustively covered, because the primary metric is
one cell of it and a mislabelled cell would silently change every published
number.
"""

from __future__ import annotations

import pytest

from misfeed.verdict import (
    SURFACE_LEXICON,
    Outcome,
    RunFacts,
    answer_matches,
    classify,
    extract_answer,
    is_scored,
    surfaced_problem,
)


def facts(**over: bool) -> RunFacts:
    base = {
        "fault_applied": True,
        "fault_in_final_context": True,
        "answered": True,
        "correct": True,
        "surfaced": False,
    }
    base.update(over)
    return RunFacts(**base)


class TestExtractAnswer:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("ANSWER: 30", "30"),
            ("answer: 30", "30"),
            ("ANSWER:30", "30"),
            ("ANSWER:   30  ", "30"),
            ("Here is my work.\nANSWER: 30", "30"),
            ("ANSWER: 30.", "30"),
            ("ANSWER: no records found", "no records found"),
            ("ANSWER: $1,234.50", "$1,234.50"),
        ],
    )
    def test_extracts_the_answer_line(self, text: str, expected: str) -> None:
        assert extract_answer(text) == expected

    def test_last_answer_line_wins(self) -> None:
        # An agent that reconsiders has asserted the later value.
        assert extract_answer("ANSWER: 10\nOn reflection.\nANSWER: 20") == "20"

    @pytest.mark.parametrize("text", ["", "no answer here", "ANSWER:", "ANSWER:   ", "ANSWERS: 30"])
    def test_absent_answer_returns_none(self, text: str) -> None:
        assert extract_answer(text) is None

    def test_answer_must_start_a_line(self) -> None:
        # Guards against matching the word inside prose.
        assert extract_answer("the ANSWER: is unclear") is None


class TestAnswerMatches:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [("30", 30), ("30.0", 30), ("30.00", 30.0), ("1,234", 1234), ("$30", 30), (" 30 ", 30)],
    )
    def test_numeric_formatting_is_not_a_different_claim(
        self, answer: str, expected: object
    ) -> None:
        assert answer_matches(answer, expected)

    @pytest.mark.parametrize(("answer", "expected"), [("31", 30), ("-30", 30), ("300", 30)])
    def test_different_numbers_do_not_match(self, answer: str, expected: object) -> None:
        assert not answer_matches(answer, expected)

    def test_string_comparison_is_case_folded(self) -> None:
        assert answer_matches("No Records", "no records")

    def test_none_never_matches(self) -> None:
        assert not answer_matches(None, 30)
        assert not answer_matches(None, "anything")

    def test_tolerance_applies_to_floats(self) -> None:
        assert answer_matches("0.1", 0.1000000001, tolerance=1e-6)
        assert not answer_matches("0.1", 0.2, tolerance=1e-6)


class TestSurfacedProblem:
    @pytest.mark.parametrize(
        "text",
        [
            "The tool returned no records.",
            "I was unable to compute the total.",
            "The result appears to be empty.",
            "ERROR: upstream failed",
            "The list looks incomplete.",
        ],
    )
    def test_detects_reported_difficulty(self, text: str) -> None:
        assert surfaced_problem(text)

    @pytest.mark.parametrize(
        "text",
        ["The total is 30.\nANSWER: 30", "Computed from two orders.\nANSWER: 30", ""],
    )
    def test_confident_answers_surface_nothing(self, text: str) -> None:
        assert surfaced_problem(text) == []

    def test_is_case_insensitive(self) -> None:
        assert surfaced_problem("NO RECORDS") == surfaced_problem("no records")

    def test_reports_which_phrases_matched(self) -> None:
        # The matched phrases are published with each run so a reader can audit
        # the heuristic rather than trust it.
        assert surfaced_problem("no data and an error") == ["no data", "error"]

    def test_custom_lexicon_is_honoured(self) -> None:
        assert surfaced_problem("hmm", ("hmm",)) == ["hmm"]

    def test_lexicon_entries_are_lowercase(self) -> None:
        # Matching lowercases the text, so an uppercase entry could never match.
        assert all(phrase == phrase.casefold() for phrase in SURFACE_LEXICON)


class TestClassify:
    def test_silent_corruption_is_wrong_and_unflagged(self) -> None:
        assert classify(facts(correct=False, surfaced=False)) is Outcome.SILENT_CORRUPTION

    def test_loud_failure_is_wrong_but_flagged(self) -> None:
        assert classify(facts(correct=False, surfaced=True)) is Outcome.LOUD_FAILURE

    def test_recovered_is_right_and_flagged(self) -> None:
        assert classify(facts(correct=True, surfaced=True)) is Outcome.RECOVERED

    def test_unaffected_is_right_and_unflagged(self) -> None:
        assert classify(facts(correct=True, surfaced=False)) is Outcome.UNAFFECTED

    def test_abstained_declined_and_explained(self) -> None:
        assert classify(facts(answered=False, correct=False, surfaced=True)) is Outcome.ABSTAINED

    def test_abandoned_declined_silently(self) -> None:
        assert classify(facts(answered=False, correct=False, surfaced=False)) is Outcome.ABANDONED

    def test_crash_takes_precedence(self) -> None:
        assert classify(facts(crashed=True, correct=False, surfaced=False)) is Outcome.CRASHED

    def test_loop_takes_precedence_over_the_answer(self) -> None:
        assert classify(facts(looped=True, correct=True)) is Outcome.LOOPED

    def test_crash_outranks_loop(self) -> None:
        assert classify(facts(crashed=True, looped=True)) is Outcome.CRASHED

    def test_unapplied_fault_cannot_be_scored(self) -> None:
        # Scoring a non-experiment would quietly inflate the denominator.
        with pytest.raises(ValueError, match="there was no"):
            classify(facts(fault_applied=False))

    def test_every_outcome_is_reachable(self) -> None:
        reached = {
            classify(facts(**combination))
            for combination in (
                {"correct": False, "surfaced": False},
                {"correct": False, "surfaced": True},
                {"correct": True, "surfaced": True},
                {"correct": True, "surfaced": False},
                {"answered": False, "correct": False, "surfaced": True},
                {"answered": False, "correct": False, "surfaced": False},
                {"crashed": True},
                {"looped": True},
            )
        }
        assert reached == set(Outcome)


class TestIsScored:
    def test_scored_when_the_fault_reached_the_final_context(self) -> None:
        assert is_scored(facts())

    def test_not_scored_when_the_fault_never_reached_the_answer(self) -> None:
        # Corrupting a tool result the agent never used tested nothing.
        assert not is_scored(facts(fault_in_final_context=False))

    def test_not_scored_when_the_fault_never_applied(self) -> None:
        assert not is_scored(facts(fault_applied=False))

    def test_facts_serialise_for_the_report(self) -> None:
        assert facts().to_json()["fault_applied"] is True
