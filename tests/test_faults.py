"""Tests for the fault taxonomy.

Three properties matter for every fault: it is deterministic (a recorded branch
must be reproducible), it reports inapplicability instead of raising (so skipped
runs are visible in the report rather than lost as errors), and it never mutates
the caller's request in place.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from misfeed.faults import (
    FAULT_IDS,
    REGISTRY,
    FaultSpec,
    apply_fault,
    inject_into_request,
)


def tool_msg(content: str, call_id: str = "c1") -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def assistant_call(name: str, call_id: str = "c1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name}}],
    }


def body_with(*messages: dict[str, Any]) -> dict[str, Any]:
    return {"model": "m", "messages": list(messages)}


ROWS = '{"rows": [{"id": 1, "amount_usd": 10}, {"id": 2, "amount_usd": 20}], "total_usd": 30}'


class TestRegistry:
    def test_every_documented_fault_is_registered(self) -> None:
        expected = {
            "empty_success",
            "missing_fields",
            "partial_list",
            "truncated",
            "schema_drift",
            "stale",
            "unit_shift",
            "error_text",
            "injected_instruction",
        }
        assert set(FAULT_IDS) == expected

    def test_unknown_fault_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown fault"):
            apply_fault("nonexistent", "{}")

    def test_unknown_fault_rejected_at_spec_construction(self) -> None:
        with pytest.raises(ValueError, match="unknown fault"):
            FaultSpec(fault="nope")

    @pytest.mark.parametrize("name", sorted(FAULT_IDS))
    def test_every_fault_is_deterministic(self, name: str) -> None:
        first = apply_fault(name, ROWS)
        second = apply_fault(name, ROWS)
        assert first == second

    @pytest.mark.parametrize("name", sorted(FAULT_IDS))
    def test_no_fault_raises_on_plain_text(self, name: str) -> None:
        # JSON-aware faults must report inapplicability, not blow up mid-run.
        result = apply_fault(name, "not json at all")
        if not result.changed:
            assert "skipped" in result.detail


class TestEmptySuccess:
    def test_blanks_the_payload(self) -> None:
        result = apply_fault("empty_success", ROWS)
        assert result.changed
        assert result.content == ""
        assert result.detail["was_bytes"] == len(ROWS)

    @pytest.mark.parametrize("shape", ["", "{}", "[]", "null"])
    def test_supported_shapes(self, shape: str) -> None:
        assert apply_fault("empty_success", ROWS, {"shape": shape}).content == shape

    def test_already_empty_is_a_noop(self) -> None:
        result = apply_fault("empty_success", "")
        assert not result.changed
        assert result.detail["skipped"] == "already empty"

    def test_invalid_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape must be"):
            apply_fault("empty_success", ROWS, {"shape": "0"})


class TestTruncated:
    def test_cuts_mid_structure(self) -> None:
        result = apply_fault("truncated", ROWS, {"keep": 0.5})
        assert result.changed
        assert len(result.content) == len(ROWS) // 2
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.content)

    def test_keep_zero_empties(self) -> None:
        assert apply_fault("truncated", ROWS, {"keep": 0.0}).content == ""

    @pytest.mark.parametrize("keep", [-0.1, 1.0, 1.5])
    def test_out_of_range_keep_rejected(self, keep: float) -> None:
        with pytest.raises(ValueError, match="keep must be in"):
            apply_fault("truncated", ROWS, {"keep": keep})

    def test_empty_content_is_a_noop(self) -> None:
        assert not apply_fault("truncated", "").changed


class TestMissingFields:
    def test_drops_named_field_at_every_depth(self) -> None:
        result = apply_fault("missing_fields", ROWS, {"fields": ["amount_usd"]})
        assert result.changed
        obj = json.loads(result.content)
        assert all("amount_usd" not in row for row in obj["rows"])
        assert obj["total_usd"] == 30  # untouched
        assert result.detail["dropped"] == ["amount_usd", "amount_usd"]

    def test_defaults_to_first_field_of_first_record(self) -> None:
        result = apply_fault("missing_fields", ROWS)
        assert result.changed
        assert result.detail["dropped"]

    def test_absent_field_reports_skip(self) -> None:
        result = apply_fault("missing_fields", ROWS, {"fields": ["nope"]})
        assert not result.changed
        assert "not present" in result.detail["skipped"]

    def test_empty_object_reports_skip(self) -> None:
        result = apply_fault("missing_fields", "{}")
        assert not result.changed
        assert result.detail["skipped"] == "no object field to drop"

    def test_scalar_json_reports_skip(self) -> None:
        assert not apply_fault("missing_fields", "42").changed


class TestPartialList:
    def test_shortens_nested_list_silently(self) -> None:
        result = apply_fault("partial_list", ROWS, {"keep_n": 1})
        assert result.changed
        obj = json.loads(result.content)
        assert len(obj["rows"]) == 1
        # The stale total is what makes this dangerous: the payload is
        # self-inconsistent but still looks complete.
        assert obj["total_usd"] == 30
        assert result.detail == {"at": "rows", "kept": 1, "was": 2}

    def test_shortens_root_list(self) -> None:
        result = apply_fault("partial_list", '[{"a": 1}, {"a": 2}, {"a": 3}]', {"keep_n": 2})
        assert json.loads(result.content) == [{"a": 1}, {"a": 2}]
        assert result.detail["at"] == "$"

    def test_keep_n_zero_empties_the_list(self) -> None:
        result = apply_fault("partial_list", ROWS, {"keep_n": 0})
        assert json.loads(result.content)["rows"] == []

    def test_list_already_short_enough_reports_skip(self) -> None:
        result = apply_fault("partial_list", '{"rows": [{"a": 1}]}', {"keep_n": 1})
        assert not result.changed
        assert "already <= 1" in result.detail["skipped"]

    def test_no_list_reports_skip(self) -> None:
        result = apply_fault("partial_list", '{"a": 1}')
        assert not result.changed
        assert result.detail["skipped"] == "no list in content"

    def test_negative_keep_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="keep_n must be >= 0"):
            apply_fault("partial_list", ROWS, {"keep_n": -1})


class TestSchemaDrift:
    def test_default_camel_cases_snake_keys(self) -> None:
        result = apply_fault("schema_drift", ROWS)
        assert result.changed
        obj = json.loads(result.content)
        assert "totalUsd" in obj
        assert "total_usd" not in obj
        assert "amountUsd" in obj["rows"][0]

    def test_explicit_rename_map(self) -> None:
        result = apply_fault("schema_drift", ROWS, {"rename": {"total_usd": "sum"}})
        obj = json.loads(result.content)
        assert obj["sum"] == 30
        assert "amount_usd" in obj["rows"][0]  # not renamed
        assert result.detail["renamed"] == {"total_usd": "sum"}

    def test_no_renameable_keys_reports_skip(self) -> None:
        result = apply_fault("schema_drift", '{"id": 1}')
        assert not result.changed
        assert result.detail["skipped"] == "no keys to rename"

    def test_values_are_preserved(self) -> None:
        obj = json.loads(apply_fault("schema_drift", ROWS).content)
        assert obj["rows"][0]["amountUsd"] == 10


class TestStale:
    def test_shifts_dates_backwards(self) -> None:
        result = apply_fault("stale", '{"as_of": "2026-09-27"}', {"shift_days": 30})
        assert result.changed
        assert json.loads(result.content)["as_of"] == "2026-08-28"
        assert result.detail["shifted"] == [{"from": "2026-09-27", "to": "2026-08-28"}]

    def test_negative_shift_moves_forward(self) -> None:
        result = apply_fault("stale", '{"d": "2026-01-01"}', {"shift_days": -1})
        assert json.loads(result.content)["d"] == "2026-01-02"

    def test_impossible_date_left_alone(self) -> None:
        result = apply_fault("stale", '{"d": "2026-02-30"}')
        assert not result.changed

    def test_no_dates_reports_skip(self) -> None:
        result = apply_fault("stale", ROWS)
        assert not result.changed
        assert result.detail["skipped"] == "no ISO dates in content"

    def test_zero_shift_rejected(self) -> None:
        with pytest.raises(ValueError, match="shift_days must be non-zero"):
            apply_fault("stale", '{"d": "2026-01-01"}', {"shift_days": 0})


class TestUnitShift:
    def test_scales_every_number(self) -> None:
        result = apply_fault("unit_shift", ROWS, {"factor": 100})
        assert result.changed
        obj = json.loads(result.content)
        assert obj["total_usd"] == 3000
        assert obj["rows"][0]["amount_usd"] == 1000

    def test_restricts_to_named_fields(self) -> None:
        result = apply_fault("unit_shift", ROWS, {"factor": 100, "fields": ["total_usd"]})
        obj = json.loads(result.content)
        assert obj["total_usd"] == 3000
        assert obj["rows"][0]["amount_usd"] == 10  # untouched
        assert obj["rows"][0]["id"] == 1

    def test_integers_stay_integers_when_whole(self) -> None:
        # A value rendered as 1000.0 would be spottable for the wrong reason.
        content = apply_fault("unit_shift", ROWS, {"factor": 100}).content
        assert ".0" not in content
        obj = json.loads(content)
        assert isinstance(obj["total_usd"], int)
        assert isinstance(obj["rows"][0]["amount_usd"], int)

    def test_fractional_results_stay_floats(self) -> None:
        obj = json.loads(apply_fault("unit_shift", '{"n": 5}', {"factor": 0.5}).content)
        assert obj["n"] == 2.5

    def test_booleans_are_not_scaled(self) -> None:
        result = apply_fault("unit_shift", '{"ok": true, "n": 2}', {"factor": 10})
        assert json.loads(result.content) == {"ok": True, "n": 20}

    def test_no_numbers_reports_skip(self) -> None:
        result = apply_fault("unit_shift", '{"a": "x"}')
        assert not result.changed
        assert result.detail["skipped"] == "no numbers to scale"

    def test_identity_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"factor must not be 1\.0"):
            apply_fault("unit_shift", ROWS, {"factor": 1.0})


class TestErrorTextAndInjection:
    def test_error_text_replaces_payload(self) -> None:
        result = apply_fault("error_text", ROWS)
        assert result.changed
        assert "503" in result.content

    def test_injection_appends_by_default(self) -> None:
        result = apply_fault("injected_instruction", ROWS)
        assert result.changed
        assert result.content.startswith(ROWS)
        assert "Ignore all previous instructions" in result.content

    @pytest.mark.parametrize(
        ("where", "check"),
        [
            ("prepend", lambda c: c.endswith(ROWS)),
            ("replace", lambda c: ROWS not in c),
        ],
    )
    def test_injection_placement(self, where: str, check: Any) -> None:
        result = apply_fault("injected_instruction", ROWS, {"where": where, "text": "X"})
        assert check(result.content)

    def test_invalid_placement_rejected(self) -> None:
        with pytest.raises(ValueError, match="where must be"):
            apply_fault("injected_instruction", ROWS, {"where": "sideways"})


class TestInjectIntoRequest:
    def test_targets_the_most_recent_tool_result(self) -> None:
        body = body_with(
            {"role": "user", "content": "q"},
            assistant_call("lookup", "c1"),
            tool_msg("old", "c1"),
            assistant_call("lookup", "c2"),
            tool_msg(ROWS, "c2"),
        )
        outcome = inject_into_request(body, FaultSpec(fault="empty_success"))
        assert outcome.applied
        assert outcome.body["messages"][4]["content"] == ""
        assert outcome.body["messages"][2]["content"] == "old"  # earlier one untouched
        assert outcome.detail["message_index"] == 4

    def test_resolves_tool_name_through_tool_call_id(self) -> None:
        body = body_with(assistant_call("orders_query", "c9"), tool_msg(ROWS, "c9"))
        outcome = inject_into_request(body, FaultSpec(fault="empty_success"))
        assert outcome.detail["tool_name"] == "orders_query"

    def test_named_targeting_picks_the_matching_tool(self) -> None:
        body = body_with(
            assistant_call("orders_query", "c1"),
            tool_msg("orders payload", "c1"),
            assistant_call("fx_rate", "c2"),
            tool_msg("fx payload", "c2"),
        )
        outcome = inject_into_request(
            body, FaultSpec(fault="empty_success", tool_name="orders_query")
        )
        assert outcome.applied
        assert outcome.body["messages"][1]["content"] == ""
        assert outcome.body["messages"][3]["content"] == "fx payload"

    def test_named_targeting_reports_skip_when_absent(self) -> None:
        body = body_with(assistant_call("fx_rate", "c2"), tool_msg("fx", "c2"))
        outcome = inject_into_request(body, FaultSpec(fault="empty_success", tool_name="orders"))
        assert not outcome.applied
        assert "no tool-result message from tool 'orders'" in outcome.detail["skipped"]

    def test_no_tool_result_reports_skip(self) -> None:
        outcome = inject_into_request(
            body_with({"role": "user", "content": "q"}), FaultSpec(fault="empty_success")
        )
        assert not outcome.applied
        assert outcome.detail["skipped"] == "no tool-result message"

    def test_missing_messages_reports_skip(self) -> None:
        outcome = inject_into_request({"model": "m"}, FaultSpec(fault="empty_success"))
        assert not outcome.applied
        assert outcome.detail["skipped"] == "no messages"

    def test_non_string_tool_content_reports_skip(self) -> None:
        body = body_with({"role": "tool", "tool_call_id": "c1", "content": [{"type": "text"}]})
        outcome = inject_into_request(body, FaultSpec(fault="empty_success"))
        assert not outcome.applied
        assert outcome.detail["skipped"] == "tool-result content is not a string"

    def test_inapplicable_fault_propagates_as_not_applied(self) -> None:
        body = body_with(assistant_call("t"), tool_msg("plain text"))
        outcome = inject_into_request(body, FaultSpec(fault="partial_list"))
        assert not outcome.applied
        assert outcome.detail["skipped"] == "content is not JSON"
        assert outcome.detail["fault"] == "partial_list"

    def test_does_not_mutate_the_callers_request(self) -> None:
        body = body_with(assistant_call("t"), tool_msg(ROWS))
        snapshot = json.dumps(body, sort_keys=True)
        inject_into_request(body, FaultSpec(fault="empty_success"))
        assert json.dumps(body, sort_keys=True) == snapshot

    def test_spec_serialises_for_the_report(self) -> None:
        spec = FaultSpec(fault="partial_list", params={"keep_n": 1}, tool_name="orders")
        assert spec.to_json() == {
            "fault": "partial_list",
            "params": {"keep_n": 1},
            "tool_name": "orders",
        }

    def test_registry_and_ids_agree(self) -> None:
        assert set(REGISTRY) == set(FAULT_IDS)
