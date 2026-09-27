"""Tests for request canonicalisation.

These focus on the cases that would make replay quietly wrong: keys that should
collide but do not, and keys that should differ but collide.
"""

from __future__ import annotations

from typing import Any

import pytest

from misfeed.canon import (
    DEFAULT_RULES,
    NormalizeRule,
    canonical_json,
    canonical_request,
    explain,
    normalize_text,
    request_key,
)


def req(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }
    base.update(over)
    return base


class TestKeysThatMustCollide:
    def test_field_order_is_irrelevant(self) -> None:
        a = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
        b = {"messages": [{"role": "user", "content": "x"}], "model": "m"}
        assert request_key(a) == request_key(b)

    def test_streaming_framing_does_not_change_the_key(self) -> None:
        # A streaming and a non-streaming call for the same content are the same
        # logical request; the framing is synthesised at replay time.
        assert request_key(req(stream=True)) == request_key(req(stream=False))
        assert request_key(req(stream=True, stream_options={"include_usage": True})) == request_key(
            req()
        )

    def test_caller_bookkeeping_does_not_change_the_key(self) -> None:
        noisy = req(user="alice", metadata={"run": "7"}, store=True)
        assert request_key(noisy) == request_key(req())

    def test_integral_float_equals_int(self) -> None:
        assert request_key(req(temperature=0.0)) == request_key(req(temperature=0))
        assert request_key(req(top_p=1.0)) == request_key(req(top_p=1))

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("run at 2026-09-27T07:30:00Z", "run at 2026-01-02T23:59:59Z"),
            ("run at 2026-09-27 07:30:00.123+05:30", "run at 2024-03-04 01:02:03Z"),
            ("id 3f2504e0-4f89-11d3-9a0c-0305e82c3301", "id 550e8400-e29b-41d4-a716-446655440000"),
            ("t=1758960000000", "t=1699999999999"),
        ],
    )
    def test_volatile_content_is_normalised(self, first: str, second: str) -> None:
        a = req(messages=[{"role": "system", "content": first}])
        b = req(messages=[{"role": "system", "content": second}])
        assert request_key(a) == request_key(b)

    def test_volatile_content_normalised_inside_nested_parts(self) -> None:
        def msgs(ts: str) -> list[dict[str, Any]]:
            return [{"role": "user", "content": [{"type": "text", "text": f"as of {ts}"}]}]

        a = req(messages=msgs("2026-09-27T07:30:00Z"))
        b = req(messages=msgs("2020-01-01T00:00:00Z"))
        assert request_key(a) == request_key(b)


class TestKeysThatMustDiffer:
    def test_prompt_change_changes_the_key(self) -> None:
        assert request_key(req()) != request_key(
            req(messages=[{"role": "user", "content": "hello."}])
        )

    def test_model_change_changes_the_key(self) -> None:
        assert request_key(req()) != request_key(req(model="other"))

    def test_sampling_change_changes_the_key(self) -> None:
        assert request_key(req()) != request_key(req(temperature=1))

    def test_tool_definitions_change_the_key(self) -> None:
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        assert request_key(req()) != request_key(req(tools=tools))

    def test_message_order_changes_the_key(self) -> None:
        one = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        two = [{"role": "user", "content": "b"}, {"role": "user", "content": "a"}]
        assert request_key(req(messages=one)) != request_key(req(messages=two))

    def test_bool_is_not_conflated_with_int(self) -> None:
        # bool is a subclass of int in Python; a naive numeric branch would make
        # parallel_tool_calls=True collide with =1.
        assert request_key(req(parallel_tool_calls=True)) != request_key(req(parallel_tool_calls=1))

    def test_non_integral_float_is_preserved(self) -> None:
        assert request_key(req(temperature=0.7)) != request_key(req(temperature=0.8))


class TestNormalisationRules:
    def test_rules_apply_in_order(self) -> None:
        rules = (
            NormalizeRule.of("a", r"foo", "bar"),
            NormalizeRule.of("b", r"bar", "baz"),
        )
        assert normalize_text("foo", rules) == "baz"

    def test_empty_ruleset_is_identity(self) -> None:
        text = "2026-09-27T07:30:00Z"
        assert normalize_text(text, ()) == text

    def test_default_rules_leave_ordinary_numbers_alone(self) -> None:
        # A 13-digit epoch is masked; a plain quantity must not be.
        assert normalize_text("total 1234", DEFAULT_RULES) == "total 1234"
        assert normalize_text("qty 20260927", DEFAULT_RULES) == "qty 20260927"


class TestCanonicalForm:
    def test_unknown_fields_are_dropped(self) -> None:
        canon = canonical_request(req(some_vendor_extension={"x": 1}))
        assert "some_vendor_extension" not in canon

    def test_canonical_json_is_stable_and_compact(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    def test_canonical_json_preserves_non_ascii(self) -> None:
        assert canonical_json({"k": "café"}) == '{"k":"café"}'.encode()

    def test_explain_reports_what_was_hashed(self) -> None:
        out = explain(req(stream=True, user="alice"))
        assert out["key"] == request_key(req())
        assert out["dropped_fields"] == ["stream", "user"]
        assert out["rules"] == ["iso8601", "uuid", "epoch_ms"]
        assert out["canonical"]["model"] == "m"


class TestNullNormalisation:
    """Explicit nulls must not make two equivalent requests look different.

    A loop that appends a response message verbatim sends `"content": null` on an
    assistant tool-call message; the official SDK omits the field. Both mean "no
    content", and treating them as different requests made cassettes
    client-specific -- caught by the SDK compatibility test, not by a unit test.
    """

    def test_explicit_null_equals_an_omitted_field(self) -> None:
        verbose = req(
            messages=[
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            ]
        )
        terse = req(messages=[{"role": "assistant", "tool_calls": [{"id": "c1"}]}])
        assert request_key(verbose) == request_key(terse)

    def test_nulls_are_dropped_at_every_depth(self) -> None:
        nested = req(
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": None}}],
                }
            ]
        )
        canon = canonical_request(nested)
        assert "arguments" not in canon["messages"][0]["tool_calls"][0]["function"]

    def test_null_top_level_option_equals_an_omitted_one(self) -> None:
        assert request_key(req(stop=None)) == request_key(req())

    def test_a_present_value_is_not_dropped(self) -> None:
        assert request_key(req(stop=["x"])) != request_key(req())

    def test_false_and_zero_survive(self) -> None:
        # Only None is dropped. Falsy-but-present values are real settings.
        canon = canonical_request(req(parallel_tool_calls=False, temperature=0))
        assert canon["parallel_tool_calls"] is False
        assert canon["temperature"] == 0

    def test_explain_reports_the_structural_rule(self) -> None:
        assert explain(req())["structural_rules"] == ["drop_null_fields"]
