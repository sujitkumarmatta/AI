"""Tests for synthesising SSE framing from a recorded response.

The content a client reassembles must be exactly what was recorded. The chunk
boundaries are invented, so what is tested about them is only that the framing is
valid and that reassembly is lossless -- not that the boundaries mean anything.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from misfeed.streaming import DONE, sse_events, sse_stream

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def completion(**message: Any) -> dict[str, Any]:
    return {
        "id": "r1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", **message},
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
        "usage": USAGE,
    }


def payloads(events: list[str]) -> list[dict[str, Any]]:
    """The JSON bodies of every event except the terminator."""
    out = []
    for event in events:
        assert event.startswith("data: "), event
        assert event.endswith("\n\n"), event
        body = event.removeprefix("data: ").rstrip()
        if body != "[DONE]":
            out.append(json.loads(body))
    return out


def reassemble(events: list[str]) -> str:
    return "".join(
        choice["delta"].get("content", "")
        for payload in payloads(events)
        for choice in payload.get("choices", [])
    )


class TestFraming:
    def test_every_event_is_valid_sse(self) -> None:
        events = list(sse_events(completion(content="hello")))
        for event in events:
            assert event.startswith("data: ")
            assert event.endswith("\n\n")

    def test_terminates_with_done(self) -> None:
        assert list(sse_events(completion(content="hello")))[-1] == DONE

    def test_chunks_are_marked_as_chunks(self) -> None:
        for payload in payloads(list(sse_events(completion(content="hi")))):
            assert payload["object"] == "chat.completion.chunk"

    def test_identity_fields_are_carried_through(self) -> None:
        for payload in payloads(list(sse_events(completion(content="hi")))):
            assert payload["id"] == "r1"
            assert payload["model"] == "m"
            assert payload["created"] == 0

    def test_opens_with_the_role(self) -> None:
        first = payloads(list(sse_events(completion(content="hi"))))[0]
        assert first["choices"][0]["delta"] == {"role": "assistant"}

    def test_closes_with_the_finish_reason(self) -> None:
        last = payloads(list(sse_events(completion(content="hi"))))[-1]
        assert last["choices"][0]["finish_reason"] == "stop"
        assert last["choices"][0]["delta"] == {}

    def test_only_the_final_chunk_carries_a_finish_reason(self) -> None:
        bodies = payloads(list(sse_events(completion(content="hi"))))
        reasons = [c["finish_reason"] for b in bodies for c in b.get("choices", [])]
        assert reasons[-1] == "stop"
        assert all(r is None for r in reasons[:-1])


class TestContentIsLossless:
    @pytest.mark.parametrize(
        "text",
        [
            "hello",
            "",
            "line one\nline two",
            "café — naïve ☕",
            "a" * 5000,
            'quotes "and" \\backslashes\\',
            "ANSWER: 6725",
        ],
    )
    def test_reassembly_returns_the_original(self, text: str) -> None:
        assert reassemble(list(sse_events(completion(content=text)))) == text

    def test_empty_content_emits_no_content_delta(self) -> None:
        bodies = payloads(list(sse_events(completion(content=""))))
        assert all("content" not in c["delta"] for b in bodies for c in b["choices"])

    def test_default_is_a_single_content_delta(self) -> None:
        # The least invention that is still valid SSE: a recorded completion has no
        # token boundaries, so none are manufactured unless asked for.
        bodies = payloads(list(sse_events(completion(content="a fairly long message"))))
        deltas = [c for b in bodies for c in b["choices"] if "content" in c["delta"]]
        assert len(deltas) == 1

    @pytest.mark.parametrize("size", [1, 3, 10])
    def test_requested_chunking_is_still_lossless(self, size: int) -> None:
        text = "the total is 6725 cents"
        events = list(sse_events(completion(content=text), chunk_chars=size))
        assert reassemble(events) == text
        deltas = [c for b in payloads(events) for c in b["choices"] if "content" in c["delta"]]
        assert len(deltas) == -(-len(text) // size)

    @pytest.mark.parametrize("size", [0, -5])
    def test_nonsensical_chunk_sizes_fall_back_to_one_delta(self, size: int) -> None:
        events = list(sse_events(completion(content="hello"), chunk_chars=size))
        assert reassemble(events) == "hello"


class TestToolCalls:
    def call(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_orders", "arguments": '{"customer_id": 1}'},
            }
        ]

    def test_tool_calls_are_emitted_with_an_index(self) -> None:
        bodies = payloads(list(sse_events(completion(content=None, tool_calls=self.call()))))
        deltas = [c["delta"] for b in bodies for c in b["choices"] if "tool_calls" in c["delta"]]
        assert len(deltas) == 1
        assert deltas[0]["tool_calls"][0]["index"] == 0
        assert deltas[0]["tool_calls"][0]["id"] == "call_1"

    def test_arguments_are_not_split_across_deltas(self) -> None:
        # Splitting a JSON document across invented boundaries buys nothing and risks
        # a client seeing unparseable fragments.
        bodies = payloads(
            list(sse_events(completion(content=None, tool_calls=self.call()), chunk_chars=2))
        )
        deltas = [c["delta"] for b in bodies for c in b["choices"] if "tool_calls" in c["delta"]]
        assert len(deltas) == 1
        assert deltas[0]["tool_calls"][0]["function"]["arguments"] == '{"customer_id": 1}'

    def test_finish_reason_is_tool_calls(self) -> None:
        bodies = payloads(list(sse_events(completion(content=None, tool_calls=self.call()))))
        assert bodies[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_multiple_calls_get_distinct_indices(self) -> None:
        calls = [
            *self.call(),
            {"id": "call_2", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ]
        bodies = payloads(list(sse_events(completion(content=None, tool_calls=calls))))
        delta = next(c["delta"] for b in bodies for c in b["choices"] if "tool_calls" in c["delta"])
        assert [tc["index"] for tc in delta["tool_calls"]] == [0, 1]


class TestUsage:
    def test_usage_is_absent_unless_requested(self) -> None:
        bodies = payloads(list(sse_events(completion(content="hi"))))
        assert all("usage" not in b for b in bodies)

    def test_requested_usage_arrives_in_a_final_choiceless_chunk(self) -> None:
        bodies = payloads(list(sse_events(completion(content="hi"), include_usage=True)))
        assert bodies[-1]["choices"] == []
        assert bodies[-1]["usage"] == USAGE

    def test_usage_chunk_precedes_done(self) -> None:
        events = list(sse_events(completion(content="hi"), include_usage=True))
        assert events[-1] == DONE
        assert "usage" in events[-2]


class TestMultipleChoices:
    def test_each_choice_is_streamed_with_its_own_index(self) -> None:
        response = completion(content="a")
        response["choices"].append(
            {
                "index": 1,
                "message": {"role": "assistant", "content": "b"},
                "finish_reason": "stop",
            }
        )
        bodies = payloads(list(sse_events(response)))
        indices = {c["index"] for b in bodies for c in b.get("choices", [])}
        assert indices == {0, 1}

    def test_no_choices_still_terminates(self) -> None:
        response = completion(content="x")
        response["choices"] = []
        assert list(sse_events(response)) == [DONE]


class TestEncoding:
    def test_sse_stream_yields_utf8_bytes(self) -> None:
        chunks = list(sse_stream(completion(content="café")))
        assert all(isinstance(chunk, bytes) for chunk in chunks)
        assert "café" in b"".join(chunks).decode("utf-8")
