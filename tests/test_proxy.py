"""Tests for the record/replay/inject proxy.

The upstream is always an httpx MockTransport, so nothing here touches the
network or needs a key. The properties that matter most:

- a replay miss fails loudly instead of going live (a silent fallthrough would be
  an invisible, billed, non-reproducible run);
- re-running a recorded experiment costs zero live calls, which is what makes
  published numbers reproducible by someone without an endpoint;
- recording is resumable, so a rate limit does not waste what was already
  captured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from starlette.testclient import TestClient

from misfeed.faults import FaultSpec
from misfeed.proxy import BudgetExceeded, Engine, Mode, ProxyConfig, create_app
from misfeed.store import CassetteMiss, Store

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def text_reply(text: str) -> dict[str, Any]:
    return {
        "id": "r",
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": USAGE,
    }


def tool_call_reply(name: str, arguments: str, call_id: str = "c1") -> dict[str, Any]:
    return {
        "id": "r",
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": USAGE,
    }


class FakeUpstream:
    """Replies in sequence, and records the request bodies it was sent."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies
        self.seen: list[dict[str, Any]] = []

    def client(self) -> httpx2.AsyncClient:
        def handler(request: httpx2.Request) -> httpx2.Response:
            self.seen.append(json.loads(request.content))
            if not self.replies:
                return httpx2.Response(500, json={"error": "fake upstream exhausted"})
            return httpx2.Response(200, json=self.replies.pop(0))

        return httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler), base_url="http://upstream/v1"
        )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "cassettes")


def ask(text: str = "q") -> dict[str, Any]:
    return {"model": "m", "messages": [{"role": "user", "content": text}]}


def with_tool_result(payload: str, tool: str = "orders") -> dict[str, Any]:
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": tool, "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": payload},
        ],
    }


class TestConfigValidation:
    def test_inject_requires_parent_fault_and_step(self, store: Store) -> None:
        with pytest.raises(ValueError, match="INJECT mode requires"):
            ProxyConfig(mode=Mode.INJECT, store=store, trace_id="b", upstream_base_url="http://u")

    def test_negative_fault_step_rejected(self, store: Store) -> None:
        with pytest.raises(ValueError, match="fault_at_step must be >= 0"):
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b",
                parent="t",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=-1,
                upstream_base_url="http://u",
            )

    def test_non_inject_mode_rejects_a_fault(self, store: Store) -> None:
        with pytest.raises(ValueError, match="must not be given a fault"):
            ProxyConfig(
                mode=Mode.RECORD,
                store=store,
                trace_id="t",
                fault=FaultSpec(fault="empty_success"),
                upstream_base_url="http://u",
            )

    def test_record_requires_an_upstream(self, store: Store) -> None:
        with pytest.raises(ValueError, match="requires upstream_base_url"):
            ProxyConfig(mode=Mode.RECORD, store=store, trace_id="t")

    def test_replay_needs_no_upstream(self, store: Store) -> None:
        ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="t")  # must not raise


class TestRecord:
    async def test_records_a_live_response(self, store: Store) -> None:
        upstream = FakeUpstream([text_reply("hello")])
        config = ProxyConfig(
            mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
        )
        engine = Engine(config, upstream.client())
        reply = await engine.handle(ask())
        assert reply["choices"][0]["message"]["content"] == "hello"
        assert engine.live_calls == 1
        trace = engine.flush()
        assert trace is not None
        assert len(trace.entries) == 1
        assert trace.meta["live_calls"] == 1
        assert trace.meta["steps"][0]["usage"] == USAGE

    async def test_second_run_serves_from_cassette_with_no_live_calls(self, store: Store) -> None:
        # This is the property that makes a recorded experiment free to re-run.
        upstream = FakeUpstream([text_reply("hello")])
        first = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            upstream.client(),
        )
        await first.handle(ask())
        first.flush()

        empty = FakeUpstream([])  # any live call would 500
        second = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            empty.client(),
        )
        reply = await second.handle(ask())
        assert reply["choices"][0]["message"]["content"] == "hello"
        assert second.live_calls == 0
        assert second.observations[0].source == "cassette"

    async def test_recording_is_resumable_after_a_cap(self, store: Store) -> None:
        # A free tier's rate limit must not waste what was already captured.
        upstream = FakeUpstream([text_reply("one")])
        config = ProxyConfig(
            mode=Mode.RECORD,
            store=store,
            trace_id="t1",
            upstream_base_url="http://upstream/v1",
            max_live_requests=1,
        )
        engine = Engine(config, upstream.client())
        await engine.handle(ask("first"))
        with pytest.raises(BudgetExceeded):
            await engine.handle(ask("second"))
        engine.flush()

        resumed = Engine(
            ProxyConfig(
                mode=Mode.RECORD,
                store=store,
                trace_id="t1",
                upstream_base_url="http://upstream/v1",
                max_live_requests=5,
            ),
            FakeUpstream([text_reply("two")]).client(),
        )
        assert (await resumed.handle(ask("first")))["choices"][0]["message"]["content"] == "one"
        assert resumed.live_calls == 0  # first request came free
        assert (await resumed.handle(ask("second")))["choices"][0]["message"]["content"] == "two"
        assert resumed.live_calls == 1
        trace = resumed.flush()
        assert trace is not None
        assert len(trace.entries) == 2

    async def test_stores_the_canonical_request_for_review(self, store: Store) -> None:
        engine = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([text_reply("hi")]).client(),
        )
        await engine.handle(ask("what is the total?"))
        trace = engine.flush()
        assert trace is not None
        blob = trace.entries[0].request
        assert blob is not None
        assert store.get_blob(blob)["messages"][0]["content"] == "what is the total?"

    async def test_can_record_without_retaining_prompts(self, store: Store) -> None:
        engine = Engine(
            ProxyConfig(
                mode=Mode.RECORD,
                store=store,
                trace_id="t1",
                upstream_base_url="http://upstream/v1",
                store_requests=False,
            ),
            FakeUpstream([text_reply("hi")]).client(),
        )
        await engine.handle(ask("sensitive"))
        trace = engine.flush()
        assert trace is not None
        assert trace.entries[0].request is None

    async def test_streaming_flags_are_stripped_from_the_upstream_call(self, store: Store) -> None:
        upstream = FakeUpstream([text_reply("hi")])
        engine = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            upstream.client(),
        )
        await engine.handle({**ask(), "stream": True, "stream_options": {"include_usage": True}})
        assert "stream" not in upstream.seen[0]
        assert "stream_options" not in upstream.seen[0]

    async def test_flush_is_a_noop_when_nothing_was_recorded(self, store: Store) -> None:
        engine = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([]).client(),
        )
        assert engine.flush() is None


class TestReplay:
    async def test_serves_recorded_run_without_any_upstream(self, store: Store) -> None:
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([text_reply("recorded")]).client(),
        )
        await recorder.handle(ask())
        recorder.flush()

        engine = Engine(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="t1"))
        assert engine.client is None
        reply = await engine.handle(ask())
        assert reply["choices"][0]["message"]["content"] == "recorded"
        assert engine.live_calls == 0

    async def test_miss_raises_instead_of_going_live(self, store: Store) -> None:
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([text_reply("recorded")]).client(),
        )
        await recorder.handle(ask("recorded question"))
        recorder.flush()

        engine = Engine(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="t1"))
        with pytest.raises(CassetteMiss) as caught:
            await engine.handle(ask("a different question"))
        assert caught.value.detail["reason"] == "key not recorded"
        assert "a different question" in json.dumps(caught.value.detail["canonical"])

    async def test_unknown_trace_fails_at_construction(self, store: Store) -> None:
        with pytest.raises(FileNotFoundError):
            Engine(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="absent"))

    async def test_unserved_entries_are_reported_as_drift(self, store: Store) -> None:
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([text_reply("a"), text_reply("b")]).client(),
        )
        await recorder.handle(ask("one"))
        await recorder.handle(ask("two"))
        recorder.flush()

        engine = Engine(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="t1"))
        await engine.handle(ask("one"))
        assert engine.cassette_remaining == 1  # the run stopped early


class TestInject:
    async def _trunk(self, store: Store) -> None:
        """A two-step run: tool call, then an answer built from the tool result."""
        engine = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([tool_call_reply("orders", "{}"), text_reply("total is 30")]).client(),
        )
        await engine.handle(ask())
        await engine.handle(with_tool_result('{"total": 30}'))
        engine.flush()

    async def test_serves_trunk_prefix_then_diverges_at_the_fault(self, store: Store) -> None:
        await self._trunk(store)
        upstream = FakeUpstream([text_reply("No records found.")])
        engine = Engine(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b1",
                parent="t1",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=1,
                upstream_base_url="http://upstream/v1",
            ),
            upstream.client(),
        )
        # Step 0 is before the fork, so it comes free from the trunk.
        await engine.handle(ask())
        assert engine.live_calls == 0

        reply = await engine.handle(with_tool_result('{"total": 30}'))
        assert engine.live_calls == 1
        assert reply["choices"][0]["message"]["content"] == "No records found."
        # The model was shown an empty tool result, not the real one.
        assert upstream.seen[0]["messages"][2]["content"] == ""
        assert engine.injection is not None
        assert engine.injection["applied"] is True
        assert engine.injection["tool_name"] == "orders"

    async def test_branch_records_only_the_continuation(self, store: Store) -> None:
        await self._trunk(store)
        engine = Engine(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b1",
                parent="t1",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=1,
                upstream_base_url="http://upstream/v1",
            ),
            FakeUpstream([text_reply("No records found.")]).client(),
        )
        await engine.handle(ask())
        await engine.handle(with_tool_result('{"total": 30}'))
        branch = engine.flush()
        assert branch is not None
        assert branch.kind == "branch"
        assert branch.parent == "t1"
        assert branch.fork_step == 1
        assert len(branch.entries) == 1  # only the divergent step
        assert branch.fault is not None
        assert branch.fault["fault"] == "empty_success"
        assert branch.fault["outcome"]["applied"] is True
        # Resolved, it is a complete two-step run.
        assert len(store.resolve("b1")) == 2

    async def test_recorded_branch_replays_with_zero_live_calls(self, store: Store) -> None:
        # The property that lets anyone reproduce a published number offline.
        await self._trunk(store)
        first = Engine(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b1",
                parent="t1",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=1,
                upstream_base_url="http://upstream/v1",
            ),
            FakeUpstream([text_reply("No records found.")]).client(),
        )
        await first.handle(ask())
        await first.handle(with_tool_result('{"total": 30}'))
        first.flush()

        again = Engine(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b1",
                parent="t1",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=1,
                upstream_base_url="http://upstream/v1",
            ),
            FakeUpstream([]).client(),  # any live call would 500
        )
        await again.handle(ask())
        reply = await again.handle(with_tool_result('{"total": 30}'))
        assert again.live_calls == 0
        assert reply["choices"][0]["message"]["content"] == "No records found."

    async def test_inapplicable_fault_is_recorded_as_not_applied(self, store: Store) -> None:
        await self._trunk(store)
        engine = Engine(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b1",
                parent="t1",
                fault=FaultSpec(fault="partial_list"),  # no list in the payload
                fault_at_step=1,
                upstream_base_url="http://upstream/v1",
            ),
            FakeUpstream([text_reply("total is 30")]).client(),
        )
        await engine.handle(ask())
        await engine.handle(with_tool_result('{"total": 30}'))
        assert engine.injection is not None
        assert engine.injection["applied"] is False
        assert engine.injection["skipped"] == "no list in content"

    async def test_inapplicable_fault_costs_nothing_and_records_nothing(self, store: Store) -> None:
        # Step 0 has no tool result, so the fault cannot apply. Nothing diverged,
        # so the run stays on the trunk: no live call, no branch, no result.
        await self._trunk(store)
        engine = Engine(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b1",
                parent="t1",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=0,
                upstream_base_url="http://upstream/v1",
            ),
            FakeUpstream([]).client(),  # any live call would 500
        )
        await engine.handle(ask())
        await engine.handle(with_tool_result('{"total": 30}'))
        assert engine.injection is not None
        assert engine.injection["applied"] is False
        assert engine.live_calls == 0
        assert engine.recorded_a_divergence is False
        assert engine.flush() is None
        assert "b1" not in store.list_traces()

    async def test_post_fork_requests_are_recorded_even_if_the_trunk_matches(
        self, store: Store
    ) -> None:
        # Regression guard. Once diverged, a request that coincidentally matches a
        # trunk key must still be recorded into the branch, or the branch would
        # replay short and the published number would be irreproducible.
        await self._trunk(store)
        config = ProxyConfig(
            mode=Mode.INJECT,
            store=store,
            trace_id="b1",
            parent="t1",
            fault=FaultSpec(fault="empty_success"),
            fault_at_step=1,
            upstream_base_url="http://upstream/v1",
        )
        engine = Engine(config, FakeUpstream([text_reply("a"), text_reply("b")]).client())
        await engine.handle(ask())
        await engine.handle(with_tool_result('{"total": 30}'))  # diverges here
        await engine.handle(ask())  # identical to trunk step 0
        assert engine.live_calls == 2
        branch = engine.flush()
        assert branch is not None
        assert [e.step for e in branch.entries] == [1, 2]

        # And the branch now replays completely, with nothing live.
        replay = Engine(
            ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="b1"),
        )
        assert len(store.resolve("b1")) == 3
        await replay.handle(ask())
        assert replay.live_calls == 0


class TestFaultPersistence:
    async def test_corruption_persists_on_every_later_request(self, store: Store) -> None:
        # The agent keeps re-sending the tool result it actually received, so a
        # fault applied to one request would evaporate on the next unless it is
        # re-applied. A real broken tool result stays broken for the whole run.
        two_hop = [
            tool_call_reply("find_customer", "{}", "c1"),
            tool_call_reply("list_orders", "{}", "c2"),
            text_reply("done"),
        ]
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream(list(two_hop)).client(),
        )
        first = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
        second = with_tool_result('{"customer_id": 1}', tool="find_customer")
        third = {
            "model": "m",
            "messages": [
                *second["messages"],
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "list_orders", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c2", "content": '{"orders": []}'},
            ],
        }
        for body in (first, second, third):
            await recorder.handle(body)
        recorder.flush()

        upstream = FakeUpstream([text_reply("a"), text_reply("b")])
        engine = Engine(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b1",
                parent="t1",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=1,
                upstream_base_url="http://upstream/v1",
            ),
            upstream.client(),
        )
        await engine.handle(first)
        await engine.handle(second)  # fault applies to the find_customer result
        await engine.handle(third)  # the agent re-sends the original content

        assert len(upstream.seen) == 2
        # Corrupted at the injection step...
        assert upstream.seen[0]["messages"][2]["content"] == ""
        # ...and still corrupted one step later, where the agent sent the original.
        carried = [
            m
            for m in upstream.seen[1]["messages"]
            if m.get("role") == "tool" and m.get("tool_call_id") == "c1"
        ]
        assert carried and carried[0]["content"] == ""
        assert engine.fault_in_final_context is True

    async def test_untouched_tool_results_are_left_alone(self, store: Store) -> None:
        two_hop = [tool_call_reply("find_customer", "{}", "c1"), text_reply("done")]
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream(list(two_hop)).client(),
        )
        first = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
        second = with_tool_result('{"customer_id": 1}', tool="find_customer")
        await recorder.handle(first)
        await recorder.handle(second)
        recorder.flush()

        upstream = FakeUpstream([text_reply("a"), text_reply("b")])
        engine = Engine(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="b1",
                parent="t1",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=1,
                upstream_base_url="http://upstream/v1",
            ),
            upstream.client(),
        )
        await engine.handle(first)
        await engine.handle(second)
        third = {
            "model": "m",
            "messages": [
                *second["messages"],
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c9",
                            "type": "function",
                            "function": {"name": "other", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c9", "content": "untouched payload"},
            ],
        }
        await engine.handle(third)
        later = {
            m.get("tool_call_id"): m["content"]
            for m in upstream.seen[1]["messages"]
            if m.get("role") == "tool"
        }
        assert later["c1"] == ""  # the faulted one stays faulted
        assert later["c9"] == "untouched payload"  # a different tool is not touched


class TestSecretHygiene:
    async def test_no_credential_reaches_a_cassette(self, store: Store) -> None:
        # NFR: no auth header or key is ever written to a cassette.
        secret = "sk-do-not-persist-0123456789"
        engine = Engine(
            ProxyConfig(
                mode=Mode.RECORD,
                store=store,
                trace_id="t1",
                upstream_base_url="http://upstream/v1",
                upstream_api_key=secret,
            ),
            FakeUpstream([text_reply("hi")]).client(),
        )
        await engine.handle(ask())
        engine.flush()
        for path in store.root.rglob("*.json"):
            blob = path.read_text()
            assert secret not in blob
            assert "authorization" not in blob.lower()


class TestHttpSurface:
    def test_replays_over_http(self, store: Store) -> None:
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([text_reply("over http")]).client(),
        )
        import anyio

        anyio.run(recorder.handle, ask())
        recorder.flush()

        app = create_app(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="t1"))
        with TestClient(app) as client:
            reply = client.post("/v1/chat/completions", json=ask())
            assert reply.status_code == 200
            assert reply.json()["choices"][0]["message"]["content"] == "over http"
            stats = client.get("/__misfeed/stats").json()
            assert stats["mode"] == "replay"
            assert stats["live_calls"] == 0
            assert stats["steps"] == 1

    def test_miss_returns_409_with_diagnostics(self, store: Store) -> None:
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([text_reply("recorded")]).client(),
        )
        import anyio

        anyio.run(recorder.handle, ask("recorded"))
        recorder.flush()

        app = create_app(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="t1"))
        with TestClient(app) as client:
            reply = client.post("/v1/chat/completions", json=ask("unrecorded"))
            assert reply.status_code == 409
            assert reply.json()["error"]["type"] == "cassette_miss"

    def test_streaming_is_refused_not_faked(self, store: Store) -> None:
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([text_reply("x")]).client(),
        )
        import anyio

        anyio.run(recorder.handle, ask())
        recorder.flush()

        app = create_app(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="t1"))
        with TestClient(app) as client:
            reply = client.post("/v1/chat/completions", json={**ask(), "stream": True})
            assert reply.status_code == 400
            assert reply.json()["error"]["type"] == "unsupported"

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [("not json", "body is not valid JSON"), ("[1, 2]", "body must be an object")],
    )
    def test_malformed_bodies_are_rejected(self, store: Store, payload: str, expected: str) -> None:
        recorder = Engine(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            FakeUpstream([text_reply("x")]).client(),
        )
        import anyio

        anyio.run(recorder.handle, ask())
        recorder.flush()

        app = create_app(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="t1"))
        with TestClient(app) as client:
            reply = client.post(
                "/v1/chat/completions",
                content=payload,
                headers={"content-type": "application/json"},
            )
            assert reply.status_code == 400
            assert reply.json()["error"]["message"] == expected

    def test_recording_is_flushed_on_shutdown(self, store: Store) -> None:
        upstream = FakeUpstream([text_reply("flushed")])
        app = create_app(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="t1", upstream_base_url="http://upstream/v1"
            ),
            client=upstream.client(),
        )
        with TestClient(app) as client:
            assert client.post("/v1/chat/completions", json=ask()).status_code == 200
        assert "t1" in store.list_traces()
        assert len(store.load_trace("t1").entries) == 1
