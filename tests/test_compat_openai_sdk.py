"""Compatibility with the official `openai` SDK, over a real socket.

The README claims misfeed needs no cooperation from the agent under test. This is
where that claim is earned rather than asserted: an unmodified `openai.AsyncOpenAI`
client, talking HTTP to a uvicorn-served misfeed proxy, records a run and then
replays it with zero live calls.

It is the only test that exercises the real server path -- uvicorn, sockets, and the
app's lifespan (which is what flushes a recording on shutdown). Everything else drives
the ASGI app in-process, which skips all three.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx2
import pytest
import uvicorn
from openai import AsyncOpenAI, ConflictError

from evals.agent import run_agent
from evals.stub_model import create_stub_model
from evals.tasks import SYSTEM_PROMPTS, get_task
from evals.world import World
from examples.openai_sdk_agent import run_with_sdk
from misfeed.faults import FaultSpec
from misfeed.proxy import Mode, ProxyConfig, create_app
from misfeed.store import Store
from misfeed.verdict import answer_matches, extract_answer

TASK = get_task("T1_shipped_total")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def _stub_client() -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=create_stub_model("naive")),
        base_url="http://stub",
    )


@contextmanager
def serving(
    config: ProxyConfig, client_factory: Callable[[], httpx2.AsyncClient] | None
) -> Iterator[str]:
    """Serve the proxy with uvicorn on a free port, and shut it down cleanly."""
    port = _free_port()
    app = create_app(config, client_factory=client_factory)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = 10.0
        while not server.started and thread.is_alive() and deadline > 0:
            asyncio.run(asyncio.sleep(0.05))
            deadline -= 0.05
        if not server.started:
            raise RuntimeError("proxy server did not start")
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "proxy server did not shut down"


async def _drive(base_url: str, *, stream: bool = False) -> str:
    """Run the SDK agent against `base_url`, returning its final message."""
    client = AsyncOpenAI(base_url=base_url, api_key="unused", max_retries=0)
    world = World()
    try:
        result = await run_with_sdk(
            client=client,
            model="stub/naive",
            question=TASK.question,
            tools=TASK.tools,
            world=world,
            system_prompt=SYSTEM_PROMPTS["baseline"],
            stream=stream,
        )
        return result.final_message
    finally:
        await client.close()
        world.close()


class TestOfficialSdkRecordAndReplay:
    def test_records_then_replays_with_zero_live_calls(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "cassettes")

        record_config = ProxyConfig(
            mode=Mode.RECORD,
            store=store,
            trace_id="sdk",
            upstream_base_url="http://stub",
        )
        with serving(record_config, _stub_client) as base_url:
            recorded = asyncio.run(_drive(base_url))
        # The lifespan flushed on shutdown; nothing else wrote this trace.
        assert "sdk" in store.list_traces()
        entries = store.resolve("sdk")
        assert len(entries) >= 2  # at least one tool hop, then an answer

        # The SDK solved the task, so the recording is a usable baseline.
        assert answer_matches(extract_answer(recorded), TASK.expected)

        # Now replay it. No upstream is configured at all, so any request the
        # cassette cannot serve fails rather than quietly going live.
        replay_config = ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="sdk")
        with serving(replay_config, None) as base_url:
            replayed = asyncio.run(_drive(base_url))
        assert replayed == recorded

    def test_sdk_request_shape_is_stable_across_separate_replays(self, tmp_path: Path) -> None:
        # The canonical key has to survive a round trip through the SDK's own request
        # construction, or replaying a real client would miss on every step. Two
        # independent replays of the same cassette prove it does.
        #
        # Each replay needs its own proxy instance. A single instance holds one engine
        # and one cassette player, which consumes a per-key queue in recorded order,
        # so a second run through the same instance exhausts it and is reported as
        # divergence. That is strict replay behaving correctly, and it is why
        # A proxy instance serves one run, by design.
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="sdk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            recorded = asyncio.run(_drive(base_url))

        replays = []
        for _ in range(2):
            config = ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="sdk")
            with serving(config, None) as base_url:
                replays.append(asyncio.run(_drive(base_url)))
        assert replays[0] == replays[1] == recorded

    def test_a_second_run_through_one_instance_is_reported_as_divergence(
        self, tmp_path: Path
    ) -> None:
        # The flip side, pinned so it cannot regress into a silent free repeat.
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="sdk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            asyncio.run(_drive(base_url))

        config = ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="sdk")
        with serving(config, None) as base_url:
            asyncio.run(_drive(base_url))
            with pytest.raises(ConflictError) as conflict:
                asyncio.run(_drive(base_url))
        assert conflict.value.status_code == 409
        body = conflict.value.response.json()
        assert body["error"]["type"] == "cassette_miss"


class TestOfficialSdkFaultInjection:
    def test_a_fault_reaches_an_unmodified_sdk_client(self, tmp_path: Path) -> None:
        # The claim in one test: an agent that knows nothing about misfeed is shown a
        # corrupted tool result and gets it wrong, while its own tools returned fine.
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="trunk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            clean = asyncio.run(_drive(base_url))
        assert answer_matches(extract_answer(clean), TASK.expected)

        fault_step = len(store.resolve("trunk")) - 1  # the request carrying the last tool result
        with serving(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="branch",
                parent="trunk",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=fault_step,
                upstream_base_url="http://stub",
            ),
            _stub_client,
        ) as base_url:
            faulted = asyncio.run(_drive(base_url))

        branch = store.load_trace("branch")
        assert branch.fault is not None
        assert branch.fault["outcome"]["applied"] is True
        # Same agent, same tools, different answer -- and it never mentioned a problem.
        assert faulted != clean
        assert not answer_matches(extract_answer(faulted), TASK.expected)


class TestCassettePortability:
    """A cassette recorded by one client must replay under another.

    This failed until explicit nulls were normalised away. This repository's own agent
    loop appends a response message verbatim, so it sends `"content": null` on an
    assistant tool-call message; the official SDK's `model_dump(exclude_none=True)`
    omits the field. Two spellings of the same request hashed differently, which made
    every cassette client-specific without anything saying so.

    A unit test would not have found it. Only driving a second, independently written
    client did.
    """

    def test_sdk_replays_a_cassette_recorded_by_the_projects_own_agent(
        self, tmp_path: Path
    ) -> None:
        store = Store(tmp_path / "cassettes")

        # Record with evals.agent, in-process, no sockets.
        own_config = ProxyConfig(
            mode=Mode.RECORD, store=store, trace_id="own", upstream_base_url="http://stub"
        )
        app = create_app(own_config, client_factory=_stub_client)

        async def record_with_own_agent() -> str:
            world = World()
            try:
                async with httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app), base_url="http://proxy/v1"
                ) as client:
                    run = await run_agent(
                        client=client,
                        model="stub/naive",
                        task=TASK,
                        system_prompt=SYSTEM_PROMPTS["baseline"],
                        world=world,
                    )
                return run.final_message
            finally:
                app.state.engine.flush()
                world.close()

        recorded = asyncio.run(record_with_own_agent())
        assert answer_matches(extract_answer(recorded), TASK.expected)

        # Replay the very same cassette with the official SDK over a socket.
        replay_config = ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="own")
        with serving(replay_config, None) as base_url:
            replayed = asyncio.run(_drive(base_url))
        assert replayed == recorded

    def test_own_agent_replays_a_cassette_recorded_by_the_sdk(self, tmp_path: Path) -> None:
        # And the other direction, so portability is not one-way.
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="sdk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            recorded = asyncio.run(_drive(base_url))

        app = create_app(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="sdk"))

        async def replay_with_own_agent() -> str:
            world = World()
            try:
                async with httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app), base_url="http://proxy/v1"
                ) as client:
                    run = await run_agent(
                        client=client,
                        model="stub/naive",
                        task=TASK,
                        system_prompt=SYSTEM_PROMPTS["baseline"],
                        world=world,
                    )
                return run.final_message
            finally:
                world.close()

        assert asyncio.run(replay_with_own_agent()) == recorded


class TestSynthesisedStreaming:
    """The SDK's own accumulator must reassemble a synthesised stream correctly.

    The framing misfeed emits is invented -- a recorded completion has no token
    boundaries. What must be true is that a real client reassembling it gets exactly
    the recorded message, including tool calls, and that the *same* cassette serves a
    streaming and a non-streaming client, since `stream` is excluded from the request
    key.
    """

    def test_a_streaming_client_replays_a_non_streaming_recording(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="sdk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            recorded = asyncio.run(_drive(base_url, stream=False))
        assert answer_matches(extract_answer(recorded), TASK.expected)

        # Same cassette, streaming client. Every step of the tool-calling loop is
        # reassembled by the SDK from synthesised SSE.
        with serving(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="sdk"), None) as base_url:
            streamed = asyncio.run(_drive(base_url, stream=True))
        assert streamed == recorded

    def test_a_non_streaming_client_replays_a_streaming_recording(self, tmp_path: Path) -> None:
        # The other direction, so the cassette is genuinely transport-agnostic.
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="sdk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            recorded = asyncio.run(_drive(base_url, stream=True))

        with serving(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="sdk"), None) as base_url:
            plain = asyncio.run(_drive(base_url, stream=False))
        assert plain == recorded

    def test_a_fault_reaches_a_streaming_client(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="trunk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            clean = asyncio.run(_drive(base_url, stream=True))
        assert answer_matches(extract_answer(clean), TASK.expected)

        fault_step = len(store.resolve("trunk")) - 1
        with serving(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="branch",
                parent="trunk",
                fault=FaultSpec(fault="empty_success"),
                fault_at_step=fault_step,
                upstream_base_url="http://stub",
            ),
            _stub_client,
        ) as base_url:
            faulted = asyncio.run(_drive(base_url, stream=True))

        branch = store.load_trace("branch")
        assert branch.fault is not None
        assert branch.fault["outcome"]["applied"] is True
        assert not answer_matches(extract_answer(faulted), TASK.expected)

    def test_chunked_content_still_reassembles(self, tmp_path: Path) -> None:
        # Forcing many small deltas exercises the client's incremental accumulation
        # path. The boundaries are invented; the reassembled message must not be.
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="sdk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            recorded = asyncio.run(_drive(base_url, stream=False))

        with serving(
            ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="sdk", stream_chunk_chars=3),
            None,
        ) as base_url:
            streamed = asyncio.run(_drive(base_url, stream=True))
        assert streamed == recorded
