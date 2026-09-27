"""Compatibility with LangGraph, over a real socket.

LangGraph is a second, independent client: requests go through `langchain-openai`,
which does its own message serialisation and generates tool schemas from Python
signatures rather than taking hand-written ones. That makes it a real test of the
claim rather than a second pass over the same code path -- and it dispatches tools
onto a thread pool, which is how `evals.world`'s thread-safety requirement was found.

Skipped unless the `compat` dependency group is installed (`uv sync --group compat`),
so LangGraph's dependency tree does not slow down an ordinary `make test`. CI installs
it and runs these.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx2
import pytest
import uvicorn

from evals.stub_model import create_stub_model
from evals.tasks import SYSTEM_PROMPTS, get_task
from evals.world import World
from misfeed.faults import FaultSpec
from misfeed.proxy import Mode, ProxyConfig, create_app
from misfeed.store import Store
from misfeed.verdict import answer_matches, extract_answer

pytest.importorskip("langchain.agents", reason="requires the 'compat' dependency group")

from examples.langgraph_agent import build_agent

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
    port = _free_port()
    app = create_app(config, client_factory=client_factory)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = 15.0
        while not server.started and thread.is_alive() and deadline > 0:
            asyncio.run(asyncio.sleep(0.05))
            deadline -= 0.05
        if not server.started:
            raise RuntimeError("proxy server did not start")
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        thread.join(timeout=15)


def _drive(base_url: str) -> tuple[str, list[str]]:
    """Run the LangGraph agent, returning its final message and the tools it called."""
    world = World()

    async def run() -> tuple[str, list[str]]:
        agent = build_agent(base_url=base_url, model="stub/naive", api_key="unused", world=world)
        state: dict[str, Any] = await agent.ainvoke(
            {
                "messages": [
                    ("system", SYSTEM_PROMPTS["baseline"]),
                    ("user", TASK.question),
                ]
            }
        )
        return str(state["messages"][-1].content), [c.name for c in world.calls]

    try:
        return asyncio.run(run())
    finally:
        world.close()


class TestLangGraphRecordAndReplay:
    def test_records_a_run_and_solves_the_task(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="lg", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            final, called = _drive(base_url)

        assert answer_matches(extract_answer(final), TASK.expected)
        # Two hops through the graph's tool node, in order.
        assert called == ["find_customer", "list_orders"]
        assert len(store.resolve("lg")) >= 2

    def test_replays_with_zero_live_calls(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="lg", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            recorded, _ = _drive(base_url)

        # No upstream at all: anything the cassette cannot serve fails loudly.
        with serving(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="lg"), None) as base_url:
            replayed, _ = _drive(base_url)
        assert replayed == recorded

    def test_tools_run_on_a_thread_pool_without_breaking_the_fixture(self, tmp_path: Path) -> None:
        # LangGraph dispatches tools onto an executor. A default sqlite3 connection
        # refuses use from another thread, which is why World holds a lock and is
        # opened with check_same_thread=False. Pinned so it cannot regress.
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="lg", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            final, called = _drive(base_url)
        assert called, "no tool ran, so the thread-pool path was never exercised"
        assert answer_matches(extract_answer(final), TASK.expected)


class TestLangGraphFaultInjection:
    def test_a_fault_reaches_an_unmodified_langgraph_agent(self, tmp_path: Path) -> None:
        # The whole claim, against a framework that knows nothing about misfeed: its
        # tools returned correct data, and it still answered wrong.
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="trunk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            clean, _ = _drive(base_url)
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
            faulted, called = _drive(base_url)

        branch = store.load_trace("branch")
        assert branch.fault is not None
        assert branch.fault["outcome"]["applied"] is True
        assert branch.fault["outcome"]["tool_name"] == "list_orders"
        # The agent's own tools returned real data throughout.
        assert called == ["find_customer", "list_orders"]
        assert not answer_matches(extract_answer(faulted), TASK.expected)

    def test_an_injected_instruction_reaches_the_graph(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "cassettes")
        with serving(
            ProxyConfig(
                mode=Mode.RECORD, store=store, trace_id="trunk", upstream_base_url="http://stub"
            ),
            _stub_client,
        ) as base_url:
            _drive(base_url)

        fault_step = len(store.resolve("trunk")) - 1
        with serving(
            ProxyConfig(
                mode=Mode.INJECT,
                store=store,
                trace_id="branch",
                parent="trunk",
                fault=FaultSpec(fault="injected_instruction"),
                fault_at_step=fault_step,
                upstream_base_url="http://stub",
            ),
            _stub_client,
        ) as base_url:
            _drive(base_url)

        branch = store.load_trace("branch")
        assert branch.fault is not None
        assert branch.fault["outcome"]["applied"] is True
