"""Why two injectors exist, demonstrated end to end.

The subject is an agent with the retry wrapper most production agents actually have:
it validates each tool result and calls the tool again if the result is unusable.

Three runs of the same agent against the same fault, differing only in where the fault
is applied and whether it clears:

1.  in flight -- the wrapper never fires, because the proxy corrupts the message on its
    way to the model and the agent's code saw real data. The agent answers wrong.
2.  at the tool, transient -- the wrapper fires, the retry gets real data, and the
    agent answers correctly.
3.  at the tool, persistent -- the wrapper fires, the retry is corrupted too, and the
    agent answers wrong.

Run 1 is the boundary of the wire-level injector, stated in the README as a limitation.
Runs 2 and 3 are what the tool-side injector adds, and the difference between them is
the question "does this agent's retry actually help?", which the wire-level injector
cannot ask at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx2
import pytest

from evals.agent import run_retrying_agent
from evals.stub_model import create_stub_model
from evals.tasks import SYSTEM_PROMPTS, get_task
from evals.world import World
from misfeed.faults import FaultSpec
from misfeed.proxy import Mode, ProxyConfig, create_app
from misfeed.store import Store
from misfeed.toolfault import ToolFaultPlan, ToolInjector
from misfeed.verdict import answer_matches, extract_answer

TASK = get_task("T1_shipped_total")


def _stub_client() -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=create_stub_model("naive")),
        base_url="http://stub",
    )


async def _run(
    store: Store,
    trace_id: str,
    *,
    parent: str | None = None,
    wire_fault: str | None = None,
    wire_step: int | None = None,
    injector: ToolInjector | None = None,
) -> tuple[str, list[dict[str, object]]]:
    config = ProxyConfig(
        mode=Mode.INJECT if wire_fault else Mode.RECORD,
        store=store,
        trace_id=trace_id,
        parent=parent,
        fault=FaultSpec(fault=wire_fault) if wire_fault else None,
        fault_at_step=wire_step,
        upstream_base_url="http://stub",
    )
    app = create_app(config, client_factory=_stub_client)
    world = World()
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://proxy/v1"
        ) as client:
            run = await run_retrying_agent(
                client=client,
                model="stub/naive",
                task=TASK,
                system_prompt=SYSTEM_PROMPTS["baseline"],
                world=world,
                injector=injector,
            )
        return run.final_message, run.tool_calls
    finally:
        app.state.engine.flush()
        world.close()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "cassettes")


def _clean(store: Store) -> str:
    final, calls = asyncio.run(_run(store, "trunk"))
    assert answer_matches(extract_answer(final), TASK.expected), "clean run must pass first"
    assert all(call.get("retries") == 0 for call in calls), "nothing should retry when clean"
    return final


class TestWhereTheFaultLands:
    def test_a_wire_fault_never_reaches_the_retry_wrapper(self, store: Store) -> None:
        _clean(store)
        step = len(store.resolve("trunk")) - 1
        final, calls = asyncio.run(
            _run(store, "wire", parent="trunk", wire_fault="empty_success", wire_step=step)
        )
        # The wrapper saw genuine data, so it never fired...
        assert all(call.get("retries") == 0 for call in calls)
        # ...and the model, which saw an empty result, answered wrong anyway.
        assert not answer_matches(extract_answer(final), TASK.expected)

    def test_a_transient_tool_fault_is_recovered_by_the_retry(self, store: Store) -> None:
        _clean(store)
        injector = ToolInjector(
            ToolFaultPlan(fault=FaultSpec(fault="empty_success"), tool_name="list_orders")
        )
        final, calls = asyncio.run(_run(store, "tool-transient", injector=injector))

        assert injector.applied is True
        assert len(injector.corruptions) == 1
        retried = [call for call in calls if call.get("retries")]
        assert [call["name"] for call in retried] == ["list_orders"]
        # The retry got real data, so the answer is right despite the fault.
        assert answer_matches(extract_answer(final), TASK.expected)

    def test_a_persistent_tool_fault_defeats_the_retry(self, store: Store) -> None:
        _clean(store)
        injector = ToolInjector(
            ToolFaultPlan(
                fault=FaultSpec(fault="empty_success"),
                tool_name="list_orders",
                persist=True,
            )
        )
        final, calls = asyncio.run(_run(store, "tool-persistent", injector=injector))

        # The wrapper fired and the retry was corrupted too.
        assert len(injector.corruptions) == 2
        assert any(call.get("retries") == 1 for call in calls)
        assert not answer_matches(extract_answer(final), TASK.expected)

    def test_the_three_runs_disagree_which_is_the_whole_point(self, store: Store) -> None:
        # One assertion for the claim: where a fault is applied, and whether it clears,
        # changes the answer. A harness with only one injection point cannot see this.
        _clean(store)
        step = len(store.resolve("trunk")) - 1
        wire, _ = asyncio.run(
            _run(store, "wire", parent="trunk", wire_fault="empty_success", wire_step=step)
        )
        transient, _ = asyncio.run(
            _run(
                store,
                "tool-transient",
                injector=ToolInjector(
                    ToolFaultPlan(fault=FaultSpec(fault="empty_success"), tool_name="list_orders")
                ),
            )
        )
        persistent, _ = asyncio.run(
            _run(
                store,
                "tool-persistent",
                injector=ToolInjector(
                    ToolFaultPlan(
                        fault=FaultSpec(fault="empty_success"),
                        tool_name="list_orders",
                        persist=True,
                    )
                ),
            )
        )
        correct = TASK.expected
        assert not answer_matches(extract_answer(wire), correct)
        assert answer_matches(extract_answer(transient), correct)
        assert not answer_matches(extract_answer(persistent), correct)


class TestReplayability:
    def test_a_tool_side_experiment_still_replays_offline(self, store: Store) -> None:
        # The corruption reaches the model, so the proxy records the divergence as
        # usual. Tool-side injection must not cost replayability.
        _clean(store)
        injector = ToolInjector(
            ToolFaultPlan(fault=FaultSpec(fault="empty_success"), tool_name="list_orders")
        )
        recorded, _ = asyncio.run(_run(store, "tool-transient", injector=injector))

        app = create_app(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id="tool-transient"))

        async def replay() -> str:
            world = World()
            replay_injector = ToolInjector(
                ToolFaultPlan(fault=FaultSpec(fault="empty_success"), tool_name="list_orders")
            )
            try:
                async with httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app), base_url="http://proxy/v1"
                ) as client:
                    run = await run_retrying_agent(
                        client=client,
                        model="stub/naive",
                        task=TASK,
                        system_prompt=SYSTEM_PROMPTS["baseline"],
                        world=world,
                        injector=replay_injector,
                    )
                return run.final_message
            finally:
                world.close()

        assert asyncio.run(replay()) == recorded
        assert app.state.engine.live_calls == 0
