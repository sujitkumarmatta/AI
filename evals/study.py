"""The study driver: record a clean run, then inject faults and score the results.

For each (prompt variant, task) it records one clean trunk, then forks a branch
per (fault, injection step). Everything is wired in-process over ASGI transports,
so a study against the stub model needs no ports, no keys and no network, and is
deterministic.

The clean run comes first and gates the rest. If the agent cannot answer a task
with nothing corrupted, its behaviour under a fault is uninterpretable, and the
report withholds that cell instead of averaging it in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2

from evals.agent import AgentRun, run_agent
from evals.stub_model import POLICIES, create_stub_model
from evals.tasks import SYSTEM_PROMPTS, TASKS, Task
from evals.world import World
from misfeed.faults import FaultSpec
from misfeed.proxy import Engine, Mode, ProxyConfig, create_app
from misfeed.report import RunResult, markdown_report, summarise
from misfeed.store import Store
from misfeed.verdict import RunFacts, answer_matches, classify, extract_answer, surfaced_problem

__all__ = ["StudyConfig", "run_study"]

# MVP fault set. The taxonomy has nine classes; the study starts with the three
# that need no assumptions about payload shape, so that it works on any task.
DEFAULT_FAULTS: tuple[str, ...] = ("empty_success", "partial_list", "missing_fields")


@dataclass(slots=True)
class StudyConfig:
    store_root: Path
    report_path: Path
    faults: tuple[str, ...] = DEFAULT_FAULTS
    max_steps: int = 6
    max_injection_step: int = 4
    max_live_requests: int = 200

    # Against the scripted stub the agent dimension is its policy, because the
    # stub does not read the system prompt. Comparing prompt variants against it
    # would produce identical rows that look like a finding about prompting.
    stub_policies: tuple[str, ...] = ("naive", "careful")
    # Against a real model the agent is fixed and the prompt is what varies.
    variants: tuple[str, ...] = ("baseline",)

    upstream_base_url: str | None = None
    upstream_api_key: str | None = None
    model: str | None = None

    @property
    def synthetic(self) -> bool:
        """True while the upstream is a stand-in rather than a model."""
        return self.upstream_base_url is None

    @property
    def agents(self) -> tuple[tuple[str, str | None], ...]:
        """(model name, stub policy) pairs to run."""
        if not self.synthetic:
            return ((self.model or "unknown", None),)
        return tuple((f"stub/{policy}", policy) for policy in self.stub_policies)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)


def _facts_from(
    run: AgentRun, task: Task, *, fault_applied: bool, fault_reached: bool
) -> tuple[RunFacts, str | None, list[str]]:
    answer = extract_answer(run.final_message)
    phrases = surfaced_problem(run.final_message)
    facts = RunFacts(
        fault_applied=fault_applied,
        fault_in_final_context=fault_reached,
        answered=answer is not None,
        correct=answer_matches(answer, task.expected),
        surfaced=bool(phrases),
        crashed=run.crashed is not None,
        looped=run.looped,
    )
    return facts, answer, phrases


def _tokens(engine: Engine) -> int:
    return sum(
        int(o.usage.get("total_tokens", 0))
        for o in engine.observations
        if isinstance(o.usage, dict)
    )


class Study:
    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self.store = Store(config.store_root)
        self.results: list[RunResult] = []

    def _upstream(self, policy: str | None) -> tuple[Callable[[], httpx2.AsyncClient] | None, str]:
        """A factory for the upstream client, and the base url to record against.

        A factory rather than a client, so that a cell already recorded never
        constructs an upstream at all.
        """
        config = self.config
        if policy is None:
            assert config.upstream_base_url is not None
            return None, config.upstream_base_url

        def build() -> httpx2.AsyncClient:
            return httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=create_stub_model(policy)),
                base_url="http://stub",
            )

        return build, "http://stub"

    async def _drive(
        self,
        *,
        task: Task,
        variant: str,
        model: str,
        policy: str | None,
        trace_id: str,
        parent: str | None = None,
        fault: str | None = None,
        fault_step: int | None = None,
    ) -> tuple[AgentRun, Engine]:
        factory, base_url = self._upstream(policy)
        config = self.config
        meta: dict[str, Any] = {"task": task.id, "variant": variant, "model": model}
        if fault is not None:
            meta |= {"fault": fault, "step": fault_step}
        proxy_config = ProxyConfig(
            mode=Mode.INJECT if fault is not None else Mode.RECORD,
            store=self.store,
            trace_id=trace_id,
            parent=parent,
            fault=FaultSpec(fault=fault) if fault is not None else None,
            fault_at_step=fault_step,
            upstream_base_url=base_url,
            upstream_api_key=config.upstream_api_key,
            max_live_requests=config.max_live_requests,
            meta=meta,
        )
        app = create_app(proxy_config, client_factory=factory)
        engine: Engine = app.state.engine
        world = World()
        try:
            async with httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://proxy/v1"
            ) as agent_client:
                run = await run_agent(
                    client=agent_client,
                    model=model,
                    task=task,
                    system_prompt=SYSTEM_PROMPTS[variant],
                    world=world,
                    max_steps=config.max_steps,
                )
        finally:
            engine.flush()
            world.close()
            if engine.client is not None:
                await engine.client.aclose()
        return run, engine

    async def run(self) -> dict[str, Any]:
        config = self.config
        for model, policy in config.agents:
            for variant in config.variants:
                for task in TASKS:
                    await self._cell(model, policy, variant, task)

        summary = summarise(
            self.results,
            synthetic=config.synthetic,
            metadata={
                "models": [model for model, _ in config.agents],
                "faults": list(config.faults),
                "variants": list(config.variants),
                "tasks": [t.id for t in TASKS],
                "store": str(config.store_root),
                "note": (
                    "Upstream is a scripted stand-in. The two stub policies differ "
                    "by construction: 'naive' uses tool output without checking it, "
                    "'careful' validates it first. This shows the harness can tell "
                    "them apart. It says nothing about any real model."
                    if config.synthetic
                    else f"Recorded against {config.upstream_base_url}."
                ),
            },
        )
        config.report_path.parent.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        config.report_path.with_suffix(".md").write_text(markdown_report(summary), encoding="utf-8")
        return summary

    async def _cell(self, model: str, policy: str | None, variant: str, task: Task) -> None:
        """One (agent, prompt, task) cell: a clean run, then its fault runs."""
        config = self.config
        stem = f"{_slug(model)}--{_slug(variant)}--{_slug(task.id)}"
        trunk_id = f"trunk--{stem}"
        run, engine = await self._drive(
            task=task, variant=variant, model=model, policy=policy, trace_id=trunk_id
        )
        facts, answer, phrases = _facts_from(run, task, fault_applied=False, fault_reached=False)
        self.results.append(
            RunResult(
                task=task.id,
                model=model,
                variant=variant,
                trace_id=trunk_id,
                steps=run.steps,
                facts=facts,
                answer=answer,
                expected=task.expected,
                surfaced_phrases=phrases,
                total_tokens=_tokens(engine),
                live_calls=engine.live_calls,
            )
        )
        if not facts.correct:
            # Precondition failed. The report withholds this cell, and spending
            # budget on its fault runs would be waste.
            return

        recorded = len(self.store.resolve(trunk_id))
        for fault in config.faults:
            for step in range(1, min(recorded, config.max_injection_step + 1)):
                await self._experiment(model, policy, variant, task, trunk_id, stem, fault, step)

    async def _experiment(
        self,
        model: str,
        policy: str | None,
        variant: str,
        task: Task,
        trunk_id: str,
        stem: str,
        fault: str,
        step: int,
    ) -> None:
        branch_id = f"branch--{stem}--s{step}--{_slug(fault)}"
        run, engine = await self._drive(
            task=task,
            variant=variant,
            model=model,
            policy=policy,
            trace_id=branch_id,
            parent=trunk_id,
            fault=fault,
            fault_step=step,
        )
        common = {
            "task": task.id,
            "model": model,
            "variant": variant,
            "trace_id": branch_id,
            "steps": run.steps,
            "fault": fault,
            "fault_step": step,
            "expected": task.expected,
            "total_tokens": _tokens(engine),
            "live_calls": engine.live_calls,
        }
        if not (engine.injection and engine.injection.get("applied")):
            reason = (engine.injection or {}).get("skipped", "fault did not apply")
            self.results.append(RunResult(**common, skipped=str(reason)))
            return

        facts, answer, phrases = _facts_from(
            run, task, fault_applied=True, fault_reached=engine.fault_in_final_context
        )
        self.results.append(
            RunResult(
                **common,
                outcome=classify(facts),
                facts=facts,
                answer=answer,
                surfaced_phrases=phrases,
            )
        )


async def run_study(config: StudyConfig) -> dict[str, Any]:
    return await Study(config).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("evals/cassettes"))
    parser.add_argument("--report", type=Path, default=Path("evals/report.json"))
    parser.add_argument(
        "--stub-policy",
        choices=POLICIES,
        action="append",
        dest="stub_policies",
        help="Stub policy to run; repeatable. Default runs naive and careful.",
    )
    parser.add_argument(
        "--upstream",
        default=None,
        help="OpenAI-compatible base URL. Omitted means use the scripted stub.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--variant",
        choices=sorted(SYSTEM_PROMPTS),
        action="append",
        dest="variants",
        help="System-prompt variant; repeatable. Only informative against a real model.",
    )
    parser.add_argument("--max-live-requests", type=int, default=200)
    args = parser.parse_args(argv)

    config = StudyConfig(
        store_root=args.store,
        report_path=args.report,
        stub_policies=tuple(args.stub_policies or POLICIES),
        variants=tuple(args.variants or ("baseline",)),
        upstream_base_url=args.upstream,
        upstream_api_key=args.api_key,
        model=args.model,
        max_live_requests=args.max_live_requests,
    )
    summary = asyncio.run(run_study(config))
    print(markdown_report(summary))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
