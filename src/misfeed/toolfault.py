"""Corrupting a tool result where the agent's own code receives it.

The proxy corrupts the tool-result message on its way to the model, which needs no
cooperation from the agent but has a boundary: the agent's own retry wrapper and
validation code see the genuine payload and are never exercised. See
`docs/adr/001-intercept-at-the-provider-boundary.md`.

This module is the other half. It is a single call inserted at the point the agent
dispatches a tool, so the corrupted value is what the agent's code gets back:

    injector = ToolInjector(ToolFaultPlan(FaultSpec(fault="empty_success")))
    ...
    result = injector.apply(tool_name=name, result=world.call(name, args))

It is opt-in by construction. Nothing here patches, wraps or inspects anything --
corrupting what a function returns to its caller cannot be done from outside that
call without patching, and a harness that monkey-patched a user's tools to measure
their reliability would be a poor advertisement for itself.

It buys one thing the in-flight injector cannot express at all: **transience**. A
fault applied in flight lives in the conversation history and is therefore permanent
by construction. Here a fault can affect one call and let the retry succeed, which is
the only way to find out whether an agent's retry logic actually helps, or it can
persist and answer a different question -- what the agent does when the failure does
not clear.

Corruptions still surface to the model, so the proxy records the divergent
continuation as a branch exactly as it would otherwise: replayability is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from misfeed.faults import FaultSpec, apply_fault

__all__ = ["Corruption", "ToolFaultPlan", "ToolInjector"]


@dataclass(frozen=True, slots=True)
class ToolFaultPlan:
    """Which tool result to corrupt, and how.

    `tool_name` of None matches any tool. `occurrence` is which matching call to hit,
    counted from zero, so a two-hop agent can have its second lookup broken and its
    first left alone.

    `persist` decides what a retry sees. False -- the default -- corrupts only the
    chosen call, so a retry receives real data and the run answers "does the retry
    help?". True corrupts that call and every matching one after it, answering "what
    happens when the failure does not clear?". The default is the transient case
    because it is the one the in-flight injector cannot model.
    """

    fault: FaultSpec
    tool_name: str | None = None
    occurrence: int = 0
    persist: bool = False

    def __post_init__(self) -> None:
        if self.occurrence < 0:
            raise ValueError(f"occurrence must be >= 0, got {self.occurrence}")

    def to_json(self) -> dict[str, Any]:
        return {
            **self.fault.to_json(),
            "tool_name": self.tool_name,
            "occurrence": self.occurrence,
            "persist": self.persist,
            "injected_at": "tool",
        }


@dataclass(frozen=True, slots=True)
class Corruption:
    """One tool result that was actually changed."""

    call_index: int
    tool_name: str
    detail: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "tool_name": self.tool_name,
            **self.detail,
        }


class ToolInjector:
    """Applies a `ToolFaultPlan` to tool results as they are produced.

    Stateful and single-run, like the proxy's engine: it counts matching calls to
    decide which to corrupt. Reuse across runs would carry the count over, so a run
    gets its own injector.
    """

    def __init__(self, plan: ToolFaultPlan | None = None) -> None:
        self.plan = plan
        self.calls = 0
        self.matches = 0
        self.corruptions: list[Corruption] = []
        self.skipped: list[dict[str, Any]] = []

    @property
    def applied(self) -> bool:
        """Whether any tool result was actually changed."""
        return bool(self.corruptions)

    @property
    def outcome(self) -> dict[str, Any] | None:
        """What happened, for the report. None when there was no plan."""
        if self.plan is None:
            return None
        return {
            **self.plan.to_json(),
            "applied": self.applied,
            "tool_calls_seen": self.calls,
            "matching_calls": self.matches,
            "corruptions": [c.to_json() for c in self.corruptions],
            "skipped": self.skipped,
        }

    def _wanted(self, index_among_matches: int) -> bool:
        assert self.plan is not None
        if index_among_matches == self.plan.occurrence:
            return True
        return self.plan.persist and index_among_matches > self.plan.occurrence

    def apply(self, *, tool_name: str, result: str) -> str:
        """Return the value the agent's code should receive for this tool call.

        Pass-through when there is no plan, when this call does not match it, or when
        the fault cannot apply to this payload -- and an inapplicable fault is recorded
        in `skipped` rather than raised, so it shows up in the report instead of
        breaking the run.
        """
        self.calls += 1
        plan = self.plan
        if plan is None:
            return result
        if plan.tool_name is not None and plan.tool_name != tool_name:
            return result

        index_among_matches = self.matches
        self.matches += 1
        if not self._wanted(index_among_matches):
            return result

        outcome = apply_fault(plan.fault.fault, result, plan.fault.params)
        if not outcome.changed:
            self.skipped.append(
                {
                    "call_index": self.calls - 1,
                    "tool_name": tool_name,
                    "fault": plan.fault.fault,
                    **outcome.detail,
                }
            )
            return result

        self.corruptions.append(
            Corruption(
                call_index=self.calls - 1,
                tool_name=tool_name,
                detail={"fault": plan.fault.fault, **outcome.detail},
            )
        )
        return outcome.content
