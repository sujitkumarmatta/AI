"""A minimal tool-calling agent loop.

Deliberately plain: one model call, execute any tool calls, append the results,
repeat. No framework, no retries, no validation of tool output. That is the point
-- it is the shape almost every agent has underneath, so what the harness measures
is the model's reaction to a degraded tool result rather than a particular
framework's error handling.

The loop talks to an injected client, so the same code runs against the proxy
in-process over an ASGI transport or against a real server over a socket.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx2

from evals.tasks import Task
from evals.world import World

__all__ = ["AgentRun", "run_agent"]


@dataclass(slots=True)
class AgentRun:
    """What the agent produced, and how."""

    final_message: str
    steps: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    looped: bool = False
    crashed: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "final_message": self.final_message,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "looped": self.looped,
            "crashed": self.crashed,
        }


async def run_agent(
    *,
    client: httpx2.AsyncClient,
    model: str,
    task: Task,
    system_prompt: str,
    world: World,
    max_steps: int = 6,
    temperature: float = 0.0,
) -> AgentRun:
    """Run `task` to completion, or until `max_steps` model calls are used.

    A transport or protocol error is raised rather than recorded as an agent
    crash: a 409 cassette miss or a 429 budget stop is the harness failing, not
    the agent, and silently scoring it would corrupt the results.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.question},
    ]
    calls: list[dict[str, Any]] = []

    for step in range(max_steps):
        reply = await client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "tools": task.tools,
                "temperature": temperature,
            },
        )
        reply.raise_for_status()
        message = reply.json()["choices"][0]["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return AgentRun(
                final_message=message.get("content") or "", steps=step + 1, tool_calls=calls
            )

        for call in tool_calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                # A model that emits unparseable arguments is the agent's problem
                # to survive, so this is reported to it as a tool result.
                arguments = {}
                result = json.dumps({"error": "arguments were not valid JSON"})
            else:
                result = world.call(name, arguments)
            calls.append({"step": step, "name": name, "arguments": arguments})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    return AgentRun(
        final_message=messages[-1].get("content") or "",
        steps=max_steps,
        tool_calls=calls,
        looped=True,
    )
