"""A tool-calling agent written against the official `openai` SDK.

This exists to demonstrate, rather than assert, that misfeed needs no cooperation
from the agent under test. The only misfeed-specific line is the `base_url`. Nothing
is wrapped, patched or subclassed; the SDK does not know the proxy exists.

Run it against a real endpoint:

    python -m examples.openai_sdk_agent --base-url https://host/v1 --model <id>

Or against a misfeed proxy, which is the point:

    misfeed --store cassettes serve --mode record --trace mytrunk \\
            --upstream https://host/v1
    python -m examples.openai_sdk_agent \\
            --base-url http://127.0.0.1:8756/v1 --model <id>

`tests/test_compat_openai_sdk.py` drives this module over a real socket to record
and then replay a run, so the compatibility claim is covered by CI rather than by
this docstring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from evals.tasks import SYSTEM_PROMPTS, get_task
from evals.world import World

__all__ = ["SdkAgentResult", "run_with_sdk"]


@dataclass(slots=True)
class SdkAgentResult:
    final_message: str
    steps: int
    tool_calls: list[str] = field(default_factory=list)
    looped: bool = False


async def run_with_sdk(
    *,
    client: AsyncOpenAI,
    model: str,
    question: str,
    tools: list[dict[str, Any]],
    world: World,
    system_prompt: str,
    max_steps: int = 6,
    stream: bool = False,
) -> SdkAgentResult:
    """A plain tool-calling loop, using the SDK's own request and response types.

    With `stream=True` it uses the SDK's streaming helper and its own accumulator, so
    the reassembled message comes from the SDK rather than from anything in this
    repository. That is what makes it worth testing against a misfeed-synthesised
    stream: the same cassette serves both paths, because `stream` is excluded from the
    request key.
    """
    messages: list[Any] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    called: list[str] = []

    for step in range(max_steps):
        # The streaming and non-streaming helpers return differently-parameterised
        # message types that share this shape, so the loop works against the shape.
        message: Any
        if stream:
            async with client.chat.completions.stream(
                model=model,
                messages=messages,
                tools=tools,  # type: ignore[arg-type]
                temperature=0.0,
            ) as streamed:
                message = (await streamed.get_final_completion()).choices[0].message
        else:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,  # type: ignore[arg-type]
                temperature=0.0,
            )
            message = completion.choices[0].message
        # model_dump drops None fields the API would reject on the way back in.
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return SdkAgentResult(
                final_message=message.content or "", steps=step + 1, tool_calls=called
            )

        for call in message.tool_calls:
            if call.type != "function":
                # Newer SDKs union function calls with custom tool calls. This loop
                # only speaks the function protocol; anything else is reported back
                # to the model rather than silently dropped.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"error": f"unsupported tool type {call.type!r}"}),
                    }
                )
                continue
            function = call.function
            try:
                arguments = json.loads(function.arguments or "{}")
            except json.JSONDecodeError:
                result = json.dumps({"error": "arguments were not valid JSON"})
            else:
                result = world.call(function.name, arguments)
            called.append(function.name)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return SdkAgentResult(final_message="", steps=max_steps, tool_calls=called, looped=True)


async def _main(args: argparse.Namespace) -> int:
    task = get_task(args.task)
    world = World()
    client = AsyncOpenAI(
        base_url=args.base_url,
        # A replaying proxy ignores this, but the SDK insists on something.
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or "unused",
    )
    try:
        result = await run_with_sdk(
            client=client,
            model=args.model,
            question=task.question,
            tools=task.tools,
            world=world,
            system_prompt=SYSTEM_PROMPTS[args.variant],
            stream=args.stream,
        )
    finally:
        await client.close()
        world.close()

    print(f"task     : {task.id}")
    print(f"expected : {task.expected}")
    print(f"tools    : {' -> '.join(result.tool_calls) or '(none)'}")
    print(f"steps    : {result.steps}{' (hit ceiling)' if result.looped else ''}")
    print(f"transport: {'streaming' if args.stream else 'non-streaming'}")
    print(f"\n{result.final_message}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="stub/naive")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--task", default="T1_shipped_total")
    parser.add_argument("--variant", default="baseline", choices=sorted(SYSTEM_PROMPTS))
    parser.add_argument(
        "--stream",
        action="store_true",
        help="use the SDK's streaming helper; misfeed synthesises the framing",
    )
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
