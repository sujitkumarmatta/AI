"""A LangGraph agent pointed at misfeed, to show the harness needs no cooperation.

LangGraph is a genuinely different client from the official SDK: requests go through
`langchain-openai`, which does its own message serialisation and its own tool-schema
generation. So this is a real portability test rather than a second look at the same
code path -- and it runs its tools in a thread pool, which is why `evals.world` is
locked and opened with `check_same_thread=False`.

The only misfeed-specific line is `base_url`. No wrapper, no callback handler, no
patching.

    misfeed --store cassettes serve --mode record --trace lgtrunk \\
            --upstream https://host/v1
    python -m examples.langgraph_agent --base-url http://127.0.0.1:8756/v1 --model <id>

One caveat worth understanding: a cassette is keyed on the whole logical request,
which includes the tool definitions. LangChain generates tool schemas from Python
signatures and docstrings, so they differ from the hand-written schemas in
`evals.world`. A cassette recorded here is therefore not interchangeable with one
recorded by `examples.openai_sdk_agent` -- correctly, because those are different
requests. Portability holds between clients sending the same logical request, which
`tests/test_compat_openai_sdk.py` demonstrates.

Requires the `compat` dependency group: `uv sync --group compat`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from evals.tasks import SYSTEM_PROMPTS, get_task
from evals.world import World

__all__ = ["build_agent", "world_tools"]


def world_tools(world: World) -> list[StructuredTool]:
    """The same two tools, described the idiomatic LangChain way.

    Schemas are derived from these signatures and docstrings rather than written by
    hand, which is exactly why a cassette from this client is not interchangeable with
    one from a client sending hand-written schemas.
    """

    def find_customer(name: str) -> str:
        """Look up a customer by exact name. Returns their id and region."""
        return world.call("find_customer", {"name": name})

    def list_orders(customer_id: int, status: str = "shipped") -> str:
        """List a customer's orders, filtered by status ('shipped' or 'pending')."""
        return world.call("list_orders", {"customer_id": customer_id, "status": status})

    return [
        StructuredTool.from_function(find_customer),
        StructuredTool.from_function(list_orders),
    ]


def build_agent(*, base_url: str, model: str, api_key: str, world: World) -> Any:
    """A LangGraph agent whose only misfeed-specific setting is `base_url`."""
    chat = ChatOpenAI(
        model=model,
        base_url=base_url,
        # SecretStr rather than a bare string: it is what ChatOpenAI's signature asks
        # for, and it keeps the key out of a repr if the model object is ever logged.
        api_key=SecretStr(api_key),
        temperature=0.0,
        max_retries=0,
    )
    return create_agent(chat, world_tools(world))


async def _main(args: argparse.Namespace) -> int:
    task = get_task(args.task)
    world = World()
    try:
        agent = build_agent(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or "unused",
            world=world,
        )
        state = await agent.ainvoke(
            {
                "messages": [
                    ("system", SYSTEM_PROMPTS[args.variant]),
                    ("user", task.question),
                ]
            }
        )
    finally:
        world.close()

    messages = state["messages"]
    print(f"task     : {task.id}")
    print(f"expected : {task.expected}")
    print(f"messages : {len(messages)}")
    print(f"tools    : {' -> '.join(c.name for c in world.calls) or '(none)'}")
    print(f"\n{messages[-1].content}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="stub/naive")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--task", default="T1_shipped_total")
    parser.add_argument("--variant", default="baseline", choices=sorted(SYSTEM_PROMPTS))
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
