"""A deterministic stand-in model, so the harness can be demonstrated offline.

This is NOT a model and its behaviour is NOT a measurement. It is a scripted
stand-in that follows the tool-calling protocol well enough to drive the whole
pipeline with no endpoint, no key and no network, which is what lets `make demo`
run on a clean clone.

Its two policies exist to show the metric moving:

- `naive` uses whatever a tool returned without checking it, so an empty or
  truncated result becomes a confident wrong number.
- `careful` validates each tool result before using it and says so when the data
  is not there.

The difference between them is true by construction, not by observation. Any
report produced from this stub is stamped `synthetic: true` and names the model
`stub/<policy>`, because presenting scripted behaviour as a finding about real
models would be precisely the kind of confident wrong answer this project is
about.

Responses are byte-for-byte deterministic -- `created` is fixed and `id` is
derived from the request -- so cassettes recorded from the stub are stable and
reproducible. Real providers vary both, which is why a real recording yields a
distinct blob per response.

`usage` is a character-count approximation, not a tokenisation. It exists so the
telemetry path is exercised end to end and must not be read as a token count.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

__all__ = ["POLICIES", "create_stub_model"]

POLICIES: tuple[str, ...] = ("naive", "careful")

_QUOTED = re.compile(r"'([^']+)'")


def _approx_tokens(text: str) -> int:
    """Roughly four characters per token. An approximation, not a tokenisation."""
    return max(1, len(text) // 4)


def _tool_results(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Tool results in order, paired with the tool name that produced each."""
    names: dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            names[call["id"]] = call["function"]["name"]
    return [
        (names.get(message.get("tool_call_id", ""), "?"), message.get("content") or "")
        for message in messages
        if message.get("role") == "tool"
    ]


def _wants_count(question: str) -> bool:
    return "how many" in question.casefold()


def _envelope(request_body: dict[str, Any], message: dict[str, Any], model: str) -> dict[str, Any]:
    prompt_text = json.dumps(request_body.get("messages", []))
    completion_text = json.dumps(message)
    digest = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    return {
        "id": f"stub-{digest}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": _approx_tokens(prompt_text),
            "completion_tokens": _approx_tokens(completion_text),
            "total_tokens": _approx_tokens(prompt_text) + _approx_tokens(completion_text),
        },
    }


def _call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _decide(body: dict[str, Any], policy: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = body.get("messages", [])
    question = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
    results = _tool_results(messages)
    careful = policy == "careful"

    if not results:
        match = _QUOTED.search(question)
        return _call("find_customer", {"name": match.group(1) if match else ""}, "call_1")

    last_name, last_content = results[-1]

    if last_name == "find_customer":
        parsed = _parse(last_content)
        customer_id = parsed.get("customer_id") if isinstance(parsed, dict) else None
        if not isinstance(customer_id, int):
            if careful:
                return _text(
                    "The customer lookup returned no usable record, so I cannot "
                    "identify the customer. The result was empty or missing the "
                    "customer_id field. I will not guess an id."
                )
            # naive: carry on with a placeholder id and see what happens.
            customer_id = 0
        return _call("list_orders", {"customer_id": customer_id, "status": "shipped"}, "call_2")

    if last_name == "list_orders":
        parsed = _parse(last_content)
        orders = parsed.get("orders") if isinstance(parsed, dict) else None
        if not isinstance(orders, list) or not orders:
            if careful:
                return _text(
                    "The order lookup returned no rows. The tool result was empty "
                    "or did not contain an 'orders' list, so the data required to "
                    "answer is missing. I am not able to compute a total from it "
                    "and will not report zero as if it were the answer."
                )
            # naive: an empty list sums to zero, and zero looks like an answer.
            orders = []
        if careful:
            broken = [o for o in orders if not isinstance(o, dict) or "amount_cents" not in o]
            if broken and not _wants_count(question):
                return _text(
                    "Some order records are missing the amount_cents field, so the "
                    "total would be incomplete. I cannot give a reliable figure."
                )
        if _wants_count(question):
            return _text(f"Counted the shipped orders.\nANSWER: {len(orders)}")
        total = sum(
            int(o["amount_cents"])
            for o in orders
            if isinstance(o, dict) and isinstance(o.get("amount_cents"), int)
        )
        return _text(f"Summed the shipped order amounts.\nANSWER: {total}")

    return _text("I do not know how to continue.\nANSWER: 0")


def _parse(content: str) -> Any:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None


def _text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def create_stub_model(policy: str = "naive") -> Starlette:
    """An OpenAI-compatible endpoint backed by a scripted policy."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    model_name = f"stub/{policy}"

    async def completions(request: Request) -> JSONResponse:
        body = await request.json()
        return JSONResponse(_envelope(body, _decide(body, policy), model_name))

    return Starlette(routes=[Route("/chat/completions", completions, methods=["POST"])])
