"""Synthesising a streamed response from a recorded one.

Cassettes store the logical response, not the wire framing: `stream` is excluded
from the request key, so one recording serves a streaming and a non-streaming
client alike. This module rebuilds the Server-Sent Events framing from that stored
response.

What is real and what is not, stated plainly because it matters:

- The **content** is real. A client reassembling the stream gets byte-for-byte the
  message that was recorded, and there is a test asserting that against the official
  SDK.
- The **chunk boundaries and timing are invented.** A recorded completion has no
  token boundaries in it, so there is nothing to reproduce. The default is therefore
  a single content delta -- the least fabrication that is still valid SSE. Anyone who
  needs a multi-chunk stream (to exercise a client's incremental accumulation path,
  say) can ask for one, and it is still invented.

Emitting fake inter-token delays would make a replay look like a live call while
telling the caller nothing true, so nothing here sleeps.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

__all__ = ["DONE", "sse_events", "sse_stream"]

DONE = "data: [DONE]\n\n"


def _event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _envelope(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": response.get("id", "misfeed-replay"),
        "object": "chat.completion.chunk",
        "created": response.get("created", 0),
        "model": response.get("model"),
    }


def _chunk(
    base: dict[str, Any],
    index: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        **base,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }


def _split(text: str, chunk_chars: int | None) -> list[str]:
    """Break content into deltas. One delta unless a size is asked for."""
    if not text:
        return []
    if chunk_chars is None or chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text]
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def sse_events(
    response: dict[str, Any],
    *,
    include_usage: bool = False,
    chunk_chars: int | None = None,
) -> Iterator[str]:
    """Yield the SSE events equivalent to `response`.

    `include_usage` mirrors `stream_options.include_usage`: a final chunk carrying
    `usage` and no choices, which is where the streaming API reports it.
    """
    base = _envelope(response)
    choices = response.get("choices") or []

    for position, choice in enumerate(choices):
        index = choice.get("index", position)
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")

        opening: dict[str, Any] = {"role": message.get("role", "assistant")}
        yield _event(_chunk(base, index, opening))

        for piece in _split(message.get("content") or "", chunk_chars):
            yield _event(_chunk(base, index, {"content": piece}))

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            # Emitted in one delta per call, with the `index` clients accumulate on.
            # Splitting an arguments string across deltas would invent boundaries
            # inside a JSON document for no benefit.
            deltas = [
                {
                    "index": position_in_call,
                    "id": call.get("id"),
                    "type": call.get("type", "function"),
                    "function": call.get("function", {}),
                }
                for position_in_call, call in enumerate(tool_calls)
            ]
            yield _event(_chunk(base, index, {"tool_calls": deltas}))

        if message.get("refusal"):
            yield _event(_chunk(base, index, {"refusal": message["refusal"]}))

        yield _event(_chunk(base, index, {}, finish_reason=finish_reason))

    if include_usage:
        usage = response.get("usage")
        yield _event({**base, "choices": [], "usage": usage})

    yield DONE


def sse_stream(
    response: dict[str, Any],
    *,
    include_usage: bool = False,
    chunk_chars: int | None = None,
) -> Iterator[bytes]:
    """`sse_events`, encoded, for an ASGI streaming response."""
    for event in sse_events(response, include_usage=include_usage, chunk_chars=chunk_chars):
        yield event.encode("utf-8")
