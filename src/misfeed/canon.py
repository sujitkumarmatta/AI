"""Canonicalisation and hashing of OpenAI-compatible chat requests.

Replay only works if "the same request" is a decidable question. Two problems
make it non-trivial:

1.  Agents inject volatile content into prompts -- timestamps, UUIDs, session
    identifiers. A byte-exact hash would miss on every run. So a small set of
    explicit, named normalisation rules is applied to string content before
    hashing.
2.  The wire format is not the request's meaning. A streaming and a
    non-streaming call for identical content are the same logical request with
    different framing, so `stream` is excluded from the key and the framing is
    synthesised at replay time instead.
3.  Clients disagree about explicit nulls. A loop that appends a response message
    verbatim sends `"content": null` on an assistant tool-call message; the official
    SDK's `model_dump(exclude_none=True)` omits the field entirely. Both mean "no
    content", so null-valued fields are dropped before hashing. Without this, a
    cassette recorded by one client cannot be replayed by another -- which was true
    of this project until a compatibility test against the official SDK showed it.
4.  Streaming leaks its framing into the next request. A client that accumulates a
    streamed reply keeps the delta's `index` on each tool call, then sends it back in
    the assistant message; a non-streaming client has no such field. Position in the
    `tool_calls` array already carries that information, and the API does not read
    `index` there, so it is dropped. Without this a streaming and a non-streaming
    client could not share a cassette -- found the same way, by driving the official
    SDK down both paths.

Both choices trade exactness for usability, so both are visible: `explain()`
returns the canonical form that was hashed, and a replay miss prints it rather
than silently falling through to a live call.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_RULES",
    "NormalizeRule",
    "canonical_json",
    "canonical_request",
    "explain",
    "fingerprint",
    "request_key",
]

# Fields that carry the request's meaning. Anything not listed is excluded from
# the key: `stream`/`stream_options` (framing, see module docstring), `user`,
# `metadata`, `store` (caller bookkeeping, no effect on the completion).
KEY_FIELDS: tuple[str, ...] = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "seed",
    "n",
    "response_format",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
)


@dataclass(frozen=True, slots=True)
class NormalizeRule:
    """A named substitution applied to string content before hashing.

    `name` exists so that a replay miss can report which rules were in force,
    and so that a project can audit its own normalisation instead of inheriting
    opaque behaviour.
    """

    name: str
    pattern: re.Pattern[str]
    replacement: str

    @classmethod
    def of(cls, name: str, pattern: str, replacement: str) -> NormalizeRule:
        return cls(name=name, pattern=re.compile(pattern), replacement=replacement)


# Deliberately conservative: each rule targets a format that is volatile per-run
# but semantically irrelevant to the completion. Anything broader risks masking
# a genuine prompt difference, which would make replay quietly wrong.
DEFAULT_RULES: tuple[NormalizeRule, ...] = (
    NormalizeRule.of(
        "iso8601",
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
        "<TS>",
    ),
    NormalizeRule.of(
        "uuid",
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "<UUID>",
    ),
    NormalizeRule.of("epoch_ms", r"\b1[6-9]\d{11}\b", "<EPOCH_MS>"),
)


def normalize_text(text: str, rules: tuple[NormalizeRule, ...] = DEFAULT_RULES) -> str:
    """Apply every rule to `text`, in order."""
    for rule in rules:
        text = rule.pattern.sub(rule.replacement, text)
    return text


def _strip_tool_call_framing(message: Any) -> Any:
    """Remove streaming-accumulation artefacts from an assistant message.

    Narrow on purpose. `index` is meaningful on a streaming delta and on
    `choices[].index`; inside a request's `messages[].tool_calls[]` it is neither part
    of the schema nor read by the API, it is just what a client's accumulator left
    behind. Dropping it anywhere else would risk masking a real difference.
    """
    if not isinstance(message, dict):
        return message
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return message
    return {
        **message,
        "tool_calls": [
            {k: v for k, v in call.items() if k != "index"} if isinstance(call, dict) else call
            for call in calls
        ],
    }


def _normalize(value: Any, rules: tuple[NormalizeRule, ...]) -> Any:
    if isinstance(value, str):
        return normalize_text(value, rules)
    if isinstance(value, bool):
        # Must precede the int/float branch: bool is a subclass of int.
        return value
    if isinstance(value, float) and value.is_integer():
        # temperature 0 and 0.0 are the same request; JSON renders them apart.
        return int(value)
    if isinstance(value, dict):
        # An explicit null and an omitted optional field mean the same thing in this
        # API, and clients disagree about which they send. Dropping nulls is what
        # makes a cassette portable between clients.
        return {k: _normalize(v, rules) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_normalize(v, rules) for v in value]
    return value


def canonical_request(
    body: dict[str, Any], rules: tuple[NormalizeRule, ...] = DEFAULT_RULES
) -> dict[str, Any]:
    """Reduce a request body to the fields that determine the completion.

    Null-valued fields are dropped here as well as inside nested objects, so the rule
    holds at every depth: a client that sends `"stop": null` and one that omits `stop`
    are making the same request. Streaming-accumulation artefacts are then stripped
    from each message, so a streaming and a non-streaming client agree.
    """
    canon = {k: _normalize(body[k], rules) for k in KEY_FIELDS if k in body and body[k] is not None}
    messages = canon.get("messages")
    if isinstance(messages, list):
        canon["messages"] = [_strip_tool_call_framing(m) for m in messages]
    return canon


def canonical_json(obj: Any) -> bytes:
    """Stable JSON: sorted keys, no incidental whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def request_key(body: dict[str, Any], rules: tuple[NormalizeRule, ...] = DEFAULT_RULES) -> str:
    """The cassette lookup key for a request."""
    return hashlib.sha256(canonical_json(canonical_request(body, rules))).hexdigest()


def fingerprint(rules: tuple[NormalizeRule, ...] = DEFAULT_RULES) -> str:
    """A short hash of the canonicalisation scheme itself.

    Every recorded trace stores this. If the scheme changes -- a field added to
    KEY_FIELDS, a normalisation rule edited, the null-dropping behaviour altered --
    previously recorded keys stop matching and every request misses. Without a stamp
    that shows up as "the cassettes are incomplete", and a run against a paid endpoint
    quietly re-records the whole study.

    Derived rather than hand-maintained, because a version integer someone has to
    remember to bump is a version integer that does not get bumped. Changing the
    scheme changes this automatically.
    """
    material = canonical_json(
        {
            "key_fields": list(KEY_FIELDS),
            "rules": [[r.name, r.pattern.pattern, r.replacement] for r in rules],
            "structural": [
                "drop_null_fields",
                "drop_tool_call_index",
                "integral_float_to_int",
                "exclude_stream",
            ],
        }
    )
    return hashlib.sha256(material).hexdigest()[:12]


def explain(
    body: dict[str, Any], rules: tuple[NormalizeRule, ...] = DEFAULT_RULES
) -> dict[str, Any]:
    """What was hashed, and under which rules -- printed on a replay miss."""
    return {
        "key": request_key(body, rules),
        "rules": [r.name for r in rules],
        "structural_rules": ["drop_null_fields", "drop_tool_call_index"],
        "canon": fingerprint(rules),
        "dropped_fields": sorted(set(body) - set(KEY_FIELDS)),
        "canonical": canonical_request(body, rules),
    }
