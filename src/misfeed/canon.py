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
        return {k: _normalize(v, rules) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v, rules) for v in value]
    return value


def canonical_request(
    body: dict[str, Any], rules: tuple[NormalizeRule, ...] = DEFAULT_RULES
) -> dict[str, Any]:
    """Reduce a request body to the fields that determine the completion."""
    return {k: _normalize(body[k], rules) for k in KEY_FIELDS if k in body}


def canonical_json(obj: Any) -> bytes:
    """Stable JSON: sorted keys, no incidental whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def request_key(body: dict[str, Any], rules: tuple[NormalizeRule, ...] = DEFAULT_RULES) -> str:
    """The cassette lookup key for a request."""
    return hashlib.sha256(canonical_json(canonical_request(body, rules))).hexdigest()


def explain(
    body: dict[str, Any], rules: tuple[NormalizeRule, ...] = DEFAULT_RULES
) -> dict[str, Any]:
    """What was hashed, and under which rules -- printed on a replay miss."""
    return {
        "key": request_key(body, rules),
        "rules": [r.name for r in rules],
        "dropped_fields": sorted(set(body) - set(KEY_FIELDS)),
        "canonical": canonical_request(body, rules),
    }
