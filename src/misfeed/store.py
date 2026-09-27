"""Cassette storage: a content-addressed blob store plus a trace tree.

A *trunk* is a clean recorded run. A *branch* is what happened after a fault was
injected at some step. Because a branch only stores its own divergent
continuation, and shares the trunk's prefix, recording N faults against a run of
M steps costs far less than N full runs -- and, more importantly, every
experiment becomes replayable offline from committed files.

Identical responses are stored once, by content hash. Trace identifiers are
caller-supplied and readable (`trunk-t1`, `branch-t1-s2-empty_success`) rather
than random, because these files are committed and diffed by humans.

Replay is strict. A lookup that has no recorded response raises rather than
falling through to a live call: a silent fallthrough would turn a divergence into
an invisible, billed, non-reproducible run, which is the exact class of bug this
project exists to expose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from misfeed.canon import canonical_json

__all__ = ["CassetteMiss", "Entry", "Player", "Store", "Trace"]


class CassetteMiss(LookupError):
    """A replay asked for a response that was never recorded."""

    def __init__(self, key: str, detail: dict[str, Any]) -> None:
        self.key = key
        self.detail = detail
        super().__init__(
            f"no recorded response for request key {key[:12]}...\n"
            f"{json.dumps(detail, indent=2, sort_keys=True)}"
        )


@dataclass(frozen=True, slots=True)
class Entry:
    """One recorded exchange.

    `request` holds the blob digest of the canonical request. It is optional so
    that a cassette can be written without retaining prompts, but it is populated
    by default: committed cassettes are reviewed by people, and a file of bare
    hashes cannot be reviewed or diagnosed. The consequence is that a cassette
    contains the prompts it was recorded from, which is why recording over
    sensitive data and then committing it is called out in the docs.
    """

    step: int
    request_key: str
    blob: str
    request: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "step": self.step,
            "request_key": self.request_key,
            "blob": self.blob,
        }
        if self.request is not None:
            out["request"] = self.request
        return out

    @staticmethod
    def from_json(raw: dict[str, Any]) -> Entry:
        return Entry(
            step=int(raw["step"]),
            request_key=raw["request_key"],
            blob=raw["blob"],
            request=raw.get("request"),
        )


@dataclass(slots=True)
class Trace:
    """A recorded run. `parent`/`fork_step`/`fault` are set only for branches."""

    id: str
    kind: str  # "trunk" | "branch"
    entries: list[Entry] = field(default_factory=list)
    parent: str | None = None
    fork_step: int | None = None
    fault: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("trunk", "branch"):
            raise ValueError(f"kind must be 'trunk' or 'branch', got {self.kind!r}")
        if self.kind == "branch" and (self.parent is None or self.fork_step is None):
            raise ValueError("a branch requires both parent and fork_step")
        if self.kind == "trunk" and (self.parent is not None or self.fork_step is not None):
            raise ValueError("a trunk must not have parent or fork_step")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "entries": [e.to_json() for e in self.entries],
            "meta": self.meta,
        }
        if self.kind == "branch":
            out["parent"] = self.parent
            out["fork_step"] = self.fork_step
            out["fault"] = self.fault
        return out

    @staticmethod
    def from_json(raw: dict[str, Any]) -> Trace:
        return Trace(
            id=raw["id"],
            kind=raw["kind"],
            entries=[Entry.from_json(e) for e in raw["entries"]],
            parent=raw.get("parent"),
            fork_step=raw.get("fork_step"),
            fault=raw.get("fault"),
            meta=raw.get("meta", {}),
        )


class Store:
    """Blobs and traces on disk, under `root`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.blobs = root / "blobs"
        self.traces = root / "traces"

    def blob_path(self, digest: str) -> Path:
        """Where a blob lives. Public because tooling and tests legitimately look."""
        return self.blobs / digest[:2] / f"{digest}.json"

    def put_blob(self, obj: Any) -> str:
        """Store `obj`, returning its content hash. Writing twice is a no-op."""
        import hashlib

        payload = canonical_json(obj)
        digest = hashlib.sha256(payload).hexdigest()
        path = self.blob_path(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return digest

    def get_blob(self, digest: str) -> Any:
        path = self.blob_path(digest)
        if not path.exists():
            raise FileNotFoundError(f"blob {digest} not in {self.blobs}")
        return json.loads(path.read_bytes())

    def has_blob(self, digest: str) -> bool:
        return self.blob_path(digest).exists()

    def save_trace(self, trace: Trace) -> Path:
        self.traces.mkdir(parents=True, exist_ok=True)
        path = self.traces / f"{trace.id}.json"
        # Indented, so that a committed cassette produces a readable diff.
        path.write_text(
            json.dumps(trace.to_json(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def load_trace(self, trace_id: str) -> Trace:
        path = self.traces / f"{trace_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"trace {trace_id!r} not in {self.traces}")
        return Trace.from_json(json.loads(path.read_text(encoding="utf-8")))

    def list_traces(self) -> list[str]:
        if not self.traces.exists():
            return []
        return sorted(p.stem for p in self.traces.glob("*.json"))

    def resolve(self, trace_id: str) -> list[Entry]:
        """Materialise the full entry sequence for a trace.

        For a branch this is the parent's entries before `fork_step`, followed by
        the branch's own. Resolution walks the chain, so a branch of a branch
        works; a cycle raises rather than hanging.
        """
        seen: set[str] = set()
        chain: list[Trace] = []
        current: str | None = trace_id
        while current is not None:
            if current in seen:
                raise ValueError(f"cycle in trace ancestry at {current!r}")
            seen.add(current)
            trace = self.load_trace(current)
            chain.append(trace)
            current = trace.parent

        entries: list[Entry] = []
        for trace in reversed(chain):  # root first
            if trace.kind == "branch":
                assert trace.fork_step is not None  # guaranteed by __post_init__
                entries = [e for e in entries if e.step < trace.fork_step]
            entries = entries + list(trace.entries)
        return entries


class Player:
    """Serves recorded responses for a resolved entry list.

    Repeats are handled by consuming a per-key queue in recorded order, so an
    agent that issues the same request twice receives the two responses it
    received when the run was recorded. Running out is a miss, not a reuse of the
    last response -- reuse would mask divergence.
    """

    def __init__(self, store: Store, entries: list[Entry]) -> None:
        self._store = store
        self._queues: dict[str, list[str]] = {}
        for entry in entries:
            self._queues.setdefault(entry.request_key, []).append(entry.blob)
        self._cursor: dict[str, int] = dict.fromkeys(self._queues, 0)
        self.served = 0

    @property
    def remaining(self) -> int:
        return sum(len(q) - self._cursor[k] for k, q in self._queues.items())

    def take(self, key: str, detail: dict[str, Any] | None = None) -> Any:
        queue = self._queues.get(key)
        index = self._cursor.get(key, 0)
        if queue is None or index >= len(queue):
            info: dict[str, Any] = dict(detail or {})
            info["reason"] = "key not recorded" if queue is None else "responses exhausted"
            info["recorded_for_key"] = 0 if queue is None else len(queue)
            info["already_served_for_key"] = index
            raise CassetteMiss(key, info)
        self._cursor[key] = index + 1
        self.served += 1
        return self._store.get_blob(queue[index])
