"""Tests for the cassette store.

The interesting cases are branch resolution (a branch must inherit exactly the
trunk prefix before its fork) and replay strictness (a miss must raise, because a
silent fallthrough is the failure mode this project exists to expose).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from misfeed.store import CassetteMiss, Entry, Player, Store, Trace


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "cassettes")


def trunk(store: Store, trace_id: str, payloads: list[str]) -> Trace:
    """A trunk whose step i has request key `k{i}` and response `payloads[i]`."""
    entries = [
        Entry(step=i, request_key=f"k{i}", blob=store.put_blob({"text": text}))
        for i, text in enumerate(payloads)
    ]
    trace = Trace(id=trace_id, kind="trunk", entries=entries)
    store.save_trace(trace)
    return trace


class TestBlobs:
    def test_identical_content_is_stored_once(self, store: Store) -> None:
        # NFR: store dedup.
        a = store.put_blob({"x": 1, "y": 2})
        b = store.put_blob({"y": 2, "x": 1})  # same content, different order
        assert a == b
        assert len(list(store.blobs.rglob("*.json"))) == 1

    def test_different_content_gets_different_digests(self, store: Store) -> None:
        assert store.put_blob({"x": 1}) != store.put_blob({"x": 2})

    def test_roundtrip(self, store: Store) -> None:
        obj = {"choices": [{"message": {"content": "café"}}], "n": 1}
        assert store.get_blob(store.put_blob(obj)) == obj

    def test_missing_blob_raises(self, store: Store) -> None:
        with pytest.raises(FileNotFoundError):
            store.get_blob("0" * 64)

    def test_rewriting_the_same_blob_is_a_noop(self, store: Store) -> None:
        digest = store.put_blob({"x": 1})
        mtime = store._blob_path(digest).stat().st_mtime_ns
        assert store.put_blob({"x": 1}) == digest
        assert store._blob_path(digest).stat().st_mtime_ns == mtime


class TestTraceValidation:
    def test_branch_requires_parent_and_fork_step(self) -> None:
        with pytest.raises(ValueError, match="requires both parent and fork_step"):
            Trace(id="b", kind="branch", parent="t")
        with pytest.raises(ValueError, match="requires both parent and fork_step"):
            Trace(id="b", kind="branch", fork_step=1)

    def test_trunk_must_not_have_parent(self) -> None:
        with pytest.raises(ValueError, match="must not have parent"):
            Trace(id="t", kind="trunk", parent="other")

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            Trace(id="t", kind="mainline")


class TestTraceRoundTrip:
    def test_trunk_roundtrip(self, store: Store) -> None:
        original = trunk(store, "t1", ["a", "b"])
        assert store.load_trace("t1").to_json() == original.to_json()

    def test_branch_roundtrip_keeps_fault(self, store: Store) -> None:
        trunk(store, "t1", ["a", "b"])
        branch = Trace(
            id="b1",
            kind="branch",
            parent="t1",
            fork_step=1,
            fault={"id": "empty_success", "step": 1},
            entries=[Entry(step=1, request_key="k1x", blob=store.put_blob({"text": "z"}))],
        )
        store.save_trace(branch)
        loaded = store.load_trace("b1")
        assert loaded.fault == {"id": "empty_success", "step": 1}
        assert loaded.fork_step == 1

    def test_saved_trace_is_human_readable(self, store: Store) -> None:
        # Cassettes are committed, so their diffs must be reviewable.
        path = store.save_trace(trunk(store, "t1", ["a"]))
        text = path.read_text()
        assert text.endswith("\n")
        assert "\n  " in text  # indented, not a single line

    def test_missing_trace_raises(self, store: Store) -> None:
        with pytest.raises(FileNotFoundError):
            store.load_trace("nope")

    def test_list_traces_is_sorted(self, store: Store) -> None:
        trunk(store, "t2", ["a"])
        trunk(store, "t1", ["a"])
        assert store.list_traces() == ["t1", "t2"]

    def test_list_traces_empty_when_nothing_saved(self, store: Store) -> None:
        assert store.list_traces() == []


class TestResolve:
    def test_trunk_resolves_to_its_own_entries(self, store: Store) -> None:
        trunk(store, "t1", ["a", "b", "c"])
        assert [e.request_key for e in store.resolve("t1")] == ["k0", "k1", "k2"]

    def test_branch_inherits_only_the_prefix_before_its_fork(self, store: Store) -> None:
        trunk(store, "t1", ["a", "b", "c"])
        store.save_trace(
            Trace(
                id="b1",
                kind="branch",
                parent="t1",
                fork_step=1,
                entries=[
                    Entry(step=1, request_key="k1x", blob=store.put_blob({"text": "bx"})),
                    Entry(step=2, request_key="k2x", blob=store.put_blob({"text": "cx"})),
                ],
            )
        )
        # step 0 from the trunk; trunk steps 1 and 2 are discarded.
        assert [e.request_key for e in store.resolve("b1")] == ["k0", "k1x", "k2x"]

    def test_fork_at_zero_inherits_nothing(self, store: Store) -> None:
        trunk(store, "t1", ["a", "b"])
        store.save_trace(
            Trace(
                id="b1",
                kind="branch",
                parent="t1",
                fork_step=0,
                entries=[Entry(step=0, request_key="k0x", blob=store.put_blob({"text": "x"}))],
            )
        )
        assert [e.request_key for e in store.resolve("b1")] == ["k0x"]

    def test_branch_of_a_branch(self, store: Store) -> None:
        trunk(store, "t1", ["a", "b", "c", "d"])
        store.save_trace(
            Trace(
                id="b1",
                kind="branch",
                parent="t1",
                fork_step=2,
                entries=[
                    Entry(step=2, request_key="k2x", blob=store.put_blob({"text": "cx"})),
                    Entry(step=3, request_key="k3x", blob=store.put_blob({"text": "dx"})),
                ],
            )
        )
        store.save_trace(
            Trace(
                id="b2",
                kind="branch",
                parent="b1",
                fork_step=3,
                entries=[Entry(step=3, request_key="k3y", blob=store.put_blob({"text": "dy"}))],
            )
        )
        assert [e.request_key for e in store.resolve("b2")] == ["k0", "k1", "k2x", "k3y"]

    def test_ancestry_cycle_raises_instead_of_hanging(self, store: Store) -> None:
        for a, b in (("b1", "b2"), ("b2", "b1")):
            path = store.traces / f"{a}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"id": a, "kind": "branch", "parent": b, "fork_step": 0, "entries": []})
            )
        with pytest.raises(ValueError, match="cycle in trace ancestry"):
            store.resolve("b1")


class TestPlayer:
    def test_serves_recorded_responses(self, store: Store) -> None:
        trunk(store, "t1", ["a", "b"])
        player = Player(store, store.resolve("t1"))
        assert player.take("k0") == {"text": "a"}
        assert player.take("k1") == {"text": "b"}
        assert player.served == 2
        assert player.remaining == 0

    def test_repeated_key_serves_recorded_order_then_misses(self, store: Store) -> None:
        # An agent that asks the same thing twice gets the two answers it got
        # when recorded -- and a third ask is a divergence, not a free repeat.
        blobs = [store.put_blob({"text": t}) for t in ("first", "second")]
        store.save_trace(
            Trace(
                id="t1",
                kind="trunk",
                entries=[Entry(step=i, request_key="same", blob=b) for i, b in enumerate(blobs)],
            )
        )
        player = Player(store, store.resolve("t1"))
        assert player.take("same") == {"text": "first"}
        assert player.take("same") == {"text": "second"}
        with pytest.raises(CassetteMiss) as caught:
            player.take("same")
        assert caught.value.detail["reason"] == "responses exhausted"
        assert caught.value.detail["recorded_for_key"] == 2
        assert caught.value.detail["already_served_for_key"] == 2

    def test_unknown_key_raises_with_diagnostics(self, store: Store) -> None:
        trunk(store, "t1", ["a"])
        player = Player(store, store.resolve("t1"))
        with pytest.raises(CassetteMiss) as caught:
            player.take("unseen", detail={"canonical": {"model": "m"}})
        assert caught.value.key == "unseen"
        assert caught.value.detail["reason"] == "key not recorded"
        assert caught.value.detail["canonical"] == {"model": "m"}
        # The message must carry the diagnostics; a bare miss is unactionable.
        assert "model" in str(caught.value)

    def test_miss_does_not_consume_or_corrupt_state(self, store: Store) -> None:
        trunk(store, "t1", ["a"])
        player = Player(store, store.resolve("t1"))
        with pytest.raises(CassetteMiss):
            player.take("unseen")
        assert player.served == 0
        assert player.take("k0") == {"text": "a"}
