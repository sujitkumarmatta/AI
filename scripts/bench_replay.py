"""Measure how long the proxy takes to serve a replayed request.

Reported as the time to serve one recorded response end to end, from committed
cassettes, with no upstream involved. That is the figure that matters for whether
a large offline study is practical.

It is not "overhead over a live call": in replay mode there is no live call to
compare against, and quoting a difference against a network round trip would
flatter the number.

Writes JSON so the README can cite a figure that traces to a committed artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from misfeed.proxy import Engine, Mode, ProxyConfig
from misfeed.store import Store


async def measure(store: Store, trace_id: str, repeats: int) -> list[float]:
    """Per-request service times in milliseconds, across whole replays."""
    entries = store.resolve(trace_id)
    requests = [store.get_blob(e.request) for e in entries if e.request is not None]
    if not requests:
        raise SystemExit(f"trace {trace_id!r} stored no requests to replay")

    samples: list[float] = []
    for _ in range(repeats):
        engine = Engine(ProxyConfig(mode=Mode.REPLAY, store=store, trace_id=trace_id))
        for body in requests:
            started = time.perf_counter()
            await engine.handle(dict(body))
            samples.append((time.perf_counter() - started) * 1000)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("evals/cassettes"))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("evals/bench-replay.json"))
    args = parser.parse_args(argv)

    store = Store(args.store)
    traces = [t for t in store.list_traces() if t.startswith("trunk--")]
    if not traces:
        raise SystemExit(f"no trunk traces in {args.store}")
    trace_id = traces[0]

    samples = asyncio.run(measure(store, trace_id, args.repeats))
    samples.sort()
    result: dict[str, Any] = {
        "trace": trace_id,
        "requests_served": len(samples),
        "replays": args.repeats,
        "ms": {
            "p50": round(statistics.median(samples), 3),
            "p95": round(samples[int(len(samples) * 0.95) - 1], 3),
            "p99": round(samples[int(len(samples) * 0.99) - 1], 3),
            "max": round(samples[-1], 3),
        },
        "note": (
            "Time to serve one recorded response from disk-backed cassettes, "
            "no upstream involved. Includes canonicalisation and hashing of the "
            "request. Machine-dependent; regenerate with scripts/bench_replay.py."
        ),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
