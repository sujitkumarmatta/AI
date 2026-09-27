"""Fail if a number in the README does not match a committed artifact.

The project's rule is that no published figure may be unbacked. A rule enforced by
good intentions decays: regenerating the benchmark once moved p95 from 0.36 to 0.33 ms
and made the README wrong within minutes. So the rule is checked.

This deliberately checks only figures that can be tied to an artifact mechanically --
the results tables, the replay bound, the test count. Prose claims are not checkable
here and are the reviewer's job.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def _table_rows(report: dict[str, object]) -> list[str]:
    results = report["results"]
    assert isinstance(results, dict)
    rows: list[str] = []
    by_variant = results["by_variant"]
    assert isinstance(by_variant, dict)
    for key, bucket in by_variant.items():
        rate = f"{bucket['silent_corruption_rate'] * 100:.1f}%"
        rows.append(f"| {key} | {bucket['scored']} | {bucket['silent_corruption']} | {rate} |")
    by_fault = results["by_fault"]
    assert isinstance(by_fault, dict)
    for fault, bucket in by_fault.items():
        rate = f"{bucket['silent_corruption_rate'] * 100:.1f}%"
        rows.append(f"| `{fault}` | {bucket['scored']} | {bucket['silent_corruption']} | {rate} |")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--report", type=Path, default=Path("evals/report.json"))
    parser.add_argument("--bench", type=Path, default=Path("evals/bench-replay.json"))
    parser.add_argument("--bench-bound-ms", type=float, default=0.4)
    args = parser.parse_args(argv)

    readme = args.readme.read_text(encoding="utf-8")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    results = report["results"]
    failures: list[str] = []

    for row in _table_rows(report):
        if row not in readme:
            failures.append(f"results row missing from README:\n    {row}")

    totals = results["totals"]
    sentence = f"{totals['scored']} of {totals['fault_runs']} fault runs scored"
    if sentence not in readme:
        failures.append(f"README does not state: {sentence!r}")
    skipped = f"{totals['skipped_inapplicable']} skipped"
    if skipped not in readme:
        failures.append(f"README does not state: {skipped!r}")

    bench = json.loads(args.bench.read_text(encoding="utf-8"))
    p95 = float(bench["ms"]["p95"])
    if p95 >= args.bench_bound_ms:
        failures.append(
            f"measured replay p95 is {p95} ms, which no longer satisfies the README's "
            f"stated bound of under {args.bench_bound_ms} ms"
        )
    if f"under {args.bench_bound_ms} ms" not in readme:
        failures.append(
            f"README no longer states the replay bound of under {args.bench_bound_ms} ms"
        )

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--co"],
        capture_output=True,
        text=True,
        check=False,
    )
    count = len([line for line in collected.stdout.splitlines() if "::" in line])
    claimed = re.search(r"# (\d+) tests", readme)
    if claimed is None:
        failures.append("README does not state a test count")
    elif int(claimed.group(1)) != count:
        failures.append(f"README claims {claimed.group(1)} tests; {count} are collected")

    if failures:
        print("README claims do not match the committed artifacts:\n")
        for failure in failures:
            print(f"  - {failure}")
        print("\nRegenerate with `make study` and `uv run python scripts/bench_replay.py`,")
        print("then update the README, or correct the claim.")
        return 1

    print(f"README claims verified against {args.report} and {args.bench}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
