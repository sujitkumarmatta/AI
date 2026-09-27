"""A narrated walkthrough of one fault experiment, from committed cassettes.

Runs entirely offline. Nothing here contacts a model: every response shown was
recorded once and is replayed from `evals/cassettes`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.tasks import get_task
from misfeed.store import Store

DEFAULT_STORE = Path("evals/cassettes")
DEFAULT_REPORT = Path("evals/report.json")

RULE = "=" * 78


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _final_text(store: Store, trace_id: str) -> str:
    entries = store.resolve(trace_id)
    response = store.get_blob(entries[-1].blob)
    return response["choices"][0]["message"].get("content") or "(no content)"


def _tool_messages(store: Store, trace_id: str) -> list[dict[str, Any]]:
    """Tool results as the model saw them on the last request of the run."""
    entries = store.resolve(trace_id)
    last_with_request = [e for e in entries if e.request is not None]
    if not last_with_request:
        return []
    request = store.get_blob(last_with_request[-1].request or "")
    return [m for m in request.get("messages", []) if m.get("role") == "tool"]


def _run_record(report: dict[str, Any], trace_id: str) -> dict[str, Any] | None:
    for run in report["results"]["runs"]:
        if run["trace_id"] == trace_id:
            assert isinstance(run, dict)
            return run
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--task", default="T1_shipped_total")
    parser.add_argument("--agent", default="stub-naive")
    parser.add_argument("--fault", default="empty_success")
    parser.add_argument("--step", type=int, default=2)
    args = parser.parse_args(argv)

    store = Store(args.store)
    if not store.list_traces():
        print(f"No cassettes in {args.store}. Run: make record")
        return 1
    report = json.loads(args.report.read_text())

    trunk = f"trunk--{args.agent}--baseline--{args.task}"
    branch = f"branch--{args.agent}--baseline--{args.task}--s{args.step}--{args.fault}"
    task = get_task(args.task)

    _heading("1. The task, and where its right answer comes from")
    print(f"question : {task.question}")
    print(f"expected : {task.expected}")
    print("source   : computed by SQL over the same rows the tools read, so the")
    print("           reference answer cannot drift from the fixture and no model")
    print("           or person grades anything.")

    _heading("2. The clean run (no fault) -- this gates everything after it")
    clean = _run_record(report, trunk)
    print(f"trace    : {args.store}/traces/{trunk}.json")
    print(f"answer   : {clean['answer'] if clean else '?'}")
    print(f"correct  : {clean['facts']['correct'] if clean else '?'}")
    print("           If the agent could not do this, its behaviour under a fault")
    print("           would be uninterpretable and the report would withhold it.")

    _heading(f"3. The same run with one tool result corrupted: {args.fault}")
    tool_messages = _tool_messages(store, branch)
    print(f"trace    : {args.store}/traces/{branch}.json")
    print(f"fault    : {args.fault} applied at step {args.step}")
    print("\nThe agent's tool really returned its normal payload. What the model")
    print("was shown on the final request:\n")
    for message in tool_messages:
        content = message.get("content") or ""
        shown = content if len(content) <= 160 else content[:157] + "..."
        print(f"  tool_call_id={message.get('tool_call_id')}  content={shown!r}")
    print("\nThe corruption persists for the rest of the run: a broken tool result")
    print("stays in the conversation, it does not heal on the next request.")

    faulted = _run_record(report, branch)
    _heading("4. What the agent did with it")
    print(_final_text(store, branch).strip())
    if faulted:
        print(f"\nanswer   : {faulted['answer']}   (expected {faulted['expected']})")
        facts = faulted.get("facts", {})
        print(f"answered : {facts.get('answered')}")
        print(f"correct  : {facts.get('correct')}")
        print(f"surfaced : {facts.get('surfaced')}  {faulted.get('surfaced_phrases', [])}")
        print(f"OUTCOME  : {str(faulted.get('outcome', '?')).upper()}")

    _heading("5. The same fault against an agent that checks its tool results")
    careful = f"branch--stub-careful--baseline--{args.task}--s{args.step}--{args.fault}"
    if careful in store.list_traces():
        record = _run_record(report, careful)
        print(_final_text(store, careful).strip()[:400])
        if record:
            print(f"\nOUTCOME  : {str(record.get('outcome', '?')).upper()}")

    _heading("6. Aggregate")
    print(args.report.with_suffix(".md").read_text())
    print("Every figure above was replayed from committed cassettes.")
    print("Live model calls made by this demo: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
