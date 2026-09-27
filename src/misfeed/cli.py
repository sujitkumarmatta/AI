"""Command line interface.

`serve` is what makes the framework-agnostic claim real: run it, point your agent's
OpenAI base URL at it, and its requests are recorded, replayed or fault-injected with
no change to your code.

The inspection commands exist because cassettes are committed and reviewed. A trace
you cannot read is a trace you cannot trust.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from misfeed.canon import fingerprint as canon_fingerprint
from misfeed.faults import FAULT_IDS, REGISTRY, FaultSpec
from misfeed.proxy import Mode, ProxyConfig, create_app
from misfeed.store import CanonMismatch, Store
from misfeed.verdict import Outcome

__all__ = ["main"]

DEFAULT_STORE = Path("cassettes")


def _store(args: argparse.Namespace) -> Store:
    return Store(args.store)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the proxy as an HTTP server."""
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - uvicorn is a declared dependency
        print("uvicorn is not installed", file=sys.stderr)
        return 1

    mode = Mode(args.mode)
    fault = None
    if args.fault is not None:
        params: dict[str, Any] = json.loads(args.fault_params) if args.fault_params else {}
        fault = FaultSpec(fault=args.fault, params=params, tool_name=args.tool)

    api_key = args.api_key or os.environ.get("MISFEED_UPSTREAM_API_KEY")
    try:
        config = ProxyConfig(
            mode=mode,
            store=_store(args),
            trace_id=args.trace,
            parent=args.parent,
            fault=fault,
            fault_at_step=args.at_step,
            upstream_base_url=args.upstream,
            upstream_api_key=api_key,
            max_live_requests=args.max_live_requests,
        )
    except ValueError as bad:
        print(f"error: {bad}", file=sys.stderr)
        return 2

    try:
        app = create_app(config)
    except CanonMismatch as stale:
        print(f"error: {stale}", file=sys.stderr)
        return 3
    print(f"misfeed {mode.value} on http://{args.host}:{args.port}/v1  trace={args.trace}")
    print(f"point your agent at:  OPENAI_BASE_URL=http://{args.host}:{args.port}/v1")
    if mode is Mode.REPLAY:
        print("replay mode: no upstream will be contacted; a miss returns 409")
    else:
        print(f"upstream: {args.upstream}  live-request cap: {args.max_live_requests}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_traces(args: argparse.Namespace) -> int:
    """List recorded traces."""
    store = _store(args)
    traces = store.list_traces()
    if not traces:
        print(f"no traces in {args.store}")
        return 1
    current = canon_fingerprint()
    for trace_id in traces:
        # Listing must survive a stale trace: this command is how you discover one.
        trace = store.load_trace(trace_id, check_canon=False)
        detail = f"{len(trace.entries):>3} entries"
        recorded = trace.meta.get("canon")
        if isinstance(recorded, str) and recorded != current:
            detail += f"  [STALE: recorded under canon {recorded}, this build is {current}]"
        if trace.kind == "branch":
            fault = (trace.fault or {}).get("fault", "?")
            detail += f"  branch of {trace.parent} at step {trace.fork_step}, fault={fault}"
            detail += f"  ({len(store.resolve(trace_id))} resolved)"
        print(f"{trace_id}\n    {trace.kind:<7} {detail}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Print a trace step by step, as the model saw it."""
    store = _store(args)
    try:
        trace = store.load_trace(args.trace)
    except (FileNotFoundError, CanonMismatch) as problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    print(f"trace   : {trace.id} ({trace.kind})")
    if trace.kind == "branch":
        print(f"parent  : {trace.parent}, forked at step {trace.fork_step}")
        print(f"fault   : {json.dumps(trace.fault, sort_keys=True)}")
    if trace.meta:
        print(f"meta    : {json.dumps({k: v for k, v in trace.meta.items() if k != 'steps'})}")

    for entry in store.resolve(args.trace):
        print(f"\n--- step {entry.step} " + "-" * 60)
        if entry.request is not None:
            request = store.get_blob(entry.request)
            for message in request.get("messages", []):
                role = message.get("role", "?")
                content = message.get("content")
                if content is None and message.get("tool_calls"):
                    calls = ", ".join(
                        f"{c['function']['name']}({c['function'].get('arguments', '')})"
                        for c in message["tool_calls"]
                    )
                    content = f"-> {calls}"
                text = str(content)
                if not args.full and len(text) > 200:
                    text = text[:197] + "..."
                print(f"  {role:>9}: {text}")
        response = store.get_blob(entry.blob)
        reply = response.get("choices", [{}])[0].get("message", {})
        answer = reply.get("content")
        if answer is None and reply.get("tool_calls"):
            answer = "-> " + ", ".join(c["function"]["name"] for c in reply["tool_calls"])
        print(f"  {'MODEL':>9}: {answer}")
        if isinstance(response.get("usage"), dict):
            print(f"  {'usage':>9}: {json.dumps(response['usage'], sort_keys=True)}")
    return 0


def cmd_faults(_: argparse.Namespace) -> int:
    """List the fault taxonomy with each class's docstring summary."""
    width = max(len(name) for name in FAULT_IDS)
    for name in FAULT_IDS:
        doc = (REGISTRY[name].__doc__ or "").strip().splitlines()[0]
        print(f"{name:<{width}}  {doc}")
    return 0


def cmd_outcomes(_: argparse.Namespace) -> int:
    """List the outcome taxonomy."""
    width = max(len(str(o)) for o in Outcome)
    for outcome in Outcome:
        print(f"{outcome!s:<{width}}  {outcome.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="misfeed",
        description="Feed an agent a broken tool result and find out whether it notices.",
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="cassette directory")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the proxy as an HTTP server")
    serve.add_argument("--mode", choices=[m.value for m in Mode], default="replay")
    serve.add_argument("--trace", required=True, help="trace to read or write")
    serve.add_argument("--parent", help="trunk to fork from (inject mode)")
    serve.add_argument("--fault", choices=list(FAULT_IDS), help="fault to inject")
    serve.add_argument("--fault-params", help="JSON object of fault parameters")
    serve.add_argument("--tool", help="only corrupt results from this tool")
    serve.add_argument("--at-step", type=int, help="zero-based request index to corrupt")
    serve.add_argument("--upstream", help="OpenAI-compatible base URL (record/inject)")
    serve.add_argument(
        "--api-key",
        help="upstream key; prefer the MISFEED_UPSTREAM_API_KEY environment variable",
    )
    serve.add_argument("--max-live-requests", type=int, default=50)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8756)
    serve.add_argument("--log-level", default="warning")
    serve.set_defaults(func=cmd_serve)

    traces = sub.add_parser("traces", help="list recorded traces")
    traces.set_defaults(func=cmd_traces)

    show = sub.add_parser("show", help="print a trace step by step")
    show.add_argument("trace")
    show.add_argument("--full", action="store_true", help="do not truncate message bodies")
    show.set_defaults(func=cmd_show)

    faults = sub.add_parser("faults", help="list the fault taxonomy")
    faults.set_defaults(func=cmd_faults)

    outcomes = sub.add_parser("outcomes", help="list the outcome taxonomy")
    outcomes.set_defaults(func=cmd_outcomes)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
