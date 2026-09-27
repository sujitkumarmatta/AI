"""The recording, replaying and fault-injecting proxy.

An OpenAI-compatible `POST /v1/chat/completions` endpoint in three modes:

- `REPLAY` serves only from cassettes. A miss is a hard 409 carrying diagnostics.
  It never contacts an upstream, so it needs no key and no network.
- `RECORD` serves from cassettes when it can and calls upstream when it cannot,
  appending what it learns. Re-running a partly-recorded run therefore costs only
  the requests that are still missing, which is what makes recording resumable
  across a free tier's rate limits.
- `INJECT` behaves like `RECORD` but corrupts the tool result at a chosen step and
  writes the divergent continuation to a branch.

`RECORD` and `INJECT` share "try the cassette, else go live and append" for a
reason: it makes an interrupted recording resumable rather than wasted, and it
makes a completed experiment free to re-run.

Two deliberate restrictions:

Streaming is rejected with a 400 rather than faked. Cassettes store the logical
response and `stream` is excluded from the request key, so synthesising SSE on
replay is a later addition that needs no cassette migration -- but a proxy that
silently returned a non-streaming body to a client that asked for a stream would
be exactly the kind of success-shaped lie this project is about.

Live calls are always made non-streaming and are capped by `max_live_requests`.
The cap exists because the intended recording budget is a free tier; exceeding it
fails loudly instead of quietly running up usage.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx2
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from misfeed.canon import DEFAULT_RULES, NormalizeRule, canonical_request, explain, request_key
from misfeed.faults import FaultSpec, inject_into_request
from misfeed.store import CassetteMiss, Entry, Player, Store, Trace

__all__ = ["BudgetExceeded", "Engine", "Mode", "ProxyConfig", "create_app"]


class Mode(StrEnum):
    REPLAY = "replay"
    RECORD = "record"
    INJECT = "inject"


class BudgetExceeded(RuntimeError):
    """The live-request cap was reached."""


@dataclass(slots=True)
class ProxyConfig:
    """How the proxy should behave for one run.

    `trace_id` is the trace being written or read. In INJECT mode it names the
    branch, `parent` names the trunk it forks from, and `fault_at_step` is the
    zero-based request index whose tool result gets corrupted.
    """

    mode: Mode
    store: Store
    trace_id: str
    parent: str | None = None
    fault: FaultSpec | None = None
    fault_at_step: int | None = None
    upstream_base_url: str | None = None
    upstream_api_key: str | None = None
    upstream_timeout_s: float = 120.0
    max_live_requests: int = 50
    rules: tuple[NormalizeRule, ...] = DEFAULT_RULES
    store_requests: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode is Mode.INJECT:
            if self.fault is None or self.fault_at_step is None or self.parent is None:
                raise ValueError("INJECT mode requires parent, fault and fault_at_step")
            if self.fault_at_step < 0:
                raise ValueError("fault_at_step must be >= 0")
        else:
            if self.fault is not None or self.fault_at_step is not None:
                raise ValueError(f"{self.mode.value} mode must not be given a fault")
        if self.mode is not Mode.REPLAY and not self.upstream_base_url:
            raise ValueError(f"{self.mode.value} mode requires upstream_base_url")


@dataclass(slots=True)
class Observation:
    """What happened on one request, for the run report."""

    step: int
    source: str  # "cassette" | "live"
    request_key: str
    latency_ms: float
    usage: dict[str, Any] | None = None
    fault: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "step": self.step,
            "source": self.source,
            "request_key": self.request_key,
            "latency_ms": round(self.latency_ms, 3),
        }
        if self.usage is not None:
            out["usage"] = self.usage
        if self.fault is not None:
            out["fault"] = self.fault
        return out


class Engine:
    """Mode logic, independent of the HTTP layer so it can be tested directly."""

    def __init__(self, config: ProxyConfig, client: httpx2.AsyncClient | None = None) -> None:
        self.config = config
        self.client = client
        self.step = 0
        self.live_calls = 0
        self.observations: list[Observation] = []
        self.injection: dict[str, Any] | None = None
        # Set when an injected fault actually changed the request. Until then the
        # run is still on the trunk's path and may be served from it; afterwards
        # the trunk is a different conversation and must not be matched.
        self.diverged = False
        self._new_entries: list[Entry] = []
        self._branch_existed = (
            config.mode is Mode.INJECT
            and (config.store.traces / f"{config.trace_id}.json").exists()
        )
        self._player = self._build_player()

    def _build_player(self) -> Player:
        store, config = self.config.store, self.config
        if config.mode is Mode.INJECT:
            assert config.parent is not None and config.fault_at_step is not None
            existing = store.traces / f"{config.trace_id}.json"
            if existing.exists():
                # The branch was recorded before: replay it for free. resolve()
                # gives the trunk prefix before the fork plus the branch's own
                # entries, which is exactly what was recorded.
                return Player(store, store.resolve(config.trace_id))
            # Fresh branch: the whole trunk is available. Divergence, not step
            # number, decides when to stop trusting it -- see `handle`.
            return Player(store, store.resolve(config.parent))
        try:
            return Player(store, store.resolve(config.trace_id))
        except FileNotFoundError:
            if config.mode is Mode.REPLAY:
                raise
            return Player(store, [])

    @property
    def recorded_a_divergence(self) -> bool:
        """Whether this run actually produced an experiment.

        A fault that could not apply (no tool result, or a JSON fault against a
        plain-text payload) leaves the run identical to the trunk. There is no
        experiment to record, and the runner reports it as skipped rather than
        counting it as a result.
        """
        return self.diverged

    @property
    def cassette_remaining(self) -> int:
        """Recorded responses not yet served -- non-zero after a run means drift."""
        return self._player.remaining

    async def handle(self, body: dict[str, Any]) -> dict[str, Any]:
        config = self.config
        step = self.step
        self.step += 1
        started = time.perf_counter()

        fault_detail: dict[str, Any] | None = None
        if config.mode is Mode.INJECT and step == config.fault_at_step:
            assert config.fault is not None
            outcome = inject_into_request(body, config.fault)
            body = outcome.body
            fault_detail = {"applied": outcome.applied, **outcome.detail}
            self.injection = fault_detail
            if outcome.applied:
                self.diverged = True

        key = request_key(body, config.rules)

        # Once a fresh branch has diverged, the trunk describes a different
        # conversation. Serving from it would leave the branch missing entries and
        # therefore un-replayable, so from here on everything is recorded.
        serve_from_cassette = not (self.diverged and not self._branch_existed)

        try:
            if not serve_from_cassette:
                raise CassetteMiss(key, {"reason": "run diverged from the trunk"})
            response = self._player.take(key, detail=explain(body, config.rules))
        except CassetteMiss:
            if config.mode is Mode.REPLAY:
                raise
        else:
            self.observations.append(
                Observation(
                    step=step,
                    source="cassette",
                    request_key=key,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    usage=response.get("usage") if isinstance(response, dict) else None,
                    fault=fault_detail,
                )
            )
            return dict(response)

        response = await self._call_upstream(body)
        self._append(step, key, body, response)
        self.observations.append(
            Observation(
                step=step,
                source="live",
                request_key=key,
                latency_ms=(time.perf_counter() - started) * 1000,
                usage=response.get("usage"),
                fault=fault_detail,
            )
        )
        return response

    async def _call_upstream(self, body: dict[str, Any]) -> dict[str, Any]:
        config = self.config
        if self.live_calls >= config.max_live_requests:
            raise BudgetExceeded(
                f"live request cap of {config.max_live_requests} reached; "
                "raise max_live_requests to continue recording"
            )
        if self.client is None:
            raise RuntimeError("no upstream client configured")
        self.live_calls += 1
        outgoing = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        headers = {"content-type": "application/json"}
        if config.upstream_api_key:
            headers["authorization"] = f"Bearer {config.upstream_api_key}"
        reply = await self.client.post(
            "/chat/completions",
            json=outgoing,
            headers=headers,
            timeout=config.upstream_timeout_s,
        )
        reply.raise_for_status()
        parsed: dict[str, Any] = reply.json()
        return parsed

    def _append(self, step: int, key: str, body: dict[str, Any], response: dict[str, Any]) -> None:
        store = self.config.store
        request_blob = (
            store.put_blob(canonical_request(body, self.config.rules))
            if self.config.store_requests
            else None
        )
        self._new_entries.append(
            Entry(
                step=step,
                request_key=key,
                blob=store.put_blob(response),
                request=request_blob,
            )
        )

    def flush(self) -> Trace | None:
        """Persist newly recorded entries. Returns None if nothing was recorded."""
        config = self.config
        if config.mode is Mode.REPLAY or not self._new_entries:
            return None
        meta = {
            **config.meta,
            "steps": [o.to_json() for o in self.observations],
            "live_calls": self.live_calls,
        }
        if config.mode is Mode.INJECT:
            assert config.parent is not None and config.fault is not None
            trace = Trace(
                id=config.trace_id,
                kind="branch",
                parent=config.parent,
                fork_step=config.fault_at_step,
                fault={**config.fault.to_json(), "outcome": self.injection},
                entries=list(self._new_entries),
                meta=meta,
            )
        else:
            existing = config.store.traces / f"{config.trace_id}.json"
            prior = config.store.load_trace(config.trace_id).entries if existing.exists() else []
            trace = Trace(
                id=config.trace_id,
                kind="trunk",
                entries=prior + list(self._new_entries),
                meta=meta,
            )
        config.store.save_trace(trace)
        self._new_entries.clear()
        return trace


def create_app(config: ProxyConfig, client: httpx2.AsyncClient | None = None) -> Starlette:
    """An ASGI app serving the OpenAI-compatible surface for one run.

    `client` overrides the upstream client the app would otherwise build, so a
    caller can supply a pre-configured or mock transport.
    """
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        owns_client = client is None
        upstream = client
        if upstream is None and config.mode is not Mode.REPLAY:
            upstream = httpx2.AsyncClient(base_url=config.upstream_base_url or "")
        engine = Engine(config, upstream)
        state["engine"] = engine
        try:
            yield
        finally:
            # Flush before closing so an interrupted recording keeps what it got.
            engine.flush()
            if upstream is not None and owns_client:
                await upstream.aclose()

    async def completions(request: Request) -> JSONResponse:
        engine: Engine = state["engine"]
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": {"message": "body is not valid JSON"}}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": {"message": "body must be an object"}}, status_code=400)
        if body.get("stream"):
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            "misfeed does not serve streaming responses yet. Set stream=false. "
                            "Cassettes are unaffected: `stream` is excluded from the request key, "
                            "so recordings made now will work once synthesis lands."
                        ),
                        "type": "unsupported",
                    }
                },
                status_code=400,
            )
        try:
            return JSONResponse(await engine.handle(body))
        except CassetteMiss as miss:
            return JSONResponse(
                {"error": {"message": str(miss), "type": "cassette_miss", "key": miss.key}},
                status_code=409,
            )
        except BudgetExceeded as over:
            return JSONResponse(
                {"error": {"message": str(over), "type": "budget_exceeded"}}, status_code=429
            )
        except httpx2.HTTPStatusError as upstream:
            return JSONResponse(
                {
                    "error": {
                        "message": f"upstream returned {upstream.response.status_code}",
                        "type": "upstream_error",
                        "body": upstream.response.text[:2000],
                    }
                },
                status_code=502,
            )

    async def stats(_: Request) -> JSONResponse:
        engine: Engine = state["engine"]
        return JSONResponse(
            {
                "mode": config.mode.value,
                "trace_id": config.trace_id,
                "steps": len(engine.observations),
                "live_calls": engine.live_calls,
                "cassette_remaining": engine.cassette_remaining,
                "injection": engine.injection,
                "observations": [o.to_json() for o in engine.observations],
            }
        )

    return Starlette(
        routes=[
            Route("/v1/chat/completions", completions, methods=["POST"]),
            Route("/__misfeed/stats", stats, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
