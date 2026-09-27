"""misfeed: feed an agent a broken tool result and find out whether it notices.

Record an agent run through an OpenAI-compatible proxy, replay it deterministically,
then corrupt a chosen tool result and record only the divergent continuation. After
that every experiment replays from committed files, with no endpoint and no key.
"""

from __future__ import annotations

from misfeed.canon import DEFAULT_RULES, NormalizeRule, request_key
from misfeed.faults import FAULT_IDS, FaultSpec, apply_fault, inject_into_request
from misfeed.proxy import BudgetExceeded, Engine, Mode, ProxyConfig, create_app
from misfeed.report import RunResult, markdown_report, summarise
from misfeed.store import CanonMismatch, CassetteMiss, Entry, Player, Store, Trace
from misfeed.streaming import sse_events
from misfeed.toolfault import Corruption, ToolFaultPlan, ToolInjector
from misfeed.verdict import Outcome, RunFacts, classify

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_RULES",
    "FAULT_IDS",
    "BudgetExceeded",
    "CanonMismatch",
    "CassetteMiss",
    "Corruption",
    "Engine",
    "Entry",
    "FaultSpec",
    "Mode",
    "NormalizeRule",
    "Outcome",
    "Player",
    "ProxyConfig",
    "RunFacts",
    "RunResult",
    "Store",
    "ToolFaultPlan",
    "ToolInjector",
    "Trace",
    "__version__",
    "apply_fault",
    "classify",
    "create_app",
    "inject_into_request",
    "markdown_report",
    "request_key",
    "sse_events",
    "summarise",
]
