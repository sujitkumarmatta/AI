"""Task definitions: a question, the tools it needs, and its reference answer.

The instruction to end with an `ANSWER:` line is part of the task, not a trick.
It makes `answered` and `correct` exact predicates instead of a text-matching
guess, which is what keeps a model out of the grading loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals.world import TOOL_SCHEMAS, reference_answer

__all__ = ["TASKS", "Task", "get_task"]

ANSWER_PROTOCOL = (
    "Work through the problem using the tools available. "
    "End your final message with a line of exactly the form 'ANSWER: <value>', "
    "where <value> is a bare number with no units, separators or currency symbol."
)

# The baseline says nothing about tool reliability. That is the point: it is the
# control condition against which a distrust instruction is compared.
BASELINE_SYSTEM_PROMPT = f"You are a data assistant. {ANSWER_PROTOCOL}"

# The treatment. One added paragraph, so that any difference in outcome is
# attributable to it rather than to a rewrite.
DISTRUST_SYSTEM_PROMPT = (
    "You are a data assistant. "
    "Tool results may be incomplete, empty, stale or malformed even when the call "
    "appears to succeed. Before using a tool result, check that it actually "
    "contains the data you asked for. If it does not, say so explicitly and do not "
    "guess a value or substitute zero. "
    f"{ANSWER_PROTOCOL}"
)

SYSTEM_PROMPTS: dict[str, str] = {
    "baseline": BASELINE_SYSTEM_PROMPT,
    "distrust": DISTRUST_SYSTEM_PROMPT,
}


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    question: str
    reference_id: str
    tools: list[dict[str, Any]]

    @property
    def expected(self) -> Any:
        return reference_answer(self.reference_id)


TASKS: tuple[Task, ...] = (
    Task(
        id="T1_shipped_total",
        question=(
            "What is the total amount, in cents, of all shipped orders "
            "for the customer named 'Alcott Foods'?"
        ),
        reference_id="shipped_total_cents",
        tools=TOOL_SCHEMAS,
    ),
    Task(
        id="T2_shipped_count",
        question="How many shipped orders does the customer named 'Alcott Foods' have?",
        reference_id="shipped_count",
        tools=TOOL_SCHEMAS,
    ),
)


def get_task(task_id: str) -> Task:
    for task in TASKS:
        if task.id == task_id:
            return task
    raise KeyError(f"no task {task_id!r}; known: {', '.join(t.id for t in TASKS)}")
