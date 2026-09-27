"""A small deterministic world, its tools, and reference answers.

Thread-safe, because agent frameworks call tools from an executor: LangGraph runs
them in a thread pool, and a default sqlite3 connection refuses use from any thread
but its creator. The connection is opened with `check_same_thread=False` and every
access is taken under a lock, which is correct here because the database is read-only
after construction and contention is irrelevant at this size.


Tasks are answerable by a SQL query over this data, which is the point: a task
whose ground truth is computed cannot drift, cannot be gamed, and needs no
labelling. That restricts the tasks to verifiable ones -- an honest cost, paid so
that no model grades another model anywhere in the evaluation.

The data is fixed, small, and has the shape that makes tool faults bite: a
multi-row result the agent must aggregate, a status filter it must respect, and
two hops (name to id, id to orders) so that a fault can be injected at either.

`list_orders` deliberately does not return a row count. If it did, an agent could
answer "how many" from the count field and a silently truncated list would not
bite -- which is exactly the kind of accidental robustness that would make a
measurement meaningless.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

__all__ = ["TOOL_SCHEMAS", "World", "reference_answer"]

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    region      TEXT NOT NULL
);
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    amount_cents INTEGER NOT NULL,
    currency     TEXT NOT NULL,
    placed_on    TEXT NOT NULL,
    status       TEXT NOT NULL
);
"""

CUSTOMERS: tuple[tuple[int, str, str], ...] = (
    (1, "Alcott Foods", "EU"),
    (2, "Brightwater Ltd", "NA"),
)

ORDERS: tuple[tuple[int, int, int, str, str, str], ...] = (
    (101, 1, 1250, "USD", "2026-03-04", "shipped"),
    (102, 1, 3400, "USD", "2026-04-11", "shipped"),
    (103, 1, 990, "USD", "2026-05-02", "pending"),
    (104, 1, 2075, "USD", "2026-06-19", "shipped"),
    (201, 2, 500, "USD", "2026-02-02", "shipped"),
)

# What the agent is told it can call. Kept minimal on purpose: a wide tool surface
# would let an agent route around a fault for reasons unrelated to its handling.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_customer",
            "description": "Look up a customer by exact name. Returns their id and region.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": (
                "List a customer's orders, optionally filtered by status "
                "('shipped' or 'pending'). Returns one entry per order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "status": {"type": "string"},
                },
                "required": ["customer_id"],
            },
        },
    },
]


@dataclass(slots=True)
class ToolCallRecord:
    """What a tool was asked and what it really returned, before any injection."""

    name: str
    arguments: dict[str, Any]
    result: str


class World:
    """The database plus the tool implementations that read it."""

    def __init__(self) -> None:
        # check_same_thread=False plus a lock: frameworks dispatch tools onto an
        # executor, and a default connection would refuse the call.
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS)
        self.db.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", ORDERS)
        self.db.commit()
        self.calls: list[ToolCallRecord] = []

    def close(self) -> None:
        self.db.close()

    def find_customer(self, name: str) -> dict[str, Any]:
        with self._lock:
            row = self.db.execute(
                "SELECT customer_id, name, region FROM customers WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return {"error": f"no customer named {name!r}"}
        return {"customer_id": row["customer_id"], "name": row["name"], "region": row["region"]}

    def list_orders(self, customer_id: int, status: str | None = None) -> dict[str, Any]:
        sql = (
            "SELECT order_id, amount_cents, currency, placed_on, status "
            "FROM orders WHERE customer_id = ?"
        )
        params: list[Any] = [customer_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY order_id"
        with self._lock:
            rows = self.db.execute(sql, params).fetchall()
        return {
            "orders": [
                {
                    "order_id": row["order_id"],
                    "amount_cents": row["amount_cents"],
                    "currency": row["currency"],
                    "placed_on": row["placed_on"],
                    "status": row["status"],
                }
                for row in rows
            ]
        }

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call, returning the JSON string the agent would see."""
        if name == "find_customer":
            result: dict[str, Any] = self.find_customer(str(arguments.get("name", "")))
        elif name == "list_orders":
            raw_status = arguments.get("status")
            result = self.list_orders(
                int(arguments["customer_id"]),
                str(raw_status) if raw_status is not None else None,
            )
        else:
            result = {"error": f"no such tool {name!r}"}
        payload = json.dumps(result)
        with self._lock:
            self.calls.append(ToolCallRecord(name=name, arguments=arguments, result=payload))
        return payload


def reference_answer(question_id: str) -> Any:
    """Ground truth, computed from the same data the tools read.

    Deliberately a separate query rather than a constant, so that changing the
    fixture data cannot leave a stale expected value behind.
    """
    world = World()
    try:
        if question_id == "shipped_total_cents":
            row = world.db.execute(
                """
                SELECT COALESCE(SUM(o.amount_cents), 0) AS total
                FROM orders o JOIN customers c USING (customer_id)
                WHERE c.name = 'Alcott Foods' AND o.status = 'shipped'
                """
            ).fetchone()
            return int(row["total"])
        if question_id == "shipped_count":
            row = world.db.execute(
                """
                SELECT COUNT(*) AS n
                FROM orders o JOIN customers c USING (customer_id)
                WHERE c.name = 'Alcott Foods' AND o.status = 'shipped'
                """
            ).fetchone()
            return int(row["n"])
        raise KeyError(f"no reference answer for {question_id!r}")
    finally:
        world.close()
