"""Projected cash positions with no funding action taken.

This is the shared input to the solver, the naive baseline and the memo: for
each entity and day, the closing balance if nobody moves any money, and the
resulting shortfall against that entity's binding floor.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .models import HORIZON_DAYS


@dataclass(frozen=True)
class Position:
    entity_id: str
    currency: str
    day: int
    date: str
    closing_balance: int
    floor: int

    @property
    def shortfall(self) -> int:
        return max(0, self.floor - self.closing_balance)

    @property
    def breaches(self) -> bool:
        return self.closing_balance < self.floor


@dataclass(frozen=True)
class EntitySummary:
    entity_id: str
    currency: str
    floor: int
    min_balance: int
    worst_day: int
    closing_balance: int

    @property
    def peak_shortfall(self) -> int:
        """What must be funded to hold the floor across the whole horizon."""
        return max(0, self.floor - self.min_balance)

    @property
    def free_cash(self) -> int:
        """What this entity could lend without ever breaching its own floor."""
        return max(0, self.min_balance - self.floor)


def binding_floors(conn: sqlite3.Connection) -> dict[str, int]:
    """The highest floor per entity, hard or soft.

    Both bind during the solve. Only the remedy ranker, reached when the solve
    fails, is allowed to treat a soft floor as negotiable.
    """
    floors: dict[str, int] = {}
    for r in conn.execute("SELECT entity_id, threshold FROM covenant"):
        floors[r["entity_id"]] = max(floors.get(r["entity_id"], 0), r["threshold"])
    return floors


def opening_balances(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT entity_id, SUM(balance) AS total FROM bank_account "
        "GROUP BY entity_id"
    )
    return {r["entity_id"]: r["total"] for r in rows}


def project(conn: sqlite3.Connection) -> list[Position]:
    floors = binding_floors(conn)
    balances = opening_balances(conn)
    currencies = {
        r["id"]: r["functional_currency"]
        for r in conn.execute("SELECT id, functional_currency FROM entity")
    }

    positions: list[Position] = []
    for entity_id, currency in currencies.items():
        running = balances.get(entity_id, 0)
        rows = conn.execute(
            "SELECT day, date, net_flow FROM cash_forecast "
            "WHERE entity_id = ? ORDER BY day",
            (entity_id,),
        )
        for row in rows:
            running += row["net_flow"]
            positions.append(Position(
                entity_id=entity_id,
                currency=currency,
                day=row["day"],
                date=row["date"],
                closing_balance=running,
                floor=floors.get(entity_id, 0),
            ))
    return positions


def summarise(conn: sqlite3.Connection) -> dict[str, EntitySummary]:
    """Per-entity worst point over the horizon.

    Small enough to hand to a language model whole, which is the point -- the
    raw projection is 84 rows and would crowd a 32k context for no benefit.
    """
    worst: dict[str, Position] = {}
    final: dict[str, Position] = {}
    for p in project(conn):
        if p.entity_id not in worst or p.closing_balance < worst[p.entity_id].closing_balance:
            worst[p.entity_id] = p
        if p.day == HORIZON_DAYS - 1:
            final[p.entity_id] = p

    return {
        entity_id: EntitySummary(
            entity_id=entity_id,
            currency=p.currency,
            floor=p.floor,
            min_balance=p.closing_balance,
            worst_day=p.day,
            closing_balance=final[entity_id].closing_balance,
        )
        for entity_id, p in worst.items()
    }


def shortfalls(conn: sqlite3.Connection) -> dict[str, EntitySummary]:
    return {k: v for k, v in summarise(conn).items() if v.peak_shortfall > 0}


def lenders(conn: sqlite3.Connection) -> dict[str, EntitySummary]:
    return {k: v for k, v in summarise(conn).items() if v.free_cash > 0}
