"""Projected cash positions with no funding action taken.

This is the shared input to the solver, the naive baseline and the memo: for
each entity and day, the closing balance if nobody moves any money, and the
resulting shortfall against that entity's binding floor.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


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

    Everything downstream compares this against a balance in the entity's
    functional currency (solver.py's balance expressions, opening_balances
    below), so a covenant denominated in a different currency would be
    compared against the wrong number without ever raising -- that failure
    mode is checked for here instead.
    """
    currencies = {
        r["id"]: r["functional_currency"]
        for r in conn.execute("SELECT id, functional_currency FROM entity")
    }
    floors: dict[str, int] = {}
    for r in conn.execute("SELECT entity_id, threshold, currency FROM covenant"):
        expected = currencies.get(r["entity_id"])
        if expected is not None and r["currency"] != expected:
            raise ValueError(
                f"covenant for {r['entity_id']} is denominated in "
                f"{r['currency']}, but its functional currency is {expected}"
            )
        floors[r["entity_id"]] = max(floors.get(r["entity_id"], 0), r["threshold"])
    return floors


def opening_balances(conn: sqlite3.Connection) -> dict[str, int]:
    """Sum of each entity's bank account balances, in its functional currency.

    Summing balances across accounts in different currencies without
    converting would silently produce a meaningless total, so this checks
    every account's currency matches before summing.
    """
    currencies = {
        r["id"]: r["functional_currency"]
        for r in conn.execute("SELECT id, functional_currency FROM entity")
    }
    totals: dict[str, int] = {}
    for r in conn.execute(
        "SELECT entity_id, currency, balance FROM bank_account"
    ):
        expected = currencies.get(r["entity_id"])
        if expected is not None and r["currency"] != expected:
            raise ValueError(
                f"bank_account for {r['entity_id']} is denominated in "
                f"{r['currency']}, but its functional currency is {expected}"
            )
        totals[r["entity_id"]] = totals.get(r["entity_id"], 0) + r["balance"]
    return totals


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

    Every entity gets an entry, even one with no cash_forecast rows at all
    (falls back to its opening balance/floor) -- an entity silently missing
    from this dict would silently disappear from shortfalls()/lenders() too.
    """
    floors = binding_floors(conn)
    balances = opening_balances(conn)
    currencies = {
        r["id"]: r["functional_currency"]
        for r in conn.execute("SELECT id, functional_currency FROM entity")
    }

    worst: dict[str, Position] = {}
    latest: dict[str, Position] = {}
    for p in project(conn):
        if p.entity_id not in worst or p.closing_balance < worst[p.entity_id].closing_balance:
            worst[p.entity_id] = p
        # Latest day actually observed, not exactly HORIZON_DAYS - 1: an
        # entity missing a forecast row for the final day would otherwise
        # raise a KeyError here instead of falling back sensibly.
        if p.entity_id not in latest or p.day > latest[p.entity_id].day:
            latest[p.entity_id] = p

    summaries: dict[str, EntitySummary] = {}
    for entity_id, currency in currencies.items():
        if entity_id in worst:
            w = worst[entity_id]
            summaries[entity_id] = EntitySummary(
                entity_id=entity_id,
                currency=w.currency,
                floor=w.floor,
                min_balance=w.closing_balance,
                worst_day=w.day,
                closing_balance=latest[entity_id].closing_balance,
            )
        else:
            opening = balances.get(entity_id, 0)
            summaries[entity_id] = EntitySummary(
                entity_id=entity_id,
                currency=currency,
                floor=floors.get(entity_id, 0),
                min_balance=opening,
                worst_day=0,
                closing_balance=opening,
            )
    return summaries


def shortfalls(conn: sqlite3.Connection) -> dict[str, EntitySummary]:
    return {k: v for k, v in summarise(conn).items() if v.peak_shortfall > 0}


def lenders(conn: sqlite3.Connection) -> dict[str, EntitySummary]:
    return {k: v for k, v in summarise(conn).items() if v.free_cash > 0}
