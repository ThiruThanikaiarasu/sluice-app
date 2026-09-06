"""Typed views over the Sluice schema, and the money conventions.

Everything below the presentation layer speaks integer minor units. `to_minor`
and `to_major` are the only sanctioned crossing points.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

HORIZON_DAYS = 14


def is_weekend(day_index: int, horizon_start: date) -> bool:
    return (horizon_start + timedelta(days=day_index)).weekday() >= 5


def next_business_day(day_index: int, horizon_start: date) -> int:
    """Push a day index forward past any weekend.

    Payment rails settle on business days only: a wire whose raw
    send_day + settlement_days lag lands on a Saturday or Sunday actually
    clears the next Monday, not the calendar day the lag arithmetic implies.
    """
    while is_weekend(day_index, horizon_start):
        day_index += 1
    return day_index

HARD = "hard"
SOFT = "soft"


def to_minor(amount: float | int | str | Decimal) -> int:
    """1500.25 -> 150025."""
    return int((Decimal(str(amount)) * 100).to_integral_value())


def to_major(amount: int) -> Decimal:
    """150025 -> Decimal('1500.25'). Display only."""
    return Decimal(amount) / 100


def format_money(amount_minor: int, currency: str) -> str:
    """150025, 'USD' -> 'USD 1,500.25'. The one shared money-display format,
    so app.py and memo.py don't each grow their own copy that can drift."""
    return f"{currency} {to_major(amount_minor):,.2f}"


def is_hard(covenant: dict) -> bool:
    """True if a covenant row (as returned by sqlite3.Row/dict) is hard --
    i.e. no remedy may ever propose breaching it."""
    return covenant["hardness"] == HARD


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    country: str
    functional_currency: str


@dataclass(frozen=True)
class BankAccount:
    id: str
    entity_id: str
    bank: str
    currency: str
    balance: int
    purpose: str | None = None


@dataclass(frozen=True)
class Covenant:
    entity_id: str
    kind: str
    threshold: int
    currency: str
    hardness: str
    source_doc: str | None = None
    source_quote: str | None = None

    @property
    def inviolable(self) -> bool:
        """True if no remedy may ever propose breaching this."""
        return self.hardness == HARD


@dataclass(frozen=True)
class CashForecast:
    entity_id: str
    day: int
    date: str
    net_flow: int
    note: str | None = None


@dataclass(frozen=True)
class ICAgreement:
    lender_id: str
    borrower_id: str
    max_limit: int
    rate_bps: int
    permitted: bool
    reason: str | None = None


@dataclass(frozen=True)
class TransferCost:
    from_bank: str
    to_bank: str
    fixed_fee: int
    settlement_days: int


@dataclass(frozen=True)
class LearnedRule:
    rule_text: str
    constraint_json: str
    created_at: str
    origin_run_id: str | None = None
