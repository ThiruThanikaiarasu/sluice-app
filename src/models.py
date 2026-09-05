"""Typed views over the Sluice schema, and the money conventions.

Everything below the presentation layer speaks integer minor units. `to_minor`
and `to_major` are the only sanctioned crossing points.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

HORIZON_DAYS = 14

HARD = "hard"
SOFT = "soft"


def to_minor(amount: float | int | str | Decimal) -> int:
    """1500.25 -> 150025."""
    return int((Decimal(str(amount)) * 100).to_integral_value())


def to_major(amount: int) -> Decimal:
    """150025 -> Decimal('1500.25'). Display only."""
    return Decimal(amount) / 100


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
