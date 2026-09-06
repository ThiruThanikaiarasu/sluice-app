"""Typed views over the Sluice schema, and the money conventions.

Everything below the presentation layer speaks integer minor units. `to_minor`
and `to_major` are the only sanctioned crossing points.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from functools import lru_cache

import holidays as _holidays_lib

HORIZON_DAYS = 14


def is_weekend(day_index: int, horizon_start: date) -> bool:
    return (horizon_start + timedelta(days=day_index)).weekday() >= 5


def next_business_day(day_index: int, horizon_start: date) -> int:
    """Push a day index forward past any weekend.

    Payment rails settle on business days only: a wire whose raw
    send_day + settlement_days lag lands on a Saturday or Sunday actually
    clears the next Monday, not the calendar day the lag arithmetic implies.

    Weekend-only: does not know about public holidays. Kept as its own
    function (rather than folded into `next_settlement_day`) because it is
    the horizon-agnostic building block `next_settlement_day` is built on,
    and existing callers/tests that only care about weekends still name it
    directly.
    """
    while is_weekend(day_index, horizon_start):
        day_index += 1
    return day_index


@lru_cache(maxsize=None)
def _country_holidays(country: str, year: int) -> frozenset[date]:
    """Public (bank) holidays for one ISO country code in one calendar
    year, cached since `holidays` recomputes moving dates (Easter-linked
    ones especially) from scratch on every call otherwise."""
    return frozenset(_holidays_lib.country_holidays(country, years=[year]).keys())


def is_bank_holiday(day_index: int, horizon_start: date, countries: frozenset[str]) -> bool:
    """True if the calendar date is a public holiday in any of `countries`.

    A wire leg has two ends (sender's country, receiver's country); if
    either side's banks are closed, the wire does not clear that day, so
    the check is "any", not "all".
    """
    d = horizon_start + timedelta(days=day_index)
    return any(d in _country_holidays(c, d.year) for c in countries)


def is_settlement_day(day_index: int, horizon_start: date, countries: frozenset[str]) -> bool:
    """True if a wire between these countries can clear on this day: not a
    weekend, and not a public holiday in either country."""
    return not is_weekend(day_index, horizon_start) and not is_bank_holiday(
        day_index, horizon_start, countries
    )


def next_settlement_day(day_index: int, horizon_start: date, countries: frozenset[str]) -> int:
    """Push a day index forward past weekends and public holidays in either
    of `countries`.

    `next_business_day` only knew about weekends; a wire scheduled to land
    on, say, a Singapore or Irish bank holiday would not actually land that
    day even though it isn't a Saturday or Sunday.
    """
    while not is_settlement_day(day_index, horizon_start, countries):
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
    term_days: int
    permitted: bool
    reason: str | None = None


def repayment_owed_minor(principal_minor: int, rate_bps: int, term_days: int) -> int:
    """Principal + accrued interest owed back to the lender, in the
    lender's currency, for a loan held its full contractual term.

    One formula, used both by the solver (to schedule the actual repayment
    leg) and by `verify()` (to independently recompute what should have
    been scheduled) -- so "the loan was repaid correctly" and "the loan
    was priced correctly" can never silently drift apart into two
    different numbers.
    """
    return round(principal_minor * (1 + rate_bps / 10_000 * term_days / 365))


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
