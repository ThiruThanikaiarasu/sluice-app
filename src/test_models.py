"""Tests for the business-day calendar helpers.

HORIZON_START used across the seeded scenarios (2026-09-07) is a Monday, so
day 5 is Saturday and day 6 is Sunday -- the concrete dates this module's
docstrings and the README caveat both cite.
"""

from __future__ import annotations

from datetime import date

from .models import (
    is_bank_holiday,
    is_settlement_day,
    is_weekend,
    next_business_day,
    next_settlement_day,
)

MONDAY = date(2026, 9, 7)

# Tuesday. day_index 3 -> 2026-12-25 (Friday, a genuine non-weekend holiday
# in both DE and US: Christmas Day) -- and day_index 4 -> 2026-12-26
# (Saturday, which happens to also be DE/IE Boxing Day / St. Stephen's Day,
# but not a US holiday). Day 3 is what distinguishes a holiday check from a
# weekend check; day 4's DE/US asymmetry is what distinguishes "holiday in
# *any* of the leg's two countries" from a single shared calendar.
DEC_TUESDAY = date(2026, 12, 22)


def test_is_weekend_identifies_saturday_and_sunday():
    assert not is_weekend(0, MONDAY)  # Mon
    assert not is_weekend(4, MONDAY)  # Fri
    assert is_weekend(5, MONDAY)      # Sat
    assert is_weekend(6, MONDAY)      # Sun
    assert not is_weekend(7, MONDAY)  # Mon


def test_next_business_day_is_identity_on_a_weekday():
    assert next_business_day(0, MONDAY) == 0
    assert next_business_day(4, MONDAY) == 4


def test_next_business_day_pushes_saturday_to_monday():
    assert next_business_day(5, MONDAY) == 7


def test_next_business_day_pushes_sunday_to_monday():
    assert next_business_day(6, MONDAY) == 7


def test_next_business_day_can_push_past_the_horizon():
    # day 13 in the seeded horizon is a Sunday; the caller (_build_legs) is
    # responsible for dropping a land_day that lands past HORIZON_DAYS - 1,
    # not this function, which just answers "which day does this actually
    # clear on" with no notion of a horizon boundary.
    assert next_business_day(13, MONDAY) == 14


def test_a_friday_send_with_a_two_day_lag_lands_monday_not_sunday():
    # This is the exact shape _build_legs calls with: send_day (4, Fri) +
    # settlement_days (2) = a raw land day (6) that is itself a Sunday, not
    # already a weekend send day like the two cases above.
    send_day, settlement_days = 4, 2
    assert next_business_day(send_day + settlement_days, MONDAY) == 7


def test_is_bank_holiday_true_only_for_a_country_that_observes_it():
    de = frozenset({"DE"})
    us = frozenset({"US"})
    assert is_bank_holiday(3, DEC_TUESDAY, de)   # Dec 25, DE Christmas
    assert is_bank_holiday(3, DEC_TUESDAY, us)   # Dec 25, US Christmas
    assert is_bank_holiday(4, DEC_TUESDAY, de)   # Dec 26, DE Boxing Day
    assert not is_bank_holiday(4, DEC_TUESDAY, us)  # Dec 26, not a US holiday


def test_is_settlement_day_checks_any_country_on_the_leg():
    # A DE<->US leg cannot clear on Dec 25 even though it is a plain Friday,
    # weekend-wise -- the entire point of the fix: is_weekend alone would
    # have said this day is fine.
    pair = frozenset({"DE", "US"})
    assert not is_settlement_day(3, DEC_TUESDAY, pair)
    # A weekday with no holiday on either side clears normally.
    assert is_settlement_day(0, DEC_TUESDAY, pair)


def test_next_settlement_day_skips_a_holiday_that_next_business_day_would_miss():
    pair = frozenset({"DE", "IE"})
    # day 3 = Dec 25, Fri (DE/IE Christmas holiday, not a weekend) --
    # next_business_day is blind to it and reports the day itself as fine.
    # day 4/5 = Sat 26th/Sun 27th are the weekend (also DE/IE Boxing Day,
    # redundantly). Day 6 = Mon 28th is the first day that actually clears.
    assert next_business_day(3, DEC_TUESDAY) == 3  # blind to the holiday
    assert next_settlement_day(3, DEC_TUESDAY, pair) == 6
