"""Tests for the business-day calendar helpers.

HORIZON_START used across the seeded scenarios (2026-09-07) is a Monday, so
day 5 is Saturday and day 6 is Sunday -- the concrete dates this module's
docstrings and the README caveat both cite.
"""

from __future__ import annotations

from datetime import date

from .models import is_weekend, next_business_day

MONDAY = date(2026, 9, 7)


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
