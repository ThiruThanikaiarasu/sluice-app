"""Tests for the solver's contract: verify() == [] on solvable scenarios,
and INFEASIBLE with a populated, specific binding_constraints otherwise."""

from __future__ import annotations

import dataclasses

import pytest

from . import baseline, solver
from .db import horizon_start
from .fx import FXTable
from .models import HORIZON_DAYS, is_weekend
from .seed import seed


def _conn(scenario, tmp_path):
    return seed(scenario, db_path=tmp_path / f"sluice_test_{scenario}.db")


@pytest.mark.parametrize("scenario", ["base", "covenant_shock"])
def test_solved_plan_has_no_violations(scenario, tmp_path):
    conn = _conn(scenario, tmp_path)
    plan = solver.solve(conn, scenario)
    assert plan.status == "OPTIMAL"
    assert plan.feasible
    violations = solver.verify(conn, plan)
    assert violations == [], violations


@pytest.mark.parametrize("scenario", ["base", "covenant_shock"])
def test_naive_baseline_has_no_violations_and_costs_more(scenario, tmp_path):
    conn = _conn(scenario, tmp_path)
    solved = solver.solve(conn, scenario)
    naive = baseline.naive_plan(conn, scenario)
    assert naive.status == "OPTIMAL"
    assert solver.verify(conn, naive) == []
    assert solved.total_cost_minor < naive.total_cost_minor


def test_infeasible_scenario_reports_specific_binding_constraints(tmp_path):
    conn = _conn("infeasible", tmp_path)
    plan = solver.solve(conn, "infeasible")
    assert plan.status == "INFEASIBLE"
    assert plan.transfers == ()
    assert plan.total_cost_minor == 0
    assert not plan.feasible
    assert plan.binding_constraints
    for line in plan.binding_constraints:
        assert len(line) > 20
        assert "problem is infeasible" not in line.lower()
    assert solver.verify(conn, plan) == []


@pytest.mark.parametrize("scenario", ["base", "covenant_shock"])
def test_no_transfer_is_sent_or_lands_on_a_weekend(scenario, tmp_path):
    conn = _conn(scenario, tmp_path)
    plan = solver.solve(conn, scenario)
    start = horizon_start(conn)
    for t in plan.transfers:
        assert not is_weekend(t.send_day, start), t
        assert not is_weekend(t.land_day, start), t


def test_build_legs_drops_a_leg_whose_adjusted_land_day_exceeds_the_horizon(tmp_path):
    """MER-SG -> MER-UK is a 2-day cross-border settlement (DBS -> Barclays,
    different countries). Sending on day 11 (Friday) raw-lands on day 13
    (Sunday); business-day-adjusted that pushes to day 14, past
    HORIZON_DAYS - 1, so _build_legs must drop this leg entirely rather than
    hand the solver a transfer that can never actually settle."""
    conn = _conn("base", tmp_path)
    entities = solver._entities(conn)
    entity_ids = list(entities.keys())
    accounts = solver.operating_accounts(conn)
    ic = solver._ic_agreements(conn)
    costs = solver._transfer_costs(conn)
    fx = FXTable(conn)
    pairs = solver._permitted_pairs(entity_ids, ic)
    start = horizon_start(conn)

    legs = solver._build_legs(entity_ids, entities, accounts, ic, costs, fx, pairs, start)

    assert ("MER-SG", "MER-UK", 11) not in legs
    for meta in legs.values():
        assert meta["land_day"] <= HORIZON_DAYS - 1


def test_verify_flags_a_hand_built_transfer_sent_on_a_weekend(tmp_path):
    conn = _conn("base", tmp_path)
    plan = solver.solve(conn, "base")
    assert plan.transfers, "base should always produce at least one real transfer"

    bad_transfer = dataclasses.replace(plan.transfers[0], send_day=5)  # Saturday
    bad_plan = dataclasses.replace(
        plan, transfers=(bad_transfer,) + plan.transfers[1:]
    )
    violations = solver.verify(conn, bad_plan)
    assert any("is not a business day" in v for v in violations), violations


def test_buffer_penalty_is_charged_once_per_entity_not_per_entity_day(tmp_path):
    """Regression guard for a real bug: an earlier revision created one
    buffer-slack variable per entity *per day* and charged all of them,
    compounding a persisting gap to ~10x a single leg's real FX spread over
    the horizon so the solver paid genuine cost chasing a notional buffer
    instead of treating it as a tie-break. The fix shares one slack variable
    per entity across every day, charged once. This pins the known-correct
    cost so the per-day version (which produced 242,755 on `base`, not
    241,455) can't silently come back.
    """
    conn = _conn("base", tmp_path)
    plan = solver.solve(conn, "base")
    assert plan.total_cost_minor == 241_455


def test_covenant_shock_ireland_flips_from_lender_to_borrower(tmp_path):
    base_conn = _conn("base", tmp_path)
    shock_conn = _conn("covenant_shock", tmp_path)

    base_plan = solver.solve(base_conn, "base")
    shock_plan = solver.solve(shock_conn, "covenant_shock")

    base_ie_lends = sum(t.amount_minor for t in base_plan.transfers if t.from_entity == "MER-IE")
    shock_ie_borrows = sum(t.amount_minor for t in shock_plan.transfers if t.to_entity == "MER-IE")

    assert base_ie_lends > 0
    assert shock_ie_borrows > 0
