"""Tests for the solver's contract: verify() == [] on solvable scenarios,
and INFEASIBLE with a populated, specific binding_constraints otherwise."""

from __future__ import annotations

import pytest

from . import baseline, solver
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


def test_covenant_shock_ireland_flips_from_lender_to_borrower(tmp_path):
    base_conn = _conn("base", tmp_path)
    shock_conn = _conn("covenant_shock", tmp_path)

    base_plan = solver.solve(base_conn, "base")
    shock_plan = solver.solve(shock_conn, "covenant_shock")

    base_ie_lends = sum(t.amount_minor for t in base_plan.transfers if t.from_entity == "MER-IE")
    shock_ie_borrows = sum(t.amount_minor for t in shock_plan.transfers if t.to_entity == "MER-IE")

    assert base_ie_lends > 0
    assert shock_ie_borrows > 0
