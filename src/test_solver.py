"""Tests for the solver's contract: verify() == [] on solvable scenarios,
and INFEASIBLE with a populated, specific binding_constraints otherwise."""

from __future__ import annotations

import dataclasses

import pytest

from . import baseline, seed as seed_mod, solver
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
    countries = solver._countries(conn)

    legs = solver._build_legs(entity_ids, entities, accounts, ic, costs, fx, pairs, start, countries)

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
    assert any("is not a settlement day" in v for v in violations), violations


def test_buffer_penalty_is_charged_once_per_entity_not_per_entity_day(tmp_path):
    """Regression guard for a real bug: an earlier revision created one
    buffer-slack variable per entity *per day* and charged all of them,
    compounding a persisting gap to ~10x a single leg's real FX spread over
    the horizon so the solver paid genuine cost chasing a notional buffer
    instead of treating it as a tie-break. The fix shares one slack variable
    per entity across every day, charged once. This pins the known-correct
    cost so the per-day version (which produced 242,755 on `base`, not the
    current pin) can't silently come back.

    The pin itself moved from 241,455 to 261,604 when the objective became
    lexicographic (minimise real FX+fee cost first, intercompany interest
    only as a tie-break) instead of one blended sum -- a different, cheaper
    combination of legs on real cost can carry a different total once
    interest is added back in for reporting, and that is the fix working as
    intended, not a regression.
    """
    conn = _conn("base", tmp_path)
    plan = solver.solve(conn, "base")
    assert plan.total_cost_minor == 261_604


def test_covenant_shock_ireland_flips_from_lender_to_borrower(tmp_path):
    base_conn = _conn("base", tmp_path)
    shock_conn = _conn("covenant_shock", tmp_path)

    base_plan = solver.solve(base_conn, "base")
    shock_plan = solver.solve(shock_conn, "covenant_shock")

    base_ie_lends = sum(t.amount_minor for t in base_plan.transfers if t.from_entity == "MER-IE")
    shock_ie_borrows = sum(t.amount_minor for t in shock_plan.transfers if t.to_entity == "MER-IE")

    assert base_ie_lends > 0
    assert shock_ie_borrows > 0


# --------------------------------------------------------------------------
# Loan repayment
# --------------------------------------------------------------------------
#
# Neither seeded scenario's chosen routes mature in time to repay within the
# horizon at the shipped `IC_TERM_DAYS = 7` (see the README caveat on why a
# shorter term was tried and reverted), so the mechanism is exercised here
# directly with a shortened term rather than through `base`/`covenant_shock`.

def _conn_with_term(scenario, tmp_path, monkeypatch, term_days):
    monkeypatch.setattr(seed_mod, "IC_TERM_DAYS", term_days)
    return seed_mod.seed(scenario, db_path=tmp_path / f"sluice_term{term_days}_{scenario}.db")


def test_a_short_enough_term_produces_a_real_repayment(tmp_path, monkeypatch):
    conn = _conn_with_term("covenant_shock", tmp_path, monkeypatch, term_days=5)
    plan = solver.solve(conn, "covenant_shock")
    assert plan.status == "OPTIMAL"
    assert plan.repayments, "term=5 should make at least one drawn loan repayable in time"
    assert solver.verify(conn, plan) == []

    for r in plan.repayments:
        assert r.total_minor == r.principal_minor + r.interest_minor
        assert r.pay_day < r.land_day
        assert 0 <= r.pay_day < HORIZON_DAYS
        assert 0 <= r.land_day < HORIZON_DAYS
        # The repaying entity actually owns the loan being closed.
        matching_draw = [
            t for t in plan.transfers
            if t.from_entity == r.lender_id and t.to_entity == r.borrower_id
        ]
        assert matching_draw, "a repayment must close a loan this same plan actually drew"


def test_verify_flags_a_repayment_with_the_wrong_amount(tmp_path, monkeypatch):
    conn = _conn_with_term("covenant_shock", tmp_path, monkeypatch, term_days=5)
    plan = solver.solve(conn, "covenant_shock")
    assert plan.repayments

    bad_repayment = dataclasses.replace(
        plan.repayments[0], total_minor=plan.repayments[0].total_minor + 10_000
    )
    bad_plan = dataclasses.replace(
        plan, repayments=(bad_repayment,) + plan.repayments[1:]
    )
    violations = solver.verify(conn, bad_plan)
    assert any("cost" in v or "!= recomputed" in v or "expected a repayment" in v
               for v in violations), violations


def test_verify_flags_a_repayment_with_no_matching_loan(tmp_path, monkeypatch):
    conn = _conn_with_term("covenant_shock", tmp_path, monkeypatch, term_days=5)
    plan = solver.solve(conn, "covenant_shock")
    assert plan.repayments

    real = plan.repayments[0]
    stray = dataclasses.replace(real, borrower_id="MER-CA", lender_id="MER-SG")
    bad_plan = dataclasses.replace(plan, repayments=plan.repayments + (stray,))
    violations = solver.verify(conn, bad_plan)
    assert any("does not match any transfer's loan" in v for v in violations), violations


def test_naive_baseline_does_not_model_repayment(tmp_path, monkeypatch):
    """Documented, disclosed simplification (see the README caveat): the
    naive baseline never schedules a repayment, even when its own loan's
    maturity would otherwise qualify -- pricing that in was tried and
    reverted because naive's one-shot sizing has no way to plan for its own
    future repayment need, and doing so made the naive baseline itself
    INFEASIBLE in a seeded scenario."""
    conn = _conn_with_term("covenant_shock", tmp_path, monkeypatch, term_days=5)
    naive = baseline.naive_plan(conn, "covenant_shock")
    assert naive.repayments == ()
